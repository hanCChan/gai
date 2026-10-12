from __future__ import annotations

import math

import torch
from torch import nn

from .cnn_stem import CNNStemPatchEmbed
from .features import build_patch_features, patchify, torch_rf_views
from .losses import DomainAdversarialHead
from .multiscale import MultiScaleTokenFusion, parse_ratios
from .oob_fusion import OOBCrossAttentionFusion, OOBGatedFusion


class RFHSTUBlock(nn.Module):
    """Minimal RF-HSTU block for the first implementation pass."""

    def __init__(self, dim: int, num_patches: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.qkvu = nn.Linear(dim, dim * 4)
        self.relative_bias = nn.Parameter(torch.zeros(2 * num_patches - 1))
        self.av_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        self.residual_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        idx = torch.arange(num_patches)
        rel = idx[:, None] - idx[None, :] + num_patches - 1
        self.register_buffer("relative_index", rel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, u = self.qkvu(x).chunk(4, dim=-1)
        score = q @ k.transpose(-2, -1) / math.sqrt(self.dim)
        score = score + self.relative_bias[self.relative_index].unsqueeze(0)
        attn = torch.sigmoid(score)
        av = attn @ v
        y = self.out(self.av_norm(av) * self.act(u))
        y = self.dropout(y)
        return self.residual_norm(x + y)


class RFHSTUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_patches: int = 32,
        oob_input_dim: int | None = None,
        patch_size: int = 256,
        sample_rate: float = 1_000_000.0,
        lora_bandwidth: float = 125_000.0,
        spreading_factor: int = 7,
        use_chirp_embedding: bool = False,
        use_oob_cross_attention: bool = False,
        use_oob_gated: bool = False,
        oob_num_heads: int = 4,
        dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.dim = dim
        self.pool = pool
        self.use_chirp_embedding = use_chirp_embedding
        self.use_oob_cross_attention = use_oob_cross_attention
        self.use_oob_gated = use_oob_gated
        self.input_proj = nn.Linear(input_dim, dim)
        self.oob_proj = nn.Linear(oob_input_dim, dim) if (use_oob_cross_attention or use_oob_gated) and oob_input_dim is not None else None
        self.oob_fusion = (
            OOBCrossAttentionFusion(dim, num_heads=oob_num_heads, dropout=dropout)
            if use_oob_cross_attention and oob_input_dim is not None
            else OOBGatedFusion(dim, dropout=dropout)
            if use_oob_gated and oob_input_dim is not None
            else None
        )
        self.position = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.chirp_id_embedding: nn.Embedding | None = None
        self.patch_in_chirp_embedding: nn.Embedding | None = None
        if use_chirp_embedding:
            samples_per_chirp = sample_rate * (2**spreading_factor) / lora_bandwidth
            patches_per_chirp = max(1, int(round(samples_per_chirp / patch_size)))
            num_chirps = max(1, math.ceil(num_patches / patches_per_chirp))
            patch_index = torch.arange(num_patches)
            self.register_buffer("chirp_ids", patch_index // patches_per_chirp, persistent=False)
            self.register_buffer("patch_in_chirp_ids", patch_index % patches_per_chirp, persistent=False)
            self.chirp_id_embedding = nn.Embedding(num_chirps, dim)
            self.patch_in_chirp_embedding = nn.Embedding(patches_per_chirp, dim)
        self.blocks = nn.ModuleList([RFHSTUBlock(dim, num_patches, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim)

    def _add_structure_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == self.num_patches:
            x = x + self.position
            if self.use_chirp_embedding:
                if self.chirp_id_embedding is None or self.patch_in_chirp_embedding is None:
                    raise RuntimeError("Chirp embedding is enabled but embedding tables are missing.")
                chirp = self.chirp_id_embedding(self.chirp_ids).unsqueeze(0)
                patch_in_chirp = self.patch_in_chirp_embedding(self.patch_in_chirp_ids).unsqueeze(0)
                x = x + chirp + patch_in_chirp
        return x

    def prepare_tokens(self, patch_features: torch.Tensor, oob_features: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(patch_features) + self.position
        if self.use_chirp_embedding:
            if self.chirp_id_embedding is None or self.patch_in_chirp_embedding is None:
                raise RuntimeError("Chirp embedding is enabled but embedding tables are missing.")
            chirp = self.chirp_id_embedding(self.chirp_ids).unsqueeze(0)
            patch_in_chirp = self.patch_in_chirp_embedding(self.patch_in_chirp_ids).unsqueeze(0)
            x = x + chirp + patch_in_chirp
        if self.use_oob_cross_attention or self.use_oob_gated:
            if oob_features is None or self.oob_proj is None or self.oob_fusion is None:
                raise RuntimeError("OOB fusion is enabled but OOB features are missing.")
            x_oob = self.oob_proj(oob_features)
            x_oob = self._add_structure_embeddings(x_oob)
            x = self.oob_fusion(x, x_oob)
        return x

    def forward_tokens(self, patch_features: torch.Tensor, oob_features: torch.Tensor | None = None) -> torch.Tensor:
        x = self.prepare_tokens(patch_features, oob_features=oob_features)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, patch_features: torch.Tensor, oob_features: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.forward_tokens(patch_features, oob_features=oob_features)
        if self.pool == "mean":
            return tokens.mean(dim=1)
        if self.pool == "first":
            return tokens[:, 0]
        raise ValueError(f"Unknown pool mode: {self.pool}")


class RFHSTUTokenEncoder(nn.Module):
    """RF-HSTU over already projected tokens, used by multi-scale branches."""

    def __init__(self, dim: int, num_patches: int, depth: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([RFHSTUBlock(dim, num_patches, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class RFPatchEmbedder(nn.Module):
    def __init__(
        self,
        window_size: int = 8192,
        patch_size: int = 256,
        sample_rate: float = 1_000_000.0,
        lora_bandwidth: float = 125_000.0,
        spreading_factor: int = 7,
        use_oob: bool = True,
        oob_fusion_type: str = "concat_oob",
        use_oob_cross_attention: bool = False,
        patch_embed_type: str = "linear",
        dim: int = 128,
        cnn_stem_dim: int = 32,
        cnn_stem_kernels: str = "7,5",
        fft_norm: str = "log_zscore",
        oob_norm: str = "zscore",
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.patch_size = patch_size
        self.sample_rate = sample_rate
        self.lora_bandwidth = lora_bandwidth
        self.spreading_factor = spreading_factor
        self.patch_embed_type = patch_embed_type
        self.dim = dim
        self.fft_norm = fft_norm
        self.oob_norm = oob_norm
        self.oob_fusion_type = "no_oob" if not use_oob else oob_fusion_type
        if self.patch_embed_type not in {"linear", "cnn_stem"}:
            raise ValueError(f"Unknown patch_embed_type={patch_embed_type!r}")
        if self.oob_fusion_type == "cross_attn_oob" and not use_oob_cross_attention:
            print("WARNING: oob_fusion_type=cross_attn_oob requires use_oob_cross_attention=True; falling back to concat_oob.")
            self.oob_fusion_type = "concat_oob"
        self.use_oob_cross_attention = use_oob_cross_attention and self.oob_fusion_type == "cross_attn_oob"
        self.use_oob_gated = self.oob_fusion_type == "gated_oob"
        self.use_oob = self.oob_fusion_type != "no_oob"
        self.cnn_stem = (
            CNNStemPatchEmbed(
                in_channels=5,
                dim=dim,
                patch_size=patch_size,
                stem_dim=cnn_stem_dim,
                kernels=cnn_stem_kernels,
            )
            if self.patch_embed_type == "cnn_stem"
            else None
        )

    @property
    def num_patches(self) -> int:
        return self.window_size // self.patch_size

    @property
    def input_dim(self) -> int:
        if self.patch_embed_type == "cnn_stem":
            channels = self.dim
            if self.oob_fusion_type == "concat_oob":
                channels += self.patch_size
            return channels
        channels = 2 + 1 + 2
        if self.oob_fusion_type == "concat_oob":
            channels += 1
        return channels * self.patch_size

    @property
    def oob_input_dim(self) -> int | None:
        if self.oob_fusion_type in {"cross_attn_oob", "gated_oob"}:
            return self.patch_size
        return None

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.patch_embed_type == "cnn_stem":
            iq_view, fft_view, oob_view, amp_phase = torch_rf_views(
                iq, self.sample_rate, self.lora_bandwidth, fft_norm=self.fft_norm, oob_norm=self.oob_norm
            )
            views = {
                "iq": patchify(iq_view, self.patch_size),
                "fft": patchify(fft_view, self.patch_size),
                "amp_phase": patchify(amp_phase, self.patch_size),
            }
            if self.use_oob:
                views["oob"] = patchify(oob_view, self.patch_size)
            if self.cnn_stem is None:
                raise RuntimeError("patch_embed_type=cnn_stem but cnn_stem module is missing.")
            x_main = torch.cat([iq_view, fft_view, amp_phase], dim=1)
            main_tokens = self.cnn_stem(x_main)
            if self.oob_fusion_type == "concat_oob":
                return torch.cat([main_tokens, views["oob"]], dim=-1), views
            if self.oob_fusion_type in {"cross_attn_oob", "gated_oob"}:
                return main_tokens, views
            return main_tokens, views
        _, views = build_patch_features(
            iq,
            patch_size=self.patch_size,
            sample_rate=self.sample_rate,
            lora_bandwidth=self.lora_bandwidth,
            use_oob=self.use_oob,
            fft_norm=self.fft_norm,
            oob_norm=self.oob_norm,
        )
        main = torch.cat([views["iq"], views["fft"], views["amp_phase"]], dim=-1)
        if self.oob_fusion_type == "concat_oob":
            return torch.cat([main, views["oob"]], dim=-1), views
        return main, views


class RFMAE(nn.Module):
    def __init__(
        self,
        embedder: RFPatchEmbedder,
        dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        mask_ratio: float = 0.4,
        use_chirp_embedding: bool = False,
        oob_num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedder.input_dim))
        self.encoder = RFHSTUEncoder(
            input_dim=embedder.input_dim,
            num_patches=embedder.num_patches,
            oob_input_dim=embedder.oob_input_dim,
            patch_size=embedder.patch_size,
            sample_rate=embedder.sample_rate,
            lora_bandwidth=embedder.lora_bandwidth,
            spreading_factor=embedder.spreading_factor,
            use_chirp_embedding=use_chirp_embedding,
            use_oob_cross_attention=embedder.use_oob_cross_attention,
            oob_num_heads=oob_num_heads,
            dim=dim,
            depth=depth,
            dropout=dropout,
        )
        self.heads = nn.ModuleDict(
            {
                "iq": nn.Linear(dim, 2 * embedder.patch_size),
                "fft": nn.Linear(dim, embedder.patch_size),
                "amp_phase": nn.Linear(dim, 2 * embedder.patch_size),
            }
        )
        if embedder.use_oob:
            self.heads["oob"] = nn.Linear(dim, embedder.patch_size)

    def make_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        patches = self.embedder.num_patches
        num_mask = max(1, int(round(patches * self.mask_ratio)))
        noise = torch.rand(batch_size, patches, device=device)
        ids = noise.argsort(dim=1)
        mask = torch.zeros(batch_size, patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids[:, :num_mask], True)
        return mask

    def forward(self, iq: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        features, targets = self.embedder(iq)
        mask = self.make_mask(features.shape[0], features.device)
        masked_features = torch.where(mask.unsqueeze(-1), self.mask_token.to(features.dtype), features)
        tokens = self.encoder.forward_tokens(masked_features, oob_features=targets.get("oob"))
        pred = {name: head(tokens) for name, head in self.heads.items()}
        return pred, targets, mask


class DeviceClassifier(nn.Module):
    def __init__(
        self,
        embedder: RFPatchEmbedder,
        num_classes: int,
        dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        domain_sizes: dict[str, int] | None = None,
        use_chirp_embedding: bool = False,
        oob_num_heads: int = 4,
        use_multiscale: bool = False,
        multiscale_ratios: str | list[int] = "1,2,4",
        multiscale_fusion_type: str = "concat_pool",
        use_cfo_feature: bool = False,
        cfo_feature_type: str = "both",
        cfo_feature_norm: str = "train_zscore",
        oob_dropout: float = 0.0,
        mixstyle: bool = False,
        mixstyle_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.use_multiscale = use_multiscale
        self.use_cfo_feature = use_cfo_feature
        self.cfo_feature_type = cfo_feature_type
        self.cfo_feature_norm = cfo_feature_norm
        self.oob_dropout = oob_dropout
        from .mixstyle import MixStyle

        self.mixstyle = MixStyle(mixstyle_alpha) if mixstyle else None
        self.num_cfo_features = 0
        self.cfo_norm: nn.Module | None = None
        if use_cfo_feature:
            self.num_cfo_features = 2 if cfo_feature_type == "both" else 1
            if cfo_feature_norm == "train_zscore":
                # BatchNorm with running stats (frozen at eval) == normalization by train statistics.
                self.cfo_norm = nn.BatchNorm1d(self.num_cfo_features, affine=False)
            else:
                raise ValueError(f"Unknown cfo_feature_norm={cfo_feature_norm!r}")
        self.encoder = RFHSTUEncoder(
            input_dim=embedder.input_dim,
            num_patches=embedder.num_patches,
            oob_input_dim=embedder.oob_input_dim,
            patch_size=embedder.patch_size,
            sample_rate=embedder.sample_rate,
            lora_bandwidth=embedder.lora_bandwidth,
            spreading_factor=embedder.spreading_factor,
            use_chirp_embedding=use_chirp_embedding,
            use_oob_cross_attention=embedder.use_oob_cross_attention,
            use_oob_gated=embedder.use_oob_gated,
            oob_num_heads=oob_num_heads,
            dim=dim,
            depth=depth,
            dropout=dropout,
        )
        self.multiscale: MultiScaleTokenFusion | None = None
        if use_multiscale:
            ratios = parse_ratios(multiscale_ratios)
            encoders = []
            for ratio in ratios:
                scale_patches = max(1, embedder.num_patches // ratio)
                encoders.append(RFHSTUTokenEncoder(dim=dim, num_patches=scale_patches, depth=depth, dropout=dropout))
            self.multiscale = MultiScaleTokenFusion(encoders, ratios=ratios, dim=dim, fusion_type=multiscale_fusion_type)
        self.embedding_dim = dim + self.num_cfo_features
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        self.domain_head = DomainAdversarialHead(dim, domain_sizes) if domain_sizes else None

    def _compute_cfo_features(self, iq: torch.Tensor) -> torch.Tensor:
        """Estimate CFO proxies (peak_offset / spectral_centroid) per window in Hz.

        No CFO compensation is performed; this only produces auxiliary features.
        """
        eps = 1e-8
        length = iq.shape[-1]
        complex_x = torch.complex(iq[:, 0], iq[:, 1])
        spectrum = torch.fft.fftshift(torch.fft.fft(complex_x, dim=-1), dim=-1)
        psd = spectrum.abs() ** 2  # [B, T]
        freq = torch.fft.fftshift(
            torch.fft.fftfreq(length, d=1.0 / self.embedder.sample_rate)
        ).to(iq.device).to(psd.dtype)  # [T]
        total = psd.sum(dim=-1).clamp_min(eps)
        centroid = (psd * freq.unsqueeze(0)).sum(dim=-1) / total  # [B]
        peak_offset = freq[psd.argmax(dim=-1)]  # [B]
        if self.cfo_feature_type == "peak_offset":
            feats = peak_offset.unsqueeze(-1)
        elif self.cfo_feature_type == "spectral_centroid":
            feats = centroid.unsqueeze(-1)
        else:  # "both"
            feats = torch.stack([peak_offset, centroid], dim=-1)
        return feats

    def forward(
        self,
        iq: torch.Tensor,
        adv_lambda: float = 1.0,
        return_features: bool = False,
        return_supcon_features: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        features, views = self.embedder(iq)
        if self.training and self.oob_dropout > 0.0 and views.get("oob") is not None:
            if torch.rand(1).item() < self.oob_dropout:
                views = dict(views)
                views["oob"] = torch.zeros_like(views["oob"])
        if self.use_multiscale:
            if self.multiscale is None:
                raise RuntimeError("Multi-scale is enabled but fusion module is missing.")
            tokens = self.encoder.prepare_tokens(features, oob_features=views.get("oob"))
            if self.mixstyle is not None:
                tokens = self.mixstyle(tokens)
            z = self.multiscale(tokens)
        else:
            tokens = self.encoder.prepare_tokens(features, oob_features=views.get("oob"))
            if self.mixstyle is not None:
                tokens = self.mixstyle(tokens)
            for block in self.encoder.blocks:
                tokens = block(tokens)
            tokens = self.encoder.final_norm(tokens)
            z = tokens.mean(dim=1) if self.encoder.pool == "mean" else tokens[:, 0]
        if self.use_cfo_feature and self.cfo_norm is not None:
            cfo = self._compute_cfo_features(iq)
            cfo = self.cfo_norm(cfo)
            embedding = torch.cat([z, cfo], dim=-1)
        else:
            embedding = z
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "embedding": embedding,
            "logits": self.classifier(embedding),
        }
        if return_features:
            out["features"] = embedding
        if return_supcon_features:
            out["supcon_features"] = embedding
        if self.domain_head is not None:
            out["domain_logits"] = self.domain_head(z, adv_lambda)
        return out
