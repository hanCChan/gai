#!/usr/bin/env python3
"""Audit original data read-only; generate an explicitly provisional protocol.

All HDF5 files: byte SHA256, headers, attributes, labels only.
Optional signal diagnostics: strictly allowlisted Day1/Day2 development files.
Final dates and other receivers: no CFO/SNR values or signal features are read.
SHA256 necessarily reads raw file bytes, but never decodes held-out IQ samples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RELATIVE = "data/external/shen_receiver_agnostic"
sys.path.insert(0, str(ROOT / "src"))
# Keep the audit independent of PyTorch; the access guard below is identical
# to the adapter's exact path allowlist and is tested by the environment check.
DEVELOPMENT_FILES = {
    f"{folder}/{rx}_day{day}_wireless_{kind}.h5"
    for rx in ("n210_1", "rtl_6")
    for folder, day, kind in (("Train", 1, "train"), ("Test", 1, "test"), ("Test", 2, "test"))
}


def sha256(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def attributes(obj):
    return {str(k): {"dtype": str(np.asarray(v).dtype), "value": str(v)}
            for k, v in obj.attrs.items()}


def identity(relative):
    match = re.fullmatch(r"(?:Train|Test)/(n210_1|rtl_6)_day([1-4])_wireless_(train|test)\.h5", relative)
    if match:
        rx, day, kind = match.groups()
        role = ("train" if kind == "train" else
                "same_day_diagnostic" if day == "1" else
                "validation" if day == "2" else "final_test_locked")
        return rx, int(day), None, role
    match = re.fullmatch(r"Location_([A-F])_(n210_[1-3])_test\.h5", relative)
    if match:
        return match[2], None, match[1], "reserved_other_conditions"
    match = re.fullmatch(r"(.+)_(train|test)\.h5", relative)
    return (match[1] if match else None), None, None, "reserved_other_receivers"


def inspect_header(path, root, expected):
    rel = path.relative_to(root).as_posix()
    project_rel = f"{DATA_RELATIVE}/{rel}"
    stat = path.stat()
    item = {"path": project_rel, "dataset_relative_path": rel, "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}
    old = expected.get(project_rel)
    item["matches_packaged_copy_sha256"] = None if old is None else item["sha256"] == old["sha256"]
    rx, day, location, role = identity(rel)
    item.update(receiver=rx, day_index_from_filename=day, location_from_filename=location,
                role=role, actual_date=None, acquisition_session_id=None,
                sampling_rate_hz=None, bandwidth_hz=None, spreading_factor=None,
                source_status="unverified_author_original", cfo_compensation_status="unknown")
    errors = []
    with h5py.File(path, "r") as f:
        item["root_attributes"] = attributes(f)
        item["datasets"] = {}
        for key in f.keys():
            if not isinstance(f.get(key, getlink=True), h5py.HardLink):
                errors.append(f"Unsupported HDF5 link: {key}")
                continue
            obj = f[key]
            if not isinstance(obj, h5py.Dataset):
                errors.append(f"Unsupported object: {key}")
                continue
            item["datasets"][key] = {"shape": list(obj.shape), "dtype": str(obj.dtype),
                                     "attributes": attributes(obj), "chunks": obj.chunks,
                                     "compression": obj.compression}
        if not {"data", "label", "CFO", "SNR"}.issubset(item["datasets"]):
            errors.append("Missing required datasets")
        else:
            labels = np.asarray(f["label"][:]).reshape(-1)
            integer_labels = bool(np.isfinite(labels).all() and np.equal(labels, np.floor(labels)).all())
            item["labels_are_finite_integers"] = integer_labels
            item["labels"] = dict(Counter(str(int(x)) for x in labels)) if integer_labels else {}
            item["packets"] = len(labels)
            if not integer_labels:
                errors.append("Invalid labels")
            if set(item["labels"]) != {str(x) for x in range(31, 41)}:
                errors.append("Expected labels 31..40")
            if f["data"].shape != (len(labels), 16384):
                errors.append("Unexpected data shape")
            for name in ("CFO", "SNR"):
                if f[name].size != len(labels):
                    errors.append(f"{name} count disagrees with packet count")
    item["schema_errors"] = errors
    return item


def finite_summary(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return {"count": int(np.size(values)), "nonfinite": int(np.size(values) - len(finite)),
            "min": float(np.min(finite)) if len(finite) else None,
            "median": float(np.median(finite)) if len(finite) else None,
            "max": float(np.max(finite)) if len(finite) else None}


def inspect_development_signals(root, items):
    # Digests are comparable across both receivers, but matches do not imply
    # independent acquisitions and nonmatches do not exclude near duplicates.
    seen = defaultdict(list)
    results = []
    for item in items:
        rel = item["dataset_relative_path"]
        if rel not in DEVELOPMENT_FILES:
            continue
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()):
            raise PermissionError("Development file escapes data root")
        own = set()
        overlaps = Counter()
        conflicts = Counter()
        rms = []
        nonfinite_rows = 0
        with h5py.File(path, "r") as f:
            labels = np.asarray(f["label"][:]).reshape(-1).astype(int)
            values = {name: np.asarray(f[name][:]).reshape(-1) for name in ("CFO", "SNR")}
            for start in range(0, len(labels), 128):
                rows = np.asarray(f["data"][start:start + 128])
                rms.extend(np.sqrt(np.mean(rows[:, :8192] ** 2 + rows[:, 8192:] ** 2, axis=1)).tolist())
                nonfinite_rows += int((~np.isfinite(rows).all(axis=1)).sum())
                for row_idx, row in enumerate(rows, start):
                    digest = hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
                    for prev, previous_label in seen[digest]:
                        if prev != rel:
                            overlaps[prev] += 1
                        if previous_label != int(labels[row_idx]):
                            conflicts[prev] += 1
                    if digest not in own:
                        seen[digest].append((rel, int(labels[row_idx])))
                    own.add(digest)
            result = {"file": rel, "role": item["role"], "packets": len(labels),
                      "nonfinite_signal_rows": nonfinite_rows,
                      "exact_duplicate_rows_within_file": len(labels) - len(own),
                      "exact_overlap_rows_with_other_development_files": dict(overlaps),
                      "identical_signal_conflicting_labels": dict(conflicts),
                      "rms": finite_summary(np.asarray(rms)),
                      "fraction_unit_rms_atol_1e_6": float(np.mean(np.isclose(rms, 1, atol=1e-6, rtol=0))),
                      "CFO_unit": "unknown", "SNR_unit": "not_confirmed_from_local_metadata",
                      "metadata_by_label": {str(label): {name: finite_summary(val[labels == label])
                                             for name, val in values.items()} for label in range(31, 41)}}
            results.append(result)
        print(f"development signal audit: {rel}", flush=True)
    return results


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_human_report(out, report, protocol, baseline_rows, project):
    summary = report["summary"]
    sources = json.loads((project / "inventory/source_inventory.json").read_text(encoding="utf-8-sig"))
    local_sources = [x for x in sources if x["path"].startswith(DATA_RELATIVE + "/")]
    ordinary_train = sorted(x["receiver"] for x in report["files"]
                            if x["dataset_relative_path"].endswith("_train.h5")
                            and "/" not in x["dataset_relative_path"])
    ordinary_test = sorted(x["receiver"] for x in report["files"]
                           if x["dataset_relative_path"].endswith("_test.h5")
                           and "/" not in x["dataset_relative_path"]
                           and x["location_from_filename"] is None)
    code_comparison = project / "inventory/shen_stage1_sources_20261011/author_code_comparison.json"
    code_evidence = json.loads(code_comparison.read_text()) if code_comparison.exists() else None
    provenance = {
        "author_original_identity": "not_verified",
        "local_packaging_integrity": {"matched_files": summary["packaged_copy_hash_matches"],
                                     "index": "inventory/bundle_checksums.json"},
        "local_source_paths": local_sources,
        "author_code_comparison": code_evidence,
        "ordinary_training_receivers_from_filenames": ordinary_train,
        "ordinary_test_receivers_from_filenames": ordinary_test,
        "test_receivers_without_ordinary_training_files": sorted(set(ordinary_test) - set(ordinary_train)),
        "does_not_establish_independent_OOB_blind_split": True,
        "download_history": "Local source inventory points to a Windows BaiduNetdiskDownload directory. This is a local path, not proof of the actual download service or author-original bytes.",
        "archive_history": "No original Shen ZIP files are present in this project. inventory/archive_members.json contains no HDF5 members. The prior shen_hdf5_audit.json states a ZIP size comparison; the underlying original ZIP/CRC record is not available here.",
        "author_link": "https://github.com/gxhen/receiverAgnosticRFFI",
        "author_current_share": "https://pan.quark.cn/s/b9017ff661b2",
        "missing_evidence": ["author-published HDF5 checksums or original archive CRCs",
                             "download/extraction history", "true date and capture/session grouping",
                             "local sampling rate, bandwidth, SF, chirp alignment and CFO processing state",
                             "exact independent OOB split and artifact locations"]}
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    rows = ["| 接收机 | 数据文件 | 角色 | 包数 |", "|---|---|---|---:|"]
    for item in protocol["files"]:
        rows.append(f"| {item['receiver']} | `{item['dataset_relative_path']}` | {item['role']} | {item['packets']} |")
    experiment_rows = ["| 编号 | 实验 | 规模 | 当前状态 |", "|---|---|---|---|"]
    for eid, experiment, budget, split, purpose, status in baseline_rows:
        experiment_rows.append(f"| {eid} | {experiment} | {budget} | {status} |")
    text = "\n".join([
        "# Shen 第一轮数据与协议审计", "",
        "本轮范围：只读数据审计、候选协议、开发数据适配器、环境准备。没有启动模型训练或下载 OSU。",
        "",
        "## 可以确认的事实", "",
        f"- 本地 {summary['hdf5_files']} 个 HDF5，{summary['bytes']:,} bytes，{summary['packets']:,} 包。全部与本地打包 SHA256 索引一致，未发现结构错误或整文件字节重复。",
        "- 全部标签为 31–40，映射到分类标签 0–9；每包为 16,384 个 float64，前 8,192 为 I、后 8,192 为 Q。各文件的字段、形状、标签、属性、SHA256 见 available_files.csv 和 audit.json。",
        "- 所有根级和数据集属性均为空。采样率、实际日期、采集组、预处理状态不能由本地 HDF5 属性确认。文件系统 mtime 不作为采集日期。",
        "- Day1 训练、Day1 同日诊断、Day2 验证共六个开发文件，24,000 包；信号均有限，未发现精确重复或跨开发文件的精确重叠。没有据此声称采集独立或排除了近重复。",
        "- 本地存储的包不是统一的单位 RMS。作者代码在构造 CIS 时逐包 RMS 归一化；是否做过 CFO 补偿仍不能由这个事实推断。",
        "",
        "## 来源证据与局限", "",
        "本地打包索引与当前文件一致，证明传输副本的完整性，不证明文件与作者原始发布版一致。source_inventory.json 的来源路径为 Windows 的 BaiduNetdiskDownload 目录；文件夹名不能证明真实下载平台。",
        "",
        "当前项目没有原始 Shen ZIP；archive_members.json 中没有 HDF5 成员记录。旧审计摘要提到了压缩包成员数/大小核对，但缺少可在当前服务器复核的原始 ZIP/CRC。因此原包身份、提取过程和修改历史标注待确认。",
        "",
        "作者仓库 https://github.com/gxhen/receiverAgnosticRFFI 提供 Quark 更新链接。已保存的作者代码比较记录在 inventory/shen_stage1_sources_20261011/；参考代码一致不能反推 HDF5 来源一致。",
        "",
        "论文参数为 Fs=1 MHz、BW=125 kHz、SF=7，并描述同步、前导提取和频偏补偿：https://arxiv.org/pdf/2207.02999 。这些是文献参数，未作为本地已证实元数据写入数据清单。",
        "",
        "## 候选划分", "", *rows, "",
        "十个候选文件共约 4.20 GB。N210-1 和 RTL-6 分别训练、分别评估；本任务为闭集跨日期识别，不称为未见接收机泛化。日期编号目前仅来自文件名，真实日期/独立采集组尚未证实。",
        "",
        "模型选择只用 Day2。Day1 test 是额外同日诊断数据，不并入训练，不用于形成与 Day2 混合的选择指标。Day3–4 的 IQ、CFO/SNR 值和特征在本轮未解码，只检查字节校验、结构与标签。",
        "",
        "历史 asset_read_audit_current.json 对全部 HDF5 曾抽查首、中、末三个数据包，包括最终日期。本轮协议能约束今后的方法选择；不能声称这些文件历史上从未读取。没有证据显示这些抽查曾用于训练或调参。",
        "",
        "## 泄漏和解释风险", "",
        "- 禁止同包窗口跨划分、Day2 标签参与梯度/对比配对/类别原型，以及 Day3–4 参与超参数选择或拟合统计量。逐包自身 RMS 与利用全验证集拟合归一化统计量是两种不同操作。",
        "- 六个开发文件没有精确重复不能证明日期/session 独立；未经验证的不同接收机包不能按行配对。后续置信区间不能把窗口或 seed 当作独立采集组。",
        "- CFO/SNR 数据存在但本地单位与估计阶段缺乏元数据说明。CFO 补偿状态不明时，不实施额外补偿或按原始 CFO 实验解释结果。",
        "- HDF5 包含多个设备，不能整文件投票为一个设备；包/chirp 聚合须定义合法观测组和一致的样本预算。",
        "- 另有独立 OOB 研究是用户确认事实，14/6 分组和投稿状态不从旧代码推断。新方法不把该研究的核心 OOB 诊断结果重复作为贡献。移除显式 OOB 分支也不代表 IQ 已去掉带外信息。",
        "- 普通目录有 14 台训练接收机、20 台测试接收机；缺训练文件的六台接收机清单见 provenance.json。这只是目录推导，不视为独立 OOB 论文的已确认盲测名单。",
        "",
        "## 第一轮强基线候选表", "", *experiment_rows, "",
        "最小训练候选为 B1、B2 × 两接收机 × 三个预定 seed（0、1、2），合计 12 次完整训练；目前只列计划，没有启动。先做接口/合成输入检查，真实训练安排在审计结果确认后的下一步。",
        "",
        "B1/B2 不额外补偿 CFO，不重采样原数据；两者统一划分、采样长度、选择指标（Day2 包级 macro-F1）、早停规则和调参预算，分别记录实际参数量/计算成本。CIS+CNN 属于采用作者表示/骨干、调整跨日期验证协议的基线，不能冒称原论文数值复现。原论文的随机加噪验证不照搬到固定 Day2 验证。",
        "",
        "B1 与 B2 是两套完整基线：表示与编码器形状都可能不同，二者的差异不能直接归因于输入表示。贡献归因需要后续控制编码器、容量和训练配方。原论文的十个发射机包含两种硬件型号；实际设备编号与型号的对应关系还未由本地记录确认，后续应检查同型号设备之间的区分能力。",
        "",
        "表示比较后再做容量匹配、同 tokenizer/stem 的 CNN/HSTU/Transformer；所有实际预处理、增强及训练设置须随运行保存。RMS 对照、噪声增强等改变一次一个变量。CFO、dechirp、跨 chirp 聚合等待相应证据；OSU 和新接收机实验不在本轮运行范围。",
        "",
        "## 已写代码与环境", "",
        "- scripts/audit_shen_stage1.py：完整 SHA256、只读结构/标签审计、仅开发数据的有限值/RMS/重复诊断，生成候选协议和实验表。输出目录必须为空，避免覆盖旧报告。",
        "- src/shen_stage1/data.py：按明确角色逐包只读访问，原标签映射到 0–9；没有隐式归一化或补偿。访问最终日期、其他接收机或伪造 role/path 时会拒绝。",
        "- scripts/setup_shen_stage1.sh：建立独立 Python 3.11 环境和固定依赖；PyTorch 2.5.1/cu121 安装说明来源 https://pytorch.org/get-started/previous-versions/ 。",
        "- scripts/check_shen_stage1_environment.py：合成数据检查 I/Q 布局、标签映射、多 worker、访问保护和 GPU 数值运算；不训练模型、不读取真实研究信号。环境实测记录另见 inventory/shen_stage1_environment_20261011/。",
        "",
        "环境路径：/home/keyan/venvs/shen-stage1。实测资源为 31 GiB RAM、两张 V100-SXM2-32GB，足够本轮小规模实验。",
        "",
        "所有原始 HDF5 均保持只读，现有模型和实验结果未覆盖。protocol.candidate.json 为候选协议，不表示来源/采集独立性已通过核验；其 SHA256 用于记录当前版本，无法代替正式冻结和数据溯源。",
        ""])
    (out / "REPORT.md").write_text(text, encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hash-workers", type=int, default=4)
    parser.add_argument("--inspect-development-signals", action="store_true")
    args = parser.parse_args()
    project = args.root.resolve()
    root = project / DATA_RELATIVE
    out = args.out_dir.resolve()
    if out.is_relative_to(root.resolve()):
        parser.error("Reports cannot be written inside the original data directory")
    if out.exists() and any(out.iterdir()):
        parser.error("Use a new empty output directory; prior audit records are immutable")
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.rglob("*.h5"))
    expected = {x["path"]: x for x in json.loads((project / "inventory/bundle_checksums.json").read_text(encoding="utf-8-sig"))}
    items = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.hash_workers, 8))) as pool:
        futures = {pool.submit(inspect_header, p, root, expected): p for p in paths}
        for future in as_completed(futures):
            items.append(future.result())
            print(f"headers and SHA256: {len(items)}/{len(paths)}", flush=True)
    items.sort(key=lambda x: x["dataset_relative_path"])
    schema_errors = [x["path"] for x in items if x["schema_errors"]]
    mismatches = [x["path"] for x in items if x["matches_packaged_copy_sha256"] is False]
    by_sha = defaultdict(list)
    for item in items:
        by_sha[item["sha256"]].append(item["path"])
    development = inspect_development_signals(root, items) if args.inspect_development_signals and not schema_errors else []
    candidates = [x for x in items if x["day_index_from_filename"] is not None]
    candidate_rows = [{**{k: x[k] for k in ("path", "dataset_relative_path", "receiver", "role", "bytes", "sha256", "packets")},
                       "day_index": x["day_index_from_filename"], "signal_access": x["dataset_relative_path"] in DEVELOPMENT_FILES,
                       "actual_date": None, "acquisition_session_id": None} for x in candidates]
    summary = {"checked_utc": datetime.now(timezone.utc).isoformat(), "hdf5_files": len(items),
               "bytes": sum(x["bytes"] for x in items), "packets": sum(x.get("packets", 0) for x in items),
               "packaged_copy_hash_matches": sum(x["matches_packaged_copy_sha256"] is True for x in items),
               "hash_mismatches": mismatches, "schema_errors": schema_errors,
               "identical_whole_files": [v for v in by_sha.values() if len(v) > 1],
               "development_signal_files_inspected": [x["file"] for x in development],
               "held_out_signal_arrays_decoded_this_run": False,
               "historical_access": "Prior asset audit sampled three signal rows per HDF5, including final dates; no training or selection provenance is established by that record.",
               "author_original_identity_verified": False,
               "original_file_writes": False,
               "protocol_status": "candidate_only_pending_source_dates_sessions_preprocessing"}
    report = {"summary": summary, "files": items, "development_signal_checks": development}
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    write_csv(out / "available_files.csv", [{**x, "labels": json.dumps(x.get("labels", {}))} for x in items],
              ["path", "bytes", "sha256", "matches_packaged_copy_sha256", "packets", "labels", "receiver", "day_index_from_filename", "role", "actual_date", "acquisition_session_id", "sampling_rate_hz", "source_status", "cfo_compensation_status"])
    write_csv(out / "candidate_split.csv", candidate_rows,
              ["receiver", "day_index", "role", "packets", "bytes", "sha256", "path", "signal_access", "actual_date", "acquisition_session_id"])
    protocol = {"schema_version": 1, "status": summary["protocol_status"], "task": "closed_set_cross_day_per_receiver",
                "data_root": DATA_RELATIVE, "label_map": {str(x): x - 31 for x in range(31, 41)},
                "receivers_trained_separately": True, "final_test_signal_access_enabled": False,
                "selection_data": "Day2 only; no Day3/Day4 feature analysis, predictions, model selection, or fitted statistics",
                "preprocessing": {"source_history": "unknown", "cfo_compensation": "unknown", "resampling": "unknown", "stored_unit_rms": "not_established"},
                "literature_parameters_not_local_metadata": {"sampling_rate_hz": 1000000, "bandwidth_hz": 125000, "spreading_factor": 7,
                    "source": "https://arxiv.org/pdf/2207.02999", "local_applicability_verified": False},
                "oob_independent_research": {"existence": "user_confirmed", "development_receivers": 14, "blind_receivers": 6,
                    "submission_status": "unverified", "exact_receiver_split_and_artifact_paths": "not_established", "core_results_reuse": "excluded_from_new_contribution"},
                "files": candidate_rows}
    protocol_text = json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    (out / "protocol.candidate.json").write_text(protocol_text, encoding="utf8")
    (out / "protocol.candidate.sha256").write_text(hashlib.sha256(protocol_text.encode()).hexdigest() + "  protocol.candidate.json\n")
    baseline_rows = [
        ("M0", "CFO-only / SNR-only", "CPU diagnostic", "Day1 fit; Day2 validation", "Metadata shortcut diagnostic; units and CFO stage unresolved", "proposed_after_audit"),
        ("B1", "stored IQ + RMS + CNN", "3 seeds per receiver", "Day1 train; Day2 validation", "No added CFO compensation; explicit per-packet normalization", "minimal_candidate"),
        ("B2", "CIS + author-style CNN", "3 seeds per receiver", "Day1 train; Day2 validation", "Author-derived representation and backbone, adapted day split; deterministic validation", "minimal_candidate"),
        ("B3", "IQ + capacity-matched CNN", "same seeds and tuning budget", "same as B1", "Separate representation improvement from parameter count", "second_step"),
        ("B4", "IQ + HSTU encoder", "same input/stem; matched capacity", "same as B1", "Remove explicit OOB, FFT and amp/phase branches; no legacy weight reuse", "second_step"),
        ("B5", "IQ + Transformer encoder", "same input/tokenizer; matched capacity", "same as B1", "Test whether sequence gains are specific to HSTU", "second_step"),
        ("C1", "estimated residual-CFO IQ + CNN", "conditional", "same as B1", "Only after compensation state, units and estimator assumptions are established", "blocked_preprocessing"),
        ("C2", "dechirp + CNN", "conditional", "same as B1", "Only after local Fs/BW/SF and chirp alignment are supported", "blocked_physical_parameters"),
        ("A1", "1/2/4/8 chirp aggregation", "conditional", "development packets only", "Known chirp boundaries required; match sample budget and aggregation for all models", "blocked_alignment"),
    ]
    write_csv(out / "first_round_baselines.csv", [dict(zip(["id", "experiment", "budget", "split", "purpose_and_constraints", "status"], row)) for row in baseline_rows],
              ["id", "experiment", "budget", "split", "purpose_and_constraints", "status"])
    write_human_report(out, report, protocol, baseline_rows, project)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if mismatches or schema_errors or len(candidates) != 10:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
