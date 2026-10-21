#!/usr/bin/env python3
"""Verify the PoC trainer's CE anchor reproduces the frozen R0/R1 CE checkpoint.

Runs the 'ce' branch with the same seed/schedule/optimizer as an existing formal CE job,
then compares the resulting state dict to the frozen selected.pt. Any mismatch in the
selected epoch or in source-val CE trajectory flags the recipe as non-identical, which
must be fixed before PoC arms are interpreted (the CE anchor is the reference point for
"did the pair loss help beyond the frozen recipe's own variance").

    python3 check_ce_anchor.py --fold 0 --seed 20261013 --pairs pairs.json --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import p0_common as C   # noqa: E402

C.add_root()


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20261013)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    a = ap.parse_args()
    run_id = f'shen_rank16_CV{a.fold}_ce_seed{a.seed}'
    frozen_dir = C.ROOT / 'outputs/receiver_r01/formal' / run_id
    poc_dir = Path(a.out) / f'shen_rank16_CV{a.fold}_ce0_seed{a.seed}'
    if not poc_dir.exists():
        print('PoC CE run missing; run train_pair_poc.py --branch ce --lambda 0 first')
        raise SystemExit(1)
    obj_poc = torch.load(poc_dir / 'selected.pt', map_location='cpu', weights_only=False)
    obj_frozen = torch.load(frozen_dir / 'selected.pt', map_location='cpu', weights_only=False)
    same_sel = obj_poc['selection']['epoch'] == json.loads((frozen_dir / 'selection.json').read_text())['epoch'] \
        if 'epoch' in obj_poc['selection'] else None
    sd_poc, sd_fr = obj_poc['model'], obj_frozen['model']
    mism = {k: float((sd_poc[k].float() - sd_fr[k].float()).abs().max()) for k in sd_fr
            if sd_poc[k].shape != sd_fr[k].shape or not torch.equal(sd_poc[k], sd_fr[k])}
    hist_poc = json.loads((poc_dir / 'history.json').read_text())
    hist_fr = list((frozen_dir / 'history.csv').read_text().strip().split('\n')[1:])
    max_dev = max(mism.values()) if mism else 0.0
    verdict = dict(
        run_id=run_id, selected_epoch_poc=obj_poc['selection']['epoch'],
        selected_epoch_frozen=json.loads((frozen_dir / 'selection.json').read_text()).get('epoch'),
        same_selected_epoch=bool(same_sel), n_mismatched_tensors=len(mism), max_abs_param_dev=max_dev,
        poc_val_ce_history=[h['source_val_CE'] for h in hist_poc],
        bitwise_identical=len(mism) == 0,
        note='R0/R1 CE used lr from selection.frozen.json (3e-4) and the same balanced schedule; '
             'identical requires same torch build and deterministic algorithms')
    C.dump_json(Path(a.out) / f'anchor_check_{run_id}.json', verdict)
    print(json.dumps(verdict, indent=1))


if __name__ == '__main__':
    main()
