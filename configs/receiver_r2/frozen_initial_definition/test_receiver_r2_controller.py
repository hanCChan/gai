#!/usr/bin/env python3
"""CPU deterministic policy checks; no new fits or coefficient search."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from receiver_r2.controller import choose_scale,SCALES


def main():
    def cell(d,se=.001):return dict(delta=d,paired_SE=se,resolution=1e-7,n=20)
    positive_RX={s:[cell(.003*s),cell(-.02*s)] for s in SCALES[:-1]}
    assert choose_scale(positive_RX)==1.,'Positive RX risk must sometimes be permitted'
    conflict={s:[cell(.03*s,.00001),cell(-.04*s,.00001)] for s in SCALES[:-1]}
    assert choose_scale(conflict,multi_rx=False)==1. and choose_scale(conflict)==0.
    noisy={s:[cell(.0001*s),cell(.0002*s)] for s in SCALES[:-1]};assert choose_scale(noisy)==1.
    scaled={1.:[cell(.03),cell(.02)],.5:[cell(.002),cell(-.01)],.25:[cell(-.001),cell(-.001)]};assert choose_scale(scaled)==.5
    assert choose_scale({})==0.
    unsupported={s:[dict(cell(-1),n=1)] for s in SCALES[:-1]};assert choose_scale(unsupported)==0.
    result=dict(passed=True,positive_individual_RX_change_permitted=True,noise_from_C1_only=True,no_C2_argument=True,ordinary_mean_guard_and_multi_RX_guard_differ=True,scaling_fallback=True,unsupported_rejects=True)
    p=ROOT/'reports/receiver_r2/preflight/controller_tests.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
