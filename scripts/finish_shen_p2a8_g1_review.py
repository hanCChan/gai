#!/usr/bin/env python3
"""Latest G1-Joint report: unchanged 24 OSU runs, complete failure accounting, conditional G2."""
import datetime,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a.cache import write_json

if __name__=='__main__':
    out=ROOT/'outputs/shen_p2a8/g1_latest_review_status.json'
    state=lambda status,**kw:write_json(out,dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
    def run(args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    try:
        state('waiting_for_frozen_OSU_24',OSU_training_configuration_changed=False,G2_0_started=False,G3_started=False)
        while True:
            orch=ROOT/'outputs/shen_p2a8/g1_osu_orchestration.json'
            status=json.loads(orch.read_text())
            if status['status']=='failed':raise RuntimeError('Frozen OSU pipeline failed; preserve it and inspect instead of changing target-driven rules: '+str(status))
            pipeline=ROOT/'outputs/shen_p2a8/g1_osu_v1/pipeline_status.json'
            if pipeline.exists():
                p=json.loads(pipeline.read_text())
                if p['status']=='failed':raise RuntimeError('Frozen OSU training/reporting failed: '+str(p))
                if p['status']=='completed':break
            time.sleep(10)
        state('auditing_OSU_saved_results',G3_started=False)
        run(['scripts/audit_shen_p2a8_g1_review.py','--dataset','osu','--pipeline','outputs/shen_p2a8/g1_osu_v1','--output','outputs/shen_p2a8/g1_osu_review_v1'])
        state('joint_reporting',G3_started=False)
        run(['scripts/report_shen_p2a8_g1_joint_review.py','--shen-review','outputs/shen_p2a8/g1_shen_review_v1','--osu-review','outputs/shen_p2a8/g1_osu_review_v1','--osu-pipeline','outputs/shen_p2a8/g1_osu_v1','--output','outputs/shen_p2a8/g1_joint_review_v2'])
        decision=json.loads((ROOT/'outputs/shen_p2a8/g1_joint_review_v2/decision.json').read_text())
        state('G1_joint_review_completed',decision=decision,G2_0_started=False,G3_started=False)
    except BaseException as e:
        state('failed',error=repr(e),OSU_training_configuration_changed=False,G2_0_started=False,G3_started=False);raise
