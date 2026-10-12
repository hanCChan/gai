"""Verify expensive-job orchestration without launching GPUs or touching data."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('finish_shen_p1',ROOT/'scripts/finish_shen_p1.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class PipelineTests(unittest.TestCase):
    def test_recipe_freezes_before_new_seeds_and_only_p1_is_launched(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for folder,lr in [('p1_seed20261012',.001),('p1_lr0003',.0003)]:
                base=root/'outputs/shen_evidence'/folder; base.mkdir(parents=True)
                (base/'launcher_results.json').write_text(json.dumps([[{'returncode':0}]]))
                for exp,kind in zip(['E01','E02','E03','E04'],['full_iq','strict_ib','oob_only','cis']):
                    for rx in ['n210_1','rtl_6']:
                        run=base/f'{exp}_{rx}'; run.mkdir()
                        # A different recipe wins for IB; the high-LR one wins elsewhere.
                        score=.9 if (kind=='strict_ib')==(lr==.0003) else .6
                        (run/'metrics.json').write_text(json.dumps({'kind':kind,'receiver':rx,'lr':lr,'macro_f1':score,'packet_accuracy':score}))
                        (run/'config.json').write_text(json.dumps({'kind':kind,'receiver':rx,'lr':lr,'seed':20261012,
                                                                'experiment_id':exp,'source_commit':'old-screening-commit'}))
            output=root/'final'; commands=[]
            def record_call(command,**kwargs):
                commands.append(command)
                return None
            with patch.object(module,'ROOT',root),patch.object(module.subprocess,'run',side_effect=record_call), \
                 patch.object(module.sys,'argv',['finish_shen_p1.py','--output',str(output)]):
                module.main()
            selection=json.loads((output/'selection.json').read_text())
            self.assertEqual(len(selection['choices']),8)
            for row in selection['choices']:
                self.assertEqual(row['lr'],.0003 if row['kind']=='strict_ib' else .001)
            matrix=json.loads((output/'confirmation_matrix.json').read_text())
            self.assertEqual(len(matrix),16)
            for job in matrix:
                self.assertEqual(job['stage'],'P1')
                cfg=json.loads(Path(job['config']).read_text())
                self.assertIn(cfg['seed'],[20261013,20261014])
                self.assertEqual(cfg['lr'],.0003 if cfg['kind']=='strict_ib' else .001)
                self.assertNotIn('source_commit',cfg)
            self.assertEqual(commands[0][commands[0].index('--stage')+1],'p1')
            self.assertIn('--selection',commands[1])
            self.assertEqual(json.loads((output/'pipeline_status.json').read_text())['stage'],'completed')

if __name__=='__main__': unittest.main()
