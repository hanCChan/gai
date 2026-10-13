#!/usr/bin/env python3
"""One strict eight-TX, five-RX fixed-epoch encoder; no outer RX access."""
import argparse
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from receiver_transform_common import IQCNN, dump, sha
ROOT=Path(__file__).resolve().parents[1]

def main(index):
    reg=json.loads((ROOT/'configs/receiver_transform/registration.json').read_text())
    assert 0<=index<20
    fold=index//10;partition=(index%10)//2;seed=reg['seeds'][index%2]
    held=[x-31 for x in reg['held_tx_partitions'][partition]]
    fitted=[x for x in range(10) if x not in held]
    rxs=reg['folds'][fold]['fitting_rx'];assert len(rxs)==5
    out=ROOT/'outputs/receiver_transform';run=out/'runs'/f'gate_CV{fold}_TX{partition}_seed{seed}'
    run.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter();steps=0
    dump(run/'status.json',dict(state='running',optimizer_steps=0,pid=os.getpid()))
    try:
        seal=json.loads((out/'pre_execution_seal.json').read_text())
        for path,digest in seal['files_sha256'].items():assert sha(ROOT/path)==digest,path
        assert sha(out/'preflight.json')==seal['preflight_sha256']
        pre=json.loads((out/'preflight.json').read_text())
        torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
        torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        arrays=[]
        for rx in rxs:
            assert rx not in reg['folds'][fold]['outer_rx']
            entry=next(x for x in pre['inputs'] if x['receiver']==rx)
            path=Path(entry['cache_path']);assert sha(path)==entry['cache_sha256']
            arrays.append(np.load(path,mmap_mode='r',allow_pickle=False))
        x=np.concatenate([a[fitted,:120].reshape(-1,2,8192) for a in arrays])
        y=np.tile(np.repeat(np.arange(8),120),5)
        group=np.repeat(np.arange(40),120)
        assert len(x)==4800 and len(y)==len(x)
        device=torch.device('cuda:0');model=IQCNN().to(device)
        import hashlib
        initial=hashlib.sha256(b''.join(p.cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
        optimizer=torch.optim.Adam(model.parameters(),lr=reg['lr'])
        rng=np.random.default_rng(seed);history=[];batch_hash=hashlib.sha256()
        dump(run/'config.json',dict(index=index,fold=fold,partition=partition,seed=seed,
            fitted_tx_ids=[x+31 for x in fitted],held_tx_ids=[x+31 for x in held],
            source_rx_opened=rxs,outer_rx_opened=[],encoder_packet_positions=[0,120],
            initial_payload_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),
            torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),
            cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),lr=reg['lr'],epochs=20,
            fitting_held_tx_access=False))
        xx=torch.from_numpy(x).to(device);yy=torch.from_numpy(y).to(device)
        for epoch in range(20):
            model.train();correct=0;total=0.
            schedule=np.stack([rng.permutation(np.flatnonzero(group==g)) for g in range(40)]).reshape(40,60,2)
            np.testing.assert_array_equal(np.sort(schedule.reshape(-1)),np.arange(4800))
            for k in range(60):
                indices=schedule[rng.permutation(40),k,:].reshape(-1)
                assert np.array_equal(np.bincount(group[indices],minlength=40),np.full(40,2))
                batch_hash.update(indices.tobytes());ids=torch.from_numpy(indices).to(device)
                optimizer.zero_grad(set_to_none=True);logits=model(xx[ids]);loss=F.cross_entropy(logits,yy[ids])
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step();steps+=1;total+=float(loss.detach())*80
                correct+=int((logits.detach().argmax(1)==yy[ids]).sum())
            row=dict(epoch=epoch+1,loss=total/4800,train_accuracy=correct/4800,optimizer_steps=steps)
            history.append(row)
            with (run/'history.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(history)
            dump(run/'status.json',dict(state='running',**row,seconds=time.perf_counter()-begin))
            print(json.dumps(dict(run=run.name,**row)),flush=True)
        assert steps==1200
        model.eval();features=np.empty((5,10,80,32),np.float32)
        with torch.no_grad():
            for ri,a in enumerate(arrays):
                for tx in range(10):
                    value=torch.from_numpy(np.array(a[tx,120:200],copy=True)).to(device)
                    features[ri,tx]=model.encode(value).cpu().numpy()
        assert np.isfinite(features).all()
        np.save(run/'features.npy',features,allow_pickle=False)
        torch.save(dict(model=model.state_dict(),fitted_tx_ids=fitted,held_tx_ids=held),run/'final.pt')
        dump(run/'status.json',dict(state='completed',optimizer_steps=steps,epochs=20,
            final_train_accuracy=history[-1]['train_accuracy'],seconds=time.perf_counter()-begin,
            checkpoint_sha256=sha(run/'final.pt'),features_sha256=sha(run/'features.npy'),
            schedule_sha256=batch_hash.hexdigest(),held_tx_used_only_after_encoder_frozen=True))
    except BaseException as e:
        dump(run/'status.json',dict(state='failed',error=repr(e),optimizer_steps=steps,seconds=time.perf_counter()-begin))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True)
    main(parser.parse_args().index)
