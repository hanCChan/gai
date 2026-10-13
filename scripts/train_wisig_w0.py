#!/usr/bin/env python3
"""One official-architecture pilot, source-only checkpoint choice."""
import argparse,csv,json,os,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from wisig_w01_common import OfficialCNN,dump,sha
ROOT=Path(__file__).resolve().parents[1]
def table(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def load(reg,out,rxs,days,positions,txs):
 xs=[];ys=[];gs=[]
 for ri in rxs:
  assert ri not in reg['roles']['final_rx']
  for di in days:
   assert di!=reg['final_date']
   with np.load(out/'cache'/f'rx{ri:02d}_day{di}.npz',allow_pickle=False) as a:
    for tx in txs:
     mask=(a['y']==tx)&(a['position']>=positions[0])&(a['position']<positions[1])
     x=a['x'][mask];xs.append(x);ys.append(a['y'][mask]);gs.append(np.full(len(x),len(gs)))
 return np.concatenate(xs),np.concatenate(ys),np.concatenate(gs)
@torch.no_grad()
def scores(model,x,y,device):
 model.eval();correct=0;loss=0.;cm=np.zeros((8,8),int)
 for begin in range(0,len(y),512):
  z=model(torch.from_numpy(x[begin:begin+512]).to(device));target=torch.from_numpy(y[begin:begin+512]).to(device)
  loss+=float(F.cross_entropy(z,target,reduction='sum'));pred=z.argmax(1).cpu().numpy()
  np.add.at(cm,(y[begin:begin+512],pred),1);correct+=int((pred==y[begin:begin+512]).sum())
 rec=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(8),where=cm.sum(1)>0)
 f1=np.divide(2*cm.diagonal(),cm.sum(1)+cm.sum(0),out=np.zeros(8),where=(cm.sum(1)+cm.sum(0))>0)
 return dict(accuracy=correct/len(y),balanced_accuracy=float(rec.mean()),macro_f1=float(f1.mean()),CE=loss/len(y),packets=len(y),class_recall=rec.tolist(),confusion=cm.tolist())
def main(index):
 started=time.perf_counter();out=ROOT/'outputs/wisig_w01';reg=json.loads((ROOT/'configs/wisig_w01/registration.json').read_text());seed=reg['pilot_seeds'][index]
 run=out/'runs'/f'pilot_seed{seed}';run.mkdir(parents=True,exist_ok=False);steps=0
 dump(run/'status.json',dict(state='running',optimizer_steps=0,pid=os.getpid()))
 try:
  seal=json.loads((out/'pre_execution_seal.json').read_text())
  for path,h in seal['files_sha256'].items():assert sha(ROOT/path)==h,path
  pre=json.loads((out/'preflight.json').read_text());assert sha(out/'preflight.json')==seal['preflight_sha256']
  for a in pre['assets']:assert sha(a['path'])==a['sha256']
  torch.set_num_threads(2);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
  device=torch.device('cuda:0');model=OfficialCNN().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.0005,eps=1e-7)
  x,y,g=load(reg,out,reg['roles']['source'],[0,1],[0,120],list(range(8)))
  xv,yv,_=load(reg,out,reg['roles']['source'],[0,1],[120,160],list(range(8)))
  groups=len(reg['roles']['source'])*2*8
  assert len(x)==groups*120 and len(xv)==groups*40 and len(np.unique(g))==groups
  assert np.array_equal(np.bincount(g),np.full(groups,120)) and not np.any(y>=8)
  init=hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
  dump(run/'config.json',dict(seed=seed,classes=8,parameters=sum(p.numel() for p in model.parameters()),feature_dimension=80,
   source_rx=reg['roles']['source'],train_dates=[0,1],held_tx=[8,9],final_rx_read=False,final_date_read=False,
   architecture='official d003 PyTorch port;NHWC flatten and asymmetric SAME padding',lr=.0005,Adam_epsilon=1e-7,initial_sha256=init,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,source_only_checkpoint_selection=True))
  xx=torch.from_numpy(x).to(device);yy=torch.from_numpy(y).to(device);rng=np.random.default_rng(seed);history=[];best=float('inf');wait=0;best_epoch=0
  for epoch in range(100):
   model.train();loss_sum=0.;correct=0
   schedule=np.stack([rng.permutation(np.flatnonzero(g==group)) for group in range(groups)]).reshape(groups,60,2)
   assert np.array_equal(np.sort(schedule.reshape(-1)),np.arange(groups*120))
   for k in range(60):
    ids=torch.from_numpy(schedule[rng.permutation(groups),k].reshape(-1)).to(device)
    optimizer.zero_grad(set_to_none=True);z=model(xx[ids]);loss=F.cross_entropy(z,yy[ids])+model.regularization()
    assert torch.isfinite(loss);loss.backward();optimizer.step();steps+=1
    loss_sum+=float(loss.detach())*len(ids);correct+=int((z.argmax(1)==yy[ids]).sum())
   val=scores(model,xv,yv,device);vl=val['CE']+float(model.regularization().detach())
   row=dict(epoch=epoch+1,train_loss=loss_sum/len(y),train_accuracy=correct/len(y),source_val_loss=vl,source_val_accuracy=val['accuracy'],optimizer_steps=steps,seconds=time.perf_counter()-started)
   history.append(row);table(run/'history.csv',history);print(json.dumps(row),flush=True)
   if vl<best:
    best=vl;wait=0;best_epoch=epoch+1;torch.save(model.state_dict(),run/'best.pt')
   else:wait+=1
   dump(run/'status.json',dict(state='running',**row,best_epoch=best_epoch))
   if wait>=5:break
  model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True));model.eval()
  train_eval=scores(model,x,y,device);validation=scores(model,xv,yv,device)
  eval_rows=[];features_dir=run/'features';features_dir.mkdir()
  for role in ['source','internal_rx']:
   for ri in reg['roles'][role]:
    for di in [0,1,2]:
     with np.load(out/'cache'/f'rx{ri:02d}_day{di}.npz',allow_pickle=False) as a:
      mask=(a['y']<8)&((a['position']>=160) if role=='source' and di<2 else np.ones(len(a['y']),bool))
      metrics=scores(model,a['x'][mask],a['y'][mask],device);eval_rows.append(dict(role=role,rx_index=ri,date_index=di,**{k:metrics[k] for k in ['accuracy','balanced_accuracy','macro_f1','packets']}))
      fm=(a['position']>=120) if role=='source' and di<2 else np.ones(len(a['y']),bool)
      values=[]
      with torch.no_grad():
       selected=a['x'][fm]
       for b in range(0,len(selected),512):values.append(model.encode(torch.from_numpy(selected[b:b+512]).to(device)).cpu().numpy())
      np.savez(features_dir/f'rx{ri:02d}_day{di}.npz',features=np.concatenate(values),y=a['y'][fm],position=a['position'][fm])
  table(run/'evaluation.csv',eval_rows)
  quality=train_eval['accuracy']>=.9 and validation['accuracy']>=.85
  dump(run/'metrics.json',dict(train_eval=train_eval,source_validation=validation,encoder_quality_passed=quality,all_evaluation=eval_rows,final_rx_evaluated=False,final_date_evaluated=False))
  dump(run/'status.json',dict(state='completed',seed=seed,epochs=len(history),best_epoch=best_epoch,optimizer_steps=steps,seconds=time.perf_counter()-started,checkpoint_sha256=sha(run/'best.pt'),encoder_quality_passed=quality,
   feature_assets=[dict(path=str(p.relative_to(run)),sha256=sha(p)) for p in sorted(features_dir.glob('*.npz'))]))
 except BaseException as error:
  dump(run/'status.json',dict(state='failed',error=repr(error),optimizer_steps=steps,seconds=time.perf_counter()-started));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,choices=[0,1],required=True);main(p.parse_args().index)
