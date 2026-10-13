#!/usr/bin/env python3
"""Numerical algorithm checks with synthetic tensors only; no fitting data."""
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from wisig_w01_common import dump
from receiver_r01.core import mldg_loss,pcgrad_backward,balanced_schedule
from receiver_r01.third_party.swa_utils import AveragedModel
from receiver_r01.third_party.swad import LossValley
from train_receiver_r01 import build


def main():
    torch.set_num_threads(2);torch.manual_seed(17);checks={};tick=time.perf_counter()
    # Central finite differences verify the Hessian term is included in MLDG.
    model=nn.Linear(2,2).double();x=torch.randn(8,2,dtype=torch.float64);y=torch.arange(8)%2
    a=torch.arange(8)<4;b=~a
    obj=mldg_loss(model,x,y,a,b,.2,1.);obj.backward();analytic=model.weight.grad[0,0].item()
    original=model.weight[0,0].item();eps=1e-5;vals=[]
    for delta in (eps,-eps):
        with torch.no_grad():model.weight[0,0]=original+delta
        vals.append(float(mldg_loss(model,x,y,a,b,.2,1.)))
    with torch.no_grad():model.weight[0,0]=original
    fd=(vals[0]-vals[1])/(2*eps);assert abs(analytic-fd)<1e-7
    checks['MLDG_full_second_order_finite_difference']=dict(analytic=analytic,finite_difference=fd)
    # Two exactly opposing scalar-task gradients must cancel under PCGrad.
    scalar=nn.Linear(1,2,bias=False).double()
    with torch.no_grad():scalar.weight.zero_()
    xx=torch.ones(2,1,dtype=torch.float64);yy=torch.tensor([0,1]);mask=torch.tensor([True,False])
    pcgrad_backward(scalar,xx,yy,mask,~mask);assert scalar.weight.grad.abs().max()<1e-12
    checks['PCGrad_opposing_tasks_cancel']=True
    # SWAD official LossValley: establish a valley then stop averaging after its death.
    tiny=nn.Linear(1,1);valley=LossValley(SimpleNamespace(logger=SimpleNamespace(error=lambda s:None)),3,6,.3)
    for step,loss in enumerate([1.,.8,.7,.72,.73,.75,2.,2.,2.,2.,2.,2.,2.],1):
        with torch.no_grad():tiny.weight.fill_(step);tiny.bias.zero_()
        segment=AveragedModel(tiny);segment.update_parameters(tiny,step=step)
        valley.update_and_evaluate(segment,0.,loss,lambda *a:None)
    assert valley.is_converged and valley.dead_valley
    checks['SWAD_official_valley']=dict(converged=True,dead=True,threshold=valley.threshold,end_step=valley.final_model.end_step)
    group=np.repeat(np.arange(6),12);schedule=balanced_schedule(group,3,21)
    for epoch in schedule:
        assert np.array_equal(np.sort(epoch.reshape(-1)),np.arange(72))
        for batch in epoch:assert np.array_equal(np.bincount(group[batch]),np.full(6,2))
    checks['Balanced_sampler_all_packets_once_and_two_per_cell']=True
    # Actual architecture synthetic CUDA optimizer steps, including full second derivatives.
    torch.use_deterministic_algorithms(True);gpu_start=time.perf_counter();architectures=[]
    for dataset,backbone,shape in [('wisig','official',(40,1,256,2)),('wisig','resnet',(40,1,256,2)),('shen','rank16',(40,1,52,126))]:
        m=build(dict(dataset=dataset,backbone=backbone)).cuda();inp=torch.randn(shape,device='cuda');target=torch.arange(40,device='cuda')%10
        ma=torch.arange(40,device='cuda')<20;opt=torch.optim.Adam(m.parameters(),lr=.0001)
        loss=mldg_loss(m,inp,target,ma,~ma);assert torch.isfinite(loss);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);opt.step()
        opt.zero_grad();value=pcgrad_backward(m,inp,target,ma,~ma);assert torch.isfinite(value)
        torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);opt.step()
        architectures.append(dict(dataset=dataset,backbone=backbone,parameters=sum(p.numel() for p in m.parameters()),synthetic_steps=2))
    torch.cuda.synchronize()
    checks['Architectures_cuda_second_order_and_PCGrad']=architectures
    result=dict(passed=True,checks=checks,seconds=time.perf_counter()-tick,reserved_gpu_seconds=time.perf_counter()-gpu_start,
                no_research_fitting=True,no_dataset_loaded=True)
    dump(ROOT/'reports/receiver_r01/2026-10-13/numerical_preflight.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
