"""Audited buffer reader and PyTorch port of the official ManyRx CNN."""
import hashlib,json,pickle,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from inspect_wisig_metadata import MetadataArray,MetadataDtype

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

class BufferArray(MetadataArray):
 def __setstate__(self,state):
  super().__setstate__(state);self.buffer=state[4];self.byteorder=state[2].state[1]
 def numeric(self):
  dtype=np.dtype(self.dtype).newbyteorder(self.byteorder)
  assert dtype.kind in 'fic' and not dtype.hasobject
  return np.frombuffer(self.buffer,dtype=dtype).reshape(self.shape)
def reconstruct(kind,shape,code):
 if kind is not BufferArray or shape!=(0,) or code!=b'b':raise ValueError('Unexpected array constructor')
 return BufferArray()
class BufferReader(pickle.Unpickler):
 def find_class(self,module,name):
  if module in ('numpy.core.multiarray','numpy._core.multiarray') and name=='_reconstruct':return reconstruct
  if module=='numpy' and name=='ndarray':return BufferArray
  if module=='numpy' and name=='dtype':return MetadataDtype
  raise pickle.UnpicklingError('Forbidden global: '+module+'.'+name)
 def persistent_load(self,value):raise pickle.UnpicklingError('Persistent IDs forbidden')
def load_archive(path):
 with zipfile.ZipFile(path) as z:
  assert z.namelist()==['ManyRx.pkl']
  with z.open('ManyRx.pkl') as f:root=BufferReader(f).load();assert f.read(1)==b''
 assert set(root)=={'tx_list','rx_list','capture_date_list','equalized_list','max_sig','data'}
 return root
class PlainReader(pickle.Unpickler):
 def find_class(self,m,n):raise pickle.UnpicklingError('Metadata global forbidden')

class SameConv(nn.Module):
 def __init__(self,cin,cout,width):
  super().__init__();self.width=width;self.conv=nn.Conv2d(cin,cout,(3,width))
 def forward(self,x):return F.relu(self.conv(F.pad(x,(0,self.width-1,1,1))))
class OfficialCNN(nn.Module):
 def __init__(self,classes=8):
  super().__init__()
  self.body=nn.Sequential(SameConv(1,8,2),nn.MaxPool2d((2,1)),
   SameConv(8,16,2),nn.MaxPool2d((2,1)),SameConv(16,16,2),nn.MaxPool2d((2,2)),
   SameConv(16,32,1),nn.MaxPool2d((2,1)),SameConv(32,16,1))
  self.fc1=nn.Linear(256,100);self.fc2=nn.Linear(100,80);self.drop=nn.Dropout(.5);self.head=nn.Linear(80,classes)
  for layer in self.modules():
   if isinstance(layer,(nn.Conv2d,nn.Linear)):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
 def encode(self,x):
  x=self.body(x).permute(0,2,3,1).contiguous().flatten(1)
  return F.relu(self.fc2(F.relu(self.fc1(x))))
 def forward(self,x):return self.head(self.drop(self.encode(x)))
 def regularization(self):return .0001*sum(layer.weight.square().sum() for layer in [self.fc1,self.fc2,self.head])
