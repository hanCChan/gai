"""Frozen IQ backbone and bounded distribution-level receiver mappings."""
import hashlib
import json
import numpy as np
import torch
from torch import nn

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

class IQCNN(nn.Module):
    def __init__(self, classes=8):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(2,16,17,stride=8,padding=8),nn.ReLU(),
            nn.Conv1d(16,32,9,stride=4,padding=4),nn.ReLU(),
            nn.Conv1d(32,64,9,stride=4,padding=4),nn.ReLU(),
            nn.Conv1d(64,64,5,stride=2,padding=2),nn.ReLU(),
            nn.AvgPool1d(2))  # Length32 ->16, exactly the registered equal-bin average.
        self.feature = nn.Sequential(nn.Flatten(),nn.Linear(1024,32),nn.ReLU())
        self.head = nn.Linear(32,classes)
    def encode(self, x):
        return self.feature(self.convs(x))
    def forward(self, x):
        return self.head(self.encode(x))

def power(matrix, exponent, floor=1e-12):
    values, vectors = np.linalg.eigh((matrix+matrix.T)/2)
    return (vectors*np.maximum(values,floor)**exponent)@vectors.T

def covariance(x, shrink=0.):
    c = np.cov(x, rowvar=False)
    return (1-shrink)*c + shrink*np.trace(c)/c.shape[0]*np.eye(c.shape[0])

def mappings(source, target, rank=2, ridge=.1, shrink=.1):
    """Lists of eight class distributions; NEVER pair individual packets."""
    assert source.shape == target.shape and source.shape[0] == 8
    dimension = source.shape[-1]
    x, y = source.mean(1), target.mean(1)
    mx, my = x.mean(0), y.mean(0)
    xc, residual = x-mx, (y-x)-(my-mx)
    regularizer = ridge*np.sum(xc**2)/dimension
    delta = np.linalg.solve(xc.T@xc+max(regularizer,1e-12)*np.eye(dimension), xc.T@residual)
    u, s, v = np.linalg.svd(delta,full_matrices=False)
    delta = (u[:,:rank]*s[:rank])@v[:rank]
    a = np.eye(dimension)+delta
    ca = covariance(source.reshape(-1,dimension),shrink)
    cb = covariance(target.reshape(-1,dimension),shrink)
    coral = power(ca,-.5)@power(cb,.5)
    return {'identity':(np.eye(dimension),np.zeros(dimension)),
            'translation':(np.eye(dimension),my-mx),
            'coral':(coral,my-mx@coral),
            'lowrank':(a,my-mx@a)}

def distances(source, target):
    mean_sq = float(np.sum((source.mean(0)-target.mean(0))**2))
    ca, cb = covariance(source), covariance(target)
    root = power(cb,.5,0.)
    eig = np.linalg.eigvalsh((root@ca@root + root@ca.T@root)/2)
    w2 = mean_sq+float(np.trace(ca)+np.trace(cb)-2*np.sqrt(np.maximum(eig,0)).sum())
    return mean_sq, max(mean_sq,w2)
