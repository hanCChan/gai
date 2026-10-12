"""CPU G2-0 diagnostic primitives; fixed features and strictly source-fitted models."""
import numpy as np
from scipy.stats import spearmanr

def fixed_cis_moments(maps):
    x=np.asarray(maps)
    if x.ndim!=4 or x.shape[1:]!=(1,52,126):raise ValueError('Expected frozen CIS geometry')
    return np.concatenate((x[:,0].mean(-1,dtype='float64'),x[:,0].std(-1,dtype='float64')),axis=1)

def band_columns(band):
    if band not in range(4):raise ValueError('Four equal 13-row bands only')
    rows=np.arange(band*13,(band+1)*13)
    return np.concatenate((rows,rows+52))

def fit_scaler(x):
    x=np.asarray(x,dtype='float64');mean=x.mean(0);std=x.std(0)
    return {'mean':mean,'std':np.where(std<1e-12,1.,std),'constant_columns':std<1e-12}

def scale(x,scaler):return (x-scaler['mean'])/scaler['std']

def fit_ridge(x,y,classes=10,alpha=1.):
    x=np.asarray(x,dtype='float64');y=np.asarray(y,dtype='int64')
    if len(x)!=len(y) or set(y)!=set(range(classes)):raise ValueError('Aligned complete source label set required')
    design=np.column_stack((x,np.ones(len(x))));target=np.eye(classes)[y]
    penalty=np.eye(design.shape[1])*alpha;penalty[-1,-1]=0.
    return np.linalg.solve(design.T@design+penalty,design.T@target)

def ridge_logits(x,weights):return np.column_stack((x,np.ones(len(x))))@weights

def source_statistics(x,y,domains,classes=10,epsilon=1e-8):
    x=np.asarray(x,dtype='float64');y=np.asarray(y);domains=np.asarray(domains)
    unique=np.unique(domains)
    if len(unique)<2:raise ValueError('One source RX cannot define cross-RX variance')
    if len(x)!=len(y) or len(y)!=len(domains):raise ValueError('Aligned source metadata required')
    centres=np.stack([x[y==c].mean(0) for c in range(classes)])
    rx_centres=[]
    for r in unique:
        if any(not np.any((domains==r)&(y==c)) for c in range(classes)):raise ValueError('Source TX-RX coverage incomplete')
        rx_centres.append(np.stack([x[(domains==r)&(y==c)].mean(0) for c in range(classes)]))
    rx_centres=np.stack(rx_centres)
    between=float(2*classes/(classes-1)*centres.var(0).sum())
    variation=float(rx_centres.var(0).sum(-1).mean())
    within=float(np.mean([x[y==c].var(0).sum() for c in range(classes)]))
    degenerate=between<epsilon
    return dict(D_between=between,V_RX_given_TX=variation,within_TX_variance=within,
        reliability_score=between/(variation+epsilon) if not degenerate else None,
        discriminability_score=between/(within+epsilon) if not degenerate else None,
        degenerate_feature=degenerate)

def rank_correlation(a,b):
    a=np.asarray(a,dtype='float64');b=np.asarray(b,dtype='float64')
    finite=np.isfinite(a)&np.isfinite(b)
    if finite.sum()<3 or np.ptp(a[finite])==0 or np.ptp(b[finite])==0:return None
    return float(spearmanr(a[finite],b[finite]).statistic)
