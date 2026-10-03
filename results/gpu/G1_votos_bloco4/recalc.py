import numpy as np, json, glob, hashlib, csv, os
from scipy import stats
G='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/G1/'
SS=[811474,54892,880099,125243,159513,739191,29675,676375,387379,259492]; SE=[42,43,44]
out={}; idxh={}
def load(m,ss,s):
    d=f'{G}g1_{m}_campinas_Q1_ss{ss}_s{s}/'
    z=np.load(d+f'predicoes_g1_{m}_campinas_Q1_ss{ss}_s{s}.npz',allow_pickle=True)
    return d,z
def mae(z):
    t=z['target'];p=z['pred'];v=(t[:,0]<299)&np.isfinite(p[:,3])&np.isfinite(t[:,3])
    return float(np.abs(p[v,3]-t[v,3]).mean()),int(v.sum())
for m in['gnn','mlp']:
    M=np.zeros((10,3));N=np.zeros((10,3),int)
    for i,ss in enumerate(SS):
        hs=set()
        for j,s in enumerate(SE):
            d,z=load(m,ss,s); M[i,j],N[i,j]=mae(z)
            hs.add(hashlib.sha256(z['idx_global'].tobytes()).hexdigest()[:12])
            r=json.load(open(d+f'run_g1_{m}_campinas_Q1_ss{ss}_s{s}.json'))
            c=r['config']
            assert r['seed']==s and r['split_seed']==ss and c['grid_km']==10 and c['buffer_km']==2,(m,ss,s)
            idxh.setdefault((m,ss,s),(c,))
        idxh[(m,ss,'hs')]=len(hs)
    a,b=10,3
    gm=M.mean(); sm=M.mean(1)
    QMe=b*((sm-gm)**2).sum()/(a-1); QMd=((M-sm[:,None])**2).sum()/(a*(b-1))
    s2=(QMe-QMd)/b; sp=np.sqrt(QMd); R=s2/QMd
    F=QMe/QMd; lo=F/stats.f.ppf(.975,9,20); hi=F/stats.f.ppf(.025,9,20)
    # IC da razao sigma2_sorteio/sigma2_semente = (F/Fcrit -1)/b
    ic=[(lo-1)/b,(hi-1)/b]
    loo=[]
    for k in range(10):
        X=np.delete(M,k,0);a2=9;sm2=X.mean(1);g=X.mean()
        e=b*((sm2-g)**2).sum()/(a2-1);d_=((X-sm2[:,None])**2).sum()/(a2*(b-1));loo.append((e-d_)/b/d_)
    out[m]=dict(QMe=QMe,QMd=QMd,s2=s2,dp=sp,R=R,IC=ic,loo_min=min(loo),loo_max=max(loo),nvals=sorted(set(N.ravel().tolist()))[:3],M=M.round(4).tolist(),
        same_idx_all=all(idxh[(m,ss,'hs')]==1 for ss in SS))
    if m=='gnn':
      pass
    out[m]['sd42_20']=None
json.dump(out,open('/tmp/claude-1000/-trabalho-HERMES/492a006d-c0f8-41c5-9896-1453e7ec0600/scratchpad/o.json','w'),indent=1)
for m in out: print(m,{k:v for k,v in out[m].items() if k!='M'})
