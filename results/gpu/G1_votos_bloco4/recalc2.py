import numpy as np, json, glob, re, hashlib, csv
G='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/G1/'
def mae(f):
    z=np.load(f,allow_pickle=True);t=z['target'];p=z['pred'];v=(t[:,0]<299)
    return np.abs(p[v,3]-t[v,3]).mean()
for m in['gnn','mlp']:
    fs=sorted(glob.glob(f'{G}g1_{m}_campinas_Q1_ss*_s42/predicoes_*.npz'));x=[mae(f) for f in fs]
    print(m,len(x),np.std(x,ddof=1))
# config comparison
def cfg(m,ss,s):
    c=json.load(open(f'{G}g1_{m}_campinas_Q1_ss{ss}_s{s}/run_g1_{m}_campinas_Q1_ss{ss}_s{s}.json'))['config'].copy()
    for k in['run_label','seed','split_seed','evid_dir']:c.pop(k,None);
    return c
for m in['gnn','mlp']:
    ref=cfg(m,811474,42);bad=[]
    for ss in[811474,54892,880099,125243,159513,739191,29675,676375,387379,259492]:
        for s in[42,43,44]:
            c=cfg(m,ss,s)
            d={k:(ref.get(k),c.get(k)) for k in set(ref)|set(c) if ref.get(k)!=c.get(k)}
            if d:bad.append((ss,s,d))
    print(m,'config diffs',bad[:3],len(bad))
# anomaly
rows={}
for m in['gnn']:
  for ss in[811474,54892,880099,125243,159513,739191,29675,676375,387379,259492]:
    for s in[42,43,44]:
        d=f'{G}g1_{m}_campinas_Q1_ss{ss}_s{s}/'
        L=list(csv.DictReader(open(d+'training_log.csv')))
        va=[float(r['val_mae_rssi']) for r in L];te=[float(r['test_mae_rssi']) for r in L]
        e=int(np.argmin(va));srt=sorted(va);marg=srt[1]-srt[0]
        z=np.load(d+f'predicoes_g1_{m}_campinas_Q1_ss{ss}_s{s}.npz',allow_pickle=True)
        t=z['target'];p=z['pred'];v=t[:,0]<299
        mp=np.abs(p[v,3]-t[v,3]).mean()
        print(ss,s,'ep_ret',e+1,'nep',len(L),'marg%.4f'%marg,'val@ret%.4f'%va[e],'test@ret%.4f'%te[e],'n',len(t),'nv',int(v.sum()),'maeval%.4f'%mp,'fin',bool(np.isfinite(p).all()),'sent',int(z['sentinela'].sum()),hashlib.sha256(z['idx_global'].tobytes()).hexdigest()[:8])
