import sys,json,numpy as np,torch
G="/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/"; out={}
def A(s):
    s=np.clip(s,0,None);t=s*20.0;t=np.where(s>0.1,t+(s-0.1)*30.0,t);return np.clip(t,0,30)
for c,q in [("lins","Q1"),("campinas","Q3")]:
    d=torch.load(G+f"transfer_dataset_{c}_v19_{q}_enriched_cftudo.pt",map_location="cpu",weights_only=False,mmap=True);T=d["terrain"]
    N=T.y.shape[0];a=N//2-1_500_000;b=a+3_000_000
    y=np.array(T.y[a:b],float);x=np.array(T.features_raw[a:b],float);dist=np.array(T.dist_nearest_m[a:b],float)
    cov=y[:,0]!=300.0;ok=cov[1:]&cov[:-1]
    dl=np.diff(np.log(dist+1));m=ok&(np.abs(dl)<0.01)
    dA=np.diff(A(x[:,1])); dn=np.diff(x[:,12]); dd=np.diff(y[:,2]); d1=np.diff(y[:,1]); d0=np.diff(y[:,0])
    X=np.c_[dA[m],dn[m],dn[m]*np.log(dist[1:][m]+1),dd[m],d1[m],dl[m],np.ones(m.sum())]
    be,res,_,_=np.linalg.lstsq(X,d0[m],rcond=None)
    r2=1-((d0[m]-X@be)**2).sum()/((d0[m]-d0[m].mean())**2).sum()
    X0=X[:,[0,1,2,5,6]];b0=np.linalg.lstsq(X0,d0[m],rcond=None)[0]
    r20=1-((d0[m]-X0@b0)**2).sum()/((d0[m]-d0[m].mean())**2).sum()
    out[f"{c}_{q}"]=dict(n=int(m.sum()),coef=dict(zip(["dA_terrain","dndvi","dndvi_x_logd","dcol2","dcol1","dlogd","cte"],map(float,be))),R2_com_col1_col2=float(r2),R2_sem_col1_col2=float(r20))
    print(c,q,out[f"{c}_{q}"],flush=True)
json.dump(out,open(sys.argv[1],"w"),indent=1)
