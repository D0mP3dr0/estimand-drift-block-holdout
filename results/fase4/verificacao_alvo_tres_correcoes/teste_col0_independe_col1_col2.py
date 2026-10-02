import sys,json,numpy as np,torch
G="/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/"; out={}
for c,q in [("lins","Q1"),("campinas","Q3")]:
    d=torch.load(G+f"transfer_dataset_{c}_v19_{q}_enriched_cftudo.pt",map_location="cpu",weights_only=False,mmap=True);T=d["terrain"]
    N=T.y.shape[0];a=N//2-1_500_000;b=a+3_000_000
    y=np.array(T.y[a:b],float);dist=np.array(T.dist_nearest_m[a:b],float)
    cov=y[:,0]!=300.0;ok=cov[1:]&cov[:-1]
    dl=np.diff(np.log(dist+1));m=ok&(np.abs(dl)<0.01)
    dy=[np.diff(y[:,i]) for i in range(5)]
    X=np.c_[dy[1][m],dy[2][m],dl[m],np.ones(m.sum())]
    be=np.linalg.lstsq(X,dy[0][m],rcond=None)[0]
    out[f"{c}_{q}"]=dict(n=int(m.sum()),coef_dcol1=float(be[0]),coef_dcol2=float(be[1]),
      corr_d0_d1=float(np.corrcoef(dy[0][m],dy[1][m])[0,1]),corr_d0_d2=float(np.corrcoef(dy[0][m],dy[2][m])[0,1]),
      corr_d0_dnegcol3=float(np.corrcoef(dy[0][m],-dy[3][m])[0,1]),
      pct_col1_zero=float((y[cov,1]==0).mean()),std_col1=float(y[cov,1].std()),std_col2=float(y[cov,2].std()))
    print(c,q,out[f"{c}_{q}"],flush=True)
json.dump(out,open(sys.argv[1],"w"),indent=1)
