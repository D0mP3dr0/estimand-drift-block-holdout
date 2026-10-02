import sys, json, numpy as np, torch
sys.path.insert(0,"/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/data_raw")
import enrich_rf_targets as E
G="/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/"
out={}
for c,q in [("lins","Q1"),("campinas","Q3")]:
    s=torch.load(G+f"{c}_v19_{q}_gpu.pt",map_location="cpu",weights_only=False,mmap=True)
    nm=s.normalization; es=float(nm["elev_std"]); del s
    d=torch.load(G+f"transfer_dataset_{c}_v19_{q}_enriched_cftudo.pt",map_location="cpu",weights_only=False,mmap=True)
    T=d["terrain"]; N=T.y.shape[0]
    a=N//2-1_500_000; b=a+3_000_000
    x=np.array(T.features_raw[a:b],np.float32); y=np.array(T.y[a:b],np.float32)
    dist=T.dist_nearest_m[a:b].float().clone()
    f=lambda tri,r: E.compute_diffraction_loss(torch.from_numpy(tri),torch.from_numpy(r),dist,1800.0).numpy().astype(np.float64)
    V={"orig(tpi*es, tri_norm)":f(x[:,5]*es,x[:,6]),
       "so_corr2(tri*es, tri_norm)":f(x[:,6]*es,x[:,6]),
       "so_corr3(tpi*es, rug*es)":f(x[:,5]*es,x[:,7]*es),
       "corr2+3(tri*es, rug*es)":f(x[:,6]*es,x[:,7]*es)}
    g=y[:,2].astype(np.float64)
    r=dict(es=es,n=int(b-a),neg_col5=float((x[:,5]<0).mean()),min_col6=float(x[:,6].min()),min_col7=float(x[:,7].min()),
           alvo_p50=float(np.median(g)),frac_piso=float((g<=6.0206004).mean()),
           std_alvo=float(g.std()),std_dist=float(dist.std()),var={})
    dl=np.diff(np.log(dist.numpy().astype(float)+1)); m=np.abs(dl)<0.01
    dy=np.diff(g)
    for k,v in V.items():
        dv=np.diff(v); mm=m
        X=np.c_[dv[mm],dl[mm],np.ones(mm.sum())]
        beta=np.linalg.lstsq(X,dy[mm],rcond=None)[0]
        r["var"][k]=dict(max_abs_dif=float(np.abs(v-g).max()),frac_igual_1em4=float((np.abs(v-g)<1e-4).mean()),
            mean_abs_dif=float(np.abs(v-g).mean()),p50_var=float(np.median(v)),
            corr_nivel=float(np.corrcoef(v,g)[0,1]) if v.std()>0 else None,
            coef_diff=float(beta[0]),corr_diff=float(np.corrcoef(dv[mm],dy[mm])[0,1]) if dv[mm].std()>0 else None)
    out[f"{c}_{q}"]=r; print(c,q,json.dumps(r,indent=1),flush=True)
json.dump(out,open(sys.argv[1],"w"),indent=1)
