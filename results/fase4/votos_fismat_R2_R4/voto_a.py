# Voto (a): hold-out em blocos com b=0 (sem aparo), particao propria por blocos de 10 km.
# Copiado (declarado): build_synthetic_grid/latlon_graus_para_metros (geometria), free_space_path_loss (baseline congelado), leitura do tensor.
import sys,json,gc,numpy as np,torch
sys.path.insert(0,"/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts")
from varredura_split_geometria import TREINOS_DIR,build_synthetic_grid,latlon_graus_para_metros
import importlib.util
sp=importlib.util.spec_from_file_location("bl","/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/scripts_congelados/baselines_v2_por_particao.py")
bl=importlib.util.module_from_spec(sp);sp.loader.exec_module(bl)
base='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/'
seeds=json.load(open(base+'fase1/1.8_referencia_nodal_200seeds.json'))['seeds']
out={}
for cid in ('bauru','lins'):
    d=json.load(open(TREINOS_DIR/f"run_c0c1cf_{cid}_s42_Q1_g10b2.json"))['geometria']
    lon,lat=build_synthetic_grid(d['lon_min_deg'],d['lon_max_deg'],d['lat_min_deg'],d['lat_max_deg'])
    pos=latlon_graus_para_metros(lon,lat)/1000.0
    ix=(pos[:,0]//10).astype(int);iy=(pos[:,1]//10).astype(int)
    gid=ix*(iy.max()+1)+iy  # particao propria
    ug,inv=np.unique(gid,return_inverse=True);N=len(ug)
    rf=torch.load(f"/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/transfer_dataset_{cid}_v19_Q1_enriched_cftudo.pt",map_location='cpu',weights_only=False,mmap=True)
    ty=torch.as_tensor(rf['terrain'].y).float().numpy();dist=torch.as_tensor(rf['terrain'].dist_nearest_m).float().numpy().astype(float);del rf
    rssi=ty[:,3].astype(float);valid=ty[:,0]<299.0
    ktr=max(1,round(.7*N));kva=max(1,round(.15*N));kte=N-ktr-kva
    def calib(trb):
        m=np.isin(inv,trb)&valid
        return np.median(rssi[m]+bl.free_space_path_loss(dist[m],900.0))
    perm=np.random.RandomState(42).permutation(N);ptx=calib(perm[:ktr])
    e=np.abs(rssi-(ptx-bl.free_space_path_loss(dist,900.0)))
    res={'N':N,'kte':kte}
    for pop,mask in(('validos',valid),('todos',np.ones_like(valid))):
        Sb=np.bincount(inv[mask],weights=e[mask],minlength=N);Mb=np.bincount(inv[mask],minlength=N).astype(float)
        mu=e[mask].mean()
        S=[];M=[]
        for s in seeds:
            p=np.random.RandomState(s).permutation(N);te=p[ktr+kva:]
            S.append(Sb[te].sum());M.append(Mb[te].sum())
        S=np.array(S);M=np.array(M);ok=M>0;S=S[ok];M=M[ok];R=S/M
        vies=R.mean()-mu
        cov=np.mean((R-R.mean())*(M-M.mean()))  # cov populacional
        rhs=-cov/M.mean()
        # identidade E[S/M]-E[S]/E[M] = -Cov(S/M,M)/E[M]
        lhs=R.mean()-S.mean()/M.mean()
        # theta=E[S]/E[M]; vies design = theta-mu
        theta=S.mean()/M.mean()
        res[pop]=dict(n_sorteios_M_pos=int(ok.sum()),mu_U=mu,Err_media=R.mean(),vies_vs_mu=vies,theta_menos_mu=theta-mu,razao=lhs,menos_cov_sobre_EM=rhs,diff_identidade=lhs-rhs,corr_Err_M=float(np.corrcoef(R,M)[0,1]),M_media=M.mean(),M_cv=M.std()/M.mean(),ep_mc_vies=R.std(ddof=1)/np.sqrt(len(R)))
        # preditor constante
        e0=np.abs(rssi-np.median(rssi[np.isin(inv,perm[:ktr])]))
        Sb0=np.bincount(inv[mask],weights=e0[mask],minlength=N);R0=np.array([Sb0[np.random.RandomState(s).permutation(N)[ktr+kva:]].sum() for s in seeds])[ok]/M
        res[pop]['constante']=dict(vies=R0.mean()-e0[mask].mean(),corr_Err_M=float(np.corrcoef(R0,M)[0,1]))
    out[cid]=res;print(cid,json.dumps(res,indent=1),flush=True)
json.dump(out,open('a.json','w'),indent=1)
