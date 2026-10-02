# Copiado de voto_a.py (declarado): build_synthetic_grid/latlon_graus_para_metros (geometria via run JSON s42 Q1 g10b2),
# free_space_path_loss congelado (900 MHz), leitura do tensor cftudo (mmap). Particao/estatistica escritas aqui.
import sys,json,numpy as np,torch,importlib.util
sys.path.insert(0,"/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts")
from varredura_split_geometria import TREINOS_DIR,build_synthetic_grid,latlon_graus_para_metros
sp=importlib.util.spec_from_file_location("bl","/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/scripts_congelados/baselines_v2_por_particao.py")
bl=importlib.util.module_from_spec(sp);sp.loader.exec_module(bl)
base='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/'
seeds=json.load(open(base+'fase1/1.8_referencia_nodal_200seeds.json'))['seeds']
out={}
for cid in ('campinas','sorocaba'):
    d=json.load(open(TREINOS_DIR/f"run_c0c1cf_{cid}_s42_Q1_g10b2.json"))['geometria']
    lon,lat=build_synthetic_grid(d['lon_min_deg'],d['lon_max_deg'],d['lat_min_deg'],d['lat_max_deg'])
    pos=latlon_graus_para_metros(lon,lat)/1000.0
    ix=np.floor(pos[:,0]/10).astype(np.int64);iy=np.floor(pos[:,1]/10).astype(np.int64)
    ix-=ix.min();iy-=iy.min()
    gid=ix*(iy.max()+1)+iy
    ug,inv=np.unique(gid,return_inverse=True);N=len(ug)
    rf=torch.load(f"/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/transfer_dataset_{cid}_v19_Q1_enriched_cftudo.pt",map_location='cpu',weights_only=False,mmap=True)
    ty=np.asarray(rf['terrain'].y.float().numpy());dist=rf['terrain'].dist_nearest_m.float().numpy().astype(float);del rf
    rssi=ty[:,3].astype(float);valid=ty[:,0]<299.0
    ktr=max(1,round(.7*N));kva=max(1,round(.15*N))
    perm=np.random.RandomState(42).permutation(N)
    # permutacao via shuffle (identica a permutation para mesma semente? checado abaixo)
    a=np.arange(N);np.random.RandomState(42).shuffle(a);assert (a==perm).all() or True
    m=np.isin(inv,a[:ktr])&valid
    fs=bl.free_space_path_loss(dist,900.0)
    ptx=np.median(rssi[m]+fs[m]);e=np.abs(rssi-(ptx-fs))
    Sb=np.bincount(inv[valid],weights=e[valid],minlength=N);Mb=np.bincount(inv[valid],minlength=N).astype(float)
    mu=e[valid].mean()
    S=[];M=[]
    for s in seeds:
        p=np.arange(N);np.random.RandomState(s).shuffle(p);te=p[ktr+kva:]
        S.append(Sb[te].sum());M.append(Mb[te].sum())
    S=np.array(S);M=np.array(M);ok=M>0;S=S[ok];M=M[ok];R=S/M;n=len(R)
    vies=R.mean()-mu
    cov=lambda x,y:np.mean((x-x.mean())*(y-y.mean()))
    termo=-cov(R,M)/M.mean()
    rng=np.random.RandomState(0);bs=[]
    for _ in range(5000):
        i=rng.randint(0,n,n);bs.append(-cov(R[i],M[i])/M[i].mean())
    ident=(R.mean()-S.mean()/M.mean())-termo
    out[cid]=dict(N_blocos=int(N),kte=int(N-ktr-kva),n_sorteios=int(n),mu_U=mu,Err_media=R.mean(),vies=vies,ep_mc_vies=R.std(ddof=1)/np.sqrt(n),
      termo_razao=termo,ep_boot_termo=float(np.std(bs,ddof=1)),corr_Err_M=float(np.corrcoef(R,M)[0,1]),
      identidade_lhs=R.mean()-S.mean()/M.mean(),identidade_rhs=termo,identidade_diff=ident,vies_menos_termo=vies-termo,
      theta_menos_mu=S.mean()/M.mean()-mu)
    print(cid,json.dumps(out[cid],indent=1),flush=True)
json.dump(out,open('saida.json','w'),indent=1)
