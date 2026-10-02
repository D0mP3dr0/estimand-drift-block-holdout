import sys,json,time
sys.path.insert(0,'/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts')
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from scipy.spatial import cKDTree
from varredura_split_geometria import TREINOS_DIR, build_synthetic_grid, latlon_graus_para_metros, assign_groups
cid,q,G,B=sys.argv[1],sys.argv[2],float(sys.argv[3]),2.0
N=int(sys.argv[4]); UP=[int(u) for u in sys.argv[5].split(',')]; NU=int(sys.argv[6])
geo=json.load(open(TREINOS_DIR/f"run_c0c1cf_{cid}_s42_{q}_g10b2.json"))['geometria']
lon,lat=build_synthetic_grid(geo['lon_min_deg'],geo['lon_max_deg'],geo['lat_min_deg'],geo['lat_max_deg'])
pos=latlon_graus_para_metros(lon,lat)/1000.; n=3600
X=pos[:,0].reshape(n,n);Y=pos[:,1].reshape(n,n)
sx=float((X[:,-1]-X[:,0]).mean()/(n-1)); sy=float(abs(Y[-1,0]-Y[0,0])/(n-1))
gid=assign_groups(pos,G); grupos=np.unique(gid); ng=len(grupos)
print('espacamento km',sx,sy,'n_blocos',ng,flush=True)
gi=gid.reshape(n,n)
def split(seed):
    r=np.random.RandomState(seed); e=grupos.copy(); r.shuffle(e)
    ntr=max(1,int(round(.7*ng))); nva=max(1,int(round(.15*ng)))
    if ntr+nva>=ng: ntr=max(1,ng-2);nva=1
    return e[:ntr],e[ntr:ntr+nva],e[ntr+nva:]
def ret(seed,up):
    gtr,gva,gte=split(seed)
    g=gi if up==1 else np.repeat(np.repeat(gi,up,0),up,1)
    mtr=np.isin(g,gtr);mva=np.isin(g,gva);mte=np.isin(g,gte)
    smp=(sy/up,sx/up)
    d=edt(~mtr,sampling=smp); nv0=mva.sum();nt0=mte.sum(); mva&=~(d<B)
    d=edt(~(mtr|mva),sampling=smp); mte&=~(d<B)
    return mte.sum()/nt0
def retexact(seed):
    gtr,gva,gte=split(seed); mtr=np.isin(gid,gtr);mva=np.isin(gid,gva);mte=np.isin(gid,gte)
    nt0=mte.sum(); kw=dict(compact_nodes=False,balanced_tree=False)
    d,_=cKDTree(pos[mtr],**kw).query(pos[mva],workers=-1); iv=np.where(mva)[0]; mva[iv[d<B]]=False
    d,_=cKDTree(pos[mtr|mva],**kw).query(pos[mte],workers=-1); it=np.where(mte)[0]; mte[it[d<B]]=False
    return mte.sum()/nt0
out=dict(cid=cid,q=q,G=G,sx=sx,sy=sy,ng=ng)
if UP[0]==1 and len(sys.argv)>7:
    s=1000; t=time.time(); out['exato_seed1000']=retexact(s); out['edt_seed1000']=ret(s,1); print(out['exato_seed1000'],out['edt_seed1000'],time.time()-t,flush=True)
S13=list(range(1000,1050)); S18=np.random.RandomState(20260926).randint(10**6,size=200).tolist(); S3=np.random.RandomState(777).randint(10**6,size=200).tolist()
res={}
for name,S in (('s13',S13),('s18',S18),('s3',S3)):
    a=np.array([ret(s,1) for s in S]); res[name]=a.tolist()
    print(name,a.mean(),a.std(ddof=1),len(a),flush=True)
out['nodal']=res
# area continua: mesmas permutacoes (s18 primeiras NU)
cont={}
for up in UP:
    if up==1: continue
    t=time.time(); S=S18[:NU]
    a=np.array([ret(s,up) for s in S]); b=np.array([ret(s,1) for s in S])
    cont[str(up)]=dict(area=a.tolist(),nodal=b.tolist()); print('up',up,'area',a.mean(),'nodal',b.mean(),'rel',a.mean()/b.mean()-1,time.time()-t,flush=True)
out['cont']=cont
json.dump(out,open(f'saida_{cid}_{q}_g{int(G)}.json','w'))
