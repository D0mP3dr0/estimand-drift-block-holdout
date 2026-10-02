"""Recalculo independente R3/R4 (lins_Q1). COPIADO (declarado): construcao da malha sintetica
3600x3600 (linspace lon/lat, meshgrid, lat decrescente), conversao lat/lon->km (111000, cos lat),
assign_groups (truncamento int), sentinela pl>=299, seeds de 1.8, shuffle RandomState(seed) dos blocos.
Proprios: vizinhanca nodal por bloco, m(i), condicao (iii), retencao sequencial, frequencias, bootstrap."""
import json, sys, time
import numpy as np, torch
from scipy.spatial import cKDTree
R="/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/"
cid="lins"; G=10.0; B=2.0; N_SIDE=3600
geo=json.load(open("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/treinos_c1/run_c0c1cf_lins_s42_Q1_g10b2.json"))["geometria"]
seeds=json.load(open(R+"_v3_2026-09-25/fase1/1.8_referencia_nodal_200seeds.json"))["seeds"]
lo=np.linspace(geo["lon_min_deg"],geo["lon_max_deg"],N_SIDE); la=np.linspace(geo["lat_max_deg"],geo["lat_min_deg"],N_SIDE)
LO,LA=np.meshgrid(lo,la); LO=LO.ravel(); LA=LA.ravel()
y=(LA-LA.min())*111000.0; x=(LO-LO.min())*111000.0*np.cos(np.radians(LA))
pos=np.stack([x,y],1)/1000.0; del LO,LA,x,y
n=pos.shape[0]
gx=(pos[:,0]/G).astype(int); gy=(pos[:,1]/G).astype(int)
raw=gx*(gy.max()+1)+gy
ids,blk=np.unique(raw,return_inverse=True); blk=blk.astype(np.int32); NB=len(ids)
bgx=np.zeros(NB,int); bgy=np.zeros(NB,int); bgx[blk]=gx; bgy[blk]=gy
print("N blocos",NB,flush=True)
order=np.argsort(blk,kind="stable"); starts=np.searchsorted(blk[order],np.arange(NB+1))
nodes_of=[order[starts[b]:starts[b+1]] for b in range(NB)]
lut={(bgx[b],bgy[b]):b for b in range(NB)}
DIRS=[(dx,dy) for dx in(-1,0,1) for dy in(-1,0,1) if (dx,dy)!=(0,0)]
nb=-np.ones((NB,8),np.int32)
mask=np.zeros(n,np.uint8); near=-np.ones((8,n),np.int32)
t0=time.time()
for a in range(NB):
    ia=nodes_of[a]
    for d,(dx,dy) in enumerate(DIRS):
        bb=lut.get((bgx[a]+dx,bgy[a]+dy))
        if bb is None: continue
        nb[a,d]=bb
        ib=nodes_of[bb]; pb=pos[ib]; pa=pos[ia]
        lob=pb.min(0); hib=pb.max(0)
        dxa=np.maximum(np.maximum(lob-pa,pa-hib),0); ca=np.hypot(dxa[:,0],dxa[:,1])<B
        if not ca.any(): continue
        loa=pa.min(0); hia=pa.max(0)
        dxb=np.maximum(np.maximum(loa-pb,pb-hia),0); cb=np.hypot(dxb[:,0],dxb[:,1])<B
        if not cb.any(): continue
        ib2=ib[cb]; ia2=ia[ca]
        dd,ix=cKDTree(pos[ib2],compact_nodes=False,balanced_tree=False).query(pos[ia2],k=1,workers=8)
        ok=dd<B
        mask[ia2[ok]]|=np.uint8(1<<d); near[d,ia2[ok]]=ib2[ix[ok]]
print("vizinhanca",time.time()-t0,flush=True)
m=np.zeros(n,np.int8)
for d in range(8): m+=((mask>>d)&1).astype(np.int8)
# condicao (iii): em frame 5x5 com centro (2,2); bit=(ox+2)*5+(oy+2)
def bit(ox,oy): return (ox+2)*5+(oy+2)
own=np.zeros(n,np.int64)  # offsets do proprio no (vizinhos+centro)
own|=1<<bit(0,0)
for d,(dx,dy) in enumerate(DIRS): own|=(((mask>>d)&1).astype(np.int64))<<bit(dx,dy)
iii_falha=np.zeros(n,bool)
for d,(dx,dy) in enumerate(DIRS):
    sel=np.where((mask>>d)&1)[0]
    j=near[d,sel]
    jb=own[j]            # offsets de j relativos ao bloco B (centro de j em (2,2))
    sh=dx*5+dy
    jb_shift=(jb<<sh) if sh>=0 else (jb>>(-sh))
    # em frame de i: offsets de j (relativos a B) + d ; usar B->i: deslocamento d
    iii_falha[sel]|=((jb_shift & ~own[sel])!=0)
print("m counts",np.bincount(m)/n,"iii falha por m",[int(iii_falha[m==k].sum()) for k in range(m.max()+1)],flush=True)
# dados e
rf=torch.load(f"/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/transfer_dataset_{cid}_v19_Q1_enriched_cftudo.pt",map_location="cpu",weights_only=False,mmap=True)
ty=torch.as_tensor(rf["terrain"].y).float().numpy(); assert ty.shape[0]==n
rssi=ty[:,3].astype(np.float64); sent=ty[:,0]>=299.0; del rf
e=np.abs(rssi+110.0); valid=~sent
mu={"todos":float(e.mean()),"validos":float(e[valid].mean())}
# retencao
Nb=NB; k_tr=max(1,round(.70*Nb)); k_va=max(1,round(.15*Nb)); k_te=Nb-k_tr-k_va
print(Nb,k_tr,k_va,k_te)
cnt=np.zeros(n,np.int16); tecount=np.zeros(NB,np.int64)
S={"todos":[],"validos":[]}; M={"todos":[],"validos":[]}
nbc=nb.copy()
def flags(idx,cls):
    """has_train, has_val por no em idx"""
    mk=mask[idx]; b=blk[idx]; ht=np.zeros(idx.size,bool); hv=np.zeros(idx.size,bool)
    for d in range(8):
        on=((mk>>d)&1).astype(bool)
        c=cls[np.where(nb[b,d]>=0,nb[b,d],0)]
        ht|=on&(c==0); hv|=on&(c==1)
    return ht,hv
t0=time.time()
for s in seeds:
    emb=np.arange(NB); np.random.RandomState(s).shuffle(emb)
    cls=np.zeros(NB,np.int8); cls[emb[k_tr:k_tr+k_va]]=1; cls[emb[k_tr+k_va:]]=2
    tb=np.where(cls==2)[0]; vb=np.where(cls==1)[0]; tecount[tb]+=1
    ti=np.concatenate([nodes_of[b] for b in tb]); vi=np.concatenate([nodes_of[b] for b in vb])
    ht,hv=flags(ti,cls); cand=~ht
    ret=cand&~hv
    need=cand&hv
    if need.any():
        hvv,_=None,None
        # val retidos: sem treino na vizinhanca; so os que tem teste vizinho importam
        htv,_h2=flags(vi,cls)
        vr=vi[~htv]
        vt=cKDTree(pos[vr],compact_nodes=False,balanced_tree=False)
        ci=ti[need]; dd,_=vt.query(pos[ci],k=1,distance_upper_bound=B,workers=8)
        ret[np.where(need)[0][~(dd<B)]]=True
    r=ti[ret]; cnt[r]+=1
    for p,sel in (("todos",None),("validos",valid)):
        rr=r if sel is None else r[valid[r]]
        S[p].append(float(e[rr].sum())); M[p].append(int(rr.size))
print("sorteios",time.time()-t0,flush=True)
# R4
pc=cnt/np.maximum(tecount[blk],1)  # freq condicional por no
cl={}
def ff(a,b_,k): 
    r=1.0
    for i in range(k): r*=(a-i)/(b_-i)
    return r
out={"N":Nb,"k_tr":k_tr,"k_va":k_va,"k_te":k_te,"blocos_te_contagem_min_max":[int(tecount.min()),int(tecount.max())]}
rows=[]
for k in range(m.max()+1):
    for nome,sel in ((f"m={k}",m==k),(f"m={k}|iii_vale",(m==k)&~iii_falha),(f"m={k}|iii_falha",(m==k)&iii_falha)):
        if sel.sum()==0: continue
        rows.append(dict(classe=nome,fracao_nos=float(sel.mean()),n=int(sel.sum()),freq_media_nodal=float(pc[sel].mean()),
          lo=ff(k_te-1,Nb-1,k),hi=ff(Nb-1-k_tr,Nb-1,k)))
out["R4_classes"]=rows
out["freq_media_total_cond"]=float(pc.mean())
out["19/131"]=19/131
# R3
out["R3"]={}
rng=np.random.RandomState(7)
for p in("todos","validos"):
    Sa=np.array(S[p]);Ma=np.array(M[p],float)
    t=Sa.sum()/Ma.sum()-mu[p]
    bs=[]
    for _ in range(4000):
        ii=rng.randint(0,200,200); bs.append(Sa[ii].sum()/Ma[ii].sum()-mu[p])
    # checagem direta com p_i=cnt/200
    pi=cnt/200.0; selp=slice(None) if p=="todos" else valid
    td=float((pi[selp]*e[selp]).sum()/pi[selp].sum()-mu[p])
    out["R3"][p]=dict(mu_U=mu[p],termo=float(t),termo_direto_cnt=td,ep_boot=float(np.std(bs)),ic95_boot=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],p_medio=float(pi[selp].mean()))
json.dump(out,open("voto_R3R4_proprio_saida.json","w"),indent=1)
print(json.dumps(out,indent=1))
