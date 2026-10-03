import os, json, sys, hashlib, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
B="/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics"
F5=B+"/_v3_2026-09-25/fase5"
D=json.load(open(F5+"/R5_por_sorteio.json"))
R=json.load(open(F5+"/R5_resumo.json"))
def est(S,M,N):
    S=np.array(S,float);M=np.array(M,float);k=len(M)
    if k<2: return None,k,(S.sum()/M.sum() if k else None)
    Mt=M.sum();E=S.sum()/Mt
    return (1-k/N)*k/(k-1)*np.sum((S-E*M)**2)/Mt**2,k,E
def cell(sorts,N,key):
    rs=[est(s[key]["S_B"],s[key]["M_B"],N) for s in sorts] if sorts and "S_B" in sorts[0].get(key,{}) else sorts
    return rs
def resumo(rs):
    E=np.array([r[2] for r in rs if r[2] is not None])
    V=[(r[0],r[2]) for r in rs if r[0] is not None]
    var=np.var(E,ddof=1);m=E.mean()
    mv=np.mean([v for v,_ in V]);Q=mv/var
    cov=np.mean([abs(e-m)<=2*np.sqrt(v) for v,e in V])
    return Q,cov,len(rs)-len(V)
out={};
for pred in("constante","fspl_b"):
  for pop in("validos","todos"):
    key=f"{pred}__{pop}";Qs=[];Cs=[];Ind=[];cells={}
    for ch,c in D["celulas"].items():
        rs=[est(s[key]["S_B"],s[key]["M_B"],c["N_blocos_ocupados"]) for s in c["sorteios"]]
        Q,cv,ind=resumo(rs);Qs.append(Q);Cs.append(cv);Ind.append(ind)
        a=R["resumo"][key]["N_ocupados"]["leitura_A"]["por_cell" if False else "por_celula"][ch]
        cells[ch]=(Q,a["Q"],cv,a["cobertura"])
    Qs=np.array(Qs);Cs=np.array(Cs)
    out[key]={"Q_mediano":float(np.median(Qs)),"Q_min":float(Qs.min()),"Q_max":float(Qs.max()),
      "n_Q_faixa":int(((Qs>=.5)&(Qs<=2)).sum()),"n_cob_ge80":int((Cs>=.8).sum()),"n_cob_lt60":int((Cs<.6).sum()),
      "k_lt2_min":int(min(Ind)),"k_lt2_max":int(max(Ind)),
      "max_abs_dif_Q_vs_gravado":max(abs(a-b) for a,b,_,_ in cells.values()),
      "max_abs_dif_cob_vs_gravado":max(abs(c-d) for _,_,c,d in cells.values()),
      "n_Q_faixa_E_cob80":int(((Qs>=.5)&(Qs<=2)&(Cs>=.8)).sum())}
# bruto
spec=importlib.util.spec_from_file_location("orig",B+"/scripts/v3_2.1_3.1_deriva_calibracao.py")
orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)
import torch
bruto={}
for ch in("lins_Q1","bauru_Q1"):
    cid,q=ch.split("_")
    rf,_=orig.carregar_tensor(orig.TENSOR_DIR/f"transfer_dataset_{cid}_v19_{q}_enriched_cftudo.pt")
    ty=torch.as_tensor(rf["terrain"].y).float().numpy().astype(float)
    dist=torch.as_tensor(rf["terrain"].dist_nearest_m).float().numpy()
    pos=orig.latlon_graus_para_metros(torch.as_tensor(rf["terrain"].pos).float().clone())
    rssi=ty[:,3];sent=ty[:,0]>=orig.PL_TARGET_MAX_VALID
    km=pos/1000.0
    gx=np.floor(km[:,0]/10).astype(int);gy=np.floor(km[:,1]/10).astype(int)
    gid=gx*(gy.max()+1)+gy  # minha definicao de bloco (agrupamento so importa)
    N=np.unique(gid).size
    fspl=orig.MODELOS["fspl"]
    rows={};
    for s in D["celulas"][ch]["sorteios"]:
        sd=s["split_seed"]
        p=orig.split_espacial_3vias(pos,10.0,2.0,(0.70,0.15,0.15),sd)
        tr,te=p["train"],p["test"]
        c0=np.median(rssi[tr]);v=~sent[tr]
        ptx=np.median(rssi[tr][v]+fspl(dist[tr][v]))
        L={"constante":np.abs(c0-rssi[te]),"fspl_b":np.abs(ptx-fspl(dist[te])-rssi[te])}
        for pr,l in L.items():
            for pop,m in(("validos",~sent[te]),("todos",np.ones(te.size,bool))):
                g=gid[te][m];u,inv=np.unique(g,return_inverse=True)
                M=np.bincount(inv);S=np.bincount(inv,weights=l[m])
                rows.setdefault(f"{pr}__{pop}",[]).append(est(S,M,N))
    for key,rs in rows.items():
        Q,cv,ind=resumo(rs)
        a=R["resumo"][key]["N_ocupados"]["leitura_A"]["por_celula"][ch]
        bruto[f"{ch}|{key}"]={"Q":Q,"Q_grav":a["Q"],"cob":cv,"cob_grav":a["cobertura"],"ind":ind,"ind_grav":a["n_indefinidos_k_lt_2"],"N":int(N)}
sha=hashlib.sha256(open(__file__,"rb").read()).hexdigest()
json.dump({"gravados_recalculo":out,"bruto":bruto,"sha_script":sha},open(os.path.dirname(os.path.abspath(__file__))+"/recalc_saida.json","w"),indent=1)
print(json.dumps(out,indent=1));print(json.dumps(bruto,indent=1));print(sha)
