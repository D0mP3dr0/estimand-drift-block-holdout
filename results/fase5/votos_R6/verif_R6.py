import json,sys,os,hashlib,importlib.util
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
B="/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics"
d=json.load(open(B+"/_v3_2026-09-25/fase5/R6_por_sorteio.json"))["celulas"]
P=["c-100","c-110","c-120","c-130","mediana_validos_treino"]
out={"a":{}}
for k in P:
    dps=[];meds=[];exc=0;dpw=[]
    for ch,L in d.items():
        v=[s["constantes"][k]["mae_validos"] for s in L if s["n_validos_teste"]>0 and s["constantes"][k]["mae_validos"] is not None]
        w=[s["n_validos_teste"] for s in L if s["n_validos_teste"]>0 and s["constantes"][k]["mae_validos"] is not None]
        exc+=len(L)-len(v)
        dps.append(np.std(v,ddof=1));meds.append(np.mean(v))
        v=np.array(v);w=np.array(w,float);m=(v*w).sum()/w.sum()
        dpw.append(np.sqrt((w*(v-m)**2).sum()/w.sum()))
    out["a"][k]={"n_cel":len(dps),"media_dp":float(np.mean(dps)),"dp_cel":float(np.std(meds,ddof=1)),"razao":float(np.mean(dps)/np.std(meds,ddof=1)),"excluidos":exc,"razao_pond_aprox_dpPop":float(np.mean(dpw)/np.std(meds,ddof=1))}
# b
sp=importlib.util.spec_from_file_location("o",B+"/scripts/v3_2.1_3.1_deriva_calibracao.py");o=importlib.util.module_from_spec(sp);sp.loader.exec_module(o)
import torch
seeds=json.load(open(B+"/_v3_2026-09-25/fase5/R6_por_sorteio.json"))["sementes"]
out["b"]={}
for ch in ["bauru_Q1","lins_Q1","sorocaba_Q4"]:
    cid,q=ch.split("_")
    rf,_=o.carregar_tensor(o.TENSOR_DIR/f"transfer_dataset_{cid}_v19_{q}_enriched_cftudo.pt")
    y=torch.as_tensor(rf["terrain"].y).float().numpy().astype(np.float64)
    pos=o.latlon_graus_para_metros(torch.as_tensor(rf["terrain"].pos).float())
    rssi=y[:,3];sent=y[:,0]>=299.0
    maxdiff=0;res={}
    for k in P: res[k]=[]
    for i,s in enumerate(seeds):
        p=o.split_espacial_3vias(pos,10.0,2.0,(0.70,0.15,0.15),s)
        tr,te=p["train"],p["test"]
        rv=rssi[te][~sent[te]]
        trv=rssi[tr][~sent[tr]]
        if rv.size==0: 
            assert d[ch][i]["n_validos_teste"]==0; continue
        for k in P:
            c=float(np.median(trv)) if k=="mediana_validos_treino" else float(k[1:])
            m=float(np.mean(np.abs(c-rv)))
            g=d[ch][i]["constantes"][k]["mae_validos"]
            maxdiff=max(maxdiff,abs(m-g)); res[k].append(m)
        assert d[ch][i]["n_validos_teste"]==rv.size
    out["b"][ch]={"max_absdiff_vs_gravado":maxdiff,"n_sorteios":len(res["c-110"]),"dp_por_nivel":{k:float(np.std(v,ddof=1)) for k,v in res.items()},"media_por_nivel":{k:float(np.mean(v)) for k,v in res.items()}}
    print(ch,out["b"][ch],flush=True)
r=[out["a"][k]["razao"] for k in P]
out["c"]={"todas_ge_2":all(x>=2 for x in r),"alguma_lt_1.5":any(x<1.5 for x in r),"min":min(r),"max":max(r),"mediana_lt_1":r[4]<1}
json.dump(out,open(B+"/_v3_2026-09-25/fase5/votos_R6/calc_R6.json","w"),indent=1)
print(json.dumps(out["a"],indent=1),out["c"])
