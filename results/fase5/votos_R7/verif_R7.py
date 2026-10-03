import json,glob,hashlib,numpy as np
G="/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/"
out={};st=set();
def mae(f):
    z=np.load(f);a=z["target"] if "target" in z.files else z["alvo"];p=z["pred"]
    m=a[:,0]<299
    return float(np.abs(a[m,3]-p[m,3]).mean()),int(m.sum()),a.shape[0],hashlib.sha256(a.tobytes()).hexdigest()[:12]
for cel in ["bauru_Q1","bauru_Q3"]:
  for mod in ["gnn","mlp"]:
    r={}
    for lote,ids,tag in (("A3",range(42,47),"s"),("A4",range(101,106),"ss")):
        v=[];h=set();seeds=set();spl=set()
        for i in ids:
            d=f"{G}{lote}/{mod}_v3_{lote.lower()}_{cel}_{tag}{i}/"
            m=mae(glob.glob(d+"predicoes_*.npz")[0]);v.append(m[0]);h.add((m[1],m[2],m[3]))
            j=json.load(open(glob.glob(d+"run_*.json")[0]));seeds.add(j.get("seed"));spl.add(j.get("split_seed"));st.add(j.get("status"))
        r[lote]={"v":v,"dp1":float(np.std(v,ddof=1)),"dp0":float(np.std(v)),"alvos_distintos":len(h),"seeds":sorted(seeds),"split_seeds":sorted(spl),"n":list(h)[0][:2]}
    r["razao1"]=r["A4"]["dp1"]/r["A3"]["dp1"];r["razao0"]=r["A4"]["dp0"]/r["A3"]["dp0"]
    out[f"{cel}_{mod}"]=r
    print(cel,mod,{k:(r[k]["dp1"],r[k]["dp0"],r[k]["alvos_distintos"],r[k]["seeds"],r[k]["split_seeds"]) for k in("A3","A4")},r["razao1"],r["razao0"])
out["status_vistos"]=sorted(map(str,st))
json.dump(out,open("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/fase5/votos_R7/calc_R7.json","w"),indent=1)
