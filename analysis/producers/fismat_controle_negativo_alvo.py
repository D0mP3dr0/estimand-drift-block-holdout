"""FISMAT — controle negativo do alvo: qual o MAE de um preditor CONSTANTE?

Se um preditor degenerado (a constante otima = mediana) ja atinge o MAE que o
paper reporta, a metrica nao mede aprendizado. Mede-se nos dois tiles que
sustentam os resultados publicados. LEITURA APENAS.
Saida: dados/alvo/fismat_controle_negativo_alvo.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch

EVID = Path(__file__).resolve().parents[1]
OUT = EVID / "dados" / "alvo" / "fismat_controle_negativo_alvo.json"
DIRS = [Path(r"F:\TOPO_RF_DOWNLOAD_DRIVE\graph_data"),
        Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\graph_data")]
PISO = float(np.float32(-20.0*np.log10(np.float32(0.5))))
NOMES = ["path_loss", "shadow_margin", "diffraction", "rssi", "coverage"]

def achar(n):
    for d in DIRS:
        if (d/n).exists(): return d/n
    return None

def medir(tile):
    c,q = tile.rsplit("_",1)
    p = achar(f"transfer_dataset_{c}_v19_{q}_enriched.pt")
    if p is None: return {"tile":tile,"ERRO":"ausente"}
    d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    y = d["terrain"].y
    y = (y.numpy() if torch.is_tensor(y) else np.asarray(y)).astype(np.float32)
    r = {"tile":tile,"arquivo":str(p),"n_nos":int(y.shape[0]),"colunas":{}}
    for ci in range(y.shape[1]):
        col = y[:,ci].astype(np.float64)
        med = float(np.median(col)); mu = float(col.mean())
        r["colunas"][f"col{ci}_{NOMES[ci]}"] = {
            "media": mu, "mediana": med, "desvio_padrao": float(col.std()),
            "MAE_preditor_constante_mediana": float(np.abs(col-med).mean()),
            "RMSE_preditor_constante_media": float(np.sqrt(((col-mu)**2).mean())),
            "amplitude_interquartil": float(np.percentile(col,75)-np.percentile(col,25)),
        }
    c2 = y[:,2].astype(np.float64)
    r["difracao_contra_o_piso"] = {
        "piso_db": PISO,
        "MAE_predizendo_sempre_o_piso_db": float(np.abs(c2-PISO).mean()),
        "frac_dentro_de_0_1dB_do_piso": float((np.abs(c2-PISO)<=0.1).mean()),
        "frac_dentro_de_1dB_do_piso": float((np.abs(c2-PISO)<=1.0).mean()),
    }
    c1 = y[:,1].astype(np.float64)
    r["shadow_contra_o_teto"] = {
        "MAE_predizendo_sempre_40dB": float(np.abs(c1-40.0).mean()),
        "frac_exatamente_40": float((c1==np.float32(40.0)).mean()),
    }
    del d
    print(json.dumps(r,indent=1), flush=True)
    return r

res=[]
for t in ["lins_Q1","bauru_Q1"]:
    try: res.append(medir(t))
    except Exception as e: res.append({"tile":t,"ERRO":repr(e)})
    OUT.write_text(json.dumps({"tiles":res},indent=2,ensure_ascii=False),encoding="utf-8")
print("gravado:",OUT)
