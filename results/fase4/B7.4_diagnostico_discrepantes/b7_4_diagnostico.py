"""B7.4 (roadmap v3-12) -- diagnostico dos dois discrepantes apontados pelo revisor cego.
Autor: forum-eng-ia, 01/10/2026. So leitura + CPU, sem treino. Numeros PROVISORIOS ate contra-auditoria.

(a) Sorocaba Q3: dp entre sorteios do MAE nos VALIDOS do FSPL calibrado (a) = 12,64 dB (Tab. 3),
    contra 1,4-4,9 nas outras celulas. Hipotese H_a: os nos validos com alvo fora da faixa do
    decodificador (PL > 200 dB ou RSSI < -150 dBm) entram no teste em proporcao que varia com o
    sorteio e, como o FSPL calibrado preve RSSI tipico, cada um pesa dezenas de dB no MAE.
    Teste: refazer os 60 sorteios aleatorios (mesmo gerador de sementes, mesma particao, mesma
    calibracao do script v3_2.1_3.1_deriva_calibracao.py, importado sem edicao), conferir o MAE
    gravado e recalcular o dp SEM os nos fora de faixa. Criterio (fixado antes de rodar):
    H_a CONFIRMADA se (i) o recalculo reproduz o gravado (|dif| < 1e-6 dB em todos os sorteios
    usaveis) e (ii) o dp sem fora-de-faixa cai para dentro da faixa das outras 15 celulas
    (<= 4,93 dB, maximo da Tab. 3 fora Sorocaba Q3). Senao: NAO confirmada.
    Controle: fracao de validos fora de faixa na celula inteira, nas 16 celulas.

(b) Campinas Q3: GNN pior que MLP nos validos (mediana +0,74 dB, A3, 5 seeds, split 42, g=5 km).
    Hipoteses testadas (criterio de cada uma fixado antes de rodar):
    H_b1 decodificador: validos de teste fora da faixa do decodificador explicam o sinal.
         Confirma se restringir aos nos dentro da faixa troca o sinal da mediana.
    H_b2 acesso a antena: o contraste concentra-se num estrato de grau de antena (arestas
         antena->terreno do no) ou de distancia a antena mais proxima, diferente das outras celulas.
         Confirma se em Campinas Q3 a GNN perde (mediana > 0) num estrato e ganha nos demais, e
         se esse estrato tem peso no teste de Campinas Q3 maior que nas outras sete celulas.
    Descritivo (nao e hipotese causal): vies medio com sinal de cada braco.
"""
from __future__ import annotations
import glob, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

RAIZ = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
V3 = RAIZ / "_v3_2026-09-25"
OUT = V3 / "fase4" / "B7.4_diagnostico_discrepantes"
SCRIPT_21 = RAIZ / "scripts" / "v3_2.1_3.1_deriva_calibracao.py"
PARCIAL_60 = V3 / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json"
TENSOR_DIR = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
A3 = V3 / "gpu" / "A3"
PL_MAX_DEC, RSSI_MIN_DEC = 200.0, -150.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


spec = importlib.util.spec_from_file_location("deriva21", SCRIPT_21)
d21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d21)          # so define funcoes; main() nao roda

out = {"script": __file__, "script_sha256": sha(__file__),
       "fontes": {"v3_2.1_3.1_deriva_calibracao.py": sha(SCRIPT_21),
                  "_v3_2.1_3.1_parcial_16x60rnd.json": sha(PARCIAL_60)},
       "natureza": "diagnostico B7.4, so leitura+CPU; PROVISORIO ate contra-auditoria"}


def carregar(cel):
    cid, q = cel.split("_")
    # weights_only=False: HeteroData do proprio ETL (mesma carga de d21.carregar_tensor)
    return torch.load(TENSOR_DIR / f"transfer_dataset_{cid}_v19_{q}_enriched_cftudo.pt",
                      mmap=True, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- controle: fracao fora de faixa
ctrl = {}
for cid in d21.CIDADES:
    for q in d21.QUADRANTES:
        cel = f"{cid}_{q}"
        rf = carregar(cel)
        y = torch.as_tensor(rf["terrain"].y).float().numpy()
        v = y[:, 0] < d21.PL_TARGET_MAX_VALID
        fora = v & ((y[:, 0] > PL_MAX_DEC) | (y[:, 3] < RSSI_MIN_DEC))
        ctrl[cel] = {"n_validos": int(v.sum()), "n_validos_fora_faixa": int(fora.sum()),
                     "frac_validos_fora_faixa": float(fora.sum() / max(v.sum(), 1)),
                     "rssi_validos_min": float(y[v, 3].min()), "pl_validos_max": float(y[v, 0].max())}
        del rf, y
        log(f"controle {cel}: {ctrl[cel]['frac_validos_fora_faixa']:.4f}")
out["controle_fracao_fora_faixa_celula_inteira"] = ctrl

# ---------------------------------------------------------------- (a) Sorocaba Q3, 60 sorteios
gravado = json.load(open(PARCIAL_60))
seeds, desc = d21.parse_seeds("aleatorios:60:20260926")
assert desc == gravado["seeds_descricao"], (desc, gravado["seeds_descricao"])
grav_cel = {s["split_seed"]: s for s in gravado["celulas"]["sorocaba_Q3"]["por_sorteio"]}

rf = carregar("sorocaba_Q3")
ty = torch.as_tensor(rf["terrain"].y).float().clone().numpy()
dist_all = torch.as_tensor(rf["terrain"].dist_nearest_m).float().clone().numpy()
pos_m = d21.latlon_graus_para_metros(torch.as_tensor(rf["terrain"].pos).float().clone())
del rf
rssi, pl = ty[:, 3].astype(np.float64), ty[:, 0].astype(np.float64)
sent = pl >= d21.PL_TARGET_MAX_VALID
fora_all = (~sent) & ((pl > PL_MAX_DEC) | (rssi < RSSI_MIN_DEC))
fspl = d21.MODELOS["fspl"]
linhas = []
for s in seeds:
    g = grav_cel[s]
    if g.get("status") != "ok" or g["n_pop_teste"]["validos"] == 0:
        continue
    parts = d21.split_espacial_3vias(pos_m, d21.GRID_KM, d21.BUFFER_KM, d21.FRACS, s)
    tr, te = parts["train"], parts["test"]
    p_tx = float(np.median(rssi[tr] + fspl(dist_all[tr])))
    pred = p_tx - fspl(dist_all[te])
    err = np.abs(pred - rssi[te])
    val = ~sent[te]; fo = fora_all[te]; dentro = val & ~fo
    mae_val = float(err[val].mean())
    linhas.append({"split_seed": s, "n_validos": int(val.sum()), "n_fora_faixa": int(fo.sum()),
                   "frac_fora_entre_validos": float(fo.sum() / val.sum()),
                   "mae_fspl_validos_db": mae_val,
                   "mae_fspl_validos_gravado_db": g["mae_modelo_a_contaminado_teste"]["fspl"]["validos"],
                   "mae_fspl_validos_sem_fora_faixa_db": float(err[dentro].mean()) if dentro.any() else None,
                   "mae_fspl_so_fora_faixa_db": float(err[fo].mean()) if fo.any() else None})
    log(f"(a) seed {s}: {mae_val:.3f} vs {linhas[-1]['mae_fspl_validos_gravado_db']:.3f}")
v_all = np.array([l["mae_fspl_validos_db"] for l in linhas])
v_in = np.array([l["mae_fspl_validos_sem_fora_faixa_db"] for l in linhas if l["mae_fspl_validos_sem_fora_faixa_db"] is not None])
frac = np.array([l["frac_fora_entre_validos"] for l in linhas])
difmax = float(max(abs(l["mae_fspl_validos_db"] - l["mae_fspl_validos_gravado_db"]) for l in linhas))
outras_max = 4.929   # Tab. 3, maior dp do FSPL (a) validos fora Sorocaba Q3 (Sorocaba Q4); conferido abaixo
dp_outras = {}
for cel, c in gravado["celulas"].items():
    xs = [s["mae_modelo_a_contaminado_teste"]["fspl"]["validos"] for s in c["por_sorteio"]
          if s.get("status") == "ok" and s["n_pop_teste"]["validos"] > 0]
    dp_outras[cel] = float(np.std(xs, ddof=1))
outras_max_calc = max(v for k, v in dp_outras.items() if k != "sorocaba_Q3")
assert abs(outras_max_calc - outras_max) < 1e-3, outras_max_calc
out["a_sorocaba_Q3"] = {
    "n_sorteios_usaveis": len(linhas), "reproducao_max_abs_dif_db": difmax,
    "dp_mae_validos_db": float(v_all.std(ddof=1)),
    "dp_mae_validos_sem_fora_faixa_db": float(v_in.std(ddof=1)),
    "mediana_mae_validos_sem_fora_faixa_db": float(np.median(v_in)),
    "frac_fora_entre_validos_teste_min_med_max": [float(frac.min()), float(np.median(frac)), float(frac.max())],
    "corr_pearson_frac_fora_vs_mae_validos": float(np.corrcoef(frac, v_all)[0, 1]),
    "dp_fspl_validos_outras_celulas_db": dp_outras, "limite_outras_celulas_db": outras_max_calc,
    "criterio": "confirma se reproducao < 1e-6 dB e dp sem fora-de-faixa <= limite_outras_celulas",
    "por_sorteio": linhas}
ok_rep = difmax < 1e-6
out["a_sorocaba_Q3"]["veredito"] = ("H_a CONFIRMADA" if ok_rep and v_in.std(ddof=1) <= outras_max_calc
                                    else ("NAO VERIFICAVEL (recalculo nao reproduz)" if not ok_rep else "H_a NAO CONFIRMADA"))
del ty, dist_all, pos_m, rssi, pl, sent, fora_all

# ---------------------------------------------------------------- (b) Campinas Q3 x 7 celulas A3
CELS = ["bauru_Q1", "bauru_Q3", "campinas_Q1", "campinas_Q3", "lins_Q1", "lins_Q3", "sorocaba_Q1", "sorocaba_Q3"]
res_b = {}
for cel in CELS:
    rf = carregar(cel)
    ei = rf["antenna", "propagates_to", "terrain"].edge_index
    grau = torch.bincount(torch.as_tensor(ei[1]), minlength=rf["terrain"].y.shape[0]).numpy()
    dist = torch.as_tensor(rf["terrain"].dist_nearest_m).float().numpy()
    del rf, ei
    porseed, cont_grau, cont_dist, cont_in, vies = [], {}, {}, [], {"gnn": [], "mlp": []}
    for s in range(42, 47):
        g = np.load(glob.glob(str(A3 / f"gnn_v3_a3_{cel}_s{s}" / "predicoes_*.npz"))[0])
        m = np.load(glob.glob(str(A3 / f"mlp_v3_a3_{cel}_s{s}" / "predicoes_*.npz"))[0])
        assert np.array_equal(g["idx_global"], m["idx_global"])
        t = g["target"].astype(np.float64); v = t[:, 0] < 299.0; y = t[v, 3]
        idx = g["idx_global"][v]
        eg = g["pred"][v, 3].astype(np.float64) - y; em = m["pred"][v, 3].astype(np.float64) - y
        d = np.abs(eg) - np.abs(em)
        porseed.append(float(d.mean()))
        inr = ~((t[v, 0] > PL_MAX_DEC) | (y < RSSI_MIN_DEC))
        cont_in.append(float(d[inr].mean()))
        vies["gnn"].append(float(eg.mean())); vies["mlp"].append(float(em.mean()))
        gr = grau[idx]
        for k in range(0, 6):
            mk = gr == k
            if mk.any():
                cont_grau.setdefault(str(k), {"peso": float(mk.mean()), "dif": []})["dif"].append(float(d[mk].mean()))
        qd = np.quantile(dist[idx], [0.25, 0.5, 0.75])
        lab = np.digitize(dist[idx], qd)
        for k in range(4):
            mk = lab == k
            cont_dist.setdefault(f"Q{k+1}", {"dist_max_m": float(dist[idx][mk].max()), "dif": []})["dif"].append(float(d[mk].mean()))
    res_b[cel] = {
        "mediana_contraste_validos_db": float(np.median(porseed)),
        "mediana_contraste_validos_dentro_faixa_db": float(np.median(cont_in)),
        "vies_medio_mediana_db": {k: float(np.median(x)) for k, x in vies.items()},
        "contraste_por_grau_antena": {k: {"peso_no_teste_valido": x["peso"], "mediana_dif_db": float(np.median(x["dif"]))} for k, x in cont_grau.items()},
        "contraste_por_quartil_distancia": {k: {"dist_max_m_seed46": x["dist_max_m"], "mediana_dif_db": float(np.median(x["dif"]))} for k, x in cont_dist.items()},
    }
    log(f"(b) {cel}: {res_b[cel]['mediana_contraste_validos_db']:+.3f}")
out["b_contraste_gnn_menos_mlp_validos_A3"] = res_b
c3 = res_b["campinas_Q3"]
hb1 = (np.sign(c3["mediana_contraste_validos_dentro_faixa_db"]) != np.sign(c3["mediana_contraste_validos_db"]))
estr_perde = [k for k, x in c3["contraste_por_grau_antena"].items() if x["mediana_dif_db"] > 0]
estr_ganha = [k for k, x in c3["contraste_por_grau_antena"].items() if x["mediana_dif_db"] < 0]
peso_campinas = {k: c3["contraste_por_grau_antena"][k]["peso_no_teste_valido"] for k in estr_perde}
peso_outras_max = {k: max(res_b[c]["contraste_por_grau_antena"].get(k, {"peso_no_teste_valido": 0})["peso_no_teste_valido"]
                          for c in CELS if c != "campinas_Q3") for k in estr_perde}
hb2 = bool(estr_perde) and bool(estr_ganha) and all(peso_campinas[k] > peso_outras_max[k] for k in estr_perde)
out["b_veredito"] = {
    "H_b1_decodificador": "CONFIRMADA" if hb1 else "DESCARTADA (o sinal nao muda dentro da faixa do decodificador)",
    "H_b2_grau_antena": ("CONFIRMADA" if hb2 else "NAO CONFIRMADA"),
    "estratos_de_grau_em_que_a_GNN_perde_em_campinas_Q3": estr_perde,
    "peso_desses_estratos_campinas_Q3": peso_campinas, "peso_maximo_nas_outras_celulas": peso_outras_max}
json.dump(out, open(OUT / "b7_4_diagnostico.json", "w"), indent=1, ensure_ascii=False)
log("gravado")
print(json.dumps({"a": out["a_sorocaba_Q3"]["veredito"], "b": out["b_veredito"]}, indent=1, ensure_ascii=False))
