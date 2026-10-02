#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""fismat_k_corrigido.py -- forum-fisico-matematico, 2026-09-25 (D6/F2, prop:inclusion).
Pergunta: o excesso de p_{i|te} na classe k=1 do 1.6 (0,168-0,170 vs 0,145) e
(i) artefato da classificacao (existe(gx,gy) do 1.6 nao checa 0<=gy<max_y: gy-1=-1
'encontra' o bloco (gx-1,max_y-1)), (ii) discretizacao (vizinho a <b da face mas nenhum
NO dele a <b), ou (iii) segunda ordem da regra sequencial (vizinho VAL cujos nos perto
de i foram aparados pelo TREINO)?
Metodo: mesma malha, mesmo assign_groups e MESMA split_uma_vez do 1.6 (import literal);
classe geometrica exata m(i) = no de blocos EXISTENTES da vizinhanca-8 (com checagem de
limites) a distancia euclidiana < b do no (face para laterais, vertice para diagonais);
forma fechada (k_te-1)_m/(N-1)_m. Lado a lado: a classe k do 1.6 (com o defeito), para
reproduzir 0,1696. Para m=1: retencao pelo papel do unico vizinho B (tr/va/te).
Seeds: os 100 do 1.6 + N_EXTRA novos (RandomState(20260928)).
"""
import argparse, hashlib, importlib.util, json, time
from math import perm
from pathlib import Path
import numpy as np

S16 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts/v3_1.6_2.3_estimando_formal.py")
spec = importlib.util.spec_from_file_location("s16", S16)
s16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s16)
G, B = s16.G, s16.B
ap = argparse.ArgumentParser()
ap.add_argument("--celula", default="bauru"); ap.add_argument("--extra", type=int, default=100); ap.add_argument("--out", required=True)
a = ap.parse_args()

d = json.load(open(s16.TREINOS_DIR / f"run_c0c1cf_{a.celula}_s42_Q1_g10b2.json")); geo = d["geometria"]
lon, lat = s16.build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"], geo["lat_min_deg"], geo["lat_max_deg"])
pos = s16.latlon_graus_para_metros(lon, lat) / 1000.0
del lon, lat
gid = s16.assign_groups(pos, G); grupos = np.unique(gid); n_g = len(grupos); gset = set(grupos.tolist())
max_y = int((pos[:, 1] / G).astype(int).max()) + 1
max_x = int((pos[:, 0] / G).astype(int).max()) + 1
k16, _ = s16.classe_k_e_dist_borda(pos, gid, gset, max_y)
gx = np.floor(pos[:, 0] / G).astype(np.int64); gy = np.floor(pos[:, 1] / G).astype(np.int64)
dx = pos[:, 0] - gx * G; dy = pos[:, 1] - gy * G


def existe_ok(i, j):
    ok = (i >= 0) & (i < max_x) & (j >= 0) & (j < max_y)
    return ok & np.isin(np.where(ok, i * max_y + j, -1), grupos)


cand = [(-1, 0, dx), (1, 0, G - dx), (0, -1, dy), (0, 1, G - dy),
        (-1, -1, np.hypot(dx, dy)), (-1, 1, np.hypot(dx, G - dy)), (1, -1, np.hypot(G - dx, dy)), (1, 1, np.hypot(G - dx, G - dy))]
m = np.zeros(gid.size, np.int8); nb1 = np.full(gid.size, -1, np.int64)
for sx, sy, dist in cand:
    hit = (dist < B) & existe_ok(gx + sx, gy + sy)
    m += hit.astype(np.int8)
    nb1[hit] = (gx[hit] + sx) * max_y + (gy[hit] + sy)
nb1[m != 1] = -1
# o defeito do 1.6, isolado: vizinho 'fantasma' por wrap de gy
fantasma = ((dy < B) & (gy == 0) & (gx >= 1) & np.isin(gx * max_y - 1, grupos))
cross = {f"k16={k}|m={j}": int(((k16 == k) & (m == j)).sum()) for k in range(5) for j in range(5) if ((k16 == k) & (m == j)).any()}
own = np.searchsorted(grupos, gid)
i1 = np.where(m == 1)[0]; nb1_idx = np.searchsorted(grupos, nb1[i1])

seeds100 = np.random.RandomState(20260927).randint(10**6, size=100).tolist(); assert seeds100 == list(s16.SEEDS)
seedsX = np.random.RandomState(20260928).randint(10**6, size=a.extra).tolist()
n_tr = int(round(0.70 * n_g)); n_va = int(round(0.15 * n_g))
cnt_te = np.zeros(gid.size, np.int16); cnt_ret = np.zeros(gid.size, np.int16)
pooled = {}; m1_papel = {}; por_seed = []
t0 = time.time()
for lote, seeds in (("s100", seeds100), ("extra", seedsX)):
    for s in seeds:
        m_te0, m_va0, m_te, m_va, kte, ktr = s16.split_uma_vez(pos, gid, grupos, n_g, s)
        emb = grupos.copy(); np.random.RandomState(s).shuffle(emb)
        papel = np.zeros(n_g, np.int8)
        papel[np.searchsorted(grupos, emb[n_tr:n_tr + n_va])] = 1; papel[np.searchsorted(grupos, emb[n_tr + n_va:])] = 2
        assert np.array_equal(papel[own] == 2, m_te0)
        if lote == "s100":
            cnt_te += m_te0; cnt_ret += m_te
        rs = {}
        for j in range(4):
            mm = m_te0 & (m == j)
            v = pooled.setdefault(f"{lote}|m={j}", [0, 0]); v[0] += int(mm.sum()); v[1] += int(m_te[mm].sum())
            rs[f"m{j}"] = [int(mm.sum()), int(m_te[mm].sum())]
        sel = m_te0[i1]; pB = papel[nb1_idx[sel]]; ret = m_te[i1][sel]
        for pb in (0, 1, 2):
            v = m1_papel.setdefault(f"{lote}|papelB={pb}", [0, 0]); v[0] += int((pB == pb).sum()); v[1] += int(ret[pB == pb].sum())
        por_seed.append({"lote": lote, "seed": int(s), **rs})
    print(lote, "ok", round(time.time() - t0), "s", flush=True)

ok = cnt_te > 0
pc = np.where(ok, cnt_ret / np.maximum(cnt_te, 1), np.nan)
def nodemean(mask):
    x = pc[mask & ok]; return float(np.nanmean(x)) if x.size else None
ff = {j: perm(19, j) / perm(131, j) for j in range(5)}
res = {"script": str(Path(__file__).resolve()), "sha256_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "sha256_script_1.6_importado": hashlib.sha256(S16.read_bytes()).hexdigest(), "celula": f"{a.celula}_Q1",
       "n_blocos": int(n_g), "max_x": max_x, "max_y": max_y, "n_nos": int(gid.size),
       "n_nos_vizinho_fantasma_1.6": int(fantasma.sum()),
       "tabela_cruzada_nos_k16_x_m": cross,
       "reproducao_1.6_media_nodal_k16": {str(k): nodemean(k16 == k) for k in range(3)},
       "media_nodal_p_cond_te_por_m_s100": {str(j): nodemean(m == j) for j in range(4)},
       "media_nodal_geral_s100": nodemean(np.ones_like(ok)),
       "frac_nos_por_m": {str(j): float((m == j).mean()) for j in range(4)},
       "forma_fechada_por_m": {str(j): ff[j] for j in range(4)},
       "pooled_por_lote_e_m": pooled, "m1_retencao_por_papel_B": m1_papel,
       "media_nodal_k16_1_restrita_a_nao_fantasma": nodemean((k16 == 1) & ~fantasma),
       "media_nodal_k16_1_restrita_a_fantasma": nodemean((k16 == 1) & fantasma),
       "por_seed": por_seed, "tempo_s": time.time() - t0}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("n_nos_vizinho_fantasma_1.6", "reproducao_1.6_media_nodal_k16", "media_nodal_p_cond_te_por_m_s100",
                                      "frac_nos_por_m", "forma_fechada_por_m", "m1_retencao_por_papel_B", "media_nodal_k16_1_restrita_a_nao_fantasma")}, indent=1))
