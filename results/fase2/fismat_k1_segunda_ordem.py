#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""fismat_k1_segunda_ordem.py -- forum-fisico-matematico, 2026-09-25 (D6/F2, prop:inclusion).
Pergunta: por que p_{i|te} empirico na classe k=1 (1.6: 0,168-0,170) excede a forma
fechada (k_te-1)/(N-1)=0,145? Decompoe, para nos k=1 de blocos de teste, a retencao
pelo papel do bloco vizinho B (treino/val/teste) e, no caso B=val, pelo no de blocos
de TREINO na vizinhanca-8 de B. Reusa LITERALMENTE split_uma_vez e classe_k_e_dist_borda
do script 1.6 (import por importlib, sem editar) -> mesma regra sequencial congelada.
Seeds: os 100 do 1.6 (RandomState(20260927)) + N_EXTRA novos (RandomState(20260928)),
para medir o erro Monte Carlo do proprio 0,169 (as 4 celulas do 1.6 usam as MESMAS
permutacoes e nao sao replicas independentes).
Uso: <venv>/python fismat_k1_segunda_ordem.py --celula bauru --extra 100 --out <json>
"""
import argparse, hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np

S16 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts/v3_1.6_2.3_estimando_formal.py")
spec = importlib.util.spec_from_file_location("s16", S16)
s16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s16)
G, B = s16.G, s16.B

ap = argparse.ArgumentParser()
ap.add_argument("--celula", default="bauru")
ap.add_argument("--extra", type=int, default=100)
ap.add_argument("--out", required=True)
a = ap.parse_args()

d = json.load(open(s16.TREINOS_DIR / f"run_c0c1cf_{a.celula}_s42_Q1_g10b2.json"))
geo = d["geometria"]
lon, lat = s16.build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"], geo["lat_min_deg"], geo["lat_max_deg"])
pos = s16.latlon_graus_para_metros(lon, lat) / 1000.0
gid = s16.assign_groups(pos, G)
grupos = np.unique(gid)
n_g = len(grupos)
gset = set(grupos.tolist())
max_y = int((pos[:, 1] / G).astype(int).max()) + 1
k_no, _ = s16.classe_k_e_dist_borda(pos, gid, gset, max_y)
gx = np.floor(pos[:, 0] / G).astype(np.int64)
gy = np.floor(pos[:, 1] / G).astype(np.int64)
dx = pos[:, 0] - gx * G
dy = pos[:, 1] - gy * G


def ex(i, j):
    return np.isin(i * max_y + j, grupos)


k1 = np.where(k_no == 1)[0]
gx1, gy1, dx1, dy1 = gx[k1], gy[k1], dx[k1], dy[k1]
nb = np.full(k1.size, -1, np.int64)
dist_lat = np.zeros(k1.size)
for (m, sx, sy, dl) in [((dx1 < B) & ex(gx1 - 1, gy1), -1, 0, dx1),
                        (((G - dx1) < B) & ex(gx1 + 1, gy1), 1, 0, G - dx1),
                        ((dy1 < B) & ex(gx1, gy1 - 1), 0, -1, dy1),
                        (((G - dy1) < B) & ex(gx1, gy1 + 1), 0, 1, G - dy1)]:
    nb[m] = (gx1[m] + sx) * max_y + (gy1[m] + sy)
    dist_lat[m] = dl[m]
assert (nb >= 0).all()
nb_idx = np.searchsorted(grupos, nb)
own_idx = np.searchsorted(grupos, gid[k1])
bx, by = grupos // max_y, grupos % max_y
viz8 = []
for t in range(n_g):
    ids = [(bx[t] + u) * max_y + (by[t] + v) for u in (-1, 0, 1) for v in (-1, 0, 1) if (u, v) != (0, 0)]
    viz8.append(np.searchsorted(grupos, [q for q in ids if q in gset]).astype(np.int64))
# larguras reais dos blocos (blocos finos de borda de dominio)
ordem = np.argsort(gid, kind="stable")
cortes = np.searchsorted(gid[ordem], grupos)
bw = np.zeros(n_g); bh = np.zeros(n_g)
for t in range(n_g):
    fim = cortes[t + 1] if t + 1 < n_g else gid.size
    sl = ordem[cortes[t]:fim]
    bw[t] = np.ptp(pos[sl, 0]); bh[t] = np.ptp(pos[sl, 1])
fino_B = (np.minimum(bw, bh) < B)[nb_idx]

seeds100 = np.random.RandomState(20260927).randint(10**6, size=100).tolist()
assert seeds100 == list(s16.SEEDS)
seedsX = np.random.RandomState(20260928).randint(10**6, size=a.extra).tolist()
cnt_te = np.zeros(k1.size, np.int32)
cnt_ret = np.zeros(k1.size, np.int32)
tab = {}
por_seed = []
t0 = time.time()
n_tr = int(round(0.70 * n_g)); n_va = int(round(0.15 * n_g))
for lote, seeds in (("s100", seeds100), ("extra", seedsX)):
    for s in seeds:
        m_te0, m_va0, m_te, m_va, kte, ktr = s16.split_uma_vez(pos, gid, grupos, n_g, s)
        emb = grupos.copy(); np.random.RandomState(s).shuffle(emb)
        papel = np.zeros(n_g, np.int8)
        papel[np.searchsorted(grupos, emb[n_tr:n_tr + n_va])] = 1
        papel[np.searchsorted(grupos, emb[n_tr + n_va:])] = 2
        sel = papel[own_idx] == 2
        assert np.array_equal(sel, m_te0[k1])
        ret = m_te[k1][sel]
        pB = papel[nb_idx[sel]]
        ntr = np.array([int((papel[viz8[t]] == 0).sum()) for t in range(n_g)])
        nB = ntr[nb_idx[sel]]
        fB = fino_B[sel]
        if lote == "s100":
            cnt_te[sel] += 1
            cnt_ret[sel] += ret.astype(np.int32)
        for pb in (0, 1, 2):
            for fino in (False, True):
                for n_ in np.unique(nB[(pB == pb) & (fB == fino)]):
                    mm = (pB == pb) & (nB == n_) & (fB == fino)
                    key = f"{lote}|papelB={pb}|finoB={int(fino)}|ntreino_viz8_B={int(n_)}"
                    v = tab.setdefault(key, [0, 0]); v[0] += int(mm.sum()); v[1] += int(ret[mm].sum())
        por_seed.append({"lote": lote, "seed": int(s), "n_k1_te": int(sel.sum()), "ret_k1": int(ret.sum()),
                         "n_B_te": int((pB == 2).sum()), "n_B_va": int((pB == 1).sum()),
                         "ret_B_te": int(ret[pB == 2].sum()), "ret_B_va": int(ret[pB == 1].sum()),
                         "ret_B_tr": int(ret[pB == 0].sum())})
    print(lote, "ok", round(time.time() - t0), "s", flush=True)

ok = cnt_te > 0
res = {"script": str(Path(__file__).resolve()),
       "sha256_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "sha256_script_1.6_importado": hashlib.sha256(S16.read_bytes()).hexdigest(),
       "celula": f"{a.celula}_Q1", "n_blocos": int(n_g), "n_tr_va_te": [n_tr, n_va, n_g - n_tr - n_va],
       "n_nos_k1": int(k1.size), "n_nos_k1_com_B_fino": int(fino_B.sum()),
       "media_nodal_p_cond_te_k1_s100": float((cnt_ret[ok] / cnt_te[ok]).mean()),
       "forma_fechada_k1": 19 / 131,
       "tabela_retencao": {"chave": "lote|papel de B (0 tr,1 va,2 te)|B com largura<b|n blocos de treino na viz-8 de B",
                           "valor": "[n no-sorteios k=1 em teste, n retidos]", "dados": tab},
       "larguras_blocos_km": {"min_largura_x": float(bw.min()), "min_altura_y": float(bh.min()),
                              "n_blocos_x_lt_b": int((bw < B).sum()), "n_blocos_y_lt_b": int((bh < B).sum()),
                              "larguras_x_unicas_arred": sorted(set(np.round(bw, 2).tolist()))[:8],
                              "alturas_y_unicas_arred": sorted(set(np.round(bh, 2).tolist()))[:8]},
       "dist_lateral_k1_km": {"min": float(dist_lat.min()), "max": float(dist_lat.max())},
       "por_seed": por_seed, "tempo_s": time.time() - t0}
json.dump(res, open(a.out, "w"), indent=1)
print("gravado", a.out)
