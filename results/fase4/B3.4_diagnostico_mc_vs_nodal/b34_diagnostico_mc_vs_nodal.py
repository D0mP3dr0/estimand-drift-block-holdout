#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B3.4 (roadmap v3-12) -- forum-geoprocessamento, 01/10/2026. SO LEITURA, CPU.

Pergunta: o que explica
  (P1) a diferenca de ~1,7 % (teste, g=10) entre o "MC nodal" (fase1/1.3, 50 seeds 1000-1049)
       e a retencao nodal media de 200 sorteios (fase1/1.8, 1.8b), main.tex:534; e
  (P2) a diferenca do MC de reticulado (Tab. 2, colunas mc_v2_*) contra a nodal de 200 sorteios
       (redacao/tables_v3/T2_retencao_v3.csv)?

Causas candidatas (pista do parecer geo G2 + inspecao dos scripts):
  S  amostragem dos sorteios: 1.3 e 1.8 rodam o MESMO codigo (split_espacial_3vias_exato /
     cKDTree sobre build_synthetic_grid) com conjuntos de seeds diferentes; o MC da Tab. 2 usa
     2500 permutacoes proprias. Como as 16 celulas compartilham a permutacao por seed, o erro de
     amostragem e' comum a todas as celulas (nao se reduz pela media entre celulas).
  W  largura: a Tab. 2 usa, para cada cidade, um retangulo com a largura norte de Q1/Q2
     (montecarlo_retencao_r3_g5.py CIDADES_EXT), enquanto a nodal da Tab. 2 e' a media de Q1 e Q3.
  Z  trapezio: com cos(phi) por no, a ultima coluna de blocos e' um trapezio (largura varia por linha).
  G  granularidade: retencao nodal conta nos a ~30 m; o aparo remove nos a distancia < b do NO de
     treino mais proximo (nao da face do bloco), o que estreita a faixa efetiva.
  V  aparo do teste contra val: o MC (e o metodo de mascara abaixo) trata um bloco de val retido
     como inteiro; o codigo real apara contra os NOS de val retidos.

Metodo (pareado por seed -- mesma permutacao em todas as variantes):
  1. Para cada geometria distinta (4 cidades x {Q1,Q3}) e g in {10,5}: grade sintetica 3600x3600
     (build_synthetic_grid + latlon_graus_para_metros, copia literal do codigo congelado);
     para cada bloco e cada um dos 8 vizinhos, mascara por no "distancia < b ao no mais proximo
     do vizinho" (cKDTree); histograma de mascaras por bloco -> retencao NODAL exata por seed com
     o aparo por blocos inteiros (variante Vnode).
  2. Retencao CONTINUA (formula de area exata, area_bloco_continua) sobre o mesmo reticulado e a
     mesma permutacao, em tres geometrias: Rt2 (retangulo com a largura da Tab. 2), Rown (retangulo
     com a largura norte da propria celula), Ztrap (ultima coluna com a largura do trapezio no
     meio de cada linha de blocos).
  3. Seeds: as 200 de 1.8 (RandomState(20260926).randint(1e6, 200)) e as 50 de 1.3 (1000-1049).
  4. Cadeia por cidade (media de Q1 e Q3, como a Tab. 2):
       MC_T2 --S--> Rt2(200) --W--> Rown(200) --Z--> Ztrap(200) --G--> Vnode(200) --V--> nodal200
     e para P1: Vnode(1000-1049) - Vnode(200) contra o publicado 1.3 - 1.8.
Validacao do metodo: Vnode(val) deve reproduzir nodal_va_media de 1.8/1.8b (val nao depende de V).

Uso: /trabalho/ambientes/s33_amb_virtual/.venv/bin/python b34_diagnostico_mc_vs_nodal.py <out.json>
"""
import csv, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
V3 = BASE / "_v3_2026-09-25"
F18 = V3 / "fase1/1.8_referencia_nodal_200seeds.json"
F18b = V3 / "fase1/1.8b_referencia_nodal_200seeds_g5.json"
F13 = V3 / "fase1/1.3_mc_nodal_vs_blocos.json"
F11 = V3 / "fase1/1.1_geo_celulas_16.json"
T2 = V3 / "redacao/tables_v3/T2_retencao_v3.csv"
GEO = BASE / "scripts/varredura_split_geometria.py"
MCR = BASE / "scripts/montecarlo_retencao_r3_g5.py"
B = 2.0
FR = (0.70, 0.15)
OFF = [("L", -1, 0), ("R", 1, 0), ("Bo", 0, -1), ("T", 0, 1),
       ("BL", -1, -1), ("BR", 1, -1), ("TL", -1, 1), ("TR", 1, 1)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def imp(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


geo = imp(GEO, "geo")
mcr = imp(MCR, "mcr")
area_cont = mcr.area_bloco_continua


def counts(n_g):
    n_tr = max(1, int(round(FR[0] * n_g))); n_va = max(1, int(round(FR[1] * n_g)))
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2); n_va = 1
    return n_tr, n_va, n_g - n_tr - n_va


def labels_for_seed(seed, n_g):
    emb = np.arange(n_g); np.random.RandomState(seed).shuffle(emb)
    n_tr, n_va, n_te = counts(n_g)
    lab = np.empty(n_g, np.int8)
    lab[emb[:n_tr]] = 0; lab[emb[n_tr:n_tr + n_va]] = 1; lab[emb[n_tr + n_va:]] = 2
    return lab


def preparar_nodal(cel, g):
    """mascaras por no -> tabela F[bloco][flags] = n nos retidos se 'flags' aparam."""
    lon, lat = geo.build_synthetic_grid(cel["bbox_lon_min_deg"], cel["bbox_lon_max_deg"],
                                        cel["bbox_lat_min_deg"], cel["bbox_lat_max_deg"])
    pos = geo.latlon_graus_para_metros(lon, lat) / 1000.0
    gid = geo.assign_groups(pos, g)
    gx = (pos[:, 0] / g).astype(int); gy = (pos[:, 1] / g).astype(int)
    nx, ny = int(gx.max()) + 1, int(gy.max()) + 1
    assert np.array_equal(np.unique(gid), np.arange(nx * ny)), "ids de bloco nao contiguos"
    order = np.argsort(gid, kind="stable"); starts = np.searchsorted(gid[order], np.arange(nx * ny + 1))
    idx = [order[starts[k]:starts[k + 1]] for k in range(nx * ny)]
    trees = [cKDTree(pos[ix], compact_nodes=False, balanced_tree=False) for ix in idx]
    n_tot = np.array([len(ix) for ix in idx])
    F = np.zeros((nx * ny, 256), np.int64)
    flags_all = np.arange(256)
    for k in range(nx * ny):
        i, j = k // ny, k % ny
        mask = np.zeros(len(idx[k]), np.int64)
        for bit, (_, di, dj) in enumerate(OFF):
            ii, jj = i + di, j + dj
            if 0 <= ii < nx and 0 <= jj < ny:
                d, _ = trees[ii * ny + jj].query(pos[idx[k]], k=1, distance_upper_bound=B, workers=-1)
                mask |= (d < B).astype(np.int64) << bit
        h = np.bincount(mask, minlength=256)
        nz = np.nonzero(h)[0]
        # F[flags] = sum_{m: m & flags == 0} h[m]
        F[k] = ((nz[None, :] & flags_all[:, None]) == 0).astype(np.int64) @ h[nz]
    # largura do trapezio na ultima coluna, por linha de blocos (no meio da linha)
    # borda leste: x do no mais a leste em cada linha de nos (grade 3600x3600, linhas = latitude),
    # media dentro de cada linha de blocos (largura do trapezio no meio da linha)
    east = pos.reshape(3600, 3600, 2)[:, -1, :]
    ey = (east[:, 1] / g).astype(int)
    x_e = np.array([east[ey == j, 0].mean() for j in range(ny)])
    return dict(nx=nx, ny=ny, F=F, n_tot=n_tot, x_e=x_e,
                ext_x_N=float(pos[:, 0].max()), ext_y=float(pos[:, 1].max()))


def vizinhos_flags(lab2, i, j, nx, ny, quem):
    out = {}
    for name, di, dj in OFF:
        ii, jj = i + di, j + dj
        out[name] = bool(0 <= ii < nx and 0 <= jj < ny and quem(ii, jj))
    return out


def ret_continua(lab, nx, ny, colw, rowh_fn, g):
    """retencao continua (val, teste) com aparo assimetrico; colw[i][j] largura por bloco."""
    L = lab.reshape(nx, ny)
    va_ret = np.zeros((nx, ny), bool); a_va = a_va0 = a_te = a_te0 = 0.0
    for i in range(nx):
        for j in range(ny):
            if L[i, j] != 1: continue
            f = vizinhos_flags(L, i, j, nx, ny, lambda a, c: L[a, c] == 0)
            wx, wy = colw[i][j], rowh_fn(j)
            A = area_cont(wx, wy, B, f["L"], f["R"], f["Bo"], f["T"], f["BL"], f["BR"], f["TL"], f["TR"])
            a_va += A; a_va0 += wx * wy; va_ret[i, j] = A > 1e-9
    for i in range(nx):
        for j in range(ny):
            if L[i, j] != 2: continue
            f = vizinhos_flags(L, i, j, nx, ny, lambda a, c: L[a, c] == 0 or (L[a, c] == 1 and va_ret[a, c]))
            wx, wy = colw[i][j], rowh_fn(j)
            A = area_cont(wx, wy, B, f["L"], f["R"], f["Bo"], f["T"], f["BL"], f["BR"], f["TL"], f["TR"])
            a_te += A; a_te0 += wx * wy
    return a_va / a_va0, a_te / a_te0


def ret_nodal_mask(lab, prep):
    nx, ny, F, n_tot = prep["nx"], prep["ny"], prep["F"], prep["n_tot"]
    L = lab.reshape(nx, ny)
    va_ret = np.zeros((nx, ny), bool); r_va = n_va0 = r_te = n_te0 = 0
    def fl(i, j, quem):
        v = 0
        for bit, (_, di, dj) in enumerate(OFF):
            ii, jj = i + di, j + dj
            if 0 <= ii < nx and 0 <= jj < ny and quem(ii, jj):
                v |= 1 << bit
        return v
    for i in range(nx):
        for j in range(ny):
            if L[i, j] != 1: continue
            k = i * ny + j; n = F[k, fl(i, j, lambda a, c: L[a, c] == 0)]
            r_va += n; n_va0 += n_tot[k]; va_ret[i, j] = n > 0
    for i in range(nx):
        for j in range(ny):
            if L[i, j] != 2: continue
            k = i * ny + j
            r_te += F[k, fl(i, j, lambda a, c: L[a, c] == 0 or (L[a, c] == 1 and va_ret[a, c]))]
            n_te0 += n_tot[k]
    return r_va / n_va0, r_te / n_te0


def main():
    out_path = Path(sys.argv[1]); t0 = time.time()
    d18 = json.load(open(F18)); d18b = json.load(open(F18b)); d13 = json.load(open(F13))["por_celula"]
    c11 = json.load(open(F11))["por_celula"]
    SEEDS200 = d18["seeds"]; assert SEEDS200 == d18b["seeds"]
    SEEDS50 = list(range(1000, 1050))
    t2 = {(r["cidade"], int(r["g_km"])): r for r in csv.DictReader(open(T2))}
    ext_t2 = {c: mcr.CIDADES_EXT[c]["ext_x_km"] for c in mcr.CIDADES_EXT}
    res = {"script": str(Path(__file__).resolve()), "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "insumos": {str(p): sha(p) for p in (F18, F18b, F13, F11, T2, GEO, MCR)},
           "seeds_200": "fase1/1.8 seeds (RandomState(20260926).randint(1e6,200))", "seeds_50": "1000-1049 (fase1/1.3)",
           "b_km": B, "por_geometria": {}}
    json.dump(res, open(out_path, "w"), indent=1)
    for g, ref in ((10.0, d18), (5.0, d18b)):
        for cid in ("bauru", "campinas", "lins", "sorocaba"):
            for q in ("Q1", "Q3"):
                key = f"{cid}_{q}_g{int(g)}"; t1 = time.time()
                cel = c11[f"{cid}_{q}"]
                prep = preparar_nodal(cel, g)
                nx, ny = prep["nx"], prep["ny"]
                ext_y = prep["ext_y"]
                rowh = lambda j: g if j < ny - 1 else ext_y - g * (ny - 1)
                def colw_rect(ext_x):
                    last = ext_x - g * (nx - 1)
                    return [[g] * ny for _ in range(nx - 1)] + [[last] * ny]
                colw_t2 = colw_rect(ext_t2[cid]); colw_own = colw_rect(prep["ext_x_N"])
                colw_trap = [[g] * ny for _ in range(nx - 1)] + [[float(prep["x_e"][j] - g * (nx - 1)) for j in range(ny)]]
                rows = {}
                for nome, seeds in (("s200", SEEDS200), ("s50", SEEDS50)):
                    acc = {v: [] for v in ("Rt2", "Rown", "Ztrap", "Vnode")}
                    for s in seeds:
                        lab = labels_for_seed(s, nx * ny)
                        acc["Rt2"].append(ret_continua(lab, nx, ny, colw_t2, rowh, g))
                        acc["Rown"].append(ret_continua(lab, nx, ny, colw_own, rowh, g))
                        acc["Ztrap"].append(ret_continua(lab, nx, ny, colw_trap, rowh, g))
                        acc["Vnode"].append(ret_nodal_mask(lab, prep))
                    rows[nome] = {v: {"va": [float(x[0]) for x in a], "te": [float(x[1]) for x in a]} for v, a in acc.items()}
                nod_te = np.array(ref["por_celula"][f"{cid}_{q}"]["retencoes_te"])
                vnode_te = np.array(rows["s200"]["Vnode"]["te"])
                ent = {
                    "nx_ny": [nx, ny], "largura_ultima_coluna_T2_km": colw_t2[-1][0],
                    "largura_ultima_coluna_norte_km": colw_own[-1][0],
                    "largura_ultima_coluna_trapezio_por_linha_km": colw_trap[-1],
                    "medias_s200": {v: {"va": float(np.mean(rows["s200"][v]["va"])), "te": float(np.mean(rows["s200"][v]["te"]))} for v in rows["s200"]},
                    "medias_s50": {v: {"va": float(np.mean(rows["s50"][v]["va"])), "te": float(np.mean(rows["s50"][v]["te"]))} for v in rows["s50"]},
                    "nodal200_te_media": float(nod_te.mean()), "nodal200_va_media": ref["por_celula"][f"{cid}_{q}"]["nodal_va_media"],
                    "vnode_vs_nodal200_por_seed_te": {"max_abs": float(np.max(np.abs(vnode_te - nod_te))),
                                                      "media": float(np.mean(vnode_te - nod_te)),
                                                      "n_seeds_iguais_1e-9": int(np.sum(np.abs(vnode_te - nod_te) < 1e-9))},
                    "tempo_s": time.time() - t1,
                }
                k13 = f"{cid}_{q}_g{int(g)}b2"
                if k13 in d13:
                    ent["mc_nodal_1.3_te"] = d13[k13]["mc_nodal"]["ret_te_media"]
                    ent["mc_nodal_1.3_va"] = d13[k13]["mc_nodal"]["ret_va_media"]
                res["por_geometria"][key] = ent
                print(key, json.dumps({k: ent[k] for k in ("medias_s200", "nodal200_te_media", "vnode_vs_nodal200_por_seed_te")}), "%.0fs" % ent["tempo_s"], flush=True)
                json.dump(res, open(out_path, "w"), indent=1)

    # ---------------- decomposicao ----------------
    dec = {}
    for g in (10, 5):
        for cid in ("bauru", "campinas", "lins", "sorocaba"):
            gs = [res["por_geometria"][f"{cid}_{q}_g{g}"] for q in ("Q1", "Q3")]
            m = lambda v, part, s="medias_s200": float(np.mean([x[s][v][part] for x in gs]))
            r = t2[(cid, g)]
            nod_te = float(np.mean([x["nodal200_te_media"] for x in gs])); nod_va = float(np.mean([x["nodal200_va_media"] for x in gs]))
            mc_te = float(r["mc_v2_te_media"])
            cad_te = [("MC_T2", mc_te), ("Rt2_200", m("Rt2", "te")), ("Rown_200", m("Rown", "te")),
                      ("Ztrap_200", m("Ztrap", "te")), ("Vnode_200", m("Vnode", "te")), ("nodal200", nod_te)]
            passos = {f"{a[0]}->{b[0]}": (b[1] - a[1]) / nod_te * 100 for a, b in zip(cad_te[:-1], cad_te[1:])}
            total = (nod_te - mc_te) / nod_te * 100
            ent = {"cadeia_te": dict(cad_te), "passos_te_pct_da_nodal": passos, "total_te_pct": total,
                   "T2_mc_vs_nodal_te_publicado_pct": float(r["mc_v2_vs_nodal200_te"]) * 100,
                   "nodal200_te_T2_csv": float(r["nodal_te_media_4cel"]), "nodal200_te_recalc": nod_te,
                   "validacao_val": {"Vnode_200_va": m("Vnode", "va"), "nodal200_va": nod_va,
                                     "erro_rel": (m("Vnode", "va") - nod_va) / nod_va}}
            # P1: 1.3 vs 1.8 -- mesmo estimador, seeds diferentes
            p1 = {}
            for x, q in zip(gs, ("Q1", "Q3")):
                if "mc_nodal_1.3_te" in x:
                    p1[q] = {"publicado_1.3_menos_1.8_pct": (x["mc_nodal_1.3_te"] / x["nodal200_te_media"] - 1) * 100,
                             "Vnode_s50_menos_Vnode_s200_pct": (x["medias_s50"]["Vnode"]["te"] / x["medias_s200"]["Vnode"]["te"] - 1) * 100,
                             "Vnode_s50_menos_1.3_pct": (x["medias_s50"]["Vnode"]["te"] / x["mc_nodal_1.3_te"] - 1) * 100}
            ent["P1_seed_set"] = p1
            dec[f"{cid}_g{g}"] = ent
    res["decomposicao"] = dec
    # erro-padrao teorico da diferenca de medias 50 x 200 (seeds independentes)
    se = {}
    for g, ref in ((10, d18), (5, d18b)):
        sd = np.mean([ref["por_celula"][k]["nodal_te_dp"] for k in ref["por_celula"]])
        mu = np.mean([ref["por_celula"][k]["nodal_te_media"] for k in ref["por_celula"]])
        se[f"g{g}"] = {"dp_por_sorteio": float(sd), "ep_diferenca_50x200_pct": float(sd * np.sqrt(1 / 50 + 1 / 200) / mu * 100),
                       "ep_media200_pct": float(sd / np.sqrt(200) / mu * 100)}
    res["erro_padrao_amostral"] = se
    res["tempo_total_s"] = time.time() - t0
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(dec, indent=1)); print(json.dumps(se, indent=1))


if __name__ == "__main__":
    main()
