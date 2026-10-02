"""
Monte Carlo de retencao para o split espacial 3-vias com reticulado REAL
(11x12 blocos em g=10km; 21x23 em g=5km, b=2km), bordas REAIS por cidade
(largura da ultima coluna/linha != g) e a regra ASSIMETRICA do codigo
congelado (val aparado contra train; test aparado contra train U
val-JA-APARADO -- nao contra val bruto).

Fonte das extensoes (READ-ONLY, so leitura): campo blocos_nao_efetivos.
por_cidade.<cidade>.{ext_x_km, ext_y_km} de
/trabalho/HERMES/AGENTES/_FIOS/2026-09-23_gnn_rf_artigo2_mathematics_opcao_b/
artefatos/matematica-estatistica-do-claim_retencao_t11.json (R1, frente
matematica-estatistica-do-claim). n_blocos por papel (g10b2: 92/20/20 de
132; g5b2 bauru: 338/72/73 de 483) do mesmo artefato (campo
retencao_por_celula[*].n_blocos), conferido contra
train_gnn_c0_spatial.py:412-472 (split_espacial_3vias): grupos =
np.unique(group_ids) de SpatialKFold._assign_groups (grid inteiro
int(pos_km/grid_km), sem excecao de borda -- confirmado por leitura,
scripts_congelados/train_gnn_c0_spatial.py e blocos_nao_efetivos.
tratamento_do_bloco_de_borda_no_codigo do JSON de R1); rng.shuffle(grupos)
seguido de fatiamento contiguo por contagem fixa (n_tr,n_va,n_te) -- e
exatamente uma PERMUTACAO SEM REPOSICAO dos ids de bloco, reproduzida aqui.

Regras do split (linhas 447-465 do arquivo acima, lidas nesta rodada):
  1) val aparado contra train: bloco de val perde a faixa de largura b em
     cada lado onde ha um bloco de TREINO adjacente (lateral ou diagonal).
  2) test aparado contra train U val_APARADO (nao o val bruto): um bloco de
     val so conta como aparador de um vizinho de teste se ELE PROPRIO
     sobreviveu ao seu proprio aparo (area retida > 0) -- efeito
     "assimetrico" pedido no protocolo (variante C).

Tres variantes (aparo geometrico axis-aligned + cantos = quarto de disco,
Proposicao 1 do rascunho, generalizada para blocos NAO quadrados):
  A) blocos INTEIROS (g x g, borda ignorada) + aparo SIMETRICO/bruto
     (test aparado contra train U val BRUTO, sem o efeito de retencao
     propria do val) -- equivalente ao MC de R1 (p2_meanfield_check.py,
     achado P2-02), aqui refeito para conferencia cruzada.
  B) + bordas REAIS (blocos de borda tem largura/altura < g) mas ainda
     aparo bruto (mesma regra de trim de A, geometria de B).
  C) + regra ASSIMETRICA do codigo (test aparado contra val JA aparado) --
     modelo completo, bordas reais + assimetria.

Hipotese/aproximacao declarada: a area retida de um bloco e calculada pela
formula geometrica continua (retangulo menos faixas de largura b removidas
por vizinho-que-apara, menos quartos de disco em cantos isolados,
generalizando a Proposicao 1 do rascunho para blocos de largura/altura
DIFERENTES de g nas bordas do dominio) -- NAO reproduz o aparo por
no-mais-proximo (cKDTree) exatamente; e a mesma aproximacao de granularidade
de bloco que P1-02/P2-02 de R1 ja registraram como limite nao verificado
(nao_verificado desta rodada). O erro de discretizacao dessa aproximacao
continua e medido por rasterizacao fina (grade de raster_m metros) de uma
amostra de blocos de borda/canto e comparado com a formula fechada.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

SEED = 20260924
N_TRIALS_MAIN = 2500
N_TRIALS_SWEEP = 800
RASTER_M = 25.0  # resolucao do raster de discretizacao (metros)

# Extensoes por cidade (READ-ONLY, citadas de R1 -- ver docstring)
CIDADES_EXT = {
    "bauru": {"ext_x_km": 103.37162264246432, "ext_y_km": 110.96908950805664},
    "campinas": {"ext_x_km": 102.95723100995667, "ext_y_km": 110.96908950805664},
    "lins": {"ext_x_km": 103.81934358302617, "ext_y_km": 110.96908950805664},
    "sorocaba": {"ext_x_km": 102.5211445573578, "ext_y_km": 110.96908950805664},
}

# n_blocos por papel, por config, do run JSON real (R1, retencao_por_celula)
N_BLOCOS = {
    "g10b2": {"train": 92, "val": 20, "test": 20},   # 132 blocos, todas as 4 cidades
    "g5b2": {"train": 338, "val": 72, "test": 73},   # 483 blocos, bauru (unica com g5 em R1)
}

# retencao observada (seed 42), R1 (retencao_por_celula, Q1 de cada cidade)
OBS_TEST = {
    ("bauru", "g10b2"): 0.4632420674893326,
    ("campinas", "g10b2"): 0.46014706310804726,
    ("lins", "g10b2"): 0.46549287319735866,
    ("sorocaba", "g10b2"): 0.45963125105236574,
    ("bauru", "g5b2"): 0.1047868809738556,
}
OBS_VAL = {
    ("bauru", "g10b2"): 0.5210508853685026,
    ("campinas", "g10b2"): 0.5201393265848172,
    ("lins", "g10b2"): 0.5209684329222332,
    ("sorocaba", "g10b2"): 0.5201061447264552,
    ("bauru", "g5b2"): 0.20190782023022533,
}


def grid_dims(ext_x, ext_y, g):
    """Reproduz int(pos_km/grid_km): blocos 0..floor(ext/g) inclusive quando
    ha resto > 0 (bloco de borda parcial); largura/altura reais por bloco."""
    nx_full = int(np.floor(ext_x / g))
    resto_x = ext_x - nx_full * g
    nx = nx_full + (1 if resto_x > 1e-9 else 0)
    colw = [g] * nx_full + ([resto_x] if resto_x > 1e-9 else [])

    ny_full = int(np.floor(ext_y / g))
    resto_y = ext_y - ny_full * g
    ny = ny_full + (1 if resto_y > 1e-9 else 0)
    rowh = [g] * ny_full + ([resto_y] if resto_y > 1e-9 else [])
    return nx, ny, np.array(colw), np.array(rowh)


def area_bloco_continua(wx, wy, b, L, R, Bo, T, BL, BR, TL, TR):
    """Area retida (formula continua, generalizacao da Prop.1 para blocos
    nao quadrados): retangulo com faixas de largura b removidas nos lados
    onde ha aparo, menos quarto de disco em cada canto ISOLADO (aparado mas
    sem lateral adjacente aparada). Clampada em [0, wx*wy]."""
    rx = max(0.0, wx - b * (L + R))
    ry = max(0.0, wy - b * (Bo + T))
    rect = rx * ry
    corner_area = 0.0
    for corner, (lat1, lat2) in (
        (BL, (L, Bo)), (BR, (R, Bo)), (TL, (L, T)), (TR, (R, T))
    ):
        if corner and not lat1 and not lat2:
            corner_area += np.pi * b * b / 4.0
    area = rect - corner_area
    return float(np.clip(area, 0.0, wx * wy))


def area_bloco_raster(wx, wy, b, L, R, Bo, T, BL, BR, TL, TR, raster_m):
    """Mesma regra, mas por contagem de pixels num raster fino (para medir
    erro de discretizacao vs a formula continua)."""
    px = raster_m / 1000.0  # km
    nxp = max(1, int(round(wx / px)))
    nyp = max(1, int(round(wy / px)))
    xs = (np.arange(nxp) + 0.5) * (wx / nxp)
    ys = (np.arange(nyp) + 0.5) * (wy / nyp)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    keep = np.ones_like(X, dtype=bool)
    if L:
        keep &= X >= b
    if R:
        keep &= X <= wx - b
    if Bo:
        keep &= Y >= b
    if T:
        keep &= Y <= wy - b
    # cantos isolados: remove quarto de disco de raio b centrado no canto
    corners = [
        (BL, not L and not Bo, 0.0, 0.0),
        (BR, not R and not Bo, wx, 0.0),
        (TL, not L and not T, 0.0, wy),
        (TR, not R and not T, wx, wy),
    ]
    for present, isolated, cx, cy in corners:
        if present and isolated:
            d2 = (X - cx) ** 2 + (Y - cy) ** 2
            keep &= d2 >= b * b
    area_px = (wx / nxp) * (wy / nyp)
    return float(keep.sum() * area_px)


def vizinhos(i, j, nx, ny):
    off = {"L": (-1, 0), "R": (1, 0), "Bo": (0, -1), "T": (0, 1),
           "BL": (-1, -1), "BR": (1, -1), "TL": (-1, 1), "TR": (1, 1)}
    out = {}
    for k, (di, dj) in off.items():
        ii, jj = i + di, j + dj
        out[k] = (ii, jj) if (0 <= ii < nx and 0 <= jj < ny) else None
    return out


def mc_variant(nx, ny, colw, rowh, b, n_tr, n_va, n_te, n_trials, seed,
                variant, raster_sample=0, raster_m=RASTER_M):
    """variant: 'A' (blocos g x g, aparo bruto), 'B' (bordas reais, aparo
    bruto), 'C' (bordas reais, aparo assimetrico/val-ja-aparado).
    Retorna arrays de retencao media por trial para val e test, e (se
    raster_sample>0) uma comparacao de discretizacao numa subamostra."""
    rng = np.random.RandomState(seed)
    N = nx * ny
    assert n_tr + n_va + n_te == N, (n_tr, n_va, n_te, N)
    labels_base = np.array([0] * n_tr + [1] * n_va + [2] * n_te)

    g_nom = float(np.median(colw))  # tamanho nominal do bloco interior
    if variant == "A":
        colw_use = np.full(nx, g_nom)
        rowh_use = np.full(ny, g_nom)
    else:
        colw_use = colw
        rowh_use = rowh

    ret_va_trials = np.empty(n_trials)
    ret_te_trials = np.empty(n_trials)

    disc_errs = []
    raster_done = 0

    for t in range(n_trials):
        labels = labels_base.copy()
        rng.shuffle(labels)
        L = labels.reshape(nx, ny)

        # --- 1) aparo de val contra train ---
        # Retencao = area RETIDA total / area ANTES do buffer total (mesma
        # definicao do run JSON, split.retencao_apos_buffer: soma de nos, nao
        # media nao-ponderada de fracao por bloco -- bloco de borda pesa
        # proporcional a sua propria area, nao igual a um bloco inteiro).
        va_area_ret = np.full((nx, ny), np.nan)
        va_area_antes = np.full((nx, ny), np.nan)
        va_retido = np.zeros((nx, ny), dtype=bool)
        for i in range(nx):
            for j in range(ny):
                if L[i, j] != 1:
                    continue
                viz = vizinhos(i, j, nx, ny)
                flags = {}
                for k, pos in viz.items():
                    flags[k] = bool(pos is not None and L[pos] == 0)  # train apara
                wx, wy = colw_use[i], rowh_use[j]
                A = area_bloco_continua(wx, wy, b, flags["L"], flags["R"],
                                          flags["Bo"], flags["T"], flags["BL"],
                                          flags["BR"], flags["TL"], flags["TR"])
                va_area_ret[i, j] = A
                va_area_antes[i, j] = wx * wy
                va_retido[i, j] = A > 1e-9

                if raster_sample and raster_done < raster_sample and (wx < g_nom - 1e-6 or wy < g_nom - 1e-6):
                    Ar = area_bloco_raster(wx, wy, b, flags["L"], flags["R"],
                                             flags["Bo"], flags["T"], flags["BL"],
                                             flags["BR"], flags["TL"], flags["TR"], raster_m)
                    denom = max(A, Ar, 1e-12)
                    disc_errs.append(abs(A - Ar) / denom)
                    raster_done += 1

        antes_va = np.nansum(va_area_antes)
        ret_va_trials[t] = float(np.nansum(va_area_ret) / antes_va) if antes_va > 0 else np.nan

        # --- 2) aparo de test contra train U val (bruto em A/B; retido em C) ---
        te_area_ret = np.full((nx, ny), np.nan)
        te_area_antes = np.full((nx, ny), np.nan)
        for i in range(nx):
            for j in range(ny):
                if L[i, j] != 2:
                    continue
                viz = vizinhos(i, j, nx, ny)
                flags = {}
                for k, pos in viz.items():
                    if pos is None:
                        flags[k] = False
                        continue
                    role = L[pos]
                    if role == 0:
                        flags[k] = True
                    elif role == 1:
                        if variant == "C":
                            flags[k] = bool(va_retido[pos])
                        else:
                            flags[k] = True  # val bruto (A, B)
                    else:
                        flags[k] = False
                wx, wy = colw_use[i], rowh_use[j]
                A = area_bloco_continua(wx, wy, b, flags["L"], flags["R"],
                                          flags["Bo"], flags["T"], flags["BL"],
                                          flags["BR"], flags["TL"], flags["TR"])
                te_area_ret[i, j] = A
                te_area_antes[i, j] = wx * wy

        antes_te = np.nansum(te_area_antes)
        ret_te_trials[t] = float(np.nansum(te_area_ret) / antes_te) if antes_te > 0 else np.nan

    out = {
        "ret_va_media": float(np.nanmean(ret_va_trials)),
        "ret_va_dp": float(np.nanstd(ret_va_trials)),
        "ret_va_p5": float(np.nanpercentile(ret_va_trials, 5)),
        "ret_va_p95": float(np.nanpercentile(ret_va_trials, 95)),
        "ret_te_media": float(np.nanmean(ret_te_trials)),
        "ret_te_dp": float(np.nanstd(ret_te_trials)),
        "ret_te_p5": float(np.nanpercentile(ret_te_trials, 5)),
        "ret_te_p95": float(np.nanpercentile(ret_te_trials, 95)),
        "n_trials": n_trials,
    }
    if raster_sample:
        out["discretizacao_erro_rel_max"] = float(max(disc_errs)) if disc_errs else None
        out["discretizacao_erro_rel_medio"] = float(np.mean(disc_errs)) if disc_errs else None
        out["discretizacao_n_amostras"] = raster_done
        out["discretizacao_raster_m"] = raster_m
    return out


def R_meanfield(g, b, q):
    """Forma fechada da Prop.2 do rascunho (conferida em R1, P2-01)."""
    return ((g - 2 * b) ** 2 + 4 * b * (g - 2 * b) * q
             + 4 * b ** 2 * (1 - np.pi / 4) * q ** 2 + np.pi * b ** 2 * q ** 3) / g ** 2


def sympy_check_R_meanfield():
    g, b, q = sp.symbols('g b q', positive=True)
    p = 1 - q
    E_lado = g - 2 * b * p
    E_rect = sp.expand(E_lado * E_lado)
    E_corner = 4 * (sp.pi * b ** 2 / 4) * p * (1 - p) ** 2
    E_area = sp.expand(E_rect - E_corner)
    R_derivado = sp.expand(E_area) / g ** 2
    R_texto_num = ((g - 2 * b) ** 2 + 4 * b * (g - 2 * b) * q
                    + 4 * b ** 2 * (1 - sp.pi / 4) * q ** 2 + sp.pi * b ** 2 * q ** 3)
    diff = sp.simplify(sp.expand(E_area) - sp.expand(R_texto_num))
    return {"diferenca_e_zero": diff == 0, "diferenca": str(diff)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true", help="so imprime o sha256 do script e sai")
    args = ap.parse_args()
    if args.hash:
        print(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return
    out = {
        "seed": SEED,
        "raster_m": RASTER_M,
        "sympy_conferencia_R_meanfield": sympy_check_R_meanfield(),
        "por_config": {},
        "extensao_barata_bauru_sweep": {},
    }

    # --- comparacao principal: g10b2 (4 cidades) e g5b2 (bauru) ---
    for config, b in (("g10b2", 2.0), ("g5b2", 2.0)):
        g = 10.0 if config == "g10b2" else 5.0
        n_roles = N_BLOCOS[config]
        cidades = list(CIDADES_EXT.keys()) if config == "g10b2" else ["bauru"]
        out["por_config"][config] = {"g_km": g, "b_km": b, "por_cidade": {}}
        for cidade in cidades:
            ext = CIDADES_EXT[cidade]
            nx, ny, colw, rowh = grid_dims(ext["ext_x_km"], ext["ext_y_km"], g)
            N = nx * ny
            assert N == n_roles["train"] + n_roles["val"] + n_roles["test"], \
                (cidade, config, N, n_roles)
            q_te = n_roles["test"] / N
            q_va = 1 - n_roles["train"] / N
            cm_te = R_meanfield(g, b, q_te)
            cm_va = R_meanfield(g, b, q_va)

            variantes = {}
            for variant in ("A", "B", "C"):
                raster_sample = 40 if (variant == "C" and cidade == cidades[0]) else 0
                res = mc_variant(nx, ny, colw, rowh, b, n_roles["train"],
                                  n_roles["val"], n_roles["test"],
                                  N_TRIALS_MAIN, SEED + hash((config, cidade, variant)) % 10000,
                                  variant, raster_sample=raster_sample)
                variantes[variant] = res

            obs_te = OBS_TEST.get((cidade, config))
            obs_va = OBS_VAL.get((cidade, config))
            entry = {
                "nx": nx, "ny": ny, "n_blocos_total": N,
                "n_blocos": n_roles,
                "colw_km": colw.tolist(), "rowh_km": rowh.tolist(),
                "campo_medio_te": cm_te, "campo_medio_va": cm_va,
                "observado_te": obs_te, "observado_va": obs_va,
                "variantes": variantes,
            }
            if obs_te is not None:
                entry["erro_rel_campo_medio_te"] = abs(cm_te - obs_te) / obs_te
                entry["erro_rel_MC_C_te"] = abs(variantes["C"]["ret_te_media"] - obs_te) / obs_te
                entry["obs_dentro_p5_p95_MC_C_te"] = bool(
                    variantes["C"]["ret_te_p5"] <= obs_te <= variantes["C"]["ret_te_p95"])
            if obs_va is not None:
                entry["erro_rel_campo_medio_va"] = abs(cm_va - obs_va) / obs_va
                entry["erro_rel_MC_C_va"] = abs(variantes["C"]["ret_va_media"] - obs_va) / obs_va
                entry["obs_dentro_p5_p95_MC_C_va"] = bool(
                    variantes["C"]["ret_va_p5"] <= obs_va <= variantes["C"]["ret_va_p95"])
            out["por_config"][config]["por_cidade"][cidade] = entry

    # --- extensao barata: sweep b x g em bauru (comparador da varredura de
    # matematica-estatistica-do-claim, entrega 1, que usa nos reais) ---
    bs = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    gs = [2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
    ext = CIDADES_EXT["bauru"]
    sweep = {}
    for g in gs:
        for b in bs:
            if b == 0.0:
                # aparo nulo: retencao = 1 trivialmente (sem custo de MC)
                sweep[f"g{g}_b{b}"] = {"g_km": g, "b_km": b, "ret_te_media": 1.0,
                                        "ret_va_media": 1.0, "trivial": True}
                continue
            nx, ny, colw, rowh = grid_dims(ext["ext_x_km"], ext["ext_y_km"], g)
            N = nx * ny
            # fracoes nominais 0.70/0.15/0.15 arredondadas (mesma regra do
            # codigo: max(1, round(frac*N)); resto vai para treino)
            n_va = max(1, int(round(0.15 * N)))
            n_te = max(1, int(round(0.15 * N)))
            n_tr = N - n_va - n_te
            if n_tr < 1:
                continue
            res = mc_variant(nx, ny, colw, rowh, b, n_tr, n_va, n_te,
                              N_TRIALS_SWEEP, SEED + hash((g, b)) % 10000, "C")
            cm_te = R_meanfield(g, b, n_te / N)
            sweep[f"g{g}_b{b}"] = {
                "g_km": g, "b_km": b, "nx": nx, "ny": ny, "N": N,
                "n_blocos": {"train": n_tr, "val": n_va, "test": n_te},
                "campo_medio_te": cm_te,
                "ret_te_media_MC_C": res["ret_te_media"],
                "ret_te_p5": res["ret_te_p5"], "ret_te_p95": res["ret_te_p95"],
            }
    out["extensao_barata_bauru_sweep"] = sweep

    src = Path(__file__).read_text()
    out["script_sha256"] = hashlib.sha256(src.encode()).hexdigest()

    dest = Path("/trabalho/HERMES/AGENTES/_FIOS/2026-09-24_gnn_rf_artigo2_mathematics_r2/"
                "artefatos/matematica-equacoes_montecarlo_retencao_RAW.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "extensao_barata_bauru_sweep"}, indent=1)[:6000])
    print("OK ->", dest)


if __name__ == "__main__":
    main()
