#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
# -*- coding: utf-8 -*-
"""R4 / roadmap v3-12 B3.3 (criterios/criterio_R4_p_fechado_vs_frequencia.json, fixado em
2026-10-01T17:13:54-03:00 antes deste codigo): por classe de borda m(i), a frequencia de inclusao no
teste CONDICIONAL ao bloco ser de teste (p_{i|te}) em 200 sorteios coincide com a forma fechada
onde a proposicao a declara exata e respeita as cotas onde ela da cotas?

Particao: a mesma de scripts/v3_1.6_2.3_estimando_formal.py, reutilizada por IMPORT dentro de
scripts/v3_12_R3R4_motor.py (importlib, sem copia/edicao do original). O motor tambem calcula
m(i) nodal (main.tex l.296), a condicao (iii) por no (l.300) e a classe alternativa m_geom; este
script le os intermediarios do motor (fase4/_v3_12_R3R4_intermediarios/) e faz as estatisticas.

Frequencia por classe c: media NODAL da frequencia condicional, f_i = (#sorteios com i retido em teste)
/ (#sorteios com beta(i) de teste); f_c = media de f_i sobre os nos de c. ERRO-PADRAO MC: bootstrap
sobre os 200 sorteios (reamostra os sorteios com reposicao e recalcula f_c inteira, incluindo os T_j por
bloco) -> captura a correlacao dos nos de um mesmo bloco (estimador por blocos); B_BOOT = 4000, semente
fixa. Secundario: ep_por_sorteio = dp entre sorteios de f_s(c) (razao agregada do sorteio s) / sqrt(200).

Forma fechada (prop:inclusion (ii)): lo(m) = (k_te-1)^{m} / (N-1)^{m};  hi(m) = (N-1-k_tr)^{m} / (N-1)^{m}
(potencias fatoriais descendentes), N=132, k_te=20, k_tr=92. (iii): lo atingida se vale a condicao por no.

LEITURAS do enunciado quanto ao que e 'exato' (ambiguidade declarada; todas implementadas e reportadas):
  L_no_a_no   : exato (= lo) onde m=0 ou a condicao (iii) VALE no no (calculada); demais nos: so cotas.
                (leitura literal do enunciado; fixada a priori como primaria)
  T_classe    : exato nas classes m=0,1,3 e so cotas em m=2 e m>=4 (texto de main.tex l.312).
  C_todas     : lo e tratada como valor exato em toda classe m (ignora a condicao (iii)).
  U_so_m0     : exato apenas m=0; toda classe m>=1 so cotas (ignora (iii) como suficiencia).
Duas definicoes de m: 'nodal' (blocos com no a < b; main.tex) e 'geom' (lados/cantos a < b, como o
criterio parafraseia e como fase2/fismat_k_corrigido.py); geom so com as leituras T, C, U.

Saidas: fase4/R4_p_por_classe.json; parciais por celula fase4/_parcial_R4_<cidade>.json.
Execucao: rodar antes v3_12_R3R4_motor.py (grava os intermediarios). Python so da venv CUDA; so CPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPTS = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts")
sys.path.insert(0, str(SCRIPTS))
import v3_12_R3R4_motor as mo  # noqa: E402

RAIZ = mo.RAIZ
FASE4 = RAIZ / "fase4"
CRIT = RAIZ / "criterios" / "criterio_R4_p_fechado_vs_frequencia.json"
OUT = FASE4 / "R4_p_por_classe.json"
B_BOOT = 4000
N_T0 = [0]   # contador global: (rep, bloco) com T_j = 0 no bootstrap
SEED_BOOT = 20261001
LEITURAS = ("L_no_a_no", "T_classe", "C_todas", "U_so_m0")
LEITURA_PRIMARIA = "L_no_a_no"


def lo_hi(m: int, N: int, kte: int, ktr: int):
    if m > kte - 1:
        lo = 0.0
    else:
        lo = math.perm(kte - 1, m) / math.perm(N - 1, m)
    hi = math.perm(N - 1 - ktr, m) / math.perm(N - 1, m) if m <= N - 1 - ktr else 0.0
    return lo, hi


def nao_exato(leitura: str, m: int, falha: bool) -> bool:
    if m == 0:
        return False
    if leitura == "L_no_a_no":
        return bool(falha)
    if leitura == "T_classe":
        return m not in (1, 3)
    if leitura == "C_todas":
        return False
    if leitura == "U_so_m0":
        return True
    raise ValueError(leitura)


def freq_linhas(R, te, n_bc, agg, W=None):
    """R: (S,J,C) retidos de teste por sorteio, bloco, classe; te: (S,J) bloco de teste;
    n_bc: (J,C) nos por bloco e classe; agg: (L,C) 0/1 (linhas = classes de saida).
    W: pesos de sorteio (S,) (contagens do bootstrap) ou None. Retorna f (L,) = media nodal de f_i."""
    if W is None:
        W = np.ones(R.shape[0])
    T = W @ te                                   # (J,)
    assert (T > 0).all()
    Rsum = np.tensordot(W, R, axes=(0, 0))       # (J,C)
    Rl = Rsum @ agg.T                            # (J,L)
    n_l = (n_bc @ agg.T).sum(0)                  # (L,)
    return (Rl / T[:, None]).sum(0) / n_l


def estatisticas(R, te, n_bc, agg, rng):
    S = R.shape[0]
    f = freq_linhas(R, te, n_bc, agg)
    boot = np.empty((B_BOOT, agg.shape[0]))
    Rf = R.reshape(S, -1)
    n_l = (n_bc @ agg.T).sum(0)
    J, C = n_bc.shape
    for b0 in range(0, B_BOOT, 500):
        nb = min(500, B_BOOT - b0)
        Wb = rng.multinomial(S, np.full(S, 1.0 / S), size=nb).astype(np.float64)   # (nb,S)
        Tb = Wb @ te                                                               # (nb,J)
        N_T0[0] += int((Tb <= 0).sum())      # bloco sem sorteio de teste na reamostra (contribuicao 0; contado)
        Tb = np.where(Tb > 0, Tb, 1.0)
        Rb = (Wb @ Rf).reshape(nb, J, C)                                           # (nb,J,C)
        Rl = Rb @ agg.T                                                            # (nb,J,L)
        boot[b0:b0 + nb] = (Rl / Tb[:, :, None]).sum(1) / n_l
    ep = boot.std(axis=0, ddof=1)
    # secundario: razao agregada por sorteio
    num = np.einsum("sjc,lc->sjl", R.astype(np.float64), agg)                      # (S,J,L)
    den = np.einsum("sj,jl->sjl", te, n_bc @ agg.T)
    fs = num.sum(1) / np.where(den.sum(1) > 0, den.sum(1), np.nan)                 # (S,L)
    ep_s = np.nanstd(fs, axis=0, ddof=1) / np.sqrt(S)
    pooled = num.sum((0, 1)) / den.sum((0, 1))
    # diagnosticos por blocos (classes pequenas: o bootstrap de sorteios subestima o EP de eventos raros)
    n_jl = n_bc @ agg.T                                                            # (J,L)
    Tj = te.sum(0)                                                                 # (J,)
    w = n_jl / n_jl.sum(0)[None, :]
    extras = dict(
        n_blocos=(n_jl > 0).sum(0), ensaios_bloco_sorteio=((n_jl > 0) * Tj[:, None]).sum(0),
        retidos_bloco_sorteio_com_algum_no=(num > 0).sum((0, 1)),
        soma_w2_sobre_T=((w ** 2) / Tj[:, None]).sum(0))
    return f, ep, ep_s, pooled, extras


def montar_linhas(nome_def, codigos, agg_rows, f, ep, ep_s, pooled, extras, n_codigo, n_total, N, kte, ktr, leituras):
    """codigos: lista (m, falha) por coluna de classe; agg_rows: lista (rotulo, [indices de colunas])"""
    linhas = []
    for li, (rot, cols) in enumerate(agg_rows):
        n_row = int(sum(n_codigo[c] for c in cols))
        if n_row == 0:
            continue
        ms = sorted({codigos[c][0] for c in cols})
        lo_m = {m: lo_hi(m, N, kte, ktr) for m in ms}
        row = dict(classe=rot, definicao_m=nome_def, n_nos=n_row, fracao_nos=n_row / n_total,
                   m_valores=ms,
                   n_nos_cond_iii_falha=int(sum(n_codigo[c] for c in cols if codigos[c][1])),
                   frequencia_media_nodal=float(f[li]), ep_mc_bootstrap_sorteios=float(ep[li]),
                   ep_mc_por_sorteio_agregado=float(ep_s[li]), frequencia_agregada_pooled=float(pooled[li]),
                   n_blocos_com_a_classe=int(extras["n_blocos"][li]),
                   ensaios_bloco_sorteio=float(extras["ensaios_bloco_sorteio"][li]),
                   retidos_bloco_sorteio_com_algum_no=int(extras["retidos_bloco_sorteio_com_algum_no"][li]),
                   cota_inferior_por_m={str(m): lo_m[m][0] for m in ms},
                   cota_superior_por_m={str(m): lo_m[m][1] for m in ms},
                   leituras={})
        for lei in leituras:
            lo_row = sum(n_codigo[c] * lo_m[codigos[c][0]][0] for c in cols) / n_row
            hi_row = sum(n_codigo[c] * (lo_m[codigos[c][0]][1] if nao_exato(lei, *codigos[c]) else lo_m[codigos[c][0]][0])
                         for c in cols) / n_row
            n_nao_ex = int(sum(n_codigo[c] for c in cols if nao_exato(lei, *codigos[c])))
            exato_puro = n_nao_ex == 0
            epv = float(ep[li])
            fr = float(f[li])
            # EP alternativo 'por blocos' sob a hipotese nula p0=lo_row (blocos independentes, nos de um bloco
            # perfeitamente correlacionados): sqrt(Sum_j w_j^2 p0(1-p0)/T_j); mais honesto que o bootstrap em
            # classes de poucos blocos/eventos raros (o bootstrap de sorteios subestima o EP nesses casos)
            ep_bn = float(math.sqrt(extras["soma_w2_sobre_T"][li] * lo_row * (1 - lo_row)))
            if epv > 0:
                dif_lo = (fr - lo_row) / epv
                if fr > hi_row:
                    dif_cota = (fr - hi_row) / epv
                elif fr < lo_row:
                    dif_cota = (fr - lo_row) / epv
                else:
                    dif_cota = 0.0
            else:
                dif_lo = 0.0 if abs(fr - lo_row) < 1e-12 else float("inf")
                dif_cota = 0.0 if (lo_row - 1e-12 <= fr <= hi_row + 1e-12) else float("inf")
            rl = dict(
                n_nos_so_cotas=n_nao_ex, exato_em_toda_a_classe=exato_puro,
                valor_fechado=lo_row if exato_puro else None,
                cota_inferior_classe=lo_row, cota_superior_classe=hi_row,
                dif_freq_menos_inferior_em_EP=dif_lo,
                ep_alternativo_por_blocos_binomial_nula=ep_bn,
                dif_freq_menos_inferior_em_EP_alternativo=((fr - lo_row) / ep_bn if ep_bn > 0 else (0.0 if abs(fr - lo_row) < 1e-12 else float("inf"))),
                dif_freq_ate_intervalo_em_EP=dif_cota,
                dentro_do_intervalo=bool(lo_row - 1e-12 <= fr <= hi_row + 1e-12),
                dentro_do_intervalo_ou_a_2EP=bool(dif_cota == 0.0 or abs(dif_cota) <= 2.0),
                coincide_a_2EP_do_valor_fechado=(bool(abs(dif_lo) <= 2.0) if exato_puro else None),
            )
            row["leituras"][lei] = rl
        linhas.append(row)
    return linhas


def veredito(linhas, leitura):
    """cumpre o criterio (leitura 'a 2 EP'): classes totalmente exatas coincidem a <=2 EP;
    classes com parte so-cotas ficam dentro do intervalo (ou a <=2 EP dele). Lista as que falham."""
    falham, avaliadas, alt_falham = [], 0, []
    for r in linhas:
        rl = r["leituras"].get(leitura)
        if rl is None:
            continue
        avaliadas += 1
        if rl["exato_em_toda_a_classe"]:
            ok = rl["coincide_a_2EP_do_valor_fechado"]
            ok_alt = abs(rl["dif_freq_menos_inferior_em_EP_alternativo"]) <= 2.0
        else:
            ok = rl["dentro_do_intervalo_ou_a_2EP"]
            ok_alt = ok
        if not ok_alt:
            alt_falham.append(r["classe"])
        if not ok:
            falham.append(dict(classe=r["classe"], freq=r["frequencia_media_nodal"], ep=r["ep_mc_bootstrap_sorteios"],
                               cota_inferior=rl["cota_inferior_classe"], cota_superior=rl["cota_superior_classe"],
                               exato=rl["exato_em_toda_a_classe"], dif_lo_EP=rl["dif_freq_menos_inferior_em_EP"],
                               dif_ate_intervalo_EP=rl["dif_freq_ate_intervalo_em_EP"], n_nos=r["n_nos"]))
    return dict(criterio_cumprido=len(falham) == 0, n_classes_avaliadas=avaliadas, classes_que_falham=falham,
                criterio_cumprido_com_EP_alternativo_por_blocos=len(alt_falham) == 0,
                classes_que_falham_com_EP_alternativo=alt_falham)


def processar(cidade: str) -> dict:
    A, meta = mo.carregar_intermediario(cidade)
    N, kte, ktr = meta["n_blocos"], meta["k_te"], meta["k_tr"]
    assert (N, kte, ktr) == (132, 20, 92)
    te = A["te_block"].astype(np.float64)
    S = te.shape[0]
    n_total = meta["n_nos"]
    rng = np.random.default_rng(SEED_BOOT)
    # ---- nodal: colunas = codigo m*2+falha (0..9) ----
    codigos_n = [(c // 2, bool(c % 2)) for c in range(mo.NCG_N)]
    n_cod_n = A["n_bc_n"].sum(0)
    rows_n = []
    for mm in range(5):
        cols_all = [mm * 2, mm * 2 + 1]
        if n_cod_n[cols_all].sum() == 0:
            continue
        rows_n.append((f"m={mm}|todas", cols_all))
        rows_n.append((f"m={mm}|(iii) vale", [mm * 2]))
        rows_n.append((f"m={mm}|(iii) falha", [mm * 2 + 1]))
    rows_n = [(rot, cols) for rot, cols in rows_n if n_cod_n[cols].sum() > 0]
    rows_n.append(("TOTAL|todos os nos", list(range(mo.NCG_N))))
    agg_n = np.zeros((len(rows_n), mo.NCG_N))
    for li, (_, cols) in enumerate(rows_n):
        agg_n[li, cols] = 1.0
    f, ep, ep_s, pooled, extras = estatisticas(A["R_n"], te, A["n_bc_n"].astype(np.float64), agg_n, rng)
    linhas_n = montar_linhas("nodal", codigos_n, rows_n, f, ep, ep_s, pooled, extras, n_cod_n, n_total, N, kte, ktr, LEITURAS)
    # ---- geometrica ----
    codigos_g = [(c, False) for c in range(mo.NCG_G)]
    n_cod_g = A["n_bc_g"].sum(0)
    rows_g = [(f"mg={c}", [c]) for c in range(mo.NCG_G) if n_cod_g[c] > 0]
    rows_g.append(("TOTAL|todos os nos", list(range(mo.NCG_G))))
    agg_g = np.zeros((len(rows_g), mo.NCG_G))
    for li, (_, cols) in enumerate(rows_g):
        agg_g[li, cols] = 1.0
    f, ep, ep_s, pooled, extras = estatisticas(A["R_g"], te, A["n_bc_g"].astype(np.float64), agg_g, rng)
    linhas_g = montar_linhas("geom", codigos_g, rows_g, f, ep, ep_s, pooled, extras, n_cod_g, n_total, N, kte, ktr,
                             ("T_classe", "C_todas", "U_so_m0"))
    # linhas por leitura: granularidade da leitura
    ver = {}
    nodal_fina = [r for r in linhas_n if r["classe"].endswith("vale") or r["classe"].endswith("falha")]
    nodal_m = [r for r in linhas_n if r["classe"].endswith("todas") and r["classe"].startswith("m=")]
    ver["nodal"] = {
        "L_no_a_no": veredito(nodal_fina, "L_no_a_no"),
        "T_classe": veredito(nodal_m, "T_classe"),
        "C_todas": veredito(nodal_m, "C_todas"),
        "U_so_m0": veredito(nodal_m, "U_so_m0"),
    }
    geo_m = [r for r in linhas_g if r["classe"].startswith("mg=")]
    ver["geom"] = {lei: veredito(geo_m, lei) for lei in ("T_classe", "C_todas", "U_so_m0")}
    cruz = {}
    m_ = A["m"]
    mg_ = A["mg"]
    for a in range(int(m_.max()) + 1):
        cruz[f"m_nodal={a}"] = {f"mg={b}": int(((m_ == a) & (mg_ == b)).sum()) for b in range(int(mg_.max()) + 1)}
    # fidelidade
    fid = dict(retencao_teste_por_sorteio_igual_1_8_a_1e12_assert_no_motor=True,
               media_retencao_nodal_te=float(np.mean(A["n_te"] / A["n_te0"])),
               n_nos=n_total, n_blocos=N, k_te=kte, k_tr=ktr, m_max_observado=meta["m_max_observado"],
               n_nos_por_m=meta["n_nos_por_m"], n_nos_falha_iii_por_m=meta["n_falha_iii_por_m"],
               frequencia_total_media_nodal_p_cond_te=[r["frequencia_media_nodal"] for r in linhas_n if r["classe"].startswith("TOTAL")][0])
    return dict(celula=meta["celula"], fidelidade=fid, classes_nodais=linhas_n, classes_geometricas=linhas_g,
                cruzamento_m_nodal_x_m_geom=cruz, veredito_por_leitura=ver,
                bootstrap_blocos_reamostra_sem_teste_acumulado=int(N_T0[0]))


def main():
    t0 = datetime.now(timezone.utc)
    faltam = [c for c in mo.CIDADES if not (mo.INTERM / f"{c}_{mo.QUAD}.npz").exists()]
    if faltam:
        raise SystemExit(f"intermediarios ausentes ({faltam}); rode scripts/v3_12_R3R4_motor.py antes")
    res = {}
    for c in mo.CIDADES:
        r = processar(c)
        (FASE4 / f"_parcial_R4_{c}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        res[r["celula"]] = r
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {c}: parcial gravado", flush=True)
    seeds, _ = mo.seeds_200()
    saida = dict(
        id="R4_p_fechado_vs_frequencia", roadmap="B3.3",
        criterio=dict(arquivo=str(CRIT), sha256=mo.sha256_file(CRIT)),
        pergunta=json.load(open(CRIT))["pergunta"],
        parametros=dict(N=132, k_te=20, k_tr=92, g_km=mo.G, b_km=mo.B, n_sorteios=200, seeds=seeds,
                        fonte_seeds=str(mo.F18), bootstrap_reps=B_BOOT, semente_bootstrap=SEED_BOOT,
                        formula_cotas="lo=(k_te-1)^{m}/(N-1)^{m}; hi=(N-1-k_tr)^{m}/(N-1)^{m} (potencia fatorial descendente)"),
        leituras={
            "L_no_a_no": "exato (=lo) onde m=0 ou a condicao (iii) vale no no (calculada por no); demais so cotas. Primaria fixada a priori (leitura literal de prop:inclusion (iii)).",
            "T_classe": "exato em m=0,1,3; so cotas em m=2 e m>=4 (texto de main.tex l.312).",
            "C_todas": "lo tratada como valor exato em toda classe (ignora (iii)).",
            "U_so_m0": "exato so em m=0; m>=1 so cotas.",
        },
        leitura_primaria=LEITURA_PRIMARIA,
        ambiguidades=[
            "A proposicao diz que a cota inferior e ATINGIDA SE vale a hipotese (iii); nao diz que ela falha fora dela nem declara por classe o que e exato. As quatro leituras (L, T, C, U) sao implementadas e reportadas; nenhuma foi escolhida depois de ver o numero.",
            "m(i) tem duas definicoes no material: nodal (main.tex l.296: blocos com >= 1 no a < b; usada na proposicao) e geometrica (lados/cantos a < b, como o criterio e fismat_k_corrigido.py descrevem). Ambas reportadas; as diferencas sao discretizacao e blocos finos.",
            "'declarada exata' no criterio: so m=0 tem cota inferior = superior (p=1). Nas demais classes a proposicao so afirma 'atingida se (iii)'.",
        ],
        por_celula=res,
        trabalho_anterior=dict(
            fismat_k_corrigido=("classe geometrica m (vizinhanca-8 por distancia a face/vertice) so em Bauru Q1, 100+100 seeds; classificado contestado na folha; "
                                "aqui: m nodal exata nas 4 celulas, condicao (iii) por no, 200 seeds de 1.8, EP por bootstrap de sorteios"),
            fismat_inclusao_decomp="decomposicao de m=1/m=2/m=3 de Bauru Q1 a partir do JSON do fismat_k_corrigido; nao recomputada",
            fismat_razao_cov="identidade da razao e decomposicao da retencao por classe k lateral (1.6); nao cobre p por classe m",
            p_inclusao_por_no_1_6="perfil por classe k lateral-only com errata; nao usado como comparador",
        ),
        scripts_sha256={str(SCRIPTS / "v3_12_R4_p_fechado_vs_frequencia.py"): mo.sha256_file(Path(__file__)),
                        str(SCRIPTS / "v3_12_R3R4_motor.py"): mo.sha256_file(SCRIPTS / "v3_12_R3R4_motor.py"),
                        str(mo.S16_PATH) + " (importado, nao editado)": mo.sha256_file(mo.S16_PATH)},
        comando="/trabalho/ambientes/s33_amb_virtual/.venv/bin/python " + " ".join(sys.argv),
        data_utc=t0.isoformat(),
    )
    OUT.write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    print("gravado", OUT)


if __name__ == "__main__":
    main()
