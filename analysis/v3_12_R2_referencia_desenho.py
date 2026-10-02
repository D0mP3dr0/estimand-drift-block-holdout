#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
# -*- coding: utf-8 -*-
"""R2 / roadmap B4.1 (v3-12): referencia de desenho. Hold-out em blocos com buffer (H),
sem buffer (H0), amostra aleatoria simples de nos (A) e Hajek sobre H (J), contra o alvo
mu_U = media de e_i no dominio inteiro. Criterio fixado ANTES (nao alterar):
  criterios/criterio_R2_referencia_desenho.json (e_i: criterios/criterio_2.3.json).

Reuso por IMPORT (sem copia) de scripts/v3_1.6_2.3_estimando_formal.py (sha256 declarado na saida):
  split_uma_vez (particao 70/15/15 + buffer cKDTree exato), carregar_alvo_dominio,
  free_space_path_loss (formula congelada), build_synthetic_grid etc.
COPIA LITERAL declarada (nao havia funcao exportavel, estava inline em processar_celula, l.220-243
do original): calibracao unica do e_i no split seed=42 (constante = mediana(RSSI_treino);
FSPL(b) = offset pela mediana dos validos do treino).

p_i por no: o JSON fase2/1.6_p_inclusao_por_no.json NAO grava p_i por no (so perfis). Recalculado
aqui como frequencia de retencao no teste em exatamente os mesmos 100 sorteios do 1.6
(RandomState(20260927).randint(1e6,size=100)); conferido contra media_p_cond_te_geral do 1.6.

Desvio declarado: os 200 sorteios de fase1/1.8 (RandomState(20260926)) e os 100 do 2.3
(RandomState(20260927)) NAO tem nenhum seed em comum; a validacao de (H) contra o 2.3 e feita
rodando (H) nos 100 seeds do 2.3 (por sorteio, contra fase2/2.3b_estimando_por_sorteio.json).

Somente CPU. Saidas em fase4/: _R2_parcial_<cidade>.json (por celula), R2_por_sorteio.json,
R2_resumo.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
V3 = BASE / "_v3_2026-09-25"
ORIG = BASE / "scripts" / "v3_1.6_2.3_estimando_formal.py"
SCRIPT_PATH = Path(__file__).resolve()
OUT = V3 / "fase4"
SEEDS_JSON = V3 / "fase1" / "1.8_referencia_nodal_200seeds.json"
P16_JSON = V3 / "fase2" / "1.6_p_inclusao_por_no.json"
H23B_JSON = V3 / "fase2" / "2.3b_estimando_por_sorteio.json"
H23_JSON = V3 / "fase2" / "2.3_estimando_ht_baselines.json"
CRIT = V3 / "criterios" / "criterio_R2_referencia_desenho.json"

spec = importlib.util.spec_from_file_location("v3_16_23_orig", ORIG)
O = importlib.util.module_from_spec(spec)
sys.modules["v3_16_23_orig"] = O
spec.loader.exec_module(O)

CIDADES = O.CIDADES
QUAD = O.QUAD
G, B = O.G, O.B
PREDS = ("constante", "fspl_calibrado_b")
POPS = ("validos", "todos")
POPS_VALID = ("validos", "sentinela", "todos")
ESTS = ("H", "H0", "A", "J", "J_p200")
SEED_A_BASE = 20261001
TOL_VALID = 1e-5


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def preparar_celula(cidade):
    d = json.load(open(O.TREINOS_DIR / f"run_c0c1cf_{cidade}_s42_{QUAD}_g10b2.json"))
    geo = d["geometria"]
    lon, lat = O.build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                      geo["lat_min_deg"], geo["lat_max_deg"])
    pos_km = O.latlon_graus_para_metros(lon, lat) / 1000.0
    gid = O.assign_groups(pos_km, G)
    grupos = np.unique(gid)
    n_g = len(grupos)
    n_nos = pos_km.shape[0]
    rssi, pl, sentinela, dist, n_total, tensor_path = O.carregar_alvo_dominio(cidade, QUAD)
    assert n_total == n_nos
    # ---- COPIA LITERAL de v3_1.6_2.3_estimando_formal.py l.220-243 (calibracao unica seed=42) ----
    rng42 = np.random.RandomState(O.SEED_REF)
    emb42 = grupos.copy(); rng42.shuffle(emb42)
    n_tr42 = max(1, int(round(O.FRACS[0] * n_g))); n_va42b = max(1, int(round(O.FRACS[1] * n_g)))
    if n_tr42 + n_va42b >= n_g:
        n_tr42 = max(1, n_g - 2); n_va42b = 1
    g_tr42 = emb42[:n_tr42]
    m_tr_42 = np.isin(gid, g_tr42)
    rssi_tr42 = rssi[m_tr_42]
    dist_tr42 = dist[m_tr_42]
    sent_tr42 = sentinela[m_tr_42]
    constante_ref = float(np.median(rssi_tr42))
    validos_tr42 = ~sent_tr42
    pl_pred_tr42_v = O.free_space_path_loss(dist_tr42[validos_tr42], O.FREQ_MHZ)
    p_tx_eff_ref = float(np.median(rssi_tr42[validos_tr42] + pl_pred_tr42_v))
    pl_pred_dom = O.free_space_path_loss(dist, O.FREQ_MHZ)
    rssi_fspl_dom = p_tx_eff_ref - pl_pred_dom
    e_constante = np.abs(rssi - constante_ref)
    e_fspl = np.abs(rssi - rssi_fspl_dom)
    # ---- fim da copia literal ----
    return dict(pos_km=pos_km, gid=gid, grupos=grupos, n_g=n_g, n_nos=n_nos, sentinela=sentinela,
                e={"constante": e_constante, "fspl_calibrado_b": e_fspl},
                constante_ref=constante_ref, p_tx_eff_ref=p_tx_eff_ref, tensor_path=tensor_path)


def medias_pop(e, idx, sent):
    """media de e sobre idx por populacao (validos, sentinela, todos); nan se vazio."""
    s = e[idx]
    sv = sent[idx]
    out = {}
    nv = int((~sv).sum())
    out["validos"] = float(s[~sv].mean()) if nv > 0 else float("nan")
    out["sentinela"] = float(s[sv].mean()) if (len(s) - nv) > 0 else float("nan")
    out["todos"] = float(s.mean()) if len(s) > 0 else float("nan")
    return out, nv


def hajek_pop(e, idx, sent, p):
    """Hajek sum(e/p)/sum(1/p) sobre idx (p>0), por populacao validos/todos; devolve tb n descartados (p=0)."""
    pp = p[idx]
    ok = pp > 0
    n_drop = int((~ok).sum())
    ii = idx[ok]
    w = 1.0 / pp[ok]
    s = e[ii]
    sv = sent[ii]
    out = {}
    wv = w[~sv]
    out["validos"] = float((s[~sv] * wv).sum() / wv.sum()) if wv.size else float("nan")
    out["todos"] = float((s * w).sum() / w.sum()) if w.size else float("nan")
    return out, n_drop


def processar_celula(cidade):
    t0 = time.time()
    cid_q = f"{cidade}_{QUAD}"
    S = preparar_celula(cidade)
    pos_km, gid, grupos, n_g, n_nos, sent, e = (S[k] for k in ("pos_km", "gid", "grupos", "n_g", "n_nos", "sentinela", "e"))
    mu_U = {pred: {"validos": float(e[pred][~sent].mean()), "todos": float(e[pred].mean()),
                   "sentinela": float(e[pred][sent].mean())} for pred in PREDS}
    seeds200 = json.load(open(SEEDS_JSON))["seeds"]
    seeds100 = np.random.RandomState(20260927).randint(10**6, size=100).tolist()
    assert len(seeds200) == 200

    # ---------- PASSO A: 100 seeds do 1.6/2.3: p_i (frequencia) + (H) para validacao ----------
    cnt_ret = np.zeros(n_nos, dtype=np.int32)
    cnt_blo = np.zeros(n_nos, dtype=np.int32)
    H_val = {pred: {pop: [] for pop in POPS_VALID} for pred in PREDS}
    M_val = []
    for s in seeds100:
        m_te0, m_va0, m_te, m_va, kte, ktr = O.split_uma_vez(pos_km, gid, grupos, n_g, s)
        cnt_ret += m_te.astype(np.int32)
        cnt_blo += m_te0.astype(np.int32)
        idx = np.flatnonzero(m_te)
        for pred in PREDS:
            mp, nv = medias_pop(e[pred], idx, sent)
            for pop in POPS_VALID:
                H_val[pred][pop].append(mp[pop])
        M_val.append(int((~sent[idx]).sum()))
        del m_te0, m_va0, m_te, m_va
    p16 = cnt_ret.astype(np.float64) / 100.0
    com_b = cnt_blo > 0
    p_cond_geral = float(cnt_ret[com_b].sum() / cnt_blo[com_b].sum())
    ref16 = json.load(open(P16_JSON))["por_celula"][cid_q]["media_p_cond_te_geral"]

    # conferencia contra 2.3b (por sorteio) e 2.3 (agregado c)
    b23 = json.load(open(H23B_JSON))["por_celula"][cid_q]
    a23 = json.load(open(H23_JSON))["por_celula"][cid_q]
    conf = {"p_cond_te_geral_recalculado": p_cond_geral, "p_cond_te_geral_1.6": ref16,
            "dif_p_cond": abs(p_cond_geral - ref16), "por_pred_pop": {}}
    ok_all = abs(p_cond_geral - ref16) < 1e-12
    for pred in PREDS:
        for pop in POPS_VALID:
            ref = b23[pred][pop]
            Msig = np.asarray(ref["M_sigma_todos_100"])
            esperado = np.asarray(ref["Err_sigma_por_sorteio_A"], dtype=float)
            mine = np.asarray(H_val[pred][pop], dtype=float)
            # populacao 'validos'/'sentinela'/'todos': sorteios com M>0 naquela populacao
            if pop == "validos":
                ok_s = np.asarray(M_val) > 0
            elif pop == "todos":
                ok_s = np.ones(100, bool)
            else:
                ok_s = ~np.isnan(mine)
            mine_A = mine[ok_s]
            assert (Msig > 0).sum() == len(esperado), (cid_q, pred, pop)
            same_mask = bool(np.array_equal(Msig > 0, ok_s))
            maxdif = float(np.max(np.abs(mine_A - esperado))) if same_mask and len(esperado) == len(mine_A) else float("inf")
            # c (razao de esperancas, p16) contra 2.3 (arredondado a 3 casas)
            mp = ~sent if pop == "validos" else (sent if pop == "sentinela" else np.ones(n_nos, bool))
            c_mine = float((p16[mp] * e[pred][mp]).sum() / p16[mp].sum())
            c_dif = abs(round(c_mine, 3) - ref["c_razao_esperancas_p_te_dB"])
            conf["por_pred_pop"][f"{pred}/{pop}"] = dict(
                n_sorteios_A=int(len(esperado)), mesma_mascara_M_maior_0=same_mask,
                max_abs_dif_Err_por_sorteio=maxdif, c_recalculado=c_mine, c_2_3=ref["c_razao_esperancas_p_te_dB"],
                dif_c_arred_3casas=c_dif,
                b_media_recalc=round(float(mine_A.mean()), 3) if len(mine_A) else None,
                b_media_2_3=ref["b_media_sorteios_mae_teste_dB"])
            ok_all &= same_mask and maxdif < TOL_VALID and c_dif < 1e-9
    conf["passou"] = bool(ok_all)
    conf["tolerancia_Err_por_sorteio"] = TOL_VALID
    log(f"{cid_q}: validacao (H) vs 2.3b passou={conf['passou']} t={time.time()-t0:.0f}s")
    if not ok_all:
        out = {"celula": cid_q, "validacao_H_vs_2.3": conf, "abortado": True}
        (OUT / f"_R2_parcial_{cidade}_VALIDACAO_FALHOU.json").write_text(json.dumps(out, indent=1))
        raise RuntimeError(f"{cid_q}: validacao de (H) falhou, ver parcial")
    n_zero_p16 = int((p16 == 0).sum())

    # ---------- PASSO B: 200 seeds do 1.8 ----------
    res = {pred: {pop: {est: [] for est in ESTS} for pop in POPS} for pred in PREDS}
    info = {"n_te_b2": [], "n_te_b0": [], "n_te_b2_validos": [], "n_A_validos": [],
            "n_descartados_p16_zero_no_teste": [], "n_descartados_p200_zero_no_teste": []}
    cnt200 = np.zeros(n_nos, dtype=np.int32)
    idx_te_guardados = []
    for j, s in enumerate(seeds200):
        m_te0, m_va0, m_te, m_va, kte, ktr = O.split_uma_vez(pos_km, gid, grupos, n_g, s)
        idx = np.flatnonzero(m_te).astype(np.int32)
        idx0 = np.flatnonzero(m_te0).astype(np.int32)
        cnt200 += m_te.astype(np.int32)
        idx_te_guardados.append(idx)
        del m_te0, m_va0, m_te, m_va
        rs = np.random.RandomState(SEED_A_BASE + j)
        idxA = rs.choice(n_nos, size=len(idx), replace=False)
        info["n_te_b2"].append(int(len(idx)))
        info["n_te_b0"].append(int(len(idx0)))
        info["n_te_b2_validos"].append(int((~sent[idx]).sum()))
        info["n_A_validos"].append(int((~sent[idxA]).sum()))
        for pred in PREDS:
            for est, ii in (("H", idx), ("H0", idx0), ("A", idxA)):
                mp, _ = medias_pop(e[pred], ii, sent)
                for pop in POPS:
                    res[pred][pop][est].append(mp[pop])
            jj, nd = hajek_pop(e[pred], idx, sent, p16)
            for pop in POPS:
                res[pred][pop]["J"].append(jj[pop])
            if pred == PREDS[0]:
                info["n_descartados_p16_zero_no_teste"].append(nd)
        if (j + 1) % 20 == 0:
            log(f"{cid_q}: passo B {j+1}/200 t={time.time()-t0:.0f}s")
    p200 = cnt200.astype(np.float64) / 200.0
    for idx in idx_te_guardados:
        for pred in PREDS:
            jj, nd = hajek_pop(e[pred], idx, sent, p200)
            for pop in POPS:
                res[pred][pop]["J_p200"].append(jj[pop])
            if pred == PREDS[0]:
                info["n_descartados_p200_zero_no_teste"].append(nd)

    out = dict(celula=cid_q, tensor_path=S["tensor_path"], n_nos=int(n_nos),
               constante_ref_dB=S["constante_ref"], p_tx_eff_ref_dB=S["p_tx_eff_ref"],
               mu_U=mu_U, n_nos_validos=int((~sent).sum()),
               n_nos_com_p16_zero=n_zero_p16, n_nos_com_p200_zero=int((p200 == 0).sum()),
               validacao_H_vs_2_3=conf, por_sorteio=res, info_por_sorteio=info,
               seeds_split=seeds200, tempo_s=time.time() - t0)
    (OUT / f"_R2_parcial_{cidade}.json").write_text(json.dumps(out, indent=1))
    log(f"{cid_q}: parcial gravado, t={time.time()-t0:.0f}s")
    return cidade


def stats(x, mu):
    a = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(a)
    if n < 2:
        return dict(n=n)
    sd = float(a.std(ddof=1))
    return dict(n=n, media_estimativas=float(a.mean()), vies=float(a.mean() - mu), ep_mc_vies=sd / math.sqrt(n),
                vies_em_ep=float((a.mean() - mu) / (sd / math.sqrt(n))) if sd > 0 else None,
                reqm=float(math.sqrt(np.mean((a - mu) ** 2))), variancia=float(a.var(ddof=1)), dp=sd,
                n_sorteios_indefinidos=int(len(x) - n))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    faltam = [c for c in CIDADES if not (OUT / f"_R2_parcial_{c}.json").exists()]
    if faltam:
        from multiprocessing import Pool
        with Pool(len(faltam)) as p:
            p.map(processar_celula, faltam)
    cel = {c: json.load(open(OUT / f"_R2_parcial_{c}.json")) for c in CIDADES}

    resumo_c = {}
    por_sorteio = {}
    for c in CIDADES:
        r = cel[c]
        cid_q = r["celula"]
        resumo_c[cid_q] = {}
        por_sorteio[cid_q] = dict(mu_U=r["mu_U"], por_sorteio=r["por_sorteio"], info_por_sorteio=r["info_por_sorteio"])
        for pred in PREDS:
            resumo_c[cid_q][pred] = {}
            for pop in POPS:
                mu = r["mu_U"][pred][pop]
                est = {k: stats(r["por_sorteio"][pred][pop][k], mu) for k in ESTS}
                vA = est["A"].get("variancia")
                deff = {"deff_H_sobre_A": est["H"]["variancia"] / vA if vA else None,
                        "deff_H0_sobre_A": est["H0"]["variancia"] / vA if vA else None,
                        "reqm_J_sobre_reqm_H": est["J"]["reqm"] / est["H"]["reqm"],
                        "reqm_J_p200_sobre_reqm_H": est["J_p200"]["reqm"] / est["H"]["reqm"]}
                resumo_c[cid_q][pred][pop] = dict(mu_U=mu, estimadores=est, razoes=deff)

    # contagens mecanicas dos limiares do criterio (sem interpretacao)
    tally = {}
    for pred in PREDS:
        for pop in POPS:
            cs = [resumo_c[f"{c}_{QUAD}"][pred][pop] for c in CIDADES]
            tally[f"{pred}/{pop}"] = dict(
                celulas_deff_H_ge_10=sum(x["razoes"]["deff_H_sobre_A"] >= 10 for x in cs),
                celulas_deff_H_lt_2=sum(x["razoes"]["deff_H_sobre_A"] < 2 for x in cs),
                celulas_abs_vies_H_le_2EP=sum(abs(x["estimadores"]["H"]["vies_em_ep"]) <= 2 for x in cs),
                celulas_abs_vies_H_gt_2EP=sum(abs(x["estimadores"]["H"]["vies_em_ep"]) > 2 for x in cs),
                celulas_reqm_J_lt_metade_H=sum(x["razoes"]["reqm_J_sobre_reqm_H"] < 0.5 for x in cs),
                celulas_reqm_J_p200_lt_metade_H=sum(x["razoes"]["reqm_J_p200_sobre_reqm_H"] < 0.5 for x in cs))

    comum = dict(
        script=str(SCRIPT_PATH), script_sha256=sha256_file(SCRIPT_PATH),
        script_original_importado=str(ORIG), script_original_sha256=sha256_file(ORIG),
        criterio=str(CRIT), criterio_sha256=sha256_file(CRIT),
        fonte_seeds_split=str(SEEDS_JSON), fonte_seeds_sha256=sha256_file(SEEDS_JSON),
        sementes=dict(split_200="fase1/1.8 'seeds' (RandomState(20260926).randint(1e6,size=200)); ver seeds_split em cada celula",
                      split_p_i_e_validacao_100="RandomState(20260927).randint(1e6,size=100) (1.6/2.3)",
                      amostra_aleatoria="RandomState(20261001 + indice_do_sorteio).choice(n_nos, size=n_te_b2, replace=False)"),
        comando=" ".join(["/trabalho/ambientes/s33_amb_virtual/.venv/bin/python", str(SCRIPT_PATH)]),
        data_utc=datetime.now(timezone.utc).isoformat(),
        desvios_do_criterio=[
            "Os 200 seeds de fase1/1.8 (RandomState(20260926)) e os 100 de fase2/2.3 (RandomState(20260927)) nao compartilham nenhum seed; "
            "nao existem '100 sorteios comuns'. A validacao de (H) foi feita rodando (H) b=2 nos 100 seeds do 2.3 contra "
            "Err_sigma_por_sorteio_A de fase2/2.3b_estimando_por_sorteio.json (por sorteio) e c de fase2/2.3.",
            "p_i por no nao esta gravado em fase2/1.6_p_inclusao_por_no.json (so perfis); recalculado como frequencia de retencao nos "
            "mesmos 100 sorteios do 1.6 e conferido contra media_p_cond_te_geral do 1.6. Nos de teste com p_i=0 nessa frequencia sao descartados no Hajek "
            "(contagem em info_por_sorteio.n_descartados_p16_zero_no_teste).",
            "Estimador suplementar J_p200 (nao pedido): Hajek com p_i = frequencia nos proprios 200 sorteios (sem p_i=0 no teste); p dependente da amostra.",
            "Nos validos, sorteios sem no valido no teste (M=0) ficam indefinidos em H/H0/J (n em estatisticas < 200; n_sorteios_indefinidos). "
            "A e sempre definido.",
            "H0 tem tamanho de teste maior que H (sem aparar); A usa o tamanho de H (teste retido com b=2), como pedido."])
    (OUT / "R2_por_sorteio.json").write_text(json.dumps(dict(**comum, celulas=por_sorteio), indent=1))
    (OUT / "R2_resumo.json").write_text(json.dumps(dict(
        **comum, validacao_H_vs_2_3={cel[c]["celula"]: cel[c]["validacao_H_vs_2_3"] for c in CIDADES},
        n_nos_p_zero={cel[c]["celula"]: dict(p16=cel[c]["n_nos_com_p16_zero"], p200=cel[c]["n_nos_com_p200_zero"]) for c in CIDADES},
        resumo_por_celula=resumo_c, contagem_mecanica_limiares=tally), indent=1))
    log(f"gravado R2_por_sorteio.json e R2_resumo.json; total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
