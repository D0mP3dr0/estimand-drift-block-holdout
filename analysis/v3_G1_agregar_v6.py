#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
# -*- coding: utf-8 -*-
"""
Agregacao do lote G1 (modelos treinados a g = 10 km, b = 2 km), VERSAO 6.
COPIA DECLARADA de scripts/v3_G1_agregar_v5.py
  sha256 do v5: 50610e19f9ea1314c899b86b21eeefb569b74e64093f28936ee1c03a6aa0e465
(v1 a v5 NAO sao editados). UNICA mudanca: a aceitacao de corrida `sem_validos` sem .npz passa a seguir
criterios/criterio_G1_adendo3.json (fixado antes de qualquer agregado do bloco 3). Fato do lote: nos sorteios
sem no valido o treinador aborta ANTES do treino com "PARTICAO DEGENERADA" (rc=1) e grava um run JSON PARCIAL
(sem selecao/insumos/modelo_v3, sem .npz, sem sidecar). Uma corrida assim e aceita como sem_validos SE E SOMENTE SE:
  (a) o 2.1 registra validos == 0 para a celula e o split_seed;
  (b) lote_G1_status.json tem, para o run_label, rc != 0 e sem_validos == true, e o log da corrida (campo `log`
      do status) contem "PARTICAO DEGENERADA";
  (c) o run JSON parcial existe, e traz: run_label, seed e split_seed do plano (topo, config e split), g = 10 e
      b = 2 (config, geometria, split), dataset com o NOME esperado e rf_data_bytes igual ao tamanho do manifest v4
      (LIDOS DO BLOCO `dataset` do parcial; o parcial NAO grava sha256 do dataset -- o manifest v4 tem
      `tamanho_bytes`, e a conferencia de tamanho e o que o parcial permite; sha nao e verificavel nele),
      particoes.test.n_pl_alvo_valido == 0 e guarda_particao_degenerada.n_pl_alvo_valido.test == 0 e `test` em
      particoes_sem_aresta_antena;
  e o run JSON NAO traz selecao/insumos/modelo_v3, e a pasta NAO tem sidecar nem .npz (senao e corrida completa
  com flag inconsistente: aborta). Conferencias de treino concluido (selecao, .npz, sidecar, amarracao, config x A4)
  nao se aplicam. Qualquer item faltando: aborta (rc=4). Run JSON sem .npz e sem registro sem_validos, com o lote ja encerrado (veredito_lote preenchido), tambem aborta; com o lote em andamento e corrida incompleta (recusa rc=2). Blocos 1 e 2 e corridas completas: sem mudanca.
O ramo `m is None` de conferir_corrida (v3-v5) fica sem uso. Saida: gpu/G1/agregado_G1_v6_bloco<N>.json.
Restante = v3/v4/v5 (ver cabecalhos). Este script NAO interpreta resultados.
"""
import argparse
import glob
import hashlib
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constantes
BASE_PADRAO = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
RAIZ_PADRAO = BASE_PADRAO / "_v3_2026-09-25"
SHA_ORIGINAL_V1 = "88d444d4c480b7758ccd197f6936d5835a78c71863d99933fe8cbcfcd2419d2b"
SHA_V2 = "d94fac29dcf6c694cab9bec55c4bdd9e2701fe5baf2675cbfe0aa77a5c155824"
SHA_V3 = "5102299e6e5b0b3d25009693ad04c3dc0d9341833252b406841e2e894bae0e16"
SHA_V4 = "4263076007a28aa76c5e51fa73c35c513b34131c8019f2b5089981a57e916b14"
SHA_V5 = "50610e19f9ea1314c899b86b21eeefb569b74e64093f28936ee1c03a6aa0e465"
TRECHO_LOG_DEGENERADA = "PARTICAO DEGENERADA"
TOL_AMARRA_MAE_REL = 0.10
# Origem dos numeros: gpu/G1_agregador_v3_contra_auditoria/veredito.json (A4: 40 runs, diferenca propria npz x run JSON
# ate 1,9 % no RMSE e 3,9 % no MAE; GNN contra MLP da mesma celula e sorteio: minimo 10,9 % no RMSE e 2,3 % no MAE) e
# corridas reais do G1: diferenca propria de MAE de 6,35 % numa corrida legitima (o MAE em todos os nos e dominado
# pelos sentinelas, onde o erro absoluto e pequeno, e por isso a diferenca relativa e instavel) e RMSE proprio maximo
# de 0,27 %. Por isso MAE 10 % (frouxo) e RMSE 3 % (quem distingue). As DUAS tolerancias sao exigidas ao mesmo tempo.
TOL_AMARRA_RMSE_REL = 0.03
SENTINELA_PL = 299.0
RUIDO_REPETICAO_DB = 0.132      # maior diferenca entre repeticoes do GNN (A2c); nao e um dp (adendo 2)
RUIDO_SENTINELA_DB = 0.117      # maior diferenca entre repeticoes nos sentinelas (A2c)
FATOR_CONDICAO1 = 3.0
FATOR_DELIMITA = 2.0
K_MIN_BLOCO3 = 10
N_SEMENTES_B2 = 3
GRID_KM, BUFFER_KM = 10.0, 2.0
POPS = ("mae_rssi_validos_db", "mae_rssi_sentinela_db", "mae_rssi_todos_db", "mae_pl_validos_db")
BASELINES = ("fspl", "hata_rural", "cost231_sub")
CELULAS_POR_BLOCO = {1: ("bauru_Q1", "campinas_Q1"), 2: ("bauru_Q1", "campinas_Q1"), 3: ("bauru_Q3", "campinas_Q3")}
# chaves do config que a conferencia A4 ignora (adendo 7: grid_km, seed, split_seed, rotulos e caminhos)
CONFIG_IGNORADAS_A4 = ("grid_km", "seed", "split_seed", "run_label", "evid_dir", "graph_dir", "graph_file",
                       "rf_data_file", "base_dir")
TOL_FLOAT = 0.0   # igualdade exata nos g/b (10.0 e 2.0 sao representaveis)


class Aborta(Exception):
    """Conferencia do item 7 violada: rc=4, nada gravado."""


class Incompleto(Exception):
    """Bloco incompleto/recusado: rc=2."""


class Caminhos:
    pass


P = Caminhos()


def definir_caminhos(raiz: Path) -> None:
    P.RAIZ = raiz
    P.BASE = raiz.parent
    P.G1 = raiz / "gpu" / "G1"
    P.A4 = raiz / "gpu" / "A4"
    P.CRIT = raiz / "criterios" / "criterio_G1_modelos_g10.json"
    P.ADENDO = raiz / "criterios" / "criterio_G1_adendo1.json"
    P.ADENDO2 = raiz / "criterios" / "criterio_G1_adendo2.json"
    P.ADENDO3 = raiz / "criterios" / "criterio_G1_adendo3.json"
    P.PLANO = P.G1 / "plano_G1.json"
    P.STATUS = P.G1 / "lote_G1_status.json"
    P.PROV = P.G1 / "proveniencia_scripts_G1.json"
    P.PARCIAL_2_1 = raiz / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json"
    P.SORTEIOS_2_1 = raiz / "fase2" / "2.1_deriva_erro_baselines_16x60rnd.json"
    P.MANIFEST_V4 = P.BASE / "manifest_mathematics_v4.jsonl"


# ------------------------------------------------------------- utilitarios (v1)
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest_cftudo() -> dict:
    out = {}
    with open(P.MANIFEST_V4, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                e = json.loads(ln)
                if e.get("grupo") == "tensores_cftudo":
                    out[Path(e.get("caminho", "")).name] = e.get("sha256")
    return out


def manifest_tamanhos() -> dict:
    out = {}
    with open(P.MANIFEST_V4, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                e = json.loads(ln)
                if e.get("grupo") == "tensores_cftudo":
                    out[Path(e.get("caminho", "")).name] = e.get("tamanho_bytes")
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


def dp(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.stdev(xs)) if len(xs) >= 2 else None


def razao(a, b):
    return (a / b) if (a is not None and b not in (None, 0)) else None


def pearson(x, y):
    if len(x) < 3:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    return pearson(rx, ry)


def _finito(nome, arr):
    if not np.isfinite(arr).all():
        raise Aborta(f"valor nao finito no .npz: {nome}")


def mae_pop(z) -> dict:
    """z: dict-like (NpzFile) com target, pred, idx_global [, sentinela]. Como o v1, com finitude de todo campo usado."""
    tgt = z["target"].astype(np.float64)
    pred = z["pred"].astype(np.float64)
    if tgt.ndim != 2 or pred.shape != tgt.shape or tgt.shape[1] < 4 or tgt.shape[0] != len(z["idx_global"]):
        raise Aborta("formato do .npz inesperado (target/pred/idx_global)")
    _finito("target[:,0]", tgt[:, 0])
    _finito("target[:,3]", tgt[:, 3])
    _finito("pred[:,3]", pred[:, 3])
    sent = tgt[:, 0] >= SENTINELA_PL
    if "sentinela" in getattr(z, "files", z) and not np.array_equal(sent, z["sentinela"].astype(bool)):
        raise Aborta("campo sentinela do .npz nao bate com target[:,0] >= 299")
    val = ~sent
    if val.any():
        _finito("pred[:,0] nos validos", pred[val, 0])
    e3s = pred[:, 3] - tgt[:, 3]
    e3 = np.abs(e3s)
    e0 = np.abs(tgt[:, 0] - pred[:, 0])
    return {
        "n": int(tgt.shape[0]), "n_validos": int(val.sum()), "n_sentinela": int(sent.sum()),
        "mae_rssi_validos_db": float(e3[val].mean()) if val.any() else None,
        "mae_rssi_sentinela_db": float(e3[sent].mean()) if sent.any() else None,
        "mae_rssi_todos_db": float(e3.mean()),
        "rmse_rssi_todos_db": float(np.sqrt((e3s ** 2).mean())),
        "mae_pl_validos_db": float(e0[val].mean()) if val.any() else None,
        "idx_sha256": hash_idx(z["idx_global"]),
    }


def hash_idx(idx) -> str:
    return hashlib.sha256(np.sort(np.asarray(idx).astype(np.int64)).tobytes()).hexdigest()


def carregar_2_1() -> dict:
    with open(P.PARCIAL_2_1, "r", encoding="utf-8") as f:
        p = json.load(f)
    out = {}
    for cel, d in p["celulas"].items():
        for s in d.get("por_sorteio", []):
            if s.get("status") == "ok":
                out[(cel, int(s["split_seed"]))] = {
                    "n_test": s["n_test"], "validos": s["n_pop_teste"]["validos"], "todos": s["n_pop_teste"]["todos"],
                    "mae_constante_validos": s["mae_constante_teste"]["validos"]}
    return out


# ------------------------------------------------- emenda 1: decomposicao de um fator
def decomposicao_um_fator(M: np.ndarray) -> dict:
    """M: n sorteios (linhas) x 3 sementes (colunas). Sementes aninhadas no sorteio.
    QM_dentro = SQ_dentro/(n*(k-1)); QM_entre = SQ_entre/(n-1)
    sigma2_semente = QM_dentro; sigma2_sorteio = max(0, (QM_entre - QM_dentro)/k), k = 3."""
    n, k = M.shape
    if k != N_SEMENTES_B2:
        raise Aborta(f"decomposicao exige {N_SEMENTES_B2} sementes por sorteio, recebeu {k}")
    if n < 2:
        return None
    gm = float(M.mean())
    mu_i = M.mean(axis=1)
    sq_entre = float(k * ((mu_i - gm) ** 2).sum())
    sq_dentro = float(((M - mu_i[:, None]) ** 2).sum())
    qm_entre = sq_entre / (n - 1)
    qm_dentro = sq_dentro / (n * (k - 1))
    bruto = (qm_entre - qm_dentro) / k
    s2_sem = qm_dentro
    s2_sor = max(0.0, bruto)
    return {"n_sorteios": int(n), "n_sementes": int(k), "qm_entre": qm_entre, "qm_dentro": qm_dentro,
            "sigma2_semente": s2_sem, "sigma2_sorteio_bruto": bruto, "sigma2_sorteio": s2_sor,
            "dp_entre_sementes_pooled_db": float(np.sqrt(qm_dentro)),
            "condicao2_sigma2_sorteio_ge_sigma2_semente": bool(s2_sor >= s2_sem)}


# -------------------------------------------------------- emendas 2, 3, 5: leitura aritmetica
def avaliar_condicoes_modelo(dp_sorteios, comparador):
    r = razao(dp_sorteios, comparador)
    return {"dp_entre_sorteios_db": dp_sorteios, "comparador_db": comparador, "razao_dp_sobre_comparador": r,
            "dp_ge_3x_comparador": (None if r is None else bool(r >= FATOR_CONDICAO1)),
            "dp_lt_2x_comparador": (None if r is None else bool(r < FATOR_DELIMITA)),
            "dp_entre_2x_e_3x_comparador": (None if r is None else bool(FATOR_DELIMITA <= r < FATOR_CONDICAO1))}


def sinalizar_celula(por_modelo: dict) -> dict:
    g, m = por_modelo["gnn"], por_modelo["mlp"]
    indet = any(x["razao_dp_sobre_comparador"] is None for x in (g, m))
    cond1 = bool(not indet and g["dp_ge_3x_comparador"] and m["dp_ge_3x_comparador"])
    delim = bool(not indet and g["dp_lt_2x_comparador"] and m["dp_lt_2x_comparador"])
    return {"condicao1_gnn_e_mlp": cond1, "ambos_lt_2x": delim,
            "faixa_intermediaria": bool(not indet and not cond1 and not delim),
            "algum_modelo_entre_2x_e_3x": bool(g["dp_entre_2x_e_3x_comparador"] or m["dp_entre_2x_e_3x_comparador"]),
            "indeterminada_dp_nao_calculavel": indet}


def contar_ramos(celulas: dict, celulas_q1: list) -> dict:
    """celulas[c] = {condicao1_gnn_e_mlp, ambos_lt_2x, faixa_intermediaria, algum_modelo_entre_2x_e_3x,
    condicao2_gnn_e_mlp (so Q1), paridade_le_0_117}. Limiar: 3 de 4 celulas; 2 de 2 se so as Q1 entram (adendo 4)."""
    ind = [c for c, d in celulas.items() if d.get("indeterminada_dp_nao_calculavel")]
    if ind:
        return {"indeterminada": True, "celulas_indeterminadas": ind, "n_celulas": len(celulas),
                "satisfaz_regra_reforca": None, "satisfaz_regra_delimita": None, "satisfaz_regra_delimita_parcial": None,
                "nota": "contagem de ramos RECUSADA: dp, razao ou condicao 2 nao calculavel em celula(s) acima"}
    n = len(celulas)
    t = 3 if n == 4 else n          # n==2 -> 2 de 2 (adendo 4); n==4 -> 3 de 4
    n1 = [c for c, d in celulas.items() if d["condicao1_gnn_e_mlp"]]
    nd = [c for c, d in celulas.items() if d["ambos_lt_2x"]]
    faixa = [c for c, d in celulas.items() if d["faixa_intermediaria"]]
    f23 = [c for c, d in celulas.items() if d["algum_modelo_entre_2x_e_3x"]]
    c2 = {c: celulas[c].get("condicao2_gnn_e_mlp") for c in celulas_q1}
    c2_todas = bool(all(v is True for v in c2.values())) if c2 else False
    reforca = bool(len(n1) >= t and c2_todas)
    delimita = bool(len(nd) >= t)
    c1_sem_c2 = bool(len(n1) >= t and not c2_todas)
    impedem = bool(len(n1) < t and len(nd) < t and len(faixa) > 0)
    parcial_literal = bool(c1_sem_c2 or impedem)
    residual = bool(not reforca and not delimita and not parcial_literal)
    parcial = bool(parcial_literal or residual)
    par = [c for c, d in celulas.items() if d.get("paridade_le_0_117")]
    return {
        "n_celulas": n, "limiar_celulas": t,
        "celulas_condicao1_gnn_e_mlp": n1, "celulas_ambos_lt_2x": nd, "celulas_faixa_intermediaria": faixa,
        "celulas_algum_modelo_entre_2x_e_3x": f23,
        "condicao2_por_celula_q1": c2, "condicao2_nas_celulas_q1_todas": c2_todas,
        "satisfaz_regra_reforca": reforca, "satisfaz_regra_delimita": delimita,
        "satisfaz_regra_delimita_parcial": parcial,
        "delimita_parcial_por": {"condicao1_atendida_condicao2_nao": c1_sem_c2,
                                 "celulas_entre_2x_e_3x_impedem_limiar": impedem,
                                 "residual_nao_prevista_literalmente_no_adendo": residual},
        "celulas_paridade_sentinela_le_0_117": par, "paridade_em_todas_as_celulas": bool(len(par) == n),
        "nota": "aplicacao aritmetica das regras do criterio/adendo; leitura e do rigor, nao deste script"}


# ---------------------------------------------------------------- proveniencia global (item 7f)
def conferir_scripts_do_lote(modo_teste: bool) -> dict:
    with open(P.PROV, "r", encoding="utf-8") as f:
        prov = json.load(f)
    falhas, linhas = [], []
    for e in prov["arquivos"]:
        p = P.BASE / e["caminho"]
        if not p.exists():
            falhas.append(f"{e['caminho']}: ausente")
            continue
        obs = sha256(p)
        linhas.append({"caminho": e["caminho"], "sha256_registrado": e["sha256"], "sha256_observado": obs,
                       "igual": bool(obs == e["sha256"])})
        if obs != e["sha256"]:
            falhas.append(f"{e['caminho']}: sha256 {obs} != registrado {e['sha256']}")
    caminhos_reg = {e["caminho"] for e in prov["arquivos"]}
    for obrig in ("_v3_2026-09-25/gpu/G1/train_v3_g10.py", "_v3_2026-09-25/gpu/G1/rodar_lote_G1.py"):
        if obrig not in caminhos_reg:
            falhas.append(f"{obrig}: nao consta de proveniencia_scripts_G1.json")
    if not modo_teste:
        orig = P.BASE / "scripts" / "v3_G1_agregar.py"
        if not orig.exists() or sha256(orig) != SHA_ORIGINAL_V1:
            falhas.append("scripts/v3_G1_agregar.py (original v1) ausente ou editado (sha != cabecalho)")
    if falhas:
        raise Aborta("proveniencia dos scripts do lote: " + "; ".join(falhas))
    return {"fonte": str(P.PROV), "arquivos": linhas}


def conferir_criterios() -> tuple:
    with open(P.CRIT, "r", encoding="utf-8") as f:
        crit = json.load(f)
    with open(P.ADENDO, "r", encoding="utf-8") as f:
        adendo = json.load(f)
    if crit.get("id") != "G1_modelos_g10" or adendo.get("id") != "G1_modelos_g10_adendo1":
        raise Aborta("id do criterio/adendo inesperado")
    m = re.search(r"sha256 ([0-9a-f]{64})", adendo.get("emenda", ""))
    sc, sa = sha256(P.CRIT), sha256(P.ADENDO)
    if not m or m.group(1) != sc:
        raise Aborta(f"sha256 do criterio ({sc}) difere do citado no adendo ({m.group(1) if m else None})")
    if not P.ADENDO2.exists():
        raise Aborta("criterio_G1_adendo2.json ausente")
    with open(P.ADENDO2, "r", encoding="utf-8") as f:
        adendo2 = json.load(f)
    m2 = re.search(r"sha256 ([0-9a-f]{64})", adendo2.get("emenda", ""))
    if adendo2.get("id") != "G1_modelos_g10_adendo2" or not m2 or m2.group(1) != sa:
        raise Aborta("adendo2: id inesperado ou sha256 citado do adendo1 difere do observado")
    if not P.ADENDO3.exists():
        raise Aborta("criterio_G1_adendo3.json ausente")
    with open(P.ADENDO3, "r", encoding="utf-8") as f:
        adendo3 = json.load(f)
    m3 = re.search(r"sha256 ([0-9a-f]{64})", adendo3.get("emenda", ""))
    s2 = sha256(P.ADENDO2)
    if adendo3.get("id") != "G1_modelos_g10_adendo3" or not m3 or m3.group(1) != s2:
        raise Aborta("adendo3: id inesperado ou sha256 citado do adendo2 difere do observado")
    return sc, sa, s2, sha256(P.ADENDO3)


def conferir_plano(plano_doc: dict) -> None:
    """plano_G1.json coerente com g/b e com a lista de 60 sorteios do 2.1 (primeiros 20 e 5)."""
    if plano_doc.get("grid_km") != GRID_KM or plano_doc.get("buffer_km") != BUFFER_KM:
        raise Aborta("plano_G1.json: grid_km/buffer_km != 10/2")
    with open(P.SORTEIOS_2_1, "r", encoding="utf-8") as f:
        lista = json.load(f)["nota_divergencia_seeds"]["seeds_usados_nesta_rodada"]
    esp = {1: lista[:20], 2: lista[:5], 3: lista[:20]}
    cs = plano_doc["corridas"]
    cont = {b: sum(1 for c in cs if c["bloco"] == b) for b in (1, 2, 3)}
    if cont != {1: 80, 2: 40, 3: 80} or len(cs) != 200 or len({c["run_label"] for c in cs}) != 200:
        raise Aborta(f"plano_G1.json: contagem por bloco {cont} / rotulos unicos != 80/40/80, 200")
    for c in plano_doc["corridas"]:
        if c["split_seed"] != esp[c["bloco"]][c["indice_sorteio"]]:
            raise Aborta(f"plano_G1.json: split_seed de {c['run_label']} fora da lista do 2.1")
        rot = f"g1_{c['tipo']}_{c['cidade']}_{c['quadrante']}_ss{c['split_seed']}_s{c['seed_treino']}"
        if rot != c["run_label"]:
            raise Aborta(f"plano_G1.json: run_label {c['run_label']} != {rot}")


# ------------------------------------------------------------ referencia A4 (item 7e)
_REF_A4 = {}


def referencia_a4(tipo: str) -> dict:
    if tipo in _REF_A4:
        return _REF_A4[tipo]
    fs = sorted(glob.glob(str(P.A4 / f"{tipo}_v3_a4_bauru_Q*_ss*" / "run_*.json")))
    if not fs:
        raise Aborta(f"A4: nenhuma corrida {tipo}_v3_a4_bauru_Q*_ss* encontrada em {P.A4}")
    refs = []
    for f in fs:
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        cfg = {k: v for k, v in (d.get("config") or {}).items() if k not in CONFIG_IGNORADAS_A4}
        mv = d.get("modelo_v3") or {}
        refs.append({"config": cfg, "artefato_tipo": d.get("artefato_tipo"), "script_sha256": d.get("script_sha256"),
                     "decoder_sha256": mv.get("decoder_sha256"), "wrapper_sha256": mv.get("wrapper_sha256")})
    r0 = refs[0]
    if any(v is None for v in (r0["artefato_tipo"], r0["script_sha256"], r0["decoder_sha256"], r0["wrapper_sha256"])) or not r0["config"]:
        raise Aborta(f"A4: referencia {tipo} com campo ausente/None (artefato_tipo, script/decoder/wrapper sha, config)")
    if any(r != refs[0] for r in refs[1:]):
        raise Aborta(f"A4: as {len(fs)} corridas {tipo} nao tem config identica fora das chaves ignoradas; referencia ambigua")
    _REF_A4[tipo] = {"n_corridas_a4": len(fs), **refs[0]}
    return _REF_A4[tipo]


# ---------------------------------------------------------------- conferencias por corrida
def _igual(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= TOL_FLOAT


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and bool(np.isfinite(x))


def _rel_ok(a, b, tol) -> bool:
    """|a-b| <= tol*|b|; None, nao numerico ou nao finito = violacao."""
    return _num(a) and _num(b) and b != 0 and abs(a - b) <= tol * abs(b)


def conferir_corrida(c: dict, dj: dict, idx_global, side, man: dict, ref21: dict, m) -> dict:
    """m = mae_pop(z) da corrida (com .npz) ou None (corrida `sem_validos` sem .npz: conferencias que dependem do
    .npz/sidecar nao se aplicam e a regra C2 passa a valer)."""
    lbl, tipo = c["run_label"], c["tipo"]
    falhas = []

    def ex(nome, ok, det=""):
        if not ok:
            falhas.append(f"{nome}{(' (' + str(det) + ')') if det else ''}")

    cfg, geo, spl = dj.get("config") or {}, dj.get("geometria") or {}, dj.get("split") or {}
    ins = dj.get("insumos") or {}
    ptest = (dj.get("particoes") or {}).get("test") or {}
    sel = (dj.get("selecao") or {}).get("test_no_melhor_ckpt") or {}
    # rotulo
    ex("run_label", dj.get("run_label") == lbl and cfg.get("run_label") == lbl, dj.get("run_label"))
    # (a) dataset
    esp_nome = f"transfer_dataset_{c['cidade']}_v19_{c['quadrante']}_enriched_cftudo.pt"
    ex("dataset_nome", Path(cfg.get("rf_data_file") or "").name == esp_nome
       and Path(ins.get("rf_data_file") or "").name == esp_nome, (cfg.get("rf_data_file"), ins.get("rf_data_file")))
    ex("dataset_sha256_x_manifest_v4", bool(ins.get("sha256_rf_data")) and man.get(esp_nome) == ins.get("sha256_rf_data"),
       (ins.get("sha256_rf_data"), man.get(esp_nome)))
    ex("dataset_flag_bate_manifest_v4", ins.get("sha256_rf_data_bate_manifest_v4") is True)
    # (b) g/b no run JSON
    gb_run = {"config.grid_km": cfg.get("grid_km"), "geometria.grid_km_usado": geo.get("grid_km_usado"),
              "split.grid_km": spl.get("grid_km")}
    bb_run = {"config.buffer_km": cfg.get("buffer_km"), "geometria.buffer_km_usado": geo.get("buffer_km_usado"),
              "split.buffer_km": spl.get("buffer_km")}
    for k, v in gb_run.items():
        ex(f"run_json_g=10[{k}]", _igual(v, GRID_KM), v)
    for k, v in bb_run.items():
        ex(f"run_json_b=2[{k}]", _igual(v, BUFFER_KM), v)
    # (c) sementes x plano
    ex("seed_x_plano", dj.get("seed") == c["seed_treino"] and cfg.get("seed") == c["seed_treino"],
       (dj.get("seed"), cfg.get("seed"), c["seed_treino"]))
    ex("split_seed_x_plano", dj.get("split_seed") == c["split_seed"] and cfg.get("split_seed") == c["split_seed"]
       and spl.get("split_seed") == c["split_seed"], (dj.get("split_seed"), cfg.get("split_seed"), spl.get("split_seed"), c["split_seed"]))
    # (C1) contagens x 2.1 (mesma celula e split_seed)
    ref = ref21.get((f"{c['cidade']}_{c['quadrante']}", c["split_seed"]))
    ex("ref_2_1_presente", ref is not None)
    n_test_run = (spl.get("n_nos_apos_buffer") or {}).get("test")
    if ref is not None:
        ex("n_test_run_x_2_1", n_test_run == ref["n_test"], (n_test_run, ref["n_test"]))
        ex("particoes_test_n_x_2_1", ptest.get("n") == ref["n_test"], (ptest.get("n"), ref["n_test"]))
    if m is None:
        # (C2) sem_validos sem .npz: run JSON lido; 2.1 validos == 0 e run JSON sem alvo valido
        ex("C2_sem_validos_2_1_validos==0", ref is not None and ref["validos"] == 0, ref and ref["validos"])
        ex("C2_sem_validos_run_json_n_pl_alvo_valido==0", ptest.get("n_pl_alvo_valido") == 0, ptest.get("n_pl_alvo_valido"))
        h = None
        dif_rel = None
    else:
        if ref is not None:
            ex("n_validos_npz_x_2_1", m["n_validos"] == ref["validos"], (m["n_validos"], ref["validos"]))
            ex("n_todos_npz_x_2_1", m["n"] == ref["todos"], (m["n"], ref["todos"]))
        # (b) sidecar
        if not isinstance(side, dict):
            ex("sidecar_shim_g10_presente", False)
        else:
            ex("sidecar_artefato", side.get("artefato") == "shim_g10" and side.get("run_label") == lbl
               and side.get("modelo") == tipo, (side.get("artefato"), side.get("run_label"), side.get("modelo")))
            ex("sidecar_g_pedido=10", _igual(side.get("grid_km_pedido"), GRID_KM), side.get("grid_km_pedido"))
            ex("sidecar_b_pedido=2", _igual(side.get("buffer_km_pedido"), BUFFER_KM), side.get("buffer_km_pedido"))
            ef = side.get("efetivo") or {}
            for k in ("config.grid_km", "geometria.grid_km_usado", "split.grid_km"):
                ex(f"sidecar_g_efetivo=10[{k}]", _igual(ef.get(k), GRID_KM), ef.get(k))
            for k in ("config.buffer_km", "geometria.buffer_km_usado", "split.buffer_km"):
                ex(f"sidecar_b_efetivo=2[{k}]", _igual(ef.get(k), BUFFER_KM), ef.get(k))
            ex("sidecar_g_b_efetivos_conferem", side.get("g_b_efetivos_conferem") is True)
            ex("sidecar_npz_igual_teste_retido", side.get("npz_igual_teste_retido") is True)
            ex("sidecar_n_nos_x_run_x_npz", n_test_run is not None and side.get("n_nos_npz") == len(idx_global)
               == side.get("n_nos_teste_retidos_run_json") == n_test_run,
               (side.get("n_nos_npz"), len(idx_global), side.get("n_nos_teste_retidos_run_json"), n_test_run))
        # (d) hash dos indices
        h = hash_idx(idx_global)
        ex("hash_idx_npz_x_run_json", ptest.get("idx_sha256_global") is not None and h == ptest.get("idx_sha256_global"),
           (h, ptest.get("idx_sha256_global")))
        ex("n_idx_npz_x_particoes_test", ptest.get("n") == len(idx_global), (ptest.get("n"), len(idx_global)))
        # (C3) amarracao do .npz ao modelo/corrida: metricas recalculadas do .npz x selecao.test_no_melhor_ckpt do run JSON
        ex("amarra_npz_n_nos", sel.get("n_nos") == m["n"], (sel.get("n_nos"), m["n"]))
        ex("amarra_npz_n_pl_alvo_valido", sel.get("n_pl_alvo_valido") == m["n_validos"] == ptest.get("n_pl_alvo_valido"),
           (sel.get("n_pl_alvo_valido"), m["n_validos"], ptest.get("n_pl_alvo_valido")))
        dif_rel = {"dif_rel_mae": (abs(m["mae_rssi_todos_db"] - sel["mae_rssi_db"]) / abs(sel["mae_rssi_db"]) if _num(sel.get("mae_rssi_db")) and sel.get("mae_rssi_db") != 0 else None),
                   "dif_rel_rmse": (abs(m["rmse_rssi_todos_db"] - sel["rmse_rssi_db"]) / abs(sel["rmse_rssi_db"]) if _num(sel.get("rmse_rssi_db")) and sel.get("rmse_rssi_db") != 0 else None)}
        ex("amarra_npz_mae_rssi_todos", _rel_ok(m["mae_rssi_todos_db"], sel.get("mae_rssi_db"), TOL_AMARRA_MAE_REL),
           "MAE do .npz difere do run JSON alem da tolerancia")
        ex("amarra_npz_rmse_rssi_todos", _rel_ok(m["rmse_rssi_todos_db"], sel.get("rmse_rssi_db"), TOL_AMARRA_RMSE_REL),
           "RMSE do .npz difere do run JSON alem da tolerancia")
    # (e) config x A4
    ref_a4 = referencia_a4(tipo)
    cfg_cmp = {k: v for k, v in cfg.items() if k not in CONFIG_IGNORADAS_A4}
    if cfg_cmp != ref_a4["config"]:
        dif = sorted(k for k in set(cfg_cmp) | set(ref_a4["config"]) if cfg_cmp.get(k) != ref_a4["config"].get(k) or
                     (k in cfg_cmp) != (k in ref_a4["config"]))
        ex("config_igual_A4", False, {k: (cfg_cmp.get(k), ref_a4["config"].get(k)) for k in dif})
    mv = dj.get("modelo_v3") or {}
    ex("script_sha_igual_A4", dj.get("script_sha256") == ref_a4["script_sha256"] and dj.get("artefato_tipo") == ref_a4["artefato_tipo"]
       and mv.get("decoder_sha256") == ref_a4["decoder_sha256"] and mv.get("wrapper_sha256") == ref_a4["wrapper_sha256"])
    if falhas:
        raise Aborta(f"{lbl}: " + "; ".join(falhas))
    return {"run_label": lbl, "passou": True, "com_npz": m is not None, "dataset_sha256": ins.get("sha256_rf_data"), "idx_sha256_teste": h, "amarracao": dif_rel}


# ------------------------------------------------------------------- agregador
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bloco", type=int, choices=(1, 2, 3), default=None,
                    help="default: agrega todo bloco COMPLETO; blocos incompletos sao recusados (rc=2)")
    ap.add_argument("--bloco3-truncado", action="store_true",
                    help="declara que o lote terminou e o bloco 3 segue incompleto: usa os primeiros k sorteios com as 4 "
                         "corridas completas (exige k validos >= 10 por celula). Adendo, emenda 4.")
    ap.add_argument("--so-conferir", action="store_true",
                    help="roda so as conferencias do item 7 nas corridas ja completas do bloco pedido; nao calcula MAE, nao grava")
    ap.add_argument("--raiz-teste", default=None,
                    help="SO PARA TESTES: raiz alternativa (pasta _v3_*); grava modo_teste=true no JSON")
    args = ap.parse_args(argv)
    argv_efetivo = sys.argv if argv is None else [sys.argv[0]] + list(argv)
    modo_teste = args.raiz_teste is not None
    definir_caminhos(Path(args.raiz_teste) if modo_teste else RAIZ_PADRAO)
    _REF_A4.clear()

    try:
        return _executar(args, argv_efetivo, modo_teste)
    except Aborta as e:
        print(f"ABORTADO (conferencia do item 7; nada gravado): {e}", file=sys.stderr)
        return 4
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ABORTADO (entrada de conferencia ausente/ilegivel; nada gravado): {type(e).__name__}: {e}", file=sys.stderr)
        return 4


def _ler_sidecar(sd):
    if not sd:
        return None
    try:
        with open(sd[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _executar(args, argv_efetivo, modo_teste) -> int:
    sha_crit, sha_adendo, sha_adendo2, sha_adendo3 = conferir_criterios()
    prov_scripts = conferir_scripts_do_lote(modo_teste)
    with open(P.PLANO, "r", encoding="utf-8") as f:
        plano_doc = json.load(f)
    conferir_plano(plano_doc)
    plano = plano_doc["corridas"]
    with open(P.STATUS, "r", encoding="utf-8") as f:
        status = json.load(f)
    sem_val = {c["run_label"] for c in status["corridas"] if c.get("sem_validos")}
    status_por_lbl = {c["run_label"]: c for c in status["corridas"]}
    tam_man = manifest_tamanhos()
    man = manifest_cftudo()
    ref21 = carregar_2_1()
    sha_plano = sha256(P.PLANO)

    # ---- completude
    def artefatos(lbl):
        """C3: exatamente (no maximo) 1 run JSON, 1 .npz e 1 sidecar por pasta, com o nome do padrao do lote."""
        d = P.G1 / lbl
        achados = []
        for pat, esp in (("run_*.json", f"run_{lbl}.json"), ("predicoes_*.npz", f"predicoes_{lbl}.npz"),
                         ("shim_g10_*.json", f"shim_g10_{lbl}.json")):
            fs = sorted(glob.glob(str(d / pat)))
            if len(fs) > 1 or (fs and Path(fs[0]).name != esp):
                raise Aborta(f"{lbl}: arquivos {pat} inesperados na pasta {[Path(x).name for x in fs]} (esperado exatamente {esp})")
            achados.append(fs)
        return (d, *achados)

    def completa(c) -> bool:
        lbl = c["run_label"]
        d, js, zs, sd = artefatos(lbl)
        if lbl in sem_val and not zs:
            return True           # aceitacao so no carregar(): exige run JSON conferido e 2.1 validos == 0 (C2)
        if js and not zs and lbl not in sem_val and status.get("veredito_lote") is not None:
            # run JSON sem predicoes e sem registro sem_validos com o lote ENCERRADO: nao e corrida em andamento (adendo 3)
            raise Aborta(f"{lbl}: run JSON sem .npz e sem registro sem_validos no status, com o lote encerrado")
        if not (js and zs):
            return False          # sidecar ausente NAO torna incompleta: e violacao da conferencia (aborta, item 7)
        try:
            with open(js[0], "r", encoding="utf-8") as f:
                r = json.load(f)
        except Exception:
            return False
        return "modelo_v3" in r and "insumos" in r

    def indices_completos(n: int):
        """indice_sorteio -> bool (as 4 corridas do sorteio completas) no bloco n, em ordem."""
        por = {}
        for c in plano:
            if c["bloco"] == n:
                por.setdefault(c["indice_sorteio"], []).append(c)
        return {i: all(completa(c) for c in por[i]) for i in sorted(por)}

    cache = {}
    conferidas = {}

    def conferir_parcial_sem_validos(c, dj, js_nome, zs, sd, d) -> dict:
        """Adendo 3: corrida sem_validos com run JSON PARCIAL (treinador abortou antes do treino)."""
        lbl = c["run_label"]
        falhas = []

        def ex(nome, ok, det=""):
            if not ok:
                falhas.append(f"{nome}{(' (' + str(det) + ')') if det else ''}")

        cl = f"{c['cidade']}_{c['quadrante']}"
        ref = ref21.get((cl, c["split_seed"]))
        # (a) 2.1
        ex("A3a_2_1_validos==0", ref is not None and ref["validos"] == 0, ref and ref["validos"])
        # (b) status + log
        e = status_por_lbl.get(lbl) or {}
        rc_ = e.get("rc")
        ex("A3b_status_rc!=0", isinstance(rc_, int) and not isinstance(rc_, bool) and rc_ != 0, rc_)
        ex("A3b_status_sem_validos==true", e.get("sem_validos") is True)
        logp = e.get("log")
        txt = None
        if isinstance(logp, str) and Path(logp).is_file():
            txt = Path(logp).read_text(encoding="utf-8", errors="replace")
        ex("A3b_log_da_corrida_lido", txt is not None, logp)
        ex("A3b_log_contem_particao_degenerada", txt is not None and TRECHO_LOG_DEGENERADA in txt)
        # (c) run JSON parcial
        for k in ("selecao", "insumos", "modelo_v3"):
            ex(f"A3c_parcial_sem_{k}", k not in dj)
        ex("A3c_sem_npz_nem_sidecar", not zs and not sd, (len(zs), len(sd)))
        cfg, geo, spl = dj.get("config") or {}, dj.get("geometria") or {}, dj.get("split") or {}
        ex("A3c_run_label", dj.get("run_label") == lbl and cfg.get("run_label") == lbl, dj.get("run_label"))
        for k, v in (("config.grid_km", cfg.get("grid_km")), ("geometria.grid_km_usado", geo.get("grid_km_usado")), ("split.grid_km", spl.get("grid_km"))):
            ex(f"A3c_g=10[{k}]", _igual(v, GRID_KM), v)
        for k, v in (("config.buffer_km", cfg.get("buffer_km")), ("geometria.buffer_km_usado", geo.get("buffer_km_usado")), ("split.buffer_km", spl.get("buffer_km"))):
            ex(f"A3c_b=2[{k}]", _igual(v, BUFFER_KM), v)
        ex("A3c_seed_x_plano", dj.get("seed") == c["seed_treino"] and cfg.get("seed") == c["seed_treino"], (dj.get("seed"), cfg.get("seed")))
        ex("A3c_split_seed_x_plano", dj.get("split_seed") == c["split_seed"] and cfg.get("split_seed") == c["split_seed"]
           and spl.get("split_seed") == c["split_seed"], (dj.get("split_seed"), cfg.get("split_seed"), spl.get("split_seed")))
        esp_nome = f"transfer_dataset_{c['cidade']}_v19_{c['quadrante']}_enriched_cftudo.pt"
        dsb = dj.get("dataset") or {}
        ex("A3c_dataset_nome", Path(dsb.get("rf_data_file") or "").name == esp_nome and Path(cfg.get("rf_data_file") or "").name == esp_nome,
           (dsb.get("rf_data_file"), cfg.get("rf_data_file")))
        ex("A3c_dataset_bytes_x_manifest_v4", dsb.get("rf_data_bytes") is not None and tam_man.get(esp_nome) == dsb.get("rf_data_bytes"),
           (dsb.get("rf_data_bytes"), tam_man.get(esp_nome)))
        pt = (dj.get("particoes") or {}).get("test") or {}
        gd = dj.get("guarda_particao_degenerada") or {}
        ex("A3c_particoes_test_n_pl_alvo_valido==0", pt.get("n_pl_alvo_valido") == 0, pt.get("n_pl_alvo_valido"))
        ex("A3c_guarda_n_pl_alvo_valido_test==0", (gd.get("n_pl_alvo_valido") or {}).get("test") == 0, (gd.get("n_pl_alvo_valido") or {}).get("test"))
        ex("A3c_guarda_test_sem_aresta_antena", "test" in (gd.get("particoes_sem_aresta_antena") or []))
        if falhas:
            raise Aborta(f"{lbl}: sem_validos (adendo 3): " + "; ".join(falhas))
        ntest = (spl.get("n_nos_apos_buffer") or {}).get("test")
        return {"run_label": lbl, "passou": True, "com_npz": False, "parcial_sem_validos_adendo3": True,
                "dataset_bytes": dsb.get("rf_data_bytes"), "n_test_parcial_igual_2_1": bool(ref is not None and ntest == ref["n_test"])}

    def carregar(c) -> dict:
        lbl = c["run_label"]
        if lbl in cache:
            return cache[lbl]
        d, js, zs, sd = artefatos(lbl)
        if lbl in sem_val and not zs:
            if not js:
                raise Aborta(f"{lbl}: registrada sem_validos so no status; run JSON ausente (adendo 3)")
            with open(js[0], "r", encoding="utf-8") as f:
                dj = json.load(f)
            conferidas[lbl] = conferir_parcial_sem_validos(c, dj, js[0], zs, sd, d)
            r = {"run_label": lbl, "sem_validos": True, "sem_predicao": True, "mae": None}
        else:
            with open(js[0], "r", encoding="utf-8") as f:
                dj = json.load(f)
            side = _ler_sidecar(sd)
            z = np.load(zs[0])
            idx = z["idx_global"]
            m = mae_pop(z)
            conferidas[lbl] = conferir_corrida(c, dj, idx, side, man, ref21, m)
            if lbl in sem_val and m["n_validos"] != 0:
                raise Aborta(f"{lbl}: status registra sem_validos, mas o .npz tem {m['n_validos']} nos validos")
            ins = dj.get("insumos") or {}
            rf = Path(ins.get("rf_data_file") or "").name
            geo = dj.get("split") or {}
            ref = ref21.get((f"{c['cidade']}_{c['quadrante']}", c["split_seed"]))
            cfg = dj.get("config") or {}
            r = {
                "run_label": lbl, "sem_validos": m["n_validos"] == 0, "sem_predicao": False, "npz": str(zs[0]), "mae": m,
                "grid_km": cfg.get("grid_km"), "buffer_km": cfg.get("buffer_km"), "split_seed_run": dj.get("split_seed"),
                "seed_treino_run": dj.get("seed"),
                "proveniencia_ok": bool(ins.get("sha256_rf_data") and man.get(rf) == ins.get("sha256_rf_data")),
                "dist_min_entre_particoes_km": {k: v.get("dist_min_km") for k, v in ((geo.get("verificacao") or {}).get("pares") or {}).items()},
                "intersecoes": (geo.get("verificacao") or {}).get("intersecoes"),
                "n_test_retido_run": (geo.get("n_nos_apos_buffer") or {}).get("test"),
                "baselines_test": {b: ((dj.get("baselines_analiticos") or {}).get("test") or {}).get(f"baseline_{b}_mae") for b in BASELINES},
                "melhor_epoca": (dj.get("selecao") or {}).get("melhor_epoca"),
                "bate_com_2_1": {"n_test_igual": True, "n_validos_igual": True, "n_todos_igual": True,
                                 "nota": "divergencia ABORTA (C1); campos mantidos por compatibilidade"},
                "mae_constante_validos_2_1": ref["mae_constante_validos"],
            }
        cache[lbl] = r
        return r

    def runs(bloco_n: int, cel: str, tipo: str, seed: int, n_sort: int, apenas_indices=None):
        sel = [c for c in plano if c["bloco"] == bloco_n and c["tipo"] == tipo
               and f"{c['cidade']}_{c['quadrante']}" == cel and c["seed_treino"] == seed]
        sel = sorted(sel, key=lambda c: c["indice_sorteio"])[:n_sort]
        if apenas_indices is not None:
            sel = [c for c in sel if c["indice_sorteio"] in apenas_indices]
        return sel

    def por_celula_modelo(bloco_n, cel, tipo, seed, n_sort, apenas_indices=None):
        return {c["split_seed"]: carregar(c) for c in runs(bloco_n, cel, tipo, seed, n_sort, apenas_indices)}

    # ---- bloco simples (1 e 3): metricas por celula e modelo + paridade (inclui sem validos)
    def metricas_bloco_simples(n: int, indices: list) -> dict:
        cels = {}
        for cel in CELULAS_POR_BLOCO[n]:
            g = por_celula_modelo(n, cel, "gnn", 42, 20, set(indices))
            m = por_celula_modelo(n, cel, "mlp", 42, 20, set(indices))
            ordem = [c["split_seed"] for c in sorted(runs(n, cel, "gnn", 42, 20, set(indices)), key=lambda c: c["indice_sorteio"])]
            ss = [s for s in ordem if s in g and s in m]
            sv = [s for s in ss if g[s]["sem_validos"] or m[s]["sem_validos"]]
            bloco = {"sorteios": ss, "n_sorteios_plano_usados": len(ss),
                     "sem_validos": sv, "n_sorteios_sem_validos": len(sv), "por_modelo": {}}
            for nome, store in (("gnn", g), ("mlp", m)):
                ok = [s for s in ss if not store[s]["sem_validos"]]
                v = [store[s]["mae"]["mae_rssi_validos_db"] for s in ok]
                d = dp(v)
                ac = avaliar_condicoes_modelo(d, RUIDO_REPETICAO_DB)
                const = [store[s]["mae_constante_validos_2_1"] for s in ok]
                bloco["por_modelo"][nome] = {
                    "n_sorteios_com_validos": len(ok),
                    "mae_validos_por_sorteio": {str(s): store[s]["mae"]["mae_rssi_validos_db"] for s in ok},
                    "dp_entre_sorteios_validos_db": d,
                    "razao_dp_sobre_0_132": razao(d, RUIDO_REPETICAO_DB),
                    "dp_ge_3x_0_132": ac["dp_ge_3x_comparador"],
                    "dp_lt_2x_0_132": ac["dp_lt_2x_comparador"],
                    "correlacao_com_constante_validos": {"pearson": pearson(v, const), "spearman": spearman(v, const), "n": len(ok)},
                    "dp_entre_sorteios_demais_populacoes_db": {p: dp([store[s]["mae"][p] for s in ok]) for p in POPS},
                    "inversao_vs_baselines_validos": {b: sum(1 for s in ok if store[s]["baselines_test"][b] is not None and
                                                              store[s]["mae"]["mae_rssi_validos_db"] < store[s]["baselines_test"][b])
                                                      for b in BASELINES},
                    "proveniencia_toda_ok": all(store[s]["proveniencia_ok"] for s in ok),
                    "geometria_declarada": sorted({(store[s]["grid_km"], store[s]["buffer_km"]) for s in ok}),
                    "dist_min_entre_particoes_km_min": min([x for s in ok for x in store[s]["dist_min_entre_particoes_km"].values() if x is not None], default=None),
                    "bate_com_2_1_todos": all((store[s]["bate_com_2_1"] or {}).get(k) for s in ok for k in ("n_test_igual", "n_validos_igual", "n_todos_igual")),
                    "melhores_epocas": [store[s]["melhor_epoca"] for s in ok],
                }
            # paridade nos sentinelas: TODOS os sorteios com predicao nas duas corridas (inclui os sem no valido) -- adendo 6
            com_pred = [s for s in ss if not g[s].get("sem_predicao") and not m[s].get("sem_predicao")]
            sem_pred = [s for s in ss if s not in com_pred]
            dif_sent = {str(s): (abs(g[s]["mae"]["mae_rssi_sentinela_db"] - m[s]["mae"]["mae_rssi_sentinela_db"])
                                 if g[s]["mae"]["mae_rssi_sentinela_db"] is not None and m[s]["mae"]["mae_rssi_sentinela_db"] is not None else None)
                        for s in com_pred}
            md = med(list(dif_sent.values()))
            bloco["paridade_sentinela"] = {
                "abs_gnn_menos_mlp_por_sorteio": dif_sent, "mediana_db": md,
                "n_sorteios_na_mediana": sum(1 for v in dif_sent.values() if v is not None),
                "n_sorteios_sem_validos_incluidos_na_mediana": sum(1 for s in com_pred if s in sv and dif_sent[str(s)] is not None),
                "sorteios_sem_predicao_fora_da_mediana": sem_pred,
                "mediana_le_0_117": (None if md is None else bool(md <= RUIDO_SENTINELA_DB)),
                "mesma_particao_gnn_mlp": all(g[s]["mae"]["idx_sha256"] == m[s]["mae"]["idx_sha256"] for s in com_pred)}
            cels[cel] = bloco
        return cels

    def sinais_simples(cels: dict) -> dict:
        """Condicao 1/2x contra o piso 0,132 (comparador das celulas Q3; para Q1 e so provisorio)."""
        out = {}
        for cel, b in cels.items():
            pm = {t: avaliar_condicoes_modelo(b["por_modelo"][t]["dp_entre_sorteios_validos_db"], RUIDO_REPETICAO_DB) for t in ("gnn", "mlp")}
            s = sinalizar_celula(pm)
            s["paridade_le_0_117"] = bool(b["paridade_sentinela"]["mediana_le_0_117"])
            s["por_modelo"] = pm
            out[cel] = s
        return out

    # ---- Q1: bloco 1 + bloco 2 (comparador pooled e condicao 2)
    def avaliar_q1() -> dict:
        i1 = indices_completos(1)
        i2 = indices_completos(2)
        if not all(i1.values()) or not all(i2.values()):
            faltam = [c["run_label"] for c in plano if c["bloco"] in (1, 2) and not completa(c)]
            raise Incompleto(f"{len(faltam)} corridas dos blocos 1/2 sem resultado")
        cels1 = metricas_bloco_simples(1, sorted(i1))
        cel_out, flags = {}, {}
        for cel in CELULAS_POR_BLOCO[2]:
            ent, pm = {"gnn": {}, "mlp": {}}, {}
            for tipo in ("gnn", "mlp"):
                por_semente = {}
                for sd in (42, 43, 44):
                    cs = runs(1 if sd == 42 else 2, cel, tipo, sd, 5)
                    por_semente[sd] = {c["split_seed"]: carregar(c) for c in cs}
                sorteios = [c["split_seed"] for c in runs(2, cel, tipo, 43, 5)]
                lin = [s for s in sorteios if all(s in por_semente[sd] and not por_semente[sd][s]["sem_validos"] for sd in (42, 43, 44))]
                M = np.array([[por_semente[sd][s]["mae"]["mae_rssi_validos_db"] for sd in (42, 43, 44)] for s in lin], float)
                dec = decomposicao_um_fator(M) if len(lin) >= 2 else None
                dp_sem = None if dec is None else dec["dp_entre_sementes_pooled_db"]
                dp20 = cels1[cel]["por_modelo"][tipo]["dp_entre_sorteios_validos_db"]
                comparador = max(RUIDO_REPETICAO_DB, dp_sem) if dp_sem is not None else None
                pm[tipo] = avaliar_condicoes_modelo(dp20, comparador)
                ent[tipo] = {
                    "sorteios_usados": lin, "n_sorteios_usados": len(lin),
                    "sorteios_do_bloco_2_sem_validos_excluidos": [s for s in sorteios if s not in lin],
                    "matriz_mae_validos_sorteio_x_semente_42_43_44": M.tolist(),
                    "decomposicao_um_fator_sementes_aninhadas_no_sorteio": dec,
                    "dp_entre_sementes_pooled_db": dp_sem,
                    "dp_entre_sementes_sobre_0_132": razao(dp_sem, RUIDO_REPETICAO_DB),
                    "comparador_db_max_0_132_dp_sementes": comparador,
                    "dp_entre_sorteios_20_semente42_db": dp20,
                    "razao_dp_sorteios20_sobre_comparador": pm[tipo]["razao_dp_sobre_comparador"],
                    "razao_dp_sorteios20_sobre_dp_sementes": razao(dp20, dp_sem),
                    "leitura_aritmetica": pm[tipo],
                    "condicao2_sigma2_sorteio_ge_sigma2_semente": (None if dec is None else dec["condicao2_sigma2_sorteio_ge_sigma2_semente"])}
            s = sinalizar_celula(pm)
            c2s = [ent[t]["condicao2_sigma2_sorteio_ge_sigma2_semente"] for t in ("gnn", "mlp")]
            s["condicao2_gnn_e_mlp"] = None if any(v is None for v in c2s) else bool(all(c2s))
            if s["condicao2_gnn_e_mlp"] is None:
                s["indeterminada_dp_nao_calculavel"] = True
            s["paridade_le_0_117"] = bool(cels1[cel]["paridade_sentinela"]["mediana_le_0_117"])
            # sensibilidade declarada: comparador pooled ENTRE as 2 celulas Q1 (leitura alternativa do adendo 2)
            cel_out[cel] = ent
            flags[cel] = s
        # leitura alternativa do comparador (dp pooled entre sementes juntando as duas celulas, por modelo) -- so informativo
        alt = {}
        for cel in CELULAS_POR_BLOCO[2]:
            pm_alt = {}
            for tipo in ("gnn", "mlp"):
                qms = [cel_out[c2][tipo]["decomposicao_um_fator_sementes_aninhadas_no_sorteio"] for c2 in CELULAS_POR_BLOCO[2]]
                if any(q is None for q in qms):
                    pm_alt[tipo] = avaliar_condicoes_modelo(cel_out[cel][tipo]["dp_entre_sorteios_20_semente42_db"], None)
                    continue
                tot = sum(q["n_sorteios"] for q in qms)
                dp_pool = float(np.sqrt(sum(q["qm_dentro"] * q["n_sorteios"] for q in qms) / tot))
                pm_alt[tipo] = avaliar_condicoes_modelo(cel_out[cel][tipo]["dp_entre_sorteios_20_semente42_db"], max(RUIDO_REPETICAO_DB, dp_pool))
            alt[cel] = sinalizar_celula(pm_alt)
            alt[cel]["por_modelo"] = pm_alt
        return {"cels1": cels1, "bloco2": cel_out, "flags": flags, "alternativa_comparador_pooled_entre_celulas": alt}

    def montar_meta(n: int) -> dict:
        return {"artefato": f"agregado_G1_v6_bloco{n}", "versao_agregador": 6,
                "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
                "script_original_copiado": {"caminho": "scripts/v3_G1_agregar_v5.py", "sha256": SHA_V5, "avo_v4_sha256": SHA_V4, "avo_v3_sha256": SHA_V3, "avo_v2_sha256": SHA_V2, "avo_v1_sha256": SHA_ORIGINAL_V1},
                "comando": [sys.executable] + list(argv_efetivo), "comando_texto": " ".join([sys.executable] + list(argv_efetivo)),
                "data": datetime.now().astimezone().isoformat(timespec="seconds"),
                "criterio_sha256": sha_crit, "adendo_sha256": sha_adendo, "adendo2_sha256": sha_adendo2, "adendo3_sha256": sha_adendo3, "script_v3_sha256_copiado": SHA_V3, "script_v2_sha256_avo": SHA_V2, "plano_sha256": sha_plano,
                "status_lote_veredito": status.get("veredito_lote"),
                "modo_teste": bool(modo_teste), "raiz": str(P.RAIZ),
                "limiares_do_criterio_db": {"ruido_repeticao_gnn_0_132": RUIDO_REPETICAO_DB, "ruido_sentinela_0_117": RUIDO_SENTINELA_DB,
                                            "fator_condicao1": FATOR_CONDICAO1, "fator_delimita": FATOR_DELIMITA, "k_min_bloco3": K_MIN_BLOCO3},
                "nota_0_132": "0,132 dB e uma diferenca absoluta entre duas repeticoes, nao um dp: o limiar e conservador (adendo 2)",
                "conferencias_item7": {"scripts_do_lote": prov_scripts,
                                       "criterio_x_adendo": "sha256 do criterio igual ao citado no adendo",
                                       "plano_x_lista_2_1_e_g_b": "ok"}}

    def fechar_conferencias(out: dict, labels: list) -> None:
        usados = {lbl: conferidas[lbl] for lbl in labels if lbl in conferidas}
        carregadas = [lbl for lbl in labels if lbl in cache]
        todas = bool(carregadas) and all(conferidas.get(lbl, {}).get("passou") is True for lbl in carregadas)
        out["conferencias_item7"]["corridas_conferidas"] = {
            "n": len(usados),
            "itens": ["run_label", "dataset_nome/sha256/flag_manifest_v4", "g=10 e b=2 no run JSON (config, geometria, split)",
                      "g=10 e b=2 no sidecar shim_g10 (pedido, efetivo, flags, n_nos)", "seed e split_seed x plano_G1.json",
                      "hash dos indices do .npz x particoes.test.idx_sha256_global", "config x A4 (fora de grid_km, seed, split_seed, rotulos e caminhos)",
                      "script/decoder/wrapper sha x A4",
                      "n_test/n_validos/n_todos x 2.1 (C1)", "nomes unicos run_/predicoes_/shim_g10_<label> (C3)",
                      "amarracao .npz x run JSON (C3)", "sem_validos sem .npz: run JSON + 2.1 validos==0 (C2)"],
            "n_corridas_carregadas": len(carregadas),
            "amarracao_por_corrida": {lbl: v.get("amarracao") for lbl, v in usados.items() if v.get("amarracao")},
            "amarracao_dif_rel_maxima": {k: max([v["amarracao"][k] for v in usados.values() if v.get("amarracao") and v["amarracao"].get(k) is not None], default=None) for k in ("dif_rel_mae", "dif_rel_rmse")},
            "todas_passaram": todas,
            "n_com_npz": sum(1 for v in usados.values() if v.get("com_npz")),
            "n_sem_validos_sem_npz_conferidas_C2": sum(1 for v in usados.values() if not v.get("com_npz")),
            "sem_validos_parcial_adendo3": {lbl: {"dataset_bytes": v.get("dataset_bytes"), "n_test_parcial_igual_2_1": v.get("n_test_parcial_igual_2_1")} for lbl, v in usados.items() if v.get("parcial_sem_validos_adendo3")},
            "amarracao_npz": "MAE e RMSE de RSSI (todos os nos) do .npz x selecao.test_no_melhor_ckpt (tol. 10 % / 3 %), n_nos, n_pl_alvo_valido; nenhum artefato do lote grava sha do .npz",
            "sem_artefatos_sem_validos_registrados": [lbl for lbl in labels if cache.get(lbl, {}).get("sem_predicao")]}

    # ---- execucao por bloco
    saidas, recusas = {}, {}
    alvo = [args.bloco] if args.bloco else [1, 2, 3]

    if args.so_conferir:
        feitas = 0
        for n in alvo:
            for c in [c for c in plano if c["bloco"] == n]:
                if completa(c):
                    carregar(c)          # aplica TODAS as conferencias (aborta se violada); nada e impresso alem do rotulo
                    feitas += 1
                    print(f"conferencia ok: {c['run_label']}")
        print(f"--so-conferir: {feitas} corridas completas conferidas (nada gravado; MAE/RMSE recalculados so para a amarracao e nao impressos)")
        return 0

    for n in alvo:
        try:
            if n == 1:
                idx = indices_completos(1)
                if not all(idx.values()):
                    raise Incompleto(f"{sum(1 for c in plano if c['bloco'] == 1 and not completa(c))} corridas sem resultado")
                cels = metricas_bloco_simples(1, sorted(idx))
                out = montar_meta(1)
                out["celulas"] = cels
                out["leitura_aritmetica_contra_piso_0_132"] = sinais_simples(cels)
                out["nota"] = ("condicao 1 das celulas Q1 usa o comparador max(0,132; dp entre sementes) do bloco 2: ver "
                               "agregado_G1_v6_bloco2.json; aqui so a comparacao com o piso 0,132")
                out["contagem_sorteios_sem_validos_por_celula"] = {c: b["n_sorteios_sem_validos"] for c, b in cels.items()}
                fechar_conferencias(out, [c["run_label"] for c in plano if c["bloco"] == 1])
            elif n == 2:
                q1 = avaliar_q1()
                out = montar_meta(2)
                out["celulas"] = q1["bloco2"]
                celulas_flags = {c: q1["flags"][c] for c in q1["flags"]}
                out["leitura_aritmetica_por_celula"] = celulas_flags
                out["alternativa_comparador_pooled_entre_celulas"] = {
                    "nota": "leitura alternativa (nao a primaria) do adendo 2: dp pooled entre sementes juntando as duas celulas Q1, por modelo",
                    "celulas": q1["alternativa_comparador_pooled_entre_celulas"]}
                out["contagem_sorteios_sem_validos_por_celula"] = {c: b["n_sorteios_sem_validos"] for c, b in q1["cels1"].items()}
                out["paridade_sentinela_celulas_q1"] = {c: q1["cels1"][c]["paridade_sentinela"] for c in q1["cels1"]}
                out["contagem_mecanica_ramos_so_celulas_q1_2_de_2"] = contar_ramos(celulas_flags, list(celulas_flags))
                fechar_conferencias(out, [c["run_label"] for c in plano if c["bloco"] in (1, 2)])
            else:
                idx3 = indices_completos(3)
                completo = all(idx3.values())
                if completo:
                    usados = sorted(idx3)
                    modo = "completo"
                else:
                    if not args.bloco3_truncado:
                        raise Incompleto(f"{sum(1 for c in plano if c['bloco'] == 3 and not completa(c))} corridas sem resultado "
                                         "(use --bloco3-truncado so se o lote terminou e o bloco segue incompleto)")
                    if status.get("veredito_lote") is None:
                        raise Incompleto("--bloco3-truncado recusado: lote_G1_status.json sem veredito_lote (lote ainda em andamento)")
                    usados = []
                    for i in sorted(idx3):
                        if not idx3[i]:
                            break
                        usados.append(i)
                    modo = "truncado_prefixo_contiguo"
                if not usados:
                    raise Incompleto("bloco 3: nenhum sorteio com as 4 corridas completas")
                cels3 = metricas_bloco_simples(3, usados)
                if modo != "completo":
                    ruins = {c: b["por_modelo"]["gnn"]["n_sorteios_com_validos"] for c, b in cels3.items()
                             if min(b["por_modelo"]["gnn"]["n_sorteios_com_validos"], b["por_modelo"]["mlp"]["n_sorteios_com_validos"]) < K_MIN_BLOCO3}
                    if ruins:
                        raise Incompleto(f"bloco 3 truncado (k={len(usados)}) com menos de {K_MIN_BLOCO3} sorteios com nos validos "
                                         f"em {sorted(ruins)}: bloco 3 nao entra (regra passa a 2 de 2 nas celulas Q1)")
                out = montar_meta(3)
                out["bloco3_modo"] = modo
                out["bloco3_k_sorteios_usados"] = len(usados)
                out["bloco3_indices_sorteio_usados"] = usados
                out["bloco3_bloqueados_depois_do_prefixo"] = [i for i in sorted(idx3) if i not in usados]
                out["comparador_celulas_q3"] = "0,132 dB (ancora medida a 5 km; adendo 2)"
                out["celulas"] = cels3
                sin3 = sinais_simples(cels3)
                out["leitura_aritmetica_por_celula"] = sin3
                out["contagem_sorteios_sem_validos_por_celula"] = {c: b["n_sorteios_sem_validos"] for c, b in cels3.items()}
                labels = [c["run_label"] for c in plano if c["bloco"] == 3 and c["indice_sorteio"] in set(usados)]
                try:
                    q1 = avaliar_q1()
                    todas = {**q1["flags"], **sin3}
                    out["contagem_mecanica_ramos_4_celulas"] = contar_ramos(todas, list(q1["flags"]))
                    labels += [c["run_label"] for c in plano if c["bloco"] in (1, 2)]
                except Incompleto as e:
                    out["contagem_mecanica_ramos_4_celulas"] = {"disponivel": False, "motivo": f"celulas Q1 indisponiveis: {e}"}
                fechar_conferencias(out, labels)
            saidas[n] = out
        except Incompleto as e:
            recusas[n] = str(e)

    for n, motivo in recusas.items():
        print(f"bloco {n}: RECUSADO ({motivo}); nao grava resultado parcial")
    for n, out in saidas.items():
        path = P.G1 / f"agregado_G1_v6_bloco{n}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        tmp.replace(path)
        print(f"bloco {n}: gravado {path}")
    return 2 if recusas else 0


if __name__ == "__main__":
    sys.exit(main())
