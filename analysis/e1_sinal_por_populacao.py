#!/usr/bin/env python3
"""
E1 -- sinal e magnitude de GNN-MLP por populacao (todos/cobertos), IQR de
predicao e vitorias sobre o preditor constante, a partir dos 160 run JSON
originais (80 pares mlpcf x c0c1cf, g10b2) em EVIDENCIA_RESUBMISSAO.

Frente ia-ablacao, rodada R2 do fio gnn_rf_artigo2_mathematics_r2, tarefa E1.
CPU apenas. So LE artefatos existentes; nao treina nada.

Uso:
  python e1_sinal_por_populacao.py --out <arquivo.json>
  python e1_sinal_por_populacao.py --hash     # imprime sha256 deste script e sai
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

E_ROOT = (
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO"
)
TREINOS_DIR = os.path.join(E_ROOT, "treinos")
DADOS_TREINOS_C1 = os.path.join(E_ROOT, "dados", "treinos_c1")
BASELINES_DIR = os.path.join(E_ROOT, "dados", "baselines_v2")

CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
SEEDS = [42, 43, 44, 45, 46]
QS = [1, 2, 3, 4]
BRACOS = ["mlpcf", "c0c1cf"]

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260924


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_self():
    return sha256_file(os.path.abspath(__file__))


def canonical_run_path(braco, cidade, seed, q):
    dirname = f"{braco}_{cidade}_s{seed}_Q{q}_g10b2"
    path = os.path.join(TREINOS_DIR, dirname, f"run_{dirname}.json")
    return path if os.path.isfile(path) else None


def duplicate_run_path(braco, cidade, seed, q):
    name = f"run_{braco}_{cidade}_s{seed}_Q{q}_g10b2.json"
    path = os.path.join(DADOS_TREINOS_C1, name)
    return path if os.path.isfile(path) else None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def coletar_runs(fontes_lidas):
    """Descobre os 160 runs canonicos, confere duplicatas por sha256,
    extrai os campos de interesse. Exclui smoke e variantes (c0c1cfsc,
    c0c1cfinv, c0c1, c0c1v2) por construcao (so monta nomes canonicos
    mlpcf_/c0c1cf_ com sufixo _g10b2, sem 'sc'/'inv'/variantes)."""
    runs = {}
    duplicatas = []
    faltantes = []
    for braco in BRACOS:
        for cidade in CIDADES:
            for seed in SEEDS:
                for q in QS:
                    key = (braco, cidade, seed, q)
                    canon = canonical_run_path(braco, cidade, seed, q)
                    if canon is None:
                        faltantes.append(
                            {"braco": braco, "cidade": cidade, "seed": seed, "Q": q,
                             "motivo": "run canonico ausente em treinos/"}
                        )
                        continue
                    sha_canon = sha256_file(canon)
                    fontes_lidas[canon] = sha_canon
                    dup = duplicate_run_path(braco, cidade, seed, q)
                    dup_info = None
                    if dup is not None:
                        sha_dup = sha256_file(dup)
                        fontes_lidas[dup] = sha_dup
                        bate = sha_canon == sha_dup
                        dup_info = {"caminho": dup, "sha256": sha_dup, "bate_com_canonico": bate}
                        duplicatas.append(
                            {"braco": braco, "cidade": cidade, "seed": seed, "Q": q,
                             "canonico": canon, "canonico_sha256": sha_canon,
                             "duplicata": dup_info}
                        )
                    data = load_json(canon)
                    sel = get_nested(data, "selecao", "test_no_melhor_ckpt") or {}
                    diag = sel.get("diag", {}) if isinstance(sel, dict) else {}
                    idx_sha_test = get_nested(data, "particoes", "test", "idx_sha256_global")
                    rf_sha = get_nested(data, "dataset", "rf_data_sha256")
                    run_label = data.get("run_label")
                    runs[key] = {
                        "run_label": run_label,
                        "caminho_canonico": canon,
                        "sha256_canonico": sha_canon,
                        "duplicata": dup_info,
                        "idx_sha256_global_test": idx_sha_test,
                        "rf_data_sha256": rf_sha,
                        "mae_rssi_db_todos": sel.get("mae_rssi_db"),
                        "mae_pl_db_cobertos": sel.get("mae_pl_db"),
                        "n_nos_avaliados": sel.get("n_nos_avaliados"),
                        "n_pl_alvo_valido": sel.get("n_pl_alvo_valido"),
                        "rssi_pred_p25": diag.get("rssi_pred_p25"),
                        "rssi_pred_p75": diag.get("rssi_pred_p75"),
                        "pl_pred_p25": diag.get("path_loss_pred_p25"),  # ausente nas runs conhecidas
                        "pl_pred_p75": diag.get("path_loss_pred_p75"),  # ausente nas runs conhecidas
                    }
    return runs, duplicatas, faltantes


def parear(runs):
    pares = {}
    invalidos = []
    for cidade in CIDADES:
        for seed in SEEDS:
            for q in QS:
                km = ("mlpcf", cidade, seed, q)
                kc = ("c0c1cf", cidade, seed, q)
                if km not in runs or kc not in runs:
                    invalidos.append({"cidade": cidade, "seed": seed, "Q": q,
                                       "motivo": "run mlpcf ou c0c1cf ausente",
                                       "mlpcf_presente": km in runs, "c0c1cf_presente": kc in runs})
                    continue
                rm, rc = runs[km], runs[kc]
                idx_ok = (rm["idx_sha256_global_test"] is not None and
                          rm["idx_sha256_global_test"] == rc["idx_sha256_global_test"])
                data_ok = (rm["rf_data_sha256"] is not None and
                           rm["rf_data_sha256"] == rc["rf_data_sha256"])
                if idx_ok and data_ok:
                    pares[(cidade, seed, q)] = {"mlpcf": rm, "c0c1cf": rc}
                else:
                    invalidos.append({
                        "cidade": cidade, "seed": seed, "Q": q,
                        "motivo": "idx_sha256_global_test ou rf_data_sha256 nao coincidem",
                        "idx_sha256_mlpcf": rm["idx_sha256_global_test"],
                        "idx_sha256_c0c1cf": rc["idx_sha256_global_test"],
                        "rf_data_sha256_mlpcf": rm["rf_data_sha256"],
                        "rf_data_sha256_c0c1cf": rc["rf_data_sha256"],
                    })
    return pares, invalidos


def carregar_constantes(fontes_lidas):
    """Le baselines_v2_<cidade>_Q<n>.json (16) e extrai o preditor
    constante piso (-110 dBm, D1 do fisico) na particao test, populacao
    'todos' (RSSI). A populacao 'cobertos' do constante so existe em
    RSSI no baselines_v2 (mesma metrica mae_db), NAO em PL -- nao
    comparavel com mae_pl_db (alvo diferente)."""
    consts = {}
    for cidade in CIDADES:
        for q in QS:
            path = os.path.join(BASELINES_DIR, f"baselines_v2_{cidade}_Q{q}.json")
            if not os.path.isfile(path):
                continue
            sha = sha256_file(path)
            fontes_lidas[path] = sha
            d = load_json(path)
            test = get_nested(d, "particoes", "test") or {}
            piso = test.get("preditor_constante_piso", {})
            consts[(cidade, q)] = {
                "caminho": path,
                "sha256": sha,
                "piso_todos_mae_rssi_db": get_nested(piso, "todos", "mae_db"),
                "piso_todos_n": get_nested(piso, "todos", "n"),
                "piso_cobertura_mae_rssi_db": get_nested(piso, "cobertura", "mae_db"),
                "piso_cobertura_n": get_nested(piso, "cobertura", "n"),
                "nota_cobertura": (
                    "piso_cobertura_mae_rssi_db e MAE em RSSI (dBm) sobre nos cobertos, "
                    "NAO em PL; nao comparavel com mae_pl_db do modelo (alvo diferente)."
                ),
            }
    return consts


def enumerar_multiconjuntos_4_de_4():
    """35 multiconjuntos distintos de tamanho 4 tirados (com reposicao) de
    um universo de 4 itens (cidades) -- C(4+4-1,4) = C(7,4) = 35."""
    idxs = range(4)
    combos = itertools.combinations_with_replacement(idxs, 4)
    return list(combos)


def bootstrap_cidade(deltas_por_cidade, seed=BOOTSTRAP_SEED, B=BOOTSTRAP_B):
    """Reamostra cidades com reposicao (n=4), B vezes, sobre a media por
    cidade do delta. Retorna estatisticas do bootstrap MAIS a enumeracao
    exata dos 35 multiconjuntos distintos e o teste de sinal exato."""
    cidades = list(deltas_por_cidade.keys())
    valores = [deltas_por_cidade[c] for c in cidades]
    n = len(valores)
    rng = random.Random(seed)
    boot_means = []
    if n > 0:
        for _ in range(B):
            amostra = [valores[rng.randrange(n)] for _ in range(n)]
            boot_means.append(sum(amostra) / n)
    boot_means.sort()

    def pctl(p):
        if not boot_means:
            return None
        idx = min(len(boot_means) - 1, max(0, int(round(p * (len(boot_means) - 1)))))
        return boot_means[idx]

    combos = enumerar_multiconjuntos_4_de_4() if n == 4 else []
    enum_means = []
    for combo in combos:
        m = sum(valores[i] for i in combo) / len(combo)
        enum_means.append(m)
    enum_means.sort()

    sinais = [1 if v > 0 else (-1 if v < 0 else 0) for v in valores]
    n_pos = sum(1 for s in sinais if s > 0)
    n_neg = sum(1 for s in sinais if s < 0)
    n_zero = sum(1 for s in sinais if s == 0)
    # teste de sinal exato (binomial p=0.5) considerando so os nao-empatados
    n_efetivo = n_pos + n_neg
    if n_efetivo > 0:
        k = max(n_pos, n_neg)
        p_dois_lados = 0.0
        for i in range(k, n_efetivo + 1):
            p_dois_lados += math.comb(n_efetivo, i) * (0.5 ** n_efetivo)
        p_dois_lados = min(1.0, 2 * p_dois_lados) if n_pos != n_neg else 1.0
    else:
        p_dois_lados = None

    return {
        "cidades_ordem": cidades,
        "valores_delta_por_cidade": {c: v for c, v in zip(cidades, valores)},
        "n_cidades": n,
        "bootstrap": {
            "B": B,
            "seed": seed,
            "media": sum(boot_means) / len(boot_means) if boot_means else None,
            "p2_5": pctl(0.025),
            "p50": pctl(0.5),
            "p97_5": pctl(0.975),
        },
        "enumeracao_exata_35_multiconjuntos": {
            "n_multiconjuntos": len(combos),
            "media_das_35_medias": sum(enum_means) / len(enum_means) if enum_means else None,
            "min": enum_means[0] if enum_means else None,
            "max": enum_means[-1] if enum_means else None,
            "mediana": enum_means[len(enum_means) // 2] if enum_means else None,
        },
        "teste_sinal_exato": {
            "n_positivos": n_pos,
            "n_negativos": n_neg,
            "n_empates": n_zero,
            "p_valor_bilateral_binomial_05": p_dois_lados,
            "nota": "n=4 cidades; teste de sinal exato tem poder baixo com n=4 (informativo, nao decisivo)",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--hash", action="store_true")
    args = ap.parse_args()

    if args.hash:
        print(sha256_self())
        return

    fontes_lidas = {}
    runs, duplicatas, faltantes = coletar_runs(fontes_lidas)
    pares, invalidos = parear(runs)
    consts = carregar_constantes(fontes_lidas)

    # -------- por corrida (descritivo, 80 pares) --------
    por_corrida = []
    for (cidade, seed, q), par in pares.items():
        rm, rc = par["mlpcf"], par["c0c1cf"]
        delta_todos = None
        if rm["mae_rssi_db_todos"] is not None and rc["mae_rssi_db_todos"] is not None:
            delta_todos = rc["mae_rssi_db_todos"] - rm["mae_rssi_db_todos"]
        delta_cobertos = None
        if rm["mae_pl_db_cobertos"] is not None and rc["mae_pl_db_cobertos"] is not None:
            delta_cobertos = rc["mae_pl_db_cobertos"] - rm["mae_pl_db_cobertos"]
        iqr_rssi_mlpcf = None
        iqr_rssi_c0c1cf = None
        if rm["rssi_pred_p75"] is not None and rm["rssi_pred_p25"] is not None:
            iqr_rssi_mlpcf = rm["rssi_pred_p75"] - rm["rssi_pred_p25"]
        if rc["rssi_pred_p75"] is not None and rc["rssi_pred_p25"] is not None:
            iqr_rssi_c0c1cf = rc["rssi_pred_p75"] - rc["rssi_pred_p25"]
        por_corrida.append({
            "cidade": cidade, "seed": seed, "Q": q,
            "delta_gnn_menos_mlp_mae_rssi_db_todos": delta_todos,
            "delta_gnn_menos_mlp_mae_pl_db_cobertos": delta_cobertos,
            "iqr_pred_rssi_db_mlpcf": iqr_rssi_mlpcf,
            "iqr_pred_rssi_db_c0c1cf": iqr_rssi_c0c1cf,
            "iqr_pred_pl_db_mlpcf": None,
            "iqr_pred_pl_db_c0c1cf": None,
        })

    # -------- por celula (cidade x Q), media sobre seeds --------
    por_celula = {}
    for cidade in CIDADES:
        for q in QS:
            regs = [r for r in por_corrida if r["cidade"] == cidade and r["Q"] == q]
            regs_s42 = [r for r in regs if r["seed"] == 42]
            deltas_todos = [r["delta_gnn_menos_mlp_mae_rssi_db_todos"] for r in regs if r["delta_gnn_menos_mlp_mae_rssi_db_todos"] is not None]
            deltas_cob = [r["delta_gnn_menos_mlp_mae_pl_db_cobertos"] for r in regs if r["delta_gnn_menos_mlp_mae_pl_db_cobertos"] is not None]
            d_todos_s42 = regs_s42[0]["delta_gnn_menos_mlp_mae_rssi_db_todos"] if regs_s42 else None
            d_cob_s42 = regs_s42[0]["delta_gnn_menos_mlp_mae_pl_db_cobertos"] if regs_s42 else None
            por_celula[f"{cidade}_Q{q}"] = {
                "cidade": cidade, "Q": q,
                "n_seeds_pareadas": len(regs),
                "media_5seeds_delta_todos_rssi_db": sum(deltas_todos) / len(deltas_todos) if deltas_todos else None,
                "media_5seeds_delta_cobertos_pl_db": sum(deltas_cob) / len(deltas_cob) if deltas_cob else None,
                "seed42_delta_todos_rssi_db": d_todos_s42,
                "seed42_delta_cobertos_pl_db": d_cob_s42,
                "sinal_seed42_todos": (None if d_todos_s42 is None else (1 if d_todos_s42 > 0 else (-1 if d_todos_s42 < 0 else 0))),
                "sinal_seed42_cobertos": (None if d_cob_s42 is None else (1 if d_cob_s42 > 0 else (-1 if d_cob_s42 < 0 else 0))),
            }

    # -------- criterio pre-registrado R1 (Bloco G) --------
    celulas_com_ambas_pop_s42 = [c for c in por_celula.values()
                                  if c["sinal_seed42_todos"] is not None and c["sinal_seed42_cobertos"] is not None]
    n_mudanca_sinal_s42 = sum(1 for c in celulas_com_ambas_pop_s42
                               if c["sinal_seed42_todos"] != c["sinal_seed42_cobertos"]
                               and c["sinal_seed42_todos"] != 0 and c["sinal_seed42_cobertos"] != 0)

    def ordem_cidades(pop_key):
        medias_cidade = {}
        for cidade in CIDADES:
            vals = [c[pop_key] for c in por_celula.values() if c["cidade"] == cidade and c[pop_key] is not None]
            medias_cidade[cidade] = sum(vals) / len(vals) if vals else None
        ordenado = sorted([c for c in CIDADES if medias_cidade[c] is not None], key=lambda c: medias_cidade[c])
        return ordenado, medias_cidade

    ordem_todos_s42, medias_todos_s42 = ordem_cidades("seed42_delta_todos_rssi_db")
    ordem_cob_s42, medias_cob_s42 = ordem_cidades("seed42_delta_cobertos_pl_db")
    ordem_todos_5s, medias_todos_5s = ordem_cidades("media_5seeds_delta_todos_rssi_db")
    ordem_cob_5s, medias_cob_5s = ordem_cidades("media_5seeds_delta_cobertos_pl_db")

    criterio_r1 = {
        "definicao": "forum-eng-ia R1 Bloco G: reportar se o sinal de GNN-MLP muda entre "
                      "populacoes (todos vs cobertos) em >=4 de 16 celulas (seed 42) e se a "
                      "ordem por cidade muda.",
        "seed42": {
            "n_celulas_avaliaveis": len(celulas_com_ambas_pop_s42),
            "n_celulas_com_mudanca_de_sinal": n_mudanca_sinal_s42,
            "limiar_atingido_ge_4_de_16": n_mudanca_sinal_s42 >= 4,
            "ordem_cidades_por_delta_todos": ordem_todos_s42,
            "ordem_cidades_por_delta_cobertos": ordem_cob_s42,
            "ordem_muda": ordem_todos_s42 != ordem_cob_s42,
            "medias_por_cidade_todos": medias_todos_s42,
            "medias_por_cidade_cobertos": medias_cob_s42,
        },
        "leitura_secundaria_media_5_seeds": {
            "ordem_cidades_por_delta_todos": ordem_todos_5s,
            "ordem_cidades_por_delta_cobertos": ordem_cob_5s,
            "ordem_muda": ordem_todos_5s != ordem_cob_5s,
            "medias_por_cidade_todos": medias_todos_5s,
            "medias_por_cidade_cobertos": medias_cob_5s,
            "rotulo": "leitura secundaria (nao pre-registrada com estas 5 seeds; so seed42 foi pre-registrado no Bloco G)",
        },
    }

    # -------- por cidade (unidade de inferencia, n=4) --------
    por_cidade = {}
    deltas_cidade_todos = {}
    deltas_cidade_cobertos = {}
    for cidade in CIDADES:
        regs = [r for r in por_corrida if r["cidade"] == cidade]
        deltas_todos = [r["delta_gnn_menos_mlp_mae_rssi_db_todos"] for r in regs if r["delta_gnn_menos_mlp_mae_rssi_db_todos"] is not None]
        deltas_cob = [r["delta_gnn_menos_mlp_mae_pl_db_cobertos"] for r in regs if r["delta_gnn_menos_mlp_mae_pl_db_cobertos"] is not None]
        media_todos = sum(deltas_todos) / len(deltas_todos) if deltas_todos else None
        media_cob = sum(deltas_cob) / len(deltas_cob) if deltas_cob else None
        por_cidade[cidade] = {
            "n_corridas_pareadas": len(regs),
            "media_delta_todos_rssi_db": media_todos,
            "media_delta_cobertos_pl_db": media_cob,
            "sinal_todos": (None if media_todos is None else (1 if media_todos > 0 else (-1 if media_todos < 0 else 0))),
            "sinal_cobertos": (None if media_cob is None else (1 if media_cob > 0 else (-1 if media_cob < 0 else 0))),
        }
        if media_todos is not None:
            deltas_cidade_todos[cidade] = media_todos
        if media_cob is not None:
            deltas_cidade_cobertos[cidade] = media_cob

    bootstrap_todos = bootstrap_cidade(deltas_cidade_todos)
    bootstrap_cobertos = bootstrap_cidade(deltas_cidade_cobertos)

    # -------- IQR de predicao por braco (media sobre as 80 corridas de cada braco) --------
    iqr_por_braco = {}
    for braco in BRACOS:
        vals_rssi = []
        for key, r in runs.items():
            if key[0] != braco:
                continue
            if r["rssi_pred_p75"] is not None and r["rssi_pred_p25"] is not None:
                vals_rssi.append(r["rssi_pred_p75"] - r["rssi_pred_p25"])
        iqr_por_braco[braco] = {
            "n_corridas": len(vals_rssi),
            "iqr_pred_rssi_db_media": sum(vals_rssi) / len(vals_rssi) if vals_rssi else None,
            "iqr_pred_rssi_db_min": min(vals_rssi) if vals_rssi else None,
            "iqr_pred_rssi_db_max": max(vals_rssi) if vals_rssi else None,
            "iqr_pred_pl_db": None,
            "iqr_pred_pl_nota": "diag do run JSON nao grava path_loss_pred_p25/p75 (so path_loss_pred_std); "
                                 "equivalente PL do IQR de predicao NAO existe nas 160 corridas -- nao_verificado.",
        }

    # -------- vitorias sobre o preditor constante --------
    vitorias = {"todos_rssi": defaultdict(lambda: {"vitorias": 0, "derrotas": 0, "n": 0}),
                "cobertos_pl": {"nao_comparavel": True,
                                 "motivo": "constante em baselines_v2 para 'cobertura' e MAE em RSSI (dBm), "
                                           "nao em PL; mae_pl_db do modelo e no alvo PL -- alvo diferente, "
                                           "comparacao nao feita para nao forcar equivalencia invalida."}}
    detalhe_vitorias = []
    for (cidade, seed, q), par in pares.items():
        const = consts.get((cidade, q))
        if const is None or const["piso_todos_mae_rssi_db"] is None:
            continue
        for braco_nome, reg in (("mlpcf", par["mlpcf"]), ("c0c1cf", par["c0c1cf"])):
            mae = reg["mae_rssi_db_todos"]
            if mae is None:
                continue
            venceu = mae < const["piso_todos_mae_rssi_db"]
            acc = vitorias["todos_rssi"][braco_nome]
            acc["n"] += 1
            if venceu:
                acc["vitorias"] += 1
            else:
                acc["derrotas"] += 1
            detalhe_vitorias.append({
                "braco": braco_nome, "cidade": cidade, "seed": seed, "Q": q,
                "mae_rssi_db_todos_modelo": mae,
                "constante_piso_mae_rssi_db_todos": const["piso_todos_mae_rssi_db"],
                "modelo_venceu_constante": venceu,
            })
    vitorias["todos_rssi"] = {k: dict(v) for k, v in vitorias["todos_rssi"].items()}
    vitorias["todos_rssi_por_particao_16"] = {}
    for cidade in CIDADES:
        for q in QS:
            const = consts.get((cidade, q))
            if const is None:
                continue
            regs = [d for d in detalhe_vitorias if d["cidade"] == cidade and d["Q"] == q and d["seed"] == 42]
            vitorias["todos_rssi_por_particao_16"][f"{cidade}_Q{q}"] = {
                r["braco"]: r["modelo_venceu_constante"] for r in regs
            }

    # -------- publicabilidade --------
    publicabilidade = {
        "sinal_delta_gnn_menos_mlp_por_cidade": {
            "classe": "publicavel (sinal)",
            "motivo": "+1/-1/0 por cidade nao revela a escala absoluta do MAE de nenhum braco.",
        },
        "magnitude_delta_gnn_menos_mlp_db": {
            "classe": "atencao -- publicavel so sem o denominador por celula",
            "motivo": "a diferenca em dB por celula, se publicada ao lado do MAE de qualquer um dos dois "
                      "bracos (ja presente na IEEE), permite reconstruir por soma/subtracao o MAE do outro "
                      "braco na mesma celula; publicar so a diferenca agregada (por cidade ou geral), sem "
                      "abrir a celula individual ao lado de qualquer MAE absoluto de origem IEEE.",
        },
        "ordem_por_cidade_ranking": {
            "classe": "publicavel (posto)",
            "motivo": "ranking ordinal entre 4 cidades nao devolve magnitude.",
        },
        "teste_de_sinal_exato_e_enumeracao_35": {
            "classe": "publicavel (contagem/teste de sinal)",
            "motivo": "p-valor e contagens de sinal sao estatisticas de posto/contagem, nao MAE.",
        },
        "iqr_predicao_rssi_db": {
            "classe": "publicavel (grandeza descritiva da predicao, nao e erro contra alvo)",
            "motivo": "p75-p25 da distribuicao PREDITA nao compara contra o alvo real; nao e MAE nem "
                      "proxy direto dele.",
        },
        "iqr_predicao_pl_db": {
            "classe": "nao_verificado -- grandeza nao existe nas 160 corridas",
            "motivo": "diag nao grava path_loss_pred_p25/p75.",
        },
        "vitorias_sobre_constante_contagem": {
            "classe": "publicavel (contagem) com ressalva",
            "motivo": "contagem de vitorias (ex.: X/16) e uma razao com denominador fixo e PUBLICO "
                      "(16 particoes, fato de desenho, nao MAE); nao vaza MAE porque o denominador nao "
                      "e um MAE de referencia -- mas se cruzada por celula com o MAE do constante (que "
                      "tambem e publicavel, pois nao e da IEEE) mais o resultado booleano, ainda nao "
                      "reconstroi o MAE do braco vencedor sem o MAE absoluto; classificar como seguro.",
        },
        "vitorias_sobre_constante_cobertos_pl": {
            "classe": "nao comparavel -- nao calculado",
            "motivo": "constante para populacao cobertos so existe em RSSI no baselines_v2; mae_pl_db "
                      "do modelo esta no alvo PL; alvos diferentes, comparacao nao forcada.",
        },
    }

    payload = {
        "artefato_tipo": "e1_sinal_por_populacao",
        "frente": "ia-ablacao",
        "rodada": "R2",
        "fio": "gnn_rf_artigo2_mathematics_r2",
        "tarefa": "E1",
        "status": "provisorio_ate_contra_auditoria",
        "script": os.path.abspath(__file__),
        "script_sha256": sha256_self(),
        "fonte_canonica_declarada": TREINOS_DIR,
        "fonte_duplicata_declarada": DADOS_TREINOS_C1,
        "runs_esperados": len(BRACOS) * len(CIDADES) * len(SEEDS) * len(QS),
        "runs_encontrados": len(runs),
        "faltantes": faltantes,
        "duplicatas_conferidas": {
            "n": len(duplicatas),
            "n_batem": sum(1 for d in duplicatas if d["duplicata"]["bate_com_canonico"]),
            "n_nao_batem": sum(1 for d in duplicatas if not d["duplicata"]["bate_com_canonico"]),
            "detalhe": duplicatas,
        },
        "pares": {
            "n_validos": len(pares),
            "n_esperados": len(CIDADES) * len(SEEDS) * len(QS),
        },
        "pares_invalidos": invalidos,
        "por_corrida_80": por_corrida,
        "por_celula": por_celula,
        "por_cidade_n4": por_cidade,
        "bootstrap_cidade": {
            "todos_rssi": bootstrap_todos,
            "cobertos_pl": bootstrap_cobertos,
            "nota_geral": "unidade de inferencia = cidade (n=4, regra do fisico R1 post 210027); "
                           "com n=4 ha so 35 multiconjuntos distintos -- enumeracao exata reportada ao "
                           "lado do bootstrap; bootstrap com B=10000 e util so como leitura suave da "
                           "mesma distribuicao discreta, nao como aproximacao assintotica valida.",
        },
        "criterio_R1_bloco_G": criterio_r1,
        "iqr_predicao_por_braco": iqr_por_braco,
        "vitorias_sobre_constante": {
            "todos_rssi_resumo": vitorias["todos_rssi"],
            "todos_rssi_por_particao_16_seed42": vitorias["todos_rssi_por_particao_16"],
            "cobertos_pl": vitorias["cobertos_pl"],
            "detalhe_80_pares_x2_bracos": detalhe_vitorias,
        },
        "publicabilidade": publicabilidade,
        "nao_verificado": [
            {
                "item": "IQR de predicao no alvo PL (path_loss_pred_p25/p75)",
                "motivo": "diag do run JSON so grava path_loss_pred_std, sem percentis",
                "experimento_que_faltaria": {
                    "o_que": "gravar percentis p10/p25/p75/p90 da predicao de PL (nao so RSSI) no diag do "
                             "script de treino/avaliacao",
                    "contra_o_que": "IQR de predicao RSSI ja disponivel, para comparar dispersao relativa "
                                     "entre os dois alvos",
                    "criterio_de_sucesso": "presenca simultanea de rssi_pred_p25/p75 e path_loss_pred_p25/p75 "
                                            "nas 160 corridas existentes ou em uma nova rodada rotulada como tal",
                },
            },
            {
                "item": "preditor constante para populacao cobertos no alvo PL",
                "motivo": "baselines_v2 so calcula o constante piso/mediana em RSSI, mesmo na chave 'cobertura'",
                "experimento_que_faltaria": {
                    "o_que": "calcular preditor constante (piso ou mediana de treino) diretamente em PL "
                             "para o subconjunto coberto (mesma definicao rf_targets[:,0] < 299.0 dB)",
                    "contra_o_que": "mae_pl_db dos bracos mlpcf/c0c1cf, mesma particao test",
                    "criterio_de_sucesso": "vitorias/derrotas comparaveis linha a linha com o mesmo alvo (PL) "
                                            "e a mesma populacao (cobertos), sem misturar unidades",
                },
            },
            {
                "item": "populacao 'sentinela' nas 160 corridas mlpcf/c0c1cf g10b2",
                "motivo": "selecao.test_no_melhor_ckpt so grava 'todos' (mae_rssi_db) e 'cobertos' "
                          "(mae_pl_db); nao ha campo gravado para sentinela nestas corridas",
                "experimento_que_faltaria": {
                    "o_que": "gravar populacao sentinela (definicao formal existente em outros artefatos "
                             "da R1, ex. transferencia_bloqueada) tambem nas corridas mlpcf/c0c1cf g10b2",
                    "contra_o_que": "populacoes todos e cobertos ja existentes",
                    "criterio_de_sucesso": "campo sentinela presente e gravado com a mesma definicao usada "
                                            "na ablacao de topografia",
                },
            },
        ],
        "_fontes": {
            "script_sha256": sha256_self(),
            "arquivos_lidos_sha256": fontes_lidas,
            "n_arquivos_lidos": len(fontes_lidas),
        },
        "_gerado_em_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_path = args.out
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
        print(f"gravado: {out_path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
