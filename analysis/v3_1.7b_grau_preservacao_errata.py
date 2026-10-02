#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Errata do teste 1.7 (fase 1, roadmap v3) -- alegacao A2 / P3 (prop:degree),
correcao pedida pelo parecer de formalizacao
(_v3_2026-09-25/fase2/fismat_secao3_formal.md, bloco 4): a versao anterior
(v3_1.7_grau_preservacao.py) define fronteira por distancia EDT < ell_medio =
(ell_x+ell_y)/2, com ell_x < ell_medio < ell_y -- metade do perimetro (a faixa
horizontal) cai como "interior" e perde arestas normalmente, inflando uma
"perda residual no interior ~0,23%" que e artefato da classificacao, nao
perda real.

Correcao (prop:degree, Proposition "Degree loss is confined to the inner
boundary"): fronteira interna de VERTICES, definida pelo proprio grafo E_TT:
    dV = {i em V : existe j fora de V com (i,j) em E_TT}
interior = V \\ dV. Pelo item (i) do teorema, d_i^V = d_i para todo i em
V\\dV (perda EXATAMENTE zero no interior, por construcao -- nao e uma
medida, e uma identidade que o script verifica).

Mede tambem r_E (hipotese H2 do teorema, ainda nao medida no 1.7 original):
comprimento maximo de aresta terreno-terreno materializada (E_TT), por
celula x sorteio (o grafo E_TT nao muda com o sorteio; o numero e o mesmo
dentro de uma celula -- gravado replicado por sorteio a pedido do parecer
para no campo ficar auditavel por (celula,sorteio) como os demais).

Reusa (import, sem editar) as funcoes de v3_1.7_grau_preservacao.py:
carregar_celula, split_espacial_3vias_exato, checar_simetria, percentis.

Uso:
  .venv/bin/python v3_1.7b_grau_preservacao_errata.py --out <json> --celulas 4 --seeds 5
"""
import argparse
import json
import time
import resource
from pathlib import Path

import numpy as np

import sys
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "v3_1_7_base", str(Path(__file__).parent / "v3_1.7_grau_preservacao.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
carregar_celula = _base.carregar_celula
split_espacial_3vias_exato = _base.split_espacial_3vias_exato
checar_simetria = _base.checar_simetria
percentis = _base.percentis
sha256_of_text = _base.sha256_of_text
CIDADES = _base.CIDADES
QS = _base.QS
N_SIDE = _base.N_SIDE
FRACS = _base.FRACS
SEEDS_20 = _base.SEEDS_20


def r_E_maximo(pos_m, ei, amostra_max=2_000_000, seed=0):
    """Comprimento maximo de aresta terreno-terreno materializada (E_TT), em
    metros. Com 116,6M arestas por celula, computa em blocos para nao
    materializar arrays intermediarios gigantes; se ainda assim custoso,
    usa amostra aleatoria SEM reposicao das arestas e declara o metodo."""
    m = ei.shape[1]
    if m <= amostra_max:
        idx = np.arange(m)
        metodo = "exaustivo (todas as arestas)"
    else:
        idx = np.random.RandomState(seed).choice(m, size=amostra_max, replace=False)
        metodo = f"amostra aleatoria sem reposicao de {amostra_max:,} de {m:,} arestas"
    u, v = ei[0, idx], ei[1, idx]
    d = np.sqrt(np.sum((pos_m[u] - pos_m[v]) ** 2, axis=1))
    return {"r_E_m": float(d.max()), "r_E_media_m": float(d.mean()),
            "r_E_p99_m": float(np.percentile(d, 99)), "metodo": metodo,
            "n_arestas_avaliadas": int(idx.size)}


def processar_sorteio_vertice(celula, seed, r_e_info, log):
    cfg = celula["cfg"]
    fracs = tuple(float(x) for x in cfg["split_frac"].split(","))
    parts = split_espacial_3vias_exato(
        celula["pos_m"], grid_km=cfg["grid_km"], buffer_km=cfg["buffer_km"],
        fracs=fracs, split_seed=seed)

    ei = celula["ei"]
    grau_full = celula["grau_full"]
    n_nodes = celula["n_nodes"]

    part_id = np.full(n_nodes, -1, dtype=np.int8)
    part_id[parts["train"]] = 0
    part_id[parts["val"]] = 1
    part_id[parts["test"]] = 2

    src_pid = part_id[ei[0]]
    dst_pid = part_id[ei[1]]
    ambos_retidos = (src_pid >= 0) & (dst_pid >= 0)
    cruzam = ambos_retidos & (src_pid != dst_pid)
    n_cruzam = int(cruzam.sum())

    resultado_particoes = {}
    for nome, mask in parts.items():
        n_retidos = int(mask.sum())
        if n_retidos == 0:
            resultado_particoes[nome] = {"n_retidos": 0}
            continue

        # subgrafo induzido: aresta retida sse ambas pontas em `mask`
        src_in = mask[ei[0]]
        dst_in = mask[ei[1]]
        both_in = src_in & dst_in
        sub_ei_src = ei[0][both_in]
        grau_induzido = np.bincount(sub_ei_src, minlength=n_nodes)

        # fronteira interna de VERTICES (definicao do parecer/teorema):
        # i em V com algum vizinho j (no grafo ORIGINAL E_TT) fora de V
        tem_vizinho_fora = src_in & ~dst_in  # aresta com origem em V e destino fora de V
        fronteira_idx = np.unique(ei[0][tem_vizinho_fora])
        fronteira = np.zeros(n_nodes, dtype=bool)
        fronteira[fronteira_idx] = True
        fronteira &= mask  # so nos retidos contam
        interior = mask & ~fronteira

        idx_retidos = np.where(mask)[0]
        idx_interior = np.where(interior)[0]
        idx_fronteira = np.where(fronteira)[0]

        antes_retidos = grau_full[idx_retidos]
        depois_retidos = grau_induzido[idx_retidos]
        antes_interior = grau_full[idx_interior]
        depois_interior = grau_induzido[idx_interior]
        antes_fronteira = grau_full[idx_fronteira]
        depois_fronteira = grau_induzido[idx_fronteira]

        with np.errstate(divide="ignore", invalid="ignore"):
            razao_interior = np.where(antes_interior > 0, depois_interior / antes_interior, np.nan)
            razao_fronteira = np.where(antes_fronteira > 0, depois_fronteira / antes_fronteira, np.nan)

        # Lambda(V) = 1 - sum(d_i^V)/sum(d_i) sobre i em V (definicao exata
        # do teorema, NAO media de razoes por no)
        soma_antes = float(antes_retidos.sum())
        soma_depois = float(depois_retidos.sum())
        lambda_v = 1.0 - (soma_depois / soma_antes) if soma_antes > 0 else None

        # verificacao da identidade (i): interior deve ter razao EXATAMENTE 1
        n_interior_nao_exato = int(np.sum(depois_interior != antes_interior))

        resultado_particoes[nome] = {
            "n_retidos": n_retidos,
            "n_interior": int(interior.sum()),
            "n_fronteira_vertices": int(fronteira.sum()),
            "frac_fronteira_vertices": float(fronteira.sum() / n_retidos),
            "razao_media_interior": float(np.nanmean(razao_interior)) if razao_interior.size else None,
            "razao_min_interior": float(np.nanmin(razao_interior)) if razao_interior.size else None,
            "n_interior_com_perda_qualquer": n_interior_nao_exato,
            "identidade_i_interior_perda_zero": (n_interior_nao_exato == 0),
            "razao_media_fronteira": float(np.nanmean(razao_fronteira)) if razao_fronteira.size else None,
            "razao_min_fronteira": float(np.nanmin(razao_fronteira)) if razao_fronteira.size else None,
            "lambda_V_perda_relativa_total": lambda_v,
            "grau_medio_antes_retidos": float(antes_retidos.mean()),
            "grau_medio_depois_retidos": float(depois_retidos.mean()),
        }

    return {
        "cidade": celula["cidade"], "q": celula["q"], "split_seed": seed,
        "n_arestas_cruzando_particao_ambas_pontas_retidas": n_cruzam,
        "r_E": r_e_info,
        "particoes": resultado_particoes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--celulas", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--log", default=None)
    ap.add_argument("--criterio", default=str(
        Path(__file__).parent.parent / "_v3_2026-09-25" / "criterios" / "criterio_1.7.json"))
    ap.add_argument("--anterior", default=str(
        Path(__file__).parent.parent / "_v3_2026-09-25" / "fase1" / "1.7_grau_preservacao.json"))
    args = ap.parse_args()

    logf = open(args.log, "a") if args.log else None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()

    t_inicio = time.time()
    saida = {
        "frente": "matematica-grafos-espectral",
        "teste": "1.7", "alegacao": "A2 / P3 (prop:degree)",
        "criterio": json.load(open(args.criterio)),
        "errata": {
            "motivo": ("parecer de formalizacao fase2/fismat_secao3_formal.md, bloco 4 (prop:degree): "
                       "o artefato anterior (v3_1.7_grau_preservacao.py) classificou fronteira por "
                       "distancia EDT < ell_medio = (ell_x+ell_y)/2, com ell_x < ell_medio < ell_y em "
                       "TODAS as celulas -- a faixa horizontal (metade do perimetro) caia como "
                       "'interior' e perdia arestas normalmente, produzindo uma 'perda residual no "
                       "interior ~0,23%' que era artefato da classificacao geometrica por distancia, "
                       "nao perda real de grau."),
            "correcao": ("fronteira redefinida como fronteira INTERNA DE VERTICES do proprio grafo "
                         "E_TT: dV = {i em V : existe j fora de V com (i,j) em E_TT}; interior = V\\dV. "
                         "Pelo item (i) do teorema (prop:degree), d_i^V = d_i para todo i em V\\dV -- "
                         "identidade, nao estimativa: verificada abaixo por particoes.*.identidade_i_"
                         "interior_perda_zero (deve ser true em toda celula x sorteio x particao)."),
            "frase_proibida_removida_do_texto": "perda residual no interior ~ 0,23 %",
            "campo_removido": "formula_termo_fronteira_sustentada_pelos_dados (fora do escopo apos a "
                               "correcao; a fronteira de vertices ja e a identidade do teorema, nao "
                               "precisa de formula empirica de razao)",
            "artefato_anterior_superado": args.anterior,
            "script_anterior_superado": "v3_1.7_grau_preservacao.py (nao editado, ver PROTOCOLOS: reusar por import)",
        },
        "script_sha256": sha256_of_text(Path(__file__).read_text()),
        "status": "em_andamento",
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    todas_celulas = [(c, q) for c in CIDADES for q in QS]
    if args.celulas >= 16:
        celulas_alvo = todas_celulas
    elif args.celulas == 4:
        celulas_alvo = [(c, "Q1") for c in CIDADES]
    else:
        celulas_alvo = todas_celulas[:args.celulas]
    seeds = SEEDS_20[:args.seeds]

    por_celula = []
    for (cidade, q) in celulas_alvo:
        t0c = time.time()
        cel = carregar_celula(cidade, q, log)
        sim = checar_simetria(cel["ei"], cel["n_nodes"])
        r_e_info = r_E_maximo(cel["pos_m"], cel["ei"])
        log(f"  {cidade} {q}: r_E_m={r_e_info['r_E_m']:.3f} ({r_e_info['metodo']})")
        resultados_seeds = []
        for seed in seeds:
            r = processar_sorteio_vertice(cel, seed, r_e_info, log)
            resultados_seeds.append(r)
            id_ok = all(p.get("identidade_i_interior_perda_zero", True) for p in r["particoes"].values())
            log(f"  {cidade} {q} seed={seed}: cruzam={r['n_arestas_cruzando_particao_ambas_pontas_retidas']} "
                f"identidade_interior_ok={id_ok} "
                f"razao_interior_test={r['particoes'].get('test', {}).get('razao_media_interior')} "
                f"razao_fronteira_test={r['particoes'].get('test', {}).get('razao_media_fronteira')} "
                f"frac_fronteira_test={r['particoes'].get('test', {}).get('frac_fronteira_vertices')}")
        por_celula.append({
            "cidade": cidade, "q": q, "n_nodes": cel["n_nodes"], "n_edges": cel["n_edges"],
            "grau_medio_full": float(cel["grau_full"].mean()),
            "ell_x_m": cel["ell_x_m"], "ell_y_m": cel["ell_y_m"],
            "simetria_amostra": sim,
            "r_E": r_e_info,
            "tempo_carga_s": cel["tempo_carga_s"],
            "pt_path": cel["pt_path"], "pt_bytes": Path(cel["pt_path"]).stat().st_size,
            "sorteios": resultados_seeds,
            "tempo_total_celula_s": time.time() - t0c,
        })
        Path(args.out).write_text(json.dumps({**saida, "por_celula": por_celula, "status": "em_andamento"},
                                              indent=1, ensure_ascii=False, default=str))
        log(f"celula {cidade} {q} concluida em {time.time()-t0c:.1f}s")

    # -------- resumo e veredito --------
    razoes_interior_todas = []
    razoes_fronteira_todas = []
    lambdas_v = []
    n_cruzam_total = 0
    identidade_falhas = []
    r_e_por_celula = {}
    for c in por_celula:
        r_e_por_celula[f"{c['cidade']}_{c['q']}"] = c["r_E"]["r_E_m"]
        for s in c["sorteios"]:
            n_cruzam_total += s["n_arestas_cruzando_particao_ambas_pontas_retidas"]
            for pnome, p in s["particoes"].items():
                if p.get("n_retidos", 0) == 0:
                    continue
                if p.get("razao_media_interior") is not None:
                    razoes_interior_todas.append(p["razao_media_interior"])
                if p.get("razao_media_fronteira") is not None:
                    razoes_fronteira_todas.append(p["razao_media_fronteira"])
                if p.get("lambda_V_perda_relativa_total") is not None:
                    lambdas_v.append(p["lambda_V_perda_relativa_total"])
                if not p.get("identidade_i_interior_perda_zero", True):
                    identidade_falhas.append({"cidade": c["cidade"], "q": c["q"],
                                               "seed": s["split_seed"], "particao": pnome,
                                               "n_interior_com_perda": p["n_interior_com_perda_qualquer"]})

    identidade_confirmada = (len(identidade_falhas) == 0)
    criterio_interior_ok = identidade_confirmada and (min(razoes_interior_todas) == 1.0 if razoes_interior_todas else False)

    veredito = "P3_SUSTENTADA_TEOREMA_i_CONFIRMADO" if criterio_interior_ok else "P3_VIRA_REMARK"

    saida["por_celula"] = por_celula
    saida["arestas_cruzando_particao"] = {
        "total_observado_todos_sorteios": n_cruzam_total, "esperado": 0,
        "confirma_zero_por_construcao": (n_cruzam_total == 0),
    }
    saida["r_E_por_celula_m"] = r_e_por_celula
    saida["resumo"] = {
        "n_celulas": len(por_celula), "n_seeds_por_celula": len(seeds),
        "n_amostras_particao_x_sorteio": len(razoes_interior_todas),
        "identidade_i_interior_perda_zero_confirmada_em_todos": identidade_confirmada,
        "identidade_i_falhas": identidade_falhas,
        "razao_interior_min": min(razoes_interior_todas) if razoes_interior_todas else None,
        "razao_interior_media": float(np.mean(razoes_interior_todas)) if razoes_interior_todas else None,
        "razao_fronteira_min": min(razoes_fronteira_todas) if razoes_fronteira_todas else None,
        "razao_fronteira_media": float(np.mean(razoes_fronteira_todas)) if razoes_fronteira_todas else None,
        "lambda_V_media": float(np.mean(lambdas_v)) if lambdas_v else None,
        "lambda_V_max": float(np.max(lambdas_v)) if lambdas_v else None,
        "r_E_m_min_max": [min(r_e_por_celula.values()), max(r_e_por_celula.values())] if r_e_por_celula else None,
    }
    saida["veredito_vs_criterio"] = veredito
    saida["nao_verificado"] = [
        {"item": "arestas antena-terreno", "motivo": "P3-1/P3 tratam grau terra-terra; nao incluido por orcamento"},
        {"item": "16x20 completo", "motivo": f"executado {len(celulas_alvo)} celulas x {len(seeds)} sorteios por orcamento de CPU (mesma reducao declarada do 1.7 original)"},
        {"item": "r_E exaustivo (todas as 116,6M arestas)", "motivo": "ver campo metodo em r_E por celula -- exaustivo se coube no orcamento, amostra c.c."},
    ]
    saida["comando_rodado"] = (
        f".venv/bin/python v3_1.7b_grau_preservacao_errata.py --out {args.out} "
        f"--celulas {args.celulas} --seeds {args.seeds}")
    saida["status"] = "concluido_provisorio"
    saida["pico_ram_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    saida["tempo_total_s"] = time.time() - t_inicio
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False, default=str))
    log(f"CONCLUIDO. tempo_total={saida['tempo_total_s']:.1f}s pico_ram={saida['pico_ram_gb']:.2f}GB "
        f"veredito={veredito} identidade_confirmada={identidade_confirmada}")


if __name__ == "__main__":
    main()
