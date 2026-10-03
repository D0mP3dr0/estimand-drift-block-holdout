#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""
Agregacao do lote G1 (modelos treinados a g = 10 km, b = 2 km) EXATAMENTE como
pre-registrada em `_v3_2026-09-25/criterios/criterio_G1_modelos_g10.json`
(campo `metricas`), fixado antes de existir qualquer script ou resultado.

COPIA DECLARADA de scripts/v3_A4_agregar.py
  sha256 do original: b5bb904429fd7ff79566a013e202f14d256794fa82cb32f427d6cdfa3bb07b6f
Reaproveitados sem mudanca de logica: sha256, manifest_cftudo, ler_run,
mae_pop (MAE de RSSI = canal 3 e de PL = canal 0 a partir do .npz fisico;
valido = target[:,0] < 299), med, dp (ddof=1), proveniencia. Removidos: braco de
sensibilidade, tolerancias e gradscaler do A4. Novos: recusa por bloco, ANOVA
de dois fatores sem repeticao (bloco 2), correlacao com o constante (2.1),
conferencia de partição contra fase2/_v3_2.1_3.1_parcial_16x60rnd.json.

Regras:
  * Cada bloco e agregado SO quando completo (todas as corridas do plano do
    bloco com run JSON completo + .npz, ou registradas `sem_validos`). Bloco
    incompleto: RECUSA (rc=2), nao grava nada. Sem flag `--parcial`.
  * O bloco 2 exige o bloco 1 completo (as 3 sementes de cada celula incluem a
    semente 42 do bloco 1 nos 5 primeiros sorteios).
  * Este script NAO interpreta: calcula as metricas e as comparacoes
    aritmeticas com os limiares do criterio (0,132 dB; 0,117 dB; 3x; 2x), sem
    atribuir REFORCA/DELIMITA. A leitura e do rigor.
  * Sorteio 'sem validos' (nenhum no valido no teste) entra como `sem_validos`
    na lista e fica FORA dos dp/correlacoes (n reportado).

Saida: gpu/G1/agregado_G1_bloco<N>.json   (uso: v3_G1_agregar.py [--bloco 1|2|3])
Nenhum numero e citavel sem contra-auditoria (reexecucao por quem nao escreveu).
"""
import argparse
import glob
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

RAIZ = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
G1 = RAIZ / "gpu" / "G1"
CRIT = RAIZ / "criterios" / "criterio_G1_modelos_g10.json"
PLANO = G1 / "plano_G1.json"
STATUS = G1 / "lote_G1_status.json"
PARCIAL_2_1 = RAIZ / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json"
MANIFEST_V4 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/manifest_mathematics_v4.jsonl")
SENTINELA_PL = 299.0
RUIDO_REPETICAO_DB = 0.132      # maior diferenca entre repeticoes do GNN (A2c), do criterio
RUIDO_SENTINELA_DB = 0.117      # maior diferenca entre repeticoes nos sentinelas (A2c), do criterio
POPS = ("mae_rssi_validos_db", "mae_rssi_sentinela_db", "mae_rssi_todos_db", "mae_pl_validos_db")
BASELINES = ("fspl", "hata_rural", "cost231_sub")
CELULAS_POR_BLOCO = {1: ("bauru_Q1", "campinas_Q1"), 2: ("bauru_Q1", "campinas_Q1"), 3: ("bauru_Q3", "campinas_Q3")}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest_cftudo() -> dict:
    out = {}
    with open(MANIFEST_V4, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                e = json.loads(ln)
                if e.get("grupo") == "tensores_cftudo":
                    out[Path(e.get("caminho", "")).name] = e.get("sha256")
    return out


def ler_run(d: Path):
    js = sorted(glob.glob(str(d / "run_*.json")))
    zs = sorted(glob.glob(str(d / "predicoes_*.npz")))
    if not js or not zs:
        raise FileNotFoundError(f"run JSON ou .npz ausente em {d}")
    with open(js[0], "r", encoding="utf-8") as f:
        dj = json.load(f)
    z = np.load(zs[0])
    return dj, {k: z[k] for k in z.files}, Path(zs[0])


def mae_pop(z: dict) -> dict:
    tgt = z["target"].astype(np.float64)
    pred = z["pred"].astype(np.float64)
    sent = tgt[:, 0] >= SENTINELA_PL
    if "sentinela" in z and not np.array_equal(sent, z["sentinela"].astype(bool)):
        raise RuntimeError("campo sentinela do .npz nao bate com target[:,0] >= 299")
    e3 = np.abs(tgt[:, 3] - pred[:, 3])
    e0 = np.abs(tgt[:, 0] - pred[:, 0])
    val = ~sent
    return {
        "n": int(tgt.shape[0]), "n_validos": int(val.sum()), "n_sentinela": int(sent.sum()),
        "mae_rssi_validos_db": float(e3[val].mean()) if val.any() else None,
        "mae_rssi_sentinela_db": float(e3[sent].mean()) if sent.any() else None,
        "mae_rssi_todos_db": float(e3.mean()),
        "mae_pl_validos_db": float(e0[val].mean()) if val.any() else None,
        "idx_sha256": hashlib.sha256(np.sort(z["idx_global"].astype(np.int64)).tobytes()).hexdigest(),
    }


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


def anova2_sem_repeticao(M: np.ndarray) -> dict:
    """M: a sorteios (linhas) x b sementes (colunas), sem celulas vazias."""
    a, b = M.shape
    gm = M.mean()
    ss_sor = b * float(((M.mean(axis=1) - gm) ** 2).sum())
    ss_sem = a * float(((M.mean(axis=0) - gm) ** 2).sum())
    ss_tot = float(((M - gm) ** 2).sum())
    ss_res = ss_tot - ss_sor - ss_sem
    ms_sor, ms_sem = ss_sor / (a - 1), ss_sem / (b - 1)
    ms_res = ss_res / ((a - 1) * (b - 1))
    var_sor_bruta = (ms_sor - ms_res) / b
    var_sem_bruta = (ms_sem - ms_res) / a
    return {"a_sorteios": a, "b_sementes": b, "ms_sorteio": ms_sor, "ms_semente": ms_sem, "ms_residuo": ms_res,
            "componente_sorteio_var_bruta": var_sor_bruta, "componente_semente_var_bruta": var_sem_bruta,
            "componente_sorteio_var": max(0.0, var_sor_bruta), "componente_semente_var": max(0.0, var_sem_bruta),
            "componente_residuo_var": ms_res,
            "componente_sorteio_ge_semente": bool(max(0.0, var_sor_bruta) >= max(0.0, var_sem_bruta))}


def carregar_2_1() -> dict:
    with open(PARCIAL_2_1, "r", encoding="utf-8") as f:
        p = json.load(f)
    out = {}
    for cel, d in p["celulas"].items():
        for s in d.get("por_sorteio", []):
            if s.get("status") == "ok":
                out[(cel, int(s["split_seed"]))] = {
                    "n_test": s["n_test"], "validos": s["n_pop_teste"]["validos"], "todos": s["n_pop_teste"]["todos"],
                    "mae_constante_validos": s["mae_constante_teste"]["validos"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bloco", type=int, choices=(1, 2, 3), default=None,
                    help="default: agrega todo bloco COMPLETO; blocos incompletos sao recusados")
    args = ap.parse_args()

    sha_crit = sha256(CRIT)
    with open(PLANO, "r", encoding="utf-8") as f:
        plano = json.load(f)["corridas"]
    with open(STATUS, "r", encoding="utf-8") as f:
        status = json.load(f)
    sem_val = {c["run_label"] for c in status["corridas"] if c.get("sem_validos")}
    man = manifest_cftudo()
    ref21 = carregar_2_1()

    def completa(c) -> bool:
        if c["run_label"] in sem_val:
            return True
        d = G1 / c["run_label"]
        return bool(glob.glob(str(d / "run_*.json")) and glob.glob(str(d / "predicoes_*.npz")))

    def bloco_completo(n: int) -> bool:
        return all(completa(c) for c in plano if c["bloco"] == n)

    cache = {}

    def carregar(c) -> dict:
        lbl = c["run_label"]
        if lbl in cache:
            return cache[lbl]
        if lbl in sem_val and not glob.glob(str(G1 / lbl / "predicoes_*.npz")):
            r = {"run_label": lbl, "sem_validos": True, "mae": None}
        else:
            dj, z, zp = ler_run(G1 / lbl)
            ins = dj.get("insumos") or {}
            rf = Path(ins.get("rf_data_file") or "").name
            geo = dj.get("split") or {}
            m = mae_pop(z)
            ref = ref21.get((f"{c['cidade']}_{c['quadrante']}", c["split_seed"]))
            cfg = dj.get("config") or {}
            r = {
                "run_label": lbl, "sem_validos": m["n_validos"] == 0, "npz": str(zp), "mae": m,
                "grid_km": cfg.get("grid_km"), "buffer_km": cfg.get("buffer_km"), "split_seed_run": dj.get("split_seed"),
                "seed_treino_run": dj.get("seed"),
                "proveniencia_ok": bool(ins.get("sha256_rf_data") and man.get(rf) == ins.get("sha256_rf_data")),
                "dist_min_entre_particoes_km": {k: v.get("dist_min_km") for k, v in ((geo.get("verificacao") or {}).get("pares") or {}).items()},
                "intersecoes": (geo.get("verificacao") or {}).get("intersecoes"),
                "n_test_retido_run": (geo.get("n_nos_apos_buffer") or {}).get("test"),
                "baselines_test": {b: ((dj.get("baselines_analiticos") or {}).get("test") or {}).get(f"baseline_{b}_mae") for b in BASELINES},
                "melhor_epoca": (dj.get("selecao") or {}).get("melhor_epoca"),
                "bate_com_2_1": (None if ref is None else {
                    "n_test_igual": geo.get("n_nos_apos_buffer", {}).get("test") == ref["n_test"],
                    "n_validos_igual": m["n_validos"] == ref["validos"],
                    "n_todos_igual": m["n"] == ref["todos"]}),
                "mae_constante_validos_2_1": None if ref is None else ref["mae_constante_validos"],
            }
        cache[lbl] = r
        return r

    def por_celula_modelo(bloco_n: int, cel: str, tipo: str, seed: int, n_sort: int):
        runs = [c for c in plano if c["bloco"] == (1 if bloco_n == 2 else bloco_n) and c["tipo"] == tipo
                and f"{c['cidade']}_{c['quadrante']}" == cel and c["seed_treino"] == seed]
        runs = sorted(runs, key=lambda c: c["indice_sorteio"])[:n_sort]
        return {c["split_seed"]: carregar(c) for c in runs}

    def agrega_bloco_simples(n: int) -> dict:
        out = {"artefato": f"agregado_G1_bloco{n}", "criterio_sha256": sha_crit, "plano_sha256": sha256(PLANO),
               "limiares_do_criterio_db": {"ruido_repeticao_gnn": RUIDO_REPETICAO_DB, "ruido_sentinela": RUIDO_SENTINELA_DB},
               "celulas": {}}
        for cel in CELULAS_POR_BLOCO[n]:
            g = por_celula_modelo(n, cel, "gnn", 42, 20)
            m = por_celula_modelo(n, cel, "mlp", 42, 20)
            ordem = [c["split_seed"] for c in plano if c["bloco"] == n and f"{c['cidade']}_{c['quadrante']}" == cel and c["tipo"] == "gnn"]
            ss = [s for s in ordem if s in g and s in m]
            bloco = {"sorteios": ss, "sem_validos": [s for s in ss if g[s]["sem_validos"] or m[s]["sem_validos"]], "por_modelo": {}}
            for nome, store in (("gnn", g), ("mlp", m)):
                ok = [s for s in ss if not store[s]["sem_validos"]]
                v = [store[s]["mae"]["mae_rssi_validos_db"] for s in ok]
                d = dp(v)
                const = [store[s]["mae_constante_validos_2_1"] for s in ok]
                bloco["por_modelo"][nome] = {
                    "n_sorteios_com_validos": len(ok),
                    "mae_validos_por_sorteio": {str(s): store[s]["mae"]["mae_rssi_validos_db"] for s in ok},
                    "dp_entre_sorteios_validos_db": d,
                    "razao_dp_sobre_0_132": razao(d, RUIDO_REPETICAO_DB),
                    "dp_ge_3x_0_132": (None if d is None else bool(d >= 3 * RUIDO_REPETICAO_DB)),
                    "dp_lt_2x_0_132": (None if d is None else bool(d < 2 * RUIDO_REPETICAO_DB)),
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
            par = [s for s in ss if s not in bloco["sem_validos"]]
            dif_sent = {str(s): (abs(g[s]["mae"]["mae_rssi_sentinela_db"] - m[s]["mae"]["mae_rssi_sentinela_db"])
                                 if g[s]["mae"]["mae_rssi_sentinela_db"] is not None and m[s]["mae"]["mae_rssi_sentinela_db"] is not None else None)
                        for s in par}
            md = med(list(dif_sent.values()))
            bloco["paridade_sentinela"] = {"abs_gnn_menos_mlp_por_sorteio": dif_sent, "mediana_db": md,
                                           "mediana_le_0_117": (None if md is None else bool(md <= RUIDO_SENTINELA_DB)),
                                           "mesma_particao_gnn_mlp": all(g[s]["mae"]["idx_sha256"] == m[s]["mae"]["idx_sha256"] for s in par)}
            out["celulas"][cel] = bloco
        # contagem mecanica entre as 2 celulas do bloco (a regra do criterio fala em >= 3 de 4 celulas: soma-se com o outro bloco)
        out["contagem_mecanica"] = {
            "celulas_com_dp_ge_3x_gnn_e_mlp": [c for c, b in out["celulas"].items()
                                                 if b["por_modelo"]["gnn"]["dp_ge_3x_0_132"] and b["por_modelo"]["mlp"]["dp_ge_3x_0_132"]],
            "celulas_com_dp_lt_2x_gnn_e_mlp": [c for c, b in out["celulas"].items()
                                                 if b["por_modelo"]["gnn"]["dp_lt_2x_0_132"] and b["por_modelo"]["mlp"]["dp_lt_2x_0_132"]],
            "celulas_paridade_le_0_117": [c for c, b in out["celulas"].items() if b["paridade_sentinela"]["mediana_le_0_117"]],
            "nota": "contagem aritmetica; REFORCA/DELIMITA e leitura do rigor, nao deste script"}
        return out

    def agrega_bloco2() -> dict:
        out = {"artefato": "agregado_G1_bloco2", "criterio_sha256": sha_crit, "plano_sha256": sha256(PLANO), "celulas": {},
               "definicao_dp_entre_sementes": ("dp pooled dentro do sorteio: raiz da media, sobre os 5 sorteios, da variancia (ddof=1) "
                                              "do MAE_validos entre as 3 sementes (42, 43, 44); o criterio nao define o estimador, "
                                              "esta e a escolha deste script (declarada)."),
               "nota_componentes": "ANOVA de dois fatores sem repeticao, sorteio (5) x semente (3); componentes por quadrados medios esperados, truncados em 0 (brutos tambem reportados)"}
        for cel in CELULAS_POR_BLOCO[2]:
            ent = {"gnn": {}, "mlp": {}}
            for tipo in ("gnn", "mlp"):
                por_semente = {}
                for sd in (42, 43, 44):
                    por_semente[sd] = {}
                    cs = [c for c in plano if c["tipo"] == tipo and f"{c['cidade']}_{c['quadrante']}" == cel
                          and c["seed_treino"] == sd and c["bloco"] in ((1,) if sd == 42 else (2,))]
                    cs = sorted(cs, key=lambda c: c["indice_sorteio"])[:5]
                    for c in cs:
                        por_semente[sd][c["split_seed"]] = carregar(c)
                sorteios = [c["split_seed"] for c in plano if c["bloco"] == 2 and c["seed_treino"] == 43
                            and c["tipo"] == tipo and f"{c['cidade']}_{c['quadrante']}" == cel]
                lin = [s for s in sorteios if all(s in por_semente[sd] and not por_semente[sd][s]["sem_validos"] for sd in (42, 43, 44))]
                M = np.array([[por_semente[sd][s]["mae"]["mae_rssi_validos_db"] for sd in (42, 43, 44)] for s in lin], float)
                var_dentro = [float(np.var(M[i], ddof=1)) for i in range(len(lin))]
                dp_sem = float(np.sqrt(np.mean(var_dentro))) if lin else None
                g20 = por_celula_modelo(1, cel, tipo, 42, 20)
                dp20 = dp([g20[s]["mae"]["mae_rssi_validos_db"] for s in g20 if not g20[s]["sem_validos"]])
                dp5 = dp(list(M.mean(axis=1))) if len(lin) >= 2 else None
                ent[tipo] = {
                    "sorteios_usados": lin, "matriz_mae_validos_sorteio_x_semente_42_43_44": M.tolist(),
                    "dp_entre_sementes_pooled_db": dp_sem,
                    "dp_entre_sorteios_20_semente42_db": dp20,
                    "razao_dp_sorteios20_sobre_dp_sementes": razao(dp20, dp_sem),
                    "dp_das_medias_por_sorteio_5_db": dp5,
                    "razao_dp_sorteios5_sobre_dp_sementes": razao(dp5, dp_sem),
                    "anova_dois_fatores_sem_repeticao": (anova2_sem_repeticao(M) if len(lin) >= 3 else None),
                    "dp_entre_sementes_sobre_0_132": razao(dp_sem, RUIDO_REPETICAO_DB)}
            out["celulas"][cel] = ent
        out["contagem_mecanica"] = {"celulas_componente_sorteio_ge_semente_gnn_e_mlp": [
            c for c, e in out["celulas"].items()
            if all((e[t]["anova_dois_fatores_sem_repeticao"] or {}).get("componente_sorteio_ge_semente") for t in ("gnn", "mlp"))],
            "nota": "contagem aritmetica; leitura e do rigor"}
        return out

    alvo = [args.bloco] if args.bloco else [1, 2, 3]
    rc = 0
    for n in alvo:
        if not bloco_completo(n) or (n == 2 and not bloco_completo(1)):
            falt = [c["run_label"] for c in plano if c["bloco"] in ((1, 2) if n == 2 else (n,)) and not completa(c)]
            print(f"bloco {n}: RECUSADO (incompleto, {len(falt)} corridas sem resultado); nao grava resultado parcial")
            rc = 2
            continue
        out = agrega_bloco2() if n == 2 else agrega_bloco_simples(n)
        path = G1 / f"agregado_G1_bloco{n}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"bloco {n}: gravado {path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
