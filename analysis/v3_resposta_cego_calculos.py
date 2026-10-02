"""Calculos da resposta ao revisor cego (criterio criterios/criterio_resposta_cego_v3-10.json).

F1: dp entre celulas num sorteio fixo (e das medias por celula) do MAE do constante.
F2: decomposicao da variancia entre sorteios do MAE em todos os nos do constante em composicao,
    condicional e resto, por celula.
F5: erro-padrao de Monte Carlo da Secao 4.3 a partir do 2.3b (valores por sorteio), com a conferencia
    de igualdade dos agregados contra o 2.3 original.
Saida: _v3_2026-09-25/fase3/3.7_resposta_cego.json
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

B = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
PARC = B / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json"
ORIG23 = B / "fase2" / "2.3_estimando_ht_baselines.json"
NOVO23 = B / "fase2" / "2.3b_estimando_por_sorteio.json"
OUT = B / "fase3" / "3.7_resposta_cego.json"


def resumo(v):
    v = sorted(v)
    return {"n": len(v), "mediana": st.median(v), "min": v[0], "max": v[-1]}


def f1_f2(d):
    cel = d["celulas"]
    seeds = [s["split_seed"] for s in next(iter(cel.values()))["por_sorteio"]]
    por_seed_val, por_seed_todos = [], []
    medias_val, medias_todos = [], []
    decomp = {}
    for i, sd in enumerate(seeds):
        vv, vt = [], []
        for k, c in cel.items():
            s = c["por_sorteio"][i]
            assert s["split_seed"] == sd
            if s.get("status") != "ok":
                continue
            m = s["mae_constante_teste"]
            if m["validos"] is not None:
                vv.append(m["validos"])
            vt.append(m["todos"])
        if len(vv) >= 2:
            por_seed_val.append(st.stdev(vv))
        por_seed_todos.append(st.stdev(vt))
    for k, c in cel.items():
        ok = [s for s in c["por_sorteio"] if s.get("status") == "ok"]
        ev = [s["mae_constante_teste"]["validos"] for s in ok if s["mae_constante_teste"]["validos"] is not None]
        et = [s["mae_constante_teste"]["todos"] for s in ok]
        medias_val.append(st.mean(ev)); medias_todos.append(st.mean(et))
        # F2: so sorteios com no valido; X = 1 - pi_te (fracao valida), Y = e_val, MAE_todos = X*Y
        xs, ys, zs = [], [], []
        n_fora = 0
        for s in ok:
            n = s["n_pop_teste"]; e = s["mae_constante_teste"]
            if not n["validos"] or e["validos"] is None:
                n_fora += 1; continue
            x = n["validos"] / n["todos"]; y = e["validos"]
            xs.append(x); ys.append(y); zs.append(e["todos"])
        # conferencia: MAE_todos = X*Y (constante exato na sentinela)
        max_dif = max(abs(z - x * y) for x, y, z in zip(xs, ys, zs))
        vz = st.pvariance(zs); comp = st.mean(ys) ** 2 * st.pvariance(xs); cond = st.mean(xs) ** 2 * st.pvariance(ys)
        decomp[k] = {"n_sorteios": len(zs), "n_sem_no_valido": n_fora, "var_mae_todos": vz,
                     "frac_composicao": comp / vz if vz else None, "frac_condicional": cond / vz if vz else None,
                     "frac_resto": (vz - comp - cond) / vz if vz else None, "max_dif_identidade_dB": max_dif}
    return {
        "F1": {"dp_entre_celulas_por_sorteio_validos_dB": resumo(por_seed_val),
               "dp_entre_celulas_por_sorteio_todos_dB": resumo(por_seed_todos),
               "dp_entre_celulas_das_medias_validos_dB": st.stdev(medias_val),
               "dp_entre_celulas_das_medias_todos_dB": st.stdev(medias_todos),
               "n_sorteios": len(seeds), "n_celulas": len(cel)},
        "F2": {"por_celula": decomp,
               "mediana_frac_composicao": st.median(v["frac_composicao"] for v in decomp.values()),
               "mediana_frac_condicional": st.median(v["frac_condicional"] for v in decomp.values()),
               "mediana_frac_resto": st.median(v["frac_resto"] for v in decomp.values()),
               "max_dif_identidade_dB_todas": max(v["max_dif_identidade_dB"] for v in decomp.values())},
    }


def f5():
    if not NOVO23.exists():
        return {"status": "2.3b ainda nao existe"}
    o = json.load(open(ORIG23))["por_celula"]; nv = json.load(open(NOVO23))["por_celula"]
    out, ok_all = {}, True
    for cel, preds in nv.items():
        for pred, pops in preds.items():
            for pop, r in pops.items():
                ro = o[cel][pred][pop]
                iguais = all(ro.get(f) is None and r.get(f) is None or (ro.get(f) is not None and r.get(f) is not None and abs(ro[f] - r[f]) < 1e-3)
                             for f in ("a_mae_dominio_dB", "b_media_sorteios_mae_teste_dB", "c_razao_esperancas_p_te_dB", "diferenca_b_menos_c_dB"))
                ok_all &= iguais
                ev = r.get("Err_sigma_por_sorteio_A") or []
                n = len(ev)
                ep = (st.stdev(ev) / math.sqrt(n)) if n >= 2 else None
                bmc = r.get("diferenca_b_menos_c_dB"); bma = r.get("vies_protocolo_b_menos_a_dB")
                out[f"{cel}|{pred}|{pop}"] = {"n_sorteios": n, "ep_mc_Err_dB": ep, "b_menos_c_dB": bmc, "b_menos_a_dB": bma,
                                              "b_menos_c_maior_que_2ep": (abs(bmc) > 2 * ep) if (ep and bmc is not None) else None,
                                              "b_menos_a_maior_que_2ep": (abs(bma) > 2 * ep) if (ep and bma is not None) else None,
                                              "agregados_iguais_ao_2.3": iguais}
    return {"conferencia_agregados_ok": ok_all, "por_celula_preditor_populacao": out}


def main():
    res = {"criterio": "criterios/criterio_resposta_cego_v3-10.json"}
    res.update(f1_f2(json.load(open(PARC))))
    res["F5"] = f5()
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("F1",)}, indent=1))
    print("F2 medianas:", {k: round(v, 3) for k, v in res["F2"].items() if k.startswith("mediana")}, "max dif identidade:", res["F2"]["max_dif_identidade_dB_todas"])
    f = res["F5"]
    if "conferencia_agregados_ok" in f:
        print("F5 conferencia:", f["conferencia_agregados_ok"])
        for k, v in f["por_celula_preditor_populacao"].items():
            if "|validos" in k or "|todos" in k:
                print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    else:
        print("F5:", f)
    print("gravado", OUT)


if __name__ == "__main__":
    sys.exit(main())
