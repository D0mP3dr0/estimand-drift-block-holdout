#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""3.7b (27/09/2026): resumo por célula da decomposição F2 de fase3/3.7_resposta_cego.json (sem recomputar).
Saída: _v3_2026-09-25/fase3/3.7b_decomposicao_por_celula.json"""
import json
from pathlib import Path
B = Path(__file__).resolve().parent.parent / "_v3_2026-09-25"
f2 = json.load(open(B / "fase3/3.7_resposta_cego.json"))["F2"]
pc = f2["por_celula"]
comp = {k: v["frac_composicao"] for k, v in pc.items()}
res = {"n_celulas": len(pc),
       "n_celulas_composicao_acima_de_meio": sum(c > 0.5 for c in comp.values()),
       "composicao_nas_demais": sorted(round(c, 3) for c in comp.values() if c <= 0.5),
       "composicao_min_max": [min(comp.values()), max(comp.values())],
       "celulas_composicao_acima_de_1": {k: round(c, 3) for k, c in comp.items() if c > 1},
       "celulas_resto_negativo": {k: round(v["frac_resto"], 3) for k, v in pc.items() if v["frac_resto"] < 0},
       "soma_das_tres_medianas": f2["mediana_frac_composicao"] + f2["mediana_frac_condicional"] + f2["mediana_frac_resto"],
       "n_sorteios_usados_min_max": [min(v["n_sorteios"] for v in pc.values()), max(v["n_sorteios"] for v in pc.values())]}
(B / "fase3/3.7b_decomposicao_por_celula.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1))
