#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""1.5b (27/09/2026): rejulga a esperança exata sem reposição contra a referência nodal de 200
sorteios nas DUAS escalas (1.8 em g=10 km, 1.8b em g=5 km). Não recomputa nada da retenção: lê
campo_medio (média de campo NO RETICULADO REAL, blocos de borda incluídos, lattice_retention de
fismat_sympy_eqR.py) e exato_sem_reposicao do 1.5. Saída: fase1/1.5b_exata_vs_nodal200.json"""
import json
from pathlib import Path
B = Path(__file__).resolve().parent.parent / "_v3_2026-09-25"
d15 = json.load(open(B / "fase1/1.5_retencao_exata_vs_mc.json"))["por_celula"]
ref = {"g10b2": json.load(open(B / "fase1/1.8_referencia_nodal_200seeds.json"))["por_celula"],
       "g5b2": json.load(open(B / "fase1/1.8b_referencia_nodal_200seeds_g5.json"))["por_celula"]}
linhas, n_red = [], 0
for k, v in d15.items():
    cel = f"{v['cidade']}_{v['Q']}"; cfg = v["config"]; r = ref[cfg][cel]
    for part, chave, nod in (("teste", "teste", "nodal_te_media"), ("val", "val", "nodal_va_media")):
        mf, ex, nd = v["campo_medio"][chave], v["exato_sem_reposicao"][chave], r[nod]
        e_mf, e_ex = abs(mf - nd) / nd, abs(ex - nd) / nd
        red = e_ex < e_mf; n_red += red
        linhas.append({"celula": cel, "config": cfg, "particao": part, "nodal200": nd, "campo_medio_reticulado": mf,
                       "exato_sem_reposicao": ex, "erro_rel_campo_medio": e_mf, "erro_rel_exato": e_ex,
                       "exato_reduz_erro": bool(red), "campo_medio_acima_do_nodal": mf > nd, "exato_abaixo_do_campo_medio": ex < mf})
res = {"n_comparacoes": len(linhas), "n_exato_reduz": n_red,
       "onde_reduz": sorted({(l["config"], l["particao"]) for l in linhas if l["exato_reduz_erro"]}),
       "n_reduz_por_config_particao": {f"{c}|{p}": sum(l["exato_reduz_erro"] for l in linhas if l["config"] == c and l["particao"] == p)
                                       for c in ("g10b2", "g5b2") for p in ("teste", "val")},
       "exato_sempre_abaixo_do_campo_medio": all(l["exato_abaixo_do_campo_medio"] for l in linhas),
       "n_campo_medio_acima_do_nodal": sum(l["campo_medio_acima_do_nodal"] for l in linhas),
       "objeto_comparado": "campo medio avaliado no reticulado real (blocos de borda incluidos), NAO a lei do bloco interior (Eq. R)"}
(B / "fase1/1.5b_exata_vs_nodal200.json").write_text(json.dumps({"resumo": res, "linhas": linhas}, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, default=list))
