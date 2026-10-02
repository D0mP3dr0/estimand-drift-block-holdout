#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""fismat_p3_fronteira.py -- forum-fisico-matematico, 2026-09-25 (D6/F2, prop:degree).
Pergunta: a 'perda residual no interior' do 1.7 (razao_interior ~0,998) e perda real no
interior ou e a fronteira da OUTRA face, classificada como interior porque o 1.7 define
fronteira = dist_EDT < ell_medio = (ell_x+ell_y)/2 e ell_x < ell_medio < ell_y (a primeira
fileira retida junto a uma face horizontal esta a ell_y > ell_medio do complemento)?
So aritmetica sobre fase1/1.7_grau_preservacao.json. Para cada (celula, sorteio, particao):
n_equiv_interior = n_interior*(grau_antes_int - grau_depois_int)/perda_por_no_de_face,
com perda_por_no_de_face = grau_antes_front - grau_depois_front (medida no proprio registro);
razao n_equiv_interior/n_fronteira (esperada ~ perimetro horizontal/vertical * ell_y/ell_x ~ 1)
e perda total prevista = (n_fronteira + n_equiv_interior)*perda_por_no / (n_retidos*grau_antes).
"""
import hashlib, json
from pathlib import Path
R = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
F = R / "fase1/1.7_grau_preservacao.json"
d = json.load(open(F))
rows = []
for cel in d["por_celula"]:
    for s in cel["sorteios"]:
        for part, p in s["particoes"].items():
            gai, gdi = p["grau_antes_interior"]["media"], p["grau_depois_interior"]["media"]
            gaf, gdf = p["grau_antes_fronteira"]["media"], p["grau_depois_fronteira"]["media"]
            gar, gdr = p["grau_antes_retidos"]["media"], p["grau_depois_retidos"]["media"]
            perda_no = gaf - gdf
            n_eq = p["n_interior"] * (gai - gdi) / perda_no
            rows.append({"celula": f'{cel["cidade"]}_{cel["q"]}', "seed": s["split_seed"], "particao": part,
                         "ell_x_m": cel["ell_x_m"], "ell_y_m": cel["ell_y_m"], "ell_medio_m": cel["ell_medio_m"],
                         "grau_medio_full": cel["grau_medio_full"], "perda_por_no_de_face": perda_no,
                         "n_fronteira": p["n_fronteira"], "n_equiv_interior": n_eq, "razao_equiv_int_sobre_front": n_eq / p["n_fronteira"],
                         "perda_rel_total_medida": 1 - gdr / gar,
                         "perda_rel_so_fronteira_1.7": p["n_fronteira"] * perda_no / (p["n_retidos"] * gar),
                         "perda_rel_residual_interior": p["n_interior"] * (gai - gdi) / (p["n_retidos"] * gar)})
def faixa(k, part=None):
    v = [r[k] for r in rows if part is None or r["particao"] == part]; return [min(v), max(v)]
out = {"script": __file__, "sha256_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "insumo": {str(F): hashlib.sha256(F.read_bytes()).hexdigest()},
       "ell_x_lt_ell_medio_lt_ell_y_em_todas": all(r["ell_x_m"] < r["ell_medio_m"] < r["ell_y_m"] for r in rows),
       "faixas": {k: {pt: faixa(k, pt) for pt in ("train", "val", "test")} for k in
                  ("perda_por_no_de_face", "razao_equiv_int_sobre_front", "perda_rel_total_medida", "perda_rel_so_fronteira_1.7", "perda_rel_residual_interior")},
       "grau_medio_full": faixa("grau_medio_full"), "registros": rows}
json.dump(out, open(R / "fase2/fismat_p3_fronteira.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("ell_x_lt_ell_medio_lt_ell_y_em_todas", "faixas", "grau_medio_full")}, indent=1))
