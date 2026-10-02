"""forum-eng-ia 26/09: larguras do MLP para paridade com o numero EFETIVO do GNN.
Convencao da casa (train_mlp_c0_spatial.py docstring [M1]): Linear mn+n, BN afim 2n,
encoder 18->w1->w2->w3->w4->256, decoder 99.363 (inclui 30 de cabecas mortas, simetricas).
Alvo: efetivos_MLP (= total - 30) == efetivos_GNN (forum-eng-ia_contagem_efetiva.json)."""
import json
from pathlib import Path
here = Path(__file__).resolve().parent
c = json.load(open(here / "forum-eng-ia_contagem_efetiva.json"))
gnn_tot = c["fanout_1_salto_producao"]["n_params_total"]
gnn_ef = c["fanout_1_salto_producao"]["n_params_efetivos"]
gnn_ef_2s = c["fanout_com_ET_TA_povoado"]["n_params_efetivos"]
DEC, MORTOS_DEC = 99_363, 30
def enc(ws):
    d = [18] + list(ws) + [256]
    return sum(m * n + 3 * n for m, n in zip(d, d[1:]))
assert enc([712, 712, 720, 688]) + DEC == 1_812_515
alvo_enc = gnn_ef + MORTOS_DEC - DEC
melhor = None
base = [712, 712, 720, 688]
for f in range(860, 920):
    ws0 = [round(b * f / 1000) for b in base]
    for d4 in range(-40, 41):
        ws = ws0[:3] + [ws0[3] + d4]
        # resolve w3 exato se possivel: enc linear em w3 dado o resto
        w1, w2, _, w4 = ws
        fixo = 18*w1 + 3*w1 + w1*w2 + 3*w2 + w4*256 + 3*w4 + 3*256
        coef = w2 + 3 + w4   # termos em w3: w2*w3 + 3*w3 + w3*w4
        resto = alvo_enc - fixo
        if resto % coef == 0:
            w3 = resto // coef
            cand = [w1, w2, w3, w4]
            dist = sum(abs(a - b * f / 1000) for a, b in zip(cand, base))
            if melhor is None or dist < melhor[0]:
                melhor = (dist, cand)
ws = melhor[1]
out = {"gnn_total": gnn_tot, "gnn_efetivos_fanout1": gnn_ef, "gnn_efetivos_se_ET_TA_povoado": gnn_ef_2s,
       "mlp_atual_total": 1_812_515, "mlp_atual_efetivos": 1_812_515 - MORTOS_DEC,
       "razao_mlp_atual_efetivos_sobre_gnn_efetivos": (1_812_515 - MORTOS_DEC) / gnn_ef,
       "frac_gnn_morta": (gnn_tot - gnn_ef) / gnn_tot,
       "mlp_novo_larguras": ws, "mlp_novo_total": enc(ws) + DEC,
       "mlp_novo_efetivos": enc(ws) + DEC - MORTOS_DEC,
       "delta_mlp_novo_efetivos_menos_gnn_efetivos": enc(ws) + DEC - MORTOS_DEC - gnn_ef}
json.dump(out, open(here / "forum-eng-ia_larguras_mlp_efetivo.json", "w"), indent=1)
print(json.dumps(out, indent=1))
