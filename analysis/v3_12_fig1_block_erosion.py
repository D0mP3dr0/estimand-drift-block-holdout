"""v3-12: copia declarada de v3_fig1_block_erosion.py (sha256 do original:
8cae02e108b193257c0847b07fa81e1cf2baef963d1db1c194890e916451a3a5; o original NAO foi alterado).
Protocolo: ROADMAP_v3-12_major_revision.md, item A24 (aprovado pelo dono em 01/10/2026,
"execute td aceito a proposta do roadmap"); parecer de origem: forum-figuras.md, R3.

Mudancas em relacao ao original (geometria, cores, composicao e figsize IDENTICOS):
  1. saida em _v3_2026-09-25/redacao/figures_v3-12/ (v3-12b: redacao_v3-12/figures_v3-12/) (pasta nova), PDF + PNG de conferencia;
  2. pdf.fonttype = 42 / ps.fonttype = 42 (o original saia em Type 3);
  3. texto sob cada painel: "(removed: ...)" -> "(trimmed by: ...)", coerente com o rodape
     "orange: area removed by the buffer" (a lista sao os vizinhos azuis que aparam o bloco,
     nao a area removida) e com o verbo do corpo ("trimming neighbours").
  4. (v3-12b, parecer forum-figuras R-F1) simbolo desenhado rho(S) com S comum -> rho(S) com S caligrafico (mathcal), o do corpo; saida em
     _v3_2026-09-25/redacao_v3-12/figures_v3-12/.
  5. (v3-12g) rotulos dos vizinhos: L, R, T_-, T -> W, E, S, N (sans-serif, west/east/south/north) e cantos pelo par de
     laterais, os simbolos de parts/parteB.tex (subsecao da area do bloco e eq:area); geometria e valores iguais.
  6. (v3-12h) os vizinhos passam a B_w, B_e, B_s, B_n (laterais) e B_ws, B_es, B_wn, B_en (diagonais), os simbolos finais de
     parts/parteB.tex (W, E, S, N colidiam com N, S_U e W da mediana); mapa por painel conferido contra a legenda atual:
     (a) B_w, B_n e a diagonal B_es; (b) B_s, B_w, B_e, B_n e a diagonal B_en.
Nenhum dado entra nesta figura (geometria sintetica com g = 10, b = 2).

Original: Figura 1 da v3 (27/09/2026): copia do trecho da Fig. 1 de draft_B/figures/make_figures.py,
so com os ROTULOS exibidos trocados para a notacao do texto.
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
HERE = "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/redacao_v3-12/figures_v3-12"
os.makedirs(HERE, exist_ok=True)
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e3"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

ROTULO = {"L": r"$B_{\mathsf{w}}$", "R": r"$B_{\mathsf{e}}$", "B": r"$B_{\mathsf{s}}$", "T": r"$B_{\mathsf{n}}$", "BL": r"$B_{\mathsf{ws}}$", "BR": r"$B_{\mathsf{es}}$", "TL": r"$B_{\mathsf{wn}}$", "TR": r"$B_{\mathsf{en}}$"}  # v3-12h: simbolos finais de parts/parteB.tex (blocos vizinhos B_w, B_e, B_s, B_n e diagonais B_ws, B_es, B_wn, B_en); as chaves internas L, R, B, T, BL.. seguem as do original
# ---------------------------------------------------------------- Figure 1
def draw_block(ax, S, title):
    g, b = 10.0, 2.0
    ax.set_aspect("equal"); ax.set_xlim(-g - 1, 2 * g + 1); ax.set_ylim(-g - 1, 2 * g + 1)
    ax.axis("off"); ax.set_title(title, fontsize=9, color=INK, loc="left")
    # neighbours (3x3 lattice)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                continue
            key = {(-1, 0): "L", (1, 0): "R", (0, -1): "B", (0, 1): "T",
                   (-1, -1): "BL", (1, -1): "BR", (-1, 1): "TL", (1, 1): "TR"}[(i, j)]
            trims = key in S
            ax.add_patch(Rectangle((i * g, j * g), g, g, facecolor=(BLUE if trims else "#f4f4f2"),
                                   alpha=0.25 if trims else 1.0, edgecolor=INK2, lw=0.6))
            ax.text(i * g + g / 2, j * g + g / 2, ROTULO[key], ha="center", va="center", fontsize=8,
                    color=INK if trims else INK2)
    # held-out block: full square, then removed regions in orange hatch
    ax.add_patch(Rectangle((0, 0), g, g, facecolor="white", edgecolor=INK, lw=1.0))
    removed_kw = dict(facecolor=ORANGE, alpha=0.30, edgecolor="none")
    if "L" in S: ax.add_patch(Rectangle((0, 0), b, g, **removed_kw))
    if "R" in S: ax.add_patch(Rectangle((g - b, 0), b, g, **removed_kw))
    if "B" in S: ax.add_patch(Rectangle((0, 0), g, b, **removed_kw))
    if "T" in S: ax.add_patch(Rectangle((0, g - b), g, b, **removed_kw))
    corners = {"BL": ((0, 0), 0, "L", "B"), "BR": ((g, 0), 90, "R", "B"),
               "TL": ((0, g), 270, "L", "T"), "TR": ((g, g), 180, "R", "T")}
    for c, (ctr, ang, s1, s2) in corners.items():
        if c in S and s1 not in S and s2 not in S:
            ax.add_patch(Wedge(ctr, b, ang, ang + 90, facecolor=ORANGE, alpha=0.30, edgecolor="none"))
    # retained core outline
    x0 = b if "L" in S else 0; x1 = g - b if "R" in S else g
    y0 = b if "B" in S else 0; y1 = g - b if "T" in S else g
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK, lw=1.2, ls="--"))
    # area label
    lat = sum(k in S for k in ("L", "R")); ver = sum(k in S for k in ("B", "T"))
    ncorner = sum(1 for c, (_, _, s1, s2) in corners.items() if c in S and s1 not in S and s2 not in S)
    area = (g - b * lat) * (g - b * ver) - math.pi * b * b / 4 * ncorner
    ax.text(g / 2, -g - 0.2, r"$\rho(\mathcal{S})=%.3f$  (trimmed by: %s)" % (area / g / g, ", ".join(ROTULO[k] for k in sorted(S))),
            ha="center", va="top", fontsize=8, color=INK)

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6))
draw_block(axes[0], {"L", "T", "BR"}, "(a) two sides + one isolated corner")
draw_block(axes[1], {"L", "R", "B", "T", "TR"}, "(b) all four sides")
fig.text(0.5, 0.005, "blue: neighbours that trim the block   |   orange: area removed by the buffer   |   dashed: retained core",
         ha="center", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(HERE, "fig1_block_erosion.pdf"))
fig.savefig(os.path.join(HERE, "fig1_block_erosion.png"), dpi=300)
plt.close(fig)
