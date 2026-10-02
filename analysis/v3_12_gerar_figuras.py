"""v3-12: copia declarada de v3_gerar_figuras.py (sha256 do original:
ad88a54a2015887f10fd7efa5bf63c989651be4207ff7b244d03f384b44951a9; o original NAO foi alterado).
Protocolo: ROADMAP_v3-12_major_revision.md, itens A24 e D10 (aprovados pelo dono em 01/10/2026,
"execute td aceito a proposta do roadmap"); parecer de origem:
redacao/_pareceres_2026-10-01_t25/forum-figuras.md (B1, R1, R2, R3 e proposta 1 de figura nova).

Nenhum dado muda: F3 le o mesmo JSON de entrada, com o mesmo filtro e a mesma conta do original.
Mudancas em relacao ao original:
  F2  (histograma; REMOVIDA em v3-12b, ver abaixo) tinha rotulo "split seed 42 (n=..)" em loc="upper left".
  F3  rotulos do eixo x no vocabulario do corpo ("Bauru Q1"); figsize na largura final
      (394,36 pt), fontes >= 7 pt.
  F4  FIGURA NOVA "deriva do erro por celula" (F4_deriva_erro_por_celula): MAE do preditor
      constante nos nos validos do teste, 60 sorteios aleatorios, 16 celulas agrupadas por cidade;
      mediana, caixa interquartil, faixa minimo-maximo, marca da media de longo prazo e os
      sorteios individuais. Entrada: fase2/_v3_2.1_3.1_parcial_16x60rnd.json (dado por sorteio).
      Conferencia dos agregados contra tables_v3/T3_deriva_erro_v3.csv e o resumo de
      fase2/2.1_deriva_erro_baselines_16x60rnd.json -> F4_conferencia_T3.json (a figura nao e
      gravada se a conferencia falhar).
      (A F4 antiga do original, |delta sentinela| GNN x MLP, nao e regerada: nao faz parte do
      manuscrito v3-11g e o nome F4 passa a designar a figura nova.)
  todas  pdf.fonttype = 42 (ja no original) e PNG de conferencia a 300 dpi; sem bbox_inches="tight",
      para que a largura gravada seja exatamente a final (394,36 pt) e o fator de escala no
      LaTeX seja 1,0 com \\includegraphics[width=\\textwidth].
Grava tambem MANIFEST_figuras_v3-12.json (entrada -> sha256, script -> sha256, saida -> sha256);
rodar ANTES v3_12_fig1_block_erosion.py para a Fig. 1 entrar no manifesto.

v3-12b (acabamento; parecer redacao_v3-12/_pareceres_2026-10-01_v3-12a/forum-figuras.md, R-F2, R-F4 e registros):
  - saida em _v3_2026-09-25/redacao_v3-12/figures_v3-12/ (o manifesto grava essa pasta em pasta_saida);
  - vocabulario dos eixos igual nas duas figuras do estudo de caso: eixo x "cell" com as cidades como grupo
    (ticks Q1..Q4 e o nome da cidade sob um filete), eixo y da fracao "valid fraction of the test set";
  - a figura do histograma dos sorteios (antiga F2, F2_k_val_k_te_seeds) sai do script, do manifesto e da pasta de
    saida: nao e mais usada no manuscrito;
  - a conferencia da F4 aponta para redacao_v3-12/tables_v3-12/T3_deriva_erro_v3-12.csv (7 colunas; as usadas sao
    celula, n_sorteios, const_validos_mediana, min, max, dp_simples).
Nenhum dado muda.

v3-12d (ordem do chefe): figura NOVA F34_fracao_e_deriva (+ PNG): dois paineis empilhados, 394,36 pt de largura e
372 pt de altura (teto 380), eixo x compartilhado ("cell", cidades como grupo); (a) a fracao valida do teste por
celula (os mesmos dados e desenho da F3, 20 sorteios) e (b) a deriva do erro por celula (os mesmos dados e desenho
da F4, 60 sorteios); rotulos "(a)" e "(b)" no canto superior esquerdo de cada painel; uma so legenda de marcas, no
topo; fontes >= 7 pt. As figuras F3 e F4 isoladas continuam sendo geradas (ficam na pasta, fora do pacote). A
conferencia contra T3_deriva_erro_v3-12.csv vale para a F4 e para o painel (b) da F34 e e regravada.

Estilo: matplotlib sem seaborn; fonte serif; sem titulo dentro da figura; paleta do skill dataviz.
"""
import csv
import hashlib
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
MDPI = os.path.abspath(os.path.join(HERE, ".."))
V3 = os.path.join(MDPI, "_v3_2026-09-25")
OUT = os.path.join(V3, "redacao_v3-12", "figures_v3-12")
FIO = "/trabalho/HERMES/AGENTES/_FIOS/2026-09-24_gnn_rf_artigo2_mathematics_r2/artefatos"

ENTRADAS = {
    "F3": os.path.join(FIO, "matematica-estatistica-do-claim_r3_fracao_valida_val_treino.json"),
    "F4_por_sorteio": os.path.join(V3, "fase2", "_v3_2.1_3.1_parcial_16x60rnd.json"),
    "F4_conferencia_resumo_2.1": os.path.join(V3, "fase2", "2.1_deriva_erro_baselines_16x60rnd.json"),
    "F4_conferencia_T3_csv": os.path.join(V3, "redacao_v3-12", "tables_v3-12", "T3_deriva_erro_v3-12.csv"),
}
SCRIPTS = {
    "v3_12_gerar_figuras.py": os.path.join(HERE, "v3_12_gerar_figuras.py"),
    "v3_12_fig1_block_erosion.py": os.path.join(HERE, "v3_12_fig1_block_erosion.py"),
    "original:v3_gerar_figuras.py": os.path.join(HERE, "v3_gerar_figuras.py"),
    "original:v3_fig1_block_erosion.py": os.path.join(HERE, "v3_fig1_block_erosion.py"),
}
SAIDAS_NOMES = [
    "fig1_block_erosion.pdf", "fig1_block_erosion.png",
    "F3_fracao_valida_teste_boxplot.pdf", "F3_fracao_valida_teste_boxplot.png",
    "F4_deriva_erro_por_celula.pdf", "F4_deriva_erro_por_celula.png",
    "F34_fracao_e_deriva.pdf", "F34_fracao_e_deriva.png",
    "F4_conferencia_T3.json",
]

TEXTWIDTH_PT = 394.36          # main.log:1406 (parecer forum-figuras R2)
W_IN = TEXTWIDTH_PT / 72.0     # largura final das figuras, em polegadas

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QS = ["Q1", "Q2", "Q3", "Q4"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300)
    plt.close(fig)


def rotulo_celula(c):
    """'bauru_Q1' -> 'Bauru Q1' (vocabulario do corpo e das Tabs. 3 e 4)."""
    cid, q = c.split("_")
    return f"{cid.capitalize()} {q}"


def posicoes_celulas():
    """x de cada celula: 4 por cidade (Q1..Q4), com folga entre cidades; grupo[cid] = (x_min, x_max)."""
    pos, grupo, x = {}, {}, 0.0
    for cid in CIDADES:
        xs = []
        for q in QS:
            pos[f"{cid}_{q}"] = x
            xs.append(x)
            x += 1.0
        grupo[cid] = (min(xs), max(xs))
        x += 0.9
    return pos, grupo, x - 0.9 - 1.0


def eixo_celulas(fig, ax, pos, grupo, x_ultimo):
    """eixo x 'cell' das figuras do estudo de caso: ticks Q1..Q4 e a cidade como grupo (filete e nome)."""
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([c.split("_")[1] for c in pos])
    ax.set_xlim(-0.7, x_ultimo + 0.7)
    ax.set_xlabel("cell", labelpad=14)
    base = offset_copy(ax.get_xaxis_transform(), fig=fig, y=-16, units="points")
    for cid, (a, b) in grupo.items():
        ax.plot([a - 0.3, b + 0.3], [0, 0], transform=base, color=INK2, lw=0.8, clip_on=False)
        ax.annotate(cid.capitalize(), xy=((a + b) / 2, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -18), textcoords="offset points", ha="center", va="top",
                    fontsize=7.5, color=INK)


# --------------------------------------------------------------- F3
def desenha_fracao(ax, pos):
    """boxplot da fracao valida do teste por celula (20 sorteios, g = 10 km, b = 2 km), nos x de `pos`."""
    d = load(ENTRADAS["F3"])
    por_celula = {}
    for r in d["resultados"]:
        if r.get("g_km") != 10.0 or r.get("b_km") != 2.0:
            continue
        por_celula.setdefault(f"{r['cidade']}_{r['Q']}", []).append(r["frac_valida"]["test"])
    celulas = sorted(por_celula.keys())
    ax.boxplot([por_celula[c] for c in celulas], positions=[pos[c] for c in celulas], patch_artist=True, widths=0.55,
               medianprops=dict(color=INK, lw=1.3),
               boxprops=dict(facecolor=BLUE, alpha=0.35, edgecolor=INK2),
               whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
               flierprops=dict(marker="o", ms=3, mfc=ORANGE, mec="none", alpha=0.8))
    ax.set_ylabel("valid fraction of the test set")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def gerar_f3():
    pos, grupo, x_ultimo = posicoes_celulas()
    fig, ax = plt.subplots(figsize=(W_IN, 2.9))
    desenha_fracao(ax, pos)
    eixo_celulas(fig, ax, pos, grupo, x_ultimo)
    fig.tight_layout()
    save(fig, "F3_fracao_valida_teste_boxplot")


# --------------------------------------------------------------- F4 (nova)
def maes_validos_por_celula():
    """MAE do constante nos nos validos do teste, um valor por sorteio utilizavel (mesmo filtro
    do 2.1: status ok e MAE finito; sorteio sem no valido no teste fica de fora)."""
    d = load(ENTRADAS["F4_por_sorteio"])
    out = {}
    for k, cel in d["celulas"].items():
        v = []
        for p in cel["por_sorteio"]:
            m = p["mae_constante_teste"]["validos"]
            if p["status"] == "ok" and m is not None and np.isfinite(m):
                v.append(float(m))
        out[k] = np.array(v)
    return out


def conferir_f4(maes):
    """Confere, celula a celula e nos agregados, contra T3_deriva_erro_v3.csv e o resumo do 2.1."""
    tab = {r["celula"]: r for r in csv.DictReader(open(ENTRADAS["F4_conferencia_T3_csv"], encoding="utf-8"))}
    resumo = load(ENTRADAS["F4_conferencia_resumo_2.1"])["resumo"]
    tol = 1e-9
    linhas, ok_total = [], True
    for k in sorted(maes):
        v, t = maes[k], tab[k]
        calc = dict(n=int(v.size), mediana=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                    dp_simples=float(np.std(v, ddof=1)))
        ref = dict(n=int(t["n_sorteios"]), mediana=float(t["const_validos_mediana"]), min=float(t["min"]),
                   max=float(t["max"]), dp_simples=float(t["dp_simples"]))
        dif = {c: abs(calc[c] - ref[c]) for c in calc}
        ok = all(dif[c] <= tol for c in dif)
        ok_total &= ok
        linhas.append(dict(celula=k, ok=ok, calculado=calc, T3_csv=ref, dif_abs_max=max(dif.values())))
    dp_medio = float(np.mean([np.std(maes[k], ddof=1) for k in maes]))
    medias = np.array([maes[k].mean() for k in sorted(maes)])
    dp_entre_celulas = float(np.std(medias, ddof=1))
    agreg = {
        "dp_medio_entre_sorteios_dB": dict(calculado=dp_medio, resumo_2_1=resumo["dp_medio_entre_sorteios_validos_dB_simples"],
                                           T3_tex_arredondado=8.017),
        "dp_entre_celulas_das_medias_dB": dict(calculado=dp_entre_celulas, resumo_2_1=resumo["dp_entre_celulas_das_medias_validos_dB"],
                                              T3_tex_arredondado=2.180, ddof=1),
    }
    for nome, a in agreg.items():
        a["ok"] = bool(abs(a["calculado"] - a["resumo_2_1"]) <= tol and round(a["calculado"], 3) == a["T3_tex_arredondado"])
        ok_total &= a["ok"]
    total_sorteios = int(sum(v.size for v in maes.values()))
    rel = dict(
        descricao="Conferencia dos agregados da F4 e do painel (b) da F34 (dado por sorteio em fase2/_v3_2.1_3.1_parcial_16x60rnd.json) "
                  "contra redacao_v3-12/tables_v3-12/T3_deriva_erro_v3-12.csv e o resumo de fase2/2.1_deriva_erro_baselines_16x60rnd.json",
        tolerancia_abs=tol, n_celulas=len(linhas), total_sorteios_utilizaveis=total_sorteios,
        tudo_confere=bool(ok_total), por_celula=linhas, agregados=agreg,
        medias_de_longo_prazo_dB={k: float(maes[k].mean()) for k in sorted(maes)},
        quartis_metodo="numpy.percentile, interpolacao linear (default)",
    )
    with open(os.path.join(OUT, "F4_conferencia_T3.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, indent=1, ensure_ascii=False)
    if not ok_total:
        raise SystemExit("F4: conferencia contra T3 FALHOU; figura nao gravada (ver F4_conferencia_T3.json)")
    return rel


def desenha_deriva(ax, maes, pos):
    """deriva do erro por celula (60 sorteios): sorteios, faixa min-max, caixa interquartil, mediana e media."""
    rng = np.random.default_rng(42)
    for k, v in maes.items():
        xc = pos[k]
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        lo, hi = float(v.min()), float(v.max())
        ax.scatter(xc + rng.uniform(-0.22, 0.22, v.size), v, s=2.5, color=INK2, alpha=0.35,
                   linewidths=0, zorder=2)
        ax.plot([xc, xc], [lo, hi], color=INK, lw=0.9, zorder=3)
        ax.plot([xc - 0.12, xc + 0.12], [lo, lo], color=INK, lw=0.9, zorder=3)
        ax.plot([xc - 0.12, xc + 0.12], [hi, hi], color=INK, lw=0.9, zorder=3)
        ax.add_patch(plt.Rectangle((xc - 0.3, q1), 0.6, q3 - q1, facecolor=BLUE, alpha=0.45,
                                   edgecolor=INK2, lw=0.7, zorder=4))
        ax.plot([xc - 0.3, xc + 0.3], [med, med], color=INK, lw=1.5, zorder=5)
        ax.scatter([xc], [v.mean()], marker="D", s=16, color=ORANGE, edgecolor=INK, linewidths=0.5, zorder=6)
    ax.set_ylim(0, 60)
    ax.set_ylabel("valid-node mean absolute error\nof the constant predictor (dB)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def gerar_f4(maes):
    pos, grupo, x_ultimo = posicoes_celulas()
    fig, ax = plt.subplots(figsize=(W_IN, 3.1))
    desenha_deriva(ax, maes, pos)
    eixo_celulas(fig, ax, pos, grupo, x_ultimo)
    handles = [
        Line2D([], [], color=INK, lw=1.5, label="median"),
        Patch(facecolor=BLUE, alpha=0.45, edgecolor=INK2, label="interquartile range"),
        Line2D([], [], color=INK, lw=0.9, marker="_", ms=6, label="minimum to maximum"),
        Line2D([], [], color=ORANGE, marker="D", ms=4, mec=INK, mew=0.5, ls="", label="mean over draws"),
        Line2D([], [], color=INK2, marker="o", ms=2.5, ls="", alpha=0.6, label="split draw"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=5, fontsize=7,
               columnspacing=0.9, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "F4_deriva_erro_por_celula")


ALTURA_F34_PT = 372.0   # teto de 380 pt (ordem do chefe, v3-12d)


def gerar_f34(maes):
    """F34: dois paineis empilhados com eixo x compartilhado ("cell", cidades como grupo):
    (a) fracao valida do teste (as mesmas caixas da F3, 20 sorteios); (b) deriva do erro (o mesmo desenho da F4, 60 sorteios).
    Uma so legenda de marcas, no topo."""
    pos, grupo, x_ultimo = posicoes_celulas()
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(W_IN, ALTURA_F34_PT / 72.0), sharex=True,
                                   gridspec_kw=dict(height_ratios=[0.85, 1.15]))
    desenha_fracao(axa, pos)
    desenha_deriva(axb, maes, pos)
    axa.tick_params(axis="x", labelbottom=False)
    eixo_celulas(fig, axb, pos, grupo, x_ultimo)
    for ax, rot in ((axa, "(a)"), (axb, "(b)")):
        ax.text(0.0, 1.03, rot, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold", color=INK)
    handles = [
        Line2D([], [], color=INK, lw=1.5, label="median"),
        Patch(facecolor=BLUE, alpha=0.45, edgecolor=INK2, label="interquartile range (IQR)"),
        Line2D([], [], color=INK2, lw=0.9, marker="_", ms=6, label="whiskers: 1.5 IQR (a), min to max (b)"),
        Line2D([], [], color=ORANGE, marker="o", ms=3, ls="", label="draw beyond whiskers (a)"),
        Line2D([], [], color=ORANGE, marker="D", ms=4, mec=INK, mew=0.5, ls="", label="mean over draws (b)"),
        Line2D([], [], color=INK2, marker="o", ms=2.5, ls="", alpha=0.6, label="split draw (b)"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, fontsize=7,
               columnspacing=1.2, handletextpad=0.4, labelspacing=0.3)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.885, bottom=0.115, hspace=0.20)
    save(fig, "F34_fracao_e_deriva")


# --------------------------------------------------------------- manifesto
def gerar_manifesto():
    man = {
        "protocolo": "ROADMAP_v3-12_major_revision.md, itens A24 e D10 (dono, 01/10/2026); acabamento v3-12b (parecer forum-figuras v3-12a)",
        "pasta_saida": OUT,
        "largura_final_pt": TEXTWIDTH_PT,
        "entradas_sha256": {k: dict(caminho=p, sha256=sha256(p)) for k, p in ENTRADAS.items()},
        "scripts_sha256": {k: dict(caminho=p, sha256=sha256(p)) for k, p in SCRIPTS.items()},
        "saidas_sha256": {n: sha256(os.path.join(OUT, n)) for n in SAIDAS_NOMES if os.path.exists(os.path.join(OUT, n))},
    }
    faltam = [n for n in SAIDAS_NOMES if not os.path.exists(os.path.join(OUT, n))]
    man["saidas_ausentes"] = faltam
    with open(os.path.join(OUT, "MANIFEST_figuras_v3-12.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    if faltam:
        print("ATENCAO: saidas ausentes no manifesto:", faltam, file=sys.stderr)


def main():
    os.makedirs(OUT, exist_ok=True)
    maes = maes_validos_por_celula()
    conferir_f4(maes)          # a figura nao e gravada se a conferencia contra a Tab. 3 falhar
    gerar_f3()
    gerar_f4(maes)
    gerar_f34(maes)
    gerar_manifesto()
    print("gerado: F3, F4, F34 + conferencia + manifesto em", OUT)


if __name__ == "__main__":
    main()
