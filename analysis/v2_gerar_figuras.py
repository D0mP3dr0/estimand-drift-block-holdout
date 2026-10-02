"""Gera F2, F3, F4 (draft_v2) a partir dos artefatos carimbados do fio
2026-09-24_gnn_rf_artigo2_mathematics_r2. Le SOMENTE os JSONs listados em
ENTRADAS; nao toca em /trabalho/ARPIA_RF fora desta pasta de scripts.
Protocolo: `/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/draft_v2/ESQUELETO_v2.md`
(secao "Figuras e tabelas"), aprovado pelo dono em
`/trabalho/HERMES/AGENTES/_DECISOES/DECISOES-2026-09-25-artigo2-tres-pontos.md`
(25/09/2026 14:40).

Saida: draft_v2/figures/{F2_k_val_k_te_seeds,F3_fracao_valida_teste_boxplot,
F4_delta_sentinela_por_celula_seed}.pdf/.png (300 dpi).

Estilo: matplotlib sem seaborn; fonte serif; sem titulo dentro da figura
(so eixos/legenda em ingles, com unidades); paleta do skill dataviz
(#2a78d6 validation/primary, #eb6834 test/secondary), como em
draft_B/figures/make_figures.py (leitura de referencia, nao alterado).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIO = "/trabalho/HERMES/AGENTES/_FIOS/2026-09-24_gnn_rf_artigo2_mathematics_r2/artefatos"
OUT = os.path.abspath(os.path.join(HERE, "..", "draft_v2", "figures"))

ENTRADAS = {
    "F2_F3": os.path.join(FIO, "matematica-estatistica-do-claim_r3_fracao_valida_val_treino.json"),
    "F4_linhas": os.path.join(FIO, "E3_linhas_v2.json"),
    "F4_agregado": os.path.join(FIO, "E3_agregado_v2.json"),
}

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    # roadmap v3, teste 0.2: fontes embutidas como Type 42 (TrueType), nao
    # Type 3, no PDF/PS de saida -- exigencia editorial MDPI/leitor de PDF.
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- F2
def gerar_f2():
    d = load(ENTRADAS["F2_F3"])
    k_val = {int(k): v for k, v in d["k_val_por_seed"].items()}
    k_te = {int(k): v for k, v in d["k_te_por_seed"].items()}
    seed42_val, seed42_te = k_val[42], k_te[42]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    bins_val = np.arange(min(k_val.values()) - 0.5, max(k_val.values()) + 1.5, 1)
    bins_te = np.arange(min(k_te.values()) - 0.5, max(k_te.values()) + 1.5, 1)
    axes[0].hist(list(k_val.values()), bins=bins_val, color=BLUE, alpha=0.75, edgecolor="white")
    axes[0].axvline(seed42_val, color=INK, lw=1.4, ls="--", label=f"seed 42 (n={seed42_val})")
    axes[0].set_xlabel("Number of cells with validation\npoorer than training (out of 16)")
    axes[0].set_ylabel("count (of 20 split seeds)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")

    axes[1].hist(list(k_te.values()), bins=bins_te, color=ORANGE, alpha=0.75, edgecolor="white")
    axes[1].axvline(seed42_te, color=INK, lw=1.4, ls="--", label=f"seed 42 (n={seed42_te})")
    axes[1].set_xlabel("Number of cells with test\nricher than training (out of 16)")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")

    for ax in axes:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "F2_k_val_k_te_seeds")


# --------------------------------------------------------------- F3
def gerar_f3():
    d = load(ENTRADAS["F2_F3"])
    resultados = d["resultados"]
    por_celula = {}
    for r in resultados:
        if r.get("g_km") != 10.0 or r.get("b_km") != 2.0:
            continue
        cel = f"{r['cidade']}_{r['Q']}"
        por_celula.setdefault(cel, []).append(r["frac_valida"]["test"])

    celulas = sorted(por_celula.keys())
    dados = [por_celula[c] for c in celulas]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    bp = ax.boxplot(dados, tick_labels=celulas, patch_artist=True, widths=0.55,
                     medianprops=dict(color=INK, lw=1.3),
                     boxprops=dict(facecolor=BLUE, alpha=0.35, edgecolor=INK2),
                     whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                     flierprops=dict(marker="o", ms=3, mfc=ORANGE, mec="none", alpha=0.8))
    ax.set_ylabel("Valid-node fraction of the test partition")
    ax.set_xlabel("City, quadrant")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "F3_fracao_valida_teste_boxplot")


# --------------------------------------------------------------- F4
def gerar_f4():
    linhas = load(ENTRADAS["F4_linhas"])
    agregado = load(ENTRADAS["F4_agregado"])
    marcadas = set()
    for seed, v in agregado["por_seed"].items():
        for cel in v["marcadas_pl"]:
            marcadas.add((cel, int(seed)))

    celulas = sorted(set(r["celula"] for r in linhas))
    x_of = {c: i for i, c in enumerate(celulas)}
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    xs_ok, ys_ok, xs_mk, ys_mk = [], [], [], []
    for r in linhas:
        x = x_of[r["celula"]] + rng.uniform(-0.15, 0.15)
        y = abs(r["delta_sent"])
        if (r["celula"], r["seed"]) in marcadas:
            xs_mk.append(x); ys_mk.append(y)
        else:
            xs_ok.append(x); ys_ok.append(y)

    ax.scatter(xs_ok, ys_ok, marker="o", s=26, color=BLUE, alpha=0.8,
               edgecolor="white", linewidth=0.4, label="GNN within PL threshold (n=%d)" % len(xs_ok))
    ax.scatter(xs_mk, ys_mk, marker="^", s=42, color=ORANGE, alpha=0.9,
               edgecolor=INK, linewidth=0.4, label="GNN PL out of threshold, marked (n=%d)" % len(xs_mk))
    ax.axhline(0.05, color=INK2, lw=1.1, ls=":", label="0.05 dB threshold")

    ax.set_xticks(range(len(celulas)))
    ax.set_xticklabels(celulas, rotation=45, ha="right")
    ax.set_ylabel(r"$|\Delta_{sentinel}|$ = |MAE$_{GNN}$ $-$ MAE$_{MLP}$| (dB)")
    ax.set_xlabel("cell (city, quadrant), 5 training seeds each")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    save(fig, "F4_delta_sentinela_por_celula_seed")


def main():
    os.makedirs(OUT, exist_ok=True)
    gerar_f2()
    gerar_f3()
    gerar_f4()
    print("gerado: F2, F3, F4 (.pdf + .png 300dpi) em", OUT)


if __name__ == "__main__":
    main()
