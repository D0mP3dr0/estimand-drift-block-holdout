"""Gera T2, T4, T5 (draft_v2) a partir dos artefatos carimbados do fio
2026-09-24_gnn_rf_artigo2_mathematics_r2. Le SOMENTE os JSONs listados em
ENTRADAS; nao toca em /trabalho/ARPIA_RF fora desta pasta de scripts.
Protocolo: `/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/draft_v2/ESQUELETO_v2.md`
(secao "Figuras e tabelas"), aprovado pelo dono em
`/trabalho/HERMES/AGENTES/_DECISOES/DECISOES-2026-09-25-artigo2-tres-pontos.md`
(25/09/2026 14:40).

Saida: draft_v2/tables/{T2_retencao,T4_e3_por_celula,T5_constante_teste}.tex/.csv

Regras de numero: dB com 3 casas decimais; fracoes (pi, retencao, razoes,
erros relativos) com 3 algarismos significativos. Nenhuma coluna VEDADA:
T4 sem MAE cobertos/todos nem delta_cob_*; T5 sem correlacao com a
distancia (r2_teto).
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIO = "/trabalho/HERMES/AGENTES/_FIOS/2026-09-24_gnn_rf_artigo2_mathematics_r2/artefatos"
OUT = os.path.abspath(os.path.join(HERE, "..", "draft_v2", "tables"))

# Diretorio dos 80 run JSON originais (dados/treinos_c1), READ-ONLY, mesmo
# TREINOS_DIR de scripts/varredura_split_geometria.py. Usado so por gerar_t3
# para ler subgrafos.{train,test}.n_arestas_ter_ter / n_terrain (grau medio
# terra-terra); nenhum tensor .pt e aberto.
RUNS_DIR_T3 = (
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/treinos_c1"
)

ENTRADAS = {
    "T2": os.path.join(FIO, "matematica-equacoes_r3_mc_g5_tres_cidades.json"),
    "T4_linhas": os.path.join(FIO, "E3_linhas_v2.json"),
    "T4_agregado": os.path.join(FIO, "E3_agregado_v2.json"),
    "T5": os.path.join(FIO, "dados-contrato_r3_alvo_pos_correcao_16celulas.json"),
    # T3 (tab:drift): fracao valida treino/validacao vem do artefato R3-a
    # (varredura_fracao_valida_r3.py), ja com portao de reproducao <=1e-3
    # contra os run JSON declarados (seed 42, 16 celulas g10b2); blocos
    # efetivos de teste vem de dados-vazamento-espacial_blocos_efetivos_20celulas.json
    # (script scripts/blocos_efetivos_20celulas.py, reusado por import
    # abaixo, sem editar); razao de grau (teste/treino) e calculada aqui
    # direto do campo subgrafos.{train,test} de cada run JSON (mesma formula
    # de fm_drift_grafo.py: n_arestas_ter_ter / n_terrain, ja contra-auditada
    # em rerun_deriva_grafo_mdpi.json).
    "T3_fracao_valida_r3a": os.path.join(FIO, "matematica-estatistica-do-claim_r3_fracao_valida_val_treino.json"),
    "T3_blocos_efetivos": os.path.join(FIO, "dados-vazamento-espacial_blocos_efetivos_20celulas.json"),
}


def sig3(x):
    """3 algarismos significativos, para fracoes/razoes/erros relativos."""
    if x == 0:
        return "0.00"
    from math import floor, log10
    d = 2 - int(floor(log10(abs(x))))
    d = max(d, 0)
    return f"{round(x, d):.{d}f}"


def db3(x):
    """3 casas decimais, para grandezas em dB."""
    return f"{x:.3f}"


def pct3(x):
    """fracao -> percentual com 3 algarismos significativos do percentual."""
    return sig3(x * 100.0)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tex_escape(s):
    return str(s).replace("_", r"\_")


CIDADES_ORDEM = ["bauru", "campinas", "lins", "sorocaba"]
CIDADE_CAP = {"bauru": "Bauru", "campinas": "Campinas", "lins": "Lins", "sorocaba": "Sorocaba"}


# --------------------------------------------------------------- T2
def gerar_t2():
    d = load(ENTRADAS["T2"])
    pc = d["por_config"]
    linhas = []
    for cfg_key, g_label in (("g10b2_controle", "10"), ("g5b2", "5")):
        cfg = pc[cfg_key]
        for cidade in CIDADES_ORDEM:
            c = cfg["por_cidade"][cidade]
            for part_label, part_key, campo_medio in (
                ("val", "va", c["campo_medio_va"]),
                ("test", "te", c["campo_medio_te"]),
            ):
                mc = c["MC_C"]
                mc_media = mc[f"ret_{part_key}_media"]
                mc_p5 = mc[f"ret_{part_key}_p5"]
                mc_p95 = mc[f"ret_{part_key}_p95"]
                nodal = c["comparador_nodal_C1"]["val" if part_label == "val" else "test"]["nodal_media_C1"]
                erro_nodal = c["erro_lei_vs_nodal"][part_key]
                linhas.append({
                    "cidade": cidade, "g": g_label, "particao": part_label,
                    "lei": campo_medio, "mc_media": mc_media, "mc_p5": mc_p5,
                    "mc_p95": mc_p95, "nodal": nodal, "erro_lei_vs_nodal_pct": erro_nodal * 100.0,
                })
    csv_path = os.path.join(OUT, "T2_retencao.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["city", "g_km", "partition", "law", "mc_mean", "mc_p5", "mc_p95", "nodal_mean", "law_vs_nodal_pct"])
        for r in linhas:
            w.writerow([
                CIDADE_CAP[r["cidade"]], r["g"], r["particao"], r["lei"], r["mc_media"],
                r["mc_p5"], r["mc_p95"], r["nodal"], r["erro_lei_vs_nodal_pct"],
            ])

    tex = []
    tex.append(r"% fonte: matematica-equacoes_r3_mc_g5_tres_cidades.json por_config.<g10b2_controle|g5b2>.por_cidade.<cidade>.campo_medio_{va,te} (coluna 'law')")
    tex.append(r"% fonte: matematica-equacoes_r3_mc_g5_tres_cidades.json por_config.<cfg>.por_cidade.<cidade>.MC_C.ret_{va,te}_{media,p5,p95} (coluna 'MC mean [P5, P95]')")
    tex.append(r"% fonte: matematica-equacoes_r3_mc_g5_tres_cidades.json por_config.<cfg>.por_cidade.<cidade>.comparador_nodal_C1.{val,test}.nodal_media_C1 (coluna 'nodal mean (20 draws)')")
    tex.append(r"% fonte: matematica-equacoes_r3_mc_g5_tres_cidades.json por_config.<cfg>.por_cidade.<cidade>.erro_lei_vs_nodal.{va,te} (coluna 'law vs nodal (%)')")
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\caption{Field-mean retention law against lattice Monte Carlo and the node-level comparator (20 split seeds), $g=10$~km and $g=5$~km, $b=2$~km, four cities.}")
    tex.append(r"\label{tab:t2_retencao}")
    tex.append(r"\begin{tabular}{llcccccc}")
    tex.append(r"\toprule")
    tex.append(r"City & $g$ (km) & Partition & Law & MC mean [P5, P95] & Nodal mean (20 draws) & Law vs.\ nodal (\%) \\")
    tex.append(r"\midrule")
    for r in linhas:
        tex.append(
            f"{tex_escape(CIDADE_CAP[r['cidade']])} & {r['g']} & {r['particao']} & {sig3(r['lei'])} & "
            f"{sig3(r['mc_media'])} [{sig3(r['mc_p5'])}, {sig3(r['mc_p95'])}] & {sig3(r['nodal'])} & "
            f"{sig3(r['erro_lei_vs_nodal_pct'])} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(os.path.join(OUT, "T2_retencao.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    return [csv_path, os.path.join(OUT, "T2_retencao.tex")]


# --------------------------------------------------------------- T4
def gerar_t4():
    linhas_json = load(ENTRADAS["T4_linhas"])
    agregado = load(ENTRADAS["T4_agregado"])
    pi_celula_inteira = load(ENTRADAS["T5"])["por_celula"]
    por_celula = {}
    for r in linhas_json:
        cel = r["celula"]
        por_celula.setdefault(cel, []).append(r)

    def mediana(vals):
        s = sorted(vals)
        n = len(s)
        m = n // 2
        return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])

    marcadas_total = set()
    for seed, v in agregado["por_seed"].items():
        for cel in v["marcadas_pl"]:
            marcadas_total.add((cel, int(seed)))

    linhas_saida = []
    for cel in sorted(por_celula.keys()):
        regs = por_celula[cel]
        pi_cel = pi_celula_inteira[cel]["pi_celula"]
        mlp = [r["mae_sent_mlp"] for r in regs]
        gnn = [r["mae_sent_gnn"] for r in regs]
        razao = [r["razao_sent_gnn_mlp"] for r in regs]
        delta_abs = [abs(r["delta_sent"]) for r in regs]
        linhas_saida.append({
            "celula": cel,
            "pi_celula": pi_cel,
            "mae_mlp_mediana": mediana(mlp), "mae_mlp_max": max(mlp),
            "mae_gnn_mediana": mediana(gnn), "mae_gnn_max": max(gnn),
            "razao_gnn_mlp_mediana": mediana(razao),
            "delta_abs_media": sum(delta_abs) / len(delta_abs),
            "n_seeds": len(regs),
            "n_marcadas_pl": sum(1 for r in regs if (cel, r["seed"]) in marcadas_total),
        })

    csv_path = os.path.join(OUT, "T4_e3_por_celula.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas_saida[0].keys()))
        w.writeheader()
        for r in linhas_saida:
            w.writerow(r)

    tex = []
    tex.append(r"% fonte: dados-contrato_r3_alvo_pos_correcao_16celulas.json por_celula.<celula>.pi_celula (coluna 'pi cell') -- pi da CELULA INTEIRA, nao da particao de teste")
    tex.append(r"% fonte: E3_linhas_v2.json[*].mae_sent_mlp, mediana e maximo sobre 5 seeds por celula (coluna 'MAE sentinel MLP')")
    tex.append(r"% fonte: E3_linhas_v2.json[*].mae_sent_gnn, mediana e maximo sobre 5 seeds por celula (coluna 'MAE sentinel GNN')")
    tex.append(r"% fonte: E3_linhas_v2.json[*].razao_sent_gnn_mlp, mediana sobre 5 seeds (coluna 'ratio GNN/MLP')")
    tex.append(r"% fonte: E3_linhas_v2.json[*].delta_sent, media de |delta_sent| sobre 5 seeds (coluna '|Delta| mean')")
    tex.append(r"% nota: exclui MAE cobertos/todos e delta_cob_* (regra de nao duplicacao, Adendo 2/4); marcadas_pl de E3_agregado_v2.json por_seed.<seed>.marcadas_pl so contam execucoes, nao excluem dado")
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\caption{E3 sentinel-population MAE by cell, five training seeds (42--46). $\pi$ is the sentinel fraction of the whole cell (not the test partition). Ratio and $|\Delta|$ computed per seed then summarized.}")
    tex.append(r"\label{tab:t4_e3_por_celula}")
    tex.append(r"\begin{tabular}{lcccccc}")
    tex.append(r"\toprule")
    tex.append(r"Cell & $\pi$ (cell) & MAE MLP med.\ (max) (dB) & MAE GNN med.\ (max) (dB) & Ratio GNN/MLP (med.) & $|\Delta|$ mean (dB) \\")
    tex.append(r"\midrule")
    for r in linhas_saida:
        tex.append(
            f"{tex_escape(r['celula'])} & {sig3(r['pi_celula'])} & "
            f"{db3(r['mae_mlp_mediana'])} ({db3(r['mae_mlp_max'])}) & "
            f"{db3(r['mae_gnn_mediana'])} ({db3(r['mae_gnn_max'])}) & "
            f"{sig3(r['razao_gnn_mlp_mediana'])} & {db3(r['delta_abs_media'])} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\multicolumn{6}{l}{\footnotesize Note: cell labels follow the convention \texttt{city\_Q\{1--4\}} (e.g.\ \texttt{bauru\_Q1} = Bauru, quadrant 1).} \\")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(os.path.join(OUT, "T4_e3_por_celula.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    return [csv_path, os.path.join(OUT, "T4_e3_por_celula.tex")]


# --------------------------------------------------------------- T5
def gerar_t5():
    d = load(ENTRADAS["T5"])
    pc = d["por_celula"]
    linhas = []
    for cel in sorted(pc.keys()):
        c = pc[cel]
        linhas.append({
            "celula": cel,
            "pi_celula": c["pi_celula"],
            "pi_treino": c["pi_treino"],
            "pi_val": c["pi_val"],
            "mae_constante_celula_inteira_db": c["mae_constante_celula_inteira_db"],
        })
    csv_path = os.path.join(OUT, "T5_constante_teste.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["celula", "pi_celula", "pi_treino", "pi_val", "mae_constante_celula_inteira_db"])
        w.writeheader()
        for r in linhas:
            w.writerow(r)

    tex = []
    tex.append(r"% fonte: dados-contrato_r3_alvo_pos_correcao_16celulas.json por_celula.<celula>.pi_celula (coluna 'pi cell')")
    tex.append(r"% fonte: dados-contrato_r3_alvo_pos_correcao_16celulas.json por_celula.<celula>.pi_treino (coluna 'pi train')")
    tex.append(r"% fonte: dados-contrato_r3_alvo_pos_correcao_16celulas.json por_celula.<celula>.pi_val (coluna 'pi val')")
    tex.append(r"% fonte: dados-contrato_r3_alvo_pos_correcao_16celulas.json por_celula.<celula>.mae_constante_celula_inteira_db (coluna 'constant predictor MAE, whole cell')")
    tex.append(r"% nota: NAO inclui correlacao com a distancia (r2_teto), conforme protocolo; colunas de particao de teste removidas em 25/09/2026 (regenerado 25/09 15:35: colunas de teste removidas por nao duplicacao com a tabela da IEEE)")
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\caption{Sentinel fraction $\pi$ (whole cell, training and validation partitions) and MAE of the constant (median) predictor on the whole cell, 16 cells.}")
    tex.append(r"\label{tab:t5_constante_teste}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r"Cell & $\pi$ cell & $\pi$ train & $\pi$ val & Constant predictor MAE, whole cell (dB) \\")
    tex.append(r"\midrule")
    for r in linhas:
        tex.append(
            f"{tex_escape(r['celula'])} & {sig3(r['pi_celula'])} & {sig3(r['pi_treino'])} & {sig3(r['pi_val'])} & {db3(r['mae_constante_celula_inteira_db'])} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\multicolumn{5}{l}{\footnotesize Note: cell labels follow the convention \texttt{city\_Q\{1--4\}} (e.g.\ \texttt{bauru\_Q1} = Bauru, quadrant 1).} \\")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(os.path.join(OUT, "T5_constante_teste.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    return [csv_path, os.path.join(OUT, "T5_constante_teste.tex")]


# --------------------------------------------------------------- T3
def gerar_t3():
    """tab:drift (roadmap v3, teste 0.2): por celula (16, seed 42, g10b2):
    razao de grau medio terra-terra teste/treino, fracao valida de TREINO e
    de VALIDACAO (a coluna de teste fica FORA -- escolha editorial do
    esqueleto v2, ja que 1 - fracao_valida_teste == "Covered fr." da tabela
    da IEEE, VEDADA por nao-duplicacao) e blocos de teste efetivos (de 20
    declarados). Fracao valida treino/validacao: artefato R3-a
    (varredura_fracao_valida_r3.py), reconstrucao EDT ja com portao <=1e-3
    contra os run JSON declarados (seed 42, 48 comparacoes, 16 celulas x 3
    papeis). Blocos efetivos de teste: dados-vazamento-espacial_blocos_efetivos_20celulas.json
    (script blocos_efetivos_20celulas.py, filtrado as 16 celulas g10b2).
    Razao de grau: calculada aqui direto do proprio run JSON
    (subgrafos.{train,test}.n_arestas_ter_ter / n_terrain), mesma formula
    contra-auditada em rerun_deriva_grafo_mdpi.json (forum-eng-dados R2)."""
    fv = load(ENTRADAS["T3_fracao_valida_r3a"])
    fv_por_celula = {
        f"{r['cidade']}_{r['Q']}": r
        for r in fv["resultados"]
        if r["split_seed"] == 42 and r["g_km"] == 10.0 and r["b_km"] == 2.0
    }
    assert len(fv_por_celula) == 16, f"esperava 16 celulas g10b2 seed42 em T3_fracao_valida_r3a, achou {len(fv_por_celula)}"

    blocos = load(ENTRADAS["T3_blocos_efetivos"])
    blocos_por_celula = {}
    for c in blocos["celulas"]:
        if not c["run_label"].endswith("_g10b2"):
            continue  # exclui as 4 celulas bauru g5b2 (fora do escopo de T3)
        partes = c["run_label"].split("_")  # c0c1cf_<cidade>_s42_<Q>_g10b2
        cidade, q = partes[1], partes[3]
        blocos_por_celula[f"{cidade}_{q}"] = c["blocos_test"]
    assert len(blocos_por_celula) == 16, f"esperava 16 celulas g10b2 em T3_blocos_efetivos, achou {len(blocos_por_celula)}"

    def grau_ratio(cidade, q):
        run_path = os.path.join(RUNS_DIR_T3, f"run_c0c1cf_{cidade}_s42_{q}_g10b2.json")
        d = load(run_path)
        sub = d["subgrafos"]
        grau_train = sub["train"]["n_arestas_ter_ter"] / sub["train"]["n_terrain"]
        grau_test = sub["test"]["n_arestas_ter_ter"] / sub["test"]["n_terrain"]
        return grau_test / grau_train

    linhas = []
    for cel in sorted(fv_por_celula.keys()):
        cidade, q = cel.split("_")
        r = fv_por_celula[cel]
        bt = blocos_por_celula[cel]
        linhas.append({
            "celula": cel,
            "razao_grau_teste_treino": grau_ratio(cidade, q),
            "frac_valida_treino": r["frac_valida"]["train"],
            "frac_valida_validacao": r["frac_valida"]["val"],
            "blocos_teste_efetivos": bt["n_efetivos_total"],
            "blocos_teste_declarados": bt["n_declarados"],
        })

    csv_path = os.path.join(OUT, "T3_deriva_por_celula.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        for r in linhas:
            w.writerow(r)

    tex = []
    tex.append(r"% fonte: matematica-estatistica-do-claim_r3_fracao_valida_val_treino.json resultados[*] (split_seed=42, g_km=10, b_km=2).frac_valida.{train,val} (colunas 'Valid frac. train'/'Valid frac. val')")
    tex.append(r"% fonte: dados-vazamento-espacial_blocos_efetivos_20celulas.json celulas[*].blocos_test.{n_efetivos_total,n_declarados}, filtrado a run_label terminando em _g10b2 (coluna 'Effective test blocks')")
    tex.append(r"% fonte: run_c0c1cf_<cidade>_s42_<Q>_g10b2.json subgrafos.{train,test}.n_arestas_ter_ter / n_terrain (coluna 'Degree ratio test/train'), formula identica a rerun_deriva_grafo_mdpi.json (forum-eng-dados R2)")
    tex.append(r"% nota: SEM coluna de fracao valida de teste (1 - Covered fr. da IEEE, VEDADO por nao duplicacao; escolha editorial do esqueleto v2); SEM densidade de antena (mesma regra)")
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\caption{Structure and composition of the retained partitions at $g = 10$~km, $b = 2$~km, split seed $42$: ratio of mean terrain degree between test and training, valid-target fraction of the training and validation partitions, and number of effective test blocks (of 20 declared), by cell.}")
    tex.append(r"\label{tab:drift}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r"Cell & Degree ratio test/train & Valid frac.\ train & Valid frac.\ val & Effective test blocks \\")
    tex.append(r"\midrule")
    for r in linhas:
        tex.append(
            f"{tex_escape(r['celula'])} & {sig3(r['razao_grau_teste_treino'])} & "
            f"{sig3(r['frac_valida_treino'])} & {sig3(r['frac_valida_validacao'])} & "
            f"{r['blocos_teste_efetivos']}/{r['blocos_teste_declarados']} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\multicolumn{5}{l}{\footnotesize Note: cell labels follow the convention \texttt{city\_Q\{1--4\}} (e.g.\ \texttt{bauru\_Q1} = Bauru, quadrant 1).} \\")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(os.path.join(OUT, "T3_deriva_por_celula.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    return [csv_path, os.path.join(OUT, "T3_deriva_por_celula.tex")]


def main():
    os.makedirs(OUT, exist_ok=True)
    saidas = []
    saidas += gerar_t2()
    saidas += gerar_t3()
    saidas += gerar_t4()
    saidas += gerar_t5()
    print("gerado:")
    for s in saidas:
        print(" -", s)


if __name__ == "__main__":
    main()
