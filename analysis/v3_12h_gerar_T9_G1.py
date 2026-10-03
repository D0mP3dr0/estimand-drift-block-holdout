#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""
v3-12h (03/10/2026): T9 -- deriva dos modelos treinados a 10 km (lote G1), 4 celulas x 2 modelos.

Protocolo: redacao_v3-12/DEMANDAS_REDATORES_v3-12h.md (item C-h1: tabela tables_v3-12/T9_deriva_modelo_G1_v3-12.tex, label
tab:drift_models_g10) e MATRIZ_DE_ACHADOS_v3-12h.md (linha 2), gravados em 03/10/2026 sob a ordem do dono de 03/10/2026
("pode executar td ... add esses dados ao texto"; registrada no cabecalho da matriz).
Criterio pre-registrado do lote: criterios/criterio_G1_modelos_g10.json + adendos 1-4; recalculo independente dos agregados em
gpu/G1_votos_bloco1|2|3/veredito.json (todos "confirmado").

Entradas (SO estas tres): gpu/G1/agregado_G1_v5_bloco1.json (Bauru Q1, Campinas Q1: 20 sorteios, semente 42),
gpu/G1/agregado_G1_v5_bloco2.json (bloco cruzado 3 sementes x 5 sorteios, so Q1: dp pooled entre sementes),
gpu/G1/agregado_G1_v8_bloco3.json (Bauru Q3, Campinas Q3).
Saida: T9_deriva_modelo_G1_v3-12.csv e .tex em tables_v3-12/ (ambiente table[H] com legenda e label; o invólucro \\tabcorpo do
manuscrito neutraliza table/caption/label, como no TN) e as entradas T9 em MANIFEST_tabelas_v3-12.json e
CONFERENCIAS_tabelas_v3-12.json.

MANIFEST e CONFERENCIAS sao regerados por inteiro por scripts/v3_12_gerar_tabelas.py (fora da pasta de trabalho, nao alterado
aqui). Este script ACRESCENTA as entradas T9 (chaves "T9_*" nas conferencias; entradas G1_* e bloco "T9" no manifesto) e e
idempotente; reexecutar o gerador principal apaga essas entradas, e basta reexecutar este script depois dele.

Nenhum numero digitado a mao: todo valor sai de campos nomeados dos tres JSON. As conferencias RELEEM cada valor por caminho
de texto ("celulas.<cel>...."), recalculam dp (ddof=1), razao, n e mediana de paridade a partir dos valores por sorteio, e
leem de volta o .csv e o .tex gravados.

Python: /trabalho/ambientes/s33_amb_virtual/.venv/bin/python (CPU).
"""
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

B = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
OUT = B / "redacao_v3-12" / "tables_v3-12"
G1 = B / "gpu" / "G1"
ENT = {
    "G1_bloco1": G1 / "agregado_G1_v5_bloco1.json",
    "G1_bloco2": G1 / "agregado_G1_v5_bloco2.json",
    "G1_bloco4": G1 / "agregado_G1_v10_bloco4.json",
    "G1_bloco3": G1 / "agregado_G1_v8_bloco3.json",
}
VEREDITOS = {
    "G1_veredito_bloco1": B / "gpu" / "G1_votos_bloco1" / "veredito.json",
    "G1_veredito_bloco2": B / "gpu" / "G1_votos_bloco2" / "veredito.json",
    "G1_veredito_bloco4": B / "gpu" / "G1_votos_bloco4" / "veredito.json",
    "G1_veredito_bloco3": B / "gpu" / "G1_votos_bloco3" / "veredito.json",
}
NOME_T = "T9_deriva_modelo_G1_v3-12"
LABEL = "tab:drift_models_g10"
PISO = 0.132  # dB; so como divisor de conferencia (a razao impressa vem do JSON)
CELULAS = [("bauru_Q1", "Bauru Q1", "G1_bloco1"), ("campinas_Q1", "Campinas Q1", "G1_bloco1"),
           ("bauru_Q3", "Bauru Q3", "G1_bloco3"), ("campinas_Q3", "Campinas Q3", "G1_bloco3")]
MODELOS = [("gnn", "GNN"), ("mlp", "MLP")]

LEGENDA = (
    "Valid-node error drift of the trained models across split draws. "
    "Trained models at the main geometry ($g = 10$~km, $b = 2$~km), training budget of eight epochs, training seed $42$ for the across-draw SD; "
    "$0.132$~dB is the largest difference between two training repeats with the same seed and partition at $g = 5$~km; "
    "pooled seed SD from three seeds $\\times$ five draws in Bauru Q1 and three seeds $\\times$ ten draws in Campinas Q1; "
    "draws without a valid test node or with a validation partition without an antenna--terrain edge were recorded and not replaced. "
    "Draws excluded are shown as draws without a valid test node plus draws whose validation partition has no antenna--terrain edge; "
    "Sentinel parity has one value per cell, printed in the GNN row, and ``same'' repeats it; --: not run, since the crossed seed block covers only the Q1 cells."
)

HEADER_TEX = ("llccccccc",
              "Cell & Model & Draws used (of 20) & Draws excluded & SD across draws of valid-node MAE (dB) & Ratio to $0.132\,\mathrm{dB}$ & "
              "Pooled SD across seeds (dB) & Spearman with constant-predictor valid-node MAE across draws & Sentinel parity: median $|\\mathrm{GNN}-\\mathrm{MLP}|$ (dB)")
HEADER_CSV = ["celula", "modelo", "n_sorteios_usados", "n_sem_validos", "n_nao_treinaveis", "sorteios_excluidos_texto",
              "dp_entre_sorteios_validos_db", "razao_dp_sobre_0_132", "dp_entre_sementes_pooled_db",
              "spearman_com_constante_validos", "paridade_sentinela_mediana_db"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fnum(x, nd):
    s = f"{x:.{nd}f}"
    if float(s) == 0.0:
        s = f"{0.0:.{nd}f}"
    return s.replace("-", "$-$")


# ---- helpers de cabecalho copiados de scripts/v3_12_gerar_tabelas.py (_quebra, _cabecalho)
def _quebra(txt, larg=11):
    if len(re.sub(r"\\[a-zA-Z]+|[{}$]", "", txt)) <= larg:
        return txt
    toks = re.findall(r"\$[^$]*\$|\S+", txt)
    linhas, cur = [], ""
    for tk in toks:
        vis = len(re.sub(r"\\[a-zA-Z]+|[{}$]", "", cur + " " + tk))
        if cur and vis > larg:
            linhas.append(cur)
            cur = tk
        else:
            cur = (cur + " " + tk).strip()
    linhas.append(cur)
    return "\\celula{" + " \\\\ \\relax ".join(linhas) + "}"


def _cabecalho(h):
    return " & ".join(_quebra(c.strip()) for c in h.split(" & "))


def construir(J):
    """Linhas (dicts) da T9, lendo os campos nomeados dos tres agregados."""
    linhas = []
    for cel, rotulo, blk in CELULAS:
        c = J[blk]["celulas"][cel]
        sv, nt = c["n_sorteios_sem_validos"], c.get("n_sorteios_nao_treinaveis", 0)  # bloco1 nao traz nao_treinaveis (ausente = 0)
        excl = f"{sv}+{nt}" if (sv or nt) else "0"
        par = c["paridade_sentinela"]["mediana_db"]
        for mod, mrot in MODELOS:
            m = c["por_modelo"][mod]
            pooled = (J["G1_bloco4"]["celulas"][cel][mod]["dp_entre_sementes_pooled_db"] if cel == "campinas_Q1" else J["G1_bloco2"]["celulas"][cel][mod]["dp_entre_sementes_pooled_db"]) if cel.endswith("Q1") else None  # v3-12j: Campinas pelo bloco 4 (3x10)
            linhas.append({
                "celula": cel, "rotulo": rotulo, "modelo": mod, "mrot": mrot, "n": m["n_sorteios_com_validos"],
                "sv": sv, "nt": nt, "excl": excl, "dp": m["dp_entre_sorteios_validos_db"], "razao": m["razao_dp_sobre_0_132"],
                "pooled": pooled, "spearman": m["correlacao_com_constante_validos"]["spearman"],
                "paridade": par if mod == "gnn" else "same"})
    return linhas


def escrever(linhas):
    csvp, texp = OUT / f"{NOME_T}.csv", OUT / f"{NOME_T}.tex"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER_CSV)
        for r in linhas:
            w.writerow([r["celula"], r["modelo"], r["n"], r["sv"], r["nt"], r["excl"], r["dp"], r["razao"],
                        "" if r["pooled"] is None else r["pooled"], r["spearman"], r["paridade"]])
    with open(texp, "w", encoding="utf-8") as f:
        f.write("% gerado por _pipeline/v3_12h_gerar_T9_G1.py -- nao editar a mao\n")
        f.write("% T9: lote G1, modelos treinados a g = 10 km, b = 2 km; 4 celulas x GNN/MLP; fontes gpu/G1/agregado_G1_v5_bloco1.json, "
                "agregado_G1_v5_bloco2.json (so a coluna de dp entre sementes, so Q1), agregado_G1_v8_bloco3.json; "
                "'Draws excluded' = sem_validos+nao_treinaveis (so onde ha); 'same' = valor da celula, impresso na linha da GNN; "
                "ambiente table[H] com legenda e label (o \\tabcorpo do manuscrito os neutraliza)\n")
        f.write("\\begin{table}[H]\n\\caption{" + LEGENDA + "}\n\\label{" + LABEL + "}\n\\centering\n")
        f.write("\\begin{tabular}{" + HEADER_TEX[0] + "}\n\\toprule\n" + _cabecalho(HEADER_TEX[1]) + " \\\\\n\\midrule\n")
        for i, r in enumerate(linhas):
            if i and r["modelo"] == "gnn":
                f.write("\\midrule\n")
            par = r["paridade"] if r["paridade"] == "same" else fnum(r["paridade"], 3)
            f.write(" & ".join([r["rotulo"] if r["modelo"] == "gnn" else "", r["mrot"], str(r["n"]), r["excl"], fnum(r["dp"], 2),
                                fnum(r["razao"], 1), "--" if r["pooled"] is None else fnum(r["pooled"], 2),
                                fnum(r["spearman"], 2), par]) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return [texp, csvp]


# ------------------------------------------------------------------ conferencias (releitura por caminho de texto)
def caminho(J, blk, cam):
    d = J[blk]
    for k in cam.split("."):
        d = d[k]
    return d


def conferir(linhas):
    """Cada valor da tabela relido por caminho no JSON (nao pelo codigo de construir) e comparado ao CSV e ao TEX gravados."""
    J = {k: load(p) for k, p in ENT.items()}
    csv_lido = list(csv.DictReader(open(OUT / f"{NOME_T}.csv", newline="", encoding="utf-8")))
    tex = (OUT / f"{NOME_T}.tex").read_text(encoding="utf-8")
    corpo = [ln for ln in tex.split("\n") if ln.endswith("\\\\") and not ln.startswith("\\celula") and "Cell" not in ln and "toprule" not in ln]
    corpo = [ln for ln in corpo if re.search(r"\b(GNN|MLP)\b", ln.split("&")[1] if "&" in ln else "")]
    assert len(corpo) == 8 and len(csv_lido) == 8, (len(corpo), len(csv_lido))
    conf, falhas = {}, []
    ncheck = 0

    def ok(chave, impresso, relido, tol=0.0):
        nonlocal ncheck
        ncheck += 1
        bom = abs(float(impresso) - float(relido)) <= tol
        if not bom:
            falhas.append((chave, impresso, relido))
        return bom

    detalhe = []
    for i, (cel, rotulo, blk) in enumerate([(c, r, b) for c, r, b in CELULAS for _ in MODELOS]):
        mod = MODELOS[i % 2][0]
        base = f"celulas.{cel}.por_modelo.{mod}"
        rc, ln = csv_lido[i], [x.strip() for x in corpo[i].rstrip("\\ ").split("&")]
        assert rc["celula"] == cel and rc["modelo"] == mod
        dp_json = caminho(J, blk, base + ".dp_entre_sorteios_validos_db")
        mae = list(caminho(J, blk, base + ".mae_validos_por_sorteio").values())
        n_json = caminho(J, blk, base + ".n_sorteios_com_validos")
        razao_json = caminho(J, blk, base + ".razao_dp_sobre_0_132")
        sp_json = caminho(J, blk, base + ".correlacao_com_constante_validos.spearman")
        sv_json = caminho(J, blk, f"celulas.{cel}.n_sorteios_sem_validos")
        try:
            nt_json = caminho(J, blk, f"celulas.{cel}.n_sorteios_nao_treinaveis")
        except KeyError:
            nt_json = 0  # bloco1 nao traz o campo (celulas Q1 sem sorteio nao treinavel registrado)
        par_json = caminho(J, blk, f"celulas.{cel}.paridade_sentinela.mediana_db")
        par_rec = statistics.median(caminho(J, blk, f"celulas.{cel}.paridade_sentinela.abs_gnn_menos_mlp_por_sorteio").values())
        # CSV == JSON (exato)
        ok(f"{cel}.{mod}.csv.dp", rc["dp_entre_sorteios_validos_db"], dp_json)
        ok(f"{cel}.{mod}.csv.razao", rc["razao_dp_sobre_0_132"], razao_json)
        ok(f"{cel}.{mod}.csv.spearman", rc["spearman_com_constante_validos"], sp_json)
        ok(f"{cel}.{mod}.csv.n", rc["n_sorteios_usados"], n_json)
        ok(f"{cel}.{mod}.csv.sem_validos", rc["n_sem_validos"], sv_json)
        ok(f"{cel}.{mod}.csv.nao_treinaveis", rc["n_nao_treinaveis"], nt_json)
        # TEX == JSON arredondado
        ok(f"{cel}.{mod}.tex.n", ln[2], n_json)
        ok(f"{cel}.{mod}.tex.dp", ln[4].replace("$-$", "-"), f"{dp_json:.2f}")
        ok(f"{cel}.{mod}.tex.razao", ln[5].replace("$-$", "-"), f"{razao_json:.1f}")
        ok(f"{cel}.{mod}.tex.spearman", ln[7].replace("$-$", "-"), f"{sp_json:.2f}")
        esperado_excl = f"{sv_json}+{nt_json}" if (sv_json or nt_json) else "0"
        ncheck += 1
        if ln[3] != esperado_excl or rc["sorteios_excluidos_texto"] != esperado_excl:
            falhas.append((f"{cel}.{mod}.excluidos", ln[3], esperado_excl))
        # recalculo a partir dos valores por sorteio
        ok(f"{cel}.{mod}.n == len(mae_validos_por_sorteio)", n_json, len(mae))
        ok(f"{cel}.{mod}.dp == stdev(ddof=1) dos MAE por sorteio", dp_json, statistics.stdev(mae), 1e-12)
        ok(f"{cel}.{mod}.razao == dp/0.132", razao_json, dp_json / PISO, 1e-9)
        # contagem de sorteios: usados = plano_usados - sem_validos (Q1) / usaveis (Q3)
        if blk == "G1_bloco3":
            ok(f"{cel}.{mod}.n == n_sorteios_usaveis", n_json, caminho(J, blk, f"celulas.{cel}.n_sorteios_usaveis"))
            ok(f"{cel}.{mod}.usaveis == considerados - sem_validos - nao_treinaveis",
               n_json, caminho(J, blk, f"celulas.{cel}.n_sorteios_plano_considerados") - sv_json - nt_json)
        else:
            ok(f"{cel}.{mod}.n == plano_usados - sem_validos", n_json, caminho(J, blk, f"celulas.{cel}.n_sorteios_plano_usados") - sv_json)
        # paridade: uma por celula, na linha da GNN; MLP = "same"
        if mod == "gnn":
            ok(f"{cel}.csv.paridade", rc["paridade_sentinela_mediana_db"], par_json)
            ok(f"{cel}.tex.paridade", ln[8], f"{par_json:.3f}")
            ok(f"{cel}.paridade == mediana(abs_gnn_menos_mlp_por_sorteio)", par_json, par_rec, 1e-15)
        else:
            ncheck += 2
            if ln[8] != "same" or rc["paridade_sentinela_mediana_db"] != "same":
                falhas.append((f"{cel}.mlp.paridade_same", ln[8], rc["paridade_sentinela_mediana_db"]))
        # dp entre sementes (so Q1; Q3 = traco / vazio)
        if cel.endswith("Q1"):
            fonte = "G1_bloco4" if cel == "campinas_Q1" else "G1_bloco2"
            pj = caminho(J, fonte, f"celulas.{cel}.{mod}.dp_entre_sementes_pooled_db")
            pj2 = caminho(J, fonte, f"celulas.{cel}.{mod}.decomposicao_um_fator_sementes_aninhadas_no_sorteio.dp_entre_sementes_pooled_db")
            ok(f"{cel}.{mod}.csv.pooled", rc["dp_entre_sementes_pooled_db"], pj)
            ok(f"{cel}.{mod}.tex.pooled", ln[6], f"{pj:.2f}")
            ok(f"{cel}.{mod}.pooled == decomposicao.dp_entre_sementes_pooled_db", pj, pj2)
            # o dp de 20 sorteios do bloco 2 (semente 42) e o do bloco 1
            ok(f"{cel}.{mod}.dp bloco1 == dp_entre_sorteios_20_semente42_db bloco2", dp_json,
               caminho(J, "G1_bloco2", f"celulas.{cel}.{mod}.dp_entre_sorteios_20_semente42_db"), 1e-12)
            # dp pooled recalculado da matriz sorteio x semente: sqrt(QM_dentro) (variancia pooled dentro do sorteio)
            mat = caminho(J, fonte, f"celulas.{cel}.{mod}.matriz_mae_validos_sorteio_x_semente_42_43_44")
            pooled_rec = (sum(statistics.variance(l) for l in mat) / len(mat)) ** 0.5
            ok(f"{cel}.{mod}.pooled == sqrt(media das variancias entre sementes por sorteio)", pj, pooled_rec, 1e-12)
        else:
            ncheck += 2
            if ln[6] != "--" or rc["dp_entre_sementes_pooled_db"] != "":
                falhas.append((f"{cel}.{mod}.pooled_vazio", ln[6], rc["dp_entre_sementes_pooled_db"]))
        detalhe.append({"celula": cel, "modelo": mod, "n": n_json, "excluidos": esperado_excl, "dp_json": dp_json,
                        "dp_recalculado_stdev_ddof1": statistics.stdev(mae), "razao_json": razao_json, "spearman_json": sp_json})
    conf["T9_conferencia_por_caminho_json"] = {"fontes": {k: str(p) for k, p in ENT.items()}, "verificacoes": ncheck, "falhas": falhas,
                                               "passou": not falhas, "linhas": detalhe}
    conf["T9_spearman"] = {"nota": "Spearman relido do agregado por caminho e conferido contra CSV e TEX; nao recalculado aqui "
                                   "(exige a MAE do preditor constante por sorteio, que esta no 2.1 e no recalculo gpu/G1_votos_bloco*/veredito.json)",
                           "veredito_independente": {k: load(p).get("veredito") for k, p in VEREDITOS.items()}}
    conf["T9_faixa_razao_0_132"] = {"min": min(r["razao"] for r in linhas), "max": max(r["razao"] for r in linhas),
                                    "impresso": f"{fnum(min(r['razao'] for r in linhas), 1)} a {fnum(max(r['razao'] for r in linhas), 1)}"}
    conf["T9_paridade_faixa_db"] = {"min": min(r["paridade"] for r in linhas if r["paridade"] != "same"),
                                    "max": max(r["paridade"] for r in linhas if r["paridade"] != "same")}
    conf["T9_campos_ausentes_no_json"] = {"n_sorteios_nao_treinaveis em agregado_G1_v5_bloco1.json (bauru_Q1, campinas_Q1)":
                                          "campo inexistente; tratado como 0 (lista sem_validos vazia; n_sorteios_plano_usados = 20 = n_sorteios_com_validos)",
                                          "dp_entre_sementes_pooled_db nas celulas Q3": "inexistente por desenho (bloco 2 so tem Q1); impresso '--' (csv vazio)"}
    return conf, not falhas


def atualizar_registros(saidas, conf):
    cpath = OUT / "CONFERENCIAS_tabelas_v3-12.json"
    C = load(cpath)
    for k in [k for k in C if k.startswith("T9_")]:
        del C[k]  # idempotente: so apaga as proprias chaves
    C.update(conf)
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(C, f, indent=1, ensure_ascii=False)
    mpath = OUT / "MANIFEST_tabelas_v3-12.json"
    M = load(mpath)
    for k, p in {**ENT, **VEREDITOS}.items():
        M["entradas"][k] = {"arquivo": str(p), "sha256": sha(p)}
    for p in saidas + [cpath]:
        M["saidas"][str(p.relative_to(B))] = sha(p)
    M["T9"] = {
        "tarefa": "T9_deriva_modelo_G1_v3-12 (csv + tex): modelos treinados a 10 km, 4 celulas x GNN/MLP, lote G1",
        "protocolo": str(B / "redacao_v3-12" / "DEMANDAS_REDATORES_v3-12h.md") + " (item C-h1) e MATRIZ_DE_ACHADOS_v3-12h.md (linha 2); ordem do dono de 03/10/2026",
        "script": {"arquivo": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "entradas": sorted(ENT) + sorted(VEREDITOS),
        "saidas": [str(p.relative_to(B)) for p in saidas],
        "conferencias": [k for k in C if k.startswith("T9_")],
        "label": LABEL,
        "comando": "/trabalho/ambientes/s33_amb_virtual/.venv/bin/python redacao_v3-12/tables_v3-12/_pipeline/v3_12h_gerar_T9_G1.py",
        "nota": "o gerador principal scripts/v3_12_gerar_tabelas.py regera MANIFEST e CONFERENCIAS por inteiro e apaga estas entradas; reexecutar este script depois dele",
    }
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(M, f, indent=1, ensure_ascii=False)


def main():
    J = {k: load(p) for k, p in ENT.items()}
    for blk in J:
        assert J[blk].get("modo_teste") is False, blk
    linhas = construir(J)
    saidas = escrever(linhas)
    conf, passou = conferir(linhas)
    atualizar_registros(saidas, conf)
    for p in saidas:
        print("gravado", p)
    print("conferencias T9:", conf["T9_conferencia_por_caminho_json"]["verificacoes"], "verificacoes;",
          "falhas:", conf["T9_conferencia_por_caminho_json"]["falhas"])
    return 0 if passou else 1


if __name__ == "__main__":
    sys.exit(main())
