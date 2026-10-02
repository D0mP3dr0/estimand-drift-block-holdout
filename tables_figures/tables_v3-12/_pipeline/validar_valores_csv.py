#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""Teste de fumaca (nao e contra-auditoria): reconta com numpy, por caminho diferente do gerador, os valores dos CSV de
tables_v3-12 a partir dos JSON de origem, e confere o balanco de colunas de cada .tex (nivel de chaves 0, com \\multicolumn).
Saida: validacao_valores.json. Tolerancia 1e-9 (relativa)."""
import csv, json, re, sys
import numpy as np
from pathlib import Path
AQUI = Path(__file__).resolve().parent
T = AQUI.parent
B = T.parent.parent  # _v3_2026-09-25
F4 = B / "fase4"
J = lambda p: json.load(open(p, encoding="utf-8"))
def R(nome):
    with open(T / nome, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
out, falhas = {}, []
def ok(chave, a, b, tol=1e-9):
    a, b = float(a), float(b)
    good = abs(a - b) <= tol * max(1.0, abs(b))
    out[chave] = good
    if not good:
        falhas.append((chave, a, b))
C = ["bauru", "campinas", "lins", "sorocaba"]
Q1 = [f"{c}_Q1" for c in C]

# ---- T2
d18, d18b = J(B / "fase1/1.8_referencia_nodal_200seeds.json")["por_celula"], J(B / "fase1/1.8b_referencia_nodal_200seeds_g5.json")["por_celula"]
for r in R("T2_retencao_v3-12.csv"):
    d = d18 if r["g_km"] == "10" else d18b
    cs = [d[f"{r['cidade']}_{q}"] for q in ("Q1", "Q2", "Q3", "Q4")]
    ok(f"T2 {r['g_km']} {r['cidade']} nodal_te", r["nodal_te_media_4cel"], np.mean([c["nodal_te_media"] for c in cs]))
    ok(f"T2 {r['g_km']} {r['cidade']} nodal_va", r["nodal_va_media_4cel"], np.mean([c["nodal_va_media"] for c in cs]))
    ok(f"T2 {r['g_km']} {r['cidade']} lei_te", r["lei_te"], cs[0]["lei_te"])
    ok(f"T2 {r['g_km']} {r['cidade']} err_te_min", r["erro_lei_te_min"], np.min([c["erro_lei_vs_nodal_te"] for c in cs]))
    ok(f"T2 {r['g_km']} {r['cidade']} err_va_max", r["erro_lei_va_max"], np.max([c["erro_lei_vs_nodal_va"] for c in cs]))

# ---- T3 e rodape (recontagem a partir do por-sorteio de b = 0 e do parcial de b = 2)
r1 = J(F4 / "R1_b0_por_sorteio.json")["celulas"]
d21 = J(B / "fase2/2.1_deriva_erro_baselines_16x60rnd.json")
rows = {r["celula"]: r for r in R("T3_deriva_erro_v3-12.csv")}
for k, c in d21["por_celula"].items():
    ok(f"T3 {k} dp", rows[k]["dp_simples"], c["constante_validos"]["dp_simples"])
    ok(f"T3 {k} med_todos", rows[k]["const_todos_mediana"], c["constante_todos"]["mediana"])
b0_mae = np.mean([np.std([s["mae_constante"]["validos"] for s in c["sorteios"] if s["mae_constante"]["validos"] is not None], ddof=1) for c in r1.values()])
b0_fr = np.mean([np.std([s["fracao_valida_teste"] for s in c["sorteios"] if s["mae_constante"]["validos"] is not None], ddof=1) for c in r1.values()])
ok("T3 rodape b0 dp mae", rows["ALL_b0_dp_entre_sorteios_media_das_celulas"]["dp_simples"], b0_mae)
ok("T3 rodape b0 dp frac", rows["ALL_dp_fracao_valida_teste_b0"]["dp_simples"], b0_fr)
p2 = J(B / "fase2/_v3_2.1_3.1_parcial_16x60rnd.json")["celulas"]
b2_fr = np.mean([np.std([s["n_pop_teste"]["validos"] / s["n_pop_teste"]["todos"] for s in c["por_sorteio"] if s["status"] == "ok" and s["n_pop_teste"]["validos"] > 0], ddof=1) for c in p2.values()])
ok("T3 rodape b2 dp frac", rows["ALL_dp_fracao_valida_teste_b2"]["dp_simples"], b2_fr)
ok("T3 rodape dp 2.1", rows["ALL_dp_entre_sorteios_media_das_celulas"]["dp_simples"], np.mean([np.std([s["mae_constante_teste"]["validos"] for s in c["por_sorteio"] if s["status"] == "ok" and s["mae_constante_teste"]["validos"] is not None], ddof=1) for c in p2.values()]))

# ---- T4 (arrays numpy)
def conta(trios):
    a = np.array([t for t in trios if t[0] is not None], dtype=float)
    return len(a), int((a[:, 1] < a[:, 0]).sum()), int((a[:, 2] < a[:, 0]).sum()), int((a[:, 3] < a[:, 4]).sum()), int((a[:, 3] < a[:, 5]).sum())
t2 = [(s["mae_constante_teste"]["validos"], s["mae_modelo_a_contaminado_teste"]["fspl"]["validos"], s["mae_modelo_b_validos_teste"]["fspl"]["validos"],
       s["mae_constante_teste"]["todos"], s["mae_modelo_a_contaminado_teste"]["fspl"]["todos"], s["mae_modelo_b_validos_teste"]["fspl"]["todos"])
      for c in p2.values() for s in c["por_sorteio"] if s["status"] == "ok"]
t0 = [(s["mae_constante"]["validos"], s["mae_fspl_a_contaminado"]["validos"], s["mae_fspl_b_validos"]["validos"],
       s["mae_constante"]["todos"], s["mae_fspl_a_contaminado"]["todos"], s["mae_fspl_b_validos"]["todos"])
      for c in r1.values() for s in c["sorteios"] if s["status"] == "ok"]
rows4 = {r["celula"]: r for r in R("T4_inversao_v3-12.csv")}
for nome, tri, chave in (("b2", t2, "ALL"), ("b0", t0, "ALL_b0_sem_buffer")):
    n, fa, fb, ca, cb = conta(tri)
    r = rows4[chave]
    for col, v in (("n_sorteios_com_validos", n), ("fspl_menor_validos_a", fa), ("fspl_menor_validos_b", fb), ("constante_menor_todos_a", ca), ("constante_menor_todos_b", cb)):
        ok(f"T4 {nome} {col}", r[col], v)
    out[f"T4 {nome} valores"] = [n, fa, fb, ca, cb]

# ---- T5 (n_nos_por_m e frequencias da lista de classes)
r4 = J(F4 / "R4_p_por_classe.json")["por_celula"]
rows5 = {r["classe_m"]: r for r in R("T5_inclusao_por_classe_v3-12.csv")}
for i, k in enumerate(Q1):
    fid = r4[k]["fidelidade"]
    for m in range(4):
        ok(f"T5 m={m} {k} fracao", rows5[str(m)][f"fracao_nos_{k}"], fid["n_nos_por_m"][m] / fid["n_nos"])
        ok(f"T5 m={m} {k} freq", rows5[str(m)][f"frequencia_200_{k}"], [x for x in r4[k]["classes_nodais"] if x["classe"] == f"m={m}|todas"][0]["frequencia_media_nodal"])
        ok(f"T5 m={m} {k} ep da frequencia (v3-12d)", rows5[str(m)][f"ep_mc_frequencia_{k}"], [x for x in r4[k]["classes_nodais"] if x["classe"] == f"m={m}|todas"][0]["ep_mc_bootstrap_sorteios"])
    soc = sum(fid["n_nos_por_m"][m] - (0 if m == 3 else 0) for m in range(4))
    # closed form: m=0,1,3 menos nos m=1 com (iii) falsa; m=2 so cotas (exceto os 'iii vale')
    exato = fid["n_nos_por_m"][0] + (fid["n_nos_por_m"][1] - fid["n_nos_falha_iii_por_m"][1]) + fid["n_nos_por_m"][3] + (fid["n_nos_por_m"][2] - fid["n_nos_falha_iii_por_m"][2])
    ok(f"T5 closed form {k}", rows5["closed_form_available"][f"fracao_nos_{k}"], exato / fid["n_nos"])

# ---- T6 (bias, REQM e deff dos vetores por sorteio)
res = J(F4 / "R2_por_sorteio.json")["celulas"]
bc = J(F4 / "votos_fismat_R2_R4/bc.json")["b"]
for r in R("T6_referencia_desenho_v3-12.csv"):
    c = res[r["celula"]]; mu = c["mu_U"][r["preditor"]][r["populacao"]]
    ps = c["por_sorteio"][r["preditor"]][r["populacao"]]
    H = np.array([np.nan if v is None else v for v in ps["H"]], float); A = np.array(ps["A"], float)
    H = H[~np.isnan(H)]
    ok(f"T6 {r['celula']} {r['preditor']} {r['populacao']} vies", r["vies_holdout_dB"], H.mean() - mu, 1e-7)
    ok(f"T6 {r['celula']} {r['preditor']} {r['populacao']} reqm_H", r["reqm_holdout_dB"], np.sqrt(np.mean((H - mu) ** 2)), 1e-7)
    ok(f"T6 {r['celula']} {r['preditor']} {r['populacao']} reqm_A", r["reqm_amostra_aleatoria_dB"], np.sqrt(np.mean((A - mu) ** 2)), 1e-7)
    ok(f"T6 {r['celula']} {r['preditor']} {r['populacao']} deff", r["deff_var_H_sobre_var_A"], bc[f"{r['celula']}|{r['preditor']}|{r['populacao']}"]["deff_calc"])
    n_nos = np.mean(c["info_por_sorteio"]["n_te_b2" if r["populacao"] == "todos" else "n_te_b2_validos"])
    ok(f"T6 {r['celula']} {r['preditor']} {r['populacao']} n/deff", r["n_sobre_deff"], n_nos / bc[f"{r['celula']}|{r['preditor']}|{r['populacao']}"]["deff_calc"])

# ---- T8
r3 = J(F4 / "R3_termo_desenho.json")["por_celula"]
for r in R("T8_termo_desenho_v3-12.csv"):
    x = r3[r["celula"]]["resultados"][f"{r['preditor']}|{r['populacao']}"]
    ok(f"T8 {r['celula']} {r['preditor']} {r['populacao']} mu", r["mu_U_dB"], x["mu_U_dB"])
    ok(f"T8 {r['celula']} {r['preditor']} {r['populacao']} Tinf", r["T_inf_dB"], x["fechado"]["m_nodal"]["L_no_a_no"]["T_inf_dB"])
    ok(f"T8 {r['celula']} {r['preditor']} {r['populacao']} Tsup", r["T_sup_dB"], x["fechado"]["m_nodal"]["L_no_a_no"]["T_sup_dB"])

# ---- T4_totais e T4_S2 (v3-12f): mesmas contagens que a T4; so os rotulos de calibracao mudam
rt = {r["celula"]: r for r in R("T4_totais_v3-12.csv")}
for nome, esperado in (("ALL", (904, 799, 796, 629, 729)), ("ALL_b0_sem_buffer", (938, 844, 847, 691, 801))):
    r = rt[nome]
    got = tuple(int(r[c]) for c in ("n_sorteios_com_validos", "fspl_menor_validos_a", "fspl_menor_validos_b", "constante_menor_todos_a", "constante_menor_todos_b"))
    out[f"T4_totais {nome} = {esperado}"] = got == esperado
    out[f"T4_totais {nome} = linha da T4"] = got == tuple(int(rows4[nome][c]) for c in ("n_sorteios_com_validos", "fspl_menor_validos_a", "fspl_menor_validos_b", "constante_menor_todos_a", "constante_menor_todos_b"))
tt = "\n".join(l for l in (T / "T4_totais_v3-12.tex").read_text(encoding="utf-8").split("\n") if not l.startswith("%"))
out["T4_totais: so 2 linhas de dados, 6 colunas de contagem, rotulos por extenso"] = (len([l for l in tt.split("\\midrule")[1].split("\\bottomrule")[0].split("\n") if l.strip()]) == 2 and "first" in tt and "second" in tt and "(a)" not in tt and "(b)" not in tt and "median" not in tt)
def corpo(f):
    return [l for l in (T / f).read_text(encoding="utf-8").split("\n") if l and not l.startswith("%")][5:]   # depois do cabecalho
out["T4_S2: corpo da tabela identico ao da T4 (so cabecalhos mudam)"] = corpo("T4_S2_v3-12.tex") == corpo("T4_inversao_v3-12.tex")
s2 = "\n".join(l for l in (T / "T4_S2_v3-12.tex").read_text(encoding="utf-8").split("\n") if not l.startswith("%"))
out["T4_S2: cabecalhos 'first/second calibration', sem (a)/(b)"] = ("first" in s2 and "second" in s2 and "lower (a)" not in s2 and "(a)" not in s2.split("\\midrule")[0])
out["T4_S2 csv = T4 csv"] = (T / "T4_S2_v3-12.csv").read_bytes() == (T / "T4_inversao_v3-12.csv").read_bytes()

# ---- T7b (v3-12d): T7 sem o contraste; valores contra o A4 e contra a T7 (CSV)
a4 = J(B / "gpu/A4/agregado_A4.json")
r7, r7b = {r["celula"]: r for r in R("T7_deriva_modelo_A4_v3-12.csv")}, {r["celula"]: r for r in R("T7b_deriva_modelo_A4_v3-12.csv")}
for k, c in a4["celulas"].items():
    for col, mod, est in (("gnn_med_val", "gnn", "media_entre_sorteios"), ("gnn_dp_val", "gnn", "dp_entre_sorteios"),
                          ("mlp_med_val", "mlp", "media_entre_sorteios"), ("mlp_dp_val", "mlp", "dp_entre_sorteios")):
        ok(f"T7b {k} {col} vs A4", r7b[k][col], c["por_modelo"][mod][est]["mae_rssi_validos_db"])
        ok(f"T7b {k} {col} vs T7 csv", r7b[k][col], r7[k][col])
    # recontagem por outro caminho: media e dp (ddof=1) dos MAE por sorteio, se o A4 os traz
for mod in ("gnn", "mlp"):
    ok(f"T7b rodape {mod} dp entre celulas vs A4", r7b["ALL_dp_entre_celulas_das_medias"][f"{mod}_dp_val"], a4["deriva_2_2"][mod]["dp_entre_celulas_das_medias_validos_db"])
    ok(f"T7b rodape {mod} dp entre celulas = std(ddof=1) das 4 medias",
       r7b["ALL_dp_entre_celulas_das_medias"][f"{mod}_dp_val"], np.std([float(r7b[k][f"{mod}_med_val"]) for k in a4["celulas"]], ddof=1), 1e-6)
out["T7b sem coluna de contraste"] = "d_val_med" not in r7b["bauru_Q1"] and "GNN$-$MLP" not in (T / "T7b_deriva_modelo_A4_v3-12.tex").read_text(encoding="utf-8")

# ---- TA1: todo numero de TA1.csv aparece na folha; nenhuma fracao de corte
ta1 = "\n".join(l for l in (T / "TA1_hiperparametros_v3-12.tex").read_text(encoding="utf-8").split("\n") if not l.startswith("%"))
folha = J(F4 / "B7.2_hiperparametros/b7_2_folha_hiperparametros.json")
out["TA1 sem fracao de corte"] = ("fraction" not in ta1.lower()) and ("clipped" not in ta1.lower()) and ("clipping, norm \\mbox{threshold} & 0.5" in ta1.lower())
out["TA1 nulo sombra x NDVI"] = folha["valor_maximo_shadowing_ndvi_penalty_registrado"] == 0.0 and "identically zero" in ta1
out["TA1 duas colunas valor"] = ta1.count("\\begin{tabular}{>{\\raggedright\\arraybackslash}p{") == 1  # v3-12b: colunas com \raggedright

# ---- TN: bloco identico ao F-10
src = (B / "_propostas_2026-10-01/blocos_formais_v3-12.tex").read_text(encoding="utf-8")
i0 = src.index("(F-10) Tabela de notacao"); a = src.index("\\begin{tabular}", i0); b = src.index("\\end{tabular}", a) + len("\\end{tabular}")
# v3-12d: TN enxuta = F-10 sem a linha do balanco de enlace, com linhas fundidas e as linhas novas de v3-12b.
# Confere: (i) o conjunto de simbolos ($...$ da coluna 1) da TN = o do F-10 menos {\hat L, \hat P, P_tx, L_max} mais {d_max, \bar d_V, MAE, MAE_valid, MAE_sigma};
# (ii) nenhum simbolo mudou; (iii) <= 26 linhas de dados.
tn = (T / "TN_notacao_v3-12.tex").read_text(encoding="utf-8")
def simbolos(txt):
    cols = [l.split(" & ")[0] for l in txt.split("\n") if l.endswith("\\\\") and " & " in l and not l.startswith("\\textbf")]
    return set(x for c in cols for x in re.findall(r"\$[^$]*\$", c))
sf, st = simbolos(src[a:b]), simbolos(tn)
retirados = {"$\\hat L$", "$\\hat P$", "$P_{\\mathrm{tx}}$", "$L_{\\max}$", "$\\ell_x$", "$\\ell_y$"}
novos = {"$B_{\\mathsf{w}}$", "$B_{\\mathsf{e}}$", "$B_{\\mathsf{s}}$", "$B_{\\mathsf{n}}$", "$B_{\\mathsf{ws}}$", "$B_{\\mathsf{es}}$", "$B_{\\mathsf{wn}}$", "$B_{\\mathsf{en}}$", "$d_{\\max}$", "$\\bar d_V$", "$\\mathrm{MAE}$", "$\\mathrm{MAE}_{\\mathrm{valid}}$", "$\\mathrm{MAE}_{\\sigma}$"}
n_linhas_tn = sum(1 for l in tn.split("\n") if l.endswith("\\\\") and " & " in l and not l.startswith("\\textbf"))
out["TN: simbolos = F-10 - balanco de enlace + novos (v3-12d)"] = (st == (sf - retirados) | novos) and (retirados <= sf) and not (retirados & st)
out[f"TN: no maximo 26 linhas ({n_linhas_tn})"] = n_linhas_tn <= 26
# v3-12g: todo \ref/\eqref da coluna "Defined in" aponta para um \label que existe nas quatro partes; nenhum simbolo da TN sem ocorrencia no texto
partes = "\n".join((T.parent / "parts" / f).read_text(encoding="utf-8") for f in ("parteA.tex", "parteB.tex", "parteC.tex", "apendices.tex"))
rotulos = set(re.findall(r"\\label\{([^}]*)\}", partes))
refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]*)\}", tn))
out[f"TN: {len(refs)} rotulos citados, todos existem nas partes"] = refs <= rotulos
# v3-12h: todo simbolo da coluna 1 existe no texto atual (ocorrencia literal, sem espacos, em parteA..C e apendices; sem comentarios)
txt = re.sub(r"(?<!\\\\)%[^\n]*", "", partes).replace(" ", "")
def forma(sim):                                   # $...$ -> conteudo sem espacos e sem chaves redundantes de um so token
    return sim.strip("$").replace(" ", "")
EXCECOES = {"$c = (c_L, c_P)$": "c_L", "$n_1 \\times n_2$": "n_1", "$R(g,b,q)$": "R(g,b", "$R_{\\mathrm{va}}(g,b)$": "R_{\\mathrm{va}}", "$R_{\\mathrm{te}}(g,b)$": "R_{\\mathrm{te}}",
            "$\\mathrm{Err}_\\sigma(U)$": "\\mathrm{Err}_", "$\\bar\\varpi$": "\\bar{\\varpi}", "$\\bar\\rho_{\\mathrm{te}}$": "\\bar{\\rho}_{\\mathrm{te}}", "$\\bar\\rho_{\\mathrm{va}}$": "\\bar{\\rho}_{\\mathrm{va}}",
            "$\\mathcal{N}_b(i)$": "\\mathcal N_b", "$\\bar d_V$": "\\bar d_V", "$\\hat\\tau$": "\\hat\\tau", "$Q_1^{\\pm}$": "Q_1"}
sem_ocorrencia = []
for sim in sorted({x for l in tn.split("\n") if l.endswith("\\\\") and " & " in l and not l.startswith("\\textbf") for x in re.findall(r"\$[^$]*\$", l.split(" & ")[0])}):
    for parte in ([sim] if sim in EXCECOES else ["$" + x + "$" for x in sim.strip("$").split(", ")]):   # varios simbolos num so $...$ (k_tr, k_va, k_te; nu_*)
        alvo = EXCECOES.get(parte, forma(parte)).replace(" ", "")
        if alvo not in txt:
            alt = alvo.replace("{", "").replace("}", "")
            if alt not in txt.replace("{", "").replace("}", ""):
                sem_ocorrencia.append(parte)
out[f"TN: todo simbolo ocorre no texto atual ({len(sem_ocorrencia)} sem ocorrencia)"] = not sem_ocorrencia
if sem_ocorrencia: falhas.append(("TN simbolos sem ocorrencia no texto", sem_ocorrencia))
if not refs <= rotulos: falhas.append(("TN rotulos inexistentes", sorted(refs - rotulos)))

# ---- balanco de colunas
def colunas_spec(spec):
    spec = spec.replace(" ", "")
    n, i, prof = 0, 0, 0
    while i < len(spec):
        ch = spec[i]
        if ch == ">" and spec[i + 1] == "{":      # v3-12b: preambulo de coluna >{...} nao conta como coluna
            d = 0
            i += 1
            while True:
                d += spec[i] == "{"; d -= spec[i] == "}"; i += 1
                if d == 0: break
        elif ch in "lcr":
            n += 1; i += 1
        elif ch == "p" and spec[i + 1] == "{":
            n += 1; i += 1
            d = 0
            while True:
                d += spec[i] == "{"; d -= spec[i] == "}"; i += 1
                if d == 0: break
        else:
            i += 1
    return n
def linhas_tabulares(txt):
    corpo = re.search(r"\\begin\{tabular\}\{(.*)\}\n(.*)\\end\{tabular\}", txt, flags=re.S)
    spec_all = txt[txt.index("\\begin{tabular}{") + len("\\begin{tabular}{"):]
    d, j = 1, 0
    while d:
        d += spec_all[j] == "{"; d -= spec_all[j] == "}"; j += 1
    spec = spec_all[:j - 1]; resto = spec_all[j:]
    resto = resto[:resto.index("\\end{tabular}")]
    rs, buf, prof = [], "", 0
    for ch in resto:
        buf += ch
    # separa linhas por \\ no nivel de chaves 0
    linhas, cur, prof, i = [], "", 0, 0
    while i < len(resto):
        ch = resto[i]
        if ch == "{": prof += 1
        if ch == "}": prof -= 1
        if resto.startswith("\\\\", i) and prof == 0:
            linhas.append(cur); cur = ""; i += 2; continue
        cur += ch; i += 1
    return spec, linhas
for tex in sorted(T.glob("T*_v3-12.tex")):
    spec, linhas = linhas_tabulares(tex.read_text(encoding="utf-8"))
    n = colunas_spec(spec); bons = True
    for ln in linhas:
        ln = "\n".join(x for x in ln.split("\n") if not x.strip().startswith(("\\toprule", "\\midrule", "\\bottomrule", "%")))
        if not ln.strip() or ln.strip() in ("\\bottomrule",): continue
        prof, amp, mc = 0, 0, 0
        for i, ch in enumerate(ln):
            if ch == "{": prof += 1
            elif ch == "}": prof -= 1
            elif ch == "&" and prof == 0 and not ln[i - 1] == "\\": amp += 1
        for m in re.finditer(r"\\multicolumn\{(\d+)\}", ln):
            mc += int(m.group(1)) - 1
        bons &= (amp + 1 + mc == n)
        if amp + 1 + mc != n: falhas.append((tex.name, n, amp + 1 + mc, ln[:60]))
    out[f"colunas {tex.name} ({n})"] = bons
for k, v in out.items():
    if v is False: falhas.append((k, "False"))
json.dump({"resultados": out, "falhas": falhas}, open(AQUI / "validacao_valores.json", "w"), indent=1, default=str)
print("verificacoes:", len(out), "falhas:", len(falhas))
for f in falhas: print(f)
sys.exit(1 if falhas else 0)
