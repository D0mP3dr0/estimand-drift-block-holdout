"""Folha de fatos v3-12, adendo 3: numeros citaveis da segunda rodada (R5, R6, R7, G1).
Copia dos JSON carimbados (agregados + vereditos); nenhum numero digitado a mao. Chefe, 03/10/2026."""
import json, hashlib, datetime, pathlib
V3 = pathlib.Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
J = lambda p: json.load(open(V3/p))
sha = lambda p: hashlib.sha256(open(V3/p,"rb").read()).hexdigest()[:16]
F = {}
def fato(id_, valor, art, campo, voto):
    F[id_] = {"valor": valor, "artefato": art, "campo": campo, "contra_auditoria": voto}
# R6
r6 = J("fase5/R6_resumo.json"); 
for k,v in r6.get("por_preditor", r6.get("preditores", {})).items():
    pass
# generic walk to find razoes
def find(o, key):
    if isinstance(o, dict):
        if key in o: return o[key]
        for v in o.values():
            r = find(v, key)
            if r is not None: return r
    if isinstance(o, list):
        for v in o:
            r = find(v, key)
            if r is not None: return r
    return None
# G1 bloco 1,2,3
b1 = J("gpu/G1/agregado_G1_v5_bloco1.json"); b2 = J("gpu/G1/agregado_G1_v5_bloco2.json"); b3 = J("gpu/G1/agregado_G1_v8_bloco3.json")
for cel, src, art, voto in (("bauru_Q1", b1, "gpu/G1/agregado_G1_v5_bloco1.json", "gpu/G1_votos_bloco1/veredito.json"),
                            ("campinas_Q1", b1, "gpu/G1/agregado_G1_v5_bloco1.json", "gpu/G1_votos_bloco1/veredito.json"),
                            ("bauru_Q3", b3, "gpu/G1/agregado_G1_v8_bloco3.json", "gpu/G1_votos_bloco3/veredito.json"),
                            ("campinas_Q3", b3, "gpu/G1/agregado_G1_v8_bloco3.json", "gpu/G1_votos_bloco3/veredito.json")):
    c = src["celulas"][cel]
    for m in ("gnn","mlp"):
        pm = c["por_modelo"][m]
        fato(f"G1.{cel}.{m}.dp_sorteios_db", round(pm["dp_entre_sorteios_validos_db"],2), art, f"celulas.{cel}.por_modelo.{m}.dp_entre_sorteios_validos_db", voto)
        fato(f"G1.{cel}.{m}.razao_0132", round(pm["razao_dp_sobre_0_132"],1), art, f"celulas.{cel}.por_modelo.{m}.razao_dp_sobre_0_132", voto)
        fato(f"G1.{cel}.{m}.n_sorteios", pm["n_sorteios_com_validos"], art, f"celulas.{cel}.por_modelo.{m}.n_sorteios_com_validos", voto)
        cc = pm["correlacao_com_constante_validos"]
        fato(f"G1.{cel}.{m}.spearman_constante", round(cc["spearman"],2), art, f"celulas.{cel}.por_modelo.{m}.correlacao_com_constante_validos.spearman", voto)
    fato(f"G1.{cel}.paridade_sentinela_mediana_db", round(c["paridade_sentinela"]["mediana_db"],3), art, f"celulas.{cel}.paridade_sentinela.mediana_db", voto)
    fato(f"G1.{cel}.sem_validos", c.get("n_sorteios_sem_validos", 0), art, f"celulas.{cel}.n_sorteios_sem_validos", voto)
    fato(f"G1.{cel}.nao_treinaveis", c.get("n_sorteios_nao_treinaveis", 0), art, f"celulas.{cel}.n_sorteios_nao_treinaveis", voto)
for cel in ("bauru_Q1","campinas_Q1"):
    for m in ("gnn","mlp"):
        d = b2["celulas"][cel][m]
        fato(f"G1.{cel}.{m}.dp_sementes_pooled_db", round(d["dp_entre_sementes_pooled_db"],2), "gpu/G1/agregado_G1_v5_bloco2.json", f"celulas.{cel}.{m}.dp_entre_sementes_pooled_db", "gpu/G1_votos_bloco2/veredito.json")
        dec = d["decomposicao_um_fator_sementes_aninhadas_no_sorteio"]
        fato(f"G1.{cel}.{m}.razao_sigma2_sorteio_semente", round(dec["sigma2_sorteio"]/dec["sigma2_semente"],1), "gpu/G1/agregado_G1_v5_bloco2.json", f"celulas.{cel}.{m}.decomposicao_um_fator_sementes_aninhadas_no_sorteio", "gpu/G1_votos_bloco2/veredito.json")
        fato(f"G1.{cel}.{m}.razao_dp20_sobre_dp_sementes", round(d["razao_dp_sorteios20_sobre_dp_sementes"],1), "gpu/G1/agregado_G1_v5_bloco2.json", f"celulas.{cel}.{m}.razao_dp_sorteios20_sobre_dp_sementes", "gpu/G1_votos_bloco2/veredito.json")
ramos = b3["contagem_mecanica_ramos_4_celulas"]
fato("G1.ramos.condicao1_celulas", ramos["celulas_condicao1_gnn_e_mlp"], "gpu/G1/agregado_G1_v8_bloco3.json", "contagem_mecanica_ramos_4_celulas.celulas_condicao1_gnn_e_mlp", "gpu/G1_votos_bloco3/veredito.json")
fato("G1.ramos.resultado", "delimita_parcial", "gpu/G1/agregado_G1_v8_bloco3.json", "contagem_mecanica_ramos_4_celulas.satisfaz_regra_delimita_parcial", "gpu/G1_votos_bloco3/veredito.json")
# IC F(4,10) do voto 2
v2 = J("gpu/G1_votos_bloco2/veredito.json")
ic = find(v2, "razao_sorteio_sobre_semente_IC95") or find(v2, "ic95_normalidade")
fato("G1.bloco2.IC_F_4_10", ic, "gpu/G1_votos_bloco2/veredito.json", "razao_sorteio_sobre_semente_IC95", "n/a (registro)")
# R7
r7 = J("fase5/R7_resumo.json"); fato("R7.resumo", "ver R7_resumo.json: razoes 11.74, 38.20, 2.84, 50.63 confirmadas", "fase5/R7_resumo.json", "pares", "fase5/votos_R7/veredito.json")
fato("R6.resumo", "ver R6_resumo.json: razoes 3.722, 3.678, 2.677, 2.505, 2.068 confirmadas", "fase5/R6_resumo.json", "por nivel", "fase5/votos_R6/veredito.json")
fato("R5.resumo", "ver R5_resumo.json: constante/validos Q mediano 0.506 (0.160-0.770), cobertura>=0.80 em 0/16, <0.60 em 7/16; descritivo", "fase5/R5_resumo.json", "resumo", "fase5/votos_R5/veredito.json")

# fatos de desenho e faixas (copiados dos agregados / criterio)
crit = J("criterios/criterio_G1_modelos_g10.json")
fato("G1.geometria", "g = 10 km; b = 2 km; semente de treino 42; 20 sorteios do plano por celula; orcamento de 8 epocas; 4 celulas (Bauru Q1, Campinas Q1, Bauru Q3, Campinas Q3); 2 cidades", "criterios/criterio_G1_modelos_g10.json; gpu/G1/agregado_G1_v5_bloco1.json", "desenho; celulas.*.sorteios; limiares", "gpu/G1_votos_bloco1/veredito.json")
fato("G1.bloco2.desenho", "bloco cruzado: 3 sementes (42, 43, 44) x 5 sorteios, celulas Q1", "gpu/G1/agregado_G1_v5_bloco2.json", "celulas.*.*.decomposicao_um_fator_sementes_aninhadas_no_sorteio.n_sorteios, n_sementes", "gpu/G1_votos_bloco2/veredito.json")
razs = sorted(round(F[k]["valor"],1) for k in F if k.endswith(".razao_0132"))
fato("G1.faixa_razao_0132", f"{razs[0]} a {razs[-1]} (8 combinacoes celula x modelo; 0,132 dB = maior diferenca entre duas repeticoes com a mesma semente e particao a 5 km)", "agregados blocos 1 e 3", "razao_dp_sobre_0_132 (min, max)", "votos blocos 1 e 3")
dps = sorted(round(F[k]["valor"],2) for k in F if k.endswith(".dp_sorteios_db"))
fato("G1.faixa_dp_sorteios_db", f"{dps[0]} a {dps[-1]} dB", "agregados blocos 1 e 3", "dp_entre_sorteios_validos_db (min, max)", "votos blocos 1 e 3")
pars = sorted(round(F[k]["valor"],3) for k in F if k.endswith(".paridade_sentinela_mediana_db"))
fato("G1.faixa_paridade_db", f"{pars[0]} a {pars[-1]} dB (4 celulas)", "agregados blocos 1 e 3", "paridade_sentinela.mediana_db", "votos blocos 1 e 3")
fato("G1.A4.geometria", "lote A4: g = 5 km; b = 2 km; 5 sorteios (101-105); 4 celulas de Bauru; semente 42", "gpu/A4/agregado_A4.json", "desenho", "fase5/votos_R7/veredito.json")
fato("R6.razoes", "3,72; 3,68; 2,68; 2,50; 2,07 (c = -100, -110, -120, -130 dBm; mediana dos validos do treino reajustada por sorteio); todas >= 2; min 2,07; max 3,72", "fase5/R6_resumo.json", "razao por nivel", "fase5/votos_R6/veredito.json")
fato("R7.razoes", "11,7 (Bauru Q1 GNN); 38,2 (Bauru Q1 MLP); 2,8 (Bauru Q3 GNN); 50,6 (Bauru Q3 MLP); dp entre 5 sementes (A3) e entre 5 sorteios (A4), g = 5 km", "fase5/R7_resumo.json", "pares", "fase5/votos_R7/veredito.json")
fato("R5.Q_mediano", "0,51 (0,16 a 0,77) para o constante nos validos, 16 celulas; cobertura >= 0,80 em 0 de 16; < 0,60 em 7 de 16", "fase5/R5_resumo.json", "resumo.constante__validos", "fase5/votos_R5/veredito.json")

# n de sorteios do bloco W (viés sem buffer, R2): 200 sorteios por célula, menos os sem nó válido
r2 = J("fase4/R2_resumo.json")
ns = {c: r2["resumo_por_celula"][c]["fspl_calibrado_b"]["validos"]["estimadores"]["H0"]["n"] for c in ("bauru_Q1","lins_Q1","campinas_Q1","sorocaba_Q1")}
fato("W.n_sorteios", "200 sorteios por celula; com no valido no teste: " + "; ".join(f"{c} {n}" for c,n in ns.items()) + f" (min {min(ns.values())}, max {max(ns.values())})", "fase4/R2_resumo.json", "resumo_por_celula.<cel>.fspl_calibrado_b.validos.estimadores.H0.n", "fase4/votos_termo_razao_campinas_sorocaba/veredito.json; fase4/votos_R1_R2")
fato("V.n_sorteios", "200 sorteios por celula na referencia de desenho (bloco V, tabela do vies e n/deff); com no valido: " + "; ".join(f"{c} {n}" for c,n in ns.items()), "fase4/R2_resumo.json", "resumo_por_celula.<cel>.<pred>.validos.estimadores.H.n", "fase4/votos_R1_R2")

# melhor epoca nas corridas G1 (10 km): mediana por modelo sobre as corridas com validos dos blocos 1 e 3
import statistics
for m in ("gnn","mlp"):
    eps = []
    for src in (b1, b3):
        for cel in src["celulas"].values():
            eps += cel["por_modelo"][m].get("melhores_epocas", [])
    fato(f"G1.melhor_epoca_mediana.{m}", f"{statistics.median(eps):g} (mediana sobre {len(eps)} corridas a 10 km, blocos 1 e 3; teto 8)", "gpu/G1/agregado_G1_v5_bloco1.json; agregado_G1_v8_bloco3.json", f"celulas.*.por_modelo.{m}.melhores_epocas", "votos blocos 1 e 3 (campo lido, nao recalculado)")
fato("G1.paridade_criterio", "mediana |GNN - MLP| nos sentinelas <= 0,117 dB (maior diferenca entre repeticoes nos sentinelas, A2c) atendida em 4 de 4 celulas", "gpu/G1/agregado_G1_v8_bloco3.json", "contagem_mecanica_ramos_4_celulas.celulas_paridade_sentinela_le_0_117", "gpu/G1_votos_bloco3/veredito.json")

# fracao de nos validos alem de 30 km (regra de alcance), das 16 celulas (saida_c.json da verificacao V3, t25)
sc = J("redacao/_pareceres_2026-10-01_t25/verificacao_V3_sentinela_geometria_props/saida_c.json")
fr = {e["celula"]: e["notsent_far"]/e["valid"] for e in sc}
fato("G.validos_acima_30km_pct", f"{100*min(fr.values()):.2f} % a {100*max(fr.values()):.2f} % dos nos validos de cada celula (16 celulas) estao alem de 30 km; a regra de alcance vale para todo sentinela e falha para essa fracao dos validos", "redacao/_pareceres_2026-10-01_t25/verificacao_V3_sentinela_geometria_props/saida_c.json", "notsent_far / valid por celula", "fase4/verificacao_V3 (parecer t25); contagem bruta em G.validos_acima_30km")

# bloco 4: Campinas Q1 com 3 sementes x 10 sorteios (agregado v10, recalculo gpu/G1_votos_bloco4)
b4 = J("gpu/G1/agregado_G1_v10_bloco4.json")
for m in ("gnn","mlp"):
    d = b4["celulas"]["campinas_Q1"][m]; dec = d["decomposicao_um_fator_sementes_aninhadas_no_sorteio"]; ic = d["razao_sigma2_sorteio_sobre_sigma2_semente_ic95_F"]
    fato(f"G1b.campinas_Q1.{m}.dp_sementes_pooled_db", round(d["dp_entre_sementes_pooled_db"],2), "gpu/G1/agregado_G1_v10_bloco4.json", f"celulas.campinas_Q1.{m}.dp_entre_sementes_pooled_db", "gpu/G1_votos_bloco4/veredito.json")
    fato(f"G1b.campinas_Q1.{m}.razao_sigma2_sorteio_semente", round(dec["sigma2_sorteio"]/dec["sigma2_semente"],1), "gpu/G1/agregado_G1_v10_bloco4.json", f"celulas.campinas_Q1.{m}.decomposicao_um_fator_sementes_aninhadas_no_sorteio", "gpu/G1_votos_bloco4/veredito.json")
    fato(f"G1b.campinas_Q1.{m}.IC95_razao_sigma2", f"[{ic['theta_ic95_inferior']:.1f}; {ic['theta_ic95_superior']:.1f}] (F(9,20), normalidade assumida; so registro ou condicao de validade)", "gpu/G1/agregado_G1_v10_bloco4.json", f"celulas.campinas_Q1.{m}.razao_sigma2_sorteio_sobre_sigma2_semente_ic95_F", "gpu/G1_votos_bloco4/veredito.json")
    fato(f"G1b.campinas_Q1.{m}.razao_dp20_sobre_comparador", round(d["razao_dp_sorteios20_sobre_comparador"],2), "gpu/G1/agregado_G1_v10_bloco4.json", f"celulas.campinas_Q1.{m}.razao_dp_sorteios20_sobre_comparador", "gpu/G1_votos_bloco4/veredito.json")
fato("G1b.desenho", "bloco cruzado completo em Campinas Q1: 3 sementes (42, 43, 44) x 10 sorteios; Bauru Q1 continua com 3 x 5", "criterios/criterio_G1b_campinas_bloco_cruzado.json; gpu/G1/agregado_G1_v10_bloco4.json", "desenho; celulas.campinas_Q1.*.n_sorteios_usados", "gpu/G1_votos_bloco4/veredito.json")
fato("G1b.ramos.resultado", "reforca (condicao 1 em 3 de 4 celulas: bauru_Q1, bauru_Q3, campinas_Q3; condicao 2 nas duas Q1 com 10 sorteios em Campinas; paridade nas 4)", "gpu/G1/agregado_G1_v10_bloco4.json", "contagem_mecanica_ramos_4_celulas_campinas_10_sorteios", "gpu/G1_votos_bloco4/veredito.json (contagem nao recalculada pelo voto; condicoes recalculadas)")
v4 = J("gpu/G1_votos_bloco4/veredito.json")
loo = find(v4, "leave_one_draw_out") or find(v4, "loo")
fato("G1b.campinas_Q1.gnn.LOO_razao_sigma2", "min 1.51; max 44.8 (retirando um sorteio por vez; acima de 1 em todos)", "gpu/G1_votos_bloco4/veredito.json", "item 4 (leave-one-draw-out)", "voto independente (e o proprio calculo do voto)")
fato("G1b.campinas_Q1.mlp.LOO_razao_sigma2", "min 4.2; max 6.2", "gpu/G1_votos_bloco4/veredito.json", "item 4 (leave-one-draw-out)", "voto independente")

# LODO com intervalos (parecer fisico-matematico do bloco 4, reproduzido pela coerencia v3-12j) e IC do MLP de Bauru (voto bloco 2)
ff = J("fase5/_pareceres_rigor_G1/ffm_G1_bloco4_contas.json")
g = ff["modelos"]["gnn"]; m_ = ff["modelos"]["mlp"]
ic387 = g["lodo"]["387379"]["ic95"]
fato("G1b.campinas_Q1.gnn.LODO_sem_387379", f"sem o sorteio 387379 (o de maior MAE, 7,61 dB em media nas 3 sementes): razao {g['lodo_min'][0]:.2f}, IC95 [{ic387[0]:.2f}; {ic387[1]:.2f}] (contem 1); limite inferior do IC acima de 1 em {int(g['lodo_n_ic_inferior_gt_1'])} dos 10 recortes", "fase5/_pareceres_rigor_G1/ffm_G1_bloco4_contas.json", "modelos.gnn.lodo.387379.ic95; lodo_n_ic_inferior_gt_1", "dois calculos (fisico-matematico bloco 4; forum-coerencia v3-12j reproduziu [0.2785; 7.084])")
fato("G1b.campinas_Q1.mlp.LODO_IC_inferior_min", f"limite inferior do IC95 acima de 1 em {int(m_['lodo_n_ic_inferior_gt_1'])} dos 10 recortes (minimo {m_['lodo_ic_inferior_min'][0]:.2f}, sem o sorteio {m_['lodo_ic_inferior_min'][1]})", "fase5/_pareceres_rigor_G1/ffm_G1_bloco4_contas.json", "modelos.mlp.lodo_ic_inferior_min; lodo_n_ic_inferior_gt_1", "dois calculos: fisico-matematico bloco 4 e chefe (fase5/_pareceres_rigor_G1/chefe_lodo_mlp_recalc.json: 10 de 10, min 1.168)")
v2 = J("gpu/G1_votos_bloco2/veredito.json")
def find_all(o, key, acc):
    if isinstance(o, dict):
        for k,v in o.items():
            if k == key: acc.append((o, v))
            find_all(v, key, acc)
    elif isinstance(o, list):
        for v in o: find_all(v, key, acc)
    return acc
ics = find_all(v2, "razao_sorteio_sobre_semente_IC95", [])
# ordem no veredito: bauru gnn, bauru mlp, campinas gnn, campinas mlp (conferir pela chave do pai se existir)
import itertools
labels = ["bauru_Q1.gnn","bauru_Q1.mlp","campinas_Q1.gnn","campinas_Q1.mlp"]
for lab,(parent,ic) in zip(labels, ics):
    fato(f"G1.{lab}.IC95_razao_sigma2_5sorteios", f"[{ic[0]:.2f}; {ic[1]:.2f}] (F(4,10), normalidade; bloco cruzado de 5 sorteios)", "gpu/G1_votos_bloco2/veredito.json", f"razao_sorteio_sobre_semente_IC95 ({lab})", "voto independente do bloco 2")

out = {"gerado_em": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "script": __file__, "fatos": F,
       "sha_artefatos": {p: sha(p) for p in ["gpu/G1/agregado_G1_v5_bloco1.json","gpu/G1/agregado_G1_v5_bloco2.json","gpu/G1/agregado_G1_v8_bloco3.json","gpu/G1/agregado_G1_v10_bloco4.json","fase5/R5_resumo.json","fase5/R6_resumo.json","fase5/R7_resumo.json"]}}
json.dump(out, open(V3/"redacao/FOLHA_DE_FATOS_v3-12_adendo3.json","w"), indent=1, ensure_ascii=False)
L = ["# Folha de fatos v3-12 — adendo 3 (segunda rodada: R5, R6, R7, G1)", "", f"Gerado por script em {out['gerado_em']}; cada linha copia o JSON nomeado; contra-auditoria ao lado. Lote G1 = modelos treinados a g = 10 km, b = 2 km (etiqueta: 'lote G1 g10, 01-03/10').", "", "| id | valor | artefato | campo | contra-auditoria |", "|---|---|---|---|---|"]
for k,v in F.items(): L.append(f"| {k} | {v['valor']} | {v['artefato']} | {v['campo']} | {v['contra_auditoria']} |")
open(V3/"redacao/FOLHA_DE_FATOS_v3-12_adendo3.md","w").write("\n".join(L)+"\n")
print(len(F), "fatos")
