#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""Gera _v3_2026-09-25/FOLHA_DE_FATOS_v3_fase1_cpu.md copiando números dos JSON (nunca à mão)."""
import json, glob, os
B = '/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/'
V = '/trabalho/HERMES/AGENTES/_FIOS/2026-09-25_gnn_rf_artigo2_v3_fase1_cpu/votos/'
L = lambda p: json.load(open(B + p))
REFUT = {}  # id -> n de votos 'refutado: true' (26/09: exibidos como CONTESTADO)
def votos(idt):
    n = 0
    for pat in (B + f'votos/verificador*_{idt}_voto.json', V + f'verificador*_{idt}_voto.json'):
        for f in glob.glob(pat):
            d = json.load(open(f))
            if d.get('refutado') is False and d.get('motivo_classe') != 'nao_verificavel': n += 1  # nao_verificavel nao conta como voto (26/09)
            if d.get('refutado') is True: REFUT[idt] = REFUT.get(idt, 0) + 1
    return n
rows = []
def g(r, *ks, default=float('nan')):
    for k in ks:
        if k in r: return r[k]
    return default
def row(aleg, texto, num, art, campo, v, nota=''):
    est = 'PROVISÓRIO: %d voto%s' % (v, '' if v == 1 else 's') if v < 3 else 'CITÁVEL: 3 votos'
    for k_, r_ in REFUT.items():
        if k_ in art or k_ in campo:
            est = 'CONTESTADO: %d confirmam / %d refuta%s (ver ata; número NÃO citável até decisão)' % (v, r_, '' if r_ == 1 else 'm')
    e = lambda x: str(x).replace('|', '\\|')  # '|' dentro de célula quebra a tabela markdown (26/09)
    rows.append(f'| {e(aleg)} | {e(texto)} | {e(num)} | `{art}` · `{e(campo)}` | {est} | {e(nota)} |')
# 1.1
d = L('fase1/1.1_geo_celulas_16.json'); ab = d['antenas_bauru_por_quadrante']
row('validade', 'As 16 células cidade-quadrante são disjuntas (interseção 0 km², 0 nós coincidentes)', f"{d['resumo']['n_celulas']} células; pares avaliados: {len(d['sobreposicao_pares_mesma_cidade'])}", 'fase1/1.1_geo_celulas_16.json', 'sobreposicao_pares_mesma_cidade', votos('1.1'), 'cada quadrante é um mosaico 3600×3600; "quarto do mosaico" sai do texto')
row('validade', 'Antenas por célula = antenas da cidade inteira (não recortadas por quadrante)', str(ab), 'fase1/1.1_geo_celulas_16.json', 'antenas_bauru_por_quadrante', votos('1.1'))
# 1.2
d = L('fase1/1.2_cadeia_buffer_violacao.json'); c = d['cadeia_buffer_violacao']
row('validade (BLOQUEIA modelos encadeados)', 'A cadeia Q1→Q2 e Q3→Q4 fura o buffer de 2 km', '; '.join(f"{x['par']}: {x['n_violacoes_buffer']}/{x['n_teste_qi1']} ({100*x['fracao_teste_afetada']:.2f} %), d_min {1000*x['d_min_km']:.1f} m" for x in c), 'fase1/1.2_cadeia_buffer_violacao.json', 'cadeia_buffer_violacao[].{n_violacoes_buffer,fracao_teste_afetada,d_min_km}', votos('1.2'), 'fase 4.3 (retreino sem cadeia) obrigatória')
# 1.3
d = L('fase1/1.3_mc_nodal_vs_blocos.json'); r = d['resumo']
row('A1', 'MC sobre nós com a regra de aparo do protocolo NÃO fecha o hiato contra o nodal de 20 seeds; "granularidade" não confirmada', f"máx teste {100*r['erro_relativo_mc_nodal_vs_nodal_real_teste_max']:.2f} %, val {100*r['erro_relativo_mc_nodal_vs_nodal_real_val_max']:.2f} %; {d['n_permutacoes_mc_nodal']} permutações", 'fase1/1.3_mc_nodal_vs_blocos.json', 'resumo', votos('1.3'), 'verificador1: diferença significativa no teste; explicada pelo 1.8 (amostra de seeds)')
# 1.4
d = L('fase1/1.4_varredura_b_lei_vs_nodal.json'); r = d['resumo_por_gb']
row('A1', 'Erro da lei vs nodal (razão de médias, 20 seeds C-1) é negativo e monotônico em b', '; '.join(f"{k}: {100*r[k]['erro_relativo_test_media_entre_celulas_RATIO_OF_MEANS']:.2f} %" for k in r), 'fase1/1.4_varredura_b_lei_vs_nodal.json', 'resumo_por_gb.*.erro_relativo_test_media_entre_celulas_RATIO_OF_MEANS', votos('1.4'), 'com errata (q por blocos nominais). MAGNITUDE NÃO CITÁVEL: verificadores F e G obtiveram ~metade (g10b2 −3,4 %) com seeds novas, coerente com o 1.8; o texto cita só o sinal negativo e a monotonicidade em b; magnitude pelo 1.8 (200 seeds)')
# 1.5
d = L('fase1/1.5_retencao_exata_vs_mc.json'); r = d['resumo']
row('A1', 'Esperança exata sem reposição NÃO reduz o erro contra o nodal', f"{r['n_exato_reduz_erro']}/{r['n_comparacoes']} comparações reduzem", 'fase1/1.5_retencao_exata_vs_mc.json', 'resumo', votos('1.5'), 'Tabela 2 mantém o campo médio, declarado como aproximação')
# 1.7
d = L('fase1/1.7_grau_preservacao.json'); r = d['resumo']
row('A2', 'Grau preservado: interior sem perda de arestas; perda só na fronteira de vértices', f"interior mín {r['razao_interior_min']:.5f}, média {r['razao_interior_media']:.5f}; fronteira média {r['razao_fronteira_media']:.3f}; {r['n_amostras_particao_x_sorteio']} amostras ({r['n_celulas']} células × {r['n_seeds_por_celula']} sorteios)", 'fase1/1.7_grau_preservacao.json', 'resumo', votos('1.7'), 'ERRATA (fronteira de vértices): interior perde exatamente zero arestas (razão 1,0 em 60/60); fronteira 0,665–0,668; r_E 42,2–42,5 m; Λ máx 0,0053; a antiga "perda residual no interior" era artefato de classificação; P3 = prop:degree com hipóteses')
# 1.8
d = L('fase1/1.8_referencia_nodal_200seeds.json'); r = d['resumo']
row('A1', 'Retenção nodal de teste, 200 seeds aleatórios, 16 células', f"média {r['nodal_te_media_16']:.4f}; dp por sorteio {r['dp_por_sorteio_media']:.4f}", 'fase1/1.8_referencia_nodal_200seeds.json', 'por_celula.*.nodal_te_media / nodal_te_ic95', votos('1.8'), 'referência da Tabela 2')
row('A1', 'Erro da lei de campo médio vs nodal-200 (teste)', f"{100*r['erro_lei_vs_nodal_te_max']:.2f} a {100*r['erro_lei_vs_nodal_te_min']:.2f} % (média {100*r['erro_lei_vs_nodal_te_media']:.2f} %)", 'fase1/1.8_referencia_nodal_200seeds.json', 'por_celula.*.erro_lei_vs_nodal_te (+ _ic95)', votos('1.8'), 'SUBSTITUI −6,2/−6,9 % (20 seeds) e −2,0/−2,7 % (60 seeds)')

if os.path.exists(B + 'fase1/1.8b_referencia_nodal_200seeds_g5.json'):
    d = L('fase1/1.8b_referencia_nodal_200seeds_g5.json'); r = d['resumo']; pc = d['por_celula']
    eva = [c['erro_lei_vs_nodal_va'] for c in pc.values()]
    row('A1', 'Retenção nodal de teste em g = 5 km (483 blocos, 338/72/73), 200 seeds aleatórios, 16 células', f"média {r['nodal_te_media_16']:.4f}; dp por sorteio {r['dp_por_sorteio_media']:.4f}", 'fase1/1.8b_referencia_nodal_200seeds_g5.json', 'por_celula.*.nodal_te_media / nodal_te_ic95', votos('1.8b'), 'Tabela 2, linhas de 5 km')
    row('A1', 'Erro da lei de campo médio vs nodal-200 em g = 5 km (teste; validação)', f"teste {100*r['erro_lei_vs_nodal_te_max']:.2f} a {100*r['erro_lei_vs_nodal_te_min']:.2f} % (média {100*r['erro_lei_vs_nodal_te_media']:.2f} %); validação {100*max(eva):.2f} a {100*min(eva):.2f} %", 'fase1/1.8b_referencia_nodal_200seeds_g5.json', 'por_celula.*.erro_lei_vs_nodal_te / _va (+ _ic95)', votos('1.8b'), 'SUBSTITUI −13,2/−15,7 % e −11,2/−12,9 % (20 seeds, v2); q_te = 73/483')
# 2.1
for suf, tag in (('', '20 seeds [42,0,1..18]'), ('_16x60rnd', '60 seeds aleatórios')):
    d = L(f'fase2/2.1_deriva_erro_baselines_16x20{suf}.json' if not suf else f'fase2/2.1_deriva_erro_baselines{suf}.json'); r = d['resumo']
    row('A3', f'Deriva do ERRO com o sorteio ({tag}): dp entre sorteios do MAE do constante nos válidos vs dp entre células', f"simples {g(r,'dp_medio_entre_sorteios_validos_dB_simples','dp_medio_entre_sorteios_validos_dB','dp_entre_sorteios_medio_validos_dB'):.2f} dB; ponderado {g(r,'dp_medio_entre_sorteios_validos_dB_ponderado'):.2f} dB; entre células {g(r,'dp_entre_celulas_das_medias_validos_dB','dp_entre_celulas_validos_dB'):.2f} dB; {g(r,'n_celulas_com_dp_entre_sorteios_ge_0_1_dB','n_celulas_acima_limiar',default='?')}/16 ≥ 0,1 dB", f'fase2/2.1_deriva_erro_baselines{suf or "_16x20"}.json', 'resumo', votos('2.1') if not suf else votos('2.1_60rnd'), 'principal = 60 seeds aleatórios' if suf else 'reprodução da C-1')
# 3.1
for suf, tag in (('', '20 seeds'), ('_16x60rnd', '60 seeds aleatórios')):
    d = L(f'fase3/3.1_calibracao_validos_vs_mediana{suf}.json'); r = d['resumo']
    row('A4', f'Inversão constante > FSPL nos válidos ({tag}): calibração na mediana contaminada (a) vs só nos válidos (b)', f"(a) {100*g(r,'fracao_inversao_validos_calibracao_a'):.1f} %; (b) {100*g(r,'fracao_inversao_validos_calibracao_b'):.1f} % de {g(r,'n_combinacoes_celula_x_sorteio',default='?')} célula×sorteio", f'fase3/3.1_calibracao_validos_vs_mediana{suf}.json', 'resumo.fracao_inversao_validos_calibracao_{a,b}', votos('3.1') if not suf else votos('3.1_60rnd'), 'a inversão não é artefato da calibração' + (' (principal)' if suf else ''))
d = L('fase3/3.1_calibracao_validos_vs_mediana.json')
lq = d['por_celula'].get('lins_Q1', {})
row('A4', 'Offsets de Lins Q1 (seed 42) por calibração', 'ver campo (a: 20,40/55,22/83,33; b: 26,42/52,19/80,29 dBm, do relatório da frente)', 'fase3/3.1_calibracao_validos_vs_mediana.json', 'por_celula.lins_Q1.<offsets>', votos('3.1'), 'o texto nomeia qual calibração cita')
# 3.2
d = L('fase3/3.2_banda_p7_bordas_reais.json'); r = d['resumo']
row('A5', 'Banda de quantis de P7 sobre sentinelas reais: 0 violações; largura ≥ 1 dB', f"violações {len(r['violacoes_teorema'])}; células com largura ≥ 1 dB: {r['n_celulas_com_largura_ge_1dB_por_modelo']}; largura média {json.dumps({k: round(v,2) for k,v in r['largura_media_banda_dB_por_modelo'].items()})} dB", 'fase3/3.2_banda_p7_bordas_reais.json', 'resumo', votos('3.2'), 'FSPL Lins Q1: offset na posição 0,976 da banda; Hata/COST na borda inferior (consequência do teorema)')
# 3.4
row('A5 (nota)', 'Coincidência 20,4026 × 20,4022 dBm decorre de P7 e da escala do intervalo de d', 'borda com distância real: ver campo', 'fase3/3.4_coincidencia_fspl.json', 'parte_B_recomputo_dado_real', votos('3.4'), 'vira nota; o texto declara que U[1,140] km imita a extensão real')

# ---- formalização do §3 (fase 2, D6) ----
def votos_nome(pat):
    n=0
    for f in glob.glob(B+f'votos/verificador*_{pat}_voto.json'):
        d=json.load(open(f))
        if d.get('refutado') is False and d.get('motivo_classe') != 'nao_verificavel': n+=1
        if d.get('refutado') is True: REFUT[pat] = REFUT.get(pat, 0) + 1
    return n
d = L('fase2/1.6_p_inclusao_por_no.json'); r = d['resumo']
row('A3/§3', 'Probabilidade de inclusão condicional p_i|te: média por célula = retenção do 1.8; perfil por classe k(i) vs forma fechada (k_te−1)_k/(N−1)_k', f"médias {[round(x,4) for x in r['medias_p_cond_te_por_celula']]} (ref. 1.8 {r['faixa_referencia_1_8']}); |Δ| máx k=0,1,2 = {r['max_abs_diff_perfil_k012']:.4f}", 'fase2/1.6_p_inclusao_por_no.json', 'resumo', votos_nome('1.6'), 'DECISÃO DO CHEFE (adendo 54, 26/09): a igualdade média p_i|te = retenção do 1.8 é sustentada por A, O e N (N: 0,4531/0,4513); a magnitude |Δ| ≤ 0,0101 do perfil por classe NÃO é citável — N obteve 0,014–0,020 em k = 1,2 com 100 sorteios próprios, coerente com o Bloco 5 do fismat (k(i) é proxy lateral de m(i)); S (adendo 55) refutou também: k = 2 → 0,0121 com 30 sorteios; o próprio A já registrava 0,011 em k = 2 com 100 seeds. Votação 2×2; a magnitude do perfil sai. Se a redação precisar do perfil, nova frente com m(i) verdadeiro (vizinhança 8).')
d = L('fase2/2.3_estimando_ht_baselines.json'); r = d['resumo']
row('A3/§3', 'Estimando: Σp_i e_i/Σp_i reproduz E_σ[Err_σ] em sentinela/todos mas não nos válidos; HT não reduz o viés', f"células com c≈b (<0,05 dB) por preditor: {r['n_celulas_c_reproduz_b_lt_0_05dB_por_preditor']}; HT reduz à metade nas 3 populações: {r['n_celulas_HT_reduz_vies_pela_metade_todas_populacoes']}/4", 'fase2/2.3_estimando_ht_baselines.json', 'resumo; por_celula.*.(M_sigma, Err_sigma, Cov/E[M])', votos_nome('2.3'), 'identidade Err − θ = −Cov(Err_σ,M)/E[M] exata (resíduo 0); HT SAI (dp por sorteio 7,9–9,1 dB nos válidos); definição formal do estimando = θ com w_i = E[1{i∈V}/M | M>0]')
for prop, txt in (('prop_ratio','Err(U) = Σ w_i e_i, w_i = E[1{i∈V}/M | M>0]; Err − θ = −Cov(Err_σ, M)/E[M]; cota dp(Err)·CV(M)'), ('cor_bias','viés vs domínio = termo de desenho Cov_U(p,e)/p̄ + termo de razão'), ('prop_inclusion','p_i = (k_te/N)·p_i|te; (k_te−1)_m/(N−1)_m ≤ p_i|te ≤ (k_te+k_va−1)_m/(N−1)_m; blindagem vale em m = 1, 3, falha em m = 2'), ('prop_degree','interior perde exatamente zero arestas; cota com constante 9π/4; r_E medido 42,2–42,5 m')):
    row('§3 teorema', txt, 'teorema (prova em LaTeX) + evidência nos artefatos 2.3/1.6/1.7', 'fase2/fismat_secao3_formal.md', prop, votos_nome(prop), 'enumeração/simulação do verificador C confirma')

hdr = ['# Folha de fatos v3 — fase 1 (rodada de CPU) + formalização do §3 (25/09/2026)',
 'Gerada por `scripts/v3_gerar_folha_fase1.py` a partir dos JSON (nenhum número digitado). Estados: CITÁVEL: 3 votos entra no .tex; PROVISÓRIO: n votos entra com `% [[PROVISÓRIO: n votos]]` e fora do abstract/conclusão; 0 votos não é alegado. Votos contados dos arquivos `verificador*_<id>_voto.json` com `refutado: false` e `motivo_classe != nao_verificavel` (voto não verificável não conta). Definições fixadas: retenção = nós retidos / nós dos blocos da partição; erro da lei = (média lei − média nodal)/média nodal, razão de médias; q = fração de blocos nominais. Decisões do dono D1–D5 valem (`_DECISOES/DECISOES-2026-09-25-artigo2-v3-independencia-e-roadmap.md`).',
 '', '| alegação | afirmação | número(s) | artefato · campo | estado | nota |', '|---|---|---|---|---|---|']
open(B + 'FOLHA_DE_FATOS_v3_fase1_cpu.md', 'w').write('\n'.join(hdr + rows) + '\n\n## Fora desta folha (fila)\n- GPU: G0–G3 (`gpu/PROTOCOLO_GPU_v3_2026-09-25.md`); 3.3 (π* por sorteio) depende de G3.\n- 2 votos restantes por número (missão m179036155797); 1.8 e as reexecuções de 60 seeds com 0 votos.\n- Condicionais D1/E1: 1.6, 2.3.\n')
print('\n'.join(rows)); print('\nlinhas:', len(rows))
