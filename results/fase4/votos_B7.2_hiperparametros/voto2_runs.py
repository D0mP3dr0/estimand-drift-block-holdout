import json,glob,statistics as st,os
G='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/'
def load(arm):
    out=[]
    for a in ('A3','A4'):
        for f in sorted(glob.glob(f'{G}{a}/{arm}_v3_*/run_*.json')):
            out.append((a,json.load(open(f)),f))
    return out
R={}
for arm in ('gnn','mlp'):
    allr=load(arm)
    R[arm]={'n_total':len(allr),'n_A3':sum(a=='A3' for a,_,_ in allr),'n_A4':sum(a=='A4' for a,_,_ in allr)}
    # separar sensibilidade MLP
    sens=[r for a,r,f in allr if 'a4s' in r['run_label']]
    main=[r for a,r,f in allr if 'a4s' not in r['run_label']]
    R[arm]['n_main']=len(main); R[arm]['n_sens']=len(sens)
    g=lambda f:[f(r) for r in main]
    def u(x): return sorted(set(map(lambda v: json.dumps(v,sort_keys=True),x)))
    R[arm]['classe']=u(g(lambda r:r['modelo']['classe']))
    R[arm]['num_layers']=u(g(lambda r:r['modelo'].get('num_layers')))
    R[arm]['heads']=u(g(lambda r:r['modelo'].get('heads')))
    R[arm]['hidden']=u(g(lambda r:r['modelo'].get('hidden_dim')))
    R[arm]['larguras']=u(g(lambda r:r['modelo'].get('larguras_encoder')))
    for k in ('terrain_dim','antenna_dim','antenna_dim_nao_usado','output_dim','dropout'):
        R[arm][k]=u(g(lambda r:r['modelo'].get(k)))
    R[arm]['nominal']=u(g(lambda r:r['capacidade']['nominal']))
    R[arm]['efetiva']=u(g(lambda r:r['capacidade']['efetiva']))
    R[arm]['n_sem_grad']=u(g(lambda r:r['capacidade'].get('n_sem_gradiente')))
    for k in ('epochs','batch_size','lr','max_norm','early_stopping_patience','variance_weight','ndvi_weight','dist_grad_weight','cosine_t0','scheduler','freq_mhz','k_antenna','k_terrain'):
        R[arm][k]=u(g(lambda r:r['config'].get(k)))
    for k in ('distance_gradient_weight','variance_weight','shadowing_ndvi_weight'):
        R[arm]['loss.'+k]=u(g(lambda r:r['loss'].get(k)))
    R[arm]['fracao_clip_min']=min(g(lambda r:r['diagnostico_treino']['fracao_passos_clipados']))
    ps=g(lambda r:r['diagnostico_treino']['n_passos']); R[arm]['n_passos']=[st.median(ps),min(ps),max(ps)]
    be=g(lambda r:r['selecao']['melhor_epoca']); R[arm]['melhor_epoca']=[st.median(be),min(be),max(be)]
    R[arm]['criterio']=u(g(lambda r:r['selecao']['criterio'])); R[arm]['test_usado']=u(g(lambda r:r['selecao']['test_usado_na_selecao']))
    R[arm]['pares_seed']=sorted(set((r['seed'],r['split_seed']) for r in main))
    R[arm]['cudnn']=u(g(lambda r:[r['ambiente']['cudnn_deterministic'],r['ambiente']['cudnn_benchmark']]))
    R[arm]['hw']=u(g(lambda r:[r['ambiente']['gpu'],r['ambiente']['torch'],r['ambiente']['cuda']]))
    t=g(lambda r:r['custo']['tempo_total_s']); R[arm]['tempo']=[st.median(t),min(t),max(t)]
    v=g(lambda r:r['custo']['vram_peak_mb']); R[arm]['vram']=[st.median(v),min(v),max(v)]
    R[arm]['escalas']=u(g(lambda r:r['modelo_v3']['escalas']))[:1] if arm=='gnn' else u(g(lambda r:r.get('modelo_v3',{}).get('escalas')))[:1]
    R[arm]['freq_baseline']=u(g(lambda r:r['baselines_analiticos']['train']['freq_mhz']))
    R[arm]['TA_amostradas']=u(g(lambda r:[r['diagnostico_avaliacao']['arestas_amostradas_por_relacao_val']['TA'],r['diagnostico_avaliacao']['arestas_amostradas_por_relacao_test']['TA']])) if arm=='gnn' else None
    R[arm]['regra_aval']=u(g(lambda r:r.get('diagnostico_avaliacao',{}).get('regra')))
    R[arm]['sens']=u([ {'larg':r['capacidade']['larguras'],'nom':r['capacidade']['nominal'],'ef':r['capacidade']['efetiva']} for r in sens])
    R[arm]['fspl_keys']=sorted(set(k for r in main for k in r['loss'] if 'fspl' in k.lower() or 'constraint' in k.lower()))
    R[arm]['loss_keys']=sorted(set(k for r in main for k in r['loss']))
json.dump(R,open(os.path.dirname(__file__)+'/voto2_saida_runs.json','w'),indent=1,ensure_ascii=False)
print(json.dumps(R,indent=1,ensure_ascii=False))
