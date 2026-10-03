import json,os,re,hashlib,numpy as np
R='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25'
G=R+'/gpu/G1'
plano=json.load(open(G+'/plano_G1.json'))['corridas']
st={c['run_label']:c for c in json.load(open(G+'/lote_G1_status.json'))['corridas']}
f21=json.load(open(R+'/fase2/_v3_2.1_3.1_parcial_16x60rnd.json'))['celulas']
agg=json.load(open(G+'/agregado_G1_v8_bloco3.json'))
sha=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest()
out={}
def mae(lbl):
    p=f'{G}/{lbl}/predicoes_{lbl}.npz'
    if not os.path.exists(p): return None
    z=np.load(p);t=z['target'];pr=z['pred'];v=t[:,0]<299
    e=np.abs(pr[:,3].astype(np.float64)-t[:,3].astype(np.float64))
    return (e[v].mean() if v.any() else None, e[~v].mean() if (~v).any() else None, int(v.sum()))
for cel,cid in [('bauru','Q3'),('campinas','Q3')]:
    key=f'{cel}_{cid}'
    sorts=[c for c in plano if c['bloco']==3 and c['cidade']==cel and c['quadrante']==cid]
    ss=sorted({c['split_seed'] for c in sorts});r={'n_sorteios_plano':len(ss)}
    res={'gnn':{},'mlp':{}};sent={'gnn':{},'mlp':{}};excl={}
    for s in ss:
        for m in ('gnn','mlp'):
            lbl=f'g1_{m}_{cel}_{cid}_ss{s}_s42'
            x=mae(lbl);stt=st.get(lbl,{})
            lg=open(f'{G}/log_{lbl}.txt',errors='ignore').read()
            deg=re.findall(r'PARTICAO DEGENERADA: (\[[^\]]*\])',lg)
            if x is None:
                excl.setdefault(s,{})[m]={'npz':False,'rc':stt.get('rc'),'degenerada':deg}
            else:
                if x[0] is None: excl.setdefault(s,{})[m]={'npz':True,'sem_validos':True,'rc':stt.get('rc'),'log_deg':deg}
                else: res[m][s]=x[0];sent[m][s]=x[1]
    r['excluidos']=excl
    us=sorted(set(res['gnn'])&set(res['mlp']))
    r['mesmos_sorteios_gnn_mlp']=set(res['gnn'])==set(res['mlp']);r['n_usaveis']=len(us)
    const=np.array([[p['mae_constante_teste']['validos'] for p in f21[key]['por_sorteio'] if p['split_seed']==s][0] for s in us])
    for m in ('gnn','mlp'):
        a=np.array([res[m][s] for s in us]);d=a.std(ddof=1)
        loo=[np.delete(a,i).std(ddof=1) for i in range(len(a))]
        r[m]={'dp_ddof1':d,'dp_ddof0':a.std(),'razao':d/0.132,'pearson_const':float(np.corrcoef(a,const)[0,1]),'loo_min':min(loo),'loo_max':max(loo),'loo_min_razao':min(loo)/.132,'loo_max_razao':max(loo)/.132,
           'agg_dp':agg['celulas'][key]['por_modelo'][m]['dp_entre_sorteios_validos_db']}
    r['paridade_mediana']=float(np.median([abs(sent['gnn'][s]-sent['mlp'][s]) for s in us]))
    r['agg_paridade']=json.dumps(agg['celulas'][key]['paridade_sentinela'])[:300]
    out[key]=r
# ramos
c1={}
for k,p in [('bauru_Q1','agregado_G1_v5_bloco1.json'),('campinas_Q1','agregado_G1_v5_bloco2.json')]:
    a=json.load(open(f'{G}/{p}'));out['agregado_'+k+'_leitura']=json.dumps(a.get('leitura_aritmetica_por_celula',{}).get(k,{}))[:900]
out['ramos_agg']=agg['contagem_mecanica_ramos_4_celulas']
out['sha_agregado']=sha(G+'/agregado_G1_v8_bloco3.json');out['sha_script']=sha(R+'/gpu/G1_votos_bloco3/verif_bloco3.py')
json.dump(out,open('/tmp/claude-1000/-trabalho-HERMES/492a006d-c0f8-41c5-9896-1453e7ec0600/scratchpad/o.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
