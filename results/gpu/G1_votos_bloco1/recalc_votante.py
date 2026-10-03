import json,glob,os,random,numpy as np
R='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/'
d21=json.load(open(R+'fase2/_v3_2.1_3.1_parcial_16x60rnd.json'))['celulas']
out={'cel':{}};dat={}
for cid in ['bauru','campinas']:
    por={s['split_seed']:s for s in d21[f'{cid}_Q1']['por_sorteio']}
    sss=[s['split_seed'] for s in d21[f'{cid}_Q1']['por_sorteio']][:20]
    for m in ['gnn','mlp']:
        mae=[];cor=[];mc=[];sent=[];nsem=0;ids={}
        for ss in sss:
            p=f'{R}gpu/G1/g1_{m}_{cid}_Q1_ss{ss}_s42/predicoes_g1_{m}_{cid}_Q1_ss{ss}_s42.npz'
            z=np.load(p);t=z['target'];pr=z['pred'];v=t[:,0]<299
            ids[ss]=(z['idx_global'],int(v.sum()),len(t),int((~z['sentinela']).sum()))
            e=np.abs(pr[:,3]-t[:,3])
            if v.sum()==0: nsem+=1;mae.append(np.nan);continue
            mae.append(e[v].mean());sent.append(e[z['sentinela']].mean() if z['sentinela'].any() else np.nan)
            mc.append(por[ss]['mae_constante_teste']['validos'])
        mae=np.array(mae);dat[(cid,m)]=(sss,mae,np.array(sent),ids)
        k=~np.isnan(mae);mc=np.array(mc)
        out['cel'][f'{cid}_{m}']=dict(n=int(k.sum()),dp=float(np.std(mae[k],ddof=1)),razao=float(np.std(mae[k],ddof=1)/0.132),r=float(np.corrcoef(mae[k],mc)[0,1]),sem_valido=nsem)
    sg,sm=dat[(cid,'gnn')][2],dat[(cid,'mlp')][2]
    out['cel'][f'{cid}_med_dif_sent']=float(np.nanmedian(np.abs(sg-sm)))
    # (a)
    random.seed(7);ch=random.sample(sss,3);
    out.setdefault('a',[])
    for ss in ch:
        ig,nv,n,_=dat[(cid,'gnn')][3][ss];im,nv2,n2,_=dat[(cid,'mlp')][3][ss]
        r=por[ss];out['a'].append(dict(cid=cid,ss=ss,n=n,n_21=r['n_test'],validos=nv,validos_21=r['n_pop_teste']['validos'],mlp_n=n2,mlp_val=nv2,idx_igual=bool(np.array_equal(ig,im))))
    # (b)
    for m in ['gnn','mlp']:
        a=dat[(cid,m)][1];a=a[~np.isnan(a)];loo=[float(np.std(np.delete(a,i),ddof=1)) for i in range(len(a))]
        out['cel'][f'{cid}_{m}'].update(loo_min=min(loo),loo_max=max(loo),maxmae=float(a.max()),dp_sem_maior=float(np.std(np.delete(a,a.argmax()),ddof=1)))
out['criterio']={k:v['dp']>=0.396 for k,v in out['cel'].items() if isinstance(v,dict) and 'dp' in v}
out['paridade_ok']={c:out['cel'][f'{c}_med_dif_sent']<=0.117 for c in ['bauru','campinas']}
json.dump(out,open(R+'gpu/G1_votos_bloco1/recalc_saida.json','w'),indent=1)
print(json.dumps(out,indent=1))
