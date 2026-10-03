import json,csv,hashlib,numpy as np
from scipy import stats
R='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/G1/'
SORT=[811474,54892,880099,125243,159513];SEM=[42,43,44]
b1={('bauru','gnn'):1.2482,('bauru','mlp'):1.1304,('campinas','gnn'):3.2603,('campinas','mlp'):2.0050}
out={'celulas':{},'checks':{}}
for c in ['bauru','campinas']:
 for m in ['gnn','mlp']:
  M=np.zeros((5,3));best=np.zeros((5,3),int);vb=np.zeros((5,3));vl=np.zeros((5,3))
  for i,s in enumerate(SORT):
   hs=set()
   for j,e in enumerate(SEM):
    l=f'g1_{m}_{c}_Q1_ss{s}_s{e}';z=np.load(R+l+'/predicoes_'+l+'.npz');rj=json.load(open(R+l+'/run_'+l+'.json'))
    t=z['target'];p=z['pred'];v=t[:,0]<299
    M[i,j]=np.abs(p[v,3]-t[v,3]).mean()
    hs.add(hashlib.sha256(np.sort(z['idx_global']).tobytes()).hexdigest()+rj['particoes']['test']['idx_sha256_global'])
    assert rj['seed']==e and rj['split_seed']==s and rj['config']['grid_km']==10 and rj['config']['buffer_km']==2,l
    best[i,j]=rj['selecao']['melhor_epoca'];vb[i,j]=rj['selecao']['melhor_val_mae_rssi_db']
    rows=list(csv.DictReader(open(R+l+'/training_log.csv')));vl[i,j]=float(rows[-1]['val_mae_rssi'])
   assert len(hs)==1,('idx diferente',c,m,s)
  qd=np.mean(M.var(1,ddof=1));qe=3*M.mean(1).var(ddof=1)
  s2s=qd;s2o=max(0,(qe-qd)/3);dp=qd**.5;comp=max(.132,dp);razao=b1[(c,m)]/comp
  lo=[np.mean((np.delete(M,i,0)).var(1,ddof=1)) for i in range(5)]
  # leave-one-run-out: dp pooled sem cada corrida (reduz graus: variancia do sorteio com 2 sementes)
  loro={}
  for i in range(5):
   for j in range(3):
    v=[np.delete(M[k],j).var(ddof=1) if k==i else M[k].var(ddof=1) for k in range(5)]
    loro[f'{SORT[i]}_s{SEM[j]}']=float(np.mean(v)**.5)
  F=(qe/qd);a=.05
  ci=[F/stats.f.ppf(1-a/2,4,10),F/stats.f.ppf(a/2,4,10)]  # QMentre/QMdentro ~ (1+3r)F
  r=[(x-1)/3 for x in ci]
  out['celulas'][f'{c}_{m}']=dict(matriz=M.tolist(),QM_entre=qe,QM_dentro=qd,dp_pooled=dp,sigma2_sorteio=s2o,comparador=comp,razao=razao,
   loro_dp=loro,melhor_epoca=best.tolist(),val_best=vb.tolist(),val_ultima=vl.tolist(),
   razao_sorteio_sobre_semente_IC95=[max(0,x)/1 for x in r] if False else [ (qe/qd/stats.f.ppf(.975,4,10)-1)/3,(qe/qd/stats.f.ppf(.025,4,10)-1)/3],F=F)
C=out['celulas']
for k,v in C.items():print(k,'QMe %.4f QMd %.5f dp %.4f s2o %.4f comp %.4f razao %.2f'%(v['QM_entre'],v['QM_dentro'],v['dp_pooled'],v['sigma2_sorteio'],v['comparador'],v['razao']))
razs={k:v['razao'] for k,v in C.items()}
c1={k:v['razao']>=3 for k,v in C.items()}
delim={c:(C[c+'_gnn']['razao']<2 and C[c+'_mlp']['razao']<2) for c in['bauru','campinas']}
cond2={c:all(C[c+'_'+m]['sigma2_sorteio']>=C[c+'_'+m]['QM_dentro'] for m in['gnn','mlp']) for c in['bauru','campinas']}
cond1c={c:c1[c+'_gnn'] and c1[c+'_mlp'] for c in['bauru','campinas']}
out['ramos']=dict(cond1_por_celula=cond1c,cond2_por_celula=cond2,delimita_por_celula=delim,
 reforca=sum(cond1c[c] and cond2[c] for c in cond1c)>=2,delimita=sum(delim.values())>=2)
out['ramos']['delimita_parcial']=(not out['ramos']['reforca']) and (not out['ramos']['delimita'])
print(out['ramos'])
for k in['campinas_gnn','campinas_mlp']:
 v=C[k];print(k);print(np.round(v['matriz'],4));print('loro',{a:round(b,4) for a,b in v['loro_dp'].items()});print('best',v['melhor_epoca']);print('valbest',np.round(v['val_best'],4).tolist());print('vallast',np.round(v['val_ultima'],4).tolist())
for k,v in C.items():print(k,'IC95 s2sorteio/s2semente',np.round(v['razao_sorteio_sobre_semente_IC95'],4))
json.dump(out,open("recalc_bloco2_saida.json","w"),indent=1,default=bool)
