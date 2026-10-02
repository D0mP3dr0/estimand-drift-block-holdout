import json
from scipy.stats import binom
d='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/fase4/'
r4=json.load(open(d+'R4_p_por_classe.json'));out={'c':{},'b':{}}
for c,v in r4['por_celula'].items():
    f=v['fidelidade'];n=f['n_nos_por_m'];fa=f['n_nos_falha_iii_por_m'];T=sum(n)
    exato=sum(n[i]-fa[i] for i in (0,1,3)); 
    out['c'][c]={'n_por_m':n,'falha_iii':fa,'frac_por_m':[x/T for x in n],'frac_exato_L_no_a_no':exato/T,'frac_m2':n[2]/T,'frac_m2_mais_falhas_m1':(n[2]+fa[1]+fa[3]+fa[0])/T,'frac_cotas_L':1-exato/T}
out['c']['binom_P(X<=1;95,0.0201)']=binom.cdf(1,95,0.0201)
r=json.load(open(d+'R2_resumo.json'))['resumo_por_celula']
for c,pp in r.items():
  for pred,pops in pp.items():
    for pop,x in pops.items():
      e=x['estimadores'];rz=x['razoes']
      out['b'][f'{c}|{pred}|{pop}']={'nH':e['H']['n'],'varH':e['H']['variancia'],'varA':e['A']['variancia'],'deff_calc':e['H']['variancia']/e['A']['variancia'],'deff_json':rz.get('deff_H_sobre_A'),'n_sobre_deff_nH':e['H']['n']/(e['H']['variancia']/e['A']['variancia'])}
json.dump(out,open(d+'votos_fismat_R2_R4/bc.json','w'),indent=1)
print(json.dumps(out,indent=0)[:6000]); print(list(rz.keys()))
