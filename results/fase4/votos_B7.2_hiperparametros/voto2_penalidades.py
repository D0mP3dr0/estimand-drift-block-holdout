import json,glob
G='/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/'
mx={}
n=0;bad=[]
for f in sorted(glob.glob(G+'A[34]/*_v3_*/run_*.json')):
    d=json.load(open(f)); n+=1
    for e in d['epocas']:
        lc=e['loss_componentes']
        # checar identidade prediction + pesos*penalidades = masked_loss
        for k in ('constraint_fspl','constraint_loss','dist_gradient_penalty','variance_penalty','shadowing_ndvi_penalty'):
            v=lc.get(k); mx.setdefault((d['modelo']['classe'],k),[0,0,0]); a=mx[(d['modelo']['classe'],k)]
            if v is None: a[2]+=1; continue
            a[0]=max(a[0],abs(v)); a[1]+=1
        pr=lc['prediction_loss']; mean5=sum(lc[c] for c in ('loss_path_loss_total','loss_path_loss_vegetation','loss_path_loss_terrain','loss_rssi','loss_coverage'))/5
        if abs(pr-mean5)>1e-4*max(1,abs(pr)): bad.append((d['run_label'],e['epoch'],pr,mean5))
        rec=pr+0.05*lc['dist_gradient_penalty']+0.02*lc['variance_penalty']+0.03*lc['shadowing_ndvi_penalty']+lc['constraint_loss']
        if abs(rec-lc['masked_loss'])>2e-3: bad.append(('masked',d['run_label'],e['epoch'],rec,lc['masked_loss']))
print(n,'runs'); 
for k,v in mx.items(): print(k,'max|v|=',v[0],'n=',v[1],'ausentes=',v[2])
print('violacoes identidade',len(bad),bad[:5])
