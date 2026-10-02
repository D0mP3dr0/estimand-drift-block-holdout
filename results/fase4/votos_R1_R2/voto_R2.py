# Voto R2, implementacao independente. Importa apenas minha particao de voto_R1.py (escrita por mim).
# Detalhe copiado do criterio_2.3: constante = mediana(RSSI do treino do split seed=42); e_i=|rssi-const| em todos os nos.
import json, sys, numpy as np
sys.path.insert(0, '/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/fase4/votos_R1_R2')
from voto_R1 import carrega, particao, BASE
c = sys.argv[1] if len(sys.argv) > 1 else 'bauru_Q1'
d = json.load(open(BASE.replace('fase4/', 'fase1/') + '1.8_referencia_nodal_200seeds.json'))
seeds = d['seeds']; assert len(seeds) == 200
P, rssi, sent = carrega(c); n = len(rssi)
tr42, _, _ = particao(P, 10.0, 2.0, 42)
const = float(np.median(rssi[tr42])); e = np.abs(rssi - const)
H = {'todos': [], 'validos': []}; A = {'todos': [], 'validos': []}; nte = []
for j, s in enumerate(seeds):
    _, _, te = particao(P, 10.0, 2.0, s)
    idx = np.flatnonzero(te); nte.append(len(idx))
    H['todos'].append(e[idx].mean())
    v = idx[~sent[idx]]; H['validos'].append(e[v].mean() if len(v) else np.nan)
    ia = np.random.RandomState(20261001 + j).choice(n, size=len(idx), replace=False)  # semente do criterio (copiada)
    A['todos'].append(e[ia].mean()); va = ia[~sent[ia]]; A['validos'].append(e[va].mean() if len(va) else np.nan)
    if j % 20 == 0: print(j, flush=True)
out = {'celula': c, 'constante_dBm': const, 'n_nos': n, 'media_n_te': float(np.mean(nte))}
for pop in ('todos', 'validos'):
    m = ~sent if pop == 'validos' else np.ones(n, bool)
    mu = float(e[m].mean())
    h = np.array(H[pop]); h = h[~np.isnan(h)]; a = np.array(A[pop]); a = a[~np.isnan(a)]
    r = {'mu_U': mu, 'n_H': len(h), 'var_H': float(h.var(ddof=1)), 'vies_H': float(h.mean() - mu),
         'ep_vies': float(h.std(ddof=1) / np.sqrt(len(h)))}
    r['vies_em_ep'] = r['vies_H'] / r['ep_vies']
    r['var_A_simulada'] = float(a.var(ddof=1))
    if pop == 'todos':
        S2 = float(e.var(ddof=1)); nn = float(np.mean(nte))
        r['var_A_teorica_srs_fpc'] = S2 / nn * (1 - nn / n)
        r['deff_teorico'] = r['var_H'] / r['var_A_teorica_srs_fpc']
    else:
        # variancia de dominio por linearizacao (estimador razao de dominio, SRS n_te_medio de todos os nos)
        nn = float(np.mean(nte)); Pv = float((~sent).mean())
        z = np.where(~sent, e - mu, 0.0)
        r['var_A_teorica_linearizada'] = float(z.var(ddof=1) / (Pv ** 2) / nn * (1 - nn / n))
        r['deff_teorico'] = r['var_H'] / r['var_A_teorica_linearizada']
    r['deff_simulado'] = r['var_H'] / r['var_A_simulada']
    out[pop] = r
print(json.dumps(out, indent=1))
json.dump(out, open(BASE + 'votos_R1_R2/saida_R2_%s.json' % c, 'w'), indent=1)
