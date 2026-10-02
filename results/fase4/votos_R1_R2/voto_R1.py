# Implementacao independente (nao importa scripts dos executores).
# Detalhes copiados do codigo existente: conversao grau->metros com origem no minimo (x com cos(lat) do ponto),
# ordem dos blocos = np.unique de id = floor(x/g)*(maxfy+1)+floor(y/g); round() do Python nas contagens;
# alvo RSSI = coluna 3 de terrain.y, sentinela = coluna 0 >= 299; constante testada: -110 e mediana(RSSI treino).
import sys, json, numpy as np, torch
from scipy.spatial import cKDTree
D = "/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/transfer_dataset_%s_v19_%s_enriched_cftudo.pt"
BASE = '/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/fase4/'

def carrega(c):
    cid, q = c.split('_')
    rf = torch.load(D % (cid, q), map_location='cpu', weights_only=False, mmap=True)
    y = torch.as_tensor(rf['terrain'].y).float().clone().numpy()
    pos = torch.as_tensor(rf['terrain'].pos).float().clone()
    lon = pos[:, 0].double().numpy(); lat = pos[:, 1].double().numpy()
    x = (lon - lon.min()) * 111000 * np.cos(np.radians(lat)) / 1000
    yy = (lat - lat.min()) * 111000 / 1000
    return np.stack([x, yy], 1), y[:, 3].astype(np.float64), y[:, 0] >= 299

def particao(P, g, b, seed):
    gx = np.floor(P[:, 0] / g).astype(np.int64); gy = np.floor(P[:, 1] / g).astype(np.int64)
    gid = gx * (gy.max() + 1) + gy
    bl = np.unique(gid); rs = np.random.RandomState(seed); rs.shuffle(bl)
    N = len(bl); ktr = max(1, int(round(.7 * N))); kva = max(1, int(round(.15 * N)))
    if ktr + kva >= N: ktr = max(1, N - 2); kva = 1
    tr = np.isin(gid, bl[:ktr]); va = np.isin(gid, bl[ktr:ktr + kva]); te = np.isin(gid, bl[ktr + kva:])
    if b > 0:
        d, _ = cKDTree(P[tr]).query(P[va], k=1, workers=-1); iv = np.where(va)[0]; va[iv[d < b]] = False
        t2 = tr | va
        d, _ = cKDTree(P[t2]).query(P[te], k=1, workers=-1); it = np.where(te)[0]; te[it[d < b]] = False
    return tr, va, te

if __name__ == "__main__":
    R = json.load(open(BASE + 'R1_b0_resumo.json'))
    seeds = R['sementes']; out = {}
    for c in sys.argv[1:]:
        P, rssi, sent = carrega(c); out[c] = {}
        for b in (0.0, 2.0):
            m_med = []; m_110 = []; fr = []; nte = []
            for s in seeds:
                tr, va, te = particao(P, 10.0, b, s)
                v = ~sent[te]; fr.append(v.mean() if te.any() else np.nan); nte.append(int(te.sum()))
                if v.any():
                    m_110.append(np.abs(rssi[te][v] + 110).mean())
                    m_med.append(np.abs(rssi[te][v] - np.median(rssi[tr])).mean())
            f = np.array(fr)
            ref = R['por_celula'][c]['b0' if b == 0 else 'b2']
            out[c]['b%d' % int(b)] = {'n_sorteios_com_valido': len(m_110),
              'dp_mae_const110': float(np.std(m_110, ddof=1)),
              'dp_mae_const_mediana_treino': float(np.std(m_med, ddof=1)),
              'dp_fracao_valida_todos': float(np.std(f, ddof=1)),
              'dp_fracao_valida_so_com_valido': float(np.std(f[f > 0], ddof=1)),
              'media_n_teste': float(np.mean(nte)),
              'json': {k: ref[k] for k in ('n_sorteios_com_no_valido', 'dp_mae_constante_validos', 'dp_fracao_valida_teste', 'dp_fracao_valida_teste_todos_os_sorteios', 'media_n_teste')}}
            print(c, b, json.dumps(out[c]['b%d' % int(b)]), flush=True)
    json.dump(out, open(BASE + 'votos_R1_R2/saida_R1.json', 'w'), indent=1)
