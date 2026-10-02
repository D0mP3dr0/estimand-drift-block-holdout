"""forum-eng-ia 26/09: contagem de parametros efetivamente treinaveis do GNNRFModel
(terrain_dim=18, antenna_dim=6, hidden=256, 4 camadas, 4 heads) num batch sintetico
com a MESMA topologia de batch do NeighborLoader de fanout 1 (arestas ET_TA presentes
como tipo, mas com 0 arestas) e, como contraste, com ET_TA povoado (fanout 2 saltos).
CPU, sem dado real: mede so a estrutura do grafo computacional."""
import sys, json, torch
sys.path.insert(0, '/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/02_models')
sys.path.insert(0, '/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2')
from torch_geometric.data import HeteroData
from gnn_rf_model import GNNRFModel
torch.manual_seed(0)
AT, TT, TA = ('antenna','propagates_to','terrain'), ('terrain','connects_to','terrain'), ('terrain','in_range_of','antenna')
def batch(com_ta, ta_attr=False):
    bs, nh, na = 64, 200, 5
    nt = bs + nh
    d = HeteroData()
    d['terrain'].x = torch.randn(nt, 18); d['antenna'].x = torch.randn(na, 6)
    src = torch.randint(0, na, (bs*3,)); dst = torch.arange(bs).repeat(3)
    d[AT].edge_index = torch.stack([src, dst]); d[AT].edge_attr = torch.randn(bs*3, 2)
    s2 = torch.randint(bs, nt, (bs*3,)); d2 = torch.arange(bs).repeat(3)
    d[TT].edge_index = torch.stack([s2, d2]); d[TT].edge_attr = torch.randn(bs*3, 2)
    if com_ta:
        s3 = torch.randint(0, nt, (40,)); d3 = torch.randint(0, na, (40,))
        d[TA].edge_index = torch.stack([s3, d3])
        if ta_attr: d[TA].edge_attr = torch.randn(40, 2)
    else:
        d[TA].edge_index = torch.zeros(2, 0, dtype=torch.long)
    return d, bs
out = {}
for nome, com_ta, ta_attr in [('fanout_1_salto_producao', False, False), ('fanout_com_ET_TA_povoado', True, False), ('ARMADILHA_ET_TA_com_edge_attr', True, True)]:
    torch.manual_seed(0)
    m = GNNRFModel(terrain_dim=18, antenna_dim=6, hidden_dim=256, num_layers=4, heads=4,
                   edge_dim=2, output_dim=5, dropout=0.1)
    m.train()
    d, bs = batch(com_ta, ta_attr)
    pred = m(d)['predictions'][:bs]
    pred.float().pow(2).mean().backward()
    tot = sum(p.numel() for p in m.parameters())
    mortos = {}
    for n, p in m.named_parameters():
        if p.grad is None or float(p.grad.abs().max()) == 0.0:
            mortos[n] = {'numel': p.numel(), 'shape': list(p.shape), 'grad': 'None' if p.grad is None else 'zero'}
    nm = sum(v['numel'] for v in mortos.values())
    por_bloco = {}
    for n, p in m.named_parameters():
        k = n.split('.')[0] + ('.' + n.split('.')[1] if n.startswith(('encoder','decoder')) else '')
        por_bloco[k] = por_bloco.get(k, 0) + p.numel()
    out[nome] = {'n_params_total': tot, 'n_params_sem_gradiente': nm,
                 'n_params_efetivos': tot - nm, 'mortos': mortos, 'por_bloco': por_bloco}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for k, v in out.items():
    print(k, v['n_params_total'], v['n_params_sem_gradiente'], v['n_params_efetivos'])
    for n, i in v['mortos'].items(): print('   ', n, i)
print(json.dumps(out['fanout_1_salto_producao']['por_bloco']))
