"""E3 generalizado — alvo + dataset v2 de QUALQUER tile (cidade x quadrante),
com seed 42 no A_terrain (decisao do dono, 2026-09-03).

Generaliza `e3_alvos_dataset_v2.py` (bauru Q2) e continua NAO reimplementando
nada: importa e chama as funcoes originais
  - prepare_transfer_dataset_v19.build_transfer_one_quadrant (alvo cols 0,3,4 + arestas)
  - enrich_rf_targets.compute_shadow_margin / compute_diffraction_loss /
    _dist_nearest_chunked / _load_antenna_pos_from_csv / _inject_antenna_metadata

Diferencas da versao bauru, cada uma por um obstaculo real:

 1. features (17 cols) e pos vem de `terrain.features_raw` / `terrain.pos` do
    dataset v1 publicado — que e exatamente o que
    prepare_transfer_dataset_v19.py:306 gravou a partir de `dem.x`. Isso vale
    com ou sem `_gpu.pt`, entao nao muda com a correcao do item 2.

    CORRECAO 2026-09-03 23h: este cabecalho afirmava "os grafos-estrutura das
    outras 3 cidades nao estao em disco". ERRADO — os 12 `_gpu.pt` de lins,
    campinas e sorocaba estao em `D:\\ARPIA_RF\\GRAPH_V19`. A busca so olhava
    GRAPH_D (onde so bauru foi copiado), nao achava, e caia na bissecao. Foi
    o que abortou campinas_Q1 as 23:21 (col2 reproduzida em 56,72%).

 2. `elev_std` LIDO quando ha carimbo, RECUPERADO quando nao ha, PROVADO
    sempre. enrich_rf_targets.py:449 usa `norm_meta.get("elev_std", 1.0)`:
    chave ausente vira 1.0 em silencio e a difracao sai errada sem erro. Aqui
    o `_gpu.pt` e procurado em STRUCT_DIRS (GRAPH_D primeiro, depois
    D:\\ARPIA_RF\\GRAPH_V19 em modo leitura, mais o que vier em --struct-dir);
    se achado, le a chave e FALHA se ela faltar. So sem nenhum carimbo em
    disco e que acha elev_std maximizando a FRACAO DE IGUALDADE BIT A BIT da
    coluna 2 contra a v1 (bissecao pela media so para achar a vizinhanca; a
    media sozinha erra ~150 ulps porque metade dos nos nao responde a
    elev_std), aceitando so com igualdade TOTAL. Em ambos os caminhos a col2
    recomputada tem de sair identica a v1 — o gate nao afrouxou. A difracao
    nao depende de NDVI, entao essa identidade vale tanto em tile limpo quanto
    em tile corrigido.

    Efeito na linhagem: bauru nao muda (o carimbo ja vinha de GRAPH_D, que
    continua sendo o primeiro da lista). lins Q1-Q4 foram construidos com
    elev_std INFERIDO porque o carimbo nao era procurado onde estava; o
    inferido bateu 100% da col2, mas quem quiser reproduzir aqueles quatro
    tiles exatamente como estao no disco tem de passar --sem-struct-externo.
    Ver `verificar_elev_std_arpia.py` (carimbado x inferido, bit a bit).

 3. TILE LIMPO x TILE CORRIGIDO. Em tile sem zero-fill nao ha E2: as features
    espectrais sao as do proprio .pt v1 e a UNICA mudanca e o sorteio semeado
    do A_terrain. Nesse caso o script exige, como gate, que y[:,1] (shadow),
    y[:,2] (difracao) e dist_nearest saiam identicas a v1 — se mudarem, algo
    fora do sorteio mudou e a corrida esta contaminada.

 4. SAIDA MAGRA POR PADRAO (--slim, ligado). Os 512-d de `terrain.x` sao 26,5
    dos 28,2 GB de cada .pt e a cadeia de treino NAO os le
    (train_gnn_c0_spatial.py:1011 e 1119-1120 montam x de features_raw + col17
    e sobrescrevem terrain.x). Transplanta-los para a v2 (a) exige ~400 GB que
    nao cabem, (b) triplica o tempo por tile (medido: 41 min so de gravacao
    contra ~20 min de calculo) e (c) enfia representacao da V1 dentro de
    artefato rotulado v2 nos tiles corrigidos. Em modo slim o campo terrain.x
    NAO e gravado: quem o ler quebra alto, em vez de treinar com embedding
    velho em silencio. `--com-embeddings` restaura o comportamento antigo.

Uso:
  ...\python.exe e3_alvos_dataset_v2_multi.py --city campinas --quad Q4 \
        --espectral-v2 dados/alvo/e2_espectral_campinas_Q4_v2.npy [--com-controle]
  ...\python.exe e3_alvos_dataset_v2_multi.py --city lins --quad Q1   # tile limpo
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

BASE_EVID = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_EVID / "dados" / "alvo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROJ = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2")
GRAPH_D = PROJ / "graph_data"
# F: primeiro para preservar a linhagem dos tiles ja construidos (bauru e lins
# foram lidos de la). ARPIA entra por ultimo, em leitura, como retaguarda: os 8
# datasets de campinas/sorocaba em F: sao copia bit-amostrada dos de ARPIA
# (duplicatas_f_arpia.json), entao se os de F: forem apagados a cadeia continua
# achando. ATENCAO: para lins os dois lados DIVERGEM em tamanho (26,3 GB em F:
# contra 29,4 GB em ARPIA) — nao sao copias, e a ordem aqui decide qual entra.
PT_DIRS_PADRAO = [Path(r"F:\TOPO_RF_DOWNLOAD_DRIVE\graph_data"), GRAPH_D,
                  Path(r"D:\ARPIA_RF\GRAPH_V19")]

# Onde procurar o grafo-estrutura `<cidade>_v19_<Q>_gpu.pt`, que carrega o
# carimbo `normalization.elev_std`. GRAPH_D vem primeiro para preservar a
# linhagem de bauru; ARPIA_GRAPH e projeto original e e lido, nunca escrito.
ARPIA_GRAPH = Path(r"D:\ARPIA_RF\GRAPH_V19")
STRUCT_DIRS_PADRAO = [GRAPH_D, ARPIA_GRAPH]

SEED_A_TERRAIN = 42
FREQ_MHZ = 1800.0
CHUNK = 200_000
COLS_ESPECTRAIS = list(range(8, 14))
COLS_NAO_ESPECTRAIS = [c for c in range(17) if c not in COLS_ESPECTRAIS]
IDX_TRI, IDX_ROUGHNESS, IDX_NDVI = 5, 6, 12

sys.path.insert(0, str(PROJ / "01_data"))
sys.path.insert(0, str(PROJ / "data_raw"))
sys.path.insert(0, str(PROJ))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stats_col(a: np.ndarray) -> dict:
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Identica a alvo_stats.py:22-25 (Pearson por momentos centrais)."""
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).mean() * (b * b).mean())
    return float((a * b).mean() / d) if d > 0 else float("nan")


def achar_pt(cidade: str, quad: str, extras: list[str]) -> Path:
    for d in [Path(x) for x in extras] + PT_DIRS_PADRAO:
        p = d / f"transfer_dataset_{cidade}_v19_{quad}_enriched.pt"
        if p.exists():
            return p
    raise SystemExit(f"ABORTA: dataset v1 de {cidade} {quad} nao encontrado")


def achar_struct(cidade: str, quad: str, extras: list[str],
                 sem_externo: bool = False) -> Path | None:
    """Localiza o grafo-estrutura que carrega o carimbo `normalization.elev_std`.

    Ordem: --struct-dir (na ordem dada) -> GRAPH_D -> D:\\ARPIA_RF\\GRAPH_V19.
    GRAPH_D antes de ARPIA preserva a linhagem de bauru bit a bit. Devolve None
    quando nenhum existe — nesse caso o chamador cai na recuperacao por
    bissecao, que continua valendo como caminho legitimo e provado.
    """
    dirs = [Path(x) for x in extras] + [GRAPH_D]
    if not sem_externo:
        dirs.append(ARPIA_GRAPH)
    for d in dirs:
        p = d / f"{cidade}_v19_{quad}_gpu.pt"
        if p.exists():
            return p
    return None


def recuperar_elev_std(E, tri_norm: torch.Tensor, rough: torch.Tensor,
                       dist: torch.Tensor, col2_v1: torch.Tensor) -> tuple[float, dict]:
    """Acha o elev_std que reproduz a col2 (difracao) da v1, por bissecao sobre
    a PROPRIA funcao original, e valida por igualdade bit a bit.

    A difracao cresce monotonicamente com elev_std (h_obs = tri_m*0.5 + ...,
    v cresce com h_obs, a perda cresce com v), entao a bissecao e legitima. Os
    nos usados na bissecao sao os de maior difracao abaixo do teto de 40 dB —
    onde a funcao ainda responde (fora do clamp)."""
    validos = torch.nonzero((col2_v1 > col2_v1.min()) & (col2_v1 < 39.9),
                            as_tuple=False).flatten()
    if validos.numel() < 1000:
        raise SystemExit("ABORTA: col2 da v1 sem variacao suficiente para "
                         "recuperar elev_std — informe --elev-std explicitamente")
    sel = validos[torch.randperm(validos.numel(), generator=torch.Generator().manual_seed(0))[:20_000]]
    t, r, d, alvo = tri_norm[sel], rough[sel], dist[sel], col2_v1[sel]
    alvo_m = float(alvo.mean())

    def media(s: float) -> float:
        return float(E.compute_diffraction_loss(t * s, r, d, FREQ_MHZ).mean())

    def frac_exata(s: float) -> float:
        """Fracao de nos da amostra em que a difracao recomputada bate BIT A BIT."""
        return float((E.compute_diffraction_loss(t * s, r, d, FREQ_MHZ)
                      == alvo).float().mean())

    # Passo 1 — bissecao pela MEDIA so para achar a vizinhanca.
    lo, hi = 1e-3, 5.0e3
    if not (media(lo) <= alvo_m <= media(hi)):
        raise SystemExit(f"ABORTA: alvo de difracao {alvo_m} fora da faixa "
                         f"[{media(lo)}, {media(hi)}] — modelo mudou?")
    for _ in range(200):
        mid = (lo + hi) / 2
        if media(mid) < alvo_m:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9 * max(1.0, hi):
            break
    base = (lo + hi) / 2

    # Passo 2 — a MEDIA e objetivo frouxo: metade dos nos tem h_obs dominado pela
    # rugosidade ou preso no clamp e nao responde a elev_std, entao a media casa
    # com ~150 ulps de erro. (Medido em lins_Q1, 2026-09-03: a bissecao deu
    # 68,591000, que reproduz a media na 6a casa e so 81,7% dos nos bit a bit;
    # o valor certo, 68,59114837646484, esta a ~1,5e-3 dali.) O objetivo certo e
    # a FRACAO DE IGUALDADE EXATA — e por isso a busca fina otimiza ela.
    melhor = (frac_exata(base), base)
    for s in np.arange(base - 0.005, base + 0.005, 2e-5):
        f = frac_exata(float(s))
        if f > melhor[0]:
            melhor = (f, float(s))
    # Passo 3 — caminhada por ulps de float32 ate a igualdade total.
    c = np.float32(melhor[1])
    for direcao in (np.float32(np.inf), np.float32(-np.inf)):
        x = c
        for _ in range(500):
            x = np.nextafter(x, direcao)
            f = frac_exata(float(x))
            if f > melhor[0]:
                melhor = (f, float(x))
            if melhor[0] >= 1.0:
                break
        if melhor[0] >= 1.0:
            break

    frac_amostra, s = melhor
    if frac_amostra < 1.0:
        raise SystemExit(
            f"ABORTA: melhor elev_std ({s!r}) reproduz apenas "
            f"{100*frac_amostra:.4f}% da col2 da v1 na amostra. Sem igualdade "
            f"total a difracao da v2 nao e comparavel com a v1 — informe "
            f"--elev-std ou investigue mudanca no modelo.")
    return s, {"metodo": "bisseção pela media para achar a vizinhanca, depois "
                         "maximizacao da FRACAO DE IGUALDADE BIT A BIT da col2 v1 "
                         "(grade fina + caminhada por ulps de float32)",
               "elev_std_m": s, "bissecao_bruta": base,
               "frac_igualdade_na_amostra": frac_amostra,
               "n_nos_amostra": int(t.shape[0])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True,
                    choices=["lins", "campinas", "bauru", "sorocaba"])
    ap.add_argument("--quad", required=True, choices=["Q1", "Q2", "Q3", "Q4"])
    ap.add_argument("--espectral-v2", default="",
                    help="npy do E2 com as 6 colunas espectrais v2; ausente = tile LIMPO")
    ap.add_argument("--pt-dir", action="append", default=[])
    ap.add_argument("--struct-dir", action="append", default=[],
                    help="onde procurar <cidade>_v19_<Q>_gpu.pt (carimbo do "
                         "elev_std); tem prioridade sobre GRAPH_D e ARPIA")
    ap.add_argument("--sem-struct-externo", action="store_true",
                    help="nao procura em D:\\ARPIA_RF\\GRAPH_V19; reproduz o "
                         "comportamento anterior a 2026-09-03 23h (lins Q1-Q4 "
                         "foram construidos assim, com elev_std inferido)")
    ap.add_argument("--out-dir", default=str(GRAPH_D))
    ap.add_argument("--elev-std", type=float, default=None,
                    help="so para diagnostico; o padrao e recuperar e provar")
    ap.add_argument("--com-controle", action="store_true",
                    help="tambem reconstroi o alvo com as features v1 e a mesma semente")
    ap.add_argument("--com-embeddings", action="store_true",
                    help="transplanta os 512-d da v1 (arquivo de 28 GB; ver docstring)")
    args = ap.parse_args()
    cidade, quad = args.city, args.quad
    t0 = time.perf_counter()

    import builtins
    import prepare_transfer_dataset_v19 as P
    import enrich_rf_targets as E

    def _tsprint(*a, **k):
        k["flush"] = True
        builtins.print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, **k)
    P.print = _tsprint    # carimba hora nos prints da funcao original (sem efeito numerico)

    ref_pt = achar_pt(cidade, quad, args.pt_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rf_v2 = out_dir / f"transfer_dataset_{cidade}_v19_{quad}_enriched_v2.pt"
    tile_limpo = not args.espectral_v2

    res = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "cidade": cidade, "quadrante": quad,
        "tile": f"{cidade}_{quad}",
        "categoria": "LIMPO (sem zero-fill; so o sorteio semeado muda)" if tile_limpo
                     else "CORRIGIDO (features espectrais v2 do E2)",
        "referencia_v1": str(ref_pt),
        "reuso": {
            "alvo_cols_0_3_4_e_arestas": "prepare_transfer_dataset_v19.build_transfer_one_quadrant (importada)",
            "alvo_col_1_shadow": "enrich_rf_targets.compute_shadow_margin (ITU-R P.833-10)",
            "alvo_col_2_diffraction": "enrich_rf_targets.compute_diffraction_loss (ITU-R P.526)",
            "dist_nearest": "enrich_rf_targets._dist_nearest_chunked + _load_antenna_pos_from_csv",
            "freq_mhz": FREQ_MHZ, "chunk": CHUNK, "device": "cpu",
        },
        "decisao_seed_a_terrain": {
            "seed": SEED_A_TERRAIN,
            "onde": "np.random.seed(42) antes de build_transfer_one_quadrant; consumido "
                    "por prepare_transfer_dataset_v19.py:141 (np.random.uniform 0.02-0.15) "
                    "que alimenta A_terrain em generate_realistic_coverage.py:109-116",
            "ratificada_pelo_dono_em": "2026-09-03 (pratica para todo rebuild v2)",
            "diverge_do_processo_v1": True,
        },
    }

    # ---------------- 1. insumos da v1 ----------------
    log(f"tile {cidade} {quad} | {res['categoria']}")
    log(f"lendo dataset v1: {ref_pt}")
    rf1 = torch.load(ref_pt, map_location="cpu", weights_only=False, mmap=True)
    fr1 = rf1["terrain"].features_raw
    x_v1 = (fr1.numpy() if torch.is_tensor(fr1) else np.asarray(fr1)).astype(np.float32).copy()
    pos = rf1["terrain"].pos.clone()
    y_v1 = np.asarray(rf1["terrain"].y.numpy() if torch.is_tensor(rf1["terrain"].y)
                      else rf1["terrain"].y).astype(np.float32).copy()
    if not hasattr(rf1["terrain"], "dist_nearest_m"):
        raise SystemExit("ABORTA: dataset v1 sem dist_nearest_m — nao ha como "
                         "conferir o enriquecimento contra a v1")
    dist_v1 = rf1["terrain"].dist_nearest_m.float().numpy().copy()
    n_edges_v1 = int(rf1["antenna", "propagates_to", "terrain"].edge_index.shape[1])
    emb_shape = list(rf1["terrain"].x.shape)
    del fr1, rf1
    gc.collect()
    N = x_v1.shape[0]
    log(f"   N={N:,}  arestas v1={n_edges_v1:,}  embeddings v1={emb_shape}")

    # ---------------- 2. features v2 ----------------
    if tile_limpo:
        x_v2 = x_v1            # mesma matriz; nada espectral muda
    else:
        esp = np.load(args.espectral_v2)
        if esp.shape != (N, 6):
            raise SystemExit(f"espectral v2 com forma {esp.shape}, esperado {(N, 6)}")
        x_v2 = x_v1.copy()
        x_v2[:, COLS_ESPECTRAIS] = esp
        del esp
        gc.collect()

    sha_ne_v1 = sha_arr(x_v1[:, COLS_NAO_ESPECTRAIS])
    sha_ne_v2 = sha_arr(x_v2[:, COLS_NAO_ESPECTRAIS])
    if sha_ne_v1 != sha_ne_v2:
        raise SystemExit("ABORTA: colunas nao espectrais divergiram")
    res["prova_identidade_features"] = {
        "definicao": "sha256 dos bytes float32 das colunas 0-7 e 14-16 (nao espectrais) "
                     "e 8-13 (espectrais), tomadas do dataset v1 publicado",
        "sha256_nao_espectrais_v1": sha_ne_v1,
        "sha256_nao_espectrais_v2": sha_ne_v2,
        "nao_espectrais_identicas_bit_a_bit": True,
        "sha256_espectrais_v1": sha_arr(x_v1[:, COLS_ESPECTRAIS]),
        "sha256_espectrais_v2": sha_arr(x_v2[:, COLS_ESPECTRAIS]),
        "espectrais_mudaram": bool(sha_arr(x_v1[:, COLS_ESPECTRAIS])
                                   != sha_arr(x_v2[:, COLS_ESPECTRAIS])),
        "espectrais_v2_vindas_de": args.espectral_v2 or "(tile limpo: as proprias da v1)",
    }
    log(f"   espectrais mudaram: {res['prova_identidade_features']['espectrais_mudaram']}")

    # ---------------- 3. elev_std: lido ou recuperado, sempre provado ----------------
    struct = achar_struct(cidade, quad, args.struct_dir, args.sem_struct_externo)
    tri_norm = torch.from_numpy(x_v2[:, IDX_TRI])
    rough = torch.from_numpy(x_v2[:, IDX_ROUGHNESS])
    dist_t_v1 = torch.from_numpy(dist_v1)
    col2_v1 = torch.from_numpy(y_v1[:, 2])
    if args.elev_std is not None:
        elev_std, meta_es = float(args.elev_std), {"metodo": "informado por --elev-std"}
    elif struct is not None:
        log(f"lendo normalization de {struct.name}")
        sd = torch.load(struct, map_location="cpu", weights_only=False, mmap=True)
        nm = getattr(sd, "normalization", None)
        del sd
        gc.collect()
        if not nm or "elev_std" not in nm:
            raise SystemExit("ABORTA: normalization/elev_std ausente no grafo — "
                             "enrich_rf_targets.py:449 assumiria 1.0 em silencio")
        elev_std = float(nm["elev_std"])
        meta_es = {"metodo": "normalization do grafo-estrutura", "fonte": str(struct),
                   "elev_mean_m": float(nm.get("elev_mean", float("nan")))}
    else:
        log("recuperando elev_std por bissecao (nenhum _gpu.pt achado em "
            f"{[str(d) for d in ([Path(x) for x in args.struct_dir] + [GRAPH_D] + ([] if args.sem_struct_externo else [ARPIA_GRAPH]))]}) ...")
        elev_std, meta_es = recuperar_elev_std(E, tri_norm, rough, dist_t_v1, col2_v1)
    meta_es["elev_std_m"] = elev_std
    col2_check = E.compute_diffraction_loss(tri_norm * elev_std, rough, dist_t_v1, FREQ_MHZ)
    meta_es["col2_recomputada_identica_a_v1"] = bool(torch.equal(col2_check, col2_v1))
    res["elev_std"] = meta_es
    log(f"   elev_std={elev_std:.4f} m | col2 identica a v1: "
        f"{meta_es['col2_recomputada_identica_a_v1']}")
    if not meta_es["col2_recomputada_identica_a_v1"]:
        raise SystemExit("ABORTA: a difracao recomputada com este elev_std nao "
                         "reproduz a col2 da v1 — insumo ou modelo divergiu")
    del col2_check, col2_v1
    gc.collect()

    # ---------------- 4. seam de I/O e construcao do alvo ----------------
    scratch = OUT_DIR / f"_scratch_e3_{cidade}_{quad}"
    scratch.mkdir(parents=True, exist_ok=True)
    fake_emb = scratch / f"{cidade}_v19_{quad}_embeddings_512.pt"
    fake_struct = scratch / f"{cidade}_v19_{quad}_gpu.pt"
    for f in (fake_emb, fake_struct):
        f.touch()
    P.GRAPH_DIR = scratch
    P.OUTPUT_DIR = scratch
    P.STRUCT_DIR = None

    torch_load_real = torch.load
    estado = {"x": None}

    def torch_load_seam(path, *a, **k):
        p = Path(path)
        if p.name == fake_emb.name:
            return {"embeddings": torch.zeros((0,), dtype=torch.float32)}
        if p.name == fake_struct.name:
            return {"dem": SimpleNamespace(x=torch.from_numpy(estado["x"]), pos=pos)}
        return torch_load_real(path, *a, **k)

    df_ant = P.load_antennas(cidade)
    log(f"Antenas {cidade} validas no CSV: {len(df_ant)}")
    res["antenas_csv"] = {"n": int(len(df_ant)), "fonte": str(P.ANTENNA_PATH)}

    def construir(x_feats: np.ndarray, etiqueta: str):
        estado["x"] = x_feats
        torch.load = torch_load_seam
        try:
            log(f"Construindo alvo/arestas [{etiqueta}] (seed {SEED_A_TERRAIN}) ...")
            np.random.seed(SEED_A_TERRAIN)
            t = time.perf_counter()
            d = P.build_transfer_one_quadrant(quad, df_ant, city=cidade)
            log(f"   [{etiqueta}] pronto em {(time.perf_counter()-t)/60:.1f} min")
        finally:
            torch.load = torch_load_real
        return d

    data = construir(x_v2, "v2")
    y_v2 = data["terrain"].y.numpy()
    n_edges_v2 = int(data["antenna", "propagates_to", "terrain"].edge_index.shape[1])

    # ---------------- 5. enriquecimento (cols 1 e 2) ----------------
    log("Enriquecimento: dist_nearest + shadow (P.833) + diffraction (P.526)")
    csv_candidates = [rf_v2.parent / "antenas_interior_sp_final.csv",
                      PROJ / "data_raw" / "antenas_interior_sp_final.csv"]
    antenna_pos, antenna_source = None, "desconhecido"
    for csv_path in csv_candidates:
        p = E._load_antenna_pos_from_csv(csv_path, pos, target_freq_mhz=FREQ_MHZ)
        if p is not None:
            antenna_pos = p
            antenna_source = f"CSV ({csv_path.name}, {p.shape[0]} torres unicas)"
            break
    if antenna_pos is None:
        antenna_pos = data["antenna"].pos.float()
        antenna_source = "antenna.pos do dataset"
    log(f"   {antenna_source}")

    dist = E._dist_nearest_chunked(pos, antenna_pos, chunk=CHUNK, device=torch.device("cpu"))
    dist_np = dist.numpy()
    dist_igual = bool(np.array_equal(dist_np, dist_v1))
    res["dist_nearest"] = {"fonte_antenas": antenna_source,
                           "identica_a_v1_bit_a_bit": dist_igual}
    if not dist_igual:
        log("[ATENCAO] dist_nearest difere da v1: shadow e difracao mudam por "
            "motivo alheio ao NDVI — numero deste tile precisa dessa etiqueta")

    shadow = E.compute_shadow_margin(torch.from_numpy(x_v2[:, IDX_NDVI]), dist, FREQ_MHZ)
    diffr = E.compute_diffraction_loss(torch.from_numpy(x_v2[:, IDX_TRI]) * elev_std,
                                       torch.from_numpy(x_v2[:, IDX_ROUGHNESS]),
                                       dist, FREQ_MHZ)
    y_enr = data["terrain"].y.float().clone()
    y_enr[:, 1] = shadow
    y_enr[:, 2] = diffr
    data["terrain"].y = y_enr
    data["terrain"].dist_nearest_m = dist
    E._inject_antenna_metadata(data, rf_v2, csv_candidates, log)
    y_v2e = y_enr.numpy()

    # gate do tile limpo: sem mudanca espectral, shadow/difracao/dist tem de ser v1
    if tile_limpo:
        gate = {
            "shadow_identica_a_v1": bool(np.array_equal(y_v2e[:, 1], y_v1[:, 1])),
            "diffraction_identica_a_v1": bool(np.array_equal(y_v2e[:, 2], y_v1[:, 2])),
            "dist_identica_a_v1": dist_igual,
        }
        gate["aprovado"] = all(gate.values())
        res["gate_tile_limpo"] = gate
        log(f"   gate tile limpo: {gate}")
        if not gate["aprovado"]:
            raise SystemExit("ABORTA: tile limpo mudou algo alem do sorteio semeado")

    # ---------------- 6. gravacao ----------------
    if args.com_embeddings:
        log("Transplantando embeddings da v1 (terrain.x) — arquivo grande")
        rf1b = torch.load(ref_pt, map_location="cpu", weights_only=False, mmap=True)
        emb = rf1b["terrain"].x
        if list(emb.shape) != emb_shape:
            raise SystemExit(f"embeddings mudaram de forma: {emb.shape} != {emb_shape}")
        data["terrain"].x = emb
        res["embeddings"] = {"origem": str(ref_pt), "shape": emb_shape,
                             "regenerados_na_v2": False,
                             "aviso": "produzidos pelo encoder sobre as features V1"}
    else:
        if "x" in data["terrain"]:
            del data["terrain"].x
        res["embeddings"] = {
            "gravados": False, "shape_na_v1": emb_shape, "origem_v1": str(ref_pt),
            "motivo": "terrain.x nao e lido pela cadeia de treino "
                      "(train_gnn_c0_spatial.py:1011 e 1119-1120 montam x de "
                      "features_raw + col17). Omitir evita 26,5 GB por tile e "
                      "impede que um consumidor leia embedding da v1 achando que "
                      "e v2: quem ler terrain.x quebra alto.",
        }
    log(f"Salvando {rf_v2.name} (slim={not args.com_embeddings}) ...")
    torch.save(data, rf_v2)
    tam = rf_v2.stat().st_size
    log(f"   salvo: {tam/1e9:.2f} GB")

    # ---------------- 7. estatisticas (mesmas de alvo_stats.py) ----------------
    pl, rssi = y_v2e[:, 0], y_v2e[:, 3]
    sent = pl >= 299.0
    val = ~sent
    ndvi_np = x_v2[:, IDX_NDVI]
    zerofill = ((x_v2[:, 12] == 0.0) & (x_v2[:, 13] == 0.0) & (x_v2[:, 11] == 0.0))
    pl1 = y_v1[:, 0]

    res[f"quadrante_{quad}_v2"] = {
        "n_nodes": int(N),
        "frac_sentinela": float(sent.mean()),
        "frac_ndvi_gt_0p5": float((ndvi_np > 0.5).mean()),
        "corr_rssi_alvo_vs_neg_dist_todos": corr(rssi, -dist_np),
        "corr_rssi_alvo_vs_neg_log10d_todos": corr(rssi, -np.log10(np.maximum(dist_np, 1.0))),
        "corr_rssi_alvo_vs_neg_dist_validos": corr(rssi[val], -dist_np[val]),
        "corr_rssi_alvo_vs_neg_log10d_validos": corr(rssi[val], -np.log10(np.maximum(dist_np[val], 1.0))),
        "corr_pl_alvo_vs_log10d_validos": corr(pl[val], np.log10(np.maximum(dist_np[val], 1.0))),
    }
    res["definicoes"] = {
        "sentinela": "y[:,0] >= 299 dB (sem cobertura; RSSI imputado -110 dBm)",
        "veg": "NDVI (features_raw col 12) > 0.5",
        "corr": "Pearson; tambem so sobre nos validos (nao-sentinela)",
    }
    res["verificacao_zerofill_v2"] = {
        "definicao": "fracao de nos com col12 (NDVI), col13 (NDWI) e col11 (B08) = 0.0 exato "
                     "— derivado de scripts/verificar_ndvi_zerofill.py",
        "frac_zerofill_tres_bandas": float(zerofill.mean()),
        "n_nos_zerofill": int(zerofill.sum()),
        "frac_zerofill_na_v1": float((((x_v1[:, 12] == 0.0) & (x_v1[:, 13] == 0.0)
                                       & (x_v1[:, 11] == 0.0)).mean())),
        "ndvi_media_nos_nao_zerofill": (float(ndvi_np[~zerofill].mean())
                                        if (~zerofill).any() else None),
        "shadow_margin_media_db": float(y_v2e[:, 1].mean()),
    }
    nomes = ["path_loss", "shadow_margin", "diffraction", "rssi", "coverage"]
    res["alvo_v2_por_canal"] = {f"col{i}_{n}": stats_col(y_v2e[:, i])
                                for i, n in enumerate(nomes)}
    res["alvo_v1_por_canal"] = {f"col{i}_{n}": stats_col(y_v1[:, i])
                                for i, n in enumerate(nomes)}
    res["comparacao_v2_vs_v1_publicado"] = {
        "aviso": "mistura o conserto do NDVI (se houver) com o novo sorteio de "
                 "A_terrain; use --com-controle para separar",
        "col2_diffraction_identica_bit_a_bit": bool(np.array_equal(y_v2e[:, 2], y_v1[:, 2])),
        "dist_nearest_identica_bit_a_bit": dist_igual,
        "delta_frac_sentinela": float(sent.mean() - (pl1 >= 299.0).mean()),
        "frac_nos_com_alvo_pl_alterado": float((y_v2e[:, 0] != pl1).mean()),
        "media_abs_delta_pl_db": float(np.abs(y_v2e[:, 0] - pl1).mean()),
        "media_abs_delta_rssi_db": float(np.abs(rssi - y_v1[:, 3]).mean()),
        "media_abs_delta_shadow_db": float(np.abs(y_v2e[:, 1] - y_v1[:, 1]).mean()),
        "n_edges_v1": n_edges_v1, "n_edges_v2": n_edges_v2,
    }
    res["arquivo_v2"] = {"path": str(rf_v2), "bytes": int(tam),
                         "slim": not args.com_embeddings}

    out_json = OUT_DIR / f"alvo_stats_{cidade}_s42_{quad}_v2.json"
    out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[OK] {out_json}")

    del data, y_enr
    gc.collect()

    # ---------------- 8. controle negativo ----------------
    if args.com_controle and not tile_limpo:
        d1 = construir(x_v1, "controle v1_s42")
        y_c = d1["terrain"].y.numpy().copy()
        n_edges_c = int(d1["antenna", "propagates_to", "terrain"].edge_index.shape[1])
        del d1
        gc.collect()
        npy_c = out_dir / f"y_controle_v1_s42_{cidade}_{quad}.npy"
        np.save(npy_c, y_c)
        res["controle_v1_s42"] = {
            "descricao": "alvo reconstruido com as features V1 e a MESMA semente 42; "
                         "slope_approx identico ao da v2, logo a diferenca isola o NDVI",
            "npy": str(npy_c), "sha256_npy": sha_arr(y_c), "n_edges": n_edges_c,
            "por_canal": {f"col{i}_{n}": stats_col(y_c[:, i])
                          for i, n in enumerate(["path_loss", "zero", "zero", "rssi", "coverage"])},
            "efeito_isolado_do_ndvi": {
                "frac_nos_com_pl_alterado": float((y_v2e[:, 0] != y_c[:, 0]).mean()),
                "media_abs_delta_pl_db": float(np.abs(y_v2e[:, 0] - y_c[:, 0]).mean()),
                "media_abs_delta_rssi_db": float(np.abs(y_v2e[:, 3] - y_c[:, 3]).mean()),
                "delta_frac_sentinela": float((y_v2e[:, 0] >= 299.0).mean()
                                              - (y_c[:, 0] >= 299.0).mean()),
                "delta_n_edges": n_edges_v2 - n_edges_c,
            },
            "efeito_do_novo_sorteio_a_terrain": {
                "descricao": "controle (features v1, seed 42) contra o alvo v1 publicado "
                             "(features v1, sem semente): so A_terrain difere",
                "frac_nos_com_pl_alterado": float((y_c[:, 0] != pl1).mean()),
                "media_abs_delta_pl_db": float(np.abs(y_c[:, 0] - pl1).mean()),
                "media_abs_delta_rssi_db": float(np.abs(y_c[:, 3] - y_v1[:, 3]).mean()),
                "delta_n_edges": n_edges_c - n_edges_v1,
            },
        }
        out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"[OK] controle gravado em {out_json}")
    elif args.com_controle and tile_limpo:
        log("--com-controle ignorado: tile limpo — o controle SERIA a propria v2 "
            "(features identicas e mesma semente); o contraste util e contra a v1.")

    for f in (fake_emb, fake_struct):
        f.unlink(missing_ok=True)
    try:
        scratch.rmdir()
    except OSError:
        pass

    log(f"TOTAL {(time.perf_counter()-t0)/60:.1f} min")
    print(json.dumps({
        "tile": f"{cidade}_{quad}", "categoria": res["categoria"],
        "frac_zerofill_v1": res["verificacao_zerofill_v2"]["frac_zerofill_na_v1"],
        "frac_zerofill_v2": res["verificacao_zerofill_v2"]["frac_zerofill_tres_bandas"],
        "frac_sentinela_v2": res[f"quadrante_{quad}_v2"]["frac_sentinela"],
        "elev_std_m": elev_std,
        "col2_identica": res["comparacao_v2_vs_v1_publicado"]["col2_diffraction_identica_bit_a_bit"],
        "arquivo": str(rf_v2), "GB": round(tam / 1e9, 2),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
