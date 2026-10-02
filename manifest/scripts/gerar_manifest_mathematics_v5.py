#!/usr/bin/env python3
"""Gera manifest_mathematics_v5.jsonl + .meta.json do Artigo 2 (MDPI Mathematics).

COPIA DECLARADA de gerar_manifest_mathematics_v4.py (sha256 do v4 no meta).
O v4 NAO e editado: o v5 o reusa POR IMPORT (importlib), como o v4 faz com o v3,
e acrescenta os grupos que o manuscrito v3-12c afirma na \\dataavailability e na
sec. 6.1 e que o v4 nao cobria. Protocolo: roadmap
`_v3_2026-09-25/_propostas_2026-10-01/ROADMAP_v3-12_major_revision.md`, sec. 5
("Manifesto v5 cobrindo fases 1-3, corridas de GPU, tabelas e scripts atuais"),
decisao D9; ordem do dono de 01/10/2026 17:12 ("execute td aceito a proposta
do roadmap"), registrada na ata `escritor_cientifico/ATAS/ata_2026-10-01_artigo2_mathematics_t25.md`
(Adendo 2). O ROADMAP em si nao traz a marca "aprovado" -- a aprovacao esta na ata.

Tarefa de REGISTRO: nenhuma leitura de dado alem do necessario para o sha256;
nenhum calculo cientifico; nenhum treino; nenhuma GPU.

Conteudo do v5 = v4 (reexecutado: 12 grupos do v3 + 4 do v4, dinamicos) +
  fase1, fase2, fase3, fase4, votos, gpu (agregados/scripts, predicoes .npz,
  checkpoints .pt), criterios restantes, tabelas e figuras v3-12 (com
  manifestos), FOLHA_DE_FATOS_v3-12_adendo.*, checkpoints das corridas
  publicadas (EVIDENCIA_RESUBMISSAO/treinos/*/checkpoints), os 16 *_gpu.pt
  (arestas) + HASHES_SHA256.txt/MANIFESTO.md, e as dependencias vivas
  (spatial_cv, config, rf_decoder, trainer congelado, os quatro modulos do
  gerador, cadeia de geracao do grafo e do alvo).

Campos por linha (formato do v4 mantido; `caminho` segue absoluto para os
validadores v1-v3; ACRESCENTADOS): `raiz` (id declarado no meta, chave
`raizes`), `caminho_rel` (relativo a essa raiz), `sha_origem`
("recalculado v5" | "v4 (mtime e tamanho inalterados)" | "pendente").
Regra (a): arquivo > 100 MB cujo tamanho e mtime nao mudaram desde o v4
REAPROVEITA o sha do v4; se mudou, recalcula.

Uso:
    python gerar_manifest_mathematics_v5.py [--saida CAMINHO] [--dry-run]
        [--adiar-grupos g1,g2]   # deixa o grupo "pendente de hash" (sha256 null)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
V4_SCRIPT_PATH = SCRIPT_PATH.parent / "gerar_manifest_mathematics_v4.py"

MDPI = SCRIPT_PATH.parent.parent
V3DIR = MDPI / "_v3_2026-09-25"
GNN_RF = MDPI.parent
PROJ = GNN_RF.parent                                   # TOPO_RF_PROJETO
EVID = GNN_RF / "gnn_rf_ieee_access" / "FIRST_RESPONSE_REVIEW_IEEE_ACESSES" / "EVIDENCIA_RESUBMISSAO"
GNN_RF_V2 = PROJ / "GNN_RF_V2"
GRAFOS = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
HERMES_AGENTES = Path("/trabalho/HERMES/AGENTES")

V4_JSONL = MDPI / "manifest_mathematics_v4.jsonl"
DEFAULT_SAIDA = MDPI / "manifest_mathematics_v5.jsonl"

LIMITE_GRANDE = 100 * 1024 * 1024  # regra (a): > 100 MB reaproveita o sha do v4

# raiz declarada (id -> caminho). Casamento por maior prefixo.
RAIZES = {
    "MDPI_Mathematics": MDPI,
    "EVIDENCIA_RESUBMISSAO": EVID,
    "GNN_RF_V2": GNN_RF_V2,
    "GRAFOS_graph_data_v3": GRAFOS,
    "HERMES_AGENTES": HERMES_AGENTES,
    "GNN_RF": GNN_RF,
    "TOPO_RF_PROJETO": PROJ,
}
RAIZ_FS = "FS"  # fallback: caminho absoluto sem a barra inicial

DIGESTS_MANUSCRITO = {  # prefixo citado no \dataavailability -> modulo esperado
    "95ea0423": "generate_realistic_coverage.py",
    "45c16d43": "prepare_transfer_dataset_v19.py",
    "5d38012e": "enrich_rf_targets.py",
    "ebeaf759": "contrafactual_alvo_completo.py",
}

# dependencias vivas apontadas pelos pareceres de 01/10 e pelo pedido.
# (caminho, papel, prefixo_sha_esperado|None)
DEPENDENCIAS_VIVAS = [
    (GNN_RF_V2 / "03_training" / "spatial_cv.py", "particao_espacial (blocos+buffer); importa `config`", None),
    (GNN_RF_V2 / "config" / "config.py", "config importado por spatial_cv (`from config import config`)", None),
    (GNN_RF_V2 / "config" / "__init__.py", "pacote `config` (import de spatial_cv)", None),
    (GNN_RF_V2 / "02_models" / "rf_decoder.py", "decodificador RF das corridas publicadas", None),
    (EVID / "scripts" / "train_gnn_c0_spatial.py", "treinador congelado (copia em scripts/; versao registrada nas corridas)", "6f955629"),
    (EVID / "dados" / "scripts_congelados" / "train_gnn_c0_spatial.py", "treinador congelado (copia em dados/scripts_congelados)", "6f955629"),
    (GNN_RF_V2 / "generate_realistic_coverage.py", "gerador: modulo 1 de 4 (digest do manuscrito 95ea0423)", "95ea0423"),
    (GNN_RF_V2 / "01_data" / "prepare_transfer_dataset_v19.py", "gerador: modulo 2 de 4 (digest do manuscrito 45c16d43)", "45c16d43"),
    (GNN_RF_V2 / "data_raw" / "enrich_rf_targets.py", "gerador: modulo 3 de 4 (digest do manuscrito 5d38012e)", "5d38012e"),
    (EVID / "scripts" / "contrafactual_alvo_completo.py", "gerador: corretor contrafactual (declive local; digest do manuscrito ebeaf759)", "ebeaf759"),
    (EVID / "scripts" / "gerar_grafo_v19_mosaico_corrigido.py", "cadeia do grafo: mosaico corrigido", None),
    (GNN_RF_V2 / "split_v19_quadrants.py", "cadeia do grafo: divisao em quadrantes", None),
    (EVID / "scripts" / "preparar_transfer_seed42.py", "cadeia do alvo: preparo do transfer (seed 42)", None),
    # ACRESCENTADAS pelo pipeline (nao listadas no pedido): importadas pelo treinador congelado
    # (`--base-dir` = GNN_RF_V2: 02_models, 03_training, 04_baselines) ou pela cadeia do grafo.
    (GNN_RF_V2 / "data_raw" / "generate_graph_v19.py", "[acrescentada] importada por gerar_grafo_v19_mosaico_corrigido.py (cadeia do grafo)", None),
    (GNN_RF_V2 / "02_models" / "gnn_rf_model.py", "[acrescentada] importada pelo treinador congelado (modelo GNN)", None),
    (GNN_RF_V2 / "02_models" / "gnn_rf_encoder.py", "[acrescentada] importada por gnn_rf_model.py", None),
    (GNN_RF_V2 / "02_models" / "physics_loss.py", "[acrescentada] importada pelo treinador congelado (perda fisica)", None),
    (GNN_RF_V2 / "04_baselines" / "empirical_models.py", "[acrescentada] importada pelo treinador congelado (baselines analiticos)", None),
    (GNN_RF_V2 / "03_training" / "rf_diagnostic_metrics.py", "[acrescentada] importada pelo treinador congelado (metricas)", None),
]

# copias divergentes conhecidas (NAO citadas no manuscrito) -- so registradas no meta
COPIAS_DIVERGENTES = [
    GNN_RF / "gnn_rf_ieee_access" / "PACOTE_REPOSITORIO_R3" / "generator" / n
    for n in ("generate_realistic_coverage.py", "prepare_transfer_dataset_v19.py",
              "enrich_rf_targets.py", "contrafactual_alvo_completo.py")
] + [
    GNN_RF / "gnn_rf_ieee_access" / "PACOTE_REPOSITORIO_R3" / "training" / "spatial_cv.py",
    GNN_RF / "gnn_rf_ieee_access" / "PACOTE_REPOSITORIO_R3" / "model" / "rf_decoder.py",
    GNN_RF / "gnn_rf_ieee_access" / "PACOTE_REPOSITORIO_R3" / "training" / "frozen" / "train_gnn_c0_spatial.py",
]


# --------------------------------------------------------------------------
# utilitarios
# --------------------------------------------------------------------------

def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    buf = bytearray(8 << 20)
    mv = memoryview(buf)
    with open(caminho, "rb", buffering=0) as f:
        while True:
            n = f.readinto(mv)
            if not n:
                break
            h.update(mv[:n])
    return h.hexdigest()


def mtime_iso(caminho: Path) -> str:
    return datetime.fromtimestamp(caminho.stat().st_mtime, tz=timezone.utc).isoformat()


def _mesmo_instante(a: str, b: str) -> bool:
    try:
        return datetime.fromisoformat(a) == datetime.fromisoformat(b)
    except Exception:
        return a == b


def raiz_e_rel(caminho: Path):
    melhor = None
    for rid, rp in RAIZES.items():
        try:
            rel = caminho.relative_to(rp)
        except ValueError:
            continue
        if melhor is None or len(str(rp)) > len(str(RAIZES[melhor[0]])):
            melhor = (rid, rel.as_posix())
    if melhor:
        return melhor
    return RAIZ_FS, str(caminho).lstrip("/")


def _importar(path: Path, nome: str):
    spec = importlib.util.spec_from_file_location(nome, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def carregar_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def completar(linha: dict) -> dict:
    """Acrescenta raiz/caminho_rel (e sha_origem se faltar)."""
    p = Path(linha["caminho"])
    rid, rel = raiz_e_rel(p)
    linha["raiz"] = rid
    linha["caminho_rel"] = rel
    return linha


def linha_nova(caminho: Path, grupo: str, sha: str | None, extra: dict | None = None,
               comando=None, erro: str | None = None) -> dict:
    st = caminho.stat()
    ln = {
        "caminho": str(caminho),
        "sha256": sha,
        "sha_recalculado": sha is not None,
        "sha_origem": "recalculado v5" if sha is not None else f"pendente: {erro or 'adiado'}",
        "tamanho_bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "grupo": grupo,
        "comando_regeneracao": comando,
    }
    if extra:
        ln.update(extra)
    return completar(ln)


def _hash_seguro(p: Path):
    try:
        return sha256_arquivo(p), None
    except Exception as e:  # nao aborta o manifesto inteiro por um arquivo
        return None, f"{type(e).__name__}: {e}"


def hashear_lote(paths, workers):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_hash_seguro, paths))
    return res, time.time() - t0


def listar(pasta: Path, filtro=lambda p: True):
    if not pasta.is_dir():
        raise FileNotFoundError(f"pasta ausente: {pasta}")
    return sorted(
        p for p in pasta.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and filtro(p)
    )


# --------------------------------------------------------------------------
# grupos novos
# --------------------------------------------------------------------------

def definir_grupos():
    """Lista (grupo, [Path...], workers, extra_fn|None)."""
    gpu = V3DIR / "gpu"
    redacao12 = V3DIR / "redacao_v3-12"
    redacao = V3DIR / "redacao"
    grupos = []
    grupos.append(("v3_fase1", listar(V3DIR / "fase1"), 8, None))
    grupos.append(("v3_fase2", listar(V3DIR / "fase2"), 8, None))
    grupos.append(("v3_fase3", listar(V3DIR / "fase3"), 8, None))
    grupos.append(("v3_fase4", listar(V3DIR / "fase4"), 8, None))
    grupos.append(("v3_votos", listar(V3DIR / "votos"), 8, None))
    grupos.append(("v3_criterios_demais", listar(V3DIR / "criterios"), 8, None))
    grupos.append(("v3_gpu_agregados_scripts_logs",
                   listar(gpu, lambda p: p.suffix not in (".pt", ".npz")), 8, None))
    grupos.append(("v3_gpu_predicoes_npz", listar(gpu, lambda p: p.suffix == ".npz"), 8, None))
    grupos.append(("v3_gpu_checkpoints_pt", listar(gpu, lambda p: p.suffix == ".pt"), 8, None))
    grupos.append(("v3_12_tabelas", listar(redacao12 / "tables_v3-12"), 8, None))
    grupos.append(("v3_12_figuras",
                   listar(redacao12 / "figures_v3-12") + listar(redacao / "figures_v3-12"), 8, None))
    grupos.append(("v3_12_folha_de_fatos_adendo",
                   sorted(redacao.glob("FOLHA_DE_FATOS_v3-12_adendo.*")), 4, None))
    grupos.append(("checkpoints_treinos_evidencia",
                   sorted(EVID.glob("treinos/*/checkpoints/*.pt")), 8, None))
    grupos.append(("dependencias_vivas",
                   [p for p, _, _ in DEPENDENCIAS_VIVAS if p.is_file()], 4, "dependencias"))
    grupos.append(("graf_data_v3_manifesto",
                   [GRAFOS / "HASHES_SHA256.txt", GRAFOS / "MANIFESTO.md"], 2, None))
    grupos.append(("grafos_gpu_arestas", sorted(GRAFOS.glob("*_gpu.pt")), 4, "gpu_pt"))
    return grupos


def ler_hashes_txt():
    d = {}
    f = GRAFOS / "HASHES_SHA256.txt"
    if f.is_file():
        for l in f.read_text().splitlines():
            if l.strip():
                s, n = l.split(None, 1)
                d[n.strip().lstrip("*")] = s
    return d


def extra_por_arquivo(tag, p: Path, sha, hashes_txt):
    if tag == "dependencias":
        for q, papel, pref in DEPENDENCIAS_VIVAS:
            if q == p:
                ex = {"papel": papel}
                if pref:
                    ex["sha_prefixo_esperado"] = pref
                    ex["sha_prefixo_confere"] = (sha or "").startswith(pref) if sha else None
                return ex
    if tag == "gpu_pt":
        decl = hashes_txt.get(p.name)
        return {
            "sha256_hashes_txt_13_09": decl,
            "confere_hashes_txt_13_09": (sha == decl) if (sha and decl) else None,
            "papel": "grafo do no/arestas (terreno-terreno e antena-terreno) da celula",
        }
    return None


def montar_grupo_novo(grupo, paths, workers, tag, adiar, hashes_txt, parciais: Path, ja: set):
    paths = [p for p in paths if str(p) not in ja]  # sem duplicar caminho ja no v4/base
    if grupo in adiar:
        linhas = [linha_nova(p, grupo, None, extra_por_arquivo(tag, p, None, hashes_txt), erro="adiado por --adiar-grupos")
                  for p in paths]
        return linhas, 0.0, len(paths)
    # reaproveita parcial anterior (mesmo tamanho+mtime)
    cache = {}
    pf = parciais / f"{grupo}.jsonl"
    if pf.is_file():
        for ln in carregar_jsonl(pf):
            cache[ln["caminho"]] = ln
    a_hashear, linhas = [], []
    for p in paths:
        c = cache.get(str(p))
        if c and c.get("sha256") and c["tamanho_bytes"] == p.stat().st_size and _mesmo_instante(c["mtime"], mtime_iso(p)):
            linhas.append(c)
        else:
            a_hashear.append(p)
    res, dt = hashear_lote(a_hashear, workers)
    for p, (sha, erro) in zip(a_hashear, res):
        linhas.append(linha_nova(p, grupo, sha, extra_por_arquivo(tag, p, sha, hashes_txt), erro=erro))
    linhas.sort(key=lambda l: l["caminho"])
    parciais.mkdir(parents=True, exist_ok=True)
    pf.write_text("\n".join(json.dumps(l, ensure_ascii=False, sort_keys=True) for l in linhas) + "\n", encoding="utf-8")
    return linhas, dt, len(a_hashear)


# --------------------------------------------------------------------------
# montagem
# --------------------------------------------------------------------------

def aplicar_regra_grandes(linhas_base, v4_idx):
    """Regra (a): > 100 MB com tamanho+mtime iguais ao v4 -> sha do v4."""
    reaproveitados, recalculados_grandes = [], []
    for ln in linhas_base:
        p = Path(ln["caminho"])
        if not p.is_file():
            ln["sha_origem"] = ln.get("sha_origem", "arquivo ausente no disco")
            continue
        tam = p.stat().st_size
        ln["tamanho_bytes"] = tam
        if tam > LIMITE_GRANDE:
            v4 = v4_idx.get(ln["caminho"])
            mt = mtime_iso(p)
            if v4 and v4.get("tamanho_bytes") == tam and _mesmo_instante(v4.get("mtime", ""), mt):
                ln["sha256"] = v4["sha256"]
                ln["sha_recalculado"] = False
                ln["mtime"] = mt
                ln["sha_origem"] = "v4 (mtime e tamanho inalterados)"
                reaproveitados.append(ln["caminho"])
            else:
                ln["sha256"] = sha256_arquivo(p)
                ln["sha_recalculado"] = True
                ln["mtime"] = mt
                ln["sha_origem"] = "recalculado v5 (mtime/tamanho mudaram desde o v4 ou arquivo novo)"
                recalculados_grandes.append(ln["caminho"])
        else:
            ln["sha_origem"] = "recalculado v5" if ln.get("sha_recalculado", True) else ln.get("sha_origem", "declarado")
    return reaproveitados, recalculados_grandes


def diferencas_contra_v4(v4_linhas, v5_por_caminho, v5_por_chave):
    mudou, sumiu_disco, sumiu_cobertura = [], [], []
    for l4 in v4_linhas:
        c = l4["caminho"]
        l5 = v5_por_chave.get((c, l4.get("grupo"))) or v5_por_caminho.get(c)
        if l5 is None:
            (sumiu_cobertura if Path(c).exists() else sumiu_disco).append(
                {"caminho": c, "grupo_v4": l4.get("grupo"), "existe_no_disco": Path(c).exists()})
            continue
        dif = {}
        if l4.get("sha256") != l5.get("sha256"):
            dif["sha256"] = {"v4": l4.get("sha256"), "v5": l5.get("sha256")}
        if l4.get("tamanho_bytes") != l5.get("tamanho_bytes"):
            dif["tamanho_bytes"] = {"v4": l4.get("tamanho_bytes"), "v5": l5.get("tamanho_bytes")}
        if l4.get("grupo") != l5.get("grupo"):
            dif["grupo"] = {"v4": l4.get("grupo"), "v5": l5.get("grupo")}
        if dif:
            mudou.append({"caminho": c, "grupo_v5": l5.get("grupo"), "diferencas": dif})
    return mudou, sumiu_disco, sumiu_cobertura


def conferir_digests(linhas):
    out = {}
    for pref, modulo in DIGESTS_MANUSCRITO.items():
        achados = [{"caminho": l["caminho"], "caminho_rel": l["caminho_rel"], "raiz": l["raiz"],
                    "grupo": l["grupo"], "sha256": l["sha256"]}
                   for l in linhas if l.get("sha256") and l["sha256"].startswith(pref)]
        out[pref] = {"modulo_esperado": modulo, "aparece_no_manifesto": bool(achados),
                     "n_linhas": len(achados), "linhas": achados}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=DEFAULT_SAIDA)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--adiar-grupos", default="", help="grupos (csv) deixados 'pendente de hash'")
    args = ap.parse_args()
    adiar = {g for g in args.adiar_grupos.split(",") if g}

    t_ini = time.time()
    v4 = _importar(V4_SCRIPT_PATH, "gerar_manifest_mathematics_v4_mod")
    v4_linhas = carregar_jsonl(V4_JSONL)
    v4_idx = {l["caminho"]: l for l in v4_linhas}

    # (a) tudo o que esta no v4, reexecutado
    base, contagens, heranca_v3, n_v3, n_v4_novas = v4.montar_manifest_v4()
    reaproveitados, recalc_grandes = aplicar_regra_grandes(base, v4_idx)
    base = [completar(l) for l in base]
    ja = {l["caminho"] for l in base}
    print(f"[base v4 reexecutado] {len(base)} linhas; grandes reaproveitados do v4: {len(reaproveitados)}; "
          f"grandes recalculados: {len(recalc_grandes)}", flush=True)

    # (b) o que faltava
    hashes_txt = ler_hashes_txt()
    parciais = args.saida.parent / (args.saida.stem + ".parciais")
    linhas_novas, contagem_novos, tempos, ja_no_base = [], {}, {}, {}
    for grupo, paths, workers, tag in definir_grupos():
        sobreposicao = sum(1 for p in paths if str(p) in ja)
        ja_no_base[grupo] = sobreposicao
        if args.dry_run:
            contagem_novos[grupo] = {"arquivos_candidatos": len(paths), "ja_no_base_v4": sobreposicao}
            continue
        ls, dt, n_hash = montar_grupo_novo(grupo, paths, workers, tag, adiar, hashes_txt, parciais, ja)
        linhas_novas.extend(ls)
        ja.update(l["caminho"] for l in ls)
        tempos[grupo] = round(dt, 1)
        contagem_novos[grupo] = {
            "linhas": len(ls),
            "ja_no_base_v4_nao_duplicadas": sobreposicao,
            "tamanho_bytes": sum(l["tamanho_bytes"] for l in ls),
            "pendentes_de_hash": sum(1 for l in ls if l["sha256"] is None),
            "hashes_calculados_agora": n_hash,
            "segundos_hash": tempos[grupo],
        }
        print(f"[{grupo}] {len(ls)} linhas, {contagem_novos[grupo]['tamanho_bytes']/1e9:.2f} GB, "
              f"{n_hash} hashes em {dt:.0f}s, pendentes={contagem_novos[grupo]['pendentes_de_hash']}", flush=True)
    if args.dry_run:
        print(json.dumps(contagem_novos, indent=1))
        return

    linhas = base + linhas_novas
    # o v3/v4 repete de proposito o mesmo caminho em grupos de papel diferente
    # (ex.: lista_fixa_tab_retention x run_c0c1cf_g10b2); a unicidade vale por (caminho, grupo).
    assert len({(l["caminho"], l["grupo"]) for l in linhas}) == len(linhas), "(caminho, grupo) duplicado no v5"
    assert not ({l["caminho"] for l in linhas_novas} & {l["caminho"] for l in base}), "grupo novo repete caminho da base"
    linhas.sort(key=lambda l: (l["grupo"], l["caminho"]))

    por_caminho = {l["caminho"]: l for l in linhas}
    por_chave = {(l["caminho"], l["grupo"]): l for l in linhas}
    n_caminhos_multigrupo = sum(1 for c, n in Counter(l["caminho"] for l in linhas).items() if n > 1)
    mudou, sumiu_disco, sumiu_cobertura = diferencas_contra_v4(v4_linhas, por_caminho, por_chave)
    novos_no_base = sorted(l["caminho"] for l in base if l["caminho"] not in v4_idx)

    # contagem e tamanho por grupo (todas as linhas)
    por_grupo = {}
    for l in linhas:
        g = por_grupo.setdefault(l["grupo"], {"linhas": 0, "tamanho_bytes": 0, "pendentes_de_hash": 0,
                                              "sha_reaproveitado_v4": 0, "sha_recalculado_v5": 0})
        g["linhas"] += 1
        g["tamanho_bytes"] += l["tamanho_bytes"]
        if l["sha256"] is None:
            g["pendentes_de_hash"] += 1
        elif l.get("sha_origem", "").startswith("v4"):
            g["sha_reaproveitado_v4"] += 1
        else:
            g["sha_recalculado_v5"] += 1

    # cobertura dos scripts v3_*.py / v3_12_*.py atuais
    scripts_atuais = sorted((MDPI / "scripts").glob("v3_*.py"))
    scripts_sem_linha = [str(p) for p in scripts_atuais if str(p.resolve()) not in por_caminho]

    dig = conferir_digests(linhas)
    deps = []
    nao_localizado = []
    for p, papel, pref in DEPENDENCIAS_VIVAS:
        l = por_caminho.get(str(p.resolve())) or por_caminho.get(str(p))
        if l is None:
            nao_localizado.append({"caminho_esperado": str(p), "papel": papel,
                                   "motivo": "arquivo ausente" if not p.exists() else "nao entrou no manifesto"})
        else:
            deps.append({"caminho_rel": l["caminho_rel"], "raiz": l["raiz"], "grupo": l["grupo"],
                         "sha256": l["sha256"], "papel": papel,
                         "prefixo_esperado": pref,
                         "prefixo_confere": (l["sha256"] or "").startswith(pref) if pref else None})
    copias_div = []
    for p in COPIAS_DIVERGENTES:
        if p.is_file():
            copias_div.append({"caminho": str(p), "sha256": sha256_arquivo(p), "tamanho_bytes": p.stat().st_size,
                               "nota": "copia no pacote de repositorio R3; sha DIFERENTE do modulo citado no manuscrito; nao entra no manifesto"})

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    texto = "\n".join(json.dumps(l, ensure_ascii=False, sort_keys=True) for l in linhas) + "\n"
    args.saida.write_text(texto, encoding="utf-8")

    agora = datetime.now(timezone.utc)
    meta = {
        "versao": "v5",
        "gerado_em": agora.isoformat(),
        "data": agora.date().isoformat(),
        "comando": f".venv/bin/python scripts/gerar_manifest_mathematics_v5.py"
                   + (f" --adiar-grupos {args.adiar_grupos}" if args.adiar_grupos else ""),
        "python": sys.executable,
        "duracao_s": round(time.time() - t_ini, 1),
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_arquivo(SCRIPT_PATH),
        "copia_declarada_de": {"script": str(V4_SCRIPT_PATH), "sha256": sha256_arquivo(V4_SCRIPT_PATH),
                               "nota": "v4 nao alterado; v5 o importa e acrescenta grupos"},
        "v4_manifest_comparado": {"caminho": str(V4_JSONL), "sha256": sha256_arquivo(V4_JSONL), "n_linhas": len(v4_linhas)},
        "saida": str(args.saida.resolve()),
        "saida_sha256": sha256_arquivo(args.saida),
        "protocolo": "ROADMAP_v3-12_major_revision.md sec. 5 / D9; ordem do dono 01/10/2026 17:12; ata escritor_cientifico 2026-10-01 (Adendo 2)",
        "raizes": {k: str(v) for k, v in RAIZES.items()} | {RAIZ_FS: "/ (caminho absoluto sem a barra inicial)"},
        "convencao_linha": "caminho (absoluto, compativel com validadores v1-v3), raiz + caminho_rel (relativo a uma raiz acima), "
                           "tamanho_bytes, mtime (UTC), sha256, sha_origem, grupo",
        "n_linhas": len(linhas),
        "n_caminhos_distintos": len({l["caminho"] for l in linhas}),
        "n_caminhos_em_mais_de_um_grupo": n_caminhos_multigrupo,
        "n_linhas_base_v4_reexecutado": len(base),
        "n_linhas_novas_v5": len(linhas_novas),
        "tamanho_total_bytes": sum(l["tamanho_bytes"] for l in linhas),
        "contagens_por_grupo": dict(sorted(por_grupo.items())),
        "detalhe_grupos_novos": contagem_novos,
        "contagens_base_v4_originais": contagens,
        "grupos_pendentes_de_hash": sorted({l["grupo"] for l in linhas if l["sha256"] is None}),
        "regra_arquivos_grandes": {
            "limite_bytes": LIMITE_GRANDE,
            "reaproveitados_do_v4_mtime_e_tamanho_inalterados": len(reaproveitados),
            "recalculados_por_mudanca_ou_novos": recalc_grandes,
        },
        "diferencas_contra_v4": {
            "n_linhas_v4": len(v4_linhas),
            "n_linhas_v5": len(linhas),
            "mudou_sha_ou_tamanho_ou_grupo": mudou,
            "sumiu_do_disco": sumiu_disco,
            "sumiu_da_cobertura_mas_existe_no_disco": sumiu_cobertura,
            "novos_nos_grupos_dinamicos_do_v4": novos_no_base,
            "novos_grupos_v5_contagem": {g: v.get("linhas") for g, v in contagem_novos.items()},
        },
        "conferencia_digests_manuscrito": dig,
        "conferencia_digests_ok": all(v["aparece_no_manifesto"] for v in dig.values()),
        "dependencias_vivas": deps,
        "nao_localizado": nao_localizado,
        "localizado_com_ressalva": [
            {"item": "GNN_RF_V2/config.py", "ressalva": "o arquivo existe em GNN_RF_V2/config/config.py (pacote `config`, com __init__.py); "
                                                       "nao ha GNN_RF_V2/config.py; spatial_cv faz `from config import config`"},
            {"item": "treinador congelado train_gnn_c0_spatial.py", "ressalva": "sha 6f955629 em EVIDENCIA/scripts e EVIDENCIA/dados/scripts_congelados (identicos); "
                                                                              "a copia em PACOTE_REPOSITORIO_R3/training/frozen tem outro sha (0e180392) e nao e o das corridas"},
            {"item": "modulos do gerador", "ressalva": "os tres modulos citados estao em GNN_RF_V2 (shas conferem); as copias em "
                                                       "PACOTE_REPOSITORIO_R3/generator e _publicar_2026-09-16/repo/generator tem shas DIFERENTES (ver copias_divergentes)"},
            {"item": "dependencias acrescentadas", "ressalva": "generate_graph_v19, gnn_rf_model, gnn_rf_encoder, physics_loss, empirical_models e rf_diagnostic_metrics "
                                                                 "nao estavam no pedido, mas o treinador congelado e o gerador do grafo os importam; entraram no grupo dependencias_vivas"},
            {"item": "proveniencia dos modulos de GNN_RF_V2", "ressalva": "os scripts congelados apontam para D:\\_ARQUIVO_SSD_F\\TOPO_RF\\GNN_RF_V2 (Windows); o manifesto registra o espelho do PC "
                                                                         "(mtimes de jan-mar/2026, anteriores as corridas de set/2026). Nao ha prova byte a byte de que sejam os lidos em D: nas corridas, "
                                                                         "exceto os quatro digests do manuscrito e o trainer (6f955629), que conferem com o que o texto e as corridas declaram"},
            {"item": "indices de particao", "ressalva": "nao existem como arquivo autonomo: sao recomputados por spatial_cv (sha no manifesto) a partir do tensor, "
                                                         "grid_km/buffer_km e split_seed; cada run JSON registra `particoes.*.idx_sha256_global` e os .npz de predicao "
                                                         "(R2 e3_predicoes, gpu/*) trazem `idx_global` dos nos de teste"},
        ],
        "copias_divergentes": copias_div,
        "scripts_v3_atuais_sem_linha": scripts_sem_linha,
        "excluidos_da_varredura": "__pycache__/ e *.pyc dentro de _v3_2026-09-25/",
        "nota_estado": "snapshot de redacao_v3-12 (tables/figures) e scripts/ em curso na data; mtime e sha registram o estado no momento da geracao",
    }
    sidecar = args.saida.with_name(args.saida.stem + ".meta.json")
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"gravado: {args.saida} ({len(linhas)} linhas = {len(base)} base v4 + {len(linhas_novas)} novas)")
    print(f"meta: {sidecar}")
    print(f"digests ok: {meta['conferencia_digests_ok']}; pendentes: {meta['grupos_pendentes_de_hash']}; "
          f"nao localizado: {len(nao_localizado)}")


if __name__ == "__main__":
    main()
