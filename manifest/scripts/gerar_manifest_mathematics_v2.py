#!/usr/bin/env python3
"""Gera manifest_mathematics_v2.jsonl do Artigo 2 (MDPI Mathematics) -- v2.

Arquivo NOVO (nao edita gerar_manifest_mathematics.py v1). Importa a logica
do v1 (grupos, cruzamento com manifest de E, idempotencia sem timestamp nas
linhas) e corrige os defeitos M1-M4 apontados em
forum-eng-dados_parecer_R2.md secao J3:

M1: junta proveniencia_fontes_rascunho.json (por BASENAME) as 14 linhas de
    alvo_stats/fismat/carimbos que a R1/R2 identificaram sem script_sha256;
    ASSERCAO: sha256 do artefato na proveniencia == recalculado hoje, senao
    aborta. Os 2 carimbos que a proveniencia nao cobre (mascara_vs_adjacencia,
    proposicoes) ficam com script_sha256=null e script_sha256_motivo
    explicito. Acrescenta grupos novos: scripts_produtores (sha recalculado,
    com assercao de igualdade contra o sha que a proveniencia/R1 declarou),
    tensores_cftudo (16 *_cftudo.pt, SEM re-hash -- sha de HASHES_SHA256.txt,
    tamanho por os.stat), scripts_artigo2 (glob de MDPI_Mathematics/scripts/
    em tempo de execucao), mlpcf_t11_E2 (incorpora E2_manifest_mlpcf_t11.jsonl,
    recalculando sha do run JSON e marcando sha_E2_confere nas linhas
    correspondentes do grupo run_mlpcf) e lista_fixa_tab_retention (20
    corridas da lista fixa do fisico, com assercao de sha contra a lista e
    contra o manifest).

M2: 'esperado' numerico e ASSERTIVO nos grupos fechados (aborta se nao bater):
    c0c1cf_g10b2 80 basenames, c0c1_bauru_g5b2_s42 4 basenames, mlpcf 80
    basenames, baselines_v2 16, alvo_stats_e_controle_negativo 10,
    fismat_carimbo_11_09 4, tensores_cftudo 16, mlpcf_t11_E2 40,
    lista_fixa_tab_retention 20. rodadas_r1_r2_fio continua dinamico
    (esperado=None), como o v1.

Sem carimbo de tempo de geracao dentro das linhas do jsonl (idempotencia
byte-a-byte entre execucoes sem mudanca no disco); o carimbo vive no sidecar
<saida>.meta.json (gerado_em, sha256 deste script, contagens).

PROIBIDO: ler ou re-hashear os *_cftudo.pt (28 GB cada) -- o sha destes vem
sempre de graph_data_v3/HASHES_SHA256.txt, nunca de leitura do tensor.

Uso:
    python gerar_manifest_mathematics_v2.py [--hash] [--saida CAMINHO] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()

RAIZ_E = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO"
)
MANIFEST_E = RAIZ_E / "manifest.jsonl"

RAIZ_FISMAT_ATAS = Path("/trabalho/HERMES/AGENTES/forum-fisico-matematico/ATAS")
RAIZ_FIOS = Path("/trabalho/HERMES/AGENTES/_FIOS")
FIO_R2 = RAIZ_FIOS / "2026-09-24_gnn_rf_artigo2_mathematics_r2"
GANCHO_E2 = FIO_R2 / "artefatos" / "E2_manifest_mlpcf_t11.jsonl"
PROVENIENCIA = FIO_R2 / "artefatos" / "proveniencia_fontes_rascunho.json"
LISTA_FIXA_TAB_RETENTION = FIO_R2 / "artefatos" / "forum-fisico-matematico_lista_corridas_tab_retention.json"

RAIZ_GRAPH_DATA_V3 = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
HASHES_SHA256_TXT = RAIZ_GRAPH_DATA_V3 / "HASHES_SHA256.txt"

DEFAULT_SAIDA = SCRIPT_PATH.parent.parent / "manifest_mathematics_v2.jsonl"

CAMPOS_PRODUTOR_CANDIDATOS = ["script", "gerado_por", "autor", "agente"]

CHECKPOINT_LIMITE_BYTES = 200 * 1024 * 1024  # 200 MB

# Scripts produtores desta rodada, com o sha256 esperado (assercao de
# igualdade contra o que a frente dados-proveniencia/coordenador R2 ja
# conferiu). None = sem assercao previa (so registra o sha recalculado hoje).
SCRIPTS_PRODUTORES = [
    {
        "id": "e3_alvos_dataset_v2_multi",
        "caminho": RAIZ_E / "scripts" / "e3_alvos_dataset_v2_multi.py",
        "sha256_esperado": "0af7c038791fe30013f74e405a05ce820a88a981b76b6bb4cfcd6ddd1df94906",
    },
    {
        "id": "alvo_stats",
        "caminho": RAIZ_E / "scripts" / "alvo_stats.py",
        "sha256_esperado": "cd3ffb513d187b5a4a2fbe1378520d92f4dc510f1f31a48e6cbe50881c2de9cd",
    },
    {
        "id": "fismat_controle_negativo_alvo",
        "caminho": RAIZ_E / "scripts" / "fismat_controle_negativo_alvo.py",
        "sha256_esperado": "65b7fd1034d7a0b26d8f8a4577ce75c7c1f9d4e86066665cba0ec8fd84d84897",
    },
    {
        "id": "contrafactual_alvo_completo",
        "caminho": RAIZ_E / "scripts" / "contrafactual_alvo_completo.py",
        "sha256_esperado": None,
    },
    {
        "id": "fm_split_geom3",
        "caminho": RAIZ_FISMAT_ATAS / "fm_split_geom3.py",
        "sha256_esperado": "f357cc84bc023ec6fcb0bf7f479a2c36597ef86b9f32eb8ac19e20e2627da401",
    },
    {
        "id": "fm_drift_grafo",
        "caminho": RAIZ_FISMAT_ATAS / "fm_drift_grafo.py",
        "sha256_esperado": "15677b743de0c368117996131846220881f858933bf0e66fc33520abdfb25750",
    },
    {
        "id": "fila_treino_16_tiles",
        "caminho": RAIZ_E / "scripts" / "fila_treino_16_tiles.ps1",
        "sha256_esperado_prefixo": "013bab6d",
    },
    {
        "id": "train_gnn_c0_spatial",
        "caminho": RAIZ_E / "dados" / "scripts_congelados" / "train_gnn_c0_spatial.py",
        "sha256_esperado": "6f955629cde164f2843f454e1ebf6977292fd80647ef48ecd0df31e464c42445",
    },
]

# Basename (apos remover qualquer sufixo " (...)" do campo 'caminho' da
# proveniencia) -> indice na lista 'alvos' de proveniencia_fontes_rascunho.json.
# Preenchido em tempo de execucao por carregar_proveniencia().

# Os 2 carimbos do fisico que a proveniencia NAO cobre nesta rodada.
CARIMBOS_SEM_PROVENIENCIA = {
    "2026-09-11_mascara_vs_adjacencia.json": (
        "proveniencia_fontes_rascunho.json nao lista este carimbo entre os "
        "'alvos' desta rodada R2 (cobriu so geometria_split e deriva_grafo); "
        "fora do escopo declarado pela frente dados-proveniencia"
    ),
    "2026-09-11_proposicoes_mdpi.json": (
        "proveniencia_fontes_rascunho.json nao lista este carimbo entre os "
        "'alvos' desta rodada R2; fora do escopo declarado pela frente "
        "dados-proveniencia"
    ),
}


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def mtime_iso(caminho: Path) -> str:
    ts = caminho.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def carregar_json_seguro(caminho: Path):
    """Retorna (dict_ou_None, motivo_se_None). Nunca lanca por conteudo invalido."""
    if caminho.suffix.lower() not in (".json",):
        return None, "arquivo nao e .json"
    try:
        texto = caminho.read_text(encoding="utf-8")
    except Exception as e:  # leitura binaria/permissao etc.
        return None, f"falha ao ler arquivo como texto: {e!r}"
    try:
        d = json.loads(texto)
    except Exception as e:
        return None, f"falha ao fazer parse JSON: {e!r}"
    if not isinstance(d, dict):
        return None, "JSON raiz nao e objeto (dict)"
    return d, None


def derivar_produtor(caminho: Path):
    """Retorna (produtor, produtor_motivo) -- so o produtor (o v2 nao
    inventa mais script_sha256 a partir de heuristica de path; quem faz isso
    e a juncao explicita com a proveniencia, funcao script_sha256_via_proveniencia)."""
    d, motivo = carregar_json_seguro(caminho)
    if d is None:
        return None, motivo
    produtor = None
    for campo in CAMPOS_PRODUTOR_CANDIDATOS:
        v = d.get(campo)
        if isinstance(v, str) and v.strip():
            produtor = v
            break
    produtor_motivo = None
    if produtor is None:
        produtor_motivo = (
            "nenhum dos campos internos reconheciveis "
            f"({', '.join(CAMPOS_PRODUTOR_CANDIDATOS)}) presente/nao-vazio"
        )
    return produtor, produtor_motivo


def carregar_manifest_e():
    """basename -> lista de (relpath, sha256); relpath -> (basename, sha256)."""
    por_basename: dict[str, list[tuple[str, str]]] = {}
    por_relpath: dict[str, tuple[str, str]] = {}
    if not MANIFEST_E.exists():
        return por_basename, por_relpath
    for linha in MANIFEST_E.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha:
            continue
        e = json.loads(linha)
        rel = e["artefato"]
        sha = e["sha256"]
        base = rel.replace("\\", "/").rsplit("/", 1)[-1]
        por_basename.setdefault(base, []).append((rel, sha))
        por_relpath[rel] = (base, sha)
    return por_basename, por_relpath


def cruzar_com_manifest_e(caminho: Path, sha256: str, por_basename, por_relpath):
    """em_manifest_E / sha_confere_E, por basename (licao R1: nunca por prefixo)."""
    try:
        rel = caminho.resolve().relative_to(RAIZ_E.resolve()).as_posix()
    except ValueError:
        rel = None

    if rel is not None and rel in por_relpath:
        _, sha_manifest = por_relpath[rel]
        return True, sha_manifest == sha256, "relpath"

    base = caminho.name
    candidatos = por_basename.get(base)
    if not candidatos:
        return False, None, None

    for rel_c, sha_c in candidatos:
        if sha_c == sha256:
            return True, True, "basename"
    return True, False, "basename"


# --------------------------------------------------------------------------
# Proveniencia (M1)
# --------------------------------------------------------------------------

def _basename_proveniencia(caminho_field: str) -> str:
    """O campo 'caminho' da proveniencia as vezes tem um sufixo livre entre
    parenteses (ex.: 'E/dados/alvo/alvo_stats_bauru_s42.json (sem sufixo Q_v2)').
    Extrai so a parte do caminho, antes do primeiro ' (' e devolve o basename."""
    p = caminho_field.split(" (", 1)[0].strip()
    return p.replace("\\", "/").rsplit("/", 1)[-1]


def carregar_proveniencia():
    """basename -> item da lista 'alvos' de proveniencia_fontes_rascunho.json."""
    if not PROVENIENCIA.exists():
        raise FileNotFoundError(f"proveniencia obrigatoria ausente: {PROVENIENCIA}")
    d = json.loads(PROVENIENCIA.read_text(encoding="utf-8"))
    por_basename = {}
    for item in d["alvos"]:
        base = _basename_proveniencia(item["caminho"])
        por_basename[base] = item
    return por_basename


def enriquecer_com_proveniencia(linha: dict, caminho: Path, por_basename_prov: dict):
    """Preenche script_caminho, script_sha256, regeneravel_por, cadeia, falta
    a partir da proveniencia, casando por BASENAME. ASSERCAO: sha256 do
    artefato na proveniencia == recalculado hoje (aborta se nao bater).
    Sem .get() com default silencioso: proveniencia sem entrada = motivo
    explicito, nao excecao (a lista de 'alvos' e propositalmente parcial:
    2 dos 4 carimbos ficam de fora, ver CARIMBOS_SEM_PROVENIENCIA)."""
    base = caminho.name
    item = por_basename_prov.get(base)
    if item is None:
        motivo = CARIMBOS_SEM_PROVENIENCIA.get(base)
        if motivo is None:
            raise KeyError(
                f"artefato {base} esperava entrada na proveniencia (grupo "
                "alvo_stats/fismat/carimbos) e nao tem, e nao esta na lista "
                "explicita CARIMBOS_SEM_PROVENIENCIA -- checar manualmente"
            )
        linha["script_caminho"] = None
        linha["script_sha256"] = None
        linha["script_sha256_motivo"] = motivo
        linha["regeneravel_por"] = None
        linha["cadeia"] = None
        linha["falta"] = None
        return linha

    sha_proveniencia = item["sha256"]
    if sha_proveniencia != linha["sha256"]:
        raise AssertionError(
            f"ABORTA: sha256 de {base} na proveniencia ({sha_proveniencia}) "
            f"!= sha256 recalculado hoje ({linha['sha256']}) -- artefato "
            "mudou desde que a frente dados-proveniencia o fechou"
        )

    script_info = item.get("script")
    if script_info is None:
        linha["script_caminho"] = None
        linha["script_sha256"] = None
        linha["script_sha256_motivo"] = "proveniencia sem bloco 'script' para este item"
    else:
        linha["script_caminho"] = script_info["caminho"]
        linha["script_sha256"] = script_info["sha256"]
        linha["script_sha256_motivo"] = None

    linha["regeneravel_por"] = item["regeneravel_por"]
    linha["cadeia"] = item["cadeia"]
    linha["falta"] = item["falta"]
    return linha


def linha_artefato(caminho: Path, grupo: str, regeneravel_por, por_basename, por_relpath):
    if not caminho.is_file():
        raise FileNotFoundError(f"artefato esperado nao existe: {caminho}")
    sha256 = sha256_arquivo(caminho)
    tamanho = caminho.stat().st_size
    mtime = mtime_iso(caminho)
    produtor, produtor_motivo = derivar_produtor(caminho)
    em_manifest_e, sha_confere_e, tipo_match = cruzar_com_manifest_e(
        caminho, sha256, por_basename, por_relpath
    )
    return {
        "caminho": str(caminho.resolve()),
        "sha256": sha256,
        "sha_recalculado": True,
        "tamanho_bytes": tamanho,
        "mtime": mtime,
        "produtor": produtor,
        "produtor_motivo": produtor_motivo,
        "grupo": grupo,
        "regeneravel_por": regeneravel_por,
        "em_manifest_E": em_manifest_e,
        "sha_confere_E": sha_confere_e,
        "tipo_match_manifest_E": tipo_match,
    }


# --------------------------------------------------------------------------
# Grupos herdados do v1 (1-5), com M1/M2 aplicados onde cabe
# --------------------------------------------------------------------------

def grupo_1_fismat_stamps():
    arquivos = sorted(RAIZ_FISMAT_ATAS.glob("2026-09-11_*_mdpi.json"))
    extra = RAIZ_FISMAT_ATAS / "2026-09-11_mascara_vs_adjacencia.json"
    if extra.exists() and extra not in arquivos:
        arquivos.append(extra)
    return sorted(set(arquivos)), "fismat_carimbo_11_09"


def grupo_2_alvo():
    pasta = RAIZ_E / "dados" / "alvo"
    arquivos = sorted(pasta.glob("alvo_stats_*.json"))
    extra = pasta / "fismat_controle_negativo_alvo.json"
    if extra.exists():
        arquivos.append(extra)
    return arquivos, "alvo_stats_e_controle_negativo"


def grupo_3_baselines():
    pasta = RAIZ_E / "dados" / "baselines_v2"
    return sorted(pasta.glob("baselines_v2_*.json")), "baselines_v2"


RE_C0C1CF = re.compile(r"^run_c0c1cf_[a-z]+_s\d+_Q\d_g10b2\.json$")
RE_C0C1_BAURU_G5B2_S42 = re.compile(r"^run_c0c1_bauru_s42_Q\d_g5b2\.json$")
RE_MLPCF = re.compile(r"^run_mlpcf_[a-z]+_s\d+_Q\d_g10b2\.json$")
RE_MLPCF_SMOKE = re.compile(r"^run_mlpcf_smoke.*\.json$")


def grupo_4_runs():
    raizes = [RAIZ_E / "dados" / "treinos_c1", RAIZ_E / "treinos"]
    achados: dict[str, list[Path]] = {
        "run_c0c1cf_g10b2": [],
        "run_c0c1_bauru_g5b2_s42": [],
        "run_mlpcf": [],
        "run_mlpcf_smoke": [],
    }
    for raiz in raizes:
        if not raiz.exists():
            continue
        candidatos = list(raiz.glob("run_*.json")) + list(raiz.glob("*/run_*.json"))
        for c in candidatos:
            nome = c.name
            if RE_MLPCF_SMOKE.match(nome):
                achados["run_mlpcf_smoke"].append(c)
            elif RE_C0C1CF.match(nome):
                achados["run_c0c1cf_g10b2"].append(c)
            elif RE_C0C1_BAURU_G5B2_S42.match(nome):
                achados["run_c0c1_bauru_g5b2_s42"].append(c)
            elif RE_MLPCF.match(nome):
                achados["run_mlpcf"].append(c)
    for k in achados:
        achados[k] = sorted(set(achados[k]))
    return achados


def grupo_5_rodadas():
    padrao_dir = "*gnn_rf_artigo2_mathematics*"
    arquivos = []
    for fio_dir in sorted(RAIZ_FIOS.glob(padrao_dir)):
        art = fio_dir / "artefatos"
        if not art.is_dir():
            continue
        arquivos.extend(sorted(art.glob("*.json")))
        arquivos.extend(sorted(art.glob("*.jsonl")))
    return arquivos, "rodadas_r1_r2_fio"


# --------------------------------------------------------------------------
# Grupos novos do v2
# --------------------------------------------------------------------------

def grupo_scripts_produtores():
    linhas = []
    for espec in SCRIPTS_PRODUTORES:
        caminho = espec["caminho"]
        if not caminho.is_file():
            raise FileNotFoundError(f"script produtor esperado nao existe: {caminho}")
        sha = sha256_arquivo(caminho)
        if "sha256_esperado_prefixo" in espec:
            pref = espec["sha256_esperado_prefixo"]
            if not sha.startswith(pref):
                raise AssertionError(
                    f"ABORTA: {espec['id']} ({caminho}) sha256={sha} nao comeca "
                    f"com o prefixo esperado {pref!r} (fila congelada do lote c0c1cf)"
                )
        elif espec.get("sha256_esperado") is not None:
            if sha != espec["sha256_esperado"]:
                raise AssertionError(
                    f"ABORTA: {espec['id']} ({caminho}) sha256={sha} != esperado "
                    f"{espec['sha256_esperado']} (proveniencia/coordenador R2)"
                )
        linhas.append({
            "caminho": str(caminho.resolve()),
            "sha256": sha,
            "sha_recalculado": True,
            "tamanho_bytes": caminho.stat().st_size,
            "mtime": mtime_iso(caminho),
            "id": espec["id"],
            "grupo": "scripts_produtores",
            "sha256_esperado_conferido": (
                espec.get("sha256_esperado") or espec.get("sha256_esperado_prefixo")
            ),
        })
    return linhas


def grupo_tensores_cftudo():
    if not HASHES_SHA256_TXT.exists():
        raise FileNotFoundError(f"HASHES_SHA256.txt ausente: {HASHES_SHA256_TXT}")
    hashes_locais: dict[str, str] = {}
    for ln in HASHES_SHA256_TXT.read_text(encoding="utf-8").splitlines():
        partes = ln.split()
        if len(partes) >= 2 and len(partes[0]) == 64:
            base = os.path.basename(partes[-1].lstrip("*"))
            hashes_locais[base] = partes[0]

    por_basename_e, _ = carregar_manifest_e()

    cidades = ["bauru", "lins", "campinas", "sorocaba"]
    quadrantes = ["Q1", "Q2", "Q3", "Q4"]
    linhas = []
    for cidade in cidades:
        for q in quadrantes:
            basename = f"transfer_dataset_{cidade}_v19_{q}_enriched_cftudo.pt"
            caminho = RAIZ_GRAPH_DATA_V3 / basename
            if not caminho.is_file():
                raise FileNotFoundError(f"tensor cftudo esperado nao existe: {caminho}")
            sha_hashes_txt = hashes_locais.get(basename)
            if sha_hashes_txt is None:
                raise KeyError(f"{basename} nao tem linha em {HASHES_SHA256_TXT}")
            tamanho_disco = os.stat(caminho).st_size

            run_json = RAIZ_E / "dados" / "treinos_c1" / f"run_c0c1cf_{cidade}_s42_{q}_g10b2.json"
            if not run_json.is_file():
                raise FileNotFoundError(f"run JSON correspondente ausente: {run_json}")
            rd = json.loads(run_json.read_text(encoding="utf-8"))
            ds = rd["dataset"]
            rf_data_bytes = ds["rf_data_bytes"]
            rf_data_sha256 = ds["rf_data_sha256"]

            if tamanho_disco != rf_data_bytes:
                raise AssertionError(
                    f"ABORTA: tamanho de {basename} em disco ({tamanho_disco}) != "
                    f"dataset.rf_data_bytes de {run_json.name} ({rf_data_bytes})"
                )
            if sha_hashes_txt != rf_data_sha256:
                raise AssertionError(
                    f"ABORTA: sha256 de {basename} em HASHES_SHA256.txt "
                    f"({sha_hashes_txt}) != dataset.rf_data_sha256 de "
                    f"{run_json.name} ({rf_data_sha256})"
                )
            sha_em_manifest_e = sha_hashes_txt in {sha for _, sha in por_basename_e.get(basename, [])}
            if not sha_em_manifest_e:
                raise AssertionError(
                    f"ABORTA: sha256 de {basename} nao encontrado em "
                    f"E/manifest.jsonl (esperado, ref. cftudo_local_vs_run.py 16/16)"
                )

            linhas.append({
                "caminho": str(caminho.resolve()),
                "sha256": sha_hashes_txt,
                "sha_recalculado": False,
                "sha_fonte": "HASHES_SHA256.txt",
                "tamanho_bytes": tamanho_disco,
                "mtime": mtime_iso(caminho),
                "grupo": "tensores_cftudo",
                "celula": f"{cidade}_{q}",
                "run_json_correspondente": str(run_json.resolve()),
                "tamanho_confere_run_json": True,
                "sha_confere_run_json": True,
                "em_manifest_E": True,
                "sha_confere_E": True,
            })
    return linhas


def grupo_scripts_artigo2():
    pasta = SCRIPT_PATH.parent
    arquivos = sorted(pasta.glob("*.py"))
    linhas = []
    for c in arquivos:
        linhas.append({
            "caminho": str(c.resolve()),
            "sha256": sha256_arquivo(c),
            "sha_recalculado": True,
            "tamanho_bytes": c.stat().st_size,
            "mtime": mtime_iso(c),
            "grupo": "scripts_artigo2",
        })
    return linhas


def grupo_mlpcf_t11_e2(runs_mlpcf: list[Path]):
    if not GANCHO_E2.exists():
        return [], {}
    entradas = []
    for ln in GANCHO_E2.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            entradas.append(json.loads(ln))

    por_run_label_para_sha_e2: dict[str, bool] = {}
    linhas = []
    for e in entradas:
        run_json_path = Path(e["caminho_run_json_linux"])
        if not run_json_path.is_file():
            raise FileNotFoundError(f"run JSON do gancho E2 nao existe: {run_json_path}")
        sha_recalc_run = sha256_arquivo(run_json_path)
        if sha_recalc_run != e["sha256_run_json"]:
            raise AssertionError(
                f"ABORTA: sha256 recalculado de {run_json_path.name} "
                f"({sha_recalc_run}) != sha256_run_json declarado no gancho E2 "
                f"({e['sha256_run_json']})"
            )

        checkpoint_path = None
        checkpoint_existe_local = False
        try:
            rd = json.loads(run_json_path.read_text(encoding="utf-8"))
            checkpoint_path = rd["selecao"]["checkpoint"]
        except Exception:
            checkpoint_path = None
        # checkpoint vive em drive Windows D:\ nao montado nesta sessao Linux
        if checkpoint_path is not None and not checkpoint_path.startswith(("D:", "F:")):
            p = Path(checkpoint_path)
            checkpoint_existe_local = p.is_file()

        if checkpoint_existe_local:
            p = Path(checkpoint_path)
            tamanho_ckpt = p.stat().st_size
            if tamanho_ckpt < CHECKPOINT_LIMITE_BYTES:
                sha_ckpt = sha256_arquivo(p)
                sha_ckpt_recalculado = True
                sha_ckpt_motivo = None
            else:
                sha_ckpt = e["sha256_checkpoint_best"]
                sha_ckpt_recalculado = False
                sha_ckpt_motivo = f"checkpoint local >= {CHECKPOINT_LIMITE_BYTES} bytes; sha copiado do gancho E2"
        else:
            sha_ckpt = e["sha256_checkpoint_best"]
            sha_ckpt_recalculado = False
            sha_ckpt_motivo = (
                "checkpoint_best.pt vive em drive Windows D:\\ nao montado nesta "
                "sessao Linux (path declarado no run JSON); sha copiado do gancho E2, nao recalculado"
            )

        por_run_label_para_sha_e2[e["run_label"]] = True

        linhas.append({
            "caminho": str(run_json_path.resolve()),
            "sha256": sha_recalc_run,
            "sha_recalculado": True,
            "tamanho_bytes": run_json_path.stat().st_size,
            "mtime": mtime_iso(run_json_path),
            "grupo": "mlpcf_t11_E2",
            "run_label": e["run_label"],
            "sha256_run_json_declarado_E2": e["sha256_run_json"],
            "sha256_run_json_confere": True,
            "sha256_checkpoint_best": sha_ckpt,
            "sha256_checkpoint_best_declarado_E2": e["sha256_checkpoint_best"],
            "sha_checkpoint_recalculado": sha_ckpt_recalculado,
            "sha_checkpoint_motivo": sha_ckpt_motivo,
            "rf_data_sha256_bate": e["rf_data_sha256_bate"],
            "script_sha256_bate": e["script_sha256_bate"],
            "manifest_status_E": e["manifest_status"],
        })

    # marca sha_E2_confere nas linhas run_mlpcf correspondentes (por run_label
    # extraido do basename run_<label>.json)
    marcadas = {}
    for c in runs_mlpcf:
        stem = c.stem  # run_<label>
        if not stem.startswith("run_"):
            continue
        label = stem[len("run_"):]
        marcadas[str(c.resolve())] = label in por_run_label_para_sha_e2

    return linhas, marcadas


def grupo_lista_fixa_tab_retention():
    if not LISTA_FIXA_TAB_RETENTION.is_file():
        raise FileNotFoundError(f"lista fixa do fisico ausente: {LISTA_FIXA_TAB_RETENTION}")
    d = json.loads(LISTA_FIXA_TAB_RETENTION.read_text(encoding="utf-8"))
    script_lista = Path(d["script"])
    if not script_lista.is_file():
        raise FileNotFoundError(f"script da lista fixa ausente: {script_lista}")
    sha_script_recalc = sha256_arquivo(script_lista)
    if sha_script_recalc != d["script_sha256"]:
        raise AssertionError(
            f"ABORTA: sha256 recalculado de {script_lista.name} "
            f"({sha_script_recalc}) != script_sha256 declarado na lista fixa "
            f"({d['script_sha256']})"
        )

    por_basename_e, _ = carregar_manifest_e()

    linhas = []
    for corrida in d["corridas"]:
        caminho = Path(corrida["arquivo"])
        if not caminho.is_file():
            raise FileNotFoundError(f"corrida da lista fixa nao existe: {caminho}")
        sha_recalc = sha256_arquivo(caminho)
        if sha_recalc != corrida["sha256"]:
            raise AssertionError(
                f"ABORTA: sha256 recalculado de {caminho.name} ({sha_recalc}) "
                f"!= sha256 declarado na lista fixa do fisico ({corrida['sha256']})"
            )
        em_manifest_e, sha_confere_e, tipo_match = cruzar_com_manifest_e(
            caminho, sha_recalc, por_basename_e, {}
        )
        linhas.append({
            "caminho": str(caminho.resolve()),
            "sha256": sha_recalc,
            "sha_recalculado": True,
            "tamanho_bytes": caminho.stat().st_size,
            "mtime": mtime_iso(caminho),
            "grupo": "lista_fixa_tab_retention",
            "cidade": corrida["cidade"],
            "quadrante": corrida["quadrante"],
            "config": corrida["config"],
            "split_seed": corrida["split_seed"],
            "arquivo_lista": str(LISTA_FIXA_TAB_RETENTION.resolve()),
            "script_sha256": d["script_sha256"],
            "em_manifest_E": em_manifest_e,
            "sha_confere_E": sha_confere_e,
        })
    return linhas


# --------------------------------------------------------------------------
# Montagem
# --------------------------------------------------------------------------

def montar_manifest():
    por_basename, por_relpath = carregar_manifest_e()
    por_basename_prov = carregar_proveniencia()
    linhas = []
    contagens = {}

    # grupo 1: carimbos fismat 11/09 (com M1)
    arquivos1, g1 = grupo_1_fismat_stamps()
    for c in arquivos1:
        ln = linha_artefato(c, g1, None, por_basename, por_relpath)
        enriquecer_com_proveniencia(ln, c, por_basename_prov)
        linhas.append(ln)
    if len(arquivos1) != 4:
        raise AssertionError(f"ABORTA: {g1} esperava 4 basenames, achou {len(arquivos1)}")
    contagens[g1] = {"esperado": 4, "encontrado": len(arquivos1), "assercao": "basenames == 4"}

    # grupo 2: alvo_stats + fismat_controle_negativo (com M1)
    arquivos2, g2 = grupo_2_alvo()
    for c in arquivos2:
        ln = linha_artefato(c, g2, None, por_basename, por_relpath)
        enriquecer_com_proveniencia(ln, c, por_basename_prov)
        linhas.append(ln)
    if len(arquivos2) != 10:
        raise AssertionError(f"ABORTA: {g2} esperava 10 basenames, achou {len(arquivos2)}")
    contagens[g2] = {"esperado": 10, "encontrado": len(arquivos2), "assercao": "basenames == 10"}

    # grupo 3: baselines_v2
    arquivos3, g3 = grupo_3_baselines()
    for c in arquivos3:
        linhas.append(linha_artefato(c, g3, None, por_basename, por_relpath))
    if len(arquivos3) != 16:
        raise AssertionError(f"ABORTA: {g3} esperava 16 basenames, achou {len(arquivos3)}")
    contagens[g3] = {"esperado": 16, "encontrado": len(arquivos3), "assercao": "basenames == 16"}

    # grupo 4: runs (M2 -- assercao por basename)
    runs = grupo_4_runs()
    esperado_basenames = {
        "run_c0c1cf_g10b2": 80,
        "run_c0c1_bauru_g5b2_s42": 4,
        "run_mlpcf": 80,
        "run_mlpcf_smoke": None,
    }
    for grupo_nome, lst in runs.items():
        for c in lst:
            linhas.append(linha_artefato(c, grupo_nome, None, por_basename, por_relpath))
        n_basenames = len({c.name for c in lst})
        esp = esperado_basenames[grupo_nome]
        if esp is not None and n_basenames != esp:
            raise AssertionError(
                f"ABORTA: {grupo_nome} esperava {esp} basenames (as copias "
                f"dados/x treinos/ dobram os arquivos, nao os basenames), "
                f"achou {n_basenames} basenames em {len(lst)} arquivos"
            )
        contagens[grupo_nome] = {
            "esperado_basenames": esp,
            "encontrado_arquivos": len(lst),
            "encontrado_basenames": n_basenames,
            "assercao": f"basenames == {esp}" if esp is not None else "dinamico (sem assercao de contagem)",
        }

    # grupo 5: rodadas do fio (dinamico, como no v1)
    arquivos5, g5 = grupo_5_rodadas()
    for c in arquivos5:
        linhas.append(linha_artefato(c, g5, None, por_basename, por_relpath))
    contagens[g5] = {
        "esperado": None,
        "encontrado": len(arquivos5),
        "assercao": "nenhuma -- grupo dinamico por construcao (glob dos artefatos do proprio fio, cresce a cada rodada)",
    }

    # grupo novo: scripts_produtores
    linhas_sp = grupo_scripts_produtores()
    linhas.extend(linhas_sp)
    contagens["scripts_produtores"] = {
        "esperado": len(SCRIPTS_PRODUTORES),
        "encontrado": len(linhas_sp),
        "assercao": "sha256 de cada script == esperado declarado (proveniencia/coordenador R2), onde houver",
    }

    # grupo novo: tensores_cftudo (SEM re-hash)
    linhas_tc = grupo_tensores_cftudo()
    linhas.extend(linhas_tc)
    if len(linhas_tc) != 16:
        raise AssertionError(f"ABORTA: tensores_cftudo esperava 16, achou {len(linhas_tc)}")
    contagens["tensores_cftudo"] = {
        "esperado": 16,
        "encontrado": len(linhas_tc),
        "assercao": "16 == 16; tamanho==rf_data_bytes e sha==rf_data_sha256 do run JSON; sha em E/manifest.jsonl",
    }

    # grupo novo: scripts_artigo2
    linhas_sa2 = grupo_scripts_artigo2()
    linhas.extend(linhas_sa2)
    contagens["scripts_artigo2"] = {
        "esperado": None,
        "encontrado": len(linhas_sa2),
        "assercao": "nenhuma -- glob dinamico de MDPI_Mathematics/scripts/*.py em tempo de execucao",
    }

    # grupo novo: mlpcf_t11_E2 (+ marca sha_E2_confere em run_mlpcf)
    linhas_e2, marcadas_e2 = grupo_mlpcf_t11_e2(runs.get("run_mlpcf", []))
    linhas.extend(linhas_e2)
    if len(linhas_e2) != 40:
        raise AssertionError(f"ABORTA: mlpcf_t11_E2 esperava 40, achou {len(linhas_e2)}")
    contagens["mlpcf_t11_E2"] = {
        "esperado": 40,
        "encontrado": len(linhas_e2),
        "assercao": "40 == 40; sha256 do run JSON recalculado == sha256_run_json do gancho E2",
    }
    for ln in linhas:
        if ln.get("grupo") == "run_mlpcf":
            ln["sha_E2_confere"] = marcadas_e2.get(ln["caminho"], False)

    # grupo novo: lista_fixa_tab_retention
    linhas_lf = grupo_lista_fixa_tab_retention()
    linhas.extend(linhas_lf)
    if len(linhas_lf) != 20:
        raise AssertionError(f"ABORTA: lista_fixa_tab_retention esperava 20, achou {len(linhas_lf)}")
    contagens["lista_fixa_tab_retention"] = {
        "esperado": 20,
        "encontrado": len(linhas_lf),
        "assercao": "20 == 20; sha256 recalculado == sha256 da lista fixa (script_sha256 250d43ce... conferido)",
    }

    linhas.sort(key=lambda ln: (ln["grupo"], ln["caminho"]))
    return linhas, contagens


def linha_para_texto(linha: dict) -> str:
    return json.dumps(linha, ensure_ascii=False, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true", help="calcula sha256 (sempre feito; flag documental)")
    ap.add_argument("--saida", type=Path, default=DEFAULT_SAIDA)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    linhas, contagens = montar_manifest()
    texto_jsonl = "\n".join(linha_para_texto(l) for l in linhas) + ("\n" if linhas else "")

    if args.dry_run:
        sys.stdout.write(texto_jsonl)
        print(f"# [dry-run] {len(linhas)} linhas; contagens={contagens}", file=sys.stderr)
        return

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(texto_jsonl, encoding="utf-8")

    script_sha = sha256_arquivo(SCRIPT_PATH)
    meta = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "script": str(SCRIPT_PATH),
        "script_sha256": script_sha,
        "saida": str(args.saida.resolve()),
        "saida_sha256": sha256_arquivo(args.saida),
        "n_linhas": len(linhas),
        "contagens_por_grupo": contagens,
        "manifest_E_usado": str(MANIFEST_E),
        "manifest_E_existe": MANIFEST_E.exists(),
        "proveniencia_usada": str(PROVENIENCIA),
        "lista_fixa_tab_retention_usada": str(LISTA_FIXA_TAB_RETENTION),
        "gancho_E2_usado": str(GANCHO_E2),
        "versao": "v2",
    }
    sidecar = args.saida.with_name(args.saida.stem + ".meta.json")
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"gravado: {args.saida} ({len(linhas)} linhas)")
    print(f"meta: {sidecar}")


if __name__ == "__main__":
    main()
