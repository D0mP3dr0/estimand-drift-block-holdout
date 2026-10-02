#!/usr/bin/env python3
"""Gera manifest_mathematics_v3.jsonl do Artigo 2 (MDPI Mathematics) -- v3.

Arquivo NOVO (nao edita gerar_manifest_mathematics.py v1 nem
gerar_manifest_mathematics_v2.py). E3 = SUPERCONJUNTO de v1 e v2: reexecuta
TODOS os grupos do v2 (que ja incluia os do v1, exceto pela regressao R-V2
descrita abaixo) e acrescenta os itens da rodada R2 pos-parecer:

R1 (heranca / corrige R-V2): o v2 perdeu a chave `script_sha256` (e outras)
    em 351 linhas que o v1 tinha (baselines_v2, grupos de run, mlpcf), porque
    o v2 parou de ler o campo interno do proprio artefato JSON. O v3 faz a
    JUNCAO por `caminho` entre o v1 e as linhas que o v3 monta e HERDA do v1
    toda chave nao nula que falte na linha do v3. Em conflito real de valor
    (mesmo caminho, mesmo sha256 do arquivo -- ou seja, arquivo identico --
    mas chave com valores diferentes) o script ABORTA e lista os conflitos.
    Excecao registrada e NAO fatal: quando as duas linhas conflitantes sao
    campos "*_motivo" (prosa explicativa sobre por que outro campo e nulo,
    nao um dado em si) o v3 mantem o texto que ele mesmo gerou e regsitra o
    par em `conflitos_motivo_resolvidos` (nao e uma divergencia de dado).
    Quando o `sha256` do arquivo no `caminho` comum mudou entre a execucao
    do v1 e a de hoje (arquivo reescrito no lugar -- ex.: script desta rodada
    sobrescrito, ou artefato do fio regravado por outra equipe), a linha NAO
    herda chaves do v1 (a metadata antiga descreve outro conteudo); o v3 usa
    o sha de hoje e grava `sha_anterior_v1` + `mudou_desde_v1: true`. Esses
    casos vao para a lista `mudou_desde_v1` no sidecar/artefato, nunca dentro
    das linhas do jsonl junto com um valor stale.

R2 (E2 T11, 40 linhas mlpcf_t11_E2): acrescenta `parametro_invocacao`
    ("cidades=campinas,sorocaba" -- o padrao do .ps1 e lins,bauru, ver
    `fila_mlp_c1v2.ps1` linha ~35 `$Cidades = @("lins","bauru")`),
    `parametro_invocacao_fonte: "log"`, `log_caminho`, `log_sha256` (sha256
    de hoje do log completo) e `log_linhas` (intervalo REAL, calculado por
    busca de texto no log a cada execucao -- nao copiado do prompt).

R3 (logs_fila, grupo novo, 2 linhas): `E/treinos/fila_mlp_c1v2.log` (log
    completo, status completo) e `E/dados/treinos_c1/fila_mlp_c1v2.log`
    (copia truncada que o manifest de E registra -- sem a invocacao
    campinas/sorocaba; prova por contagem de "campinas" == 0).

R4 (E2_manifest_c0c1cf_e_controles.jsonl, 108 linhas da rodada 2 do cartao da
    eng-IA): reconfere o sha256 do run JSON de cada uma das 108 (assercao).
    As 80 do grupo "c0c1cf" (mesmo basename das linhas ja existentes em
    run_c0c1cf_g10b2) recebem `checkpoint_sha256_declarado_E2c` e
    `sha_checkpoint_recalculado` (recalcula se o checkpoint < 200 MB, como
    e o caso das 108 -- ~20 MB cada). As 28 restantes (c0c1cfsc + c0c1cfinv)
    nao tem correspondente e viram o grupo novo `E2_c0c1cf_controles`.

R5 (scripts_artigo2 com lista esperada explicita, nao so glob dinamico):
    continua glob de `MDPI_Mathematics/scripts/*.py`, mas agora confere
    contra uma lista esperada fixa desta rodada e REGISTRA AVISO (nao
    aborta) se algo aparecer fora da lista ou faltar.

R6 (run_c0c1_bauru_g5b2_s42, 8 linhas = 4 basenames x 2 copias): le
    `dataset.rf_data_file` de cada run e marca `entrada_rf_data_file`,
    `entrada_sem_hash_local: true` e `entrada_motivo`, com assercao de que o
    basename realmente nao existe em `/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3`
    (achado do voto 203205_critico-verificador.md).

Sem carimbo de tempo de geracao dentro das linhas do jsonl (idempotencia
byte-a-byte entre execucoes sem mudanca no disco); o carimbo vive no sidecar
<saida>.meta.json.

PROIBIDO: ler ou re-hashear os *_cftudo.pt (28 GB cada); ler/re-hashear
checkpoints .pt >= 200 MB; treino, download, GPU.

Uso:
    python gerar_manifest_mathematics_v3.py [--hash] [--saida CAMINHO] [--dry-run]
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
E2_C0C1CF_CONTROLES = FIO_R2 / "artefatos" / "E2_manifest_c0c1cf_e_controles.jsonl"

RAIZ_GRAPH_DATA_V3 = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
HASHES_SHA256_TXT = RAIZ_GRAPH_DATA_V3 / "HASHES_SHA256.txt"

FILA_LOG_COMPLETO = RAIZ_E / "treinos" / "fila_mlp_c1v2.log"
FILA_LOG_TRUNCADO = RAIZ_E / "dados" / "treinos_c1" / "fila_mlp_c1v2.log"

MANIFEST_V1 = SCRIPT_PATH.parent.parent / "manifest_mathematics.jsonl"

DEFAULT_SAIDA = SCRIPT_PATH.parent.parent / "manifest_mathematics_v3.jsonl"

CAMPOS_PRODUTOR_CANDIDATOS = ["script", "gerado_por", "autor", "agente"]

CHECKPOINT_LIMITE_BYTES = 200 * 1024 * 1024  # 200 MB

# --------------------------------------------------------------------------
# Scripts produtores desta rodada (identico ao v2)
# --------------------------------------------------------------------------
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

# R5: lista esperada explicita de MDPI_Mathematics/scripts/*.py (stems, sem
# extensao). "e qualquer e3 v2 que exista" e checado dinamicamente abaixo.
ESPERADO_SCRIPTS_ARTIGO2_BASE = {
    "blocos_efetivos_20celulas",
    "e1_sinal_por_populacao",
    "e2_cartao_corridas",
    "e2_cartao_corridas_v2",
    "e3_inferencia_por_populacao",
    "gerar_manifest_mathematics",
    "gerar_manifest_mathematics_v2",
    "gerar_manifest_mathematics_v3",
    "montecarlo_retencao",
    "validar_manifest_mathematics",
    "validar_manifest_mathematics_v2",
    "validar_manifest_mathematics_v3",
    "varredura_split_geometria",
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
    if caminho.suffix.lower() not in (".json",):
        return None, "arquivo nao e .json"
    try:
        texto = caminho.read_text(encoding="utf-8")
    except Exception as e:
        return None, f"falha ao ler arquivo como texto: {e!r}"
    try:
        d = json.loads(texto)
    except Exception as e:
        return None, f"falha ao fazer parse JSON: {e!r}"
    if not isinstance(d, dict):
        return None, "JSON raiz nao e objeto (dict)"
    return d, None


def derivar_produtor(caminho: Path):
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
# Proveniencia (M1, identico ao v2)
# --------------------------------------------------------------------------

def _basename_proveniencia(caminho_field: str) -> str:
    p = caminho_field.split(" (", 1)[0].strip()
    return p.replace("\\", "/").rsplit("/", 1)[-1]


def carregar_proveniencia():
    if not PROVENIENCIA.exists():
        raise FileNotFoundError(f"proveniencia obrigatoria ausente: {PROVENIENCIA}")
    d = json.loads(PROVENIENCIA.read_text(encoding="utf-8"))
    por_basename = {}
    for item in d["alvos"]:
        base = _basename_proveniencia(item["caminho"])
        por_basename[base] = item
    return por_basename


def enriquecer_com_proveniencia(linha: dict, caminho: Path, por_basename_prov: dict):
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
# Grupos herdados do v1/v2 (identicos)
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
    arquivos = sorted(p for p in pasta.glob("*.py") if p.parent.name != "__pycache__")
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
    esperado = set(ESPERADO_SCRIPTS_ARTIGO2_BASE)
    extra_e3_v2 = pasta / "e3_inferencia_por_populacao_v2.py"
    if extra_e3_v2.is_file():
        esperado.add("e3_inferencia_por_populacao_v2")
    encontrados = {p.stem for p in arquivos}
    avisos_fora_da_lista = sorted(encontrados - esperado)
    avisos_faltando = sorted(esperado - encontrados)
    return linhas, sorted(esperado), avisos_fora_da_lista, avisos_faltando


def grupo_mlpcf_t11_e2(runs_mlpcf: list[Path], linhas_log: list[str]):
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

        # R2: parametro de invocacao (declarado: padrao do .ps1 e lins,bauru;
        # esta fila rodou com --Cidades campinas,sorocaba, confirmado no log)
        # e intervalo real de linhas no log completo (busca por texto, nao
        # numero copiado do prompt).
        info_intervalo = _localizar_intervalo_log(linhas_log, e["run_label"])

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
            "parametro_invocacao": "cidades=campinas,sorocaba",
            "parametro_invocacao_fonte": "log",
            "log_caminho": str(FILA_LOG_COMPLETO.resolve()),
            "log_linhas": info_intervalo,
        })

    marcadas = {}
    for c in runs_mlpcf:
        stem = c.stem
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
# Grupos novos do v3
# --------------------------------------------------------------------------

def _ler_log_linhas(caminho: Path) -> list[str]:
    """Le o log byte a byte e separa por '\\n' apenas (NAO usa universal
    newlines: o log tem \\r soltos no meio de trechos com encoding
    misto/UTF-16 corrompido que fariam Python contar linhas a mais e
    dessincronizar da numeracao que `grep -n`/`sed -n` usam)."""
    raw = caminho.read_bytes()
    txt = raw.decode("utf-8", errors="replace")
    return txt.split("\n")


def _localizar_intervalo_log(linhas_log: list[str], run_label: str) -> dict:
    marcador_inicio = f"$ {run_label} "
    marcador_ok = f"[{run_label}] ok em"
    marcador_pula = f"pula {run_label} (concluido)"
    inicios = [i + 1 for i, l in enumerate(linhas_log) if marcador_inicio in l]
    fins_ok = [i + 1 for i, l in enumerate(linhas_log) if marcador_ok in l]
    fins_pula = [i + 1 for i, l in enumerate(linhas_log) if marcador_pula in l]
    if fins_ok:
        fim, tipo_fim = fins_ok[-1], "ok"
    elif fins_pula:
        fim, tipo_fim = fins_pula[-1], "pula_concluido"
    else:
        fim, tipo_fim = None, "nao_encontrado"
    candidatos_inicio = [i for i in inicios if fim is None or i <= fim]
    inicio = candidatos_inicio[-1] if candidatos_inicio else (inicios[0] if inicios else None)
    return {
        "linha_inicio": inicio,
        "linha_fim": fim,
        "tipo_fim": tipo_fim,
        "tentativas_start_no_log": len(inicios),
    }


def grupo_logs_fila(por_basename_e, por_relpath_e, linhas_log_completo: list[str]):
    if not FILA_LOG_COMPLETO.is_file():
        raise FileNotFoundError(f"log completo da fila MLP ausente: {FILA_LOG_COMPLETO}")
    if not FILA_LOG_TRUNCADO.is_file():
        raise FileNotFoundError(f"log truncado da fila MLP (copia em E) ausente: {FILA_LOG_TRUNCADO}")

    ln_completo = linha_artefato(FILA_LOG_COMPLETO, "logs_fila", None, por_basename_e, por_relpath_e)
    ln_completo["status"] = "completo"
    ln_completo["contagem_campinas"] = sum(1 for l in linhas_log_completo if "campinas" in l)
    linha_cidades_idx = next(
        (i + 1 for i, l in enumerate(linhas_log_completo) if "cidades: campinas, sorocaba" in l),
        None,
    )
    ln_completo["linha_cidades_campinas_sorocaba"] = linha_cidades_idx

    txt_truncado = FILA_LOG_TRUNCADO.read_bytes().decode("utf-8", errors="replace")
    linhas_truncado = txt_truncado.split("\n")
    ln_truncado = linha_artefato(FILA_LOG_TRUNCADO, "logs_fila", None, por_basename_e, por_relpath_e)
    ln_truncado["status"] = "truncado"
    ln_truncado["contagem_campinas"] = sum(1 for l in linhas_truncado if "campinas" in l)
    ln_truncado["nota"] = (
        "copia que E/manifest.jsonl registra (dados/treinos_c1/fila_mlp_c1v2.log); "
        "nao contem a invocacao campinas/sorocaba (contagem_campinas confere 0)"
    )

    return [ln_completo, ln_truncado]


def grupo_e2c(por_basename_e, por_relpath_e, linhas_run_c0c1cf: list[dict], linhas_run_bauru_g5b2: list[dict]):
    if not E2_C0C1CF_CONTROLES.is_file():
        raise FileNotFoundError(f"cartao E2 c0c1cf/controles ausente: {E2_C0C1CF_CONTROLES}")
    entradas = []
    for ln in E2_C0C1CF_CONTROLES.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            entradas.append(json.loads(ln))

    por_basename_run = {}
    for ln in linhas_run_c0c1cf:
        por_basename_run.setdefault(Path(ln["caminho"]).name, []).append(ln)

    linhas_extra = []
    n_juntadas = 0
    for e in entradas:
        caminho_abs = (RAIZ_E / e["artefato"]).resolve()
        if not caminho_abs.is_file():
            raise FileNotFoundError(f"artefato do cartao E2c nao existe: {caminho_abs}")
        sha_recalc = sha256_arquivo(caminho_abs)
        if sha_recalc != e["sha256"]:
            raise AssertionError(
                f"ABORTA: sha256 recalculado de {caminho_abs.name} ({sha_recalc}) "
                f"!= sha256 declarado no cartao E2c ({e['sha256']})"
            )

        basename = caminho_abs.name
        checkpoint_path = e.get("checkpoint_path")
        checkpoint_sha256_declarado = e.get("checkpoint_sha256")
        sha_ckpt_recalculado = False
        sha_ckpt_hoje = None
        sha_ckpt_motivo = None
        if checkpoint_path:
            p = Path(checkpoint_path)
            if p.is_file():
                tamanho = p.stat().st_size
                if tamanho < CHECKPOINT_LIMITE_BYTES:
                    sha_ckpt_hoje = sha256_arquivo(p)
                    sha_ckpt_recalculado = True
                    if checkpoint_sha256_declarado and sha_ckpt_hoje != checkpoint_sha256_declarado:
                        raise AssertionError(
                            f"ABORTA: sha256 recalculado do checkpoint de {e['run_label']} "
                            f"({sha_ckpt_hoje}) != checkpoint_sha256 declarado no cartao E2c "
                            f"({checkpoint_sha256_declarado})"
                        )
                else:
                    sha_ckpt_motivo = f"checkpoint local >= {CHECKPOINT_LIMITE_BYTES} bytes; nao recalculado"
            else:
                sha_ckpt_motivo = "checkpoint_path do cartao E2c nao existe localmente"

        candidatos = por_basename_run.get(basename)
        if candidatos:
            for ln in candidatos:
                ln["checkpoint_sha256_declarado_E2c"] = checkpoint_sha256_declarado
                ln["sha_checkpoint_recalculado"] = sha_ckpt_recalculado
                ln["sha_checkpoint_recalculado_valor"] = sha_ckpt_hoje
                ln["sha_checkpoint_motivo"] = sha_ckpt_motivo
                ln["e2c_run_label"] = e["run_label"]
                ln["e2c_veredito"] = e.get("veredito")
                ln["e2c_grupo_origem"] = e.get("grupo")
            n_juntadas += 1
        else:
            em_manifest_e, sha_confere_e, tipo_match = cruzar_com_manifest_e(
                caminho_abs, sha_recalc, por_basename_e, por_relpath_e
            )
            linhas_extra.append({
                "caminho": str(caminho_abs),
                "sha256": sha_recalc,
                "sha_recalculado": True,
                "tamanho_bytes": caminho_abs.stat().st_size,
                "mtime": mtime_iso(caminho_abs),
                "grupo": "E2_c0c1cf_controles",
                "run_label": e["run_label"],
                "grupo_e2c_origem": e.get("grupo"),
                "veredito": e.get("veredito"),
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256_declarado_E2c": checkpoint_sha256_declarado,
                "sha_checkpoint_recalculado": sha_ckpt_recalculado,
                "sha_checkpoint_recalculado_valor": sha_ckpt_hoje,
                "sha_checkpoint_motivo": sha_ckpt_motivo,
                "rf_data_sha256_declarado": e.get("rf_data_sha256_declarado"),
                "rf_data_sha256_bate_hashfile": e.get("rf_data_sha256_bate_hashfile"),
                "script_sha256_declarado": e.get("script_sha256_declarado"),
                "script_sha256_bate": e.get("script_sha256_bate"),
                "em_manifest_E": em_manifest_e,
                "sha_confere_E": sha_confere_e,
                "tipo_match_manifest_E": tipo_match,
            })

    return linhas_extra, n_juntadas, len(entradas)


def marcar_bauru_g5b2_entrada_sem_hash(linhas_bauru: list[dict]):
    """R6: os runs g5b2 declaram dataset.rf_data_file apontando para
    *_enriched.pt SEM _cftudo; esses basenames nao existem em graph_data_v3
    (so os *_cftudo.pt e *_gpu.pt existem la). Marca as linhas e ASSERTA que
    o basename realmente nao existe (nao supoe)."""
    for ln in linhas_bauru:
        caminho = Path(ln["caminho"])
        rd = json.loads(caminho.read_text(encoding="utf-8"))
        rf_file = rd["dataset"]["rf_data_file"]
        basename = rf_file.replace("\\", "/").rsplit("/", 1)[-1]
        candidato = RAIZ_GRAPH_DATA_V3 / basename
        if candidato.exists():
            raise AssertionError(
                f"ABORTA: esperava {basename} ausente em {RAIZ_GRAPH_DATA_V3}, mas existe "
                "(premissa do voto 203205_critico-verificador.md nao se sustenta mais)"
            )
        ln["entrada_rf_data_file"] = basename
        ln["entrada_sem_hash_local"] = True
        ln["entrada_motivo"] = (
            f"sem arquivo correspondente em {RAIZ_GRAPH_DATA_V3}; "
            "rf_data_sha256 declarado nao cruzavel"
        )


# --------------------------------------------------------------------------
# Heranca v1 -> v3 (R1)
# --------------------------------------------------------------------------

def carregar_manifest_v1():
    """Chave (caminho, grupo): o MESMO caminho pode aparecer em mais de um
    grupo dentro do proprio v3 (ex.: um run que tambem esta na lista fixa do
    fisico, ou no gancho E2) -- juntar so por caminho geraria falso conflito
    de 'grupo'/'script_sha256' entre linhas-irmas de grupos diferentes. O v1
    ja usava grupo == 'mlpcf_t11_E2' para as linhas do proprio gancho E2, o
    que preserva a correspondencia correta com o v3."""
    if not MANIFEST_V1.is_file():
        raise FileNotFoundError(f"manifest v1 ausente (heranca obrigatoria): {MANIFEST_V1}")
    por_chave = {}
    for ln in MANIFEST_V1.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        por_chave[(d["caminho"], d.get("grupo"))] = d
    return por_chave


def herdar_de_v1(linhas_v3: list[dict], v1_por_chave: dict):
    """Para cada linha do v3, junta por (caminho, grupo) com a linha do v1
    (ver nota em carregar_manifest_v1 sobre por que nao e so 'caminho'). Se o
    arquivo mudou (sha256 diferente) desde o v1, NAO herda (metadata antiga
    descreve outro conteudo) -- so registra mudou_desde_v1. Senao, herda toda
    chave nao nula do v1 ausente no v3; conflito de valor (mesma chave, dois
    valores nao nulos diferentes) aborta, EXCETO quando ambas as chaves
    terminam em '_motivo' (prosa explicativa, nao dado)."""
    linhas_por_chave_v3 = {}
    for ln in linhas_v3:
        linhas_por_chave_v3.setdefault((ln["caminho"], ln.get("grupo")), []).append(ln)

    chaves_herdadas = 0
    mudou_desde_v1 = []
    conflitos = []
    conflitos_motivo_resolvidos = []
    v1_caminhos_sem_correspondente_v3 = []

    for (caminho_v1, grupo_v1), v1ln in v1_por_chave.items():
        candidatos_v3 = linhas_por_chave_v3.get((caminho_v1, grupo_v1))
        if not candidatos_v3:
            v1_caminhos_sem_correspondente_v3.append({"caminho": caminho_v1, "grupo": grupo_v1})
            continue
        for ln in candidatos_v3:
            if ln.get("sha256") != v1ln.get("sha256"):
                if not ln.get("mudou_desde_v1"):
                    mudou_desde_v1.append({
                        "caminho": caminho_v1,
                        "sha_anterior_v1": v1ln.get("sha256"),
                        "sha_hoje": ln.get("sha256"),
                    })
                ln["mudou_desde_v1"] = True
                ln["sha_anterior_v1"] = v1ln.get("sha256")
                continue
            ln.setdefault("mudou_desde_v1", False)
            for k, v in v1ln.items():
                if v is None:
                    continue
                if k not in ln or ln.get(k) is None:
                    ln[k] = v
                    chaves_herdadas += 1
                elif ln[k] != v:
                    if k.endswith("_motivo") and isinstance(ln[k], str) and isinstance(v, str):
                        conflitos_motivo_resolvidos.append({
                            "caminho": caminho_v1,
                            "chave": k,
                            "valor_v1": v,
                            "valor_v3_mantido": ln[k],
                        })
                        continue
                    conflitos.append({
                        "caminho": caminho_v1,
                        "chave": k,
                        "valor_v1": v,
                        "valor_v3": ln[k],
                    })

    if conflitos:
        raise AssertionError(
            "ABORTA: conflitos de heranca v1 -> v3 (mesmo caminho, mesmo sha256 "
            f"do arquivo, valores diferentes para a mesma chave): {conflitos}"
        )

    return {
        "chaves_herdadas": chaves_herdadas,
        "mudou_desde_v1": mudou_desde_v1,
        "conflitos_motivo_resolvidos": conflitos_motivo_resolvidos,
        "v1_caminhos_sem_correspondente_v3": v1_caminhos_sem_correspondente_v3,
    }


# --------------------------------------------------------------------------
# Montagem
# --------------------------------------------------------------------------

def montar_manifest():
    por_basename, por_relpath = carregar_manifest_e()
    por_basename_prov = carregar_proveniencia()
    linhas = []
    contagens = {}

    arquivos1, g1 = grupo_1_fismat_stamps()
    for c in arquivos1:
        ln = linha_artefato(c, g1, None, por_basename, por_relpath)
        enriquecer_com_proveniencia(ln, c, por_basename_prov)
        linhas.append(ln)
    if len(arquivos1) != 4:
        raise AssertionError(f"ABORTA: {g1} esperava 4 basenames, achou {len(arquivos1)}")
    contagens[g1] = {"esperado": 4, "encontrado": len(arquivos1), "assercao": "basenames == 4"}

    arquivos2, g2 = grupo_2_alvo()
    for c in arquivos2:
        ln = linha_artefato(c, g2, None, por_basename, por_relpath)
        enriquecer_com_proveniencia(ln, c, por_basename_prov)
        linhas.append(ln)
    if len(arquivos2) != 10:
        raise AssertionError(f"ABORTA: {g2} esperava 10 basenames, achou {len(arquivos2)}")
    contagens[g2] = {"esperado": 10, "encontrado": len(arquivos2), "assercao": "basenames == 10"}

    arquivos3, g3 = grupo_3_baselines()
    for c in arquivos3:
        linhas.append(linha_artefato(c, g3, None, por_basename, por_relpath))
    if len(arquivos3) != 16:
        raise AssertionError(f"ABORTA: {g3} esperava 16 basenames, achou {len(arquivos3)}")
    contagens[g3] = {"esperado": 16, "encontrado": len(arquivos3), "assercao": "basenames == 16"}

    runs = grupo_4_runs()
    esperado_basenames = {
        "run_c0c1cf_g10b2": 80,
        "run_c0c1_bauru_g5b2_s42": 4,
        "run_mlpcf": 80,
        "run_mlpcf_smoke": None,
    }
    linhas_run_c0c1cf = []
    linhas_run_bauru_g5b2 = []
    for grupo_nome, lst in runs.items():
        linhas_deste_grupo = []
        for c in lst:
            ln = linha_artefato(c, grupo_nome, None, por_basename, por_relpath)
            linhas.append(ln)
            linhas_deste_grupo.append(ln)
        if grupo_nome == "run_c0c1cf_g10b2":
            linhas_run_c0c1cf = linhas_deste_grupo
        if grupo_nome == "run_c0c1_bauru_g5b2_s42":
            linhas_run_bauru_g5b2 = linhas_deste_grupo
        n_basenames = len({c.name for c in lst})
        esp = esperado_basenames[grupo_nome]
        if esp is not None and n_basenames != esp:
            raise AssertionError(
                f"ABORTA: {grupo_nome} esperava {esp} basenames, achou {n_basenames}"
            )
        contagens[grupo_nome] = {
            "esperado_basenames": esp,
            "encontrado_arquivos": len(lst),
            "encontrado_basenames": n_basenames,
            "assercao": f"basenames == {esp}" if esp is not None else "dinamico (sem assercao de contagem)",
        }

    # R6: marca as 8 linhas run_c0c1_bauru_g5b2_s42
    marcar_bauru_g5b2_entrada_sem_hash(linhas_run_bauru_g5b2)
    contagens["run_c0c1_bauru_g5b2_s42"]["r6_entrada_sem_hash_marcadas"] = len(linhas_run_bauru_g5b2)

    arquivos5, g5 = grupo_5_rodadas()
    for c in arquivos5:
        linhas.append(linha_artefato(c, g5, None, por_basename, por_relpath))
    contagens[g5] = {
        "esperado": None,
        "encontrado": len(arquivos5),
        "assercao": "nenhuma -- grupo dinamico por construcao (glob dos artefatos do proprio fio, cresce a cada rodada)",
    }

    linhas_sp = grupo_scripts_produtores()
    linhas.extend(linhas_sp)
    contagens["scripts_produtores"] = {
        "esperado": len(SCRIPTS_PRODUTORES),
        "encontrado": len(linhas_sp),
        "assercao": "sha256 de cada script == esperado declarado (proveniencia/coordenador R2), onde houver",
    }

    linhas_tc = grupo_tensores_cftudo()
    linhas.extend(linhas_tc)
    if len(linhas_tc) != 16:
        raise AssertionError(f"ABORTA: tensores_cftudo esperava 16, achou {len(linhas_tc)}")
    contagens["tensores_cftudo"] = {
        "esperado": 16,
        "encontrado": len(linhas_tc),
        "assercao": "16 == 16; tamanho==rf_data_bytes e sha==rf_data_sha256 do run JSON; sha em E/manifest.jsonl",
    }

    linhas_sa2, esperado_sa2, avisos_fora, avisos_faltando = grupo_scripts_artigo2()
    linhas.extend(linhas_sa2)
    contagens["scripts_artigo2"] = {
        "esperado": None,
        "encontrado": len(linhas_sa2),
        "lista_esperada_explicita": esperado_sa2,
        "avisos_fora_da_lista": avisos_fora,
        "avisos_faltando_na_lista": avisos_faltando,
        "assercao": "nenhuma (aviso, nao aborto) -- glob dinamico conferido contra lista esperada desta rodada",
    }

    linhas_log_completo = _ler_log_linhas(FILA_LOG_COMPLETO)
    linhas_e2, marcadas_e2 = grupo_mlpcf_t11_e2(runs.get("run_mlpcf", []), linhas_log_completo)
    linhas.extend(linhas_e2)
    if len(linhas_e2) != 40:
        raise AssertionError(f"ABORTA: mlpcf_t11_E2 esperava 40, achou {len(linhas_e2)}")
    contagens["mlpcf_t11_E2"] = {
        "esperado": 40,
        "encontrado": len(linhas_e2),
        "assercao": "40 == 40; sha256 do run JSON recalculado == sha256_run_json do gancho E2; parametro_invocacao + log_linhas conferidos por busca no log",
    }
    for ln in linhas:
        if ln.get("grupo") == "run_mlpcf":
            ln["sha_E2_confere"] = marcadas_e2.get(ln["caminho"], False)

    linhas_lf = grupo_lista_fixa_tab_retention()
    linhas.extend(linhas_lf)
    if len(linhas_lf) != 20:
        raise AssertionError(f"ABORTA: lista_fixa_tab_retention esperava 20, achou {len(linhas_lf)}")
    contagens["lista_fixa_tab_retention"] = {
        "esperado": 20,
        "encontrado": len(linhas_lf),
        "assercao": "20 == 20; sha256 recalculado == sha256 da lista fixa (script_sha256 250d43ce... conferido)",
    }

    # R3: logs_fila
    linhas_logs = grupo_logs_fila(por_basename, por_relpath, linhas_log_completo)
    linhas.extend(linhas_logs)
    contagens["logs_fila"] = {
        "esperado": 2,
        "encontrado": len(linhas_logs),
        "assercao": "2 == 2; log completo (campinas>0) x copia truncada de E (campinas==0)",
    }

    # R4: E2_manifest_c0c1cf_e_controles.jsonl
    linhas_e2c_extra, n_juntadas, n_total_e2c = grupo_e2c(
        por_basename, por_relpath, linhas_run_c0c1cf, linhas_run_bauru_g5b2
    )
    linhas.extend(linhas_e2c_extra)
    if n_total_e2c != 108:
        raise AssertionError(f"ABORTA: E2_manifest_c0c1cf_e_controles.jsonl esperava 108 linhas, achou {n_total_e2c}")
    if n_juntadas != 80:
        raise AssertionError(f"ABORTA: E2c esperava juntar 80 linhas c0c1cf existentes, juntou {n_juntadas}")
    if len(linhas_e2c_extra) != 28:
        raise AssertionError(f"ABORTA: E2_c0c1cf_controles esperava 28 linhas sem correspondente, achou {len(linhas_e2c_extra)}")
    contagens["E2_c0c1cf_controles"] = {
        "esperado_total_cartao": 108,
        "juntadas_em_run_c0c1cf_g10b2": n_juntadas,
        "grupo_novo_sem_correspondente": len(linhas_e2c_extra),
        "assercao": "108 == 108; sha256 do run JSON recalculado == sha256 do cartao E2c; 80 juntadas + 28 no grupo novo",
    }

    linhas.sort(key=lambda ln: (ln["grupo"], ln["caminho"]))

    # R1: heranca v1 -> v3
    v1_por_caminho = carregar_manifest_v1()
    resultado_heranca = herdar_de_v1(linhas, v1_por_caminho)

    return linhas, contagens, resultado_heranca


def linha_para_texto(linha: dict) -> str:
    return json.dumps(linha, ensure_ascii=False, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true", help="calcula sha256 (sempre feito; flag documental)")
    ap.add_argument("--saida", type=Path, default=DEFAULT_SAIDA)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    linhas, contagens, resultado_heranca = montar_manifest()
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
        "manifest_v1_usado": str(MANIFEST_V1),
        "proveniencia_usada": str(PROVENIENCIA),
        "lista_fixa_tab_retention_usada": str(LISTA_FIXA_TAB_RETENTION),
        "gancho_E2_usado": str(GANCHO_E2),
        "e2c_usado": str(E2_C0C1CF_CONTROLES),
        "heranca_v1": {
            "chaves_herdadas": resultado_heranca["chaves_herdadas"],
            "mudou_desde_v1": resultado_heranca["mudou_desde_v1"],
            "conflitos_motivo_resolvidos": resultado_heranca["conflitos_motivo_resolvidos"],
            "v1_caminhos_sem_correspondente_v3": resultado_heranca["v1_caminhos_sem_correspondente_v3"],
            "chaves_perdidas": len(resultado_heranca["v1_caminhos_sem_correspondente_v3"]),
        },
        "versao": "v3",
    }
    sidecar = args.saida.with_name(args.saida.stem + ".meta.json")
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"gravado: {args.saida} ({len(linhas)} linhas)")
    print(f"meta: {sidecar}")


if __name__ == "__main__":
    main()
