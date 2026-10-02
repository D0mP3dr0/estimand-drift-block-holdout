#!/usr/bin/env python3
"""Gera manifest_mathematics.jsonl do Artigo 2 (MDPI Mathematics).

Reexecutavel e idempotente: percorre os grupos de artefatos que o rascunho
draft_B usa (fisico 11/09, alvo_stats, baselines_v2, runs c0c1cf/c0c1/mlpcf,
artefatos das rodadas R1/R2 do fio, gancho E2 mlpcf T11) e escreve uma linha
JSON por artefato com sha256, tamanho, mtime, produtor (quando derivavel do
proprio arquivo), script_sha256 e cruzamento com o manifest canonico de E
(gnn_rf_ieee_access/.../EVIDENCIA_RESUBMISSAO/manifest.jsonl) por basename e
por caminho relativo.

Uso:
    python gerar_manifest_mathematics.py [--hash] [--saida CAMINHO] [--dry-run]

--hash      calcula sha256 de cada arquivo (padrao: sim; a flag existe para
            deixar explicito no comando de reexecucao, o script sempre hasheia).
--saida     caminho de saida (padrao: manifest_mathematics.jsonl ao lado deste
            script, em MDPI_Mathematics/).
--dry-run   imprime no stdout, nao grava nada em disco.

Sem carimbo de tempo de geracao dentro das linhas do jsonl (idempotencia
byte-a-byte entre execucoes sem mudanca no disco); o carimbo vive no sidecar
<saida>.meta.json (gerado_em, sha256 deste script, contagens).
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

DEFAULT_SAIDA = SCRIPT_PATH.parent.parent / "manifest_mathematics.jsonl"

CAMPOS_PRODUTOR_CANDIDATOS = ["script", "gerado_por", "autor", "agente"]


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


def derivar_produtor_e_script_sha(caminho: Path):
    """Retorna (produtor, produtor_motivo, script_sha256, script_sha256_motivo)."""
    d, motivo = carregar_json_seguro(caminho)
    if d is None:
        return None, motivo, None, "sem JSON interno para localizar campo script_sha256"

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

    script_sha256 = None
    script_sha_motivo = None
    v = d.get("script_sha256")
    if isinstance(v, str) and v.strip():
        script_sha256 = v
    else:
        # tenta localizar o script produtor nesta maquina, se o caminho for local
        if produtor and Path(produtor).is_absolute() and not produtor.startswith("D:") \
                and not produtor.startswith("F:") and Path(produtor).exists():
            try:
                script_sha256 = sha256_arquivo(Path(produtor))
            except Exception as e:
                script_sha_motivo = f"produtor localizado mas falha ao hashear: {e!r}"
        else:
            script_sha_motivo = (
                "campo script_sha256 ausente no artefato e o caminho do produtor "
                "(quando presente) e de maquina/drive Windows nao montado aqui"
            )

    return produtor, produtor_motivo, script_sha256, script_sha_motivo


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
    # existe por basename mas nenhum sha bate
    return True, False, "basename"


def linha_artefato(caminho: Path, grupo: str, regeneravel_por, por_basename, por_relpath):
    if not caminho.is_file():
        raise FileNotFoundError(f"artefato esperado nao existe: {caminho}")
    sha256 = sha256_arquivo(caminho)
    tamanho = caminho.stat().st_size
    mtime = mtime_iso(caminho)
    produtor, produtor_motivo, script_sha256, script_sha_motivo = derivar_produtor_e_script_sha(caminho)
    em_manifest_e, sha_confere_e, tipo_match = cruzar_com_manifest_e(
        caminho, sha256, por_basename, por_relpath
    )
    return {
        "caminho": str(caminho.resolve()),
        "sha256": sha256,
        "tamanho_bytes": tamanho,
        "mtime": mtime,
        "produtor": produtor,
        "produtor_motivo": produtor_motivo,
        "script_sha256": script_sha256,
        "script_sha256_motivo": script_sha_motivo,
        "grupo": grupo,
        "regeneravel_por": regeneravel_por,
        "em_manifest_E": em_manifest_e,
        "sha_confere_E": sha_confere_e,
        "tipo_match_manifest_E": tipo_match,
    }


# --------------------------------------------------------------------------
# Grupos
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
    """Retorna lista de (caminho, grupo) para as 3 familias + smoke a parte."""
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
        # dados/treinos_c1: arquivos diretos; treinos/: um nivel de subpasta
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


def grupo_6_gancho_e2():
    if not GANCHO_E2.exists():
        return None
    linhas = []
    for ln in GANCHO_E2.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            linhas.append(json.loads(ln))
    return linhas


def resolver_campo_gancho(d: dict, chaves: list[str]):
    for k in chaves:
        if k in d:
            return d[k]
    return None


def montar_manifest():
    por_basename, por_relpath = carregar_manifest_e()
    linhas = []
    contagens = {}

    arquivos1, g1 = grupo_1_fismat_stamps()
    for c in arquivos1:
        linhas.append(linha_artefato(c, g1, None, por_basename, por_relpath))
    contagens[g1] = {"esperado": 4, "encontrado": len(arquivos1)}

    arquivos2, g2 = grupo_2_alvo()
    for c in arquivos2:
        linhas.append(linha_artefato(c, g2, None, por_basename, por_relpath))
    contagens[g2] = {"esperado": 10, "encontrado": len(arquivos2)}

    arquivos3, g3 = grupo_3_baselines()
    for c in arquivos3:
        cmd = None
        linhas.append(linha_artefato(c, g3, cmd, por_basename, por_relpath))
    contagens[g3] = {"esperado": 16, "encontrado": len(arquivos3)}

    runs = grupo_4_runs()
    for grupo_nome, lst in runs.items():
        for c in lst:
            linhas.append(linha_artefato(c, grupo_nome, None, por_basename, por_relpath))
        esperado = {
            "run_c0c1cf_g10b2": None,  # 2 raizes x 80 = 160 esperado quando espelhado
            "run_c0c1_bauru_g5b2_s42": None,  # 2 raizes x 4 = 8 esperado
            "run_mlpcf": None,  # ate 2 raizes x 80 (bauru/lins espelhado, campinas/sorocaba so em treinos/)
            "run_mlpcf_smoke": None,
        }[grupo_nome]
        contagens[grupo_nome] = {"esperado": esperado, "encontrado": len(lst)}

    arquivos5, g5 = grupo_5_rodadas()
    for c in arquivos5:
        linhas.append(linha_artefato(c, g5, None, por_basename, por_relpath))
    contagens[g5] = {"esperado": None, "encontrado": len(arquivos5)}

    gancho = grupo_6_gancho_e2()
    if gancho is None:
        gancho_estado = "ausente"
    else:
        gancho_estado = f"presente ({len(gancho)} linhas)"
        for d in gancho:
            caminho_str = resolver_campo_gancho(d, ["caminho", "artefato", "path"])
            sha_declarado = resolver_campo_gancho(d, ["sha256"])
            if caminho_str is None:
                raise KeyError(
                    "gancho E2: linha sem campo de caminho reconhecivel "
                    f"(caminho/artefato/path): {d}"
                )
            if sha_declarado is None:
                raise KeyError(f"gancho E2: linha sem campo sha256: {d}")
            p = Path(caminho_str)
            if not p.is_absolute():
                p = RAIZ_E / caminho_str
            linha = linha_artefato(p, "mlpcf_t11_E2", None, por_basename, por_relpath)
            linha["sha256_declarado_no_gancho"] = sha_declarado
            linha["sha_confere_gancho"] = sha_declarado == linha["sha256"]
            linhas.append(linha)
        contagens["mlpcf_t11_E2"] = {"esperado": 40, "encontrado": len(gancho)}

    linhas.sort(key=lambda ln: ln["caminho"])
    return linhas, contagens, gancho_estado


def linha_para_texto(linha: dict) -> str:
    return json.dumps(linha, ensure_ascii=False, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true", help="calcula sha256 (sempre feito; flag documental)")
    ap.add_argument("--saida", type=Path, default=DEFAULT_SAIDA)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    linhas, contagens, gancho_estado = montar_manifest()
    texto_jsonl = "\n".join(linha_para_texto(l) for l in linhas) + ("\n" if linhas else "")

    if args.dry_run:
        sys.stdout.write(texto_jsonl)
        print(f"# [dry-run] {len(linhas)} linhas; contagens={contagens}; gancho_E2={gancho_estado}", file=sys.stderr)
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
        "gancho_E2": gancho_estado,
        "manifest_E_usado": str(MANIFEST_E),
        "manifest_E_existe": MANIFEST_E.exists(),
    }
    sidecar = args.saida.with_suffix("").with_suffix(".meta.json") if args.saida.suffix == ".jsonl" \
        else args.saida.with_name(args.saida.stem + ".meta.json")
    # nome fixo esperado: manifest_mathematics.meta.json ao lado do jsonl
    sidecar = args.saida.with_name(args.saida.stem + ".meta.json")
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"gravado: {args.saida} ({len(linhas)} linhas)")
    print(f"meta: {sidecar}")


if __name__ == "__main__":
    main()
