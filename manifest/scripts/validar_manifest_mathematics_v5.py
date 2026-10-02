#!/usr/bin/env python3
"""Valida manifest_mathematics_v5.jsonl.

COPIA DECLARADA de validar_manifest_mathematics_v3.py (v3 nao editado). Mantem:
a regra de `sha_recalculado: false` (nao rele arquivo grande; confere so o
tamanho), a matriz por grupo (linhas, sha recalculado x declarado, script_sha256
preenchido) e a classificacao de caminhos ausentes. ACRESCENTA, para o v5:
  - rehash de uma AMOSTRA de N arquivos pequenos (padrao 200, < 1 MB, semente
    fixa) e de TODOS os scripts (.py/.ps1/.sh) do manifesto;
  - existencia + tamanho de TODAS as linhas (inclusive as sem rehash);
  - consistencia `raiz` + `caminho_rel` == `caminho` (raizes lidas do .meta.json);
  - unicidade de (`caminho`, `grupo`); linhas com sha256 nulo ("pendente de hash");
  - conferencia dos digests citados no manuscrito (95ea0423, 45c16d43, 5d38012e,
    ebeaf759) contra o manifesto E contra o arquivo no disco.

Uso:
    python validar_manifest_mathematics_v5.py [--manifest CAMINHO] [--amostra 200]
        [--semente 0] [--saida-json CAMINHO]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "manifest_mathematics_v5.jsonl"

PEQUENO_BYTES = 1024 * 1024
EXT_SCRIPT = (".py", ".ps1", ".sh")
DIGESTS_MANUSCRITO = ("95ea0423", "45c16d43", "5d38012e", "ebeaf759")


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def classificar_ausente(caminho_str: str) -> str:
    if caminho_str[:2] in ("D:", "F:") or caminho_str.startswith("D:\\") or caminho_str.startswith("F:\\"):
        return "drive_externo_nao_montado"
    if "\\" in caminho_str:
        return "convencao_caminho_barra_invertida"
    if not caminho_str.startswith("/"):
        return "convencao_caminho_relativo"
    return "convencao_caminho_outro"


def validar_manifest_v5(caminho_manifest: Path, n_amostra: int, semente: int):
    linhas = []
    for ln in caminho_manifest.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            linhas.append(json.loads(ln))

    meta_path = caminho_manifest.with_name(caminho_manifest.stem + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    raizes = meta.get("raizes", {})

    problemas = {
        "convencao_caminho": [],
        "drive_externo_nao_montado": [],
        "tamanho_diverge": [],
        "divergencia_real": [],
        "raiz_rel_inconsistente": [],
        "caminho_duplicado": [],
    }

    # unicidade
    # o v3/v4 repete de proposito o caminho em grupos de papel diferente; duplicata = mesmo (caminho, grupo)
    cont = Counter((e["caminho"], e.get("grupo")) for e in linhas)
    problemas["caminho_duplicado"] = [list(c) for c, n in cont.items() if n > 1]
    n_multigrupo = sum(1 for c, n in Counter(e["caminho"] for e in linhas).items() if n > 1)

    # existencia + tamanho de TODAS as linhas; raiz+rel
    existentes = []
    for e in linhas:
        caminho_str = e["caminho"]
        p = Path(caminho_str)
        if not p.exists():
            cat = classificar_ausente(caminho_str)
            if cat == "drive_externo_nao_montado":
                problemas["drive_externo_nao_montado"].append(caminho_str)
            else:
                problemas["convencao_caminho"].append({"caminho": caminho_str, "motivo": cat})
            continue
        if p.stat().st_size != e["tamanho_bytes"]:
            problemas["tamanho_diverge"].append({"caminho": caminho_str, "manifest": e["tamanho_bytes"],
                                                 "disco": p.stat().st_size})
            continue
        existentes.append(e)
        rid, rel = e.get("raiz"), e.get("caminho_rel")
        if rid is None or rel is None:
            problemas["raiz_rel_inconsistente"].append({"caminho": caminho_str, "motivo": "sem raiz/caminho_rel"})
        else:
            if rid == "FS":
                recomposto = "/" + rel
            elif rid in raizes:
                recomposto = str(Path(raizes[rid]) / rel)
            else:
                recomposto = None
            if recomposto != caminho_str:
                problemas["raiz_rel_inconsistente"].append({"caminho": caminho_str, "raiz": rid, "rel": rel})

    # rehash: amostra de pequenos + todos os scripts
    candidatos_amostra = [e for e in existentes
                          if e.get("sha256") and e.get("sha_recalculado", True)
                          and e["tamanho_bytes"] < PEQUENO_BYTES
                          and not e["caminho"].endswith(EXT_SCRIPT)]
    rnd = random.Random(semente)
    amostra = rnd.sample(candidatos_amostra, min(n_amostra, len(candidatos_amostra)))
    scripts = [e for e in existentes if e["caminho"].endswith(EXT_SCRIPT) and e.get("sha256")]
    a_rehash = {e["caminho"]: e for e in amostra}  # por caminho (o mesmo arquivo em 2 grupos conta 1x)
    a_rehash.update({e["caminho"]: e for e in scripts})

    rehash_ok = 0
    for c, e in a_rehash.items():
        sha = sha256_arquivo(Path(c))
        if sha != e["sha256"]:
            problemas["divergencia_real"].append({"caminho": c, "sha256_manifest": e["sha256"], "sha256_disco": sha})
        else:
            rehash_ok += 1

    sem_rehash = [e["caminho"] for e in existentes if e.get("sha_recalculado", True) is False]
    pendentes = [e["caminho"] for e in linhas if e.get("sha256") is None]

    # digests do manuscrito: no manifesto e no disco
    digests = {}
    for pref in DIGESTS_MANUSCRITO:
        achados = [e for e in linhas if e.get("sha256") and e["sha256"].startswith(pref)]
        no_disco = []
        for e in achados:
            if Path(e["caminho"]).exists() and e["caminho"].endswith(EXT_SCRIPT):
                no_disco.append(sha256_arquivo(Path(e["caminho"])).startswith(pref))
        digests[pref] = {"linhas_no_manifesto": len(achados),
                         "caminhos": [e["caminho"] for e in achados],
                         "rehash_no_disco_confere": (all(no_disco) if no_disco else None)}
    digests_ok = all(v["linhas_no_manifesto"] >= 1 and v["rehash_no_disco_confere"] is True for v in digests.values())

    # matriz por grupo (como o v3)
    por_grupo = {}
    for e in linhas:
        g = por_grupo.setdefault(e.get("grupo", "?"), {"linhas": 0, "tamanho_bytes": 0, "sha_recalculado_true": 0,
                                                       "sha_recalculado_false": 0, "sha256_nulo": 0,
                                                       "script_sha256_preenchido": 0})
        g["linhas"] += 1
        g["tamanho_bytes"] += e["tamanho_bytes"]
        g["sha_recalculado_true" if e.get("sha_recalculado", True) else "sha_recalculado_false"] += 1
        if e.get("sha256") is None:
            g["sha256_nulo"] += 1
        if e.get("script_sha256"):
            g["script_sha256_preenchido"] += 1

    segue_sem_cadeia = [{"caminho": e["caminho"], "grupo": e.get("grupo"),
                         "script_sha256_motivo": e.get("script_sha256_motivo")}
                        for e in linhas if "script_sha256" in e and e.get("script_sha256") is None]

    total_problemas = sum(len(v) for v in problemas.values())
    return {
        "n_linhas": len(linhas),
        "n_caminhos_em_mais_de_um_grupo_informativo": n_multigrupo,
        "existentes_com_tamanho_confere": len(existentes),
        "rehash": {"amostra_pequenos_pedida": n_amostra, "amostra_pequenos_usada": len(amostra),
                   "scripts_rehash": len(scripts), "total_arquivos_rehash": len(a_rehash),
                   "rehash_ok": rehash_ok, "semente": semente},
        "sem_rehash_sha_recalculado_false": len(sem_rehash),
        "caminhos_sem_rehash": sem_rehash,
        "pendentes_de_hash_sha256_nulo": pendentes,
        "problemas_total": total_problemas,
        "problemas_por_categoria": {k: len(v) for k, v in problemas.items()},
        "detalhe_problemas": problemas,
        "digests_manuscrito": digests,
        "digests_manuscrito_ok": digests_ok,
        "matriz_por_grupo": dict(sorted(por_grupo.items())),
        "segue_sem_cadeia_total": len(segue_sem_cadeia),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--amostra", type=int, default=200)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--saida-json", type=Path, default=None)
    args = ap.parse_args()

    resultado = validar_manifest_v5(args.manifest, args.amostra, args.semente)
    saida = {
        "manifest_validado": str(args.manifest.resolve()),
        "nota": ("validador v5 (copia declarada do v3): existencia+tamanho de todas as linhas; rehash de amostra de "
                 "arquivos pequenos e de todos os scripts; arquivos com sha_recalculado=false so tem o tamanho conferido; "
                 "conferencia dos digests do manuscrito"),
        "resultado_manifest_mathematics_v5": resultado,
    }
    txt = json.dumps(saida, ensure_ascii=False, indent=2, sort_keys=True)
    if args.saida_json:
        args.saida_json.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    raise SystemExit(1 if resultado["problemas_total"] or resultado["pendentes_de_hash_sha256_nulo"] else 0)


if __name__ == "__main__":
    main()
