#!/usr/bin/env python3
"""Valida manifest_mathematics_v2.jsonl (arquivo NOVO; nao edita
validar_manifest_mathematics.py v1 nem o validador original de E).

Por que nao usar o v1 direto: o v1 recalcula sha256 de TODO artefato listado
(`sha256_arquivo`), inclusive dos que o proprio manifest ja marca com
`sha_recalculado: false` -- caso dos 16 `_cftudo.pt` (28 GB cada, grupo
`tensores_cftudo`), cujo sha vem de `HASHES_SHA256.txt` e cuja leitura/rehash
esta EXPRESSAMENTE PROIBIDA nesta rodada. Rodar o v1 sobre o manifest v2
leria 16 x 28 GB. Este validador v2 respeita o campo `sha_recalculado`:
quando False, so confere caminho + tamanho (os.stat, sem abrir o arquivo
para hashear); quando True (ou ausente, tratado como True por seguranca),
recalcula e confere sha256 normalmente, como o v1 fazia para tudo.

Uso:
    python validar_manifest_mathematics_v2.py [--manifest CAMINHO]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "manifest_mathematics_v2.jsonl"


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


def validar_manifest_v2(caminho_manifest: Path):
    validos = 0
    validos_sem_rehash = 0
    problemas = {
        "convencao_caminho": [],
        "drive_externo_nao_montado": [],
        "divergencia_real": [],
    }
    ignorados_rehash = []
    for ln in caminho_manifest.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        e = json.loads(ln)
        caminho_str = e["caminho"]
        p = Path(caminho_str)
        if not p.exists():
            categoria = classificar_ausente(caminho_str)
            if categoria == "drive_externo_nao_montado":
                problemas["drive_externo_nao_montado"].append(caminho_str)
            else:
                problemas["convencao_caminho"].append({"caminho": caminho_str, "motivo": categoria})
            continue

        sha_recalculado = e.get("sha_recalculado", True)
        tamanho_disco = p.stat().st_size
        if sha_recalculado is False:
            # NAO abre o arquivo (proibido reler/re-hashear tensor .pt de 28 GB);
            # confere so o tamanho contra o manifest.
            if tamanho_disco != e["tamanho_bytes"]:
                problemas["divergencia_real"].append(
                    {
                        "caminho": caminho_str,
                        "motivo": "tamanho em disco diverge (sha nao recalculado por proibicao explicita)",
                        "tamanho_manifest": e["tamanho_bytes"],
                        "tamanho_disco": tamanho_disco,
                    }
                )
                continue
            validos += 1
            validos_sem_rehash += 1
            ignorados_rehash.append(caminho_str)
            continue

        sha_disco = sha256_arquivo(p)
        if sha_disco != e["sha256"] or tamanho_disco != e["tamanho_bytes"]:
            problemas["divergencia_real"].append(
                {
                    "caminho": caminho_str,
                    "sha256_manifest": e["sha256"],
                    "sha256_disco": sha_disco,
                    "tamanho_manifest": e["tamanho_bytes"],
                    "tamanho_disco": tamanho_disco,
                }
            )
            continue
        validos += 1

    total_problemas = sum(len(v) for v in problemas.values())
    return {
        "validos": validos,
        "validos_sem_rehash_sha_recalculado_false": validos_sem_rehash,
        "caminhos_sem_rehash": ignorados_rehash,
        "problemas_total": total_problemas,
        "problemas_por_categoria": {k: len(v) for k, v in problemas.items()},
        "detalhe_problemas": problemas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    resultado = validar_manifest_v2(args.manifest)

    saida = {
        "manifest_validado": str(args.manifest.resolve()),
        "nota": (
            "validador v2: NAO re-hasheia artefatos com sha_recalculado=false "
            "(os 16 tensores _cftudo.pt de graph_data_v3, 28 GB cada) -- so "
            "confere caminho + tamanho via os.stat para esses; recalcula "
            "sha256 normalmente para todo o resto, como o v1 fazia"
        ),
        "resultado_manifest_mathematics_v2": resultado,
    }
    print(json.dumps(saida, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(1 if resultado["problemas_total"] else 0)


if __name__ == "__main__":
    main()
