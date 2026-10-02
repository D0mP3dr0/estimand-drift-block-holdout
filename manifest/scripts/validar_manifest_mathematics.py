#!/usr/bin/env python3
"""Valida manifest_mathematics.jsonl (arquivo NOVO; nao edita o validador
original de E).

O validador canonico de E
(gnn_rf_ieee_access/.../EVIDENCIA_RESUBMISSAO/scripts/validar_manifest.py) e
um script de topo (sem funcoes) que roda a validacao e chama SystemExit no
proprio import — importa-lo literalmente encerraria este processo e nao daria
para "so acrescentar a classificacao" por cima. A adaptacao documentada aqui:
1) reexecutamos o ORIGINAL sem modifica-lo, via `runpy.run_path`, dentro de um
   `try/except SystemExit`, capturando stdout — isso reaproveita o algoritmo
   dele byte a byte sobre `manifest.jsonl` de E (contexto, nao repete a
   auditoria da R1: so registra o veredito agregado que ele devolveu agora);
2) para o manifest NOVO (`manifest_mathematics.jsonl`), cujo esquema usa
   `caminho` absoluto em vez de `artefato` relativo a uma BASE, reimplementamos
   o MESMO algoritmo (existe? sha256 bate?) com o campo certo, e classificamos
   cada problema em: (i) convencao de caminho (ex.: prefixo D:\\, barra
   invertida, caminho relativo que na verdade existe sob outra base), (ii)
   drive externo nao montado (caminho comeca com letra de unidade tipo D:\\ ou
   F:\\ e nao existe montado aqui), (iii) divergencia real de sha256/tamanho
   (arquivo existe, sha nao bate).

Uso:
    python validar_manifest_mathematics.py [--manifest CAMINHO]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import runpy
from pathlib import Path

E_VALIDADOR_ORIGINAL = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/scripts/validar_manifest.py"
)

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "manifest_mathematics.jsonl"


def reexecutar_validador_original_de_E():
    """Roda o validador ORIGINAL de E sem editar/copiar seu conteudo, so para
    contexto (nao substitui a auditoria da R1, so registra o veredito atual)."""
    buf = io.StringIO()
    codigo = None
    with contextlib.redirect_stdout(buf):
        try:
            runpy.run_path(str(E_VALIDADOR_ORIGINAL), run_name="__main__")
            codigo = 0
        except SystemExit as e:
            codigo = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    saida = buf.getvalue()
    ultima_linha = [l for l in saida.splitlines() if l.strip()]
    resumo = ultima_linha[-1] if ultima_linha else None
    return {"codigo_saida": codigo, "resumo": resumo, "linhas_stdout": len(saida.splitlines())}


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


def validar_manifest_novo(caminho_manifest: Path):
    validos = 0
    problemas = {
        "convencao_caminho": [],
        "drive_externo_nao_montado": [],
        "divergencia_real": [],
    }
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
        sha_disco = sha256_arquivo(p)
        tamanho_disco = p.stat().st_size
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
        "problemas_total": total_problemas,
        "problemas_por_categoria": {k: len(v) for k, v in problemas.items()},
        "detalhe_problemas": problemas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    contexto_E = reexecutar_validador_original_de_E()
    resultado_novo = validar_manifest_novo(args.manifest)

    saida = {
        "manifest_validado": str(args.manifest.resolve()),
        "contexto_validador_original_E": contexto_E,
        "resultado_manifest_mathematics": resultado_novo,
    }
    print(json.dumps(saida, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(1 if resultado_novo["problemas_total"] else 0)


if __name__ == "__main__":
    main()
