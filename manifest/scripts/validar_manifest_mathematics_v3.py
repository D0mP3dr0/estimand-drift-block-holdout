#!/usr/bin/env python3
"""Valida manifest_mathematics_v3.jsonl (arquivo NOVO; nao edita os
validadores v1/v2). Reusa a regra do v2 (respeita `sha_recalculado: false`
para nao reler os 16 tensores _cftudo.pt de 28 GB) e acrescenta a matriz por
grupo pedida na rodada: esperado x encontrado, quantas linhas tem
`script_sha256` preenchido (prova da heranca v1) e quantas linhas vieram com
sha recalculado hoje x sha so declarado/copiado (E2, checkpoints, tensores).

Uso:
    python validar_manifest_mathematics_v3.py [--manifest CAMINHO]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "manifest_mathematics_v3.jsonl"

CHECKPOINT_LIMITE_BYTES = 200 * 1024 * 1024

ESPERADO_POR_GRUPO = {
    "fismat_carimbo_11_09": 4,
    "alvo_stats_e_controle_negativo": 10,
    "baselines_v2": 16,
    "scripts_produtores": 8,
    "tensores_cftudo": 16,
    "mlpcf_t11_E2": 40,
    "lista_fixa_tab_retention": 20,
    "logs_fila": 2,
}
ESPERADO_BASENAMES_POR_GRUPO = {
    "run_c0c1cf_g10b2": 80,
    "run_c0c1_bauru_g5b2_s42": 4,
    "run_mlpcf": 80,
}


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


def validar_manifest_v3(caminho_manifest: Path):
    validos = 0
    validos_sem_rehash = 0
    problemas = {
        "convencao_caminho": [],
        "drive_externo_nao_montado": [],
        "divergencia_real": [],
    }
    ignorados_rehash = []

    linhas = []
    for ln in caminho_manifest.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        linhas.append(json.loads(ln))

    for e in linhas:
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
            if tamanho_disco != e["tamanho_bytes"]:
                problemas["divergencia_real"].append({
                    "caminho": caminho_str,
                    "motivo": "tamanho em disco diverge (sha nao recalculado por proibicao explicita)",
                    "tamanho_manifest": e["tamanho_bytes"],
                    "tamanho_disco": tamanho_disco,
                })
                continue
            validos += 1
            validos_sem_rehash += 1
            ignorados_rehash.append(caminho_str)
            continue

        sha_disco = sha256_arquivo(p)
        if sha_disco != e["sha256"] or tamanho_disco != e["tamanho_bytes"]:
            problemas["divergencia_real"].append({
                "caminho": caminho_str,
                "sha256_manifest": e["sha256"],
                "sha256_disco": sha_disco,
                "tamanho_manifest": e["tamanho_bytes"],
                "tamanho_disco": tamanho_disco,
            })
            continue
        validos += 1

    # ---------------------------------------------------------------
    # Matriz por grupo: linhas, basenames, esperado x encontrado,
    # script_sha256 preenchido, sha recalculado x declarado.
    # ---------------------------------------------------------------
    por_grupo = {}
    for e in linhas:
        g = e.get("grupo", "?")
        info = por_grupo.setdefault(g, {
            "linhas": 0,
            "basenames": set(),
            "script_sha256_preenchido": 0,
            "sha_recalculado_true": 0,
            "sha_recalculado_false": 0,
            "mudou_desde_v1": 0,
        })
        info["linhas"] += 1
        info["basenames"].add(Path(e["caminho"]).name)
        if e.get("script_sha256"):
            info["script_sha256_preenchido"] += 1
        if e.get("sha_recalculado", True):
            info["sha_recalculado_true"] += 1
        else:
            info["sha_recalculado_false"] += 1
        if e.get("mudou_desde_v1"):
            info["mudou_desde_v1"] += 1

    matriz = {}
    for g, info in sorted(por_grupo.items()):
        esperado = ESPERADO_POR_GRUPO.get(g)
        esperado_basenames = ESPERADO_BASENAMES_POR_GRUPO.get(g)
        matriz[g] = {
            "linhas": info["linhas"],
            "basenames": len(info["basenames"]),
            "esperado_linhas": esperado,
            "esperado_basenames": esperado_basenames,
            "linhas_ok": (esperado is None) or (info["linhas"] == esperado),
            "basenames_ok": (esperado_basenames is None) or (len(info["basenames"]) == esperado_basenames),
            "script_sha256_preenchido": info["script_sha256_preenchido"],
            "sha_recalculado_true": info["sha_recalculado_true"],
            "sha_recalculado_false": info["sha_recalculado_false"],
            "mudou_desde_v1": info["mudou_desde_v1"],
        }

    # "segue sem cadeia" -- linhas com script_sha256 nulo e sem motivo de
    # exclusao explicita conhecida (carimbos fora do escopo da proveniencia).
    segue_sem_cadeia = []
    for e in linhas:
        if "script_sha256" in e and e.get("script_sha256") is None:
            motivo = e.get("script_sha256_motivo")
            segue_sem_cadeia.append({
                "caminho": e["caminho"],
                "grupo": e.get("grupo"),
                "script_sha256_motivo": motivo,
            })

    total_problemas = sum(len(v) for v in problemas.values())
    return {
        "validos": validos,
        "validos_sem_rehash_sha_recalculado_false": validos_sem_rehash,
        "caminhos_sem_rehash": ignorados_rehash,
        "problemas_total": total_problemas,
        "problemas_por_categoria": {k: len(v) for k, v in problemas.items()},
        "detalhe_problemas": problemas,
        "n_linhas": len(linhas),
        "matriz_por_grupo": matriz,
        "segue_sem_cadeia": segue_sem_cadeia,
        "segue_sem_cadeia_total": len(segue_sem_cadeia),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    resultado = validar_manifest_v3(args.manifest)

    saida = {
        "manifest_validado": str(args.manifest.resolve()),
        "nota": (
            "validador v3: mesma regra do v2 para sha_recalculado=false (nao "
            "re-hasheia _cftudo.pt nem checkpoints >= 200 MB); acrescenta "
            "matriz por grupo (esperado x encontrado, script_sha256 "
            "preenchido, sha recalculado x declarado) e a lista 'segue sem "
            "cadeia' (script_sha256 nulo)"
        ),
        "resultado_manifest_mathematics_v3": resultado,
    }
    print(json.dumps(saida, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(1 if resultado["problemas_total"] else 0)


if __name__ == "__main__":
    main()
