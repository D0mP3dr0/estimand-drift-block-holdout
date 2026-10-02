#!/usr/bin/env python3
"""Gera manifest_mathematics_v4.jsonl do Artigo 2 (MDPI Mathematics) -- v4.

Arquivo NOVO (nao edita gerar_manifest_mathematics_v3.py). Roadmap v3, teste
0.2 (m179036156098; draft_v2/ROADMAP_TESTES_v3_2026-09-25.md, Fase 0),
aprovacao do dono em
`/trabalho/HERMES/AGENTES/_DECISOES/DECISOES-2026-09-25-artigo2-v3-independencia-e-roadmap.md`
(D4, roadmap encadeado, 25/09/2026).

v4 = SUPERCONJUNTO do v3: reusa `montar_manifest()` de
gerar_manifest_mathematics_v3.py POR IMPORT (importlib, sem editar o
arquivo), preservando as 516 linhas/12 grupos e as asserções/heranca v1 do
v3 (inclusive o grupo `scripts_artigo2`, que ja e glob DINAMICO de
`MDPI_Mathematics/scripts/*.py` -- portanto ja cobre todos os scripts novos
da rodada R3/v2/v3, ex.: v2_gerar_tabelas.py, v2_gerar_figuras.py,
varredura_fracao_valida_r3.py, blocos_efetivos_20celulas.py,
varredura_fracao_valida_estratificada.py, montecarlo_retencao_r3_g5.py,
r3_alvo_pos_correcao_16celulas.py, r3_prova_antenas_bauru_q1.py,
v2_montar_main.py, v3_limpar_fonte.py, e3_agregar*.py,
e3_registrar_oficina.py, e os scripts de fase 1 do roadmap v3
(v3_1.1_1.2_*.py, v3_1.3_*.py, v3_1.4_*.py, v3_1.7_*.py, v3_2.1_3.1_*.py,
etc., a medida que forem gravados em scripts/); so ATUALIZA a lista
esperada explicita (ESPERADO_SCRIPTS_ARTIGO2_BASE) para essa rodada, para
que deixem de aparecer como "aviso_fora_da_lista" e passem a ser
reconhecidos nominalmente.

Grupos NOVOS do v4 (task 0.2, item 2):
  - `r2_2026_09_24_artefatos`: todo arquivo sob `_R2_2026-09-24/` (cache_r3,
    e3_predicoes, e3_calibracao, logs, RAW json) -- proveniencia dos
    artefatos CPU do R2/R3, sem tocar nos tensores *_cftudo.pt (28 GB, fora
    desta pasta).
  - `draft_v2_tables`: `draft_v2/tables/*.tex` e `*.csv` (T2-T5, incluindo o
    T3 novo desta rodada).
  - `draft_v2_figures`: `draft_v2/figures/*.pdf` e `*.png` (F1-F4).
  - `v3_2026_09_25_criterios`: `_v3_2026-09-25/criterios/*.json` e
    `MANIFEST_criterios.jsonl` (roadmap v3, D1: criterio antes do numero).

Cada linha nova traz `caminho`, `sha256`, `tamanho_bytes`, `mtime`, `grupo`
e `comando_regeneracao` (script + argumentos que regeram o artefato, quando
aplicavel; None para artefatos que sao saida direta de outra frente e nao
tem um comando unico de regeneracao neste repositorio).

PROIBIDO: ler ou re-hashear os *_cftudo.pt (28 GB cada); ler/re-hashear
checkpoints .pt >= 200 MB; treino, download, GPU (idem v3).

Uso:
    python gerar_manifest_mathematics_v4.py [--saida CAMINHO] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
V3_SCRIPT_PATH = SCRIPT_PATH.parent / "gerar_manifest_mathematics_v3.py"

MDPI_MATH_ROOT = SCRIPT_PATH.parent.parent
R2_DIR = MDPI_MATH_ROOT / "_R2_2026-09-24"
TABLES_DIR = MDPI_MATH_ROOT / "draft_v2" / "tables"
FIGURES_DIR = MDPI_MATH_ROOT / "draft_v2" / "figures"
CRITERIOS_DIR = MDPI_MATH_ROOT / "_v3_2026-09-25" / "criterios"

DEFAULT_SAIDA = MDPI_MATH_ROOT / "manifest_mathematics_v4.jsonl"

# R3/v2/v3: lista esperada explicita desta rodada, para que o grupo
# scripts_artigo2 do v3 (glob dinamico) deixe de reportar estes nomes como
# "aviso_fora_da_lista" -- eles SAO esperados no v4.
SCRIPTS_R3_V2_V3_NOVOS = {
    "e3_agregar",
    "e3_agregar_v2",
    "e3_registrar_oficina",
    "montecarlo_retencao_r3_g5",
    "r3_alvo_pos_correcao_16celulas",
    "r3_prova_antenas_bauru_q1",
    "v2_gerar_figuras",
    "v2_gerar_tabelas",
    "v2_montar_main",
    "v3_limpar_fonte",
    "varredura_fracao_valida_estratificada",
    "varredura_fracao_valida_r3",
    "gerar_manifest_mathematics_v4",
}


def _importar_v3():
    spec = importlib.util.spec_from_file_location("gerar_manifest_mathematics_v3_mod", V3_SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def mtime_iso(caminho: Path) -> str:
    ts = caminho.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def linha_generica(caminho: Path, grupo: str, comando_regeneracao):
    return {
        "caminho": str(caminho.resolve()),
        "sha256": sha256_arquivo(caminho),
        "sha_recalculado": True,
        "tamanho_bytes": caminho.stat().st_size,
        "mtime": mtime_iso(caminho),
        "grupo": grupo,
        "comando_regeneracao": comando_regeneracao,
    }


# --------------------------------------------------------------------------
# Grupos novos do v4
# --------------------------------------------------------------------------

def grupo_r2_2026_09_24_artefatos():
    if not R2_DIR.is_dir():
        raise FileNotFoundError(f"pasta R2 ausente: {R2_DIR}")
    arquivos = sorted(p for p in R2_DIR.rglob("*") if p.is_file())
    linhas = []
    for p in arquivos:
        comando = None
        rel = p.relative_to(R2_DIR).as_posix()
        if rel.startswith("cache_r3/"):
            comando = (
                ".venv/bin/python scripts/varredura_fracao_valida_r3.py "
                "--out _R2_2026-09-24/... (extrai cache_r3/<cidade>_<Q>_valid2d.npz "
                "da 1a leitura do tensor _cftudo.pt correspondente)"
            )
        elif rel.startswith("e3_predicoes/"):
            comando = "E3 (inferencia GNN/MLP por populacao); ver e3_inferencia_por_populacao.py / e3_agregar_v2.py"
        elif rel.startswith("e3_calibracao/"):
            comando = "E3 calibracao de baselines; ver scripts do fio 2026-09-24_gnn_rf_artigo2_mathematics_r2"
        elif rel.startswith("r3_fracao_valida"):
            comando = ".venv/bin/python scripts/varredura_fracao_valida_r3.py --out <json> --log <log>"
        elif rel.startswith("r3_alvo_pos_correcao"):
            comando = ".venv/bin/python scripts/r3_alvo_pos_correcao_16celulas.py"
        linhas.append(linha_generica(p, "r2_2026_09_24_artefatos", comando))
    return linhas


def grupo_draft_v2_tables():
    if not TABLES_DIR.is_dir():
        raise FileNotFoundError(f"pasta de tabelas ausente: {TABLES_DIR}")
    arquivos = sorted(p for p in TABLES_DIR.iterdir() if p.is_file() and p.suffix in (".tex", ".csv"))
    linhas = []
    for p in arquivos:
        if p.stem.startswith("T3_"):
            comando = ".venv/bin/python scripts/v2_gerar_tabelas.py  # gerar_t3() (roadmap v3, teste 0.2)"
        else:
            comando = ".venv/bin/python scripts/v2_gerar_tabelas.py"
        linhas.append(linha_generica(p, "draft_v2_tables", comando))
    return linhas


def grupo_draft_v2_figures():
    if not FIGURES_DIR.is_dir():
        raise FileNotFoundError(f"pasta de figuras ausente: {FIGURES_DIR}")
    arquivos = sorted(
        p for p in FIGURES_DIR.iterdir()
        if p.is_file() and p.suffix in (".pdf", ".png")
    )
    linhas = []
    for p in arquivos:
        if p.stem.startswith(("F2_", "F3_", "F4_")):
            comando = ".venv/bin/python scripts/v2_gerar_figuras.py  # pdf.fonttype=42 desde 25/09 (roadmap v3, teste 0.2)"
        else:
            comando = None  # F1 (esquema) e outras figuras nao geradas por v2_gerar_figuras.py
        linhas.append(linha_generica(p, "draft_v2_figures", comando))
    return linhas


def grupo_v3_2026_09_25_criterios():
    if not CRITERIOS_DIR.is_dir():
        raise FileNotFoundError(f"pasta de criterios v3 ausente: {CRITERIOS_DIR}")
    arquivos = sorted(
        p for p in CRITERIOS_DIR.iterdir()
        if p.is_file() and p.suffix in (".json", ".jsonl")
    )
    linhas = []
    for p in arquivos:
        linhas.append(linha_generica(p, "v3_2026_09_25_criterios", None))
    return linhas


# --------------------------------------------------------------------------
# Montagem
# --------------------------------------------------------------------------

def montar_manifest_v4():
    v3 = _importar_v3()

    # nao edita o arquivo v3 em disco: so ajusta o global do MODULO em
    # memoria, para esta execucao, com a lista esperada de scripts desta
    # rodada (evita "aviso_fora_da_lista" para scripts ja esperados no v4).
    v3.ESPERADO_SCRIPTS_ARTIGO2_BASE = set(v3.ESPERADO_SCRIPTS_ARTIGO2_BASE) | SCRIPTS_R3_V2_V3_NOVOS

    linhas_v3, contagens_v3, heranca_v3 = v3.montar_manifest()

    linhas_novas = []
    contagens_novas = {}

    l_r2 = grupo_r2_2026_09_24_artefatos()
    linhas_novas.extend(l_r2)
    contagens_novas["r2_2026_09_24_artefatos"] = {"encontrado": len(l_r2)}

    l_tab = grupo_draft_v2_tables()
    linhas_novas.extend(l_tab)
    contagens_novas["draft_v2_tables"] = {"encontrado": len(l_tab)}

    l_fig = grupo_draft_v2_figures()
    linhas_novas.extend(l_fig)
    contagens_novas["draft_v2_figures"] = {"encontrado": len(l_fig)}

    l_crit = grupo_v3_2026_09_25_criterios()
    linhas_novas.extend(l_crit)
    contagens_novas["v3_2026_09_25_criterios"] = {"encontrado": len(l_crit)}

    linhas_novas.sort(key=lambda ln: (ln["grupo"], ln["caminho"]))

    todas = linhas_v3 + linhas_novas
    contagens = dict(contagens_v3)
    contagens.update(contagens_novas)
    contagens["scripts_artigo2"]["esperado_atualizado_v4"] = sorted(v3.ESPERADO_SCRIPTS_ARTIGO2_BASE)

    return todas, contagens, heranca_v3, len(linhas_v3), len(linhas_novas)


def linha_para_texto(linha: dict) -> str:
    return json.dumps(linha, ensure_ascii=False, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=DEFAULT_SAIDA)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    linhas, contagens, heranca_v3, n_v3, n_novas = montar_manifest_v4()
    texto_jsonl = "\n".join(linha_para_texto(l) for l in linhas) + ("\n" if linhas else "")

    if args.dry_run:
        sys.stdout.write(texto_jsonl)
        print(f"# [dry-run] {len(linhas)} linhas ({n_v3} herdadas do v3 + {n_novas} novas)", file=sys.stderr)
        return

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(texto_jsonl, encoding="utf-8")

    script_sha = sha256_arquivo(SCRIPT_PATH)
    v3_sha = sha256_arquivo(V3_SCRIPT_PATH)
    meta = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "data": datetime.now(timezone.utc).date().isoformat(),
        "script": str(SCRIPT_PATH),
        "script_sha256": script_sha,
        "v3_script_usado": str(V3_SCRIPT_PATH),
        "v3_script_sha256": v3_sha,
        "saida": str(args.saida.resolve()),
        "saida_sha256": sha256_arquivo(args.saida),
        "n_linhas": len(linhas),
        "n_linhas_herdadas_v3": n_v3,
        "n_linhas_novas_v4": n_novas,
        "contagens_por_grupo": contagens,
        "heranca_v1_via_v3": {
            "chaves_herdadas": heranca_v3["chaves_herdadas"],
            "mudou_desde_v1": heranca_v3["mudou_desde_v1"],
            "conflitos_motivo_resolvidos": heranca_v3["conflitos_motivo_resolvidos"],
            "chaves_perdidas": len(heranca_v3["v1_caminhos_sem_correspondente_v3"]),
        },
        "versao": "v4",
        "roadmap": "draft_v2/ROADMAP_TESTES_v3_2026-09-25.md, Fase 0, teste 0.2",
        "aprovacao_dono": "AGENTES/_DECISOES/DECISOES-2026-09-25-artigo2-v3-independencia-e-roadmap.md (D4)",
    }
    sidecar = args.saida.with_name(args.saida.stem + ".meta.json")
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"gravado: {args.saida} ({len(linhas)} linhas = {n_v3} v3 + {n_novas} novas)")
    print(f"meta: {sidecar}")


if __name__ == "__main__":
    main()
