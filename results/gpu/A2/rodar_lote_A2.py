#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A2/rodar_lote_A2.py -- oficina-experimento, PLANO A / item A2 (G1
reprodutibilidade), protocolo aprovado
PROTOCOLO_GPU_v3_2026-09-25.md, criterio criterios/criterio_A2.json.

Roda as 8 corridas (Bauru Q1 e Lins Q1 x GNN/MLP x 2 repeticoes da MESMA
seed=42), sem cadeia, 8 epocas, grafo inteiro (--mmap, sem --max-nodes),
chamando os wrappers congelados gpu/modelo_v3/train_gnn_v3.py e
train_mlp_v3.py via subprocess com o MESMO interpretador que executa este
driver (deve ser a venv CUDA).

Orcamento duro: para de iniciar NOVAS corridas quando o tempo decorrido
+ a estimativa da proxima corrida (1.3x a maior corrida ja medida, ou a
projecao do criterio se nenhuma corrida rodou ainda) ultrapassaria
ORCAMENTO_GPU_S. Corrida em andamento NAO e abortada no meio (checkpoint
atomico e responsabilidade do script congelado, ja testado nos gates
anteriores) -- o corte vale para a PROXIMA corrida.

Grava, por corrida, um registro em gpu/A2/lote_A2_status.json (append via
reescrita atomica do arquivo inteiro) com: run_label, comando, rc,
tempo_s, iniciado_em, concluido_em, erro (se houver). Esse status.json e
a fonte que o script de registro (registrar_A2.py) le -- nao numero
copiado a mao.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

A2_DIR = Path(__file__).resolve().parent
MODELO_V3_DIR = A2_DIR.parent / "modelo_v3"
CRITERIO_PATH = A2_DIR.parent.parent / "criterios" / "criterio_A2.json"
STATUS_PATH = A2_DIR / "lote_A2_status.json"

ORCAMENTO_GPU_S = 180 * 60  # teto duro do protocolo (orcamento_gpu_min=180)
PROJECAO_INICIAL_S = 20 * 60  # projecao conservadora por corrida antes de medir a 1a (errata: ~15,7min + overhead)

PYTHON = sys.executable  # este driver so e valido se invocado pela venv CUDA


def carregar_status() -> dict:
    if STATUS_PATH.exists():
        with open(STATUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"iniciado_em": datetime.now(timezone.utc).isoformat(), "corridas": [],
            "orcamento_gpu_s": ORCAMENTO_GPU_S, "veredito_lote": None}


def salvar_status(status: dict) -> None:
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    tmp.replace(STATUS_PATH)  # escrita atomica (rename), nunca sobrescrita parcial


def main() -> int:
    with open(CRITERIO_PATH, "r", encoding="utf-8") as f:
        criterio = json.load(f)
    corridas_planejadas = criterio["corridas"]
    cfg = criterio["config_fixa_por_corrida"]

    status = carregar_status()
    ja_feitas = {c["run_label"] for c in status["corridas"] if c.get("rc") == 0}

    t0_lote = time.perf_counter()
    duracoes_ok = [c["tempo_s"] for c in status["corridas"] if c.get("rc") == 0]

    for corrida in corridas_planejadas:
        run_label = corrida["run_label"]
        if run_label in ja_feitas:
            print(f"[lote_A2] {run_label} ja concluida (rc=0) -- pulando (retomavel).", flush=True)
            continue

        decorrido = time.perf_counter() - t0_lote
        estimativa_proxima = (max(duracoes_ok) * 1.3) if duracoes_ok else PROJECAO_INICIAL_S
        if decorrido + estimativa_proxima > ORCAMENTO_GPU_S:
            print(f"[lote_A2] ORCAMENTO: decorrido={decorrido:.0f}s + estimativa={estimativa_proxima:.0f}s "
                  f"> teto={ORCAMENTO_GPU_S}s -- NAO inicia {run_label}. Parando aqui (abortado_por_orcamento).",
                  flush=True)
            status["veredito_lote"] = "abortado_por_orcamento"
            salvar_status(status)
            return 2

        tipo = corrida["tipo"]
        cidade = corrida["cidade"]
        quadrante = corrida["quadrante"]
        script = "train_gnn_v3.py" if tipo == "gnn" else "train_mlp_v3.py"

        argv = [
            PYTHON, str(MODELO_V3_DIR / script),
            "--cidade", cidade,
            "--quadrante", quadrante,
            "--seed-treino", str(cfg["seed_treino"]),
            "--split-seed", str(cfg["split_seed"]),
            "--epochs", str(cfg["epochs"]),
            "--evid-dir", str(A2_DIR),
            "--run-label", run_label,
            *(["--hidden-dim", str(cfg["hidden_dim"])] if corrida["tipo"] == "gnn" else []),
            "--batch-size", str(cfg["batch_size"]),
            "--lr", str(cfg["lr"]),
        ]
        if tipo == "gnn":
            argv += ["--k-antenna", str(cfg["k_antenna"]), "--k-terrain", str(cfg["k_terrain"])]
        if cfg["mmap"]:
            argv += ["--mmap"]

        print(f"[lote_A2] iniciando {run_label}: {' '.join(argv)}", flush=True)
        iniciado_em = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        log_path = A2_DIR / f"log_{run_label}.txt"
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT, cwd=str(MODELO_V3_DIR))
        tempo_s = time.perf_counter() - t0
        concluido_em = datetime.now(timezone.utc).isoformat()

        registro = {
            "run_label": run_label, "tipo": tipo, "cidade": cidade, "quadrante": quadrante,
            "rep": corrida["rep"], "comando": argv, "rc": proc.returncode,
            "tempo_s": tempo_s, "iniciado_em": iniciado_em, "concluido_em": concluido_em,
            "log": str(log_path),
        }
        status["corridas"] = [c for c in status["corridas"] if c["run_label"] != run_label] + [registro]
        salvar_status(status)

        if proc.returncode == 0:
            duracoes_ok.append(tempo_s)
            print(f"[lote_A2] {run_label} OK em {tempo_s:.1f}s", flush=True)
        else:
            print(f"[lote_A2] {run_label} FALHOU rc={proc.returncode} -- ver {log_path}", flush=True)
            status["veredito_lote"] = "abortado_por_falha_corrida"
            salvar_status(status)
            return 1

    status["veredito_lote"] = "concluido"
    status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
    salvar_status(status)
    print("[lote_A2] TODAS as 8 corridas concluidas.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
