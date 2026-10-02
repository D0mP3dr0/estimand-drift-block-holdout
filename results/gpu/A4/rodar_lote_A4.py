#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A4/rodar_lote_A4.py -- oficina-experimento, PLANO A / item A3 (G2
paridade), protocolo aprovado PROTOCOLO_GPU_v3_2026-09-25.md,
criterio gpu/A3/criterio_A4.json.

Adaptado de gpu/A2c/rodar_lote_A2c.py (mesma logica de retomada por
run_label/rc==0, escrita atomica de status, env com
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True). Diferencas:

  1. 80 corridas planejadas (8 celulas x 5 seeds x {gnn,mlp}), cada uma
     com `seed_treino` PROPRIO (42-46) em vez de repeticao da mesma seed
     -- este e o gate de PARIDADE (G2), nao o de reprodutibilidade (G1).
  2. Orcamento estimado POR TIPO (gnn/mlp), nao um unico max() global:
     GNN e MLP tem custos MUITO diferentes (~835s vs ~104s medidos em
     A2c, mesmo codigo/dataset/epocas) -- usar max() global penalizaria
     a estimativa de toda corrida MLP com o pior caso GNN e pararia o
     lote cedo demais. A guarda de orcamento aqui usa
     `max(duracoes_ok[tipo]) * 1.3` (ou projecao inicial por tipo,
     calibrada em A2c) para decidir se cabe a PROXIMA corrida.
  3. Guarda de VRAM livre (< 1 GB no repouso, ANTES de iniciar a proxima
     corrida) -- protocolo PLANO A: "abortar lote se VRAM livre < 1 GB".
  4. Trata rc=3 (verificar_proveniencia_dataset abortou por sha256 nao
     bater com o manifest v4) como achado de proveniencia, NAO falha de
     execucao generica -- grava e PARA o lote (nao decide sozinho se
     deve continuar; achado vai para o rigor).

Orcamento duro: ORCAMENTO_GPU_S vem do protocolo desta chamada
(orcamento_gpu_min=840). Corrida em andamento nunca e abortada no meio
(checkpoint atomico e responsabilidade do script congelado). Retomavel:
rodar de novo pula toda run_label com rc==0 ja registrado.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

A4_DIR = Path(__file__).resolve().parent
MODELO_V3_DIR = A4_DIR.parent / "modelo_v3"
CRITERIO_PATH = A4_DIR / "criterio_A4.json"
STATUS_PATH = A4_DIR / "lote_A4_status.json"

ORCAMENTO_GPU_S = 840 * 60  # teto duro desta chamada (orcamento_gpu_min=840)
PROJECAO_INICIAL_S = {"gnn": 20 * 60, "mlp": 4 * 60}  # calibrado em A2c (835s/104s) + margem
VRAM_LIVRE_MINIMA_MIB = 1024

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


def vram_livre_mib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0]
        return float(out)
    except Exception as e:  # pragma: no cover - guarda defensiva, nao interpretativa
        print(f"[lote_A4] AVISO: nao consegui ler VRAM livre via nvidia-smi ({e}); "
              f"assumindo 0 MiB (guarda conservadora).", flush=True)
        return 0.0


def main() -> int:
    with open(CRITERIO_PATH, "r", encoding="utf-8") as f:
        criterio = json.load(f)
    corridas_planejadas = criterio["corridas"]
    cfg = criterio["config_fixa_por_corrida"]

    status = carregar_status()
    ja_feitas = {c["run_label"] for c in status["corridas"] if c.get("rc") == 0}

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 26/09 20:15 (chefe): limite de threads de CPU por corrida -- so ambiente, nao protocolo
    # (Tctl 92-94 C com 12 threads OpenMP em spin; treino e na GPU). Vale das corridas seguintes em diante.
    env["OMP_NUM_THREADS"] = "4"; env["MKL_NUM_THREADS"] = "4"

    t0_lote = time.perf_counter()
    duracoes_ok = {"gnn": [], "mlp": []}
    for c in status["corridas"]:
        if c.get("rc") == 0:
            duracoes_ok.setdefault(c["tipo"], []).append(c["tempo_s"])

    for corrida in corridas_planejadas:
        run_label = corrida["run_label"]
        if run_label in ja_feitas:
            print(f"[lote_A4] {run_label} ja concluida (rc=0) -- pulando (retomavel).", flush=True)
            continue

        tipo = corrida["tipo"]
        cidade = corrida["cidade"]
        quadrante = corrida["quadrante"]
        seed = corrida["seed_treino"]
        split_seed = corrida.get("split_seed", cfg["split_seed"])

        decorrido = time.perf_counter() - t0_lote
        obs = duracoes_ok.get(tipo) or []
        estimativa_proxima = (max(obs) * 1.3) if obs else PROJECAO_INICIAL_S[tipo]
        if decorrido + estimativa_proxima > ORCAMENTO_GPU_S:
            print(f"[lote_A4] ORCAMENTO: decorrido={decorrido:.0f}s + estimativa({tipo})={estimativa_proxima:.0f}s "
                  f"> teto={ORCAMENTO_GPU_S}s -- NAO inicia {run_label}. Parando aqui (abortado_por_orcamento).",
                  flush=True)
            status["veredito_lote"] = "abortado_por_orcamento"
            status["tempo_total_lote_s"] = decorrido
            salvar_status(status)
            return 2

        vram_livre = vram_livre_mib()
        if vram_livre < VRAM_LIVRE_MINIMA_MIB:
            print(f"[lote_A4] VRAM livre={vram_livre:.0f} MiB < {VRAM_LIVRE_MINIMA_MIB} MiB -- "
                  f"guarda do protocolo (PLANO A: 'abortar lote se VRAM livre < 1 GB'). "
                  f"NAO inicia {run_label}. Parando aqui.", flush=True)
            status["veredito_lote"] = "abortado_por_vram"
            status["tempo_total_lote_s"] = decorrido
            salvar_status(status)
            return 3

        script = "train_gnn_v3.py" if tipo == "gnn" else "train_mlp_v3.py"
        argv = [
            PYTHON, str(MODELO_V3_DIR / script),
            "--cidade", cidade,
            "--quadrante", quadrante,
            "--seed-treino", str(seed),
            "--split-seed", str(split_seed),
            "--epochs", str(cfg["epochs"]),
            "--evid-dir", str(A4_DIR),
            "--run-label", run_label,
            *(["--hidden-dim", str(cfg["hidden_dim"])] if tipo == "gnn" else []),
            "--batch-size", str(cfg["batch_size"]),
            "--lr", str(cfg["lr"]),
        ]
        if tipo == "gnn":
            argv += ["--k-antenna", str(cfg["k_antenna"]), "--k-terrain", str(cfg["k_terrain"])]
        if cfg["mmap"]:
            argv += ["--mmap"]
        if tipo == "mlp" and corrida.get("larguras_antigas"):
            argv += ["--larguras-antigas"]

        print(f"[lote_A4] iniciando {run_label} (vram_livre={vram_livre:.0f} MiB): {' '.join(argv)}", flush=True)
        iniciado_em = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        log_path = A4_DIR / f"log_{run_label}.txt"
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT,
                                   cwd=str(MODELO_V3_DIR), env=env)
        tempo_s = time.perf_counter() - t0
        concluido_em = datetime.now(timezone.utc).isoformat()

        registro = {
            "run_label": run_label, "tipo": tipo, "cidade": cidade, "quadrante": quadrante,
            "seed_treino": seed, "split_seed": split_seed, "larguras_antigas": bool(corrida.get("larguras_antigas")), "comando": argv, "rc": proc.returncode,
            "tempo_s": tempo_s, "iniciado_em": iniciado_em, "concluido_em": concluido_em,
            "log": str(log_path),
        }
        status["corridas"] = [c for c in status["corridas"] if c["run_label"] != run_label] + [registro]
        salvar_status(status)

        if proc.returncode == 0:
            duracoes_ok.setdefault(tipo, []).append(tempo_s)
            print(f"[lote_A4] {run_label} OK em {tempo_s:.1f}s", flush=True)
        elif proc.returncode == 3:
            print(f"[lote_A4] {run_label} ABORTOU rc=3 (sha256 do dataset NAO bate com manifest v4 -- "
                  f"achado de proveniencia, ver {log_path}). Parando o lote aqui.", flush=True)
            status["veredito_lote"] = "abortado_por_proveniencia"
            status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
            salvar_status(status)
            return 4
        else:
            print(f"[lote_A4] {run_label} FALHOU rc={proc.returncode} -- ver {log_path}", flush=True)
            status["veredito_lote"] = "abortado_por_falha_corrida"
            status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
            salvar_status(status)
            return 1

    status["veredito_lote"] = "concluido"
    status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
    salvar_status(status)
    print("[lote_A4] TODAS as corridas planejadas foram concluidas.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
