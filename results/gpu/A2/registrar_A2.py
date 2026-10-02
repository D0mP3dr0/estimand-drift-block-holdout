#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A2/registrar_A2.py -- oficina-experimento. Le os artefatos REAIS
gravados pelo driver (lote_A2_status.json) e pelos wrappers congelados
(run_<label>.json, quando existirem) e escreve, por script (nunca a mao):
  - cartao_experimento_A2.md  (narrativa + campos do protocolo-experimento)
  - manifest_A2.json          (sha256 de cada artefato citavel desta rodada)
  - versions_A2.csv           (uma linha por corrida planejada, com role)

Nao interpreta se o resultado e bom ou ruim (isso e do rigor) -- so
recomputa e registra o que os artefatos dizem.
"""
from __future__ import annotations

import hashlib
import json
import csv
from datetime import datetime, timezone
from pathlib import Path

A2_DIR = Path(__file__).resolve().parent
V3_DIR = A2_DIR.parent / "_v3_2026-09-25" if False else A2_DIR.parent.parent
CRITERIO_PATH = A2_DIR.parent.parent / "criterios" / "criterio_A2.json"
STATUS_PATH = A2_DIR / "lote_A2_status.json"
AMBIENTE_FREEZE_PATH = A2_DIR / "ambiente_freeze_A2.txt"


def sha256_arquivo(p: Path, bloco: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bloco)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    with open(CRITERIO_PATH, "r", encoding="utf-8") as f:
        criterio = json.load(f)
    with open(STATUS_PATH, "r", encoding="utf-8") as f:
        status = json.load(f)
    ambiente_hash = sha256_arquivo(AMBIENTE_FREEZE_PATH)
    criterio_hash = sha256_arquivo(CRITERIO_PATH)

    corridas_status = {c["run_label"]: c for c in status["corridas"]}
    corridas_planejadas = criterio["corridas"]

    # ---- manifest ----
    artefatos = []
    for p in [CRITERIO_PATH, STATUS_PATH, AMBIENTE_FREEZE_PATH,
              A2_DIR / "ia-bug-silencioso.json", A2_DIR / "bug_silencioso_modelo_v3.json",
              A2_DIR / "rodar_lote_A2.py"]:
        if p.exists():
            artefatos.append({
                "arquivo": str(p), "sha256": sha256_arquivo(p), "bytes": p.stat().st_size,
                "regeneravel_por": None if p.name.endswith(".json") and "status" not in p.name
                else "gpu/A2/rodar_lote_A2.py (status) ou nao aplicavel (registro/criterio)",
            })
    for label, c in corridas_status.items():
        log_p = Path(c["log"])
        if log_p.exists():
            artefatos.append({"arquivo": str(log_p), "sha256": sha256_arquivo(log_p),
                               "bytes": log_p.stat().st_size,
                               "regeneravel_por": "gpu/A2/rodar_lote_A2.py (nao determinístico se GPU mudar de estado)"})
        run_json = A2_DIR / label / f"run_{label}.json"
        if run_json.exists():
            artefatos.append({"arquivo": str(run_json), "sha256": sha256_arquivo(run_json),
                               "bytes": run_json.stat().st_size,
                               "regeneravel_por": f"gpu/modelo_v3/train_{c['tipo']}_v3.py --cidade {c['cidade']} --quadrante {c['quadrante']} --run-label {label}"})
        ckpt = A2_DIR / label / "checkpoints" / "checkpoint_best.pt"
        if ckpt.exists():
            artefatos.append({"arquivo": str(ckpt), "sha256": sha256_arquivo(ckpt),
                               "bytes": ckpt.stat().st_size,
                               "regeneravel_por": f"gpu/modelo_v3/train_{c['tipo']}_v3.py --cidade {c['cidade']} --quadrante {c['quadrante']} --run-label {label} (SEM CADEIA, seed=42)"})
        npz = A2_DIR / label / f"predicoes_{label}.npz"
        if npz.exists():
            artefatos.append({"arquivo": str(npz), "sha256": sha256_arquivo(npz),
                               "bytes": npz.stat().st_size,
                               "regeneravel_por": f"reconstrucao da particao de teste pelo mesmo wrapper, run-label {label}"})

    manifest = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "gerado_por": "gpu/A2/registrar_A2.py",
        "source_root": str(A2_DIR),
        "criterio_sha256": criterio_hash,
        "ambiente_hash": ambiente_hash,
        "artefatos": artefatos,
        "custo": {
            "orcamento_gpu_s_planejado": status.get("orcamento_gpu_s"),
            "tempo_total_lote_s": status.get("tempo_total_lote_s"),
            "veredito_lote": status.get("veredito_lote"),
        },
    }
    manifest_path = A2_DIR / "manifest_A2.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ---- versions_A2.csv ----
    versions_path = A2_DIR / "versions_A2.csv"
    fieldnames = ["run_label", "tipo", "cidade", "quadrante", "rep", "rc", "tempo_s",
                  "iniciado_em", "concluido_em", "ambiente_hash", "role"]
    with open(versions_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for corrida in corridas_planejadas:
            label = corrida["run_label"]
            c = corridas_status.get(label)
            if c is None:
                role = "nao_iniciado_orcamento_vram"
                row = {"run_label": label, "tipo": corrida["tipo"], "cidade": corrida["cidade"],
                       "quadrante": corrida["quadrante"], "rep": corrida["rep"], "rc": "",
                       "tempo_s": "", "iniciado_em": "", "concluido_em": "",
                       "ambiente_hash": ambiente_hash, "role": role}
            else:
                if c["rc"] == 0:
                    role = f"{corrida['cidade']}_{corrida['quadrante']}_{corrida['tipo']}_rep{corrida['rep']}_ok"
                else:
                    role = "descartado_cuda_oom"
                row = {"run_label": label, "tipo": corrida["tipo"], "cidade": corrida["cidade"],
                       "quadrante": corrida["quadrante"], "rep": corrida["rep"], "rc": c["rc"],
                       "tempo_s": round(c["tempo_s"], 2), "iniciado_em": c["iniciado_em"],
                       "concluido_em": c["concluido_em"], "ambiente_hash": ambiente_hash, "role": role}
            w.writerow(row)

    # ---- cartao_experimento_A2.md ----
    ok = [c for c in corridas_status.values() if c["rc"] == 0]
    falha = [c for c in corridas_status.values() if c["rc"] != 0]
    nao_iniciadas = [c["run_label"] for c in corridas_planejadas if c["run_label"] not in corridas_status]
    cartao_path = A2_DIR / "cartao_experimento_A2.md"
    linhas = []
    linhas.append("---")
    linhas.append("experimento: A2 (PLANO A / G1 reprodutibilidade)")
    linhas.append("protocolo: PROTOCOLO_GPU_v3_2026-09-25.md")
    linhas.append(f"criterio: criterios/criterio_A2.json (sha256 {criterio_hash})")
    linhas.append("tipo: confirmatorio")
    linhas.append("seed_treino: 42")
    linhas.append("split_seed: 42")
    linhas.append(f"ambiente_hash: {ambiente_hash}")
    linhas.append(f"gerado_em: {datetime.now(timezone.utc).isoformat()}")
    linhas.append("agente: oficina-experimento")
    linhas.append("veredito_execucao: abortado_por_orcamento (guarda de VRAM do proprio protocolo)")
    linhas.append("---")
    linhas.append("")
    linhas.append("# Cartao de experimento -- A2 (G1 reprodutibilidade)")
    linhas.append("")
    linhas.append("## Criterio de aceite congelado (copiado de criterios/criterio_A2.json)")
    linhas.append("")
    linhas.append(f"> {criterio['criterio_aceite']}")
    linhas.append("")
    linhas.append("## O que rodou de fato")
    linhas.append("")
    linhas.append(f"Corridas planejadas: {len(corridas_planejadas)}. Concluidas com rc=0: {len(ok)}. "
                   f"Falharam: {len(falha)}. Nao iniciadas: {len(nao_iniciadas)}.")
    linhas.append("")
    for c in status["corridas"]:
        linhas.append(f"- `{c['run_label']}`: rc={c['rc']}, tempo={c['tempo_s']:.1f}s, "
                       f"log=`{Path(c['log']).name}`")
    if nao_iniciadas:
        linhas.append("")
        linhas.append("Nao iniciadas (lote abortado antes de chegar nelas): " + ", ".join(nao_iniciadas))
    linhas.append("")
    linhas.append("## Achado desta rodada (execucao, nao auditoria)")
    linhas.append("")
    linhas.append(
        "A primeira corrida (`gnn_v3_a2_bauru_Q1_rep1`, grafo inteiro, `--mmap`, "
        "sem `--max-nodes`, `k_antenna=k_terrain=-1`, `batch_size=24576`, `hidden_dim=256`) "
        "estourou VRAM (`torch.OutOfMemoryError`) aos 36,7 s, dentro do primeiro epoch, na "
        "chamada `scaler.scale(loss).backward()`. VRAM livre no INICIO do lote: 8544 MiB "
        "(processo do motor de embeddings, pid 1630, usando 5390 MiB; total usado em repouso "
        "7297 MiB de 16303 MiB). O processo de treino sozinho chegou a 8,22 GiB em uso antes "
        "de falhar ao alocar mais 406 MiB, com apenas 122 MiB livres reportados pelo driver no "
        "momento da falha -- ou seja, o lote esbarrou exatamente na guarda que o proprio "
        "protocolo ja previa (\"abortar lote se VRAM livre < 1 GB\"), so que a guarda disparou "
        "DENTRO da corrida (CUDA OOM), nao antes de inicia-la, porque a checagem de VRAM livre "
        "e feita uma vez no repouso e o pico real de uma corrida com vizinhanca completa "
        "(k=-1) no treino inteiro so aparece depois que o backward comeca.")
    linhas.append("")
    linhas.append(
        "Nenhuma tentativa de mitigacao (reduzir batch_size, desligar o motor de embeddings, "
        "trocar hidden_dim) foi feita nesta rodada -- mudar qualquer parametro do "
        "`config_fixa_por_corrida` do criterio_A2.json e decisao que muda a comparabilidade "
        "da propria pergunta de reprodutibilidade e nao cabe ao executor decidir sozinho "
        "(\"quem executa nunca audita\"; \"nunca estender o teto sozinho\"). O lote foi "
        "interrompido aqui e o achado registrado para decisao do dono/rigor.")
    linhas.append("")
    linhas.append("## Contrato de dados")
    linhas.append("")
    linhas.append(
        "NAO HA script de contrato-de-dados cobrindo os artefatos efetivamente usados por "
        "este lote (`GNN_RF_V2/graph_data/{bauru,lins}_v19_Q1_gpu.pt` e "
        "`transfer_dataset_{bauru,lins}_v19_Q1_enriched_v2.pt`). Existe um contrato de rodada "
        "anterior (`EVIDENCIA_RESUBMISSAO/_campanha_2026-09-13/dados_rodada2/contrato_dataset_rodada2.py`) "
        "mas ele mira 16 arquivos `*_enriched_cftudo.pt` num caminho Windows removido "
        "(`F:\\TOPO_RF_DOWNLOAD_DRIVE\\graph_data_v3`), escopo e populacao diferentes dos "
        "arquivos `_v2` usados por `train_gnn_v3.py`/`train_mlp_v3.py`. ACHADO: "
        "\"contrato de dados nao implementado\" para os arquivos `_gpu.pt`/`_enriched_v2.pt` "
        "de `GNN_RF_V2/graph_data` -- nao e motivo para pular o registro deste achado, mas "
        "tambem nao foi motivo para bloquear esta rodada, ja que o proprio protocolo A2 nao "
        "lista contrato-de-dados como portao de saida (os portoes desta fila sao G0/A1/A1bis, "
        "ja cumpridos) e o gate de bug silencioso desta rodada (`ia-bug-silencioso.json`) ja "
        "cobre vazamento por correlacao (item e, reusado do gate A1) e normalizacao 2x (item c).")
    linhas.append("")
    linhas.append("## Ambiente")
    linhas.append("")
    linhas.append(f"Freeze completo em `ambiente_freeze_A2.txt` (sha256/ambiente_hash "
                   f"`{ambiente_hash}`): Python 3.11.16, torch 2.10.0+cu128, CUDA runtime 12.8, "
                   f"driver 595.91.07, GPU RTX 5070 Ti 16303 MiB.")
    linhas.append("")
    linhas.append("## Cobertura")
    linhas.append("")
    linhas.append(f"registrado: criterio_A2.json, lote_A2_status.json, ambiente_freeze_A2.txt, "
                   f"log da corrida 1, run JSON parcial da corrida 1 (sem checkpoint, crash antes "
                   f"do 1o save), manifest_A2.json, versions_A2.csv.")
    linhas.append("nao_registrado: cartao completo com os 5 blocos do roadmap (nao aplicavel -- "
                   "A2 e portao de reprodutibilidade do PLANO A, nao \"resultado final\" do "
                   "roadmap v22); MAE por populacao das 8 corridas (0/8 corridas completas); "
                   "hash de checkpoint/npz (nenhum foi gravado, corrida 1 falhou antes do 1o save).")
    conteudo = "\n".join(linhas) + "\n"
    with open(cartao_path, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print(json.dumps({
        "manifest": str(manifest_path), "versions": str(versions_path),
        "cartao": str(cartao_path), "ambiente_hash": ambiente_hash,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
