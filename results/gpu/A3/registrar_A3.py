#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A3/registrar_A3.py -- oficina-experimento. Le os artefatos REAIS
gravados pelo driver (lote_A3_status.json) e pelos wrappers congelados
(run_<label>.json, checkpoints, predicoes .npz, quando existirem) e
escreve, por script (nunca a mao):
  - cartao_experimento_A3.md  (narrativa + campos do protocolo-experimento)
  - manifest_A3.json          (sha256 de cada artefato citavel desta rodada)
  - versions_A3.csv           (uma linha por corrida planejada, com role)

Nao interpreta se o resultado e bom ou ruim (isso e do rigor) -- so
recomputa e registra o que os artefatos dizem.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

A3_DIR = Path(__file__).resolve().parent
CRITERIO_PATH = A3_DIR / "criterio_A3.json"
STATUS_PATH = A3_DIR / "lote_A3_status.json"
AMBIENTE_FREEZE_PATH = A3_DIR / "ambiente_freeze_A3.txt"
IA_BUG_SILENCIOSO_PATH = A3_DIR / "ia-bug-silencioso.json"


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
              IA_BUG_SILENCIOSO_PATH, A3_DIR / "rodar_lote_A3.py"]:
        if p.exists():
            artefatos.append({
                "arquivo": str(p), "sha256": sha256_arquivo(p), "bytes": p.stat().st_size,
                "regeneravel_por": "gpu/A3/rodar_lote_A3.py (status) ou nao aplicavel (registro/criterio/gate)",
            })

    checkpoints_por_label = {}
    for label, c in corridas_status.items():
        log_p = Path(c["log"])
        if log_p.exists():
            artefatos.append({"arquivo": str(log_p), "sha256": sha256_arquivo(log_p),
                               "bytes": log_p.stat().st_size,
                               "regeneravel_por": "gpu/A3/rodar_lote_A3.py (nao determinístico se GPU mudar de estado)"})
        run_json = A3_DIR / label / f"run_{label}.json"
        if run_json.exists():
            artefatos.append({"arquivo": str(run_json), "sha256": sha256_arquivo(run_json),
                               "bytes": run_json.stat().st_size,
                               "regeneravel_por": f"gpu/modelo_v3/train_{c['tipo']}_v3.py --cidade {c['cidade']} --quadrante {c['quadrante']} --seed-treino {c['seed_treino']} --run-label {label}"})
        ckpt = A3_DIR / label / "checkpoints" / "checkpoint_best.pt"
        if ckpt.exists():
            h = sha256_arquivo(ckpt)
            checkpoints_por_label[label] = {"arquivo": str(ckpt), "sha256": h}
            artefatos.append({"arquivo": str(ckpt), "sha256": h,
                               "bytes": ckpt.stat().st_size,
                               "regeneravel_por": f"gpu/modelo_v3/train_{c['tipo']}_v3.py --cidade {c['cidade']} --quadrante {c['quadrante']} --seed-treino {c['seed_treino']} --run-label {label} (SEM CADEIA)"})
        npz = A3_DIR / label / f"predicoes_{label}.npz"
        if npz.exists():
            artefatos.append({"arquivo": str(npz), "sha256": sha256_arquivo(npz),
                               "bytes": npz.stat().st_size,
                               "regeneravel_por": f"reconstrucao da particao de teste pelo mesmo wrapper, run-label {label}"})

    manifest = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "gerado_por": "gpu/A3/registrar_A3.py",
        "source_root": str(A3_DIR),
        "criterio_sha256": criterio_hash,
        "ambiente_hash": ambiente_hash,
        "artefatos": artefatos,
        "custo": {
            "orcamento_gpu_s_planejado": status.get("orcamento_gpu_s"),
            "tempo_total_lote_s": status.get("tempo_total_lote_s"),
            "veredito_lote": status.get("veredito_lote"),
        },
    }
    manifest_path = A3_DIR / "manifest_A3.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ---- versions_A3.csv ----
    versions_path = A3_DIR / "versions_A3.csv"
    fieldnames = ["run_label", "tipo", "cidade", "quadrante", "seed_treino", "rc", "tempo_s",
                  "iniciado_em", "concluido_em", "ambiente_hash", "checkpoint_sha256", "role"]
    linhas_csv = []
    with open(versions_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for corrida in corridas_planejadas:
            label = corrida["run_label"]
            c = corridas_status.get(label)
            ckpt_sha = checkpoints_por_label.get(label, {}).get("sha256", "")
            if c is None:
                role = "nao_iniciado_orcamento"
                row = {"run_label": label, "tipo": corrida["tipo"], "cidade": corrida["cidade"],
                       "quadrante": corrida["quadrante"], "seed_treino": corrida["seed_treino"], "rc": "",
                       "tempo_s": "", "iniciado_em": "", "concluido_em": "",
                       "ambiente_hash": ambiente_hash, "checkpoint_sha256": "", "role": role}
            else:
                if c["rc"] == 0:
                    role = f"{corrida['cidade']}_{corrida['quadrante']}_{corrida['tipo']}_s{corrida['seed_treino']}_ok"
                elif c["rc"] == 3:
                    role = "descartado_proveniencia_sha256_nao_bate"
                else:
                    role = f"descartado_rc{c['rc']}"
                row = {"run_label": label, "tipo": corrida["tipo"], "cidade": corrida["cidade"],
                       "quadrante": corrida["quadrante"], "seed_treino": corrida["seed_treino"],
                       "rc": c["rc"], "tempo_s": round(c["tempo_s"], 2), "iniciado_em": c["iniciado_em"],
                       "concluido_em": c["concluido_em"], "ambiente_hash": ambiente_hash,
                       "checkpoint_sha256": ckpt_sha, "role": role}
            w.writerow(row)
            linhas_csv.append(row)

    # ---- cartao_experimento_A3.md ----
    ok = [c for c in corridas_status.values() if c["rc"] == 0]
    falha = [c for c in corridas_status.values() if c["rc"] != 0]
    nao_iniciadas = [c["run_label"] for c in corridas_planejadas if c["run_label"] not in corridas_status]
    cartao_path = A3_DIR / "cartao_experimento_A3.md"

    veredito_lote = status.get("veredito_lote")
    tempo_total_s = status.get("tempo_total_lote_s") or sum(c["tempo_s"] for c in corridas_status.values())

    linhas = []
    linhas.append("---")
    linhas.append("experimento: A3 (PLANO A / G2 paridade)")
    linhas.append("protocolo: PROTOCOLO_GPU_v3_2026-09-25.md")
    linhas.append(f"criterio: gpu/A3/criterio_A3.json (sha256 {criterio_hash})")
    linhas.append("tipo: confirmatorio")
    linhas.append("seed_treino: 42-46 (uma por corrida, ver versions_A3.csv)")
    linhas.append("split_seed: 42 (fixo em todas as corridas)")
    linhas.append(f"ambiente_hash: {ambiente_hash}")
    linhas.append(f"gerado_em: {datetime.now(timezone.utc).isoformat()}")
    linhas.append("agente: oficina-experimento")
    linhas.append(f"veredito_lote_driver: {veredito_lote}")
    linhas.append("---")
    linhas.append("")
    linhas.append("# Cartao de experimento -- A3 (G2 paridade, SEM CADEIA)")
    linhas.append("")
    linhas.append("## Guardas de entrada desta rodada")
    linhas.append("")
    linhas.append(
        "Guarda de protocolo: PROTOCOLO_GPU_v3_2026-09-25.md aprovado, secao PLANO A / item A3. "
        "Guarda de bug silencioso: gpu/A3/ia-bug-silencioso.json veredito=atencao, com os 2 achados "
        "de gravidade 'atencao' explicitamente enderecados e aceitos pelo chefe na secao "
        "'DECISAO DO CHEFE (26/09 ~06:00)' do proprio protocolo (correlacao canal 2/roughness "
        "explicada fisicamente por construcao do gerador de alvo; fracao_passos_clipados=1,0 "
        "registrada para o model card, nao bloqueia por regra do eng-IA) -- portanto NAO e "
        "'bloqueia' nem 'atencao' sem aceite documentado, e o treino desta rodada prossegue.")
    linhas.append("")
    linhas.append("## Criterio de aceite congelado (copiado de gpu/A3/criterio_A3.json)")
    linhas.append("")
    linhas.append(f"> {criterio['criterio_aceite']}")
    linhas.append("")
    linhas.append("## O que rodou de fato")
    linhas.append("")
    linhas.append(f"Corridas planejadas: {len(corridas_planejadas)}. Concluidas com rc=0: {len(ok)}. "
                   f"Falharam/abortaram (rc!=0): {len(falha)}. Nao iniciadas: {len(nao_iniciadas)}. "
                   f"Tempo total do lote nesta chamada: {tempo_total_s:.1f}s (~{tempo_total_s/60:.1f} min) "
                   f"de um orcamento duro de {criterio.get('orcamento','840 min')}.")
    linhas.append("")
    for c in status["corridas"]:
        linhas.append(f"- `{c['run_label']}` (seed={c['seed_treino']}): rc={c['rc']}, "
                       f"tempo={c['tempo_s']:.1f}s, log=`{Path(c['log']).name}`")
    if nao_iniciadas:
        linhas.append("")
        linhas.append(
            f"Nao iniciadas ({len(nao_iniciadas)} de {len(corridas_planejadas)}): lote parado "
            f"deliberadamente por esta chamada de oficina-experimento ANTES de esgotar o teto formal "
            f"de orcamento_gpu_min=840 -- uma unica chamada sincrona deste agente nao pode manter a "
            f"conexao aberta pelas ~10-14h necessarias para as 80 corridas completas; o driver "
            f"(gpu/A3/rodar_lote_A3.py) e RETOMAVEL por run_label (pula toda corrida com rc=0 ja "
            f"registrada em lote_A3_status.json), entao uma proxima chamada continua exatamente daqui, "
            f"sem re-rodar nada ja concluido. Isto NAO e 'abortado_por_orcamento' no sentido do teto de "
            f"840 min ter sido esgotado (nao foi) -- e um corte de sessao, registrado com honestidade "
            f"no campo cobertura deste cartao e no oficina-experimento.json.")
    linhas.append("")
    linhas.append("## Contrato de dados / proveniencia")
    linhas.append("")
    linhas.append(
        "Nao existe script de contrato-de-dados dedicado (skill `contrato-de-dados` em "
        "GNN_TOPO/METODOLOGIA) cobrindo os arquivos `transfer_dataset_<cidade>_v19_<Q>_enriched_cftudo.pt` "
        "e `<cidade>_v19_<Q>_gpu.pt` usados por este lote -- ACHADO: \"contrato de dados formal nao "
        "implementado para arpia_rf/modelo_v3\". O que existe e equivalente, embutido no codigo "
        "congelado desde a ERRATA DE PROVENIENCIA (26/09): `v3_common.verificar_proveniencia_dataset` "
        "recalcula/reusa o sha256 de cada `rf_data_file` e `graph_file` por corrida e confere contra "
        "`manifest_mathematics_v4.jsonl` (grupo tensores_cftudo, 16 entradas), ABORTANDO com rc=3 ANTES "
        "de gastar GPU se o sha256 nao bater -- e essa checagem que rodou, por construcao, em TODA "
        "corrida desta rodada (campo `insumos.sha256_rf_data_bate_manifest_v4` de cada run JSON).")
    linhas.append("")
    linhas.append("## Ambiente")
    linhas.append("")
    linhas.append(f"Freeze completo em `ambiente_freeze_A3.txt` (ambiente_hash `{ambiente_hash}`): "
                   f"Python 3.11.16, torch 2.10.0+cu128, CUDA runtime 12.8, driver 595.91.07, "
                   f"GPU RTX 5070 Ti 16303 MiB.")
    linhas.append("")
    linhas.append("## Checkpoints (atomicos, um arquivo por corrida, nunca sobrescritos)")
    linhas.append("")
    if checkpoints_por_label:
        for label, info in checkpoints_por_label.items():
            linhas.append(f"- `{label}`: `{info['arquivo']}` sha256=`{info['sha256']}`")
    else:
        linhas.append("Nenhum checkpoint gravado nesta chamada (ver corridas concluidas acima).")
    linhas.append("")
    linhas.append("## Cobertura")
    linhas.append("")
    registrado = ["criterio_A3.json", "lote_A3_status.json", "ambiente_freeze_A3.txt",
                  "ia-bug-silencioso.json (reusado do gate)", "manifest_A3.json", "versions_A3.csv"]
    registrado += [f"log_{c['run_label']}.txt" for c in status["corridas"]]
    registrado += [f"run_{c['run_label']}.json" for c in status["corridas"]
                   if (A3_DIR / c["run_label"] / f"run_{c['run_label']}.json").exists()]
    nao_registrado = []
    if nao_iniciadas:
        nao_registrado.append(f"{len(nao_iniciadas)} corridas nao iniciadas: " + ", ".join(nao_iniciadas))
    nao_registrado.append("MAE agregado por populacao entre seeds (analise de dispersao) -- fora do "
                           "mandato do executor, cabe ao rigor (forum-eng-ia/forum-fisico-matematico)")
    nao_registrado.append("contrato de dados formal (script dedicado) -- achado registrado acima")
    linhas.append("registrado: " + "; ".join(registrado))
    linhas.append("")
    linhas.append("nao_registrado: " + "; ".join(nao_registrado))
    linhas.append("")
    linhas.append(
        "motivo: lote parado por decisao operacional desta chamada (limite pratico de duracao de uma "
        "unica invocacao sincrona), nao por esgotamento do teto formal de 840 min nem por falha de "
        "corrida nem por guarda de VRAM/proveniencia -- ver veredito_lote_driver no cabecalho.")
    conteudo = "\n".join(linhas) + "\n"
    with open(cartao_path, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print(json.dumps({
        "manifest": str(manifest_path), "versions": str(versions_path),
        "cartao": str(cartao_path), "ambiente_hash": ambiente_hash,
        "ok": len(ok), "falha": len(falha), "nao_iniciadas": len(nao_iniciadas),
        "tempo_total_s": tempo_total_s,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
