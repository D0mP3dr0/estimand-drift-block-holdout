#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
# -*- coding: utf-8 -*-
"""
gpu/G1/rodar_lote_G1_bloco4.py -- LANCADOR DO BLOCO 4 do lote G1 (03/10/2026),
conforme criterios/criterio_G1b_campinas_bloco_cruzado.json (fixado
2026-10-03T16:18:37-03:00, antes de existir este arquivo e antes de qualquer
corrida do bloco 4). Autorizado pelo dono em 03/10/2026 (16h).

COPIA DECLARADA de gpu/G1/rodar_lote_G1_retomada.py
  sha256 do original: 4f4f61cf170319ea9f3752fb78d83681065b4bc034b347751777a5d9edd3fc79
(que por sua vez copia rodar_lote_G1.py, sha e05a9214...bc59). Os originais, o shim
train_v3_g10.py, gpu/modelo_v3, lote_G1_status.json, plano_G1.json e
nao_treinaveis_G1_retomada.json NAO sao editados nem reescritos.

DIFERENCAS em relacao ao retomada (unicas):
  1. Plano: SO as 20 corridas do bloco 4 (campo `bloco: 4`): Campinas Q1,
     sementes de treino 43 e 44, sorteios 739191, 29675, 676375, 387379, 259492,
     GNN e MLP; run_label g1_<modelo>_campinas_Q1_ss<split>_s<seed>. Ordem: por
     sorteio, por semente, gnn depois mlp (alterna os modelos a cada par, de modo
     que, se o lote parar, cada par GNN/MLP fique completo). Os sorteios e a
     config sao validados contra o criterio G1b ao gerar o plano.
  2. Arquivos de saida PROPRIOS: status em lote_G1_bloco4_status.json, log em
     lote_G1_bloco4_driver.log, plano em plano_G1_bloco4.json, nao treinaveis em
     nao_treinaveis_G1_bloco4.json (os do lote G1 ficam intactos).
  3. Teto 4 h (criterio G1b: ~3 h previstas), t0_lote zerado no inicio.
  4. Status novo (nao herda o do lote G1); logica de reconhecimento de falhas
     anteriores fica inerte (status vazio) e e mantida por ser copia declarada.
Tudo o mais (guarda de VRAM, guarda de sha do shim, escrita atomica, config,
criterio de corrida completa, sem_validos, rc=3, nao_treinavel pelos adendos 3 e 4)
e igual ao retomada. As corridas gravam em G1_DIR (mesmo diretorio do lote G1), sem
colisao de run_label com as existentes (prova de anterioridade do criterio G1b).

--- cabecalho do retomada, preservado ---
gpu/G1/rodar_lote_G1_retomada.py -- LANCADOR DE RETOMADA do lote G1 (03/10/2026),
conforme criterios/criterio_G1_adendo4.json.

COPIA DECLARADA de gpu/G1/rodar_lote_G1.py
  sha256 do original: e05a921488a13f4fb34e262968cddeb9cb3134bd44c9cfcb05bbaaa03ff3bc59
O original, o shim train_v3_g10.py e gpu/modelo_v3 NAO sao editados. Tudo o que
segue (teto, guarda de VRAM, guarda de sha do shim, escrita atomica do status,
config, plano, ordem, criterio de corrida completa, sem_validos, rc=3) e igual
ao original. O teto e zerado na retomada (t0_lote reinicia), como no desenho.

UNICA diferenca de comportamento (adendo 4):
  Ao encontrar rc != 0 cujo log contem 'PARTICAO DEGENERADA' em particao
  diferente de 'test', registra no status `nao_treinavel: true` (com a particao
  nomeada e a nota "excluido da celula para GNN e MLP; nao substituido") e SEGUE,
  em vez de parar o lote com abortado_por_falha_corrida. Particao degenerada
  que inclua 'test' continua parando o lote (falha nao prevista).
  Para a corrida ja falhada antes da emenda (g1_gnn_bauru_Q3_ss769856_s42, com
  registro de falha rc=1 no status e run JSON parcial sem .npz), o lancador a
  reconhece pelo log existente, ACRESCENTA o marcador ao registro dela no status
  (sem apagar nenhum campo do registro original) e NAO a reexecuta. O mesmo
  marcador e gravado em arquivo proprio gpu/G1/nao_treinaveis_G1_retomada.json.
  O MLP do mesmo sorteio e executado normalmente; se falhar pelo mesmo motivo,
  recebe o mesmo tratamento.
  Aditivo apenas: o `veredito_lote` anterior fica em `veredito_lote_anterior`
  e o status ganha `retomadas` (log das retomadas); o veredito final sobrescreve
  `veredito_lote` ao fim, como no original.

--- cabecalho do original, preservado ---
gpu/G1/rodar_lote_G1.py -- oficina-experimento, teste G1_modelos_g10,
criterio criterios/criterio_G1_modelos_g10.json (fixado 2026-10-01T22:47:16-03:00,
antes de existir este script).

COPIA DECLARADA de gpu/A4/rodar_lote_A4.py
  sha256 do original: 524ddab5b79047a4645cf5ee31ff41fb7571e7bf6526e8e47c4c0f61adb55ee5
Mantidos sem mudanca: logica de retomada, escrita atomica de status, env
(PYTORCH_CUDA_ALLOC_CONF, OMP/MKL 4 threads), guarda de VRAM livre < 1 GB,
tratamento de rc=3 (proveniencia), config fixa (8 epocas, hidden 256, batch
12288, lr 1e-3, k -1/-1, mmap, sem cadeia, dataset canonico cftudo por sha).

Diferencas declaradas em relacao ao A4:
  1. Geometria g = 10 km, b = 2 km (A4: 5/2). Os wrappers v3 nao expoem
     --grid-km/--buffer-km; por decisao do chefe (opcao B, 01/10/2026) o lancador
     chama o shim gpu/G1/train_v3_g10.py (--modelo gnn|mlp), que importa o wrapper
     original sem altera-lo e faz chegar `--grid-km 10 --buffer-km 2` ao main do
     congelado. GUARDA: o lancador RECUSA iniciar (rc=5) se o shim nao existir ou
     se o sha256 de qualquer original (wrappers, v3_common, rf_decoder_v3,
     congelados) divergir dos registrados no shim. Nenhum arquivo de
     gpu/modelo_v3 nem o congelado e editado.
  2. O plano nao vem de lista no criterio: e gerado aqui, deterministicamente,
     do criterio_G1 + lista de 60 sorteios de
     fase2/2.1_deriva_erro_baselines_16x60rnd.json (nota_divergencia_seeds.
     seeds_usados_nesta_rodada), tomando-se os primeiros 20 (blocos 1 e 3) e
     os primeiros 5 (bloco 2).
  3. Ordem de prioridade (bloco 1, 2, 3). Dentro de cada sorteio:
     {cidade A, cidade B} x {gnn, mlp} em ordem gnn-A, gnn-B, mlp-A, mlp-B,
     com A = bauru nos sorteios de indice par e campinas nos impares, de modo
     que, se o lote parar, as duas cidades tenham o mesmo numero de sorteios
     concluidos (diferenca maxima de um modelo em um sorteio).
  4. Retomavel por corrida: pula a que ja tem run JSON completo (run JSON com
     blocos `modelo_v3` e `insumos`, sha256 do dataset == manifest v4, e .npz
     de predicoes presente) ou ja registrada como `sem_validos`.
  5. Sorteio sem no valido no teste apos o buffer (n_pop_teste.validos == 0 na
     analise 2.1) e registrado como `sem_validos` e NAO substituido: se a
     corrida terminar com rc != 0 para um sorteio conhecido como sem validos,
     registra e SEGUE; se terminar com rc == 0, segue normalmente. Qualquer
     outra falha (rc != 0 em sorteio com validos) para o lote, como no A4.
  6. Log em gpu/G1/lote_G1_driver.log (append) alem do stdout.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

G1_DIR = Path(__file__).resolve().parent
RAIZ = G1_DIR.parent.parent                      # _v3_2026-09-25
MODELO_V3_DIR = G1_DIR.parent / "modelo_v3"
CRITERIO_PATH = RAIZ / "criterios" / "criterio_G1b_campinas_bloco_cruzado.json"
SORTEIOS_PATH = RAIZ / "fase2" / "2.1_deriva_erro_baselines_16x60rnd.json"
PARCIAL_2_1_PATH = RAIZ / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json"
MANIFEST_V4 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/manifest_mathematics_v4.jsonl")
STATUS_PATH = G1_DIR / "lote_G1_bloco4_status.json"
LOG_PATH = G1_DIR / "lote_G1_bloco4_driver.log"
PLANO_PATH = G1_DIR / "plano_G1_bloco4.json"
NAO_TREINAVEIS_PATH = G1_DIR / "nao_treinaveis_G1_bloco4.json"
CRITERIO_ADENDO4_PATH = RAIZ / "criterios" / "criterio_G1_adendo4.json"
NOTA_NAO_TREINAVEL = "excluido da celula para GNN e MLP; nao substituido"

GRID_KM = 10.0
BUFFER_KM = 2.0
CFG = {"epochs": 8, "hidden_dim": 256, "batch_size": 12288, "lr": 0.001,
       "k_antenna": -1, "k_terrain": -1, "mmap": True}
CIDADES = ("bauru", "campinas")

# Teto duro do bloco 4: criterio G1b estima ~3 h (10 GNN x ~15,4 min + 10 MLP x ~1,8 min); teto = 4 h. Retomar zera o contador de teto (como no A4); estender e decisao do chefe.
ORCAMENTO_GPU_S = 4 * 3600
PROJECAO_INICIAL_S = {"gnn": 20 * 60, "mlp": 4 * 60}
VRAM_LIVRE_MINIMA_MIB = 1024

PYTHON = sys.executable  # so e valido se invocado pela venv CUDA


def log(msg: str) -> None:
    linha = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(linha, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


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
    tmp.replace(STATUS_PATH)  # atomica


def particoes_degeneradas(log_path: Path) -> list[str]:
    """Particoes nomeadas em 'PARTICAO DEGENERADA: [...]' no log (lista vazia se nao houver)."""
    try:
        txt = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    achadas: list[str] = []
    for m in re.finditer(r"PARTICAO DEGENERADA:\s*\[([^\]]*)\]", txt):
        for nome in re.findall(r"['\"]?([A-Za-z_]+)['\"]?", m.group(1)):
            if nome not in achadas:
                achadas.append(nome)
    return achadas


def marcar_nao_treinavel(status: dict, registro: dict, particoes: list[str], origem: str) -> None:
    """Acrescenta o marcador ao registro (nunca remove campos) e grava arquivo proprio."""
    registro["nao_treinavel"] = True
    registro["nao_treinavel_particao"] = particoes
    registro["nao_treinavel_nota"] = NOTA_NAO_TREINAVEL
    registro["nao_treinavel_marcado_em"] = datetime.now(timezone.utc).isoformat()
    registro["nao_treinavel_origem"] = origem
    registro["nao_treinavel_criterio"] = str(CRITERIO_ADENDO4_PATH)
    salvar_status(status)
    try:
        cur = json.loads(NAO_TREINAVEIS_PATH.read_text(encoding="utf-8")) if NAO_TREINAVEIS_PATH.exists() else {"criterio": str(CRITERIO_ADENDO4_PATH), "nao_treinaveis": []}
    except Exception:
        cur = {"criterio": str(CRITERIO_ADENDO4_PATH), "nao_treinaveis": []}
    cur["nao_treinaveis"] = [x for x in cur["nao_treinaveis"] if x["run_label"] != registro["run_label"]] + [{
        "run_label": registro["run_label"], "tipo": registro["tipo"], "cidade": registro["cidade"],
        "quadrante": registro["quadrante"], "split_seed": registro["split_seed"],
        "seed_treino": registro["seed_treino"], "rc": registro["rc"], "particao_degenerada": particoes,
        "nota": NOTA_NAO_TREINAVEL, "origem": origem, "log": registro["log"],
        "marcado_em": registro["nao_treinavel_marcado_em"]}]
    tmp = NAO_TREINAVEIS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(NAO_TREINAVEIS_PATH)  # atomica


def vram_livre_mib() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15, check=True).stdout.strip().splitlines()[0]
        return float(out)
    except Exception as e:
        log(f"[lote_G1] AVISO: nao consegui ler VRAM livre via nvidia-smi ({e}); assumindo 0 MiB (guarda conservadora).")
        return 0.0


def manifest_cftudo() -> dict:
    out = {}
    with open(MANIFEST_V4, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                e = json.loads(ln)
                if e.get("grupo") == "tensores_cftudo":
                    out[Path(e.get("caminho", "")).name] = e.get("sha256")
    return out


SHIM_PATH = G1_DIR / "train_v3_g10.py"


def guarda_shim() -> tuple[bool, str]:
    """Shim existe e os sha256 dos originais batem com os registrados no proprio shim."""
    if not SHIM_PATH.exists():
        return False, f"shim ausente: {SHIM_PATH}"
    sys.path.insert(0, str(G1_DIR))
    import train_v3_g10 as shim  # so importa stdlib; nao executa treino
    ruins = {}
    for p, esp in shim.SHA_ESPERADOS.items():
        if not Path(p).exists():
            ruins[str(p)] = "ausente"
        elif shim.sha256(Path(p)) != esp:
            ruins[str(p)] = "sha256 divergente"
    if shim.GRID_KM != GRID_KM or shim.BUFFER_KM != BUFFER_KM:
        ruins["g/b"] = f"shim={shim.GRID_KM}/{shim.BUFFER_KM}"
    return (not ruins), (json.dumps(ruins) if ruins else "ok")


def gerar_plano() -> list[dict]:
    with open(CRITERIO_PATH, "r", encoding="utf-8") as f:
        crit = json.load(f)
    assert crit["id"] == "G1b_campinas_bloco_cruzado_completo"
    d = crit["desenho"]
    assert d["bloco"] == 4 and d["celula"] == "campinas_Q1" and d["n_corridas"] == 20
    assert d["sementes_treino"] == [43, 44] and d["modelos"] == ["gnn", "mlp"]
    sorteios = [int(x) for x in d["sorteios"]]
    assert len(sorteios) == 5
    with open(SORTEIOS_PATH, "r", encoding="utf-8") as f:
        lista = json.load(f)["nota_divergencia_seeds"]["seeds_usados_nesta_rodada"]
    assert len(lista) == 60 and all(s in lista[:20] for s in sorteios)
    plano: list[dict] = []
    for ss in sorteios:
        for sd in d["sementes_treino"]:
            for tipo in d["modelos"]:
                plano.append({"bloco": 4, "run_label": f"g1_{tipo}_campinas_Q1_ss{ss}_s{sd}",
                              "tipo": tipo, "cidade": "campinas", "quadrante": "Q1",
                              "seed_treino": sd, "split_seed": ss, "indice_sorteio": lista.index(ss)})
    assert len(plano) == 20 and len({c["run_label"] for c in plano}) == 20
    return plano


def validos_conhecidos() -> dict:
    """(cidade_quadrante, split_seed) -> n_pop_teste.validos da analise 2.1 (so para classificar `sem_validos`)."""
    out = {}
    with open(PARCIAL_2_1_PATH, "r", encoding="utf-8") as f:
        p = json.load(f)
    for cel, d in p["celulas"].items():
        for s in d.get("por_sorteio", []):
            if s.get("status") == "ok":
                out[(cel, int(s["split_seed"]))] = s["n_pop_teste"]["validos"]
    return out


def corrida_completa(run_label: str, man: dict, cidade: str, quadrante: str) -> bool:
    d = G1_DIR / run_label
    rj, npz = d / f"run_{run_label}.json", d / f"predicoes_{run_label}.npz"
    if not (rj.exists() and npz.exists()):
        return False
    try:
        with open(rj, "r", encoding="utf-8") as f:
            r = json.load(f)
    except Exception:
        return False
    ins = r.get("insumos") or {}
    nome = f"transfer_dataset_{cidade}_v19_{quadrante}_enriched_cftudo.pt"
    return bool("modelo_v3" in r and ins.get("sha256_rf_data_bate_manifest_v4") is True
                and ins.get("sha256_rf_data") and ins.get("sha256_rf_data") == man.get(nome))


def main() -> int:
    ok_shim, motivo_shim = guarda_shim()
    if not ok_shim:
        log(f"[lote_G1] RECUSADO: guarda do shim falhou ({motivo_shim}). rc=5")
        return 5
    plano = gerar_plano()
    with open(PLANO_PATH, "w", encoding="utf-8") as f:
        json.dump({"criterio": str(CRITERIO_PATH), "grid_km": GRID_KM, "buffer_km": BUFFER_KM, "n": len(plano),
                   "corridas": plano}, f, indent=1, ensure_ascii=False)
    man = manifest_cftudo()
    vconh = validos_conhecidos()

    status = carregar_status()
    ja_sem_validos = {c["run_label"] for c in status["corridas"] if c.get("sem_validos")}
    status.setdefault("retomadas", []).append({"iniciada_em": datetime.now(timezone.utc).isoformat(),
                                               "lancador": "rodar_lote_G1_bloco4.py",
                                               "criterio": str(CRITERIO_PATH)})
    if status.get("veredito_lote") is not None:
        status["veredito_lote_anterior"] = status["veredito_lote"]
    salvar_status(status)

    # Reconhecimento das corridas ja falhadas antes da emenda: registro rc != 0 + log com
    # PARTICAO DEGENERADA fora de 'test' -> acrescenta marcador, NAO reexecuta.
    for c in status["corridas"]:
        if c.get("rc") not in (None, 0) and not c.get("nao_treinavel") and not c.get("sem_validos"):
            pd = particoes_degeneradas(Path(c.get("log", "")))
            if pd and "test" not in pd:
                marcar_nao_treinavel(status, c, pd, "falha anterior a emenda; reconhecida pelo log existente")
                log(f"[lote_G1_retomada] {c['run_label']}: rc={c['rc']} com particao degenerada {pd} -- marcada nao_treinavel (registro original mantido), NAO reexecutada.")
    ja_nao_treinavel = {c["run_label"] for c in status["corridas"] if c.get("nao_treinavel")}

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"

    t0_lote = time.perf_counter()
    duracoes_ok = {"gnn": [], "mlp": []}
    for c in status["corridas"]:
        if c.get("rc") == 0:
            duracoes_ok.setdefault(c["tipo"], []).append(c["tempo_s"])

    log(f"[lote_G1_bloco4] inicio: {len(plano)} corridas planejadas, g={GRID_KM} b={BUFFER_KM}, teto={ORCAMENTO_GPU_S}s")
    for corrida in plano:
        run_label, tipo = corrida["run_label"], corrida["tipo"]
        cidade, quadrante = corrida["cidade"], corrida["quadrante"]
        seed, split_seed = corrida["seed_treino"], corrida["split_seed"]

        if run_label in ja_nao_treinavel:
            log(f"[lote_G1] {run_label} ja marcada nao_treinavel -- pulando (nao reexecuta).")
            continue
        if run_label in ja_sem_validos or corrida_completa(run_label, man, cidade, quadrante):
            log(f"[lote_G1] {run_label} ja concluida -- pulando (retomavel).")
            continue

        decorrido = time.perf_counter() - t0_lote
        obs = duracoes_ok.get(tipo) or []
        estimativa = (max(obs) * 1.3) if obs else PROJECAO_INICIAL_S[tipo]
        if decorrido + estimativa > ORCAMENTO_GPU_S:
            log(f"[lote_G1] ORCAMENTO: decorrido={decorrido:.0f}s + estimativa({tipo})={estimativa:.0f}s "
                f"> teto={ORCAMENTO_GPU_S}s -- NAO inicia {run_label}. abortado_por_orcamento.")
            status["veredito_lote"] = "abortado_por_orcamento"
            status["tempo_total_lote_s"] = decorrido
            salvar_status(status)
            return 2

        vram_livre = vram_livre_mib()
        if vram_livre < VRAM_LIVRE_MINIMA_MIB:
            log(f"[lote_G1] VRAM livre={vram_livre:.0f} MiB < {VRAM_LIVRE_MINIMA_MIB} MiB -- NAO inicia {run_label}. Parando.")
            status["veredito_lote"] = "abortado_por_vram"
            status["tempo_total_lote_s"] = decorrido
            salvar_status(status)
            return 3

        argv = [
            PYTHON, str(SHIM_PATH), "--modelo", tipo,
            "--cidade", cidade, "--quadrante", quadrante,
            "--seed-treino", str(seed), "--split-seed", str(split_seed),
            "--epochs", str(CFG["epochs"]),
            "--evid-dir", str(G1_DIR), "--run-label", run_label,
            *(["--hidden-dim", str(CFG["hidden_dim"])] if tipo == "gnn" else []),
            "--batch-size", str(CFG["batch_size"]), "--lr", str(CFG["lr"]),
        ]
        if tipo == "gnn":
            argv += ["--k-antenna", str(CFG["k_antenna"]), "--k-terrain", str(CFG["k_terrain"])]
        if CFG["mmap"]:
            argv += ["--mmap"]

        log(f"[lote_G1] iniciando {run_label} (bloco {corrida['bloco']}, vram_livre={vram_livre:.0f} MiB): {' '.join(argv)}")
        iniciado_em = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        log_path = G1_DIR / f"log_{run_label}.txt"
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT, cwd=str(MODELO_V3_DIR), env=env)
        tempo_s = time.perf_counter() - t0
        concluido_em = datetime.now(timezone.utc).isoformat()

        registro = {"run_label": run_label, "bloco": corrida["bloco"], "tipo": tipo, "cidade": cidade,
                    "quadrante": quadrante, "seed_treino": seed, "split_seed": split_seed,
                    "grid_km": GRID_KM, "buffer_km": BUFFER_KM, "comando": argv, "rc": proc.returncode,
                    "tempo_s": tempo_s, "iniciado_em": iniciado_em, "concluido_em": concluido_em, "log": str(log_path)}
        status["corridas"] = [c for c in status["corridas"] if c["run_label"] != run_label] + [registro]

        if proc.returncode == 0:
            salvar_status(status)
            duracoes_ok.setdefault(tipo, []).append(tempo_s)
            log(f"[lote_G1] {run_label} OK em {tempo_s:.1f}s")
        elif proc.returncode == 3:
            log(f"[lote_G1] {run_label} ABORTOU rc=3 (sha256 do dataset NAO bate com manifest v4 -- achado de proveniencia, ver {log_path}). Parando.")
            status["veredito_lote"] = "abortado_por_proveniencia"
            status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
            salvar_status(status)
            return 4
        elif vconh.get((f"{cidade}_{quadrante}", split_seed)) == 0:
            registro["sem_validos"] = True
            registro["nota"] = "n_pop_teste.validos == 0 na analise 2.1; registrada como 'sem validos', nao substituida"
            salvar_status(status)
            log(f"[lote_G1] {run_label} rc={proc.returncode} em sorteio SEM VALIDOS (2.1) -- registrada como sem_validos, segue.")
        elif (lambda pd: bool(pd) and "test" not in pd)(particoes_degeneradas(log_path)):
            pd = particoes_degeneradas(log_path)
            marcar_nao_treinavel(status, registro, pd, "detectada nesta retomada")
            ja_nao_treinavel.add(run_label)
            log(f"[lote_G1] {run_label} rc={proc.returncode} PARTICAO DEGENERADA {pd} -- registrada nao_treinavel, segue.")
        else:
            log(f"[lote_G1] {run_label} FALHOU rc={proc.returncode} -- ver {log_path}")
            status["veredito_lote"] = "abortado_por_falha_corrida"
            status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
            salvar_status(status)
            return 1

    status["veredito_lote"] = "concluido"
    status["tempo_total_lote_s"] = time.perf_counter() - t0_lote
    salvar_status(status)
    log("[lote_G1] TODAS as corridas planejadas foram concluidas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
