#!/usr/bin/env python
"""
e2_cartao_corridas.py -- frente ia-experiment-card, fio R2 (2026-09-24), tarefa E2.

Varre as corridas de treino do universo do Artigo2/Mathematics em
EVIDENCIA_RESUBMISSAO (dados/treinos_c1/run_*.json e treinos/<label>/run_*.json),
verifica campo a campo os 8 campos exigidos do cartao de experimento, recomputa
sha256 de run JSON / checkpoint_best.pt / dataset rf_data (.pt em
/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3), cruza com manifest.jsonl
(presente/ausente/sha diverge), confere script_sha256 contra os scripts
congelados em disco e identifica a fila (.ps1) que disparou o grupo.

Uso:
    /trabalho/ambientes/s33_amb_virtual/.venv/bin/python e2_cartao_corridas.py --hash
    /trabalho/ambientes/s33_amb_virtual/.venv/bin/python e2_cartao_corridas.py \
        --grupo mlpcf_campinas_sorocaba --out /caminho/saida.json
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

E = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
          "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO")
RF_DIR = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")

CAMPOS_EXIGIDOS = [
    "dataset.rf_data_file", "dataset.rf_data_sha256",
    "split.grid_km", "split.buffer_km", "split_seed",
    "particoes.train.idx_sha256_global", "particoes.val.idx_sha256_global",
    "particoes.test.idx_sha256_global",
    "seed", "config", "script_sha256",
    "ambiente.gpu", "ambiente.cuda", "ambiente.torch",
    "ambiente.cudnn_deterministic", "ambiente.cudnn_benchmark",
    "custo.tempo_total_s",
]

# fila congelada por prefixo de run_label (achado por leitura de conteudo dos .ps1,
# nao por convencao de nome -- ver post do fio)
FILA_POR_GRUPO = {
    "mlpcf": E / "dados/scripts_congelados/fila_mlp_c1v2.ps1",  # cobre bauru/lins E campinas/sorocaba (mesmo script, -Cidades diferente)
    "c0c1cfsc": E / "dados/scripts_congelados/fila_controles_c1v2.ps1",
    "c0c1cfinv": E / "dados/scripts_congelados/fila_controles_c1v2.ps1",
    "c0c1cf": E / "scripts/fila_treino_16_tiles.ps1",  # NAO esta em dados/scripts_congelados nem no manifest.jsonl
}

SCRIPT_TREINO = {
    "mlp": E / "dados/scripts_congelados/train_mlp_c0_spatial.py",
    "gnn": E / "dados/scripts_congelados/train_gnn_c0_spatial.py",
}

_HASH_CACHE = {}


def sha256_arquivo(caminho: Path, bufsize=8 * 1024 * 1024):
    caminho = Path(caminho)
    key = str(caminho.resolve()) if caminho.exists() else str(caminho)
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    if not caminho.exists():
        _HASH_CACHE[key] = None
        return None
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    v = h.hexdigest()
    _HASH_CACHE[key] = v
    return v


def get(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def win_para_linux(p):
    if not p:
        return None
    if p.startswith("F:\\TOPO_RF_DOWNLOAD_DRIVE"):
        rest = p[len("F:\\TOPO_RF_DOWNLOAD_DRIVE"):].replace("\\", "/")
        return "/trabalho/TOPO_RF_DOWNLOAD_DRIVE" + rest
    if p.startswith("D:\\_ARQUIVO_SSD_F\\TOPO_RF\\GNN_RF"):
        rest = p[len("D:\\_ARQUIVO_SSD_F\\TOPO_RF\\GNN_RF"):].replace("\\", "/")
        return "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF" + rest
    return None


def carregar_manifest():
    idx = {}
    for linha in (E / "manifest.jsonl").read_text().splitlines():
        linha = linha.strip()
        if not linha:
            continue
        try:
            obj = json.loads(linha)
        except json.JSONDecodeError:
            continue
        idx[obj.get("artefato")] = obj
    return idx


def descobrir_run_jsons():
    """Retorna lista de (grupo, run_label, caminho) para todo run_*.json do universo."""
    achados = []
    for p in sorted((E / "dados/treinos_c1").glob("run_*.json")):
        label = p.stem[len("run_"):]
        grupo = re.match(r"^(mlpcf|c0c1cfsc|c0c1cfinv|c0c1cf|c0c1v2|c0c1)", label)
        grupo = grupo.group(1) if grupo else "outro"
        achados.append((grupo, label, p, "dados/treinos_c1"))
    for d in sorted((E / "treinos").iterdir()):
        if not d.is_dir():
            continue
        rj = d / f"run_{d.name}.json"
        if rj.exists():
            grupo = re.match(r"^(mlpcf|c0c1cfsc|c0c1cfinv|c0c1cf|c0c1v2|c0c1)", d.name)
            grupo = grupo.group(1) if grupo else "outro"
            achados.append((grupo, d.name, rj, "treinos"))
    return achados


def script_treino_esperado(grupo):
    return SCRIPT_TREINO["mlp"] if grupo == "mlpcf" else SCRIPT_TREINO["gnn"]


def avaliar_corrida(grupo, label, caminho_run, origem_pasta, manifest_idx):
    out = {
        "grupo": grupo, "run_label": label, "origem_pasta": origem_pasta,
        "caminho_run_json": str(caminho_run),
    }
    try:
        d = json.loads(caminho_run.read_text())
    except Exception as e:
        out["erro_leitura"] = str(e)
        out["campos_faltando"] = CAMPOS_EXIGIDOS
        out["veredito"] = "bloqueia"
        return out

    faltando = [c for c in CAMPOS_EXIGIDOS if get(d, c) in (None, "", {})]
    out["campos_faltando"] = faltando

    out["run_json_sha256"] = sha256_arquivo(caminho_run)

    # checkpoint
    ck_path = caminho_run.parent / "checkpoints" / "checkpoint_best.pt"
    out["checkpoint_path"] = str(ck_path)
    out["checkpoint_existe"] = ck_path.exists()
    out["checkpoint_sha256"] = sha256_arquivo(ck_path) if ck_path.exists() else None

    # script_sha256 declarado vs script congelado em disco
    script_decl_sha = d.get("script_sha256")
    script_esperado = script_treino_esperado(grupo)
    script_disco_sha = sha256_arquivo(script_esperado)
    out["script_congelado_path"] = str(script_esperado)
    out["script_sha256_declarado"] = script_decl_sha
    out["script_sha256_disco"] = script_disco_sha
    out["script_sha256_bate"] = (script_decl_sha == script_disco_sha) if script_decl_sha and script_disco_sha else None

    # rf_data
    rf_win = get(d, "dataset.rf_data_file")
    rf_linux = win_para_linux(rf_win)
    rf_decl_sha = get(d, "dataset.rf_data_sha256")
    rf_real_sha = sha256_arquivo(rf_linux) if rf_linux else None
    out["rf_data_linux_path"] = rf_linux
    out["rf_data_sha256_declarado"] = rf_decl_sha
    out["rf_data_sha256_recomputado"] = rf_real_sha
    out["rf_data_sha256_bate"] = (rf_decl_sha == rf_real_sha) if rf_decl_sha and rf_real_sha else None

    # idx_sha256_global das 3 particoes
    out["idx_sha256_global"] = {
        p: get(d, f"particoes.{p}.idx_sha256_global") for p in ("train", "val", "test")
    }

    # manifest
    rel_dados = f"dados/treinos_c1/run_{label}.json"
    rel_treinos = f"treinos/{label}/run_{label}.json"
    entrada = manifest_idx.get(rel_dados) or manifest_idx.get(rel_treinos)
    if entrada is None:
        out["manifest_status"] = "ausente"
    else:
        sha_manifest = entrada.get("sha256")
        sha_arquivo_correspondente = out["run_json_sha256"] if origem_pasta == "dados/treinos_c1" or rel_dados not in manifest_idx else None
        # confere contra o proprio arquivo lido (mesmo que caminho de origem seja outro)
        out["manifest_status"] = "presente" if sha_manifest == out["run_json_sha256"] else "sha_diverge"
        out["manifest_sha256"] = sha_manifest
        out["manifest_artefato"] = entrada.get("artefato")

    # determinismo
    out["determinismo"] = {
        "cudnn_deterministic": get(d, "ambiente.cudnn_deterministic"),
        "cudnn_benchmark": get(d, "ambiente.cudnn_benchmark"),
    }

    # seed: config declarada (run json top-level / config) vs checkpoint (config efetiva)
    seed_declarado = d.get("seed")
    split_seed_declarado = d.get("split_seed")
    seed_config = get(d, "config.seed")
    split_seed_config = get(d, "config.split_seed")
    out["seed_declarado_topo"] = seed_declarado
    out["seed_config"] = seed_config
    out["split_seed_declarado_topo"] = split_seed_declarado
    out["split_seed_config"] = split_seed_config
    out["seed_diverge_topo_vs_config"] = (seed_declarado != seed_config) or (split_seed_declarado != split_seed_config)

    if faltando or out.get("script_sha256_bate") is False or out.get("rf_data_sha256_bate") is False or out["seed_diverge_topo_vs_config"] or out["manifest_status"] == "sha_diverge":
        out["veredito"] = "bloqueia"
    elif out["manifest_status"] == "ausente" or out.get("rf_data_sha256_bate") is None or not caminho_run.exists():
        out["veredito"] = "atencao"
    else:
        out["veredito"] = "ok"

    fila = FILA_POR_GRUPO.get(grupo)
    out["fila_congelada_path"] = str(fila) if fila else None
    out["fila_congelada_sha256"] = sha256_arquivo(fila) if fila else None
    out["fila_congelada_no_manifest"] = (str(fila.relative_to(E)) in manifest_idx) if fila else None

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true", help="imprime sha256 do proprio script e sai")
    ap.add_argument("--grupo", default=None, help="filtra por prefixo de grupo (mlpcf, c0c1cf, c0c1cfsc, c0c1cfinv)")
    ap.add_argument("--cidade", default=None, help="filtra por substring do run_label (ex.: campinas, sorocaba)")
    ap.add_argument("--out", default=None, help="grava JSON de saida neste caminho")
    args = ap.parse_args()

    if args.hash:
        print(sha256_arquivo(Path(__file__)))
        return

    manifest_idx = carregar_manifest()
    corridas = descobrir_run_jsons()
    if args.grupo:
        corridas = [c for c in corridas if c[0] == args.grupo]
    if args.cidade:
        corridas = [c for c in corridas if args.cidade in c[1]]

    resultados = []
    for grupo, label, caminho, origem in corridas:
        resultados.append(avaliar_corrida(grupo, label, caminho, origem, manifest_idx))

    saida = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_arquivo(Path(__file__)),
        "n_corridas_avaliadas": len(resultados),
        "corridas": resultados,
    }
    texto = json.dumps(saida, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(texto)
    else:
        print(texto)


if __name__ == "__main__":
    main()
