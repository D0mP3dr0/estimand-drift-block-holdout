#!/usr/bin/env python
"""
e2_cartao_corridas_v2.py -- frente ia-experiment-card, fio R2 (2026-09-24), tarefa E2 v2.

Diferenca do v1 (e2_cartao_corridas.py): NAO recomputa sha256 dos rf_data (.pt de ate
28 GB); usa o sha256 declarado no run JSON e confere contra
/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/HASHES_SHA256.txt (hash local, capturado
antes desta rodada, ver post do fio). Recomputa apenas run JSON e checkpoint_best.pt
(pequenos, ~20 MB). Fonte canonica de cada corrida: treinos/<label>/run_<label>.json
(co-localizado com o checkpoint); confere que a copia em dados/treinos_c1/run_<label>.json
(quando existe) e byte-identica (diff), sem reler nada a olho.

Uso:
    /trabalho/ambientes/s33_amb_virtual/.venv/bin/python e2_cartao_corridas_v2.py --hash
    /trabalho/ambientes/s33_amb_virtual/.venv/bin/python e2_cartao_corridas_v2.py \
        --grupo c0c1cf --out /caminho/saida.json
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
HASHES_LOCAIS = RF_DIR / "HASHES_SHA256.txt"

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

FILA_POR_GRUPO = {
    "mlpcf": E / "dados/scripts_congelados/fila_mlp_c1v2.ps1",
    "c0c1cfsc": E / "dados/scripts_congelados/fila_controles_c1v2.ps1",
    "c0c1cfinv": E / "dados/scripts_congelados/fila_controles_c1v2.ps1",
    "c0c1cf": E / "scripts/fila_treino_16_tiles.ps1",
}

SCRIPT_TREINO = {
    "mlp": E / "dados/scripts_congelados/train_mlp_c0_spatial.py",
    "gnn": E / "dados/scripts_congelados/train_gnn_c0_spatial.py",
}

# labels ja verificadas por leitura manual completa (R1, 2026-09-23) ou por script v1
# com rehash total do rf_data (R2, 2026-09-24) -- nao reprocessadas aqui, so citadas.
JA_VERIFICADAS = {
    "mlpcf_bauru_s42_Q1_g10b2", "c0c1cf_bauru_s42_Q1_g10b2",
    "c0c1cfsc_bauru_s42_Q2_g10b2", "c0c1cfinv_lins_s42_Q1_g10b2",
}
JA_VERIFICADAS_PREFIXO = "mlpcf_campinas_", "mlpcf_sorocaba_"

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


def carregar_hashes_locais():
    idx = {}
    if not HASHES_LOCAIS.exists():
        return idx
    for linha in HASHES_LOCAIS.read_text().splitlines():
        linha = linha.strip()
        if not linha:
            continue
        partes = linha.split(None, 1)
        if len(partes) != 2:
            continue
        sha, nome = partes
        idx[nome.strip()] = sha.strip()
    return idx


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


def descobrir_run_jsons_canonicos():
    """Fonte canonica: treinos/<label>/run_<label>.json (co-localizado com checkpoint).
    Retorna lista de (grupo, label, caminho_treinos, caminho_dados_ou_None)."""
    achados = []
    for d in sorted((E / "treinos").iterdir()):
        if not d.is_dir():
            continue
        rj = d / f"run_{d.name}.json"
        if not rj.exists():
            continue
        grupo = re.match(r"^(mlpcf|c0c1cfsc|c0c1cfinv|c0c1cf|c0c1v2|c0c1)", d.name)
        grupo = grupo.group(1) if grupo else "outro"
        rj_dados = E / "dados/treinos_c1" / f"run_{d.name}.json"
        achados.append((grupo, d.name, rj, rj_dados if rj_dados.exists() else None))
    return achados


def script_treino_esperado(grupo):
    return SCRIPT_TREINO["mlp"] if grupo == "mlpcf" else SCRIPT_TREINO["gnn"]


def ja_verificada(label):
    if label in JA_VERIFICADAS:
        return True
    return any(label.startswith(p) for p in JA_VERIFICADAS_PREFIXO)


def avaliar_corrida(grupo, label, caminho_treinos, caminho_dados, manifest_idx, hashes_locais):
    out = {
        "grupo": grupo, "run_label": label,
        "caminho_run_json_treinos": str(caminho_treinos),
        "caminho_run_json_dados": str(caminho_dados) if caminho_dados else None,
    }
    try:
        d = json.loads(caminho_treinos.read_text())
    except Exception as e:
        out["erro_leitura"] = str(e)
        out["campos_faltando"] = CAMPOS_EXIGIDOS
        out["veredito"] = "bloqueia"
        return out

    faltando = [c for c in CAMPOS_EXIGIDOS if get(d, c) in (None, "", {})]
    out["campos_faltando"] = faltando

    out["run_json_sha256"] = sha256_arquivo(caminho_treinos)

    # copia em dados/treinos_c1: byte-identica?
    if caminho_dados:
        sha_dados = sha256_arquivo(caminho_dados)
        out["run_json_dados_sha256"] = sha_dados
        out["run_json_dados_identico"] = (sha_dados == out["run_json_sha256"])
    else:
        out["run_json_dados_sha256"] = None
        out["run_json_dados_identico"] = None

    # checkpoint (pequeno, recomputa)
    ck_path = caminho_treinos.parent / "checkpoints" / "checkpoint_best.pt"
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

    # rf_data -- SEM rehash: compara sha declarado contra HASHES_SHA256.txt local (por basename)
    rf_win = get(d, "dataset.rf_data_file")
    rf_linux = win_para_linux(rf_win)
    rf_basename = Path(rf_win.replace("\\", "/")).name if rf_win else None
    rf_decl_sha = get(d, "dataset.rf_data_sha256")
    rf_sha_local = hashes_locais.get(rf_basename) if rf_basename else None
    out["rf_data_linux_path"] = rf_linux
    out["rf_data_basename"] = rf_basename
    out["rf_data_existe_em_disco"] = Path(rf_linux).exists() if rf_linux else None
    out["rf_data_sha256_declarado"] = rf_decl_sha
    out["rf_data_sha256_local_hashfile"] = rf_sha_local
    out["rf_data_sha256_bate_hashfile"] = (rf_decl_sha == rf_sha_local) if rf_decl_sha and rf_sha_local else None
    out["rf_data_sha256_metodo"] = "comparado contra HASHES_SHA256.txt local (nao recomputado; arquivo de ate 28 GB)"

    # idx_sha256_global das 3 particoes
    out["idx_sha256_global"] = {
        p: get(d, f"particoes.{p}.idx_sha256_global") for p in ("train", "val", "test")
    }

    # manifest
    rel_dados = f"dados/treinos_c1/run_{label}.json"
    rel_treinos = f"treinos/{label}/run_{label}.json"
    entrada = manifest_idx.get(rel_treinos) or manifest_idx.get(rel_dados)
    if entrada is None:
        out["manifest_status"] = "ausente"
    else:
        sha_manifest = entrada.get("sha256")
        sha_ref = out["run_json_sha256"] if entrada.get("artefato") == rel_treinos else (out["run_json_dados_sha256"] or out["run_json_sha256"])
        out["manifest_status"] = "presente" if sha_manifest == sha_ref else "sha_diverge"
        out["manifest_sha256"] = sha_manifest
        out["manifest_artefato"] = entrada.get("artefato")

    out["determinismo"] = {
        "cudnn_deterministic": get(d, "ambiente.cudnn_deterministic"),
        "cudnn_benchmark": get(d, "ambiente.cudnn_benchmark"),
    }

    seed_declarado = d.get("seed")
    split_seed_declarado = d.get("split_seed")
    seed_config = get(d, "config.seed")
    split_seed_config = get(d, "config.split_seed")
    out["seed_declarado_topo"] = seed_declarado
    out["seed_config"] = seed_config
    out["split_seed_declarado_topo"] = split_seed_declarado
    out["split_seed_config"] = split_seed_config
    out["seed_diverge_topo_vs_config"] = (seed_declarado != seed_config) or (split_seed_declarado != split_seed_config)

    bloqueia = (
        bool(faltando)
        or out.get("script_sha256_bate") is False
        or out.get("rf_data_sha256_bate_hashfile") is False
        or out["seed_diverge_topo_vs_config"]
        or out["manifest_status"] == "sha_diverge"
        or out["run_json_dados_identico"] is False
        or not out["checkpoint_existe"]
    )
    if bloqueia:
        out["veredito"] = "bloqueia"
    elif (out["manifest_status"] == "ausente" or out.get("rf_data_sha256_bate_hashfile") is None
          or out.get("rf_data_existe_em_disco") is False):
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
    ap.add_argument("--cidade", default=None, help="filtra por substring do run_label")
    ap.add_argument("--incluir-ja-verificadas", action="store_true",
                     help="por padrao pula labels ja verificadas em rodadas anteriores")
    ap.add_argument("--out", default=None, help="grava JSON de saida neste caminho")
    args = ap.parse_args()

    if args.hash:
        print(sha256_arquivo(Path(__file__)))
        return

    manifest_idx = carregar_manifest()
    hashes_locais = carregar_hashes_locais()
    corridas = descobrir_run_jsons_canonicos()
    if args.grupo:
        corridas = [c for c in corridas if c[0] == args.grupo]
    if args.cidade:
        corridas = [c for c in corridas if args.cidade in c[1]]
    if not args.incluir_ja_verificadas:
        corridas = [c for c in corridas if not ja_verificada(c[1])]

    resultados = []
    for grupo, label, caminho_treinos, caminho_dados in corridas:
        resultados.append(avaliar_corrida(grupo, label, caminho_treinos, caminho_dados, manifest_idx, hashes_locais))

    saida = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_arquivo(Path(__file__)),
        "hashes_locais_fonte": str(HASHES_LOCAIS),
        "hashes_locais_n_entradas": len(hashes_locais),
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
