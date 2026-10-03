#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
# -*- coding: utf-8 -*-
"""
gpu/G1/train_v3_g10.py -- shim do teste G1 (decisao do chefe, opcao B).

Importa o wrapper v3 ORIGINAL (gpu/modelo_v3/train_gnn_v3.py ou train_mlp_v3.py)
SEM alterar nenhum arquivo de gpu/modelo_v3 nem os scripts congelados, e faz
chegar `--grid-km 10 --buffer-km 2` ao main() do congelado pela mesma tecnica
que os wrappers ja usam (monkeypatch em tempo de execucao): substitui
`v3_common.carregar_modulo_congelado` por uma versao que devolve o mesmo modulo
com `main` embrulhado para anexar esses dois argumentos. O pos-treino do
wrapper le `geometria.grid_km_usado` do run JSON (gravado pelo congelado), logo
as predicoes .npz saem na MESMA geometria do treino; o shim confere isso.

Uso (igual ao do wrapper, mais --modelo):
  train_v3_g10.py --modelo gnn|mlp <demais argumentos do wrapper>

sha256 dos originais (o shim RECUSA executar, rc=6, se algum divergir):
  train_gnn_v3.py            903b1ffaa19f7d6d732e958516a9e3dd98110cc3ac0f1aed704582e6e23a914f
  train_mlp_v3.py            86484b957e8507d8626a3e0ac386ce8c87fdca33f2548a814b02bee26c9085b8
  v3_common.py               57ffd38d869450a63d5263b101b440c02e19b738447c7013e5e9d7e48eff0268
  rf_decoder_v3.py           645f169755926b1525b7a8082da7e8dbfcb8d4bdbac1dfbf3aa5c45884fbfe0a
  train_gnn_c0_spatial.py    6f955629cde164f2843f454e1ebf6977292fd80647ef48ecd0df31e464c42445
  train_mlp_c0_spatial.py    4b75093f52e8b7fdec45c7f2b8c5bcd68c38093c420697927b49fa99a474b31c
(os mesmos que constam nos run JSON do A4: wrapper_sha256, decoder_sha256, script_sha256)

Ao fim de cada corrida grava, ao lado do run JSON (sem tocar nele),
`<evid-dir>/<run-label>/shim_g10_<run-label>.json` com os sha e os valores
efetivos (config.grid_km, config.buffer_km, geometria.grid_km_usado/
buffer_km_usado) e a conferencia de que o numero de nos do .npz == nos de teste
retidos apos o buffer do run JSON (rc=7 se g/b efetivos != 10/2 ou se divergir).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GRID_KM = 10.0
BUFFER_KM = 2.0

MODELO_V3_DIR = Path(__file__).resolve().parent.parent / "modelo_v3"
FROZEN_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/"
                  "EVIDENCIA_RESUBMISSAO/scripts")
SHA_ESPERADOS = {
    MODELO_V3_DIR / "train_gnn_v3.py": "903b1ffaa19f7d6d732e958516a9e3dd98110cc3ac0f1aed704582e6e23a914f",
    MODELO_V3_DIR / "train_mlp_v3.py": "86484b957e8507d8626a3e0ac386ce8c87fdca33f2548a814b02bee26c9085b8",
    MODELO_V3_DIR / "v3_common.py": "57ffd38d869450a63d5263b101b440c02e19b738447c7013e5e9d7e48eff0268",
    MODELO_V3_DIR / "rf_decoder_v3.py": "645f169755926b1525b7a8082da7e8dbfcb8d4bdbac1dfbf3aa5c45884fbfe0a",
    FROZEN_DIR / "train_gnn_c0_spatial.py": "6f955629cde164f2843f454e1ebf6977292fd80647ef48ecd0df31e464c42445",
    FROZEN_DIR / "train_mlp_c0_spatial.py": "4b75093f52e8b7fdec45c7f2b8c5bcd68c38093c420697927b49fa99a474b31c",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verificar_sha() -> dict:
    obs = {str(p): sha256(p) for p in SHA_ESPERADOS}
    ruins = {str(p): (obs[str(p)], esp) for p, esp in SHA_ESPERADOS.items() if obs[str(p)] != esp}
    if ruins:
        print(f"[shim_g10] RECUSADO: sha256 divergente dos registrados: {ruins}", flush=True)
        sys.exit(6)
    return obs


def main() -> int:
    sha_obs = verificar_sha()
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--modelo", choices=("gnn", "mlp"), required=True)
    ap.add_argument("--evid-dir", required=True)
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--grid-km", default=None)
    ap.add_argument("--buffer-km", default=None)
    known, resto = ap.parse_known_args()
    if known.grid_km is not None or known.buffer_km is not None:
        print("[shim_g10] RECUSADO: g e b sao fixos (10/2) neste shim; nao passe --grid-km/--buffer-km.", flush=True)
        return 6
    # argv para o wrapper: tudo menos --modelo, (mantendo --evid-dir e --run-label)
    argv_wrapper = ["--evid-dir", known.evid_dir, "--run-label", known.run_label] + resto

    sys.path.insert(0, str(MODELO_V3_DIR))
    import v3_common as v3  # o mesmo objeto que o wrapper importa como `v3`

    orig_carregar = v3.carregar_modulo_congelado

    def carregar_g10(script_path, nome_modulo):
        mod = orig_carregar(script_path, nome_modulo)
        main_orig = mod.main

        def main_g10(argv=None):
            argv = list(argv or [])
            if "--grid-km" in argv or "--buffer-km" in argv:
                raise RuntimeError("wrapper ja passou --grid-km/--buffer-km; shim nao deve duplicar")
            argv += ["--grid-km", str(GRID_KM), "--buffer-km", str(BUFFER_KM)]
            print(f"[shim_g10] mod.main recebe + --grid-km {GRID_KM} --buffer-km {BUFFER_KM}", flush=True)
            return main_orig(argv)

        mod.main = main_g10
        return mod

    v3.carregar_modulo_congelado = carregar_g10

    if known.modelo == "gnn":
        import train_gnn_v3 as wrapper
    else:
        import train_mlp_v3 as wrapper
    sys.argv = [wrapper.__file__] + argv_wrapper
    rc = wrapper.main()
    if rc != 0:
        return rc

    # conferencia pos-corrida (nao edita o run JSON)
    import numpy as np
    pasta = Path(known.evid_dir) / known.run_label
    run_json = pasta / f"run_{known.run_label}.json"
    npz = pasta / f"predicoes_{known.run_label}.npz"
    with open(run_json, "r", encoding="utf-8") as f:
        rec = json.load(f)
    cfg, geo = rec.get("config", {}), rec.get("geometria", {})
    n_teste_run = (rec.get("split", {}).get("n_nos_apos_buffer", {}) or {}).get("test")
    n_npz = int(np.load(npz)["idx_global"].shape[0]) if npz.exists() else None
    ok_gb = (cfg.get("grid_km") == GRID_KM and cfg.get("buffer_km") == BUFFER_KM
             and geo.get("grid_km_usado") == GRID_KM and geo.get("buffer_km_usado") == BUFFER_KM
             and (rec.get("split", {}).get("grid_km") == GRID_KM) and (rec.get("split", {}).get("buffer_km") == BUFFER_KM))
    ok_npz = (n_npz is not None and n_npz == n_teste_run)
    side = {"artefato": "shim_g10", "modelo": known.modelo, "run_label": known.run_label,
            "grid_km_pedido": GRID_KM, "buffer_km_pedido": BUFFER_KM,
            "efetivo": {"config.grid_km": cfg.get("grid_km"), "config.buffer_km": cfg.get("buffer_km"),
                        "geometria.grid_km_usado": geo.get("grid_km_usado"), "geometria.buffer_km_usado": geo.get("buffer_km_usado"),
                        "split.grid_km": rec.get("split", {}).get("grid_km"), "split.buffer_km": rec.get("split", {}).get("buffer_km")},
            "g_b_efetivos_conferem": bool(ok_gb),
            "n_nos_teste_retidos_run_json": n_teste_run, "n_nos_npz": n_npz, "npz_igual_teste_retido": bool(ok_npz),
            "sha256_observados": sha_obs}
    with open(pasta / f"shim_g10_{known.run_label}.json", "w", encoding="utf-8") as f:
        json.dump(side, f, indent=1, ensure_ascii=False)
    print(f"[shim_g10] g/b efetivos conferem={ok_gb}; npz({n_npz}) == teste retido do run JSON({n_teste_run}): {ok_npz}", flush=True)
    return 0 if (ok_gb and ok_npz) else 7


if __name__ == "__main__":
    sys.exit(main())
