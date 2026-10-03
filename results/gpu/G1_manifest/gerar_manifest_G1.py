"""D2(1) forum-eng-dados: gera gpu/G1/MANIFEST_G1.jsonl (somente leitura sobre o lote).
Primeira linha = cabecalho (data, comando, sha256 deste script); demais = um artefato por linha
{caminho relativo a MDPI_Mathematics, categoria, sha256, bytes, mtime_utc}.
Comando: venv s33_amb_virtual, script gpu/G1_manifest/gerar_manifest_G1.py (sem argumentos).
Falha se MANIFEST_G1.jsonl ja existir (nao sobrescreve). __pycache__ fora (nao e artefato do lote)."""
import hashlib, json, sys, datetime
from pathlib import Path

BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
V3 = BASE / "_v3_2026-09-25"
G1 = V3 / "gpu/G1"
OUT = G1 / "MANIFEST_G1.jsonl"
if OUT.exists():
    sys.exit("MANIFEST_G1.jsonl ja existe; nao sobrescrevo")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


itens = []  # (Path, categoria)
for p in sorted(G1.iterdir()):
    if p.is_file():
        n = p.name
        if n.startswith("log_g1_"):
            cat = "log_corrida"
        elif n.endswith((".py",)):
            cat = "script_lote"
        elif n.startswith("agregado_"):
            cat = "agregado"
        else:
            cat = "controle_lote"
        itens.append((p, cat))
for d in sorted(G1.glob("g1_*")):
    for p in sorted(d.rglob("*")):
        if p.is_file():
            n = p.name
            cat = ("run_json" if n.startswith("run_") else "predicoes_npz" if n.endswith(".npz")
                   else "shim_sidecar" if n.startswith("shim_g10_") else "training_log_csv" if n == "training_log.csv"
                   else "checkpoint" if n.endswith(".pt") else "outro")
            itens.append((p, cat))
for i in range(1, 9):
    itens.append((BASE / f"scripts/v3_G1_agregar_v{i}.py" if i > 1 else BASE / "scripts/v3_G1_agregar.py", "script_agregador"))
for n in ["criterio_G1_modelos_g10.json"] + [f"criterio_G1_adendo{i}.json" for i in range(1, 5)]:
    itens.append((V3 / "criterios" / n, "criterio"))

linhas = []
for p, cat in itens:
    st = p.stat()
    linhas.append({"caminho": str(p.relative_to(BASE)), "categoria": cat, "sha256": sha(p), "bytes": st.st_size,
                   "mtime_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat()})
cab = {"_cabecalho": True, "gerado_em_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "comando": "venv s33_amb_virtual: python _v3_2026-09-25/gpu/G1_manifest/gerar_manifest_G1.py",
       "gerador": "_v3_2026-09-25/gpu/G1_manifest/gerar_manifest_G1.py",
       "gerador_sha256": sha(Path(__file__)), "raiz_relativa": str(BASE),
       "n_artefatos": len(linhas), "bytes_total": sum(l["bytes"] for l in linhas),
       "nota": "nota sobre v1 do agregador: scripts/v3_G1_agregar.py e o v1 (original); __pycache__ excluido; os 4 tensores _cftudo.pt estao em verificacao_tensores_G1.json"}
with open(OUT, "x", encoding="utf-8") as f:
    f.write(json.dumps(cab, ensure_ascii=False) + "\n")
    for l in linhas:
        f.write(json.dumps(l, ensure_ascii=False) + "\n")
print(cab)
