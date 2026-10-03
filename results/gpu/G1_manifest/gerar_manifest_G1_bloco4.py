"""D2(1) forum-eng-dados: gera gpu/G1/MANIFEST_G1_bloco4.jsonl (copia declarada de gerar_manifest_G1.py com a lista de alvos trocada para o bloco 4) (somente leitura sobre o lote).
Primeira linha = cabecalho (data, comando, sha256 deste script); demais = um artefato por linha
{caminho relativo a MDPI_Mathematics, categoria, sha256, bytes, mtime_utc}.
Comando: venv s33_amb_virtual, script gpu/G1_manifest/gerar_manifest_G1_bloco4.py (sem argumentos).
Falha se MANIFEST_G1.jsonl ja existir (nao sobrescreve). __pycache__ fora (nao e artefato do lote)."""
import hashlib, json, sys, datetime
from pathlib import Path

BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
V3 = BASE / "_v3_2026-09-25"
G1 = V3 / "gpu/G1"
OUT = G1 / "MANIFEST_G1_bloco4.jsonl"
if OUT.exists():
    sys.exit("MANIFEST_G1_bloco4.jsonl ja existe; nao sobrescrevo")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


SS = ["739191", "29675", "676375", "387379", "259492"]
SEEDS = [43, 44]
NOMES = [f"g1_{m}_campinas_Q1_ss{ss}_s{s}" for m in ("gnn", "mlp") for ss in SS for s in SEEDS]
assert len(NOMES) == 20
# ausentes declarados: pedidos no escopo mas inexistentes em disco em 03/10; registrados no cabecalho, nao inventados
ausentes = []
itens = []  # (Path, categoria)
for nome in NOMES:
    d = G1 / nome
    if not d.is_dir():
        sys.exit(f"pasta ausente: {d}")
    lg = G1 / f"log_{nome}.txt"
    if not lg.is_file():
        sys.exit(f"log ausente: {lg}")
    itens.append((lg, "log_corrida"))
    for p in sorted(d.rglob("*")):
        if p.is_file():
            n = p.name
            cat = ("run_json" if n.startswith("run_") else "predicoes_npz" if n.endswith(".npz")
                   else "shim_sidecar" if n.startswith("shim_g10_") else "training_log_csv" if n == "training_log.csv"
                   else "checkpoint" if n.endswith(".pt") else "outro")
            itens.append((p, cat))
for n, cat in [("lote_G1_bloco4_status.json", "controle_lote"), ("lote_G1_bloco4_driver.log", "controle_lote"),
               ("plano_G1_bloco4.json", "controle_lote"), ("nao_treinaveis_G1_bloco4.json", "controle_lote"),
               ("proveniencia_scripts_G1_bloco4.json", "controle_lote"), ("rodar_lote_G1_bloco4.py", "script_lote"),
               ("agregado_G1_v10_bloco4.json", "agregado")]:
    p = G1 / n
    (itens if p.is_file() else ausentes).append((p, cat) if p.is_file() else str(p.relative_to(BASE)))
for n in ["v3_G1_agregar_v9.py", "v3_G1_agregar_v10.py"]:
    itens.append((BASE / "scripts" / n, "script_agregador"))
for n in ["diff_v3_G1_agregar_v8_para_v9.patch", "diff_v3_G1_agregar_v9_para_v10.patch"]:
    p = BASE / "scripts" / n
    if p.is_file():
        itens.append((p, "diff_agregador"))
    else:
        ausentes.append(str(p.relative_to(BASE)))
itens.append((V3 / "criterios/criterio_G1b_campinas_bloco_cruzado.json", "criterio"))
for p in sorted((V3 / "gpu/G1_votos_bloco4").glob("*")):
    if p.is_file():
        itens.append((p, "votos_bloco4"))
for p, _ in itens:
    if not p.is_file():
        sys.exit(f"alvo ausente: {p}")

linhas = []
for p, cat in itens:
    st = p.stat()
    linhas.append({"caminho": str(p.relative_to(BASE)), "categoria": cat, "sha256": sha(p), "bytes": st.st_size,
                   "mtime_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat()})
cab = {"_cabecalho": True, "gerado_em_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "comando": "venv s33_amb_virtual: python _v3_2026-09-25/gpu/G1_manifest/gerar_manifest_G1_bloco4.py",
       "gerador": "_v3_2026-09-25/gpu/G1_manifest/gerar_manifest_G1_bloco4.py",
       "gerador_sha256": sha(Path(__file__)), "raiz_relativa": str(BASE),
       "n_artefatos": len(linhas), "bytes_total": sum(l["bytes"] for l in linhas),
       "ausentes_declarados": ausentes, "nota": "bloco 4 (20 corridas Campinas Q1, seeds 43/44 x 5 sorteios); __pycache__ excluido; sha dos tensores: ver verificacao_tensores_G1.json; manifest do lote anterior em MANIFEST_G1.jsonl"}
with open(OUT, "x", encoding="utf-8") as f:
    f.write(json.dumps(cab, ensure_ascii=False) + "\n")
    for l in linhas:
        f.write(json.dumps(l, ensure_ascii=False) + "\n")
print(cab)
