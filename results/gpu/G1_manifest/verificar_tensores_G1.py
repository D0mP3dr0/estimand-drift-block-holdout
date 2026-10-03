"""D2(2) forum-eng-dados: recalcula sha256 (arquivo inteiro, blocos de 16 MiB) dos 4 tensores
_enriched_cftudo.pt usados no lote G1 e compara com (a) manifest v4 (grupo tensores_cftudo) e
(b) sha256_rf_data de TODOS os run JSON do lote que o trazem. Somente leitura.
Comando: venv s33_amb_virtual, script gpu/G1_manifest/verificar_tensores_G1.py (sem argumentos)."""
import hashlib, json, os, datetime, time
from pathlib import Path
BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
G1 = BASE / "_v3_2026-09-25/gpu/G1"
OUT = BASE / "_v3_2026-09-25/gpu/G1_manifest/verificacao_tensores_G1.json"
CELULAS = [("bauru", "Q1"), ("bauru", "Q3"), ("campinas", "Q1"), ("campinas", "Q3")]

man = {}
with open(BASE / "manifest_mathematics_v4.jsonl", encoding="utf-8") as f:
    for ln in f:
        if ln.strip():
            e = json.loads(ln)
            if e.get("grupo") == "tensores_cftudo":
                man[e["caminho"]] = e
runs = {}
for p in sorted(G1.glob("g1_*/run_*.json")):
    d = json.loads(p.read_text())
    ins = d.get("insumos")
    if ins and ins.get("rf_data_file"):
        runs.setdefault(ins["rf_data_file"], []).append((d["run_label"], ins.get("sha256_rf_data"), ins.get("metodo_sha_rf_data")))
res = []
for cid, q in CELULAS:
    cam = f"/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/transfer_dataset_{cid}_v19_{q}_enriched_cftudo.pt"
    st0 = os.stat(cam)
    t0 = time.time()
    h = hashlib.sha256()
    n = 0
    with open(cam, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
            n += len(b)
    st1 = os.stat(cam)
    obs = h.hexdigest()
    m = man.get(cam, {})
    rl = runs.get(cam, [])
    shas_run = sorted({r[1] for r in rl}, key=str)
    r = {"celula": f"{cid}_{q}", "caminho": cam, "sha256_observado": obs, "bytes_lidos": n,
         "bytes_stat": st1.st_size,
         "mtime_stat": datetime.datetime.fromtimestamp(st1.st_mtime, datetime.timezone.utc).isoformat(),
         "estavel_durante_leitura": (st0.st_size, st0.st_mtime_ns) == (st1.st_size, st1.st_mtime_ns),
         "manifest_v4_sha256": m.get("sha256"), "manifest_v4_tamanho_bytes": m.get("tamanho_bytes"),
         "manifest_v4_mtime": m.get("mtime"),
         "bate_manifest_v4": bool(m.get("sha256")) and obs == m.get("sha256"),
         "tamanho_bate_manifest_v4": n == m.get("tamanho_bytes"),
         "n_run_json_com_insumos": len(rl), "sha256_distintos_nos_run_json": shas_run,
         "metodos_sha_nos_run_json": sorted({str(x[2]) for x in rl}),
         "bate_todos_run_json": bool(rl) and shas_run == [obs],
         "tempo_leitura_s": round(time.time() - t0, 1)}
    res.append(r)
    print(json.dumps(r), flush=True)
    OUT.with_suffix(".parcial.json").write_text(json.dumps(res, indent=1))
final = {"gerado_em": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "comando": "venv s33_amb_virtual: python gpu/G1_manifest/verificar_tensores_G1.py",
         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "manifest_v4": str(BASE / "manifest_mathematics_v4.jsonl"),
         "nota": "run JSON do lote registram metodo_sha_rf_data=reuso_manifest_v4_mtime_e_tamanho_identicos (sha NAO recomputado no lote); esta leitura integral e a primeira verificacao de conteudo.",
         "resultados": res}
OUT.write_text(json.dumps(final, indent=1, ensure_ascii=False))
