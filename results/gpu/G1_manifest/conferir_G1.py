"""D2(3)(4) forum-eng-dados: reconfere sha das proveniencias do lote G1 e conta o lote. Somente leitura.
Grava gpu/G1_manifest/conferencia_G1.json. Comando: venv s33_amb_virtual, este script, sem argumentos."""
import hashlib, json, collections, datetime
from pathlib import Path
BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
G1 = BASE / "_v3_2026-09-25/gpu/G1"
OUT = BASE / "_v3_2026-09-25/gpu/G1_manifest/conferencia_G1.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


r = {"gerado_em_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "script_sha256": sha(Path(__file__))}
# (3) proveniencias
prov = {}
for nome in ["proveniencia_scripts_G1.json", "proveniencia_scripts_G1_retomada.json"]:
    d = json.loads((G1 / nome).read_text())
    L = []
    for e in d["arquivos"]:
        p = BASE / e["caminho"]
        obs = sha(p) if p.exists() else None
        L.append({"caminho": e["caminho"], "sha256_registrado": e["sha256"], "sha256_observado": obs,
                  "bate_sha": obs == e["sha256"], "bytes_registrado": e["bytes"],
                  "bytes_observado": p.stat().st_size if p.exists() else None,
                  "bate_bytes": p.exists() and p.stat().st_size == e["bytes"]})
    prov[nome] = {"todos_batem_sha": all(x["bate_sha"] for x in L), "arquivos": L}
r["proveniencias"] = prov
# (4) contagens
plano = json.loads((G1 / "plano_G1.json").read_text())
status = json.loads((G1 / "lote_G1_status.json").read_text())
nt = json.loads((G1 / "nao_treinaveis_G1_retomada.json").read_text())["nao_treinaveis"]
nt_lab = {x["run_label"] for x in nt}
st_lab = {c["run_label"]: c for c in status["corridas"]}
C = collections.defaultdict(lambda: collections.Counter())
sem_par, sem_valido_lab, no_plano_sem_run = [], [], []
for c in plano["corridas"]:
    lab = c["run_label"]
    d = G1 / lab
    run = d / f"run_{lab}.json"
    npz = d / f"predicoes_{lab}.npz"
    shim = d / f"shim_g10_{lab}.json"
    tl = d / "training_log.csv"
    ck = d / "checkpoints/checkpoint_best.pt"
    log = G1 / f"log_{lab}.txt"
    b = f"bloco{c['bloco']}"
    C[b]["planejadas"] += 1
    C[b]["run_json"] += run.exists()
    C[b]["npz"] += npz.exists()
    C[b]["shim"] += shim.exists()
    C[b]["training_log"] += tl.exists()
    C[b]["checkpoint"] += ck.exists()
    C[b]["log"] += log.exists()
    completa = run.exists() and npz.exists() and shim.exists()
    C[b]["completas_run_npz_shim"] += completa
    if not run.exists():
        no_plano_sem_run.append(lab)
    if not completa:
        sem_par.append({"run_label": lab, "bloco": c["bloco"], "tem_run": run.exists(), "tem_npz": npz.exists(),
                        "tem_shim": shim.exists(), "nao_treinavel_registrado": lab in nt_lab,
                        "rc_no_status": st_lab.get(lab, {}).get("rc", "ausente_no_status"),
                        "status_run_json": json.loads(run.read_text()).get("status") if run.exists() else None,
                        "chaves_run_json": len(json.loads(run.read_text())) if run.exists() else None,
                        "guarda_particao_degenerada": json.loads(run.read_text()).get("guarda_particao_degenerada") if run.exists() else None})
r["corridas_por_bloco"] = {k: dict(v) for k, v in sorted(C.items())}
r["corridas_planejadas_total"] = len(plano["corridas"]); r["n_plano_declarado"] = plano["n"]
r["corridas_no_status"] = len(status["corridas"]); r["rc_no_status"] = dict(collections.Counter(str(c["rc"]) for c in status["corridas"]))
r["plano_fora_do_status"] = sorted(set(c["run_label"] for c in plano["corridas"]) - set(st_lab))
r["status_fora_do_plano"] = sorted(set(st_lab) - set(c["run_label"] for c in plano["corridas"]))
r["corridas_sem_par_completo"] = sem_par
r["nao_treinaveis_registrados"] = sorted(nt_lab)
r["n_nao_treinaveis_corridas"] = len(nt_lab)
# sem_validos / nao_treinaveis por sorteio, conforme agregado v8 (bloco 3)
ag = json.loads((G1 / "agregado_G1_v8_bloco3.json").read_text())
for k in ["contagem_sorteios_sem_validos_por_celula", "contagem_sorteios_nao_treinaveis_por_celula", "contagem_sorteios_usaveis_por_celula"]:
    r["agregado_v8_" + k] = ag[k]
for f in ["agregado_G1_v5_bloco1.json", "agregado_G1_v5_bloco2.json"]:
    r[f + "_sem_validos"] = json.loads((G1 / f).read_text())["contagem_sorteios_sem_validos_por_celula"]
# arquivos
arqs = [p for p in G1.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
r["arquivos_lote_sem_pycache"] = len(arqs)
r["bytes_lote_sem_pycache"] = sum(p.stat().st_size for p in arqs)
r["arquivos_pycache_excluidos"] = [str(p.relative_to(G1)) for p in G1.rglob("*") if p.is_file() and "__pycache__" in p.parts]
cats = collections.Counter()
for p in arqs:
    n = p.name
    cats[("run_json" if n.startswith("run_") else "npz" if n.endswith(".npz") else "shim" if n.startswith("shim_") else
          "training_log.csv" if n == "training_log.csv" else "checkpoint" if n.endswith(".pt") else "log_corrida" if n.startswith("log_g1_") else "outro_lote")] += 1
r["arquivos_por_categoria"] = dict(cats)
OUT.write_text(json.dumps(r, indent=1, ensure_ascii=False))
print(json.dumps(r, indent=1, ensure_ascii=False))
