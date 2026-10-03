#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""Monta _v3_2026-09-25/redacao_v3-12/pacote_limpo/ a partir de redacao_v3-12/main_<versao>.tex (03/10/2026).
Uso: v3_12_montar_pacote_limpo.py <versao>   (ex.: v3-12k)
Copia SO o que o main referencia (tables_v3-12/*.tex por \\tabcorpo, figures_v3-12/* por \\includegraphics),
mais reference.bib, Definitions/ e os PDFs do suplementar (Table_S*.pdf). Limpa comentarios e marcadores de
processo com scripts/v3_limpar_fonte.py (limpar_tex/limpar_bib). Recria a pasta do zero (a anterior vai para
pacote_limpo_anterior/). Compila com o TeX Live do ecossistema e grava MANIFESTO.txt com sha256 de cada arquivo."""
import hashlib, re, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts")
from v3_limpar_fonte import limpar_tex, limpar_bib

ver = sys.argv[1]
RED = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/redacao_v3-12")
DST = RED / "pacote_limpo"
main = (RED / f"main_{ver}.tex").read_text()
tabs = sorted(set(re.findall(r"\\tabcorpo\{(tables_v3-12/[^}]+)\}", main)))
figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main)))
if DST.exists():
    ant = DST.with_name("pacote_limpo_anterior"); shutil.rmtree(ant, ignore_errors=True); DST.rename(ant)
DST.mkdir(parents=True)
rel = {}
t, rel["main.tex"] = limpar_tex(main); (DST / "main.tex").write_text(t)
t, rel["reference.bib"] = limpar_bib((RED / "reference.bib").read_text()); (DST / "reference.bib").write_text(t)
for tb in tabs:
    q = DST / tb; q.parent.mkdir(parents=True, exist_ok=True)
    t, rel[tb] = limpar_tex((RED / tb).read_text()); q.write_text(t)
for f in figs:
    src = RED / f
    if not src.exists():
        cand = sorted((RED / f).parent.glob(Path(f).name + ".*")); src = cand[0]; f = str(src.relative_to(RED))
    q = DST / f; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, q)
shutil.copytree(RED / "Definitions", DST / "Definitions")
for s in sorted((RED / "suplementar").glob("Table_S*.pdf")):
    shutil.copy2(s, DST / s.name)
tl = "/trabalho/ambientes/texlive-2025/bin/x86_64-linux"
env = {"PATH": tl + ":/usr/bin:/bin"}
for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], ["bibtex", "main"],
            ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
    r = subprocess.run(cmd, cwd=DST, env=env, capture_output=True, text=True)
    if r.returncode != 0 and cmd[0] == "pdflatex" and "-halt-on-error" in cmd:
        print("ERRO pdflatex", r.stdout[-1500:]); sys.exit(1)
log = (DST / "main.log").read_text(errors="ignore")
pags = re.search(r"Output written on main.pdf \((\d+) pages", log)
undef = len(re.findall(r"undefined", log, re.I))
for ext in ("aux", "log", "out", "blg", "toc"):
    p = DST / f"main.{ext}"
    if p.exists(): p.unlink()
linhas = []
for p in sorted(DST.rglob("*")):
    if p.is_file():
        linhas.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(DST)}")
(DST / "MANIFESTO.txt").write_text(f"# pacote_limpo de main_{ver}.tex; {pags.group(1) if pags else '?'} paginas; undefined no log: {undef}\n" + "\n".join(linhas) + "\n")
print(f"pacote_limpo: {len(linhas)} arquivos; {pags.group(1) if pags else '?'} paginas; undefined: {undef}; comentarios restantes no main: {sum(1 for l in (DST/'main.tex').read_text().splitlines() if l.lstrip().startswith('%'))}; % fato: {(DST/'main.tex').read_text().count('% fato')}")
