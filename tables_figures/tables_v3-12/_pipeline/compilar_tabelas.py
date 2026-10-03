#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""Teste de fumaca LaTeX (nao e validacao cientifica): cada tabela de ../ compila num documento minimo (article, booktabs, array, amsmath,
macro \\celula do manuscrito, largura de texto 394,36 pt = a do MDPI) e o log e lido por Overfull/Underfull/erros. Saida: compilacao.json."""
import json, re, subprocess, sys
from pathlib import Path
AQUI = Path(__file__).resolve().parent
TAB = AQUI.parent
W = AQUI / "_compila"
W.mkdir(exist_ok=True)
DOC = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=600pt,paperheight=800pt,text={394.36pt,700pt}]{geometry}
\usepackage{booktabs,array,amsmath,amssymb}
\newcommand{\celula}[1]{\begin{tabular}[c]{@{}c@{}}#1\end{tabular}}
\newsavebox{\tabcaixa}
\newcommand{\tabajusta}[1]{\sbox{\tabcaixa}{#1}\ifdim\wd\tabcaixa>\linewidth\resizebox{\linewidth}{!}{\usebox{\tabcaixa}}\else\usebox{\tabcaixa}\fi}
\usepackage{graphicx}
\begin{document}
\noindent\begin{minipage}{\linewidth}
\input{%s}
\end{minipage}

\newsavebox{\bx}\sbox{\bx}{\input{%s}}\typeout{LARGURA_NATURAL=\the\wd\bx}
\end{document}
"""
# v3-12h: arquivos com ambiente table (T9: table[H] + caption + label) entram pelo invólucro \tabcorpo do manuscrito (main.tex linha 51), como em parts/*.tex
DOC_TAB = DOC.replace(r"\begin{document}", r"\providecommand{\tabcorpo}[1]{\begingroup\renewenvironment{table}[1][]{}{}\renewcommand{\caption}[1]{}\renewcommand{\label}[1]{}\input{#1}\endgroup}" + "\n" + r"\begin{document}", 1) \
    .replace(r"\input{%s}" + "\n" + r"\end{minipage}", r"\tabajusta{\tabcorpo{%s}}" + "\n" + r"\end{minipage}", 1) \
    .replace(r"\sbox{\bx}{\input{%s}}", r"\sbox{\bx}{\tabcorpo{%s}}", 1)
res = {}
ok_todas = True
for t in sorted(TAB.glob("T*_v3-12.tex")):
    nome = t.stem
    modelo = DOC_TAB if "\\begin{table}" in t.read_text(encoding="utf-8") else DOC
    (W / f"{nome}.tex").write_text(modelo % (t, t), encoding="utf-8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{nome}.tex"], cwd=W, capture_output=True, text=True, timeout=120)
    log = (W / f"{nome}.log").read_text(encoding="latin-1")
    over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)
    larg = re.findall(r"LARGURA_NATURAL=([\d.]+)pt", log)
    erros = re.findall(r"^! .*$", log, flags=re.M)
    ok = r.returncode == 0 and not erros
    ok_todas &= ok
    res[nome] = {"compilou": ok, "returncode": r.returncode, "erros": erros[:5], "largura_natural_pt": float(larg[0]) if larg else None, "largura_texto_pt": 394.36,
                 "overfull_hbox_pt": over, "pdf_gerado": (W / f"{nome}.pdf").exists()}
ANT = TAB.parent.parent / "redacao" / "tables_v3"
for t in sorted(ANT.glob("T*_v3.tex")):
    nome = "antiga_" + t.stem
    (W / f"{nome}.tex").write_text(DOC % (t, t), encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{nome}.tex"], cwd=W, capture_output=True, text=True, timeout=120)
    log = (W / f"{nome}.log").read_text(encoding="latin-1")
    larg = re.findall(r"LARGURA_NATURAL=([\d.]+)pt", log)
    res[nome] = {"so_comparacao_v3-11g": True, "largura_natural_pt": float(larg[0]) if larg else None}
json.dump(res, open(AQUI / "compilacao.json", "w"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if ok_todas else 1)
