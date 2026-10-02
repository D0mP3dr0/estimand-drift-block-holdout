#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""v3-11 (27/09/2026): confere o encolhimento das partes contra a cópia anterior.
Regras: (1) blocos formais (definition, proposition, lemma, corollary, theorem, remark, proof,
equation, align, gather) byte-idênticos, na mesma ordem; (2) todas as chaves \\cite preservadas no
conjunto das partes; (3) \\label e \\ref/\\eqref: nenhum label some; (4) números da prosa que
somem: listados para o chefe/coerência conferirem se estão numa tabela; (5) abstract idêntico.
Uso: v3_11_conferir_encolhimento.py <dir_antes> <dir_depois>"""
import re, sys, json, hashlib
from pathlib import Path
ENV = r"(definition|proposition|lemma|corollary|theorem|remark|proof|equation|align|gather)\*?"
def ler(d): return {p.name: p.read_text() for p in sorted(Path(d).glob("parte*.tex")) + sorted(Path(d).glob("apendices.tex"))}
def formais(t): return [m.group(0) for m in re.finditer(r"\\begin\{" + ENV + r"\}.*?\\end\{\1\*?\}", t, re.S)]
def cites(t): return sorted(k.strip() for g in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", t) for k in g.split(","))
def labels(t): return sorted(re.findall(r"\\label\{([^}]*)\}", t))
def sem_coment(t): return re.sub(r"(?<!\\)%.*", "", t)
def numeros(t):
    t = sem_coment(t)
    for m in re.finditer(r"\\begin\{" + ENV + r"\}.*?\\end\{\1\*?\}", t, re.S): t = t.replace(m.group(0), " ")
    return re.findall(r"(?<![\w.])[−-]?\d+(?:[.,{}]\d+)*(?:\.\d+)?", t)
A, B = ler(sys.argv[1]), ler(sys.argv[2])
res = {"arquivos": {}}
tA = "\n".join(A.values()); tB = "\n".join(B.values())
for n in A:
    fa, fb = formais(sem_coment(A[n])), formais(sem_coment(B.get(n, "")))
    res["arquivos"][n] = {"formais_antes": len(fa), "formais_depois": len(fb),
                          "formais_identicos": fa == fb,
                          "formais_alterados": [i for i, (x, y) in enumerate(zip(fa, fb)) if x != y][:10]}
res["cites_perdidas"] = sorted(set(cites(tA)) - set(cites(tB)))
res["labels_perdidos"] = sorted(set(labels(tA)) - set(labels(tB)))
ab = lambda t: (re.search(r"\\abstract\{.*?\}\n", t, re.S) or [None])[0] if isinstance(t, str) else None
res["abstract_identico"] = (re.search(r"\\abstract\{.*?\}\n", A.get("parteA.tex",""), re.S).group(0) == re.search(r"\\abstract\{.*?\}\n", B.get("parteA.tex",""), re.S).group(0))
from collections import Counter
ca, cb = Counter(numeros(tA)), Counter(numeros(tB))
res["numeros_que_sumiram_da_prosa"] = sorted((k for k in ca if cb[k] < ca[k]), key=str)
res["numeros_novos_na_prosa"] = sorted((k for k in cb if cb[k] > ca[k]), key=str)
print(json.dumps(res, indent=1, ensure_ascii=False))
