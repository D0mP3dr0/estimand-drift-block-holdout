#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
# -*- coding: utf-8 -*-
"""v3-12: gera POR SCRIPT o adendo da folha de fatos da v3-12 (roadmap
_propostas_2026-10-01/ROADMAP_v3-12_major_revision.md, secao 6, passo 3; "va" do dono de 01/10/2026 17:12).

Regras do script
  - Nenhum numero digitado a mao: todo valor e LIDO de um JSON/CSV de artefato (ou derivado por
    conta simples sobre valores lidos, e a derivacao fica declarada na linha e no JSON espelho).
  - Cada linha traz o valor para o texto (arredondado SO aqui), o valor cheio ao lado, o caminho do
    artefato e do campo, e o estado do bloco (copiado do protocolo).
  - Arredondamento: Decimal(repr(float)) com ROUND_HALF_UP; virgula decimal; nada de arredondar duas vezes.
  - O script ABORTA (exit 2, nada gravado) se um artefato ou campo esperado faltar, se uma
    conferencia interna falhar ou se um valor lido for nulo onde se espera numero.
  - Grava o sha256 de cada artefato lido no JSON espelho.
  - Nao toca na folha existente (redacao/FOLHA_DE_FATOS_v3_consolidada.md).

Saidas
  redacao/FOLHA_DE_FATOS_v3-12_adendo.md
  redacao/FOLHA_DE_FATOS_v3-12_adendo.json   (espelho)

Execucao: /trabalho/ambientes/s33_amb_virtual/.venv/bin/python scripts/v3_12_gerar_folha_fatos.py
Somente CPU, leitura de arquivos pequenos (o maior e o parcial fase2 de 2,2 MB, usado so na conferencia
da recontagem de b = 2).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

RAIZ = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
SAIDA_MD = RAIZ / "redacao" / "FOLHA_DE_FATOS_v3-12_adendo.md"
SAIDA_JSON = RAIZ / "redacao" / "FOLHA_DE_FATOS_v3-12_adendo.json"
SCRIPT = Path(__file__).resolve()

CELULAS = ["bauru_Q1", "bauru_Q2", "bauru_Q3", "bauru_Q4",
           "campinas_Q1", "campinas_Q2", "campinas_Q3", "campinas_Q4",
           "lins_Q1", "lins_Q2", "lins_Q3", "lins_Q4",
           "sorocaba_Q1", "sorocaba_Q2", "sorocaba_Q3", "sorocaba_Q4"]
Q1 = ["bauru_Q1", "campinas_Q1", "lins_Q1", "sorocaba_Q1"]
PRED_ROT = {"constante": "constante", "fspl_calibrado_b": "FSPL (b)"}
POP_ROT = {"validos": "nós válidos", "todos": "todos os nós"}


# ----------------------------------------------------------------------------- utilidades
def die(msg: str):
    sys.stderr.write("ABORTADO: " + msg + "\n")
    sys.exit(2)


def rotulo_celula(c: str) -> str:
    cid, q = c.split("_")
    return cid.capitalize() + " " + q


class Artefato:
    def __init__(self, rel: str, kind: str = "json"):
        p = Path(rel)
        self.path = p if p.is_absolute() else RAIZ / rel
        try:
            self.rel = str(self.path.relative_to(RAIZ))
        except ValueError:
            self.rel = str(self.path)
        if not self.path.is_file():
            die(f"artefato ausente: {self.path}")
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.bytes = len(raw)
        if kind == "json":
            try:
                self.data = json.loads(raw.decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                die(f"JSON ilegivel {self.path}: {e}")
        elif kind == "csv":
            self.data = list(csv.DictReader(raw.decode("utf-8").splitlines()))
        elif kind == "bin":
            self.data = None
        else:
            die("kind desconhecido " + kind)

    def g(self, *keys):
        o = self.data
        trilha = []
        for k in keys:
            trilha.append(str(k))
            try:
                o = o[k]
            except (KeyError, IndexError, TypeError):
                die(f"campo ausente em {self.rel}: {'.'.join(trilha)}")
        if o is None:
            die(f"campo nulo em {self.rel}: {'.'.join(trilha)}")
        return o

    def g_nulo_ok(self, *keys):
        o = self.data
        for k in keys:
            try:
                o = o[k]
            except (KeyError, IndexError, TypeError):
                die(f"campo ausente em {self.rel}: {'.'.join(map(str, keys))}")
        return o

    def campo(self, *keys) -> str:
        return f"{self.rel}:{'.'.join(map(str, keys))}"


ARTEFATOS: dict[str, Artefato] = {}


def A(rel: str, kind: str = "json") -> Artefato:
    a = Artefato(rel, kind)
    ARTEFATOS[a.rel] = a
    return a


def _dec(v) -> Decimal:
    return Decimal(repr(float(v)))


def q(v, nd: int) -> str:
    d = _dec(v).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    s = format(d, "f")
    if d == 0 and s.startswith("-"):
        s = s[1:]
    return s.replace(".", ",")


def sig(v, n: int) -> str:
    if v == 0:
        return "0"
    d = _dec(v)
    nd = n - 1 - d.adjusted()
    return q(v, max(nd, 0))


def fmt(v, kind: str = "raw", unit: str = "") -> str:
    if kind == "raw":
        if isinstance(v, bool):
            s = "verdadeiro" if v else "falso"
        elif v is None:
            s = "nulo"
        elif isinstance(v, (list, dict)):
            s = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        else:
            s = str(v)
    elif kind == "int":
        s = str(int(_dec(v).quantize(Decimal(1), rounding=ROUND_HALF_UP)))
    elif kind.startswith("f"):
        s = q(v, int(kind[1:]))
    elif kind.startswith("pct"):
        s = q(float(_dec(v) * 100), int(kind[3:]))
        unit = unit or " %"
    elif kind == "sig3":
        s = sig(v, 3)
    else:
        die("formato desconhecido " + kind)
    return s + unit


class Item:
    def __init__(self, rotulo, cheio, texto, campo, derivado=None):
        self.rotulo, self.cheio, self.texto, self.campo, self.derivado = rotulo, cheio, texto, campo, derivado

    def asdict(self):
        d = {"rotulo": self.rotulo, "valor_cheio": self.cheio, "valor_texto": self.texto, "campo": self.campo}
        if self.derivado:
            d["derivado"] = self.derivado
        return d


def F(art: Artefato, keys, rot, kind="raw", unit="", texto=None, nulo_ok=False) -> Item:
    """Item copiado de um campo do artefato."""
    keys = tuple(keys)
    v = art.g_nulo_ok(*keys) if nulo_ok else art.g(*keys)
    if not nulo_ok and isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        die(f"valor nao finito em {art.campo(*keys)}")
    t = texto if texto is not None else fmt(v, kind, unit)
    return Item(rot, v, t, art.campo(*keys))


def D(rot, v, kind, unit, derivacao, campo, texto=None) -> Item:
    """Item derivado por conta simples sobre valores lidos (derivacao declarada)."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        die(f"derivado nao finito: {rot}")
    return Item(rot, v, texto if texto is not None else fmt(v, kind, unit), campo, derivacao)


LINHAS: list[dict] = []
CONFERENCIAS: list[dict] = []
BLOCOS: dict[str, dict] = {}


def conf(desc: str, ok: bool, detalhe=None):
    CONFERENCIAS.append({"descricao": desc, "passou": bool(ok), "detalhe": detalhe})
    if not ok:
        die(f"conferencia falhou: {desc} :: {detalhe}")


def proximo(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def bloco(letra, titulo, estado, nota_bloco=""):
    BLOCOS[letra] = {"titulo": titulo, "estado": estado, "nota": nota_bloco, "linhas": 0}


def linha(letra, id_, afirmacao, itens, estado=None, nota=""):
    est = estado if estado is not None else BLOCOS[letra]["estado"]
    LINHAS.append({"bloco": letra, "id": id_, "afirmacao": afirmacao, "itens": itens, "estado": est, "nota": nota})
    BLOCOS[letra]["linhas"] += 1


def argextremo(d: dict, cmp):
    """Retorna (valor, [celulas]) do minimo ou maximo (empates incluidos)."""
    v = cmp(d.values())
    return v, [k for k, x in d.items() if x == v]


# ============================================================================= BLOCO T
def bloco_T():
    bloco("T", "T — deriva do erro com o sorteio × células",
          "CITÁVEL (recálculo independente V1 + coerência + físico-matemático)",
          "Fonte: V1 (saida_V1.json) e T3 (T3_deriva_erro_v3.csv). dp = desvio-padrão, conforme o artefato; "
          "população = nós válidos do teste; sorteios sem nó válido excluídos (contagens em V1.sorteios_total e V1.sorteios_com_no_valido).")
    V1 = A("redacao/_pareceres_2026-10-01_t25/verificacao_V1_deriva/saida_V1.json")
    T3 = A("redacao/tables_v3/T3_deriva_erro_v3.csv", "csv")

    # --- conferencia V1 x T3
    rows = T3.data
    ncel = [r["celula"] for r in rows]
    conf("T3 tem as 16 células na ordem esperada", ncel == CELULAS, ncel)
    s_simples = [float(r["dp_simples"]) for r in rows]
    s_pond = [float(r["dp_ponderado"]) for r in rows]
    n_sort = [int(r["n_sorteios"]) for r in rows]
    conf("T3: média dos dp_simples = V1.dp_intra_media_simples",
         proximo(sum(s_simples) / 16, V1.g("dp_intra_media_simples"), 1e-12),
         [sum(s_simples) / 16, V1.g("dp_intra_media_simples")])
    conf("T3: média dos dp_ponderado = V1.dp_intra_ponderado_media",
         proximo(sum(s_pond) / 16, V1.g("dp_intra_ponderado_media"), 1e-12),
         [sum(s_pond) / 16, V1.g("dp_intra_ponderado_media")])
    conf("T3: n_sorteios por célula = V1.n_por_celula e soma = V1.sorteios_com_no_valido",
         n_sort == V1.g("n_por_celula") and sum(n_sort) == V1.g("sorteios_com_no_valido"),
         [sum(n_sort), V1.g("sorteios_com_no_valido")])

    linha("T", "T.dp_sorteio",
          "Desvio-padrão entre sorteios do MAE do constante nos nós válidos, média das 16 células (simples e ponderada)",
          [F(V1, ["dp_intra_media_simples"], "média simples", "f2", " dB"),
           F(V1, ["dp_intra_ponderado_media"], "média ponderada", "f2", " dB")])
    linha("T", "T.dp_entre_medias",
          "Desvio-padrão entre as médias das 16 células do MAE do constante nos nós válidos",
          [F(V1, ["dp_entre_medias"], "dp entre médias", "f2", " dB")])
    linha("T", "T.razoes",
          "Razão entre o dp entre sorteios e o dp entre médias das células (média simples; média ponderada)",
          [F(V1, ["razao"], "razão (simples)", "f1"),
           F(V1, ["razao_ponderada_vs_dp_entre_medias"], "razão (ponderada)", "f1")])
    linha("T", "T.razao_corrigida",
          "Razão e componentes depois de descontar o ruído das médias",
          [F(V1, ["razao_corrigida"], "razão corrigida", "f1"),
           F(V1, ["razao_corrigida_poolado"], "razão corrigida (dp poolado)", "f1"),
           F(V1, ["dp_entre_corrigido"], "dp entre médias corrigido", "f2", " dB"),
           F(V1, ["dp_intra_poolado"], "dp intra poolado", "f2", " dB"),
           F(V1, ["var_ruido_medias"], "variância do ruído das médias", "f2", " dB²")])
    linha("T", "T.razao_ic",
          "Intervalo de 95 % por bootstrap da razão",
          [F(V1, ["razao_boot_IC95", 0], "limite inferior", "f1"),
           F(V1, ["razao_boot_IC95", 1], "limite superior", "f1")])
    linha("T", "T.anova",
          "Componentes de variância (ANOVA de um fator, células): dp de célula, dp residual de sorteio e correlação intraclasse",
          [F(V1, ["anova", "dp_celula"], "dp de célula", "f2", " dB"),
           F(V1, ["anova", "dp_sorteio_residual"], "dp residual (sorteio)", "f2", " dB"),
           F(V1, ["anova", "ICC"], "correlação intraclasse", "f2")])
    nc = V1.g("A2", "n_celulas")
    linha("T", "T.sinal",
          "Número de células em que o sinal de delta_val e de delta_te muda entre os sorteios",
          [F(V1, ["A2", "sinal_muda_val"], "delta_val", "int",
             texto=f"{V1.g('A2', 'sinal_muda_val')} de {nc}"),
           F(V1, ["A2", "sinal_muda_te"], "delta_te", "int",
             texto=f"{V1.g('A2', 'sinal_muda_te')} de {nc}"),
           F(V1, ["A2", "n_celulas"], "n de células", "int"),
           F(V1, ["A2", "n_seeds_por_celula", 0], "sorteios por célula", "int")])
    for campo, rot in (("frac_val_mais_pobre", "validação"), ("frac_te_mais_rico", "teste")):
        d = V1.g("A2", campo)
        conf(f"V1.A2.{campo}: 16 células", sorted(d) == sorted(CELULAS), sorted(d))
        vmin, cmin = argextremo(d, min)
        vmax, cmax = argextremo(d, max)
        mm = V1.g("A2", campo + "_min_max")
        conf(f"V1.A2.{campo}_min_max coincide com min/max do dicionário", proximo(mm[0], vmin) and proximo(mm[1], vmax), [mm, vmin, vmax])
        linha("T", f"T.faixa_{campo}",
              f"Faixa entre células da fração de sorteios do campo {campo} ({rot}), com a célula de cada extremo",
              [F(V1, ["A2", campo + "_min_max", 0], "mínimo", "f2",
                 texto=f"{q(mm[0], 2)} ({', '.join(cmin)})"),
               F(V1, ["A2", campo + "_min_max", 1], "máximo", "f2",
                 texto=f"{q(mm[1], 2)} ({', '.join(cmax)})")],
              nota="células nos extremos deduzidas do dicionário " + f"A2.{campo} (empates listados)")
    # T3: faixa dos dp por célula
    for col, rot in (("dp_simples", "dp simples entre sorteios"), ("dp_ponderado", "dp ponderado entre sorteios")):
        d = {r["celula"]: float(r[col]) for r in rows}
        vmin, cmin = argextremo(d, min)
        vmax, cmax = argextremo(d, max)
        linha("T", f"T.t3_{col}",
              f"Faixa entre as 16 células do {rot} do MAE do constante nos nós válidos (tabela T3)",
              [D("mínimo", vmin, "f2", " dB", f"mínimo da coluna {col} do CSV", T3.campo(col), f"{q(vmin, 2)} dB ({', '.join(cmin)})"),
               D("máximo", vmax, "f2", " dB", f"máximo da coluna {col} do CSV", T3.campo(col), f"{q(vmax, 2)} dB ({', '.join(cmax)})")])


# ============================================================================= BLOCO U
def norm_b2(s):
    return dict(status=s["status"], nv=s["n_pop_teste"]["validos"], c=s["mae_constante_teste"],
                a=s["mae_modelo_a_contaminado_teste"]["fspl"], b=s["mae_modelo_b_validos_teste"]["fspl"])


def norm_b0(s):
    return dict(status=s["status"], nv=s["n_validos_teste"], c=s["mae_constante"],
                a=s["mae_fspl_a_contaminado"], b=s["mae_fspl_b_validos"])


def recontar(lista_norm):
    """Recontagem SOBRE OS SORTEIOS COM NO VALIDO no teste (n_validos > 0)."""
    r = dict(n_sorteios=0, n_com_valido=0, const_menor_todos_a=0, const_menor_todos_b=0,
             fspl_menor_validos_a=0, fspl_menor_validos_b=0)
    for s in lista_norm:
        r["n_sorteios"] += 1
        if s["status"] != "ok":
            die("sorteio com status != ok na recontagem")
        if s["nv"] <= 0:
            continue
        r["n_com_valido"] += 1
        r["const_menor_todos_a"] += int(s["c"]["todos"] < s["a"]["todos"])
        r["const_menor_todos_b"] += int(s["c"]["todos"] < s["b"]["todos"])
        r["fspl_menor_validos_a"] += int(s["a"]["validos"] < s["c"]["validos"])
        r["fspl_menor_validos_b"] += int(s["b"]["validos"] < s["c"]["validos"])
    return r


def somar(d):
    t = {}
    for r in d.values():
        for k, v in r.items():
            t[k] = t.get(k, 0) + v
    return t


def bloco_U():
    bloco("U", "U — buffer zero (b = 0) × b = 2: deriva do MAE do constante e da fração válida",
          "CITÁVEL (voto independente em fase4/votos_R1_R2/ para 2 células + reagregação do forum-eng-dados)",
          "Fonte: fase4/R1_b0_resumo.json e R1_b0_por_sorteio.json. dp entre sorteios por célula (ddof = 1, conforme o campo estatisticas do artefato), sorteios sem nó válido no teste excluídos. "
          "Inversões recontadas por este script sobre os sorteios com nó válido (conferido contra T4_inversao_v3.csv para b = 2 antes de gravar b = 0).")
    R = A("fase4/R1_b0_resumo.json")
    S = A("fase4/R1_b0_por_sorteio.json")
    P2 = A("fase2/_v3_2.1_3.1_parcial_16x60rnd.json")  # sorteios de b = 2 (conferencia da recontagem)
    T4 = A("redacao/tables_v3/T4_inversao_v3.csv", "csv")
    VR1 = A("fase4/votos_R1_R2/saida_R1.json")

    pc = R.g("por_celula")
    conf("R1_resumo: 16 células", sorted(pc) == sorted(CELULAS), sorted(pc))

    # ---- sorteios com no valido
    scv = R.g("sorteios_com_no_valido")
    linha("U", "U.sorteios_validos",
          "Sorteios (célula × sorteio) com pelo menos um nó válido no teste",
          [F(R, ["sorteios_com_no_valido", "b0", "n_sorteios_com_no_valido_total"], "b = 0", "int",
             texto=f"{scv['b0']['n_sorteios_com_no_valido_total']} de {scv['b0']['de_total']}"),
           F(R, ["sorteios_com_no_valido", "b2", "n_sorteios_com_no_valido_total"], "b = 2", "int",
             texto=f"{scv['b2']['n_sorteios_com_no_valido_total']} de {scv['b2']['de_total']}"),
           F(R, ["sorteios_com_no_valido", "b0", "n_sorteios_sem_no_valido_excluidos_total"], "b = 0, excluídos", "int"),
           F(R, ["sorteios_com_no_valido", "b2", "n_sorteios_sem_no_valido_excluidos_total"], "b = 2, excluídos", "int")])

    # ---- medias de 16 celulas e razoes do criterio
    m16 = R.g("medias_16_celulas")
    dp0 = {c: R.g("por_celula", c, "b0", "dp_mae_constante_validos") for c in CELULAS}
    dp2 = {c: R.g("por_celula", c, "b2", "dp_mae_constante_validos") for c in CELULAS}
    f0 = {c: R.g("por_celula", c, "b0", "dp_fracao_valida_teste") for c in CELULAS}
    f2 = {c: R.g("por_celula", c, "b2", "dp_fracao_valida_teste") for c in CELULAS}
    conf("R1: média das 16 células dp MAE b0 = medias_16_celulas",
         proximo(sum(dp0.values()) / 16, m16["dp_mae_constante_validos_b0"], 1e-12), None)
    conf("R1: média das 16 células dp MAE b2 = medias_16_celulas",
         proximo(sum(dp2.values()) / 16, m16["dp_mae_constante_validos_b2_artefato_2.1"], 1e-12), None)
    conf("R1: média das 16 células dp fração b0/b2 = medias_16_celulas",
         proximo(sum(f0.values()) / 16, m16["dp_fracao_valida_teste_b0"], 1e-12)
         and proximo(sum(f2.values()) / 16, m16["dp_fracao_valida_teste_b2"], 1e-12), None)
    conf("R1: R_err e R_frac = razões das médias de 16",
         proximo(R.g("razoes_criterio", "R_err"), (sum(dp0.values()) / 16) / (sum(dp2.values()) / 16), 1e-12)
         and proximo(R.g("razoes_criterio", "R_frac"), (sum(f0.values()) / 16) / (sum(f2.values()) / 16), 1e-12), None)
    linha("U", "U.medias16",
          "Médias das 16 células do dp entre sorteios (b = 0 e b = 2): MAE do constante nos válidos e fração válida do teste; razões b = 0 sobre b = 2",
          [F(R, ["medias_16_celulas", "dp_mae_constante_validos_b0"], "dp MAE, b = 0", "f2", " dB"),
           F(R, ["medias_16_celulas", "dp_mae_constante_validos_b2_artefato_2.1"], "dp MAE, b = 2", "f2", " dB"),
           F(R, ["razoes_criterio", "R_err"], "razão dp MAE (b0/b2)", "f2"),
           F(R, ["medias_16_celulas", "dp_fracao_valida_teste_b0"], "dp fração válida, b = 0", "f3"),
           F(R, ["medias_16_celulas", "dp_fracao_valida_teste_b2"], "dp fração válida, b = 2", "f3"),
           F(R, ["razoes_criterio", "R_frac"], "razão dp fração (b0/b2)", "f2")])

    # ---- min e max das razoes por celula
    rm = {c: R.g("por_celula", c, "razao_dp_mae_b0_sobre_b2") for c in CELULAS}
    rf = {c: R.g("por_celula", c, "razao_dp_fracao_b0_sobre_b2") for c in CELULAS}
    for c in CELULAS:
        conf(f"R1: razão por célula {c} = dp_b0/dp_b2", proximo(rm[c], dp0[c] / dp2[c], 1e-12) and proximo(rf[c], f0[c] / f2[c], 1e-12), None)
    for nome, d, camp, rot in (("mae", rm, "razao_dp_mae_b0_sobre_b2", "do dp do MAE do constante nos válidos"),
                               ("fracao", rf, "razao_dp_fracao_b0_sobre_b2", "do dp da fração válida do teste")):
        vmin, cmin = argextremo(d, min)
        vmax, cmax = argextremo(d, max)
        linha("U", f"U.razao_{nome}_minmax",
              f"Razão b = 0 sobre b = 2 {rot}, por célula: mínimo e máximo entre as 16 células",
              [D("mínimo", vmin, "f2", "", f"mínimo entre células de por_celula.<c>.{camp}", R.campo("por_celula", "<célula>", camp),
                 f"{q(vmin, 2)} ({', '.join(cmin)})"),
               D("máximo", vmax, "f2", "", f"máximo entre células de por_celula.<c>.{camp}", R.campo("por_celula", "<célula>", camp),
                 f"{q(vmax, 2)} ({', '.join(cmax)})")])

    # ---- por celula (16 linhas)
    for c in CELULAS:
        linha("U", f"U.cel.{c}",
              f"{rotulo_celula(c)}: dp entre sorteios do MAE do constante nos válidos e da fração válida do teste, b = 0 e b = 2, e razões; sorteios com nó válido",
              [F(R, ["por_celula", c, "b0", "dp_mae_constante_validos"], "dp MAE b = 0", "f2", " dB"),
               F(R, ["por_celula", c, "b2", "dp_mae_constante_validos"], "dp MAE b = 2", "f2", " dB"),
               F(R, ["por_celula", c, "razao_dp_mae_b0_sobre_b2"], "razão MAE", "f2"),
               F(R, ["por_celula", c, "b0", "dp_fracao_valida_teste"], "dp fração b = 0", "f3"),
               F(R, ["por_celula", c, "b2", "dp_fracao_valida_teste"], "dp fração b = 2", "f3"),
               F(R, ["por_celula", c, "razao_dp_fracao_b0_sobre_b2"], "razão fração", "f2"),
               F(R, ["por_celula", c, "b0", "n_sorteios_com_no_valido"], "sorteios com nó válido, b = 0", "int"),
               F(R, ["por_celula", c, "b2", "n_sorteios_com_no_valido"], "sorteios com nó válido, b = 2", "int")])

    # ---- conferencia: voto independente (2 celulas)
    for c in ("lins_Q1", "sorocaba_Q3"):
        for b in ("b0", "b2"):
            v = VR1.g(c, b, "json")
            conf(f"voto R1 ({c},{b}): dp MAE e dp fração do voto = resumo",
                 proximo(VR1.g(c, b, "dp_mae_const110"), R.g("por_celula", c, b, "dp_mae_constante_validos"))
                 and proximo(VR1.g(c, b, "dp_fracao_valida_so_com_valido"), R.g("por_celula", c, b, "dp_fracao_valida_teste"))
                 and VR1.g(c, b, "n_sorteios_com_valido") == R.g("por_celula", c, b, "n_sorteios_com_no_valido"),
                 None)

    # ---- recontagem das inversoes (b = 2 primeiro: tem de reproduzir T4; depois b = 0)
    b2 = {c: recontar([norm_b2(s) for s in P2.g("celulas", c, "por_sorteio")]) for c in CELULAS}
    t2 = somar(b2)
    allrow = [r for r in T4.data if r["celula"] == "ALL"]
    conf("T4_inversao_v3.csv tem a linha ALL", len(allrow) == 1, None)
    allrow = allrow[0]
    esperado = dict(n_com_valido=int(allrow["n_sorteios_com_validos"]),
                    fspl_menor_validos_a=int(allrow["fspl_menor_validos_a"]),
                    fspl_menor_validos_b=int(allrow["fspl_menor_validos_b"]),
                    const_menor_todos_a=int(allrow["constante_menor_todos_a"]),
                    const_menor_todos_b=int(allrow["constante_menor_todos_b"]))
    conf("recontagem b = 2 sobre os sorteios com nó válido reproduz a linha ALL de T4_inversao_v3.csv (904; 629/729; 799/796)",
         all(t2[k] == v for k, v in esperado.items()) and t2["n_com_valido"] == R.g("sorteios_com_no_valido", "b2", "n_sorteios_com_no_valido_total"),
         {"recontagem": {k: t2[k] for k in esperado}, "T4_ALL": esperado})
    b0 = {c: recontar([norm_b0(s) for s in S.g("celulas", c, "sorteios")]) for c in CELULAS}
    t0 = somar(b0)
    conf("recontagem b = 0: sorteios com nó válido = resumo (938)",
         t0["n_com_valido"] == R.g("sorteios_com_no_valido", "b0", "n_sorteios_com_no_valido_total"), [t0["n_com_valido"]])
    for c in CELULAS:
        conf(f"recontagem b = 0 por célula {c}: n com nó válido = resumo", b0[c]["n_com_valido"] == R.g("por_celula", c, "b0", "n_sorteios_com_no_valido"), None)
    inv = R.g("inversoes_constante_gt_fspl", "b0")
    conf("recontagem b = 0 de FSPL menor nos válidos = resumo (a, b); constante menor em todos os nós = n válidos − (a_todos, b_todos) do resumo",
         t0["fspl_menor_validos_a"] == inv["a_validos"] and t0["fspl_menor_validos_b"] == inv["b_validos"]
         and t0["const_menor_todos_a"] == t0["n_com_valido"] - inv["a_todos"]
         and t0["const_menor_todos_b"] == t0["n_com_valido"] - inv["b_todos"],
         {"recontagem": t0, "resumo": inv})
    CONFERENCIAS.append({"descricao": "(informativo) recontagem de b = 2 sobre os 960 do parcial fase2: n sorteios", "passou": True,
                         "detalhe": t2["n_sorteios"]})

    def pct(a, b):
        return q(a / b * 100, 1) + " %"

    def itens_inv(t, rot_b):
        n = t["n_com_valido"]
        mk = lambda rot, k, der: D(rot, t[k], "int", "", der, "recontagem sobre " + rot_b, f"{t[k]} de {n} ({pct(t[k], n)})")
        return [mk("constante com menor erro em todos os nós, calibração (a)", "const_menor_todos_a",
                   "contagem de sorteios com n_validos>0 e mae_constante.todos < mae_fspl_a.todos"),
                mk("constante com menor erro em todos os nós, calibração (b)", "const_menor_todos_b",
                   "contagem de sorteios com n_validos>0 e mae_constante.todos < mae_fspl_b.todos"),
                mk("FSPL com menor erro nos válidos, calibração (a)", "fspl_menor_validos_a",
                   "contagem de sorteios com n_validos>0 e mae_fspl_a.validos < mae_constante.validos"),
                mk("FSPL com menor erro nos válidos, calibração (b)", "fspl_menor_validos_b",
                   "contagem de sorteios com n_validos>0 e mae_fspl_b.validos < mae_constante.validos")]
    linha("U", "U.inversoes_b2_conferencia",
          "Recontagem para b = 2 sobre os sorteios com nó válido (conferência: reproduz a linha ALL de T4_inversao_v3.csv)",
          itens_inv(t2, "fase2/_v3_2.1_3.1_parcial_16x60rnd.json:celulas.<c>.por_sorteio (b = 2)"),
          nota="conferência passou: 629/729 e 799/796 sobre 904; denominador = sorteios com nó válido")
    linha("U", "U.inversoes_b0",
          "Recontagem para b = 0 sobre os sorteios com nó válido (938)",
          itens_inv(t0, "fase4/R1_b0_por_sorteio.json:celulas.<c>.sorteios (b = 0)"))
    # por celula em JSON detalhe (nao vira linha no md)
    DETALHE["U_inversoes_por_celula"] = {"b2": b2, "b0": b0}


DETALHE: dict = {}


# ============================================================================= BLOCO V
def bloco_V():
    bloco("V", "V — referência por desenho (hold-out com buffer × amostra aleatória de nós)",
          "CITÁVEL para deff, viés e REQM (voto em fase4/votos_R1_R2/ para Bauru; fase4/votos_fismat_R2_R4/ para n/deff)",
          "Fonte: fase4/R2_resumo.json e R2_por_sorteio.json. Legenda (docstring de scripts/v3_12_R2_referencia_desenho.py): H = hold-out em blocos com buffer (b = 2); "
          "H0 = hold-out sem buffer; A = amostra aleatória simples de nós, com o tamanho de H; J = Hájek sobre H; μ_U = média do erro no domínio inteiro. "
          "viés em unidades de EP = viés / EP de Monte Carlo do viés. Nos válidos, sorteios sem nó válido ficam indefinidos em H/H0/J (campo n_sorteios_indefinidos).")
    R = A("fase4/R2_resumo.json")
    S = A("fase4/R2_por_sorteio.json")
    BC = A("fase4/votos_fismat_R2_R4/bc.json")
    VT = A("fase4/votos_fismat_R2_R4/veredito.json")
    VB = A("fase4/votos_R1_R2/saida_R2_bauru_Q1.json")
    VR = A("fase4/votos_R1_R2/veredito.json")
    rc = R.g("resumo_por_celula")
    conf("R2_resumo: 4 células Q1", sorted(rc) == sorted(Q1), sorted(rc))

    # conferencia deff: fismat bc.json b == R2_resumo
    for c in Q1:
        for pred in PRED_ROT:
            for pop in POP_ROT:
                dj = BC.g("b", f"{c}|{pred}|{pop}", "deff_json")
                conf(f"deff de R2_resumo = deff_json do voto fismat ({c}|{pred}|{pop})",
                     proximo(dj / R.g("resumo_por_celula", c, pred, pop, "razoes", "deff_H_sobre_A") - 1, 0, 1e-9), None)
    # conferencia voto Bauru Q1 (4 digitos)
    est_b = rc["bauru_Q1"]["constante"]
    conf("voto R2 (Bauru Q1, constante, válidos): viés e EP reproduzem R2_resumo a 1e-3 relativo",
         abs(VB.g("validos", "vies_H") / est_b["validos"]["estimadores"]["H"]["vies"] - 1) < 1e-3
         and abs(VB.g("validos", "ep_vies") / est_b["validos"]["estimadores"]["H"]["ep_mc_vies"] - 1) < 1e-3
         and abs(VB.g("validos", "var_H") / est_b["validos"]["estimadores"]["H"]["variancia"] - 1) < 1e-3
         and abs(VB.g("validos", "deff_simulado") / est_b["validos"]["razoes"]["deff_H_sobre_A"] - 1) < 1e-3, None)
    conf("voto R2 (Bauru Q1, constante, todos): viés e EP reproduzem R2_resumo a 1e-3 relativo",
         abs(VB.g("todos", "vies_H") / est_b["todos"]["estimadores"]["H"]["vies"] - 1) < 1e-2
         and abs(VB.g("todos", "ep_vies") / est_b["todos"]["estimadores"]["H"]["ep_mc_vies"] - 1) < 1e-2,
         [VB.g("todos", "vies_H"), est_b["todos"]["estimadores"]["H"]["vies"]])

    n_ranges = {"todos": [], "validos": []}
    for c in Q1:
        info = S.g("celulas", c, "info_por_sorteio")
        for pred, prot in PRED_ROT.items():
            for pop, popr in POP_ROT.items():
                base = ["resumo_por_celula", c, pred, pop]
                pre = f"{rotulo_celula(c)}, preditor {prot}, {popr}"
                idb = f"V.{c}.{pred}.{pop}"
                E = lambda est, campo: base + ["estimadores", est, campo]
                linha("V", idb + ".vies_H", f"{pre}: viés do estimador H em relação a μ_U, com EP de Monte Carlo e em unidades de EP",
                      [F(R, E("H", "vies"), "viés de H", "f2", " dB"),
                       F(R, E("H", "ep_mc_vies"), "EP", "f2", " dB"),
                       F(R, E("H", "vies_em_ep"), "viés em EP", "f2", " EP")])
                linha("V", idb + ".reqm", f"{pre}: REQM de H e de A",
                      [F(R, E("H", "reqm"), "REQM de H", "f2", " dB"),
                       F(R, E("A", "reqm"), "REQM de A", "f3", " dB")])
                linha("V", idb + ".deff", f"{pre}: efeito de desenho (variância de H sobre a de A) e de H0 sobre A",
                      [F(R, base + ["razoes", "deff_H_sobre_A"], "deff (H/A)", "int"),
                       F(R, base + ["razoes", "deff_H0_sobre_A"], "H0/A", "int")])
                # n medio de nos pontuados e n/deff
                chave = "n_te_b2" if pop == "todos" else "n_te_b2_validos"
                lst = info[chave]
                conf(f"R2_por_sorteio {c}: {chave} tem 200 sorteios", len(lst) == 200, len(lst))
                n_med = sum(lst) / len(lst)
                deff = R.g(*base, "razoes", "deff_H_sobre_A")
                n_ranges[pop].append((n_med / deff, c, pred))
                linha("V", idb + ".n_deff", f"{pre}: n médio de nós pontuados por sorteio e n/deff",
                      [D("n médio de nós pontuados", n_med, "int", "",
                         f"média dos 200 sorteios de info_por_sorteio.{chave} (inclui sorteios sem nó válido, como no voto fismat)",
                         S.campo("celulas", c, "info_por_sorteio", chave)),
                       D("n/deff", n_med / deff, "f1", "", "n médio / razoes.deff_H_sobre_A",
                         S.campo("celulas", c, "info_por_sorteio", chave) + " ÷ " + R.campo(*base, "razoes", "deff_H_sobre_A"))])
                linha("V", idb + ".vies_H0", f"{pre}: viés do estimador H0 (sem buffer) em relação a μ_U, com EP de Monte Carlo e em unidades de EP",
                      [F(R, E("H0", "vies"), "viés de H0", "f2", " dB"),
                       F(R, E("H0", "ep_mc_vies"), "EP", "f2", " dB"),
                       F(R, E("H0", "vies_em_ep"), "viés em EP", "f2", " EP")])
                linha("V", idb + ".razao_J_H",
                      f"{pre}: razão REQM(J)/REQM(H) (Hájek truncado)",
                      [F(R, base + ["razoes", "reqm_J_sobre_reqm_H"], "REQM(J)/REQM(H)", "f2",
                         texto="NÃO CITAR o valor (Hájek truncado); citar só que não reduz")],
                      nota="NÃO CITAR o valor (Hájek truncado); citar só que não reduz")
    # conferencia das faixas de n/deff com o voto fismat (texto do voto a.json: todos 14.9-22.0; validos 1.1-4.8)
    texto_voto = VT.g("b", "todos_n/deff") + " | " + VT.g("b", "validos_n/deff")
    for pop in ("todos", "validos"):
        lo = min(x[0] for x in n_ranges[pop])
        hi = max(x[0] for x in n_ranges[pop])
        DETALHE.setdefault("V_n_deff_faixa", {})[pop] = {"min": lo, "max": hi,
                                                         "min_em": [x[1:] for x in n_ranges[pop] if x[0] == lo],
                                                         "max_em": [x[1:] for x in n_ranges[pop] if x[0] == hi]}
        conf(f"faixa de n/deff ({pop}) arredondada a 1 casa aparece no voto fismat a.json (b.{pop}_n/deff)",
             q(lo, 1).replace(",", ".") in texto_voto and q(hi, 1).replace(",", ".") in texto_voto,
             [q(lo, 1), q(hi, 1), texto_voto])
        linha("V", f"V.faixa_n_deff.{pop}", f"Faixa de n/deff entre as 4 células e os 2 preditores, {POP_ROT[pop]}",
              [D("mínimo", lo, "f1", "", "mínimo de n médio/deff entre as 8 combinações célula × preditor", "ver linhas V.<célula>.<preditor>." + pop + ".n_deff"),
               D("máximo", hi, "f1", "", "máximo de n médio/deff entre as 8 combinações célula × preditor", "ver linhas V.<célula>.<preditor>." + pop + ".n_deff")],
              nota=f"conferido contra {VT.rel} campo b.{pop}_n/deff")
    # contagem mecanica J
    for chave in ("constante/validos", "constante/todos", "fspl_calibrado_b/validos", "fspl_calibrado_b/todos"):
        pred, pop = chave.split("/")
        rj = [R.g("resumo_por_celula", c, pred, pop, "razoes", "reqm_J_sobre_reqm_H") for c in Q1]
        linha("V", f"V.J_nao_reduz.{pred}.{pop}",
              f"Hájek truncado (J), preditor {PRED_ROT[pred]}, {POP_ROT[pop]}: número de células em que REQM(J) é menor que REQM(H)",
              [D("células com REQM(J) < REQM(H)", sum(1 for x in rj if x < 1.0), "int", "",
                 "contagem sobre as 4 células de razoes.reqm_J_sobre_reqm_H < 1", R.campo("resumo_por_celula", "<célula>", pred, pop, "razoes", "reqm_J_sobre_reqm_H"),
                 f"{sum(1 for x in rj if x < 1.0)} de {len(Q1)}"),
               F(R, ["contagem_mecanica_limiares", chave, "celulas_reqm_J_lt_metade_H"], "células com REQM(J) < metade de REQM(H)", "int",
                 texto=f"{R.g('contagem_mecanica_limiares', chave, 'celulas_reqm_J_lt_metade_H')} de {len(Q1)}")],
              nota="NÃO CITAR o valor da razão (Hájek truncado); citar só que não reduz")
    CONFERENCIAS.append({"descricao": "voto R2/R1 (veredito)", "passou": VR.g("R2", "veredito") == "CONFIRMADO",
                         "detalhe": VR.g("R2", "veredito")})


# ============================================================================= BLOCO W
def bloco_W():
    bloco("W", "W — termo de razão (b = 0), FSPL (b) nos nós válidos",
          "CITÁVEL para Bauru e Lins (dois cálculos independentes); PROVISÓRIO para Campinas e Sorocaba (um cálculo)",
          "Fonte: fase4/_pareceres_rigor_R1-R4/fismat_scripts/fismat_R2_termo_razao.json (fismat) e fase4/votos_fismat_R2_R4/a.json (voto, Bauru e Lins). "
          "Marca: leitura post hoc; as 4 células compartilham as permutações. "
          "-Cov(Err,M)/E[M] = campo termo_razao; EP bootstrap = ep_boot_razao (do termo de razão); EP de Monte Carlo = ep_mc_vies de R2_resumo (do viés).")
    FM = A("fase4/_pareceres_rigor_R1-R4/fismat_scripts/fismat_R2_termo_razao.json")
    VT = A("fase4/votos_fismat_R2_R4/a.json")
    R2 = A("fase4/R2_resumo.json")
    for c in Q1:
        cid = c.split("_")[0]
        base = [c, "H0_fspl_calibrado_b_validos"]
        # conferencia: viés do fismat = viés H0 do R2_resumo
        conf(f"W {c}: viés de H0 (fismat) = R2_resumo.estimadores.H0.vies (FSPL, válidos)",
             proximo(FM.g(*base, "vies"), R2.g("resumo_por_celula", c, "fspl_calibrado_b", "validos", "estimadores", "H0", "vies"), 1e-9), None)
        conf(f"W {c}: identidade viés = termo de razão + termo de desenho (campo fecha ≈ 0)",
             abs(FM.g(*base, "fecha")) < 1e-9, FM.g(*base, "fecha"))
        if cid in ("bauru", "lins"):
            v = VT.g(cid, "validos")
            conf(f"W {c}: voto a.json reproduz o fismat (viés, termo de razão, corr, EP de MC)",
                 proximo(v["vies_vs_mu"], FM.g(*base, "vies"), 1e-9) and proximo(v["razao"], FM.g(*base, "termo_razao"), 1e-9)
                 and proximo(v["corr_Err_M"], FM.g(*base, "corr_err_M"), 1e-9)
                 and proximo(v["ep_mc_vies"], R2.g("resumo_por_celula", c, "fspl_calibrado_b", "validos", "estimadores", "H0", "ep_mc_vies"), 1e-9),
                 None)
            estado = "CITÁVEL (dois cálculos independentes: fismat_R2_termo_razao.json e voto a.json)"
        else:
            estado = "PROVISÓRIO (um cálculo: fismat_R2_termo_razao.json)"
        linha("W", f"W.{c}",
              f"{rotulo_celula(c)}, FSPL (b), nós válidos, hold-out sem buffer (H0): viés, -Cov(Err,M)/E[M], correlação entre Err e M e EP",
              [F(FM, base + ["vies"], "viés de H0", "f2", " dB"),
               F(FM, base + ["termo_razao"], "-Cov(Err,M)/E[M]", "f2", " dB"),
               F(FM, base + ["corr_err_M"], "corr(Err, M)", "f2"),
               F(FM, base + ["ep_boot_razao"], "EP bootstrap do termo de razão", "f2", " dB"),
               F(R2, ["resumo_por_celula", c, "fspl_calibrado_b", "validos", "estimadores", "H0", "ep_mc_vies"], "EP de Monte Carlo do viés", "f2", " dB")]
              + ([F(VT, [cid, "validos", "vies_vs_mu"], "viés (voto a.json)", "f2", " dB"),
                  F(VT, [cid, "validos", "razao"], "-Cov(Err,M)/E[M] (voto a.json)", "f2", " dB"),
                  F(VT, [cid, "validos", "corr_Err_M"], "corr(Err, M) (voto a.json)", "f2"),
                  F(VT, [cid, "validos", "ep_mc_vies"], "EP de Monte Carlo (voto a.json)", "f2", " dB")] if cid in ("bauru", "lins") else []),
              estado=estado,
              nota="leitura post hoc; as 4 células compartilham as permutações")


# ============================================================================= BLOCO X
def bloco_X():
    bloco("X", "X — termo de desenho (intervalo por cotas, leitura primária L_no_a_no, m nodal)",
          "CITÁVEL (voto em fase4/votos_R3_R4/ para Lins; forum-eng-dados)",
          "Fonte: fase4/R3_termo_desenho.json, leitura primária L_no_a_no, m nodal. T = termo de desenho Σ p_i e_i / Σ p_i − μ_U; T_inf e T_sup são as cotas; "
          "o termo por frequência de 200 sorteios (± EP bootstrap) entra só como conferência.")
    X = A("fase4/R3_termo_desenho.json")
    conf("R3: leitura primária = L_no_a_no", X.g("leitura_primaria") == "L_no_a_no", X.g("leitura_primaria"))
    A("fase4/votos_R3_R4/veredito.json")
    for c in Q1:
        for pred, prot in PRED_ROT.items():
            for pop, popr in POP_ROT.items():
                ch = f"{pred}|{pop}"
                base = ["por_celula", c, "resultados", ch]
                pre = f"{rotulo_celula(c)}, preditor {prot}, {popr}"
                cl = base + ["fechado", "m_nodal", "L_no_a_no"]
                linha("X", f"X.{c}.{pred}.{pop}.intervalo",
                      f"{pre}: μ_U e cotas do termo de desenho (T_inf, T_sup) e largura do intervalo",
                      [F(X, base + ["mu_U_dB"], "μ_U", "f2", " dB"),
                       F(X, cl + ["T_inf_dB"], "T_inf", "f3", " dB"),
                       F(X, cl + ["T_sup_dB"], "T_sup", "f3", " dB"),
                       F(X, cl + ["largura_dB"], "largura", "f3", " dB")])
                linha("X", f"X.{c}.{pred}.{pop}.freq200",
                      f"{pre}: termo de desenho por frequência de 200 sorteios, com EP bootstrap (só conferência)",
                      [F(X, base + ["frequencia_200", "T_freq200_dB"], "T por frequência (200)", "f3", " dB"),
                       F(X, base + ["frequencia_200", "ep_bootstrap_200"], "EP bootstrap", "f3", " dB")],
                      nota="só conferência")


# ============================================================================= BLOCO Y
def classe_nodal(R4: Artefato, c: str, nome: str):
    for i, x in enumerate(R4.g("por_celula", c, "classes_nodais")):
        if x["classe"] == nome:
            return i, x
    die(f"classe {nome} ausente em R4 {c}")


def bloco_Y():
    bloco("Y", "Y — inclusão por classe de borda m (nodal): frequência × forma fechada/cotas",
          "CITÁVEL",
          "Fonte: fase4/R4_p_por_classe.json e fase4/votos_fismat_R2_R4/bc.json. Leitura primária L_no_a_no; classes nodais; sorteios de fase1/1.8; "
          "forma fechada onde m = 0 ou a condição (iii) vale no nó; cotas nas demais.")
    R4 = A("fase4/R4_p_por_classe.json")
    BC = A("fase4/votos_fismat_R2_R4/bc.json")
    VT = A("fase4/votos_R3_R4/voto_R3R4_proprio_saida.json")
    FB = A("fase4/_pareceres_rigor_R1-R4/fismat_scripts/fismat_R4_teste_exato.json")
    A("fase4/votos_R3_R4/veredito.json")
    A("fase4/votos_fismat_R2_R4/veredito.json")
    P = R4.g("parametros")
    N, kte, ktr = R4.g("parametros", "N"), R4.g("parametros", "k_te"), R4.g("parametros", "k_tr")
    kva = VT.g("k_va")
    conf("N = k_tr + k_va + k_te", N == ktr + kva + kte, [N, ktr, kva, kte])
    conf("R4: leitura primária = L_no_a_no", R4.g("leitura_primaria") == "L_no_a_no", None)
    linha("Y", "Y.parametros", "Partição em blocos: número de blocos N e contagens de treino, validação e teste",
          [F(R4, ["parametros", "N"], "N", "int"),
           F(R4, ["parametros", "k_tr"], "k_tr", "int"),
           F(VT, ["k_va"], "k_va", "int"),
           F(R4, ["parametros", "k_te"], "k_te", "int")],
          nota="k_va lido do voto votos_R3_R4/voto_R3R4_proprio_saida.json; conferido N = k_tr + k_va + k_te")
    # cotas 19/131 e 39/131
    i1, x1 = classe_nodal(R4, "bauru_Q1", "m=1|todas")
    inf1, sup1 = x1["cota_inferior_por_m"]["1"], x1["cota_superior_por_m"]["1"]
    conf("cota inferior de m = 1 = (k_te − 1)/(N − 1)", proximo(inf1, (kte - 1) / (N - 1), 1e-12), [inf1, (kte - 1) / (N - 1)])
    conf("cota superior de m = 1 = (N − 1 − k_tr)/(N − 1)", proximo(sup1, (N - 1 - ktr) / (N - 1), 1e-12), [sup1, (N - 1 - ktr) / (N - 1)])
    for c in Q1:
        _, xc = classe_nodal(R4, c, "m=1|todas")
        conf(f"cotas de m = 1 iguais em {c}", proximo(xc["cota_inferior_por_m"]["1"], inf1, 1e-15) and proximo(xc["cota_superior_por_m"]["1"], sup1, 1e-15), None)
    campo_cotas = R4.campo("por_celula", "<célula>", "classes_nodais", "[classe m=1|todas]", "cota_inferior_por_m.1 / cota_superior_por_m.1")
    linha("Y", "Y.cotas_m1", "Cotas da probabilidade de inclusão condicional no teste para a classe m = 1: inferior (k_te − 1)/(N − 1) e superior (N − 1 − k_tr)/(N − 1)",
          [D("cota inferior", inf1, "f4", "", "valor lido de cota_inferior_por_m.1; rótulo (k_te−1)/(N−1) conferido com N e k_te do JSON", campo_cotas,
             f"{kte - 1}/{N - 1} = {q(inf1, 4)}"),
           D("cota superior", sup1, "f4", "", "valor lido de cota_superior_por_m.1; rótulo (N−1−k_tr)/(N−1) conferido com N e k_tr do JSON", campo_cotas,
             f"{N - 1 - ktr}/{N - 1} = {q(sup1, 4)}")],
          nota="idênticas nas 4 células")

    # fracao com forma fechada / so cotas
    for c in Q1:
        cb = ["c", c]
        _, x0 = classe_nodal(R4, c, "m=0|todas")
        _, x1i = classe_nodal(R4, c, "m=1|(iii) vale")
        _, x3i = classe_nodal(R4, c, "m=3|(iii) vale")
        soma = x0["fracao_nos"] + x1i["fracao_nos"] + x3i["fracao_nos"]
        conf(f"fração com forma fechada de bc.json = m=0 + m=1 com (iii) + m=3 com (iii) de R4 ({c})",
             proximo(BC.g(*cb, "frac_exato_L_no_a_no"), soma, 1e-9) and proximo(BC.g(*cb, "frac_exato_L_no_a_no") + BC.g(*cb, "frac_cotas_L"), 1.0, 1e-9),
             [BC.g(*cb, "frac_exato_L_no_a_no"), soma])
        linha("Y", f"Y.fracao.{c}",
              f"{rotulo_celula(c)}: fração de nós com forma fechada (m = 0, m = 1 com (iii), m = 3 com (iii)) e fração de nós só com cotas",
              [F(BC, cb + ["frac_exato_L_no_a_no"], "forma fechada", "pct2"),
               F(BC, cb + ["frac_cotas_L"], "só cotas", "pct2")],
              nota="conferido contra a soma de fracao_nos das classes m=0, m=1|(iii) vale e m=3|(iii) vale de R4_p_por_classe.json")

    # por classe
    for c in Q1:
        for nome, rot in (("m=0|todas", "m = 0"), ("m=1|todas", "m = 1 (todos os nós da classe)"),
                          ("m=1|(iii) vale", "m = 1 com (iii)"), ("m=2|todas", "m = 2"), ("m=3|todas", "m = 3")):
            i, x = classe_nodal(R4, c, nome)
            base = ["por_celula", c, "classes_nodais", i]
            L = base + ["leituras", "L_no_a_no"]
            m = x["m_valores"][0]
            itens = [F(R4, base + ["fracao_nos"], "fração de nós", "pct2"),
                     F(R4, base + ["frequencia_media_nodal"], "frequência", "f4" if x["frequencia_media_nodal"] > 0.01 else "sig3"),
                     F(R4, base + ["ep_mc_bootstrap_sorteios"], "EP", "sig3")]
            fechado = R4.g_nulo_ok(*L, "valor_fechado")
            if fechado is not None:
                itens.append(F(R4, L + ["valor_fechado"], "valor fechado", "f4" if fechado > 0.01 else "sig3"))
            else:
                itens.append(F(R4, base + ["cota_inferior_por_m", str(m)], "cota inferior", "f4" if x["cota_inferior_por_m"][str(m)] > 0.01 else "sig3"))
                itens.append(F(R4, base + ["cota_superior_por_m", str(m)], "cota superior", "f4"))
            linha("Y", "Y.classe." + c + "." + re.sub(r"[^0-9A-Za-z]+", "_", nome).strip("_"),
                  f"{rotulo_celula(c)}, classe {rot}: fração de nós, frequência de inclusão condicional ao teste com EP, e valor fechado ou cotas",
                  itens)
    # binomial de Sorocaba
    c = "sorocaba_Q1"
    i, x = classe_nodal(R4, c, "m=2|(iii) vale")
    ev, ens = x["retidos_bloco_sorteio_com_algum_no"], x["ensaios_bloco_sorteio"]
    conf("binomial de Sorocaba: eventos e ensaios de R4 = fismat_R4_teste_exato", ev == FB.g(c, "m2_iii", "eventos") and proximo(ens, FB.g(c, "m2_iii", "ensaios")), [ev, ens])
    linha("Y", "Y.binomial_sorocaba",
          "Sorocaba Q1, classe m = 2 com (iii): eventos em ensaios (bloco × sorteio) e probabilidade binomial P(X ≤ eventos) sob a cota inferior",
          [F(R4, ["por_celula", c, "classes_nodais", i, "retidos_bloco_sorteio_com_algum_no"], "eventos", "int"),
           F(R4, ["por_celula", c, "classes_nodais", i, "ensaios_bloco_sorteio"], "ensaios", "int"),
           F(BC, ["c", "binom_P(X<=1;95,0.0201)"], "P (bc.json, p = 0,0201)", "f2"),
           F(FB, [c, "m2_iii", "P_X_le_obs_binom_p0"], "P (fismat, p = cota inferior exata)", "f2")],
          nota="só para resposta a revisor")


# ============================================================================= BLOCO Z
def bloco_Z():
    bloco("Z", "Z — modelos treinados (GNN e MLP): sentinelas, repetições, deriva sobre o piso, melhor época, erro relativo ao constante",
          "CITÁVEL (V2 + forum-eng-ia)",
          "Fonte: redacao/_pareceres_2026-10-01_t25/verificacao_V2_modelos/saida_V2.json. dp amostral = ddof 1 (campo dp_amostral). Contagens de corridas nos campos n de cada linha.")
    V = A("redacao/_pareceres_2026-10-01_t25/verificacao_V2_modelos/saida_V2.json")
    A("redacao/_pareceres_2026-10-01_t25/verificacao_V2_modelos/veredito_V2.json")
    med = V.g("a_mediana_delta_sentinela_por_celula")
    absm = {c: abs(v["mediana"]) for c, v in med.items()}
    vmax, cmax = argextremo(absm, max)
    conf("a_max_abs_mediana = max |mediana| das células", proximo(vmax, V.g("a_max_abs_mediana"), 1e-12), [vmax, V.g("a_max_abs_mediana")])
    linha("Z", "Z.sentinela_max",
          "Maior mediana, entre as células do campo a_mediana_delta_sentinela_por_celula, do módulo da diferença GNN − MLP no MAE dos nós sentinela",
          [F(V, ["a_max_abs_mediana"], "maior |mediana|", "f3", " dB", texto=f"{q(V.g('a_max_abs_mediana'), 3)} dB ({', '.join(cmax)})")])
    for c in ("bauru_Q1", "lins_Q1"):
        linha("Z", f"Z.A2c.{c}",
              f"{rotulo_celula(c)}: diferença absoluta entre repetições da mesma GNN (A2c), nos sentinelas e nos válidos",
              [F(V, ["a_A2c_gnn_delta_abs", c, "sen"], "sentinelas", "f3", " dB"),
               F(V, ["a_A2c_gnn_delta_abs", c, "val"], "válidos", "f3", " dB")])
    piso = V.g("b_piso_val_calc")
    conf("piso = A2c Bauru Q1 válidos", proximo(piso, V.g("a_A2c_gnn_delta_abs", "bauru_Q1", "val"), 1e-12), None)
    linha("Z", "Z.piso", "Piso de reprodutibilidade usado na razão (diferença entre repetições da GNN, Bauru Q1, válidos)",
          [F(V, ["b_piso_val_calc"], "piso", "f3", " dB")])
    for mod in ("gnn", "mlp"):
        # o artefato divide pelo piso ARREDONDADO que consta no nome do campo (..._sobre_0.132)
        den = float(f"b_razao_{mod}_dp_amostral_sobre_0.132".rsplit("_sobre_", 1)[1])
        conf("denominador do nome do campo (0.132) = piso calculado arredondado a 3 casas", q(piso, 3) == q(den, 3), [piso, den])
        razoes = {k: V.g("b", k, "dp_amostral") / den for k in V.g("b") if k.startswith(mod + "_")}
        vmin, cmin = argextremo(razoes, min)
        vmax, cmax = argextremo(razoes, max)
        rg = V.g(f"b_razao_{mod}_dp_amostral_sobre_0.132")
        conf(f"razão {mod} dp amostral/piso: faixa recomputada = faixa do artefato", proximo(vmin, rg[0], 1e-9) and proximo(vmax, rg[1], 1e-9), [vmin, vmax, rg])
        linha("Z", f"Z.deriva_piso.{mod}",
              f"Razão entre o dp amostral entre sorteios do MAE nos válidos ({mod.upper()}, séries {mod}_Q1 a {mod}_Q4 do campo b) e o piso: mínimo e máximo",
              [F(V, [f"b_razao_{mod}_dp_amostral_sobre_0.132", 0], "mínimo", "f1", "",
                 texto=f"{q(rg[0], 1)} ({cmin[0]})"),
               F(V, [f"b_razao_{mod}_dp_amostral_sobre_0.132", 1], "máximo", "f1", "",
                 texto=f"{q(rg[1], 1)} ({cmax[0]})")],
              nota="série de cada extremo deduzida de b.<modelo>_Q<k>.dp_amostral ÷ 0.132 (denominador que consta no nome do campo; piso calculado b_piso_val_calc arredondado a 3 casas; conferido contra a faixa do artefato)")
    for mod in ("gnn", "mlp"):
        linha("Z", f"Z.melhor_epoca.{mod}",
              f"Melhor época ({mod.upper()}): mediana e fração das corridas em que a última época é a melhor",
              [F(V, ["c", mod, "mediana"], "mediana", "f1"),
               F(V, ["c", mod, "frac_ultima"], "fração na última época", "pct1"),
               F(V, ["c", mod, "n"], "corridas", "int")])
    for k in ("A3_gnn", "A3_mlp", "A4_gnn", "A4_mlp"):
        linha("Z", f"Z.melhor_epoca.{k}",
              f"Melhor época, lote {k.split('_')[0]} ({k.split('_')[1].upper()}): mediana e fração na última época",
              [F(V, ["c_A3_A4_separados", k, "mediana"], "mediana", "f1"),
               F(V, ["c_A3_A4_separados", k, "frac_ultima"], "fração na última época", "pct1"),
               F(V, ["c_A3_A4_separados", k, "n"], "corridas", "int")])
    for mod in ("gnn", "mlp"):
        linha("Z", f"Z.mae_sobre_constante.{mod}",
              f"MAE de {mod.upper()} como fração do MAE do constante (campo e_{mod}): mediana, mínimo e máximo",
              [F(V, [f"e_{mod}", "mediana"], "mediana", "pct1"),
               F(V, [f"e_{mod}", "min"], "mínimo", "pct1"),
               F(V, [f"e_{mod}", "max"], "máximo", "pct1"),
               F(V, [f"e_{mod}", "n"], "corridas", "int")])
    linha("Z", "Z.gnn_melhor_que_mlp",
          "Corridas do lote A4 (campo f_A4) em que a GNN é melhor que o MLP",
          [F(V, ["f_A4", "gnn_melhor"], "GNN melhor", "int",
             texto=f"{int(V.g('f_A4', 'gnn_melhor'))} de {V.g('f_A4', 'n')}"),
           F(V, ["f_A4", "n"], "n", "int")])
    linha("Z", "Z.gnn_melhor_por_celula",
          "Corridas do lote A4 em que a GNN é melhor, por célula (campo f_agregado)",
          [F(V, ["f_agregado", c], rotulo_celula(c), "int") for c in ("bauru_Q1", "bauru_Q2", "bauru_Q3", "bauru_Q4")])


# ============================================================================= BLOCO H
def bloco_H():
    bloco("H", "H — hiperparâmetros e configuração (51 campos)",
          "PROVISÓRIO (um extrator, conferido por script contra o código; sem segundo voto)",
          "Fonte: fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.json (lista campos); a coluna de campo traz a fonte caminho:linha do próprio JSON. "
          "Valores copiados sem arredondar; listas muito longas aparecem abreviadas aqui e completas no JSON espelho.")
    H = A("fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.json")
    A("fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.csv", "csv")
    campos = H.g("campos")
    conf("B7.2: 51 campos", len(campos) == 51, len(campos))
    for i, c in enumerate(campos):
        for k in ("grupo", "campo", "gnn", "mlp", "igual_nos_dois_bracos", "fonte", "nota"):
            if k not in c:
                die(f"B7.2 campos[{i}] sem {k}")
        gtxt = fmt(c["gnn"], "raw")
        mtxt = fmt(c["mlp"], "raw")

        def abrevia(s):
            return s if len(s) <= 400 else s[:200] + " […] " + s[-80:] + f" [valor completo ({len(s)} caracteres) no JSON espelho]"
        it = [Item("GNN", c["gnn"], abrevia(gtxt), H.campo("campos", i, "gnn")),
              Item("MLP", c["mlp"], abrevia(mtxt), H.campo("campos", i, "mlp")),
              Item("igual nos dois braços", c["igual_nos_dois_bracos"], fmt(c["igual_nos_dois_bracos"]), H.campo("campos", i, "igual_nos_dois_bracos")),
              Item("fonte (caminho:linha)", c["fonte"], c["fonte"], H.campo("campos", i, "fonte"))]
        linha("H", f"H.{i + 1:02d}", f"[{c['grupo']}] {c['campo']}", it, nota=c["nota"])
    # campos extras do JSON (registrados na nota do bloco)
    linha("H", "H.extra.penalidades_max",
          "Valores máximos registrados dos termos de penalidade de restrição FSPL e de sombra × NDVI (campos do JSON, fora dos 51)",
          [F(H, ["valor_maximo_constraint_fspl_registrado"], "restrição FSPL, máximo registrado", "raw"),
           F(H, ["valor_maximo_shadowing_ndvi_penalty_registrado"], "sombra × NDVI, máximo registrado", "raw")])
    BLOCOS["H"]["linhas"] -= 1  # a linha extra nao conta entre os 51
    BLOCOS["H"]["nota_contagem"] = "51 linhas H.01–H.51 + 1 linha extra H.extra.penalidades_max (referência da linha C.penalidades)"


# ============================================================================= BLOCO G
def bloco_G():
    bloco("G", "G — geometria do bloco e sentinela",
          "CITÁVEL",
          "Fonte: redacao/_pareceres_2026-10-01_t25/verificacao_V3_sentinela_geometria_props/saida_abd.json (dimensões do bloco por latitude) e saida_c.json (16 células).")
    D_ = A("redacao/_pareceres_2026-10-01_t25/verificacao_V3_sentinela_geometria_props/saida_abd.json")
    C = A("redacao/_pareceres_2026-10-01_t25/verificacao_V3_sentinela_geometria_props/saida_c.json")
    A("redacao/_pareceres_2026-10-01_t25/verificacao_V3_sentinela_geometria_props/veredito.json")
    d = D_.g("d")
    conf("saida_abd.d tem 9 latitudes", len(d) == 9, len(d))
    ew = {x["lat"]: x["ew_km"] for x in d}
    ns = {x["lat"]: x["ns_km"] for x in d}
    ar = {x["lat"]: x["area_pct"] for x in d}
    lat_ref = f"latitudes de {str(min(ew)).replace('.', ',')} a {str(max(ew)).replace('.', ',')}"

    def mm(dic, nd, un, nome, campo):
        vmin, lmin = argextremo(dic, min)
        vmax, lmax = argextremo(dic, max)
        return [D("mínimo", vmin, f"f{nd}", un, f"mínimo de d[].{campo} entre as latitudes", D_.campo("d", "[]", campo), f"{q(vmin, nd)}{un} (lat {str(lmin[0]).replace('.', ',')})"),
                D("máximo", vmax, f"f{nd}", un, f"máximo de d[].{campo} entre as latitudes", D_.campo("d", "[]", campo), f"{q(vmax, nd)}{un} (lat {str(lmax[0]).replace('.', ',')})")]
    linha("G", "G.bloco_EO", "Dimensão leste–oeste do bloco de 10 km, faixa entre as latitudes avaliadas (" + lat_ref + ")", mm(ew, 3, " km", "E-O", "ew_km"))
    linha("G", "G.bloco_NS", "Dimensão norte–sul do bloco de 10 km, faixa entre as latitudes avaliadas", mm(ns, 3, " km", "N-S", "ns_km"))
    linha("G", "G.area", "Diferença de área do bloco em relação a 100 km², em porcentagem, faixa entre as latitudes avaliadas", mm(ar, 2, " %", "área", "area_pct"))
    cel = {x["celula"]: x for x in C.data}
    conf("saida_c.json tem as 16 células", sorted(cel) == sorted(CELULAS), sorted(cel))
    for c in CELULAS:
        for k in ("sent_notfar", "notsent_far", "min_dist_sent", "min_rssi_valid", "frac_valid_below110", "valid", "sent_far"):
            if k not in cel[c]:
                die(f"saida_c.json {c} sem {k}")
    n_ok = sum(1 for c in CELULAS if cel[c]["sent_notfar"] == 0)
    mind = {c: cel[c]["min_dist_sent"] for c in CELULAS}
    vmin, cmin = argextremo(mind, min)
    linha("G", "G.sentinela_30km",
          "Células em que todo nó sentinela dista mais de 30 km do transmissor mais próximo (sent_notfar = 0), e menor distância de um sentinela ao transmissor",
          [D("células com sent_notfar = 0", n_ok, "int", "", "contagem de células com sent_notfar == 0", C.campo("[]", "sent_notfar"), f"{n_ok} de 16"),
           D("menor distância de sentinela ao transmissor mais próximo", vmin, "f2", " m", "mínimo de min_dist_sent entre as 16 células",
             C.campo("[]", "min_dist_sent"), f"{q(vmin, 2)} m ({cmin[0]})")])
    mr = {c: cel[c]["min_rssi_valid"] for c in CELULAS}
    rmin, cr1 = argextremo(mr, min)
    rmax, cr2 = argextremo(mr, max)
    linha("G", "G.rssi_min_validos", "Faixa entre as 16 células do RSSI mínimo dos nós válidos",
          [D("mais baixo", rmin, "f2", " dBm", "mínimo de min_rssi_valid entre as células", C.campo("[]", "min_rssi_valid"), f"{q(rmin, 2)} dBm ({cr1[0]})"),
           D("mais alto", rmax, "f2", " dBm", "máximo de min_rssi_valid entre as células", C.campo("[]", "min_rssi_valid"), f"{q(rmax, 2)} dBm ({cr2[0]})")])
    fb = {c: cel[c]["frac_valid_below110"] for c in CELULAS}
    f1, cf1 = argextremo(fb, min)
    f2, cf2 = argextremo(fb, max)
    linha("G", "G.validos_abaixo_110", "Fração de nós válidos com RSSI abaixo de −110 dBm: mínimo e máximo entre as 16 células",
          [D("mínimo", f1, "pct1", "", "mínimo de frac_valid_below110 entre as células", C.campo("[]", "frac_valid_below110"), f"{q(f1 * 100, 1)} % ({cf1[0]})"),
           D("máximo", f2, "pct1", "", "máximo de frac_valid_below110 entre as células", C.campo("[]", "frac_valid_below110"), f"{q(f2 * 100, 1)} % ({cf2[0]})")])
    nf = {c: cel[c]["notsent_far"] for c in CELULAS}
    n1, cn1 = argextremo(nf, min)
    n2, cn2 = argextremo(nf, max)
    linha("G", "G.validos_acima_30km",
          "Contexto, fora do pedido: nós válidos (não sentinela) a mais de 30 km do transmissor (campo notsent_far), mínimo e máximo entre as 16 células",
          [D("mínimo", n1, "int", "", "mínimo de notsent_far entre as células", C.campo("[]", "notsent_far"), f"{n1} ({cn1[0]})"),
           D("máximo", n2, "int", "", "máximo de notsent_far entre as células", C.campo("[]", "notsent_far"), f"{n2} ({cn2[0]})")],
          nota="o campo mostra que a implicação sentinela ⇒ distância > 30 km não é bicondicional; incluído só para registro")


# ============================================================================= BLOCO C
def bloco_C():
    bloco("C", "C — correções de fatos da folha antiga (sem número novo; texto fixo)",
          "CORREÇÃO DE FATO (texto fixo do protocolo; o protocolo não atribui estado de votos a este bloco)",
          "Texto fixo vindo do protocolo (passo 3 do roadmap v3-12); o script só confere a existência dos artefatos citados, o sha do corretor e os campos listados.")
    corretor = A("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/scripts/contrafactual_alvo_completo.py", "bin")
    conf("sha256 do corretor contrafactual_alvo_completo.py começa com ebeaf759", corretor.sha256.startswith("ebeaf759"), corretor.sha256)
    P1v = A("redacao/_pareceres_2026-10-01_t25/verificacao_P1_declive/veredito.json")
    A("redacao/_pareceres_2026-10-01_t25/verificacao_P1_declive/saida_teste.json")
    G45 = A("redacao/_pareceres_2026-10-01_t25/verificacao_G4_G5_celulas/veredito.json")
    A("redacao/_pareceres_2026-10-01_t25/verificacao_G4_G5_celulas/g5_saida.json")
    g1 = A("fase1/1.1_geo_celulas_16.json")
    pares = g1.g("sobreposicao_pares_mesma_cidade")
    conf("1.1: todos os pares da mesma cidade com interseção 0 km² e 0 nós coincidentes",
         len(pares) > 0 and all(p["intersecao_km2"] == 0 and p["n_nos_coincidentes"] == 0 for p in pares), len(pares))
    Hh = A("fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.json")
    B34 = A("fase4/verificacao_B3.4/P1_resultado.json")
    A("fase4/verificacao_B3.4/VEREDITO_verificacao_B3.4.md", "bin")

    linha("C", "C.N2_declive",
          "Linha N2 da folha antiga: o alvo gravado usa o declive local (corretor contrafactual_alvo_completo.py)",
          [Item("texto fixo", "o alvo gravado usa o declive local", "o alvo gravado usa o declive local (corretor contrafactual_alvo_completo.py, sha256 ebeaf759…)",
                corretor.path.as_posix() + " (sha256 " + corretor.sha256 + ")"),
           F(P1v, ["veredito"], "veredito.json (literal)", "raw")],
          nota="o veredito.json de verificacao_P1_declive grava o valor literal ao lado (esqueleto pendente); a evidência dos dois testes está em verificacao_P1_declive/saida_teste.json; nenhum número dela é copiado aqui")
    linha("C", "C.disjuntas",
          "'Células disjuntas' vale só dentro da cidade; entre cidades há sobreposição",
          [D("pares da mesma cidade avaliados, com interseção 0 km² e 0 nós coincidentes", len(pares), "int", "",
             "contagem de pares em sobreposicao_pares_mesma_cidade (todos com intersecao_km2 == 0 e n_nos_coincidentes == 0)",
             g1.campo("sobreposicao_pares_mesma_cidade", "[]", "intersecao_km2 / n_nos_coincidentes"), str(len(pares))),
           F(G45, ["G4", "veredito"], "verificacao_G4_G5_celulas/veredito.json, G4.veredito (literal)", "raw")],
          nota="sem percentuais por célula (as duas medidas de sobreposição divergem; nenhum percentual é copiado)")
    linha("C", "C.reticulado_permutacao",
          "As 16 células usam o mesmo reticulado e a mesma permutação por sorteio",
          [Item("texto fixo", "as 16 células usam o mesmo reticulado e a mesma permutação por sorteio",
                "as 16 células usam o mesmo reticulado e a mesma permutação por sorteio", G45.campo("G5", "veredito"))],
          nota="artefato: verificacao_G4_G5_celulas/veredito.json (G5) e saída g5_saida.json na mesma pasta; sem percentuais por célula")
    camposH = Hh.g("campos")
    idx_pen = [i for i, c in enumerate(camposH) if c["grupo"] == "perda" and ("(peso)" in c["campo"] or "restricao" in c["campo"])]
    conf("B7.2: quatro campos de penalidade/restrição no grupo perda (peso de gradiente de distância, variância, sombra × NDVI; restrição FSPL)", len(idx_pen) == 4, idx_pen)
    linha("C", "C.penalidades",
          "Quatro termos de penalidade na perda, um deles identicamente nulo",
          [F(Hh, ["campos", i, "campo"], f"termo de penalidade (H.{i + 1:02d})", "raw") for i in idx_pen]
          + [F(Hh, ["valor_maximo_shadowing_ndvi_penalty_registrado"], "sombra × NDVI, máximo registrado", "raw"),
             F(Hh, ["valor_maximo_constraint_fspl_registrado"], "restrição FSPL, máximo registrado", "raw")],
          nota="texto fixo do protocolo: quatro termos, um identicamente nulo; os máximos registrados são lidos de B7.2 (campos fora da lista dos 51)")
    linha("C", "C.mc_1_7pct",
          "A diferença de 1,7 % entre os dois Monte Carlo não se distingue de ruído de semente",
          [F(B34, ["s13_vs_s18", "diff_rel_pct"], "diferença relativa (MC 50 × 200)", "f2", " %"),
           F(B34, ["s13_vs_s18", "se_rel_pct"], "erro-padrão relativo", "f2", " %"),
           F(B34, ["s13_vs_s18", "z"], "z", "f2")],
          nota="texto fixo do protocolo: 'não se distingue de ruído de semente'; fonte fase4/verificacao_B3.4/ (P1_resultado.json e VEREDITO_verificacao_B3.4.md)")


# ============================================================================= montagem
def esc(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def md_celula_valor(itens: list[Item], campo: str) -> str:
    if len(itens) == 1:
        return esc(itens[0].texto if campo == "texto" else json.dumps(itens[0].cheio, ensure_ascii=False) if campo == "cheio" else "")
    partes = []
    for it in itens:
        v = it.texto if campo == "texto" else (json.dumps(it.cheio, ensure_ascii=False) if not isinstance(it.cheio, str) else it.cheio)
        if campo == "cheio" and len(v) > 400:
            v = v[:200] + " […] (completo no JSON espelho)"
        partes.append(f"{it.rotulo}: {v}")
    return esc("; ".join(partes))


def md_campos(itens: list[Item]) -> str:
    """Agrupa os campos por artefato: `artefato:campo1, campo2`."""
    grupos: dict[str, list[str]] = {}
    for it in itens:
        for campo in it.campo.split(" ÷ "):
            chave, resto = None, campo
            for rel in sorted(ARTEFATOS, key=len, reverse=True):
                if campo.startswith(rel + ":"):
                    chave, resto = rel, campo[len(rel) + 1:]
                    break
            grupos.setdefault(chave if chave else campo, [])
            if chave and resto not in grupos[chave]:
                grupos[chave].append(resto)
    partes = []
    for k, v in grupos.items():
        partes.append(f"`{k}:{', '.join(v)}`" if v else f"`{k}`")
    return esc("; ".join(partes))


def gerar_md(agora: str) -> str:
    L = []
    L.append("# FOLHA DE FATOS v3-12 — adendo (gerado por script, 01/10/2026)")
    L.append("")
    L.append("Gerada por `scripts/v3_12_gerar_folha_fatos.py` (sha256 do script no JSON espelho `FOLHA_DE_FATOS_v3-12_adendo.json`) a partir dos artefatos listados no espelho, cada um com o sha256. "
             "Nenhum número foi digitado: cada valor é lido de um JSON/CSV; onde há conta (mínimo, máximo, contagem, razão), a derivação consta no espelho. "
             "O script aborta se um artefato ou campo faltar. Não altera `FOLHA_DE_FATOS_v3_consolidada.md`; este adendo vale para a v3-12 e acrescenta ou corrige linhas dela.")
    L.append("")
    L.append("Colunas: `id | afirmação | valor para o texto | valor cheio | artefato:campo | estado | nota`. "
             "Arredondamento só na coluna \"valor para o texto\" (meio para cima sobre a representação decimal mais curta do número; vírgula decimal); o valor cheio é o do artefato. "
             "Raiz dos artefatos: `/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/` (caminhos abaixo relativos a ela, salvo os absolutos). "
             "\"<célula>\" e \"[]\" nos campos indicam iteração sobre as células ou sobre os elementos da lista.")
    L.append("")
    L.append("Regras de estado (da folha v3): CITÁVEL entra sem marcador; PROVISÓRIO entra com `% [[PROVISÓRIO: n votos]]` e fora do Abstract e das Conclusions.")
    L.append("")
    for letra in ["T", "U", "V", "W", "X", "Y", "Z", "H", "G", "C"]:
        b = BLOCOS[letra]
        n = b["linhas"]
        L.append(f"## {b['titulo']}")
        L.append("")
        L.append(f"**Estado do bloco:** {b['estado']}. **Linhas:** {n}" + (f" ({b['nota_contagem']})" if b.get("nota_contagem") else "") + ".")
        if b["nota"]:
            L.append("")
            L.append(b["nota"])
        L.append("")
        L.append("| id | afirmação | valor para o texto | valor cheio | artefato:campo | estado | nota |")
        L.append("|---|---|---|---|---|---|---|")
        for ln in LINHAS:
            if ln["bloco"] != letra:
                continue
            it = ln["itens"]
            L.append("| " + " | ".join([esc(ln["id"]), esc(ln["afirmacao"]), md_celula_valor(it, "texto"), md_celula_valor(it, "cheio"),
                                          md_campos(it), esc(ln["estado"]), esc(ln["nota"])]) + " |")
        L.append("")
    L.append(f"_Gerado em {agora}._")
    L.append("")
    return "\n".join(L)


def main():
    bloco_T()
    bloco_U()
    bloco_V()
    bloco_W()
    bloco_X()
    bloco_Y()
    bloco_Z()
    bloco_H()
    bloco_G()
    bloco_C()
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = gerar_md(agora)
    sha_script = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    espelho = {
        "id": "FOLHA_DE_FATOS_v3-12_adendo",
        "gerado_em_utc": agora,
        "script": str(SCRIPT),
        "script_sha256": sha_script,
        "venv": sys.executable,
        "raiz_artefatos": str(RAIZ),
        "protocolo": "_propostas_2026-10-01/ROADMAP_v3-12_major_revision.md, secao 6, passo 3 (va do dono, sessao principal, 01/10/2026 17:12)",
        "folha_existente_nao_alterada": "redacao/FOLHA_DE_FATOS_v3_consolidada.md",
        "arredondamento": "Decimal(repr(float)), ROUND_HALF_UP, so na coluna valor_texto; valor_cheio e o do artefato",
        "linhas_por_bloco": {k: BLOCOS[k]["linhas"] for k in BLOCOS},
        "total_linhas": len(LINHAS),
        "campos_nao_achados": [],
        "conferencias": CONFERENCIAS,
        "artefatos": {k: {"sha256": a.sha256, "bytes": a.bytes, "caminho_absoluto": str(a.path)} for k, a in sorted(ARTEFATOS.items())},
        "blocos": BLOCOS,
        "linhas": [{"bloco": l["bloco"], "id": l["id"], "afirmacao": l["afirmacao"], "estado": l["estado"], "nota": l["nota"],
                    "itens": [i.asdict() for i in l["itens"]]} for l in LINHAS],
        "detalhe": DETALHE,
    }
    SAIDA_MD.write_text(md, encoding="utf-8")
    SAIDA_JSON.write_text(json.dumps(espelho, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("OK", SAIDA_MD)
    print("OK", SAIDA_JSON)
    print("linhas por bloco:", json.dumps(espelho["linhas_por_bloco"]))
    print("total:", len(LINHAS), "| artefatos:", len(ARTEFATOS), "| conferencias:", len(CONFERENCIAS))


if __name__ == "__main__":
    main()
