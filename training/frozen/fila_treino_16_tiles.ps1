# =====================================================================
# TREINO C1 - 16 tiles x 5 seeds, sobre a linhagem corrigida.
#
# ISTO E O QUE OS REVISORES PEDIRAM. O ponto 1 do revisor 1, que o roadmap
# classifica como a lacuna mais critica: avaliacao com separacao espacial em
# blocos de 10 km e buffer de 2 km, teste jamais usado na selecao. E o unico
# item do parecer que exige treino longo.
#
# O QUE MUDOU NO DADO ATE CHEGAR AQUI (tudo medido, artefatos em dados/alvo/):
#  - mosaico DEM refeito: o antigo devolvia 0,0 em 100% dos pontos de
#    sorocaba Q3/Q4, ~90% de campinas Q3/Q4, ~50% de sorocaba Q1/Q2 e 32,5%
#    de bauru Q3/Q4;
#  - 10 tiles regerados dessa base, com seed 42 no A_terrain;
#  - alvo com os 3 defeitos corrigidos nos 16: atenuacao de terreno vinda do
#    relevo em vez de sorteio, coluna certa na difracao (era TPI lido como
#    TRI, o que anulava o termo) e unidades consistentes em h_obs;
#  - portao de integridade: 15 dos 16 aprovados; sorocaba_Q1 reprova por 9
#    nos sem banda espectral em 12,96 M - decisao de criterio pendente.
#
# CADA TILE USA A PASTA ONDE ELE ESTA. lins e bauru Q1/Q2 em graph_data;
# campinas, sorocaba e bauru Q3/Q4 em graph_data_v3. Os dois discos nao
# aceitam hardlink entre si, e nao e preciso: o programa recebe uma pasta por
# execucao, e em cada pasta o dataset e o grafo ja convivem.
#
# PROTOCOLO, identico ao das corridas anteriores para permitir comparacao:
# grid 10 km, buffer 2 km, split-seed 42, split 70/15/15, 900 MHz na perda e
# 1800 MHz no diagnostico, transferencia encadeada Q1->Q4 dentro da cidade,
# 35 epocas no Q1 e 20 nos demais.
#
# ROTULO c0c1cf_* - nao colide com c0c1_* (v1, 01-02/09) nem com c0c1v2_*
# (lins, 03-05/09).
#
# CUSTO: ~40 min por corrida x 16 tiles x 5 seeds = ~53 h. Nao cabe numa
# noite. A fila e retomavel: pula run ja concluido (com custo.tempo_total_s
# gravado, nao so o JSON, que nasce no inicio).
# =====================================================================
param(
    [int[]] $Seeds = @(42, 43, 44, 45, 46),
    [string[]] $Cidades = @("lins", "bauru", "campinas", "sorocaba")
)
$ErrorActionPreference = "Stop"

$PY   = "F:\S33_dslm\s33_amb_virtual\.venv\Scripts\python.exe"
$BASE = "D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO"
$S    = Join-Path $BASE "scripts"
$TREI = Join-Path $BASE "treinos"
$CONG = Join-Path $BASE "dados\scripts_congelados"
$GD   = "D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\graph_data"
$V3   = "F:\TOPO_RF_DOWNLOAD_DRIVE\graph_data_v3"
$LOG  = Join-Path $TREI "fila_treino_16.log"
$VERD = Join-Path $BASE "VERDITO_TREINO_16.md"

$estado = New-Object System.Collections.Specialized.OrderedDictionary
function L($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content -Encoding utf8 $LOG }
function Marcar($k, $v) { $estado[$k] = $v; L "  [$k] $v" }

function Concluido($runJson) {
    if (-not (Test-Path $runJson)) { return $false }
    try {
        $d = Get-Content $runJson -Raw | ConvertFrom-Json
        return ($null -ne $d.custo.tempo_total_s)
    } catch { return $false }
}

# devolve @(pasta, nome_do_dataset) do tile, ou $null
function Localizar($c, $q) {
    # SO o caminho novo. A versao anterior caia no `_enriched_v2_cftudo.pt` de
    # graph_data quando nao achava o regerado - e aquele veio do processo
    # antigo, que apenas COPIAVA as features do v1 publicado. Misturar os dois
    # foi o motivo de o dono mandar interromper o treino de 07/09: em
    # bauru_Q3, o unico tile que passou pelos dois caminhos, as derivadas do
    # terreno coincidiam em 2,3% a 2,6%. Faltando o regerado, e melhor a fila
    # abortar aquele tile do que treinar linhagem mista em silencio.
    $a = "$V3\transfer_dataset_${c}_v19_${q}_enriched_cftudo.pt"
    if ((Test-Path $a) -and ((Get-Item $a).Length -gt 20e9)) {
        return @($V3, (Split-Path $a -Leaf))
    }
    return $null
}

function Treinar($c, $q, $seed, $prev, $ep, $lr) {
    $label = "c0c1cf_${c}_s${seed}_${q}_g10b2"
    $runJson = Join-Path $TREI "$label\run_$label.json"
    $ckpt = Join-Path $TREI "$label\checkpoints\checkpoint_best.pt"
    if (Concluido $runJson) { L "pula $label (concluido)"; return $ckpt }
    if (Test-Path $runJson) { L "$label tem JSON incompleto - refazendo" }

    $loc = Localizar $c $q
    if (-not $loc) { L "ABORTA $label : dataset com alvo corrigido nao encontrado"; return $null }
    $pasta = $loc[0]; $rf = $loc[1]
    $gpu = "${c}_v19_${q}_gpu.pt"
    if (-not (Test-Path (Join-Path $pasta $gpu))) { L "ABORTA $label : falta $gpu em $pasta"; return $null }

    $argv = @((Join-Path $CONG "train_gnn_c0_spatial.py"),
              "--run-label", $label,
              "--rf-data-file", $rf,
              "--graph-file", $gpu,
              "--graph-dir", $pasta,
              "--evid-dir", $TREI,
              "--epochs", $ep, "--lr", $lr,
              "--seed", $seed, "--split-seed", 42,
              "--grid-km", 10.0, "--buffer-km", 2.0,
              "--split-frac", "0.70,0.15,0.15",
              "--freq-mhz", 900, "--diag-freq-mhz", 1800, "--hash")
    if ($prev) { $argv = $argv + @("--transfer-from", $prev) }

    L "$ $label  (pasta: $(Split-Path $pasta -Leaf))"
    $t0 = Get-Date
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PY -u @argv *>> $LOG
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $eap
    $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    L "   rc=$rc em $min min"

    if ($rc -ne 0) { Marcar $label "FALHOU rc=$rc"; return $null }
    Marcar $label "ok em $min min"
    return $ckpt
}

L "===== fila_treino_16 iniciada (pid $PID) ====="
L ("seeds: " + ($Seeds -join ", ") + " | cidades: " + ($Cidades -join ", "))

foreach ($seed in $Seeds) {
    foreach ($c in $Cidades) {
        L "--- $c seed $seed ---"
        $prev = ""
        foreach ($q in @("Q1", "Q2", "Q3", "Q4")) {
            if ($q -eq "Q1") { $ep = 35; $lr = "1e-3" } else { $ep = 20; $lr = "2e-4" }
            $ck = Treinar $c $q $seed $prev $ep $lr
            if (-not $ck) { L "[PARE] cadeia $c s$seed interrompida em $q"; break }
            $prev = $ck
        }
    }
}

# --- agregacao por cidade ---
foreach ($c in $Cidades) {
    $rc = 1
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PY -u (Join-Path $S "agregar_c1_multiseed_v2.py") --cidade $c --prefixo "c0c1cf" --parcial *>> $LOG
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $eap
    Marcar "agregado_$c" $(if ($rc -eq 0) { "ok" } else { "rc=$rc" })
}

# --- veredito ---
$linhas = New-Object System.Collections.Generic.List[string]
$linhas.Add('# Veredito - treino C1 sobre a linhagem corrigida')
$linhas.Add('')
$linhas.Add('Gerado em ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '. Log: ' + $LOG)
$linhas.Add('')
$linhas.Add('| corrida | resultado |')
$linhas.Add('|---|---|')
foreach ($k in $estado.Keys) { $linhas.Add("| $k | $($estado[$k]) |") }
$linhas.Add('')
$linhas.Add('## Runs gravados')
$linhas.Add('')
$linhas.Add('| run | json | checkpoint |')
$linhas.Add('|---|---|---|')
foreach ($r in @(Get-ChildItem $TREI -Directory -Filter "c0c1cf_*" -EA SilentlyContinue | Sort-Object Name)) {
    $j = Join-Path $r.FullName "run_$($r.Name).json"
    $ck = Join-Path $r.FullName "checkpoints\checkpoint_best.pt"
    if (Test-Path $j) { $tj = 'sim' } else { $tj = 'NAO' }
    if (Test-Path $ck) { $tc = 'sim' } else { $tc = 'NAO' }
    $linhas.Add("| $($r.Name) | $tj | $tc |")
}
$linhas.Add('')
$linhas.Add('## Limitacoes que nao se resolvem com este treino')
$linhas.Add('')
$linhas.Add('- 84-93% do alvo de perda e o sentinela de 300 dB, marcador de')
$linhas.Add('  "fora do alcance", nao medida - em todos os 16 tiles.')
$linhas.Add('- ha nos com perda menor que a de espaco livre da propria distancia.')
$linhas.Add('- distancia mediana a antena: 6,5-9,5 km em lins e bauru contra')
$linhas.Add('  68-75 km em campinas e sorocaba. Regimes diferentes.')
$linhas.Add('- falta a comparacao contra um modelo sem estrutura de grafo; como o')
$linhas.Add('  alvo e calculado ponto a ponto, um modelo simples deveria empatar.')
$linhas.Add('- sorocaba_Q1 entrou com 9 nos sem banda espectral (7e-07 do tile).')
$linhas.Add('')
$linhas.Add('Nenhum numero daqui e citavel antes de contra-auditoria por quem nao')
$linhas.Add('produziu a corrida.')

Set-Content -Path $VERD -Value ($linhas -join "`r`n") -Encoding utf8
L "veredito: $VERD"
L "===== fila_treino_16 concluida ====="
