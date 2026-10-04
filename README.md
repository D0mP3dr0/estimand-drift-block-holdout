# Estimand Drift under Spatial Block Hold-Out: reproducibility package

Code, criteria, result records, tables, figures, and artifact manifest for the
manuscript *Estimand Drift under Spatial Block Hold-Out* by L. F. C.
Seelig and R. M. Salles, manuscript submitted to MDPI Mathematics (revision of 3 October 2026).

The study works on 16 city x quadrant cells of a radio-frequency (RF) field
reconstruction task (Bauru, Campinas, Lins, Sorocaba; quadrants Q1-Q4) and asks
what a buffered spatial block hold-out actually estimates: how many nodes the
buffer removes from the test partition, how the surviving test population
differs from the sampling population, how the error of constant and analytical
baselines drifts with the split, and what that does to a model comparison. The
repository holds the generator and training code the manuscript cites by digest,
the pre-registered criteria, the numerical result records, and the scripts that
turn those records into every table and figure.

## What is here

- `generator/` - the reference-field generator and the target corrector cited in
  the manuscript by SHA-256 digest: `generate_realistic_coverage.py` (95ea0423),
  `prepare_transfer_dataset_v19.py` (45c16d43), `enrich_rf_targets.py`
  (5d38012e), `contrafactual_alvo_completo.py` (ebeaf759, the generator
  self-consistency check). `generator/graph_build/` holds the graph builders the
  pipeline depends on (`generate_graph_v19.py`, `split_v19_quadrants.py`,
  `gerar_grafo_v19_mosaico_corrigido.py`, `preparar_transfer_seed42.py`).
- `partition/` - `spatial_cv.py` (the buffered spatial block partition) and the
  `config/` package it imports.
- `training/` - `frozen/train_gnn_c0_spatial.py` (the frozen trainer, digest
  6f955629), the frozen graph-free control `frozen/train_mlp_c0_spatial.py`
  (loaded by `modelo_v3/v3_common.py`), the PowerShell queue
  `frozen/fila_treino_16_tiles.ps1`, the modules the trainer imports
  (`models/`, `rf_diagnostic_metrics.py`, `baselines/empirical_models.py`), and
  `modelo_v3/` with the training and evaluation code of the third campaign
  (`train_gnn_v3.py`, `train_mlp_v3.py`, `v3_common.py`, `rf_decoder_v3.py`,
  `e3_inferencia_checkpoints_antigos.py`). `G1/` holds the shim and the three launchers of
  the 10 km campaign (`train_v3_g10.py`, `rodar_lote_G1.py`, `rodar_lote_G1_retomada.py`,
  `rodar_lote_G1_bloco4.py`).
- `analysis/` - the scripts of the article: one script per numerical test
  (`v3_*.py`), the revision-round scripts (`v3_12_*.py`), the table, figure, and
  fact-sheet generators, the earlier Monte Carlo and sweep modules the tests
  import (`varredura_*.py`, `montecarlo_*.py`, `r3_*.py`, ...), and
  `analysis/producers/` with the small scripts that produced target statistics
  and negative controls.
- `criteria/` - the decision criteria (`criterio_*.json`, `MANIFEST_criterios.jsonl`),
  written before the corresponding runs.
- `results/` - JSON, CSV, and the small driver scripts of each phase:
  `fase1/` (geometry, retention, nodal references), `fase2/` (inclusion
  probabilities, error drift, estimands), `fase3/` (calibration, edge bands,
  FSPL coincidence, feature columns), `fase4/` (b = 0 drift, design reference,
  design term, closed-form p versus frequency, hyper-parameter sheet, and the
  verification and vote folders with their scripts), `fase5/` (checks R5 to R7),
  `gpu/` (aggregates and per-run JSON/CSV of the GNN and MLP campaigns A0-A4
  with their drivers, and the G1 campaign at 10 km), `verificacoes/` (two verification
  outputs), and `inputs/` (the one data file consumed by the figure script).
- `tables_figures/` - `tables_v3-12/` (LaTeX tabulars, CSVs, manifest and checks
  of every table), `figures_v3-12/` (PDF/PNG, manifest, check of the F4 figure),
  `suplementar/` (Table S1, Table S2, and Table S3, sources and PDFs), and
  `prior_version_csv/` (four CSVs of the previous table set, read by the table
  script as cross-checks).
- `manifest/` - `manifest_mathematics_v5.jsonl` and `.meta.json` (file, size, and
  SHA-256 of every artifact, including those not distributed here), the
  validator output, and the manifest generator and validator scripts.
- `SHA256SUMS.txt` - SHA-256 of every file in this repository.
- `environment.txt` - the frozen package list of the environment.
- `LICENSE` (MIT, code) and `LICENSE-DATA` (CC BY 4.0, results and tables).

## What is not here

- The reference fields (`*_cftudo.pt`), the graph tensors (`*_gpu.pt`), the
  per-node prediction files (`*.npz`), and all model checkpoints (`*.pt`) are not
  in the repository: they are large (the reference fields alone are of the order
  of gigabytes each). They are **available from the corresponding author upon
  request**. Their file names, sizes, and SHA-256 digests are listed in
  `manifest/manifest_mathematics_v5.jsonl` (groups `tensores_cftudo`,
  `grafos_gpu_arestas`, `checkpoints_treinos_evidencia`, `v3_gpu_checkpoints_pt`,
  `v3_gpu_predicoes_npz`), so that any copy obtained separately can be verified
  with `sha256sum`.
- The manuscript source and PDF, review material, and internal working notes are
  not part of this package.
- Training logs (`*.log`, `*.txt`), Markdown notes, diffs, and LaTeX
  intermediates were left out, except the two G1 driver logs and the G1 aggregator diffs
  (see "Revision of 3 October 2026"); the JSON records carry the provenance fields that
  matter (script and input digests).
- No file above the 5 MB per-file limit was found among the candidates; none was left out for size.

## Revision of 3 October 2026

This revision adds the material behind the model campaign at a block size of 10 km and
three robustness checks of the revised manuscript. Nothing from the earlier package was
removed; the files listed below were added and the table, manifest, and checksum files
were regenerated.

- **Campaign G1** (`results/gpu/G1/`, `training/G1/`). 220 trained runs (110 GNN and
  110 MLP) at g = 10 km and b = 2 km in four cells of two cities (Bauru Q1 and Q3, Campinas
  Q1 and Q3), 20 split draws per cell at training seed 42, plus a crossed block of training
  seeds 42 to 44 in the two Q1 cells (5 draws in Bauru Q1, 10 draws in Campinas Q1), for
  the GNN and the MLP. The folder holds the plans, the launcher status and driver logs
  (`lote_G1_*`), the gate and the script provenance records, the four aggregates
  (`agregado_G1_v5_bloco1.json`, `..._v5_bloco2.json`, `..._v8_bloco3.json`,
  `..._v10_bloco4.json`), the run manifests (`MANIFEST_G1*.jsonl`), and, per run, the run
  record `run_<label>.json` and the shim record `shim_g10_<label>.json`. The recomputation
  votes are in `results/gpu/G1_votos_bloco1` to `bloco4`, the manifest builders and tensor
  checks in `results/gpu/G1_manifest/`, and the aggregator tests and counter-audits in
  `results/gpu/G1_agregador_v*`. The shim and the three launchers are in `training/G1/`; the
  aggregators (`v3_G1_agregar.py` and versions v2 to v10, with the step-by-step diffs in
  `analysis/diffs/`) are in `analysis/`. The pre-registered criteria are
  `criteria/criterio_G1_modelos_g10.json`, `criterio_G1_adendo1.json` to `adendo4.json`, and
  `criterio_G1b_campinas_bloco_cruzado.json`.
- **Checks R5, R6, R7** (`results/fase5/`). R5: variance of a single draw estimated from its
  own test blocks; R6: whether the between-draw ratio depends on the level of the constant
  predictor; R7: between-draw versus between-seed spread in the earlier 5 km runs. Scripts
  `analysis/v3_13_*.py` (`v3_13_laco_comum.py` is the loop shared by R5 and R6); the
  recomputation votes are in `results/fase5/votos_R5`, `votos_R6`, `votos_R7`; criteria in
  `criteria/criterio_R5_*`, `criterio_R6_*`, `criterio_R7_*`.
- **Tables.** Table 7 of the manuscript (file prefix T9, `T9_deriva_modelo_G1_v3-12.{tex,csv}`;
  written by `analysis/v3_12h_gerar_T9_G1.py`, to be run after `v3_12_gerar_tabelas.py`) and
  Table S3 (`tables_figures/suplementar/Table_S3.{tex,pdf}`, built from the T7b fragment).
  T3 summary, TN, T6, T8, `CONFERENCIAS_tabelas_v3-12.json`, `Table_S1`, and the table generator
  were regenerated. `results/verificacoes/` holds the outputs of the two checks that source the
  0.132 dB repeat difference (`saida_V2.json`) and the 30 km fraction (`saida_c.json`).
- **Manifest.** `manifest/manifest_mathematics_v5.jsonl` was regenerated with
  `scripts/gerar_manifest_mathematics_v5.py` (now with the G1 files) and revalidated; its
  `.meta.json` and the validator output are alongside.

**Available on request.** The prediction files (`predicoes_*.npz`), the checkpoints
(`checkpoints/`), the per-run `training_log.csv`, and the per-run training logs
(`log_g1_*.txt`) of the G1 runs are not in this repository because of their size. Their
names, sizes, and SHA-256 digests are in `manifest/manifest_mathematics_v5.jsonl` (groups
`v3_gpu_predicoes_npz`, `v3_gpu_checkpoints_pt`, `v3_gpu_agregados_scripts_logs`), and the
corresponding author supplies them upon request. 10 G1 runs have no shim record in the
source folder and none is shown for them.
- Test E1 (added later on 3 October): block K-fold scored jointly against the block hold-out, criterion `criteria/criterio_E1_kfold_conjunto.json`, records `results/fase5/E1_*.json`, independent recalculation `results/fase5/votos_E1/`, script `analysis/v3_13_E1_kfold_conjunto.py`. The fact sheet of the revision (`results/FOLHA_DE_FATOS_v3-12_adendo3.md`, generated by `analysis/v3_13_folha_adendo3.py`) maps every number of the revised sections to its record.

## Environment

Python 3.11.16, PyTorch 2.10.0+cu128 (CUDA 12.8), PyTorch Geometric 2.8.0.post1,
NumPy 2.4.6, SciPy 1.17.0, pandas 2.3.3, matplotlib 3.10.8, SymPy 1.14.0,
scikit-learn 1.8.0, on a single NVIDIA GPU for the training campaigns. The
analysis scripts of `analysis/` run on CPU. The complete package list of the
environment used by the authors is in `environment.txt`; install it with
`pip install -r environment.txt` (the CUDA build of PyTorch must come from the
matching wheel index).

## Absolute paths in the scripts (read this before running anything)

The scripts were written and run on the authors' machine and carry absolute
paths of that machine; 859 files in this repository contain such paths.
Nothing was rewritten, so that the SHA-256 of every file still matches the
manifest and the digests cited in the manuscript. To run a script elsewhere,
adjust the following before executing:

- `analysis/*.py` and the scripts under `results/`: the constants that point at
  the project tree (typically `B`, `V3`, `MDPI`, `OUT`, `OUT_DIR`, `OUT_JSON`,
  `HERE`, and the literal prefix
  `/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25`).
  Outputs are written to those locations, not to the folders of this repository.
- `training/modelo_v3/v3_common.py`: `GNN_RF_V2` (the tree that holds
  `01_data`, `02_models`, `03_training`, `04_baselines`, `config`, `data_raw`),
  `GRAPH_DIR_DEFAULT` (directory with the `*_enriched_cftudo.pt` and `*_gpu.pt`
  tensors), `MANIFEST_V4_PATH` (the loader expects the v4 manifest; the v5
  manifest in `manifest/` carries the same `tensores_cftudo` rows, but that
  substitution was not tested), and `FROZEN_SCRIPTS_DIR` (the folder with the two
  frozen trainers in `training/frozen/`).
- `training/frozen/train_gnn_c0_spatial.py`: the `--base-dir` argument, which
  must point at a tree with the original module layout. The modules were
  gathered here by role: `02_models` -> `training/models/`, `03_training` ->
  `partition/spatial_cv.py` and `training/rf_diagnostic_metrics.py`,
  `04_baselines` -> `training/baselines/`, `config` -> `partition/config/`,
  `data_raw` and `01_data` -> `generator/`. Recreate the original layout (copies
  or symbolic links) to run the trainer unchanged.
- `tables_figures/` scripts: the table and figure scripts read result files from
  `fase1`-`fase4` and `gpu` and write to `redacao_v3-12/tables_v3-12/` and
  `redacao_v3-12/figures_v3-12/` under the project tree. The sources of Table S1
  and Table S2 `\input` the table fragments with the relative path
  `../tables_v3-12/`, which is preserved here (`tables_figures/suplementar/` and
  `tables_figures/tables_v3-12/` are siblings).
- One input of the table script is not distributed: the LaTeX block with the
  notation table, read to build table TN. TN therefore cannot be regenerated from
  this repository; its tabular and CSV are provided. The input of the F3 and F34
  figures is shipped as `results/inputs/fracao_valida_val_treino_20_splits.json`.

## How to reproduce each table and figure

Commands assume the environment above and the data obtained on request. `python`
denotes the environment's interpreter; run from the directory that holds the
scripts after fixing the paths. Every JSON record stores the command that
produced it (`comando_rodado` / `comando`) and the SHA-256 of the script and the
inputs, which is the authoritative record; the commands below were taken from
those fields and from the script headers. Table and figure labels (T2, F3, ...)
are the file prefixes in `tables_figures/`.

**Numerical tests (inputs of the tables).**

| Test | Command | Output |
|---|---|---|
| Geometry of the 16 cells, buffer-violation chain (1.1, 1.2) | `python analysis/v3_1.1_1.2_celulas_cadeia.py --out-11 <out> --out-12 <out> --fig <fig>` | `results/fase1/1.1_geo_celulas_16.json`, `1.2_cadeia_buffer_violacao.json` (and the cell map figure, not shipped) |
| Nodal Monte Carlo versus block model (1.3) | `python analysis/v3_1.3_mc_nodal_vs_blocos.py --out fase1/1.3_mc_nodal_vs_blocos.json` | `results/fase1/1.3_mc_nodal_vs_blocos.json` |
| Sweep of b, law versus nodal (1.4) | `python analysis/v3_1.4_varredura_b_lei_vs_nodal.py --out <out> --seeds 20` | `results/fase1/1.4_varredura_b_lei_vs_nodal.json` |
| Exact retention versus Monte Carlo (1.5, 1.5b) | `python analysis/v3_1.5_retencao_exata_vs_mc.py --out fase1/1.5_retencao_exata_vs_mc.json`; `python analysis/v3_1.5b_exata_vs_nodal200.py` | `results/fase1/1.5_*.json`, `1.5b_exata_vs_nodal200.json` |
| Degree preservation (1.7, with the 1.7b erratum) | `python analysis/v3_1.7b_grau_preservacao_errata.py --out <out> --celulas 4 --seeds 5` | `results/fase1/1.7_grau_preservacao.json` |
| Nodal reference, 200 seeds (1.8, 1.8b) | `N_SEEDS=200 python analysis/v3_1.8_referencia_nodal_200seeds.py`; `N_SEEDS=200 python analysis/v3_1.8b_referencia_nodal_200seeds_g5.py` | `results/fase1/1.8_*.json`, `1.8b_*.json` |
| Inclusion probabilities, estimands (1.6, 2.3, 2.3b) | `python analysis/v3_1.6_2.3_estimando_formal.py`; `python analysis/v3_2.3b_por_sorteio.py` | `results/fase2/1.6_*.json`, `2.3_*.json`, `2.3b_*.json` |
| Error drift and calibration (2.1, 3.1) | `python analysis/v3_2.1_3.1_deriva_calibracao.py --seeds aleatorios:60:20260926 --sufixo-saida _16x60rnd --sem-sha` | `results/fase2/2.1_deriva_erro_baselines_16x60rnd.json`, `results/fase3/3.1_calibracao_validos_vs_mediana_16x60rnd.json` |
| Edge band p7 at real borders (3.2) | `python analysis/v3_3.2_banda_p7_bordas_reais.py --out <out>` | `results/fase3/3.2_banda_p7_bordas_reais.json` |
| FSPL coincidence (3.4) | `python analysis/v3_3.4_coincidencia_fspl.py --out <out>` | `results/fase3/3.4_coincidencia_fspl.json` |
| Feature columns (3.9), per-cell decomposition (3.7b), further calculations (3.7, 3.8) | `python analysis/v3_3.9_colunas_lidar.py`; `v3_3.7b_decomposicao_por_celula.py`; `v3_resposta_cego_calculos.py`; `v3_resposta_cego_G1G2.py` | `results/fase3/3.9_*.json`, `3.7b_*.json`, `3.7_*.json`, `3.8_*.json` |
| b = 0 drift (R1) | `python analysis/v3_12_R1_b0_deriva.py --celula <cell>` for the 16 cells, then `--consolidar` (`--validar` re-derives the b = 2 totals) | `results/fase4/R1_b0_por_sorteio.json`, `R1_b0_resumo.json` |
| Design reference (R2) | `python analysis/v3_12_R2_referencia_desenho.py` | `results/fase4/R2_por_sorteio.json`, `R2_resumo.json` |
| Design term (R3) | `python analysis/v3_12_R3_termo_desenho.py` | `results/fase4/R3_termo_desenho.json` |
| Closed-form p versus frequency (R4) | `python analysis/v3_12_R4_p_fechado_vs_frequencia.py` | `results/fase4/R4_p_por_classe.json` |
| Hyper-parameter sheet (B7.2) | `python results/fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.py` | `results/fase4/B7.2_hiperparametros/b7_2_folha_hiperparametros.{json,csv}` |
| Model campaigns A3 and A4 (needs GPU and the tensors) | drivers `results/gpu/A3/rodar_lote_A3.py`, `results/gpu/A4/rodar_lote_A4.py`, which call `training/modelo_v3/train_gnn_v3.py` and `train_mlp_v3.py`; then `python analysis/v3_A3_agregar.py`, `python analysis/v3_A4_agregar.py`; `python analysis/v3_A4_suplemento_inversao_validos.py` | `results/gpu/A3/agregado_A3.json`, `results/gpu/A4/agregado_A4.json`, `results/gpu/A4/suplemento_inversao_validos_A4.json` |

**Tables.** All tables of the article are produced by one script,
`python analysis/v3_12_gerar_tabelas.py`, which reads the named JSON records
above, checks several totals before writing, and writes the `.tex`/`.csv` pairs
plus `CONFERENCIAS_tabelas_v3-12.json` (the checks) and
`MANIFEST_tabelas_v3-12.json` (inputs, script, and outputs with SHA-256; this
manifest is regenerated by the script and is not shipped, because a secret
scanner flagged its path-plus-digest lines as a false positive). The result ->
table map is:

| Table | Content | Source records | Output (in `tables_figures/tables_v3-12/`) |
|---|---|---|---|
| T2 | retention by cell | `fase1/1.8`, `1.8b` | `T2_retencao_v3-12.{tex,csv}` |
| T3 (and summary) | error drift of the constant predictor | `fase2/2.1_..16x60rnd`, `fase2/_v3_2.1_3.1_parcial_16x60rnd`, `fase4/R1_b0_*` | `T3_deriva_erro_v3-12`, `T3_resumo_v3-12` |
| T4 (totals, S2) | inversion counts, with and without the buffer | `fase2/_v3_2.1_3.1_parcial_16x60rnd`, `fase4/R1_b0_*` | `T4_inversao_v3-12`, `T4_totais_v3-12`, `T4_S2_v3-12` (Table S2) |
| T5 | inclusion by class | `fase4/R4_p_por_classe.json` | `T5_inclusao_por_classe_v3-12` |
| T6 | reference by design | `fase4/R2_*`, `fase4/votos_fismat_R2_R4/` | `T6_referencia_desenho_v3-12` |
| T7, T7b | model drift (campaign A4) | `gpu/A4/agregado_A4.json` | `T7_deriva_modelo_A4_v3-12`, `T7b_...` |
| T9 (Table 7) | model drift at 10 km (campaign G1) | `gpu/G1/agregado_G1_v5_bloco1.json`, `..._v5_bloco2.json`, `..._v8_bloco3.json`, `..._v10_bloco4.json`, `gpu/G1_votos_bloco4/veredito.json` | `T9_deriva_modelo_G1_v3-12` (by `analysis/v3_12h_gerar_T9_G1.py`) |
| Table S3 | model drift at 5 km (T7b fragment) | `T7b_deriva_modelo_A4_v3-12.tex` | `tables_figures/suplementar/Table_S3.{tex,pdf}` |
| T8 | design term | `fase4/R3_termo_desenho.json` | `T8_termo_desenho_v3-12` |
| TA1 (Table S1) | hyper-parameters | `fase4/B7.2_hiperparametros/` | `TA1_hiperparametros_v3-12`, `TA1a_...`, `TA1b_...` |
| TN | notation | notation block of the manuscript (not distributed) | `TN_notacao_v3-12` |

Table S1, Table S2, and Table S3 are compiled with `pdflatex Table_S1.tex`,
`pdflatex Table_S2.tex`, and `pdflatex Table_S3.tex` inside `tables_figures/suplementar/`. The script
`tables_figures/tables_v3-12/_pipeline/validar_valores_csv.py` re-checks the
values of the CSVs against the records.

**Figures.** Run `python analysis/v3_12_fig1_block_erosion.py` first
(`fig1_block_erosion`, synthetic geometry with g = 10 and b = 2, no data), then
`python analysis/v3_12_gerar_figuras.py` (`F3_...`, `F4_...`, and the two-panel
`F34_...`; it needs `results/fase2/_v3_2.1_3.1_parcial_16x60rnd.json`,
`results/fase2/2.1_deriva_erro_baselines_16x60rnd.json`,
`results/inputs/fracao_valida_val_treino_20_splits.json`, and the T3 CSV, and
writes `MANIFEST_figuras_v3-12.json` and the check `F4_conferencia_T3.json`).

**Fact sheet.** `python analysis/v3_12_gerar_folha_fatos.py` gathers the
numerical facts of the article from the records; its output is not part of this
package.

**Re-checking the artifacts.** `python manifest/scripts/validar_manifest_mathematics_v5.py`
recomputes the digests listed in `manifest/manifest_mathematics_v5.jsonl` against
the files of the authors' tree; with the data obtained on request placed at the
listed locations it verifies each file, and `sha256sum` on a single file can be
compared with its line in the manifest.

## Digests cited in the manuscript

| Prefix | Module | Path in this repository |
|---|---|---|
| 95ea0423 | `generate_realistic_coverage.py` | `generator/generate_realistic_coverage.py` |
| 45c16d43 | `prepare_transfer_dataset_v19.py` | `generator/prepare_transfer_dataset_v19.py` |
| 5d38012e | `enrich_rf_targets.py` | `generator/enrich_rf_targets.py` |
| ebeaf759 | `contrafactual_alvo_completo.py` | `generator/contrafactual_alvo_completo.py` |
| 6f955629 | `train_gnn_c0_spatial.py` (frozen trainer) | `training/frozen/train_gnn_c0_spatial.py` |

Check with `sha256sum <file> | cut -c1-8`.

## Data availability

The reference fields, graph tensors, per-node predictions, and checkpoints are
available from the corresponding author upon request. File names and SHA-256
digests are in `manifest/manifest_mathematics_v5.jsonl`.

## License

Code is released under the MIT License (`LICENSE`). Results, criteria, tables,
figures, and the manifest are released under CC BY 4.0 (`LICENSE-DATA`).

## Citation

The manuscript was submitted to MDPI Mathematics (revision of 3 October 2026). See `CITATION.cff`.
