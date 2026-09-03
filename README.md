Python source code used for FDC-zone assignment, consolidation of Imaris exports, trajectory and feature preprocessing, UMAP/clustering, and integration of single-cell RNA-sequencing data with the expanded cell-behavior space. The core calculations and selected analysis parameters from the original workflow are retained in organized scripts.

## System requirements

- Tested operating system: Microsoft Windows
- Tested Python: 3.9.25 (Conda environment `py39`)
- No non-standard hardware is required for the included demonstrations.
- GPU/OpenCL-capable hardware may accelerate image processing with `pyclesperanto-prototype`, but was not tested as part of this package.
- Imaris is third-party software used upstream to create CSV exports consumed by `02_merge_imaris_exports.py`; Imaris is not distributed here.

The tested package versions are recorded in `environment.yml`. On a normal desktop computer, creating the complete environment may take approximately 10-30 minutes, depending on network speed. This estimate has not been independently benchmarked on a clean computer.

## Installation

Install Miniconda or Anaconda, open a terminal in this folder, and run:

```shell
conda env create -f environment.yml
conda activate gc-behavior-analysis
```

If the solver cannot reproduce an older package combination on a non-Windows system, install Python 3.9 and the versions listed in `environment.yml` using conda/pip as appropriate.

## Demonstrations

Two scientific demonstrations use stratified subsets of the real manuscript data:

```shell
python demo/01_motility_space/run_motility_demo.py
python demo/02_scmotiph_integration/run_scmotiph_demo.py
```

Demo 1 reconstructs a nine-cluster motility UMAP from 450 observations sampled from `GCB_no_inhibit_all_features_20.parquet`. Demo 2 integrates 450 scRNA-seq cells and 450 expanded-behavior observations using the manuscript's label- and motility-informed entropic fused Gromov-Wasserstein approach, then shows both sources in a merged UMAP. Each demo directory contains its real-data subset, detailed instructions, and expected plots and tables.

## Main analysis workflow

The scripts reflect the original analysis workflow and execute from top to bottom:

1. `code/gc_analysis/01_set_FDC_zones.py` assigns image-derived FDC zones.
2. `code/gc_analysis/02_merge_imaris_exports.py` consolidates Imaris CSV exports.
3. `code/gc_analysis/03_preprocess_features.py` constructs trajectories and calculates motility and interaction features.
4. `code/gc_analysis/04_construct_motility_space.py` performs feature scaling, PCA, nine-cluster K-means, and UMAP.
5. `code/gc_analysis/05_scmotiph_integration.py` performs the fixed-parameter scRNA-seq/expanded-behavior FGW integration and exports the merged UMAP.

Before running the full workflow, replace the JHU network paths in the analysis scripts with local paths to the corresponding deposited data. The required input categories are listed in `DATA_AND_PATHS.md`. The full workflow cannot run using this code folder alone unless the manuscript datasets are downloaded separately and those paths are updated.

Run the full-workflow scripts from the package root with `code/` on the Python module search path. In PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path code).Path
python code/gc_analysis/01_set_FDC_zones.py
python code/gc_analysis/02_merge_imaris_exports.py
python code/gc_analysis/03_preprocess_features.py
python code/gc_analysis/04_construct_motility_space.py
python code/gc_analysis/05_scmotiph_integration.py
```

On macOS or Linux, first run `export PYTHONPATH="$PWD/code"`, then use the same five `python` commands. Scripts 01 and 02 require the upstream FDC-mask and Imaris-export inputs; later scripts consume the outputs identified in `DATA_AND_PATHS.md`.

## Directory contents

- `code/gc_analysis/`: five core analysis scripts.
- `code/features/`: custom motility, interaction, directionality, and time-series calculations.
- `code/utils/`: shared trajectory, image, RNA, plotting, and general utilities.
- `demo/01_motility_space/`: real-data subset, deterministic workflow, and expected motility-cluster UMAP.
- `demo/02_scmotiph_integration/`: real scRNA/behavior subsets, deterministic FGW workflow, and expected merged-space UMAP.
