# Data and path requirements

The full-workflow scripts contain absolute paths from the computer on which the analyses were performed. They document the original data locations; reviewers should replace them with paths to the separately deposited manuscript data. The two directly executable demos use portable paths relative to their script locations.

Required input categories include:

- FDC-mask TIFF hyperstacks and corresponding Imaris CSV directories for script 01.
- Imaris per-video and per-cell-type CSV exports for script 02.
- `Intravital Data_all.parquet` for script 03.
- Intermediate trajectory and feature parquet files, including `traj_duration_20_all.parquet` and `all_features_20_all.parquet`, for script 04.
- `GCB_only_cluster_final.h5ad`, `Expanded_behavior.parquet`, and `Motility-Associated_Genes.csv` for the scRNA/behavior integration script.

Small, stratified subsets of the motility, expanded-behavior, and scRNA-seq data are included under `demo/` for direct execution. The complete datasets remain required to reproduce the full manuscript results.

The data repository DOI/accession and a manifest of the complete data should be added here when full-data deposition is finalized.

Imaris version and the export settings used to generate the CSV input files should also be added to the manuscript Methods or this file.
