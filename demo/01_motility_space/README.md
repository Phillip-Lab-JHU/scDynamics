# Demo 1: motility-space construction

This demo uses 450 real observations sampled from `GCB_no_inhibit_all_features_20.parquet`: 50 observations from each of the nine manuscript motility clusters.

From the package root, run:

```shell
python demo/01_motility_space/run_motility_demo.py
```

The wrapper repeats the feature selection, standardization, 95%-variance PCA, nine-cluster K-means, speed-based cluster ordering, and UMAP settings used in `code/gc_analysis/04_construct_motility_space.py`. Expected runtime is approximately 1-3 minutes.

Expected outputs are supplied in `expected_output/`:

- `motility_umap_clusters.png`
- `motility_umap_coordinates.csv`
- `motility_cluster_summary.csv`