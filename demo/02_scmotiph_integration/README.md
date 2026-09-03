# Demo 2: scMOTIPh integration

This demo uses real, stratified subsets of the manuscript datasets:

- 450 scRNA-seq cells: 75 cells per genotype (`WT`, `EZH2`) and zone (`DZ`, `DZ-LZ`, `LZ`).
- 450 expanded-behavior observations: 75 cells per genotype and corresponding zone.
- The manuscript motility-associated gene signature.

From the package root, run:

```shell
python demo/02_scmotiph_integration/run_scmotiph_demo.py
```

The wrapper performs RNA normalization, log transformation, gene scoring, highly-variable-gene selection, scaling and PCA; performs the manuscript feature selection, scaling and PCA for expanded behavior; applies label- and motility-informed entropic fused Gromov-Wasserstein transport; and generates a shared UMAP from RNA cells and behavior observations transported into RNA PCA space.

Expected runtime is approximately 2-10 minutes depending on the computer. Expected outputs are supplied in `expected_output/`:

- `merged_umap.png`
- `merged_umap_coordinates.csv`
- `merged_space_summary.csv`

