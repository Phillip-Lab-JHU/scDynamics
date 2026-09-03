# Author: Chanhong Min <cmin11@jhmi.edu>

"""Compact real-data demonstration of scRNA/expanded-behavior FGW integration."""

from pathlib import Path
import os

os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from umap import UMAP


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

# RNA preprocessing follows the central workflow, with reduced dimensions for the demo.
adata_rna = sc.read_h5ad(DATA / "GCB_scRNA_subset.h5ad")
adata_rna.layers["counts"] = adata_rna.X.copy()
sc.pp.normalize_total(adata_rna, target_sum=10000)
sc.pp.log1p(adata_rna)
adata_rna.raw = adata_rna

signature = pd.read_csv(DATA / "Motility-Associated_Genes.csv")["Gene"].dropna().astype(str)
signature = [gene for gene in signature if gene in adata_rna.var_names]
sc.tl.score_genes(adata_rna, signature, score_name="motility_signature_score", random_state=0)

sc.pp.highly_variable_genes(adata_rna, flavor="seurat", n_top_genes=1000)
adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()
sc.pp.scale(adata_rna, max_value=10)
sc.tl.pca(adata_rna, n_comps=30, svd_solver="arpack", random_state=0)
xs = adata_rna.obsm["X_pca"]

# Expanded-behavior preprocessing uses the same column ranges as the manuscript script.
behavior = pd.read_parquet(DATA / "Expanded_behavior_subset.parquet")
behavior["Zone"] = pd.cut(
    behavior["avg_zone"], [-1e-9, 0.4, 0.8, 2.0000001],
    labels=["DZ", "DZ-LZ", "LZ"], right=False
).astype(str)
motility = behavior.iloc[:, 8:134].drop(
    columns=["speed_distribution_x", "speed_distribution_y", "speed_distribution_z"]
)
drop_columns = [
    "FDC_diff_distance_cov", "T_diff_distance_cov", "Core_diff_distance_cov",
    "LZ_diff_distance_cov", "DZ_diff_distance_cov",
    "DZ_distance_autocorr_1", "DZ_distance_autocorr_2", "DZ_distance_autocorr_3",
    "DZ_diff_distance_autocorr_1", "DZ_diff_distance_autocorr_2", "DZ_diff_distance_autocorr_3",
    "LZ_diff_distance_autocorr_1", "LZ_diff_distance_autocorr_2", "LZ_diff_distance_autocorr_3",
    "Core_distance_autocorr_1", "Core_distance_autocorr_2", "Core_distance_autocorr_3",
    "Core_diff_distance_autocorr_1", "Core_diff_distance_autocorr_2", "Core_diff_distance_autocorr_3",
    "Core_distance_variance", "Core_diff_distance_variance", "DZ_distance_variance",
    "DZ_diff_distance_variance", "FDC_diff_distance_variance", "FDC_distance_variance",
    "LZ_distance_variance", "LZ_diff_distance_variance", "T_diff_distance_variance",
    "T_distance_variance", "PC1", "PC2", "kmeans",
]
colocalization = behavior.iloc[:, 148:289].drop(columns=drop_columns)
colocalization = colocalization.drop(columns=colocalization.columns[colocalization.isna().any()])
behavior_features = pd.concat([motility, colocalization], axis=1)
adata_behavior = sc.AnnData(behavior_features.astype(float))
sc.pp.scale(adata_behavior, max_value=10)
sc.tl.pca(adata_behavior, n_comps=30, svd_solver="arpack", random_state=0)
xt = adata_behavior.obsm["X_pca"]

# The published script's selected FGW settings.
label_coeff, alpha, epsilon = 100, 0.1, 1.3
label_mapping = {"DZ": 0, "DZ-LZ": 1, "LZ": 2}
ys = adata_rna.obs["cluster"].map(label_mapping).to_numpy()
yt = behavior["Zone"].map(label_mapping).to_numpy()

source_shared = StandardScaler().fit_transform(
    adata_rna.obs[["motility_signature_score"]]
)
target_shared = StandardScaler().fit_transform(
    behavior[["morpho_avg_speed"]]
)
feature_cost = ot.dist(source_shared, target_shared)
label_cost = (ys[:, None] != yt[None, :]) * label_coeff
M = np.maximum(feature_cost, label_cost)
C1, C2 = cdist(xs, xs), cdist(xt, xt)
a = np.ones(len(xs)) / len(xs)
b = np.ones(len(xt)) / len(xt)

coupling = ot.gromov.entropic_fused_gromov_wasserstein(
    M, C1, C2, a, b, alpha=alpha, epsilon=epsilon,
    max_iter=10000, random_state=0
)
barycentric = coupling.T / np.sum(coupling, axis=0)[:, None]
barycentric = np.nan_to_num(barycentric)
transported_behavior = barycentric @ xs

combined = np.concatenate([xs, transported_behavior])
embedding = UMAP(
    metric="euclidean", n_components=2, n_neighbors=30,
    min_dist=0.5, random_state=0
).fit_transform(combined)

rna_result = pd.DataFrame({
    "dataset": "scRNA-seq", "UMAP1": embedding[:len(xs), 0],
    "UMAP2": embedding[:len(xs), 1],
    "zone": adata_rna.obs["cluster"].astype(str).to_numpy(),
    "type": adata_rna.obs["Type"].astype(str).to_numpy(),
    "behavior_cluster": np.nan,
})
behavior_result = pd.DataFrame({
    "dataset": "expanded behavior", "UMAP1": embedding[len(xs):, 0],
    "UMAP2": embedding[len(xs):, 1],
    "zone": behavior["Zone"].astype(str).to_numpy(),
    "type": behavior["Type"].astype(str).to_numpy(),
    "behavior_cluster": behavior["kmeans"].to_numpy(),
})
result = pd.concat([rna_result, behavior_result], ignore_index=True)
result.to_csv(OUTPUT / "merged_umap_coordinates.csv", index=False)
pd.DataFrame(coupling).to_csv(OUTPUT / "fgw_coupling.csv", index=False)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for dataset, color, marker in [("scRNA-seq", "#4F609C", "o"),
                                ("expanded behavior", "#E36A5C", "^")]:
    part = result[result.dataset == dataset]
    ax.scatter(part.UMAP1, part.UMAP2, s=10, alpha=0.65,
               color=color, marker=marker, label=dataset)
ax.set(xlabel="UMAP1", ylabel="UMAP2", title="scMOTIPh merged space: real-data subsets")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT / "merged_umap.png", dpi=200, bbox_inches="tight")
plt.close(fig)

summary = result.groupby(["dataset", "zone"]).size().rename("n").reset_index()
summary.to_csv(OUTPUT / "merged_space_summary.csv", index=False)
print(f"RNA cells: {len(xs)}; behavior cells: {len(xt)}")
print(f"RNA PCA dimensions: {xs.shape[1]}; behavior PCA dimensions: {xt.shape[1]}")
print(f"Coupling mass: {coupling.sum():.8f}")
print(summary.to_string(index=False))
print(f"Outputs written to {OUTPUT}")
