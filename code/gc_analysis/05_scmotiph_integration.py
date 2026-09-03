# Author: Chanhong Min <cmin11@jhmi.edu>

"""Integrate scRNA-seq and expanded-behavior spaces with entropic FGW."""

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


RNA_DIR = Path(r"\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected")
BEHAVIOR_DIR = Path(r"\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure6. Expanded behavior")
OUTPUT_DIR = RNA_DIR / "analysis" / "RNA-behavior"
SIGNATURE_FILE = OUTPUT_DIR / "signatures" / "Motility-Associated_Genes.csv"

# RNA preprocessing.
adata_rna = sc.read_h5ad(RNA_DIR / "GCB_only_cluster_final.h5ad")
adata_rna = adata_rna[
    adata_rna.obs["cluster"].isin(["DZ", "DZ-LZ", "LZ", "Recycle"])
].copy()
adata_rna.obs["cluster"] = adata_rna.obs["cluster"].replace({"Recycle": "DZ-LZ"})
adata_rna.layers["counts"] = adata_rna.X.copy()
sc.pp.normalize_total(adata_rna, target_sum=10000)
sc.pp.log1p(adata_rna)
adata_rna.raw = adata_rna

signature = pd.read_csv(SIGNATURE_FILE)["Gene"].dropna().astype(str)
signature = [gene for gene in signature if gene in adata_rna.var_names]
sc.tl.score_genes(
    adata_rna, signature,
    score_name="deformability_score_Motility associated signatures",
    random_state=0,
)

sc.pp.highly_variable_genes(adata_rna, flavor="seurat", n_top_genes=3000)
adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()
sc.pp.scale(adata_rna, max_value=10)
sc.tl.pca(adata_rna, n_comps=330, svd_solver="arpack", random_state=0)
xs = adata_rna.obsm["X_pca"]

# Expanded-behavior preprocessing.
behavior = pd.read_parquet(BEHAVIOR_DIR / "Expanded_behavior.parquet")
behavior["Zone"] = pd.cut(
    behavior["avg_zone"], [-1e-9, 0.4, 0.8, 2.0000001],
    labels=["DZ", "DZ-LZ", "LZ"], right=False,
).astype(str)
behavior["Zone1"] = pd.cut(
    behavior["avg_zone"], [-1e-9, 0.4, 0.8, 1.2, 1.6, 2.0000001],
    labels=["DZ", "DZ-sLZ", "sLZ", "sLZ-dLZ", "dLZ"], right=False,
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
colocalization = colocalization.drop(
    columns=colocalization.columns[colocalization.isna().any()]
)
behavior_features = pd.concat([motility, colocalization], axis=1)
adata_behavior = sc.AnnData(behavior_features.astype(float))
sc.pp.scale(adata_behavior, max_value=10)
sc.tl.pca(adata_behavior, n_comps=36, svd_solver="arpack", random_state=0)
xt = adata_behavior.obsm["X_pca"]

# Label- and motility-informed entropic fused Gromov-Wasserstein integration.
label_coeff, alpha, epsilon = 100, 0.1, 1.3

label_mapping = {"DZ": 0, "DZ-LZ": 1, "LZ": 2}
ys = adata_rna.obs["cluster"].map(label_mapping).to_numpy()
yt = behavior["Zone"].map(label_mapping).to_numpy()
source_shared = StandardScaler().fit_transform(
    adata_rna.obs[["deformability_score_Motility associated signatures"]]
)
target_shared = StandardScaler().fit_transform(behavior[["morpho_avg_speed"]])
feature_cost = ot.dist(source_shared, target_shared)
label_cost = (ys[:, None] != yt[None, :]) * label_coeff
M = np.maximum(feature_cost, label_cost)
C1, C2 = cdist(xs, xs), cdist(xt, xt)
a = np.ones(len(xs)) / len(xs)
b = np.ones(len(xt)) / len(xt)
coupling = ot.gromov.entropic_fused_gromov_wasserstein(
    M, C1, C2, a, b, alpha=alpha, epsilon=epsilon,
    max_iter=10000, random_state=0,
)
barycentric = coupling.T / np.sum(coupling, axis=0)[:, None]
transported_behavior = np.nan_to_num(barycentric) @ xs

combined = np.concatenate([xs, transported_behavior])
embedding = UMAP(
    metric="euclidean", n_components=2, n_neighbors=30,
    min_dist=0.5, random_state=0,
).fit_transform(combined)

rna_result = pd.DataFrame({
    "dataset": "scRNA-seq",
    "UMAP1": embedding[:len(xs), 0],
    "UMAP2": embedding[:len(xs), 1],
    "zone": adata_rna.obs["cluster"].astype(str).to_numpy(),
    "type": adata_rna.obs["Type"].astype(str).to_numpy(),
    "behavior_cluster": np.nan,
})
behavior_cluster_column = "beh_kmeans" if "beh_kmeans" in behavior else "kmeans"
behavior_result = pd.DataFrame({
    "dataset": "expanded behavior",
    "UMAP1": embedding[len(xs):, 0],
    "UMAP2": embedding[len(xs):, 1],
    "zone": behavior["Zone"].astype(str).to_numpy(),
    "type": behavior["Type"].astype(str).to_numpy(),
    "behavior_cluster": behavior[behavior_cluster_column].to_numpy(),
})
result = pd.concat([rna_result, behavior_result], ignore_index=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_DIR / "scMOTIPh_merged_umap_coordinates.csv", index=False)
pd.DataFrame(coupling).to_parquet(OUTPUT_DIR / "scMOTIPh_FGW_coupling.parquet", index=False)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for dataset, color, marker in [
    ("scRNA-seq", "#4F609C", "o"),
    ("expanded behavior", "#E36A5C", "^"),
]:
    part = result[result["dataset"] == dataset]
    ax.scatter(part["UMAP1"], part["UMAP2"], s=5, alpha=0.5,
               color=color, marker=marker, label=dataset)
ax.set(xlabel="UMAP1", ylabel="UMAP2", title="scMOTIPh merged space")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "scMOTIPh_merged_umap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Integrated {len(xs)} RNA cells and {len(xt)} behavior observations.")
print(f"FGW coupling mass: {coupling.sum():.8f}")
