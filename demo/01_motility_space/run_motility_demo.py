# Author: Chanhong Min <cmin11@jhmi.edu>

"""Reconstruct a compact motility UMAP and nine clusters from real study data."""

from pathlib import Path
import os

os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP


HERE = Path(__file__).resolve().parent
INPUT = HERE / "data" / "GCB_no_inhibit_all_features_20_subset.parquet"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

df = pd.read_parquet(INPUT)
motility = df.iloc[:, 8:126].drop(
    columns=["speed_distribution_x", "speed_distribution_y", "speed_distribution_z"]
)
scaled = StandardScaler().fit_transform(motility)
pcs = PCA(0.95).fit_transform(scaled)

raw_clusters = KMeans(
    n_clusters=9, random_state=0, init="k-means++", n_init=10
).fit_predict(pcs)

# Match the manuscript convention: order cluster numbers by mean speed.
means = pd.DataFrame({"cluster": raw_clusters, "avg_speed": df["avg_speed"]}).groupby("cluster")["avg_speed"].mean()
cluster_map = {old: new for new, old in enumerate(means.sort_values().index)}
clusters = pd.Series(raw_clusters).map(cluster_map).to_numpy()

embedding = UMAP(
    metric="euclidean", n_components=2, n_neighbors=20,
    min_dist=0.1, random_state=0
).fit_transform(pcs)

result = pd.DataFrame({
    "sample_id": df["traj_Label"].astype(str),
    "Type": df["Type"].astype(str),
    "UMAP1": embedding[:, 0],
    "UMAP2": embedding[:, 1],
    "motility_cluster": clusters,
})
result.to_csv(OUTPUT / "motility_umap_coordinates.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 4))
scatter = ax.scatter(result.UMAP1, result.UMAP2, c=result.motility_cluster,
                     cmap="tab10", s=12, alpha=0.8)
ax.set(xlabel="UMAP1", ylabel="UMAP2", title="Motility space: real-data subset")
ax.legend(*scatter.legend_elements(), title="Cluster", frameon=False,
          bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT / "motility_umap_clusters.png", dpi=200, bbox_inches="tight")
plt.close(fig)

summary = result.groupby(["motility_cluster", "Type"]).size().rename("n").reset_index()
summary.to_csv(OUTPUT / "motility_cluster_summary.csv", index=False)
print(f"Input rows: {len(df)}; features: {motility.shape[1]}; PCA components: {pcs.shape[1]}")
print(summary.to_string(index=False))
print(f"Outputs written to {OUTPUT}")
