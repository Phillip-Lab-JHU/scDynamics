# Author: Chanhong Min <cmin11@jhmi.edu>

"""Construct the nine-cluster motility space used in the manuscript."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from utils.misc_utils import order_cluster_by_feature


DURATION = 20
ANALYSIS_DIR = Path(r"\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis")
OUTPUT_DIR = ANALYSIS_DIR / "feature_csvs"

# These are the feature and trajectory tables produced by script 03.
features_all = pd.read_parquet(ANALYSIS_DIR / f"all_features_{DURATION}_all.parquet")
trajectories_all = pd.read_parquet(ANALYSIS_DIR / f"traj_duration_{DURATION}_all.parquet")

# Select untreated experiments and exclude Group A videos, following the original analysis.
features = features_all[features_all["Exp"].isin(["Exp1", "Exp2", "Exp3"])].reset_index(drop=True)
trajectories = trajectories_all[trajectories_all["Exp"].isin(["Exp1", "Exp2", "Exp3"])].reset_index(drop=True)
videos = np.unique(features["Video"])
group_a_videos = [videos[1], videos[2], videos[4], videos[-1]]
features = features[~features["Video"].isin(group_a_videos)].reset_index(drop=True)
trajectories = trajectories[~trajectories["Video"].isin(group_a_videos)].reset_index(drop=True)
features = features[features["Type"] != "T-cell"].reset_index(drop=True)
trajectories = trajectories[trajectories["Type"] != "T-cell"].reset_index(drop=True)

# Feature selection, scaling, PCA, clustering, and UMAP match the original workflow.
motility = features.iloc[:, 8:126].drop(
    columns=["speed_distribution_x", "speed_distribution_y", "speed_distribution_z"]
)
scaled = StandardScaler().fit_transform(motility)
pcs = PCA(0.95).fit_transform(scaled)
raw_clusters = KMeans(
    n_clusters=9, random_state=0, init="k-means++", n_init=10
).fit_predict(pcs)
embedding = UMAP(
    metric="euclidean", n_components=2, n_neighbors=20,
    min_dist=0.1, random_state=0
).fit_transform(pcs)

result = pd.concat(
    [features, pd.DataFrame(embedding, columns=["PC1", "PC2"]),
     pd.DataFrame({"kmeans": raw_clusters})],
    axis=1,
)
result, _ = order_cluster_by_feature(
    result, cluster_name="kmeans", feature_name="avg_speed"
)

expanded_labels = np.repeat(result["kmeans"].to_numpy(), DURATION)
if len(expanded_labels) != len(trajectories):
    raise ValueError("Cluster labels do not match the trajectory table length.")
trajectories["kmeans"] = expanded_labels

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT_DIR / f"GCB_no_inhibit_all_features_{DURATION}.parquet", index=False)
result.to_csv(OUTPUT_DIR / f"GCB_no_inhibit_all_features_{DURATION}.csv", index=False)
trajectories.to_parquet(OUTPUT_DIR / f"GCB_no_inhibit_traj_duration_{DURATION}.parquet", index=False)
trajectories.to_csv(OUTPUT_DIR / f"GCB_no_inhibit_traj_duration_{DURATION}.csv", index=False)

print(f"Saved {len(result)} cells with {motility.shape[1]} motility features and {pcs.shape[1]} PCA components.")
