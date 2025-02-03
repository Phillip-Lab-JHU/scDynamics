# Chanhong Min <cmin11@jhmi.edu>

# Copyright 2023 The Phillip tiME Lab at the Johns Hopkins University
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/Phillip-Lab-JHU/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generates Data for Figure 1."""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast

#################################### motility feature space ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Anshika motility project\\'
df_duration = pd.read_parquet(path+'traj_duration_96.parquet')
df_features = pd.read_parquet(path+'motility_features_96.parquet')

# #################################### Filter high max speed cells ####################################
# remove_rows = df_features[df_features['max_speed']>10]
# df_features = df_features[df_features['max_speed']<=10].reset_index(drop=True)
# df_duration = remove_trajs_condition(df_duration, duration=30, remove_traj_idxs=list(remove_rows.index))


#################################### PCA, UMAP and clustering ####################################
df_features.columns.get_loc('inst_angle_cosine_similarity_entropies')
motility_data = df_features.iloc[:,8:103].drop(['speed_distribution_x', 'speed_distribution_y',
                                                'inst_speed_cosine_similarity_entropies', 'inst_angle_cosine_similarity_entropies'],axis=1)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)

m = Morphodynamics(df_features, 'umap')
cluster = m.get_cluster(pcs, n_clusters=9, cluster_type='kmeans')
df_features_ = df_features.copy()
df_features_['kmeans'] = cluster
m.evaluate_umap(df_features_, pcs, path, n_neighbors_list=[10, 20, 30, 40, 50, 60],
                      min_dist_list=[0.005, 0.01, 0.05, 0.1, 0.3, 0.5], condition_name ='kmeans')

umap = m.get_umap(pcs, 50, 0.5)
#m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
cluster = m.get_cluster(pcs, n_clusters=8, cluster_type='kmeans')
df = pd.concat([df_features, umap, cluster], axis=1)

df, replace_map = order_cluster_by_feature(df, cluster_name='kmeans', feature_name='avg_speed')

draw_umap_space(df, path, file_name='space_kmeans_ordered', condition_name='kmeans', label_name='pseudo_label',
                colors=cmc.lipari, dot_size=0.07, x_name='PC1', y_name='PC2')

df.to_csv(path + 'motility_features_96.csv', index=False)
df.to_parquet(path + 'motility_features_96.parquet')

duration=96
label_expanded = np.repeat(df['kmeans'], duration).reset_index(drop=True)
df_duration['kmeans'] = label_expanded

df_duration.to_parquet(path + 'traj_duration_96.parquet')
df_duration.to_csv(path + 'traj_duration_96.csv', index=False)


#################################### Outlier detection  ####################################
m = Morphodynamics(df_features, 'umap')
df_features.columns.get_loc('inst_angle_cosine_similarity_entropies')
motility_data = df_features.iloc[:,8:103].drop(['speed_distribution_x', 'speed_distribution_y',
                                                'inst_speed_cosine_similarity_entropies', 'inst_angle_cosine_similarity_entropies'],axis=1)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)

umap = m.get_umap(pcs, 30, 0.005)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.001, min_samples=10)  # eps = radius around the point, min_samples = min number of samples within the circle
dbscan.fit(umap)
labels = dbscan.labels_
df_features_ = df_features.copy()
df_features_['dbscan'] = labels
df_features_.loc[(df_features_['dbscan'] != -1), 'outlier'] = 'outlier'
df_features_.loc[(df_features_['dbscan'] == -1), 'outlier'] = 'data'
df_ = pd.concat([df_features_, umap], axis=1)

outliers = np.where(labels != -1)[0]
print(outliers.shape)

plt.scatter(umap.iloc[:, 0], umap.iloc[:, 1])
plt.scatter(umap.iloc[outliers, 0], umap.iloc[outliers, 1], c="red", marker="x")
plt.show()
plt.clf()
plt.close()



from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=4, min_samples=10)  # eps = radius around the point, min_samples = min number of samples within the circle
dbscan.fit(pcs)
labels = dbscan.labels_
df_features_ = df_features.copy()
df_features_['dbscan'] = labels
df_features_.loc[(df_features_['dbscan'] == -1), 'outlier'] = 'data'
df_features_.loc[(df_features_['dbscan'] != -1), 'outlier'] = 'outlier'

df_ = pd.concat([df_features_, umap], axis=1)

outliers = np.where(labels != -1)[0]
print(outliers.shape)

plt.scatter(umap.iloc[:, 0], umap.iloc[:, 1])
plt.scatter(umap.iloc[outliers, 0], umap.iloc[outliers, 1], c="red", marker="x")
plt.show()
plt.clf()
plt.close()

draw_umap_space(df_, path, file_name='outlier_detection', condition_name='outlier', label_name='pseudo_particle',
                colors=cmc.batlow, dot_size=0.07, x_name='PC1', y_name='PC2')


# ############################## latent vector space #######################################
#
# path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Charles Post Optimizations\Low Density (25k cells)\0.5 Gel\analysis 01_27_24\\'
# df_features = pd.read_parquet(path+'all_features_removed_30.parquet')
# df_duration = pd.read_parquet(path + 'traj_duration_removed_30.parquet')
#
# ts = Morphodynamics(df_duration, 'umap')
# cluster, cluster_expanded, cluster_center = ts.get_ts_cluster(df_duration, 11,  duration=30, normalize=False, feature_name=['reg_x', 'reg_y'])
#
# motility_data = df_features.iloc[:,:128]
#
# scaler = StandardScaler()  # if not normalize, UMAP space is completely different
# motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
#
# m = Morphodynamics(df_features, 'umap')
# umap = m.get_umap(motility_data_scaled, 20, 0.5)
# df = pd.concat([df_features, umap, pd.DataFrame(cluster, columns=['tskmeans'])], axis=1)
# df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # strip() removes any leading, and trailing whitespaces
#
# df.to_csv(path + 'all_features_removed_30_PC.csv', index=False)
# df.to_parquet(path + 'all_features_removed_30_PC.parquet')

