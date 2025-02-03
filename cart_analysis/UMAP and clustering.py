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

############################## Figure 2. Overall behavior dynamics of lymphocytes in GC LZ. #######################################

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
df_features = pd.read_parquet(path+'motility_features_nan_removed.parquet')
df_features = df_features[df_features['type']!='M5CARHD'].reset_index(drop=True)

df_duration = pd.read_parquet(path + 'traj_duration_nan_removed.parquet')
df_duration = df_duration[df_duration['type']!='M5CARHD'].reset_index(drop=True)

#df_features = df_features.drop(columns=['PC1', 'PC2', 'kmeans', 'pancreatic_effect', 'lung_effect', 'ovarian_effect'])
#df_duration = df_duration.drop(columns=['pancreatic_effect', 'lung_effect', 'ovarian_effect'])
# df_lv = pd.read_parquet(path+'classifier_latent_vector.parquet')
# #df_lv = pd.read_parquet(path+'latent_vector_31.parquet')
# df_lv = pd.concat([df_lv, df_features], axis=1)

df_features.columns.get_loc('inst_angle_pulseindicator')
motility_data = df_features.iloc[:,8:90].drop(['speed_distribution_x', 'speed_distribution_y'],axis=1)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)

m = Morphodynamics(df_features, 'umap')
cluster = m.get_cluster(pcs, n_clusters=8, cluster_type='kmeans')
df_features_ = df_features.copy()
df_features_['kmeans'] = cluster
# m.evaluate_umap(df_features_, pcs, path, n_neighbors_list=[10, 20, 30, 40, 50, 60],
#                       min_dist_list=[0.005, 0.01, 0.05, 0.1, 0.3, 0.5], condition_name ='kmeans')
umap = m.get_umap(pcs, 40, 0.1)
#m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
cluster = m.get_cluster(pcs, n_clusters=8, cluster_type='kmeans')
df = pd.concat([df_features, umap, cluster], axis=1)

df = order_cluster_by_feature(df, cluster_name='kmeans', feature_name='avg_speed')

draw_umap_space(df, path, file_name='space_kmeans_ordered', condition_name='kmeans', label_name='label',
                colors=cmc.batlow, dot_size=0.07, x_name='PC1', y_name='PC2')

# df = df.drop(columns=['pancreatic_phenotype', 'pancreatic_tumor_volume', 'pancreatic_infiltration',
#                       'lung_phenotype', 'lung_tumor_volume', 'lung_infiltration', 'ovarian_phenotype'])
# df_duration = df_duration.drop(columns=['pancreatic_phenotype', 'pancreatic_tumor_volume', 'pancreatic_infiltration',
#                       'lung_phenotype', 'lung_tumor_volume', 'lung_infiltration', 'ovarian_phenotype'])

conditions = [
    (df['type'] == 'M5CAR'),
    #(df['type'] == 'M5CARHD'),
    (df['type'] == 'V5'),
    (df['type'] == 'VR5aIL5'),
    (df['type'] == 'VR5aIL8'),
    (df['type'] == 'VR5aTNFa'),
    (df['type'] == 'Vgsig'),
]

#values = [0, 2, 2, 1, 0, 1]
phenotypes = [0, 1, 1, 0, 0, 0]
tumor_volumes = [830.896, 158.025, 263.541, 661.106, 883.055, 550.116]
infiltrations = [np.mean([3924.181, 25.912, 6.957, 2501.243, 1754.223, 28787.27, 31968.56]),
                 np.mean([130135.7, 127906.3, 332571.6, 353724.8, 109638.6, 96546.2, 178802.1, 171382.3]),
                 np.mean([210617.8, 212892.9, 102898.9, 97300.36, 198191.8, 210626, 148633.9, 195361.2]),
                 np.mean([1339.736, 1347.296, 18471.22, 18855.73, 30926.7, 33410.55, 466.711, 408.531]),
                 np.mean([37.494, 20.501, 119.638, 97.132, 74.755, 59.093, 42.76, 25.343]),
                 np.mean([34283.26, 32999.39, 114693.2, 103467.8, 95370.37, 101408.1, 1046.29, 1009.819])]


df['pancreatic_phenotype'] = np.select(conditions, phenotypes, default='').astype(int)
df['pancreatic_tumor_volume'] = np.select(conditions, tumor_volumes, default='').astype(float)
df['pancreatic_infiltration'] = np.select(conditions, infiltrations, default='').astype(float)


conditions = [
    (df['type'] == 'M5CAR'),
    #(df['type'] == 'M5CARHD'),
    (df['type'] == 'V5'),
    (df['type'] == 'VR5aIL5'),
    (df['type'] == 'VR5aIL8'),
    (df['type'] == 'VR5aTNFa'),
    (df['type'] == 'Vgsig'),
]




#values = [0, 2, 2, 1, 0, 1]
phenotypes = [0, 1, 1, 0, 0, 0]

tumor_volumes = [690.059, 569.809, 268.15, 560.478, 649.062, 484.936]

infiltrations = [np.mean([0, 0, 423.549, 1205.147, 104.374, 270.308, 34.414, 69.353]),
                 np.mean([21454.87, 23557.92, 128112.8, 119128.9, 192770.7, 190788.9, 42215.31, 46104.5]),
                 np.mean([20934.08, 26553.05, 230641.4, 212683, 80769, 95654.39, 19.210, 29.903]),
                 np.mean([86853.25, 102549.1, 7848.761, 272175.6, 47092.14, 54644.61, 56.944, 225.662]),
                 np.mean([139.95, 133.886, 70.831, 36.709, 438.169, 350.486, 373.155, 347.340]),
                 np.mean([772.944, 1453.158, 11555.1, 15356.47, 139378.1, 150817.9, 21381.98, 16876.04])]

df['lung_phenotype'] = np.select(conditions, phenotypes, default='').astype(int)
df['lung_tumor_volume'] = np.select(conditions, tumor_volumes, default='').astype(float)
df['lung_infiltration'] = np.select(conditions, infiltrations, default='').astype(float)


conditions = [
    (df['type'] == 'M5CAR'),
    #(df['type'] == 'M5CARHD'),
    (df['type'] == 'V5'),
    (df['type'] == 'VR5aIL5'),
    (df['type'] == 'VR5aIL8'),
    (df['type'] == 'VR5aTNFa'),
    (df['type'] == 'Vgsig'),
]

#values = [0, 2, 2, 1, 0, 1]
phenotypes = [0, 0, 1, 1, 0, 0]

tumor_volumes = [418.52, 214.78, 174.67, 232.34, 190.06, 192.55]

infiltrations = [np.mean([1784.4, 14699.1, 751.13, 757.24]),
                 np.mean([3499.29, 631.7, 4632.26, 2439.63]),
                 np.mean([30202.78, 10856.81, 8007.22, 31127.79, 45886.69]),
                 np.mean([99515.97, 10238.23, 25223.42]),
                 np.mean([4580.35, 312.40, 203.97, 823.52]),
                 np.mean([9012.64, 15392.46, 149.25])]

df['ovarian_phenotype'] = np.select(conditions, phenotypes, default='').astype(int)
df['ovarian_tumor_volume'] = np.select(conditions, tumor_volumes, default='').astype(float)
df['ovarian_infiltration'] = np.select(conditions, infiltrations, default='').astype(float)
#TODO add ovarian tumor volume and infiltration



# VR5aIL8 -> 1
# VR5aIL5 -> 2
# Based on infiltration

df_duration['kmeans'] = np.repeat(df['kmeans'].values, 31)
df_duration['pancreatic_phenotype'] = np.repeat(df['pancreatic_phenotype'].values, 31)
df_duration['pancreatic_tumor_volume'] = np.repeat(df['pancreatic_tumor_volume'].values, 31)
df_duration['pancreatic_infiltration'] = np.repeat(df['pancreatic_infiltration'].values, 31)

df_duration['lung_phenotype'] = np.repeat(df['lung_phenotype'].values, 31)
df_duration['lung_tumor_volume'] = np.repeat(df['lung_tumor_volume'].values, 31)
df_duration['lung_infiltration'] = np.repeat(df['lung_infiltration'].values, 31)

df_duration['ovarian_phenotype'] = np.repeat(df['ovarian_phenotype'].values, 31)
df_duration['ovarian_tumor_volume'] = np.repeat(df['ovarian_tumor_volume'].values, 31)
df_duration['ovarian_infiltration'] = np.repeat(df['ovarian_infiltration'].values, 31)

df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # strip() removes any leading, and trailing whitespaces

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
df.to_csv(path + 'motility_features_nan_removed.csv', index=False)
df.to_parquet(path + 'motility_features_nan_removed.parquet')
df_duration.to_csv(path + 'traj_duration_nan_removed.csv', index=False)
df_duration.to_parquet(path + 'traj_duration_nan_removed.parquet')

#################################### latent space ####################################

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
df_duration = pd.read_parquet(path + 'traj_duration_nan_removed.parquet')

ts = Morphodynamics(df_duration, 'umap')
cluster, cluster_expanded, cluster_center = ts.get_ts_cluster(df_duration, 11,  duration=31, normalize=False, feature_name=['rotated_x', 'rotated_y'])

motility_data = df_lv.iloc[:,:128]

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)

m = Morphodynamics(df_lv, 'umap')
umap = m.get_umap(motility_data_scaled, 20, 0.5)
df = pd.concat([df_lv, umap, pd.DataFrame(cluster, columns=['tskmeans'])], axis=1)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # strip() removes any leading, and trailing whitespaces

df.to_csv(path + 'all_features.csv', index=False)
df.to_parquet(path + 'all_features.parquet')

#################################### motility space ####################################

df_features.columns.get_loc('displ_autocorr_y_3')
motility_data = df_features.iloc[:,2:75].drop(['phi'],axis=1)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)

m = Morphodynamics(df_features, 'umap')
umap = m.get_umap(pcs, 20, 0.5)
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
cluster = m.get_cluster(pcs, n_clusters=11, cluster_type='kmeans')
df = pd.concat([df_features, umap, cluster], axis=1)

df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # strip() removes any leading, and trailing whitespaces

df.to_csv(path + 'motility_features_nan_removed.csv', index=False)
df.to_parquet(path + 'motility_features_nan_removed.parquet')