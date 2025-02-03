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
"""Generates Data for Figure 1-2: Non-moving cells removed."""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast

#################################### motility space ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\\'
df = pd.read_parquet(path+'all_features_30_PC.parquet')
df_duration = pd.read_parquet(path+'traj_duration_30.parquet')

duration=30
label_expanded = np.repeat(df['kmeans'], duration).reset_index(drop=True)
df_duration['kmeans'] = label_expanded
#################################### Box plot comparing all motility features by cell types ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis\Figure1-2. Only moving\\'
color_list = ('#888888', '#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100')
draw_umap_space(df, path, file_name='motility space_kmeans', condition_name='kmeans', label_name='pseudo_particle',
                colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')



color_list = ('#888888', '#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100')
df.columns.get_loc('inst_angle_cosine_similarity_entropies')
#feature_list = list( df.columns[2:103].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )
feature_list = ['avg_speed', 'total_distance', 'max_speed', 'Time_span', 'net_distance']
condition_name = 'kmeans'

if not os.path.isdir(path + 'motility_feature_box_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'motility_feature_box_plot_type/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    draw_custom_box_plot(dataset, path+ 'motility_feature_box_plot_type/', file_name=feature_name, colors = color_list,
    strip_plot=False, test='mann-whitney', pvalue=False, figsize=(6,6))

#################################### Trajectories for each cluster ####################################
draw_2D_trajectories_one_figure(df_duration, df, path, duration=30, n_examples=40, label_name='kmeans', feature_name=['x', 'y'], lim=200)
# -> Cluster 3, 5 and 8 are non-moving cluster, so let's remove it

#################################### Box plot comparing all motility features by cell types ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\\'
df_removed = df[(df['kmeans']!=3)&(df['kmeans']!=5)&(df['kmeans']!=8)].reset_index(drop=True)
df_duration_removed = df_duration[(df_duration['kmeans']!=3)&(df_duration['kmeans']!=5)&(df_duration['kmeans']!=8)].reset_index(drop=True)

df_removed = df_removed.drop(['PC1', 'PC2', 'kmeans'], axis=1)

df_removed.columns.get_loc('inst_angle_cosine_similarity_entropies')
motility_data = df_removed.iloc[:,2:103].drop(['phi', 'speed_distribution_x', 'speed_distribution_y'],axis=1)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)

m = Morphodynamics(df_removed, 'umap')
umap = m.get_umap(pcs, 20, 0.5)
m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
cluster = m.get_cluster(pcs, n_clusters=12, cluster_type='kmeans')
df_removed_PC = pd.concat([df_removed, umap, cluster], axis=1)

df_removed_PC.to_csv(path + 'nonmoving_removed_all_features_30_PC.csv', index=False)
df_removed_PC.to_parquet(path + 'nonmoving_removed_all_features_30_PC.parquet')

duration=30
label_expanded = np.repeat(df_removed_PC['kmeans'], duration).reset_index(drop=True)
df_duration_removed['kmeans'] = label_expanded

df_duration_removed.to_csv(path + 'nonmoving_removed_traj_duration_30.csv', index=False)
df_duration_removed.to_parquet(path + 'nonmoving_removed_traj_duration_30.parquet')

#################################### Box plot comparing all motility features by cell types ####################################