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
"""Do kmeans and UMAP """

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.misc_utils import order_cluster_by_feature
from utils.traj_utils import to_timeseries_fast

duration=20
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\\'
df_features_all = pd.read_parquet(path+'all_features_%s.parquet'%duration)
df_duration_all = pd.read_parquet(path+'traj_duration_%s.parquet'%duration)

#################################### Without inhibition GCB + inhibition (Figure 2, 3, 4) ####################################
df_features = df_features_all[(df_features_all['Exp']=='Exp1')|(df_features_all['Exp']=='Exp2')|(df_features_all['Exp']=='Exp3')
                              |(df_features_all['Exp']=='Exp5')].reset_index(drop=True)
df_duration = df_duration_all[(df_duration_all['Exp']=='Exp1')|(df_duration_all['Exp']=='Exp2')|(df_duration_all['Exp']=='Exp3')
                              |(df_duration_all['Exp']=='Exp5')].reset_index(drop=True)

videos = np.unique(df_features['Video'])
df_features = df_features[(df_features['Video'] != videos[1])&(df_features['Video'] != videos[2])&(df_features['Video'] != videos[4])
                          &(df_features['Video'] != videos[-1])].reset_index(drop=True)

df_duration = df_duration[(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])&(df_duration['Video'] != videos[4])
                          &(df_duration['Video'] != videos[-1])].reset_index(drop=True)

df_features = df_features[df_features['Type']!='T-cell'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='T-cell'].reset_index(drop=True)


########### With inhibition GCB ############

df_features_inhibit = df_features_all[(df_features_all['Exp']=='CD40L')|(df_features_all['Exp']=='IgG')|(df_features_all['Exp']=='mLT')].reset_index(drop=True)
df_duration_inhibit = df_duration_all[(df_duration_all['Exp']=='CD40L')|(df_duration_all['Exp']=='IgG')|(df_duration_all['Exp']=='mLT')].reset_index(drop=True)


# videos = np.unique(df_features['Video'])
# df_features = df_features[(df_features['Video'] != videos[1])&(df_features['Video'] != videos[2])&(df_features['Video'] != videos[12])
#                           &(df_features['Video'] != videos[-1])].reset_index(drop=True)
#
# df_duration = df_duration[(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])&(df_duration['Video'] != videos[12])
#                  &(df_duration['Video'] != videos[-1])].reset_index(drop=True)

df_features_inhibit = df_features_inhibit[df_features_inhibit['Type']!='T-cell'].reset_index(drop=True)
df_duration_inhibit = df_duration_inhibit[df_duration_inhibit['Type']!='T-cell'].reset_index(drop=True)



#################### UMAP and PCA ###################

df_features.columns.get_loc('inst_angle_cosine_similarity_entropies')
motility_data = df_features.iloc[:,8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)
motility_data_inhibit = df_features_inhibit.iloc[:,8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
motility_data_scaled_inhibit= pd.DataFrame(scaler.transform( motility_data_inhibit ), columns=motility_data_inhibit.columns)


from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
pcs_inhibit = pca.transform(motility_data_scaled_inhibit)

# m = Morphodynamics(df_features, 'umap')
# cluster = m.get_cluster(pcs, n_clusters=9, cluster_type='kmeans')
#
# df_features_ = df_features.copy()
# df_features_['kmeans'] = cluster
# m.evaluate_umap(df_features_, pcs, path, n_neighbors_list=[10, 20, 30, 40, 50, 60],
#                       min_dist_list=[0.005, 0.01, 0.05, 0.1, 0.3, 0.5], condition_name ='kmeans')


from sklearn.cluster import KMeans
km = KMeans(n_clusters=9, random_state=0, init='k-means++')
# k-means++: Initialize centroids that are far away each other
kmeans_predicted = km.fit_predict(pcs)
cluster = pd.DataFrame(kmeans_predicted, columns=['kmeans'])

kmeans_predicted_inhibit = km.predict(pcs_inhibit)
cluster_inhibit = pd.DataFrame(kmeans_predicted_inhibit, columns=['kmeans'])


from umap import UMAP
__umap = UMAP(metric='euclidean', n_components=2, n_neighbors=40, min_dist=0.5, random_state=0)
pcs_array = __umap.fit_transform(pcs)
umap = pd.DataFrame(pcs_array, columns=['PC1', 'PC2'])

pcs_array_inhibit = __umap.transform(pcs_inhibit)
umap_inhibit = pd.DataFrame(pcs_array_inhibit, columns=['PC1', 'PC2'])

# umap = m.get_umap(pcs, 20, 0.01)
# #m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
# cluster = m.get_cluster(pcs, n_clusters=9, cluster_type='kmeans')
df = pd.concat([df_features, umap, cluster], axis=1)
df, replace_map = order_cluster_by_feature(df, cluster_name='kmeans', feature_name='avg_speed')

df_inhibit = pd.concat([df_features_inhibit, umap_inhibit, cluster_inhibit], axis=1)
df_inhibit = df_inhibit.replace({'kmeans': replace_map})

df_with_inhibit = pd.concat([df, df_inhibit], axis=0).reset_index(drop=True)
df_duration_with_inhibit = pd.concat([df_duration, df_duration_inhibit], axis=0).reset_index(drop=True)


draw_umap_space(df, path, file_name='space_kmeans_ordered', condition_name='kmeans', label_name='pseudo_Label',
                colors=cmc.batlow, dot_size=0.07, x_name='PC1', y_name='PC2')

draw_umap_space(df_with_inhibit, path, file_name='with inhibit space_kmeans_ordered', condition_name='Exp', label_name='pseudo_Label',
                colors=cmc.batlow, dot_size=0.07, x_name='PC1', y_name='PC2')


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df.to_parquet(path + 'GCB_no_inhibit_all_features_%s.parquet'%duration)
df.to_csv(path + 'GCB_no_inhibit_all_features_%s.csv'%duration, index=False)

label_expanded = np.repeat(df['kmeans'], duration).reset_index(drop=True)
df_duration['kmeans'] = label_expanded

df_duration.to_parquet(path + 'GCB_no_inhibit_traj_duration_%s.parquet'%duration)
df_duration.to_csv(path + 'GCB_no_inhibit_traj_duration_%s.csv'%duration, index=False)

df_with_inhibit.to_parquet(path + 'GCB_with_inhibit_all_features_%s.parquet'%duration)
df_with_inhibit.to_csv(path + 'GCB_with_inhibit_all_features_%s.csv'%duration, index=False)

label_expanded = np.repeat(df_with_inhibit['kmeans'], duration).reset_index(drop=True)
df_duration_with_inhibit['kmeans'] = label_expanded

df_duration_with_inhibit.to_parquet(path + 'GCB_with_inhibit_traj_duration_%s.parquet'%duration)
df_duration_with_inhibit.to_csv(path + 'GCB_with_inhibit_traj_duration_%s.csv'%duration, index=False)








#################################### Only Without inhibition Group A (Figure 1) ####################################
df_features = df_features_all[(df_features_all['Exp']=='Exp1')|(df_features_all['Exp']=='Exp2')|(df_features_all['Exp']=='Exp3')
                              |(df_features_all['Exp']=='Exp5')].reset_index(drop=True)
df_duration = df_duration_all[(df_duration_all['Exp']=='Exp1')|(df_duration_all['Exp']=='Exp2')|(df_duration_all['Exp']=='Exp3')
                              |(df_duration_all['Exp']=='Exp5')].reset_index(drop=True)

videos = np.unique(df_features['Video'])
df_features = df_features[(df_features['Video'] == videos[1])|(df_features['Video'] == videos[2])|(df_features['Video'] == videos[4])|
                 (df_features['Video'] == videos[-1])].reset_index(drop=True)

df_duration = df_duration[(df_duration['Video'] == videos[1])|(df_duration['Video'] == videos[2])|(df_duration['Video'] == videos[4])
                 |(df_duration['Video'] == videos[-1])].reset_index(drop=True)


################## UMAP and PCA ##################

df_features.columns.get_loc('inst_angle_cosine_similarity_entropies')
motility_data = df_features.iloc[:,8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)


from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)

# m = Morphodynamics(df_features, 'umap')
# cluster = m.get_cluster(pcs, n_clusters=9, cluster_type='kmeans')
#
# df_features_ = df_features.copy()
# df_features_['kmeans'] = cluster
# m.evaluate_umap(df_features_, pcs, path, n_neighbors_list=[10, 20, 30, 40, 50, 60],
#                       min_dist_list=[0.005, 0.01, 0.05, 0.1, 0.3, 0.5], condition_name ='kmeans')


from sklearn.cluster import KMeans
km = KMeans(n_clusters=9, random_state=0, init='k-means++')
# k-means++: Initialize centroids that are far away each other
kmeans_predicted = km.fit_predict(pcs)
cluster = pd.DataFrame(kmeans_predicted, columns=['kmeans'])


from umap import UMAP
__umap = UMAP(metric='euclidean', n_components=2, n_neighbors=20, min_dist=0.01, random_state=0)
pcs_array = __umap.fit_transform(pcs)
umap = pd.DataFrame(pcs_array, columns=['PC1', 'PC2'])

# umap = m.get_umap(pcs, 20, 0.01)
# #m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
# cluster = m.get_cluster(pcs, n_clusters=9, cluster_type='kmeans')
df = pd.concat([df_features, umap, cluster], axis=1)
df, replace_map = order_cluster_by_feature(df, cluster_name='kmeans', feature_name='avg_speed')


df_with_inhibit = pd.concat([df, df_inhibit], axis=0).reset_index(drop=True)


draw_umap_space(df, path, file_name='groupA_space_kmeans_ordered', condition_name='kmeans', label_name='pseudo_Label',
                colors=cmc.batlow, dot_size=0.07, x_name='PC1', y_name='PC2')


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df.to_parquet(path + 'GroupA_no_inhibit_all_features_%s.parquet'%duration)
df.to_csv(path + 'GroupA_no_inhibit_all_features_%s.csv'%duration, index=False)

label_expanded = np.repeat(df['kmeans'], duration).reset_index(drop=True)
df_duration['kmeans'] = label_expanded

df_duration.to_parquet(path + 'GroupA_no_inhibit_traj_duration_%s.parquet'%duration)
df_duration.to_csv(path + 'GroupA_no_inhibit_traj_duration_%s.csv'%duration, index=False)









# ############################## latent vector space #######################################
#
# path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'
# df_features = pd.read_parquet(path+'inhibit_all_features_20.parquet')
#
#
# motility_data = df_features.iloc[:,:128]
#
# scaler = StandardScaler()  # if not normalize, UMAP space is completely different
# #motility_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( motility_data+abs(min(motility_data.min()))+1e-10 ) ), columns=motility_data.columns)
# motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
#
# # correlation_matrix = df.iloc[:,128:240]
# # correlation_matrix = correlation_matrix.drop(['angle_distribution', 'speed_distribution', 'speed_distribution_x',
# #                                               'speed_distribution_y', 'speed_distribution_z'], axis=1)
# # test = Morphodynamics(df, 'pfa')
# # test.evaluate_pfa(correlation_matrix)
# # plt.figure(figsize=(20, 15))
# # sns.set(font_scale=0.7)
# # heatmap = sns.heatmap(test.correlation, annot=False,  yticklabels=True, xticklabels=True,
# #                       # yticklabels = ['clone 1-1','clone 1-2','clone 1-3','clone 3-3'],
# #                       cmap='RdBu_r'
# #                       )
# # #heatmap.ax_heatmap.set_xticklabels(heatmap.ax_heatmap.get_xmajorticklabels(), fontsize = 16, )
# # plt.savefig(path+'features_heatmap.png')
#
# duration=20
# path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/'
# df_duration = pd.read_parquet(path + 'GCB_traj_duration_20.parquet')
#
# path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/'
# df_duration = pd.read_parquet(path + 'inhibit_traj_duration_%s.parquet' %duration)
#
# ts = Morphodynamics(df_duration, 'umap')
# cluster, cluster_expanded, cluster_center = ts.get_ts_cluster(df_duration, 11,  duration=20, normalize=False, feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])
# cluster, cluster_expanded, cluster_center = ts.get_ts_cluster(df_duration, 11,  duration=20, normalize=False,
#                                                               feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z',
#                                                                             'Shortest_Distance_to_Surfaces_Surfaces=FDC',
#                                                                             'Shortest_Distance_to_Surfaces_Surfaces=T-cell'])
# #cluster2, cluster_expanded2, cluster_center = ts.get_ts_cluster(df_duration, 11,  duration=20, normalize=True, feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])
#
# m = Morphodynamics(df_features, 'umap')
# umap = m.get_umap(motility_data_scaled, 20, 0.5)
# df = pd.concat([df_features, umap, pd.DataFrame(cluster, columns=['tskmeans'])], axis=1)
#
# # color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')
# # draw_umap_space(df, path, file_name='space_tskmeans', condition_name='tskmeans', colors=color_list, label_name='pseudo_Label', x_name='PC1', y_name='PC2', dot_size=0.07)
#
# #m = Morphodynamics(df_features, 'umap')
# #m.evaluate_cluster(motility_data_scaled, path, cluster_type='kmeans', k_max=50)
# #cluster = m.get_cluster(motility_data_scaled, n_clusters=11, cluster_type='kmeans')
#
# path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'
# df.to_parquet(path + 'inhibit_all_features_20.parquet')
# df.to_csv(path + 'inhibit_all_features_20.csv', index=False)





