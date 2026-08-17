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
"""Generates Data for Supplement: DNN classification of WT & MT / WT & Tfh"""

from scipy import stats
from utils.draw_utils import *
from Morphology import Morphodynamics
from utils.misc_utils import *

############################## WT vs MT #######################################

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'
df_lv = pd.read_parquet(path+'GCB_with_interaction_latent_vector_20.parquet')
df_interaction = pd.read_parquet(path+'GCB_interaction_features_20.parquet')
df_motility = pd.read_parquet(path+'GCB_motility_features_20.parquet')
df = pd.concat([df_lv, df_motility, df_interaction], axis=1)
df['tskmeans'] = pd.read_parquet(path+'GCB_all_features_20.parquet')['tskmeans']


motility_data = df.iloc[:,:32]

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)

from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)

m = Morphodynamics(df, 'umap')
umap = m.get_umap(pcs, 20, 0.5)
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/supplements/DNN classification of GCB/'
m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
cluster = m.get_cluster(pcs, n_clusters=12, cluster_type='kmeans')
df = pd.concat([df, umap], axis=1)
df['clf_kmeans'] = cluster


color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')

draw_umap_space(df, path, file_name='latent space_type', condition_name='Type', label_name='pseudo_Label',
                colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df, path, file_name='latent space_clf_kmeans', condition_name='clf_kmeans', label_name='pseudo_Label',
                colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df, path, file_name='latent space_tskmeans', condition_name='tskmeans', label_name='pseudo_Label',
                colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')

draw_contour(df, path, file_name='space_contour_type', condition_name='Type', colors=color_list, x_name='PC1', y_name='PC2', bin_num=50, num_contours=3)


df.columns.get_loc('inst_angle_symbolic_dynamic_entropies')
df.columns.get_loc('FDC_total_distance')
df.columns.get_loc('norm_T_noncontact_persistences')
feature_list = df.columns[:187].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
interaction_features = df.columns[228:312]
feature_list = list(feature_list) + list(interaction_features)
draw_space_feature_magnitude(df, path, feature_list, dot_size=0.07, x_name='PC1', y_name='PC2', vmax=None)

draw_cluster_distribution_heatmap(df, path, file_name='type_clf_kmeans_heatmap', condition_name='Type', cluster_type='clf_kmeans')
draw_cluster_distribution_heatmap(df, path, file_name='tskmeans_clf_kmeans_heatmap', condition_name='tskmeans', cluster_type='clf_kmeans')


#################################### All motility feature clustermap for conditions ####################################

Z_avg_df = pd.DataFrame()
for feature_name in feature_list:
    Z_avg_df_temp = pd.DataFrame()
    Z_df = pd.DataFrame()
    pdf_temp = pd.DataFrame()
    cluster_list = []
    for cluster in np.unique(df['clf_kmeans']):
        Z_matrix = pd.DataFrame()
        Z_avg_matrix = []
        data = df[(df['clf_kmeans'] == cluster)][feature_name]
        avg = np.mean(data)
        cluster_list.append(avg)



    Z_avg_df = pd.concat([Z_avg_df,pd.DataFrame(cluster_list, columns=[feature_name])], axis=1)

Z_avg_df = Z_avg_df.replace([np.inf, -np.inf], np.nan)  # Convert inf to nan
Z_avg_df = Z_avg_df.dropna(axis=1, how='any')
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
Z_avg_df= pd.DataFrame(scaler.fit_transform( Z_avg_df ), columns=Z_avg_df.columns)

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)

if np.max(np.max(Z_avg_df)) >= abs(np.min(np.min(Z_avg_df))):
    kws = dict(cbar_kws=dict(ticks=[-round(np.max(np.max(Z_avg_df)), 1), 0, round(np.max(np.max(Z_avg_df)), 1)], orientation='horizontal'),
               vmin=-round(np.max(np.max(Z_avg_df)), 1))
else:
    kws = dict(cbar_kws=dict(ticks=[round(np.min(np.min(Z_avg_df)), 1), 0, -round(np.min(np.min(Z_avg_df)), 1)],orientation='horizontal'),
               vmin=round(np.min(np.min(Z_avg_df)), 1) )
g=sns.clustermap(Z_avg_df.T, annot=False, cmap='RdBu_r',
#cbar_pos=(1, 0.2, 0.03, 0.8),
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (6, 40),
#dendrogram_ratio=0.1
)
g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=16, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width, 0.02])

g.ax_cbar.set_title('Z score', fontsize=16)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=16)

for spine in g.ax_cbar.spines:
    #g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'clf_kmeans_features_heatmap.png', dpi=300,bbox_inches='tight')
plt.clf()
plt.close()


#################################### Correlation heatmap ####################################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'
corrs = np.load(path+'GCB_corr.npy')



path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/supplements/DNN classification of GCB/'

for type in np.unique(df['Type']):
    corr_sum = np.zeros(shape=(5, 5))
    for corr in corrs[df['Type'] == type]:
        corr_sum = corr_sum + corr

    mirror = corr_sum.copy().T
    np.fill_diagonal(mirror, 0)

    array = corr_sum + mirror

    lower_triangle_mask = np.tril(np.ones_like(array), k=0)
    result_array = array * lower_triangle_mask

    font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
    matplotlib.rc('font', **font)
    #matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    #matplotlib.rcParams['lines.linewidth'] = 1

    sns.heatmap(result_array, annot=True, cmap='OrRd',
        #cbar_pos=(1, 0.2, 0.03, 0.8),
        linewidths=0.5, linecolor='black',
        alpha=0.7,
        #**kws,
        xticklabels=['P1', 'P2', 'P3', 'FDC', 'Tfh'], yticklabels=['P1', 'P2', 'P3', 'FDC', 'Tfh']
        )

    plt.savefig(path + 'corr_heatmap_%s.png' % (type), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()





############################## WT vs Tfh #######################################

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'
df_lv = pd.read_parquet(path+'no_MT_classification_latent_vector_20.parquet')
df_interaction = pd.read_parquet(path+'no_MT_interaction_features_20.parquet')
df_motility = pd.read_parquet(path+'no_MT_motility_features_20.parquet')
df = pd.concat([df_lv, df_motility, df_interaction], axis=1)
df['tskmeans'] = pd.read_parquet(path+'no_MT_all_features_20.parquet')['tskmeans']


motility_data = df.iloc[:,:32]

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)

from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)

m = Morphodynamics(df, 'umap')
umap = m.get_umap(pcs, 20, 0.5)
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/supplements/DNN classification of WT and Tfh/'
m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
cluster = m.get_cluster(pcs, n_clusters=10, cluster_type='kmeans')
df = pd.concat([df, umap], axis=1)
df['clf_kmeans'] = cluster


color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')

draw_umap_space(df, path, file_name='latent space_type', condition_name='Type', label_name='pseudo_Label',
                colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df, path, file_name='latent space_clf_kmeans', condition_name='clf_kmeans', label_name='pseudo_Label',
                colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df, path, file_name='latent space_tskmeans', condition_name='tskmeans', label_name='pseudo_Label',
                colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')

draw_contour(df, path, file_name='space_contour_type', condition_name='Type', colors=color_list, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)


df.columns.get_loc('inst_angle_symbolic_dynamic_entropies')
df.columns.get_loc('FDC_total_distance')
df.columns.get_loc('norm_T_noncontact_persistences')
feature_list = df.columns[:187].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
interaction_features = df.columns[228:312]
feature_list = list(feature_list) + list(interaction_features)
draw_space_feature_magnitude(df, path, feature_list, dot_size=0.07, x_name='PC1', y_name='PC2', vmax=None)

draw_cluster_distribution_heatmap(df, path, file_name='type_clf_kmeans_heatmap', condition_name='Type', cluster_type='clf_kmeans')
draw_cluster_distribution_heatmap(df, path, file_name='tskmeans_clf_kmeans_heatmap', condition_name='tskmeans', cluster_type='clf_kmeans')


#################################### All motility feature clustermap for conditions ####################################

Z_avg_df = pd.DataFrame()
for feature_name in feature_list:
    Z_avg_df_temp = pd.DataFrame()
    Z_df = pd.DataFrame()
    pdf_temp = pd.DataFrame()
    cluster_list = []
    for cluster in np.unique(df['clf_kmeans']):
        Z_matrix = pd.DataFrame()
        Z_avg_matrix = []
        data = df[(df['clf_kmeans'] == cluster)][feature_name]
        avg = np.mean(data)
        cluster_list.append(avg)



    Z_avg_df = pd.concat([Z_avg_df,pd.DataFrame(cluster_list, columns=[feature_name])], axis=1)

Z_avg_df = Z_avg_df.replace([np.inf, -np.inf], np.nan)  # Convert inf to nan
Z_avg_df = Z_avg_df.dropna(axis=1, how='any')
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
Z_avg_df= pd.DataFrame(scaler.fit_transform( Z_avg_df ), columns=Z_avg_df.columns)

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)

if np.max(np.max(Z_avg_df)) >= abs(np.min(np.min(Z_avg_df))):
    kws = dict(cbar_kws=dict(ticks=[-round(np.max(np.max(Z_avg_df)), 1), 0, round(np.max(np.max(Z_avg_df)), 1)], orientation='horizontal'),
               vmin=-round(np.max(np.max(Z_avg_df)), 1))
else:
    kws = dict(cbar_kws=dict(ticks=[round(np.min(np.min(Z_avg_df)), 1), 0, -round(np.min(np.min(Z_avg_df)), 1)],orientation='horizontal'),
               vmin=round(np.min(np.min(Z_avg_df)), 1) )
g=sns.clustermap(Z_avg_df.T, annot=False, cmap='RdBu_r',
#cbar_pos=(1, 0.2, 0.03, 0.8),
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (6, 40),
#dendrogram_ratio=0.1
)
g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=16, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width, 0.02])

g.ax_cbar.set_title('Z score', fontsize=16)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=16)

for spine in g.ax_cbar.spines:
    #g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'clf_kmeans_features_heatmap.png', dpi=300,bbox_inches='tight')
plt.clf()
plt.close()


#################################### Correlation heatmap ####################################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'
corrs = np.load(path+'no_MT_corr.npy')


path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/supplements/DNN classification of WT and Tfh/'
for type in np.unique(df['Type']):
    corr_sum = np.zeros(shape=(4, 4))
    for corr in corrs[df['Type'] == type]:
        corr_sum = corr_sum + corr

    mirror = corr_sum.copy().T
    np.fill_diagonal(mirror, 0)

    array = corr_sum + mirror

    lower_triangle_mask = np.tril(np.ones_like(array), k=0)
    result_array = array * lower_triangle_mask

    font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
    matplotlib.rc('font', **font)
    #matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    #matplotlib.rcParams['lines.linewidth'] = 1

    sns.heatmap(result_array, annot=True, cmap='OrRd',
        #cbar_pos=(1, 0.2, 0.03, 0.8),
        linewidths=0.5, linecolor='black',
        alpha=0.7,
        #**kws,
        xticklabels=['P1', 'P2', 'P3', 'FDC'], yticklabels=['P1', 'P2', 'P3', 'FDC']
        )

    plt.savefig(path + 'corr_heatmap_%s.png' % (type), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
