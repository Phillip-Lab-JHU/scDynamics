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

"""Generates Data for Figure6. define GCB cells with motillity + colocalization"""

from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import DistanceSignal

#################################### Pull out interaction features ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GCB_no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_no_inhibit_traj_duration_20.parquet')


_, _, Zone_series = to_timeseries_fast(df_duration, duration=20, feature_name='Zone')

feature_list = ['average']
FDC_dist = DistanceSignal(Zone_series)
df_distance = FDC_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'Zone_average'}, inplace=True)

df['Zone_average'] = df_distance


df.loc[(df['Zone_average'] < 0.4) & (df['Zone_average'] >= 0), 'Zone'] = 'DZ'
df.loc[(df['Zone_average'] < 0.8) & (df['Zone_average'] >= 0.4), 'Zone'] = 'DZ-sLZ'
df.loc[(df['Zone_average'] < 1.2) & (df['Zone_average'] >= 0.8), 'Zone'] = 'sLZ'
df.loc[(df['Zone_average'] < 1.6) & (df['Zone_average'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
df.loc[(df['Zone_average'] <= 2) & (df['Zone_average'] >= 1.6), 'Zone'] = 'dLZ'


print(df[df['Zone']=='DZ'].shape[0], df[df['Zone']=='DZ-sLZ'].shape[0], df[df['Zone']=='sLZ'].shape[0],
      df[df['Zone']=='sLZ-dLZ'].shape[0], df[df['Zone']=='dLZ'].shape[0])

duration=20
label_expanded = np.repeat(df['Zone'], duration).reset_index(drop=True)
df_duration['Zone_label'] = label_expanded

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure6. Test\\'

#################################### all motility features wrt avg Zone distance ####################################
from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
df.columns.get_loc('morpho_displ_autocorr_3')
motility_data = df.iloc[:,8:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)


df.columns.get_loc('quality_FDC_approach_times')
df.columns.get_loc('Core_diff_distance_autocorr_3')

colocalization_data = df.iloc[:,148:280].drop(['FDC_diff_distance_cov', 'T_diff_distance_cov',
                                         'Core_diff_distance_cov', 'LZ_diff_distance_cov', 'DZ_diff_distance_cov',
                                         'DZ_distance_autocorr_1', 'DZ_distance_autocorr_2', 'DZ_distance_autocorr_3',
                                         'DZ_diff_distance_autocorr_1', 'DZ_diff_distance_autocorr_2', 'DZ_diff_distance_autocorr_3',
                                         'LZ_diff_distance_autocorr_1', 'LZ_diff_distance_autocorr_2', 'LZ_diff_distance_autocorr_3',
                                         'Core_distance_autocorr_1', 'Core_distance_autocorr_2', 'Core_distance_autocorr_3',
                                         'Core_diff_distance_autocorr_1', 'Core_diff_distance_autocorr_2', 'Core_diff_distance_autocorr_3',
                                         'Core_distance_variance', 'Core_diff_distance_variance', 'DZ_distance_variance', 'DZ_diff_distance_variance',
                                         'FDC_diff_distance_variance', 'FDC_distance_variance', 'LZ_distance_variance', 'LZ_diff_distance_variance',
                                         'T_diff_distance_variance', 'T_distance_variance',
                                         ], axis=1)
columns_with_nan = colocalization_data.columns[colocalization_data.isna().any()].tolist()
colocalization_data = colocalization_data.drop(columns_with_nan, axis=1)

input_data = pd.concat([motility_data, colocalization_data], axis=1)
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
input_data_scaled= pd.DataFrame(scaler.fit_transform( input_data ), columns=input_data.columns)


from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(input_data_scaled)

from Morphology import Morphodynamics
m = Morphodynamics(df, 'umap')
cluster = m.get_cluster(pcs, n_clusters=9, cluster_type='kmeans')
df_ = df.copy()
df_['kmeans'] = cluster
m.evaluate_umap(df_, pcs, path, n_neighbors_list=[10, 20, 30, 40, 50, 60],
                      min_dist_list=[0.005, 0.01, 0.05, 0.1, 0.3, 0.5], condition_name ='kmeans')

umap = m.get_umap(pcs, 30, 0.1)
umap.rename(columns={'PC1':'UMAP1', 'PC2':'UMAP2'}, inplace=True)#m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
cluster = m.get_cluster(pcs, n_clusters=9, cluster_type='kmeans')
df_all = pd.concat([df, umap], axis=1)
df_all['all_kmeans'] = cluster

df_all, mapping = order_cluster_by_feature(df_all, cluster_name='all_kmeans', feature_name='avg_speed')

draw_umap_space(df_all, path, file_name='space_kmeans_ordered', condition_name='all_kmeans', label_name='pseudo_Label',
                colors=cmc.batlow, dot_size=0.07, x_name='UMAP1', y_name='UMAP2')

df_ = df_all.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
#################################### Basic analysis ####################################

color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')
#df_['kmeans'] = df['kmeans'].astype(str)

draw_umap_space(df_, path, file_name='space_Type', condition_name='Type', label_name='pseudo_Label', colors=('#CC6677', '#888888'), x_name='UMAP1', y_name='UMAP2', dot_size=0.07)
draw_umap_space(df_, path, file_name='space_kmeans', condition_name='kmeans', label_name='pseudo_Label', colors=cmc.batlow, x_name='UMAP1', y_name='UMAP2', dot_size=0.07)


draw_jointplot(xs='UMAP1', y='UMAP2', df=df_, path=path, file_name='jointplot_type', hue="Type", colors=('#CC6677', '#888888', ), hue_order=['mt GCB', 'wt GCB'],
               legend=False, fill=False, thresh=0.15, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')
draw_jointplot(xs='UMAP1', y='UMAP2', df=df_, path=path, file_name='jointplot_kmeans', hue="kmeans", colors=cmc.batlow,
               legend=False, fill=True, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')


feature_list = input_data.columns
draw_space_feature_magnitude(df_, path, feature_list, dot_size=0.07, x_name='UMAP1', y_name='UMAP2', vmax=None)



draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_type_heatmap', condition_name='Type', cluster_type='all_kmeans',
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(8,2))

draw_relative_cluster_distribution_heatmap(df_, path, file_name='relative_kmeans_type_heatmap', condition_name='Type', cluster_type='all_kmeans',
                                  annot=True, transpose=False, col_cluster=False, row_cluster=True, vmax=80, cmap=cmc.oslo_r, figsize=(8,2))

#################################### Z scores of all motility features wrt kmeans ####################################
from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma

Z_avg_df = pd.DataFrame()
for feature_name in feature_list:
    Z_avg_df_temp = pd.DataFrame()
    Z_df = pd.DataFrame()
    pdf_temp = pd.DataFrame()
    cluster_list = []
    for cluster in np.unique(df_['all_kmeans']):
        Z_matrix = pd.DataFrame()
        Z_avg_matrix = []
        data = df_[(df_['all_kmeans'] == cluster)][feature_name]
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

kws = dict(cbar_kws=dict(ticks=[1.5, 0, -1.5], orientation='horizontal'), vmin=-1.5, vmax=1.5 )

g=sns.clustermap(Z_avg_df, annot=False, cmap=cmc.vik,
#cbar_pos=(1, 0.2, 0.03, 0.8),
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (65, 6),
dendrogram_ratio=0.05
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=24, rotation=0, va='center')


x0, _y0, _w, _h = g.cbar_pos
#g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*10, 0.02])
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Z score', fontsize=16)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=16)

for spine in g.ax_cbar.spines:
    #g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'kmeans Z score features_heatmap.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/kmeans Z score features_heatmap.svg', bbox_inches='tight')
plt.clf()
plt.close()