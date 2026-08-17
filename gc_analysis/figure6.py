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
from features.interaction import ZoneSignal

#################################### Pull out interaction features ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GCB_no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_no_inhibit_traj_duration_20.parquet')


_, _, Zone_series = to_timeseries_fast(df_duration, duration=20, feature_name='Zone')

feature_list = ['dz_resident_times', 'dz_resident_persistences', 'slz_resident_times',
                'slz_resident_persistences', 'dlz_resident_times', 'dlz_resident_persistences', 'avg_zone']
Zone_func = ZoneSignal(Zone_series)
df_zone = Zone_func.extract_features(feature_list)
df = pd.concat([df, df_zone], axis=1)

#df = df.drop(columns=['PC1', 'PC2', 'kmeans'])

df.loc[(df['avg_zone'] < 0.4) & (df['avg_zone'] >= 0), 'Zone'] = 'DZ'
df.loc[(df['avg_zone'] < 0.8) & (df['avg_zone'] >= 0.4), 'Zone'] = 'DZ-sLZ'
df.loc[(df['avg_zone'] < 1.2) & (df['avg_zone'] >= 0.8), 'Zone'] = 'sLZ'
df.loc[(df['avg_zone'] < 1.6) & (df['avg_zone'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
df.loc[(df['avg_zone'] <= 2) & (df['avg_zone'] >= 1.6), 'Zone'] = 'dLZ'


print(df[df['Zone']=='DZ'].shape[0], df[df['Zone']=='DZ-sLZ'].shape[0], df[df['Zone']=='sLZ'].shape[0],
      df[df['Zone']=='sLZ-dLZ'].shape[0], df[df['Zone']=='dLZ'].shape[0])

duration=20
label_expanded = np.repeat(df['Zone'], duration).reset_index(drop=True)
df_duration['Zone_label'] = label_expanded

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure6. Expanded behavior\\'

#################################### all motility features wrt avg Zone distance ####################################
from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
df.columns.get_loc('morpho_displ_autocorr_3')
motility_data = df.iloc[:,8:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)


df.columns.get_loc('quality_FDC_approach_times')
df.columns.get_loc('dlz_resident_persistences')

colocalization_data = df.iloc[:,148:289].drop(['FDC_diff_distance_cov', 'T_diff_distance_cov',
                                         'Core_diff_distance_cov', 'LZ_diff_distance_cov', 'DZ_diff_distance_cov',
                                         'DZ_distance_autocorr_1', 'DZ_distance_autocorr_2', 'DZ_distance_autocorr_3',
                                         'DZ_diff_distance_autocorr_1', 'DZ_diff_distance_autocorr_2', 'DZ_diff_distance_autocorr_3',
                                         'LZ_diff_distance_autocorr_1', 'LZ_diff_distance_autocorr_2', 'LZ_diff_distance_autocorr_3',
                                         'Core_distance_autocorr_1', 'Core_distance_autocorr_2', 'Core_distance_autocorr_3',
                                         'Core_diff_distance_autocorr_1', 'Core_diff_distance_autocorr_2', 'Core_diff_distance_autocorr_3',
                                         'Core_distance_variance', 'Core_diff_distance_variance', 'DZ_distance_variance', 'DZ_diff_distance_variance',
                                         'FDC_diff_distance_variance', 'FDC_distance_variance', 'LZ_distance_variance', 'LZ_diff_distance_variance',
                                         'T_diff_distance_variance', 'T_distance_variance',
                                        'PC1', 'PC2', 'kmeans'
                                         ], axis=1)
columns_with_nan = colocalization_data.columns[colocalization_data.isna().any()].tolist()
colocalization_data = colocalization_data.drop(columns_with_nan, axis=1)

input_data = pd.concat([motility_data, colocalization_data], axis=1)


scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
input_data_scaled= pd.DataFrame(scaler.fit_transform( input_data ), columns=input_data.columns)


from sklearn.decomposition import PCA
pca = PCA(0.75)
pcs = pca.fit_transform(input_data_scaled)

from Morphology import Morphodynamics
m = Morphodynamics(df, 'umap')
# cluster = m.get_cluster(pcs, n_clusters=9, cluster_type='kmeans')
# df_ = df.copy()
# df_['beh_kmeans'] = cluster
# m.evaluate_umap(df_, pcs, path, n_neighbors_list=[10, 20, 30, 40, 50, 60],
#                       min_dist_list=[0.005, 0.01, 0.05, 0.1, 0.3, 0.5], condition_name ='kmeans')

umap = m.get_umap(pcs, 30, 0.1)
umap.rename(columns={'PC1':'beh_PC1', 'PC2':'beh_PC2'}, inplace=True)
cluster = m.get_cluster(pcs, n_clusters=8, cluster_type='kmeans')
cluster.rename(columns={'kmeans':'beh_kmeans'}, inplace=True)
df_all = pd.concat([df, umap, cluster], axis=1)

df_all, mapping = order_cluster_by_feature(df_all, cluster_name='beh_kmeans', feature_name='avg_speed')

color_list = ('#BAC8DA', '#4F609C', '#8A4F21', '#F06293', '#E9C61D', '#BCBCBC', '#BEDCB0', '#F5A9F5', )
draw_umap_space(df_all, path, file_name='space_beh_kmeans_ordered', condition_name='beh_kmeans', label_name='pseudo_Label',
                colors=color_list, dot_size=0.07, x_name='beh_PC1', y_name='beh_PC2')

df_all.to_parquet(path + 'Expanded_behavior.parquet')
df_all.to_csv(path + 'Expanded_behavior.csv', index=False)




#################################### Basic analysis ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure6. Expanded behavior\\'
df = pd.read_parquet(path+'Expanded_behavior.parquet')

df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

color_list = ('#BAC8DA', '#4F609C', '#8A4F21', '#F06293', '#E9C61D', '#BCBCBC', '#BEDCB0', '#F5A9F5' )

#df_['kmeans'] = df['kmeans'].astype(str)

draw_umap_space(df_, path, file_name='space_Type', condition_name='Type', label_name='pseudo_Label', colors=('#CC6677', '#888888'), x_name='beh_PC1', y_name='beh_PC2', dot_size=0.07)
draw_umap_space(df_, path, file_name='space_kmeans', condition_name='beh_kmeans', label_name='pseudo_Label', colors=color_list, x_name='beh_PC1', y_name='beh_PC2', dot_size=0.07)

draw_umap_space(df_, path, file_name='space_zone', condition_name='Zone', label_name='pseudo_Label', colors=('#BAC8DA', '#BEDCB0', '#8A4F21', '#4F609C', '#E9C61D'),
                x_name='beh_PC1', y_name='beh_PC2', dot_size=0.07)

draw_jointplot(xs='beh_PC1', y='beh_PC2', df=df_, path=path, file_name='jointplot_type', hue="Type", colors=('#CC6677', '#888888'), hue_order=['MT', 'WT'],
               legend=False, fill=False, thresh=0.15, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')
draw_jointplot(xs='beh_PC1', y='beh_PC2', df=df_, path=path, file_name='jointplot_kmeans', hue="beh_kmeans",
               colors=['#BAC8DA', '#4F609C', '#8A4F21', '#F06293', '#E9C61D', '#BCBCBC', '#BEDCB0', '#F5A9F5'],
               legend=False, fill=True, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')
draw_jointplot(xs='beh_PC1', y='beh_PC2', df=df_, path=path, file_name='jointplot_zone', hue="Zone", colors=('#BAC8DA', '#BEDCB0', '#8A4F21', '#4F609C', '#E9C61D'),
               legend=False, fill=True, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

feature_list = input_data.columns
draw_space_feature_magnitude(df_, path, feature_list, dot_size=0.07, x_name='beh_PC1', y_name='beh_PC2', vmax=None)



draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_type_heatmap', condition_name='Type', cluster_type='beh_kmeans',
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(8,2))
p_dict = permutation_test(df_, group_name='Type', class_name='beh_kmeans', iteration=10000)



min_count = df['Type'].value_counts().min()
sampled_df = df.groupby('Type', group_keys=False).sample(n=min_count, random_state=42)

sampled_df_ = sampled_df.copy()
sampled_df_ = sampled_df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
draw_cluster_distribution_heatmap(sampled_df_, path, file_name='kmeans_type_heatmap_per_MC', condition_name='beh_kmeans', cluster_type='Type',
                                  annot=True, col_cluster=False, row_cluster=False, transpose=True, cmap=cmc.oslo_r, figsize=(8,2))




draw_cluster_distribution_heatmap(df_, path, file_name='row_beh_kmeans_heatmap', condition_name='beh_kmeans', cluster_type='kmeans',
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(4,4))
draw_cluster_distribution_heatmap(df_, path, file_name='row_kmeans_heatmap', condition_name='kmeans', cluster_type='beh_kmeans',
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(4,4))

draw_relative_cluster_distribution_heatmap(df_, path, file_name='relative_kmeans_type_heatmap', condition_name='Type', cluster_type='beh_kmeans',
                                  annot=True, transpose=False, col_cluster=False, row_cluster=True, vmax=80, cmap=cmc.oslo_r, figsize=(8,2))

#################################### Box plot comparing all motility features by kmeans ####################################
df.columns.get_loc('morpho_displ_autocorr_3')
motility_data = df.iloc[:,8:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)


df.columns.get_loc('quality_FDC_approach_times')
df.columns.get_loc('dlz_resident_persistences')

colocalization_data = df.iloc[:,148:286].drop(['FDC_diff_distance_cov', 'T_diff_distance_cov',
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

feature_list = input_data.columns

condition_name = 'kmeans'
if not os.path.isdir(path + 'feature_violin_plot_kmeans/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_kmeans/')

n_colors = np.unique(df[condition_name]).shape[0]
cm = cmc.batlow
cmap = [cm(1. * i / n_colors) for i in range(n_colors)]

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    draw_custom_violin_plot(dataset, path + 'feature_violin_plot_kmeans/', file_name=feature_name,
    colors=cmap, test='mann-whitney', pvalue=False, figsize=(4, 4))


#################################### Z scores of all motility features wrt kmeans ####################################
from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma

Z_avg_df = pd.DataFrame()
for feature_name in feature_list:
    Z_avg_df_temp = pd.DataFrame()
    Z_df = pd.DataFrame()
    pdf_temp = pd.DataFrame()
    cluster_list = []
    for cluster in np.unique(df_['kmeans']):
        Z_matrix = pd.DataFrame()
        Z_avg_matrix = []
        data = df_[(df_['kmeans'] == cluster)][feature_name]
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


#################################### Kmeans radar plot ####################################

features = ['avg_speed', 'progressivity', 'displ_cov', 'dz_resident_times', 'slz_resident_times', 'dlz_resident_times', 'FDC_contact_persistences', 'T_contact_persistences']
hue='beh_kmeans'
colors =  ['#BAC8DA', '#4F609C', '#8A4F21', '#F06293', '#E9C61D', '#BCBCBC', '#BEDCB0', '#F5A9F5']
df_keyfeatures = df.loc[:, features]
df_keyfeatures = df_keyfeatures.rename(columns={'avg_speed':'Speed', 'progressivity':'Progressivity', 'displ_cov': 'Disp CV', 'dz_resident_times': 'DZ residence',
                                                'slz_resident_times': 'sLZ residence', 'dlz_resident_times': 'dLZ residence',
                                                'FDC_contact_persistences': 'FDC contact' , 'T_contact_persistences': 'Tfh contact'})


from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
df_keyfeatures_scaled= pd.DataFrame(scaler.fit_transform( df_keyfeatures ), columns=df_keyfeatures.columns)


# min/max normalization
# df_minmax_normalized_features = (df_keyfeatures-df_keyfeatures.min())/(df_keyfeatures.max()-df_keyfeatures.min())

df_keyfeatures_scaled['beh_kmeans'] = df['beh_kmeans']
theta = df_keyfeatures.columns
theta.tolist()

n_colors = np.unique(df[hue]).shape[0]
from collections.abc import Iterable
if isinstance(colors, Iterable):
    cmap = colors
else:
    #cmap = [colors(1. * i / n_colors) for i in range(n_colors)]
    cmap = ['rgb' + str(colors(1. * i / (n_colors-1))[:-1]) for i in range(n_colors)]


dict_clusters={'r':[], 'theta':[], 'beh_kmeans':[]}
#theta_manuallist = ['avg_speed', 'max_speed', 'min_speed', 'net_distance', 'progressivity', 'alpha', 'avg_angle']

df_testing = pd.DataFrame()
full_dataaslist = []
full_thetaaslist = []
full_KMEANSaslist = []

for cluster in np.unique(df_keyfeatures_scaled['beh_kmeans']):
    per_cluster = df_keyfeatures_scaled[df_keyfeatures_scaled['beh_kmeans'] == cluster]
    # color = kmeans_colors[cluster]

    df_spider = pd.DataFrame()
    for feature in theta:
        df_spider.loc[0, feature] = per_cluster.loc[:,feature].mean()

    data_aslist = df_spider.loc[0, :].values.tolist()
    dict_clusters['r'].append(data_aslist)
    dict_clusters['theta'].append(theta)
    cluster_list_expanded = np.repeat(cluster, n_colors)

    cluster_list = cluster
    dict_clusters['beh_kmeans'].append(cluster_list)

    full_dataaslist.append(data_aslist)
    full_thetaaslist.append(theta)
    full_KMEANSaslist.append(cluster_list_expanded)


fullfull_dataaslist = []
fullfull_thetaaslist = []
fullfull_KMEANSaslist = []
for i in range(len(np.unique(df_keyfeatures_scaled['beh_kmeans']))):
    for j in range(len(theta)):
        fullfull_dataaslist.append(full_dataaslist[i][j])
        fullfull_thetaaslist.append(full_thetaaslist[i][j])
        fullfull_KMEANSaslist.append(full_KMEANSaslist[i][j])


df_testing['r'] = list((fullfull_dataaslist))
df_testing['theta'] = list((fullfull_thetaaslist))
df_testing['beh_kmeans'] = list((fullfull_KMEANSaslist))


# df_spiderplot = pd.DataFrame(dict(r=dict_clusters['r'], theta=dict_clusters['theta'],KMEANS=dict_clusters['KMEANS']))
df_spiderplot = pd.DataFrame(dict(r=df_testing['r'], theta=df_testing['theta'], beh_kmeans=df_testing['beh_kmeans']))
rmin=-1
rmax=2

tick_range = np.arange(rmin, rmax+1, 1)
df_spiderplot['r'] = df_spiderplot['r'].clip(lower=rmin, upper=rmax)


fig = px.line_polar(df_spiderplot, color='beh_kmeans',
                    color_discrete_sequence =cmap,
                    r='r', theta='theta', line_close=True, range_r = [rmin, rmax], width=900, height=900,
                    #markers=True,
                    start_angle=90,template="plotly_white")
fig.update_traces(fill='toself')
fig.update_layout(font=dict(size=26, color="black"), margin=dict(l=190, r=150, b=150, t=150))
fig.update_traces(marker={'size': 20})
fig.update_layout(polar=dict(radialaxis=dict(gridcolor='black', angle=90, tickangle=90, tickvals=tick_range,
                                             showline=False, linecolor='black', linewidth=3)))
fig.update_layout(polar=dict(angularaxis=dict(showline=True, linecolor='black', linewidth=5, gridcolor='black')),plot_bgcolor='rgba(0,0,0,0)',legend=dict(
        x=1.55,  # x-coordinate of the legend
        y=1.2))
fig.write_image(path + "kmeans_radarplot.png", format='png',engine='orca', scale=10, width=1000, height=1000, )
#fig.write_html(path + "kmeans_radarplot.html")

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')

fig.write_image(path + 'svg/kmeans_radarplot.svg', format='svg', engine='orca')


for idx, cluster in enumerate(np.unique(df_keyfeatures_scaled['beh_kmeans'])):
    fig = px.line_polar(df_spiderplot[df_spiderplot['beh_kmeans']==cluster].reset_index(drop=True), color='beh_kmeans',
                        color_discrete_sequence =[cmap[idx]],
                        r='r', theta='theta', line_close=True, range_r = [rmin, rmax], width=900, height=900,
                        #markers=True,
                        start_angle=90,template="plotly_white")
    fig.update_traces(fill='toself')
    fig.update_layout(font=dict(size=26, color="black"), margin=dict(l=190, r=150, b=150, t=150))
    fig.update_traces(marker={'size': 20})
    fig.update_layout(polar=dict(radialaxis=dict(gridcolor='black', angle=90, tickangle=90, tickvals=tick_range,
                                                 showline=False, linecolor='black', linewidth=3)))
    fig.update_layout(polar=dict(angularaxis=dict(showline=True, linecolor='black', linewidth=5, gridcolor='black')),plot_bgcolor='rgba(0,0,0,0)',showlegend=False)
    fig.write_image(path + "kmeans_radarplot_%s.png"%cluster, format='png',engine='orca', scale=10,width=1000, height=1000, )
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')

    fig.write_image(path + 'svg/kmeans_radarplot_%s.svg' % cluster, format='svg', engine='orca' )


############# Plot LZ vs DZ jointplot for each cell type ###############

for cell_type in ['mt_B-cell', 'wt_B-cell']:

    df_part = df[df['Type']==cell_type].reset_index(drop=True)

    df_part.loc[(df_part['Zone'] == 'DZ'), 'Zone1'] = 'DZ'
    #df_part.loc[(df_part['Zone'] == 'DZ-sLZ'), 'Zone1'] = 'DZ-sLZ'
    df_part.loc[( (df_part['Zone'] == 'sLZ') | (df_part['Zone'] == 'dLZ') | (df_part['Zone'] == 'sLZ-dLZ') ), 'Zone1'] = 'LZ'
    df_part['Zone1'] = df_part.Zone1.astype(str)

    print(cell_type, df_part[df_part['Zone1']=='DZ'].shape[0], df_part[df_part['Zone1']=='DZ-sLZ'].shape[0], df_part[df_part['Zone1']=='LZ'].shape[0])

    draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s_DZ vs LZ jointplot'%cell_type, hue="Zone1", hue_order=['LZ', 'DZ'],
                   colors=('#E69965', '#BAC8DA'), fill=False, legend=False, thresh=0.25, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')


############# Plot sLZ vs dLZ jointplot for each cell type ###############

for cell_type in ['mt_B-cell', 'wt_B-cell']:

    df_part = df[df['Type']==cell_type].reset_index(drop=True)

    df_part.loc[(df_part['Zone'] == 'sLZ'), 'Zone1'] = 'sLZ'
    #df_part.loc[(df_part['Zone'] == 'sLZ-dLZ'), 'Zone1'] = 'sLZ-dLZ'
    df_part.loc[(df_part['Zone'] == 'dLZ'), 'Zone1'] = 'dLZ'
    df_part['Zone1'] = df_part.Zone1.astype(str)

    print(cell_type, df_part[df_part['Zone1']=='sLZ'].shape[0], df_part[df_part['Zone1']=='sLZ-dLZ'].shape[0], df_part[df_part['Zone1']=='dLZ'].shape[0])

    draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s_sLZ vs dLZ jointplot'%cell_type, hue="Zone1", hue_order=['sLZ', 'dLZ'],
                   colors=('#4F609C', '#8A4F21'), fill=False, legend=False, thresh=0.25, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

############# Plot Interzone kmeans cross correlation for each cell type ###############
font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

condition_name = 'Type'
cluster_type = 'kmeans'

df_corr_data = pd.DataFrame()
group_clones=[]
for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
#for group in ['DZ', 'sLZ', 'dLZ']:
    corrcoef = []
    aaa = df[df['Zone'] == group]

    group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    group_clone_T = group_clone.T
    for cluster in list(pd.unique(df[cluster_type])):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=[0, 0])
    group_clone = group_clone_T.T

    for column in group_clone.columns:
        group_clone.rename(columns={column:column+'_%s'%group}, inplace=True)
    group_clones.append(group_clone)
    df_corr_data = pd.concat([df_corr_data, group_clone], axis=1)

df_corr = df_corr_data.corr()

df_corr.columns.name = None # Remove name of column = 'Type'
df_corr.index.name = None # Remove name of index = 'Type'


rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_DZ-sLZ': 'WT DZ-sLZ', 'wt_B-cell_sLZ': 'WT sLZ',
               'wt_B-cell_sLZ-dLZ': 'WT sLZ-dLZ', 'wt_B-cell_dLZ': 'WT dLZ',
               'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_DZ-sLZ': 'MT DZ-sLZ', 'mt_B-cell_sLZ': 'MT sLZ',
               'mt_B-cell_sLZ-dLZ': 'MT sLZ-dLZ', 'mt_B-cell_dLZ': 'MT dLZ',
               }
df_corr.rename(columns=rename_keys, inplace=True)
df_corr.rename(index=rename_keys, inplace=True)

mask = np.triu(df_corr) # Mask for only lower triangle

mask = mask[1:, :-1]  # mask that doesn't contain very top left and very bottom right block
corr = df_corr.iloc[1:,:-1].copy()  # df that doesn't contain very top left and very bottom right block

fig, ax = plt.subplots(figsize=(2, 2))

ax = sns.heatmap(corr, annot=True, mask=mask, cmap=cmc.vik, alpha=0.9, linewidths=1, linecolor='white', fmt=".2f", square= True,
                 annot_kws={'size': 4, 'weight':'normal'},
                 cbar_kws= {"shrink":0.7, 'label':'Correlation'})

cax = ax.figure.axes[-1]  # colorbar
cax.tick_params(labelsize=4)  # fontsize of tick label
cax.yaxis.label.set_size(6)  # fontsize of color bar y label

yticks = [i for i in corr.index]
xticks = [i for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=4, weight='normal')
plt.xticks(plt.xticks()[0], labels=xticks, fontsize=4, rotation=35, rotation_mode='anchor', ha='right', weight='normal')

plt.savefig(path+'3 zone correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/3 zones correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

############# All Kmeans distribution heatmap ###############
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_['Type_Zone'] = df_['Type'].astype(str) + ' ' + df_['Zone']
group_df = draw_cluster_distribution_heatmap(df_, path, file_name='all_kmeans_heatmap', condition_name='Type_Zone', annot=False,
                                  cluster_type='kmeans', col_cluster=False, figsize=(6,5))
group_df = draw_cluster_distribution_heatmap(df_, path, file_name='zone_kmeans_heatmap', condition_name='Zone', annot=False,
                                  cluster_type='kmeans', col_cluster=False, figsize=(6,4), vmax=40)
draw_relative_cluster_distribution_heatmap(df_, path, file_name='relative_zone_kmeans_heatmap', condition_name='Zone', cluster_type='kmeans',
                                  annot=False, transpose=False, col_cluster=False, row_cluster=True, vmax=35, cmap=cmc.oslo_r, figsize=(6,4))


#p_dict = permutation_test(df_, group_name='Type_Zone', class_name='kmeans', iteration=50000)

############# Plot linear regression btw FDC zones and cluster enrichment ###############


mt_enrichments = pd.DataFrame()
wt_enrichments = pd.DataFrame()
for group_clone in group_clones:  # 'DZ', 'sLz', 'dLZ'
    mt_enrichments = pd.concat( [mt_enrichments, group_clone[list(group_clone.columns)[0]]], axis=1 )
    wt_enrichments = pd.concat([wt_enrichments, group_clone[list(group_clone.columns)[1]]], axis=1)

n_colors = np.unique(df['kmeans']).shape[0]
colors=cmc.batlow
cmap = [colors(1. * i / n_colors) for i in range(n_colors)]

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(4,4))  # 2 inch by 2 inch
# almost verbatim from question

rs = []
ps = []
for idx, kmeans in enumerate(mt_enrichments.index):
    r, p = scipy.stats.spearmanr(np.arange(len(list(mt_enrichments.iloc[kmeans, :].index))),
                                mt_enrichments.iloc[kmeans, :].values)
    if p>0.11:
        continue
    sns.regplot(x=np.arange(len(list(mt_enrichments.iloc[kmeans, :].index))), y=mt_enrichments.iloc[kmeans, :].values,
                ci=None, line_kws={'color':cmap[idx], 'linewidth':3}, label=kmeans, scatter=False, scatter_kws={'s': 10, 'color':cmap[idx]})
    rs.append(r)
    ps.append(p)
    # plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
    #          fontsize=20, fontdict={'weight': 'normal'}, color="blue")

plt.title('r: %s, p: %s'%(rs, ps) , fontsize=4, fontdict={'weight': 'normal'})
#plt.xticks(plt.xticks()[0], datasets['Exact_Age'], fontsize=12, fontdict={'weight': 'normal'})
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

ax.tick_params(width=2, color='0.2')

ax.set_ylabel('fraction of cluster (%)', fontsize=16, weight='normal', color='0.2')
plt.xticks(np.arange(len(list(mt_enrichments.iloc[kmeans, :].index))), ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ'], fontsize=16, rotation=35, # fontname = "Arial",
           rotation_mode='anchor', ha='right', color='0.2', weight='normal')

plt.yticks(fontsize=16, color='0.2', weight='normal')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
           loc='best')

plt.savefig(path + 'MT cluster fraction regplot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/MT cluster fraction regplot.svg', bbox_inches='tight')
plt.clf()
plt.close()



font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(4,4))  # 2 inch by 2 inch
# almost verbatim from question

rs = []
ps = []
for idx, kmeans in enumerate(wt_enrichments.index):
    r, p = scipy.stats.spearmanr(np.arange(len(list(wt_enrichments.iloc[kmeans, :].index))),
                                wt_enrichments.iloc[kmeans, :].values)
    if p>0.11:
        continue
    sns.regplot(x=np.arange(len(list(wt_enrichments.iloc[kmeans, :].index))), y=wt_enrichments.iloc[kmeans, :].values,
                ci=None, line_kws={'color':cmap[idx], 'linewidth':3}, label=kmeans, scatter=False, scatter_kws={'s': 10, 'color':cmap[idx]})
    rs.append(r)
    ps.append(p)
    # plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
    #          fontsize=20, fontdict={'weight': 'normal'}, color="blue")

plt.title('r: %s, p: %s'%(rs, ps) , fontsize=4, fontdict={'weight': 'normal'})
#plt.xticks(plt.xticks()[0], datasets['Exact_Age'], fontsize=12, fontdict={'weight': 'normal'})
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

ax.tick_params(width=2, color='0.2')

ax.set_ylabel('fraction of cluster (%)', fontsize=16, weight='normal', color='0.2')
plt.xticks(np.arange(len(list(wt_enrichments.iloc[kmeans, :].index))), ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ'], fontsize=16, rotation=35, # fontname = "Arial",
           rotation_mode='anchor', ha='right', color='0.2', weight='normal')

plt.yticks(fontsize=16, color='0.2', weight='normal')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
           loc='best')

plt.savefig(path + 'WT cluster fraction regplot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/WT cluster fraction regplot.svg', bbox_inches='tight')
plt.clf()
plt.close()


############# Shannon entropy of DZ vs sLZ vs dLZ for each video ###############

df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

group_name = 'Video'
groups = np.unique(df_[group_name])

entropies_dz = {'MT': [], 'WT':[]}
for group in groups:
    df_part = df_[(df_[group_name]==group)&(df_['Zone'] == 'DZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df_, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_dz[type].append(entropy[type])

entropies_dz_slz = {'MT': [], 'WT':[]}
for group in groups:
    df_part = df_[(df_[group_name]==group)&(df_['Zone'] == 'DZ-sLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df_, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_dz_slz[type].append(entropy[type])

entropies_slz = {'MT': [], 'WT':[]}
for group in groups:
    df_part = df_[(df_[group_name]==group)&(df_['Zone'] == 'sLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df_, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_slz[type].append(entropy[type])

entropies_slz_dlz = {'MT': [], 'WT':[]}
for group in groups:
    df_part = df_[(df_[group_name]==group)&(df_['Zone'] == 'sLZ-dLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df_, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_slz_dlz[type].append(entropy[type])

entropies_dlz = {'MT': [], 'WT':[]}
for group in groups:
    df_part = df_[(df_[group_name]==group)&(df_['Zone'] == 'dLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df_, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_dlz[type].append(entropy[type])

test = 'mann-whitney'
color_list=['#CC6677', '#888888']
marker_list=['o', '^']
entropies = {}

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(4,4))

p_value_data = {}
for idx, cell_type in enumerate(['MT', 'WT']):
    entropies = {'%s_DZ'%cell_type:entropies_dz[cell_type], '%s_DZ-sLZ'%cell_type:entropies_dz_slz[cell_type], '%s_sLZ'%cell_type:entropies_slz[cell_type],
                  '%s_sLZ-dLZ'%cell_type:entropies_slz_dlz[cell_type], '%s_dLZ'%cell_type:entropies_dlz[cell_type],}

    # entropies = {'%s_DZ' % cell_type: entropies_dz[cell_type], '%s_sLZ' % cell_type: entropies_slz[cell_type],
    #              '%s_dLZ' % cell_type: entropies_dlz[cell_type], }

    rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_DZ-sLZ': 'WT DZ-sLZ', 'wt_B-cell_sLZ': 'WT sLZ',
                   'wt_B-cell_sLZ-dLZ': 'WT sLZ-dLZ', 'wt_B-cell_dLZ': 'WT dLZ',
                   'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_DZ-sLZ': 'MT DZ-sLZ', 'mt_B-cell_sLZ': 'MT sLZ',
                    'mt_B-cell_sLZ-dLZ': 'MT sLZ-dLZ', 'mt_B-cell_dLZ': 'MT dLZ',}
    entropies_ = {rename_keys.get(k, k): v for (k, v) in entropies.items()}

    p_value_data[cell_type] = entropies_
    mean_dataset = {}
    error_dataset = {}

    for key in entropies_:
        values = entropies_[key]
        mean = np.mean(values)
        mean_dataset[key] = mean
        # error = np.std(values)
        # interval = stats.norm.interval(confidence=0.95, loc=np.mean(values), scale=stats.sem(values))
        # error = np.mean(values) - interval[0]
        error = stats.sem(values)
        error_dataset[key] = error

    sns.lineplot(data=mean_dataset, x=np.arange(len(list(mean_dataset))), y=mean_dataset.values(),
                 label=cell_type, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars',
                 color=color_list[idx])
    ax.errorbar(np.arange(len(list(mean_dataset))), mean_dataset.values(), [x for x in error_dataset.values()],
                color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

handles, labels = ax.get_legend_handles_labels()
# ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['MT'], 0.1*std_dataset['MT'], color='#CC6677',
#            capsize=3, capthick=1, elinewidth=1.5)
# ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['WT'], 0.1*std_dataset['WT'], color='#888888',
#            capsize=3, capthick=1, elinewidth=1.5)

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_ylabel('Shannon entropy', fontsize=16, weight='normal', color='0.2')

#ax.set_xlabel('%s' % x_label, fontsize=16, weight='normal', color='0.2')

plt.xticks(np.arange(len(list(mean_dataset))), ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ'], fontsize=16, rotation=35, # fontname = "Arial",
           rotation_mode='anchor', ha='right', color='0.2', weight='normal')

plt.yticks(fontsize=16, color='0.2', weight='normal')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
           loc='best')

from scipy import stats
p_values = []
pairs = []
for (mt_key, mt_values), (wt_keys, wt_values) in zip(p_value_data['MT'].items(), p_value_data['WT'].items()):
    if test == 'mann-whitney':
        stat_test = stats.mannwhitneyu(mt_values, wt_values)
    elif test == 't-test':
        stat_test = stats.ttest_ind(mt_values, wt_values)
    elif test == 'wilcoxon-ranksum':
        stat_test = stats.ranksums(mt_values, wt_values)
    #print(mt_key, stat_test.pvalue)
    p_values.append(stat_test.pvalue)
    pairs.append(mt_key)

plt.title('%s:%s' % (pairs, p_values), fontsize=4)
plt.savefig(path + 'entropy of different zones.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/' ):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/entropy of different zones.svg', bbox_inches='tight')
plt.clf()
plt.close()



############################### Motility cluster transitions of GCB for two nodes ################################
entropies_all = {}
cluster_type = 'beh_kmeans'
for cell_type in ['wt_B-cell', 'mt_B-cell']:
    cluster_size = np.unique(df[cluster_type]).size

    df_trans = df[df['Type']==cell_type].reset_index(drop=True)

    transit_array_temp = np.empty((cluster_size,cluster_size), dtype = 'object')
    for row in range(0,transit_array_temp.shape[0]):
        for col in range(0,transit_array_temp.shape[1]):
            transit_array_temp[row,col] = []

    for label in np.unique(df_trans['Label']):
        each_label = df_trans[df_trans['Label']==label].reset_index(drop=True)
        if each_label.shape[0] >= 2:
            for idx in range(each_label.shape[0] - 1):
                row = each_label[cluster_type][idx]
                col = each_label[cluster_type][idx + 1]
                transit_array_temp[row, col].append(1)

    transit_array = np.empty((cluster_size,cluster_size), dtype = 'float')
    for row in range(0,transit_array_temp.shape[0]):
        for col in range(0,transit_array_temp.shape[1]):
            transit_array[row,col] = len(transit_array_temp[row,col])

    ############## Heatmap ##############
    if cell_type == 'wt_B-cell':
        cutoff = 5
    elif cell_type == 'mt_B-cell':
        cutoff = 9

    transit_array_heatmap = 100 * transit_array / np.sum(transit_array, axis=1)[:, np.newaxis]
    zero_transition_clusters = np.sum(transit_array, axis=1) <= cutoff
    transit_array_heatmap[zero_transition_clusters, :] = 0

    draw_clustermap(transit_array_heatmap, path, file_name='MC_transition_%s'%cell_type, vmax=30, annot=False, metric='euclidean', transpose=False,
                        row_cluster=False, col_cluster=False, cmap='OrRd', figsize=(4,4))

    transit_array_heatmap_col_norm = 100 * transit_array / np.sum(transit_array, axis=0)

    draw_clustermap(transit_array_heatmap_col_norm, path, file_name='MC_transition_%s_col_norm' % cell_type, vmax=30, annot=False,
                    metric='euclidean', transpose=False,
                    row_cluster=False, col_cluster=False, cmap='OrRd', figsize=(4, 4))

    #zero_transition_clusters = np.sum(transit_array, axis=0) <= cutoff
    #transit_array_heatmap_col_norm[zero_transition_clusters, :] = 0
    ################## Row-wise entropy ##################
    entropies = {}
    for idx, row in enumerate(transit_array_heatmap):
        prob = row / 100
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log(prob))
        entropies[idx] = entropy

    entropies_all[cell_type] = entropies


    max_entropy = - np.log(1/9)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.lineplot(data=entropies, x=np.arange(len(list(entropies))), y=entropies.values(),
                 label=cell_type, lw=2.5, marker='o', dashes=False, markersize=8, err_style='bars',
                 color='#661100')
    ax.axhline(max_entropy, linestyle='--', linewidth=1, color='0.2')

    handles, labels = ax.get_legend_handles_labels()


    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')
    ax.set_ylabel('Shannon entropy', fontsize=16, weight='normal', color='0.2')

    plt.xticks(np.arange(len(list(entropies))), fontsize=16, color='0.2', weight='normal')
    plt.yticks(fontsize=16, color='0.2', weight='normal')
    plt.ylim(-0.1, max_entropy+0.1)

    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
               loc='best')
    ax.legend().remove()

    #plt.title('row entropy_%s' % (cell_type), fontsize=4)
    plt.savefig(path + 'row entropy_%s.png' % (cell_type), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/row entropy_%s.svg' % (cell_type), bbox_inches='tight')
    plt.clf()
    plt.close()

    ############## Sankey plot ##############
    values = list(transit_array.flatten())

    node_dict = {}
    chars = ['A', 'B']
    for idx, char in enumerate(chars):
        for i in range(cluster_size):
            key = f'%s{i}'% char
            node_dict[key] = idx*cluster_size + i

    source = []
    for char in chars:
        for i in range(cluster_size):
            for j in range(cluster_size):
                source.append(f'%s{i}'% char)

    target = []
    chars = ['B']
    for idx, char in enumerate(chars):
        for i in range(cluster_size):
            for j in range(cluster_size):
                target.append(f'%s{j}'%char)


    source_node = [node_dict[x] for x in source]
    target_node = [node_dict[x] for x in target]

    node_label = ['B%s'%i for i in range(cluster_size)]*2


    n_colors = np.unique(df[cluster_type]).shape[0]
    colors=cmc.batlow
    cmap = ['rgb'+str(colors(1. * i / n_colors)[:-1]) for i in range(n_colors)]
    #cmap = (colors(1. * i / n_colors) for i in range(n_colors)]

    link_color_list = []
    for i in range(cluster_size):
        for color in cmap:
            link_color_list.append(color)

    import plotly.graph_objects as go # Import the graphical object

    fig = go.Figure(
        data=[go.Sankey( # The plot we are interest
            # This part is for the node information
            arrangement = 'snap',
            orientation = 'h',
            node = dict(
                label = node_label,
                #x = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
                #y = [0.1, 0.1, 0.1, 0.7, 0.5, 0.3, 0.7, 0.5, 0.3],
                color = cmap*2
            ),
            # This part is for the link information
            link = dict(
                source = source_node,
                target = target_node,
                value = values,
                color = link_color_list
            )
        )
             ]
    )

    fig.write_html(path + '2 node sankey-plot_%s.html' % cell_type)


max_entropy = - np.log(1/9)
fig, ax = plt.subplots(figsize=(4, 2))
sns.lineplot(data=entropies_all['wt_B-cell'], x=np.arange(len(list(entropies_all['wt_B-cell']))), y=entropies_all['wt_B-cell'].values(),
             label='wt_B-cell', lw=2.5, marker='o', dashes=False, markersize=8, err_style='bars',
             color='#888888')

sns.lineplot(data=entropies_all['mt_B-cell'], x=np.arange(len(list(entropies_all['mt_B-cell']))), y=entropies_all['mt_B-cell'].values(),
             label='mt_B-cell', lw=2.5, marker='o', dashes=False, markersize=8, err_style='bars',
             color='#CC6677')

ax.axhline(max_entropy, linestyle='--', linewidth=2, color='0.2')

handles, labels = ax.get_legend_handles_labels()


for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')
ax.set_ylabel('Shannon entropy', fontsize=16, weight='normal', color='0.2')

plt.xticks(np.arange(len(list(entropies))), fontsize=16, color='0.2', weight='normal')
plt.yticks(fontsize=16, color='0.2', weight='normal')
plt.ylim(-0.1, max_entropy+0.1)

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
           loc='best')
ax.legend().remove()

#plt.title('row entropy_%s' % (cell_type), fontsize=4)
plt.savefig(path + 'row entropy in one figure.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/row entropy in one figure.svg', bbox_inches='tight')
plt.clf()
plt.close()

############################### Zonal MC transitions of GCB for two nodes ################################

cluster_type = 'kmeans'
for cell_type in ['wt_B-cell', 'mt_B-cell']:
    cluster_size = np.unique(df[cluster_type]).size

    for zone in ['DZ', 'sLZ', 'dLZ']:
        df_trans = df[(df['Type']==cell_type)&(df['Zone']==zone)].reset_index(drop=True)

        transit_array_temp = np.empty((cluster_size,cluster_size), dtype = 'object')
        for row in range(0,transit_array_temp.shape[0]):
            for col in range(0,transit_array_temp.shape[1]):
                transit_array_temp[row,col] = []

        for label in np.unique(df_trans['Label']):
            each_label = df_trans[df_trans['Label']==label].reset_index(drop=True)
            if each_label.shape[0] >= 2:
                for idx in range(each_label.shape[0] - 1):
                    row = each_label[cluster_type][idx]
                    col = each_label[cluster_type][idx + 1]
                    transit_array_temp[row, col].append(1)

        transit_array = np.empty((cluster_size,cluster_size), dtype = 'float')
        for row in range(0,transit_array_temp.shape[0]):
            for col in range(0,transit_array_temp.shape[1]):
                transit_array[row,col] = len(transit_array_temp[row,col])

        ############## Heatmap ##############
        if cell_type =='wt_B-cell':
            cutoff = 5
        elif cell_type =='mt_B-cell':
            cutoff = 9

        transit_array_heatmap = 100*transit_array/np.sum(transit_array, axis=1)[:, np.newaxis]
        zero_transition_clusters = np.sum(transit_array, axis=1)<=cutoff
        transit_array_heatmap[zero_transition_clusters,:] = 0
        draw_clustermap(transit_array_heatmap, path, file_name='MC_transition_%s_%s'%(cell_type, zone), vmax=30, annot=False, metric='euclidean', transpose=False,
                            row_cluster=False, col_cluster=False, cmap='OrRd', figsize=(4,4))