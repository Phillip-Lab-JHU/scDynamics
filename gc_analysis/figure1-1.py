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
"""Generates Data for Figure1-1. General Behavior of T and wt GCB """
import matplotlib.pyplot as plt

from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast


#################################### Plot whole state space ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GroupA_no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GroupA_no_inhibit_traj_duration_20.parquet')



path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure1. Dynamic Behavior of T and wt GCB\Overall behavior\\'

# xmin = math.floor(df['PC1'].min()) - 1
# xmax = math.ceil(df['PC1'].max()) + 1
# ymin = math.floor(df['PC2'].min()) - 1
# ymax = math.ceil(df['PC2'].max()) + 1

# fig, ax = plt.subplots(figsize=(2, 2))
# # plt.subplot(len(n_neighbors_list), len(min_dist_list), i)
#
# font = {'family': 'arial',
#             'weight': 'normal',
#             'size': 8}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
# matplotlib.rcParams['lines.linewidth'] = 1
#
# scatter = ax.scatter(df['PC1'], df['PC2'], s=0.05,color='grey')
# format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
#
# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax)
# plt.savefig(path + '%s.png' % 'plain_space', dpi=300)
# plt.clf()
# plt.close()

#color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'T-cell': 'Tfh'}})

draw_umap_space(df_, path, file_name='space_Type', condition_name='Type', label_name='pseudo_Label', colors=('#CC6677', '#6699CC'), x_name='PC1', y_name='PC2', dot_size=0.07)
draw_umap_space(df_, path, file_name='space_kmeans', condition_name='kmeans', label_name='pseudo_Label', colors=cmc.hawaii, x_name='PC1', y_name='PC2', dot_size=0.07)

draw_contour(df_, path, file_name='space_contour', condition_name='Type', colors=('#CC6677', '#6699CC'), x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df_, path, file_name='space_kmeans_contour', condition_name='kmeans', colors=cmc.batlow, x_name='PC1', y_name='PC2', bin_num=50, num_contours=1)

draw_jointplot(xs='PC1', y='PC2', df=df_, path=path, file_name='jointplot_type', hue="Type", colors=('#CC6677', '#6699CC'),
               legend=False, fill=True, thresh=0.3, alpha=0.7, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')
draw_jointplot(xs='PC1', y='PC2', df=df_, path=path, file_name='jointplot_kmeans', hue="kmeans", colors=cmc.batlow,
               legend=False, fill=True, thresh=0.3, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

df.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df.columns[8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
draw_space_feature_magnitude(df, path, feature_list, dot_size=0.07, x_name='PC1', y_name='PC2', vmax=None)

plt.get_cmap('Set1')
#################################### Heatmap of cluster enrichment and shannon entropy ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'T-cell': 'Tfh'}})
draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_type_heatmap', condition_name='Type', cluster_type='kmeans',
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(8,2))
draw_relative_cluster_distribution_heatmap(df_, path, file_name='relative_kmeans_type_heatmap', condition_name='Type', cluster_type='kmeans',
                                  annot=True, transpose=False, col_cluster=False, row_cluster=True, cmap=cmc.oslo_r, figsize=(8,2), vmax=80)

draw_heatmap_with_circles(df_, path, file_name='kmeans_type_circleheatmap', condition_name='Type', cluster_type='kmeans',
                          vmax=None, transpose=False, row_cluster=False, col_cluster=False, figsize=(4,4))

entropies = {'T-cell':[], 'wt_B-cell':[]}
for video in np.unique(df['Video']):
    df_part = df[df['Video']==video]
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        entropies[type].append(entropy[type])

new_order = ['wt_B-cell', 'T-cell']
ordered_entropies = change_dict_order(entropies, new_order)

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
entropies_ = {replace_keys.get(k, k):v  for (k,v) in ordered_entropies.items() }

dict_datasets = entropies_
file_name='entropy'
test='mann-whitney'

colors = ('#6699CC', '#CC6677')
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(1, 2))
sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

ax = sns.barplot(data=sorted_vals, capsize=0.5, edgecolor='0.2', lw=1, errwidth=1, palette=colors)
plot_params = {'edgecolor': '0.2', 'linewidth': 1, 'fc': 'none'}
ax = sns.stripplot(data=sorted_vals, marker='s', s=1.5, **plot_params)
# marker='s'(square), s = marker size

# format_figure(ax, title=None, xlabel=None, ylabel=None, despine=True, detick=True)
ax.axhline(max_entropy, linestyle='--', linewidth=1, color='0.2')
# plt.xticks(plt.xticks()[0], sorted_keys, fontsize=12, fontdict={'weight': 'bold'})
# ax.set_xticks(plt.xticks()[0], sorted_keys, fontsize=4)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylabel('Shannon entropy', fontsize=8, weight='bold', color='0.2')
plt.xticks(plt.xticks()[0], sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
           weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')
# plt.ylabel('%s' % feature_name, fontsize=4)
# category labels
plt.grid(False)

from scipy import stats
from itertools import combinations

p_values = []
pairs = []
for pair in combinations(range(0, len(dict_datasets)), 2):  # 2 for pairs, 3 for triplets, etc
    if test == 'mann-whitney':
        stat_test = stats.mannwhitneyu(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
    elif test == 't-test':
        stat_test = stats.ttest_ind(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
    elif test == 'wilcoxon-ranksum':
        stat_test = stats.ranksums(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
    p_values.append(stat_test.pvalue)
    pairs.append(pair)
    print(pair, stat_test.pvalue)
plt.title('%s:%s' % (pairs, p_values), fontsize=4)
plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()


#################################### Box plot comparing all motility features by cell types ####################################
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
condition_name = 'Type'
replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'wt GCB'}
if not os.path.isdir(path + 'feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_type/')

if not os.path.isdir(path + 'feature_box_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_box_plot_type/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell', 'T-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    draw_custom_violin_plot(dict_datasets, path+'feature_violin_plot_type/', file_name=feature_name, colors=('#6699CC', '#CC6677'),
                            test='mann-whitney', pvalue=True, figsize=(1,2))

    draw_custom_box_plot(dict_datasets, path+ 'feature_box_plot_type/', file_name=feature_name, colors=('#6699CC', '#CC6677'),
    strip_plot=False, test='mann-whitney', pvalue=True, figsize=(1,2))


#################################### Experimental(video by video) motility features of mt vs wt ####################################

videos = np.unique(df['Video'])
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'wt GCB'}
colors=('#6699CC', '#CC6677')

if not os.path.isdir(path + 'experimental_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_motility_feature/')

for feature_name in feature_list:
    dataset = {}
    for cell_type in np.unique(df['Type']):
        df_part = df[df['Type'] == cell_type]
        avgs = []
        for video in videos:
            df_video = df_part[df_part['Video'] == video]
            if df_video.shape[0] == 0:
                continue
            data = df_video[feature_name]
            avg = np.mean(data)
            avgs.append(avg)
        dataset[cell_type] = avgs
    new_order = ['wt_B-cell', 'T-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}

    draw_custom_bar_plot(dict_datasets, path+'experimental_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, figsize=(1, 2))

#################################### Volcano plot of all motility features ####################################
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

df_p = pd.DataFrame()
for feature_name in feature_list:
    dataset = {}
    for condition in np.unique(df['Type']):
        data = df[df['Type'] == condition][feature_name]
        dataset[condition] = np.array(data)

    pvalue = get_pvalue(dataset, test='mann-whitney')
    logp = -np.log10(pvalue)

    avgZ = get_avgZ(dataset, ref_name='T-cell', data_name='wt_B-cell')

    row = pd.DataFrame()
    row['Feature'] = [feature_name]
    row['Pvalue'] = [pvalue]
    row['-Logp'] = [logp]
    row['AvgZ'] = [avgZ]
    df_p = pd.concat([df_p, row], axis=0)

df_p['Adj_p'] = adjusted_pvalues(df_p['Pvalue'], correction_type='Benjamini-Hochberg')
df_p['Adj_Logp'] = -np.log(df_p['Adj_p'])
df_p = df_p[~df_p.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf


draw_volcano_plot(df_p, path, file_name='motility volcano plot', z_thresh=0.4, p_thresh=20, z_name='AvgZ', p_name='Adj_Logp',
                  feature_name='Feature', figsize=(6,6))

#################################### draw 3D trajectories by kmeans ####################################
draw_3D_trajectory_one_figure(df_duration, path, folder_name='kmeans trajectory2', duration=20,
                              n_examples=30, label_name='kmeans', feature_name=['Position X', 'Position Y', 'Position Z'], lim=150)

draw_3D_trajectory_one_figure(df_duration, path, folder_name='lymphocyte trajectory', duration=20,
                              n_examples=20, label_name='Type', feature_name=['Position X', 'Position Y', 'Position Z'], lim=100)

#################################### Box plot comparing all motility features by kmeans ####################################
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

condition_name = 'kmeans'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
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
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in dataset.items() }
    draw_custom_violin_plot(dict_datasets, path + 'feature_violin_plot_kmeans/', file_name=feature_name,
    colors=cmap,
    test='mann-whitney', pvalue=False, figsize=(4, 4))


#################################### Z scores of all motility features wrt kmeans ####################################
df.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df.columns[8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma

Z_avg_df = pd.DataFrame()
for feature_name in feature_list:
    Z_avg_df_temp = pd.DataFrame()
    Z_df = pd.DataFrame()
    pdf_temp = pd.DataFrame()
    cluster_list = []
    for cluster in np.unique(df['kmeans']):
        Z_matrix = pd.DataFrame()
        Z_avg_matrix = []
        data = df[(df['kmeans'] == cluster)][feature_name]
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

g=sns.clustermap(Z_avg_df.T, annot=False, cmap=cmc.vik,
#cbar_pos=(1, 0.2, 0.03, 0.8),
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (6, 30),
dendrogram_ratio=0.05
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=16, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*10, 0.02])

g.ax_cbar.set_title('Z score', fontsize=16)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=16)

for spine in g.ax_cbar.spines:
    #g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'kmeans Z score features_heatmap.png', dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/kmeans Z score features_heatmap.svg', bbox_inches='tight')
plt.clf()
plt.close()

#################################### Plot total displacement for trajectory duration ####################################

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df_duration = pd.read_parquet(path+'long_traj_duration_20.parquet')


df_duration = df_duration[(df_duration['Exp']=='Exp1')|(df_duration['Exp']=='Exp2')|(df_duration['Exp']=='Exp3')].reset_index(drop=True)

videos = np.unique(df_duration['Video'])

df_duration = df_duration[(df_duration['Video'] == videos[1])|(df_duration['Video'] == videos[2])|(df_duration['Video'] == videos[4])|
                 (df_duration['Video'] == videos[-1])].reset_index(drop=True)


label_series = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['Type', 'Time_span', 'Label'], equal_length=False, frame_name='Time')
trajectories = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['Position X', 'Position Y', 'Position Z'],
                                               equal_length=False, frame_name='Time')

label_list = []
for idx, typs in label_series.items():
    label_list_temp = []
    n_columns = typs.shape[1]
    for col in range(n_columns):
        col_data = typs[:, col][0]
        label_list_temp.append(col_data)
    label_list.append(label_list_temp)

label_list = np.array(label_list)


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure1. Dynamic Behavior of T and wt GCB\Overall behavior\\'

from features.basic_motility import BasicMotility
feature_list = ['total_distance', 'avg_speed', 'max_speed', 'min_speed', 'net_distance',  'progressivity', 'alphas',
                'total_angle', 'avg_angle', 'max_angle', 'min_angle',
                'displ_variance', 'displ_cov','displ_skewness', 'displ_kurtosis', 'displ_ngaussalpha', 'displ_gini',
                'angle_variance', 'angle_cov', 'angle_skewness', 'angle_kurtosis', 'angle_ngaussalpha', 'angle_gini',
                'msds', 'displ_autocorr', 'displ_partial_autocorr', 'angle_autocorr', 'angle_partial_autocorr',
                #'displ_hurst_RS', 'angle_hurst_RS',
                ]
basic_motil = BasicMotility(trajectories, time_unit=0.5, feature_list=feature_list)
#basic_motil.plot_msd_alpha(path)
#basic_motil.plot_rotated_trajectories(path)
#basic_motil.plot_original_trajectories(path)

df_basic = basic_motil.extract_features(tau_limit=3)

df_long = pd.concat([df_basic, pd.DataFrame(label_list, columns=['Type', 'Time_span', 'Label'])], axis=1)
df_long['Time_span'] = df_long['Time_span'].astype(float) / 2  # Change frame -> min
df_long_ = df_long.replace({'Type': {'wt_B-cell': 'wt GCB', 'T-cell': 'Tfh'}})

df_long_.columns.get_loc('angle_partial_autocorr_3')
feature_list = df_long_.columns[:37]

draw_lineplot_by_custom_ranges(df_long_, path, folder_name='motility_feature_wrt_elapsed_time', feature_list=feature_list,
                                   condition_name='Type', custsom_range=(10, 20), stepsize=1, range_feature='Time_span',
                                       color_list=['#CC6677', '#6699CC'], marker_list=['o', '^', ], figsize=(4,4), x_label='Length of trajectory (min)',
                                   estimator='median', replace_keys=None, pvalue=True, test='mann-whitney')

