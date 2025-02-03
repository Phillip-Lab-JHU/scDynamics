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
"""Generates Data for Supplement."""

from utils.draw_utils import *
from Morphology import Morphodynamics
from utils.misc_utils import *
from Quantify import calculate_entropy
from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma

#################################### For GC B cells only ####################################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/'
df_features = pd.read_parquet(path+'GCB_all_features_20.parquet')

motility_data = df_features.iloc[:,:128]

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#motility_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( motility_data+abs(min(motility_data.min()))+1e-10 ) ), columns=motility_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)


# correlation_matrix = df.iloc[:,128:240]
# correlation_matrix = correlation_matrix.drop(['angle_distribution', 'speed_distribution', 'speed_distribution_x',
#                                               'speed_distribution_y', 'speed_distribution_z'], axis=1)
# test = Morphodynamics(df, 'pfa')
# test.evaluate_pfa(correlation_matrix)
# plt.figure(figsize=(20, 15))
# sns.set(font_scale=0.7)
# heatmap = sns.heatmap(test.correlation, annot=False,  yticklabels=True, xticklabels=True,
#                       # yticklabels = ['clone 1-1','clone 1-2','clone 1-3','clone 3-3'],
#                       cmap='RdBu_r'
#                       )
# #heatmap.ax_heatmap.set_xticklabels(heatmap.ax_heatmap.get_xmajorticklabels(), fontsize = 16, )
# plt.savefig(path+'features_heatmap.png')
#path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/'

#df_duration = pd.read_parquet(path + 'traj_duration_%s.parquet' %time)

m = Morphodynamics(df_features, 'umap')
umap = m.get_umap(motility_data_scaled, 20, 0.5)
df_features = pd.concat([df_features, umap], axis=1)

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/'
df_duration = pd.read_parquet(path + 'GCB_traj_duration_20.parquet')

ts = Morphodynamics(df_duration, 'umap')
cluster, cluster_expanded, cluster_center = ts.get_ts_cluster(df_duration, 5, duration=20, normalize=False, feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])
df_features['tskmeans_5'] = cluster
df_duration['tskmeans_5'] = cluster_expanded

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Figure1. Basic gc_analysis - GCB/'
xmin = math.floor(df_features['PC1'].min()) - 1
xmax = math.ceil(df_features['PC1'].max()) + 1
ymin = math.floor(df_features['PC2'].min()) - 1
ymax = math.ceil(df_features['PC2'].max()) + 1
draw_umap_space(df_features, path, file_name='space_tskmeans_5', condition_name='tskmeans_5', label_name='pseudo_Label',
                    x_name='PC1', y_name='PC2', dot_size=25,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

df_cluster_0 = df_duration[df_duration['tskmeans_5']==0]
df_cluster_1 = df_duration[df_duration['tskmeans_5']==1]
df_cluster_3 = df_duration[df_duration['tskmeans_5']==3]
df_cluster_4 = df_duration[df_duration['tskmeans_5']==4]
df_rest = df_duration[(df_duration['tskmeans_5']==2)]

cluster0, cluster_expanded0, cluster_center = ts.get_ts_cluster(df_cluster_0, 2, duration=20, normalize=False, feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])
cluster_expanded0 = pd.DataFrame(cluster_expanded0).set_index(df_cluster_0.index)
cluster_expanded0 = cluster_expanded0.replace({0: {0: 0, 1: 1}})
#cluster0 = cluster0.rename(columns={'tskmeans':0})

cluster1, cluster_expanded1, cluster_center = ts.get_ts_cluster(df_cluster_1, 2, duration=20, normalize=False, feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])
cluster_expanded1 = pd.DataFrame(cluster_expanded1).set_index(df_cluster_1.index)
cluster_expanded1 = cluster_expanded1.replace({0: {0: 3, 1: 4}})
#cluster1 = cluster1.rename(columns={'tskmeans':0})

cluster3, cluster_expanded3, cluster_center  = ts.get_ts_cluster(df_cluster_3, 2, duration=20, normalize=False, feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])
cluster_expanded3 = pd.DataFrame(cluster_expanded3).set_index(df_cluster_3.index)
cluster_expanded3 = cluster_expanded3.replace({0: {0: 5, 1: 6}})
#cluster3 = cluster3.rename(columns={'tskmeans':0})

cluster4, cluster_expanded4, cluster_center  = ts.get_ts_cluster(df_cluster_4, 2, duration=20, normalize=False, feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])
cluster_expanded4 = pd.DataFrame(cluster_expanded4).set_index(df_cluster_4.index)
cluster_expanded4 = cluster_expanded4.replace({0: {0: 7, 1: 8}})
#cluster4 = cluster4.rename(columns={'tskmeans':0})

intracluster_1 = pd.concat([cluster_expanded0, cluster_expanded1, cluster_expanded3, cluster_expanded4, df_rest['tskmeans_5']])
intracluster_sorted = intracluster_1.sort_index()

reduced_intracluster = reduced_labels(intracluster_sorted, duration=20)
df_features['tskmeans_9'] = reduced_intracluster

draw_umap_space(df_features, path, file_name='space_tskmeans_9', condition_name='tskmeans_9', label_name='pseudo_Label',
                    x_name='PC1', y_name='PC2', dot_size=25,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

df_features = df_features.rename(columns={'tskmeans_9':'tskmeans'})
df_features = df_features.replace({'tskmeans': {2:0, 8:1, 7:2, 3:3, 4:4, 5:5, 6:6, 0:7, 1:8}})

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/'
df_features.to_parquet(path + 'GCB_all_features_20.parquet')
df_features.to_excel(path + 'GCB_all_features_20.xlsx', index=False)

#################################### Plot whole state space ####################################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/'
df = pd.read_parquet(path+'GCB_all_features_20.parquet')

videos = np.unique(df['Video'])
df = df[(df['Video'] != videos[0])&(df['Video'] != videos[1])&(df['Video'] != videos[2])].reset_index(drop=True)

xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Figure4 supp. Overall behavior of wt and mt GCB/'
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
draw_umap_space(df_, path, file_name='space_Type', condition_name='Type', label_name='pseudo_Label', colors = ('#CC6677', '#888888'),
                x_name='PC1', y_name='PC2', dot_size=0.07)
draw_umap_space(df_, path, file_name='space_tskmeans', condition_name='tskmeans', label_name='pseudo_Label',
                colors = ('#888888', '#CC6677', '#44AA99', '#6699CC', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100'),
                x_name='PC1', y_name='PC2', dot_size=0.07)
draw_contour(df_, path, file_name='space_contour', condition_name='Type', colors= ('Reds', 'Greys'),
             x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df_, path, file_name='space_tskmeans_contour', condition_name='tskmeans',
             colors= ('Reds', 'Greens', 'Blues', 'Greys', 'Oranges', 'Purples', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
                  'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'plasma'),
             x_name='PC1', y_name='PC2', bin_num=50, num_contours=1)


df.columns.get_loc('inst_angle_symbolic_dynamic_entropies')
feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
draw_space_feature_magnitude(df, path, feature_list, dot_size=0.07, x_name='PC1', y_name='PC2', vmax=None)


#################################### Draw 3D trajectories ####################################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/'
df_duration = pd.read_parquet(path + 'GCB_traj_duration_20.parquet')
traj_list, _, trajectories = to_timeseries_fast(df_duration, duration=20, feature_name=['Position X', 'Position Y', 'Position Z'])

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Figure4 supp. Overall behavior of wt and mt GCB/'
draw_3D_trajectory_labels(path, trajectories, folder_name='trajectory by tskmeans', label=df['tskmeans'], idx_range=None)

#################################### Heatmap of cluster enrichment and shannon entropy ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB', 'T-cell': 'Tfh'}})
draw_cluster_distribution_heatmap(df_, path, condition_name='Type', cluster_type='tskmeans')

entropies = {'mt_B-cell':[], 'wt_B-cell':[]}
for video in np.unique(df['Video']):
    df_part = df[df['Video']==video]
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='tskmeans')
    for type in entropy:
        entropies[type].append(entropy[type])

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_entropies = change_dict_order(entropies, new_order)

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
entropies_ = {replace_keys.get(k, k):v  for (k,v) in ordered_entropies.items() }

dict_datasets = entropies_
file_name='entropy'
test='mann-whitney'

colors = ('#888888', '#CC6677')
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
# plt.title('%s:%s' % (pairs, p_values))
plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()

#draw_custom_box_plot(entropies, path, feature_name='entropy', test='mann-whitney')

#################################### Box plot comparing all motility features by cell types ####################################
df.columns.get_loc('Rotated_X')
feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
condition_name = 'Type'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
if not os.path.isdir(path + 'feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_type/')

if not os.path.isdir(path + 'feature_box_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_box_plot_type/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    draw_custom_violin_plot(dict_datasets, path+'feature_violin_plot_type/', file_name=feature_name, colors=('#888888', '#CC6677'),
                            test='mann-whitney', pvalue=True, figsize=(1,2))

    draw_custom_box_plot(dict_datasets, path+ 'feature_box_plot_type/', file_name=feature_name, colors=('#888888', '#CC6677'),
    strip_plot=False, test='mann-whitney', pvalue=True, figsize=(1,2))

#################################### Box plot comparing all motility features by tskmeans ####################################
feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
condition_name = 'tskmeans'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
if not os.path.isdir(path + 'feature_violin_plot_tskmeans/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_tskmeans/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in dataset.items() }
    draw_custom_violin_plot(dict_datasets, path + 'feature_violin_plot_tskmeans/', file_name=feature_name,
    colors=('#888888', '#CC6677', '#44AA99', '#6699CC', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', ),
    test='mann-whitney', pvalue=False, figsize=(2, 2))

#################################### Volcano plot of all motility features ####################################
feature_list = df.columns[130:283].drop(['phi', 'speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z', ])

df_p = pd.DataFrame()
for feature_name in feature_list:
    dataset = {}
    for condition in np.unique(df['Type']):
        data = df[df['Type'] == condition][feature_name]
        dataset[condition] = np.array(data)

    pvalue = get_pvalue(dataset, test='mann-whitney')
    logp = -np.log10(pvalue)

    avgZ = get_avgZ(dataset, ref_name='wt_B-cell', data_name='mt_B-cell')

    row = pd.DataFrame()
    row['Feature'] = [feature_name]
    row['Pvalue'] = [pvalue]
    row['-Logp'] = [logp]
    row['AvgZ'] = [avgZ]
    df_p = pd.concat([df_p, row], axis=0)

df_p['Adj_p'] = adjusted_pvalues(df_p['Pvalue'], correction_type='Benjamini-Hochberg')
df_p['Adj_Logp'] = -np.log(df_p['Adj_p'])
df_p = df_p[~df_p.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf


draw_volcano_plot(df_p, path, file_name='motility volcano plot', z_thresh=0.2, p_thresh=20, z_name='AvgZ', p_name='Adj_Logp',
                  feature_name='Feature', figsize=(6,6))