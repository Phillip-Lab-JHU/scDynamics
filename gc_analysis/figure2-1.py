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
"""Generates Data for Figure2-1. General characterization of WT and MT motility """

from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast
from features.interaction import ZoneSignal


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GCB_no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_no_inhibit_traj_duration_20.parquet')

_, _, Zone_series = to_timeseries_fast(df_duration, duration=20, feature_name='Zone')

feature_list = ['avg_zone', 'dz_resident_times', 'dz_resident_persistences', 'slz_resident_times',
                'slz_resident_persistences', 'dlz_resident_times', 'dlz_resident_persistences']
Zone_func = ZoneSignal(Zone_series)
df_zone = Zone_func.extract_features(feature_list)

df = pd.concat([df, df_zone], axis=1)

for typ in np.unique(df['Type']):
    print(typ, df[df['Type']==typ].shape[0])

for typ in np.unique(df['Video']):
    print(typ, df[df['Video']==typ].shape[0])

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure2-1. Overall behavior of wt and mt GCB\\'

#################################### PCA by video-wise (average motility features -> PCA) ###################################
df.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df.columns[8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',])


df_pca_inputs = pd.DataFrame()

for patient in np.unique(df['Video']):
    df_patient = df[(df['Video']==patient)].reset_index(drop=True)
    df_pca_input = pd.DataFrame()
    for cond in np.unique(df_patient['Type']):

        df_part = df_patient[(df_patient['Type']==cond)].reset_index(drop=True)

        motility_input = df_part[feature_list]

        pca_input = motility_input.median(axis=0).to_frame().T

        df_pca_input = pca_input
        df_pca_input['Video'] = np.unique(df_part['Video'])[0]
        df_pca_input['Type'] = np.unique(df_part['Type'])[0]

        df_pca_inputs = pd.concat([df_pca_inputs, df_pca_input], axis=0)

df_pca_inputs = df_pca_inputs.reset_index(drop=True)

pca_inputs_only = df_pca_inputs.iloc[:, :-2]

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
pca_inputs_only= pd.DataFrame(scaler.fit_transform( pca_inputs_only ), columns=pca_inputs_only.columns)

from sklearn.decomposition import PCA
pca = PCA(n_components=4)

pcs_array = pca.fit_transform(pca_inputs_only)  # factor scores for non-rotated data
df_pcs = pd.DataFrame(pcs_array, columns=['PC1', 'PC2', 'PC3', 'PC4'])

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'])
#loadings = pd.concat([df_title, loadings], axis=1)

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC1', 'PC2', 'PC3', 'PC4'])

df_pca_inputs_final = pd.concat([df_pca_inputs, df_pcs], axis=1)


x_name='PC1'
y_name='PC2'
draw_umap_space(df_pca_inputs_final, path,file_name='per video space_type',
                condition_name='Type', label_name=None, colors=('#CC6677', '#888888'), dot_size=5,
                x_name=x_name, y_name=y_name)

x_name='PC1'
y_name='PC2'
draw_umap_space(df_pca_inputs_final, path,file_name='per video space_video',
                condition_name='Video', label_name=None, colors=plt.get_cmap('Set3'), dot_size=5,
                x_name=x_name, y_name=y_name)

#################################### Plot feature correlation ####################################
df.columns.get_loc('inst_angle_cosine_similarity_entropies')

df_corr = df.iloc[:,8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1).corr()
font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)


kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

g=sns.clustermap(df_corr, annot=False, cmap=cmc.cork,
#cbar_pos=(1, 0.2, 0.03, 0.8),
linewidths=0.05, linecolor='black',
alpha=0.7,
**kws,
figsize = (36, 36),
dendrogram_ratio=0.05
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=12, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, rotation=0, va='center')


x0, _y0, _w, _h = g.cbar_pos
#g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*10, 0.02])
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation', fontsize=16)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=16)

for spine in g.ax_cbar.spines:
    #g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'feature correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/feature correlation.svg', bbox_inches='tight')
plt.clf()
plt.close()

#################################### Plot whole state space ####################################

color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
#df_['kmeans'] = df['kmeans'].astype(str)

draw_umap_space(df_, path, file_name='space_Type', condition_name='Type', label_name='pseudo_Label', colors=('#CC6677', '#888888'), x_name='PC1', y_name='PC2', dot_size=0.07)
draw_umap_space(df_, path, file_name='space_kmeans', condition_name='kmeans', label_name='pseudo_Label', colors=cmc.batlow, x_name='PC1', y_name='PC2', dot_size=0.07)


draw_jointplot(xs='PC1', y='PC2', df=df_, path=path, file_name='jointplot_type', hue="Type", colors=('#CC6677', '#888888', ), hue_order=['MT', 'WT'],
               legend=False, fill=False, thresh=0.15, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')
draw_jointplot(xs='PC1', y='PC2', df=df_, path=path, file_name='jointplot_kmeans', hue="kmeans", colors=cmc.batlow,
               legend=False, fill=True, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

draw_contour(df_, path, file_name='space_contour', condition_name='Type', colors=('#CC6677', '#888888'), x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df_, path, file_name='space_kmeans_contour', condition_name='kmeans', colors=cmc.batlow, x_name='PC1', y_name='PC2', bin_num=50, num_contours=1)

df.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df.columns[8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
draw_space_feature_magnitude(df, path, feature_list, dot_size=0.07, x_name='PC1', y_name='PC2', vmax=None)


#################################### Heatmap of cluster enrichment and shannon entropy ####################################

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_type_heatmap', condition_name='Type', cluster_type='kmeans',
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(8,2))


min_count = df['Type'].value_counts().min()
sampled_df = df.groupby('Type', group_keys=False).sample(n=min_count, random_state=42)

sampled_df_ = sampled_df.copy()
sampled_df_ = sampled_df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
draw_cluster_distribution_heatmap(sampled_df_, path, file_name='kmeans_type_heatmap_per_MC', condition_name='kmeans', cluster_type='Type',
                                  annot=True, col_cluster=False, row_cluster=False, transpose=True, vmax=80, cmap=cmc.oslo_r, figsize=(8,2))

p_dict = permutation_test(df_, group_name='Type', class_name='kmeans', iteration=50000)

draw_heatmap_with_circles(df_, path, file_name='kmeans_type_circleheatmap', condition_name='Type', cluster_type='kmeans',
                          vmax=None, transpose=False, row_cluster=False, col_cluster=False, figsize=(4,4))

draw_relative_cluster_distribution_heatmap(df_, path, file_name='relative_kmeans_type_heatmap', condition_name='Type', cluster_type='kmeans',
                                  annot=True, transpose=False, col_cluster=False, row_cluster=True, vmax=80, cmap=cmc.oslo_r, figsize=(8,2))

p_dict = permutation_test(df, group_name='Type', class_name='kmeans', iteration=10000)

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_['type_day'] = df_['Type'].astype(str) + ' ' + df_['Day'].astype(str)
draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_type_day_heatmap', condition_name='type_day', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,4))



entropies = {'mt_B-cell':[], 'wt_B-cell':[]}
for video in np.unique(df['Video']):
    df_part = df[df['Video']==video]
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        entropies[type].append(entropy[type])

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_entropies = change_dict_order(entropies, new_order)

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
entropies_ = {replace_keys.get(k, k):v  for (k,v) in ordered_entropies.items() }

dict_datasets = entropies_
file_name='entropy'
test='mann-whitney'

colors = ('#888888', '#CC6677', )
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
# plt.xticks(plt.xticks()[0], sorted_keys, fontsize=12, fontdict={'weight': 'normal'})
# ax.set_xticks(plt.xticks()[0], sorted_keys, fontsize=4)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylabel('Shannon entropy', fontsize=8, weight='normal', color='0.2')
plt.xticks(plt.xticks()[0], sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
           weight='normal')
plt.yticks(fontsize=8, color='0.2', weight='normal')
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
plt.title('%s:%s' % (pairs, p_values))
plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()

############# Plot day kmeans cross correlation for each cell type ###############
df_ = df.copy()
df_['type_day'] = df_['Type'].astype(str) + ' ' + df_['Day'].astype(str)

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

condition_name = 'Day'
cluster_type = 'kmeans'
file_name = '%s cross-correlation'%condition_name

df_corr_data = pd.DataFrame()
group_clones=[]
for group in np.unique(df[condition_name]):
#for group in ['DZ', 'sLZ', 'dLZ']:
    corrcoef = []
    aaa = df_[df_['Day'] == group].reset_index(drop=True)

    group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    ###### Find missing cluster and put 0 ######
    group_clone_T = group_clone.T
    for cluster in sorted(list(pd.unique(df[cluster_type]))):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=0)
            group_clone_T.sort_index(axis=1, inplace=True)
    group_clone = group_clone_T.T
    for column in group_clone.columns:
        group_clone.rename(columns={column: column}, inplace=True)
    group_clones.append(group_clone)
    df_corr_data = pd.concat([df_corr_data, group_clone], axis=1)

df_corr = df_corr_data.corr(method='spearman')

df_corr.columns.name = None # Remove name of column = 'Type'
df_corr.index.name = None # Remove name of index = 'Type'


mask = np.triu(df_corr) # Mask for only lower triangle

mask = mask[1:, :-1]  # mask that doesn't contain very top left and very bottom right block
corr = df_corr.iloc[1:,:-1].copy()  # df that doesn't contain very top left and very bottom right block

fig, ax = plt.subplots(figsize=(2, 2))

ax = sns.heatmap(corr, annot=True, mask=mask, cmap='OrRd', alpha=0.9, linewidths=1, linecolor='white', fmt=".2f", square= True,
                 annot_kws={'size': 4, 'weight':'normal'},
                 cbar_kws= {"shrink":0.7, 'label':'Correlation'})

cax = ax.figure.axes[-1]  # colorbar
cax.tick_params(labelsize=4)  # fontsize of tick label
cax.yaxis.label.set_size(6)  # fontsize of color bar y label

yticks = [i for i in corr.index]
xticks = [i for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=4, weight='normal')
plt.xticks(plt.xticks()[0], labels=xticks, fontsize=4, rotation=35, rotation_mode='anchor', ha='right', weight='normal')

plt.savefig(path+'%s.png'%file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg'%file_name, bbox_inches='tight')

plt.close()
plt.clf()



#################################### Box plot comparing all motility features by cell types ####################################
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
condition_name = 'Type'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
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

    draw_custom_violin_plot(dict_datasets, path+'feature_violin_plot_type/', file_name=feature_name, colors = ('#888888', '#CC6677'),
                            test='mann-whitney', pvalue=True, figsize=(1,2))

    draw_custom_box_plot(dict_datasets, path+ 'feature_box_plot_type/', file_name=feature_name, colors = ('#888888', '#CC6677'),
    strip_plot=False, test='mann-whitney', pvalue=True, figsize=(1,2))


#################################### Experimental(video by video) motility features of mt vs wt ####################################

videos = np.unique(df['Video'])
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
colors = ('#888888', '#CC6677')

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
    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}

    draw_custom_bar_plot(dict_datasets, path+'experimental_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, figsize=(1, 2))


#################################### Box plot comparing all int features by cell types ####################################
df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2'])

condition_name = 'Type'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}


if not os.path.isdir(path + 'int_feature_bar_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'int_feature_bar_plot_type/')
if not os.path.isdir(path + 'int_feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'int_feature_violin_plot_type/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    try:
        # draw_custom_box_plot(dict_datasets, path+'int_feature_box_plot_type/', file_name=feature_name, colors = ('#888888', '#CC6677'),
        #                         strip_plot=False, test='mann-whitney', pvalue=True, figsize=(1,2))
        draw_custom_violin_plot(dict_datasets, path + 'int_feature_violin_plot_type/', file_name=feature_name,colors=('#888888', '#CC6677'),
                                test='mann-whitney', pvalue=True, figsize=(1, 2))

    except:
        pass

    draw_custom_bar_plot(dict_datasets, path + 'int_feature_bar_plot_type/', file_name=feature_name,
                         strip_plot=False, colors = ('#888888', '#CC6677') , test='mann-whitney', pvalue=True, figsize=(1, 2))

#################################### Experimental(video by video) int features of mt vs wt ####################################

videos = np.unique(df['Video'])
df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2'])

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
colors = ('#888888', '#CC6677')

if not os.path.isdir(path + 'experimental_int_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_int_feature/')

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
            avg = np.median(data)
            avgs.append(avg)
        dataset[cell_type] = avgs
    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}

    draw_custom_bar_plot(dict_datasets, path+'experimental_int_feature/', file_name=feature_name,
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


draw_volcano_plot(df_p, path, file_name='motility volcano plot', z_thresh=0.4, p_thresh=20, z_name='AvgZ', p_name='Adj_Logp',
                  feature_name='Feature', figsize=(6,6))

#################################### draw 3D trajectories by kmeans ####################################
for i in np.unique(df['kmeans']):
    print('cluster: ', i, 'Cell number: ', df[df['kmeans']==i].shape[0])

np.mean( df[df['Type']=='wt_B-cell']['total_distance'] )


draw_3D_trajectory_one_figure(df_duration, path, folder_name='kmeans trajectory', duration=20,
                              n_examples=30, label_name='kmeans', feature_name=['Position X', 'Position Y', 'Position Z'], lim=150)

draw_3D_trajectory_one_figure(df_duration, path, folder_name='GCB trajectory', duration=20,
                              n_examples=30, label_name='Type', feature_name=['Position X', 'Position Y', 'Position Z'], lim=100)



################# Find traj closest to centroid #################
# motility_data = df.iloc[:,8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)
# from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
# scaler = StandardScaler()  # if not normalize, UMAP space is completely different
# #aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
# motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
#
# from sklearn.decomposition import PCA
# pca = PCA(0.95)
# pcs = pca.fit_transform(motility_data_scaled)
#
# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=9, random_state=0, init='k-means++')
# # k-means++: Initialize centroids that are far away each other
# kmeans_predicted = km.fit_predict(pcs)
# centroids = km.cluster_centers_
#
# nearest_trajs = {}
# for idx, centroid in enumerate(centroids):
#     # Calculate the Euclidean distance between each row in data and the current vector
#     distances = np.linalg.norm(pcs - centroid, axis=1)
#     nearest_trajs[idx] = np.argmin(distances)
#
#
#
# nearest_trajs = {}
# for cluster in np.unique(kmeans_predicted):
#     centroid = centroids[cluster]
#     data = pcs[kmeans_predicted == cluster]
#     distances = np.linalg.norm(data - centroid, axis=1)
#     nearest_trajs[cluster] = np.argmin(distances)
#     #nearest_trajs[cluster] = np.argsort(distances)[:10]
#
# df_ = df.copy()
# df_['kmeans'] = kmeans_predicted
# df_, replace_map = order_cluster_by_feature(df_, cluster_name='kmeans', feature_name='avg_speed')
#
# nearest_trajs = {replace_map[old_key]: value for old_key, value in nearest_trajs.items()}
# nearest_trajs = dict(sorted(nearest_trajs.items()))

draw_3D_trajectory_one_figure_GC(df_duration, path, folder_name='representative kmeans trajectory', duration=20,
                              n_examples=1, label_name='kmeans', feature_name=['Position X', 'Position Y', 'Position Z'], lim=100)

#################################### Box plot comparing all motility features by kmeans ####################################
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

condition_name = 'kmeans'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
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

g=sns.clustermap(Z_avg_df, annot=False, cmap=cmc.vik,
#cbar_pos=(1, 0.2, 0.03, 0.8),
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (44, 6),
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

#################################### Box plot comparing all interaction features by kmeans ####################################
df.columns.get_loc('quality_FDC_approach_times')
df.columns.get_loc('Core_diff_distance_autocorr_3')

motility_data = df.iloc[:,148:280].drop(['FDC_diff_distance_cov', 'T_diff_distance_cov',
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
columns_with_nan = motility_data.columns[motility_data.isna().any()].tolist()
feature_list = motility_data.drop(columns_with_nan, axis=1).columns

condition_name = 'kmeans'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
if not os.path.isdir(path + 'int_feature_violin_plot_kmeans/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'int_feature_violin_plot_kmeans/')

n_colors = np.unique(df[condition_name]).shape[0]
cm = cmc.batlow
cmap = [cm(1. * i / n_colors) for i in range(n_colors)]

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in dataset.items() }
    draw_custom_bar_plot(dict_datasets, path + 'int_feature_violin_plot_kmeans/', file_name=feature_name,
    strip_plot=False, colors=cmap, test='mann-whitney', pvalue=False, figsize=(4, 4))

#################################### Kmeans radar plot ####################################

features = ['avg_speed', 'progressivity', 'angle_cov', 'displ_cov', 'avg_angle', 'phi_max', 'exy_max', 'displ_autocorr_1']
hue='kmeans'
colors = cmc.batlow
df_keyfeatures = df.loc[:, features]
df_keyfeatures = df_keyfeatures.rename(columns={'avg_speed':'Average Speed', 'progressivity': 'Progressivity', 'angle_cov': 'Angle CV', 'displ_cov': 'Displacement CV',
                                                'avg_angle':'Average Turning Angle', 'phi_max': 'Sphericity Index', 'exy_max': 'Elongation Factor', 'displ_autocorr_1': 'Autocorrelation'})


from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
df_keyfeatures_scaled= pd.DataFrame(scaler.fit_transform( df_keyfeatures ), columns=df_keyfeatures.columns)


# min/max normalization
# df_minmax_normalized_features = (df_keyfeatures-df_keyfeatures.min())/(df_keyfeatures.max()-df_keyfeatures.min())

df_keyfeatures_scaled['kmeans'] = df['kmeans']
theta = df_keyfeatures.columns
theta.tolist()

n_colors = np.unique(df[hue]).shape[0]
from collections.abc import Iterable
if isinstance(colors, Iterable):
    cmap = colors
else:
    #cmap = [colors(1. * i / n_colors) for i in range(n_colors)]
    cmap = ['rgb' + str(colors(1. * i / (n_colors-1))[:-1]) for i in range(n_colors)]


dict_clusters={'r':[], 'theta':[], 'kmeans':[]}
#theta_manuallist = ['avg_speed', 'max_speed', 'min_speed', 'net_distance', 'progressivity', 'alpha', 'avg_angle']

df_testing = pd.DataFrame()
full_dataaslist = []
full_thetaaslist = []
full_KMEANSaslist = []

for cluster in np.unique(df_keyfeatures_scaled['kmeans']):
    per_cluster = df_keyfeatures_scaled[df_keyfeatures_scaled['kmeans'] == cluster]
    # color = kmeans_colors[cluster]

    df_spider = pd.DataFrame()
    for feature in theta:
        df_spider.loc[0, feature] = per_cluster.loc[:,feature].mean()

    data_aslist = df_spider.loc[0, :].values.tolist()
    dict_clusters['r'].append(data_aslist)
    dict_clusters['theta'].append(theta)
    cluster_list_expanded = np.repeat(cluster, n_colors)

    cluster_list = cluster
    dict_clusters['kmeans'].append(cluster_list)

    full_dataaslist.append(data_aslist)
    full_thetaaslist.append(theta)
    full_KMEANSaslist.append(cluster_list_expanded)


fullfull_dataaslist = []
fullfull_thetaaslist = []
fullfull_KMEANSaslist = []
for i in range(len(np.unique(df_keyfeatures_scaled['kmeans']))):
    for j in range(len(theta)):
        fullfull_dataaslist.append(full_dataaslist[i][j])
        fullfull_thetaaslist.append(full_thetaaslist[i][j])
        fullfull_KMEANSaslist.append(full_KMEANSaslist[i][j])


df_testing['r'] = list((fullfull_dataaslist))
df_testing['theta'] = list((fullfull_thetaaslist))
df_testing['kmeans'] = list((fullfull_KMEANSaslist))


# df_spiderplot = pd.DataFrame(dict(r=dict_clusters['r'], theta=dict_clusters['theta'],KMEANS=dict_clusters['KMEANS']))
df_spiderplot = pd.DataFrame(dict(r=df_testing['r'], theta=df_testing['theta'],kmeans=df_testing['kmeans']))

fig = px.line_polar(df_spiderplot, color='kmeans',
                    color_discrete_sequence =cmap,
                    r='r', theta='theta', line_close=True, range_r = [-2.0, 3], width=900, height=900,
                    #markers=True,
                    start_angle=90,template="plotly_white")
fig.update_traces(fill='toself')
fig.update_layout(font=dict(size=26, color="black"), margin=dict(l=190, r=150, b=150, t=150))
fig.update_traces(marker={'size': 20})
fig.update_layout(polar=dict(radialaxis=dict(gridcolor='black', angle=90, tickangle=90, tickvals=[-2, -1, 0, 1, 2, 3],
                                             showline=False, linecolor='black', linewidth=3)))


fig.update_layout(polar=dict(angularaxis=dict(showline=True, linecolor='black', linewidth=5, gridcolor='black')),plot_bgcolor='rgba(0,0,0,0)',legend=dict(
        x=1.55,  # x-coordinate of the legend
        y=1.2))
fig.write_image(path + "kmeans_radarplot.png", format='png',engine='kaleido', scale=10, width=1000, height=1000, )
#fig.write_html(path + "kmeans_radarplot.html")

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')

fig.write_image(path + 'svg/kmeans_radarplot.svg', format='svg', engine='kaleido')


for idx, cluster in enumerate(np.unique(df_keyfeatures_scaled['kmeans'])):
    fig = px.line_polar(df_spiderplot[df_spiderplot['kmeans']==cluster].reset_index(drop=True), color='kmeans',
                        color_discrete_sequence =[cmap[idx]],
                        r='r', theta='theta', line_close=True, range_r = [-2.0, 3], width=900, height=900,
                        #markers=True,
                        start_angle=90,template="plotly_white")
    fig.update_traces(fill='toself')
    fig.update_layout(font=dict(size=26, color="black"), margin=dict(l=190, r=150, b=150, t=150))
    fig.update_traces(marker={'size': 20})
    fig.update_layout(polar=dict(radialaxis=dict(gridcolor='black', angle=90, tickangle=90, tickvals=[-1, 0, 1, 2, 3],
                                                 showline=False, linecolor='black', linewidth=3)))
    fig.update_layout(polar=dict(angularaxis=dict(showline=True, linecolor='black', linewidth=5, gridcolor='black')),plot_bgcolor='rgba(0,0,0,0)',legend=dict(
            x=1.55,  # x-coordinate of the legend
            y=1.2))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(showticklabels=False),
            angularaxis=dict(showticklabels=False)
        )
    )
    fig.update_layout(showlegend=False)

    fig.write_image(path + "new_kmeans_radarplot_%s.png"%cluster, format='png',engine='kaleido', scale=10,width=1000, height=1000, )
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')

    fig.write_image(path + 'svg/new_kmeans_radarplot_%s.svg' % cluster, format='svg', engine='kaleido' )

#################################### Z scores of all interaction features wrt kmeans ####################################
df.columns.get_loc('quality_FDC_approach_times')
df.columns.get_loc('Core_diff_distance_autocorr_3')


condition_name = 'Type'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}


feature_list = ['quality_FDC_approach_times', 'quality_FDC_approach_persistences','quality_FDC_departure_times',
                'quality_FDC_departure_persistences','quality_FDC_stay_times', 'quality_FDC_stay_persistences',
                'quality_T_approach_times', 'quality_T_approach_persistences','quality_T_departure_times',
                'quality_T_departure_persistences','quality_T_stay_times', 'quality_T_stay_persistences',
                'quality_DZ_approach_times', 'quality_DZ_approach_persistences','quality_DZ_departure_times',
                'quality_DZ_departure_persistences',#'quality_DZ_stay_times', 'quality_DZ_stay_persistences',
                'quality_LZ_approach_times', 'quality_LZ_approach_persistences','quality_LZ_departure_times',
                'quality_LZ_departure_persistences',#'quality_LZ_stay_times',
                'quality_LZ_stay_persistences',
                #'quality_Core_approach_times',
                'quality_Core_approach_persistences','quality_Core_departure_times',
                'quality_Core_departure_persistences','quality_Core_stay_times', 'quality_Core_stay_persistences',
                'FDC_distance_distance_slopes', 'FDC_distance_cov','FDC_distance_average', 'FDC_diff_distance_average', 'FDC_avg_overlap', 'FDC_overlap_slopes', 'FDC_contact_times', 'FDC_contact_persistences', 'FDC_noncontact_times','FDC_noncontact_persistences',
                'T_distance_distance_slopes', 'T_distance_cov', 'T_distance_average', 'T_diff_distance_average', 'T_avg_overlap', 'T_overlap_slopes', 'T_contact_times','T_contact_persistences', 'T_noncontact_times','T_noncontact_persistences',
                'DZ_distance_distance_slopes', #'DZ_distance_cov',
                'DZ_distance_average', 'DZ_diff_distance_average',
                'LZ_distance_distance_slopes', #'LZ_distance_cov',
                'LZ_distance_average', 'LZ_diff_distance_average',
                'Core_distance_distance_slopes', 'Core_distance_cov','Core_distance_average', 'Core_diff_distance_average',\
                'dz_resident_times', 'dz_resident_persistences', 'slz_resident_times',
                'slz_resident_persistences', 'dlz_resident_times', 'dlz_resident_persistences']

important_features = []
for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }
    pairs, p_values, cohen_ds = get_various_statistics(dict_datasets, test='mann-whitney')
    if p_values[0] <= 0.05:
        important_features.append(feature_name)


from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma

Z_avg_df = pd.DataFrame()
for feature_name in important_features:
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

g=sns.clustermap(Z_avg_df, annot=False, cmap=cmc.vik,
#cbar_pos=(1, 0.2, 0.03, 0.8),
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (13, 5),
dendrogram_ratio=0.1
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=14, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=16, rotation=0, va='center')

x0, _y0, _w, _h = g.cbar_pos
#g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*10, 0.02])
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Z score', fontsize=16)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=16)

for spine in g.ax_cbar.spines:
    #g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'kmeans Z score interaction features_heatmap.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/kmeans Z score interaction features_heatmap.svg', bbox_inches='tight')
plt.clf()
plt.close()

#################################### Z scores of all interaction features wrt kmeans per cell type ####################################
df.columns.get_loc('quality_FDC_approach_times')
df.columns.get_loc('Core_diff_distance_autocorr_3')


condition_name = 'Type'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}


feature_list = ['quality_FDC_approach_times', 'quality_FDC_approach_persistences','quality_FDC_departure_times',
                'quality_FDC_departure_persistences','quality_FDC_stay_times', 'quality_FDC_stay_persistences',
                'quality_T_approach_times', 'quality_T_approach_persistences','quality_T_departure_times',
                'quality_T_departure_persistences','quality_T_stay_times', 'quality_T_stay_persistences',
                'quality_DZ_approach_times', 'quality_DZ_approach_persistences','quality_DZ_departure_times',
                'quality_DZ_departure_persistences',#'quality_DZ_stay_times', 'quality_DZ_stay_persistences',
                'quality_LZ_approach_times', 'quality_LZ_approach_persistences','quality_LZ_departure_times',
                'quality_LZ_departure_persistences',#'quality_LZ_stay_times',
                'quality_LZ_stay_persistences',
                #'quality_Core_approach_times',
                'quality_Core_approach_persistences','quality_Core_departure_times',
                'quality_Core_departure_persistences','quality_Core_stay_times', 'quality_Core_stay_persistences',
                'FDC_distance_distance_slopes', 'FDC_distance_cov','FDC_distance_average', 'FDC_diff_distance_average', 'FDC_avg_overlap', 'FDC_overlap_slopes', 'FDC_contact_times', 'FDC_contact_persistences', 'FDC_noncontact_times','FDC_noncontact_persistences',
                'T_distance_distance_slopes', 'T_distance_cov', 'T_distance_average', 'T_diff_distance_average', 'T_avg_overlap', 'T_overlap_slopes', 'T_contact_times','T_contact_persistences', 'T_noncontact_times','T_noncontact_persistences',
                'DZ_distance_distance_slopes', #'DZ_distance_cov',
                'DZ_distance_average', 'DZ_diff_distance_average',
                'LZ_distance_distance_slopes', #'LZ_distance_cov',
                'LZ_distance_average', 'LZ_diff_distance_average',
                'Core_distance_distance_slopes', 'Core_distance_cov','Core_distance_average', 'Core_diff_distance_average',\
                'dz_resident_times', 'dz_resident_persistences', 'slz_resident_times',
                'slz_resident_persistences', 'dlz_resident_times', 'dlz_resident_persistences']

important_features = []
for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }
    pairs, p_values, cohen_ds = get_various_statistics(dict_datasets, test='mann-whitney')
    if p_values[0] <=0.05:
        important_features.append(feature_name)




cell_type = 'wt_B-cell'
df_ = df[df['Type']==cell_type].reset_index(drop=True)

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma

Z_avg_df = pd.DataFrame()
for feature_name in important_features:
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
figsize = (13, 5),
dendrogram_ratio=0.1,
row_cluster=False,

)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=14, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=16, rotation=0, va='center')

x0, _y0, _w, _h = g.cbar_pos
#g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*10, 0.02])
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Z score', fontsize=16)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=16)

for spine in g.ax_cbar.spines:
    #g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'kmeans Z score interaction features_heatmap_%s.png'%cell_type, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/kmeans Z score interaction features_heatmap_%s.svg'%cell_type, bbox_inches='tight')
plt.clf()
plt.close()
####################################### Number of Ezh2 GCB vs WT #############################################
replace_keys = {'mt_B-cell':'MT', 'wt_B-cell':'WT'}

dataset = {'mt_B-cell':[], 'wt_B-cell':[]}
videos = np.unique(df['Video'])
for video in videos:
    if '-A' in video:
        continue
    df_video = df[(df['Video'] == video)&(df['Type'] != 'T-cell')]
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        df_part = df_video[df_video['Type'] == cell_type]
        cell_number_fraction = df_part.shape[0]/df_video.shape[0]
        dataset[cell_type].append(cell_number_fraction)

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dataset, new_order)

cellnumber_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }
colors=('#888888', '#CC6677')
draw_custom_bar_plot(cellnumber_datasets, path, file_name='Total GCB cell number',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))


#################################### Plot total displacement for trajectory duration ####################################

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df_duration = pd.read_parquet(path+'long_traj_duration_20.parquet')

videos = np.unique(df_duration['Video']) # Remove Group A, IgG and CD40L
df_duration = df_duration[(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])&
                          (df_duration['Video'] != videos[4])&(df_duration['Video'] != videos[5])&
                          (df_duration['Video'] != videos[6])&(df_duration['Video'] != videos[7])&
                          (df_duration['Video'] != videos[8])&(df_duration['Video'] != videos[9])&
                          (df_duration['Video'] != videos[10])&(df_duration['Video'] != videos[11])&
                          (df_duration['Video'] != videos[12])&(df_duration['Video'] != videos[-1])].reset_index(drop=True)
videos = np.unique(df_duration['Video'])

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


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure2-1. Overall behavior of wt and mt GCB\\'

############# Calculate Motility features #############
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



############# Calculate FDC interaction features #############
FDC_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['Shortest_Distance_to_Surfaces_Surfaces=FDC'],
                                               equal_length=False, frame_name='Time')

FDC_overlap = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC'],
                                               equal_length=False, frame_name='Time')

from features.interaction import DistanceSignal, OverlapSignal
feature_list = ['average']
FDC_dist = DistanceSignal(FDC_distances)
df_distance = FDC_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)


feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences']
FDC_over = OverlapSignal(FDC_overlap)
df_overlap = FDC_over.extract_features(feature_list)

df_inter_FDC = pd.concat([df_distance, df_overlap], axis=1)
for column in df_inter_FDC.columns:
    df_inter_FDC.rename(columns={column:'FDC_'+column}, inplace=True)



############# Calculate Tfh interaction features #############
T_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['Shortest_Distance_to_Surfaces_Surfaces=T-cell'],
                                               equal_length=False, frame_name='Time')

T_overlap = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell'],
                                               equal_length=False, frame_name='Time')

from features.interaction import DistanceSignal, OverlapSignal, ZoneSignal
feature_list = ['average']
T_dist = DistanceSignal(T_distances)
df_distance = T_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)


feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences']
T_over = OverlapSignal(T_overlap)
df_overlap = T_over.extract_features(feature_list)

df_inter_T = pd.concat([df_distance, df_overlap], axis=1)
for column in df_inter_T.columns:
    df_inter_T.rename(columns={column:'T_'+column}, inplace=True)



df_long = pd.concat([df_basic, df_inter_FDC, df_inter_T, pd.DataFrame(label_list, columns=['Type', 'Time_span', 'Label'])], axis=1)
df_long = df_long[df_long['Type']!='T-cell'].reset_index(drop=True)
df_long['Time_span'] = df_long['Time_span'].astype(float) / 2  # Change frame -> min


df_long_ = df_long.replace({'Type': {'wt B-cell': 'WT', 'mt B-cell': 'MT'}})
df_long_.columns.get_loc('T_contact_persistences')
feature_list = df_long_.columns[:47]

draw_lineplot_by_custom_ranges(df_long_, path, folder_name='motility_feature_wrt_elapsed_time', feature_list=feature_list,
                                   condition_name='Type', custsom_range=(10, 22), stepsize=1, range_feature='Time_span',
                                       color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label='Length of trajectory (min)',
                                   estimator='mean', error_type='ci_norm', replace_keys=None, pvalue=True, test='mann-whitney')







df_long = pd.concat([df_basic, df_inter_FDC, df_inter_T, pd.DataFrame(label_list, columns=['Type', 'Time_span', 'Label'])], axis=1)
df_long['Time_span'] = df_long['Time_span'].astype(float) / 2  # Change frame -> min
df_long_ = df_long.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT', 'T-cell': 'Tfh'}})
df_long_.columns.get_loc('angle_partial_autocorr_3')
feature_list = df_long_.columns[:37]

draw_lineplot_by_custom_ranges(df_long_, path, folder_name='all cells motility_feature_wrt_elapsed_time', feature_list=feature_list,
                                   condition_name='Type', custsom_range=(10, 20), stepsize=1, range_feature='Time_span',
                                       color_list=['#DDCC77', '#CC6677', '#888888'], marker_list=['.', 'o', '^', ], figsize=(4,4), x_label='Length of trajectory (min)',
                                   estimator='median', replace_keys=None, pvalue=True, test='mann-whitney')




#################################### Correlation between morphodynamics and speed ####################################

linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = 1
ncols = np.unique(df['Type']).size
from matplotlib.ticker import MaxNLocator
font = {'family': 'arial',
            'weight': 'normal',
            'size': 10}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = linewidth

fig,axes = plt.subplots(nrows, ncols, figsize=(8, 4), sharey='row')
for row, type in enumerate(['wt_B-cell', 'mt_B-cell']):
    ax = axes[row]
    df_part_= df[ (df['Type']==type)  ].reset_index(drop=True)
    sns.regplot(x='avg_speed', y='morpho_avg_speed', data=df_part_, scatter_kws={"color":"black", "alpha":0.3, 's':4}, line_kws={"color":"black"}, ax=ax)

    r, p = scipy.stats.pearsonr(df_part_['avg_speed'], df_part_['morpho_avg_speed'])
    plt.text(0.7, 0.2, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
             fontsize=14, fontdict={'weight': 'normal'}, color="black")
    plt.text(0.7, 0.1, "p = " + str(format(p, ".2e")), ha='left', va='top', transform=ax.transAxes,
             fontsize=12, fontdict={'weight': 'normal'}, color="black")

    ax.spines["left"].set_visible(True)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['left'].set_color('0.2')

    ax.spines["bottom"].set_visible(True)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['bottom'].set_color('0.2')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=linewidth, color='0.2', labelsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel('Average speed (µm/min)', fontsize=12, weight='normal', color='0.2', labelpad=5)
    ax.set_ylabel('Shape Deformability', fontsize=12, weight='normal', color='0.2', labelpad=5)
    #ax.set_xlim(16, 98)

plt.savefig(path + 'corr btw speed and deformability.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/corr btw speed and deformability.svg', bbox_inches='tight')
plt.clf()
plt.close()


#################################### Correlation between int features and speed ####################################

df.columns.get_loc('quality_FDC_approach_times')
df.columns.get_loc('Core_diff_distance_autocorr_3')

motility_data = df.iloc[:,148:280].drop(['FDC_diff_distance_cov', 'T_diff_distance_cov',
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
columns_with_nan = motility_data.columns[motility_data.isna().any()].tolist()
feature_list = motility_data.drop(columns_with_nan, axis=1).columns

if not os.path.isdir(path + 'corr between speed and int feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'corr between speed and int feature/')

for feature in feature_list:

    linewidth = 1.5
    fontsize = 16
    width=6
    ratio=5
    space=0.2
    nrows = 1
    ncols = np.unique(df['Type']).size
    from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig,axes = plt.subplots(nrows, ncols, figsize=(8, 4), sharey='row')
    for row, type in enumerate(['wt_B-cell', 'mt_B-cell']):
        ax = axes[row]
        df_part_= df[ (df['Type']==type)&(df[feature]!=0) ].reset_index(drop=True)
        sns.regplot(x='avg_speed', y='%s'%feature, data=df_part_, scatter_kws={"color":"black", "alpha":0.3, 's':4}, line_kws={"color":"black"}, ax=ax)

        r, p = scipy.stats.pearsonr(df_part_['avg_speed'], df_part_['%s'%feature])
        plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                 fontsize=14, fontdict={'weight': 'normal'}, color="black")
        plt.text(0.1, 0.88, "p = " + str(p), ha='left', va='top', transform=ax.transAxes,
                 fontsize=12, fontdict={'weight': 'normal'}, color="black")

        ax.spines["left"].set_visible(True)
        ax.spines['left'].set_linewidth(linewidth)
        ax.spines['left'].set_color('0.2')

        ax.spines["bottom"].set_visible(True)
        ax.spines['bottom'].set_linewidth(linewidth)
        ax.spines['bottom'].set_color('0.2')

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=linewidth, color='0.2', labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('Average speed (µm/min)', fontsize=12, weight='normal', color='0.2', labelpad=5)
        ax.set_ylabel('%s'%feature, fontsize=12, weight='normal', color='0.2', labelpad=5)
        #ax.set_xlim(16, 98)

    plt.savefig(path + 'corr between speed and int feature/%s.png'%feature, dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/corr between speed and int feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/corr between speed and int feature/')
    plt.savefig(path + 'svg/corr between speed and int feature/%s.svg'%feature, bbox_inches='tight')
    plt.clf()
    plt.close()



#################################### Correlation between int features and speed ####################################

df.columns.get_loc('quality_FDC_approach_times')
df.columns.get_loc('Core_diff_distance_autocorr_3')

motility_data = df.iloc[:,148:280].drop(['FDC_diff_distance_cov', 'T_diff_distance_cov',
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
columns_with_nan = motility_data.columns[motility_data.isna().any()].tolist()
feature_list = motility_data.drop(columns_with_nan, axis=1).columns

if not os.path.isdir(path + 'corr between angle and int feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'corr between angle and int feature/')

for feature in feature_list:

    linewidth = 1.5
    fontsize = 16
    width=6
    ratio=5
    space=0.2
    nrows = 1
    ncols = np.unique(df['Type']).size
    from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig,axes = plt.subplots(nrows, ncols, figsize=(8, 4), sharey='row')
    for row, type in enumerate(['wt_B-cell', 'mt_B-cell']):
        ax = axes[row]
        df_part_= df[ (df['Type']==type)&(df[feature]!=0) ].reset_index(drop=True)
        sns.regplot(x='avg_angle', y='%s'%feature, data=df_part_, scatter_kws={"color":"black", "alpha":0.3, 's':4}, line_kws={"color":"black"}, ax=ax)

        r, p = scipy.stats.pearsonr(df_part_['avg_angle'], df_part_['%s'%feature])
        plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                 fontsize=14, fontdict={'weight': 'normal'}, color="black")
        plt.text(0.1, 0.88, "p = " + str(p), ha='left', va='top', transform=ax.transAxes,
                 fontsize=12, fontdict={'weight': 'normal'}, color="black")

        ax.spines["left"].set_visible(True)
        ax.spines['left'].set_linewidth(linewidth)
        ax.spines['left'].set_color('0.2')

        ax.spines["bottom"].set_visible(True)
        ax.spines['bottom'].set_linewidth(linewidth)
        ax.spines['bottom'].set_color('0.2')

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=linewidth, color='0.2', labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('Average angle (rad/min)', fontsize=12, weight='normal', color='0.2', labelpad=5)
        ax.set_ylabel('%s'%feature, fontsize=12, weight='normal', color='0.2', labelpad=5)
        #ax.set_xlim(16, 98)

    plt.savefig(path + 'corr between angle and int feature/%s.png'%feature, dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/corr between angle and int feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/corr between angle and int feature/')
    plt.savefig(path + 'svg/corr between angle and int feature/%s.svg'%feature, bbox_inches='tight')
    plt.clf()
    plt.close()