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
"""Generates Data for Figure5-1. Inhibition analysis"""

from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GCB_with_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_with_inhibit_traj_duration_20.parquet')

df_inhibit = df[(df['Exp']=='IgG')|(df['Exp']=='CD40L')|(df['Exp']=='mLT')].reset_index(drop=True)
df_duration_inhibit = df_duration[(df_duration['Exp']=='IgG')|(df_duration['Exp']=='CD40L')|(df_duration['Exp']=='mLT')].reset_index(drop=True)

df['Inhibition'] = df['Exp'].copy()
df['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)

df_duration['Inhibition'] = df_duration['Exp'].copy()
df_duration['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)

for i in np.unique(df['Type']):
    for j in np.unique(df['Inhibition']):
        print(i, j, df[(df['Type']==i)&(df['Inhibition']==j)].shape[0])

for i in np.unique(df['Type']):
    for j in np.unique(df['Inhibition']):
        print(i, j, np.unique(df[(df['Type']==i)&(df['Inhibition']==j)]['Video']).shape[0])

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure5. Inhibition\Overall\\'


#################################### Plot whole state space ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_inhibit_ = df_inhibit.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')

xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

for typ in np.unique(df_['Type']):
    df_part = df_[df_['Type']==typ].reset_index(drop=True)
    df_part_inhibit = df_inhibit_[df_inhibit_['Type'] == typ].reset_index(drop=True)
    print(typ, df_part_inhibit.shape[0])
    # draw_umap_space(df_part, path, file_name='%s space_exp'%typ, condition_name='Inhibition', label_name='pseudo_Label', colors=color_list, x_name='PC1', y_name='PC2', dot_size=0.07)
    # draw_umap_space(df_part_inhibit, path, file_name='%s inhibit_space_exp'%typ, condition_name='Exp', label_name='pseudo_Label', colors=color_list, x_name='PC1', y_name='PC2', dot_size=0.07)

    draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s jointplot_exp'%typ, hue="Inhibition", colors = ('#888888', '#CC6677', '#6699CC', '#44AA99'),
                   hue_order=['Control', 'IgG', 'CD40L', 'mLT'], legend=True, fill=False, thresh=0.2, height=4, ratio=5, space=0, n_contours=3,
                   xlabels='UMAP1', ylabel='UMAP2', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    # draw_jointplot(xs='PC1', y='PC2', df=df_part_inhibit, path=path, file_name='%s inhibit_jointplot_exp'%typ, hue="Exp", colors=color_list,
    #                hue_order=['IgG', 'CD40L', 'mLT'], legend=True, fill=False, thresh=0.2, height=4, ratio=5, space=0,
    #                xlabels='UMAP1', ylabel='UMAP2', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)



df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_['type_exp'] = df_['Type'].astype(str) + ' ' + df_['Exp'].astype(str)
df_['type_inhibition'] = df_['Type'].astype(str) + ' ' + df_['Inhibition'].astype(str)

draw_cluster_distribution_heatmap(df_, path, file_name='exp_kmeans_heatmap', condition_name='type_exp', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,4))
draw_cluster_distribution_heatmap(df_, path, file_name='inhibition_kmeans_heatmap', condition_name='type_inhibition', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,3))

df_inhibit_ = df_inhibit.copy()
df_inhibit_ = df_inhibit_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_inhibit_['type_exp'] = df_inhibit_['Type'].astype(str) + ' ' + df_inhibit_['Exp'].astype(str)

draw_cluster_distribution_heatmap(df_inhibit_, path, file_name='no control inhibition_kmeans_heatmap', condition_name='type_exp', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,2))

draw_relative_cluster_distribution_heatmap(df_inhibit_, path, file_name='relative_no control inhibition_kmeans_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=40, cmap=cmc.oslo_r, transpose=False, condition_name='type_exp', cluster_type='kmeans', figsize=(4,2))


group_name = 'Video'
groups = np.unique(df[group_name])

entropies_control = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Inhibition'] == 'Control')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_control[type].append(entropy[type])

entropies_igg = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Inhibition'] == 'IgG')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_igg[type].append(entropy[type])

entropies_cd40l = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Inhibition'] == 'CD40L')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_cd40l[type].append(entropy[type])

entropies_mlt = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Inhibition'] == 'mLT')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_mlt[type].append(entropy[type])


entropies = {}
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    entropies = {
        '%s Control'%cell_type:entropies_control[cell_type],
                 '%s IgG'%cell_type:entropies_igg[cell_type], '%s CD40L'%cell_type:entropies_cd40l[cell_type],
        '%s mLT' % cell_type: entropies_mlt[cell_type],
                  }

    # entropies = {'%s_DZ' % cell_type: entropies_dz[cell_type], '%s_sLZ' % cell_type: entropies_slz[cell_type],
    #              '%s_dLZ' % cell_type: entropies_dlz[cell_type], }

    # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_DZ-sLZ': 'WT DZ-sLZ', 'wt_B-cell_sLZ': 'WT sLZ',
    #                'wt_B-cell_sLZ-dLZ': 'WT sLZ-dLZ', 'wt_B-cell_dLZ': 'WT dLZ',
    #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_DZ-sLZ': 'MT DZ-sLZ', 'mt_B-cell_sLZ': 'MT sLZ',
    #                 'mt_B-cell_sLZ-dLZ': 'MT sLZ-dLZ', 'mt_B-cell_dLZ': 'MT dLZ',}
    # entropies_ = {rename_keys.get(k, k): v for (k, v) in entropies.items()}
    # draw_custom_bar_plot(entropies_, path, file_name='entropy of DZ vs sLZ vs dLZ for %s' %cell_type, colors=('#888888', '#CC6677', '#6699CC'),
    #                      strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))
    dict_datasets = entropies
    file_name = 'entropy of IgG vs CD40L for %s' %cell_type
    test = 'mann-whitney'

    colors = ('#888888', '#CC6677', '#6699CC', '#44AA99', '#DDCC77')
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
        #print(pair, stat_test.pvalue)
    plt.title('%s:%s' % (pairs, p_values), fontsize=4)
    plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()

############# Plot type+day kmeans cross correlation for each cell type ###############
df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_['type_inhibition'] = df_['Type'].astype(str) + ' ' + df_['Inhibition'].astype(str)

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

condition_name = 'type_inhibition'
cluster_type = 'kmeans'

df_corr_data = pd.DataFrame()
group_clones=[]
for group in ['WT Control', 'WT IgG', 'WT CD40L', 'WT mLT', 'MT Control', 'MT IgG', 'MT CD40L', 'MT mLT']:
#for group in ['DZ', 'sLZ', 'dLZ']:
    corrcoef = []
    aaa = df_[df_['type_inhibition'] == group].reset_index(drop=True)

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
    group_clones.append(group_clone)
    df_corr_data = pd.concat([df_corr_data, group_clone], axis=1)

df_corr = df_corr_data.corr(method='spearman')

df_corr.columns.name = None # Remove name of column = 'Type'
df_corr.index.name = None # Remove name of index = 'Type'


mask = np.triu(df_corr) # Mask for only lower triangle

mask = mask[1:, :-1]  # mask that doesn't contain very top left and very bottom right block
corr = df_corr.iloc[1:,:-1].copy()  # df that doesn't contain very top left and very bottom right block

fig, ax = plt.subplots(figsize=(2, 2))

ax = sns.heatmap(abs(corr), annot=True, mask=mask, cmap='OrRd', alpha=0.9, linewidths=1, linecolor='white', fmt=".2f", square= True,
                 annot_kws={'size': 4, 'weight':'normal'},
                 cbar_kws= {"shrink":0.7, 'label':'Correlation'})

cax = ax.figure.axes[-1]  # colorbar
cax.tick_params(labelsize=4)  # fontsize of tick label
cax.yaxis.label.set_size(6)  # fontsize of color bar y label

yticks = [i for i in corr.index]
xticks = [i for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=4, weight='normal')
plt.xticks(plt.xticks()[0], labels=xticks, fontsize=4, rotation=35, rotation_mode='anchor', ha='right', weight='normal')

plt.savefig(path+'inhibition correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/inhibition correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

#################################### Box plot comparing all motility features by cell types ####################################
if not os.path.isdir(path + 'feature_violin_plot_inhibit/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_inhibit/')

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['WT', 'MT']:
        for group in ['Control','IgG', 'CD40L', 'mLT']:
        #for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
            data = df_[(df_[condition_name] == cell_type)&(df_['Inhibition'] == group)][feature_name]

            dataset[cell_type+' '+str(group)] = np.array(data)

    #dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_violin_plot(dataset, path + 'feature_violin_plot_inhibit/', file_name=feature_name,
                            colors=('#888888', '#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677', '#CC6677'), test='kruskal-wallis_dunn', pvalue=True, return_sig=True, figsize=(2, 2))

###################### Plot Experimental motility feature plots for Exp  ############################
videos = np.unique(df['Video'])

if not os.path.isdir(path + 'experimental_inhibit_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_inhibit_motility_feature/')

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    dataset = {}
    for cell_type in ['WT', 'MT']:
        for group in ['Control','IgG', 'CD40L', 'mLT']:
            df_part = df_[(df_['Type'] == cell_type)&(df_['Inhibition'] == group)].reset_index(drop=True)
            avgs = []
            for video in videos:
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] == 0:
                    continue
                data = df_video[feature_name]
                avg = np.mean(data)
                avgs.append(avg)
            dataset[cell_type + ' ' + str(group)] = avgs
    #new_order = ['wt_B-cell', 'wt_B-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    #dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_bar_plot(dataset, path+'experimental_inhibit_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=('#888888', '#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677', '#CC6677'), test='kruskal-wallis_dunn', pvalue=True, return_sig=True, figsize=(2, 2))

######################## interaction features mt vs WT box plot  ###########################
if not os.path.isdir(path + 'Inhibit int feature violin plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Inhibit int feature violin plot/')

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Inhibition', 'traj_Label'])

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})


# k = df.iloc[:,324:417].isnull().any()
# null_features = k.index[k==True]
# feature_list = df.columns[324:417].drop(null_features).drop(['PC1', 'PC2', 'tskmeans'])
for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['WT', 'MT']:
        for group in ['Control','IgG', 'CD40L', 'mLT']:
        #for group in ['IgG', 'CD40L', 'mLT']:
            data = df_[(df_['Inhibition'] == group) & (df_[condition_name] == cell_type)][feature_name]
            data = pd.to_numeric(data, errors='coerce').to_numpy(dtype=float)
            dataset[cell_type + ' ' + str(group)] = data[np.isfinite(data)]

    if any(len(v) == 0 for v in dataset.values()):
        print(feature_name)
        continue

    # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
    #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ', }
    # dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}


    # draw_custom_violin_plot(dataset, path + 'Inhibit int feature violin plot/', file_name=feature_name,
    #                         colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'),
    #                         test='mann-whitney', pvalue=True, figsize=(2, 2))

    draw_custom_bar_plot(dataset, path + 'Inhibit int feature violin plot/', file_name=feature_name,
                         strip_plot=False, colors=('#888888', '#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677','#CC6677'),
                         test='kruskal-wallis_dunn', pvalue=True, return_sig=True, figsize=(2, 2))

###################### Plot Experimental int feature plots for Exp  ############################
videos = np.unique(df['Video'])

if not os.path.isdir(path + 'experimental_inhibit_int_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_inhibit_int_feature/')

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Inhibition', 'traj_Label'])

for feature_name in feature_list:
    dataset = {}
    for cell_type in ['WT', 'MT']:
        for group in ['Control','IgG', 'CD40L', 'mLT']:
            df_part = df_[(df_['Type'] == cell_type)&(df_['Inhibition'] == group)].reset_index(drop=True)
            avgs = []
            for video in videos:
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] == 0:
                    continue
                data = pd.to_numeric(df_video[feature_name], errors='coerce').to_numpy(dtype=float)
                if np.isfinite(data).any():
                    avgs.append(np.nanmean(data[np.isfinite(data)]))
            dataset[cell_type + ' ' + str(group)] = np.array(avgs)
    #new_order = ['wt_B-cell', 'wt_B-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    #dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    if any(len(v) == 0 for v in dataset.values()):
        print(feature_name)
        continue

    draw_custom_bar_plot(dataset, path+'experimental_inhibit_int_feature/', file_name=feature_name,
                         strip_plot=True, colors=('#888888', '#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677', '#CC6677'),
                         test='kruskal-wallis_dunn', pvalue=True, return_sig=True, figsize=(2, 2))


#################################### Volcano plot of all motility features ####################################
# feature_list = df_inhibit.columns[128:284].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
# cell_type = 'T-cell'
# df_part = df[df['Inhibition']!='Control'].reset_index(drop=True)
# df_part_ = df_part[df_part['Type']==cell_type]
#
#
# df_p = pd.DataFrame()
# for feature_name in feature_list:
#     dataset = {}
#     for condition in np.unique(df_part_['Inhibition']):
#         data = df_part_[df_part_['Inhibition'] == condition][feature_name]
#         dataset[condition] = np.array(data)
#
#     pvalue = get_pvalue(dataset, test='mann-whitney')
#     logp = -np.log10(pvalue)
#
#     avgZ = get_avgZ(dataset, ref_name='ControlAb', data_name='CD40LAb')
#
#     row = pd.DataFrame()
#     row['Feature'] = [feature_name]
#     row['Pvalue'] = [pvalue]
#     row['-Logp'] = [logp]
#     row['AvgZ'] = [avgZ]
#     df_p = pd.concat([df_p, row], axis=0)
#
# df_p['Adj_p'] = adjusted_pvalues(df_p['Pvalue'], correction_type='Benjamini-Hochberg')
# df_p['Adj_Logp'] = -np.log(df_p['Adj_p'])
# df_p = df_p[~df_p.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf
#
#
# draw_volcano_plot(df_p, path, file_name='motility volcano plot_%s'%cell_type, z_thresh=0.5, p_thresh=2, z_name='AvgZ', p_name='Adj_Logp',
#                   feature_name='Feature', figsize=(6,6))

