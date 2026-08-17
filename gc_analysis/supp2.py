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
"""Generates Data for Supplement"""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
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

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\supplements\\'

# #################################### APRW space ####################################
# path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/'
# df = pd.read_parquet(path+'all_features_20.parquet')
#
# aprw_data = df.iloc[:,128:136]
#
# scaler = StandardScaler()  # if not normalize, UMAP space is completely different
# aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
#
#
# m = Morphodynamics(df, 'umap')
# umap = m.get_umap(aprw_data_scaled, 20, 0.5)
#
#
# df = df.drop(['PC1', 'PC2'], axis=1)
# df = pd.concat([df, umap], axis=1)
#
#
# xmin = math.floor(df['PC1'].min()) - 1
# xmax = math.ceil(df['PC1'].max()) + 1
# ymin = math.floor(df['PC2'].min()) - 1
# ymax = math.ceil(df['PC2'].max()) + 1
#
# path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Figure1. Basic gc_analysis - supp/'
# draw_umap_space(df, path, file_name='APRW space_tskmeans', condition_name='tskmeans', label_name='pseudo_Label', x_name='PC1', y_name='PC2', dot_size=12,
#                 xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
# draw_umap_space(df, path, file_name='APRW space_Type', condition_name='Type', label_name='pseudo_Label', x_name='PC1', y_name='PC2', dot_size=12,
#                 xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
#
# feature_list = df.columns[128:136]
# draw_space_feature_magnitude(df, path, feature_list, dot_size=12, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, vmax=None)


################# sanity check ####################
draw_jointplot(xs='PC1', y='PC2', df=df, path=path, file_name='jointplot_exp_group', hue="Exp_group",
               colors=plt.get_cmap('Set1'), hue_order=['B', 'C'],
               legend=True, fill=False, thresh=0.15, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')


df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_['type_exp_group'] = df_['Type'].astype(str) + ' ' + df_['Exp_group'].astype(str)
draw_cluster_distribution_heatmap(df_, path, file_name='kmeans__exp_group', condition_name='Exp_group', vmax=20,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(8,2))
draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_type_type_exp_group', condition_name='type_exp_group', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,2))

df_['day_exp_group'] = df_['Day'].astype(str) + ' ' + df_['Exp_group'].astype(str)
df_ = df_[(df_['day_exp_group']!='D9 B')&(df_['day_exp_group']!='D9 C')]

draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_day_exp_group', condition_name='day_exp_group', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,2))

p_dict = permutation_test(df_, group_name='day_exp_group', class_name='kmeans', iteration=50000)

draw_jointplot(xs='PC1', y='PC2', df=df_, path=path, file_name='jointplot_day_exp_group', hue="day_exp_group",
               colors=plt.get_cmap('Set2'), hue_order=['D10 B', 'D10 C', 'D11 B', 'D11 C'],
               legend=True, fill=False, thresh=0.15, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_type_day_exp_group', condition_name='day_exp_group', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,2))


############# Plot type+day kmeans cross correlation for each cell type ###############
df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_['type_day'] = df_['Type'].astype(str) + ' ' + df_['Day'].astype(str)
df_['type_exp_group'] = df_['Type'].astype(str) + ' ' + df_['Exp_group'].astype(str)
df_['day_exp_group'] = df_['Day'].astype(str) + ' ' + df_['Exp_group'].astype(str)
df_['exp_exp_group'] = df_['Exp'].astype(str) + ' ' + df_['Exp_group'].astype(str)

#df_ = df_[(df_['day_exp_group']!='D9 B')&(df_['day_exp_group']!='D9 C')]
condition_name = 'Day'
cluster_type = 'kmeans'


font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

#file_name = '%s no D9 cross-correlation'%condition_name
file_name = '%s cross-correlation'%condition_name
df_corr_data = pd.DataFrame()
group_clones=[]
for group in np.unique(df_[condition_name]):
#for group in ['DZ', 'sLZ', 'dLZ']:
    corrcoef = []
    aaa = df_[df_[condition_name] == group].reset_index(drop=True)

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
        group_clone.rename(columns={column:column}, inplace=True)
    group_clones.append(group_clone)
    df_corr_data = pd.concat([df_corr_data, group_clone], axis=1)

df_corr = df_corr_data.corr(method='spearman')

df_corr.columns.name = None # Remove name of column = 'Type'
df_corr.index.name = None # Remove name of index = 'Type'


mask = np.triu(df_corr) # Mask for only lower triangle

mask = mask[1:, :-1]  # mask that doesn't contain very top left and very bottom right block
corr = df_corr.iloc[1:,:-1].copy()  # df that doesn't contain very top left and very bottom right block

fig, ax = plt.subplots(figsize=(1.5, 1.5))
kws = dict(cbar_kws=dict(ticks=[0, 0.5, 1], orientation='horizontal'), vmin=0)

ax = sns.heatmap(corr, annot=True, mask=mask, cmap='OrRd', alpha=0.9, linewidths=1, linecolor='white', fmt=".2f", square= True,
                 annot_kws={'size': 4, 'weight':'normal'},vmin=0, vmax=1,
                 cbar_kws= {"shrink":0.7, 'label':'Correlation', 'ticks':[0, 0.5, 1]})

cax = ax.figure.axes[-1]  # colorbar
cax.tick_params(labelsize=4)  # fontsize of tick label
cax.yaxis.label.set_size(6)  # fontsize of color bar y label

# yticks = [i for i in corr.index]
# xticks = [i for i in corr.columns]
# plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=4, weight='normal')
# plt.xticks(plt.xticks()[0], labels=xticks, fontsize=4, rotation=35, rotation_mode='anchor', ha='right', weight='normal')

plt.xticks(fontsize=4, rotation=35, rotation_mode='anchor', ha='right', color='0.2', weight='normal')
plt.yticks(fontsize=4, rotation=0,  color='0.2', weight='normal')

plt.savefig(path+'%s.png'%file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg'%file_name, bbox_inches='tight')

plt.close()
plt.clf()

###################### Plot Experimental motility feature plots for Zones  ############################
videos = np.unique(df['Video'])

if not os.path.isdir(path + 'experimental_exp_group_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_exp_group_motility_feature/')

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    dataset = {}
    for group in ['B','C']:
        df_part = df[(df['Exp_group'] == group)].reset_index(drop=True)
        avgs = []
        for video in videos:
            df_video = df_part[df_part['Video'] == video]
            if df_video.shape[0] == 0:
                continue
            data = df_video[feature_name]
            avg = np.mean(data)
            avgs.append(avg)
        dataset[str(group)] = avgs
    #new_order = ['wt_B-cell', 'wt_B-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
                   'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ', }
    dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_bar_plot(dict_datasets, path+'experimental_exp_group_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))





###################### Plot Zone motility feature violin plot for MT and WT  ############################

if not os.path.isdir(path + 'Exp_group motility box plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Exp_group motility box plot/')

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for group in ['B','C']:
        #for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
            data = df[(df[condition_name] == cell_type)&(df['Exp_group'] == group)][feature_name]

            dataset[cell_type+'_'+str(group)] = np.array(data)

    rename_keys = {'wt_B-cell_B': 'WT B', 'wt_B-cell_C': 'WT C',
                   'mt_B-cell_B': 'MT B', 'mt_B-cell_C': 'MT C'}
    # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_DZ-sLZ': 'WT DZ-sLZ', 'wt_B-cell_sLZ': 'WT sLZ',
    #                'wt_B-cell_sLZ-dLZ': 'WT sLZ-dLZ', 'wt_B-cell_dLZ': 'WT dLZ',
    #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_DZ-sLZ': 'MT DZ-sLZ', 'mt_B-cell_sLZ': 'MT sLZ',
    #                 'mt_B-cell_sLZ-dLZ': 'MT sLZ-dLZ', 'mt_B-cell_dLZ': 'MT dLZ',}

    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_violin_plot(dataset_renamed, path + 'Exp_group motility box plot/', file_name=feature_name,
                            colors=('#888888', '#888888', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))
    # draw_custom_violin_plot(dataset_renamed, path + 'Zone motility box plot/', file_name=feature_name,
    #                         colors=('#888888', '#888888', '#888888', '#888888', '#888888',
    #                                 '#CC6677', '#CC6677', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))


###################### Plot Experimental motility feature plots for Zones  ############################
############# Plot type+day kmeans cross correlation for each cell type ###############
df_ = df.copy()
df_['day_exp_group'] = df_['Day'].astype(str) + ' ' + df_['Exp_group'].astype(str)
df_ = df_[(df_['day_exp_group']!='D9 B')&(df_['day_exp_group']!='D9 C')]

videos = np.unique(df['Video'])

if not os.path.isdir(path + 'experimental_exp_group_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_exp_group_motility_feature/')

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    dataset = {}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for group in ['B','C']:
            df_part = df_[(df_['Type'] == cell_type)&(df['Exp_group'] == group)].reset_index(drop=True)
            avgs = []
            for video in videos:
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] == 0:
                    continue
                data = df_video[feature_name]
                avg = np.mean(data)
                avgs.append(avg)
            dataset[cell_type + '_' + str(group)] = avgs
    #new_order = ['wt_B-cell', 'wt_B-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    rename_keys = {'wt_B-cell_B': 'WT B', 'wt_B-cell_C': 'WT C',
                   'mt_B-cell_B': 'MT B', 'mt_B-cell_C': 'MT C'}
    dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_bar_plot(dict_datasets, path+'experimental_exp_group_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=('#888888', '#888888', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))


######################## Zone interaction features mt vs WT box plot  ###########################
if not os.path.isdir(path + 'Exp_group int feature violin plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Exp_group int feature violin plot/')
if not os.path.isdir(path + 'Exp_group int feature bar plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Exp_group int feature bar plot/')

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2'])


for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for group in ['B','C']:
        #for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
            data = df[(df[condition_name] == cell_type)&(df['Exp_group'] == group)][feature_name]

            dataset[cell_type+'_'+str(group)] = np.array(data)

    values = flatten_nested_dict(dataset)
    if np.isnan(values).any() == True:  # Check at least one nan
        continue
    elif np.isfinite(values).all() == False:  # Check everything is not inf
        continue

    rename_keys = {'wt_B-cell_B': 'WT B', 'wt_B-cell_C': 'WT C',
                   'mt_B-cell_B': 'MT B', 'mt_B-cell_C': 'MT C'}
    # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_DZ-sLZ': 'WT DZ-sLZ', 'wt_B-cell_sLZ': 'WT sLZ',
    #                'wt_B-cell_sLZ-dLZ': 'WT sLZ-dLZ', 'wt_B-cell_dLZ': 'WT dLZ',
    #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_DZ-sLZ': 'MT DZ-sLZ', 'mt_B-cell_sLZ': 'MT sLZ',
    #                 'mt_B-cell_sLZ-dLZ': 'MT sLZ-dLZ', 'mt_B-cell_dLZ': 'MT dLZ',}

    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_violin_plot(dataset_renamed, path + 'Exp_group int feature violin plot/', file_name=feature_name,
                            colors=('#888888', '#888888', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))

    draw_custom_bar_plot(dataset_renamed, path + 'Exp_group int feature bar plot/', file_name=feature_name,
                         strip_plot=False, colors=('#888888', '#888888', '#CC6677', '#CC6677'),
                         test='mann-whitney', pvalue=True, figsize=(2, 2))

    # draw_custom_violin_plot(dataset_renamed, path + 'Zone motility box plot/', file_name=feature_name,
    #                         colors=('#888888', '#888888', '#888888', '#888888', '#888888',
    #                                 '#CC6677', '#CC6677', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))

######################## DZ/sLZ/dLZ interaction features mt vs WT box plot  ###########################

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'Zone'])

for zone in ['DZ','sLZ', 'dLZ']:
    df_part = df[(df['Zone'] == zone)].reset_index(drop=True)
    if not os.path.isdir(path + '%s int feature violin plot/'%zone):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s int feature violin plot/'%zone)
    for feature_name in feature_list:
        condition_name = 'Type'
        dataset={}
        for cell_type in ['wt_B-cell', 'mt_B-cell']:
            data = df_part[df_part[condition_name] == cell_type][feature_name]
            dataset[cell_type] = np.array(data)

        values = flatten_nested_dict(dataset)
        if np.isnan(values).any() == True:  # Check at least one nan
            continue
        elif np.isfinite(values).all() == False:  # Check everything is not inf
            continue

        rename_keys = {'wt_B-cell': 'WTssss', 'mt_B-cell': 'MT' }
        dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

        # draw_custom_bar_plot(dataset_renamed, path + '%s int feature violin plot/'%zone, file_name=feature_name,
        #                      strip_plot=False, colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))
        draw_custom_violin_plot(dataset_renamed, path + '%s int feature violin plot/'%zone, file_name=feature_name,
                                colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))

###################### Plot Experimental motility feature plots for Zones  ############################
df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone'])

videos = np.unique(df['Video'])

if not os.path.isdir(path + 'experimental_zone_int_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_zone_int_feature/')

for feature_name in feature_list:
    dataset = {}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for group in ['DZ','sLZ', 'dLZ']:
            df_part = df[(df['Type'] == cell_type)&(df['Zone'] == group)].reset_index(drop=True)
            avgs = []
            for video in videos:
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] == 0:
                    continue
                data = df_video[feature_name]
                avg = np.mean(data)
                avgs.append(avg)
            dataset[cell_type + '_' + str(group)] = avgs
    #new_order = ['wt_B-cell', 'wt_B-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
                   'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ', }
    dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_bar_plot(dict_datasets, path+'experimental_zone_int_feature/', file_name=feature_name,
                         strip_plot=True, colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))

