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
"""Generates Data for Figure 1."""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast

#################################### Correct csv file patient info ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')
patient_info = pd.read_excel(path+'Aging donors Jude NAM study.xlsx')

test = df[(df['Patient'] == 'F612C')]
df.loc[(df['Patient'] == 'F1061'), 'Type'] = 'Prefrail'
df.loc[(df['Patient'] == 'F1089'), 'Age'] = 55
df = df[df['Patient']!='F1122'].reset_index(drop=True)
df.loc[(df['Patient'] == 'F612C'), 'Type'] = 'Prefrail'

df_duration.loc[(df_duration['Patient'] == 'F1061'), 'Type'] = 'Prefrail'
df_duration.loc[(df_duration['Patient'] == 'F1089'), 'Age'] = 55
df_duration = df_duration[df_duration['Patient']!='F1122'].reset_index(drop=True)
df_duration.loc[(df_duration['Patient'] == 'F612C'), 'Type'] = 'Prefrail'

for patient in np.unique(patient_info['Patient']):
    df.loc[(df['Patient'] == patient), 'Weakness'] = patient_info[patient_info['Patient']==patient]['Weakness'].values[0]
    df.loc[(df['Patient'] == patient), 'Weight_loss'] = patient_info[patient_info['Patient'] == patient]['Weight loss'].values[0]
    df.loc[(df['Patient'] == patient), 'Exhaustion'] = patient_info[patient_info['Patient'] == patient]['Exhaustion'].values[0]
    df.loc[(df['Patient'] == patient), 'Activity'] = patient_info[patient_info['Patient'] == patient]['Activity'].values[0]
    df.loc[(df['Patient'] == patient), 'Gait'] = patient_info[patient_info['Patient'] == patient]['Gait (avg)'].values[0]
    df.loc[(df['Patient'] == patient), 'Grip'] = patient_info[patient_info['Patient'] == patient]['Grip strength (max)'].values[0]
    df.loc[(df['Patient'] == patient), 'Frailty_score'] = patient_info[patient_info['Patient'] == patient]['Frailty score (0-5)'].values[0]


for patient in np.unique(patient_info['Patient']):
    df_duration.loc[(df_duration['Patient'] == patient), 'Weakness'] = patient_info[patient_info['Patient']==patient]['Weakness'].values[0]
    df_duration.loc[(df_duration['Patient'] == patient), 'Weight_loss'] = patient_info[patient_info['Patient'] == patient]['Weight loss'].values[0]
    df_duration.loc[(df_duration['Patient'] == patient), 'Exhaustion'] = patient_info[patient_info['Patient'] == patient]['Exhaustion'].values[0]
    df_duration.loc[(df_duration['Patient'] == patient), 'Activity'] = patient_info[patient_info['Patient'] == patient]['Activity'].values[0]
    df_duration.loc[(df_duration['Patient'] == patient), 'Gait'] = patient_info[patient_info['Patient'] == patient]['Gait (avg)'].values[0]
    df_duration.loc[(df_duration['Patient'] == patient), 'Grip'] = patient_info[patient_info['Patient'] == patient]['Grip strength (max)'].values[0]
    df_duration.loc[(df_duration['Patient'] == patient), 'Frailty_score'] = patient_info[patient_info['Patient'] == patient]['Frailty score (0-5)'].values[0]


df.to_csv(path + 'cleaned_all_features_30.csv', index=False)
df.to_parquet(path + 'cleaned_all_features_30.parquet')

df_duration.to_csv(path + 'cleaned_traj_duration_30.csv', index=False)
df_duration.to_parquet(path + 'cleaned_traj_duration_30.parquet')


#################################### motility space ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df_ctrl = df[df['Condition']=='Control'].reset_index(drop=True)
df_ctrl = df_ctrl[df_ctrl['Type']!='Prefrail'].reset_index(drop=True)

df_ctrl = df_ctrl[df_ctrl['Age']!=55].reset_index(drop=True)

# df.loc[(df['Type'] != 'Young'), 'Type2'] = 'Old'
# df.loc[(df['Type'] == 'Young'), 'Type2'] = 'Young'
#
# df_patient = pd.DataFrame()
# for typ in ['Young', 'Old']:
#     df_part = df[df['Type2']==typ].reset_index(drop=True)
#     typ_temp = []
#     frailty_temp = []
#     age_temp = []
#     sex_temp = []
#     patient_temp = []
#     df_temp = pd.DataFrame()
#     for patient in np.unique(df_part['Patient']):
#         each_patient = df_part[df_part['Patient']==patient].reset_index(drop=True)
#         typ = each_patient['Type2'][0]
#         frailty = each_patient['Type'][0]
#         age = each_patient['Age'][0]
#         sex = each_patient['Sex'][0]
#         patient = each_patient['Patient'][0]
#
#         typ_temp.append(typ)
#         frailty_temp.append(frailty)
#         age_temp.append(age)
#         sex_temp.append(sex)
#         patient_temp.append(patient)
#
#     df_temp['Type2'] = typ_temp
#     df_temp['Frailty'] = frailty_temp
#     df_temp['Age'] = age_temp
#     df_temp['Sex'] = sex_temp
#     df_temp['Patient'] = patient_temp
#     df_temp = df_temp.sort_values(by='Age')
#     df_patient = pd.concat([df_patient, df_temp.reset_index(drop=True)], axis=0)
#
# df_patient = df_patient.reset_index(drop=True)

for i in np.unique(df_ctrl['Type']):
    print(i, df_ctrl[df_ctrl['Type']==i].shape[0])


path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure1. Control\\'

#path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis\Figure1-2. Only moving\\'
color_list = ('#888888', '#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100')

draw_umap_space(df_ctrl, path, file_name='motility space_type', condition_name='Type', label_name='pseudo_particle',
                colors = ('#fdc086', '#beaed4', '#7fc97f'), dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_ctrl, path, file_name='motility space_age', condition_name='Age', label_name='pseudo_particle',
                colors =cmc.turku, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_ctrl, path, file_name='motility space_patient', condition_name='Patient', label_name='pseudo_particle',
                colors =cmc.turku, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df, path, file_name='motility space_kmeans', condition_name='kmeans', label_name='pseudo_particle',
                colors=cmc.batlow, dot_size=0.07, x_name='PC1', y_name='PC2')

draw_umap_space(df_ctrl[~df_ctrl['Weakness'].isnull()], path, file_name='motility space_weakness', condition_name='Weakness', label_name='pseudo_particle',
                colors=cmc.roma, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_ctrl[~df_ctrl['Weight_loss'].isnull()], path, file_name='motility space_weightloss', condition_name='Weight_loss', label_name='pseudo_particle',
                colors=cmc.roma, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_ctrl[~df_ctrl['Exhaustion'].isnull()], path, file_name='motility space_exhaustion', condition_name='Exhaustion', label_name='pseudo_particle',
                colors=cmc.roma, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_ctrl[~df_ctrl['Activity'].isnull()], path, file_name='motility space_activity', condition_name='Activity', label_name='pseudo_particle',
                colors=cmc.roma, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_ctrl[~df_ctrl['Gait'].isnull()], path, file_name='motility space_gait', condition_name='Gait', label_name='pseudo_particle',
                colors=cmc.batlow, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_ctrl[~df_ctrl['Grip'].isnull()], path, file_name='motility space_grip', condition_name='Grip', label_name='pseudo_particle',
                colors=cmc.batlow, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_ctrl[~df_ctrl['Frailty_score'].isnull()], path, file_name='motility space_frailty_score', condition_name='Frailty_score', label_name='pseudo_particle',
                colors=cmc.batlow, dot_size=0.07, x_name='PC1', y_name='PC2')
# draw_umap_space(df_ctrl, path, file_name='motility space_tskmeans', condition_name='tskmeans', label_name='pseudo_particle',
#                 colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')

xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

draw_jointplot(xs='PC1', y='PC2', df=df_ctrl, path=path, file_name='jointplot_type', hue="Type", colors=('#fdc086', '#beaed4', '#7fc97f'),
               legend=False, fill=True, thresh=0.3, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2',
               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
draw_jointplot(xs='PC1', y='PC2', df=df_ctrl, path=path, file_name='jointplot_kmeans', hue="kmeans", colors=cmc.batlow,
               legend=False, fill=True, thresh=0.3, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2',
               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
draw_jointplot(xs='PC1', y='PC2', df=df_ctrl, path=path, file_name='jointplot_age', hue="Age", colors=cmc.turku,
               legend=False, fill=True, thresh=0.6, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2',
               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

draw_contour(df_ctrl, path, file_name='space_contour_type', condition_name='Type', colors=('#fdc086', '#beaed4', '#7fc97f'), x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df_ctrl, path, file_name='space_contour_age', condition_name='Age', colors=color_list*3, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df_ctrl, path, file_name='space_contour_patient', condition_name='Patient', colors=color_list*4, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)


df_ctrl.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list( df_ctrl.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )
feature_list.append('Time_span')

draw_space_feature_magnitude(df_ctrl, path, feature_list, dot_size=0.07, x_name='PC1', y_name='PC2', vmax=None)

draw_space_feature_magnitude(df_ctrl, path, feature_list=['avg_speed'], dot_size=0.07, x_name='PC1', y_name='PC2', vmax=3)
draw_space_feature_magnitude(df_ctrl, path, feature_list=['displ_cov'], dot_size=0.07, x_name='PC1', y_name='PC2', vmax=1.4)
draw_space_feature_magnitude(df_ctrl, path, feature_list=['angle_cov'], dot_size=0.07, x_name='PC1', y_name='PC2', vmax=1)
draw_space_feature_magnitude(df_ctrl, path, feature_list=['morpho_avg_speed'], dot_size=0.07, x_name='PC1', y_name='PC2', vmax=2)

#################################### Trajectories for each cluster ####################################
draw_2D_trajectories_one_figure(df_duration, df, path, duration=30, n_examples=30, label_name='kmeans', feature_name=['x', 'y'], lim=30) # lim=150
for i in np.unique(df['kmeans']):
    print('cluster: ', i, 'Cell number: ', df[df['kmeans']==i].shape[0])

#################################### Heatmap of cluster enrichment  ####################################
#df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB', 'T-cell': 'Tfh'}})
draw_cluster_distribution_heatmap(df_ctrl, path, file_name='kmeans_type_heatmap', condition_name='Type', cluster_type='kmeans',
                                  annot=True, col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(6,2))

draw_relative_cluster_distribution_heatmap(df_ctrl, path, file_name='relative_kmeans_type_heatmap', condition_name='Type', cluster_type='kmeans',
                                  annot=True, transpose=False, col_cluster=False, row_cluster=True, cmap=cmc.oslo_r, figsize=(6,2))


draw_heatmap_with_circles(df_ctrl, path, file_name='kmeans_type_circleheatmap', condition_name='Type', cluster_type='kmeans',
                          vmax=None, transpose=False, row_cluster=True, col_cluster=False, figsize=(4,4))


draw_cluster_distribution_heatmap(df_ctrl, path, file_name='all_patient_kmeans_heatmap', condition_name='Patient',
                                  cluster_type='kmeans', transpose=False, row_cluster=True, col_cluster=False, cmap=cmc.bilbao_r, figsize=(4,12))

draw_relative_cluster_distribution_heatmap(df_ctrl, path, file_name='relative_all_patient_kmeans_heatmap', condition_name='Patient',
                                           cluster_type='kmeans', transpose=False, row_cluster=False, col_cluster=False,
                                           cmap=cmc.oslo_r, figsize=(4,12))



draw_cluster_distribution_heatmap(df_ctrl[df_ctrl['Type']!='Frail'].reset_index(drop=True), path, file_name='age_kmeans_heatmap', condition_name='Age',
                                  cluster_type='kmeans', transpose=True, row_cluster=False, col_cluster=False, cmap=cmc.bilbao_r, figsize=(10,4))

draw_relative_cluster_distribution_heatmap(df_ctrl[df_ctrl['Type']!='Frail'].reset_index(drop=True), path, file_name='relative_age_kmeans_heatmap',
                                           condition_name='Age', cluster_type='kmeans', transpose=True, row_cluster=False, col_cluster=False,
                                           cmap=cmc.oslo_r, figsize=(10,4))

#################################### Total entropy ####################################
entropy, max_entropy = calculate_entropy(df_ctrl, df_ctrl, condition_name='Type', cluster_type='kmeans')

dict_datasets={}
for key in entropy:
    dict_datasets[key] = np.array([entropy[key]])

new_order = ['Young', 'Old', 'Frail']
dict_datasets = change_dict_order(dict_datasets, new_order)

file_name='total entropy'
test='mann-whitney'

colors = ('#7fc97f',  '#beaed4', '#fdc086')
#colors = ('#888888', '#6699CC', '#CC6677')


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
#ax = sns.stripplot(data=sorted_vals, marker='s', s=1.5, **plot_params)
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
plt.xticks(np.array([0,1,2]), sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
           weight='bold')

plt.yticks(fontsize=8, color='0.2', weight='bold')
# plt.ylabel('%s' % feature_name, fontsize=4)
# category labels
plt.grid(False)
plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()


#################################### Patient-wise entropy ####################################

entropies = {'Young':[], 'Old':[], 'Frail':[]}
for video in np.unique(df_ctrl['Patient']):
    df_part = df_ctrl[df_ctrl['Patient']==video]
    entropy, max_entropy = calculate_entropy(df_part, df_ctrl, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        entropies[type].append(entropy[type])

# new_order = ['wt_B-cell', 'T-cell']
# ordered_entropies = change_dict_order(entropies, new_order)
#replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
#entropies_ = {replace_keys.get(k, k):v  for (k,v) in ordered_entropies.items() }

dict_datasets = entropies
file_name='entropy'
test='mann-whitney'

colors = ('#7fc97f', '#beaed4', '#fdc086')
#colors = ('#888888', '#6699CC', '#CC6677')


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


pairs, p_values, cohen_d = get_various_statistics(dict_datasets, test='kruskal-wallis_dunn')

plt.title('%s: %s, %s' % (pairs, p_values, cohen_d), fontsize=4)
plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()

#################################### Cross correlation of young vs old vs frail ###################################

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
corrcoef = []

group_clone = pd.DataFrame(df_ctrl.groupby([condition_name, cluster_type]).size())
group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
group_clone = group_clone.unstack(level=0)
group_clone[np.isnan(group_clone)] = 0
group_clone_T = group_clone.T
for cluster in list(pd.unique(df[cluster_type])):
    if cluster in group_clone_T.columns:
        continue
    else:
        group_clone_T.insert(loc=int(cluster), column=cluster, value=[0]*np.unique(df[condition_name]).size)
group_clone = group_clone_T.T

df_corr = group_clone.corr()

df_corr.columns.name = None # Remove name of column = 'Type'
df_corr.index.name = None # Remove name of index = 'Type'


# rename_keys = {'wt_B-cell_far': 'wt GCB far', 'wt_B-cell_close': 'wt GCB close',
#                'mt_B-cell_far': 'mt GCB far', 'mt_B-cell_close': 'mt GCB close',}
# df_corr.rename(columns=rename_keys, inplace=True)
# df_corr.rename(index=rename_keys, inplace=True)

mask = np.triu(df_corr) # Mask for only lower triangle

mask = mask[1:, :-1]  # mask that doesn't contain very top left and very bottom right block
corr = df_corr.iloc[1:,:-1].copy()  # df that doesn't contain very top left and very bottom right block

fig, ax = plt.subplots(figsize=(2, 2))

ax = sns.heatmap(corr, annot=True, mask=mask, cmap='OrRd', alpha=0.9, linewidths=1, linecolor='white', fmt=".2f", square= True,
                 annot_kws={'size': 4, 'weight':'bold'},
                 cbar_kws= {"shrink":0.7, 'label':'Correlation'})

cax = ax.figure.axes[-1]  # colorbar
cax.tick_params(labelsize=4)  # fontsize of tick label
cax.yaxis.label.set_size(6)  # fontsize of color bar y label

yticks = [i for i in corr.index]
xticks = [i for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=4, weight='bold')
plt.xticks(plt.xticks()[0], labels=xticks, fontsize=4, rotation=35, rotation_mode='anchor', ha='right', weight='bold')

plt.savefig(path+'Type correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/Type correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

#################################### Box plot comparing all motility features by cell types ####################################
df_ctrl.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list( df_ctrl.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )
feature_list.append('Time_span')
condition_name = 'Type'
#replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'wt GCB'}
if not os.path.isdir(path + 'motility_feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'motility_feature_violin_plot_type/')

if not os.path.isdir(path + 'motility_feature_box_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'motility_feature_box_plot_type/')


for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df_ctrl[condition_name]):
        data = df_ctrl[df_ctrl[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    # new_order = ['wt_B-cell', 'T-cell']
    # ordered_dataset = change_dict_order(dataset, new_order)
    # dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    new_order = ['Young', 'Old', 'Frail']
    dataset = change_dict_order(dataset, new_order)

    draw_custom_violin_plot(dataset, path+'motility_feature_violin_plot_type/', file_name=feature_name, colors = ('#7fc97f', '#beaed4', '#fdc086'),
                            test='kruskal-wallis_dunn', pvalue=True, figsize=(1,2))

    draw_custom_box_plot(dataset, path+ 'motility_feature_box_plot_type/', file_name=feature_name, colors = ('#7fc97f', '#beaed4', '#fdc086'),
    strip_plot=False, test='kruskal-wallis_dunn', pvalue=True, figsize=(1,2))

#################################### Box plot comparing experimental(video by video) motility features of age groups ###################################

condition_name = 'Type'
batch_name = 'Patient'
batches = np.unique(df_ctrl[batch_name])

colors = ('#7fc97f', '#beaed4', '#fdc086')

if not os.path.isdir(path + 'experimental_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_motility_feature/')

for feature_name in feature_list:
    dataset = {}
    for type in np.unique(df_ctrl[condition_name]):
        df_part = df_ctrl[df_ctrl[condition_name] == type]
        avgs = []
        for batch in batches:
            #if 'A' in video and cell_type == 'mt_B-cell':
            df_patient = df_part[df_part[batch_name] == batch]
            if df_patient.shape[0] == 0:
                continue
            data = df_patient[feature_name]
            avg = np.mean(data)
            #print(type, batch, avg)
            avgs.append(avg)
        dataset[type] = avgs
    new_order = ['Young', 'Old', 'Frail']
    dataset = change_dict_order(dataset, new_order)

    draw_custom_bar_plot(dataset, path+'experimental_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=colors, test='kruskal-wallis_dunn', pvalue=True, figsize=(1, 2))

#################################### Box plot comparing experimental(video by video) motility features of kmeans ###################################

condition_name = 'kmeans'
batch_name = 'Patient'
batches = np.unique(df[batch_name])

n_colors = np.unique(df[condition_name]).shape[0]
cm = cmc.batlow
cmap = [cm(1. * i / n_colors) for i in range(n_colors)]

if not os.path.isdir(path + 'kmeans_experimental_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'kmeans_experimental_motility_feature/')

for feature_name in feature_list:
    dataset = {}
    for type in np.unique(df[condition_name]):
        df_part = df[df[condition_name] == type]
        avgs = []
        for batch in batches:
            #if 'A' in video and cell_type == 'mt_B-cell':
            df_patient = df_part[df_part[batch_name] == batch]
            if df_patient.shape[0] == 0:
                continue
            data = df_patient[feature_name]
            avg = np.mean(data)
            avgs.append(avg)
        dataset[type] = avgs
    #new_order = ['Young', 'Old', 'Frail']
    #dataset = change_dict_order(dataset, new_order)

    draw_custom_bar_plot(dataset, path+'kmeans_experimental_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=cmap, test='kruskal-wallis_dunn', pvalue=False, figsize=(4, 4))

#################################### Z scores of all motility features wrt kmeans ####################################

df_ctrl.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df_ctrl.columns[8:103].drop(['speed_distribution_x', 'speed_distribution_y',
                                            'inst_speed_cosine_similarity_entropies', 'inst_angle_cosine_similarity_entropies'])

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
plt.savefig(path + 'svg/' + 'kmeans Z score features_heatmap.svg', bbox_inches='tight',)
plt.clf()
plt.close()

#################################### Volcano plot of all motility features ####################################

condition_name = 'Type'
ref = 'Old'
compare = 'Frail'

feature_list = df_ctrl.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y'])

df_part = df_ctrl[(df_ctrl[condition_name]==ref)|(df_ctrl[condition_name]==compare)].reset_index(drop=True)

df_p = pd.DataFrame()
for feature_name in feature_list:
    dataset = {}
    for condition in np.unique(df_part[condition_name]):
        data = df_part[df_part[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)

    pvalue = get_pvalue(dataset, test='mann-whitney')
    logp = -np.log10(pvalue)

    avgZ = get_avgZ(dataset, ref_name=ref, data_name=compare)

    row = pd.DataFrame()
    row['Feature'] = [feature_name]
    row['Pvalue'] = [pvalue]
    row['-Logp'] = [logp]
    row['AvgZ'] = [avgZ]
    df_p = pd.concat([df_p, row], axis=0)

df_p['Adj_p'] = adjusted_pvalues(df_p['Pvalue'], correction_type='Benjamini-Hochberg')
df_p['Adj_Logp'] = -np.log(df_p['Adj_p'])
df_p = df_p[~df_p.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf


draw_volcano_plot(df_p, path, file_name='volcano plot of %s vs %s'%(ref, compare), z_thresh=0.4, p_thresh=80, z_name='AvgZ', p_name='Adj_Logp',
                  feature_name='Feature', figsize=(6,6))


