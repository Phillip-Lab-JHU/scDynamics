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
"""Generates Data for Supp 1."""

from utils.draw_utils import *
import pandas as pd

############################## Figure 1. Optimization #######################################

#################################### Draw 3D trajectories ####################################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'
df = pd.read_parquet(path+'no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path + 'no_inhibit_traj_duration_20.parquet')
traj_list, _, trajectories = to_timeseries_fast(df_duration, duration=20, feature_name=['Position X', 'Position Y', 'Position Z'])

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/supplements/supp1/'
draw_3D_trajectory_labels(path, trajectories, folder_name='trajectory by tskmeans', label=df['tskmeans'], idx_range=None)
################# Save data for number of time frames #################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/'
excel = pd.read_parquet(path + 'Intravital Data.parquet')

condition_name='Type'
label_name='Label'
label_data = excel.groupby([condition_name, label_name]).apply(lambda x: x.name)  # contain (cell type, TrackID) tuple

time_list = []
for traj_idx in tqdm(range(0, label_data.shape[0])):  # For each cell trajectory(time 1~t)
    traj_data_temp = excel.groupby([condition_name, label_name]).get_group(label_data.iloc[traj_idx]).copy()
    time_list.append(traj_data_temp.shape[0])

time_list = np.array(time_list)
#np.save(path+'time_list.npy', time_list)

################# Histogram for number of time frames #################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/'
time_list = np.load(path+'time_list.npy')

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Figure1. Optimization/'

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(2, 2))
ax = sns.distplot(time_list, color='black', bins=50, kde=True, hist_kws=dict(color='#888888', edgecolor="black", linewidth=1))
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=1)

ax.set_xlabel('Number of frames', fontsize=8, weight='normal')
ax.set_ylabel('Density of tracked cells', fontsize=8, weight='normal')
plt.xticks(fontsize=8, color='0.2', weight='normal')
plt.yticks(fontsize=8, color='0.2', weight='normal')

plt.xlim(-2, 60)

plt.savefig(path + 'time frame histogram.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/time frame histogram.svg', bbox_inches='tight')
plt.close()
plt.clf()

################# Bar plot of cell number GCB vs T cell #################

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/'
df = pd.read_parquet(path+'all_features_20.parquet')

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Figure1. Optimization/'
videos = np.unique(df['Video'])
df_A = df[(df['Video'] == videos[0])|(df['Video'] == videos[1])|(df['Video'] == videos[2])].reset_index(drop=True)

cellnumber_datasets={'T-cell':[], 'GCB':[]}
for video in np.unique(df_A['Video']):
    df_video = df_A[df_A['Video']==video]

    for cell_type in ['T-cell', 'wt_B-cell']:
        df_part = df_video[df_video['Type'] == cell_type]
        cell_number = df_part.shape[0]
        if cell_type=='wt_B-cell':
            cellnumber_datasets['GCB'].append(cell_number)
        elif cell_type=='T-cell':
            cellnumber_datasets['T-cell'].append(cell_number)

df_noA = df[(df['Video'] != videos[0])&(df['Video'] != videos[1])&(df['Video'] != videos[2])].reset_index(drop=True)
for video in np.unique(df_noA['Video']):
    df_video = df_noA[df_noA['Video']==video]

    cellnumber_temp={}
    for cell_type in ['T-cell', 'wt_B-cell', 'mt_B-cell',]:
        df_part = df_video[df_video['Type'] == cell_type]
        cell_number = df_part.shape[0]
        cellnumber_temp[cell_type] = cell_number

        if cell_type=='T-cell':
            cellnumber_datasets['T-cell'].append(cellnumber_temp['T-cell'])

    cellnumber_datasets['GCB'].append(cellnumber_temp['wt_B-cell'] + cellnumber_temp['mt_B-cell'])

new_order = ['GCB', 'T-cell']
cellnumber_datasets = change_dict_order(cellnumber_datasets, new_order)

replace_keys = {'T-cell':'T cell', 'GCB':'GC B cell'}
cellnumber_datasets = {replace_keys.get(k, k):v  for (k,v) in cellnumber_datasets.items() }

dict_datasets = cellnumber_datasets
file_name='cell number'
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
# plt.xticks(plt.xticks()[0], sorted_keys, fontsize=12, fontdict={'weight': 'normal'})
# ax.set_xticks(plt.xticks()[0], sorted_keys, fontsize=4)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=1, color='0.2')
ax.set_ylabel('Number of tracked cells', fontsize=8, weight='normal', color='0.2')
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
#plt.title('%s:%s' % (pairs, p_values))
plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()

############################### Permutation test for cell number ################################

videos = np.unique(df['Video'])
df_noA = df[(df['Video'] != videos[0])&(df['Video'] != videos[1])&(df['Video'] != videos[2])].reset_index(drop=True)

average_distances_to_FDC = []
average_distances_to_T = []
for i in range(df_noA.shape[0]):
    avg_distance_FDC = np.mean(df_noA[shortest_distance_features[0]][i])
    avg_distance_T = np.mean(df_noA[shortest_distance_features[3]][i])
    average_distances_to_FDC.append(avg_distance_FDC)
    average_distances_to_T.append(avg_distance_T)
df_noA['Average_distance_to_FDC'] = average_distances_to_FDC
df_noA['Average_distance_to_T'] = average_distances_to_T
df_wt = df_noA[df_noA['Type']=='wt_B-cell']
df_mt = df_noA[df_noA['Type']=='mt_B-cell']
df_T = df_noA[df_noA['Type']=='T-cell']

iteration = 1000
feature = 'Average_distance_to_FDC'
pvalues = {}

test_statistics = []
numbers = []
for video in videos:
    if '-A' in video:
        continue
    df_mt_each_video = df_mt[df_mt['Video'] == video]
    df_wt_each_video = df_wt[df_wt['Video'] == video]
    df_T_each_video = df_T[df_T['Video'] == video]

    mt_number = df_mt_each_video.shape[0]
    wt_number = df_wt_each_video.shape[0]
    T_number = df_T_each_video.shape[0]

    test_statistic = (np.mean(df_mt_each_video[feature]) - np.mean(df_wt_each_video[feature]))
    #test_statistic = (np.mean(df_T_each_video[feature]) - np.mean(df_wt_each_video[feature]))

    test_statistics.append(test_statistic)
    numbers.append(mt_number - wt_number)
    #print(video, test_statistic, T_number - wt_number)
    k=0
    for j in range(iteration):
        if mt_number > wt_number:
            # k += test_statistic < ( np.mean(df_mt.sample(n=wt_number)[feature]) - np.mean(df_wt_each_video[feature]) )
            k += np.mean(df_mt.sample(n=wt_number)[feature]) > np.mean(df_wt_each_video[feature])
            # k += test_statistics > (np.mean(df_T.sample(n=wt_number)[feature]) - np.mean(df_wt_each_video[feature]))

        elif mt_number < wt_number:
            # k += test_statistic < (np.mean(df_mt_each_video[feature]) - np.mean(df_wt.sample(n=mt_number)[feature]))
            k += np.mean(df_mt_each_video[feature]) > np.mean(df_wt.sample(n=mt_number)[feature])

    pvalues[video] = k/iteration



