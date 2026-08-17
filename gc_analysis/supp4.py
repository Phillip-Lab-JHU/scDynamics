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

from scipy import stats
from utils.draw_utils import *
from Morphology import Morphodynamics
from utils.misc_utils import *

############################## Figure 3. Dynamics of cell-cell interaction #######################################

directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/'
df_lv = pd.read_parquet(directory+'latent_vector_20.parquet')

# directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/GCB/'
# df_lv = pd.read_parquet(directory+'GCB_latent_vector_20.parquet')

# directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/2D trajectory/'
# draw_2D_trajectories_one_figure(df_int, df_lv, directory, duration=20, number_of_ex=30, label_name='tskmeans', feature_name=['Rotated_X', 'Rotated_Y'])

interaction_features = []
overlapped_volume_features = []
shortest_distance_features = []
for column_name in df_lv.columns:
    if any(txt in column_name for txt in ('Overlapped',
                                          'Shortest_Distance')):  # Search whether 'Overlapped', or 'Shortest_Distance' word is in the column_name
        interaction_features.append(column_name)
    if ('Overlapped_Volume_Ratio' in column_name) and ('instant' not in column_name):
        overlapped_volume_features.append(column_name)

    if ('Shortest_Distance' in column_name) and ('instant' not in column_name):  # Search whether 'Overlapped', or 'Shortest_Distance' word is in the column_name
        shortest_distance_features.append(column_name)


df_interaction = pd.DataFrame()
tp_interactions = []
#df_interaction = pd.DataFrame()
for i in range(df_lv.shape[0]):
    each_row_interactions = pd.DataFrame()
    for feature_name in shortest_distance_features: # for each column
        each_row = df_lv.iloc[i, df_lv.columns.get_loc(feature_name)] # iloc[row_idx, column_idx]
        each_row_interactions = pd.concat([ each_row_interactions, pd.DataFrame(each_row, columns=[feature_name] )], axis=1)
        # each_row_interactions = (duration=20, Shortest Distance to Surface 1~n)
    tp_interaction = 0
    for t in range(0, each_row_interactions.shape[0]):
        each_time = each_row_interactions.iloc[t,:]
        if (each_time == 0).any() == True: # any returns True when at least one is True
            tp_interaction = tp_interaction + 1
    tp_interactions.append(tp_interaction)
df_interaction['tp_interaction_whole'] = tp_interactions


for feature_name in shortest_distance_features: # for each column
    tp_int_types = []
    for i in range(df_lv.shape[0]): # for each row
        each_row = df_lv.iloc[i, df_lv.columns.get_loc(feature_name)]
        tp_int_type = 0
        for t in range(each_row.shape[0]):
            each_time = each_row[t]
            if each_time == 0:
                tp_int_type = tp_int_type + 1
        tp_int_types.append(tp_int_type)

        #if each_row == 20:

    df_interaction['tp_interaction_' + feature_name[feature_name.find('=')+1:]] = tp_int_types

df = pd.concat([df_lv, df_interaction], axis = 1)

####################################### Instantaneous shortest distance histogram Ezh2 GCB vs WT wrt to FDC and T cell #############################################
directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/'
df_duration = pd.read_parquet(directory + 'traj_duration_20.parquet')
videos = np.unique(df_duration['Video'])
df_video = df_duration[(df_duration['Video'] != videos[0])&(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])].reset_index(drop=True)
df_wt = df_video[df_video['Type']=='wt_B-cell']
df_mt = df_video[df_video['Type']=='mt_B-cell']

directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Figure3. Interaction dynamics - supp/'

fig = plt.figure()
sns.set(context='notebook', style='ticks', font_scale=1)
kdeplot = sns.kdeplot(df_wt[shortest_distance_features[0]], fill=True, linewidth=2, clip=(0.0, 40), label='WT-FDC')
kdeplot = sns.kdeplot(df_mt[shortest_distance_features[0]], fill=True, linewidth=2, clip=(0.0, 40), label='MT-FDC')
plt.legend()
plt.savefig(directory+'GCB-FDC shortest distance distribution.png')
plt.close()
plt.clf()

fig = plt.figure()
sns.set(context='notebook', style='ticks', font_scale=1)
kdeplot = sns.kdeplot(df_wt[shortest_distance_features[3]], fill=True, linewidth=2, clip=(0.0, 40), label='WT-T')
kdeplot = sns.kdeplot(df_mt[shortest_distance_features[3]], fill=True, linewidth=2, clip=(0.0, 40), label='MT-T')
plt.legend()
plt.savefig(directory+'GCB-T shortest distance distribution.png')
plt.close()
plt.clf()

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

####################################### Quantify interaction frequency for T cell#############################################
directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Figure3. Interaction dynamics - supp/'
int_time = 20
freq_total_datasets = {}
freq_per_cellnumber_datasets = {}
total_int_freq_per_cellnumber_datasets = {}
persistent_int_per_cellcontact_datasets = {}
low_contact_freq_per_cellnumbers_datasets = {}

videos = np.unique(df['Video'])
for interaction_type in ['FDC', 'wt_B-cell', 'mt_B-cell']:
    for cell_type in ['T-cell']:
        df_part = df[df['Type'] == cell_type]
        int_freqs = []
        int_freq_per_cellnumbers = []
        total_int_freq_per_cellnumbers = []
        persistent_int_per_cellcontacts = []
        low_contact_freq_per_cellnumbers = []
        for video in videos:
            #if 'A' in video and cell_type == 'mt_B-cell':
            if '-A' in video:
                continue
            df_video = df_part[df_part['Video'] == video]
            if df_video.shape[0] == 0:
                continue
            data = df_video['tp_interaction_%s' % interaction_type]
            mask = ~np.isnan(data)
            data = data[mask]
            int_freq = sum(data >= int_time)
            int_freq_per_cellnumber = sum(data >= int_time) / df_video.shape[0]
            total_int_freq_per_cellnumber = sum(data) / df_video.shape[0]
            persistent_int_per_cellcontact = sum(data >= int_time) / sum(data)
            low_contact_freq_per_cellnumber = sum((1<=data) & (data<=3)) / df_video.shape[0]
            int_freqs.append(int_freq)
            int_freq_per_cellnumbers.append(int_freq_per_cellnumber)
            total_int_freq_per_cellnumbers.append(total_int_freq_per_cellnumber)
            persistent_int_per_cellcontacts.append(persistent_int_per_cellcontact)
            low_contact_freq_per_cellnumbers.append(low_contact_freq_per_cellnumber)

        freq_total_datasets[cell_type + '-' + interaction_type] = int_freqs
        freq_per_cellnumber_datasets[cell_type + '-' + interaction_type] = int_freq_per_cellnumbers
        total_int_freq_per_cellnumber_datasets[cell_type + '-' + interaction_type] = total_int_freq_per_cellnumbers
        persistent_int_per_cellcontact_datasets[cell_type + '-' + interaction_type] = persistent_int_per_cellcontacts
        low_contact_freq_per_cellnumbers_datasets[cell_type + '-' + interaction_type] = low_contact_freq_per_cellnumbers

directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Interaction/Quality of interaction/T cell/'
draw_custom_box_plot(freq_total_datasets, directory, feature_name='Total T interaction frequency')
draw_custom_box_plot(freq_per_cellnumber_datasets, directory, feature_name='T persistent interaction frequency per cell number')
draw_custom_box_plot(total_int_freq_per_cellnumber_datasets, directory, feature_name='T number of contacts per cell number')
draw_custom_box_plot(persistent_int_per_cellcontact_datasets, directory, feature_name='T persistent interaction frequency per number of contacts')
draw_custom_box_plot(low_contact_freq_per_cellnumbers_datasets, directory, feature_name='T low contact time frequency per cell number')

####################################### Quantify FDC interaction frequency for all lymphocytes #############################################
int_time = 20
freq_total_datasets = {}
freq_per_cellnumber_datasets = {}
total_int_freq_per_cellnumber_datasets = {}
persistent_int_per_cellcontact_datasets = {}
low_contact_freq_per_cellnumbers_datasets = {}

videos = np.unique(df['Video'])
for interaction_type in ['FDC']:
    for cell_type in ['T-cell', 'mt_B-cell', 'wt_B-cell']:
        df_part = df[df['Type'] == cell_type]
        int_freqs = []
        int_freq_per_cellnumbers = []
        total_int_freq_per_cellnumbers = []
        persistent_int_per_cellcontacts = []
        low_contact_freq_per_cellnumbers = []
        for video in videos:
            #if 'A' in video and cell_type == 'mt_B-cell':
            if '-A' in video:
                continue
            df_video = df_part[df_part['Video'] == video]
            if df_video.shape[0] == 0:
                continue
            data = df_video['tp_interaction_%s' % interaction_type]
            mask = ~np.isnan(data)
            data = data[mask]
            int_freq = sum(data >= int_time)
            int_freq_per_cellnumber = sum(data >= int_time) / df_video.shape[0]
            total_int_freq_per_cellnumber = sum(data) / df_video.shape[0]
            persistent_int_per_cellcontact = sum(data >= int_time) / sum(data)
            low_contact_freq_per_cellnumber = sum((1<=data) & (data<=3)) / df_video.shape[0]
            int_freqs.append(int_freq)
            int_freq_per_cellnumbers.append(int_freq_per_cellnumber)
            total_int_freq_per_cellnumbers.append(total_int_freq_per_cellnumber)
            persistent_int_per_cellcontacts.append(persistent_int_per_cellcontact)
            low_contact_freq_per_cellnumbers.append(low_contact_freq_per_cellnumber)

        freq_total_datasets[cell_type + '-' + interaction_type] = int_freqs
        freq_per_cellnumber_datasets[cell_type + '-' + interaction_type] = int_freq_per_cellnumbers
        total_int_freq_per_cellnumber_datasets[cell_type + '-' + interaction_type] = total_int_freq_per_cellnumbers
        persistent_int_per_cellcontact_datasets[cell_type + '-' + interaction_type] = persistent_int_per_cellcontacts
        low_contact_freq_per_cellnumbers_datasets[cell_type + '-' + interaction_type] = low_contact_freq_per_cellnumbers

directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Interaction/Quality of interaction/all cells/'
draw_custom_box_plot(freq_total_datasets, directory, feature_name='Total FDC interaction frequency')
draw_custom_box_plot(freq_per_cellnumber_datasets, directory, feature_name='FDC persistent interaction frequency per cell number')
draw_custom_box_plot(total_int_freq_per_cellnumber_datasets, directory, feature_name='FDC number of contacts per cell number')
draw_custom_box_plot(persistent_int_per_cellcontact_datasets, directory, feature_name='FDC persistent interaction frequency per number of contacts')
draw_custom_box_plot(low_contact_freq_per_cellnumbers_datasets, directory, feature_name='FDC low contact time frequency per cell number')

####################################### Number of Ezh2 GCB vs WT with Exp group 2 and Exp group 3 #############################################
cellnumber_group = {'B': [], 'C': []}
groups = np.unique(df['Exp_group'])
videos = np.unique(df['Video'])[3:]
for group in groups:
    if 'A' in group:
        continue
    df_group = df[df['Exp_group'] == group]
    for video in videos:
        if '-%s'%group not in video:
            continue
        df_video = df_group[(df_group['Video'] == video)&(df_group['Type'] != 'T-cell')]
        df_part = df_video[df_video['Type'] == 'wt_B-cell']
        cell_number_fraction = df_part.shape[0]/df_video.shape[0]
        cellnumber_group[group].append(cell_number_fraction)

directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Interaction/Quality of interaction/'
draw_custom_box_plot(cellnumber_group, directory, feature_name='Exp B vs Exp C GC B cell number')

####################################### Quantify interaction frequency histogram #############################################
df_part = df[df['Type']=='T-cell']
bin_num=20
plt.figure(figsize=(10,8))
for i, interaction_type in enumerate(['FDC', 'wt_B-cell', 'mt_B-cell']):
    data = df_part['tp_interaction_%s'%interaction_type]
    mask = ~np.isnan(data) # True when it is not nan (for T cell ~ mt B interaction)
    data = data[mask]
    kde = scipy.stats.gaussian_kde(data)
    x = np.linspace(0,20,bin_num)
    pdf = kde.evaluate(x)
    plt.subplot(3, 1, i+1)
    #plt.plot(x, pdf, label='T-cell-'+interaction_type,)
    #plt.xticks(plt.xticks()[0], sorted_keys, fontsize=9.5, fontdict={'weight': 'normal'})
    plt.hist(data, path, alpha = 0.5, label='T-cell-'+interaction_type)
    plt.legend(loc=1, borderaxespad=0.0, fontsize=10, markerscale=5)
    #plt.xlabel('Time(frames)')
    #plt.ylabel('Number of cell trajectories')
    plt.ylim(0, 3000)
plt.show()
plt.close()
plt.clf()


plt.figure(figsize=(10,8))
bin_num=20
num_colors = 4
cm = plt.cm.get_cmap(name='Set2')
currentColors = [cm(1.*i/num_colors) for i in range(num_colors)]
i=1
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    df_part = df[df['Type']==cell_type]
    for interaction_type in ['FDC', 'T-cell']:
        data = df_part['tp_interaction_%s'%interaction_type]
        mask = ~np.isnan(data)
        data = data[mask]
        kde = scipy.stats.gaussian_kde(data)
        x = np.linspace(0,20,bin_num)
        pdf = kde.evaluate(x)
        plt.subplot(4, 1, i)
        #plt.plot(x, pdf, label=cell_type+'-'+interaction_type, color =currentColors[i])


        plt.hist(data, path, alpha = 0.5, label=cell_type+'-'+interaction_type)
        plt.legend(loc=1, borderaxespad=0.0, fontsize=10, markerscale=5, )
        #plt.xlabel('Time(frames)')
        #plt.ylabel('Number of cell trajectories')
        i = i + 1
        plt.ylim(0, 1000)
plt.show()
plt.close()
plt.clf()

############################### Shortest distance to FDC profiles ################################
directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/'
df_duration = pd.read_parquet(directory + 'traj_duration_20.parquet')

ts = Morphodynamics(df_duration, 'umap')
cluster, cluster_center = ts.get_ts_cluster(df_duration, 20, expand=False, duration=20, feature_name=[shortest_distance_features[0]])

feature_name = 'Distance to FDC'
directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/FDC/'
for traj_idx, signal in enumerate(cluster_center):
    series = signal.flatten()
    time_range = range(0, 20)
    fig, ax = plt.subplots(figsize=(15, 6))

    sns.lineplot(x=time_range, y=series, linewidth=3)
    # slope, intercept, r_val, p_val, SE = scipy.stats.linregress(time_range, series)
    # sns.lineplot(x=time_range, y=[slope * t for t in time_range] + intercept, color='red')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(feature_name, fontsize=12)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    ax.set_xticks(ticks=time_range)
    # ax.set_yticks(ticks=np.linspace(min(a), max(a), 5))
    # ax.grid(True)
    plt.xlim(0, 20 - 1)
    # plt.ylim(, )

    if not os.path.isdir(
            directory + '%s_series/' % feature_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(directory + '%s_series/' % feature_name)
    plt.savefig(directory + '%s_series/%s.png' % (feature_name, traj_idx))
    plt.clf()
    plt.close()

df['tskmeans_FDC'] = cluster

directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/FDC/'
draw_cluster_distribution_heatmap(df, directory, condition_name='Type', cluster_type='tskmeans_FDC')

############################### FDC overlapped volume profiles for all cells ################################

duration=20

df1 = df[overlapped_volume_features[0]]
empty = np.empty((df1.shape[0], duration))
for i, row in enumerate(df1):
    empty[i] = row

df_empty = pd.DataFrame(empty)

colors = ['red', 'green', 'blue', 'gray', 'cyan', 'violet', 'coral', 'brown', 'powderblue', 'olive',
                  'hotpink', 'indigo', 'bisque', 'lawngreen', 'darksalmon', 'wheat', 'steelblue', 'black', 'lime',
                  'gold', 'orange', 'cornsilk', 'ivory', 'aliceblue', 'peru', 'mistyrose', 'chocolate', 'slategrey',
                  'cornflowerblue', 'silver']

df_row_colors = pd.DataFrame()
for condition in ['Type', 'FDC_tskmeans']:
    my_palette = dict(zip(df[condition].unique(), colors[:df[condition].unique().size]))
    row_colors = df[condition].map(my_palette)
    df_row_colors_temp = pd.DataFrame(row_colors, columns=[condition])
    df_row_colors = pd.concat([df_row_colors, df_row_colors_temp], axis=1)


plt.figure(dpi=150)
#sns.set(font_scale=0.5)
sm = sns.clustermap(df_empty, annot=False, cmap='RdYlGn_r', xticklabels=True, col_cluster=False, vmax=0.1, method='ward',
                    figsize=(10,100),  cbar_kws=dict(use_gridspec=False,pad=0.01,shrink=0.001,), row_colors=df_row_colors
                    #dendrogram_ratio=0.1
                    )
#sm.cax.set_visible(False)

from matplotlib.patches import Patch

handles = [Patch(facecolor=my_palette[name]) for name in my_palette]
plt.legend(handles, my_palette, title=condition,
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')

plt.show()


############################### T-cell -> FDC, mt B-cell, wt B-cell interaction profiles ################################
directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Interaction/interaction dynamics/T cell interaction/'

duration=20

df_part = df[df['Type']=='T-cell'].reset_index(drop=True)
# overlapped_volume_features[0]
# overlapped_volume_features[2]
# overlapped_volume_features[4]
df1 = df_part[overlapped_volume_features[4]]
empty = np.empty((df1.shape[0], duration))
for i, row in enumerate(df1):
    empty[i] = row

df_empty = pd.DataFrame(empty)

colors = ['red', 'green', 'blue', 'gray', 'cyan', 'violet', 'coral', 'brown', 'powderblue', 'olive',
                  'hotpink', 'indigo', 'bisque', 'lawngreen', 'darksalmon', 'wheat', 'steelblue', 'black', 'lime',
                  'gold', 'orange', 'cornsilk', 'ivory', 'aliceblue', 'peru', 'mistyrose', 'chocolate', 'slategrey',
                  'cornflowerblue', 'silver']

from matplotlib.patches import Patch
df_row_colors = pd.DataFrame()
#'wt_B-cell_tskmeans', 'mt_B-cell_tskmeans', 'FDC_tskmeans'
for condition in ['mt_B-cell_tskmeans']:
    my_palette = dict(zip(df_part[condition].unique(), colors[:df_part[condition].unique().size]))
    row_colors = df_part[condition].map(my_palette)
    df_row_colors_temp = pd.DataFrame(row_colors, columns=[condition])
    df_row_colors = pd.concat([df_row_colors, df_row_colors_temp], axis=1)


plt.figure(dpi=150)
#sns.set(font_scale=0.5)
sm = sns.clustermap(df_empty, annot=False, cmap='RdYlGn_r', xticklabels=True, col_cluster=False, vmax=0.1,
                    figsize=(10,100),  cbar_kws=dict(use_gridspec=False,pad=0.01,shrink=0.001,), row_colors=df_row_colors, method='ward',
                    #dendrogram_ratio=0.1
                    )
handles = [Patch(facecolor=my_palette[name]) for name in my_palette]
plt.legend(handles, my_palette, title=condition,
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
#sm.cax.set_visible(False)
#plt.show()
plt.savefig(directory+'mt B cell interaction profiles.png')



draw_cluster_distribution_heatmap(df_part, directory, condition_name='tskmeans', cluster_type='mt_B-cell_tskmeans', vmax=70)


############################### B-cell -> FDC, T-cell interaction profiles ################################
directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Interaction/interaction dynamics/wt B cell interaction/'

duration=20

df_part = df[(df['Type']=='wt_B-cell')|(df['Type']=='mt_B-cell')].reset_index(drop=True)
# overlapped_volume_features[0]
# overlapped_volume_features[2]
# overlapped_volume_features[4]
df1 = df_part[overlapped_volume_features[0]]
empty = np.empty((df1.shape[0], duration))
for i, row in enumerate(df1):
    empty[i] = row

df_empty = pd.DataFrame(empty)

colors = ['red', 'green', 'blue', 'gray', 'cyan', 'violet', 'coral', 'brown', 'powderblue', 'olive',
                  'hotpink', 'indigo', 'bisque', 'lawngreen', 'darksalmon', 'wheat', 'steelblue', 'black', 'lime',
                  'gold', 'orange', 'cornsilk', 'ivory', 'aliceblue', 'peru', 'mistyrose', 'chocolate', 'slategrey',
                  'cornflowerblue', 'silver']

from matplotlib.patches import Patch
df_row_colors = pd.DataFrame()
#'wt_B-cell_tskmeans', 'mt_B-cell_tskmeans', 'FDC_tskmeans'
for condition in ['Type', 'FDC_tskmeans']:
    my_palette = dict(zip(df_part[condition].unique(), colors[:df_part[condition].unique().size]))
    row_colors = df_part[condition].map(my_palette)
    df_row_colors_temp = pd.DataFrame(row_colors, columns=[condition])
    df_row_colors = pd.concat([df_row_colors, df_row_colors_temp], axis=1)


plt.figure(dpi=150)
#sns.set(font_scale=0.5)
sm = sns.clustermap(df_empty, annot=False, cmap='RdYlGn_r', xticklabels=True, col_cluster=False, vmax=0.1,
                    figsize=(10,100),  cbar_kws=dict(use_gridspec=False,pad=0.01,shrink=0.001,), row_colors=df_row_colors, method='ward',
                    #dendrogram_ratio=0.1
                    )
handles = [Patch(facecolor=my_palette[name]) for name in my_palette]
plt.legend(handles, my_palette, title=condition,
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
#sm.cax.set_visible(False)
#plt.show()
plt.savefig(directory+'FDC interaction profiles.png')

draw_cluster_distribution_heatmap(df_part, directory, condition_name='tskmeans', cluster_type='FDC_tskmeans', vmax=70)

############################### Point process gc_analysis ################################

# Find thresholds of T-FDC, T-wt, T-mt, wt-FDC, wt-T, mt-FDC, mt-T (in order)
directory = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Interaction/Point process gc_analysis/'
from kneed import KneeLocator
duration=20
threshes=[]
for cell_type in ['T-cell', 'wt_B-cell', 'mt_B-cell',]:
    if cell_type == 'T-cell':
        idxs = [0,2,4]
    if cell_type == 'wt_B-cell' or cell_type == 'mt_B-cell':
        idxs = [0, 3]
    for idx in idxs: # 0 = FDC, 1 = macrophage, 2 = wt B cell, 3 = T cell, 4 = mt B cell
        df_partial = df[df['Type']==cell_type]
        new_df = pd.DataFrame()
        for i in range(df_partial.shape[0]):
            row = df_partial.iloc[i,:][overlapped_volume_features[idx]] # overlapped_volume_features[0]
            if any(np.unique(row) != 0): # Delete rows that only has 0 for overlapped volumes
                # any returns True when at least one is True
                new_df = pd.concat([new_df, df_partial[:][i:i+1]])

        df1 = new_df[overlapped_volume_features[idx]]
        empty = np.empty((df1.shape[0], duration))
        for i, row in enumerate(df1):
            empty[i] = row

        plt.figure()
        kdeplot = sns.kdeplot(empty.flatten(), fill=True, linewidth=2, clip=(0.0, 1.0))

        x, y = sns.kdeplot(empty.flatten(), clip=(0.0, 1.0)).lines[0].get_data()
        kl = KneeLocator(x, y, curve='convex', direction='decreasing')

        thresh = kl.elbow
        threshes.append(thresh)

        plt.scatter(kl.elbow, kl.elbow_y, zorder=2, color='red') # zorder determines which image is in front (zorder = 2 is very front)
        #plt.vlines(kl.elbow, 0, kl.elbow_y, linestyle="dashed")
        #plt.hlines(kl.elbow_y, 0, kl.elbow, linestyle="dashed")
        print('threshold for overlapped volume: ', kl.elbow)  # find point of maximum curvature
        plt.title('threshold: %s' %kl.elbow)
        plt.savefig(directory+'%s_%s.png'%(cell_type, overlapped_volume_features[idx][overlapped_volume_features[idx].find('=')+1:]), dpi=300)
        plt.clf()
        plt.close()



df_concat = pd.DataFrame()
j = 0
for cell_type in ['T-cell', 'mt_B-cell', 'wt_B-cell']:
    new_df = pd.DataFrame()
    df_concat_temp = pd.DataFrame()
    df_partial = df[df['Type'] == cell_type].copy()
    if cell_type == 'T-cell':
        idxs = [0, 2, 4]
    if cell_type == 'wt_B-cell' or cell_type == 'mt_B-cell':
        idxs = [0, 3]
    for idx in idxs:  # 0 = FDC, 1 = macrophage, 2 = wt B cell, 3 = T cell, 4 = mt B cell
        interaction_scores = []
        interaction_covs = []
        for i in range(df_partial.shape[0]):
            row = df_partial.iloc[i, :][overlapped_volume_features[idx]]  # overlapped_volume_features[0]
            interaction_profile = row > threshes[j]
            interaction_profile = interaction_profile * 1  # convert false -> 0, true -> 1
            if np.mean(interaction_profile) == 0:
                interaction_score = np.nan
                interaction_cov = np.nan
            else:
                interaction_score = np.sum(interaction_profile)
                interaction_cov = np.std(interaction_profile) / np.mean(interaction_profile)
            interaction_covs.append(interaction_cov)
            interaction_scores.append(interaction_score)
        interaction_type = overlapped_volume_features[idx][overlapped_volume_features[idx].find('=') + 1:]
        new_df['%s_%s_score' % (cell_type, interaction_type)] = interaction_scores
        new_df['%s_%s_cov' % (cell_type, interaction_type)] = interaction_covs
        j = j + 1
    # dfs.append(new_df)
    new_df = new_df.set_index( pd.Index([i for i in df_partial.index]) ) # match index of new_df to df_partial for correct concatenatation
    df_concat_temp = pd.concat([df_partial, new_df], axis=1)
    df_concat = pd.concat([df_concat, df_concat_temp], axis=0)

df_concat