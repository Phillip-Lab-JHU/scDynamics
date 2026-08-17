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
"""Generates Data for Figure2-3. wt and MT transition dynamics"""

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

# conditions = [
#     (df['kmeans'] == 0),
#     (df['kmeans'] == 1),
#     (df['kmeans'] == 2),
#     (df['kmeans'] == 3),
#     (df['kmeans'] == 4),
#     (df['kmeans'] == 5),
#     (df['kmeans'] == 6),
#     (df['kmeans'] == 7),
#     (df['kmeans'] == 8),
# ]
#
# values = [0, 0, 0, 1, 1, 1, 2, 2, 2]
#
# df['big_kmeans'] = np.select(conditions, values, default='').astype(int)

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure2-3. Transition dynamics of wt and mt GCB\\'

############################### Motility cluster transitions of GCB for two nodes ################################
entropies_all = {}
cluster_type = 'kmeans'
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

############################### Motility cluster transitions of GCB for three nodes ################################

cluster_type = 'kmeans'
for cell_type in ['wt_B-cell', 'mt_B-cell']:
    cluster_size = np.unique(df[cluster_type]).size
    df_trans = df[df['Type']==cell_type].reset_index(drop=True)

    transit_array1_temp = np.empty((cluster_size,cluster_size), dtype = 'object')
    transit_array2_temp = np.empty((cluster_size,cluster_size), dtype = 'object')
    for row in range(0,transit_array1_temp.shape[0]):
        for col in range(0,transit_array1_temp.shape[1]):
            transit_array1_temp[row,col] = []
            transit_array2_temp[row,col] = []
    for label in np.unique(df_trans['Label']):
        each_label = df_trans[df_trans['Label']==label].reset_index(drop=True)
        if each_label.shape[0] >= 3:
            row1 = each_label[:3][cluster_type][0]
            col1 = each_label[:3][cluster_type][1]
            row2 = each_label[:3][cluster_type][1]
            col2 = each_label[:3][cluster_type][2]
            transit_array1_temp[row1, col1].append(1)
            transit_array2_temp[row2, col2].append(1)

    transit_array1 = np.empty((cluster_size,cluster_size), dtype = 'object')
    transit_array2 = np.empty((cluster_size,cluster_size), dtype = 'object')
    for row in range(0,transit_array1_temp.shape[0]):
        for col in range(0,transit_array1_temp.shape[1]):
            transit_array1[row,col] = len(transit_array1_temp[row,col])
            transit_array2[row, col] = len(transit_array2_temp[row, col])
    transit_array1.flatten()

    values = list(transit_array1.flatten())
    for i in transit_array2.flatten():
        values.append(i)

    node_dict = {}
    chars = ['A', 'B', 'C']
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
    chars = ['B', 'C']
    for idx, char in enumerate(chars):
        for i in range(cluster_size):
            for j in range(cluster_size):
                target.append(f'%s{j}'%char)


    source_node = [node_dict[x] for x in source]
    target_node = [node_dict[x] for x in target]

    node_label = ['MC%s'%i for i in range(cluster_size)]*3

    # color_list = ['red', 'green', 'blue', 'gray', 'cyan', 'violet', 'coral', 'brown', 'powderblue', 'olive',
    #               'hotpink', 'indigo', 'bisque', 'lawngreen', 'darksalmon', 'wheat', 'steelblue', 'black', 'lime', 'gold',
    #               'orange', 'cornsilk', 'ivory', 'aliceblue', 'peru', 'mistyrose', 'chocolate', 'slategrey',
    #               'cornflowerblue', 'silver'][:cluster_size]

    n_colors = np.unique(df[cluster_type]).shape[0]
    colors=cmc.batlow
    cmap = ['rgb'+str(colors(1. * i / n_colors)[:-1]) for i in range(n_colors)]

    link_color_list = []
    for j in range(2):
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
                color = cmap*3
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

    fig.write_html(path + '3 node sankey-plot_%s.html' % cell_type)



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

        transit_array_heatmap
        # transit_array_only_trans = transit_array.copy()
        # np.fill_diagonal(transit_array_only_trans, 0)
        # transit_array_only_trans_heatmap = 100 * transit_array_only_trans / np.sum(transit_array_only_trans, axis=1)[:, np.newaxis]
        # zero_transition_clusters = np.sum(transit_array_only_trans, axis=1) <= cutoff
        # transit_array_only_trans_heatmap[zero_transition_clusters, :] = 0



#################################### transition matrix of DZ, LZ, FDC Core ####################################

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df_duration = pd.read_parquet(path+'long_traj_duration_20.parquet')

# Remove Group A, IgG , mLT and CD40L
df_duration = df_duration[(df_duration['Exp']=='Exp1')|(df_duration['Exp']=='Exp2')|(df_duration['Exp']=='Exp3')].reset_index(drop=True)

videos = np.unique(df_duration['Video'])
df_duration = df_duration[(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])
                          &(df_duration['Video'] != videos[-1])].reset_index(drop=True)
videos = np.unique(df_duration['Video'])

zone_series = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Zone', equal_length=False, frame_name='Time')
type_series = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Type', equal_length=False, frame_name='Time')
video_series = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Video', equal_length=False, frame_name='Time')

trajectories = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['Position X', 'Position Y', 'Position Z'],
                                               equal_length=False, frame_name='Time')

type_list = []
for idx, types in type_series.items():
    type = np.unique(types)
    type_list.append(type[0])

type_list = np.array(type_list)
print(type_list)

video_list = []
for idx, videos in video_series.items():
    video = np.unique(videos)
    video_list.append(video[0])

video_list = np.array(video_list)
print(video_list)

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure2-3. Transition dynamics of wt and mt GCB\\'

#################################### transition matrix of each 1 frame for each video ####################################
def do_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

from collections import Counter

custom_range = (1, 40)
stepsize = 1
tran_matrix_norms = {}
#for n_frame in np.arange(custom_range[0], custom_range[1] + stepsize, stepsize):

n_frame = 20
transition_matrix_list = []
for idx, transition in zone_series.items():  # For each cell
    new_transition = []
    for batch in do_batch(transition, n_frame):  # do_batch(transition, 4) = [2, 2, 1, 2], [1, 2, 1, 2], [2, 2, 1, 1,], ... group by 4 elements
        state = Counter(batch).most_common(1)[0][0]  # Counter(batch).most_common(1) return [(highest frequency value, frequency)]
        # Counter(batch).most_common(1)[0][0] therefore returns highest frequency value
        # total_freq = batch.size
        # freq = Counter(batch).most_common(1)[0][1]
        # if freq / total_freq ==0.5:
        #     state = np.nan
        # else:
        #     state = Counter(batch).most_common(1)[0][0]  # Counter(batch).most_common(1) return [(highest frequency value, frequency)]
        # How this works: zone_ts = [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2]
        # do_batch(zone_ts, 11) = [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2], [1, 1, 1, 2, 2, 1, 1, 2, 2, 2]
        # Counter(batch).most_common(1)[0][0] = [2], [1] (most frequent value from each batch)
        # If same freq -> Pick first value

        #state = round( np.median(batch) )  # np.median([1, 2]) = 1.5
        new_transition.append(state)
    trans_matrix = compute_transition_matrix(new_transition, n_states=3)
    transition_matrix_list.append(trans_matrix)

transition_matrix_list = np.array(transition_matrix_list)
np.sum(transition_matrix_list, axis=0)

one_to_twos = {'WT': [], 'MT': []}
for video in np.unique(video_list):
    wt_per_video = np.logical_and(video_list == video, type_list == 'wt B-cell')
    mt_per_video = np.logical_and(video_list == video, type_list == 'mt B-cell')

    if np.sum(wt_per_video) <= 5 or np.sum(mt_per_video) <= 5:
        continue
    transtion_matrix_wt = transition_matrix_list[wt_per_video]
    transtion_matrix_mt = transition_matrix_list[mt_per_video]

    transtion_matrix_wt = np.sum(transtion_matrix_wt, axis=0)
    transtion_matrix_mt = np.sum(transtion_matrix_mt, axis=0)

    transtion_matrix_wt = transtion_matrix_wt / np.sum(transtion_matrix_wt, axis=1)[:, np.newaxis]
    transtion_matrix_mt = transtion_matrix_mt / np.sum(transtion_matrix_mt, axis=1)[:, np.newaxis]
    one_to_twos['WT'].append(transtion_matrix_wt[1, 2])
    one_to_twos['MT'].append(transtion_matrix_mt[1, 2])
    print(video, np.sum(wt_per_video), np.sum(mt_per_video), transtion_matrix_wt[1, 2], transtion_matrix_mt[1, 2])

test = 'mann-whitney'
colors = ('#888888', '#CC6677')
draw_custom_box_plot(one_to_twos, path, file_name='test_%s'%n_frame,
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1, 2))




#################################### transition matrix of each 1 frame ####################################

transition_matrix_list = []
for idx, transition in zone_series.items():
    trans_matrix = compute_transition_matrix(transition, n_states=3)
    transition_matrix_list.append(trans_matrix)

transition_matrix_list = np.array(transition_matrix_list)

tran_matrix_wt = transition_matrix_list[type_list == 'wt B-cell']
tran_matrix_wt = np.sum(tran_matrix_wt, axis=0)

tran_matrix_wt = np.sum(tran_matrix_wt, axis=0)
tran_matrix_wt = tran_matrix_wt/np.sum(tran_matrix_wt, axis=1)[:, np.newaxis]

tran_matrix_mt = transition_matrix_list[type_list == 'mt B-cell']
tran_matrix_mt = np.sum(tran_matrix_mt, axis=0)
tran_matrix_mt = tran_matrix_mt/np.sum(tran_matrix_mt, axis=1)[:, np.newaxis]





#################################### Zone transition matrix of each elapsed time ####################################

custom_range = (1, 40)
stepsize = 1
label_stepsize = 4
#color_list=['#CC6677', '#44AA99', '#6699CC']
color_list=['#888888', '#CC6677']

marker_list=['.', '.', 'o', ]
xlabel = 'elapsed time (min)'

def do_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

from collections import Counter

only_transs = {}
transs = {}
tran_matrix_normss = {}
trans_othernorms = {}
for typ in ['wt B-cell', 'mt B-cell', 'T-cell']:
    only_trans = {'sLZ -> DZ': [], 'sLZ -> dLZ': [], 'dLZ -> sLZ':[], 'DZ -> sLZ':[]}
    trans = {'sLZ -> DZ': [], 'sLZ -> dLZ': [], 'dLZ -> sLZ':[], 'DZ -> sLZ':[]}
    trans_othernorm = {'sLZ -> DZ': [], 'sLZ -> dLZ': [], 'dLZ -> sLZ':[], 'DZ -> sLZ':[]}
    tran_matrix_norms = {}
    for n_frame in np.arange(custom_range[0], custom_range[1] + stepsize, stepsize):
        #n_frame = 20
        transition_matrix_list = []
        for idx, transition in zone_series.items():  # For each cell
            new_transition = []
            for batch in do_batch(transition, n_frame):  # do_batch(transition, 4) = [2, 2, 1, 2], [1, 2, 1, 2], [2, 2, 1, 1,], ... group by 4 elements
                state = Counter(batch).most_common(1)[0][0]  # Counter(batch).most_common(1) return [(highest frequency value, frequency)]
                # Counter(batch).most_common(1)[0][0] therefore returns highest frequency value
                # total_freq = batch.size
                # freq = Counter(batch).most_common(1)[0][1]
                # if freq / total_freq ==0.5:
                #     state = np.nan
                # else:
                #     state = Counter(batch).most_common(1)[0][0]  # Counter(batch).most_common(1) return [(highest frequency value, frequency)]
                # How this works: zone_ts = [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2]
                # do_batch(zone_ts, 11) = [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2], [1, 1, 1, 2, 2, 1, 1, 2, 2, 2]
                # Counter(batch).most_common(1)[0][0] = [2], [1] (most frequent value from each batch)
                # If same freq -> Pick first value

                #state = round( np.median(batch) )  # np.median([1, 2]) = 1.5
                new_transition.append(state)
            trans_matrix = compute_transition_matrix(new_transition, n_states=3)
            transition_matrix_list.append(trans_matrix)

        transition_matrix_list = np.array(transition_matrix_list)

        ####### Regular transition #######
        tran_matrix = transition_matrix_list[type_list == typ]
        tran_matrix_whole = np.sum(tran_matrix, axis=0)
        tran_matrix_norm = tran_matrix_whole / np.sum(tran_matrix_whole, axis=1)[:, np.newaxis]
        one_to_zero = tran_matrix_norm[1, 0]
        one_to_two = tran_matrix_norm[1, 2]
        two_to_one = tran_matrix_norm[2, 1]
        zero_to_one = tran_matrix_norm[0, 1]

        trans['sLZ -> DZ'].append(one_to_zero)
        trans['sLZ -> dLZ'].append(one_to_two)
        trans['dLZ -> sLZ'].append(two_to_one)
        trans['DZ -> sLZ'].append(zero_to_one)

        tran_matrix_norms[n_frame] = tran_matrix_norm

        ####### Relative transition #######
        tran_matrix_only_trans = tran_matrix_whole.copy()
        np.fill_diagonal(tran_matrix_only_trans, 0)
        # tran_matrix_only_trans = tran_matrix_only_trans / np.sum(tran_matrix_only_trans, axis=1)[:, np.newaxis]
        # one_to_zero_only_trans = tran_matrix_only_trans[1, 0]
        # one_to_two_only_trans = tran_matrix_only_trans[1, 2]
        # two_to_one_only_trans = tran_matrix_only_trans[2, 1]
        # zero_to_one_only_trans = tran_matrix_only_trans[0, 1]
        # only_trans['sLZ -> DZ'].append(one_to_zero_only_trans)
        # only_trans['sLZ -> dLZ'].append(one_to_two_only_trans)
        # only_trans['dLZ -> sLZ'].append(two_to_one_only_trans)
        # only_trans['DZ -> sLZ'].append(zero_to_one_only_trans)
        tran_matrix_only_trans = tran_matrix_only_trans / np.sum(tran_matrix_only_trans)
        one_to_zero_only_trans = tran_matrix_only_trans[1, 0]
        one_to_two_only_trans = tran_matrix_only_trans[1, 2]
        two_to_one_only_trans = tran_matrix_only_trans[2, 1]
        zero_to_one_only_trans = tran_matrix_only_trans[0, 1]
        only_trans['sLZ -> DZ'].append(one_to_zero_only_trans)
        only_trans['sLZ -> dLZ'].append(one_to_two_only_trans)
        only_trans['dLZ -> sLZ'].append(two_to_one_only_trans)
        only_trans['DZ -> sLZ'].append(zero_to_one_only_trans)
        # ####### Other normalization transition #######
        # tran_matrix_othernorm = tran_matrix_whole.copy()
        # tran_matrix_othernorm = tran_matrix_othernorm / np.sum(tran_matrix_othernorm, axis=0)[np.newaxis, :]
        # one_to_zero_othernorm = tran_matrix_othernorm[1, 0]
        # one_to_two_othernorm = tran_matrix_othernorm[1, 2]
        # two_to_one_othernorm = tran_matrix_othernorm[2, 1]
        # zero_to_one_othernorm = tran_matrix_othernorm[0, 1]
        #
        # trans_othernorm['sLZ -> DZ'].append(one_to_zero_othernorm)
        # trans_othernorm['sLZ -> dLZ'].append(one_to_two_othernorm)
        # trans_othernorm['dLZ -> sLZ'].append(two_to_one_othernorm)
        # trans_othernorm['DZ -> sLZ'].append(zero_to_one_othernorm)

    transs[typ] = trans
    tran_matrix_normss[typ] = tran_matrix_norms
    only_transs[typ] = only_trans
    trans_othernorms[typ] = trans_othernorm



for name in ['sLZ -> DZ', 'sLZ -> dLZ', 'dLZ -> sLZ', 'DZ -> sLZ']:
    if name == 'sLZ -> DZ':
        file_name = 'sLZ-DZ transition'
    elif name == 'sLZ -> dLZ':
        file_name = 'sLZ-dLZ transition'
    elif name == 'dLZ -> sLZ':
        file_name = 'dLZ-sLZ transition'
    elif name == 'DZ -> sLZ':
        file_name = 'DZ-sLZ transition'

    dataset = {}
    for typ, values in transs.items():
        if typ == 'T-cell':
            continue
        dataset[typ] = values[name]


    font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(4,4))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    for idx, key in enumerate(dataset):
        sns.lineplot(data=dataset, x=np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), y=dataset[key],
                     label=key, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars', color=color_list[idx])
        # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
        #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

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
    ax.set_xlabel('%s'%xlabel, fontsize=16, weight='normal', color='0.2')
    ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize)*0.5,
                       rotation=35, rotation_mode='anchor', ha='right', fontsize=12, weight='normal')
    plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
               weight='normal')
    plt.yticks(fontsize=16, color='0.2', weight='normal')

    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
               loc='best')

    plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/' )
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
    plt.clf()
    plt.close()

    draw_custom_bar_plot(dataset, path, file_name='bar plot %s' % file_name,
                         colors=color_list,
                         vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))

for name in ['sLZ -> DZ', 'sLZ -> dLZ', 'dLZ -> sLZ', 'DZ -> sLZ']:
    if name == 'sLZ -> DZ':
        file_name = 'relative sLZ-DZ transition'
    elif name == 'sLZ -> dLZ':
        file_name = 'relative sLZ-dLZ transition'
    elif name == 'dLZ -> sLZ':
        file_name = 'relative dLZ-sLZ transition'
    elif name == 'DZ -> sLZ':
        file_name = 'relative DZ-sLZ transition'

    dataset = {}
    for typ, values in only_transs.items():
        if typ == 'T-cell':
            continue
        dataset[typ] = values[name]


    font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(4,4))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    for idx, key in enumerate(dataset):
        sns.lineplot(data=dataset, x=np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), y=dataset[key],
                     label=key, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars', color=color_list[idx])
        # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
        #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

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
    ax.set_xlabel('%s'%xlabel, fontsize=16, weight='normal', color='0.2')
    ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize)*0.5,
                       rotation=35, rotation_mode='anchor', ha='right', fontsize=12, weight='normal')
    plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
               weight='normal')
    plt.yticks(fontsize=16, color='0.2', weight='normal')

    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
               loc='best')

    plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/' )
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
    plt.clf()
    plt.close()

    draw_custom_bar_plot(dataset, path, file_name='bar plot %s' % file_name,
                         colors=color_list,
                         vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))


name = 'sLZ -> dLZ'
file_name = 'short time sLZ-dLZ transition'

dataset = {}
for typ, values in transs.items():
    if typ == 'T-cell':
        continue
    dataset[typ] = values[name][:20]

# custom_range = (1, 20)
# stepsize = 1
font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(4,4))
# sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
#                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
for idx, key in enumerate(dataset):
    sns.lineplot(data=dataset, x=np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), y=dataset[key],
                 label=key, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars', color=color_list[idx])
    # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
    #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

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
ax.set_xlabel('%s'%xlabel, fontsize=16, weight='normal', color='0.2')
ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize)*0.5,
                   rotation=35, rotation_mode='anchor', ha='right', fontsize=12, weight='normal')
plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
           weight='normal')
plt.yticks(fontsize=16, color='0.2', weight='normal')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
           loc='best')

plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()

draw_custom_bar_plot(dataset, path, file_name='bar plot %s' % file_name,
                     colors=color_list,
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))


# #################################### Association between zone transition and motilty ####################################
#
# wt_initial_idx = np.where(type_list=='wt B-cell')[0]
#
# wt_one_2_twos = []
# for idx, tras_matrix_wt_each in enumerate(tran_matrix_wt):
#     if tras_matrix_wt_each[1, 2] == 1:
#         wt_one_2_twos.append(idx)
#
# wt_one_2_twos = [x + wt_initial_idx for x in wt_one_2_twos]
#
# one_2_two_trajectories = np.array( [trajectories[key][:20] for key in wt_one_2_twos] ) # Not exactly this, for trajs that have 3 or more transitions
# wt_one_2_two_trajectories = array_to_dict(one_2_two_trajectories)
#
#
#
#
# mt_initial_idx = np.where(type_list=='mt B-cell')[0][0]
#
# mt_one_2_twos = []
# for idx, tras_matrix_mt_each in enumerate(tran_matrix_mt):
#     if tras_matrix_mt_each[1, 2] == 1:
#         mt_one_2_twos.append(idx)
#
# mt_one_2_twos = [x + mt_initial_idx for x in mt_one_2_twos]
#
# one_2_two_trajectories = np.array( [trajectories[key][:20] for key in mt_one_2_twos] ) # Not exactly this, for trajs that have 3 or more transitions
# mt_one_2_two_trajectories = array_to_dict(one_2_two_trajectories)
#
#
# mt_initial_idx = np.where(type_list=='mt B-cell')[0][0]
#
# mt_one_2_ones = []
# for idx, tras_matrix_mt_each in enumerate(tran_matrix_mt):
#     if tras_matrix_mt_each[1, 1] == 1:
#         mt_one_2_ones.append(idx)
#
# mt_one_2_ones = [x + mt_initial_idx for x in mt_one_2_ones]
#
# one_2_one_trajectories = np.array( [trajectories[key][:20] for key in mt_one_2_ones] ) # Not exactly this, for trajs that have 3 or more transitions
# one_2_one_trajectories = array_to_dict(one_2_one_trajectories)
#
#
#
# from features.basic_motility import BasicMotility
# feature_list = ['total_distance', 'avg_speed', 'max_speed', 'min_speed', 'net_distance',  'progressivity', 'alphas',
#                 'total_angle', 'avg_angle', 'max_angle', 'min_angle',
#                 'displ_variance', 'displ_cov','displ_skewness', 'displ_kurtosis', 'displ_ngaussalpha', 'displ_gini',
#                 'angle_variance', 'angle_cov', 'angle_skewness', 'angle_kurtosis', 'angle_ngaussalpha', 'angle_gini',
#                 'msds', 'displ_autocorr', 'displ_partial_autocorr', 'angle_autocorr', 'angle_partial_autocorr',
#                 #'displ_hurst_RS', 'angle_hurst_RS',
#                 ]
# basic_motil = BasicMotility(mt_one_2_two_trajectories, time_unit=0.5, feature_list=feature_list)
# df_mt = basic_motil.extract_features(tau_limit=3)
# df_mt['Type'] = 'mt_B-cell_transit'
#
# basic_motil = BasicMotility(one_2_one_trajectories, time_unit=0.5, feature_list=feature_list)
# df_mt2 = basic_motil.extract_features(tau_limit=3)
# df_mt2['Type'] = 'mt_B-cell'
#
#
#
# basic_motil = BasicMotility(wt_one_2_two_trajectories, time_unit=0.5, feature_list=feature_list)
# df_wt = basic_motil.extract_features(tau_limit=3)
# df_wt['Type'] = 'wt_B-cell'
#
# df_one_2_two = pd.concat([df_mt, df_mt2], axis=0).reset_index(drop=True)
#
#
# condition_name = 'Type'
# if not os.path.isdir(path + 'one_2_one_feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'one_2_one_feature_violin_plot_type/')
#
# for feature_name in feature_list:
#     dataset={}
#     for condition in np.unique(df_one_2_two[condition_name]):
#         data = df_one_2_two[df_one_2_two[condition_name] == condition][feature_name]
#         dataset[condition] = np.array(data)
#     new_order = ['mt_B-cell', 'mt_B-cell_transit']
#     ordered_dataset = change_dict_order(dataset, new_order)
#     dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }
#
#     draw_custom_violin_plot(dict_datasets, path+'one_2_one_feature_violin_plot_type/', file_name=feature_name, colors=('#6699CC', '#CC6677'),
#                             test='mann-whitney', pvalue=True, figsize=(1,2))


#################################### Causal relationship between interaction and MC ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df_duration = pd.read_parquet(path+'long_traj_duration_20.parquet')

# Remove Group A, IgG , mLT and CD40L
df_duration = df_duration[(df_duration['Exp']=='Exp1')|(df_duration['Exp']=='Exp2')|(df_duration['Exp']=='Exp3')
                              |(df_duration['Exp']=='Exp5')].reset_index(drop=True)

videos = np.unique(df_duration['Video'])
df_duration = df_duration[(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])&(df_duration['Video'] != videos[4])
                          &(df_duration['Video'] != videos[-1])].reset_index(drop=True)
videos = np.unique(df_duration['Video'])

# int_features = ['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell',
#                 'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell', 'Zone', 'Distance_to_DZ', 'Distance_to_LZ',
#        'Distance_to_FDC_core',]

FDC_overlap = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC',
                                              equal_length=False, frame_name='Time')
FDC_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Shortest_Distance_to_Surfaces_Surfaces=FDC',
                                              equal_length=False, frame_name='Time')
T_overlap = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell',
                                              equal_length=False, frame_name='Time')
T_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Shortest_Distance_to_Surfaces_Surfaces=T-cell',
                                              equal_length=False, frame_name='Time')
DZ_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Distance_to_DZ',
                                              equal_length=False, frame_name='Time')
LZ_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Distance_to_LZ',
                                              equal_length=False, frame_name='Time')
Core_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Distance_to_FDC_core',
                                              equal_length=False, frame_name='Time')
Zones = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Zone',
                                              equal_length=False, frame_name='Time')

label_series = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['Type', 'Time_span', 'Label'],
                                               equal_length=False, frame_name='Time')
trajectories = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['Position X', 'Position Y', 'Position Z'],
                                               equal_length=False, frame_name='Time')

from features.interaction import DistanceSignal, OverlapSignal, ZoneSignal
feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr', 'autocorr',
                ]
FDC_dist = DistanceSignal(FDC_distances)
df_distance = FDC_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)

feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
FDC_over = OverlapSignal(FDC_overlap)
df_overlap = FDC_over.extract_features(feature_list)

df_inter_FDC = pd.concat([df_distance, df_overlap], axis=1)
for column in df_inter_FDC.columns:
    df_inter_FDC.rename(columns={column:'FDC_'+column}, inplace=True)

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr', 'autocorr',
                ]

T_dist = DistanceSignal(T_distances)
df_distance = T_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)

feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
T_over = OverlapSignal(T_overlap)
df_overlap = T_over.extract_features(feature_list)

df_inter_T = pd.concat([df_distance, df_overlap], axis=1)
for column in df_inter_T.columns:
    df_inter_T.rename(columns={column:'T_'+column}, inplace=True)


feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr', 'autocorr',
                ]
DZ_dist = DistanceSignal(DZ_distances)
df_inter_DZ = DZ_dist.extract_features(feature_list, tau_limit=3)
for column in df_inter_DZ.columns:
    df_inter_DZ.rename(columns={column:'DZ_distance_'+column}, inplace=True)

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr', 'autocorr',
                ]
LZ_dist = DistanceSignal(LZ_distances)
df_inter_LZ = LZ_dist.extract_features(feature_list, tau_limit=3)
for column in df_inter_LZ.columns:
    df_inter_LZ.rename(columns={column:'LZ_distance_'+column}, inplace=True)

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr', 'autocorr',
                ]
Core_dist = DistanceSignal(Core_distances)
df_inter_Core = Core_dist.extract_features(feature_list, tau_limit=3)
for column in df_inter_Core.columns:
    df_inter_Core.rename(columns={column:'Core_distance_'+column}, inplace=True)


feature_list = ['avg_zone', 'dz_resident_times', 'dz_resident_persistences', 'slz_resident_times',
                'slz_resident_persistences', 'dlz_resident_times', 'dlz_resident_persistences']
Zone_func = ZoneSignal(Zones)
df_zone = Zone_func.extract_features(feature_list)


df_inter = pd.concat([df_inter_FDC, df_inter_T, df_inter_DZ, df_inter_LZ, df_inter_Core, df_zone], axis=1)


label_list = []
for idx, typs in label_series.items():
    label_list_temp = []
    n_columns = typs.shape[1]
    for col in range(n_columns):
        col_data = typs[:, col][0]
        label_list_temp.append(col_data)
    label_list.append(label_list_temp)

label_list = np.array(label_list)


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure2-3. Transition dynamics of wt and mt GCB\\'

# from features.basic_motility import BasicMotility
# feature_list = ['total_distance', 'avg_speed', 'max_speed', 'min_speed', 'net_distance',  'progressivity', 'alphas',
#                 'total_angle', 'avg_angle', 'max_angle', 'min_angle',
#                 'displ_variance', 'displ_cov','displ_skewness', 'displ_kurtosis', 'displ_ngaussalpha', 'displ_gini',
#                 'angle_variance', 'angle_cov', 'angle_skewness', 'angle_kurtosis', 'angle_ngaussalpha', 'angle_gini',
#                 'msds', 'displ_autocorr', 'displ_partial_autocorr', 'angle_autocorr', 'angle_partial_autocorr',
#                 #'displ_hurst_RS', 'angle_hurst_RS',
#                 ]
# basic_motil = BasicMotility(trajectories, time_unit=0.5, feature_list=feature_list)
# #basic_motil.plot_msd_alpha(path)
# #basic_motil.plot_rotated_trajectories(path)
# #basic_motil.plot_original_trajectories(path)
#
# df_basic = basic_motil.extract_features(tau_limit=3)


df_long = pd.concat([df_inter, pd.DataFrame(label_list, columns=['Type', 'Time_span', 'Label'])], axis=1)
df_long = df_long[df_long['Type']!='T-cell'].reset_index(drop=True)
df_long['Time_span'] = df_long['Time_span'].astype(float) / 2  # Change frame -> min

df_long['dz_resident_times'] = df_long['dz_resident_times'].astype(float) / 2  # Change frame -> min
df_long['dz_resident_persistences'] = df_long['dz_resident_persistences'].astype(float) / 2  # Change frame -> min
df_long['slz_resident_times'] = df_long['slz_resident_times'].astype(float) / 2  # Change frame -> min
df_long['slz_resident_persistences'] = df_long['slz_resident_persistences'].astype(float) / 2  # Change frame -> min
df_long['dlz_resident_times'] = df_long['dlz_resident_times'].astype(float) / 2  # Change frame -> min
df_long['dlz_resident_persistences'] = df_long['dlz_resident_persistences'].astype(float) / 2  # Change frame -> min

df_long.loc[(df_long['avg_zone'] < 0.4) & (df_long['avg_zone'] >= 0), 'Zone'] = 'DZ'
df_long.loc[(df_long['avg_zone'] < 0.8) & (df_long['avg_zone'] >= 0.4), 'Zone'] = 'DZ-sLZ'
df_long.loc[(df_long['avg_zone'] < 1.2) & (df_long['avg_zone'] >= 0.8), 'Zone'] = 'sLZ'
df_long.loc[(df_long['avg_zone'] < 1.6) & (df_long['avg_zone'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
df_long.loc[(df_long['avg_zone'] <= 2) & (df_long['avg_zone'] >= 1.6), 'Zone'] = 'dLZ'


df_long_ = df_long.replace({'Type': {'wt B-cell': 'WT', 'mt B-cell': 'MT'}})
df_long_.columns.get_loc('Type')
feature_list = df_long_.columns[:49]

############################### interaction feature wrt elapsed time ################################
draw_lineplot_by_custom_ranges(df_long_, path, folder_name='interaction_feature_wrt_elapsed_time', feature_list=feature_list,
                                   condition_name='Type', custsom_range=(10, 20), stepsize=1, range_feature='Time_span',
                                       color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label='Length of trajectory (min)',
                                   estimator='mean', replace_keys=None, pvalue=True, test='mann-whitney')


print(df_long[df_long['Zone']=='DZ'].shape[0], df_long[df_long['Zone']=='DZ-sLZ'].shape[0], df_long[df_long['Zone']=='sLZ'].shape[0],
      df_long[df_long['Zone']=='sLZ-dLZ'].shape[0], df_long[df_long['Zone']=='dLZ'].shape[0])

for zone in ['DZ', 'sLZ', 'dLZ']:
    df_long_part = df_long_[df_long_['Zone']==zone].reset_index(drop=True)
    # if not os.path.isdir(path + '%s int feature violin plot/'%zone):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    #     os.makedirs(path + '%s int feature violin plot/'%zone)
    draw_lineplot_by_custom_ranges(df_long_part, path, folder_name='%s interaction_feature_wrt_elapsed_time'%zone,feature_list=feature_list,
                                   condition_name='Type', custsom_range=(10, 20), stepsize=1, range_feature='Time_span',
                                   color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4, 4),
                                   x_label='Length of trajectory (min)',
                                   estimator='mean', replace_keys=None, pvalue=True, test='mann-whitney')


for zone in ['DZ','sLZ', 'dLZ']:
    df_part = df_long_[(df_long_['Zone'] == zone)].reset_index(drop=True)
    if not os.path.isdir(path + '%s int feature violin plot/'%zone):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s int feature violin plot/'%zone)
    if not os.path.isdir(path + '%s int feature box plot/'%zone):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s int feature box plot/'%zone)
    for feature_name in feature_list:
        condition_name = 'Type'
        dataset={}
        for cell_type in ['WT', 'MT']:
            data = df_part[df_part[condition_name] == cell_type][feature_name]
            dataset[cell_type] = np.array(data)

        values = flatten_nested_dict(dataset)
        if np.isnan(values).any() == True:  # Check at least one nan
            continue
        elif np.isfinite(values).all() == False:  # Check everything is not inf
            continue

        # draw_custom_bar_plot(dataset_renamed, path + '%s int feature violin plot/'%zone, file_name=feature_name,
        #                      strip_plot=False, colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))
        draw_custom_violin_plot(dataset, path + '%s int feature violin plot/'%zone, file_name=feature_name,
                                colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))
        draw_custom_box_plot(dataset, path + '%s int feature box plot/' % zone, file_name=feature_name,
                                colors=('#888888', '#CC6677'), strip_plot=False, test='mann-whitney', pvalue=True, figsize=(1, 2))

############################### interaction feature of wt vs mt ###############################

feature = 'dz_resident_times'
condition_name = 'Type'
dataset={}
for cell_type in ['WT', 'MT']:
    #data = df_long_[(df_long_[condition_name] == cell_type)&( (df_long_['Zone'] == 'sLZ')|(df_long_['Zone'] == 'sLZ-dLZ')|(df_long_['Zone'] == 'dLZ') )][feature_name]
    data = df_long_[(df_long_[condition_name] == cell_type)][feature]
    dataset[cell_type] = np.array(data)

[print(key, np.array(value).size) for key, value in dataset.items()]
# values = flatten_nested_dict(dataset)
# if np.isnan(values).any() == True:  # Check at least one nan
#     continue
# elif np.isfinite(values).all() == False:  # Check everything is not inf
#     continue

# draw_custom_bar_plot(dataset_renamed, path + '%s int feature violin plot/'%zone, file_name=feature_name,
#                      strip_plot=False, colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))
draw_custom_violin_plot(dataset, path, file_name='%s_violin_plot'%feature,
                        colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))
draw_custom_box_plot(dataset, path, file_name='%s_box_plot'%feature,
                        colors=('#888888', '#CC6677'), strip_plot=False, test='mann-whitney', pvalue=True, figsize=(1, 2))


