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
"""Generates Data for Figure2-3. """
import scipy.stats
from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import ZoneSignal




# probs = {'WT':[], 'MT':[]}
# for i in range(100):
#     wt_prob = simulate_markov_chain(transit_array_heatmaps[0] / 100, start_state=2, n_steps=100)
#     mt_prob = simulate_markov_chain(transit_array_heatmaps[1] / 100, start_state=2, n_steps=100)
#     probs['WT'].append(wt_prob[-1])
#     probs['MT'].append(mt_prob[-1])
#
#
# test = 'mann-whitney'
# colors = ('#888888', '#CC6677')
# draw_custom_bar_plot(probs, path, file_name='test',
#                      strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1, 2))



########################################################################################################

###1. From segmentation from Imaris, you combine all excel files into one excel file using data curation.py
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Imaris csvs\\'
excel = pd.read_parquet(path + 'Intravital Data_all.parquet')

########################################################################################################
### 2. to_trajectory_duration to sort the dataframe by cells and fixed duration
for duration in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
#for duration in [20]:
    df = pd.DataFrame()
    for idx, video in enumerate(np.unique(excel['Video'])):
        print(idx, video)
        excel_temp = excel[excel['Video']==video].reset_index(drop=True)
        df_duration_temp = to_trajectory_duration(excel_temp, duration=duration, condition_name='Type', frame_name='Time', label_name='Label', verbose=False)
        #df_duration_temp = to_trajectory_variable_duration(excel_temp, min_duration=duration, condition_name='Type', label_name='Label')
        df = pd.concat([df, df_duration_temp], axis=0)
    df = df.reset_index(drop=True)
    path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\traj_durations_all\\'
    df.to_csv(path + 'traj_duration_%s.csv' % duration, index=False)
    df.to_parquet(path + 'traj_duration_%s.parquet' % duration)

########################################################################################################
### 3. Save df and df_duration files
for duration in range(1, 40):
    path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\traj_durations_all\\'
    df_duration = pd.read_parquet(path+'traj_duration_%s.parquet'%duration)
    df_duration.rename(columns=lambda x: x.replace('Tfh', 'T-cell').replace('wtGCB', 'wt_B-cell').replace('mtGCB', 'mt_B-cell'), inplace=True)
    df_duration['Type'].replace({'Tfh': 'T-cell', 'wtGCB': 'wt_B-cell', 'mtGCB': 'mt_B-cell'}, inplace=True)

    df_labels = reduced_label_for_overlapped_volume(df_duration, duration=duration)
    df = df_labels[['TrackID', 'Time', 'pseudo_Time', 'Label', 'pseudo_Label', 'Type', 'Video', 'Exp', 'Day',
                           'Exp_group', 'Time_span']]

    _, _, Zone_series = to_timeseries_fast(df_duration, duration=duration, feature_name='Zone')

    feature_list = ['avg_zone', 'dz_resident_times', 'dz_resident_persistences', 'slz_resident_times',
                    'slz_resident_persistences', 'dlz_resident_times', 'dlz_resident_persistences']
    Zone_func = ZoneSignal(Zone_series)
    df_zone = Zone_func.extract_features(feature_list)

    df = pd.concat([df, df_zone], axis=1)

    df.loc[(df['avg_zone'] < 0.4) & (df['avg_zone'] >= 0), 'Zone'] = 'DZ'
    df.loc[(df['avg_zone'] < 0.8) & (df['avg_zone'] >= 0.4), 'Zone'] = 'DZ-sLZ'
    df.loc[(df['avg_zone'] < 1.2) & (df['avg_zone'] >= 0.8), 'Zone'] = 'sLZ'
    df.loc[(df['avg_zone'] < 1.6) & (df['avg_zone'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
    df.loc[(df['avg_zone'] <= 2) & (df['avg_zone'] >= 1.6), 'Zone'] = 'dLZ'

    #################################### Without inhibition GCBs ####################################
    df = df[(df['Exp'] == 'CD40L') | (df['Exp'] == 'IgG') | (df['Exp'] == 'mLT')].reset_index(drop=True)
    df_duration = df_duration[(df_duration['Exp'] == 'CD40L') | (df_duration['Exp'] == 'IgG') | (df_duration['Exp'] == 'mLT')].reset_index(drop=True)

    df = df[df['Type'] != 'T-cell'].reset_index(drop=True)
    df_duration = df_duration[df_duration['Type'] != 'T-cell'].reset_index(drop=True)

    zone_to_num = {'DZ': 0, 'DZ-sLZ': 1, 'sLZ': 2, 'sLZ-dLZ': 3, 'dLZ': 4}
    df['Zone_num'] = df['Zone'].map(zone_to_num)

    label_expanded = np.repeat(df['Zone'], duration).reset_index(drop=True)
    df_duration['Zone_label'] = label_expanded

    path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\traj_durations_all\transition\\'
    df.to_parquet(path + 'df_%s.parquet' % duration)
    df_duration.to_parquet(path + 'df_duration_%s.parquet' % duration)



############################### Zone transitions of GCB ################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\traj_durations_all\transition\\'
duration = 30
for duration in range (2, 30):
    print('%s duration (frames): '%duration)
    path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\traj_durations_all\transition\\'
    df = pd.read_parquet(path+'df_%s.parquet'%duration)
    df_duration = pd.read_parquet(path+'df_duration_%s.parquet'%duration)

    df = df[df['Exp'] == 'mLT'].reset_index(drop=True)
    df_duration = df_duration[df_duration['Exp'] == 'mLT'].reset_index(drop=True)

    # Validate and convert zone indices
    if df['Zone_num'].isna().any():
        missing = df.loc[df['Zone_num'].isna(), 'Zone'].value_counts(
            dropna=False
        )
        raise ValueError(f'Unmapped zones detected:\n{missing}')

    df['Zone_num'] = df['Zone_num'].astype(int)

    path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\rebuttal\mLT\\'

    cluster_type = 'Zone_num'
    transit_array_list = []
    for cell_type in ['wt_B-cell', 'mt_B-cell']:

        cluster_size = np.unique(df[cluster_type]).size
        df_trans = df[(df['Type']==cell_type)].reset_index(drop=True)
        count = 0
        transit_array_temp = np.empty((cluster_size,cluster_size), dtype = 'object')
        for row in range(0,transit_array_temp.shape[0]):
            for col in range(0,transit_array_temp.shape[1]):
                transit_array_temp[row,col] = []

        for label in np.unique(df_trans['Label']):
            each_label = df_trans[df_trans['Label']==label].reset_index(drop=True)
            if each_label.shape[0] >= 2:
                count = count + 1
                for idx in range(each_label.shape[0] - 1):
                    row = each_label[cluster_type][idx]
                    col = each_label[cluster_type][idx + 1]
                    transit_array_temp[row, col].append(1)

        #print(cell_type, count)
        transit_array = np.empty((cluster_size,cluster_size), dtype = 'float')
        for row in range(0,transit_array_temp.shape[0]):
            for col in range(0,transit_array_temp.shape[1]):
                transit_array[row,col] = len(transit_array_temp[row,col])
        ############## Heatmap ##############
        transit_array_heatmap = 100 * transit_array / np.sum(transit_array, axis=1)[:, np.newaxis]
        transit_array_list.append(transit_array_heatmap)
        #print(transit_array_heatmap/100)



    def simulate_markov_chain(trans_matrix, start_state, n_steps):
        n_states = trans_matrix.shape[0]
        curr_state = start_state
        pi = np.array([0] * n_states)
        pi[start_state] = 1

        i = 0
        while i < n_steps:
            curr_state = np.random.choice(range(0, n_states), p=trans_matrix[curr_state])
            pi[curr_state] += 1
            i += 1
        #print("π = ", pi / steps)
        return pi / n_steps

    wt_prob = simulate_markov_chain(transit_array_list[0]/100, start_state=2, n_steps=10**6)
    mt_prob = simulate_markov_chain(transit_array_list[1]/100, start_state=2, n_steps=10**6)

    print('WT:', wt_prob)
    print('MT:', mt_prob)

dict_datasets = {'WT': np.array([0.2161, 0.2060, 0.2033, 0.2541, 0.2273, 0.2429]), 'MT': np.array([0.1934, 0.1827, 0.1783, 0.1908, 0.2088, 0.2287])}
# dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='simulated DZ localization result (duration 13 to 18)', colors=('#888888', '#CC6677'),
                     vmax=None, strip_plot=True, test='t-test', pvalue=True, figsize=(1, 2))

dict_datasets = {'WT': np.array([0.1074, 0.1173, 0.1159, 0.1140, 0.1079, 0.1111]), 'MT': np.array([0.1147, 0.1313, 0.1628, 0.1485, 0.1388, 0.1280])}
# dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='simulated dLZ localization result (duration 13 to 18)', colors=('#888888', '#CC6677'),
                     vmax=None, strip_plot=True, test='t-test', pvalue=True, figsize=(1, 2))



############################### Zone transitions of GCB per video ################################
entropies_all = {}
cluster_type = 'Zone_num'

transit_array_heatmaps={}
for cell_type in ['wt_B-cell', 'mt_B-cell']:

    cluster_size = np.unique(df[cluster_type]).size
    transit_array_list = []
    for video in np.unique(df['Video']):
        df_trans = df[(df['Type']==cell_type)&(df['Video']==video)].reset_index(drop=True)
        count = 0
        transit_array_temp = np.empty((cluster_size,cluster_size), dtype = 'object')
        for row in range(0,transit_array_temp.shape[0]):
            for col in range(0,transit_array_temp.shape[1]):
                transit_array_temp[row,col] = []

        for label in np.unique(df_trans['Label']):
            each_label = df_trans[df_trans['Label']==label].reset_index(drop=True)
            if each_label.shape[0] >= 2:
                count = count + 1
                for idx in range(each_label.shape[0] - 1):
                    row = each_label[cluster_type][idx]
                    col = each_label[cluster_type][idx + 1]
                    transit_array_temp[row, col].append(1)
        #print(cell_type, video, count)
        # if (cell_type == 'wt_B-cell') and (count<30) or (cell_type == 'mt_B-cell') and (count<60) :
        #     continue
        if (count<50) :
            continue

        print(cell_type, video, count)
        transit_array = np.empty((cluster_size,cluster_size), dtype = 'float')
        for row in range(0,transit_array_temp.shape[0]):
            for col in range(0,transit_array_temp.shape[1]):
                transit_array[row,col] = len(transit_array_temp[row,col])
        #print(transit_array)
        ############## Heatmap ##############
        transit_array_heatmap = 100 * transit_array / np.sum(transit_array, axis=1)[:, np.newaxis]
        transit_array_list.append(transit_array_heatmap)
        #print(transit_array_heatmap)
        # if cell_type == 'wt_B-cell':
        #     cutoff = 5
        # elif cell_type == 'mt_B-cell':
        #     cutoff = 9
        #zero_transition_clusters = np.sum(transit_array, axis=1) <= cutoff
        #transit_array_heatmap[zero_transition_clusters, :] = 0
    transit_array_list = np.array(transit_array_list)
    transit_array_heatmaps[cell_type] = transit_array_list


DZ_sLZ_2_DZs = {}
sLZ_2_DZs = {}
sLZ_2_sLZ_DZs = {}
sLZ_2_sLZ_dLZs = {}
sLZ_2_dLZs = {}
sLZ_dLZ_2_dLZs = {}

for cell_type, trans_matrices in transit_array_heatmaps.items():
    DZ_sLZ_2_DZ = trans_matrices[:, 1, 0]
    # for i in range(trans_matrices.shape[0]):
    #     trans_matrices[i, 1, :]
    #     print(i, np.sum(trans_matrices[i, 1, :]))

    DZ_sLZ_2_DZ = DZ_sLZ_2_DZ[~np.isnan(DZ_sLZ_2_DZ)]  # Remove NaN

    sLZ_2_DZ = trans_matrices[:, 2, 0]
    sLZ_2_DZ = sLZ_2_DZ[~np.isnan(sLZ_2_DZ)]  # Remove NaN

    sLZ_2_sLZ_DZ = trans_matrices[:, 2, 1]
    sLZ_2_sLZ_DZ = sLZ_2_sLZ_DZ[~np.isnan(sLZ_2_sLZ_DZ)]  # Remove NaN

    sLZ_2_sLZ_dLZ = trans_matrices[:, 2, 3]
    sLZ_2_sLZ_dLZ = sLZ_2_sLZ_dLZ[~np.isnan(sLZ_2_sLZ_dLZ)]  # Remove NaN

    sLZ_2_dLZ = trans_matrices[:, 2, 4]
    sLZ_2_dLZ = sLZ_2_dLZ[~np.isnan(sLZ_2_dLZ)]  # Remove NaN

    sLZ_dLZ_2_dLZ = trans_matrices[:, 3, 4]
    sLZ_dLZ_2_dLZ = sLZ_dLZ_2_dLZ[~np.isnan(sLZ_dLZ_2_dLZ)]  # Remove NaN

    DZ_sLZ_2_DZs[cell_type] = DZ_sLZ_2_DZ
    sLZ_2_DZs[cell_type] = sLZ_2_DZ
    sLZ_2_sLZ_DZs[cell_type] = sLZ_2_sLZ_DZ
    sLZ_2_sLZ_dLZs[cell_type] = sLZ_2_sLZ_dLZ
    sLZ_2_dLZs[cell_type] = sLZ_2_dLZ
    sLZ_dLZ_2_dLZs[cell_type] = sLZ_dLZ_2_dLZ


test = 't-test'
colors = ('#888888', '#CC6677')
# draw_custom_box_plot(sLZ_2_dLZs, path, file_name='test',
#                      strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1, 2))

for name, data in zip(['DZ-sLZ to DZ', 'sLZ to DZ', 'sLZ to DZ-sLZ', 'sLZ to sLZ-dLZ', 'sLZ to dLZ', 'sLZ-dLZ to dLZ'],
                [DZ_sLZ_2_DZs, sLZ_2_DZs, sLZ_2_sLZ_DZs, sLZ_2_sLZ_dLZs, sLZ_2_dLZs, sLZ_dLZ_2_dLZs]):
    draw_custom_bar_plot(data, path, file_name='test %s'%name,
                         strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1, 2))



def simulate_markov_chain(trans_matrix, start_state, n_steps):
    n_states = trans_matrix.shape[0]
    curr_state = start_state
    pi = np.array([0] * n_states)
    pi[start_state] = 1

    i = 0
    while i < n_steps:
        curr_state = np.random.choice(range(0, n_states), p=trans_matrix[curr_state])
        pi[curr_state] += 1
        i += 1
    #print("π = ", pi / steps)
    return pi / n_steps

wt_prob = simulate_markov_chain(transit_array_heatmaps[0]/100, start_state=2, n_steps=10**6)
mt_prob = simulate_markov_chain(transit_array_heatmaps[1]/100, start_state=2, n_steps=10**6)

############################### Zone transitions of GCB for two nodes ################################

max_duration = 26 # 16
DZ_sLZ_2_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
sLZ_2_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
sLZ_2_sLZ_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
sLZ_2_sLZ_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}
sLZ_2_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}
sLZ_dLZ_2_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}

for duration in range(2, max_duration+1):
    print('---Duration %s frame---'%duration)
    path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\traj_durations\transition\\'
    df = pd.read_parquet(path+'df_%s.parquet'%duration)


    entropies_all = {}
    cluster_type = 'Zone_num'

    transit_array_heatmaps={}
    DZ_sLZ_2_DZ_ps_temp, sLZ_2_DZ_ps_temp, sLZ_2_sLZ_DZ_ps_temp, sLZ_2_sLZ_dLZ_ps_temp, sLZ_2_dLZ_ps_temp, sLZ_dLZ_2_dLZ_ps_temp = {}, {}, {}, {}, {}, {}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:

        cluster_size = np.unique(df[cluster_type]).size
        transit_array_list = []

        df_trans = df[(df['Type']==cell_type)].reset_index(drop=True)

        count = 0
        transit_array_temp = np.empty((cluster_size,cluster_size), dtype = 'object')
        for row in range(0,transit_array_temp.shape[0]):
            for col in range(0,transit_array_temp.shape[1]):
                transit_array_temp[row,col] = []


        for label in np.unique(df_trans['Label']):
            each_label = df_trans[df_trans['Label']==label].reset_index(drop=True)
            simulated_each_label = df_to_shuffle[df_to_shuffle['Label']==label].reset_index(drop=True)
            # shuffled_label = df_to_shuffle[df_to_shuffle['Label']==label].reset_index(drop=True)
            if each_label.shape[0] >= 2:
                count = count + 1
                for idx in range(each_label.shape[0] - 1):
                    row = each_label[cluster_type][idx]
                    col = each_label[cluster_type][idx + 1]
                    transit_array_temp[row, col].append(1)

        print(cell_type, count)
        transit_array = np.empty((cluster_size,cluster_size), dtype='float')
        for row in range(0,transit_array_temp.shape[0]):
            for col in range(0,transit_array_temp.shape[1]):
                transit_array[row,col] = len(transit_array_temp[row,col])

        ############## Heatmap ##############
        transit_array_heatmap = 100 * transit_array / np.sum(transit_array, axis=1)[:, np.newaxis]
        transit_array_heatmaps[cell_type] = transit_array_heatmap

        DZ_sLZ_2_DZ_k_enrich, sLZ_2_DZ_k_enrich, sLZ_2_sLZ_DZ_k_enrich, sLZ_2_sLZ_dLZ_k_enrich, sLZ_2_dLZ_k_enrich, sLZ_dLZ_2_dLZ_k_enrich = 0, 0, 0, 0, 0, 0
        DZ_sLZ_2_DZ_k_deplete, sLZ_2_DZ_k_deplete, sLZ_2_sLZ_DZ_k_deplete, sLZ_2_sLZ_dLZ_k_deplete, sLZ_2_dLZ_k_deplete, sLZ_dLZ_2_dLZ_k_deplete = 0, 0, 0, 0, 0, 0
        iteration = 100

        for _ in tqdm(range(iteration)):

            #df_to_shuffle = df.copy()
            #df_to_shuffle[cluster_type] = np.random.permutation(df[cluster_type].values)  # Random shuffle zones
            shuffled = df[cluster_type].sample(frac=1).values
            df_to_shuffle = df.copy()
            df_to_shuffle[cluster_type] = shuffled

            simulated_transit_array_temp = np.empty((cluster_size, cluster_size), dtype='object')
            for row in range(0, simulated_transit_array_temp.shape[0]):
                for col in range(0, simulated_transit_array_temp.shape[1]):
                    simulated_transit_array_temp[row, col] = []

            for label in np.unique(df_trans['Label']):
                each_label = df_trans[df_trans['Label']==label].reset_index(drop=True)
                simulated_each_label = df_to_shuffle[df_to_shuffle['Label']==label].reset_index(drop=True)
                # shuffled_label = df_to_shuffle[df_to_shuffle['Label']==label].reset_index(drop=True)
                if each_label.shape[0] >= 2:
                    count = count + 1
                    for idx in range(each_label.shape[0] - 1):
                        simulated_row = simulated_each_label[cluster_type][idx]
                        simulated_col = simulated_each_label[cluster_type][idx + 1]
                        simulated_transit_array_temp[simulated_row, simulated_col].append(1)

            simulated_transit_array = np.empty((cluster_size, cluster_size), dtype='float')
            for row in range(0, simulated_transit_array_temp.shape[0]):
                for col in range(0, simulated_transit_array_temp.shape[1]):
                    simulated_transit_array[row, col] = len(simulated_transit_array_temp[row, col])

            simulated_transit_array_heatmap = 100 * simulated_transit_array / np.sum(simulated_transit_array, axis=1)[:, np.newaxis]


            DZ_sLZ_2_DZ_k_enrich += transit_array_heatmap[1, 0] > simulated_transit_array_heatmap[1, 0] # Count how many times it is enrichment
            DZ_sLZ_2_DZ_k_deplete += transit_array_heatmap[1, 0] < simulated_transit_array_heatmap[1, 0] # Count how many times it is depletion

            sLZ_2_DZ_k_enrich += transit_array_heatmap[2, 0] > simulated_transit_array_heatmap[2, 0] # Count how many times it is enrichment
            sLZ_2_DZ_k_deplete += transit_array_heatmap[2, 0] < simulated_transit_array_heatmap[2, 0]  # Count how many times it is depletion

            sLZ_2_sLZ_DZ_k_enrich += transit_array_heatmap[2, 1] > simulated_transit_array_heatmap[2, 1]  # Count how many times it is enrichment
            sLZ_2_sLZ_DZ_k_deplete += transit_array_heatmap[2, 1] < simulated_transit_array_heatmap[2, 1]  # Count how many times it is depletion

            sLZ_2_sLZ_dLZ_k_enrich += transit_array_heatmap[2, 3] > simulated_transit_array_heatmap[2, 3]  # Count how many times it is enrichment
            sLZ_2_sLZ_dLZ_k_deplete += transit_array_heatmap[2, 3] < simulated_transit_array_heatmap[2, 3]  # Count how many times it is depletion

            sLZ_2_dLZ_k_enrich += transit_array_heatmap[2, 4] > simulated_transit_array_heatmap[2, 4]  # Count how many times it is enrichment
            sLZ_2_dLZ_k_deplete += transit_array_heatmap[2, 4] < simulated_transit_array_heatmap[2, 4]  # Count how many times it is depletion

            sLZ_dLZ_2_dLZ_k_enrich += transit_array_heatmap[3, 4] > simulated_transit_array_heatmap[3, 4]  # Count how many times it is enrichment
            sLZ_dLZ_2_dLZ_k_deplete += transit_array_heatmap[3, 4] < simulated_transit_array_heatmap[3, 4]  # Count how many times it is depletion

        DZ_sLZ_2_DZ_p_enrich = 1 - DZ_sLZ_2_DZ_k_enrich / iteration
        DZ_sLZ_2_DZ_p_deplete = 1 - DZ_sLZ_2_DZ_k_deplete / iteration

        sLZ_2_DZ_p_enrich = 1 - sLZ_2_DZ_k_enrich / iteration
        sLZ_2_DZ_p_deplete = 1 - sLZ_2_DZ_k_deplete / iteration

        sLZ_2_sLZ_DZ_p_enrich = 1 - sLZ_2_sLZ_DZ_k_enrich / iteration
        sLZ_2_sLZ_DZ_p_deplete = 1 - sLZ_2_sLZ_DZ_k_deplete / iteration

        sLZ_2_sLZ_dLZ_p_enrich = 1 - sLZ_2_sLZ_dLZ_k_enrich / iteration
        sLZ_2_sLZ_dLZ_p_deplete = 1 - sLZ_2_sLZ_dLZ_k_deplete / iteration

        sLZ_2_dLZ_p_enrich = 1 - sLZ_2_dLZ_k_enrich / iteration
        sLZ_2_dLZ_p_deplete = 1 - sLZ_2_dLZ_k_deplete / iteration

        sLZ_dLZ_2_dLZ_p_enrich = 1 - sLZ_dLZ_2_dLZ_k_enrich / iteration
        sLZ_dLZ_2_dLZ_p_deplete = 1 - sLZ_dLZ_2_dLZ_k_deplete / iteration

        DZ_sLZ_2_DZ_ps_temp[cell_type] = (DZ_sLZ_2_DZ_p_enrich, DZ_sLZ_2_DZ_p_deplete)
        sLZ_2_DZ_ps_temp[cell_type] = (sLZ_2_DZ_p_enrich, sLZ_2_DZ_p_deplete)
        sLZ_2_sLZ_DZ_ps_temp[cell_type] = (sLZ_2_sLZ_DZ_p_enrich, sLZ_2_sLZ_DZ_p_deplete)
        sLZ_2_sLZ_dLZ_ps_temp[cell_type] = (sLZ_2_sLZ_dLZ_p_enrich, sLZ_2_sLZ_dLZ_p_deplete)
        sLZ_2_dLZ_ps_temp[cell_type] = (sLZ_2_dLZ_p_enrich, sLZ_2_dLZ_p_deplete)
        sLZ_dLZ_2_dLZ_ps_temp[cell_type] = (sLZ_dLZ_2_dLZ_p_enrich, sLZ_dLZ_2_dLZ_p_deplete)

    for cell_type, trans_matrices in transit_array_heatmaps.items():
        DZ_sLZ_2_DZ = trans_matrices[1, 0]
        sLZ_2_DZ = trans_matrices[2, 0]
        sLZ_2_sLZ_DZ = trans_matrices[2, 1]
        sLZ_2_sLZ_dLZ = trans_matrices[2, 3]
        sLZ_2_dLZ = trans_matrices[2, 4]
        sLZ_dLZ_2_dLZ = trans_matrices[3, 4]

        DZ_sLZ_2_DZs[cell_type].append(DZ_sLZ_2_DZ)
        sLZ_2_DZs[cell_type].append(sLZ_2_DZ)
        sLZ_2_sLZ_DZs[cell_type].append(sLZ_2_sLZ_DZ)
        sLZ_2_sLZ_dLZs[cell_type].append(sLZ_2_sLZ_dLZ)
        sLZ_2_dLZs[cell_type].append(sLZ_2_dLZ)
        sLZ_dLZ_2_dLZs[cell_type].append(sLZ_dLZ_2_dLZ)


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure2-3. Transition dynamics of wt and mt GCB\\'
custom_range = (2, max_duration)
stepsize=1
label_stepsize = 4
color_list=['#888888', '#CC6677']
marker_list=['.', '.']
xlabel = 'elapsed time (min)'
file_name = 'test'

for name, dataset in zip(['DZ-sLZ to DZ', 'sLZ to DZ', 'sLZ to DZ-sLZ', 'sLZ to sLZ-dLZ', 'sLZ to dLZ', 'sLZ-dLZ to dLZ'],
                [DZ_sLZ_2_DZs, sLZ_2_DZs, sLZ_2_sLZ_DZs, sLZ_2_sLZ_dLZs, sLZ_2_dLZs, sLZ_dLZ_2_dLZs]):

    font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(2,3))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    for idx, key in enumerate(dataset):
        sns.lineplot(data=dataset, x=np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), y=dataset[key],
                     label=key, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars', color=color_list[idx], alpha=0.9)
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
    ax.set_xticklabels(np.array( np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize)*0.5, dtype=int ),
                       rotation=0, rotation_mode='anchor', ha='center', fontsize=12, weight='normal')
    plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
               weight='normal')
    plt.yticks(fontsize=12, color='0.2', weight='normal')

    legend = plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
               loc='best')

    legend.remove()

    plt.savefig(path + 'zone all transition per duration_%s.png' % (name), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/' )
    plt.savefig(path + 'svg/zone all transition per duration_%s.svg' % (name), bbox_inches='tight')
    plt.clf()
    plt.close()








def run_analysis(df, condition='Type', transition_zone='Zone_num', iteration=100):
    def transition_matrix(df, transition_zone, label_groups, cluster_size):
        mat = np.zeros((cluster_size, cluster_size), dtype=int)
        for label, indices in label_groups.items():
            if len(indices) >= 2:
                zones = df[transition_zone].values[indices]
                for i in range(len(zones) - 1):
                    mat[zones[i], zones[i + 1]] += 1
        return mat

    results = {}

    for cell_type in np.unique(df[condition]):
        df_trans = df[df['Type'] == cell_type].reset_index(drop=True)
        cluster_size = df_trans[transition_zone].nunique()

        # Precompute label -> index mapping
        label_groups = {}
        for label in df_trans['Label'].unique():
            idx = df_trans[df_trans['Label'] == label].index.to_numpy()
            if len(idx) >= 2:
                label_groups[label] = idx

        # Actual transition matrix
        true_mat = transition_matrix(df_trans, transition_zone, label_groups, cluster_size)
        true_heatmap = 100 * true_mat / true_mat.sum(axis=1, keepdims=True)

        # Prepare for permutations
        enrich_counts = np.zeros((cluster_size, cluster_size), dtype=int)
        deplete_counts = np.zeros((cluster_size, cluster_size), dtype=int)

        shuffle_heatmaps = []
        for _ in tqdm(range(iteration), desc=f"Permuting {cell_type}"):
            shuffled = df[transition_zone].sample(frac=1).values
            df_shuffled = df.copy()
            df_shuffled[transition_zone] = shuffled

            shuffle_mat = transition_matrix(df_shuffled, transition_zone, label_groups, cluster_size)
            shuffle_mat[0, 3], shuffle_mat[0, 4], shuffle_mat[1, 3], shuffle_mat[1, 4], \
            shuffle_mat[3, 0], shuffle_mat[3, 1], shuffle_mat[4, 0], shuffle_mat[4, 1] = 0, 0, 0, 0, 0, 0, 0, 0
            shuffle_heatmap = 100 * shuffle_mat / shuffle_mat.sum(axis=1, keepdims=True)

            enrich_counts += true_heatmap > shuffle_heatmap
            deplete_counts += true_heatmap < shuffle_heatmap
            shuffle_heatmaps.append(shuffle_heatmap)

        avg_simulated_heatmap = np.sum(shuffle_heatmaps, axis=0)/iteration

        # Store enrichment/depletion p-values
        enrich_p = 1 - enrich_counts / iteration
        deplete_p = 1 - deplete_counts / iteration
        results[cell_type] = {
            'transition_heatmap': true_heatmap,
            'enrich_p': enrich_p,
            'deplete_p': deplete_p,
            'simulated_transition_heatmap':avg_simulated_heatmap
        }

    return results






DZ_sLZ_2_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
sLZ_2_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
sLZ_2_sLZ_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
sLZ_2_sLZ_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}
sLZ_2_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}
sLZ_dLZ_2_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}

simulated_DZ_sLZ_2_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
simulated_sLZ_2_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
simulated_sLZ_2_sLZ_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
simulated_sLZ_2_sLZ_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}
simulated_sLZ_2_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}
simulated_sLZ_dLZ_2_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}

p_DZ_sLZ_2_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
p_sLZ_2_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
p_sLZ_2_sLZ_DZs = {'wt_B-cell':[], 'mt_B-cell':[]}
p_sLZ_2_sLZ_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}
p_sLZ_2_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}
p_sLZ_dLZ_2_dLZs = {'wt_B-cell':[], 'mt_B-cell':[]}


for name, p_data in zip(['DZ-sLZ to DZ', 'sLZ to DZ', 'sLZ to DZ-sLZ', 'sLZ to sLZ-dLZ', 'sLZ to dLZ', 'sLZ-dLZ to dLZ'],
                                        [p_DZ_sLZ_2_DZs, p_sLZ_2_DZs, p_sLZ_2_sLZ_DZs, p_sLZ_2_sLZ_dLZs, p_sLZ_2_dLZs, p_sLZ_dLZ_2_dLZs]):
    print('\n--- %s ---' % name)
    for cell_type, values in p_data.items():
        print('--- %s ---'%cell_type)

        for idx, value in enumerate(values):
            t = (idx+2)/2
            p_enrich, p_deplete = value
            if p_enrich <= 0.05:
                print(t, cell_type, 'enriched')
            if p_deplete <= 0.05:
                print(t, cell_type, 'depleted')


max_duration=26
for duration in range(2, max_duration+1):
    print('---Duration %s frame---'%duration)
    path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\traj_durations\transition\\'
    df = pd.read_parquet(path+'df_%s.parquet'%duration)

    results = run_analysis(df, condition='Type', transition_zone='Zone_num', iteration=1000)

    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        heatmap_wt = results[cell_type]['transition_heatmap']
        simulated_heatmap_wt = results[cell_type]['simulated_transition_heatmap']

        DZ_sLZ_2_DZs[cell_type].append(heatmap_wt[1, 0])
        sLZ_2_DZs[cell_type].append(heatmap_wt[2, 0])
        sLZ_2_sLZ_DZs[cell_type].append(heatmap_wt[2, 1])
        sLZ_2_sLZ_dLZs[cell_type].append(heatmap_wt[2, 3])
        sLZ_2_dLZs[cell_type].append(heatmap_wt[2, 4])
        sLZ_dLZ_2_dLZs[cell_type].append(heatmap_wt[3, 4])

        simulated_DZ_sLZ_2_DZs[cell_type].append(simulated_heatmap_wt[1, 0])
        simulated_sLZ_2_DZs[cell_type].append(simulated_heatmap_wt[2, 0])
        simulated_sLZ_2_sLZ_DZs[cell_type].append(simulated_heatmap_wt[2, 1])
        simulated_sLZ_2_sLZ_dLZs[cell_type].append(simulated_heatmap_wt[2, 3])
        simulated_sLZ_2_dLZs[cell_type].append(simulated_heatmap_wt[2, 4])
        simulated_sLZ_dLZ_2_dLZs[cell_type].append(simulated_heatmap_wt[3, 4])

        p_DZ_sLZ_2_DZ = results[cell_type]['enrich_p'][1, 0], results[cell_type]['deplete_p'][1, 0]
        p_sLZ_2_DZ = results[cell_type]['enrich_p'][2, 0], results[cell_type]['deplete_p'][2, 0]
        p_DZ_sLZ_2_sLZ_DZ = results[cell_type]['enrich_p'][2, 1], results[cell_type]['deplete_p'][2, 1]
        p_DZ_sLZ_2_sLZ_dLZ = results[cell_type]['enrich_p'][2, 3], results[cell_type]['deplete_p'][2, 3]
        p_DZ_sLZ_2_dLZ = results[cell_type]['enrich_p'][2, 4], results[cell_type]['deplete_p'][2, 4]
        p_DZ_sLZ_dLZ_2_dLZ = results[cell_type]['enrich_p'][3, 4], results[cell_type]['deplete_p'][3, 4]

        p_DZ_sLZ_2_DZs[cell_type].append(p_DZ_sLZ_2_DZ)
        p_sLZ_2_DZs[cell_type].append(p_sLZ_2_DZ)
        p_sLZ_2_sLZ_DZs[cell_type].append(p_DZ_sLZ_2_sLZ_DZ)
        p_sLZ_2_sLZ_dLZs[cell_type].append(p_DZ_sLZ_2_sLZ_dLZ)
        p_sLZ_2_dLZs[cell_type].append(p_DZ_sLZ_2_dLZ)
        p_sLZ_dLZ_2_dLZs[cell_type].append(p_DZ_sLZ_dLZ_2_dLZ)



path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure2-3. Transition dynamics of wt and mt GCB\\'
custom_range = (2, max_duration)
stepsize=1
label_stepsize = 4
color_list=['#888888', '#CC6677']
marker_list=['.', '.']
xlabel = 'elapsed time (min)'

for name, dataset, simul_dataset in zip(['DZ-sLZ to DZ', 'sLZ to DZ', 'sLZ to DZ-sLZ', 'sLZ to sLZ-dLZ', 'sLZ to dLZ', 'sLZ-dLZ to dLZ'],
                                        [DZ_sLZ_2_DZs, sLZ_2_DZs, sLZ_2_sLZ_DZs, sLZ_2_sLZ_dLZs, sLZ_2_dLZs, sLZ_dLZ_2_dLZs],
                                        [simulated_DZ_sLZ_2_DZs, simulated_sLZ_2_DZs, simulated_sLZ_2_sLZ_DZs,
                                         simulated_sLZ_2_sLZ_dLZs, simulated_sLZ_2_dLZs, simulated_sLZ_dLZ_2_dLZs],
                                        ):

    font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(2,3))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    for idx, key in enumerate(dataset):
        sns.lineplot(data=dataset, x=np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), y=dataset[key],
                     label=key, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars', color=color_list[idx], alpha=0.9)
        # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
        #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

    sns.lineplot(data=simul_dataset, x=np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), y=simul_dataset[key],
                 lw=1, linestyle='--', markersize=8, color='0.2', alpha=0.9)
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
    ax.set_xticklabels(np.array( np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize)*0.5, dtype=int ),
                       rotation=0, rotation_mode='anchor', ha='center', fontsize=12, weight='normal')
    plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
               weight='normal')
    plt.yticks(fontsize=12, color='0.2', weight='normal')

    legend = plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
               loc='best')

    legend.remove()

    plt.savefig(path + 'zone all transition with simulated per duration %s.png' % (name), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/' )
    plt.savefig(path + 'svg/zone all transition with simulated per duration %s.svg' % (name), bbox_inches='tight')
    plt.clf()
    plt.close()


for name, dataset, simul_dataset in zip(['DZ-sLZ to DZ', 'sLZ to DZ', 'sLZ to DZ-sLZ', 'sLZ to sLZ-dLZ', 'sLZ to dLZ', 'sLZ-dLZ to dLZ'],
                                        [DZ_sLZ_2_DZs, sLZ_2_DZs, sLZ_2_sLZ_DZs, sLZ_2_sLZ_dLZs, sLZ_2_dLZs, sLZ_dLZ_2_dLZs],
                                        [simulated_DZ_sLZ_2_DZs, simulated_sLZ_2_DZs, simulated_sLZ_2_sLZ_DZs,
                                         simulated_sLZ_2_sLZ_dLZs, simulated_sLZ_2_dLZs, simulated_sLZ_dLZ_2_dLZs],
                                        ):
    test = 'mann-whitney'
    colors = ('#888888', '#CC6677')
    draw_custom_bar_plot(dataset, path, file_name='bar plot %s' % name,
                         strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1, 2))