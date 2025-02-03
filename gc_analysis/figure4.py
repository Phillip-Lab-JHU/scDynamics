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
"""Generates Data for Figure4. GCB - Tfh - FDC interaction"""

from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import DistanceSignal

############################### Prepare long_traj_duration dataset ################################

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

df_duration = df_duration[df_duration['Type']!='T-cell'].reset_index(drop=True)
df_duration = get_instant_movements_variable_duration(df_duration, frame_name='Time_span', time_unit=0.5,
                                                      feature_name=['Position X', 'Position Y', 'Position Z'])

df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=FDC')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=T-cell')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_DZ')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_LZ')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_FDC_core')

#df_duration = df_duration[(df_duration['pseudo_frame']!=0)&(df_duration['pseudo_frame']!=1)].reset_index(drop=True)

df_duration = df_duration.replace({'Type': {'wt B-cell': 'wt_B-cell', 'mt B-cell': 'mt_B-cell'}})

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure4. Tri-interaction\\'

############################### Prepare Persistent Contact dataset ################################
from itertools import groupby

min_duration = 15
test = 'wilcoxon-ranksum'

other_features = ['Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'Zone', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC',
            'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell', 'instant_speed', 'instant_angle',
                  'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC', 'diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell',
                  'diff_Distance_to_DZ', 'diff_Distance_to_LZ', 'diff_Distance_to_FDC_core']
#traj = df_partial[(df_partial['TrackID']==1000000129)&(df_partial['Video']=='4-Good-D10-C1-ZT4-40-127-FOV230-256px_Statistics')].reset_index(drop=True)
df_both_PI = pd.DataFrame()
df_FDC_PI = pd.DataFrame()
df_Tfh_PI = pd.DataFrame()
for cell_type in ['wt_B-cell', 'mt_B-cell']:
    df_partial = df_duration[df_duration['Type'] == cell_type].reset_index(drop=True)
    befores = []
    durings = []
    afters = []

    datas = []
    df_temporal = pd.DataFrame()

    for i in range(0, df_partial.shape[0]):
        if (i == 0) or (i == duration + i0):
            duration = df_partial['Time_span'][i]
            traj = df_partial[i: duration + i].reset_index(drop=True)
            i0 = i

            #traj = traj[(traj['pseudo_frame'] != 0) & (traj['pseudo_frame'] != 1)].reset_index(drop=True)

            FDC_interaction_profile = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC'] > 0
            FDC_interaction_profile = FDC_interaction_profile * 1  # Binary interaction profile

            Tfh_interaction_profile = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell'] > 0
            Tfh_interaction_profile = Tfh_interaction_profile * 1  # Binary interaction profile

            FDC_elements = []
            FDC_indexes = []
            FDC_idx0 = 0
            for FDC_element, FDC_group in groupby(FDC_interaction_profile) :
                #print((FDC_element, FDC_group), (Tfh_element, Tfh_group) )
                # Groups consistent values
                # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
                FDC_element_list = list(FDC_group)
                FDC_idx1 = len(FDC_element_list) + FDC_idx0
                FDC_idx_list = list(range(FDC_idx0, FDC_idx1))
                FDC_idx0 = FDC_idx1
                FDC_elements.append(FDC_element_list)
                FDC_indexes.append(FDC_idx_list)

            Tfh_elements = []
            Tfh_indexes = []
            Tfh_idx0 = 0
            for Tfh_element, Tfh_group in groupby(Tfh_interaction_profile) :
                #print((FDC_element, FDC_group), (Tfh_element, Tfh_group) )
                # Groups consistent values
                # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
                Tfh_element_list = list(Tfh_group)
                Tfh_idx1 = len(Tfh_element_list) + Tfh_idx0
                Tfh_idx_list = list(range(Tfh_idx0, Tfh_idx1))
                Tfh_idx0 = Tfh_idx1
                Tfh_elements.append(Tfh_element_list)
                Tfh_indexes.append(Tfh_idx_list)

            # for (FDC_element, FDC_group), (Tfh_element, Tfh_group) in zip( groupby(FDC_interaction_profile), groupby(Tfh_interaction_profile) ):
            #     #print((FDC_element, FDC_group), (Tfh_element, Tfh_group) )
            #     # Groups consistent values
            #     # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
            #     FDC_element_list = list(FDC_group)
            #     FDC_idx1 = len(FDC_element_list) + FDC_idx0
            #     FDC_idx_list = list(range(FDC_idx0, FDC_idx1))
            #     FDC_idx0 = FDC_idx1
            #     FDC_elements.append(FDC_element_list)
            #     FDC_indexes.append(FDC_idx_list)
            #
            #     Tfh_element_list = list(Tfh_group)
            #     Tfh_idx1 = len(Tfh_element_list) + Tfh_idx0
            #     Tfh_idx_list = list(range(Tfh_idx0, Tfh_idx1))
            #     Tfh_idx0 = Tfh_idx1
            #     Tfh_elements.append(Tfh_element_list)
            #     Tfh_indexes.append(Tfh_idx_list)

            FDC_check_minduration = np.array([np.sum(element) for element in FDC_elements]) >= min_duration
            Tfh_check_minduration = np.array([np.sum(element) for element in Tfh_elements]) >= min_duration
            # list of booleans whether each element have more than 20 persistent contact
            if (any(FDC_check_minduration) == False):
                continue
            elif (np.sum(FDC_check_minduration)==1):
                FDC_PI_first_frame = np.array(FDC_indexes)[FDC_check_minduration][0][0]
                FDC_PI_last_frame = np.array(FDC_indexes)[FDC_check_minduration][0][-1]
                FDC_duration = len(np.array(FDC_indexes)[FDC_check_minduration][0])
                df_PI_temp = pd.DataFrame()
                df_PI_temp['FDC_first'] = [FDC_PI_first_frame]
                df_PI_temp['FDC_last'] = [FDC_PI_last_frame]
                df_PI_temp['FDC_duration'] = [FDC_duration]
                df_PI_temp['Type'] = [cell_type]
                df_PI_temp['Exp'] = [traj['Exp'][0]]
                df_PI_temp['Video'] = [traj['Video'][0]]
                df_PI_temp['TrackID'] = [traj['TrackID'][0]]
                df_PI_temp['Time'] = [traj['Time'][0]]
                df_PI_temp[other_features] = np.mean(traj[other_features])

                trans_matrix = compute_transition_matrix(traj['Zone'], n_states=3)
                norm_trans_matrix = trans_matrix / np.sum(trans_matrix, axis=1)[:, np.newaxis]
                norm_trans_matrix[np.isnan(norm_trans_matrix)] = 0
                df_PI_temp['dz_residence'] = norm_trans_matrix[0, 0]
                df_PI_temp['slz_residence'] = norm_trans_matrix[1, 1]
                df_PI_temp['dlz_residence'] = norm_trans_matrix[2, 2]
                df_PI_temp['dz_slz'] = norm_trans_matrix[0, 1]
                df_PI_temp['slz_dz'] = norm_trans_matrix[1, 0]
                df_PI_temp['slz_dlz'] = norm_trans_matrix[1, 2]
                df_PI_temp['dlz_slz'] = norm_trans_matrix[2, 1]

                df_FDC_PI = pd.concat([df_FDC_PI, df_PI_temp], axis=0, ignore_index=True)

            if (any(Tfh_check_minduration) == False):
                continue
            elif (np.sum(Tfh_check_minduration)==1):
                Tfh_PI_first_frame = np.array(Tfh_indexes)[Tfh_check_minduration][0][0]
                Tfh_PI_last_frame = np.array(Tfh_indexes)[Tfh_check_minduration][0][-1]
                Tfh_duration = len(np.array(Tfh_indexes)[Tfh_check_minduration][0])
                df_PI_temp = pd.DataFrame()
                df_PI_temp['Tfh_first'] = [Tfh_PI_first_frame]
                df_PI_temp['Tfh_last'] = [Tfh_PI_last_frame]
                df_PI_temp['Tfh_duration'] = [Tfh_duration]
                df_PI_temp['Type'] = [cell_type]
                df_PI_temp['Exp'] = [traj['Exp'][0]]
                df_PI_temp['Video'] = [traj['Video'][0]]
                df_PI_temp['TrackID'] = [traj['TrackID'][0]]
                df_PI_temp['Time'] = [traj['Time'][0]]
                df_PI_temp[other_features] = np.mean(traj[other_features])

                trans_matrix = compute_transition_matrix(traj['Zone'], n_states=3)
                norm_trans_matrix = trans_matrix / np.sum(trans_matrix, axis=1)[:, np.newaxis]
                norm_trans_matrix[np.isnan(norm_trans_matrix)] = 0
                df_PI_temp['dz_residence'] = norm_trans_matrix[0, 0]
                df_PI_temp['slz_residence'] = norm_trans_matrix[1, 1]
                df_PI_temp['dlz_residence'] = norm_trans_matrix[2, 2]
                df_PI_temp['dz_slz'] = norm_trans_matrix[0, 1]
                df_PI_temp['slz_dz'] = norm_trans_matrix[1, 0]
                df_PI_temp['slz_dlz'] = norm_trans_matrix[1, 2]
                df_PI_temp['dlz_slz'] = norm_trans_matrix[2, 1]

                df_Tfh_PI = pd.concat([df_Tfh_PI, df_PI_temp], axis=0, ignore_index=True)

            if (any(FDC_check_minduration) == False) or (any(Tfh_check_minduration) == False):  # false -> No 20 persistent contact
                continue
            elif (np.sum(FDC_check_minduration)==1) and (np.sum(Tfh_check_minduration)==1): # only one segment that is 20 persistent contact

                FDC_PI_first_frame = np.array(FDC_indexes)[FDC_check_minduration][0][0]
                Tfh_PI_first_frame = np.array(Tfh_indexes)[Tfh_check_minduration][0][0]

                FDC_PI_last_frame = np.array(FDC_indexes)[FDC_check_minduration][0][-1]
                Tfh_PI_last_frame = np.array(Tfh_indexes)[Tfh_check_minduration][0][-1]

                FDC_duration = len(np.array(FDC_indexes)[FDC_check_minduration][0])
                Tfh_duration = len(np.array(Tfh_indexes)[Tfh_check_minduration][0])

                df_PI_temp = pd.DataFrame()
                df_PI_temp['FDC_first'] = [FDC_PI_first_frame]
                df_PI_temp['FDC_last'] = [FDC_PI_last_frame]
                df_PI_temp['FDC_duration'] = [FDC_duration]
                df_PI_temp['Tfh_first'] = [Tfh_PI_first_frame]
                df_PI_temp['Tfh_last'] = [Tfh_PI_last_frame]
                df_PI_temp['Tfh_duration'] = [Tfh_duration]
                df_PI_temp['Type'] = [cell_type]
                df_PI_temp['Exp'] = [traj['Exp'][0]]
                df_PI_temp['Video'] = [traj['Video'][0]]
                df_PI_temp['TrackID'] = [traj['TrackID'][0]]
                df_PI_temp['Time'] = [traj['Time'][0]]
                df_PI_temp[other_features] = np.mean(traj[other_features])

                trans_matrix = compute_transition_matrix(traj['Zone'], n_states=3)
                norm_trans_matrix = trans_matrix / np.sum(trans_matrix, axis=1)[:, np.newaxis]
                norm_trans_matrix[np.isnan(norm_trans_matrix)] = 0
                df_PI_temp['dz_residence'] = norm_trans_matrix[0, 0]
                df_PI_temp['slz_residence'] = norm_trans_matrix[1, 1]
                df_PI_temp['dlz_residence'] = norm_trans_matrix[2, 2]
                df_PI_temp['dz_slz'] = norm_trans_matrix[0, 1]
                df_PI_temp['slz_dz'] = norm_trans_matrix[1, 0]
                df_PI_temp['slz_dlz'] = norm_trans_matrix[1, 2]
                df_PI_temp['dlz_slz'] = norm_trans_matrix[2, 1]

                print(cell_type, traj['Exp'][0], traj['Video'][0], traj['TrackID'][0])

                df_both_PI = pd.concat([df_both_PI, df_PI_temp], axis=0, ignore_index=True)

for df_temp in [df_FDC_PI, df_Tfh_PI, df_both_PI]:

    df_temp.loc[(df_temp['Zone'] < 0.4) & (df_temp['Zone'] >= 0), 'Zone1'] = 'DZ'
    df_temp.loc[(df_temp['Zone'] < 0.8) & (df_temp['Zone'] >= 0.4), 'Zone1'] = 'DZ-sLZ'
    df_temp.loc[(df_temp['Zone'] < 1.2) & (df_temp['Zone'] >= 0.8), 'Zone1'] = 'sLZ'
    df_temp.loc[(df_temp['Zone'] < 1.6) & (df_temp['Zone'] >= 1.2), 'Zone1'] = 'sLZ-dLZ'
    df_temp.loc[(df_temp['Zone'] <= 2) & (df_temp['Zone'] >= 1.6), 'Zone1'] = 'dLZ'

############################### FDC PC Interaction analysis ################################
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
for cell_type in np.unique(df_FDC_PI['Type']):
    df_part = df_FDC_PI[df_FDC_PI['Type']==cell_type].reset_index(drop=True)
    #df_part['FDC_duration']
    dict_datasets_temp[cell_type] = df_part['FDC_duration'].values

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_bar_plot(dict_datasets, path, file_name='FDC persistent contacts',
                         strip_plot=True, colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))


if not os.path.isdir(path + 'FDC_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'FDC_PC/')
if not os.path.isdir(path + 'svg/FDC_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/FDC_PC/')


feature_list = ['Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'Zone', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC',
            'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell', 'instant_speed', 'instant_angle',
                  'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC', 'diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell',
                  'diff_Distance_to_DZ', 'diff_Distance_to_LZ', 'diff_Distance_to_FDC_core', 'FDC_duration',
                  'dz_residence', 'slz_residence', 'dlz_residence', 'dz_slz', 'slz_dz', 'slz_dlz', 'dlz_slz']

for feature in feature_list:
    replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
    dict_datasets_temp = {}
    for cell_type in np.unique(df_FDC_PI['Type']):
        df_part = df_FDC_PI[df_FDC_PI['Type']==cell_type].reset_index(drop=True)
        #df_part['FDC_duration']
        dict_datasets_temp[cell_type] = df_part[feature].values

    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
    dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}

    draw_custom_violin_plot(dict_datasets, path, file_name='FDC_PC/FDC %s'%feature,
                             colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))



if not os.path.isdir(path + 'zone FDC_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'zone FDC_PC/')
if not os.path.isdir(path + 'svg/zone FDC_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/zone FDC_PC/')

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for zone in ['DZ','sLZ', 'dLZ']:
            data = df_FDC_PI[(df_FDC_PI['Zone1'] == zone) & (df_FDC_PI[condition_name] == cell_type)][feature_name]
            dataset[cell_type + '_' + str(zone)] = np.array(data)

    values = flatten_nested_dict(dataset)
    if np.isnan(values).any() == True:  # Check at least one nan
        continue
    elif np.isfinite(values).all() == False:  # Check everything is not inf
        continue

    rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_sLZ': 'wt GCB sLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
                   'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_sLZ': 'mt GCB sLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ', }
    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    # draw_custom_violin_plot(dataset_renamed, path + 'Zone int feature violin plot/', file_name=feature_name,
    #                         colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))

    draw_custom_bar_plot(dataset_renamed, path + 'zone FDC_PC/', file_name=feature_name,
                         strip_plot=False, colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'),
                         test='kruskal-wallis_dunn', pvalue=True, figsize=(2, 2))




############################### Tfh PC Interaction analysis ################################
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
for cell_type in np.unique(df_Tfh_PI['Type']):
    df_part = df_Tfh_PI[df_Tfh_PI['Type']==cell_type].reset_index(drop=True)
    #df_part['FDC_duration']
    dict_datasets_temp[cell_type] = df_part['Tfh_duration'].values

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_bar_plot(dict_datasets, path, file_name='Tfh persistent contacts',
                         strip_plot=True, colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))


feature_list = ['Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'Zone', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC',
            'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell', 'instant_speed', 'instant_angle',
                  'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC', 'diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell',
                  'diff_Distance_to_DZ', 'diff_Distance_to_LZ', 'diff_Distance_to_FDC_core', 'Tfh_duration',
                  'dz_residence', 'slz_residence', 'dlz_residence', 'dz_slz', 'slz_dz', 'slz_dlz', 'dlz_slz']

if not os.path.isdir(path + 'Tfh_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Tfh_PC/')
if not os.path.isdir(path + 'svg/Tfh_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/Tfh_PC/')
for feature in feature_list:
    replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
    dict_datasets_temp = {}
    for cell_type in np.unique(df_Tfh_PI['Type']):
        df_part = df_Tfh_PI[df_Tfh_PI['Type']==cell_type].reset_index(drop=True)
        #df_part['FDC_duration']
        dict_datasets_temp[cell_type] = df_part[feature].values

    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
    dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}

    draw_custom_bar_plot(dict_datasets, path, file_name='Tfh_PC/Tfh %s'%feature, strip_plot=True,
                             colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))


if not os.path.isdir(path + 'zone Tfh_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'zone Tfh_PC/')
if not os.path.isdir(path + 'svg/zone Tfh_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/zone Tfh_PC/')

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for zone in ['sLZ', 'dLZ']:
            data = df_Tfh_PI[(df_Tfh_PI['Zone1'] == zone) & (df_Tfh_PI[condition_name] == cell_type)][feature_name]
            dataset[cell_type + '_' + str(zone)] = np.array(data)


    rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_sLZ': 'wt GCB sLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
                   'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_sLZ': 'mt GCB sLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ', }
    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    # draw_custom_violin_plot(dataset_renamed, path + 'Zone int feature violin plot/', file_name=feature_name,
    #                         colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))
    try:
        draw_custom_bar_plot(dataset_renamed, path + 'zone Tfh_PC/', file_name=feature_name,
                             strip_plot=True, colors=('#888888', '#888888', '#CC6677', '#CC6677'),
                             test='kruskal-wallis_dunn', pvalue=True, figsize=(2, 2))
    except:
        pass
############################### FDC & Tfh PC Interaction analysis ################################
if not os.path.isdir(path + 'both_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'both_PC/')
if not os.path.isdir(path + 'svg/both_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/both_PC/')
for feature in other_features:
    replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
    dict_datasets_temp = {}
    for cell_type in np.unique(df_both_PI['Type']):
        df_part = df_both_PI[df_both_PI['Type']==cell_type].reset_index(drop=True)
        #df_part['FDC_duration']
        dict_datasets_temp[cell_type] = df_part[feature].values

    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
    dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}

    draw_custom_bar_plot(dict_datasets, path, file_name='both_PC/both %s'%feature, strip_plot=True,
                             colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))


############################### Interaction clock analysis ################################
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
for cell_type in np.unique(df_both_PI['Type']):
    df_part = df_both_PI[(df_both_PI['Type']==cell_type)].reset_index(drop=True)
    #df_part['FDC_duration']
    dict_datasets_temp[cell_type] = (df_part['Tfh_first'].values - df_part['FDC_first'].values) / 2

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_bar_plot(dict_datasets, path, file_name='Tfh PC first or FDC PC first',
                         strip_plot=True, colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))


replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
for cell_type in np.unique(df_both_PI['Type']):
    df_part = df_both_PI[(df_both_PI['Type']==cell_type)].reset_index(drop=True)
    #df_part['FDC_duration']
    dict_datasets_temp[cell_type] = (df_part['Tfh_last'].values - df_part['FDC_last'].values) / 2

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_bar_plot(dict_datasets, path, file_name='Tfh PC last or FDC PC last',
                         strip_plot=True, colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
for cell_type in np.unique(df_both_PI['Type']):
    df_part = df_both_PI[(df_both_PI['Type']==cell_type)&(df_both_PI['Zone']>=1.7)].reset_index(drop=True)
    #df_part['FDC_duration']
    dict_datasets_temp[cell_type] = (df_part['Tfh_first'].values - df_part['FDC_first'].values) / 2

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_bar_plot(dict_datasets, path, file_name='dLZ Tfh PC first or FDC PC first',
                         strip_plot=True, colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))


replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
for cell_type in np.unique(df_both_PI['Type']):
    df_part = df_both_PI[(df_both_PI['Type']==cell_type)&(df_both_PI['Zone']<=1.4)].reset_index(drop=True)
    #df_part['FDC_duration']
    dict_datasets_temp[cell_type] = (df_part['Tfh_first'].values - df_part['FDC_first'].values) / 2

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_bar_plot(dict_datasets, path, file_name='non dLZ Tfh PC first or FDC PC first',
                         strip_plot=True, colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))


replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
for cell_type in np.unique(df_both_PI['Type']):
    df_part = df_both_PI[(df_both_PI['Type']==cell_type)].reset_index(drop=True)
    #df_part['FDC_duration']
    #criteria = (df_part['Tfh_first'].values - df_part['FDC_first'].values) / 2

    value = (df_part['Tfh_first'].values - df_part['FDC_last'].values) / 2
    #nonzero = value[criteria>0]
    dict_datasets_temp[cell_type] = value

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_bar_plot(dict_datasets, path, file_name='Time delay between FDC PC finish and Tfh PC starts',
                         strip_plot=True, colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))


replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
for cell_type in np.unique(df_Tfh_PI['Type']):
    df_part = df_Tfh_PI[(df_Tfh_PI['Type']==cell_type)].reset_index(drop=True)
    #df_part['FDC_duration']
    dict_datasets_temp[cell_type] = (df_part['Time'].values - 1)/2

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_bar_plot(dict_datasets, path, file_name='test Tfh PC elapsed time', strip_plot=True,
                          colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))




replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
for cell_type in np.unique(df_both_PI['Type']):
    df_part = df_both_PI[(df_both_PI['Type']==cell_type)&(df_both_PI['Zone']<=1.4)].reset_index(drop=True)
    #df_part['FDC_duration']
    dict_datasets_temp[cell_type] = df_part['Tfh_first'].values - df_part['FDC_first'].values

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_bar_plot(dict_datasets, path, file_name='non dLZ Tfh PC first or FDC PC first',
                         strip_plot=True, colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))


df_both_PI_ = df_both_PI.copy()
df_both_PI_['int_clock'] = df_both_PI['Tfh_first'].values - df_both_PI['FDC_first'].values

coloc_features = ['Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core', 'Zone']

for idx in [0,1,2,3]:
    coloc_feature = coloc_features[idx]
    if idx == 0:
        xlabel = 'Distance to DZ (µm)'
        custsom_range = (0, 40)
        step = 4
    elif idx == 1:
        xlabel = 'Distance to sLZ (µm)'
        custsom_range = (0, 8)
        step = 1
    elif idx == 2:
        xlabel = 'Distance to dLZ (µm)'
        custsom_range = (0, 40)
        step = 4
    elif idx == 3:
        xlabel = 'Zone labels'
        custsom_range = (0.5, 2)
        step = 0.2
    draw_lineplot_by_custom_ranges(df_both_PI_, path, folder_name='int_clock %s'%coloc_feature, feature_list=['int_clock'],
                                   condition_name='Type', custsom_range=custsom_range, stepsize=step, range_feature=coloc_feature,
                                       color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label=xlabel,
                                   replace_keys=None, pvalue=False, test='mann-whitney')



replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
mt = []
wt = []
for video in np.unique(df_Tfh_PI['Video']):
    df_part = df_Tfh_PI[df_Tfh_PI['Video'] == video].reset_index(drop=True)
    if np.unique(df_part['Type']).size == 2:
        mt.append(df_part[df_part['Type']=='mt_B-cell']['Time'].values)
        wt.append(df_part[df_part['Type'] == 'wt_B-cell']['Time'].values)
        #dict_datasets_temp['mt_B-cell'] = df_part[df_part['Type']=='mt_B-cell']['Time'].values



replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
mt = []
wt = []
delays = []
for video in np.unique(df_FDC_PI['Video']):
    df_part = df_FDC_PI[df_FDC_PI['Video'] == video].reset_index(drop=True)
    if np.unique(df_part['Type']).size == 2:

        mt_time_dist = (df_part[df_part['Type' ]== 'mt_B-cell']['Time'].values - 1) / 2
        wt_time_dist = (df_part[df_part['Type'] == 'wt_B-cell']['Time'].values - 1) / 2

        mt.append(np.mean(mt_time_dist))
        wt.append(np.mean(wt_time_dist))

dict_datasets_temp['mt_B-cell'] = mt
dict_datasets_temp['wt_B-cell'] = wt

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_bar_plot(dict_datasets, path, file_name='Elapsed time for FDC PC',
                         strip_plot=True, colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))



replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
dict_datasets_temp = {}
mt = []
wt = []
delays = []
for video in np.unique(df_FDC_PI['Video']):
    df_part = df_FDC_PI[df_FDC_PI['Video'] == video].reset_index(drop=True)
    if np.unique(df_part['Type']).size == 2:

        mt_time_dist = (df_part[df_part['Type' ]== 'mt_B-cell']['Time'].values - 1) / 2
        wt_time_dist = (df_part[df_part['Type'] == 'wt_B-cell']['Time'].values - 1) / 2

        mt.append(mt_time_dist)
        wt.append(wt_time_dist)

mt_flatten = flatten_list_of_list(mt)
wt_flatten = flatten_list_of_list(wt)


dict_datasets_temp['mt_B-cell'] = mt_flatten
dict_datasets_temp['wt_B-cell'] = wt_flatten

new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_violin_plot(dict_datasets, path, file_name='Elapsed time for FDC PC distribution',
                        colors = ('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))


replace_keys = {'T-cell': 'Tfh', 'mt_B-cell': 'mt GCB', 'wt_B-cell': 'wt GCB'}
dict_datasets_temp = {}
mt = []
wt = []
delays = []
for video in np.unique(df_Tfh_PI['Video']):
    df_part = df_Tfh_PI[df_Tfh_PI['Video'] == video].reset_index(drop=True)
    if np.unique(df_part['Type']).size == 2:
        mt_time_dist = (df_part[df_part['Type'] == 'mt_B-cell']['Time'].values - 1) / 2
        wt_time_dist = (df_part[df_part['Type'] == 'wt_B-cell']['Time'].values - 1) / 2

        mt.append(mt_time_dist)
        wt.append(wt_time_dist)

mt_flatten = flatten_list_of_list(mt)
wt_flatten = flatten_list_of_list(wt)

dict_datasets_temp['mt_B-cell'] = mt_flatten
dict_datasets_temp['wt_B-cell'] = wt_flatten


new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
draw_custom_bar_plot(dict_datasets, path, file_name='Elapsed time for Tfh PC',
                     strip_plot=True, colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))


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


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure4. Tri-interaction\\'



#################### Define interaction dynamics clusters ###################

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
motility_data = motility_data.drop(columns_with_nan, axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)


from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)

# m = Morphodynamics(df_features, 'umap')
# cluster = m.get_cluster(pcs, n_clusters=9, cluster_type='kmeans')
#
# df_features_ = df_features.copy()
# df_features_['kmeans'] = cluster
# m.evaluate_umap(df_features_, pcs, path, n_neighbors_list=[10, 20, 30, 40, 50, 60],
#                       min_dist_list=[0.005, 0.01, 0.05, 0.1, 0.3, 0.5], condition_name ='kmeans')


from sklearn.cluster import KMeans
km = KMeans(n_clusters=9, random_state=0, init='k-means++')
# k-means++: Initialize centroids that are far away each other
kmeans_predicted = km.fit_predict(pcs)
cluster = pd.DataFrame(kmeans_predicted, columns=['int_kmeans'])
for i in np.unique(cluster):
    print(i, cluster[cluster['int_kmeans']==i].shape[0])


from umap import UMAP
__umap = UMAP(metric='euclidean', n_components=2, n_neighbors=20, min_dist=0.01, random_state=0)
pcs_array = __umap.fit_transform(pcs)
umap = pd.DataFrame(pcs_array, columns=['int_PC1', 'int_PC2'])


# umap = m.get_umap(pcs, 20, 0.01)
# #m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
# cluster = m.get_cluster(pcs, n_clusters=9, cluster_type='kmeans')
df_int = pd.concat([df, umap, cluster], axis=1)
df_int, replace_map = order_cluster_by_feature(df_int, cluster_name='int_kmeans', feature_name='FDC_contact_times')

for c in np.unique(df_int['int_kmeans']):
    print(c, df_int[df_int['int_kmeans']==c].shape[0])
#################### interaction space ###################

color_list = ['#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933'][:np.unique(df_int['int_kmeans']).size]

df_ = df_int.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
draw_umap_space(df_, path, file_name='int_kmeans_space', condition_name='int_kmeans', label_name='pseudo_Label',
                colors=color_list, dot_size=0.07, x_name='int_PC1', y_name='int_PC2')

draw_umap_space(df_, path, file_name='int_type_space', condition_name='Type', label_name='pseudo_Label',
                colors=('#CC6677', '#888888'), dot_size=0.07, x_name='int_PC1', y_name='int_PC2')

motility_data.columns.get_loc('quality_FDC_approach_times')
motility_data.columns.get_loc('Core_diff_distance_average')
feature_list = motility_data.columns[:90]
draw_space_feature_magnitude(df_, path, feature_list, dot_size=0.07, x_name='int_PC1', y_name='int_PC2', vmax=None)

draw_jointplot(xs='int_PC1', y='int_PC2', df=df_, path=path, file_name='jointplot_type', hue="Type", colors=('#CC6677', '#888888', ), hue_order=['mt GCB', 'wt GCB'],
               legend=False, fill=True, thresh=0.4, alpha=0.5, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

draw_jointplot(xs='int_PC1', y='int_PC2', df=df_, path=path, file_name='jointplot_kmeans', hue="int_kmeans", colors=color_list,
               legend=False, fill=True, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

draw_cluster_distribution_heatmap(df_, path, file_name='int_kmeans_type_heatmap', condition_name='Type', cluster_type='int_kmeans', vmax=None,
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(8,2))

draw_relative_cluster_distribution_heatmap(df_, path, file_name='relative_int_kmeans_type_heatmap', condition_name='Type', cluster_type='int_kmeans',
                                  annot=True, transpose=False, col_cluster=False, row_cluster=True, cmap=cmc.oslo_r, figsize=(6,2))

draw_cluster_distribution_heatmap(df_, path, file_name='IDC vs MC', condition_name='kmeans', cluster_type='int_kmeans', vmax=25,
                                  annot=False, col_cluster=False, row_cluster=True,figsize=(4,4))

############# All Kmeans distribution heatmap ###############
df_ = df_int.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
df_['Type_Zone'] = df_['Type'].astype(str) + ' ' + df_['Zone']
group_df = draw_cluster_distribution_heatmap(df_, path, file_name='all_kmeans_heatmap', condition_name='Type_Zone', annot=False, vmax=50,
                                  cluster_type='int_kmeans', col_cluster=False, figsize=(6,5))



############################### Locate GCB cells wrt FDC and T cells and project features ################################
videos = np.unique(df['Video'])
df_noA = df_duration[(df_duration['Video'] != videos[0])&(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])].reset_index(drop=True)
df_A = df_duration[(df_duration['Video'] == videos[0])|(df_duration['Video'] == videos[1])|(df_duration['Video'] == videos[2])].reset_index(drop=True)
df_wt = df_noA[df_noA['Type']=='wt_B-cell']
df_mt = df_noA[df_noA['Type']=='mt_B-cell']

feature_name = 'instant_speed'

for cell_type in ['mt_B-cell', 'wt_B-cell']:
    df_partial = df_noA[df_noA['Type']==cell_type].reset_index(drop=True)
    #df_partial = df_A[df_A['Type']=='wt_B-cell'].reset_index(drop=True)

    bin_num=50
    xmin = 0
    xmax = 20
    ymin = 0
    ymax = 20

    xgrid = np.linspace(xmin, xmax, bin_num) # (100, ) 1d x coordinate
    ygrid = np.linspace(ymin, ymax, bin_num) # (100, ) 1d y coordinate
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing = 'xy')

    lattice_temp = np.empty((bin_num,bin_num), dtype = 'object')
    for row in range(0,lattice_temp.shape[0]):
        for col in range(0,lattice_temp.shape[1]):
            lattice_temp[row,col] = [0]

    for idx, row in df_partial.iterrows():
        x = row['Shortest_Distance_to_Surfaces_Surfaces=FDC']
        y = row['Shortest_Distance_to_Surfaces_Surfaces=T-cell']

        residual = (x - Xgrid) ** 2 + (y - Ygrid) ** 2  # residual = (bin,bin) 2d array
        min_arr_index = np.unravel_index(np.argmin(residual), residual.shape)  # residual이 minimum인 array index 반환, (x,y)꼴

        feature = row[feature_name]
        lattice_temp[min_arr_index].append(feature)

    for row in range(0,lattice_temp.shape[0]):
        for col in range(0,lattice_temp.shape[1]):
            if len(lattice_temp[row,col]) > 1:
                lattice_temp[row,col].remove(0)

    lattice = np.empty((bin_num,bin_num))
    for row in range(0,lattice_temp.shape[0]):
        for col in range(0,lattice_temp.shape[1]):
            lattice[row, col] = np.mean(lattice_temp[row, col])

    from skimage.filters import gaussian
    gaussian_lattice = gaussian(lattice, sigma=1, preserve_range=True)

    plt.figure(figsize=(20,15))
    plt.imshow(gaussian_lattice, origin='lower', aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='coolwarm', vmax=7)
    plt.colorbar()
    plt.xlabel('Distance to FDC')
    plt.ylabel('Distance to T')
    plt.savefig(directory+'%s_%s_A_position wrt FDC & T.png'%(cell_type, feature_name), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()


############################### Locate GCB cells wrt FDC and T cells and project approach, departure features ################################
feature_name = 'instant_speed'
for move_type in ['approach', 'departure']:
    for cell_type in ['mt_B-cell', 'wt_B-cell']:
        df_partial = df_noA[(df_noA['Type']==cell_type)&(df_noA['instant_Shortest_Distance_to_Surfaces_Surfaces=FDC']==move_type)].reset_index(drop=True)
        #df_partial = df_A[(df_A['Type']=='wt_B-cell')&(df_A['instant_Shortest_Distance_to_Surfaces_Surfaces=FDC']=='approach')].reset_index(drop=True)

        bin_num=50
        xgrid = np.linspace(0, 20, bin_num) # (100, ) 1d x coordinate
        ygrid = np.linspace(0, 20, bin_num) # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing = 'xy')

        lattice_temp = np.empty((bin_num,bin_num), dtype = 'object')
        for row in range(0,lattice_temp.shape[0]):
            for col in range(0,lattice_temp.shape[1]):
                lattice_temp[row,col] = [0]

        for idx, row in df_partial.iterrows():
            x = row['Shortest_Distance_to_Surfaces_Surfaces=FDC']
            y = row['Shortest_Distance_to_Surfaces_Surfaces=T-cell']

            residual = (x - Xgrid) ** 2 + (y - Ygrid) ** 2  # residual = (bin,bin) 2d array
            min_arr_index = np.unravel_index(np.argmin(residual), residual.shape)  # residual이 minimum인 array index 반환, (x,y)꼴

            feature = row[feature_name]
            lattice_temp[min_arr_index].append(feature)

        for row in range(0,lattice_temp.shape[0]):
            for col in range(0,lattice_temp.shape[1]):
                if len(lattice_temp[row,col]) > 1:
                    lattice_temp[row,col].remove(0)

        lattice = np.empty((bin_num,bin_num))
        for row in range(0,lattice_temp.shape[0]):
            for col in range(0,lattice_temp.shape[1]):
                lattice[row, col] = np.mean(lattice_temp[row, col])

        from skimage.filters import gaussian
        gaussian_lattice = gaussian(lattice, sigma=1, preserve_range=True)

        plt.figure(figsize=(20,15))
        plt.imshow(gaussian_lattice, origin='lower', aspect='auto', extent=[0, 20, 0, 20], cmap='coolwarm', vmax=7)
        plt.colorbar()
        plt.xlabel('Distance to FDC')
        plt.ylabel('Distance to T')
        plt.savefig(directory+'%s_%s_%s_NoA_position wrt FDC & T.png'%(cell_type, move_type, feature_name), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()