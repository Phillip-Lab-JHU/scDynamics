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
"""Generates Data for Figure3. wt and MT interaction with Tfh"""

from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import DistanceSignal


############################### Motility of CD40L and mLT ################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis_test\feature_csvs\\'
df = pd.read_parquet(path+'GCB_with_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_with_inhibit_traj_duration_20.parquet')

df['Inhibition'] = df['Exp'].copy()
df['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)

df_ = df.copy()
df_['type_exp'] = df_['Type'].astype(str) + ' ' + df_['Exp'].astype(str)
for typ in np.unique(df_['type_exp']):
    print(typ, df_[df_['type_exp']==typ].shape[0])

df_ = df.copy()
df_['type_Inhibition'] = df_['Type'].astype(str) + ' ' + df_['Inhibition'].astype(str)
for typ in np.unique(df_['type_Inhibition']):
    print(typ, df_[df_['type_Inhibition']==typ].shape[0])


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\FigureX. Direct cauality btw motility and interaction\\'

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_ = df_[(df_['Inhibition']!='IgG')&(df_['Inhibition']!='mLT')].reset_index(drop=True)
df_['type_inhibition'] = df_['Type'].astype(str) + ' ' + df_['Inhibition'].astype(str)
draw_cluster_distribution_heatmap(df_, path, file_name='type_inhibition_kmeans_heatmap', condition_name='type_inhibition', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,2))

p_dict = permutation_test(df_, group_name='type_inhibition', class_name='kmeans', iteration=10000)


############################### Zone-dependence ################################

df.loc[(df['avg_zone'] < 0.4) & (df['avg_zone'] >= 0), 'Zone'] = 'DZ'
df.loc[(df['avg_zone'] < 0.8) & (df['avg_zone'] >= 0.4), 'Zone'] = 'DZ-sLZ'
df.loc[(df['avg_zone'] < 1.2) & (df['avg_zone'] >= 0.8), 'Zone'] = 'sLZ'
df.loc[(df['avg_zone'] < 1.6) & (df['avg_zone'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
df.loc[(df['avg_zone'] <= 2) & (df['avg_zone'] >= 1.6), 'Zone'] = 'dLZ'

df_ = df.copy()
df_['type_Inhibition_zone'] = df_['Type'].astype(str) + ' ' + df_['Inhibition'].astype(str)+ ' ' + df_['Zone'].astype(str)
for typ in np.unique(df_['type_Inhibition_zone']):
    print(typ, df_[df_['type_Inhibition_zone']==typ].shape[0])

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_ = df_[(df_['Inhibition']!='IgG')&(df_['Inhibition']!='mLT')].reset_index(drop=True)
df_['type_Inhibition_zone'] = df_['Type'].astype(str) + ' ' + df_['Inhibition'].astype(str)+ ' ' + df_['Zone'].astype(str)
group_df = draw_cluster_distribution_heatmap(df_, path, file_name='type_Inhibition_zone_kmeans_heatmap', condition_name='type_Inhibition_zone',
                                             annot=False,
                                             cluster_type='kmeans', col_cluster=False, cmap=cmc.bilbao_r, figsize=(6, 10))

for zone in np.unique(df['Zone']):
    df_part = df[df['Zone']==zone].reset_index(drop=True)
    df_part = df_part[(df_part['Inhibition'] != 'IgG') & (df_part['Inhibition'] != 'mLT')].reset_index(drop=True)
    df_part = df_part.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
    df_part['type_Inhibition_zone'] = df_part['Type'].astype(str) + ' ' + df_part['Inhibition'].astype(str)+ ' ' + df_part['Zone'].astype(str)
    group_df = draw_cluster_distribution_heatmap(df_part, path, file_name='type_Inhibition_zone_kmeans_heatmap_%s'%zone, condition_name='type_Inhibition_zone',
                                                 annot=False,
                                                 cluster_type='kmeans', col_cluster=False, cmap=cmc.bilbao_r, figsize=(4, 2))



#################################### BC analysis ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis_test\feature_csvs\\'
df = pd.read_parquet(path+'GCB_with_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_with_inhibit_traj_duration_20.parquet')
df_inhibit = df[(df['Exp']=='CD40L')|(df['Exp']=='IgG')].reset_index(drop=True)
df_duration_inhibit = df_duration[(df_duration['Exp']=='CD40L')|(df_duration['Exp']=='IgG')].reset_index(drop=True)


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GCB_no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_no_inhibit_traj_duration_20.parquet')

_, _, Zone_series = to_timeseries_fast(df_duration, duration=20, feature_name='Zone')

feature_list = ['dz_resident_times', 'dz_resident_persistences', 'slz_resident_times',
                'slz_resident_persistences', 'dlz_resident_times', 'dlz_resident_persistences', 'avg_zone']
from features.interaction import ZoneSignal
Zone_func = ZoneSignal(Zone_series)
df_zone = Zone_func.extract_features(feature_list)
df = pd.concat([df, df_zone], axis=1)



from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
df.columns.get_loc('morpho_displ_autocorr_3')
motility_data = df.iloc[:,8:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)
motility_data_inhibit = df_inhibit.iloc[:,8:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)


df.columns.get_loc('quality_FDC_approach_times')
df.columns.get_loc('dlz_resident_persistences')

colocalization_data = df.iloc[:,148:289].drop(['FDC_diff_distance_cov', 'T_diff_distance_cov',
                                         'Core_diff_distance_cov', 'LZ_diff_distance_cov', 'DZ_diff_distance_cov',
                                         'DZ_distance_autocorr_1', 'DZ_distance_autocorr_2', 'DZ_distance_autocorr_3',
                                         'DZ_diff_distance_autocorr_1', 'DZ_diff_distance_autocorr_2', 'DZ_diff_distance_autocorr_3',
                                         'LZ_diff_distance_autocorr_1', 'LZ_diff_distance_autocorr_2', 'LZ_diff_distance_autocorr_3',
                                         'Core_distance_autocorr_1', 'Core_distance_autocorr_2', 'Core_distance_autocorr_3',
                                         'Core_diff_distance_autocorr_1', 'Core_diff_distance_autocorr_2', 'Core_diff_distance_autocorr_3',
                                         'Core_distance_variance', 'Core_diff_distance_variance', 'DZ_distance_variance', 'DZ_diff_distance_variance',
                                         'FDC_diff_distance_variance', 'FDC_distance_variance', 'LZ_distance_variance', 'LZ_diff_distance_variance',
                                         'T_diff_distance_variance', 'T_distance_variance',
                                        'PC1', 'PC2', 'kmeans'
                                         ], axis=1)
colocalization_data_inhibit = df_inhibit.iloc[:,148:289].drop(['FDC_diff_distance_cov', 'T_diff_distance_cov',
                                         'Core_diff_distance_cov', 'LZ_diff_distance_cov', 'DZ_diff_distance_cov',
                                         'DZ_distance_autocorr_1', 'DZ_distance_autocorr_2', 'DZ_distance_autocorr_3',
                                         'DZ_diff_distance_autocorr_1', 'DZ_diff_distance_autocorr_2', 'DZ_diff_distance_autocorr_3',
                                         'LZ_diff_distance_autocorr_1', 'LZ_diff_distance_autocorr_2', 'LZ_diff_distance_autocorr_3',
                                         'Core_distance_autocorr_1', 'Core_distance_autocorr_2', 'Core_distance_autocorr_3',
                                         'Core_diff_distance_autocorr_1', 'Core_diff_distance_autocorr_2', 'Core_diff_distance_autocorr_3',
                                         'Core_distance_variance', 'Core_diff_distance_variance', 'DZ_distance_variance', 'DZ_diff_distance_variance',
                                         'FDC_diff_distance_variance', 'FDC_distance_variance', 'LZ_distance_variance', 'LZ_diff_distance_variance',
                                         'T_diff_distance_variance', 'T_distance_variance',
                                        'PC1', 'PC2', 'avg_zone'
                                         ], axis=1)
columns_with_nan = colocalization_data.columns[colocalization_data.isna().any()].tolist()
#columns_with_nan_inhibit = colocalization_data_inhibit.columns[colocalization_data_inhibit.isna().any()].tolist()
#merged_list = list(set(columns_with_nan) | set(columns_with_nan_inhibit))
colocalization_data = colocalization_data.drop(columns_with_nan, axis=1)
colocalization_data_inhibit = colocalization_data_inhibit.drop(columns_with_nan, axis=1)

input_data = pd.concat([motility_data, colocalization_data], axis=1)
input_data_inhibit = pd.concat([motility_data_inhibit, colocalization_data_inhibit], axis=1)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
input_data_scaled= pd.DataFrame(scaler.fit_transform( input_data ), columns=input_data.columns)
input_data_scaled_inhibit= pd.DataFrame(scaler.transform( input_data_inhibit ), columns=input_data_inhibit.columns)

from sklearn.decomposition import PCA
pca = PCA(0.75)
pcs = pca.fit_transform(input_data_scaled)
pcs_inhibit = pca.transform(input_data_scaled_inhibit)


from sklearn.cluster import KMeans
km = KMeans(n_clusters=8, random_state=0, init='k-means++')
# k-means++: Initialize centroids that are far away each other
kmeans_predicted = km.fit_predict(pcs)
cluster = pd.DataFrame(kmeans_predicted, columns=['beh_kmeans'])

kmeans_predicted_inhibit = km.predict(pcs_inhibit)
cluster_inhibit = pd.DataFrame(kmeans_predicted_inhibit, columns=['beh_kmeans'])

from umap import UMAP
__umap = UMAP(metric='euclidean', n_components=2, n_neighbors=30, min_dist=0.1, random_state=0)
pcs_array = __umap.fit_transform(pcs)
umap = pd.DataFrame(pcs_array, columns=['beh_PC1', 'beh_PC2'])

pcs_array_inhibit = __umap.transform(pcs_inhibit)
umap_inhibit = pd.DataFrame(pcs_array_inhibit, columns=['beh_PC1', 'beh_PC2'])

df = pd.concat([df, umap, cluster], axis=1)
df, replace_map = order_cluster_by_feature(df, cluster_name='beh_kmeans', feature_name='avg_speed')

df_inhibit = pd.concat([df_inhibit, umap_inhibit, cluster_inhibit], axis=1)
df_inhibit = df_inhibit.replace({'kmeans': replace_map})

df_with_inhibit = pd.concat([df, df_inhibit], axis=0).reset_index(drop=True)

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\FigureX. Direct cauality btw motility and interaction\\'
df_with_inhibit.to_parquet(path + 'Expanded_behavior.parquet')
df_with_inhibit.to_csv(path + 'Expanded_behavior.csv', index=False)


#################################### Basic analysis ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\FigureX. Direct cauality btw motility and interaction\\'
df = pd.read_parquet(path+'Expanded_behavior.parquet')

df['Inhibition'] = df['Exp'].copy()
df['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_ = df_[(df_['Inhibition']!='IgG')&(df_['Inhibition']!='mLT')].reset_index(drop=True)
df_['type_inhibition'] = df_['Type'].astype(str) + ' ' + df_['Inhibition'].astype(str)
draw_cluster_distribution_heatmap(df_, path, file_name='type_inhibition_BC_heatmap', condition_name='type_inhibition', vmax=24, annot=True,
                                  cluster_type='beh_kmeans', col_cluster=False, row_cluster=True, figsize=(5,2.5))

p_dict = permutation_test(df_, group_name='type_inhibition', class_name='beh_kmeans', iteration=10000)

for typ in np.unique(df_['type_inhibition']):
    print(typ, df_[df_['type_inhibition']==typ].shape[0])

############################### Prepare long_traj_duration dataset ################################

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df_duration = pd.read_parquet(path+'long_traj_duration_20.parquet')
df_duration = df_duration[df_duration['Exp_group']!='A'].reset_index(drop=True)

df_duration['Inhibition'] = df_duration['Exp'].copy()
df_duration['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)


videos = np.unique(df_duration['Video']) # Remove Group A, IgG and CD40L

df_duration = df_duration[df_duration['Type']!='T-cell'].reset_index(drop=True)
df_duration = get_instant_movements_variable_duration(df_duration, frame_name='Time_span', time_unit=0.5,
                                                      feature_name=['Position X', 'Position Y', 'Position Z'])

df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=FDC')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=T-cell')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_DZ')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_LZ')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_FDC_core')


type_series = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Type', equal_length=False, frame_name='Time')

type_list = []
for idx, types in type_series.items():
    type = np.unique(types)
    type_list.append(type[0])

type_list = np.array(type_list)
print(type_list)


#df_duration = df_duration[(df_duration['pseudo_frame']!=0)&(df_duration['pseudo_frame']!=1)].reset_index(drop=True)

df_duration = df_duration.replace({'Type': {'wt B-cell': 'wt_B-cell', 'mt B-cell': 'mt_B-cell'}})

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\FigureX. Direct cauality btw motility and interaction\\'

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
                FDC_PI_first_frame = np.array(FDC_indexes, dtype='object')[FDC_check_minduration][0][0]
                FDC_PI_last_frame = np.array(FDC_indexes, dtype='object')[FDC_check_minduration][0][-1]
                FDC_duration = len(np.array(FDC_indexes, dtype='object')[FDC_check_minduration][0])
                df_PI_temp = pd.DataFrame()
                df_PI_temp['FDC_first'] = [FDC_PI_first_frame]
                df_PI_temp['FDC_last'] = [FDC_PI_last_frame]
                df_PI_temp['FDC_duration'] = [FDC_duration]
                df_PI_temp['Type'] = [cell_type]
                df_PI_temp['Inhibition'] = [traj['Inhibition'][0]]
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
                Tfh_PI_first_frame = np.array(Tfh_indexes, dtype='object')[Tfh_check_minduration][0][0]
                Tfh_PI_last_frame = np.array(Tfh_indexes, dtype='object')[Tfh_check_minduration][0][-1]
                Tfh_duration = len(np.array(Tfh_indexes, dtype='object')[Tfh_check_minduration][0])
                df_PI_temp = pd.DataFrame()
                df_PI_temp['Tfh_first'] = [Tfh_PI_first_frame]
                df_PI_temp['Tfh_last'] = [Tfh_PI_last_frame]
                df_PI_temp['Tfh_duration'] = [Tfh_duration]
                df_PI_temp['Type'] = [cell_type]
                df_PI_temp['Inhibition'] = [traj['Inhibition'][0]]
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

                FDC_PI_first_frame = np.array(FDC_indexes, dtype='object')[FDC_check_minduration][0][0]
                Tfh_PI_first_frame = np.array(Tfh_indexes, dtype='object')[Tfh_check_minduration][0][0]

                FDC_PI_last_frame = np.array(FDC_indexes, dtype='object')[FDC_check_minduration][0][-1]
                Tfh_PI_last_frame = np.array(Tfh_indexes, dtype='object')[Tfh_check_minduration][0][-1]

                FDC_duration = len(np.array(FDC_indexes, dtype='object')[FDC_check_minduration][0])
                Tfh_duration = len(np.array(Tfh_indexes, dtype='object')[Tfh_check_minduration][0])

                df_PI_temp = pd.DataFrame()
                df_PI_temp['FDC_first'] = [FDC_PI_first_frame]
                df_PI_temp['FDC_last'] = [FDC_PI_last_frame]
                df_PI_temp['FDC_duration'] = [FDC_duration]
                df_PI_temp['Tfh_first'] = [Tfh_PI_first_frame]
                df_PI_temp['Tfh_last'] = [Tfh_PI_last_frame]
                df_PI_temp['Tfh_duration'] = [Tfh_duration]
                df_PI_temp['Type'] = [cell_type]
                df_PI_temp['Inhibition'] = [traj['Inhibition'][0]]
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
    replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
    dict_datasets_temp = {}
    for cell_type in np.unique(df_FDC_PI['Type']):
        for inhibition in ['Control', 'CD40L']:
            df_part = df_FDC_PI[(df_FDC_PI['Type']==cell_type)&(df_FDC_PI['Inhibition']==inhibition)].reset_index(drop=True)
            #df_part['FDC_duration']
            dict_datasets_temp[cell_type + ' ' + inhibition] = df_part[feature].values

    new_order = ['wt_B-cell Control', 'mt_B-cell Control', 'wt_B-cell CD40L', 'mt_B-cell CD40L']
    ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
    dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}

    draw_custom_violin_plot(dict_datasets, path, file_name='FDC_PC/FDC %s'%feature,
                             colors = ('#888888', '#CC6677', '#888888', '#CC6677'), test='kruskal-wallis_dunn', return_sig=True, pvalue=True, figsize=(2, 2))



if not os.path.isdir(path + 'zone FDC_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'zone FDC_PC/')
if not os.path.isdir(path + 'svg/zone FDC_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/zone FDC_PC/')

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for zone in ['DZ','sLZ', 'dLZ']:
            for inhibition in ['Control', 'CD40L']:
                data = df_FDC_PI[(df_FDC_PI['Zone1'] == zone) & (df_FDC_PI[condition_name] == cell_type)&(df_FDC_PI['Inhibition']==inhibition)][feature_name]
                dataset[cell_type + ' ' + str(zone) + ' ' + inhibition] = np.array(data)

    values = flatten_nested_dict(dataset)
    if np.isnan(values).any() == True:  # Check at least one nan
        continue
    elif np.isfinite(values).all() == False:  # Check everything is not inf
        continue

    rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
                   'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ', }
    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    # draw_custom_violin_plot(dataset_renamed, path + 'Zone int feature violin plot/', file_name=feature_name,
    #                         colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))

    draw_custom_bar_plot(dataset_renamed, path + 'zone FDC_PC/', file_name=feature_name,
                         strip_plot=False, colors=('#888888', '#888888', '#888888', '#888888', '#888888', '#888888',
                                                   '#CC6677', '#CC6677', '#CC6677', '#CC6677', '#CC6677', '#CC6677'),
                         return_sig=True,
                         test='kruskal-wallis_dunn', pvalue=True, figsize=(4, 2))




############################### Tfh PC Interaction analysis ################################

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
    replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
    dict_datasets_temp = {}
    for cell_type in np.unique(df_Tfh_PI['Type']):
        for inhibition in ['Control', 'CD40L']:
            df_part = df_Tfh_PI[(df_Tfh_PI['Type']==cell_type)&(df_Tfh_PI['Inhibition']==inhibition)].reset_index(drop=True)
            #df_part['FDC_duration']
            dict_datasets_temp[cell_type + ' ' + inhibition] = df_part[feature].values

    new_order = ['wt_B-cell Control', 'mt_B-cell Control', 'wt_B-cell CD40L', 'mt_B-cell CD40L']
    ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
    dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
    del dict_datasets['mt_B-cell CD40L']
    draw_custom_violin_plot(dict_datasets, path, file_name='Tfh_PC/Tfh %s'%feature,
                             colors = ('#888888', '#CC6677', '#888888', '#CC6677'), test='kruskal-wallis_dunn', return_sig=True, pvalue=True, figsize=(2, 2))


[print(key, np.array(value).size) for key, value in dict_datasets.items()]



if not os.path.isdir(path + 'zone Tfh_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'zone Tfh_PC/')
if not os.path.isdir(path + 'svg/zone Tfh_PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/zone Tfh_PC/')

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for zone in ['sLZ', 'dLZ']:
            for inhibition in ['Control', 'CD40L']:
                data = df_Tfh_PI[(df_Tfh_PI['Zone1'] == zone) & (df_Tfh_PI[condition_name] == cell_type) & (df_Tfh_PI['Inhibition'] == inhibition)][feature_name]
                dataset[cell_type + ' ' + str(zone) + ' ' + inhibition] = np.array(data)


    rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
                   'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ', }
    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    # draw_custom_violin_plot(dataset_renamed, path + 'Zone int feature violin plot/', file_name=feature_name,
    #                         colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))
    try:
        draw_custom_bar_plot(dataset_renamed, path + 'zone Tfh_PC/', file_name=feature_name,
                             strip_plot=True, colors=('#888888', '#888888', '#CC6677', '#CC6677'),
                             test='kruskal-wallis_dunn', pvalue=True, return_sig=True, figsize=(2, 2))
    except:
        pass

[print(key, np.array(value).size) for key, value in dataset_renamed.items()]

############################### Interaction clock analysis ################################
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
dict_datasets_temp = {}
for cell_type in np.unique(df_both_PI['Type']):
    for inhibition in np.unique(df_both_PI['Inhibition']):
        if inhibition == 'IgG':
            continue
        else:

            df_part = df_both_PI[(df_both_PI['Type']==cell_type)&(df_both_PI['Inhibition']==inhibition)].reset_index(drop=True)
    #df_part['FDC_duration']
            dict_datasets_temp[cell_type+' '+inhibition] = (df_part['Tfh_first'].values - df_part['FDC_first'].values) / 2

del dict_datasets_temp['mt_B-cell CD40L']
new_order = ['wt_B-cell Control', 'mt_B-cell Control', 'wt_B-cell CD40L',]
ordered_dataset = change_dict_order(dict_datasets_temp, new_order)
dict_datasets = {replace_keys.get(k, k): v for (k, v) in ordered_dataset.items()}
vmin = np.min(flatten_nested_dict(dict_datasets))
vmax = np.max(flatten_nested_dict(dict_datasets))
draw_custom_bar_plot(dict_datasets, path, file_name='Tfh PC first or FDC PC first',
                         strip_plot=True, colors = ('#888888', '#CC6677', '#888888'), test='kruskal-wallis_dunn', pvalue=True, figsize=(1.5, 2),)
                     #vmin=vmin, vmax=vmax)
del dict_datasets['mt_B-cell Control']
draw_custom_bar_plot(dict_datasets, path, file_name='Tfh PC first or FDC PC first_PBS_vs_CD40L',
                         strip_plot=True, colors = ('#888888', '#888888'), test='mann-whitney', pvalue=True, figsize=(1, 2),)


[print(key, np.array(value).size) for key, value in dict_datasets.items()]


############################### inst speed and angle before during after ################################

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df_duration = pd.read_parquet(path+'long_traj_duration_20.parquet')

df_duration = df_duration[df_duration['Exp_group']!='A'].reset_index(drop=True)

df_duration['Inhibition'] = df_duration['Exp'].copy()
df_duration['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)


videos = np.unique(df_duration['Video']) # Remove Group A, IgG and CD40L

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

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\FigureX. Direct cauality btw motility and interaction\\'
threshold = 0
min_duration = 15
interaction_type = 'T-cell' #'FDC', 'T-cell'

features = ['instant_speed', 'instant_angle', 'Zone',
            'Sphericity', 'Volume', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core',
            'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC','diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'diff_Distance_to_DZ', 'diff_Distance_to_LZ', 'diff_Distance_to_FDC_core']


df_bda_motility = pd.DataFrame()
for feature in tqdm(features):
    #traj = df_partial[(df_partial['TrackID']==1000008597)&(df_partial['Video']=='2-Good-D9-B-M1L-ZT1-070-157-FOV230-320px_Statistics')].reset_index(drop=True)
    df_final = pd.DataFrame()
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        df_partial = df_duration[df_duration['Type'] == cell_type].reset_index(drop=True)
        befores = []
        durings = []
        afters = []
        labels = []
        datas = []
        inhibitions = []
        df_temporal = pd.DataFrame()
        count = 0
        for i in range(0, df_partial.shape[0]):
            if (i == 0) or (i == duration + i0):
                duration = df_partial['Time_span'][i]
                traj = df_partial[i: duration + i].reset_index(drop=True)
                i0 = i
                count = count + 1
                traj = traj[(traj['pseudo_frame'] != 0) & (traj['pseudo_frame'] != 1)].reset_index(drop=True)

                interaction_profile = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=%s' % interaction_type] > threshold
                interaction_profile = interaction_profile * 1  # Binary interaction profile

                elements = []
                indexes = []
                idx0 = 0
                for element, group in groupby(interaction_profile):  # Groups consistent values
                    # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
                    element_list = list(group)
                    idx1 = len(element_list) + idx0
                    idx_list = list(range(idx0, idx1))
                    idx0 = idx1
                    elements.append(element_list)
                    indexes.append(idx_list)

                check_minduration = np.array([np.sum(element) for element in elements]) >= min_duration
                # list of booleans whether each element have more than 20 persistent contact

                if any(check_minduration) == False:  # false -> No 20 persistent contact
                    continue

                elif np.sum(check_minduration)==1: # only one segment that is 20 persistent contact
                    n_groups = len(elements)
                    for group_idx, (index, element) in enumerate(zip(indexes, elements)):
                        if np.sum(element) >= min_duration:
                            if (group_idx != 0) and (group_idx != n_groups-1): # persistent segment should not be first nor last
                                #print(cell_type, interaction_type, traj['Exp'][0], traj['Video'][0], traj['TrackID'][0])
                                label = traj['Label'][0]
                                inhibition = traj['Inhibition'][0]
                                data = []
                                during= traj[feature][index].values
                                during_mean = np.median(during)
                                durings.append(during_mean)

                                before= traj[feature][indexes[group_idx - 1]].values
                                before_mean = np.median(before)
                                befores.append(before_mean)

                                after = traj[feature][indexes[group_idx + 1]].values
                                after_mean = np.median(after)
                                afters.append(after_mean)

                                data.append(before_mean)
                                data.append(during_mean)
                                data.append(after_mean)
                                labels.append(label)
                                datas.append(data)
                                inhibitions.append(inhibition)
        # befores = flatten_list(befores)
        # durings = flatten_list(durings)
        # afters = flatten_list(afters)
        print(cell_type, count)
        for d_type in ['before', 'during', 'after']:
            if d_type == 'before':
                aaa = befores
            elif d_type == 'during':
                aaa = durings
            elif d_type == 'after':
                aaa = afters

            df_temp = pd.DataFrame()
            df_temp['value'] = aaa
            df_temp['data_type'] = d_type
            df_temp['Label'] = labels
            df_temp['Inhibition'] = inhibitions
            df_temporal = pd.concat([df_temporal, df_temp], axis=0)

        df_temporal = df_temporal.reset_index(drop=True)
        df_temporal['Type'] = cell_type

        # df_temporal['before'] = befores
        # df_temporal['during'] = durings
        # df_temporal['after'] = afters
        # df_temporal['Type'] = cell_type

        df_final = pd.concat([df_final, df_temporal], axis=0)

    df_final = df_final.reset_index(drop=True)

    df_bda_motility[feature] = df_final['value']

df_bda_motility['data_type'] = df_final['data_type']
df_bda_motility['Type'] = df_final['Type']
df_bda_motility['Label'] = df_final['Label']
df_bda_motility['Inhibition'] = df_final['Inhibition']
df_bda_motility['type_label'] = df_final['Type'].astype(str) + '_' +df_final['Label'].astype(str)


########## pre/post contact motility mode identification by tskmeans ################
feature = 'instant_angle'

df_ts = pd.DataFrame()
time_series = []
for label in np.unique(df_bda_motility['type_label']):
    df_bda_motility_label = df_bda_motility[df_bda_motility['type_label'] == label]
    cell_type = df_bda_motility_label['Type'].values[0]
    inhibition = df_bda_motility_label['Inhibition'].values[0]
    before = df_bda_motility_label[df_bda_motility_label['data_type'] == 'before'][feature].values[0]
    during = df_bda_motility_label[df_bda_motility_label['data_type'] == 'during'][feature].values[0]
    after = df_bda_motility_label[df_bda_motility_label['data_type'] == 'after'][feature].values[0]

    ts = np.array([before, during, after])
    time_series.append(ts)

    df_ts_temp = pd.DataFrame()
    df_ts_temp['Type'] = [cell_type]
    df_ts_temp['Inhibition'] = [inhibition]
    df_ts_temp['Label'] = [label]
    df_ts_temp['before'] = [before]
    df_ts_temp['during'] = [during]
    df_ts_temp['after'] = [after]
    df_ts = pd.concat([df_ts, df_ts_temp], axis=0)

time_series = np.array(time_series)

time_series_dict = array_to_dict(time_series)
time_series_dict_scaled = normalize_timeseries(time_series_dict)
time_series_scaled = dict_to_array(time_series_dict_scaled)
from tslearn.clustering import TimeSeriesKMeans
tskm = TimeSeriesKMeans(n_clusters=4, metric='euclidean', random_state=0, verbose=True, max_iter=100)
tskmeans_predicted = tskm.fit_predict(time_series_scaled)

df_ts['tskmeans'] = tskmeans_predicted
df_ts, replace_map = order_cluster_by_feature(df_ts, cluster_name='tskmeans', feature_name='before')
tskm_cluster_centers_temp = tskm.cluster_centers_

tskm_cluster_centers= np.zeros_like(tskm_cluster_centers_temp)
# Apply the mapping to reorder rows
for old_idx, new_idx in replace_map.items():
    tskm_cluster_centers[new_idx] = tskm_cluster_centers_temp[old_idx]

df_ts['before_scaled'] = time_series_scaled[:, 0]
df_ts['during_scaled'] = time_series_scaled[:, 1]
df_ts['after_scaled'] = time_series_scaled[:, 2]


########## pre/post contact motility mode distribution ################
df_ts_ = df_ts.copy()
df_ts_ = df_ts_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_ts_['type_inhibition'] = df_ts_['Type'].astype(str) + ' ' + df_ts_['Inhibition'].astype(str)
df_ts_ = df_ts_[(df_ts_['type_inhibition']!='MT IgG')&(df_ts_['type_inhibition']!='MT CD40L')]
draw_cluster_distribution_heatmap(df_ts_, path, file_name='tskmeans_type_inhibition_%s_heatmap'%feature, condition_name='type_inhibition', cluster_type='tskmeans',
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(3.5,2))


p_dict = permutation_test(df_ts_, group_name='Type', class_name='tskmeans', iteration=10000)

########## pre/post contact motility representatibe cluster ################
for cluster, tskm_cluster_center in enumerate(tskm_cluster_centers):
    ts = tskm_cluster_center.flatten()
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.lineplot(x=np.arange(0, 2 + 1, 1),
                 y=ts, lw=2.5,
                 dashes=False, markersize=8, err_style='bars',
                 color='#888888')
    if not os.path.isdir(path + 'tskmeans_cluster_%s/'%feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'tskmeans_cluster_%s/'%feature)
    plt.savefig(path + 'tskmeans_cluster_%s/cluster_%s.png' % (feature, cluster), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/tskmeans_cluster_%s/'%feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/tskmeans_cluster_%s/'%feature)
    plt.savefig(path + 'svg/tskmeans_cluster_%s/cluster_%s.svg' % (feature, cluster), dpi=300, bbox_inches='tight')

    plt.clf()
    plt.close()

########## pre/post contact motility mode actual time series per cluster ################
for cluster in np.unique(df_ts['tskmeans']):
    df_ts_part = df_ts[df_ts['tskmeans'] == cluster]
    for idx, each_cell in df_ts_part.iterrows():
        label = each_cell['Label']
        values = each_cell[['before', 'during', 'after']].values
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.lineplot(x=np.arange(0, 2 + 1, 1),
                     y=values, lw=2.5,
                     dashes=False, markersize=8, err_style='bars',
                     color='#888888')
        if not os.path.isdir(path + 'actual_ts_%s/cluster_%s/'%(feature, cluster)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'actual_ts_%s/cluster_%s/'%(feature, cluster))

        plt.savefig(path + 'actual_ts_%s/cluster_%s/%s.png' % (feature, cluster, label), dpi=300, bbox_inches='tight')

        if not os.path.isdir(path + 'svg/actual_ts_%s/cluster_%s/'%(feature, cluster)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/actual_ts_%s/cluster_%s/'%(feature, cluster))

        plt.savefig(path + 'svg/actual_ts_%s/cluster_%s/%s.svg' % (feature, cluster, label), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()











############################### Determining which MC is signature for B-T interaction ################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GCB_no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'long_traj_duration_20.parquet')

# Remove Group A, IgG , mLT and CD40L
df_duration = df_duration[(df_duration['Exp']=='Exp1')|(df_duration['Exp']=='Exp2')|(df_duration['Exp']=='Exp3')
                              |(df_duration['Exp']=='Exp5')].reset_index(drop=True)

videos = np.unique(df_duration['Video'])
df_duration = df_duration[(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])&(df_duration['Video'] != videos[4])
                          &(df_duration['Video'] != videos[-1])].reset_index(drop=True)
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



from itertools import groupby

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\FigureX. Direct cauality btw motility and interaction\\'
threshold = 0
min_duration = 15
interaction_type = 'T-cell' #'FDC', 'T-cell'

# features = ['instant_speed', 'instant_angle', 'Zone',
#             'Sphericity', 'Volume', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core',
#             'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
#             'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC','diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell',
#             'diff_Distance_to_DZ', 'diff_Distance_to_LZ', 'diff_Distance_to_FDC_core']
features = ['instant_speed', 'instant_angle', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core',
            'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell']

df_bda_motility = pd.DataFrame()
for feature in tqdm(features):
    #traj = df_partial[(df_partial['TrackID']==1000008597)&(df_partial['Video']=='2-Good-D9-B-M1L-ZT1-070-157-FOV230-320px_Statistics')].reset_index(drop=True)
    df_final = pd.DataFrame()
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        df_partial = df_duration[df_duration['Type'] == cell_type].reset_index(drop=True)
        befores = []
        durings = []
        afters = []
        labels = []
        datas = []
        df_temporal = pd.DataFrame()
        count = 0
        for i in range(0, df_partial.shape[0]):
            if (i == 0) or (i == duration + i0):
                duration = df_partial['Time_span'][i]
                traj = df_partial[i: duration + i].reset_index(drop=True)
                i0 = i
                count = count + 1
                traj = traj[(traj['pseudo_frame'] != 0) & (traj['pseudo_frame'] != 1)].reset_index(drop=True)

                interaction_profile = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=%s' % interaction_type] > threshold
                interaction_profile = interaction_profile * 1  # Binary interaction profile

                elements = []
                indexes = []
                idx0 = 0
                for element, group in groupby(interaction_profile):  # Groups consistent values
                    # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
                    element_list = list(group)
                    idx1 = len(element_list) + idx0
                    idx_list = list(range(idx0, idx1))
                    idx0 = idx1
                    elements.append(element_list)
                    indexes.append(idx_list)

                check_minduration = np.array([np.sum(element) for element in elements]) >= min_duration
                # list of booleans whether each element have more than 20 persistent contact

                if any(check_minduration) == False:  # false -> No 20 persistent contact
                    continue

                elif np.sum(check_minduration)==1: # only one segment that is 20 persistent contact
                    n_groups = len(elements)
                    for group_idx, (index, element) in enumerate(zip(indexes, elements)):
                        if np.sum(element) >= min_duration:
                            if (group_idx != 0) and (group_idx != n_groups-1): # persistent segment should not be first nor last
                                #print(cell_type, interaction_type, traj['Exp'][0], traj['Video'][0], traj['TrackID'][0])
                                label = traj['Label'][0]
                                data = []
                                during= traj[feature][index].values
                                during_mean = np.mean(during)
                                durings.append(during_mean)

                                before= traj[feature][indexes[group_idx - 1]].values
                                before_mean = np.mean(before)
                                befores.append(before_mean)

                                after = traj[feature][indexes[group_idx + 1]].values
                                after_mean = np.mean(after)
                                afters.append(after_mean)

                                data.append(before_mean)
                                data.append(during_mean)
                                data.append(after_mean)
                                labels.append(label)
                                datas.append(data)
        # befores = flatten_list(befores)
        # durings = flatten_list(durings)
        # afters = flatten_list(afters)
        print(cell_type, count)
        for d_type in ['before', 'during', 'after']:
            if d_type == 'before':
                aaa = befores
            elif d_type == 'during':
                aaa = durings
            elif d_type == 'after':
                aaa = afters

            df_temp = pd.DataFrame()
            df_temp['value'] = aaa
            df_temp['data_type'] = d_type
            df_temp['Label'] = labels
            df_temporal = pd.concat([df_temporal, df_temp], axis=0)

        df_temporal = df_temporal.reset_index(drop=True)
        df_temporal['Type'] = cell_type

        # df_temporal['before'] = befores
        # df_temporal['during'] = durings
        # df_temporal['after'] = afters
        # df_temporal['Type'] = cell_type

        df_final = pd.concat([df_final, df_temporal], axis=0)

    df_final = df_final.reset_index(drop=True)

    df_bda_motility[feature] = df_final['value']

df_bda_motility['data_type'] = df_final['data_type']
df_bda_motility['Type'] = df_final['Type']
df_bda_motility['Label'] = df_final['Label']
df_bda_motility['type_label'] = df_final['Type'].astype(str) + '_' +df_final['Label'].astype(str)


############ MC imputation ############
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

knn = KNeighborsClassifier(n_neighbors=10)

scaler = StandardScaler()
X = scaler.fit_transform(df[['avg_speed', 'avg_angle', 'DZ_distance_average', 'LZ_distance_average',
                                            'Core_distance_average', 'FDC_distance_average', 'T_distance_average']].values)
y = df['kmeans'].values
knn.fit(X,y)

X_new = scaler.transform(df_bda_motility[features].values)
y_new = knn.predict(X_new)
df_bda_motility['pred_MC'] = y_new

############ pred_MC_distribution ############
df_ = df_bda_motility.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_['Type_data_type'] = df_['Type'].astype(str) + ' ' + df_['data_type'].astype(str)
draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_order_type_T_heatmap_int_feature_included', condition_name='Type_data_type', cluster_type='pred_MC',
                                  annot=True, col_cluster=False, row_cluster=True,figsize=(6,4.5))
p_dict = permutation_test(df_, group_name='Type_data_type', class_name='pred_MC', iteration=10000)

############ UMAP imputation ############
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

#reg = MLPRegressor()
reg = RandomForestRegressor()
scaler = StandardScaler()
X = scaler.fit_transform(df[['avg_speed', 'avg_angle', 'DZ_distance_average', 'LZ_distance_average',
                                            'Core_distance_average', 'FDC_distance_average', 'T_distance_average']].values)
y = df[['PC1', 'PC2']].values
reg.fit(X,y)

X_new = scaler.transform(df_bda_motility[features].values)
y_new = reg.predict(X_new)
df_bda_motility[['pred_PC1', 'pred_PC2']] = y_new

# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=10, weights="uniform")
# imputer.fit_transform(X)

############ Draw UMAP space ############
xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

df_bda_motility_ = df_bda_motility.copy()
df_bda_motility_ = df_bda_motility_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

for d_type in np.unique(df_bda_motility_['data_type']):
    df_bda_motility_part = df_bda_motility_[df_bda_motility_['data_type']==d_type].reset_index(drop=True)
    draw_jointplot(xs='pred_PC1', y='pred_PC2', df=df_bda_motility_part, path=path, file_name='jointplot_type_%s'%d_type, hue="Type",
                   colors=('#CC6677', '#888888', ), hue_order=['MT', 'WT'],
                   legend=False, fill=False, thresh=0.15, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2',
                   xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)




####### Motility cluster transitions of GCB for three nodes ####
df_ts = pd.DataFrame()
for label in np.unique(df_bda_motility['type_label']):
    df_bda_motility_label = df_bda_motility[df_bda_motility['type_label'] == label]
    cell_type = df_bda_motility_label['Type'].values[0]

    before = df_bda_motility_label[df_bda_motility_label['data_type'] == 'before']['pred_MC'].values[0]
    during = df_bda_motility_label[df_bda_motility_label['data_type'] == 'during']['pred_MC'].values[0]
    after = df_bda_motility_label[df_bda_motility_label['data_type'] == 'after']['pred_MC'].values[0]

    ts = np.array([before, during, after])

    df_ts_temp = pd.DataFrame()
    df_ts_temp['Type'] = [cell_type]
    df_ts_temp['Label'] = [label]
    df_ts_temp['before'] = [before]
    df_ts_temp['during'] = [during]
    df_ts_temp['after'] = [after]
    df_ts = pd.concat([df_ts, df_ts_temp], axis=0)


####### Motility cluster transitions of GCB for three nodes ####

cluster_type = 'kmeans'
for cell_type in ['wt_B-cell', 'mt_B-cell']:
    cluster_size = np.unique(df[cluster_type]).size
    df_trans = df_ts[df_ts['Type']==cell_type].reset_index(drop=True)

    transit_array1_temp = np.empty((cluster_size,cluster_size), dtype = 'object')
    transit_array2_temp = np.empty((cluster_size,cluster_size), dtype = 'object')
    for row in range(0,transit_array1_temp.shape[0]):
        for col in range(0,transit_array1_temp.shape[1]):
            transit_array1_temp[row,col] = []
            transit_array2_temp[row,col] = []
    for label in np.unique(df_trans['Label']):
        each_label = df_trans[df_trans['Label']==label].reset_index(drop=True)
        row1 = each_label['before'].values[0]
        col1 = each_label['during'].values[0]
        row2 = each_label['during'].values[0]
        col2 = each_label['after'].values[0]
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
    fig.update_traces(textfont=dict(size=45))
    fig.write_html(path + 'T interaction sankey-plot_%s_int_feature_included.html' % cell_type)
############################### Determining which MC is signature for B-FDC interaction ################################

from itertools import groupby

threshold = 0
min_duration = 15
interaction_type = 'FDC' #'FDC', 'T-cell'

# features = ['instant_speed', 'instant_angle', 'Zone',
#             'Sphericity', 'Volume', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core',
#             'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
#             'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC','diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell',
#             'diff_Distance_to_DZ', 'diff_Distance_to_LZ', 'diff_Distance_to_FDC_core']
features = ['instant_speed', 'instant_angle', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core',
            'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell']

df_bda_motility = pd.DataFrame()
for feature in tqdm(features):
    #traj = df_partial[(df_partial['TrackID']==1000008597)&(df_partial['Video']=='2-Good-D9-B-M1L-ZT1-070-157-FOV230-320px_Statistics')].reset_index(drop=True)
    df_final = pd.DataFrame()
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        df_partial = df_duration[df_duration['Type'] == cell_type].reset_index(drop=True)
        befores = []
        durings = []
        afters = []
        labels = []
        datas = []
        df_temporal = pd.DataFrame()
        count = 0
        for i in range(0, df_partial.shape[0]):
            if (i == 0) or (i == duration + i0):
                duration = df_partial['Time_span'][i]
                traj = df_partial[i: duration + i].reset_index(drop=True)
                i0 = i
                count = count + 1
                traj = traj[(traj['pseudo_frame'] != 0) & (traj['pseudo_frame'] != 1)].reset_index(drop=True)

                interaction_profile = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=%s' % interaction_type] > threshold
                interaction_profile = interaction_profile * 1  # Binary interaction profile

                elements = []
                indexes = []
                idx0 = 0
                for element, group in groupby(interaction_profile):  # Groups consistent values
                    # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
                    element_list = list(group)
                    idx1 = len(element_list) + idx0
                    idx_list = list(range(idx0, idx1))
                    idx0 = idx1
                    elements.append(element_list)
                    indexes.append(idx_list)

                check_minduration = np.array([np.sum(element) for element in elements]) >= min_duration
                # list of booleans whether each element have more than 20 persistent contact

                if any(check_minduration) == False:  # false -> No 20 persistent contact
                    continue

                elif np.sum(check_minduration)==1: # only one segment that is 20 persistent contact
                    n_groups = len(elements)
                    for group_idx, (index, element) in enumerate(zip(indexes, elements)):
                        if np.sum(element) >= min_duration:
                            if (group_idx != 0) and (group_idx != n_groups-1): # persistent segment should not be first nor last
                                #print(cell_type, interaction_type, traj['Exp'][0], traj['Video'][0], traj['TrackID'][0])
                                label = traj['Label'][0]
                                data = []
                                during= traj[feature][index].values
                                during_mean = np.mean(during)
                                durings.append(during_mean)

                                before= traj[feature][indexes[group_idx - 1]].values
                                before_mean = np.mean(before)
                                befores.append(before_mean)

                                after = traj[feature][indexes[group_idx + 1]].values
                                after_mean = np.mean(after)
                                afters.append(after_mean)

                                data.append(before_mean)
                                data.append(during_mean)
                                data.append(after_mean)
                                labels.append(label)
                                datas.append(data)
        # befores = flatten_list(befores)
        # durings = flatten_list(durings)
        # afters = flatten_list(afters)
        print(cell_type, count)
        for d_type in ['before', 'during', 'after']:
            if d_type == 'before':
                aaa = befores
            elif d_type == 'during':
                aaa = durings
            elif d_type == 'after':
                aaa = afters

            df_temp = pd.DataFrame()
            df_temp['value'] = aaa
            df_temp['data_type'] = d_type
            df_temp['Label'] = labels
            df_temporal = pd.concat([df_temporal, df_temp], axis=0)

        df_temporal = df_temporal.reset_index(drop=True)
        df_temporal['Type'] = cell_type

        # df_temporal['before'] = befores
        # df_temporal['during'] = durings
        # df_temporal['after'] = afters
        # df_temporal['Type'] = cell_type

        df_final = pd.concat([df_final, df_temporal], axis=0)

    df_final = df_final.reset_index(drop=True)

    df_bda_motility[feature] = df_final['value']

df_bda_motility['data_type'] = df_final['data_type']
df_bda_motility['Type'] = df_final['Type']
df_bda_motility['Label'] = df_final['Label']
df_bda_motility['type_label'] = df_final['Type'].astype(str) + '_' +df_final['Label'].astype(str)

############ MC imputation ############
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

knn = KNeighborsClassifier(n_neighbors=10)

scaler = StandardScaler()
X = scaler.fit_transform(df[['avg_speed', 'avg_angle', 'DZ_distance_average', 'LZ_distance_average',
                                            'Core_distance_average', 'FDC_distance_average', 'T_distance_average']].values)
y = df['kmeans'].values
knn.fit(X,y)

X_new = scaler.transform(df_bda_motility[features].values)
y_new = knn.predict(X_new)
df_bda_motility['pred_MC'] = y_new

############ imputed MC distribution ############
df_ = df_bda_motility.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_['Type_data_type'] = df_['Type'].astype(str) + ' ' + df_['data_type'].astype(str)
draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_order_type_FDC_heatmap_int_feature_included', condition_name='Type_data_type', cluster_type='pred_MC',
                                  annot=True, col_cluster=False, row_cluster=True,figsize=(6,4.5))
p_dict = permutation_test(df_, group_name='Type_data_type', class_name='pred_MC', iteration=10000)

for i in np.unique(df['kmeans']):
    print('cluster: ', i, 'Total distance: ', np.mean( df[df['kmeans']==i]['total_distance'] ))



df_ts = pd.DataFrame()
time_series = []
for label in np.unique(df_bda_motility['type_label']):
    df_bda_motility_label = df_bda_motility[df_bda_motility['type_label'] == label]
    cell_type = df_bda_motility_label['Type'].values[0]

    before = df_bda_motility_label[df_bda_motility_label['data_type'] == 'before']['pred_MC'].values[0]
    during = df_bda_motility_label[df_bda_motility_label['data_type'] == 'during']['pred_MC'].values[0]
    after = df_bda_motility_label[df_bda_motility_label['data_type'] == 'after']['pred_MC'].values[0]

    ts = np.array([before, during, after])
    time_series.append(ts)

    df_ts_temp = pd.DataFrame()
    df_ts_temp['Type'] = [cell_type]
    df_ts_temp['Label'] = [label]
    df_ts_temp['before'] = [before]
    df_ts_temp['during'] = [during]
    df_ts_temp['after'] = [after]
    df_ts = pd.concat([df_ts, df_ts_temp], axis=0)


####### Motility cluster transitions of GCB for three nodes ####

cluster_type = 'kmeans'
for cell_type in ['wt_B-cell', 'mt_B-cell']:
    cluster_size = np.unique(df[cluster_type]).size
    df_trans = df_ts[df_ts['Type']==cell_type].reset_index(drop=True)

    transit_array1_temp = np.empty((cluster_size,cluster_size), dtype = 'object')
    transit_array2_temp = np.empty((cluster_size,cluster_size), dtype = 'object')
    for row in range(0,transit_array1_temp.shape[0]):
        for col in range(0,transit_array1_temp.shape[1]):
            transit_array1_temp[row,col] = []
            transit_array2_temp[row,col] = []
    for label in np.unique(df_trans['Label']):
        each_label = df_trans[df_trans['Label']==label].reset_index(drop=True)
        row1 = each_label['before'].values[0]
        col1 = each_label['during'].values[0]
        row2 = each_label['during'].values[0]
        col2 = each_label['after'].values[0]
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
    fig.update_traces(textfont=dict(size=45))
    fig.write_html(path + 'FDC interaction sankey-plot_%s_int_feature_included.html' % cell_type)












