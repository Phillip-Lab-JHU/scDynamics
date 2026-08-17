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
"""Preprocess dataframe"""

from utils.traj_utils import *
from utils.draw_utils import *
from features.basic_motility import BasicMotility
from utils.traj_utils import morpho_trajectory
from features.aprw import APRW
from features.decomposed_motility import DecomposedMotility3D
from features.interaction import DistanceSignal, OverlapSignal, ZoneSignal
from features.timeseries import Timeseries
from features.directionality import Directionality

########################################################################################################
###1. From segmentation from Imaris, you combine all excel files into one excel file using data curation.py
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Imaris csvs\\'
excel = pd.read_parquet(path + 'Intravital Data_all.parquet')

excel['Type'].replace({'Tfh': 'T-cell', 'wtGCB': 'wt B-cell', 'mtGCB': 'mt B-cell'}, inplace=True)
excel.rename(columns=lambda x: x.replace('Tfh','T-cell').replace('wtGCB', 'wt_B-cell').replace('mtGCB', 'mt_B-cell'), inplace=True)

########################################################################################################
### 2. to_trajectory_duration to sort the dataframe by cells and fixed duration
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\\'
duration = 20
duration_int15s = 40
df = pd.DataFrame()
for idx, video in enumerate(np.unique(excel['Video'])):
    frame_interval = 15 if 'int15s' in str(video).lower() else 30
    video_duration = duration_int15s if frame_interval == 15 else duration
    print(idx, video, video_duration, frame_interval)
    excel_temp = excel[excel['Video']==video].reset_index(drop=True)
    df_duration_temp = to_trajectory_duration(excel_temp, duration=video_duration, condition_name='Type', frame_name='Time', label_name='Label', verbose=False)
    #df_duration_temp = to_trajectory_variable_duration(excel_temp, min_duration=video_duration, condition_name='Type', label_name='Label')
    if video_duration == duration_int15s:
        group_names = ['Type', 'pseudo_Label']
        row_in_traj = df_duration_temp.groupby(group_names).cumcount()
        df_duration_temp = df_duration_temp[row_in_traj % 2 == 0].copy()
        new_time = df_duration_temp.groupby(group_names).cumcount()
        df_duration_temp['Time'] = new_time + 1
        df_duration_temp['pseudo_Time'] = new_time
    df_duration_temp['frame_interval'] = frame_interval
    df = pd.concat([df, df_duration_temp], axis=0)
df = df.reset_index(drop=True)

#df.to_csv(path + 'traj_duration_%s.csv' % duration, index=False)
#df.to_parquet(path + 'traj_duration_%s.parquet' % duration)
########################################################################################################
### 2. to_trajectory_duration to sort the dataframe by cells and fixed duration
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\\'
duration = 20
duration_int15s = 40
df = pd.DataFrame()
for idx, video in enumerate(np.unique(excel['Video'])):
    frame_interval = 15 if 'int15s' in str(video).lower() else 30
    video_duration = duration_int15s if frame_interval == 15 else duration
    print(idx, video, video_duration, frame_interval)
    excel_temp = excel[excel['Video']==video].reset_index(drop=True)
    df_duration_temp = to_trajectory_variable_duration(excel_temp, min_duration=video_duration, condition_name='Type', label_name='Label')
    df_duration_temp['frame_interval'] = frame_interval
    df = pd.concat([df, df_duration_temp], axis=0)
df = df.reset_index(drop=True)

df.to_csv(path + 'long_traj_duration_%s_all.csv' % duration, index=False)
df.to_parquet(path + 'long_traj_duration_%s_all.parquet' % duration)

df_part = df[df['Video']=='202506a-Good-D10-B1L-zt1-z45-90-step3-fov230-mLT-int15s_Statistics']
#df = to_trajectory_duration(excel, duration=duration, condition_name='Type', frame_name='Time', label_name='Label')
#df = to_trajectory_variable_duration(excel, min_duration=duration, condition_name='Type', label_name='Label')
traj_list, trajectories_array, trajectories = to_timeseries_fast(df, duration=duration, feature_name=['Position X', 'Position Y', 'Position Z'])
rotated_trajectories = register_traj_disp(trajectories)
rotated_trajectories = dict_to_array(rotated_trajectories)
df_traj = rotated_trajectories.reshape(rotated_trajectories.shape[0] * rotated_trajectories.shape[1], 3)
df_traj = pd.DataFrame(df_traj, columns=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])
df_duration = pd.concat([df_traj, df], axis=1)
#df_duration.to_csv(path + 'traj_duration_%s.csv' % duration, index=False)
#df_duration.to_parquet(path + 'traj_duration_%s.parquet' % duration)

########################################################################################################
### 3. Preprocess dataframe using some filters and make data as trajectory dictionary
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\\'
# duration = 20
# df_duration = pd.read_parquet(path + 'traj_duration_%s.parquet' %duration)

#df_removed, removed_index = remove_stationary_trajs(df, duration=time, feature_name=['x', 'y'])

interaction_features = []
overlapped_volume_features = []
shortest_distance_features = []
for column_name in df_duration.columns:
    if any(txt in column_name for txt in ('Overlapped',
                                          'Shortest_Distance')):  # Search whether 'Overlapped', or 'Shortest_Distance' word is in the column_name
        interaction_features.append(column_name)
    if 'Overlapped_Volume_Ratio' in column_name:
        overlapped_volume_features.append(column_name)

    if 'Shortest_Distance' in column_name:  # Search whether 'Overlapped', or 'Shortest_Distance' word is in the column_name
        shortest_distance_features.append(column_name)

new_df = pd.DataFrame()
for traj_idx in tqdm(range(0, int(df_duration.shape[0]/duration))):
    traj_data_temp = df_duration[duration*traj_idx:duration*(traj_idx+1)]
    row0 = traj_data_temp[overlapped_volume_features[0]]
    row1 = traj_data_temp[overlapped_volume_features[1]]
    row2 = traj_data_temp[overlapped_volume_features[2]]
    row3 = traj_data_temp[overlapped_volume_features[3]]
    row4 = traj_data_temp[overlapped_volume_features[4]]
    if all(row0 < 0.6) and all(row1 < 0.6) and all(row2 < 0.6) and all(row3 < 0.6) and all(row4 < 0.6):
        new_df = pd.concat([new_df, traj_data_temp])

traj_final = new_df.reset_index(drop=True)
traj_final = traj_final.replace({'Type': {'wt B-cell': 'wt_B-cell', 'mt B-cell': 'mt_B-cell'}})
df_duration = traj_final

########################################################################################################
### 4. Calculate instantaneous features(inst speed, angle and directionality wrt FDC and T cell)
df_duration = get_instant_movements(df_duration, duration=duration, time_unit=0.5, feature_name=['Position X', 'Position Y', 'Position Z'])
df_duration = get_instant_direction(df_duration, duration=duration, thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=FDC')
df_duration = get_instant_direction(df_duration, duration=duration, thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=T-cell')

df_duration = get_instant_direction(df_duration, duration=duration, thresh=0.1, feature_name='Distance_to_DZ')
df_duration = get_instant_direction(df_duration, duration=duration, thresh=0.1, feature_name='Distance_to_LZ')
df_duration = get_instant_direction(df_duration, duration=duration, thresh=0.1, feature_name='Distance_to_FDC_core')

# videos = np.unique(df_duration['Video'])
# df_duration_noA = df_duration[(df_duration['Video'] != videos[0])&(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])].reset_index(drop=True)
# df_duration_A = df_duration[(df_duration['Video'] == videos[0])|(df_duration['Video'] == videos[1])|(df_duration['Video'] == videos[2])].reset_index(drop=True)
# df_duration_noMT = df_duration[df_duration['Type'] != 'mt_B-cell'].reset_index(drop=True)

df_duration.to_csv(path + 'traj_duration_%s_all.csv'%duration, index=False)
df_duration.to_parquet(path + 'traj_duration_%s_all.parquet'%duration)

# df_duration.to_csv(path + 'traj_duration_%s.csv'%duration, index=False)
# df_duration.to_parquet(path + 'traj_duration_%s.parquet'%duration)

########################################################################################################
### 5. Calculate motility features(APRW + basic motility + morphodynamic)
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\\'
duration = 20
df_duration = pd.read_parquet(path + 'traj_duration_%s_all.parquet' %duration)

###### Make trajectory as dictionary form ######
traj_list, _, trajectories = to_timeseries_fast(df_duration, duration=duration, feature_name=['Position X', 'Position Y', 'Position Z'])

###### Calculate basic motility features ######
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

k = df_basic.isnull().any()
print(np.where(np.isnan(df_basic['alphas'])))

###### Calculate anisotropic motility features ######
ani_motil = DecomposedMotility3D(trajectories, time_unit=0.5)
feature_list = ['avg_speed_x', 'max_speed_x', 'min_speed_x', 'net_distance_x', 'progressivity_x',
                'avg_speed_y', 'max_speed_y', 'min_speed_y', 'net_distance_y', 'progressivity_y',
                'avg_speed_z', 'max_speed_z', 'min_speed_z', 'net_distance_z', 'progressivity_z',
                'exy_max', 'eyz_max', 'exz_max', 'phi_max', 'exy_total', 'eyz_total', 'exz_total', 'phi_total',
                'msd_x', 'msd_y', 'msd_z',
                #'alpha_x', 'alpha_y', 'alpha_z',
                'displ_variance_x', 'displ_cov_x', 'displ_skewness_x', 'displ_kurtosis_x', 'displ_ngaussalpha_x',
                'displ_variance_y', 'displ_cov_y', 'displ_skewness_y', 'displ_kurtosis_y', 'displ_ngaussalpha_y',
                'displ_variance_z', 'displ_cov_z', 'displ_skewness_z', 'displ_kurtosis_z', 'displ_ngaussalpha_z',
                'displ_autocorr_x', 'displ_autocorr_y', 'displ_autocorr_z',
                #'displ_partial_autocorr_x', 'displ_partial_autocorr_y', 'displ_partial_autocorr_z',
                #'displ_hurst_RS_x', 'displ_hurst_RS_y', 'displ_hurst_RS_z'
                ]
#df_aniso = ani_motil(feature_list, tau_limit=3)
df_aniso = ani_motil.extract_features(feature_list=feature_list, tau_limit=3)
k = df_aniso.isnull().any()

###### Calculate APRW features ######
aprw = APRW()
df_aprw = aprw.get_APRW(trajectories, dt=0.5, max_speed=np.max(df_basic['avg_speed']))

###### Calculate morphodynamic features ######
morpho_trajectories = morpho_trajectory(df_duration, features = ['Area', 'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Sphericity', 'Volume'], duration=duration, dim=3)
feature_list = ['avg_speed', 'max_speed', 'min_speed',  'progressivity', 'displ_autocorr', 'displ_cov',
                #'avg_angle', 'max_angle', 'min_angle',
                #'displ_variance', ,'displ_skewness', 'displ_kurtosis', 'displ_ngaussalpha',
                #'angle_variance', 'angle_cov', 'angle_skewness', 'angle_kurtosis', 'angle_ngaussalpha',
                #'msd', , 'displ_partial_autocorr', 'angle_autocorr', 'angle_partial_autocorr',
                #'displ_hurst_RS', 'angle_hurst_RS','net_distance','alpha',
                ]
morpho_basic = BasicMotility(morpho_trajectories, time_unit=0.5, feature_list=feature_list)
df_morpho = morpho_basic.extract_features(tau_limit=3)
#df_morpho = df_morpho.drop(['angle_distribution', 'speed_distribution'], axis=1)
for column in df_morpho.columns:
    df_morpho.rename(columns={column:'morpho_'+column}, inplace=True)

###### Calculate instantaneous speed timeseries features ######
df_inst_speed = df_duration[df_duration['pseudo_Time']!=0].reset_index(drop=True)
_, _, inst_speed_ts = to_timeseries_fast(df_inst_speed, duration=duration-1, feature_name='instant_speed')
ts_features = Timeseries(inst_speed_ts)
feature_list = ['peak_to_peak', 'slopes', 'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',
                'approximate_entropies', 'attention_entropies', 'permutation_entropies', 'bubble_entropies', 'cosine_similarity_entropies',
               #'corrected_conditional_entropies', 'dispersion_entropies', 'distribution_entropies', 'entropy_of_entropies', 'fuzzy_entropies',
               #'gridded_distribution_entropies', 'increment_entropies', 'kolmogorov_entropies', 'phase_entropies', 'sample_entropies', 'slope_entropies',
               #'spectral_entropies', 'symbolic_dynamic_entropies'
                ]
df_inst_speed_features= ts_features.extract_features(feature_list=feature_list)
for column in df_inst_speed_features.columns:
    df_inst_speed_features.rename(columns={column:'inst_speed_'+column}, inplace=True)

k = df_inst_speed_features.isnull().any()
null_features = k.index[k==True]
df_inst_speed_features = df_inst_speed_features.drop(null_features, axis=1)

###### Calculate instantaneous angle timeseries features ######
df_inst_angle = df_duration[(df_duration['pseudo_Time']!=0)&(df_duration['pseudo_Time']!=1)].reset_index(drop=True)
_, _, inst_angle_ts = to_timeseries_fast(df_inst_angle, duration=duration-2, feature_name='instant_angle')
ts_features = Timeseries(inst_angle_ts)
feature_list = ['peak_to_peak', 'slopes', 'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',
                'approximate_entropies', 'attention_entropies', 'permutation_entropies', 'bubble_entropies', 'cosine_similarity_entropies',
               #'corrected_conditional_entropies', 'dispersion_entropies', 'distribution_entropies', 'entropy_of_entropies', 'fuzzy_entropies',
               #'gridded_distribution_entropies', 'increment_entropies', 'kolmogorov_entropies', 'phase_entropies', 'sample_entropies', 'slope_entropies',
               #'spectral_entropies', 'symbolic_dynamic_entropies'
                ]
df_inst_angle_features= ts_features.extract_features(feature_list=feature_list)
for column in df_inst_angle_features.columns:
    df_inst_angle_features.rename(columns={column:'inst_angle_'+column}, inplace=True)

k = df_inst_angle_features.isnull().any()
null_features = k.index[k==True]
df_inst_angle_features = df_inst_angle_features.drop(null_features, axis=1)

###### Compute label features ######
df_labels = reduced_label_for_overlapped_volume(df_duration, duration=duration)
df_labels = df_labels[['TrackID', 'Time', 'pseudo_Time', 'Label', 'pseudo_Label', 'Type', 'Video', 'Exp', 'Day',
                       'Exp_group', 'Time_span', 'instant_speed', 'instant_angle', 'frame_interval']]
df_labels['traj_Label'] = df_labels['Video'].astype(str)+'_'+df_labels['Type'].astype(str)+'_'\
                         +df_labels['Label'].astype(str)

###### Concatenate all features ######
df_motility = pd.concat([df_aprw, df_basic, df_aniso, df_inst_speed_features, df_inst_angle_features, df_morpho, df_labels], axis=1)
a = df_motility.isnull().any()

#path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'
#path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Long term/GCB/'
df_motility.to_csv(path + 'motility_features_%s.csv'%duration, index=False)
df_motility.to_parquet(path + 'motility_features_%s.parquet'%duration)

########################################################################################################
### 6. Calculate interaction properties

###### Calculate directional quality features ######
feature_list = ['approach_times', 'approach_persistences', 'departure_times', 'departure_persistences', 'stay_times', 'stay_persistences']

_, _, FDC_direction = to_timeseries_fast(df_duration, duration=duration, feature_name='quality_Shortest_Distance_to_Surfaces_Surfaces=FDC' )
FDC_direction_features = Directionality(FDC_direction)
df_FDC_direction_features= FDC_direction_features.extract_features(feature_list=feature_list)
for column in df_FDC_direction_features.columns:
    df_FDC_direction_features.rename(columns={column:'quality_FDC_'+column}, inplace=True)

_, _, T_direction = to_timeseries_fast(df_duration, duration=duration, feature_name='quality_Shortest_Distance_to_Surfaces_Surfaces=T-cell' )
T_direction_features = Directionality(T_direction)
df_T_direction_features= T_direction_features.extract_features(feature_list=feature_list)
for column in df_T_direction_features.columns:
    df_T_direction_features.rename(columns={column:'quality_T_'+column}, inplace=True)

_, _, DZ_direction = to_timeseries_fast(df_duration, duration=duration, feature_name='quality_Distance_to_DZ')
DZ_direction_features = Directionality(DZ_direction)
df_DZ_direction_features= DZ_direction_features.extract_features(feature_list=feature_list)
for column in df_DZ_direction_features.columns:
    df_DZ_direction_features.rename(columns={column:'quality_DZ_'+column}, inplace=True)

_, _, LZ_direction = to_timeseries_fast(df_duration, duration=duration, feature_name='quality_Distance_to_LZ' )
LZ_direction_features = Directionality(LZ_direction)
df_LZ_direction_features= LZ_direction_features.extract_features(feature_list=feature_list)
for column in df_LZ_direction_features.columns:
    df_LZ_direction_features.rename(columns={column:'quality_LZ_'+column}, inplace=True)

_, _, Core_direction = to_timeseries_fast(df_duration, duration=duration, feature_name='quality_Distance_to_FDC_core' )
Core_direction_features = Directionality(Core_direction)
df_Core_direction_features= Core_direction_features.extract_features(feature_list=feature_list)
for column in df_Core_direction_features.columns:
    df_Core_direction_features.rename(columns={column:'quality_Core_'+column}, inplace=True)

df_quality_features = pd.concat([df_FDC_direction_features, df_T_direction_features, df_DZ_direction_features,
                                     df_LZ_direction_features, df_Core_direction_features], axis=1)

############################ Calculate Quantitative FDC features ############################
_, _, FDC_distances = to_timeseries_fast(df_duration, duration, feature_name='Shortest_Distance_to_Surfaces_Surfaces=FDC')
_, _, FDC_overlap = to_timeseries_fast(df_duration, duration, feature_name='Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC')

df_duration_diff = df_duration[(df_duration['pseudo_Time']!=0)].reset_index(drop=True)
_, _, FDC_diff_distances = to_timeseries_fast(df_duration_diff, duration=duration-1, feature_name='diff_Shortest_Distance_to_Surfaces_Surfaces=FDC')

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average', 'autocorr',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr'
                ]
FDC_dist = DistanceSignal(FDC_distances)
df_distance = FDC_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)

FDC_diff_dist = DistanceSignal(FDC_diff_distances)
df_diff_distance = FDC_diff_dist.extract_features(feature_list, tau_limit=3)
for column in df_diff_distance.columns:
    df_diff_distance.rename(columns={column:'diff_distance_'+column}, inplace=True)

feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
FDC_over = OverlapSignal(FDC_overlap)
df_overlap = FDC_over.extract_features(feature_list)

df_inter_FDC = pd.concat([df_distance, df_diff_distance, df_overlap], axis=1)
for column in df_inter_FDC.columns:
    df_inter_FDC.rename(columns={column:'FDC_'+column}, inplace=True)


###### Calculate Quantitative Tfh features ######
_, _, T_distances = to_timeseries_fast(df_duration, duration, feature_name='Shortest_Distance_to_Surfaces_Surfaces=T-cell')
_, _, T_overlap = to_timeseries_fast(df_duration, duration, feature_name='Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell')

df_duration_diff = df_duration[(df_duration['pseudo_Time']!=0)].reset_index(drop=True)
_, _, T_diff_distances = to_timeseries_fast(df_duration_diff, duration=duration-1, feature_name='diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell')

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average', 'autocorr',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr'
                ]

T_dist = DistanceSignal(T_distances)
df_distance = T_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)

T_diff_dist = DistanceSignal(T_diff_distances)
df_diff_distance = T_diff_dist.extract_features(feature_list, tau_limit=3)
for column in df_diff_distance.columns:
    df_diff_distance.rename(columns={column:'diff_distance_'+column}, inplace=True)

feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
T_over = OverlapSignal(T_overlap)
df_overlap = T_over.extract_features(feature_list)

df_inter_T = pd.concat([df_distance, df_diff_distance, df_overlap], axis=1)
for column in df_inter_T.columns:
    df_inter_T.rename(columns={column:'T_'+column}, inplace=True)

###### Calculate Quantitative DZ features ######
_, _, DZ_distances = to_timeseries_fast(df_duration, duration, feature_name='Distance_to_DZ')
df_duration_diff = df_duration[(df_duration['pseudo_Time']!=0)].reset_index(drop=True)
_, _, DZ_diff_distances = to_timeseries_fast(df_duration_diff, duration=duration-1, feature_name='diff_Distance_to_DZ')

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average', 'autocorr',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr'
                ]

DZ_dist = DistanceSignal(DZ_distances)
df_distance = DZ_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)

DZ_diff_dist = DistanceSignal(DZ_diff_distances)
df_diff_distance = DZ_diff_dist.extract_features(feature_list, tau_limit=3)
for column in df_diff_distance.columns:
    df_diff_distance.rename(columns={column:'diff_distance_'+column}, inplace=True)

df_inter_DZ = pd.concat([df_distance, df_diff_distance], axis=1)
for column in df_inter_DZ.columns:
    df_inter_DZ.rename(columns={column:'DZ_'+column}, inplace=True)

###### Calculate Quantitative LZ features ######
_, _, LZ_distances = to_timeseries_fast(df_duration, duration, feature_name='Distance_to_LZ')
df_duration_diff = df_duration[(df_duration['pseudo_Time']!=0)].reset_index(drop=True)
_, _, LZ_diff_distances = to_timeseries_fast(df_duration_diff, duration=duration-1, feature_name='diff_Distance_to_LZ')

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average', 'autocorr',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr'
                ]

LZ_dist = DistanceSignal(LZ_distances)
df_distance = LZ_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)

LZ_diff_dist = DistanceSignal(LZ_diff_distances)
df_diff_distance = LZ_diff_dist.extract_features(feature_list, tau_limit=3)
for column in df_diff_distance.columns:
    df_diff_distance.rename(columns={column:'diff_distance_'+column}, inplace=True)

df_inter_LZ = pd.concat([df_distance, df_diff_distance], axis=1)
for column in df_inter_LZ.columns:
    df_inter_LZ.rename(columns={column:'LZ_'+column}, inplace=True)


###### Calculate Quantitative Core features ######
_, _, Core_distances = to_timeseries_fast(df_duration, duration, feature_name='Distance_to_FDC_core')
df_duration_diff = df_duration[(df_duration['pseudo_Time']!=0)].reset_index(drop=True)
_, _, Core_diff_distances = to_timeseries_fast(df_duration_diff, duration=duration-1, feature_name='diff_Distance_to_FDC_core')

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average', 'autocorr',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr'
                ]

Core_dist = DistanceSignal(Core_distances)
df_distance = Core_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)

Core_diff_dist = DistanceSignal(Core_diff_distances)
df_diff_distance = Core_diff_dist.extract_features(feature_list, tau_limit=3)
for column in df_diff_distance.columns:
    df_diff_distance.rename(columns={column:'diff_distance_'+column}, inplace=True)

df_inter_Core = pd.concat([df_distance, df_diff_distance], axis=1)
for column in df_inter_Core.columns:
    df_inter_Core.rename(columns={column:'Core_'+column}, inplace=True)

###### Calculate Zone features ######
_, _, Zones = to_timeseries_fast(df_duration, duration, feature_name='Zone')

feature_list = ['avg_zone', 'dz_resident_times', 'dz_resident_persistences', 'slz_resident_times',
                'slz_resident_persistences', 'dlz_resident_times', 'dlz_resident_persistences']
Zone_func = ZoneSignal(Zones)
df_zone = Zone_func.extract_features(feature_list)

df_inter = pd.concat([df_quality_features, df_inter_FDC, df_inter_T, df_inter_DZ, df_inter_LZ, df_inter_Core, df_zone], axis=1)


# ############# Normalized timeseries #############
#
# norm_FDC_distances = normalize_timeseries(FDC_distances)
# norm_FDC_overlap = normalize_timeseries(FDC_overlap)
#
# feature_list = ['total_distance', 'max_distance', 'peak_to_peak', 'distance_slopes', \
#         'RMSs', 'crestfactor', 'formfactor', 'pulseindicator', 'variance', 'cov', 'skewness', 'kurtosis', 'ngaussalpha',
#         #'autocorr', 'partial_autocorr'
#                 ]
#
# norm_FDC_dist = DistanceSignal(norm_FDC_distances)
# df_distance_norm = norm_FDC_dist.extract_features(feature_list, tau_limit=3)
#
# feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
#
# norm_FDC_over = OverlapSignal(norm_FDC_overlap)
# df_overlap_norm = norm_FDC_over.extract_features(feature_list)
#
# df_inter_FDC_norm = pd.concat([df_distance_norm, df_overlap_norm], axis=1)
# for column in df_inter_FDC_norm.columns:
#     df_inter_FDC_norm.rename(columns={column: 'norm_FDC_' + column}, inplace=True)
#
# norm_T_distances = normalize_timeseries(T_distances)
# norm_T_overlap = normalize_timeseries(T_overlap)
#
# feature_list = ['total_distance', 'max_distance', 'peak_to_peak', 'distance_slopes', \
#         'RMSs', 'crestfactor', 'formfactor', 'pulseindicator', 'variance', 'cov', 'skewness', 'kurtosis', 'ngaussalpha',
#         #'autocorr', 'partial_autocorr'
#                 ]
#
# norm_T_dist = DistanceSignal(norm_T_distances)
# df_distance_norm = norm_T_dist.extract_features(feature_list, tau_limit=3)
#
# feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
#
# norm_T_over = OverlapSignal(norm_T_overlap)
# df_overlap_norm = norm_T_over.extract_features(feature_list)
#
# df_inter_T_norm = pd.concat([df_distance_norm, df_overlap_norm], axis=1)
# for column in df_inter_T_norm.columns:
#     df_inter_T_norm.rename(columns={column: 'norm_T_' + column}, inplace=True)
#
# df_inter_norm = pd.concat([df_inter_FDC_norm, df_inter_T_norm], axis=1)
#
# df_interaction = pd.concat([df_inter, df_inter_norm], axis=1)

df_inter.to_parquet(path + 'interaction_features_%s.parquet'%duration)
df_inter.to_csv(path + 'interaction_features_%s.csv'%duration, index=False)


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\\'
df_motility = pd.read_parquet(path + 'motility_features_20.parquet')
df_inter = pd.read_parquet(path + 'interaction_features_20.parquet')

df_all = pd.concat([df_motility, df_inter], axis=1)

df_all.to_parquet(path + 'all_features_%s_all.parquet'%duration)
df_all.to_csv(path + 'all_features_%s_all.csv'%duration, index=False)












########################################################################################################
### 7. Calculate latent vectors
from dnn.autoencoders import Temporal_Conv1D_2D
from utils.traj_utils import *
from tensorflow.keras import models, optimizers

duration=20
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/'
df_duration = pd.read_parquet(path + 'GCB_traj_duration_%s.parquet' %duration)

df_duration_inhibit = pd.read_parquet(path + 'inhibit_traj_duration_%s.parquet' %duration)
df_duration_ctrl = pd.read_parquet(path + 'traj_duration_%s.parquet' %duration)
df_duration = pd.concat([df_duration_ctrl, df_duration_inhibit], axis=0).reset_index(drop=True)

traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration, duration=duration, feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])

traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration, duration=duration, feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z',
                                                                                                               'Shortest_Distance_to_Surfaces_Surfaces=FDC',
                                                                                                               'Shortest_Distance_to_Surfaces_Surfaces=T-cell'])
X_train = dict_to_array(trajectories)

# df_traj = rotated_trajectories.reshape(rotated_trajectories.shape[0]*rotated_trajectories.shape[1],3)
# df_traj = pd.DataFrame(df_traj, columns=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])
# df = pd.concat([df_traj, df], axis=1)
# df.to_csv(path + 'GCB_traj_duration_%s.csv' %duration, index=False)
# df.to_parquet(path + 'GCB_traj_duration_%s.parquet'%duration)

model = Temporal_Conv1D_2D(duration=duration, coor_dim=X_train.shape[2], dimension=128)
result = model.fit(X_train, X_train, batch_size=256, epochs=10000, verbose=1, validation_split=0.1, shuffle=True)
########################### Saving & Loading model ################################
model.save('saved_model/Temporal_Conv1D_2D_FDC_Tfh_interaction_10000epochs')
#ls saved_model

model = models.load_model('saved_model/Temporal_Conv1D_2D_10000epochs_with_inhibition', compile = False)
model.compile(loss='mse', optimizer=optimizers.Adadelta(learning_rate=0.1))
result = model.fit(X_train, X_train, batch_size=512, epochs=1, verbose=1, validation_split=0.1, shuffle=True)
####################################################################################

########################### Plot errors of model  ################################
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
t = fig.suptitle('Performance', fontsize=12)
fig.subplots_adjust(top=0.85,wspace=0.3)

max_epoch = len(result.history['accuracy']) # 25(epoch 수)
epoch_list = list(range(1,max_epoch+1)) # range(1,26) = 1~25

ax1.plot(epoch_list, result.history['accuracy'], label = 'training accuracy')
ax1.plot(epoch_list, result.history['val_accuracy'], label = 'validation accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 1000))
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.set_title('accuracy test')
ax1.legend(loc='best')

ax2.plot(epoch_list, result.history['loss'], label = 'training loss')
ax2.plot(epoch_list, result.history['val_loss'], label = 'validation loss')
ax2.set_xticks(np.arange(1, max_epoch, 1000))
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.set_title('loss test')
ax2.legend(loc='best')
# training accuracy(training data) = 매우 높아짐, but validation accuracy(test data) = 높지않음  ------> Overfitting
# 이 때는 epoch number 증가, training images 증가 필요
plt.savefig(path+'with_FDC_Tfh_interaction_loss_10000.png')
####################################################################################


########################### Draw reconstructed trajectories  ################################
pred = model.predict(X_train)
recons_trajectories = array_to_dict(pred)
rotated_trajectories = array_to_dict(rotated_trajectories)

draw_3D_trajectory(path, rotated_trajectories, folder_name='original_trajectory',idx_range=None, matplotlib_plot=False)
draw_3D_trajectory(path, recons_trajectories, folder_name='reconstructed_trajectory', idx_range=None, matplotlib_plot=False)

########################### Extract latent vectors  ################################
bottleneck = models.Model(inputs=model.inputs, outputs=model.layers[64].output)
lvs = bottleneck.predict(X_train)
df_lv = pd.DataFrame(lvs, columns=[ 'lv_%s' % str(i) for i in range(0,lvs.shape[1]) ])

df_lv.to_parquet(path + 'GCB_with_interaction_latent_vector_%s.parquet'%duration)
df_lv.to_csv(path + 'GCB_with_interaction_latent_vector_%s.csv'%duration, index=False)

########################### Combine all dataframe ################################
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Cornell LN Spleen\Analysis\feature_csvs\\'
df_lv = pd.read_parquet(path+'inhibit_latent_vector_%s.parquet'%duration)
df_motility = pd.read_parquet(path+'inhibit_motility_features_%s.parquet'%duration)
df_interaction = pd.read_parquet(path+'inhibit_interaction_features_%s.parquet'%duration)
df_all = pd.concat([df_lv, df_motility, df_interaction], axis=1)

df_all.to_parquet(path + 'inhibit_all_features_%s.parquet'%duration)
df_all.to_csv(path + 'inhibit_all_features_%s.csv'%duration, index=False)





########################################################################################################
### 8. Classification of mt & WT / WT & Tfh
from dnn.classification import GC_classifier
from utils.traj_utils import *
from tensorflow.keras import models, optimizers

duration=20
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/'
df_duration = pd.read_parquet(path + 'GCB_traj_duration_%s.parquet' %duration)
#df_duration = pd.read_parquet(path + 'no_MT_traj_duration_%s.parquet' %duration)


traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration, duration=duration, feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z',
                                                                                                               'Shortest_Distance_to_Surfaces_Surfaces=FDC',
                                                                                                               'Shortest_Distance_to_Surfaces_Surfaces=T-cell'])

# traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration, duration=duration, feature_name=['Rotated_X', 'Rotated_Y', 'Rotated_Z',
#                                                                                                                'Shortest_Distance_to_Surfaces_Surfaces=FDC'])

X_train = dict_to_array(trajectories)

df_labels = reduced_label_for_overlapped_volume(df_duration, duration=duration)
y = df_labels['Type']
print('Classes:', pd.unique(y))
y = y.replace(list(pd.unique(y)), [i for i in range(pd.unique(y).shape[0])])
y_train = np.array(y)
print('Number of classes:', np.unique(y).size)

model = GC_classifier(duration, coor_dim=X_train.shape[2], n_classes=np.unique(y).size)
# df_traj = rotated_trajectories.reshape(rotated_trajectories.shape[0]*rotated_trajectories.shape[1],3)
# df_traj = pd.DataFrame(df_traj, columns=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])
# df = pd.concat([df_traj, df], axis=1)
# df.to_csv(path + 'GCB_traj_duration_%s.csv' %duration, index=False)
# df.to_parquet(path + 'GCB_traj_duration_%s.parquet'%duration)

result = model.fit(X_train, y_train, batch_size=256, epochs=10000, verbose=1, validation_split=0.1, shuffle=True)

########################### Saving & Loading model ################################
model.save('saved_model/GCB_classification_100000epochs')  # AssertionError: Tried to export a function which references an 'untracked' resource.
# Solution: This happens when model includes custom objects (tf.Variable, subclassed layer ...). Must use get_config() method.
#ls saved_model

model = models.load_model('saved_model/GCB_classification_10000epochs', compile = False)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adadelta(learning_rate=0.1))
result = model.fit(X_train, y_train, batch_size=256, epochs=10, verbose=1, validation_split=0.1, shuffle=True)
####################################################################################

########################### Plot errors of model  ################################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
t = fig.suptitle('Performance', fontsize=12)
fig.subplots_adjust(top=0.85,wspace=0.3)

max_epoch = len(result.history['accuracy']) # 25(epoch 수)
epoch_list = list(range(1,max_epoch+1)) # range(1,26) = 1~25

ax1.plot(epoch_list, result.history['accuracy'], label = 'training accuracy')
ax1.plot(epoch_list, result.history['val_accuracy'], label = 'validation accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 1000))
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.set_title('accuracy test')
ax1.legend(loc='best')

ax2.plot(epoch_list, result.history['loss'], label = 'training loss')
ax2.plot(epoch_list, result.history['val_loss'], label = 'validation loss')
ax2.set_xticks(np.arange(1, max_epoch, 1000))
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.set_title('loss test')
ax2.legend(loc='best')
# training accuracy(training data) = 매우 높아짐, but validation accuracy(test data) = 높지않음  ------> Overfitting
# 이 때는 epoch number 증가, training images 증가 필요
plt.savefig(path+'no_MT_classification_loss_10000.png')
####################################################################################

########################### Extract latent vectors  ################################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'
weights = models.Model(inputs=model.inputs, outputs=model.get_layer('weights').output)
corr = weights.predict(X_train)
np.save(path+'no_MT_corr.npy', corr)

bottleneck = models.Model(inputs=model.inputs, outputs=model.get_layer('latent_vector').output)
lvs = bottleneck.predict(X_train)
df_lv = pd.DataFrame(lvs, columns=[ 'lv_%s' % str(i) for i in range(0,lvs.shape[1]) ])

df_lv.to_parquet(path + 'no_MT_classification_latent_vector_%s.parquet'%duration)
df_lv.to_csv(path + 'no_MT_classification_latent_vector_%s.csv'%duration, index=False)









