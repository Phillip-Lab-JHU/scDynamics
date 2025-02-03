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
from features.decomposed_motility import DecomposedMotility2D
from features.interaction import DistanceSignal, OverlapSignal
from features.timeseries import Timeseries
from features.directionality import Directionality
import os
import pandas as pd
from tqdm import tqdm

# ########################################################################################################
# ###1. From segmentation from Imaris, you combine all excel files into one excel file using imaris.py
# path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\csv_files\\'
# csvs = next(os.walk(path))[2]
# duration = 30
# # um_per_pix = 0.568
#
# excel = pd.DataFrame()
# for csv in tqdm(csvs):
#     excel_temp = pd.read_csv(path + csv)
#     excel_temp = excel_temp.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # strip() removes any leading, and trailing whitespaces on the elements
#     excel_temp.rename(columns=lambda x: x.replace(' ', ''), inplace=True)  # Removes whitespace on the column names
#     df_duration_temp = to_trajectory_duration(excel_temp, duration=duration, condition_name='Patient', frame_name='frame', label_name='particle', verbose=False)
#     excel = pd.concat([excel, df_duration_temp], axis=0)
#
# excel = excel.reset_index(drop=True)
#
# traj_list, trajectories_array, trajectories = to_timeseries_fast(excel, duration=duration, feature_name=['x', 'y'])
# reg_trajectories = register_traj_disp_reflection(trajectories)
# reg_trajectories = dict_to_array(reg_trajectories)
#
# df_traj = reg_trajectories.reshape(reg_trajectories.shape[0] * reg_trajectories.shape[1], 2)
# df_traj = pd.DataFrame(df_traj, columns=['reg_x', 'reg_y' ])
# df = pd.concat([df_traj, excel], axis=1)
#
# df_duration = get_instant_movements(df, duration=duration, time_unit=2, feature_name=['x', 'y'])
# df_duration = get_instant_direction(df_duration, duration=duration, thresh=0.1, feature_name='shortest_distance')
# df_duration = get_instant_direction(df_duration, duration=duration, thresh=0.1, feature_name='avg_shortest_distance')
#
# # df_duration['x'] = df_duration['x']*um_per_pix
# # df_duration['y'] = df_duration['y']*um_per_pix
# # df_duration['reg_x'] = df_duration['reg_x']*um_per_pix
# # df_duration['reg_y'] = df_duration['reg_y']*um_per_pix
#
# path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\\'
# df_duration.to_csv(path + 'traj_duration_30.csv', index=False)
# df_duration.to_parquet(path + 'traj_duration_30.parquet')

########################################################################################################
### 5. Calculate motility features(APRW + basic motility + morphodynamic)
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Anshika motility project\\'
df_duration = pd.read_parquet(path+'cleaned_traj_duration_96.parquet')
###### Make trajectory as dictionary form ######

duration = 96
traj_list, _, trajectories = to_timeseries_fast(df_duration, duration=duration, feature_name=['X', 'Y'])

###### Calculate basic motility features ######
feature_list = ['total_distance', 'avg_speed', 'max_speed', 'min_speed', 'net_distance',  'progressivity', 'alphas',
                'total_angle', 'avg_angle', 'max_angle', 'min_angle',
                'displ_variance', 'displ_cov','displ_skewness', 'displ_kurtosis', 'displ_ngaussalpha', 'displ_gini',
                'angle_variance', 'angle_cov', 'angle_skewness', 'angle_kurtosis', 'angle_ngaussalpha', 'angle_gini',
                'msds', 'displ_autocorr', 'displ_partial_autocorr', 'angle_autocorr', 'angle_partial_autocorr',
                #'displ_hurst_RS', 'angle_hurst_RS',
                ]

basic_motil = BasicMotility(trajectories, time_unit=2, feature_list=feature_list)
#basic_motil.plot_msd_alpha(path)
#basic_motil.plot_rotated_trajectories(path)
#basic_motil.plot_original_trajectories(path)

df_basic = basic_motil.extract_features(tau_limit=3)

k = df_basic.isnull().any()
print(np.where(np.isnan(df_basic['alphas'])))

###### Calculate anisotropic motility features ######
ani_motil = DecomposedMotility2D(trajectories, time_unit=2)
feature_list = ['avg_speed_x', 'max_speed_x', 'min_speed_x', 'net_distance_x', 'progressivity_x',
                'avg_speed_y', 'max_speed_y', 'min_speed_y', 'net_distance_y', 'progressivity_y',
                'exy_max', 'exy_total',
                'msd_x', 'msd_y',
                #'alpha_x', 'alpha_y',
                'displ_variance_x', 'displ_cov_x', 'displ_skewness_x', 'displ_kurtosis_x', 'displ_ngaussalpha_x',
                'displ_variance_y', 'displ_cov_y', 'displ_skewness_y', 'displ_kurtosis_y', 'displ_ngaussalpha_y',
                'displ_autocorr_x', 'displ_autocorr_y',
                #'displ_partial_autocorr_x', 'displ_partial_autocorr_y',
                #'displ_hurst_RS_x', 'displ_hurst_RS_y', 'displ_hurst_RS_z'
                ]
#df_aniso = ani_motil(feature_list, tau_limit=3)
df_aniso = ani_motil.extract_features(feature_list=feature_list, tau_limit=3)
k = df_aniso.isnull().any()

###### Calculate APRW features ######
aprw = APRW()
df_aprw = aprw.get_APRW(trajectories, dt=2, max_speed=np.max(df_basic['avg_speed']))

###### Calculate morphodynamic features ######
morpho_trajectories = morpho_trajectory(df_duration,
                                        features = ['area', 'perimeter', 'convex_area', 'solidity', 'eccentricity',
                                                    'equivalent_diameter', 'extent', 'major_axis_length', 'minor_axis_length',
                                                    'aspect_ratio', 'elongation', 'compactness', 'roundness', 'circularity', 'rectangularity'],
                                        duration=duration, dim=2)
feature_list = ['avg_speed', 'max_speed', 'min_speed', 'displ_autocorr', 'displ_cov',
                'avg_angle', 'max_angle', 'min_angle', 'progressivity', 'angle_cov',
                #'displ_variance', ,'displ_skewness', 'displ_kurtosis', 'displ_ngaussalpha',
                #'angle_variance', 'angle_skewness', 'angle_kurtosis', 'angle_ngaussalpha',
                #'msd', , 'displ_partial_autocorr', 'angle_autocorr', 'angle_partial_autocorr',
                #'displ_hurst_RS', 'angle_hurst_RS','net_distance','alpha',
                ]
morpho_basic = BasicMotility(morpho_trajectories, time_unit=2, feature_list=feature_list)
df_morpho = morpho_basic.extract_features(tau_limit=3)
#df_morpho = df_morpho.drop(['angle_distribution', 'speed_distribution'], axis=1)
for column in df_morpho.columns:
    df_morpho.rename(columns={column:'morpho_'+column}, inplace=True)

###### Calculate instantaneous speed timeseries features ######
df_inst_speed = df_duration[df_duration['Frame #']!=1].reset_index(drop=True)
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
df_inst_angle = df_duration[(df_duration['Frame #']!=1)&(df_duration['Frame #']!=2)].reset_index(drop=True)
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
df_labels_part = df_labels[['reg_x', 'reg_y', 'X', 'Y', 'Object #', 'Frame #', 'Age', 'Cell Line', 'Senescence', 'Gender', 'instant_speed', 'instant_angle']]
df_labels_part['pseudo_label'] = df_labels_part['Cell Line'].astype(str)+'_'+df_labels_part['Gender'].astype(str)+'_'\
                         +df_labels_part['Object #'].astype(str)

######### Concatenate all features #########

#df_motility = pd.concat([df_aprw, df_basic, df_aniso, df_morpho, df_labels], axis=1)
df_motility = pd.concat([df_aprw, df_basic, df_aniso, df_inst_speed_features, df_inst_angle_features, df_labels_part], axis=1)
a = df_motility.isnull().any()
nan_row_idxs = df_motility[df_motility.isnull().any(axis=1)].index

df_motility_dropped = df_motility.drop(nan_row_idxs, axis=0)
df_motility_dropped = df_motility_dropped.reset_index(drop=True)

df_duration_dropped = remove_trajs_condition(df_duration, duration, nan_row_idxs)

df_motility_dropped.to_csv(path + 'motility_features_96.csv', index=False)
df_motility_dropped.to_parquet(path + 'motility_features_96.parquet')

df_duration_dropped.to_csv(path + 'traj_duration_96.csv', index=False)
df_duration_dropped.to_parquet(path + 'traj_duration_96.parquet')

########################################################################################################
### 6. Calculate interaction properties

###### Calculate directional quality features ######
feature_list = ['approach_times', 'approach_persistences', 'departure_times', 'departure_persistences', 'stay_times', 'stay_persistences']

_, _, shortest_distance = to_timeseries_fast(df_duration, duration=duration, feature_name='quality_shortest_distance')

shortest_direction_features = Directionality(shortest_distance)
df_shortest_direction_features = shortest_direction_features.extract_features(feature_list=feature_list)
for column in df_shortest_direction_features.columns:
    df_shortest_direction_features.rename(columns={column:'nearest_'+column}, inplace=True)

_, _, avg_shortest_distance = to_timeseries_fast(df_duration, duration=duration, feature_name='quality_avg_shortest_distance')
avg_direction_features = Directionality(avg_shortest_distance)
df_avg_direction_features = avg_direction_features.extract_features(feature_list=feature_list)
for column in df_avg_direction_features.columns:
    df_avg_direction_features.rename(columns={column:'group_'+column}, inplace=True)

df_direction_features = pd.concat([df_shortest_direction_features, df_avg_direction_features], axis=1)


############################ Calculate Quantitative features ############################
_, _, shortest_distance = to_timeseries_fast(df_duration, duration, feature_name='shortest_distance')
_, _, avg_shortest_distance = to_timeseries_fast(df_duration, duration, feature_name='avg_shortest_distance')
_, _, n_neighbors = to_timeseries_fast(df_duration, duration, feature_name='n_neighbors')

df_duration_diff = df_duration[(df_duration['pseudo_frame']!=0)].reset_index(drop=True)
_, _, diff_shortest_distance = to_timeseries_fast(df_duration_diff, duration-1, feature_name='diff_shortest_distance')
_, _, diff_avg_shortest_distance = to_timeseries_fast(df_duration_diff, duration-1, feature_name='diff_avg_shortest_distance')

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
                'autocorr',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr'
                ]
shortest_dist = DistanceSignal(shortest_distance)
df_shortest = shortest_dist.extract_features(feature_list, tau_limit=3)
for column in df_shortest.columns:
    df_shortest.rename(columns={column:'nearest_distance_'+column}, inplace=True)

diff_shortest_dist = DistanceSignal(diff_shortest_distance)
df_diff_shortest = diff_shortest_dist.extract_features(feature_list, tau_limit=3)
for column in df_diff_shortest.columns:
    df_diff_shortest.rename(columns={column:'diff_nearest_distance_'+column}, inplace=True)

avg_shortest_dist = DistanceSignal(avg_shortest_distance)
df_avg = avg_shortest_dist.extract_features(feature_list, tau_limit=3)
for column in df_avg.columns:
    df_avg.rename(columns={column:'group_distance_'+column}, inplace=True)

diff_avg_shortest_dist = DistanceSignal(diff_avg_shortest_distance)
df_diff_avg = diff_avg_shortest_dist.extract_features(feature_list, tau_limit=3)
for column in df_diff_avg.columns:
    df_diff_avg.rename(columns={column:'diff_group_distance_'+column}, inplace=True)


feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
                #'autocorr',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr'
                ]
n_neighbors_dist = DistanceSignal(n_neighbors)
df_n_neighbors = n_neighbors_dist.extract_features(feature_list, tau_limit=3)

for column in df_n_neighbors.columns:
    df_n_neighbors.rename(columns={column:'n_neighbors_'+column}, inplace=True)

df_inter = pd.concat([df_shortest, df_avg, df_n_neighbors, df_diff_shortest, df_diff_avg], axis=1)

df_interaction = pd.concat([df_direction_features, df_inter], axis=1)
a = df_interaction.isnull().any()

df_interaction.to_parquet(path + 'interaction_features_%s.parquet'%duration)
df_interaction.to_csv(path + 'interaction_features_%s.csv'%duration, index=False)

df_all = pd.concat([df_motility, df_interaction], axis=1)

df_all.to_parquet(path + 'all_features_%s.parquet'%duration)
df_all.to_csv(path + 'all_features_%s.csv'%duration, index=False)

######### motility data postprocessing (remove non-moving cells) #########
thresh = 0.8 # (um/min)

fig, ax = plt.subplots()
counts, bins, patches = plt.hist(df_motility['max_speed'], bins=1000)
print('max freq of avg speed:', bins[np.argmax(counts)])
plt.savefig(path+'mas_speed histogram.png')
plt.close()
plt.clf()

fig, ax = plt.subplots()
counts, bins, patches = plt.hist(df_motility[df_motility['avg_speed']>=thresh]['avg_speed'], bins=1000)
print('max freq of avg speed:', bins[np.argmax(counts)])
plt.savefig(path+'avg_speed removed histogram.png')
plt.close()
plt.clf()

remove_rows = df_motility[df_motility['avg_speed']<thresh]
remove_idxs = list(remove_rows.index)
df_removed = remove_trajs_condition(df_duration, duration=duration, remove_idxs=remove_idxs)

df_removed.to_csv(path + 'traj_duration_removed_30.csv', index=False)
df_removed.to_parquet(path + 'traj_duration_removed_30.parquet')

#nan_rows = df_motility[df_motility.isna().any(axis=1)]
#df_motility = df_motility.dropna().reset_index(drop=True)
df_motility_removed = df_motility[df_motility['avg_speed']>=thresh].reset_index(drop=True)

df_motility_removed.to_csv(path + 'motility_features_removed_30.csv', index=False)
df_motility_removed.to_parquet(path + 'motility_features_removed_30.parquet')

########################################################################################################
### 7. Calculate latent vectors
from dnn.autoencoders import Temporal_Conv1D_2D, set_duration_for_autoencoder
from utils.traj_utils import *
from tensorflow.keras import models, optimizers

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Charles Post Optimizations\Low Density (25k cells)\0.5 Gel\analysis 01_27_24\\'
df_duration = pd.read_parquet(path + 'traj_duration_removed_30.parquet')
duration = 30


traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration, duration=duration, feature_name=['x', 'y'])
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
rotated_trajectories = register_traj_disp(trajectories)
#draw_3D_trajectory(path, trajectories, idx_range=range(0,30))
#draw_3D_trajectory(path, rotated_trajectories, idx_range=range(0,30))
rotated_trajectories = dict_to_array(rotated_trajectories)
X_train = rotated_trajectories

model = Temporal_Conv1D_2D(duration=new_duration, coor_dim=2, dimension=128)
result = model.fit(X_train, X_train, batch_size=256, epochs=10000, verbose=1, validation_split=0.1, shuffle=True)
########################### Saving & Loading model ################################
model.save('saved_model/Temporal_Conv1D_2D_10000epochs_monocyte')
#ls saved_model

model = models.load_model('saved_model/Temporal_Conv1D_2D_10000epochs_monocyte', compile = False)
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
plt.savefig(path+'loss_10000.png')
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

df_lv.to_parquet(path + 'latent_vector_removed_%s.parquet'%duration)
df_lv.to_csv(path + 'latent_vector_removed_%s.csv'%duration, index=False)


########################### Combine all dataframe ################################
df_lv = pd.read_parquet(path+'latent_vector_removed_%s.parquet'%duration)
df_motility = pd.read_parquet(path+'motility_features_removed_%s.parquet'%duration)
df_all = pd.concat([df_lv, df_motility], axis=1)

df_all.to_parquet(path + 'all_features_removed_%s.parquet'%duration)
df_all.to_csv(path + 'all_features_removed_%s.csv'%duration, index=False)





