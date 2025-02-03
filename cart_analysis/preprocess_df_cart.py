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

########################################################################################################
###1. From segmentation from Imaris, you combine all excel files into one excel file using imaris.py
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
excel = pd.read_csv(path + 'traj_duration.csv')[['id', 'frame', 'x', 'y', 'type']]
duration = 31
traj_list, trajectories_array, trajectories = to_timeseries_fast(excel, duration=31, feature_name=['x', 'y'])
reg_trajectories = register_traj_disp_reflection(trajectories)
reg_trajectories = dict_to_array(reg_trajectories)

df_traj = reg_trajectories.reshape(reg_trajectories.shape[0] * reg_trajectories.shape[1], 2)
df_traj = pd.DataFrame(df_traj, columns=['reg_x', 'reg_y' ])
df = pd.concat([df_traj, excel], axis=1)

duration = 31
df_duration = get_instant_movements(df, duration=duration, time_unit=2, feature_name=['x', 'y'])

df_duration.to_csv(path + 'traj_duration.csv', index=False)
df_duration.to_parquet(path + 'traj_duration.parquet')

########################################################################################################
### 5. Calculate motility features(APRW + basic motility + morphodynamic)
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
df_duration = pd.read_parquet(path + 'traj_duration.parquet')
df_duration['label'] = df_duration['type'].astype(str) + '_' +df_duration['id'].astype(str)
###### Make trajectory as dictionary form ######

duration = 31
traj_list, _, trajectories = to_timeseries_fast(df_duration, duration=duration, feature_name=['x', 'y'])

###### Calculate basic motility features ######
feature_list = ['total_distance', 'avg_speed', 'max_speed', 'min_speed', 'net_distance',  'progressivity', 'alphas',
                'total_angle', 'avg_angle', 'max_angle', 'min_angle',
                'displ_variance', 'displ_cov','displ_skewness', 'displ_kurtosis', 'displ_ngaussalpha', 'displ_gini',
                'angle_variance', 'angle_cov', 'angle_skewness', 'angle_kurtosis', 'angle_ngaussalpha', 'angle_gini',
                 'displ_autocorr', 'displ_partial_autocorr', 'angle_autocorr', 'angle_partial_autocorr',
                #'displ_hurst_RS', 'angle_hurst_RS', 'msds',
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

###### Calculate instantaneous speed timeseries features ######
df_inst_speed = df_duration[df_duration['frame']!=1].reset_index(drop=True)
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
df_inst_angle = df_duration[(df_duration['frame']!=1)&(df_duration['frame']!=2)].reset_index(drop=True)
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

###### Concatenate all features ######
#df_motility = pd.concat([df_aprw, df_basic, df_aniso, df_morpho, df_labels], axis=1)
df_motility = pd.concat([df_aprw, df_basic, df_aniso, df_inst_speed_features, df_inst_angle_features, df_labels], axis=1)
nan_rows = df_motility[df_motility.isna().any(axis=1)]
df_motility = df_motility.dropna().reset_index(drop=True)

a = df_motility.isnull().any()

#path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Long term/GCB/'
df_motility.to_csv(path + 'motility_features_nan_removed.csv', index=False)
df_motility.to_parquet(path + 'motility_features_nan_removed.parquet')

df_duration_removed = remove_trajs_condition(df_duration, duration=duration, remove_idxs=list(nan_rows.index))
df_duration_removed.to_csv(path + 'traj_duration_nan_removed.csv', index=False)
df_duration_removed.to_parquet(path + 'traj_duration_nan_removed.parquet')

########################################################################################################
### 7. Calculate latent vectors
from dnn.autoencoders import Temporal_Conv1D_2D, set_duration_for_autoencoder
from utils.traj_utils import *
from tensorflow.keras import models, optimizers


path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
df_duration_removed = pd.read_parquet(path + 'traj_duration_nan_removed.parquet')
duration=31

traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_removed, duration=duration, feature_name=['x', 'y'])
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
rotated_trajectories = register_traj_disp(trajectories)
rotated_trajectories = dict_to_array(rotated_trajectories)
X_train = rotated_trajectories

model = Temporal_Conv1D_2D(duration=new_duration, coor_dim=2, dimension=128)
result = model.fit(X_train, X_train, batch_size=256, epochs=10000, verbose=1, validation_split=0.1, shuffle=True)
########################### Saving & Loading model ################################
model.save('saved_model/Temporal_Conv1D_2D_10000epochs_CART')
#ls saved_model

model = models.load_model('saved_model/Temporal_Conv1D_2D_10000epochs_CART', compile = False)
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

df_lv.to_parquet(path + 'latent_vector_%s.parquet'%duration)
df_lv.to_csv(path + 'latent_vector_%s.csv'%duration, index=False)


########################### Combine all dataframe ################################
df_lv = pd.read_parquet(path+'latent_vector_%s.parquet'%duration)
df_motility = pd.read_parquet(path+'motility_features_nan_removed.parquet')

df_all = pd.concat([df_lv, df_motility], axis=1)

df_all.to_parquet(path + 'all_features.parquet')
df_all.to_csv(path + 'all_features.csv', index=False)





