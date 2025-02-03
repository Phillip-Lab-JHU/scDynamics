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

import os
import pandas as pd
from tqdm import tqdm

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\Xihe_collab\\'
excel_file_name = 'Motility Data'

exp_groups = next(os.walk(path))[1] #['Exp1', 'Exp2', 'Exp3']
excel_file = pd.DataFrame()
for exp_group in tqdm(exp_groups):  # each Exp1, Exp2, Exp3
    exp_folders = next(os.walk(path + exp_group))[1]  # ['2-Bad-D09-A1-ZT1-73-160-FOV230-256px_Statistics', '4-Good-D10-B1-ZT1-45-132-FOV230-256px_Statistics', ...]

    df3 = pd.DataFrame()
    for exp_folder in exp_folders: # each '2-Bad-D09-A1-ZT1-73-160-FOV230-256px_Statistics', ...
        file_names = next(os.walk(path + exp_group + '/' + exp_folder))[2] # ['macrophage', 'mt B-cell', 'T-cell', 'wt B-cell']
        #drift_file_name = next(os.walk(path + exp_group + '/' + exp_folder))[2][0]
        #drift_file = pd.read_csv(path + exp_group + '/' + exp_folder + '/' + drift_file_name, skiprows=3)
        print(exp_folder)
        df2 = pd.DataFrame()
        for l, file_name in enumerate(file_names):  # for each feature Excel file

            import chardet
            with open(path + exp_group + '/' + exp_folder + '/' + file_name, 'rb') as f:
                result = chardet.detect(f.read())
                print(result)

            df_temp = pd.read_csv(path + exp_group + '/' + exp_folder + '/' + file_name, skiprows=2)

            df2 = df_temp[['Position X', 'Position Y', 'Position Z', 'Time', 'TrackID']]
            df2['Type'] = exp_folder
            df2['Exp'] = exp_group
            df2['Video'] = df2['Type'].astype(str)+'_'+df2['Exp'].astype(str)

        df3 = pd.concat([df3, df2])
    excel_file = pd.concat([excel_file, df3])

excel_file = excel_file.reset_index(drop=True)

excel_file.to_csv(path + excel_file_name + '.csv', index=False)
excel_file.to_parquet(path + excel_file_name + '.parquet')


########################################################################################################
###1. From segmentation from Imaris, you combine all excel files into one excel file using imaris.py
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\Xihe_collab\\'
excel = pd.read_parquet(path + 'Motility Data.parquet')

########################################################################################################
### 2. to_trajectory_duration to sort the dataframe by cells and fixed duration
duration = 22
df = pd.DataFrame()
for idx, video in enumerate(np.unique(excel['Video'])):
    print(idx, video)
    excel_temp = excel[excel['Video']==video].reset_index(drop=True)
    df_duration_temp = to_trajectory_duration(excel_temp, duration=duration, condition_name='Type', frame_name='Time', label_name='TrackID', verbose=False)
    #df_duration_temp = to_trajectory_variable_duration(excel_temp, min_duration=duration, condition_name='Type', label_name='Label')
    df = pd.concat([df, df_duration_temp], axis=0)
df = df.reset_index(drop=True)

df.to_csv(path + 'traj_duration_%s.csv' % duration, index=False)
df.to_parquet(path + 'traj_duration_%s.parquet' % duration)

traj_list, trajectories_array, trajectories = to_timeseries_fast(df, duration=duration, feature_name=['Position X', 'Position Y', 'Position Z'])
rotated_trajectories = register_traj_disp(trajectories)
rotated_trajectories = dict_to_array(rotated_trajectories)
df_traj = rotated_trajectories.reshape(rotated_trajectories.shape[0] * rotated_trajectories.shape[1], 3)
df_traj = pd.DataFrame(df_traj, columns=['Rotated_X', 'Rotated_Y', 'Rotated_Z'])
df_duration = pd.concat([df_traj, df], axis=1)
df_duration.to_csv(path + 'traj_duration_%s.csv' % duration, index=False)
df_duration.to_parquet(path + 'traj_duration_%s.parquet' % duration)

########################################################################################################
### 3. Calculate instantaneous features(inst speed, angle and directionality wrt FDC and T cell)
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\Xihe_collab\\'
duration = 22
df_duration = pd.read_parquet(path + 'traj_duration_%s.parquet' %duration)

df_duration = get_instant_movements(df_duration, duration=duration, time_unit=0.5, feature_name=['Position X', 'Position Y', 'Position Z'])
df_duration.to_csv(path + 'traj_duration_%s.csv'%duration, index=False)
df_duration.to_parquet(path + 'traj_duration_%s.parquet'%duration)


########################################################################################################
### 5. Calculate motility features(APRW + basic motility + morphodynamic)
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\Xihe_collab\\'
duration = 22
df_duration = pd.read_parquet(path + 'traj_duration_%s.parquet' %duration)

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
df_labels = df_labels[['TrackID', 'Time', 'pseudo_Time', 'pseudo_TrackID', 'Type', 'Video', 'Exp', 'Time_span', 'instant_speed', 'instant_angle']]
df_labels['traj_Label'] = df_labels['Video'].astype(str)+'_'+df_labels['Type'].astype(str)+'_'\
                         +df_labels['TrackID'].astype(str)

###### Concatenate all features ######
#df_motility = pd.concat([df_aprw, df_basic, df_aniso, df_inst_speed_features, df_inst_angle_features, df_morpho, df_labels], axis=1)
df_motility = pd.concat([df_aprw, df_basic, df_aniso, df_inst_speed_features, df_inst_angle_features, df_labels], axis=1)
a = df_motility.isnull().any()

#path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'
#path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Long term/GCB/'
df_motility.to_csv(path + 'motility_features_%s.csv'%duration, index=False)
df_motility.to_parquet(path + 'motility_features_%s.parquet'%duration)


############################## Kmeans clustering and UMAP #######################################

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\Xihe_collab\\'
df_features = pd.read_parquet(path+'motility_features_22.parquet')
df_duration = pd.read_parquet(path + 'traj_duration_22.parquet')

#df_features = df_features.drop(columns=['PC1', 'PC2', 'kmeans', 'pancreatic_effect', 'lung_effect', 'ovarian_effect'])
#df_duration = df_duration.drop(columns=['pancreatic_effect', 'lung_effect', 'ovarian_effect'])
# df_lv = pd.read_parquet(path+'classifier_latent_vector.parquet')
# #df_lv = pd.read_parquet(path+'latent_vector_31.parquet')
# df_lv = pd.concat([df_lv, df_features], axis=1)

df_features.columns.get_loc('inst_angle_cosine_similarity_entropies')
motility_data = df_features.iloc[:,8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)

m = Morphodynamics(df_features, 'umap')
cluster = m.get_cluster(pcs, n_clusters=8, cluster_type='kmeans')
df_features_ = df_features.copy()
df_features_['kmeans'] = cluster
m.evaluate_umap(df_features_, pcs, path, n_neighbors_list=[10, 20, 30, 40, 50, 60],
                      min_dist_list=[0.005, 0.01, 0.05, 0.1, 0.3, 0.5], condition_name ='kmeans')
umap = m.get_umap(pcs, 40, 0.3)
#m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
cluster = m.get_cluster(pcs, n_clusters=4, cluster_type='kmeans')
df = pd.concat([df_features, umap, cluster], axis=1)

df, replace_map = order_cluster_by_feature(df, cluster_name='kmeans', feature_name='avg_speed')

draw_umap_space(df, path, file_name='space_kmeans_ordered', condition_name='kmeans', label_name='TrackID',
                colors=cmc.batlow, dot_size=0.4, x_name='PC1', y_name='PC2')

duration=22
df.to_parquet(path + 'motility_features_%s.parquet'%duration)
df.to_csv(path + 'motility_features_%s.csv'%duration, index=False)

label_expanded = np.repeat(df['kmeans'], duration).reset_index(drop=True)
df_duration['kmeans'] = label_expanded

df_duration.to_parquet(path + 'traj_duration_%s.parquet'%duration)
df_duration.to_csv(path + 'traj_duration_%s.csv'%duration, index=False)