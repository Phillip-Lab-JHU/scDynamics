from utils.traj_utils import *
from utils.draw_utils import *
from utils.draw_utils import *
from features.basic_motility import BasicMotility
from features.aprw import APRW
from features.decomposed_motility import DecomposedMotility2D
from features.timeseries import Timeseries
import os
import pandas as pd
from tqdm import tqdm

########################################################################################################
###1. From segmentation from Imaris, you combine all excel files into one excel file using imaris.py
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\NiaLab_TcellTimelapse\\'
excel = pd.read_csv(path + 'track_File2_KR270_YummerMetastasis.tif.csv')
duration = 11

excel['Type'] = 'KR270'
df = to_trajectory_duration(excel, duration=duration, condition_name='Type', frame_name='frame', label_name='particle', verbose=False)
traj_list, trajectories_array, trajectories = to_timeseries_fast(df, duration=duration, feature_name=['x', 'y'])
reg_trajectories = register_traj_disp_reflection(trajectories)
reg_trajectories = dict_to_array(reg_trajectories)

df_traj = reg_trajectories.reshape(reg_trajectories.shape[0] * reg_trajectories.shape[1], 2)
df_traj = pd.DataFrame(df_traj, columns=['reg_x', 'reg_y' ])
df = pd.concat([df_traj, df], axis=1)

df_duration = get_instant_movements(df, duration=duration, time_unit=2, feature_name=['x', 'y'])

df_duration.to_csv(path + 'traj_duration_11.csv', index=False)
df_duration.to_parquet(path + 'traj_duration_11.parquet')


########################################################################################################
### 5. Calculate motility features(APRW + basic motility + morphodynamic)
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\NiaLab_TcellTimelapse\\'
df_duration = pd.read_parquet(path + 'traj_duration_11.parquet')
#df_duration['label'] = df_duration['type'].astype(str) + '_' +df_duration['id'].astype(str)
###### Make trajectory as dictionary form ######

duration = 11
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
df_motility.to_csv(path + 'motility_features.csv', index=False)
#df_motility.to_parquet(path + 'motility_features.parquet')

############################## Kmeans clustering and UMAP #######################################

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\NiaLab_TcellTimelapse\\'
df_features = pd.read_csv(path+'motility_features.csv')
df_duration = pd.read_parquet(path + 'traj_duration_11.parquet')

#df_features = df_features.drop(columns=['PC1', 'PC2', 'kmeans', 'pancreatic_effect', 'lung_effect', 'ovarian_effect'])
#df_duration = df_duration.drop(columns=['pancreatic_effect', 'lung_effect', 'ovarian_effect'])
# df_lv = pd.read_parquet(path+'classifier_latent_vector.parquet')
# #df_lv = pd.read_parquet(path+'latent_vector_31.parquet')
# df_lv = pd.concat([df_lv, df_features], axis=1)

df_features.columns.get_loc('inst_angle_pulseindicator')
motility_data = df_features.iloc[:,8:90].drop(['speed_distribution_x', 'speed_distribution_y',],axis=1)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)

m = Morphodynamics(df_features, 'umap')
cluster = m.get_cluster(pcs, n_clusters=5, cluster_type='kmeans')
df_features_ = df_features.copy()
df_features_['kmeans'] = cluster
m.evaluate_umap(df_features_, pcs, path, n_neighbors_list=[10, 20, 30, 40, 50, 60],
                      min_dist_list=[0.005, 0.01, 0.05, 0.1, 0.3, 0.5], condition_name ='kmeans')
umap = m.get_umap(pcs, 50, 0.5)
#m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
cluster = m.get_cluster(pcs, n_clusters=4, cluster_type='kmeans')
df = pd.concat([df_features, umap, cluster], axis=1)

df, replace_map = order_cluster_by_feature(df, cluster_name='kmeans', feature_name='avg_speed')

draw_umap_space(df, path, file_name='space_kmeans_ordered', condition_name='kmeans', label_name='pseudo_particle',
                colors=cmc.batlow, dot_size=0.4, x_name='PC1', y_name='PC2')

duration=11
#df.to_parquet(path + 'motility_features_%s.parquet'%duration)
df.to_csv(path + 'motility_features.csv', index=False)

label_expanded = np.repeat(df['kmeans'], duration).reset_index(drop=True)
df_duration['kmeans'] = label_expanded

df_duration.to_parquet(path + 'traj_duration_%s.parquet'%duration)
df_duration.to_csv(path + 'traj_duration_%s.csv'%duration, index=False)