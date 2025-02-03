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

"""Generates Data for Figure 7 Back project prediction with known features."""

import pandas as pd
from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast
from dnn.autoencoders import set_duration_for_autoencoder
import tensorflow as tf
from dnn.classification import scTRAIT, Temporal_Conv1D_2D_classifier

#################################### Training dataset ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df_frail = df[df['Type']=='Frail'].reset_index(drop=True)
df_duration_frail = df_duration[df_duration['Type']=='Frail'].reset_index(drop=True)

# Young and Old only
df = df[df['Type']=='Old'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']=='Old'].reset_index(drop=True)


duration = 30
traj_scale_data = df_duration.loc[:, ['reg_x', 'reg_y']]
traj_scaler = StandardScaler()
traj_scale_data = pd.DataFrame(traj_scaler.fit_transform(traj_scale_data), columns=['reg_x', 'reg_y'])

traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration,
                                                                 feature_name=['reg_x', 'reg_y'])
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
rotated_trajectories = dict_to_array(trajectories)
X_traj = rotated_trajectories

local_features = ['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance']
local_scale_data = df_duration.loc[:, local_features]
local_scaler = StandardScaler()
local_scale_data = pd.DataFrame(local_scaler.fit_transform(local_scale_data), columns=local_features)

traj_list, trajectories_array, trajectories = to_timeseries_fast(local_scale_data, duration=duration,
                                                                 feature_name=local_features)
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
rotated_trajectories = dict_to_array(trajectories)
X_local = rotated_trajectories

morphology_features = ['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length', 'minor_axis_length',
                       'aspect_ratio', 'circularity']
scale_data = df_duration.loc[:, morphology_features]
scaler = StandardScaler()
scale_data = pd.DataFrame(scaler.fit_transform(scale_data), columns=morphology_features)

traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration,
                                                                 feature_name=morphology_features)
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
rotated_trajectories = dict_to_array(trajectories)
X_morph = rotated_trajectories

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X_covar = le.fit_transform(np.array(df['Condition']))
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# X_covar = ohe.fit_transform( np.array(df_nonyoung['Condition']).reshape(-1,1) )

y= np.array(df['Age'])

from sklearn.model_selection import train_test_split
X_traj_train, X_traj_test_nonfrail, X_local_train, X_local_test_nonfrail, X_morph_train, X_morph_test_nonfrail, \
X_covar_train, X_covar_test_nonfrail, y_train, y_test_nonfrail, df_train, df_test_nonfrail = train_test_split(X_traj, X_local, X_morph, X_covar, y, df, test_size=0.2, random_state=0, stratify=y)
print(X_traj_train.shape, X_traj_test_nonfrail.shape, X_local_train.shape, X_local_test_nonfrail.shape,
      X_morph_train.shape, X_morph_test_nonfrail.shape, X_covar_train.shape, X_covar_test_nonfrail.shape, y_train.shape, y_test_nonfrail.shape)


############# frail testing set #############
traj_scale_data = df_duration.loc[:, ['reg_x', 'reg_y']]
traj_scaler = StandardScaler()
traj_scaler.fit(traj_scale_data)
traj_scale_data = df_duration_frail.loc[:, ['reg_x', 'reg_y']]
traj_scale_data = pd.DataFrame(traj_scaler.transform(traj_scale_data), columns=['reg_x', 'reg_y'])

traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration,
                                                                 feature_name=['reg_x', 'reg_y'])
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
# rotated_trajectories = register_traj_disp_reflection(trajectories)
rotated_trajectories = dict_to_array(trajectories)
X_traj_frail = rotated_trajectories

local_features = ['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance']
local_scale_data = df_duration.loc[:, local_features]
local_scaler = StandardScaler()
local_scaler.fit(local_scale_data)
local_scale_data = df_duration_frail.loc[:, local_features]
local_scale_data = pd.DataFrame(local_scaler.transform(local_scale_data), columns=local_features)

traj_list, trajectories_array, trajectories = to_timeseries_fast(local_scale_data, duration=duration,
                                                                 feature_name=local_features)
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
# rotated_trajectories = register_traj_disp_reflection(trajectories)
rotated_trajectories = dict_to_array(trajectories)
X_local_frail = rotated_trajectories

morphology_features = ['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length', 'minor_axis_length',
                       'aspect_ratio', 'circularity']
scale_data = df_duration.loc[:, morphology_features]
scaler = StandardScaler()
scaler.fit(scale_data)
scale_data = df_duration_frail.loc[:, morphology_features]
scale_data = pd.DataFrame(scaler.transform(scale_data), columns=morphology_features)

traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration,
                                                                 feature_name=morphology_features)
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
# rotated_trajectories = register_traj_disp_reflection(trajectories)
rotated_trajectories = dict_to_array(trajectories)
X_morph_frail = rotated_trajectories

le = LabelEncoder()
le.fit(np.array(df['Condition']))
X_covar_frail = le.transform(np.array(df_frail['Condition']))
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
y_frail = np.array(df_frail['Age'])


X_traj_test = np.concatenate((X_traj_test_nonfrail, X_traj_frail), axis=0)
X_local_test = np.concatenate((X_local_test_nonfrail, X_local_frail), axis=0)
X_morph_test = np.concatenate((X_morph_test_nonfrail, X_morph_frail), axis=0)
X_covar_test = np.concatenate((X_covar_test_nonfrail, X_covar_frail), axis=0)
y_test = np.concatenate((y_test_nonfrail, y_frail), axis=0)
df_test = pd.concat([df_test_nonfrail, df_frail], axis=0).reset_index(drop=True)


from tensorflow.keras import callbacks
stop_early = callbacks.EarlyStopping(monitor='loss', patience=30, min_delta=0.0001)

from dnn.classification import scTRAIT
sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=0)
result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=1000,
                     verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
sctrait.save('saved_model/monocyte_project/regression interpretation scTRAIT', save_format='tf')


#################################### Get predicted cellular age ####################################
from tensorflow.keras import models
sctrait = tf.keras.models.load_model('saved_model/monocyte_project/regression interpretation scTRAIT', compile=True)

y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
df_test['pred'] = y_pred.flatten()

#################################### Extract pre perturbation vectors ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis\Figure7. Interpretation\\'

traj = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('emb_traj').output)
traj_vector = traj.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])

local = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('emb_local').output)
local_vector = local.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])

morph = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('emb_morph').output)
morph_vector = morph.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])


df_traj = pd.DataFrame(traj_vector, columns=[ 'pre_traj_%s' % str(i) for i in range(0,traj_vector.shape[1]) ])
df_local = pd.DataFrame(local_vector, columns=[ 'pre_local_%s' % str(i) for i in range(0,local_vector.shape[1]) ])
df_morph = pd.DataFrame(morph_vector, columns=[ 'pre_morph_%s' % str(i) for i in range(0,morph_vector.shape[1]) ])


scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( df_traj ), columns=df_traj.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_traj_pcs = pd.DataFrame(pcs, columns=['pre_traj_PC_%s' % str(i) for i in range(0, pcs.shape[1])])

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC%s'%pc for pc in range(pca.explained_variance_.size)])

variance.to_csv(path + 'pre_traj_variance.csv', index=False)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( df_local ), columns=df_local.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_local_pcs = pd.DataFrame(pcs, columns=['pre_local_PC_%s' % str(i) for i in range(0, pcs.shape[1])])

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC%s'%pc for pc in range(pca.explained_variance_.size)])

variance.to_csv(path + 'pre_local_variance.csv', index=False)

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( df_morph ), columns=df_morph.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_morph_pcs = pd.DataFrame(pcs, columns=['pre_morph_PC_%s' % str(i) for i in range(0, pcs.shape[1])])

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC%s'%pc for pc in range(pca.explained_variance_.size)])

variance.to_csv(path + 'pre_morph_variance.csv', index=False)

df_test_prepcs = pd.concat([df_test, df_traj_pcs, df_local_pcs, df_morph_pcs], axis=1)

#################################### Extract post perturbation vectors ####################################

traj = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('MLP_perturbed_emb_traj').output)
traj_vector = traj.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])

local = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('MLP_perturbed_emb_local').output)
local_vector = local.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])

morph = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('MLP_perturbed_emb_morph').output)
morph_vector = morph.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])


df_traj = pd.DataFrame(traj_vector, columns=[ 'post_traj_%s' % str(i) for i in range(0,traj_vector.shape[1]) ])
df_local = pd.DataFrame(local_vector, columns=[ 'post_local_%s' % str(i) for i in range(0,local_vector.shape[1]) ])
df_morph = pd.DataFrame(morph_vector, columns=[ 'post_morph_%s' % str(i) for i in range(0,morph_vector.shape[1]) ])


scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( df_traj ), columns=df_traj.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_traj_pcs = pd.DataFrame(pcs, columns=['post_traj_PC_%s' % str(i) for i in range(0, pcs.shape[1])])

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC%s'%pc for pc in range(pca.explained_variance_.size)])

variance.to_csv(path + 'post_traj_variance.csv', index=False)


scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( df_local ), columns=df_local.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_local_pcs = pd.DataFrame(pcs, columns=['post_local_PC_%s' % str(i) for i in range(0, pcs.shape[1])])

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC%s'%pc for pc in range(pca.explained_variance_.size)])

variance.to_csv(path + 'post_local_variance.csv', index=False)


scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( df_morph ), columns=df_morph.columns)
from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_morph_pcs = pd.DataFrame(pcs, columns=['post_morph_PC_%s' % str(i) for i in range(0, pcs.shape[1])])

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC%s'%pc for pc in range(pca.explained_variance_.size)])

variance.to_csv(path + 'post_morph_variance.csv', index=False)

df_test_postpcs = pd.concat([df_test_prepcs, df_traj_pcs, df_local_pcs, df_morph_pcs], axis=1)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis\Figure7. Interpretation\\'
df_test_postpcs.to_csv(path + 'latent_vectors.csv', index=False)
df_test_postpcs.to_parquet(path + 'latent_vectors.parquet')


#################################### Read latent vector dataset ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis\Figure7. Interpretation\\'
df = pd.read_parquet(path + 'latent_vectors.parquet')

#################################### Get PCA ####################################
from sklearn.decomposition import PCA

df.columns.get_loc('inst_angle_cosine_similarity_entropies')
motility_data = df.iloc[:, 2:103].drop(['speed_distribution_x', 'speed_distribution_y'], axis=1)
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_motility_pc = pd.DataFrame(pcs, columns=['motility_PC_%s' % str(i) for i in range(0, pcs.shape[1])])

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC%s'%pc for pc in range(pca.explained_variance_.size)])

variance.to_csv(path + 'known_motility_variance.csv', index=False)

df.columns.get_loc('morpho_avg_speed')
df.columns.get_loc('morpho_displ_autocorr_3')
morpho_data = df.iloc[:, 103:115]
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
morpho_data_scaled= pd.DataFrame(scaler.fit_transform( morpho_data ), columns=morpho_data.columns)
pca = PCA(0.95)
pcs = pca.fit_transform(morpho_data_scaled)
df_morpho_pc = pd.DataFrame(pcs, columns=['morpho_PC_%s' % str(i) for i in range(0, pcs.shape[1])])

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC%s'%pc for pc in range(pca.explained_variance_.size)])

variance.to_csv(path + 'known_morph_variance.csv', index=False)

df.columns.get_loc('nearest_approach_times')
df.columns.get_loc('diff_group_distance_autocorr_3')
colocalize_data = df.iloc[:, 133:187]
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
colocalize_data_scaled= pd.DataFrame(scaler.fit_transform( colocalize_data ), columns=colocalize_data.columns)
pca = PCA(0.95)
pcs = pca.fit_transform(colocalize_data_scaled)
df_colocalize_pc = pd.DataFrame(pcs, columns=['colocalize_PC_%s' % str(i) for i in range(0, pcs.shape[1])])

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC%s'%pc for pc in range(pca.explained_variance_.size)])

variance.to_csv(path + 'known_colocalize_variance.csv', index=False)



#################################### Chronological Age vs Predicted Age ####################################
font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(4, 4))
#sns.scatterplot(x=y_test, y=prediction, lw=2.5, s=5, alpha=0.5, palette=('black',), ax=ax)


sns.kdeplot(x='Age', y='pred', data=df, alpha=1, hue='Type',hue_order=['Old', 'Frail'], ax=ax,
            zorder=1, linewidths=1.5, palette =('#beaed4', '#fdc086'), fill=False,  common_norm=False, thresh=0.1,)


file_name = 'Overlayed Trained on Old Frail Age Regression'

#ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )
old = df[df['Type']=='Old'].reset_index(drop=True)
frail = df[df['Type']=='Frail'].reset_index(drop=True)

sns.regplot(x='Age', y='pred', data=old, scatter=False, #scatter_kws={"color":"black", "alpha":0.7, 's':10},
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)

sns.regplot(x='Age', y='pred', data=frail, scatter=False, #scatter_kws={"color":"black", "alpha":0.7, 's':10},
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)

mae = np.mean([ abs(i - j) for i, j in zip(frail['Age'], frail['pred']) ])
r, p = scipy.stats.pearsonr(frail['Age'], frail['pred'])

plt.text(0.1, 0.95,"MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="#fdc086")
plt.text(0.1, 0.88, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="#fdc086")
# plt.text(0.1, 0.9, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
#          fontsize=14, fontdict={'weight': 'bold'}, color="black")

mae = np.mean([ abs(i - j) for i, j in zip(old['Age'], old['pred']) ])
r, p = scipy.stats.pearsonr(old['Age'], old['pred'])
plt.text(0.4, 0.3,"MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="#beaed4")
plt.text(0.4, 0.23, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="#beaed4")
# plt.text(0.8, 0.1, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
#          fontsize=14, fontdict={'weight': 'bold'}, color="black")

handles, labels = ax.get_legend_handles_labels()

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_xlabel('Chronological Age (yr)', fontsize=16, weight='bold', color='0.2')
ax.set_ylabel('Predicted Age (yr)', fontsize=16, weight='bold', color='0.2')

#custom_range = (70, 95)
label_stepsize = 5
# ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
# ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

plt.xticks(fontsize=12, color='0.2', weight='bold')
#plt.yticks(np.arange(age_range[0], age_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2', weight='bold')
plt.yticks(fontsize=12, color='0.2', weight='bold')

legend = plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
           loc='best')
legend.remove()
plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()


#################################### latent vector vs Predicted Age ####################################
features = ['pre_traj_PC_0', 'pre_traj_PC_1', 'pre_traj_PC_2', 'pre_local_PC_0', 'pre_local_PC_1', 'pre_local_PC_2',
            'pre_morph_PC_0', 'pre_morph_PC_1', 'pre_morph_PC_2', 'post_traj_PC_0', 'post_traj_PC_1', 'post_traj_PC_2',
            'post_local_PC_0', 'post_local_PC_1', 'post_local_PC_2', 'post_morph_PC_0', 'post_morph_PC_1', 'post_morph_PC_2',]

for feature in features:

    font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(4, 4))
    #sns.scatterplot(x=y_test, y=prediction, lw=2.5, s=5, alpha=0.5, palette=('black',), ax=ax)


    sns.kdeplot(x=feature, y='pred', data=df, alpha=1, hue='Type',hue_order=['Old', 'Frail'], ax=ax,
                zorder=1, linewidths=1.5, palette =('#beaed4', '#fdc086'), fill=False,  common_norm=False, thresh=0.1,)


    file_name = '%s vs Predicted age'%feature

    #ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )
    old = df[df['Type']=='Old'].reset_index(drop=True)
    frail = df[df['Type']=='Frail'].reset_index(drop=True)

    sns.regplot(x=feature, y='pred', data=old, scatter=False, #scatter_kws={"color":"#beaed4", "alpha":0.7, 's':2},
                line_kws={"color":"red", 'linewidth':1.5}, ax=ax)

    sns.regplot(x=feature, y='pred', data=frail, scatter=False, #scatter_kws={"color":"#fdc086", "alpha":0.7, 's':2},
                line_kws={"color":"red", 'linewidth':1.5}, ax=ax)

    r, p = scipy.stats.pearsonr(old[feature], old['pred'])

    plt.text(0.4, 0.23, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                             fontsize=14, fontdict={'weight': 'bold'}, color="#beaed4")

    r, p = scipy.stats.pearsonr(frail[feature], frail['pred'])

    plt.text(0.1, 0.88, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                             fontsize=14, fontdict={'weight': 'bold'}, color="#fdc086")
    # plt.text(0.1, 0.9, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
    #          fontsize=14, fontdict={'weight': 'bold'}, color="black")


    # plt.text(0.8, 0.1, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
    #          fontsize=14, fontdict={'weight': 'bold'}, color="black")

    handles, labels = ax.get_legend_handles_labels()

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')

    ax.set_xlabel('%s'%feature, fontsize=16, weight='bold', color='0.2')
    ax.set_ylabel('Predicted Age (yr)', fontsize=16, weight='bold', color='0.2')

    #custom_range = (70, 95)
    label_stepsize = 5
    # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
    # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

    plt.xticks(fontsize=12, color='0.2', weight='bold')
    #plt.yticks(np.arange(age_range[0], age_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2', weight='bold')
    plt.yticks(fontsize=12, color='0.2', weight='bold')

    legend = plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
               loc='best')
    legend.remove()
    plt.savefig(path + 'feature vs predicted age/%s.png' % (file_name), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/feature vs predicted age/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/feature vs predicted age/' )
    plt.savefig(path + 'svg/feature vs predicted age/%s.svg' % (file_name), bbox_inches='tight')
    plt.clf()
    plt.close()

#################################### Known features correlation  ####################################
df_corr = pd.concat([df_motility_pc, df_morpho_pc, df_colocalize_pc], axis=1)
corr = df_corr.corr(method='pearson')

fig, ax = plt.subplots()
# ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
#                  annot_kws={'size': 4, 'weight':'bold'},
#                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

g=sns.clustermap(corr, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=1,
**kws,
figsize = (32, 32),
dendrogram_ratio=0.1,
cbar=True
)
# g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=4, rotation=35, rotation_mode='anchor', ha='right')
# g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=4, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# cax = ax.figure.axes[-1]  # colorbar
# cax.tick_params(labelsize=4)  # fontsize of tick label
# cax.yaxis.label.set_size(6)  # fontsize of color bar y label

plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
plt.yticks(fontsize=4, color='0.2', weight='bold')

# ax.set_xticklabels(list(corr.columns), fontsize=1, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
# ax.set_yticklabels(list(corr.index), fontsize=1, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')


plt.savefig(path+'known features correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/known features correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()


#################################### preperturb latent vectors correlation  ####################################
df.columns.get_loc('pre_traj_PC_0')
df.columns.get_loc('pre_morph_PC_33')

df_corr = df.iloc[:, 191:248]
corr = df_corr.corr(method='pearson')

fig, ax = plt.subplots()
# ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
#                  annot_kws={'size': 4, 'weight':'bold'},
#                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

g=sns.clustermap(corr, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=1,
**kws,
figsize = (16, 16),
dendrogram_ratio=0.1,
cbar=True
)
# g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=4, rotation=35, rotation_mode='anchor', ha='right')
# g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=4, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# cax = ax.figure.axes[-1]  # colorbar
# cax.tick_params(labelsize=4)  # fontsize of tick label
# cax.yaxis.label.set_size(6)  # fontsize of color bar y label

plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
plt.yticks(fontsize=4, color='0.2', weight='bold')

plt.savefig(path+'pre perturb vectors features correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/pre perturb vectors features correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

#################################### correlation btw preperturb latent vectors & PC known features ####################################
fig_sizes = [(14, 14), (14, 12), (14, 5)]
dataset = [df_motility_pc, df_colocalize_pc, df_morpho_pc]
for idx, name in enumerate(['motility', 'colocalize', 'morpho']):
    df_corr1 = dataset[idx]
    #df_corr1 = pd.concat([df_motility_pc, df_morpho_pc, df_colocalize_pc], axis=1)
    df_corr2 = df.iloc[:, 191:248]

    #corr = df_corr1.corrwith(df_corr2, axis=0)

    corr = pd.DataFrame() # Correlation matrix
    df_p = pd.DataFrame()  # Matrix of p-values
    for x in df_corr1.columns:
        for y in df_corr2.columns:
            r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
            corr.loc[x,y] = r
            df_p.loc[x,y] = p

    columns_to_drop = corr.columns[(abs(corr) < 0.5).all(axis=0)]
    corr_filtered = corr.drop(columns=columns_to_drop)

    fig, ax = plt.subplots()
    # ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
    #                  annot_kws={'size': 4, 'weight':'bold'},
    #                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

    kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

    g=sns.clustermap(corr_filtered, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
    #cbar_pos=(1, 0.2, 0.03, 0.8),
    metric='correlation', method='average',
    linewidths=0.5, linecolor='black',
    alpha=1,
    **kws,
    figsize = fig_sizes[idx],
    dendrogram_ratio=0.1,
    cbar=True
    )
    # g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=4, rotation=35, rotation_mode='anchor', ha='right')
    # g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=4, va='center')

    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*1, 0.02])

    g.ax_cbar.set_title('Correlation', fontsize=12)
    g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

    # cax = ax.figure.axes[-1]  # colorbar
    # cax.tick_params(labelsize=4)  # fontsize of tick label
    # cax.yaxis.label.set_size(6)  # fontsize of color bar y label

    # plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
    # plt.yticks(fontsize=4, color='0.2', weight='bold')
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=7, rotation=35, rotation_mode='anchor',ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=7, va='center')

    plt.savefig(path+'pre perturb vectors vs %s PC correlation.png'%name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/pre perturb vectors vs %s PC correlation.svg'%name, bbox_inches='tight')

    plt.close()
    plt.clf()

#################################### filtered correlation btw preperturb latent vectors & PC known features ####################################
df_corr1 = pd.concat([df_motility_pc, df_colocalize_pc, df_morpho_pc], axis=1)
df_corr2 = df.iloc[:, 191:248]

#corr = df_corr1.corrwith(df_corr2, axis=0)

corr = pd.DataFrame() # Correlation matrix
df_p = pd.DataFrame()  # Matrix of p-values
for x in df_corr1.columns:
    for y in df_corr2.columns:
        r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
        corr.loc[x,y] = r
        df_p.loc[x,y] = p

columns_to_drop = corr.columns[(abs(corr) < 0.4).all(axis=0)]
corr_filtered1 = corr.drop(columns=columns_to_drop)

index_to_drop = corr_filtered1.index[(abs(corr_filtered1) < 0.4).all(axis=1)]
corr_filtered = corr_filtered1.drop(index=index_to_drop)

fig, ax = plt.subplots()
# ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
#                  annot_kws={'size': 4, 'weight':'bold'},
#                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

g=sns.clustermap(corr_filtered, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=1,
**kws,
figsize = (3.3, 2.7),
dendrogram_ratio=0.1,
cbar=True
)
# g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=4, rotation=35, rotation_mode='anchor', ha='right')
# g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=4, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, 0.1*1, 0.02])

g.ax_cbar.set_title('Correlation', fontsize=7)
g.ax_cbar.tick_params(axis='x', length=5, labelsize=7)

# cax = ax.figure.axes[-1]  # colorbar
# cax.tick_params(labelsize=4)  # fontsize of tick label
# cax.yaxis.label.set_size(6)  # fontsize of color bar y label

# plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
# plt.yticks(fontsize=4, color='0.2', weight='bold')
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=7, rotation=35, rotation_mode='anchor',ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=7, va='center')

plt.savefig(path+'all pre perturb vectors vs PC correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/all pre perturb vectors vs PC correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()


#################################### correlation btw preperturb latent vectors PC0 & known features PC0 ####################################

df_corr1 = pd.concat([df_motility_pc.iloc[:,0:3], df_colocalize_pc.iloc[:,0:3], df_morpho_pc.iloc[:,0:3]], axis=1)
#df_corr1 = pd.concat([df_motility_pc, df_morpho_pc, df_colocalize_pc], axis=1)
df_corr2 = df.loc[:, ['pre_traj_PC_0', 'pre_traj_PC_1', 'pre_traj_PC_2', 'pre_local_PC_0', 'pre_local_PC_1', 'pre_local_PC_2',
                      'pre_morph_PC_0', 'pre_morph_PC_1', 'pre_morph_PC_2']]

#corr = df_corr1.corrwith(df_corr2, axis=0)

corr = pd.DataFrame() # Correlation matrix
df_p = pd.DataFrame()  # Matrix of p-values
for x in df_corr1.columns:
    for y in df_corr2.columns:
        r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
        corr.loc[x,y] = r
        df_p.loc[x,y] = p


fig, ax = plt.subplots()
# ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
#                  annot_kws={'size': 4, 'weight':'bold'},
#                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

g=sns.clustermap(corr, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=1,
**kws,
figsize = (4, 4),
dendrogram_ratio=0.1,
cbar=True,
)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=12, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

x0, _y0, _w, _h = g.cbar_pos
#g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*4, 0.02])
g.ax_cbar.set_position([0.8, 1, 0.06*4, 0.02])

g.ax_cbar.set_title('Correlation', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# cax = ax.figure.axes[-1]  # colorbar
# cax.tick_params(labelsize=4)  # fontsize of tick label
# cax.yaxis.label.set_size(6)  # fontsize of color bar y label

# plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
# plt.yticks(fontsize=4, color='0.2', weight='bold')

plt.savefig(path+'PC0 only pre perturb vectors vs PC correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/PC0 only pre perturb vectors vs PC correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

#################################### correlation btw preperturb latent vectors & known features ####################################


df_corr1 = pd.concat([motility_data, colocalize_data, morpho_data], axis=1)
df_corr2 = df.loc[:, ['pre_traj_PC_0', 'pre_traj_PC_1', 'pre_traj_PC_2', 'pre_local_PC_0', 'pre_local_PC_1', 'pre_local_PC_2',
                      'pre_morph_PC_0', 'pre_morph_PC_1', 'pre_morph_PC_2']]


corr = pd.DataFrame() # Correlation matrix
df_p = pd.DataFrame()  # Matrix of p-values
for x in df_corr1.columns:
    for y in df_corr2.columns:
        r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
        corr.loc[x,y] = r
        df_p.loc[x,y] = p

index_to_drop = corr.index[(abs(corr) < 0.4).all(axis=1)]
corr_filtered = corr.drop(index=index_to_drop)

max_corr = np.max(abs(corr_filtered), axis=1)
fig, ax = plt.subplots()
# ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
#                  annot_kws={'size': 4, 'weight':'bold'},
#                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )


g=sns.clustermap(corr_filtered, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=1,
**kws,
figsize = (5, 18), # (4, 38) for all features
dendrogram_ratio=0.1,
cbar=True
)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=6, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=8, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)


# cax = ax.figure.axes[-1]  # colorbar
# cax.tick_params(labelsize=4)  # fontsize of tick label
# cax.yaxis.label.set_size(6)  # fontsize of color bar y label

# plt.xticks(fontsize=1, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
# plt.yticks(fontsize=1, color='0.2', weight='bold')

plt.savefig(path+'pre perturb vectors vs actual features correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/pre perturb vectors vs actual features correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()


#################################### correlation btw preperturb latent vectors & known features ####################################
fig_sizes = [(4, 28), (4, 16), (4, 6)]
fontsizes = [8, 9, 16]
dataset = [motility_data, colocalize_data, morpho_data]
for idx, name in enumerate(['motility', 'colocalize', 'morpho']):
    df_corr1 = dataset[idx]
#df_corr1 = pd.concat([df_motility_pc, df_morpho_pc, df_colocalize_pc], axis=1)
    #df_corr2 = df.iloc[:, 191:248]
    df_corr2 = df.loc[:, ['pre_traj_PC_0', 'pre_traj_PC_1', 'pre_traj_PC_2', 'pre_local_PC_0', 'pre_local_PC_1', 'pre_local_PC_2',
                          'pre_morph_PC_0','pre_morph_PC_1', 'pre_morph_PC_2']]

#corr = df_corr1.corrwith(df_corr2, axis=0)

    corr = pd.DataFrame() # Correlation matrix
    df_p = pd.DataFrame()  # Matrix of p-values
    for x in df_corr1.columns:
        for y in df_corr2.columns:
            r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
            corr.loc[x,y] = r
            df_p.loc[x,y] = p


    fig, ax = plt.subplots()
    # ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
    #                  annot_kws={'size': 4, 'weight':'bold'},
    #                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

    kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

    g=sns.clustermap(corr, annot=False, cmap='PuOr', col_cluster=False, row_cluster=True,
    #cbar_pos=(1, 0.2, 0.03, 0.8),
    metric='correlation', method='average',
    linewidths=0.5, linecolor='black',
    alpha=1,
    **kws,
    figsize = fig_sizes[idx],
    dendrogram_ratio=0.1,
    cbar=True
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=fontsizes[idx]-2, rotation=35, rotation_mode='anchor', ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=fontsizes[idx], va='center')

    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*1, 0.02])

    g.ax_cbar.set_title('Correlation', fontsize=12)
    g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

    # cax = ax.figure.axes[-1]  # colorbar
    # cax.tick_params(labelsize=4)  # fontsize of tick label
    # cax.yaxis.label.set_size(6)  # fontsize of color bar y label

    # plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
    # plt.yticks(fontsize=4, color='0.2', weight='bold')

    plt.savefig(path+'pre perturb vectors vs %s actual features correlation.png'%name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/pre perturb vectors vs %s actual features correlation.svg'%name, bbox_inches='tight')

    plt.close()
    plt.clf()

#################################### filtered correlation btw postperturb latent vectors & PC known features ####################################
df_corr1 = pd.concat([df_motility_pc, df_colocalize_pc, df_morpho_pc], axis=1)
df_corr2 = df.iloc[:, 248:257]

#corr = df_corr1.corrwith(df_corr2, axis=0)

corr = pd.DataFrame() # Correlation matrix
df_p = pd.DataFrame()  # Matrix of p-values
for x in df_corr1.columns:
    for y in df_corr2.columns:
        r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
        corr.loc[x,y] = r
        df_p.loc[x,y] = p

columns_to_drop = corr.columns[(abs(corr) < 0.4).all(axis=0)]
corr_filtered1 = corr.drop(columns=columns_to_drop)

index_to_drop = corr_filtered1.index[(abs(corr_filtered1) < 0.4).all(axis=1)]
corr_filtered = corr_filtered1.drop(index=index_to_drop)

fig, ax = plt.subplots()
# ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
#                  annot_kws={'size': 4, 'weight':'bold'},
#                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

g=sns.clustermap(corr_filtered, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=1,
**kws,
figsize = (4, 4),
dendrogram_ratio=0.1,
cbar=True
)
# g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=4, rotation=35, rotation_mode='anchor', ha='right')
# g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=4, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# cax = ax.figure.axes[-1]  # colorbar
# cax.tick_params(labelsize=4)  # fontsize of tick label
# cax.yaxis.label.set_size(6)  # fontsize of color bar y label

# plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
# plt.yticks(fontsize=4, color='0.2', weight='bold')
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=7, rotation=35, rotation_mode='anchor',ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=7, va='center')

plt.savefig(path+'all post perturb vectors vs PC correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/all post perturb vectors vs PC correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

#################################### per condition filtered correlation btw postperturb latent vectors & PC known features ####################################

for perturb in np.unique(df['Condition']):

    idxs = df['Condition'] == perturb

    df_corr1 = pd.concat([df_motility_pc, df_colocalize_pc, df_morpho_pc], axis=1)[idxs]
    df_corr2 = df.iloc[:, 248:257][idxs]

    #corr = df_corr1.corrwith(df_corr2, axis=0)

    corr = pd.DataFrame() # Correlation matrix
    df_p = pd.DataFrame()  # Matrix of p-values
    for x in df_corr1.columns:
        for y in df_corr2.columns:
            r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
            corr.loc[x,y] = r
            df_p.loc[x,y] = p

    columns_to_drop = corr.columns[(abs(corr) < 0.4).all(axis=0)]
    corr_filtered1 = corr.drop(columns=columns_to_drop)

    index_to_drop = corr_filtered1.index[(abs(corr_filtered1) < 0.4).all(axis=1)]
    corr_filtered = corr_filtered1.drop(index=index_to_drop)

    fig, ax = plt.subplots()
    # ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
    #                  annot_kws={'size': 4, 'weight':'bold'},
    #                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

    kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

    g=sns.clustermap(corr_filtered, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
    #cbar_pos=(1, 0.2, 0.03, 0.8),
    metric='correlation', method='average',
    linewidths=0.5, linecolor='black',
    alpha=1,
    **kws,
    figsize = (4, 4),
    dendrogram_ratio=0.1,
    cbar=True
    )
    # g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=4, rotation=35, rotation_mode='anchor', ha='right')
    # g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=4, va='center')

    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*1, 0.02])

    g.ax_cbar.set_title('Correlation', fontsize=12)
    g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

    # cax = ax.figure.axes[-1]  # colorbar
    # cax.tick_params(labelsize=4)  # fontsize of tick label
    # cax.yaxis.label.set_size(6)  # fontsize of color bar y label

    # plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
    # plt.yticks(fontsize=4, color='0.2', weight='bold')
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=7, rotation=35, rotation_mode='anchor',ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=7, va='center')

    plt.savefig(path+'all %s post perturb vectors vs PC correlation.png'%perturb, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/all %s post perturb vectors vs PC correlation.svg'%perturb, bbox_inches='tight')

    plt.close()
    plt.clf()


#################################### correlation btw postperturb latent vectors PC0 & known features PC0 ####################################

df_corr1 = pd.concat([df_motility_pc.iloc[:,0:3], df_colocalize_pc.iloc[:,0:3], df_morpho_pc.iloc[:,0:3]], axis=1)
#df_corr1 = pd.concat([df_motility_pc, df_morpho_pc, df_colocalize_pc], axis=1)
#df_corr2 = df.loc[:, ['post_traj_PC_0', 'post_traj_PC_1', 'post_local_PC_0', 'post_local_PC_1', 'post_morph_PC_0', 'post_morph_PC_1']]
df_corr2 = df.iloc[:, 248:257]
#corr = df_corr1.corrwith(df_corr2, axis=0)

corr = pd.DataFrame() # Correlation matrix
df_p = pd.DataFrame()  # Matrix of p-values
for x in df_corr1.columns:
    for y in df_corr2.columns:
        r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
        corr.loc[x,y] = r
        df_p.loc[x,y] = p


fig, ax = plt.subplots()
# ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
#                  annot_kws={'size': 4, 'weight':'bold'},
#                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

g=sns.clustermap(corr, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=1,
**kws,
figsize = (4, 4),
dendrogram_ratio=0.1,
cbar=True,
)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=12, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, 0.06*4, 0.02])

g.ax_cbar.set_title('Correlation', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# cax = ax.figure.axes[-1]  # colorbar
# cax.tick_params(labelsize=4)  # fontsize of tick label
# cax.yaxis.label.set_size(6)  # fontsize of color bar y label

# plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
# plt.yticks(fontsize=4, color='0.2', weight='bold')

plt.savefig(path+'PC0 only post perturb vectors vs PC correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/PC0 only post perturb vectors vs PC correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

#################################### each perturbation correlation btw postperturb latent vectors PC0 & known features PC0 ####################################

for perturb in np.unique(df['Condition']):

    idxs = df['Condition'] == perturb
    df_corr1 = pd.concat([df_motility_pc.iloc[:,0:3], df_colocalize_pc.iloc[:,0:3], df_morpho_pc.iloc[:,0:3]], axis=1)[idxs]
    #df_corr1 = pd.concat([df_motility_pc, df_morpho_pc, df_colocalize_pc], axis=1)
    #df_corr2 = df.loc[:, ['post_traj_PC_0', 'post_traj_PC_1', 'post_local_PC_0', 'post_local_PC_1', 'post_morph_PC_0', 'post_morph_PC_1']]
    df_corr2 = df.iloc[:, 248:257][idxs]
    #corr = df_corr1.corrwith(df_corr2, axis=0)

    corr = pd.DataFrame() # Correlation matrix
    df_p = pd.DataFrame()  # Matrix of p-values
    for x in df_corr1.columns:
        for y in df_corr2.columns:
            r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
            corr.loc[x,y] = r
            df_p.loc[x,y] = p


    fig, ax = plt.subplots()
    # ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
    #                  annot_kws={'size': 4, 'weight':'bold'},
    #                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

    kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

    g=sns.clustermap(corr, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
    #cbar_pos=(1, 0.2, 0.03, 0.8),
    metric='correlation', method='average',
    linewidths=0.5, linecolor='black',
    alpha=1,
    **kws,
    figsize = (4, 4),
    dendrogram_ratio=0.1,
    cbar=True,
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=12, rotation=35, rotation_mode='anchor', ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([0.8, 1, 0.06*4, 0.02])

    g.ax_cbar.set_title('Correlation', fontsize=12)
    g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

    # cax = ax.figure.axes[-1]  # colorbar
    # cax.tick_params(labelsize=4)  # fontsize of tick label
    # cax.yaxis.label.set_size(6)  # fontsize of color bar y label

    # plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
    # plt.yticks(fontsize=4, color='0.2', weight='bold')

    plt.savefig(path+'PC0 %s only post perturb vectors vs PC correlation.png'%perturb, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/PC0 %s only post perturb vectors vs PC correlation.svg'%perturb, bbox_inches='tight')

    plt.close()
    plt.clf()


#################################### correlation btw postperturb latent vectors & known features ####################################


df_corr1 = pd.concat([motility_data, colocalize_data, morpho_data], axis=1)
#df_corr2 = df.loc[:, ['post_traj_PC_0', 'post_traj_PC_1', 'post_local_PC_0', 'post_local_PC_1', 'post_morph_PC_0', 'post_morph_PC_1']]
df_corr2 = df.iloc[:, 248:257]

corr = pd.DataFrame() # Correlation matrix
df_p = pd.DataFrame()  # Matrix of p-values
for x in df_corr1.columns:
    for y in df_corr2.columns:
        r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
        corr.loc[x,y] = r
        df_p.loc[x,y] = p

index_to_drop = corr.index[(abs(corr) < 0.4).all(axis=1)]
corr_filtered = corr.drop(index=index_to_drop)

max_corr = np.max(abs(corr_filtered), axis=1)

fig, ax = plt.subplots()
# ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
#                  annot_kws={'size': 4, 'weight':'bold'},
#                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

g=sns.clustermap(corr_filtered, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=1,
**kws,
figsize = (6, 6),
dendrogram_ratio=0.1,
cbar=True
)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=8, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=8, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)


# cax = ax.figure.axes[-1]  # colorbar
# cax.tick_params(labelsize=4)  # fontsize of tick label
# cax.yaxis.label.set_size(6)  # fontsize of color bar y label

# plt.xticks(fontsize=1, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
# plt.yticks(fontsize=1, color='0.2', weight='bold')

plt.savefig(path+'post perturb vectors vs actual features correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/post perturb vectors vs actual features correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

#################################### each perturbation correlation btw postperturb latent vectors & known features ####################################

for perturb in np.unique(df['Condition']):

    idxs = df['Condition'] == perturb
    df_corr1 = pd.concat([motility_data, colocalize_data, morpho_data], axis=1)[idxs]
    #df_corr2 = df.loc[:, ['post_traj_PC_0', 'post_traj_PC_1', 'post_local_PC_0', 'post_local_PC_1', 'post_morph_PC_0', 'post_morph_PC_1']]
    df_corr2 = df.iloc[:, 248:258][idxs]

    corr = pd.DataFrame() # Correlation matrix
    df_p = pd.DataFrame()  # Matrix of p-values
    for x in df_corr1.columns:
        for y in df_corr2.columns:
            r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
            corr.loc[x,y] = r
            df_p.loc[x,y] = p

    index_to_drop = corr.index[(abs(corr) < 0.4).all(axis=1)]
    corr_filtered = corr.drop(index=index_to_drop)

    max_corr = np.max(abs(corr_filtered), axis=1)

    fig, ax = plt.subplots()
    # ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
    #                  annot_kws={'size': 4, 'weight':'bold'},
    #                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

    kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

    g=sns.clustermap(corr_filtered, annot=False, cmap='PuOr', col_cluster=False, row_cluster=False,
    #cbar_pos=(1, 0.2, 0.03, 0.8),
    metric='correlation', method='average',
    linewidths=0.5, linecolor='black',
    alpha=1,
    **kws,
    figsize = (6, 14),
    dendrogram_ratio=0.1,
    cbar=True
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=8, rotation=35, rotation_mode='anchor', ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=8, va='center')

    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*1, 0.02])

    g.ax_cbar.set_title('Correlation', fontsize=12)
    g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)


    # cax = ax.figure.axes[-1]  # colorbar
    # cax.tick_params(labelsize=4)  # fontsize of tick label
    # cax.yaxis.label.set_size(6)  # fontsize of color bar y label

    # plt.xticks(fontsize=1, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
    # plt.yticks(fontsize=1, color='0.2', weight='bold')

    plt.savefig(path+'post perturb vectors %s vs actual features correlation.png'%perturb, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/post perturb vectors %s vs actual features correlation.svg'%perturb, bbox_inches='tight')

    plt.close()
    plt.clf()


#################################### Correlation btw age and PC motility features ####################################

feature_list = list(df.iloc[:, 248:258].columns)

df_r = pd.DataFrame()
df_p = pd.DataFrame()
for feature in tqdm(feature_list):
    df_r_temp = pd.DataFrame()
    df_p_temp = pd.DataFrame()

    datasets = {}
    ages = []
    mean_values = []
    age_types = []
    media_types = []

    for batch in np.unique(df['Patient']):
        df_part = df[df['Patient'] == batch].reset_index(drop=True)
        medias = np.unique(df_part['Condition'])

        for media in medias:
            df_part_part = df_part[df_part['Condition'] == media].reset_index(drop=True)
            age = np.unique( df_part_part['Age'] )[0]
            pred = np.unique(df_part_part['pred'])[0]
            age_type = np.unique( df_part_part['Type'] )[0]
            mean_value = np.mean(np.array(df_part_part[feature]))

            ages.append(age)
            age_types.append(age_type)
            mean_values.append(mean_value)
            media_types.append(media)

    datasets['Age'] = np.array(ages)
    datasets['pred'] = np.array(pred)
    datasets['Value'] = np.array(mean_values)
    datasets['Type'] = np.array(age_types)
    datasets['Condition'] = np.array(media_types)

    df_ = pd.DataFrame(datasets)

    for type in ['Old', 'Frail']:
        condition_rs = []
        condition_ps = []

        for cond in np.unique(df['Condition']):
            df_part_= df_[ (df_['Type']==type) & (df_['Condition']==cond) ].reset_index(drop=True)
            #sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)
            r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
            #print(type, cond, r, p)
            condition_rs.append(r)
            condition_ps.append(p)
        each_rs = pd.DataFrame(condition_rs, columns=[feature], index='%s ' % type + np.unique(df['Condition']))
        each_ps = pd.DataFrame(condition_ps, columns=[feature], index='%s ' % type + np.unique(df['Condition']))
        df_r_temp = pd.concat([df_r_temp, each_rs], axis=0)
        df_p_temp = pd.concat([df_p_temp, each_ps], axis=0)

    df_r = pd.concat([df_r, df_r_temp], axis=1)
    df_p = pd.concat([df_p, df_p_temp], axis=1)


columns_to_drop = df_r.columns[(df_p > 0.05).all()]
df_r_filtered = df_r.drop(columns=columns_to_drop)

df_p = -np.log10(df_p)

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)

fig, ax = plt.subplots()
# if np.max(np.max(Z_avg_df)) >= abs(np.min(np.min(Z_avg_df))):
#     kws = dict(cbar_kws=dict(ticks=[-round(np.max(np.max(Z_avg_df)), 1), 0, round(np.max(np.max(Z_avg_df)), 1)], orientation='horizontal'),
#                vmin=-round(np.max(np.max(Z_avg_df)), 1))
# else:
#     kws = dict(cbar_kws=dict(ticks=[round(np.min(np.min(Z_avg_df)), 1), 0, -round(np.min(np.min(Z_avg_df)), 1)],orientation='horizontal'),
#                vmin=round(np.min(np.min(Z_avg_df)), 1) )
#
kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

g=sns.clustermap(df_r, annot=False, cmap=cmc.cork, col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=1,
**kws,
figsize = (6, 5),
dendrogram_ratio=0.1,
cbar=True
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=12, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation Coefficient', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# for spine in g.ax_cbar.spines:
#     #g.ax_cbar.spines[spine].set_color('crimson')
#     g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'correlation with age for PC features.png', dpi=300,bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/' + 'correlation with age for PC features.svg', bbox_inches='tight')
plt.clf()
plt.close()



#################################### Correlation btw pred age and kmeans  ####################################
from sklearn.cluster import KMeans
km = KMeans(n_clusters=9, random_state=0, init='k-means++')
# k-means++: Initialize centroids that are far away each other
kmeans_predicted = km.fit_predict(df.iloc[:, 248:251])
'PT' + list(kmeans_predicted.astype(str) )
kmeans_predicted = np.array('PT') + kmeans_predicted.astype(str)
cluster = pd.DataFrame(kmeans_predicted, columns=['post_traj_kmeans'])

df_ = pd.concat([df, cluster], axis=1)

draw_cluster_distribution_heatmap(df_, path, file_name='post_traj_kmeans_kmeans_heatmap', condition_name='kmeans', annot=False,
                                  cluster_type='post_traj_kmeans', col_cluster=False, figsize=(6,5))


draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_post_traj_kmeans_heatmap', condition_name='post_traj_kmeans', annot=False,
                                  cluster_type='kmeans', col_cluster=False, figsize=(6,5))

condition_name = 'Type'
cluster_type = 'kmeans'


group_clones=[]
for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
#for group in ['DZ', 'sLZ', 'dLZ']:
    corrcoef = []
    aaa = df[df['Zone'] == group]

    group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    group_clone_T = group_clone.T
    for cluster in list(pd.unique(df[cluster_type])):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=[0, 0])
    group_clone = group_clone_T.T

    for column in group_clone.columns:
        group_clone.rename(columns={column:column+'_%s'%group}, inplace=True)
    group_clones.append(group_clone)