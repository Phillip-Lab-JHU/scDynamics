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

df = df[df['Type']!='Young'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Young'].reset_index(drop=True)

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

y_names = df['Type']
print('Classes:', pd.unique(y_names))
y = y_names.replace(list(pd.unique(y_names)), [i for i in range(pd.unique(y_names).shape[0])])
y = np.array(y)
print('Number of classes:', np.unique(y).size)

from sklearn.model_selection import train_test_split
X_traj_train, X_traj_test, X_local_train, X_local_test, X_morph_train, X_morph_test, \
X_covar_train, X_covar_test, y_train, y_test, df_train, df_test = train_test_split(X_traj, X_local, X_morph, X_covar, y, df, test_size=0.2, random_state=0, stratify=y)
print(X_traj_train.shape, X_traj_test.shape, X_local_train.shape, X_local_test.shape,
      X_morph_train.shape, X_morph_test.shape, X_covar_train.shape, X_covar_test.shape, y_train.shape, y_test.shape)
df_test = df_test.reset_index(drop=True)


from tensorflow.keras import callbacks
stop_early = callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001)

from dnn.classification import scTRAIT
sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=np.unique(y).size)
result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=500,
                     verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])

sctrait.save('saved_model/monocyte_project_v2/classification interpretation scTRAIT old vs frail', save_format='tf')


#################################### Get predicted cellular age ####################################
from tensorflow.keras import models
sctrait = tf.keras.models.load_model('saved_model/monocyte_project_v2/classification interpretation scTRAIT old vs frail', compile=True)

y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
df_test['pred'] = y_pred.flatten()

#################################### Extract pre perturbation vectors ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure7. Interpretation_classification\old vs frail\\'

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

#################################### Extract Delta embeddings ####################################

traj_pert_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('delta_emb_traj').output)
traj_pert_embeding = traj_pert_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
df_traj_pert_embeding = pd.DataFrame(traj_pert_embeding, columns=['traj_delta_%s' % str(i) for i in range(0, traj_pert_embeding.shape[1])])

local_pert_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('delta_emb_local').output)
local_pert_embeding = local_pert_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
df_local_pert_embeding = pd.DataFrame(local_pert_embeding, columns=['local_delta_%s' % str(i) for i in range(0, local_pert_embeding.shape[1])])

morph_pert_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('delta_emb_morph').output)
morph_pert_embeding = morph_pert_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
df_morph_pert_embeding = pd.DataFrame(morph_pert_embeding, columns=['morph_delta_%s' % str(i) for i in range(0, morph_pert_embeding.shape[1])])

df_final = pd.concat([df_test_postpcs, df_traj_pert_embeding, df_local_pert_embeding, df_morph_pert_embeding], axis=1)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure7. Interpretation_classification\old vs frail\\'
df_final.to_csv(path + 'latent_vectors.csv', index=False)
df_final.to_parquet(path + 'latent_vectors.parquet')

#################################### Read latent vector dataset ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure7. Interpretation_classification\old vs frail\\'
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

#################################### Pre perturbation space ####################################
umaps = pd.DataFrame()
for name in ['pre_traj', 'pre_local', 'pre_morph']:
    col_idxs = np.array([name in i for i in df.columns])
    col_list = df.columns[col_idxs]
    data = df.loc[:, col_list]
    m = Morphodynamics(data, 'umap')
    umap = m.get_umap(data, 20, 0.5)
    for idx, column in enumerate(umap.columns):
        umap.rename(columns={column:'%s_UMAP%s'%(name,str(idx+1))}, inplace=True)

    umaps = pd.concat([umaps, umap], axis=1)

df = pd.concat([df, umaps], axis=1)


#['pre_traj_PC_0', 'pre_traj_PC_1', 'pre_local_PC_0', 'pre_local_PC_1', 'pre_morph_PC_0', 'pre_morph_PC_1']
fig, ax = plt.subplots()
grid = sns.PairGrid(data=df, vars=['pre_traj_UMAP1', 'pre_traj_UMAP2', 'pre_local_UMAP1', 'pre_local_UMAP2', 'pre_morph_UMAP1', 'pre_morph_UMAP2'],
                    height=6, aspect=1, hue='Type', hue_order=['Old', 'Frail'], palette=('#beaed4', '#fdc086'), diag_sharey=False, despine=True, corner=True,)
grid.map_upper(sns.kdeplot,  alpha=0.8, linewidths=1.5,
                    fill=True, legend=False, common_norm=False, thresh=0.2)
#grid = grid.map_upper(corr)
grid.map_lower(sns.kdeplot, alpha=0.8, linewidths=1.5,
                    fill=True, legend=False, common_norm=False, thresh=0.2)
grid.map_diag(sns.kdeplot, alpha=0.8, linewidths=1.5,
                    fill=True, legend=False, common_norm=False, thresh=0.2);

plt.tick_params(left=False, right=False, labelleft=False,
            labelbottom=False, bottom=False)

plt.tight_layout()

plt.savefig(path + '/pre perturbation pair plot all embedding.png', dpi=300)
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/pre perturbation pair plot all embedding.svg')
plt.clf()
plt.close()


#################################### Post perturbation space ####################################
umaps = pd.DataFrame()
for name in ['post_traj', 'post_local', 'post_morph']:
    col_idxs = np.array([name in i for i in df.columns])
    col_list = df.columns[col_idxs]
    data = df.loc[:, col_list]
    m = Morphodynamics(data, 'umap')
    umap = m.get_umap(data, 20, 0.5)
    for idx, column in enumerate(umap.columns):
        umap.rename(columns={column:'%s_UMAP%s'%(name,str(idx+1))}, inplace=True)

    umaps = pd.concat([umaps, umap], axis=1)

df = pd.concat([df, umaps], axis=1)


#['pre_traj_PC_0', 'pre_traj_PC_1', 'pre_local_PC_0', 'pre_local_PC_1', 'pre_morph_PC_0', 'pre_morph_PC_1']
fig, ax = plt.subplots()
grid = sns.PairGrid(data=df, vars=['post_traj_UMAP1', 'post_traj_UMAP2', 'post_local_UMAP1', 'post_local_UMAP2', 'post_morph_UMAP1', 'post_morph_UMAP2'],
                    height=6, aspect=1, hue='Type', hue_order=['Old', 'Frail'], palette=('#beaed4', '#fdc086'), diag_sharey=False, despine=True, corner=True,)
grid.map_upper(sns.kdeplot,  alpha=0.8, linewidths=1.5,
                    fill=True, legend=False, common_norm=False, thresh=0.2)
#grid = grid.map_upper(corr)
grid.map_lower(sns.kdeplot, alpha=0.8, linewidths=1.5,
                    fill=True, legend=False, common_norm=False, thresh=0.2)
grid.map_diag(sns.kdeplot, alpha=0.8, linewidths=1.5,
                    fill=True, legend=False, common_norm=False, thresh=0.2);

plt.tick_params(left=False, right=False, labelleft=False,
            labelbottom=False, bottom=False)

plt.tight_layout()

plt.savefig(path + '/post perturbation pair plot all embedding.png', dpi=300)
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/post perturbation pair plot all embedding.svg')
plt.clf()
plt.close()

#################################### Perturbation embedding space ####################################
df_pcs = pd.DataFrame()
variances = []
for name in ['traj_delta', 'local_delta', 'morph_delta']:
    col_idxs = np.array([name in i for i in df.columns])
    col_list = df.columns[col_idxs]
    data = df.loc[:, col_list]
    from sklearn.decomposition import PCA
    scaler = StandardScaler()  # if not normalize, UMAP space is completely different
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    pca = PCA(2)
    pcs = pca.fit_transform(data_scaled)
    pcs = pd.DataFrame(pcs, columns=['%s_PC1'%name, '%s_PC2'%name])

    df_pcs = pd.concat([df_pcs, pcs], axis=1)

    variance = pd.DataFrame(np.array(
        [pca.explained_variance_, pca.explained_variance_ratio_,
         np.cumsum(pca.explained_variance_ratio_)]),
        index=['eigen value', 'variance', 'cumulative variance'],
        columns=['PC%s' % pc for pc in range(pca.explained_variance_.size)])

    #variance.to_csv(path + '%s_variance.csv'%name, index=False)
    variances.append(variance)
df = pd.concat([df, df_pcs], axis=1)




font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 0.5

# xmin = math.floor(df[x_name].min()) - 1
# xmax = math.ceil(df[x_name].max()) + 1
# ymin = math.floor(df[y_name].min()) - 1
# ymax = math.ceil(df[y_name].max()) + 1

colors = ('#888888','#CC6677', '#44AA99', '#6699CC') # Grey, Red, Green, Blue
cmap = ListedColormap(colors)

for idx, name in enumerate(['traj_delta', 'local_delta', 'morph_delta']):

    fig, ax = plt.subplots(figsize=(2, 2))
    x_name = name+'_PC1'
    y_name = name+'_PC2'

    condition_name = 'Condition'
#plt.figure(figsize=(15, 10))
    scatter = ax.scatter(df[x_name], df[y_name],
                          c=df[condition_name].replace(list(np.unique(df[condition_name])),
                            [i for i in range(np.unique(df[condition_name]).shape[0])]),
                          # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
                          s=0.1, label=df[condition_name], marker='o',
                          cmap=cmap)
    # for idx, (pc1, pc2) in enumerate( zip(df[x_name], df[y_name]) ):
    #     ax.plot([0, pc1], [0, pc2], '-', color=colors[idx], linewidth=0.7)

# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax)

    plt.axvline(0, color='0.2', linewidth=0.3, zorder=3, linestyle='--')
    plt.axhline(0, color='0.2', linewidth=0.3, zorder=3, linestyle='--')
    [x.set_linewidth(0.5) for x in ax.spines.values()]

    variance = variances[idx]
    xlabel = 'PC1(' + str(round(variance['PC0'][1] * 100, ndigits=1)) + '%)'
    ylabel = 'PC2(' + str(round(variance['PC1'][1] * 100, ndigits=1)) + '%)'
    ax.set_xlabel(xlabel, labelpad=3, fontsize=6)
    ax.set_ylabel(ylabel, labelpad=3, fontsize=6)
    #sns.despine()
    #plt.axis('off')
    #format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
    handles, labels = scatter.legend_elements(num=None)
    # plt.legend(handles=handles, labels=list(np.unique(pcss[condition_name])),
    #            bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
    #            fontsize=3, frameon=False, markerscale=0.3)

    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    plt.savefig(path + 'perturbation embedding %s.png' % name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/perturbation embedding %s.svg' % name, bbox_inches='tight')
    plt.clf()
    plt.close()



font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 0.5

# xmin = math.floor(df[x_name].min()) - 1
# xmax = math.ceil(df[x_name].max()) + 1
# ymin = math.floor(df[y_name].min()) - 1
# ymax = math.ceil(df[y_name].max()) + 1

colors = ('#888888','#CC6677', '#44AA99', '#6699CC') # Grey, Red, Green, Blue
cmap = ListedColormap(colors)

for idx, name in enumerate(['traj_delta', 'local_delta', 'morph_delta']):

    fig, ax = plt.subplots(figsize=(2, 2))
    x_name = name+'_PC1'
    y_name = name+'_PC2'

    condition_name = 'Condition'

    for cond_idx, cond in enumerate(np.unique(df[condition_name])):
        df_part = df[df[condition_name]==cond].reset_index(drop=True)
        pc1 = np.mean(df_part[x_name])
        pc2 = np.mean(df_part[y_name])
        ax.scatter(pc1, pc2, color=colors[cond_idx], s=10, marker='o')
        ax.plot([0, pc1], [0, pc2], '-', color=colors[cond_idx], linewidth=0.7)

    # for idx, (pc1, pc2) in enumerate( zip(df[x_name], df[y_name]) ):
    #     ax.plot([0, pc1], [0, pc2], '-', color=colors[idx], linewidth=0.7)

# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax)

    plt.axvline(0, color='0.2', linewidth=0.3, zorder=3, linestyle='--')
    plt.axhline(0, color='0.2', linewidth=0.3, zorder=3, linestyle='--')
    [x.set_linewidth(0.5) for x in ax.spines.values()]

    variance = variances[idx]
    xlabel = 'PC1(' + str(round(variance['PC0'][1] * 100, ndigits=1)) + '%)'
    ylabel = 'PC2(' + str(round(variance['PC1'][1] * 100, ndigits=1)) + '%)'
    ax.set_xlabel(xlabel, labelpad=3, fontsize=6)
    ax.set_ylabel(ylabel, labelpad=3, fontsize=6)
    #sns.despine()
    #plt.axis('off')
    #format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
    handles, labels = scatter.legend_elements(num=None)
    # plt.legend(handles=handles, labels=list(np.unique(pcss[condition_name])),
    #            bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
    #            fontsize=3, frameon=False, markerscale=0.3)

    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    plt.savefig(path + 'global perturbation embedding %s.png' % name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/global perturbation embedding %s.svg' % name, bbox_inches='tight')
    plt.clf()
    plt.close()

#################################### preperturb latent vectors correlation  ####################################
df.columns.get_loc('pre_traj_PC_0')
df.columns.get_loc('pre_morph_PC_26')

df_corr = df.iloc[:, 198:294]
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
plt.savefig(path + 'svg/pre perturb vectors correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()


#################################### correlation btw preperturb latent vectors & PC known features ####################################
df_corr1 = pd.concat([df_motility_pc, df_colocalize_pc, df_morpho_pc], axis=1)
df_corr2 = df.iloc[:, 198:294]

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
figsize = (3, 3.3),
dendrogram_ratio=0.1,
cbar=True
)
# g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=4, rotation=35, rotation_mode='anchor', ha='right')
# g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=4, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, 0.06*3, 0.02])

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
figsize = (5, 18),
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


#################################### filtered correlation btw postperturb latent vectors & PC known features ####################################
df_corr1 = pd.concat([df_motility_pc, df_colocalize_pc, df_morpho_pc], axis=1)
df_corr2 = df.loc[:, ['post_traj_PC_0', 'post_traj_PC_1', 'post_traj_PC_2', 'post_local_PC_0', 'post_local_PC_1', 'post_local_PC_2',
                      'post_morph_PC_0','post_morph_PC_1', 'post_morph_PC_2']]

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
figsize = (3, 3),
dendrogram_ratio=0.1,
cbar=True
)
# g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=4, rotation=35, rotation_mode='anchor', ha='right')
# g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=4, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation', fontsize=7)
g.ax_cbar.tick_params(axis='x', length=5, labelsize=7)

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
    df_corr2 = df.loc[:, ['post_traj_PC_0', 'post_traj_PC_1', 'post_traj_PC_2', 'post_local_PC_0', 'post_local_PC_1', 'post_local_PC_2',
                      'post_morph_PC_0','post_morph_PC_1', 'post_morph_PC_2']][idxs]

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
df_corr2 = df.loc[:, ['post_traj_PC_0', 'post_traj_PC_1', 'post_traj_PC_2', 'post_local_PC_0', 'post_local_PC_1', 'post_local_PC_2',
                      'post_morph_PC_0','post_morph_PC_1', 'post_morph_PC_2']]
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
    #df_corr2 = df.iloc[:, 248:258][idxs]
    df_corr2 = df.loc[:, ['post_traj_PC_0', 'post_traj_PC_1', 'post_traj_PC_2', 'post_local_PC_0', 'post_local_PC_1',
                          'post_local_PC_2',
                          'post_morph_PC_0', 'post_morph_PC_1', 'post_morph_PC_2']][idxs]
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
df_corr2 = df.loc[:, ['post_traj_PC_0', 'post_traj_PC_1', 'post_traj_PC_2', 'post_local_PC_0', 'post_local_PC_1', 'post_local_PC_2',
                      'post_morph_PC_0','post_morph_PC_1', 'post_morph_PC_2',]]

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
    #df_corr2 = df.iloc[:, 248:258][idxs]
    df_corr2 = df.loc[:, ['post_traj_PC_0', 'post_traj_PC_1', 'post_traj_PC_2', 'post_local_PC_0', 'post_local_PC_1',
                          'post_local_PC_2',
                          'post_morph_PC_0', 'post_morph_PC_1', 'post_morph_PC_2', 'post_morph_PC_3']][idxs]
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

#################################### Patient-wise behavior projection ####################################
df_patient = pd.DataFrame()

for patient in np.unique(df['Patient']):
    each_patient = df[(df['Patient']==patient)].reset_index(drop=True)
    for cond in np.unique(each_patient['Condition']):
        df_part = each_patient[(each_patient['Condition']==cond)].reset_index(drop=True)
        motility_input = df_part[['pre_traj_PC_0', 'pre_traj_PC_1', 'pre_local_PC_0', 'pre_local_PC_1', 'pre_morph_PC_0', 'pre_morph_PC_1',
                                  'post_traj_PC_0', 'post_traj_PC_1', 'post_local_PC_0', 'post_local_PC_1', 'post_morph_PC_0', 'post_morph_PC_1']]
        df_patient_each = motility_input.median(axis=0).to_frame().T
        df_patient_each['Patient'] = np.unique(df_part['Patient'])[0]
        df_patient_each['Type'] = np.unique(df_part['Type'])[0]
        df_patient_each['Condition'] = np.unique(df_part['Condition'])[0]
        df_patient_each['Age'] = np.unique(df_part['Age'])[0]

        df_patient_each['kmeans'] = np.median(df_part['kmeans'])
        df_patient_each['Weakness'] = np.unique(df_part['Weakness'])[0]
        df_patient_each['Weight_loss'] = np.unique(df_part['Weight_loss'])[0]
        df_patient_each['Exhaustion'] = np.unique(df_part['Exhaustion'])[0]
        df_patient_each['Activity'] = np.unique(df_part['Activity'])[0]
        df_patient_each['Gait'] = np.unique(df_part['Gait'])[0]
        df_patient_each['Grip'] = np.unique(df_part['Grip'])[0]
        df_patient_each['Frailty_score'] = np.unique(df_part['Frailty_score'])[0]

        df_patient = pd.concat([df_patient, df_patient_each], axis=0)

df_patient = df_patient.reset_index(drop=True)



x_name='pre_traj_PC_0'
y_name='pre_traj_PC_1'

for x_name, y_name in [('pre_traj_PC_0', 'pre_traj_PC_1'), ('pre_local_PC_0', 'pre_local_PC_1'), ('pre_morph_PC_0', 'pre_morph_PC_1')]:
# xlabel = '%s('%x_name + str(round(variance['%s'%x_name][1] * 100, ndigits=1)) + '%)'
# ylabel = '%s('%y_name + str(round(variance['%s'%y_name][1] * 100, ndigits=1)) + '%)'
    draw_diff_arrow_scatter(df_patient, path,
                            file_name='annot %s %s PCA first differential response patients'%(x_name, y_name), condition_name='Type',
                            diff_condition_name='Condition', ind_name='Patient', ref='Control', colors=('#fdc086', '#beaed4'),
                            dot_size=5, x_name=x_name, y_name=y_name, xlabel=x_name, ylabel=y_name)



for x_name, y_name in [('post_traj_PC_0', 'post_traj_PC_1'), ('post_local_PC_0', 'post_local_PC_1'), ('post_morph_PC_0', 'post_morph_PC_1')]:
    for cond in np.unique(df_patient['Condition']):
        df_part = df_patient[df_patient['Condition'] == cond].reset_index(drop=True)
        # draw_umap_space(df_part, path, file_name='post PCA first patient space %s %s %s'%(cond, x_name, y_name),
        #                 condition_name='Type', label_name=None, colors=('#fdc086', '#beaed4'), dot_size=3,
        #                 x_name=x_name, y_name=y_name)
        file_name='post PCA first patient space %s %s %s'%(cond, x_name, y_name)
        font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 2

        fig, ax = plt.subplots(figsize=(4, 4))
        #
        sns.scatterplot(data=df_part, x=x_name, y=y_name, hue='Type', lw=0, s=32,
                        hue_order=['Old', 'Frail'], palette=('#beaed4', '#fdc086'))
        # sns.scatterplot(data=pred_df_, x='FAMD0', y='FAMD1', hue='pred_type', lw=0,  s=32, hue_order=['Old', 'Frail'],
        #                          palette=('#beaed4', '#fdc086'))

        handles, labels = ax.get_legend_handles_labels()

        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_color('0.2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(width=2, color='0.2')

        ax.set_xlabel('%s' % x_name, fontsize=16, weight='bold', color='0.2')
        ax.set_ylabel('%s' % y_name, fontsize=16, weight='bold', color='0.2')

        texts = []

        for i in range(df_part.shape[0]):
            texts.append(
                plt.text(x=df_part[x_name].iloc[i], y=df_part[y_name].iloc[i], s=df_part['Patient'].iloc[i],
                         fontsize=8, weight='bold', color='0.2'))
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='0.2'))

        plt.xticks(fontsize=12, color='0.2', weight='bold')
        plt.yticks(fontsize=12, color='0.2', weight='bold')

        # plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
        #            loc='best')
        ax.legend_.remove()
        plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

        if not os.path.isdir(
                path + '/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + '/svg/')
        plt.savefig(path + '/svg/%s.svg' % (file_name), bbox_inches='tight')
        plt.clf()
        plt.close()




################### Plot patient info vs Old frac ###################
for feature in ['Age', 'Gait', 'Grip', 'Frailty_score']:
    for model in ['post_traj_PC_0', 'post_local_PC_0', 'post_morph_PC_0', 'pre_traj_PC_0', 'pre_local_PC_0', 'pre_morph_PC_0']:

        file_name = '%s vs %s'%(feature, model)
        font = {'family': 'arial',
                            'weight': 'normal',
                            'size': 16}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 2
        df_patient_ = df_patient[~df_patient[feature].isnull()].reset_index(drop=True)
        df_patient_ = df_patient_[df_patient_['Condition']=='Control'].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(4,4))
        # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
        #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
        #for idx, key in enumerate(dataset):

        sns.regplot(data=df_patient_, x=feature, y=model,  scatter=False, line_kws={"color": "black"}, ax=ax)
        r, p = scipy.stats.pearsonr(df_patient_[feature], df_patient_[model])
        print(feature, r, p)
        if feature == 'FAMD1':
            plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            plt.text(0.8, 0.88, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
        else:
            plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            plt.text(0.1, 0.88, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")

        sns.scatterplot(data=df_patient_, x=feature, y=model, hue='Type', lw=0,  s=32, hue_order=['Old', 'Frail'],
                                 palette=('#beaed4', '#fdc086'))


        #ax.plot([15, 95], [15, 95], linestyle='--', color='black',  linewidth=1 )
        #sns.lineplot(x=x, y=y, label='Old', lw=2.5, dashes=False, markersize=8, err_style='bars')
            # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
            #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

        #mae = np.mean([abs(i - j) for i, j in zip(pred_ages_[model], pred_ages_['Chronological Age'])])

        # plt.text(0.2, 1.1, "MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
        #          fontsize=16, fontdict={'weight': 'bold'}, color="black")

        handles, labels = ax.get_legend_handles_labels()
        # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['mt GCB'], 0.1*std_dataset['mt GCB'], color='#CC6677',
        #            capsize=3, capthick=1, elinewidth=1.5)
        # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['wt GCB'], 0.1*std_dataset['wt GCB'], color='#888888',
        #            capsize=3, capthick=1, elinewidth=1.5)

        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_color('0.2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(width=2, color='0.2')

        ax.set_xlabel('%s'%feature, fontsize=16, weight='bold', color='0.2')
        ax.set_ylabel('%s'%model, fontsize=16, weight='bold', color='0.2')

        # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
        # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

        plt.xticks(fontsize=12, color='0.2', weight='bold')
        plt.yticks(fontsize=12, color='0.2', weight='bold')

        # plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
        #            loc='best')
        ax.legend_.remove()
        plt.savefig(path + 'various correlations/%s.png' % (file_name), dpi=300,bbox_inches='tight')

        if not os.path.isdir(path + 'various correlations/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'various correlations/svg/' )
        plt.savefig(path + 'various correlations/svg/%s.svg' % (file_name), bbox_inches='tight')
        plt.clf()
        plt.close()