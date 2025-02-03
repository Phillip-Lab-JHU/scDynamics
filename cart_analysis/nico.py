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
"""NICO"""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast
from dnn.autoencoders import set_duration_for_autoencoder
from dnn.classification import Temporal_Conv1D_2D_classifier, Res_Conv1D_LSTM_classifier
import tensorflow as tf

########################### Split dataset for training and rep ###########################
path = r'\\philliplab-server.wse.jhu.edu\data\NICO\Immune Reprogramming\CD4_DIFFERENTIATION_KINETICS\COMPLETE_ANALYSIS\SPLICE_30\\'
df1 = pd.read_parquet(path+'2d_motility_features_BR1.parquet')
df2 = pd.read_parquet(path+'2d_motility_features_BR2.parquet')
df = pd.concat([df1, df2], axis=0, ignore_index=True)

df_duration1 = pd.read_parquet(path+'2D_traj_duration_30_BR1.parquet')
df_duration2 = pd.read_parquet(path+'2D_traj_duration_30_BR2.parquet')
df_duration = pd.concat([df_duration1, df_duration2], axis=0, ignore_index=True)


#
# df['Type2'] = 'Young'
# df_duration['Type2'] = 'Young'
# df.loc[df['Type'] != 'Young', 'Type2'] = 'Old'
# df_duration.loc[df_duration['Type'] != 'Young', 'Type2'] = 'Old'


#################################### SUBTYPES classification ####################################

#################################### Training data preparation ####################################

path = r'\\philliplab-server.wse.jhu.edu\data\NICO\Immune Reprogramming\CD4_DIFFERENTIATION_KINETICS\COMPLETE_ANALYSIS\SPLICE_30\\'

duration=30
traj_scale_data = df_duration.loc[:, ['rotated_x', 'rotated_y']]
traj_scaler = StandardScaler()
traj_scale_data= pd.DataFrame(traj_scaler.fit_transform( traj_scale_data ), columns=['rotated_x', 'rotated_y'])

traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration, feature_name=['rotated_x', 'rotated_y'])
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
rotated_trajectories = dict_to_array(trajectories)
X_traj = rotated_trajectories

local_features = ['rotated_x', 'rotated_y', 'n_neighbors', 'shortest_distance']
local_scale_data = df_duration.loc[:, local_features]
local_scaler = StandardScaler()
local_scale_data= pd.DataFrame(local_scaler.fit_transform( local_scale_data ), columns=local_features)

traj_list, trajectories_array, trajectories = to_timeseries_fast(local_scale_data, duration=duration, feature_name=local_features)
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
rotated_trajectories = dict_to_array(trajectories)
X_local = rotated_trajectories

morphology_features = ['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length','minor_axis_length', 'aspect_ratio', 'circularity']
scale_data = df_duration.loc[:, morphology_features]
scaler = StandardScaler()
scale_data= pd.DataFrame(scaler.fit_transform( scale_data ), columns=morphology_features)

traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration, feature_name=morphology_features)
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
#rotated_trajectories = register_traj_disp_reflection(trajectories)
rotated_trajectories = dict_to_array(trajectories)
X_morph = rotated_trajectories


# X_covar = df_nonyoung['Condition']
# print('Classes:', pd.unique(X_covar))
# X_covar = X_covar.replace(list(pd.unique(X_covar)), [i for i in range(pd.unique(X_covar).shape[0])])
# X_covar = np.array(X_covar)
# print('Number of classes:', np.unique(X_covar).size)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_covar = le.fit_transform( np.array(df['DAY']) )
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# X_covar = ohe.fit_transform( np.array(df_nonyoung['Condition']).reshape(-1,1) )

y_names = df['SUBTYPE']
print('Classes:', pd.unique(y_names))
y = y_names.replace(list(pd.unique(y_names)), [i for i in range(pd.unique(y_names).shape[0])])
y = np.array(y)
print('Number of classes:', np.unique(y).size)

from sklearn.model_selection import train_test_split
X_traj_train, X_traj_test, X_local_train, X_local_test, X_morph_train, X_morph_test, \
X_covar_train, X_covar_test, y_train, y_test = train_test_split(X_traj, X_local, X_morph, X_covar, y, test_size=0.2, random_state=0, stratify=y)
print(X_traj_train.shape, X_traj_test.shape, X_local_train.shape, X_local_test.shape,
      X_morph_train.shape, X_morph_test.shape, X_covar_train.shape, X_covar_test.shape, y_train.shape, y_test.shape)


#################################### Model training ####################################

from tensorflow.keras import callbacks
stop_early = callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001)
# checkpointer = callbacks.ModelCheckpoint('saved_model/monocyte_project/%s'%model_name, monitor='accuracy', verbose=1,
#                 save_weights_only=False, save_freq='epoch', every_n_epochs=10, mode='auto', save_best_only=True)  # Save model every epoch

from dnn.classification import scTRAIT
sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=np.unique(y).size)
result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=500,
                     verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
#3 class scTRAIT 2
### model2: Vanilla trajectory ###
vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=np.unique(y).size)
result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=500, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])

### model3: Vanilla morphology ###
vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=np.unique(y).size)
result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=500, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])

#################################### Model prediction ####################################

y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
y_pred2 = vt.predict([X_traj_test])
y_pred3 = vm.predict([X_morph_test])

#################################### Model accuracy evaluation ####################################
draw_confusion_matrix(y_pred, y_test, y_names, path, figsize=(8,8), file_name='scTRAIT confusion matrix', vmax=0.9)
draw_confusion_matrix(y_pred2, y_test, y_names, path, figsize=(8,8), file_name='vanilla traj confusion matrix', vmax=0.9)
draw_confusion_matrix(y_pred3, y_test, y_names, path, figsize=(8,8), file_name='vanilla morpho confusion matrix', vmax=0.9)


from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

if y_pred.shape[1]>=3:
    y_class = np.argmax(y_pred, axis=1)
    y_class2 = np.argmax(y_pred2, axis=1)
    y_class3 = np.argmax(y_pred3, axis=1)

else:
    y_class = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred)])
    y_class2 = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred2)])
    y_class3 = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred3)])

# f1 = f1_score(y_test, y_class)
# f1_2 = f1_score(y_test, y_class2)
# f1_3 = f1_score(y_test, y_class3)

acs = accuracy_score(y_test, y_class)
acs2 = accuracy_score(y_test, y_class2)
acs3 = accuracy_score(y_test, y_class3)

# ps = precision_score(y_test, y_class)
# ps2 = precision_score(y_test, y_class2)
# ps3 = precision_score(y_test, y_class3)
#
# rs = recall_score(y_test, y_class)
# rs2 = recall_score(y_test, y_class2)
# rs3 = recall_score(y_test, y_class3)

#print(f1, f1_2, f1_3, acs, acs2, acs3, ps, ps2, ps3, rs, rs2, rs3)
print(acs, acs2, acs3)

dict_datasets = {'VM': np.array(acs3), 'VT': np.array(acs2), 'scTRAIT': np.array(acs)}
#dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='accuracy scores', colors=('#6699CC', '#CC6677', '#aa4499'), vmax=1,
                     strip_plot=False, test='t-test', pvalue=False, figsize=(1,2))



