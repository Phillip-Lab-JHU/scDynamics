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

"""Generates Data for Figure 6 age and frailty score (regression)."""
import pandas as pd
from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast
from dnn.autoencoders import set_duration_for_autoencoder
import tensorflow as tf
from dnn.classification import scTRAIT, Temporal_Conv1D_2D_classifier

#################################### Regression: leave-one-out for only young ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

# Old and young only
df = df[df['Type']=='Young'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']=='Young'].reset_index(drop=True)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\age regression\\'

for patient in tqdm(np.unique(df['Patient'])):
    print('Patient %s starting'%patient)
    df_part = df[df['Patient']!=patient].reset_index(drop=True)
    df_duration_part = df_duration[df_duration['Patient']!=patient].reset_index(drop=True)

    df_part_test = df[df['Patient']==patient].reset_index(drop=True)
    df_duration_part_test = df_duration[df_duration['Patient']==patient].reset_index(drop=True)

    ############# training set #############
    duration=30
    traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part, duration=duration, feature_name=['reg_x', 'reg_y'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
    #rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_traj_train = rotated_trajectories


    traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part, duration=duration, feature_name=['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
    #rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_local_train = rotated_trajectories


    traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part, duration=duration,
                                                                     feature_name=['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length',
                                                                                   'minor_axis_length', 'aspect_ratio', 'circularity'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
    #rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_morph_train = rotated_trajectories

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_covar_train = le.fit_transform( np.array(df_part['Condition']) )
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    #print(le_name_mapping)

    # from sklearn.preprocessing import OneHotEncoder
    # ohe = OneHotEncoder()
    # X_covar = ohe.fit_transform( np.array(df_nonyoung['Condition']).reshape(-1,1) )

    y_train = np.array(df_part['Age'])
    #print('Ages:', pd.unique(y_train))
    #print('Number of ages:', np.unique(y_train).size)


    ############# testing set #############
    traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part_test, duration=duration, feature_name=['reg_x', 'reg_y'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
    #rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_traj_test = rotated_trajectories


    traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part_test, duration=duration, feature_name=['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
    #rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_local_test = rotated_trajectories


    traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part_test, duration=duration,
                                                                     feature_name=['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length',
                                                                                   'minor_axis_length', 'aspect_ratio', 'circularity'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
    #rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_morph_test = rotated_trajectories


    #le = LabelEncoder()
    X_covar_test = le.transform( np.array(df_part_test['Condition']) )
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    #print(le_name_mapping)

    # from sklearn.preprocessing import OneHotEncoder
    # ohe = OneHotEncoder()
    # X_covar = ohe.fit_transform( np.array(df_nonyoung['Condition']).reshape(-1,1) )

    y_test = np.array(df_part_test['Age'])
    #print('Ages:', pd.unique(y_test))
    #print('Number of ages:', np.unique(y_test).size)


    #################################### Model training ####################################

    from tensorflow.keras import callbacks
    stop_early = callbacks.EarlyStopping(monitor='loss', patience=30, min_delta=0.0001)
    # checkpointer = callbacks.ModelCheckpoint('saved_model/monocyte_project_v2/%s'%model_name, monitor='accuracy', verbose=1,
    #                 save_weights_only=False, save_freq='epoch', every_n_epochs=10, mode='auto', save_best_only=True)  # Save model every epoch

    from dnn.classification import scTRAIT, Temporal_Conv1D_2D_classifier
    sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=0)
    result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=1000,
                         verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
    sctrait.save('saved_model/monocyte_project_v2/young regression scTRAIT %s'%patient, save_format='tf')

    ### model2: Vanilla trajectory ###
    vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=0)
    result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                         callbacks=[stop_early])
    vt.save('saved_model/monocyte_project_v2/young regression vanilla traj %s'%patient, save_format='tf')

    ### model3: Vanilla morphology ###
    vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=0)
    result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                         callbacks=[stop_early])
    vm.save('saved_model/monocyte_project_v2/young regression vanilla morpho %s'%patient, save_format='tf')

    y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
    y_pred2 = vt.predict([X_traj_test])
    y_pred3 = vm.predict([X_morph_test])


    # draw_confusion_matrix(y_pred, y_test, y_names, path, figsize=(8,8), file_name='%s scTRAIT confusion matrix'%patient)
    # draw_confusion_matrix(y_pred2, y_test, y_names, path, figsize=(8,8), file_name='%s vanilla traj confusion matrix'%patient)
    # draw_confusion_matrix(y_pred3, y_test, y_names, path, figsize=(8,8), file_name='%s vanilla morph confusion matrix'%patient)
    print(patient, y_test[0], np.mean(y_pred), np.mean(y_pred2), np.mean(y_pred3))

    np.save(path + 'young only scTRAIT %s result.npy' % patient, y_pred)
    np.save(path + 'young only VT %s result.npy' % patient, y_pred2)
    np.save(path + 'young only VM %s result.npy' % patient, y_pred3)

#################################### Regression: leave-one-out for nonfrail ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Frail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Frail'].reset_index(drop=True)

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df = df[df['Age']!=55].reset_index(drop=True)
df_duration = df_duration[df_duration['Age']!=55].reset_index(drop=True)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\age regression\\'

for patient in tqdm(np.unique(df['Patient'])):
    print('Patient %s starting'%patient)
    df_part = df[df['Patient']!=patient].reset_index(drop=True)
    df_duration_part = df_duration[df_duration['Patient']!=patient].reset_index(drop=True)

    df_part_test = df[df['Patient']==patient].reset_index(drop=True)
    df_duration_part_test = df_duration[df_duration['Patient']==patient].reset_index(drop=True)

    duration = 30

    traj_scale_data = df_duration_part.loc[:, ['reg_x', 'reg_y']]
    traj_scaler = StandardScaler()
    traj_scale_data = pd.DataFrame(traj_scaler.fit_transform(traj_scale_data), columns=['reg_x', 'reg_y'])

    traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration,
                                                                     feature_name=['reg_x', 'reg_y'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
    rotated_trajectories = dict_to_array(trajectories)
    X_traj_train = rotated_trajectories

    local_features = ['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance']
    local_scale_data = df_duration_part.loc[:, local_features]
    local_scaler = StandardScaler()
    local_scale_data = pd.DataFrame(local_scaler.fit_transform(local_scale_data), columns=local_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(local_scale_data, duration=duration,
                                                                     feature_name=local_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
    rotated_trajectories = dict_to_array(trajectories)
    X_local_train = rotated_trajectories

    morphology_features = ['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length', 'minor_axis_length',
                           'aspect_ratio', 'circularity']
    scale_data = df_duration_part.loc[:, morphology_features]
    scaler = StandardScaler()
    scale_data = pd.DataFrame(scaler.fit_transform(scale_data), columns=morphology_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration,
                                                                     feature_name=morphology_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
    rotated_trajectories = dict_to_array(trajectories)
    X_morph_train = rotated_trajectories

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    X_covar_train = le.fit_transform(np.array(df_part['Condition']))
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    y_train = np.array(df_part['Age'])

    from sklearn.utils import shuffle
    X_traj_train, X_local_train, X_morph_train, X_covar_train, y_train = shuffle(X_traj_train, X_local_train,
                                                                                 X_morph_train, X_covar_train, y_train,
                                                                                 random_state=0)
    ############# testing set #############
    traj_scale_data = df_duration_part.loc[:, ['reg_x', 'reg_y']]
    traj_scaler = StandardScaler()
    traj_scaler.fit(traj_scale_data)
    traj_scale_data = df_duration_part_test.loc[:, ['reg_x', 'reg_y']]
    traj_scale_data = pd.DataFrame(traj_scaler.transform(traj_scale_data), columns=['reg_x', 'reg_y'])

    traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration,
                                                                     feature_name=['reg_x', 'reg_y'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
    # rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_traj_test = rotated_trajectories

    local_features = ['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance']
    local_scale_data = df_duration_part.loc[:, local_features]
    local_scaler = StandardScaler()
    local_scaler.fit(local_scale_data)
    local_scale_data = df_duration_part_test.loc[:, local_features]
    local_scale_data = pd.DataFrame(local_scaler.transform(local_scale_data), columns=local_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(local_scale_data, duration=duration,
                                                                     feature_name=local_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
    # rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_local_test = rotated_trajectories

    morphology_features = ['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length', 'minor_axis_length',
                           'aspect_ratio', 'circularity']
    scale_data = df_duration_part.loc[:, morphology_features]
    scaler = StandardScaler()
    scaler.fit(scale_data)
    scale_data = df_duration_part_test.loc[:, morphology_features]
    scale_data = pd.DataFrame(scaler.transform(scale_data), columns=morphology_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration,
                                                                     feature_name=morphology_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
    # rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_morph_test = rotated_trajectories

    le = LabelEncoder()
    le.fit(np.array(df_part['Condition']))
    X_covar_test = le.transform(np.array(df_part_test['Condition']))
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    y_test = np.array(df_part_test['Age'])

    #################################### Model training ####################################

    from tensorflow.keras import callbacks
    stop_early = callbacks.EarlyStopping(monitor='loss', patience=30, min_delta=0.0001)
    # checkpointer = callbacks.ModelCheckpoint('saved_model/monocyte_project_v2/%s'%model_name, monitor='accuracy', verbose=1,
    #                 save_weights_only=False, save_freq='epoch', every_n_epochs=10, mode='auto', save_best_only=True)  # Save model every epoch
    #
    from dnn.classification import scTRAIT, Temporal_Conv1D_2D_classifier
    sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=0)
    result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=1000,
                         verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
    sctrait.save('saved_model/monocyte_project_v2/wo55 frailty score regression scTRAIT %s'%patient, save_format='tf')

    ### model2: Vanilla trajectory ###
    vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=0)
    result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                         callbacks=[stop_early])
    vt.save('saved_model/monocyte_project_v2/wo55 frailty score regression vanilla traj %s'%patient, save_format='tf')

    ### model3: Vanilla morphology ###
    vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=0)
    result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                         callbacks=[stop_early])
    vm.save('saved_model/monocyte_project_v2/wo55 frailty score regression vanilla morpho %s'%patient, save_format='tf')

    y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
    y_pred2 = vt.predict([X_traj_test])
    y_pred3 = vm.predict([X_morph_test])


    # print(patient, y_test[0], np.mean(y_pred),
    #       #np.mean(y_pred2), np.mean(y_pred3)
    #       )

    np.save(path + 'nonfrail scTRAIT %s result.npy' % patient, y_pred)
    np.save(path + 'nonfrail VT %s result.npy' % patient, y_pred2)
    np.save(path + 'nonfrail VM %s result.npy' % patient, y_pred3)

#################################### Regression: Model evaluation with training with nonfrail(young+old) and predicting with frail ####################################

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df_frail = df[df['Type']=='Frail'].reset_index(drop=True)
df_duration_frail = df_duration[df_duration['Type']=='Frail'].reset_index(drop=True)

df_prefrail = df[df['Type']=='Prefrail'].reset_index(drop=True)
df_duration_prefrail = df_duration[df_duration['Type']=='Prefrail'].reset_index(drop=True)

# Young and Old only
df = df[df['Type']!='Frail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Frail'].reset_index(drop=True)
df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df = df[df['Age']!=55].reset_index(drop=True)
df_duration = df_duration[df_duration['Age']!=55].reset_index(drop=True)

duration = 30

traj_scale_data = df_duration.loc[:, ['reg_x', 'reg_y']]
traj_scaler = StandardScaler()
traj_scale_data = pd.DataFrame(traj_scaler.fit_transform(traj_scale_data), columns=['reg_x', 'reg_y'])

traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration,
                                                                 feature_name=['reg_x', 'reg_y'])
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
rotated_trajectories = dict_to_array(trajectories)
X_traj_train = rotated_trajectories

local_features = ['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance']
local_scale_data = df_duration.loc[:, local_features]
local_scaler = StandardScaler()
local_scale_data = pd.DataFrame(local_scaler.fit_transform(local_scale_data), columns=local_features)

traj_list, trajectories_array, trajectories = to_timeseries_fast(local_scale_data, duration=duration,
                                                                 feature_name=local_features)
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
rotated_trajectories = dict_to_array(trajectories)
X_local_train = rotated_trajectories

morphology_features = ['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length', 'minor_axis_length',
                       'aspect_ratio', 'circularity']
scale_data = df_duration.loc[:, morphology_features]
scaler = StandardScaler()
scale_data = pd.DataFrame(scaler.fit_transform(scale_data), columns=morphology_features)

traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration,
                                                                 feature_name=morphology_features)
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
rotated_trajectories = dict_to_array(trajectories)
X_morph_train = rotated_trajectories

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X_covar_train = le.fit_transform(np.array(df['Condition']))
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# X_covar = ohe.fit_transform( np.array(df_nonyoung['Condition']).reshape(-1,1) )

y_train = np.array(df['Age'])

from sklearn.utils import shuffle
X_traj_train, X_local_train, X_morph_train, X_covar_train, y_train = shuffle(X_traj_train, X_local_train,
                                                                             X_morph_train, X_covar_train, y_train,
                                                                             random_state=0)
from tensorflow.keras import callbacks
stop_early = callbacks.EarlyStopping(monitor='loss', patience=30, min_delta=0.0001)

from dnn.classification import scTRAIT, Temporal_Conv1D_2D_classifier
sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=0)
result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=1000,
                     verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
sctrait.save('saved_model/monocyte_project_v2/wo55 frailty score nonfrail regression scTRAIT', save_format='tf')

### model2: Vanilla trajectory ###
vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=0)
result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vt.save('saved_model/monocyte_project_v2/wo55 frailty score nonfrail regression vanilla traj', save_format='tf')

### model3: Vanilla morphology ###
vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=0)
result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vm.save('saved_model/monocyte_project_v2/wo55 frailty score nonfrail regression vanilla morpho', save_format='tf')

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\age regression\\'

for patient in tqdm(np.unique(df_frail['Patient'])):
    df_part_test = df_frail[df_frail['Patient']==patient].reset_index(drop=True)
    df_duration_part_test = df_duration_frail[df_duration_frail['Patient'] == patient].reset_index(drop=True)
    ############# testing set #############
    traj_scale_data = df_duration.loc[:, ['reg_x', 'reg_y']]
    traj_scaler = StandardScaler()
    traj_scaler.fit(traj_scale_data)
    traj_scale_data = df_duration_part_test.loc[:, ['reg_x', 'reg_y']]
    traj_scale_data = pd.DataFrame(traj_scaler.transform(traj_scale_data), columns=['reg_x', 'reg_y'])

    traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration,
                                                                     feature_name=['reg_x', 'reg_y'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
    # rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_traj_test = rotated_trajectories

    local_features = ['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance']
    local_scale_data = df_duration.loc[:, local_features]
    local_scaler = StandardScaler()
    local_scaler.fit(local_scale_data)
    local_scale_data = df_duration_part_test.loc[:, local_features]
    local_scale_data = pd.DataFrame(local_scaler.transform(local_scale_data), columns=local_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(local_scale_data, duration=duration,
                                                                     feature_name=local_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
    # rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_local_test = rotated_trajectories

    morphology_features = ['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length', 'minor_axis_length',
                           'aspect_ratio', 'circularity']
    scale_data = df_duration.loc[:, morphology_features]
    scaler = StandardScaler()
    scaler.fit(scale_data)
    scale_data = df_duration_part_test.loc[:, morphology_features]
    scale_data = pd.DataFrame(scaler.transform(scale_data), columns=morphology_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration,
                                                                     feature_name=morphology_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
    # rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_morph_test = rotated_trajectories

    le = LabelEncoder()
    le.fit(np.array(df['Condition']))
    X_covar_test = le.transform(np.array(df_part_test['Condition']))
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    y_test = np.array(df_part_test['Age'])

    y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
    y_pred2 = vt.predict([X_traj_test])
    y_pred3 = vm.predict([X_morph_test])


    #print(patient, y_test[0], np.mean(y_pred), np.mean(y_pred2), np.mean(y_pred3))

    np.save(path + 'nonfrail predict frail scTRAIT %s result.npy' % patient, y_pred)
    np.save(path + 'nonfrail predict frail VT %s result.npy' % patient, y_pred2)
    np.save(path + 'nonfrail predict frail VM %s result.npy' % patient, y_pred3)

for patient in tqdm(np.unique(df_prefrail['Patient'])):
    df_part_test = df_prefrail[df_prefrail['Patient']==patient].reset_index(drop=True)
    df_duration_part_test = df_duration_prefrail[df_duration_prefrail['Patient'] == patient].reset_index(drop=True)
    ############# testing set #############
    traj_scale_data = df_duration.loc[:, ['reg_x', 'reg_y']]
    traj_scaler = StandardScaler()
    traj_scaler.fit(traj_scale_data)
    traj_scale_data = df_duration_part_test.loc[:, ['reg_x', 'reg_y']]
    traj_scale_data = pd.DataFrame(traj_scaler.transform(traj_scale_data), columns=['reg_x', 'reg_y'])

    traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration,
                                                                     feature_name=['reg_x', 'reg_y'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
    # rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_traj_test = rotated_trajectories

    local_features = ['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance']
    local_scale_data = df_duration.loc[:, local_features]
    local_scaler = StandardScaler()
    local_scaler.fit(local_scale_data)
    local_scale_data = df_duration_part_test.loc[:, local_features]
    local_scale_data = pd.DataFrame(local_scaler.transform(local_scale_data), columns=local_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(local_scale_data, duration=duration,
                                                                     feature_name=local_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
    # rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_local_test = rotated_trajectories

    morphology_features = ['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length', 'minor_axis_length',
                           'aspect_ratio', 'circularity']
    scale_data = df_duration.loc[:, morphology_features]
    scaler = StandardScaler()
    scaler.fit(scale_data)
    scale_data = df_duration_part_test.loc[:, morphology_features]
    scale_data = pd.DataFrame(scaler.transform(scale_data), columns=morphology_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration,
                                                                     feature_name=morphology_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
    # rotated_trajectories = register_traj_disp_reflection(trajectories)
    rotated_trajectories = dict_to_array(trajectories)
    X_morph_test = rotated_trajectories

    le = LabelEncoder()
    le.fit(np.array(df['Condition']))
    X_covar_test = le.transform(np.array(df_part_test['Condition']))
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    y_test = np.array(df_part_test['Age'])

    y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
    y_pred2 = vt.predict([X_traj_test])
    y_pred3 = vm.predict([X_morph_test])


    #print(patient, y_test[0], np.mean(y_pred), np.mean(y_pred2), np.mean(y_pred3))

    np.save(path + 'nonfrail predict prefrail scTRAIT %s result.npy' % patient, y_pred)
    np.save(path + 'nonfrail predict prefrail VT %s result.npy' % patient, y_pred2)
    np.save(path + 'nonfrail predict prefrail VM %s result.npy' % patient, y_pred3)

#################################### Read regression accuracies ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

# df = df[df['Age']!=55].reset_index(drop=True)
# df_duration = df_duration[df_duration['Age']!=55].reset_index(drop=True)

patients = np.unique(df['Patient'])

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\age regression\\'
load_path = path+'results/'

young_thresh = 30
old_thresh = 80

pred_ages = {'scTRAIT': [], 'scTRAIT_sd': [], 'VT':[], 'VT_sd':[], 'VM': [], 'VM_sd': [],
             'scTRAIT_youngfrac': [], 'VT_youngfrac': [], 'VM_youngfrac': [],
             'scTRAIT_oldfrac': [], 'VT_oldfrac': [], 'VM_oldfrac': [],
             'scTRAIT_diff': [], 'VT_diff': [], 'VM_diff': [],
             'gmm_young_frac':[], 'gmm_old_frac':[], 'gmm_diff':[],
             'Chronological Age': [], 'Type':[], 'Patient':[],
             'Weakness':[], 'Weight loss':[], 'Exhaustion':[], 'Activity':[], 'Gait':[], 'Grip':[], 'Frailty score': []}

pred_age_distribution = {'Old':[], 'Frail':[], 'Young':[]}
old_temp = []
frail_temp = []
young_temp = []
for patient in patients:
    age_group = np.unique(df[df['Patient']==patient]['Type'])[0]
    age = np.unique(df[df['Patient'] == patient]['Age'])[0]

    weakness = np.unique(df[df['Patient'] == patient]['Weakness'])[0]
    weight_loss = np.unique(df[df['Patient'] == patient]['Weight_loss'])[0]
    exhaustion = np.unique(df[df['Patient'] == patient]['Exhaustion'])[0]
    activity = np.unique(df[df['Patient'] == patient]['Activity'])[0]
    gait = np.unique(df[df['Patient'] == patient]['Gait'])[0]
    grip = np.unique(df[df['Patient'] == patient]['Grip'])[0]
    fs = np.unique(df[df['Patient'] == patient]['Frailty_score'])[0]

    # if age_group == 'Young':
    #     pred_sctrait = np.load(load_path + 'young only scTRAIT %s result.npy' % patient)
    #     pred_vt = np.load(load_path + 'young only VT %s result.npy' % patient)
    #     pred_vm = np.load(load_path + 'young only VM %s result.npy' % patient)
    # else:
    #     pred_sctrait = np.load(load_path + 'nonyoung scTRAIT %s result.npy' % patient)
    #     pred_vt = np.load(load_path + 'nonyoung VT %s result.npy' % patient)
    #     pred_vm = np.load(load_path + 'nonyoung VM %s result.npy' % patient)

    if age_group == 'Frail':
        pred_sctrait = np.load(load_path + 'nonfrail predict frail scTRAIT %s result.npy' % patient)
        pred_vt = np.load(load_path + 'nonfrail predict frail VT %s result.npy' % patient)
        pred_vm = np.load(load_path + 'nonfrail predict frail VM %s result.npy' % patient)

    elif age_group == 'Prefrail':
        pred_sctrait = np.load(load_path + 'nonfrail predict prefrail scTRAIT %s result.npy' % patient)
        pred_vt = np.load(load_path + 'nonfrail predict prefrail VT %s result.npy' % patient)
        pred_vm = np.load(load_path + 'nonfrail predict prefrail VM %s result.npy' % patient)

    else:
        pred_sctrait = np.load(load_path + 'nonfrail scTRAIT %s result.npy' % patient)
        pred_vt = np.load(load_path + 'nonfrail VT %s result.npy' % patient)
        pred_vm = np.load(load_path + 'nonfrail VM %s result.npy' % patient)

    #pred_sctrait = np.load(path+'no frail scTRAIT %s result.npy' % patient)
    #pred_vt = np.load(path + 'no frail VT %s result.npy' % patient)
    #pred_vm = np.load(path + 'no frail VM %s result.npy' % patient)
    pred_sctrait = pred_sctrait.flatten()
    pred_vt = pred_vt.flatten()
    pred_vm = pred_vm.flatten()

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2, covariance_type='tied', random_state=0)
    gmm_predicted = gmm.fit_predict(pred_sctrait.reshape(-1, 1))
    mean0, mean1 = gmm.means_.flatten()
    if mean0 <= mean1:
        pass
    elif mean0 > mean1:
        mean0, mean1 = mean1, mean0
        gmm_predicted = 1 - gmm_predicted

    young_fraction = gmm_predicted[gmm_predicted==0].size / gmm_predicted.size
    old_fraction = gmm_predicted[gmm_predicted==1].size / gmm_predicted.size
    diff = mean1 - mean0
    pred_ages['gmm_young_frac'].append(young_fraction)
    pred_ages['gmm_old_frac'].append(old_fraction)
    pred_ages['gmm_diff'].append(diff)

    #pred_ages['scTRAIT_sd'].append(np.std(pred_sctrait)/np.mean(pred_sctrait))
    pred_ages['scTRAIT'].append(np.median(pred_sctrait))
    pred_ages['scTRAIT_sd'].append(np.std(pred_sctrait))
    young_fraction = pred_sctrait[pred_sctrait <= young_thresh].size/pred_sctrait.size
    pred_ages['scTRAIT_youngfrac'].append(young_fraction)
    old_fraction = pred_sctrait[pred_sctrait >= old_thresh].size / pred_sctrait.size
    pred_ages['scTRAIT_oldfrac'].append(old_fraction)
    pred_ages['scTRAIT_diff'].append(old_fraction - young_fraction)

    #pred_ages['VT_sd'].append(np.std(pred_vt)/np.mean(pred_vt))
    pred_ages['VT'].append(np.median(pred_vt))
    pred_ages['VT_sd'].append(np.std(pred_vt))
    young_fraction = pred_vt[pred_vt <= young_thresh].size / pred_vt.size
    pred_ages['VT_youngfrac'].append(young_fraction)
    old_fraction = pred_vt[pred_vt >= old_thresh].size / pred_vt.size
    pred_ages['VT_oldfrac'].append(old_fraction)
    pred_ages['VT_diff'].append(old_fraction - young_fraction)

    #pred_ages['VM_sd'].append(np.std(pred_vm)/np.mean(pred_vm))
    pred_ages['VM'].append(np.median(pred_vm))
    pred_ages['VM_sd'].append(np.std(pred_vm))
    young_fraction = pred_vm[pred_vm <= young_thresh].size / pred_vm.size
    pred_ages['VM_youngfrac'].append(young_fraction)
    old_fraction = pred_vm[pred_vm >= old_thresh].size / pred_vm.size
    pred_ages['VM_oldfrac'].append(old_fraction)
    pred_ages['VM_diff'].append(old_fraction - young_fraction)


    pred_ages['Chronological Age'].append(age)
    pred_ages['Type'].append(age_group)
    pred_ages['Patient'].append(patient)

    pred_ages['Weakness'].append(weakness)
    pred_ages['Weight loss'].append(weight_loss)
    pred_ages['Exhaustion'].append(exhaustion)
    pred_ages['Activity'].append(activity)
    pred_ages['Gait'].append(gait)
    pred_ages['Grip'].append(grip)
    pred_ages['Frailty score'].append(fs)

    #print(patient, age_group, age, np.mean(pred_sctrait), np.mean(pred_vt), np.mean(pred_vm))
    #print(patient, age_group, age, np.median(pred_sctrait), np.median(pred_vt), np.median(pred_vm))
    #print(patient, age_group, age, np.median(pred_sctrait))

    if (age_group=='Old'):
        old_temp.append(pred_sctrait)

    elif (age_group=='Frail'):
        frail_temp.append(pred_sctrait)

    elif (age_group=='Young'):
        young_temp.append(pred_sctrait)

old_temp = flatten_list_of_list(old_temp)
frail_temp = flatten_list_of_list(frail_temp)
young_temp = flatten_list_of_list(young_temp)

pred_age_distribution['Old'] = old_temp
pred_age_distribution['Frail'] = frail_temp
pred_age_distribution['Young'] = young_temp

pred_df = pd.DataFrame(pred_ages)


pred_df.loc[(pred_df['scTRAIT_oldfrac'] >=0.5), 'pred_type'] = 'Frail'
pred_df.loc[(pred_df['scTRAIT_oldfrac'] <0.5), 'pred_type'] = 'Old'




################### Confusion matrix ###################
df_confusion = pred_df[(pred_df['Type']=='Old') | (pred_df['Type']=='Frail')].reset_index(drop=True)

y_class = df_confusion['pred_type']
y_test = df_confusion['Type']

accuracy = np.sum(y_test == y_class) / y_test.size

vmax=0.9
file_name = 'scTRAIT confusion matrix'
figsize=(4,4)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_class)
norm_cm = cm / np.sum(cm, axis=1)[:, np.newaxis]

numer = cm
d = np.sum(cm, axis=1)[:, np.newaxis]
denom = np.repeat(d, 2, axis=1)

result = np.core.defchararray.add(numer.astype('str'), '/')
annot = np.core.defchararray.add(result, denom.astype('str'))

import seaborn as sns

font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=figsize)
ax = sns.heatmap(norm_cm, annot=annot, fmt='', annot_kws={'size': 18, 'weight': 'bold'}, linewidths=0.5, linecolor='black', alpha=0.8, cmap='Blues', vmax=vmax)
ax.set_xticklabels(['Frail', 'Old'], rotation=0, fontsize=16, weight='bold')
ax.set_yticklabels(['Frail', 'Old'], rotation=0, fontsize=16, weight='bold')
ax.set_xlabel('Predicted', fontsize=16, weight='bold', color='0.2')
ax.set_ylabel('Truth', fontsize=16, weight='bold', color='0.2')

plt.savefig(path + '%s.png'%file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()

################### Plot Predicted Age KDE of Frail vs Old ###################
sorted_keys, sorted_vals = list(pred_age_distribution.keys()), list(pred_age_distribution.values())

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

colors=('#beaed4', '#fdc086', '#7fc97f')


for_data = sns.kdeplot([*old_temp, *frail_temp, *young_temp], fill=False, linewidth=0.7)
data = for_data.lines[-1].get_xydata()
xs, ys = data[:,0], data[:, 1]
max_idx = np.argmax(ys)
max_idx2 = np.argmax(ys[xs<=50])

print('two peaks: ', xs[max_idx], xs[max_idx2])
fig, ax = plt.subplots(figsize=(2,2))
for i, key in enumerate(pred_age_distribution):
    ax = sns.kdeplot(data=pred_age_distribution[key], fill=True, linewidth=1, color=colors[i], label=key) #clip=(0, 0.5),



# ax = sns.kdeplot(data=dict_datasets['wt GCB'], fill=True, linewidth=1, clip=(0, 40), color='#888888', label='wt GCB')
# ax = sns.kdeplot(data=dict_datasets['mt GCB'], fill=True, linewidth=1, clip=(0, 40), color='#CC6677', label='mt GCB')

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Predicted age', fontsize=8, weight='bold', color='0.2')
ax.set_ylabel('Density', fontsize=8, weight='bold', color='0.2')
plt.xticks(fontsize=8, color='0.2', weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')

#plt.legend(frameon=False, prop = {'weight':'bold', 'size':8}, labelcolor='0.2')

plt.savefig(path+'predicted age distribution.png', dpi=300, bbox_inches='tight')
plt.close()
plt.clf()

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/predicted age distribution.svg', bbox_inches='tight')
plt.close()
plt.clf()

#pred_df['pred_fs'] = pred_df['scTRAIT_oldfrac'] / ( pred_df['scTRAIT_youngfrac'] + 1)

################### Plot Chronological age vs Old frac ###################
for model in ['scTRAIT', 'VT', 'VM']:

    file_name = 'Old fraction prediction_%s'%model
    font = {'family': 'arial',
                        'weight': 'normal',
                        'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(4,4))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    #for idx, key in enumerate(dataset):
    sns.scatterplot(data=pred_df, x='Chronological Age', y='%s_oldfrac'%model, hue='Type', lw=0,  s=32, hue_order=['Young', 'Old', 'Frail', 'Prefrail'],
                             palette=('#7fc97f', '#beaed4', '#fdc086', '#888888'))
    #ax.plot([15, 95], [15, 95], linestyle='--', color='black',  linewidth=1 )
    #sns.lineplot(x=x, y=y, label='Old', lw=2.5, dashes=False, markersize=8, err_style='bars')
        # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
        #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

    #mae = np.mean([abs(i - j) for i, j in zip(pred_ages[model], pred_ages['Chronological Age'])])

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

    ax.set_xlabel('Chronological Age (yr)', fontsize=16, weight='bold', color='0.2')
    ax.set_ylabel('Old fraction (%)', fontsize=16, weight='bold', color='0.2')

    custom_range = (15, 95)
    #custom_range = (70, 95)
    label_stepsize = 10
    # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
    # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

    plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
               weight='bold')
    plt.yticks(fontsize=12, color='0.2', weight='bold')

    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
               loc='best')
    plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/' )
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
    plt.clf()
    plt.close()

################### Plot Chronological age vs GMM prediction ###################
for model in ['gmm_young_frac', 'gmm_old_frac', 'gmm_diff']:

    file_name = 'GMM prediction_%s'%model
    font = {'family': 'arial',
                        'weight': 'normal',
                        'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(4,4))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    #for idx, key in enumerate(dataset):
    sns.scatterplot(data=pred_df, x='Chronological Age', y='%s'%model, hue='Type', lw=0,  s=32, hue_order=['Young', 'Old', 'Frail', 'Prefrail'],
                             palette=('#7fc97f', '#beaed4', '#fdc086', '#888888'))
    #ax.plot([15, 95], [15, 95], linestyle='--', color='black',  linewidth=1 )
    #sns.lineplot(x=x, y=y, label='Old', lw=2.5, dashes=False, markersize=8, err_style='bars')
        # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
        #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

    #mae = np.mean([abs(i - j) for i, j in zip(pred_ages[model], pred_ages['Chronological Age'])])

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

    ax.set_xlabel('Chronological Age (yr)', fontsize=16, weight='bold', color='0.2')
    ax.set_ylabel('%s'%model, fontsize=16, weight='bold', color='0.2')

    custom_range = (15, 95)
    #custom_range = (70, 95)
    label_stepsize = 10
    # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
    # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

    plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
               weight='bold')
    plt.yticks(fontsize=12, color='0.2', weight='bold')

    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
               loc='best')
    plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/' )
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
    plt.clf()
    plt.close()


################### Plot Chronological age vs Predicted Age ###################
for model in ['scTRAIT', 'VT', 'VM']:

    file_name = 'Age prediction_%s'%model
    font = {'family': 'arial',
                        'weight': 'normal',
                        'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(4,4))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    #for idx, key in enumerate(dataset):
    sns.scatterplot(data=pred_ages, x='Chronological Age', y=model, hue='Type', lw=0,  s=32, hue_order=['Young', 'Old', 'Frail', 'Prefrail'],
                             palette=('#7fc97f', '#beaed4', '#fdc086', '#888888'))
    #ax.plot([15, 95], [15, 95], linestyle='--', color='black',  linewidth=1 )
    #sns.lineplot(x=x, y=y, label='Old', lw=2.5, dashes=False, markersize=8, err_style='bars')
        # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
        #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

    #mae = np.mean([abs(i - j) for i, j in zip(pred_ages[model], pred_ages['Chronological Age'])])
    #
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

    ax.set_xlabel('Chronological Age (yr)', fontsize=16, weight='bold', color='0.2')
    ax.set_ylabel('Predicted Age (yr)', fontsize=16, weight='bold', color='0.2')

    #custom_range = (15, 95)
    #label_stepsize = 10
    # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
    # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

    plt.xticks(fontsize=12, color='0.2', weight='bold')
    plt.yticks(fontsize=12, color='0.2', weight='bold')

    # plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
    #            weight='bold')
    # plt.yticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2', weight='bold')

    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
               loc='best')
    plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/' )
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
    plt.clf()
    plt.close()

################### Plot Predicted young frac vs Predicted Old frac ###################

file_name = 'pred young frac vs pred old frac'
font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(4,4))
# sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
#                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
#for idx, key in enumerate(dataset):
sns.scatterplot(data=pred_ages, x='scTRAIT_youngfrac', y='scTRAIT_oldfrac', hue='Type', lw=0,  s=32, hue_order=['Young', 'Old', 'Frail', 'Prefrail'],
                         palette=('#7fc97f', '#beaed4', '#fdc086', '#888888'))
#ax.plot([15, 95], [15, 95], linestyle='--', color='black',  linewidth=1 )
#sns.lineplot(x=x, y=y, label='Old', lw=2.5, dashes=False, markersize=8, err_style='bars')
    # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
    #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)


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

ax.set_xlabel('Predicted young frac', fontsize=16, weight='bold', color='0.2')
ax.set_ylabel('Predicted Old frac', fontsize=16, weight='bold', color='0.2')

plt.xticks(fontsize=12, color='0.2', weight='bold')
plt.yticks(fontsize=12, color='0.2', weight='bold')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
           loc='best')
plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()

################### KDEPlot overlayed Predicted young frac vs Predicted Old frac ###################

file_name = 'KDEplot pred young frac vs pred old frac'
font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(4,4))
# sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
#                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
#for idx, key in enumerate(dataset):
sns.kdeplot(data=pred_df[pred_df['Type']!='Prefrail'], x='scTRAIT_youngfrac', y='scTRAIT_oldfrac',  hue='Type', alpha=0.7, ax=ax, zorder=2, linewidths=1.5, palette=('#7fc97f', '#beaed4', '#fdc086',),
            hue_order=['Young', 'Old', 'Frail',], fill=False, common_norm=False, thresh=0.2, levels=3)

sns.scatterplot(data=pred_df[pred_df['Type']!='Prefrail'], x='scTRAIT_youngfrac', y='scTRAIT_oldfrac', hue='Type', lw=0,  s=32, hue_order=['Young', 'Old', 'Frail'],
                         palette=('#7fc97f', '#beaed4', '#fdc086'))



#ax.plot([15, 95], [15, 95], linestyle='--', color='black',  linewidth=1 )
#sns.lineplot(x=x, y=y, label='Old', lw=2.5, dashes=False, markersize=8, err_style='bars')
    # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
    #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)


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

ax.set_xlabel('Predicted young frac', fontsize=16, weight='bold', color='0.2')
ax.set_ylabel('Predicted Old frac', fontsize=16, weight='bold', color='0.2')

plt.xticks(fontsize=12, color='0.2', weight='bold')
plt.yticks(fontsize=12, color='0.2', weight='bold')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
           loc='best')
plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()

# ################### Plot patient info vs Predicted Age ###################
# for feature in ['Weakness', 'Weight loss', 'Exhaustion', 'Activity', 'Gait', 'Grip', 'Frailty score']:
#     for model in ['scTRAIT', 'VT', 'VM']:
#     #for model in ['scTRAIT', 'VT', 'VM']:
#
#         file_name = '%s vs predicted age_%s'%(feature, model)
#         font = {'family': 'arial',
#                             'weight': 'normal',
#                             'size': 16}
#         matplotlib.rc('font', **font)
#         matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
#         matplotlib.rcParams['lines.linewidth'] = 2
#
#         pred_df_ = pred_df[~pred_df[feature].isnull()].reset_index(drop=True)
#         fig, ax = plt.subplots(figsize=(4,4))
#         # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
#         #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
#         #for idx, key in enumerate(dataset):
#         sns.scatterplot(data=pred_df_, x=feature, y=model, hue='Type', lw=0,  s=32, hue_order=['Old', 'Frail', 'Prefrail'],
#                                  palette=('#beaed4', '#fdc086', '#888888'))
#         #ax.plot([15, 95], [15, 95], linestyle='--', color='black',  linewidth=1 )
#         #sns.lineplot(x=x, y=y, label='Old', lw=2.5, dashes=False, markersize=8, err_style='bars')
#             # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
#             #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)
#
#         #mae = np.mean([abs(i - j) for i, j in zip(pred_ages_[model], pred_ages_['Chronological Age'])])
#
#         # plt.text(0.2, 1.1, "MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
#         #          fontsize=16, fontdict={'weight': 'bold'}, color="black")
#
#         handles, labels = ax.get_legend_handles_labels()
#         # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['mt GCB'], 0.1*std_dataset['mt GCB'], color='#CC6677',
#         #            capsize=3, capthick=1, elinewidth=1.5)
#         # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['wt GCB'], 0.1*std_dataset['wt GCB'], color='#888888',
#         #            capsize=3, capthick=1, elinewidth=1.5)
#
#         for axis in ['bottom', 'left']:
#             ax.spines[axis].set_linewidth(2)
#             ax.spines[axis].set_color('0.2')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#
#         ax.tick_params(width=2, color='0.2')
#
#         ax.set_xlabel('%s'%feature, fontsize=16, weight='bold', color='0.2')
#         ax.set_ylabel('Predicted frailty score', fontsize=16, weight='bold', color='0.2')
#
#         # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#         #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
#         # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#         #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
#
#         plt.xticks(fontsize=12, color='0.2', weight='bold')
#         plt.yticks(fontsize=12, color='0.2', weight='bold')
#
#         plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
#                    loc='best')
#         plt.savefig(path + 'patient info vs pred age/%s.png' % (file_name), dpi=300,bbox_inches='tight')
#
#         if not os.path.isdir(path + 'patient info vs pred age/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#             os.makedirs(path + 'patient info vs pred age/svg/' )
#         plt.savefig(path + 'patient info vs pred age/svg/%s.svg' % (file_name), bbox_inches='tight')
#         plt.clf()
#         plt.close()
#
# ################### Linear correlation of patient info vs Old frac ###################
# for feature in ['Gait', 'Grip', 'Frailty score']:
#     for model in ['gmm_old_frac']:
#
#         file_name = 'regression %s vs predicted age_%s'%(feature, model)
#         font = {'family': 'arial',
#                             'weight': 'normal',
#                             'size': 16}
#         matplotlib.rc('font', **font)
#         matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
#         matplotlib.rcParams['lines.linewidth'] = 2
#
#         pred_df_ = pred_df[~pred_df[feature].isnull()].reset_index(drop=True)
#         fig, ax = plt.subplots(figsize=(4,4))
#
#         # sns.regplot(x=feature, y=model, data=pred_df_,
#         #             scatter_kws={"color": ('#beaed4', '#fdc086', '#888888'),  'hue':'Type', 'hue_order': ['Old', 'Frail', 'Prefrail'], "alpha": 0.7, 's': 20},
#         #             line_kws={"color": "black"}, ax=ax)
#
#         sns.regplot(x=feature, y=model, data=pred_df_, scatter_kws={"color": "black", "alpha": 0.7, 's': 20},
#                     line_kws={"color": "black"}, ax=ax)
#         r, p = scipy.stats.pearsonr(pred_df_[feature], pred_df_[model])
#         plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
#                  fontsize=12, fontdict={'weight': 'bold'}, color="black")
#         plt.text(0.8, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
#                  fontsize=12, fontdict={'weight': 'bold'}, color="black")
#
#         #mae = np.mean([abs(i - j) for i, j in zip(pred_ages_[model], pred_ages_['Chronological Age'])])
#
#         # plt.text(0.2, 1.1, "MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
#         #          fontsize=16, fontdict={'weight': 'bold'}, color="black")
#
#         handles, labels = ax.get_legend_handles_labels()
#         # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['mt GCB'], 0.1*std_dataset['mt GCB'], color='#CC6677',
#         #            capsize=3, capthick=1, elinewidth=1.5)
#         # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['wt GCB'], 0.1*std_dataset['wt GCB'], color='#888888',
#         #            capsize=3, capthick=1, elinewidth=1.5)
#
#         ax.spines["left"].set_visible(True)
#         ax.spines['left'].set_linewidth(1.5)
#         ax.spines['left'].set_color('0.2')
#
#         ax.spines["bottom"].set_visible(True)
#         ax.spines['bottom'].set_linewidth(1.5)
#         ax.spines['bottom'].set_color('0.2')
#
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.tick_params(width=1.5, color='0.2', labelsize=10)
#         from matplotlib.ticker import MaxNLocator
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#
#         ax.set_xlabel('%s' % feature, fontsize=10, weight='bold', color='0.2', labelpad=5)
#         ax.set_ylabel('Predicted frailty score', fontsize=10, weight='bold', color='0.2', labelpad=5)
#
#         plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
#                    loc='best')
#         plt.savefig(path + 'patient info vs pred age/%s.png' % (file_name), dpi=300,bbox_inches='tight')
#
#         if not os.path.isdir(path + 'patient info vs pred age/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#             os.makedirs(path + 'patient info vs pred age/svg/' )
#         plt.savefig(path + 'patient info vs pred age/svg/%s.svg' % (file_name), bbox_inches='tight')
#         plt.clf()
#         plt.close()


################### Plot patient info vs Old frac ###################
for feature in ['Weakness', 'Weight loss', 'Exhaustion', 'Activity', 'Gait', 'Grip', 'Frailty score', 'Chronological Age']:
    for model in ['scTRAIT_oldfrac', 'VT_oldfrac', 'VM_oldfrac']:

        file_name = '%s vs old frac_%s'%(feature, model)
        font = {'family': 'arial',
                            'weight': 'normal',
                            'size': 16}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 2

        pred_df_ = pred_df[~pred_df['Weakness'].isnull()].reset_index(drop=True)
        #pred_df_ = pred_df_[pred_df_['Chronological Age']!=55].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(4,4))
        # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
        #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
        #for idx, key in enumerate(dataset):
        if (feature == 'Gait') or (feature == 'Grip') or (feature == 'Frailty score') or (feature == 'Chronological Age'):
            sns.regplot(data=pred_df_, x=feature, y=model,  scatter=False, line_kws={"color": "black"}, ax=ax)
            r, p = scipy.stats.pearsonr(pred_df_[feature], pred_df_[model])
            if feature == 'Grip':
                plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
                plt.text(0.8, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
            else:
                plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
                plt.text(0.1, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")

        sns.scatterplot(data=pred_df_, x=feature, y=model, hue='Type', lw=0,  s=32, hue_order=['Old', 'Frail', 'Prefrail'],
                                 palette=('#beaed4', '#fdc086', '#888888'))


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
        ax.set_ylabel('Predicted frailty score', fontsize=16, weight='bold', color='0.2')

        # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
        # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

        plt.xticks(fontsize=12, color='0.2', weight='bold')
        plt.yticks(fontsize=12, color='0.2', weight='bold')

        # plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
        #            loc='best')
        ax.legend_.remove()
        plt.savefig(path + 'patient info vs pred old frac/%s.png' % (file_name), dpi=300,bbox_inches='tight')

        if not os.path.isdir(path + 'patient info vs pred old frac/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'patient info vs pred old frac/svg/' )
        plt.savefig(path + 'patient info vs pred old frac/svg/%s.svg' % (file_name), bbox_inches='tight')
        plt.clf()
        plt.close()

################### Plot Grip vs Gait ###################

feature1= 'Grip'
feature2= 'Chronological Age'
file_name = '%s vs %s Type'%(feature1, feature2)
font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

pred_df_ = pred_df[~pred_df['Grip'].isnull()].reset_index(drop=True)
#pred_df_ = pred_df_[pred_df_['Chronological Age']!=55].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(4,4))
# sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
#                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
#for idx, key in enumerate(dataset):
sns.regplot(data=pred_df_, x=feature1, y=feature2,  scatter=False, line_kws={"color": "black"}, ax=ax)
r, p = scipy.stats.pearsonr(pred_df_[feature1], pred_df_[feature2])
plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
         fontsize=12, fontdict={'weight': 'bold'}, color="black")
plt.text(0.8, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
         fontsize=12, fontdict={'weight': 'bold'}, color="black")


# sns.scatterplot(data=pred_df_, x='Grip', y='Gait', hue='Type', lw=0,  s=32, hue_order=['Old', 'Frail', 'Prefrail'],
#                          palette=('#beaed4', '#fdc086', '#888888'))

sns.scatterplot(data=pred_df_, x=feature1, y=feature2, hue='Type', lw=0,  s=32, hue_order=['Old', 'Frail', 'Prefrail'],
                         palette=('#beaed4', '#fdc086', '#888888'))


handles, labels = ax.get_legend_handles_labels()


for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_xlabel('%s'%feature1, fontsize=16, weight='bold', color='0.2')
ax.set_ylabel('%s'%feature2, fontsize=16, weight='bold', color='0.2')

# ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
# ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

plt.xticks(fontsize=12, color='0.2', weight='bold')
plt.yticks(fontsize=12, color='0.2', weight='bold')

# plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
#            loc='best')
ax.legend_.remove()
plt.savefig(path + 'patient info vs pred old frac/%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'patient info vs pred old frac/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'patient info vs pred old frac/svg/' )
plt.savefig(path + 'patient info vs pred old frac/svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()



################### FAMD(Factor Analysis on Mixed Data) on patient info ###################
from FAMD.famd import FAMD
famd = FAMD(n_components=2, n_iter=3, copy=True, check_input=True, engine='sklearn', random_state=0)
famd_data = pred_df_[['Weakness', 'Weight loss', 'Exhaustion', 'Activity', 'Gait', 'Grip']]


for col in ['Weakness', 'Weight loss', 'Exhaustion', 'Activity']:
    famd_data[col] = famd_data[col].astype('int').astype('object')

famd_pred_df = famd.fit_transform(famd_data)

pred_df_['FAMD0'] = famd_pred_df.iloc[:, 0]
pred_df_['FAMD1'] = famd_pred_df.iloc[:, 1]

print(famd.column_contributions_)

aa = famd.eigenvalues_summary.values[0, 1]



file_name = 'FAMD of patient info annotated'
font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(4,4))
#
sns.scatterplot(data=pred_df_, x='FAMD0', y='FAMD1', hue='Type', lw=0,  s=32, hue_order=['Old', 'Frail', 'Prefrail'],
                         palette=('#beaed4', '#fdc086', '#888888'))
# sns.scatterplot(data=pred_df_, x='FAMD0', y='FAMD1', hue='pred_type', lw=0,  s=32, hue_order=['Old', 'Frail'],
#                          palette=('#beaed4', '#fdc086'))

handles, labels = ax.get_legend_handles_labels()


for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_xlabel('FAMD0 (%s)'%famd.eigenvalues_summary.values[0, 1], fontsize=16, weight='bold', color='0.2')
ax.set_ylabel('FAMD1 (%s)'%famd.eigenvalues_summary.values[1, 1], fontsize=16, weight='bold', color='0.2')

# ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
# ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

texts = []

for i in range(pred_df_.shape[0]):
    texts.append(plt.text(x=pred_df_['FAMD0'].iloc[i], y=pred_df_['FAMD1'].iloc[i], s=pred_df_['Patient'].iloc[i], fontsize=8,
                 weight='bold', color='0.2'))
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='0.2'))

plt.xticks(fontsize=12, color='0.2', weight='bold')
plt.yticks(fontsize=12, color='0.2', weight='bold')

# plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
#            loc='best')
ax.legend_.remove()
plt.savefig(path + 'patient info vs pred old frac/%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'patient info vs pred old frac/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'patient info vs pred old frac/svg/' )
plt.savefig(path + 'patient info vs pred old frac/svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()



frail_old = pred_df_[(pred_df_['Type'] =='Frail')&(pred_df_['pred_type'] =='Old')].reset_index(drop=True)
frail_frail = pred_df_[(pred_df_['Type'] =='Frail')&(pred_df_['pred_type'] =='Frail')].reset_index(drop=True)

for feature in ['Gait', 'Grip', 'Frailty score', 'FAMD0', 'FAMD1']:
    dict_datasets = {'Predicted as Frail': np.array(frail_frail[feature]), 'Predicted as Old': np.array(frail_old[feature])}
    # dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
    draw_custom_bar_plot(dict_datasets, path+'patient info vs pred old frac/', file_name='pred_old vs pred_frail %s'%feature, colors=('#fdc086', '#beaed4'),
                         vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))

################### Plot patient info vs Old frac ###################
for feature in ['FAMD0', 'FAMD1', ]:
    for model in ['scTRAIT_oldfrac', 'VT_oldfrac', 'VM_oldfrac', 'gmm_old_frac', 'scTRAIT_diff', 'Chronological Age']:

        file_name = '%s vs old frac_%s'%(feature, model)
        font = {'family': 'arial',
                            'weight': 'normal',
                            'size': 16}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 2

        #pred_df_ = pred_df_[pred_df_['Chronological Age']!=55].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(4,4))
        # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
        #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
        #for idx, key in enumerate(dataset):
        sns.regplot(data=pred_df_, x=feature, y=model,  scatter=False, line_kws={"color": "black"}, ax=ax)
        r, p = scipy.stats.pearsonr(pred_df_[feature], pred_df_[model])
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

        sns.scatterplot(data=pred_df_, x=feature, y=model, hue='Type', lw=0,  s=32, hue_order=['Old', 'Frail', 'Prefrail'],
                                 palette=('#beaed4', '#fdc086', '#888888'))


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
        plt.savefig(path + 'patient info vs pred old frac/%s.png' % (file_name), dpi=300,bbox_inches='tight')

        if not os.path.isdir(path + 'patient info vs pred old frac/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'patient info vs pred old frac/svg/' )
        plt.savefig(path + 'patient info vs pred old frac/svg/%s.svg' % (file_name), bbox_inches='tight')
        plt.clf()
        plt.close()


################### Plot patient info vs Old frac ###################
for feature in ['FAMD0', 'FAMD1', ]:
    for model in ['scTRAIT_oldfrac', 'Chronological Age', 'gmm_old_frac']:

        file_name = 'only frail %s vs old frac_%s'%(feature, model)
        font = {'family': 'arial',
                            'weight': 'normal',
                            'size': 16}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 2

        pred_df__ = pred_df_[pred_df_['Type']=='Frail'].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(4,4))
        # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
        #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
        #for idx, key in enumerate(dataset):
        sns.regplot(data=pred_df__, x=feature, y=model,  scatter=False, line_kws={"color": "black"}, ax=ax)
        r, p = scipy.stats.pearsonr(pred_df__[feature], pred_df__[model])
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

        sns.scatterplot(data=pred_df__, x=feature, y=model, hue='Type', lw=0,  s=32, hue_order=['Frail'],
                                 palette=('#fdc086',  ))


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
        plt.savefig(path + 'patient info vs pred old frac/%s.png' % (file_name), dpi=300,bbox_inches='tight')

        if not os.path.isdir(path + 'patient info vs pred old frac/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'patient info vs pred old frac/svg/' )
        plt.savefig(path + 'patient info vs pred old frac/svg/%s.svg' % (file_name), bbox_inches='tight')
        plt.clf()
        plt.close()


################### Plot patient info vs Old frac ###################
for feature in ['FAMD0', 'FAMD1', 'Gait', 'Grip', 'Chronological Age']:
    for model in ['Frailty score']:

        file_name = 'frailty score vs %s'%(feature)
        font = {'family': 'arial',
                            'weight': 'normal',
                            'size': 16}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 2

        #pred_df_ = pred_df_[pred_df_['Chronological Age']!=55].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(4,4))
        # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
        #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
        #for idx, key in enumerate(dataset):
        sns.regplot(data=pred_df_, x=feature, y=model,  scatter=False, line_kws={"color": "black"}, ax=ax)
        r, p = scipy.stats.pearsonr(pred_df_[feature], pred_df_[model])
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

        sns.scatterplot(data=pred_df_, x=feature, y=model, hue='Type', lw=0,  s=32, hue_order=['Old', 'Frail', 'Prefrail'],
                                 palette=('#beaed4', '#fdc086', '#888888'))


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
        ax.set_ylabel('Clinical frailty score', fontsize=16, weight='bold', color='0.2')

        # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
        # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

        plt.xticks(fontsize=12, color='0.2', weight='bold')
        plt.yticks(fontsize=12, color='0.2', weight='bold')

        # plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
        #            loc='best')
        ax.legend_.remove()
        plt.savefig(path + 'patient info vs pred old frac/%s.png' % (file_name), dpi=300,bbox_inches='tight')

        if not os.path.isdir(path + 'patient info vs pred old frac/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'patient info vs pred old frac/svg/' )
        plt.savefig(path + 'patient info vs pred old frac/svg/%s.svg' % (file_name), bbox_inches='tight')
        plt.clf()
        plt.close()

################### Plot Chronological age vs Predicted Age CoV ###################
for model in ['scTRAIT_sd', 'VT_sd', 'VM_sd']:

    file_name = 'SD Age prediction_%s'%model
    font = {'family': 'arial',
                        'weight': 'normal',
                        'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(4,4))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    #for idx, key in enumerate(dataset):
    sns.scatterplot(data=pred_ages, x='Chronological Age', y=model, hue='Type', lw=0,  s=32, hue_order=['Young', 'Old', 'Frail', 'Prefrail'],
                             palette=('#7fc97f', '#beaed4', '#fdc086', '#888888'))
    #ax.plot([15, 95], [15, 95], linestyle='--', color='black',  linewidth=1 )
    #sns.lineplot(x=x, y=y, label='Old', lw=2.5, dashes=False, markersize=8, err_style='bars')
        # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
        #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

    # mae = np.mean([abs(i - j) for i, j in zip(pred_ages[model], pred_ages['Chronological Age'])])

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

    ax.set_xlabel('Chronological Age (yr)', fontsize=16, weight='bold', color='0.2')
    ax.set_ylabel('Predicted Age Heterogeneity', fontsize=16, weight='bold', color='0.2')

    custom_range = (15, 95)
    #custom_range = (70, 95)
    label_stepsize = 10
    # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
    # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

    plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
               weight='bold')
    plt.yticks(fontsize=12, color='0.2', weight='bold')

    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
               loc='best')
    plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/' )
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
    plt.clf()
    plt.close()

################### Young old separate plot Chronological age vs Old frac ###################
for model in ['scTRAIT', 'VT', 'VM']:


    for typ in ['Young', 'Old']:

        font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 2
        fig, ax = plt.subplots(figsize=(4, 4))

        if typ == 'Young':
            pred_df_part = pred_df[pred_df['Type'] == 'Young'].reset_index(drop=True)
            age_range = [15, 50]
            sns.scatterplot(data=pred_df_part, x='Chronological Age', y='%s_oldfrac'%model, hue='Type', lw=2.5, s=32,
                            palette=('#7fc97f',))
        elif typ == 'Old':
            pred_df_part = pred_df[pred_df['Type'] != 'Young'].reset_index(drop=True)
            age_range = [65, 95]
            sns.scatterplot(data=pred_df_part, x='Chronological Age', y='%s_oldfrac'%model, hue='Type', lw=2.5, s=32,
                            hue_order=['Old', 'Frail', 'Prefrail'],
                            palette=('#beaed4', '#fdc086', '#888888'))
        file_name = 'Oldfrac_%s_%s' % (typ,model)

        #ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )

        mae = np.mean([abs(i - j) for i, j in zip(pred_df_part[model], pred_df_part['Chronological Age'])])

        # plt.text(0.2, 1.1, "MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
        #          fontsize=16, fontdict={'weight': 'bold'}, color="black")

        handles, labels = ax.get_legend_handles_labels()


        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_color('0.2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(width=2, color='0.2')

        ax.set_xlabel('Chronological Age (yr)', fontsize=16, weight='bold', color='0.2')
        ax.set_ylabel('Old fraction (%)', fontsize=16, weight='bold', color='0.2')

        #custom_range = (70, 95)
        label_stepsize = 5
        # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
        # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

        plt.xticks(np.arange(age_range[0], age_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
                   weight='bold')
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

################### Linear correlation of chronological age vs Old frac ###################

model='scTRAIT'
file_name = 'Regression Oldfrac vs age_%s' % (model)

font = {'family': 'arial',
        'weight': 'normal',
        'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(4, 4))

pred_df_frail = pred_df[pred_df['Type'] == 'Frail'].reset_index(drop=True)
sns.regplot(data=pred_df_frail, x='Chronological Age', y='%s_oldfrac'%model, scatter=False,
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)
sns.scatterplot(data=pred_df_frail, x='Chronological Age', y='%s_oldfrac'%model, hue='Type',
                lw=2.5, s=32, palette=('#fdc086',), ax=ax)

pred_df_old = pred_df[(pred_df['Type'] == 'Old')&(pred_df['Chronological Age'] != 55)].reset_index(drop=True)
sns.regplot(data=pred_df_old, x='Chronological Age', y='%s_oldfrac'%model, scatter=False,
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)
sns.scatterplot(data=pred_df_old, x='Chronological Age', y='%s_oldfrac'%model, hue='Type',
                lw=2.5, s=32, palette=('#beaed4',), ax=ax)

r, p = scipy.stats.pearsonr(pred_df_frail['Chronological Age'], pred_df_frail['%s_oldfrac'%model])
plt.text(0.1, 0.88, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="#fdc086")

r, p = scipy.stats.pearsonr(pred_df_old['Chronological Age'], pred_df_old['%s_oldfrac'%model])
plt.text(0.1, 0.3, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="#beaed4")

handles, labels = ax.get_legend_handles_labels()


for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_xlabel('Chronological Age (yr)', fontsize=16, weight='bold', color='0.2')
ax.set_ylabel('Old fraction (%)', fontsize=16, weight='bold', color='0.2')

#custom_range = (70, 95)
label_stepsize = 5

plt.xticks(fontsize=12, color='0.2', weight='bold')
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


################### Young old separate plot Chronological age vs Predicted Age ###################
for model in ['scTRAIT', 'VT', 'VM']:


    for typ in ['Young', 'Old']:

        font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 2
        fig, ax = plt.subplots(figsize=(4, 4))

        if typ == 'Young':
            pred_df_part = pred_df[pred_df['Type'] == 'Young'].reset_index(drop=True)
            age_range = [15, 50]
            sns.scatterplot(data=pred_df_part, x='Chronological Age', y=model, hue='Type', lw=2.5, s=32,
                            palette=('#7fc97f',))
        elif typ == 'Old':
            pred_df_part = pred_df[pred_df['Type'] != 'Young'].reset_index(drop=True)
            age_range = [65, 95]
            sns.scatterplot(data=pred_df_part, x='Chronological Age', y=model, hue='Type', lw=2.5, s=32,
                            hue_order=['Old', 'Frail', 'Prefrail'],
                            palette=('#beaed4', '#fdc086', '#888888'))
        file_name = 'Separate Age prediction_%s_%s' % (typ,model)

        #ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )

        mae = np.mean([abs(i - j) for i, j in zip(pred_df_part[model], pred_df_part['Chronological Age'])])

        # plt.text(0.2, 1.1, "MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
        #          fontsize=16, fontdict={'weight': 'bold'}, color="black")

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

        plt.xticks(np.arange(age_range[0], age_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
                   weight='bold')
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


################### Young old separate plot Chronological age vs Predicted Age CoV ###################
for model in ['scTRAIT_sd', 'VT_sd', 'VM_sd']:


    for typ in ['Young', 'Old']:

        font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 2
        fig, ax = plt.subplots(figsize=(4, 4))

        if typ == 'Young':
            pred_df_part = pred_df[pred_df['Type'] == 'Young'].reset_index(drop=True)
            age_range = [15, 50]
            sns.scatterplot(data=pred_df_part, x='Chronological Age', y=model, hue='Type', lw=2.5, s=32,
                            palette=('#7fc97f',))
        elif typ == 'Old':
            pred_df_part = pred_df[pred_df['Type'] != 'Young'].reset_index(drop=True)
            age_range = [65, 95]
            sns.scatterplot(data=pred_df_part, x='Chronological Age', y=model, hue='Type', lw=2.5, s=32,
                            hue_order=['Old', 'Frail'],
                            palette=('#beaed4', '#fdc086',))
        file_name = 'Separate SD Age prediction_%s_%s' % (typ,model)

        #ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )

        mae = np.mean([abs(i - j) for i, j in zip(pred_df_part[model], pred_df_part['Chronological Age'])])

        # plt.text(0.2, 1.1, "MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
        #          fontsize=16, fontdict={'weight': 'bold'}, color="black")

        handles, labels = ax.get_legend_handles_labels()


        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_color('0.2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(width=2, color='0.2')

        ax.set_xlabel('Chronological Age (yr)', fontsize=16, weight='bold', color='0.2')
        ax.set_ylabel('Predicted Age Heterogeneity', fontsize=16, weight='bold', color='0.2')

        #custom_range = (70, 95)
        label_stepsize = 5
        # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
        # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

        plt.xticks(np.arange(age_range[0], age_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
                   weight='bold')
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


################### Young old separate plot Chronological age vs Predicted Age CoV ###################
for model in ['scTRAIT', 'VT', 'VM']:


    for typ in ['Young', 'Old']:

        font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 2
        fig, ax = plt.subplots(figsize=(4, 4))

        if typ == 'Young':
            pred_df_part = pred_df[pred_df['Type'] == 'Young'].reset_index(drop=True)
            #age_range = [15, 50]
            sns.scatterplot(data=pred_df_part, x=model, y=model+'_sd', hue='Type', lw=2.5, s=32,
                            palette=('#7fc97f',))
        elif typ == 'Old':
            pred_df_part = pred_df[pred_df['Type'] != 'Young'].reset_index(drop=True)
            #age_range = [65, 95]
            sns.scatterplot(data=pred_df_part, x=model, y=model+'_sd', hue='Type', lw=2.5, s=32,
                            hue_order=['Old', 'Frail'],
                            palette=('#beaed4', '#fdc086',))
        file_name = 'Pred Age vs SD Age_%s_%s' % (typ,model)

        #ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )

        mae = np.mean([abs(i - j) for i, j in zip(pred_df_part[model], pred_df_part['Chronological Age'])])

        # plt.text(0.2, 1.1, "MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
        #          fontsize=16, fontdict={'weight': 'bold'}, color="black")

        handles, labels = ax.get_legend_handles_labels()


        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_color('0.2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(width=2, color='0.2')

        ax.set_xlabel('Predicted Age', fontsize=16, weight='bold', color='0.2')
        ax.set_ylabel('Predicted Age Heterogeneity', fontsize=16, weight='bold', color='0.2')

        #custom_range = (70, 95)
        label_stepsize = 5
        # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
        # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
        #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

        plt.xticks(fontsize=12, color='0.2',
                   weight='bold')
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


################### Plot Chronological age vs Age Differential ###################
pred_df_nonyoung = pred_df[pred_df['Type']!='Young'].reset_index(drop=True)

for model in ['scTRAIT', 'VT', 'VM']:
    file_name = 'Differential Age with 80%s'%model
    font = {'family': 'arial',
                        'weight': 'normal',
                        'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    #pred_df['delta'] = pred_df[model] - pred_df['Chronological Age']
    pred_df_nonyoung['delta'] = pred_df_nonyoung[model] - 80

    fig, ax = plt.subplots(figsize=(4,4))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    #for idx, key in enumerate(dataset):
    sns.scatterplot(data=pred_df_nonyoung, x='Chronological Age', y='delta', hue='Type', lw=2.5,  s=32, hue_order=['Old', 'Frail', 'Prefrail'],
                             palette=('#beaed4', '#fdc086', '#888888'))
    ax.axhline(0, linestyle='--', linewidth=1, color='0.2')

    #sns.lineplot(x=x, y=y, label='Old', lw=2.5, dashes=False, markersize=8, err_style='bars')
        # ax.errorbar(np.arange(custom_range[0], custom_range[1] + stepsize, stepsize), mean_dataset[key], 0.1 * std_dataset[key],
        #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

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

    ax.set_xlabel('Chronological Age (yr)', fontsize=16, weight='bold', color='0.2')
    ax.set_ylabel('Age Differential (yr)', fontsize=16, weight='bold', color='0.2')

    custom_range = (65, 95)
    #custom_range = (70, 95)
    label_stepsize = 5
    # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
    # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

    plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
               weight='bold')
    plt.yticks(fontsize=12, color='0.2', weight='bold')

    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
               loc='best')
    plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/' )
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
    plt.clf()
    plt.close()


model = 'scTRAIT'
pred_df['delta'] = pred_df[model] - 80

young = pred_df[pred_df['Type'] =='Young'].reset_index(drop=True)
old = pred_df[pred_df['Type'] =='Old'].reset_index(drop=True)
frail = pred_df[pred_df['Type'] == 'Frail'].reset_index(drop=True)

dict_datasets = {'Old': np.array(old['delta']), 'Frail': np.array(frail['delta'])}
# dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='Differential age bar graph %s'%model, colors=('#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))

# dict_datasets = {'Young': np.array(young['scTRAIT']), 'Old': np.array(old['scTRAIT']), 'Frail': np.array(frail['scTRAIT'])}
# # dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
# draw_custom_bar_plot(dict_datasets, path, file_name='Predicted age bar graph %s' % model,
#                      colors=('#7fc97f', '#beaed4', '#fdc086'),
#                      vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))

dict_datasets = {'Old': np.array(old['scTRAIT']), 'Frail': np.array(frail['scTRAIT'])}
# dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='Predicted age bar graph %s' % model,
                     colors=('#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))

dict_datasets = {'Old': np.array(old['scTRAIT_sd']), 'Frail': np.array(frail['scTRAIT_sd'])}
# dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='Predicted SD age bar graph %s' % model,
                     colors=('#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))


dict_datasets = {'Young': np.array(young['scTRAIT']), 'Old': np.array(old['scTRAIT']), 'Frail': np.array(frail['scTRAIT'])}
# dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='All Predicted age bar graph %s' % model,
                     colors=('#7fc97f', '#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))

dict_datasets = {'Young': np.array(young['scTRAIT_oldfrac']), 'Old': np.array(old['scTRAIT_oldfrac']), 'Frail': np.array(frail['scTRAIT_oldfrac'])}
# dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='All Old fraction bar graph %s' % model,
                     colors=('#7fc97f', '#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))

dict_datasets = {'Old': np.array(old['scTRAIT_oldfrac']), 'Frail': np.array(frail['scTRAIT_oldfrac'])}
# dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name= 'Nonyoung Old fraction bar graph %s' % model,
                     colors=('#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))


dict_datasets = {'Young': np.array(young['scTRAIT_youngfrac']), 'Old': np.array(old['scTRAIT_youngfrac']), 'Frail': np.array(frail['scTRAIT_youngfrac'])}
# dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='All Young fraction bar graph %s' % model,
                     colors=('#7fc97f', '#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))


old_part = old[old['scTRAIT']>=60].reset_index(drop=True)
frail_part = frail[frail['scTRAIT']>=60].reset_index(drop=True)
dict_datasets = {'Old': np.array(old_part['scTRAIT_sd']), 'Frail': np.array(frail_part['scTRAIT_sd'])}
# dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='Predicted SD age bar graph only for high biological age %s' % model,
                     colors=('#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))

old_predicted_as_old = np.sum(old['scTRAIT']>=60) / old['scTRAIT'].size
frail_predicted_as_old = np.sum(frail['scTRAIT'] >= 60) / frail['scTRAIT'].size
dict_datasets = {'Old': np.array(old_predicted_as_old), 'Frail': np.array(frail_predicted_as_old)}
draw_custom_bar_plot(dict_datasets, path, file_name='Percentage predicted as old bar graph %s' % model,
                     colors=('#beaed4', '#fdc086'),
                     vmax=1, strip_plot=False, test='mann-whitney', pvalue=True, figsize=(1, 2))


old_part = old[old['scTRAIT']>=60].reset_index(drop=True)
frail_part = frail[frail['scTRAIT']>=60].reset_index(drop=True)
dict_datasets = {'Old': np.array(old_part['scTRAIT_youngfrac']), 'Frail': np.array(frail_part['scTRAIT_youngfrac'])}
draw_custom_bar_plot(dict_datasets, path, file_name='Young fraction %s' % model,
                     colors=('#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))



old_part = old[old['scTRAIT']>=60].reset_index(drop=True)
frail_part = frail[frail['scTRAIT']>=60].reset_index(drop=True)
dict_datasets = {'Old': np.array(old_part['scTRAIT_oldfrac']), 'Frail': np.array(frail_part['scTRAIT_oldfrac'])}
draw_custom_bar_plot(dict_datasets, path, file_name='Old fraction %s' % model,
                     colors=('#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))



young_part = young[young['scTRAIT']>=60].reset_index(drop=True)
old_part = old[old['scTRAIT']>=60].reset_index(drop=True)
frail_part = frail[frail['scTRAIT']>=60].reset_index(drop=True)
dict_datasets = {'Young': np.array(young_part['scTRAIT_youngfrac']), 'Old': np.array(old_part['scTRAIT_youngfrac']),
                 'Frail': np.array(frail_part['scTRAIT_youngfrac'])}
draw_custom_bar_plot(dict_datasets, path, file_name='Old predicted Young fraction %s' % model,
                     colors=('#7fc97f', '#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))


young_part = young[young['scTRAIT']<60].reset_index(drop=True)
old_part = old[old['scTRAIT']<60].reset_index(drop=True)
frail_part = frail[frail['scTRAIT']<60].reset_index(drop=True)
dict_datasets = {'Young': np.array(young_part['scTRAIT_youngfrac']), 'Old': np.array(old_part['scTRAIT_youngfrac']),
                 'Frail': np.array(frail_part['scTRAIT_youngfrac'])}
draw_custom_bar_plot(dict_datasets, path, file_name='Young predicted Young fraction %s' % model,
                     colors=('#7fc97f', '#beaed4', '#fdc086'),
                     vmax=None, strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1, 2))

################### Plot Predicted age vs Protein Secretion ###################
pred_df = pd.DataFrame(pred_ages)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\\'
df_secretion = pd.read_excel(path+'Ageing samples--Walston donor registry.xlsx', sheet_name='secretion data _1', skiprows=1)[:16]
df_secretion = df_secretion.rename(columns={'Donor': 'Patient'})

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\age regression\secretion\\'

merged_df = pd.merge(pred_df, df_secretion, on='Patient', how='inner')
# how='inner' will only include patients present in both DataFrames.
# how='left' will include all patients from the left DataFrame.
# how='right' will include all patients from the right DataFrame.
# how='outer' will include all patients from both DataFrames.
features = merged_df.columns[7:-2]
for feature in features:
    fig, ax = plt.subplots(figsize=(4,4))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    #for idx, key in enumerate(dataset):
    sns.scatterplot(data=merged_df, x='scTRAIT', y=feature, hue='Type', lw=2.5,  s=32, hue_order=['Old', 'Young'],
                             palette=('#beaed4', '#fdc086'))

    handles, labels = ax.get_legend_handles_labels()

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')

    ax.set_xlabel('Predicted Age (yr)', fontsize=16, weight='bold', color='0.2')
    ax.set_ylabel('%s Secretion'%feature, fontsize=16, weight='bold', color='0.2')

    #custom_range = (70, 95)
    # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
    # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

    # plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
    #            weight='bold')
    plt.xticks(fontsize=12, color='0.2', weight='bold')
    plt.yticks(fontsize=12, color='0.2', weight='bold')

    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
               loc='best')
    plt.savefig(path + '%s.png' % (feature), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/' )
    plt.savefig(path + 'svg/%s.svg' % (feature), bbox_inches='tight')
    plt.clf()
    plt.close()


################### Plot Predicted age vs Other markers ###################
pred_df = pd.DataFrame(pred_ages)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\\'
df_other = pd.read_excel(path+'Ageing samples--Walston donor registry.xlsx', sheet_name='Sheet1', skiprows=2).iloc[:, 1:].drop(['UNITS/description', 'Normal range'], axis=1)
df_other.set_index('Unnamed: 1', inplace=True)
df_other = df_other.T
df_other.reset_index(inplace=True)
df_other.rename(columns={'index': 'Patient'}, inplace=True)

df_other = df_other.dropna(axis=1)


#path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\age regression\blood counts\\'
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\age regression\blood counts vs chronological age\\'

merged_df = pd.merge(pred_df, df_other, on='Patient', how='inner')
# how='inner' will only include patients present in both DataFrames.
# how='left' will include all patients from the left DataFrame.
# how='right' will include all patients from the right DataFrame.
# how='outer' will include all patients from both DataFrames.
features = merged_df.columns[4:-10]
for feature in features:
    fig, ax = plt.subplots(figsize=(4,4))

    # sns.scatterplot(data=merged_df, x='scTRAIT', y=feature, hue='Type', lw=2.5,  s=32, hue_order=['Old', 'Young'],
    #                          palette=('#beaed4', '#fdc086'))
    sns.scatterplot(data=merged_df, x='Chronological Age', y=feature, hue='Type', lw=2.5, s=32, hue_order=['Old', 'Young'],
                    palette=('#beaed4', '#fdc086'))

    handles, labels = ax.get_legend_handles_labels()

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')

    #ax.set_xlabel('Predicted Age (yr)', fontsize=16, weight='bold', color='0.2')
    ax.set_xlabel('Chronological Age (yr)', fontsize=16, weight='bold', color='0.2')
    ax.set_ylabel('%s'%feature, fontsize=16, weight='bold', color='0.2')

    #custom_range = (70, 95)
    # ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')
    # ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
    #                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='bold')

    # plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
    #            weight='bold')
    plt.xticks(fontsize=12, color='0.2', weight='bold')
    plt.yticks(fontsize=12, color='0.2', weight='bold')

    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
               loc='best')
    plt.savefig(path + '%s.png' % (feature), dpi=300,bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/' )
    plt.savefig(path + 'svg/%s.svg' % (feature), bbox_inches='tight')
    plt.clf()
    plt.close()