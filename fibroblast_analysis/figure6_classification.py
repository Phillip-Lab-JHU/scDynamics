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

"""Generates Data for Figure 6 age and frailty score (classification)."""
import pandas as pd
from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast
from dnn.autoencoders import set_duration_for_autoencoder
import tensorflow as tf
from dnn.classification import scTRAIT, Temporal_Conv1D_2D_classifier

#################################### Classification 3 class: Model evaluation with new batch (leave-one-out) ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\leave one out 3 class\\'

duration=30
for patient in tqdm(np.unique(df['Patient'])):
    patient = np.unique(df['Patient'])[0]
    print('Patient %s starting'%patient)

    df_part = df[df['Patient']!=patient].reset_index(drop=True)
    df_duration_part = df_duration[df_duration['Patient']!=patient].reset_index(drop=True)

    df_part_test = df[df['Patient']==patient].reset_index(drop=True)
    df_duration_part_test = df_duration[df_duration['Patient']==patient].reset_index(drop=True)

    ############# training set #############
    traj_scale_data = df_duration_part.loc[:, ['reg_x', 'reg_y']]
    traj_scaler = StandardScaler()
    traj_scale_data = pd.DataFrame(traj_scaler.fit_transform(traj_scale_data), columns=['reg_x', 'reg_y'])

    traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration, feature_name=['reg_x', 'reg_y'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
    rotated_trajectories = dict_to_array(trajectories)
    X_traj_train = rotated_trajectories

    local_features = ['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance']
    local_scale_data = df_duration_part.loc[:, local_features]
    local_scaler = StandardScaler()
    local_scale_data = pd.DataFrame(local_scaler.fit_transform(local_scale_data), columns=local_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(local_scale_data, duration=duration, feature_name=local_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
    rotated_trajectories = dict_to_array(trajectories)
    X_local_train = rotated_trajectories

    morphology_features = ['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length', 'minor_axis_length',
                           'aspect_ratio', 'circularity']
    scale_data = df_duration_part.loc[:, morphology_features]
    scaler = StandardScaler()
    scale_data = pd.DataFrame(scaler.fit_transform(scale_data), columns=morphology_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration, feature_name=morphology_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
    rotated_trajectories = dict_to_array(trajectories)
    X_morph_train = rotated_trajectories

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_covar_train = le.fit_transform( np.array(df_part['Condition']) )
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    # from sklearn.preprocessing import OneHotEncoder
    # ohe = OneHotEncoder()
    # X_covar = ohe.fit_transform( np.array(df_nonyoung['Condition']).reshape(-1,1) )

    y_names = df_part['Type']
    print('Classes:', pd.unique(y_names))
    y_train = y_names.replace({'Frail': 0, 'Old':1, 'Young':2}, inplace=False)
    #y = y_names.replace(list(pd.unique(y_names)), [i for i in range(pd.unique(y_names).shape[0])])
    y_train = np.array(y_train)
    print('Number of classes:', np.unique(y_train).size)

    from sklearn.utils import shuffle
    X_traj_train, X_local_train, X_morph_train, X_covar_train, y_train = shuffle(X_traj_train, X_local_train,
                                                                                 X_morph_train, X_covar_train, y_train, random_state=0)

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
    X_covar_test = le.transform( np.array(df_part_test['Condition']) )
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    # from sklearn.preprocessing import OneHotEncoder
    # ohe = OneHotEncoder()
    # X_covar = ohe.fit_transform( np.array(df_nonyoung['Condition']).reshape(-1,1) )

    y_names_test = df_part_test['Type']
    print('Classes:', pd.unique(y_names_test))
    y_test = y_names_test.replace({'Frail': 0, 'Old': 1, 'Young':2}, inplace=False)
    #y = y_names_test.replace(list(pd.unique(y_names_test)), [i for i in range(pd.unique(y_names_test).shape[0])])
    y_test = np.array(y_test)
    print('Number of classes:', np.unique(y_test).size)

    #################################### Model training ####################################

    from tensorflow.keras import callbacks
    stop_early = callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001)
    # checkpointer = callbacks.ModelCheckpoint('saved_model/monocyte_project_v2/%s'%model_name, monitor='accuracy', verbose=1,
    #                 save_weights_only=False, save_freq='epoch', every_n_epochs=10, mode='auto', save_best_only=True)  # Save model every epoch

    from dnn.classification import scTRAIT
    sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=np.unique(y_train).size)
    result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=500,
                         verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
    sctrait.save('saved_model/monocyte_project_v2/leave one out 3 class scTRAIT %s'%patient, save_format='tf')
    # ### model2: Vanilla trajectory ###
    # vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=np.unique(y_train).size)
    # result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=500, verbose=1, validation_split=0.1, shuffle=True,
    #                      callbacks=[stop_early])
    # vt.save('saved_model/monocyte_project_v2/leave one out 3 class vanilla traj %s'%patient, save_format='tf')
    #
    # ### model3: Vanilla morphology ###
    # vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=np.unique(y_train).size)
    # result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=500, verbose=1, validation_split=0.1, shuffle=True,
    #                      callbacks=[stop_early])
    # vm.save('saved_model/monocyte_project_v2/leave one out 3 class vanilla morpho %s'%patient, save_format='tf')

    y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
    # y_pred2 = vt.predict([X_traj_test])
    # y_pred3 = vm.predict([X_morph_test])

    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

    if y_pred.shape[1]>=3:
        y_class = np.argmax(y_pred, axis=1)
        # y_class2 = np.argmax(y_pred2, axis=1)
        # y_class3 = np.argmax(y_pred3, axis=1)

    else:
        y_class = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred)])
        # y_class2 = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred2)])
        # y_class3 = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred3)])

    #f1 = f1_score(y_test, y_class)
    # f1_2 = f1_score(y_test, y_class2)
    # f1_3 = f1_score(y_test, y_class3)

    acs = accuracy_score(y_test, y_class)
    # acs2 = accuracy_score(y_test, y_class2)
    # acs3 = accuracy_score(y_test, y_class3)

    #ps = precision_score(y_test, y_class)
    # ps2 = precision_score(y_test, y_class2)
    # ps3 = precision_score(y_test, y_class3)

    #rs = recall_score(y_test, y_class)
    # rs2 = recall_score(y_test, y_class2)
    # rs3 = recall_score(y_test, y_class3)

    np.save(path + 'scTRAIT %s result.npy' %(patient), y_pred)
    # np.save(path + 'VT %s result.npy' % patient, y_pred2)
    # np.save(path + 'VM %s result.npy' % patient, y_pred3)

    np.savetxt(path+'%s result.txt'%patient, ['Accuracy: ', acs], fmt='%s', delimiter ='')

    # np.savetxt(path+'%s result.txt'%patient, ['Accuracy: ', acs, acs2, acs3,
    #                                           'F1 score: ', f1, f1_2, f1_3,
    #                                           'Precision: ', ps, ps2, ps3,
    #                                           'Recall: ', rs, rs2, rs3], fmt='%s', delimiter ='')



#################################### Classification young vs old: Model evaluation with new batch (leave-one-out) ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df['Type2'] = 'Young'
df_duration['Type2'] = 'Young'
df.loc[df['Type'] != 'Young', 'Type2'] = 'Old'
df_duration.loc[df_duration['Type'] != 'Young', 'Type2'] = 'Old'

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\leave one out old vs young\\'

duration=30
for patient in tqdm(np.unique(df['Patient'])):
    print('Patient %s starting'%patient)

    df_part = df[df['Patient'] != patient].reset_index(drop=True)
    df_duration_part = df_duration[df_duration['Patient'] != patient].reset_index(drop=True)

    df_part_test = df[df['Patient'] == patient].reset_index(drop=True)
    df_duration_part_test = df_duration[df_duration['Patient'] == patient].reset_index(drop=True)

    ############# training set #############

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

    y_names = df_part['Type2']
    print('Classes:', pd.unique(y_names))
    y_train = y_names.replace({'Old': 0, 'Young': 1}, inplace=False)
    # y = y_names.replace(list(pd.unique(y_names)), [i for i in range(pd.unique(y_names).shape[0])])
    y_train = np.array(y_train)
    print('Number of classes:', np.unique(y_train).size)

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


    y_names_test = df_part_test['Type2']
    print('Classes:', pd.unique(y_names_test))
    y_test = y_names_test.replace({'Old': 0, 'Young': 1}, inplace=False)
    # y = y_names_test.replace(list(pd.unique(y_names_test)), [i for i in range(pd.unique(y_names_test).shape[0])])
    y_test = np.array(y_test)
    print('Number of classes:', np.unique(y_test).size)


    #################################### Model training ####################################

    from tensorflow.keras import callbacks
    stop_early = callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001)
    # checkpointer = callbacks.ModelCheckpoint('saved_model/monocyte_project_v2/%s'%model_name, monitor='accuracy', verbose=1,
    #                 save_weights_only=False, save_freq='epoch', every_n_epochs=10, mode='auto', save_best_only=True)  # Save model every epoch

    from dnn.classification import scTRAIT, scTRAIT_old
    sctrait = scTRAIT_old(duration=new_duration, embed_dim=64, n_classes=np.unique(y_train).size)
    result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=500,
                         verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
    sctrait.save('saved_model/monocyte_project_v2/leave one out young vs old scTRAIT %s'%patient, save_format='tf')

    ### model2: Vanilla trajectory ###
    # vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=np.unique(y_train).size)
    # result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=500, verbose=1, validation_split=0.1, shuffle=True,
    #                      callbacks=[stop_early])
    # vt.save('saved_model/monocyte_project_v2/leave one out young vs old vanilla traj %s'%patient, save_format='tf')
    vt = tf.keras.models.load_model('saved_model/monocyte_project_v2/leave one out young vs old vanilla traj %s'%patient, compile=True)

    ### model3: Vanilla morphology ###
    # vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=np.unique(y_train).size)
    # result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=500, verbose=1, validation_split=0.1, shuffle=True,
    #                      callbacks=[stop_early])
    # vm.save('saved_model/monocyte_project_v2/leave one out young vs old vanilla morpho %s'%patient, save_format='tf')
    vm = tf.keras.models.load_model('saved_model/monocyte_project_v2/leave one out young vs old vanilla morpho %s'%patient,compile=True)

    y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
    y_pred2 = vt.predict([X_traj_test])
    y_pred3 = vm.predict([X_morph_test])


    # draw_confusion_matrix(y_pred, y_test, y_names, path, figsize=(8,8), file_name='%s scTRAIT confusion matrix'%patient)
    # draw_confusion_matrix(y_pred2, y_test, y_names, path, figsize=(8,8), file_name='%s vanilla traj confusion matrix'%patient)
    # draw_confusion_matrix(y_pred3, y_test, y_names, path, figsize=(8,8), file_name='%s vanilla morph confusion matrix'%patient)


    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

    if y_pred.shape[1]>=3:
        y_class = np.argmax(y_pred, axis=1)
        y_class2 = np.argmax(y_pred2, axis=1)
        y_class3 = np.argmax(y_pred3, axis=1)

    else:
        y_class = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred)])
        y_class2 = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred2)])
        y_class3 = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred3)])


    f1 = f1_score(y_test, y_class)
    f1_2 = f1_score(y_test, y_class2)
    f1_3 = f1_score(y_test, y_class3)

    acs = accuracy_score(y_test, y_class)
    acs2 = accuracy_score(y_test, y_class2)
    acs3 = accuracy_score(y_test, y_class3)

    ps = precision_score(y_test, y_class)
    ps2 = precision_score(y_test, y_class2)
    ps3 = precision_score(y_test, y_class3)

    rs = recall_score(y_test, y_class)
    rs2 = recall_score(y_test, y_class2)
    rs3 = recall_score(y_test, y_class3)

    np.save(path + 'scTRAIT %s result.npy' % (patient), y_pred)
    np.save(path + 'VT %s result.npy' % patient, y_pred2)
    np.save(path + 'VM %s result.npy' % patient, y_pred3)

    np.savetxt(path+'%s result.txt'%patient,
               ['F1:', f1, f1_2, f1_3, 'Accuracy: ', acs, acs2, acs3, 'Precision: ', ps, ps2, ps3, 'Recall: ', rs, rs2, rs3],
               fmt='%s', delimiter ='')



#################################### Read classification accuracies ####################################
patients = np.unique(df['Patient'])

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\leave one out old vs young\\'

scores = {'scTRAIT': [], 'VT':[], 'VM': []}
for patient in patients:
    age_group = np.unique(df[df['Patient']==patient]['Type'])[0]
    age = np.unique(df[df['Patient'] == patient]['Age'])[0]

    txt = np.loadtxt(path+'%s result.txt'%patient, dtype=str)
    sctrait_score = float(txt[5])
    vt_score = float(txt[6])
    vm_score = float(txt[7])
    print(patient, age_group, age, sctrait_score)
    # if sctrait_score <=0.7:
    #     print(patient, age_group, age, sctrait_score)
    scores['scTRAIT'].append(sctrait_score)
    scores['VT'].append(vt_score)
    scores['VM'].append(vm_score)

draw_custom_violin_plot(scores, path, file_name='leave one out score', colors = ('#aa4499', '#CC6677', '#6699CC'),
                            test='mann-whitney', pvalue=True, figsize=(1,2))

draw_custom_bar_plot(scores, path, file_name='leave one out score', colors=('#aa4499', '#CC6677', '#6699CC'), vmax=1,
                     strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))

draw_custom_box_plot(scores, path, file_name='leave one out score', colors=('#aa4499', '#CC6677', '#6699CC'),
                     strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))

#################################### Classification frail vs old: Model evaluation with new batch (leave-one-out) ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

# Old and frail only
df = df[df['Type']!='Young'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Young'].reset_index(drop=True)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\leave one out frail vs old\\'

frail_patients = []
old_patients = []
for patient in np.unique(df['Patient']):
    df_part = df[df['Patient'] == patient].reset_index(drop=True)
    typ = np.unique(df_part['Type'])[0]
    if typ == 'Frail':
        frail_patients.append(patient)
    elif typ =='Old':
        old_patients.append(patient)

duration=30
for patient in tqdm(np.unique(df['Patient'])):
# for add_factor in np.arange(0, 1, 0.1):
    print('Patient %s starting'%patient)

    df_part = df[df['Patient']!=patient].reset_index(drop=True)
    df_duration_part = df_duration[df_duration['Patient']!=patient].reset_index(drop=True)

    df_part_test = df[df['Patient']==patient].reset_index(drop=True)
    df_duration_part_test = df_duration[df_duration['Patient']==patient].reset_index(drop=True)

    # ################### Add subset of testing to the training ###################
    # test_size = df_part_test.shape[0]
    # df_cut_off = int(test_size*add_factor)
    # df_duration_cut_off = df_cut_off*duration
    # df_add_train = df_part_test.iloc[:df_cut_off]
    # df_duration_add_train = df_duration_part_test.iloc[:df_duration_cut_off]
    #
    # df_part = pd.concat([df_part, df_add_train]).reset_index(drop=True)
    # df_duration_part = pd.concat([df_duration_part, df_duration_add_train]).reset_index(drop=True)
    # df_part_test = df_part_test.iloc[df_cut_off:]
    # df_duration_part_test = df_duration_part_test.iloc[df_duration_cut_off:]


# for frail, old in tqdm(zip(frail_patients, old_patients)):
#     print('Patient %s %s starting'%(frail, old))
    # df_part = df[(df['Patient'] != frail)&(df['Patient'] != old)].reset_index(drop=True)
    # df_duration_part = df_duration[(df_duration['Patient'] != frail)&(df_duration['Patient'] != old)].reset_index(drop=True)
    #
    # df_part_test = df[(df['Patient'] == frail)|(df['Patient'] == old)].reset_index(drop=True)
    # df_duration_part_test = df_duration[(df_duration['Patient'] == frail)|(df_duration['Patient'] == old)].reset_index(drop=True)

    ############# training set #############



    traj_scale_data = df_duration_part.loc[:, ['reg_x', 'reg_y']]
    traj_scaler = StandardScaler()
    traj_scale_data = pd.DataFrame(traj_scaler.fit_transform(traj_scale_data), columns=['reg_x', 'reg_y'])

    traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration, feature_name=['reg_x', 'reg_y'])
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
    rotated_trajectories = dict_to_array(trajectories)
    X_traj_train = rotated_trajectories

    local_features = ['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance']
    local_scale_data = df_duration_part.loc[:, local_features]
    local_scaler = StandardScaler()
    local_scale_data = pd.DataFrame(local_scaler.fit_transform(local_scale_data), columns=local_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(local_scale_data, duration=duration, feature_name=local_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
    rotated_trajectories = dict_to_array(trajectories)
    X_local_train = rotated_trajectories

    morphology_features = ['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length', 'minor_axis_length',
                           'aspect_ratio', 'circularity']
    scale_data = df_duration_part.loc[:, morphology_features]
    scaler = StandardScaler()
    scale_data = pd.DataFrame(scaler.fit_transform(scale_data), columns=morphology_features)

    traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration, feature_name=morphology_features)
    trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
    rotated_trajectories = dict_to_array(trajectories)
    X_morph_train = rotated_trajectories

    # traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part, duration=duration, feature_name=['reg_x', 'reg_y'])
    # trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
    # #rotated_trajectories = register_traj_disp_reflection(trajectories)
    # rotated_trajectories = dict_to_array(trajectories)
    # X_traj_train = rotated_trajectories
    #
    #
    # traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part, duration=duration, feature_name=['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance'])
    # trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
    # #rotated_trajectories = register_traj_disp_reflection(trajectories)
    # rotated_trajectories = dict_to_array(trajectories)
    # X_local_train = rotated_trajectories
    #
    #
    # traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part, duration=duration,
    #                                                                  feature_name=['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length',
    #                                                                                'minor_axis_length', 'aspect_ratio', 'circularity'])
    # trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
    # #rotated_trajectories = register_traj_disp_reflection(trajectories)
    # rotated_trajectories = dict_to_array(trajectories)
    # X_morph_train = rotated_trajectories

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_covar_train = le.fit_transform( np.array(df_part['Condition']) )
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    # from sklearn.preprocessing import OneHotEncoder
    # ohe = OneHotEncoder()
    # X_covar = ohe.fit_transform( np.array(df_nonyoung['Condition']).reshape(-1,1) )

    y_names = df_part['Type']
    print('Classes:', pd.unique(y_names))
    y_train = y_names.replace({'Frail': 0, 'Old':1}, inplace=False)
    #y = y_names.replace(list(pd.unique(y_names)), [i for i in range(pd.unique(y_names).shape[0])])
    y_train = np.array(y_train)
    print('Number of classes:', np.unique(y_train).size)

    from sklearn.utils import shuffle
    X_traj_train, X_local_train, X_morph_train, X_covar_train, y_train = shuffle(X_traj_train, X_local_train,
                                                                                 X_morph_train, X_covar_train, y_train, random_state=0)

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


    # traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part_test, duration=duration, feature_name=['reg_x', 'reg_y'])
    # trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
    # #rotated_trajectories = register_traj_disp_reflection(trajectories)
    # rotated_trajectories = dict_to_array(trajectories)
    # X_traj_test = rotated_trajectories
    #
    #
    # traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part_test, duration=duration, feature_name=['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance'])
    # trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
    # #rotated_trajectories = register_traj_disp_reflection(trajectories)
    # rotated_trajectories = dict_to_array(trajectories)
    # X_local_test = rotated_trajectories
    #
    #
    # traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration_part_test, duration=duration,
    #                                                                  feature_name=['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length',
    #                                                                                'minor_axis_length', 'aspect_ratio', 'circularity'])
    # trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
    # #rotated_trajectories = register_traj_disp_reflection(trajectories)
    # rotated_trajectories = dict_to_array(trajectories)
    # X_morph_test = rotated_trajectories

    le = LabelEncoder()
    le.fit(np.array(df_part['Condition']))
    X_covar_test = le.transform( np.array(df_part_test['Condition']) )
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    # from sklearn.preprocessing import OneHotEncoder
    # ohe = OneHotEncoder()
    # X_covar = ohe.fit_transform( np.array(df_nonyoung['Condition']).reshape(-1,1) )

    y_names_test = df_part_test['Type']
    print('Classes:', pd.unique(y_names_test))
    y_test = y_names_test.replace({'Frail': 0, 'Old': 1}, inplace=False)
    #y = y_names_test.replace(list(pd.unique(y_names_test)), [i for i in range(pd.unique(y_names_test).shape[0])])
    y_test = np.array(y_test)
    print('Number of classes:', np.unique(y_test).size)

    #################################### Model training ####################################

    from tensorflow.keras import callbacks
    stop_early = callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001)
    # checkpointer = callbacks.ModelCheckpoint('saved_model/monocyte_project_v2/%s'%model_name, monitor='accuracy', verbose=1,
    #                 save_weights_only=False, save_freq='epoch', every_n_epochs=10, mode='auto', save_best_only=True)  # Save model every epoch

    from dnn.classification import scTRAIT
    sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=np.unique(y_train).size)
    result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=500,
                         verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
    sctrait.save('saved_model/monocyte_project_v2/leave one out old vs frail scTRAIT %s'%patient, save_format='tf')
    #sctrait = tf.keras.models.load_model('saved_model/monocyte_project_v2/leave one out old vs frail scTRAIT %s'%patient, compile=True)
    ### model2: Vanilla trajectory ###
    # vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=np.unique(y_train).size)
    # result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=500, verbose=1, validation_split=0.1, shuffle=True,
    #                      callbacks=[stop_early])
    # vt.save('saved_model/monocyte_project_v2/leave one out old vs frail vanilla traj %s'%patient, save_format='tf')
    vt = tf.keras.models.load_model('saved_model/monocyte_project_v2/leave one out old vs frail vanilla traj %s'%patient, compile=True)

    ### model3: Vanilla morphology ###
    # vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=np.unique(y_train).size)
    # result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=500, verbose=1, validation_split=0.1, shuffle=True,
    #                      callbacks=[stop_early])
    # vm.save('saved_model/monocyte_project_v2/leave one out old vs frail vanilla morpho %s'%patient, save_format='tf')
    vm = tf.keras.models.load_model('saved_model/monocyte_project_v2/leave one out old vs frail vanilla morpho %s'%patient, compile=True)
    y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
    y_pred2 = vt.predict([X_traj_test])
    y_pred3 = vm.predict([X_morph_test])


    # draw_confusion_matrix(y_pred, y_test, y_names, path, figsize=(8,8), file_name='%s scTRAIT confusion matrix'%patient)
    # draw_confusion_matrix(y_pred2, y_test, y_names, path, figsize=(8,8), file_name='%s vanilla traj confusion matrix'%patient)
    # draw_confusion_matrix(y_pred3, y_test, y_names, path, figsize=(8,8), file_name='%s vanilla morph confusion matrix'%patient)


    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

    if y_pred.shape[1]>=3:
        y_class = np.argmax(y_pred, axis=1)
        y_class2 = np.argmax(y_pred2, axis=1)
        y_class3 = np.argmax(y_pred3, axis=1)

    else:
        y_class = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred)])
        y_class2 = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred2)])
        y_class3 = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred3)])

    f1 = f1_score(y_test, y_class)
    f1_2 = f1_score(y_test, y_class2)
    f1_3 = f1_score(y_test, y_class3)

    acs = accuracy_score(y_test, y_class)
    acs2 = accuracy_score(y_test, y_class2)
    acs3 = accuracy_score(y_test, y_class3)

    ps = precision_score(y_test, y_class)
    ps2 = precision_score(y_test, y_class2)
    ps3 = precision_score(y_test, y_class3)

    rs = recall_score(y_test, y_class)
    rs2 = recall_score(y_test, y_class2)
    rs3 = recall_score(y_test, y_class3)

    np.save(path + 'scTRAIT %s result.npy' %(patient), y_pred)
    np.save(path + 'VT %s result.npy' % patient, y_pred2)
    np.save(path + 'VM %s result.npy' % patient, y_pred3)

    np.savetxt(path+'%s result.txt'%patient, ['Accuracy: ', acs, acs2, acs3,
                                              'F1 score: ', f1, f1_2, f1_3,
                                              'Precision: ', ps, ps2, ps3,
                                              'Recall: ', rs, rs2, rs3], fmt='%s', delimiter ='')

    # np.save(path + 'scTRAIT %s %s result.npy' % (frail, old), y_pred)
    # np.save(path + 'VT %s %s result.npy' % (frail, old), y_pred2)
    # np.save(path + 'VM %s %s result.npy' % (frail, old), y_pred3)
    #
    # np.savetxt(path + '%s %s result.txt' % (frail, old), ['Accuracy: ', acs, acs2, acs3,
    #                                             'F1 score: ', f1, f1_2, f1_3,
    #                                             'Precision: ', ps, ps2, ps3,
    #                                             'Recall: ', rs, rs2, rs3], fmt='%s', delimiter ='')


#################################### Read classification accuracies ####################################
patients = np.unique(df['Patient'])

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\leave one out frail vs old\\'

scores = {'scTRAIT': [], 'VT':[], 'VM': []}
for patient in patients:
#for frail, old in tqdm(zip(frail_patients, old_patients)):
    age_group = np.unique(df[df['Patient']==patient]['Type'])[0]
    age = np.unique(df[df['Patient'] == patient]['Age'])[0]

    txt = np.loadtxt(path+'%s result.txt'%patient, dtype=str, usecols=0)
    #print(frail, old)
    # txt = np.loadtxt(path + '%s %s result.txt' % (frail, old), dtype=str)


    sctrait_score = float(txt[1])
    vt_score = float(txt[2])
    vm_score = float(txt[3])
    #print(patient, age_group, age, sctrait_score)
    if sctrait_score <=0.3:
        print(patient, age_group, age, sctrait_score)

    scores['scTRAIT'].append(sctrait_score)
    scores['VT'].append(vt_score)
    scores['VM'].append(vm_score)

# draw_custom_violin_plot(scores, path, file_name='leave one out score', colors = ('#aa4499', '#CC6677', '#6699CC'),
#                             test='mann-whitney', pvalue=True, figsize=(1,2))

draw_custom_bar_plot(scores, path, file_name='leave one out score', colors=('#aa4499', '#CC6677', '#6699CC'), vmax=1,
                     strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))

draw_custom_box_plot(scores, path, file_name='leave one out score', colors=('#aa4499', '#CC6677', '#6699CC'),
                     strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))

