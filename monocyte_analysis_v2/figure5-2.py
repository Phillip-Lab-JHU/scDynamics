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

"""Generates Data for Figure 5-2 non-frail vs frail."""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast
from dnn.autoencoders import set_duration_for_autoencoder
from dnn.classification import Temporal_Conv1D_2D_classifier, Res_Conv1D_LSTM_classifier
import tensorflow as tf

# ########################### Split dataset for training and rep ###########################
# path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\\'
# df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
# df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')
#
# df_rep = pd.read_parquet(path+'cleaned_rep_motility_features_30.parquet')
# df_duration_rep = pd.read_parquet(path+'cleaned_rep_traj_duration_30.parquet')
#
#
# df_train = df[df['Patient']!='F1095'].reset_index(drop=True)
# df_train = df_train[(df_train['Patient']!='F1044')|(df_train['Condition']!='IL6')].reset_index(drop=True)
#
# df_duration_train = df_duration[df_duration['Patient']!='F1095'].reset_index(drop=True)
# df_duration_train = df_duration_train[(df_duration_train['Patient']!='F1044')|(df_duration_train['Condition']!='IL6')].reset_index(drop=True)
#
#
#
# df_rep_summed = pd.concat([df_rep, df[df['Patient']=='F1095'].reset_index(drop=True)], axis=0)
# df_rep_summed = pd.concat([df_rep_summed, df[(df['Patient']=='F1044')&(df['Condition']=='IL6')].reset_index(drop=True)], axis=0)
# df_rep_summed = df_rep_summed.reset_index(drop=True)
#
# df_duration_rep_summed = pd.concat([df_duration_rep, df_duration[df_duration['Patient']=='F1095'].reset_index(drop=True)], axis=0)
# df_duration_rep_summed = pd.concat([df_duration_rep_summed, df_duration[(df_duration['Patient']=='F1044')&(df_duration['Condition']=='IL6')].reset_index(drop=True)], axis=0)
# df_duration_rep_summed = df_duration_rep_summed.reset_index(drop=True)
#
# path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure5. Classification\\'
# df_train.to_parquet(path+'df.parquet')
# df_duration_train.to_parquet(path+'df_duration.parquet')
#
# df_rep_summed.to_parquet(path+'rep_df.parquet')
# df_duration_rep_summed.to_parquet(path+'rep_df_duration.parquet')


########################### Prepare dataset for training ###########################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df = df[df['Type']!='Young'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Young'].reset_index(drop=True)

#df = pd.read_parquet(path+'nonmoving_removed_all_features_30_PC.parquet')
#df_duration = pd.read_parquet(path+'nonmoving_removed_traj_duration_30.parquet')

#################################### Training data preparation ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure5. Classification\frail vs non frail\\'

duration=30
traj_scale_data = df_duration.loc[:, ['reg_x', 'reg_y']]
traj_scaler = StandardScaler()
traj_scale_data= pd.DataFrame(traj_scaler.fit_transform( traj_scale_data ), columns=['reg_x', 'reg_y'])

traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration, feature_name=['reg_x', 'reg_y'])
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
rotated_trajectories = dict_to_array(trajectories)
X_traj = rotated_trajectories

local_features = ['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance']
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
X_covar = le.fit_transform( np.array(df['Condition']) )
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
X_covar_train, X_covar_test, y_train, y_test = train_test_split(X_traj, X_local, X_morph, X_covar, y, test_size=0.2, random_state=0, stratify=y)
print(X_traj_train.shape, X_traj_test.shape, X_local_train.shape, X_local_test.shape,
      X_morph_train.shape, X_morph_test.shape, X_covar_train.shape, X_covar_test.shape, y_train.shape, y_test.shape)


#################################### Model training ####################################

from tensorflow.keras import callbacks
stop_early = callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001)
# checkpointer = callbacks.ModelCheckpoint('saved_model/monocyte_project_v2/%s'%model_name, monitor='accuracy', verbose=1,
#                 save_weights_only=False, save_freq='epoch', every_n_epochs=10, mode='auto', save_best_only=True)  # Save model every epoch

from dnn.classification import scTRAIT
sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=np.unique(y).size)
result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=500,
                     verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
sctrait.save('saved_model/monocyte_project_v2/frail vs nonfrail scTRAIT', save_format='tf')
#sctrait.save('saved_model/monocyte_project_v2/frail vs prefrail vs nonfrail scTRAIT', save_format='tf')
### model2: Vanilla trajectory ###
vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=np.unique(y).size)
result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=500, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vt.save('saved_model/monocyte_project_v2/frail vs nonfrail vanilla traj', save_format='tf')
#vt.save('saved_model/monocyte_project_v2/frail vs prefrail vs nonfrail vanilla traj', save_format='tf')
### model3: Vanilla morphology ###
vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=np.unique(y).size)
result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=500, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vm.save('saved_model/monocyte_project_v2/frail vs nonfrail vanilla morpho', save_format='tf')
#vm.save('saved_model/monocyte_project_v2/frail vs prefrail vs nonfrail vanilla morpho', save_format='tf')
#################################### Load trained model ####################################
sctrait = tf.keras.models.load_model('saved_model/monocyte_project_v2/frail vs nonfrail scTRAIT', compile=True)
# result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=2,
#                      verbose=1, validation_split=0.1, shuffle=True,)

vt = tf.keras.models.load_model('saved_model/monocyte_project_v2/frail vs nonfrail vanilla traj', compile=True)
# result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

vm = tf.keras.models.load_model('saved_model/monocyte_project_v2/frail vs nonfrail vanilla morpho', compile=True)
# result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

#################################### Model prediction ####################################
y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
y_pred2 = vt.predict([X_traj_test])
y_pred3 = vm.predict([X_morph_test])

#################################### Model accuracy evaluation ####################################

draw_confusion_matrix(y_pred, y_test, y_names, path, figsize=(4,4), file_name='scTRAIT confusion matrix', vmax=0.9)
draw_confusion_matrix(y_pred2, y_test, y_names, path, figsize=(4,4), file_name='vanilla traj confusion matrix', vmax=0.9)
draw_confusion_matrix(y_pred3, y_test, y_names, path, figsize=(4,4), file_name='vanilla morpho confusion matrix', vmax=0.9)


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

print(f1, f1_2, f1_3, acs, acs2, acs3, ps, ps2, ps3, rs, rs2, rs3)

dict_datasets = {'VM': np.array(f1_3), 'VT': np.array(f1_2), 'scTRAIT': np.array(f1)}
#dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='f1 scores', colors=('#6699CC', '#CC6677', '#aa4499'), vmax=1,
                     strip_plot=False, test='t-test', pvalue=False, figsize=(1,2))

dict_datasets = {'VM': np.array(ps3), 'VT': np.array(ps2), 'scTRAIT': np.array(ps)}
#dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='precision scores', colors=('#6699CC', '#CC6677', '#aa4499'), vmax=1,
                     strip_plot=False, test='t-test', pvalue=False, figsize=(1,2))

dict_datasets = {'VM': np.array(rs3), 'VT': np.array(rs2), 'scTRAIT': np.array(rs)}
#dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='recall scores', colors=('#6699CC', '#CC6677', '#aa4499'), vmax=1,
                     strip_plot=False, test='t-test', pvalue=False, figsize=(1,2))

#################################### ROC curve and AUC ####################################
from sklearn.metrics import roc_curve
fpr1, tpr1, thresh1 = roc_curve(y_test, y_pred, pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, y_pred2, pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_test, y_pred3, pos_label=1)

auc_score = roc_auc_score(y_test, y_pred)
auc_score2 = roc_auc_score(y_test, y_pred2)
auc_score3 = roc_auc_score(y_test, y_pred3)


font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(2.7,2.7))

#plt.style.use('seaborn') # plot
plt.plot(fpr1, tpr1, linestyle='-.', color='#aa4499', label='scTRAIT (AUC = %.2f)'%auc_score,  linewidth=3)
plt.plot(fpr2, tpr2, linestyle='-.', color='#CC6677', label='VT (AUC = %.2f)'%auc_score2,  linewidth=3)
plt.plot(fpr3, tpr3, linestyle='-.', color='#6699CC', label='VM (AUC = %.2f)'%auc_score3,  linewidth=3)
plt.plot([0, 1], [0, 1], linestyle='--', color='black',  linewidth=1, )
#plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=1, color='0.2')

plt.xticks(fontsize=8, color='0.2', weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')
#plt.title('AUC: %s'%auc_score)
ax.set_xlabel('False Positive Rate', fontsize=8, weight='bold', color='0.2')
ax.set_ylabel('True Positive Rate', fontsize=8, weight='bold', color='0.2')

plt.legend(frameon=False, prop = {'weight':'bold', 'size':8}, labelcolor='0.2', loc='best')
plt.savefig(path + 'ROC curve', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/ROC curve.svg', bbox_inches='tight')
plt.clf()
plt.close()


#################################### Model evaluation with replicate (new batch) ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df_rep = pd.read_parquet(path+'cleaned_rep_motility_features_30.parquet')
df_duration_rep = pd.read_parquet(path+'cleaned_rep_traj_duration_30.parquet')

df_rep = df_rep[df_rep['Type']!='Young'].reset_index(drop=True)
df_duration_rep = df_duration_rep[df_duration_rep['Type']!='Young'].reset_index(drop=True)

# df_rep = df_rep[(df_rep['Patient']!='F1095')&(df_rep['Patient']!='F1044')].reset_index(drop=True)
# df_duration_rep = df_duration_rep[(df_duration_rep['Patient']!='F1095')&(df_duration_rep['Patient']!='F1044')].reset_index(drop=True)


duration=30
traj_scale_data = df_duration.loc[:, ['reg_x', 'reg_y']]
traj_scaler = StandardScaler()
traj_scaler.fit(traj_scale_data)
traj_scale_data = df_duration_rep.loc[:, ['reg_x', 'reg_y']]
traj_scale_data= pd.DataFrame(traj_scaler.transform( traj_scale_data ), columns=['reg_x', 'reg_y'])

traj_list, trajectories_array, trajectories = to_timeseries_fast(traj_scale_data, duration=duration, feature_name=['reg_x', 'reg_y'])
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
#rotated_trajectories = register_traj_disp_reflection(trajectories)
rotated_trajectories = dict_to_array(trajectories)
X_traj_rep = rotated_trajectories

local_features = ['reg_x', 'reg_y', 'n_neighbors', 'shortest_distance']
local_scale_data = df_duration.loc[:, local_features]
local_scaler = StandardScaler()
local_scaler.fit(local_scale_data)
local_scale_data = df_duration_rep.loc[:, local_features]
local_scale_data= pd.DataFrame(local_scaler.transform( local_scale_data ), columns=local_features)

traj_list, trajectories_array, trajectories = to_timeseries_fast(local_scale_data, duration=duration, feature_name=local_features)
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=4)
#rotated_trajectories = register_traj_disp_reflection(trajectories)
rotated_trajectories = dict_to_array(trajectories)
X_local_rep = rotated_trajectories


morphology_features=['area', 'solidity', 'eccentricity', 'extent', 'major_axis_length','minor_axis_length', 'aspect_ratio', 'circularity']
scale_data = df_duration.loc[:, morphology_features]
scaler = StandardScaler()
scaler.fit(scale_data)
scale_data = df_duration_rep.loc[:, morphology_features]
scale_data= pd.DataFrame(scaler.transform( scale_data ), columns=morphology_features)

traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration, feature_name=morphology_features)
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
#rotated_trajectories = register_traj_disp_reflection(trajectories)
rotated_trajectories = dict_to_array(trajectories)
X_morph_rep = rotated_trajectories

le = LabelEncoder()
le.fit( np.array(df['Condition']) )
X_covar_rep = le.transform( np.array(df_rep['Condition']) )
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

y_names_rep = df_rep['Type']
print('Classes:', pd.unique(y_names_rep))
y_rep = y_names_rep.replace(list(pd.unique(y_names_rep)), [i for i in range(pd.unique(y_names_rep).shape[0])])
y_rep = np.array(y_rep)
print('Number of classes:', np.unique(y_rep).size)

#################################### Load trained model ####################################
sctrait = tf.keras.models.load_model('saved_model/monocyte_project_v2/frail vs nonfrail scTRAIT', compile=True)
# result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=2,
#                      verbose=1, validation_split=0.1, shuffle=True,)

vt = tf.keras.models.load_model('saved_model/monocyte_project_v2/frail vs nonfrail vanilla traj', compile=True)
# result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

vm = tf.keras.models.load_model('saved_model/monocyte_project_v2/frail vs nonfrail vanilla morpho', compile=True)
# result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

#################################### prediction ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure5. Classification\rep frail vs non frail\\'

y_pred_rep = sctrait.predict([X_traj_rep, X_local_rep, X_morph_rep, X_covar_rep])
y_pred_rep2 = vt.predict([X_traj_rep])
y_pred_rep3 = vm.predict([X_morph_rep])


draw_confusion_matrix(y_pred_rep, y_rep, y_names_rep, path, figsize=(4,4), file_name='rep scTRAIT confusion matrix', vmax=0.9)
draw_confusion_matrix(y_pred_rep2, y_rep, y_names_rep, path, figsize=(4,4), file_name='rep vanilla traj confusion matrix', vmax=0.9)
draw_confusion_matrix(y_pred_rep3, y_rep, y_names_rep, path, figsize=(4,4), file_name='rep vanilla morph confusion matrix', vmax=0.9)


from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

if y_pred_rep.shape[1]>=3:
    y_class_rep = np.argmax(y_pred_rep, axis=1)
    y_class_rep2 = np.argmax(y_pred_rep2, axis=1)
    y_class_rep3 = np.argmax(y_pred_rep3, axis=1)

else:
    y_class_rep = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred_rep)])
    y_class_rep2 = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred_rep2)])
    y_class_rep3 = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred_rep3)])

f1 = f1_score(y_rep, y_class_rep)
f1_2 = f1_score(y_rep, y_class_rep2)
f1_3 = f1_score(y_rep, y_class_rep3)

acs = accuracy_score(y_rep, y_class_rep)
acs2 = accuracy_score(y_rep, y_class_rep2)
acs3 = accuracy_score(y_rep, y_class_rep3)

ps = precision_score(y_rep, y_class_rep)
ps2 = precision_score(y_rep, y_class_rep2)
ps3 = precision_score(y_rep, y_class_rep3)

rs = recall_score(y_rep, y_class_rep)
rs2 = recall_score(y_rep, y_class_rep2)
rs3 = recall_score(y_rep, y_class_rep3)

print(f1, f1_2, f1_3, acs, acs2, acs3, ps, ps2, ps3, rs, rs2, rs3)

dict_datasets = {'VM': np.array(f1_3), 'VT': np.array(f1_2), 'scTRAIT': np.array(f1)}
#dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='f1 scores', colors=('#6699CC', '#CC6677', '#aa4499'), vmax=1,
                     strip_plot=False, test='t-test', pvalue=False, figsize=(1,2))

dict_datasets = {'VM': np.array(ps3), 'VT': np.array(ps2), 'scTRAIT': np.array(ps)}
#dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='precision scores', colors=('#6699CC', '#CC6677', '#aa4499'), vmax=1,
                     strip_plot=False, test='t-test', pvalue=False, figsize=(1,2))

dict_datasets = {'VM': np.array(rs3), 'VT': np.array(rs2), 'scTRAIT': np.array(rs)}
#dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='recall scores', colors=('#6699CC', '#CC6677', '#aa4499'), vmax=1,
                     strip_plot=False, test='t-test', pvalue=False, figsize=(1,2))



from sklearn.metrics import roc_curve
fpr1, tpr1, thresh1 = roc_curve(y_rep, y_pred_rep, pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_rep, y_pred_rep2, pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_rep, y_pred_rep3, pos_label=1)

auc_score = roc_auc_score(y_rep, y_pred_rep)
auc_score2 = roc_auc_score(y_rep, y_pred_rep2)
auc_score3 = roc_auc_score(y_rep, y_pred_rep3)


font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(2.7,2.7))

#plt.style.use('seaborn') # plot
plt.plot(fpr1, tpr1, linestyle='-.', color='#aa4499', label='scTRAIT (AUC = %.2f)'%auc_score,  linewidth=3)
plt.plot(fpr2, tpr2, linestyle='-.', color='#CC6677', label='VT (AUC = %.2f)'%auc_score2,  linewidth=3)
plt.plot(fpr3, tpr3, linestyle='-.', color='#6699CC', label='VM (AUC = %.2f)'%auc_score3,  linewidth=3)
plt.plot([0, 1], [0, 1], linestyle='--', color='black',  linewidth=1, )
#plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=1, color='0.2')

plt.xticks(fontsize=8, color='0.2', weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')
#plt.title('AUC: %s'%auc_score)
ax.set_xlabel('False Positive Rate', fontsize=8, weight='bold', color='0.2')
ax.set_ylabel('True Positive Rate', fontsize=8, weight='bold', color='0.2')

plt.legend(frameon=False, prop = {'weight':'bold', 'size':8}, labelcolor='0.2', loc='best')
plt.savefig(path + 'rep ROC curve', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/rep ROC curve.svg', bbox_inches='tight')
plt.clf()
plt.close()




# #################################### Extract pre perturbation embedding ####################################
#
# from tensorflow.keras import models
# sctrait = tf.keras.models.load_model('saved_model/monocyte_project_v2/frail vs nonfrail scTRAIT', compile=True)
#
# traj_pert_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('delta_emb_traj').output)
# traj_pert_embeding = traj_pert_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
# traj_pert_embeding[X_covar_test==0][0]
#
# traj_pert_embeddings = []
# for i in np.unique(X_covar_test):
#     traj_pert_embeddings.append( traj_pert_embeding[X_covar_test == i][0] )
# traj_pert_embeddings = np.array(traj_pert_embeddings)
#
#
# local_pert_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('delta_emb_local').output)
# local_pert_embeding = local_pert_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#
# local_pert_embeddings = []
# for i in np.unique(X_covar_test):
#     local_pert_embeddings.append( local_pert_embeding[X_covar_test == i][0] )
# local_pert_embeddings = np.array(local_pert_embeddings)
#
# morph_pert_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('delta_emb_morph').output)
# morph_pert_embeding = morph_pert_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#
# morph_pert_embeddings = []
# for i in np.unique(X_covar_test):
#     morph_pert_embeddings.append( morph_pert_embeding[X_covar_test == i][0] )
# morph_pert_embeddings = np.array(morph_pert_embeddings)
#
#
# traj_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('emb_traj').output)
# traj_embbeding = traj_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#
# local_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('emb_local').output)
# local_embbeding = local_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#
# morph_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('emb_morph').output)
# morph_embbeding = morph_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#
# traj_concat = np.concatenate([traj_embbeding, traj_pert_embeddings], axis=0)
# local_concat = np.concatenate([local_embbeding, local_pert_embeddings], axis=0)
# morph_concat = np.concatenate([morph_embbeding, morph_pert_embeddings], axis=0)
#
#
# df_traj = pd.DataFrame(traj_concat, columns=[ 'traj_emb_%s' % str(i) for i in range(0,traj_embbeding.shape[1]) ])
# df_local = pd.DataFrame(local_concat, columns=[ 'local_emb_%s' % str(i) for i in range(0,local_embbeding.shape[1]) ])
# df_morph = pd.DataFrame(morph_concat, columns=[ 'morph_emb_%s' % str(i) for i in range(0,morph_embbeding.shape[1]) ])
#
# df_traj['Label'] = 'Data'
# df_traj.loc[df_traj.index[-4:], 'Label'] = 'Vector'
# df_traj['Type'] = np.concatenate([y_test, np.array(['Vector', 'Vector', 'Vector', 'Vector'])])
# pd.unique(y_names)
# #print(le_name_mapping)
# df_traj.loc[df_traj['Type'] == '0', 'Type'] = 'Frail'
# df_traj.loc[df_traj['Type'] == '1', 'Type'] = 'Old'
#
# df_local['Label'] = 'Data'
# df_local.loc[df_local.index[-4:], 'Label'] = 'Vector'
# df_local['Type'] = np.concatenate([y_test, np.array(['Vector', 'Vector', 'Vector', 'Vector'])])
# pd.unique(y_names)
# #print(le_name_mapping)
# df_local.loc[df_local['Type'] == '0', 'Type'] = 'Frail'
# df_local.loc[df_local['Type'] == '1', 'Type'] = 'Old'
#
# df_morph['Label'] = 'Data'
# df_morph.loc[df_morph.index[-4:], 'Label'] = 'Vector'
# df_morph['Type'] = np.concatenate([y_test, np.array(['Vector', 'Vector', 'Vector', 'Vector'])])
# pd.unique(y_names)
# #print(le_name_mapping)
# df_morph.loc[df_morph['Type'] == '0', 'Type'] = 'Frail'
# df_morph.loc[df_morph['Type'] == '1', 'Type'] = 'Old'
#
# path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure5. Classification\frail vs non frail\\'
#
# df_traj.to_csv(path + 'traj_emb.csv', index=False)
# df_traj.to_parquet(path + 'traj_emb.parquet')
#
# df_local.to_csv(path + 'local_emb.csv', index=False)
# df_local.to_parquet(path + 'local_emb.parquet')
#
# df_morph.to_csv(path + 'morph_emb.csv', index=False)
# df_morph.to_parquet(path + 'morph_emb.parquet')
#
# #################################### Extract post perturbation embedding ####################################
#
# from tensorflow.keras import models
# traj_pert_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('delta_emb_traj').output)
# traj_pert_embeding = traj_pert_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
# traj_pert_embeding[X_covar_test==0][0]
#
# traj_pert_embeddings = []
# for i in np.unique(X_covar_test):
#     traj_pert_embeddings.append( traj_pert_embeding[X_covar_test == i][0] )
# traj_pert_embeddings = np.array(traj_pert_embeddings)
#
#
# local_pert_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('delta_emb_local').output)
# local_pert_embeding = local_pert_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#
# local_pert_embeddings = []
# for i in np.unique(X_covar_test):
#     local_pert_embeddings.append( local_pert_embeding[X_covar_test == i][0] )
# local_pert_embeddings = np.array(local_pert_embeddings)
#
# morph_pert_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('delta_emb_morph').output)
# morph_pert_embeding = morph_pert_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#
# morph_pert_embeddings = []
# for i in np.unique(X_covar_test):
#     morph_pert_embeddings.append( morph_pert_embeding[X_covar_test == i][0] )
# morph_pert_embeddings = np.array(morph_pert_embeddings)
#
#
# traj_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('MLP_perturbed_emb_traj').output)
# traj_embbeding = traj_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#
# local_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('MLP_perturbed_emb_local').output)
# local_embbeding = local_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#
# morph_emb = models.Model(inputs=sctrait.inputs, outputs=sctrait.get_layer('MLP_perturbed_emb_morph').output)
# morph_embbeding = morph_emb.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#
# traj_concat = np.concatenate([traj_embbeding, traj_pert_embeddings[:, :32]], axis=0)
# local_concat = np.concatenate([local_embbeding, local_pert_embeddings[:, :32]], axis=0)
# morph_concat = np.concatenate([morph_embbeding, morph_pert_embeddings[:, :32]], axis=0)
#
#
# df_traj = pd.DataFrame(traj_concat, columns=[ 'traj_emb_%s' % str(i) for i in range(0,traj_embbeding.shape[1]) ])
# df_local = pd.DataFrame(local_concat, columns=[ 'local_emb_%s' % str(i) for i in range(0,local_embbeding.shape[1]) ])
# df_morph = pd.DataFrame(morph_concat, columns=[ 'morph_emb_%s' % str(i) for i in range(0,morph_embbeding.shape[1]) ])
#
# df_traj['Label'] = 'Data'
# df_traj.loc[df_traj.index[-4:], 'Label'] = 'Vector'
# df_traj['Type'] = np.concatenate([y_test, np.array(['Vector', 'Vector', 'Vector', 'Vector'])])
# pd.unique(y_names)
# #print(le_name_mapping)
# df_traj.loc[df_traj['Type'] == '0', 'Type'] = 'Frail'
# df_traj.loc[df_traj['Type'] == '1', 'Type'] = 'Old'
#
# df_local['Label'] = 'Data'
# df_local.loc[df_local.index[-4:], 'Label'] = 'Vector'
# df_local['Type'] = np.concatenate([y_test, np.array(['Vector', 'Vector', 'Vector', 'Vector'])])
# pd.unique(y_names)
# #print(le_name_mapping)
# df_local.loc[df_local['Type'] == '0', 'Type'] = 'Frail'
# df_local.loc[df_local['Type'] == '1', 'Type'] = 'Old'
#
# df_morph['Label'] = 'Data'
# df_morph.loc[df_morph.index[-4:], 'Label'] = 'Vector'
# df_morph['Type'] = np.concatenate([y_test, np.array(['Vector', 'Vector', 'Vector', 'Vector'])])
# pd.unique(y_names)
# #print(le_name_mapping)
# df_morph.loc[df_morph['Type'] == '0', 'Type'] = 'Frail'
# df_morph.loc[df_morph['Type'] == '1', 'Type'] = 'Old'
#
# path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure5. Classification\frail vs non frail\\'
#
# df_traj.to_csv(path + 'perturbed_traj_emb.csv', index=False)
# df_traj.to_parquet(path + 'perturbed_traj_emb.parquet')
#
# df_local.to_csv(path + 'perturbed_local_emb.csv', index=False)
# df_local.to_parquet(path + 'perturbed_local_emb.parquet')
#
# df_morph.to_csv(path + 'perturbed_morph_emb.csv', index=False)
# df_morph.to_parquet(path + 'perturbed_morph_emb.parquet')
#
# #################################### Pre perturbation Embedding space ####################################
# path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure5. Classification\frail vs non frail\\'
# # df_traj = pd.read_csv(path+'traj_emb.csv')
# # df_local = pd.read_csv(path+'local_emb.csv')
# # df_morph = pd.read_csv(path+'morph_emb.csv')
#
# umaps = pd.DataFrame()
# motility_datas = pd.DataFrame()
# for name in ['traj_emb', 'local_emb', 'morph_emb']:
#     df_temp = pd.read_parquet(path+'%s.parquet'%name)
#     df_temp = df_temp.iloc[:-4, :]
#     motility_data = df_temp.iloc[:, :64]
#     print(df_temp.shape)
#     scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#     #aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
#     motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
#     from sklearn.decomposition import PCA
#     pca = PCA(0.95)
#     pcs = pca.fit_transform(motility_data_scaled)
#
#     m = Morphodynamics(df_temp, 'umap')
#     umap = m.get_umap(pcs, 20, 0.5)
#     for column in umap.columns:
#         umap.rename(columns={column:'%s_'%name+column}, inplace=True)
#
#     #m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
#     #cluster = m.get_cluster(pcs, n_clusters=11, cluster_type='kmeans')
#     umaps = pd.concat([umaps, umap], axis=1)
#     motility_datas = pd.concat([motility_datas, motility_data], axis=1)
#
# df = pd.concat([motility_datas, umaps], axis=1)
# df['Label'] = df_temp['Label']
# df['Type'] = df_temp['Type']
#
#
# # df_traj = pd.read_csv(path+'traj_emb.csv')
# # df_local = pd.read_csv(path+'local_emb.csv')
# # df_morph = pd.read_csv(path+'morph_emb.csv')
#
# # from itertools import combinations
# # for pair in list(combinations(['traj_emb_PC1', 'traj_emb_PC2', 'local_emb_PC1', 'local_emb_PC2', 'morph_emb_PC1', 'morph_emb_PC2'], 2)):
# #
# #     draw_umap_space(df, path, file_name='%s_%s_type'%(pair[0], pair[1]), condition_name='Type', label_name='Type',
# #                     colors = ('#6699CC', '#888888', ), dot_size=0.07, x_name=pair[0], y_name=pair[1])
# #
# # from itertools import combinations
# # for pair in list(combinations(['traj_emb_PC1', 'traj_emb_PC2', 'local_emb_PC1', 'local_emb_PC2', 'morph_emb_PC1', 'morph_emb_PC2'], 2)):
# #
# #     draw_contour(df, path, file_name='contour_%s_%s'%(pair[0], pair[1]), condition_name='Type',
# #                  colors=('#6699CC', '#888888',), x_name=pair[0], y_name=pair[1], bin_num=50, num_contours=5)
#
# fig, ax = plt.subplots()
# grid = sns.PairGrid(data=df, vars=['traj_emb_PC1', 'traj_emb_PC2', 'local_emb_PC1', 'local_emb_PC2', 'morph_emb_PC1', 'morph_emb_PC2'],
#                     height=6, aspect=1, hue='Type', hue_order=['Frail', 'Old'], palette=('#fdc086', '#beaed4'), diag_sharey=False, despine=True, corner=True,)
# grid.map_upper(sns.kdeplot,  alpha=0.8, linewidths=1.5,
#                     fill=True, legend=False, common_norm=False, thresh=0.2)
# #grid = grid.map_upper(corr)
# grid.map_lower(sns.kdeplot, alpha=0.8, linewidths=1.5,
#                     fill=True, legend=False, common_norm=False, thresh=0.2)
# grid.map_diag(sns.kdeplot, alpha=0.8, linewidths=1.5,
#                     fill=True, legend=False, common_norm=False, thresh=0.2);
#
# plt.tick_params(left=False, right=False, labelleft=False,
#             labelbottom=False, bottom=False)
#
# plt.tight_layout()
#
# plt.savefig(path + '/pre perturbation pair plot all embedding.png', dpi=300)
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/pre perturbation pair plot all embedding.svg')
# plt.clf()
# plt.close()
#
# #################################### Post perturbation Embedding space ####################################
# path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure5. Classification\frail vs non frail\\'
#
# umaps = pd.DataFrame()
# motility_datas = pd.DataFrame()
# for name in ['perturbed_traj_emb', 'perturbed_local_emb', 'perturbed_morph_emb']:
#     df_temp = pd.read_parquet(path+'%s.parquet'%name)
#     df_temp = df_temp.iloc[:-4, :]
#     motility_data = df_temp.iloc[:, :32]
#     print(df_temp.shape)
#     scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#     #aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
#     motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
#     from sklearn.decomposition import PCA
#     pca = PCA(0.95)
#     pcs = pca.fit_transform(motility_data_scaled)
#
#     m = Morphodynamics(df_temp, 'umap')
#     umap = m.get_umap(pcs, 20, 0.5)
#     for column in umap.columns:
#         umap.rename(columns={column:'%s_'%name+column}, inplace=True)
#
#     #m.evaluate_cluster(pcs, path, cluster_type='kmeans', k_max=50)
#     #cluster = m.get_cluster(pcs, n_clusters=11, cluster_type='kmeans')
#     umaps = pd.concat([umaps, umap], axis=1)
#     motility_datas = pd.concat([motility_datas, motility_data], axis=1)
#
# df = pd.concat([motility_datas, umaps], axis=1)
# df['Label'] = df_temp['Label']
# df['Type'] = df_temp['Type']
#
#
# # df_traj = pd.read_csv(path+'traj_emb.csv')
# # df_local = pd.read_csv(path+'local_emb.csv')
# # df_morph = pd.read_csv(path+'morph_emb.csv')
#
# # from itertools import combinations
# # for pair in list(combinations(['traj_emb_PC1', 'traj_emb_PC2', 'local_emb_PC1', 'local_emb_PC2', 'morph_emb_PC1', 'morph_emb_PC2'], 2)):
# #
# #     draw_umap_space(df, path, file_name='%s_%s_type'%(pair[0], pair[1]), condition_name='Type', label_name='Type',
# #                     colors = ('#6699CC', '#888888', ), dot_size=0.07, x_name=pair[0], y_name=pair[1])
# #
# # from itertools import combinations
# # for pair in list(combinations(['traj_emb_PC1', 'traj_emb_PC2', 'local_emb_PC1', 'local_emb_PC2', 'morph_emb_PC1', 'morph_emb_PC2'], 2)):
# #
# #     draw_contour(df, path, file_name='contour_%s_%s'%(pair[0], pair[1]), condition_name='Type',
# #                  colors=('#6699CC', '#888888',), x_name=pair[0], y_name=pair[1], bin_num=50, num_contours=5)
#
# fig, ax = plt.subplots()
# grid = sns.PairGrid(data=df, vars=['perturbed_traj_emb_PC1', 'perturbed_traj_emb_PC2',
#                                    'perturbed_local_emb_PC1', 'perturbed_local_emb_PC2',
#                                    'perturbed_morph_emb_PC1', 'perturbed_morph_emb_PC2'],
#                     height=6, aspect=1, hue='Type', hue_order=['Frail', 'Old'], palette=('#fdc086', '#beaed4'), diag_sharey=False, despine=True, corner=True,)
# grid.map_upper(sns.kdeplot,  alpha=0.8, linewidths=1.5,
#                     fill=True, legend=False, common_norm=False, thresh=0.2)
# #grid = grid.map_upper(corr)
# grid.map_lower(sns.kdeplot, alpha=0.8, linewidths=1.5,
#                     fill=True, legend=False, common_norm=False, thresh=0.2)
# grid.map_diag(sns.kdeplot, alpha=0.8, linewidths=1.5,
#                     fill=True, legend=False, common_norm=False, thresh=0.2);
#
# plt.tick_params(left=False, right=False, labelleft=False,
#             labelbottom=False, bottom=False)
#
# plt.tight_layout()
#
# plt.savefig(path + '/post perturbation pair plot all embedding.png', dpi=300)
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/post perturbation pair plot all embedding.svg')
# plt.clf()
# plt.close()
#
# # condition_name='Type'
# # x_name='PC1'
# # y_name='PC2'
# # colors = ('#CC6677', '#6699CC', '#888888')
# # df = df_traj[df_traj['Label']=='Data']
# # file_name='traj_space_type'
# # dot_size=0.07,
# #
# # cmap = ListedColormap(colors[:pd.unique(df[condition_name]).shape[0]])
# #
# # xmin = math.floor(df[x_name].min()) - 1
# # xmax = math.ceil(df[x_name].max()) + 1
# # ymin = math.floor(df[y_name].min()) - 1
# # ymax = math.ceil(df[y_name].max()) + 1
# #
# # font = {'family': 'arial',
# #             'weight': 'normal',
# #             'size': 8}
# # matplotlib.rc('font', **font)
# # matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
# # matplotlib.rcParams['lines.linewidth'] = 1
# #
# # fig, ax = plt.subplots(figsize=(2, 2))
# # #plt.figure(figsize=(15, 10))
# # scatter = ax.scatter(df[x_name], df[y_name],
# #                       c=df[condition_name].replace(list(pd.unique(df[condition_name])),
# #                                                             [i for i in range(
# #                                                                 pd.unique(df[condition_name]).shape[0])]),
# #                       # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
# #                       s=dot_size, label=df[condition_name],
# #                       cmap=cmap)
# # vectors = df_traj[df_traj['Label']=='Vector'].reset_index(drop=True)[['PC1', 'PC2']].values
# # plt.plot([0, 0], vectors[0], color='black')
# # plt.plot([0, 0], vectors[1], color='blue')
# # plt.plot([0, 0], vectors[2], color='red')
# # plt.plot([0, 0], vectors[3], color='green')
# #
# # plt.xlim(xmin, xmax)
# # plt.ylim(ymin, ymax)
# #
# # format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
# # handles, labels = scatter.legend_elements(num=None)
# # plt.legend(handles=handles, labels=list(pd.unique(df[condition_name])),
# #            bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
# #            fontsize=3, frameon=False, markerscale=0.3)
# #
# # # bbox_to_anchor is position of labels (x, y) (increasing x moves right, increasing y moves top)
# # # frameon=False removes bounding box around label
# # # font size adjust size of letter
# # # markerscale adjust size of marker
# #
# # plt.savefig(path + '%s.png' % file_name, dpi=300)
# #
# # if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
# #     os.makedirs(path + 'svg/')
# # plt.savefig(path + 'svg/%s.svg' % file_name)
# # plt.clf()
# # plt.close()
#
#
# #################################### Perturbation Embedding space ####################################
# path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure5. Classification\frail vs non frail\\'
# # df_traj = pd.read_csv(path+'traj_emb.csv')
# # df_local = pd.read_csv(path+'local_emb.csv')
# # df_morph = pd.read_csv(path+'morph_emb.csv')
#
# pcss = pd.DataFrame()
# motility_datas = pd.DataFrame()
# variances = []
# for name in ['traj_emb', 'local_emb', 'morph_emb']:
#     df_temp = pd.read_parquet(path + '%s.parquet' % name)
#     df_temp = df_temp.iloc[-4:, :]
#     motility_data = df_temp.iloc[:, :64]
#     print(df_temp.shape)
#     scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#     # aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
#     motility_data_scaled = pd.DataFrame(scaler.fit_transform(motility_data), columns=motility_data.columns)
#     from sklearn.decomposition import PCA
#
#     pca = PCA(2)
#     pcs = pca.fit_transform(motility_data_scaled)
#     pcs = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
#
#     variance = pd.DataFrame(np.array(
#         [pca.explained_variance_, pca.explained_variance_ratio_,
#          np.cumsum(pca.explained_variance_ratio_)]),
#         index=['eigen value', 'variance', 'cumulative variance'], columns=['PC1', 'PC2'])
#
#     variances.append(variance)
#     for column in pcs.columns:
#         pcs.rename(columns={column: '%s_' % name + column}, inplace=True)
#
#     pcss = pd.concat([pcss, pcs], axis=1)
#
# pcss['Type'] = ['Control', 'DNA', 'IL6', 'LPS']
#
# font = {'family': 'arial',
#         'weight': 'normal',
#         'size': 8}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
# matplotlib.rcParams['lines.linewidth'] = 0.5
#
# # xmin = math.floor(df[x_name].min()) - 1
# # xmax = math.ceil(df[x_name].max()) + 1
# # ymin = math.floor(df[y_name].min()) - 1
# # ymax = math.ceil(df[y_name].max()) + 1
#
# colors = ('#888888', '#CC6677', '#44AA99', '#6699CC')
# cmap = ListedColormap(colors)
#
# for name_idx, name in enumerate(['traj_emb', 'local_emb', 'morph_emb']):
#
#     fig, ax = plt.subplots(figsize=(2, 2))
#     x_name = name + '_PC1'
#     y_name = name + '_PC2'
#
#     condition_name = 'Type'
#     # plt.figure(figsize=(15, 10))
#     scatter = ax.scatter(pcss[x_name], pcss[y_name],
#                          c=pcss[condition_name].replace(list(np.unique(pcss[condition_name])),
#                                                         [i for i in range(np.unique(pcss[condition_name]).shape[0])]),
#                          # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
#                          s=10, label=pcss[condition_name], marker='o',
#                          cmap=cmap)
#     for idx, (pc1, pc2) in enumerate(zip(pcss[x_name], pcss[y_name])):
#         ax.plot([0, pc1], [0, pc2], '-', color=colors[idx], linewidth=0.7)
#
#     # plt.xlim(xmin, xmax)
#     # plt.ylim(ymin, ymax)
#
#     plt.axvline(0, color='0.2', linewidth=0.3, zorder=3, linestyle='--')
#     plt.axhline(0, color='0.2', linewidth=0.3, zorder=3, linestyle='--')
#     [x.set_linewidth(0.5) for x in ax.spines.values()]
#
#     variance = variances[name_idx]
#     xlabel = 'PC1(' + str(round(variance['PC1'][1] * 100, ndigits=1)) + '%)'
#     ylabel = 'PC2(' + str(round(variance['PC2'][1] * 100, ndigits=1)) + '%)'
#     # ax.set_xlabel(xlabel, labelpad=5, fontsize=8)
#     # ax.set_ylabel(ylabel, labelpad=5, fontsize=8)
#     plt.xlabel(xlabel, fontsize=8)
#     plt.ylabel(ylabel, fontsize=8)
#     # sns.despine()
#     # plt.axis('off')
#     # format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
#     handles, labels = scatter.legend_elements(num=None)
#     # plt.legend(handles=handles, labels=list(np.unique(pcss[condition_name])),
#     #            bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
#     #            fontsize=3, frameon=False, markerscale=0.3)
#
#     plt.savefig(path + 'perturbation embedding %s.png' % name, dpi=300)
#
#     if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'svg/')
#     plt.savefig(path + 'svg/perturbation embedding %s.svg' % name)
#     plt.clf()
#     plt.close()
#
# #################################### MLP Classification on Age Types ####################################
#
#
#
# df_data = df_ctrl
#
# X = df_data.iloc[:,2:103].drop(['phi', 'speed_distribution_x', 'speed_distribution_y'],axis=1)
# y = df_data['Type']
#
# from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
# scaler = StandardScaler()  # if not normalize, UMAP space is completely different
# X= pd.DataFrame(scaler.fit_transform( X ), columns=X.columns)
#
# print('Classes:', pd.unique(y))
# y = y.replace(list(pd.unique(y)), [i for i in range(pd.unique(y).shape[0])])
# y = np.array(y)
# print('Number of classes:', np.unique(y).size)
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
#
# import tensorflow as tf
# from tensorflow.keras import models, layers, regularizers
#
# tf.keras.backend.clear_session()  # clear the TF session and reset the parameters
#
# inp = layers.Input(shape=(98))
# d1 = layers.Dense(50, activation='relu')(inp)
# d2 = layers.Dense(40, activation='relu')(d1)
# d3 = layers.Dense(30, activation='relu')(d2)
# #d4 = layers.Dense(60, activation='relu')(d3)
# #d5 = layers.Dense(50, activation='relu')(d4)
# #d6 = layers.Dense(30, activation='relu')(d5)
# out = layers.Dense(3, activation='softmax')(d3)
#
# model = models.Model(inputs=inp, outputs=out)
# model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
# model.summary()
#
# result = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.1, shuffle=True)
#
# ########################### Plot errors of model  ################################
# fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
# t = fig.suptitle('Performance', fontsize=12)
# fig.subplots_adjust(top=0.85,wspace=0.3)
#
# max_epoch = len(result.history['accuracy']) # 25(epoch 수)
# epoch_list = list(range(1,max_epoch+1)) # range(1,26) = 1~25
#
# ax1.plot(epoch_list, result.history['accuracy'], label = 'training accuracy')
# ax1.plot(epoch_list, result.history['val_accuracy'], label = 'validation accuracy')
# ax1.set_xticks(np.arange(1, max_epoch, 1000))
# ax1.set_xlabel('epoch')
# ax1.set_ylabel('accuracy')
# ax1.set_title('accuracy test')
# ax1.legend(loc='best')
#
# ax2.plot(epoch_list, result.history['loss'], label = 'training loss')
# ax2.plot(epoch_list, result.history['val_loss'], label = 'validation loss')
# ax2.set_xticks(np.arange(1, max_epoch, 1000))
# ax2.set_xlabel('epoch')
# ax2.set_ylabel('loss')
# ax2.set_title('loss test')
# ax2.legend(loc='best')
# # training accuracy(training data) = 매우 높아짐, but validation accuracy(test data) = 높지않음  ------> Overfitting
# # 이 때는 epoch number 증가, training images 증가 필요
# plt.show()
# plt.clf()
# plt.close()
#
# ###########################################################
# y_pred = model.predict(X_test)
# y_class = np.argmax(y_pred, axis=1)
# accuracy = np.sum(y_test == y_class) / y_test.size
#
#
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_class)
# norm_cm = cm / np.sum(cm, axis=1)[:, np.newaxis]
#
# import seaborn as sns
# plt.figure(figsize=(10,6))
# sns.heatmap(norm_cm, annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
#
# plt.savefig(path + 'IL6 confision matrix.png', dpi=300, bbox_inches='tight')
# plt.clf()
# plt.close()
#
#
# #################################### XGBoost Classification on Age Types ####################################
# df_data = df_nonfrail
#
# X = df_data.iloc[:,2:103].drop(['phi', 'speed_distribution_x', 'speed_distribution_y'],axis=1)
# y = df_data['Type']
#
# from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
# scaler = StandardScaler()  # if not normalize, UMAP space is completely different
# X= pd.DataFrame(scaler.fit_transform( X ), columns=X.columns)
#
# print('Classes:', pd.unique(y))
# y = y.replace(list(pd.unique(y)), [i for i in range(pd.unique(y).shape[0])])
# y = np.array(y)
# print('Number of classes:', np.unique(y).size)
#
#
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.inspection import permutation_importance
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
#
# model = xgb.XGBClassifier(n_estimators=500, max_depth=5, eta=0.05)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = np.sum(y_test == y_pred) / y_test.size
#
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# norm_cm = cm / np.sum(cm, axis=1)[:, np.newaxis]
#
# import seaborn as sns
# plt.figure(figsize=(10,6))
# sns.heatmap(norm_cm, annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
#
# plt.savefig(path + 'XGBoost All confision matrix.png', dpi=300, bbox_inches='tight')
# plt.clf()
# plt.close()


