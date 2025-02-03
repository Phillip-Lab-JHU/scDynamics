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

"""Generates Data for Figure 6_sc single-cell age regression."""
import pandas as pd
from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast
from dnn.autoencoders import set_duration_for_autoencoder
import tensorflow as tf
from dnn.classification import scTRAIT, Temporal_Conv1D_2D_classifier

#################################### Young single-cell age regression ####################################


#################################### Training data preparation ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df = df[df['Type']=='Young'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']=='Young'].reset_index(drop=True)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\\'

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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_covar = le.fit_transform( np.array(df['Condition']) )
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)



y = np.array(df['Age'])
print('Number of classes:', np.unique(y).size)

from sklearn.model_selection import train_test_split
X_traj_train, X_traj_test, X_local_train, X_local_test, X_morph_train, X_morph_test, \
X_covar_train, X_covar_test, y_train, y_test = train_test_split(X_traj, X_local, X_morph, X_covar, y, test_size=0.2, random_state=0, stratify=y)
print(X_traj_train.shape, X_traj_test.shape, X_local_train.shape, X_local_test.shape,
      X_morph_train.shape, X_morph_test.shape, X_covar_train.shape, X_covar_test.shape, y_train.shape, y_test.shape)


#################################### Model training ####################################

from tensorflow.keras import callbacks
stop_early = callbacks.EarlyStopping(monitor='loss', patience=30, min_delta=0.0001)
# checkpointer = callbacks.ModelCheckpoint('saved_model/monocyte_project_v2/%s'%model_name, monitor='accuracy', verbose=1,
#                 save_weights_only=False, save_freq='epoch', every_n_epochs=10, mode='auto', save_best_only=True)  # Save model every epoch

from dnn.classification import scTRAIT
sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=0)
result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=1000,
                     verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
sctrait.save('saved_model/monocyte_project_v2/regression scTRAIT young', save_format='tf')
#3 class scTRAIT 2
### model2: Vanilla trajectory ###
vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=0)
result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vt.save('saved_model/monocyte_project_v2/regression vanilla traj young', save_format='tf')

### model3: Vanilla morphology ###
vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=0)
result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vm.save('saved_model/monocyte_project_v2/regression vanilla morpho young', save_format='tf')

#################################### Load trained model ####################################
sctrait = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression scTRAIT young', compile=True)
# result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=2,
#                      verbose=1, validation_split=0.1, shuffle=True,)

vt = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression vanilla traj young', compile=True)
# result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

vm = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression vanilla morpho young', compile=True)
# result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

#################################### Model prediction ####################################

y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#y_pred = sctrait.predict([X_traj_test, X_local_test, X_covar_test])
y_pred2 = vt.predict([X_traj_test])
y_pred3 = vm.predict([X_morph_test])

#################################### Graph ####################################
model = 'VM'

if model == 'scTRAIT':
    prediction = y_pred.flatten()
elif model == 'VT':
    prediction = y_pred2.flatten()
elif model == 'VM':
    prediction = y_pred3.flatten()

font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2
fig, ax = plt.subplots(figsize=(4, 4))

#sns.scatterplot(x=y_test, y=prediction, lw=2.5, s=5, palette=('black',), ax=ax)
sns.regplot(x=y_test, y=prediction, scatter=False, #scatter_kws={"color":"black", "alpha":0.7, 's':10},
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)
sns.kdeplot(x=y_test, y=prediction, alpha=1, ax=ax, zorder=1, linewidths=1.5, color ='#7fc97f',
                    fill=False,  common_norm=False, thresh=0.15,)

file_name = 'Young Age Regression_%s' % (model)

#ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )

mae = np.mean([ abs(i - j) for i, j in zip(y_test, prediction) ])

plt.text(0.2, 1.1, "MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
         fontsize=16, fontdict={'weight': 'bold'}, color="black")

r, p = scipy.stats.pearsonr(y_test, prediction)
plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="black")
plt.text(0.1, 0.9, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
         fontsize=14, fontdict={'weight': 'bold'}, color="black")

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

# legend = plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
#            loc='best')
# legend.remove()
plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()


#################################### Old single-cell age regression ####################################


#################################### Training data preparation ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df = df[df['Type']=='Old'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']=='Old'].reset_index(drop=True)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\\'

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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_covar = le.fit_transform( np.array(df['Condition']) )
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)



y = np.array(df['Age'])
print('Number of classes:', np.unique(y).size)

from sklearn.model_selection import train_test_split
X_traj_train, X_traj_test, X_local_train, X_local_test, X_morph_train, X_morph_test, \
X_covar_train, X_covar_test, y_train, y_test = train_test_split(X_traj, X_local, X_morph, X_covar, y, test_size=0.2, random_state=0, stratify=y)
print(X_traj_train.shape, X_traj_test.shape, X_local_train.shape, X_local_test.shape,
      X_morph_train.shape, X_morph_test.shape, X_covar_train.shape, X_covar_test.shape, y_train.shape, y_test.shape)


#################################### Model training ####################################

from tensorflow.keras import callbacks
stop_early = callbacks.EarlyStopping(monitor='loss', patience=30, min_delta=0.0001)
# checkpointer = callbacks.ModelCheckpoint('saved_model/monocyte_project_v2/%s'%model_name, monitor='accuracy', verbose=1,
#                 save_weights_only=False, save_freq='epoch', every_n_epochs=10, mode='auto', save_best_only=True)  # Save model every epoch

from dnn.classification import scTRAIT
sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=0)
result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=1000,
                     verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
sctrait.save('saved_model/monocyte_project_v2/regression scTRAIT old', save_format='tf')
#3 class scTRAIT 2
### model2: Vanilla trajectory ###
vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=0)
result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vt.save('saved_model/monocyte_project_v2/regression vanilla traj old', save_format='tf')

### model3: Vanilla morphology ###
vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=0)
result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vm.save('saved_model/monocyte_project_v2/regression vanilla morpho old', save_format='tf')

#################################### Load trained model ####################################
sctrait = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression scTRAIT old', compile=True)
# result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=2,
#                      verbose=1, validation_split=0.1, shuffle=True,)

vt = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression vanilla traj old', compile=True)
# result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

vm = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression vanilla morpho old', compile=True)
# result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

#################################### Model prediction ####################################

y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#y_pred = sctrait.predict([X_traj_test, X_local_test, X_covar_test])
y_pred2 = vt.predict([X_traj_test])
y_pred3 = vm.predict([X_morph_test])

#################################### Graph ####################################

model = 'scTRAIT'

if model == 'scTRAIT':
    prediction = y_pred.flatten()
elif model == 'VT':
    prediction = y_pred2.flatten()
elif model == 'VM':
    prediction = y_pred3.flatten()

font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(4, 4))
#sns.scatterplot(x=y_test, y=prediction, lw=2.5, s=5, alpha=0.5, palette=('black',), ax=ax)
sns.regplot(x=y_test, y=prediction, scatter=False, #scatter_kws={"color":"black", "alpha":0.7, 's':10},
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)
sns.kdeplot(x=y_test, y=prediction, alpha=1, ax=ax, zorder=1, linewidths=1.5, color ='#beaed4',
                    fill=False,  common_norm=False, thresh=0.1,)

file_name = 'Old Age Regression_%s' % (model)

#ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )

mae = np.mean([ abs(i - j) for i, j in zip(y_test, prediction) ])

plt.text(0.2, 1.1, "MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
         fontsize=16, fontdict={'weight': 'bold'}, color="black")

r, p = scipy.stats.pearsonr(y_test, prediction)
plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="black")
plt.text(0.1, 0.9, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
         fontsize=14, fontdict={'weight': 'bold'}, color="black")

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

# legend = plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
#            loc='best')
# legend.remove()
plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()


#################################### Predict Frail single-cell age regression trained on old ####################################

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df_prefrail = df[df['Type']=='Prefrail'].reset_index(drop=True)
df_duration_prefrail = df_duration[df_duration['Type']=='Prefrail'].reset_index(drop=True)

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df_frail = df[df['Type']=='Frail'].reset_index(drop=True)
df_duration_frail = df_duration[df_duration['Type']=='Frail'].reset_index(drop=True)

df = df[df['Type']=='Old'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']=='Old'].reset_index(drop=True)

############# testing set #############
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
X_traj_test = rotated_trajectories

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
X_local_test = rotated_trajectories

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
X_morph_test = rotated_trajectories

le = LabelEncoder()
le.fit(np.array(df['Condition']))
X_covar_test = le.transform(np.array(df_frail['Condition']))
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
y_test_frail = np.array(df_frail['Age'])

y_pred_frail = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
y_pred2_frail = vt.predict([X_traj_test])
y_pred3_frail = vm.predict([X_morph_test])


#################################### Graph ####################################
model = 'scTRAIT'

if model == 'scTRAIT':
    prediction_frail = y_pred_frail.flatten()
elif model == 'VT':
    prediction_frail = y_pred2_frail.flatten()
elif model == 'VM':
    prediction_frail = y_pred3_frail.flatten()

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\\'

font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(4, 4))
#sns.scatterplot(x=y_test, y=prediction, lw=2.5, s=5, alpha=0.5, palette=('black',), ax=ax)
sns.regplot(x=y_test_frail, y=prediction_frail, scatter=False, #scatter_kws={"color":"black", "alpha":0.7, 's':10},
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)
sns.regplot(x=y_test, y=prediction, scatter=False, #scatter_kws={"color":"black", "alpha":0.7, 's':10},
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)
sns.kdeplot(x=y_test_frail, y=prediction_frail, alpha=1, ax=ax, zorder=1, linewidths=1.5, color ='#fdc086',
                    fill=False,  common_norm=False, thresh=0.1,)
sns.kdeplot(x=y_test, y=prediction, alpha=1, ax=ax, zorder=1, linewidths=1.5, color ='#beaed4',
                    fill=False,  common_norm=False, thresh=0.1,)

file_name = 'Overlayed Trained on Old Frail Age Regression_%s' % (model)

#ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )

mae = np.mean([ abs(i - j) for i, j in zip(y_test_frail, prediction_frail) ])
r, p = scipy.stats.pearsonr(y_test_frail, prediction_frail)

plt.text(0.1, 0.95,"MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="#fdc086")
plt.text(0.1, 0.88, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="#fdc086")
# plt.text(0.1, 0.9, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
#          fontsize=14, fontdict={'weight': 'bold'}, color="black")

mae = np.mean([ abs(i - j) for i, j in zip(y_test, prediction) ])
r, p = scipy.stats.pearsonr(y_test, prediction)
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

# legend = plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
#            loc='best')
# legend.remove()
plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()

#################################### Predict Prefrail single-cell age regression trained on old ####################################

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df_prefrail = df[df['Type']=='Prefrail'].reset_index(drop=True)
df_duration_prefrail = df_duration[df_duration['Type']=='Prefrail'].reset_index(drop=True)


df = df[df['Type']=='Old'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']=='Old'].reset_index(drop=True)

############# testing set #############
traj_scale_data = df_duration.loc[:, ['reg_x', 'reg_y']]
traj_scaler = StandardScaler()
traj_scaler.fit(traj_scale_data)
traj_scale_data = df_duration_prefrail.loc[:, ['reg_x', 'reg_y']]
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
local_scale_data = df_duration_prefrail.loc[:, local_features]
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
scale_data = df_duration_prefrail.loc[:, morphology_features]
scale_data = pd.DataFrame(scaler.transform(scale_data), columns=morphology_features)

traj_list, trajectories_array, trajectories = to_timeseries_fast(scale_data, duration=duration,
                                                                 feature_name=morphology_features)
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=8)
# rotated_trajectories = register_traj_disp_reflection(trajectories)
rotated_trajectories = dict_to_array(trajectories)
X_morph_test = rotated_trajectories

le = LabelEncoder()
le.fit(np.array(df['Condition']))
X_covar_test = le.transform(np.array(df_prefrail['Condition']))
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
y_test_frail = np.array(df_prefrail['Age'])

y_pred_frail = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
y_pred2_frail = vt.predict([X_traj_test])
y_pred3_frail = vm.predict([X_morph_test])


#################################### Graph ####################################
model = 'scTRAIT'

if model == 'scTRAIT':
    prediction_frail = y_pred_frail.flatten()
elif model == 'VT':
    prediction_frail = y_pred2_frail.flatten()
elif model == 'VM':
    prediction_frail = y_pred3_frail.flatten()

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\\'

font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(4, 4))
#sns.scatterplot(x=y_test, y=prediction, lw=2.5, s=5, alpha=0.5, palette=('black',), ax=ax)
sns.regplot(x=y_test_frail, y=prediction_frail, scatter=False, #scatter_kws={"color":"black", "alpha":0.7, 's':10},
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)
sns.regplot(x=y_test, y=prediction, scatter=False, #scatter_kws={"color":"black", "alpha":0.7, 's':10},
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)
sns.kdeplot(x=y_test_frail, y=prediction_frail, alpha=1, ax=ax, zorder=1, linewidths=1.5, color ='#fdc086',
                    fill=False,  common_norm=False, thresh=0.1,)
sns.kdeplot(x=y_test, y=prediction, alpha=1, ax=ax, zorder=1, linewidths=1.5, color ='#beaed4',
                    fill=False,  common_norm=False, thresh=0.1,)

file_name = 'Overlayed Trained on Old Prefrail Age Regression_%s' % (model)

#ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )

mae = np.mean([ abs(i - j) for i, j in zip(y_test_frail, prediction_frail) ])
r, p = scipy.stats.pearsonr(y_test_frail, prediction_frail)

plt.text(0.1, 0.95,"MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="#fdc086")
plt.text(0.1, 0.88, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="#fdc086")
# plt.text(0.1, 0.9, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
#          fontsize=14, fontdict={'weight': 'bold'}, color="black")

mae = np.mean([ abs(i - j) for i, j in zip(y_test, prediction) ])
r, p = scipy.stats.pearsonr(y_test, prediction)
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

# legend = plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
#            loc='best')
# legend.remove()
plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()


#################################### Frail single-cell age regression ####################################


#################################### Training data preparation ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df = df[df['Type']=='Frail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']=='Frail'].reset_index(drop=True)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\\'

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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_covar = le.fit_transform( np.array(df['Condition']) )
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)



y = np.array(df['Age'])
print('Number of classes:', np.unique(y).size)

from sklearn.model_selection import train_test_split
X_traj_train, X_traj_test, X_local_train, X_local_test, X_morph_train, X_morph_test, \
X_covar_train, X_covar_test, y_train, y_test = train_test_split(X_traj, X_local, X_morph, X_covar, y, test_size=0.2, random_state=0, stratify=y)
print(X_traj_train.shape, X_traj_test.shape, X_local_train.shape, X_local_test.shape,
      X_morph_train.shape, X_morph_test.shape, X_covar_train.shape, X_covar_test.shape, y_train.shape, y_test.shape)


#################################### Model training ####################################

from tensorflow.keras import callbacks
stop_early = callbacks.EarlyStopping(monitor='loss', patience=30, min_delta=0.0001)
# checkpointer = callbacks.ModelCheckpoint('saved_model/monocyte_project_v2/%s'%model_name, monitor='accuracy', verbose=1,
#                 save_weights_only=False, save_freq='epoch', every_n_epochs=10, mode='auto', save_best_only=True)  # Save model every epoch

from dnn.classification import scTRAIT
sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=0)
result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=1000,
                     verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
sctrait.save('saved_model/monocyte_project_v2/regression scTRAIT frail', save_format='tf')
#3 class scTRAIT 2
### model2: Vanilla trajectory ###
vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=0)
result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vt.save('saved_model/monocyte_project_v2/regression vanilla traj frail', save_format='tf')

### model3: Vanilla morphology ###
vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=0)
result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vm.save('saved_model/monocyte_project_v2/regression vanilla morpho frail', save_format='tf')

#################################### Load trained model ####################################
sctrait = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression scTRAIT frail', compile=True)
# result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=2,
#                      verbose=1, validation_split=0.1, shuffle=True,)

vt = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression vanilla traj frail', compile=True)
# result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

vm = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression vanilla morpho frail', compile=True)
# result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

#################################### Model prediction ####################################

y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#y_pred = sctrait.predict([X_traj_test, X_local_test, X_covar_test])
y_pred2 = vt.predict([X_traj_test])
y_pred3 = vm.predict([X_morph_test])

#################################### Graph ####################################

model = 'VM'

if model == 'scTRAIT':
    prediction = y_pred.flatten()
elif model == 'VT':
    prediction = y_pred2.flatten()
elif model == 'VM':
    prediction = y_pred3.flatten()

font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(4, 4))
# sns.scatterplot(x=y_test, y=prediction, lw=2.5, s=5, alpha=0.5, palette=('black',), ax=ax)
sns.regplot(x=y_test, y=prediction, scatter=False, #scatter_kws={"color":"black", "alpha":0.7, 's':10},
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)
sns.kdeplot(x=y_test, y=prediction, alpha=1, ax=ax, zorder=1, linewidths=1.5, color ='#fdc086',
                    fill=False,  common_norm=False, thresh=0.1,)

file_name = 'Frail Age Regression_%s' % (model)

#ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )

mae = np.mean([ abs(i - j) for i, j in zip(y_test, prediction) ])

plt.text(0.2, 1.1, "MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
         fontsize=16, fontdict={'weight': 'bold'}, color="black")

r, p = scipy.stats.pearsonr(y_test, prediction)
plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="black")
plt.text(0.1, 0.9, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
         fontsize=14, fontdict={'weight': 'bold'}, color="black")

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

# legend = plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
#            loc='best')
# legend.remove()
plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()



#################################### Nonfrail single-cell age regression ####################################


#################################### Training data preparation ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df = df[df['Type']!='Frail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Frail'].reset_index(drop=True)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure6. Age score\\'

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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_covar = le.fit_transform( np.array(df['Condition']) )
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)



y = np.array(df['Age'])
print('Number of classes:', np.unique(y).size)

from sklearn.model_selection import train_test_split
X_traj_train, X_traj_test, X_local_train, X_local_test, X_morph_train, X_morph_test, \
X_covar_train, X_covar_test, y_train, y_test = train_test_split(X_traj, X_local, X_morph, X_covar, y, test_size=0.2, random_state=0, stratify=y)
print(X_traj_train.shape, X_traj_test.shape, X_local_train.shape, X_local_test.shape,
      X_morph_train.shape, X_morph_test.shape, X_covar_train.shape, X_covar_test.shape, y_train.shape, y_test.shape)


#################################### Model training ####################################

from tensorflow.keras import callbacks
stop_early = callbacks.EarlyStopping(monitor='loss', patience=30, min_delta=0.0001)
# checkpointer = callbacks.ModelCheckpoint('saved_model/monocyte_project_v2/%s'%model_name, monitor='accuracy', verbose=1,
#                 save_weights_only=False, save_freq='epoch', every_n_epochs=10, mode='auto', save_best_only=True)  # Save model every epoch

from dnn.classification import scTRAIT
sctrait = scTRAIT(duration=new_duration, embed_dim=64, n_classes=0)
result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=1000,
                     verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])
sctrait.save('saved_model/monocyte_project_v2/regression scTRAIT nonfrail', save_format='tf')
#3 class scTRAIT 2
### model2: Vanilla trajectory ###
vt = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=0)
result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vt.save('saved_model/monocyte_project_v2/regression vanilla traj nonfrail', save_format='tf')

### model3: Vanilla morphology ###
vm = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=8, n_classes=0)
result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=1000, verbose=1, validation_split=0.1, shuffle=True,
                     callbacks=[stop_early])
vm.save('saved_model/monocyte_project_v2/regression vanilla morpho nonfrail', save_format='tf')

#################################### Load trained model ####################################
sctrait = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression scTRAIT nonfrail', compile=True)
# result = sctrait.fit(x=[X_traj_train, X_local_train, X_morph_train, X_covar_train], y=y_train, batch_size=256, epochs=2,
#                      verbose=1, validation_split=0.1, shuffle=True,)

vt = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression vanilla traj nonfrail', compile=True)
# result2 = vt.fit(x=X_traj_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

vm = tf.keras.models.load_model('saved_model/monocyte_project_v2/regression vanilla morpho nonfrail', compile=True)
# result3 = vm.fit(x=X_morph_train, y=y_train, batch_size=256, epochs=2, verbose=1, validation_split=0.1, shuffle=True)

#################################### Model prediction ####################################

y_pred = sctrait.predict([X_traj_test, X_local_test, X_morph_test, X_covar_test])
#y_pred = sctrait.predict([X_traj_test, X_local_test, X_covar_test])
y_pred2 = vt.predict([X_traj_test])
y_pred3 = vm.predict([X_morph_test])

#################################### Graph ####################################
model = 'VM'

if model == 'scTRAIT':
    prediction = y_pred.flatten()
elif model == 'VT':
    prediction = y_pred2.flatten()
elif model == 'VM':
    prediction = y_pred3.flatten()

font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2
fig, ax = plt.subplots(figsize=(4, 4))

#sns.scatterplot(x=y_test, y=prediction, lw=2.5, s=5, palette=('black',), ax=ax)
sns.regplot(x=y_test, y=prediction, scatter=False, #scatter_kws={"color":"black", "alpha":0.7, 's':10},
            line_kws={"color":"red", 'linewidth':1.5}, ax=ax)
sns.kdeplot(x=y_test, y=prediction, alpha=1, ax=ax, zorder=1, linewidths=1.5, color ='black',
                    fill=False,  common_norm=False, thresh=0.2,)

file_name = 'Nonfrail Age Regression_%s' % (model)

#ax.plot(age_range, age_range, linestyle='--', color='black',  linewidth=1 )

mae = np.mean([ abs(i - j) for i, j in zip(y_test, prediction) ])

plt.text(0.2, 1.1, "MAE = %s years" %str(round(mae, 2)), ha='left', va='top', transform=ax.transAxes,
         fontsize=16, fontdict={'weight': 'bold'}, color="black")

r, p = scipy.stats.pearsonr(y_test, prediction)
plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=14, fontdict={'weight': 'bold'}, color="black")
plt.text(0.1, 0.9, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
         fontsize=14, fontdict={'weight': 'bold'}, color="black")

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

# legend = plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
#            loc='best')
# legend.remove()
plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()



mae = np.mean([ abs(i - j) for i, j in zip(y_test, y_pred.flatten()) ])
mae2 = np.mean([ abs(i - j) for i, j in zip(y_test, y_pred2.flatten()) ])
mae3 = np.mean([ abs(i - j) for i, j in zip(y_test, y_pred3.flatten()) ])

dict_datasets = {'VM': np.array(mae3), 'VT': np.array(mae2), 'scTRAIT': np.array(mae)}
#dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='MAE', colors=('#6699CC', '#CC6677', '#aa4499'), vmax=None,
                     strip_plot=False, test='t-test', pvalue=False, figsize=(1,2))