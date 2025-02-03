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
"""Generates Data for Figure 1."""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast
from dnn.autoencoders import set_duration_for_autoencoder
from dnn.classification import Temporal_Conv1D_2D_classifier, Res_Conv1D_LSTM_classifier
from utils.traj_utils import *
from tensorflow.keras import models, optimizers
from sklearn.model_selection import train_test_split

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
df = pd.read_parquet(path+'motility_features_nan_removed.parquet')
df_duration = pd.read_parquet(path + 'traj_duration_nan_removed.parquet')
duration=31

#df_duration = df_duration[df_duration['type']!='M5CARHD'].reset_index(drop=True)

traj_list, trajectories_array, trajectories = to_timeseries_fast(df_duration, duration=duration, feature_name=['reg_x', 'reg_y'])
trajectories, new_duration = set_duration_for_autoencoder(trajectories, duration=duration, dim=2)
#rotated_trajectories = register_traj_disp_reflection(trajectories)
rotated_trajectories = dict_to_array(trajectories)
X = rotated_trajectories
X = X+1

# y = df['type']
# print('Classes:', pd.unique(y))
# y = y.replace(list(pd.unique(y)), [i for i in range(pd.unique(y).shape[0])])
# y = np.array(y)
# print('Number of classes:', np.unique(y).size)

# lung cancer
# df = df[df['type']!='M5CARHD'].reset_index(drop=True)
# conditions = [
#     (df['type'] == 'M5CAR'),
#     #(df['type'] == 'M5CARHD'),
#     (df['type'] == 'V5'),
#     (df['type'] == 'VR5aIL5'),
#     (df['type'] == 'VR5aIL8'),
#     (df['type'] == 'VR5aTNFa'),
#     (df['type'] == 'Vgsig'),
# ]
#
# #values = [0, 2, 2, 1, 0, 1]
# values = [0, 0, 1, 0, 0, 0]
#
# df['killing'] = np.select(conditions, values, default='').astype(int)
y = df['lung_phenotype']

for i in np.unique(df['lung_phenotype']):
    print(i, df[df['lung_phenotype']==i].shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)


model = Temporal_Conv1D_2D_classifier(duration=new_duration, coor_dim=2, n_classes=np.unique(y_train).size)
#model = Res_Conv1D_LSTM_classifier(duration=new_duration, coor_dim=2, n_classes=np.unique(y_train).size)

from tensorflow.keras import callbacks
stop_early = callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001)

result = model.fit(X_train, y_train, batch_size=64, epochs=500, verbose=1, validation_split=0.1, shuffle=True, callbacks=[stop_early])

########################### Saving & Loading model ################################
model.save('saved_model/Temporal_Conv1D_2D_classifier_1000epochs_CART')
#ls saved_model

model = models.load_model('saved_model/Temporal_Conv1D_2D_10000epochs_CART', compile = False)
model.compile(loss='mse', optimizer=optimizers.Adadelta(learning_rate=0.1))
result = model.fit(X_train, X_train, batch_size=512, epochs=1, verbose=1, validation_split=0.1, shuffle=True)
####################################################################################

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
# #plt.savefig(path+'loss_10000.png')
# plt.show()

###########################################################
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\analysis\\'

y_pred = model.predict(X_test)
draw_confusion_matrix(y_pred, y_test, y_test, path, figsize=(4,4), file_name='confusion matrix', vmax=0.9)

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

if y_pred.shape[1]>=3:
    y_class = np.argmax(y_pred, axis=1)

else:
    y_class = np.array([1 if prob >= 0.5 else 0 for prob in np.ravel(y_pred)])


f1 = f1_score(y_test, y_class)
acs = accuracy_score(y_test, y_class)
ps = precision_score(y_test, y_class)
rs = recall_score(y_test, y_class)

print(f1, acs, ps, rs)

dict_datasets = {'F1 score': np.array(f1), 'Precision': np.array(ps), 'Recall': np.array(rs),}
#dict_datasets = {'scTRAIT': np.array(acs), 'Vanilla traj': np.array(acs2), 'Vanilla morpho': np.array(acs3),}
draw_custom_bar_plot(dict_datasets, path, file_name='f1 scores', colors=('#888888', '#888888', '#888888'), vmax=1, strip_plot=False, test='t-test', pvalue=False, figsize=(2,2))


#################################### ROC curve and AUC ####################################
from sklearn.metrics import roc_curve
fpr1, tpr1, thresh1 = roc_curve(y_test, y_pred, pos_label=1)

auc_score = roc_auc_score(y_test, y_pred)


font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(4,4))

#plt.style.use('seaborn') # plot
plt.plot(fpr1, tpr1, linestyle='-.', color='purple', label='scTRAIT (AUC = %.2f)'%auc_score,  linewidth=3)
# plt.plot(fpr2, tpr2, linestyle='-.', color='red', label='Vanilla traj model (AUC = %.2f)'%auc_score2,  linewidth=3)
# plt.plot(fpr3, tpr3, linestyle='-.', color='blue', label='Vanilla morph model (AUC = %.2f)'%auc_score3,  linewidth=3)
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


#################################### Classification based on motility features ####################################
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
df = pd.read_parquet(path+'motility_features_nan_removed.parquet')
#df = pd.read_parquet(path+'all_features.parquet')

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\analysis\\'
color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')


df.columns.get_loc('inst_angle_pulseindicator')
feature_list = list(df.columns[2:90].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']))

corrs = {}
ps = {}
for feature in feature_list:
    r, p = scipy.stats.pearsonr(df['pancreatic_effect'], df[feature])
    corrs[feature] = r
    ps[feature] = p


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

X = df.iloc[:,2:90].drop(['speed_distribution_x', 'speed_distribution_y'], axis=1)
y = df['lung_phenotype']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

model = xgb.XGBClassifier(n_estimators=500, max_depth=5, eta=0.05)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

np.sum(y_test==y_pred)/y_test.size
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(30, 12))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.show()

perm_importance = permutation_importance(model, np.ascontiguousarray(X_test), y_test, n_repeats=10, random_state=1066)
sorted_idx = perm_importance.importances_mean.argsort()
fig = plt.figure(figsize=(30, 12))
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.show()

#################################### XGBoost Classification ####################################
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
df = pd.read_parquet(path+'motility_features_nan_removed.parquet')
#df = pd.read_parquet(path+'all_features.parquet')

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\analysis\\'
color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')

# df = df[df['type']!='M5CARHD'].reset_index(drop=True)
# conditions = [
#     (df['type'] == 'M5CAR'),
#     (df['type'] == 'V5'),
#     (df['type'] == 'VR5aIL5'),
#     (df['type'] == 'VR5aIL8'),
#     (df['type'] == 'VR5aTNFa'),
#     (df['type'] == 'Vgsig'),
# ]
#
# #values = [0, 1.8, 1.6, 0.1, 0, 0.6]
# values = [0, 1, 1, 0, 0, 0]
#
# df['killing'] = np.select(conditions, values, default='').astype(float)
#
# df.columns.get_loc('displ_autocorr_y_3')
df.columns.get_loc('inst_angle_pulseindicator')

X = df.iloc[:,2:90].drop(['phi', 'speed_distribution_x', 'speed_distribution_y'], axis=1)
y = df['lung_phenotype']

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
X= pd.DataFrame(scaler.fit_transform( X ), columns=X.columns)

print('Classes:', pd.unique(y))
y = y.replace(list(pd.unique(y)), [i for i in range(pd.unique(y).shape[0])])
y = np.array(y)
print('Number of classes:', np.unique(y).size)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


param_grid = {'gamma': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4, 200],
              #'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.300000012, 0.4, 0.5, 0.6, 0.7],
              'max_depth': [5,10,15,20],
              'n_estimators': [50,100,150,200,500],
              'reg_alpha': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
              'reg_lambda': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200]}

gs = GridSearchCV(xgb.XGBClassifier(), param_grid, verbose = 3, cv=3, n_jobs = -1)
#model = xgb.XGBClassifier(n_estimators=500, max_depth=5, eta=0.05)
g_res = gs.fit(X_train, y_train)
print(g_res.best_score_)
print(g_res.best_params_)

model = xgb.XGBClassifier(n_estimators=500, max_depth=5, eta=0.05)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = np.sum(y_test == y_pred) / y_test.size

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
plt.clf()
plt.close()

#################################### DNN Classification on subtypes ####################################
X = df.iloc[:,2:75]
y = df['type']

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
X= pd.DataFrame(scaler.fit_transform( X ), columns=X.columns)

print('Classes:', pd.unique(y))
y = y.replace(list(pd.unique(y)), [i for i in range(pd.unique(y).shape[0])])
y = np.array(y)
print('Number of classes:', np.unique(y).size)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers

tf.keras.backend.clear_session()  # clear the TF session and reset the parameters

inp = layers.Input(shape=(73))
d1 = layers.Dense(50, activation='relu')(inp)
d2 = layers.Dense(40, activation='relu')(d1)
d3 = layers.Dense(30, activation='relu')(d2)
#d4 = layers.Dense(60, activation='relu')(d3)
#d5 = layers.Dense(50, activation='relu')(d4)
#d6 = layers.Dense(30, activation='relu')(d5)
out = layers.Dense(6, activation='softmax')(d3)

model = models.Model(inputs=inp, outputs=out)
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()

result = model.fit(X_train, y_train, batch_size=32, epochs=500, validation_split=0.1, shuffle=True)

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
plt.show()
plt.clf()
plt.close()

###########################################################
y_pred = model.predict(X_test)
y_class = np.argmax(y_pred, axis=1)
accuracy = np.sum(y_test == y_class) / y_test.size


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_class)

import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()
plt.clf()
plt.close()





