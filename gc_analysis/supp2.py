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
"""Generates Data for Supplement"""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *

#################################### APRW space ####################################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/'
df = pd.read_parquet(path+'all_features_20.parquet')

aprw_data = df.iloc[:,128:136]

scaler = StandardScaler()  # if not normalize, UMAP space is completely different
aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)


m = Morphodynamics(df, 'umap')
umap = m.get_umap(aprw_data_scaled, 20, 0.5)


df = df.drop(['PC1', 'PC2'], axis=1)
df = pd.concat([df, umap], axis=1)


xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Figure1. Basic gc_analysis - supp/'
draw_umap_space(df, path, file_name='APRW space_tskmeans', condition_name='tskmeans', label_name='pseudo_Label', x_name='PC1', y_name='PC2', dot_size=12,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
draw_umap_space(df, path, file_name='APRW space_Type', condition_name='Type', label_name='pseudo_Label', x_name='PC1', y_name='PC2', dot_size=12,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

feature_list = df.columns[128:136]
draw_space_feature_magnitude(df, path, feature_list, dot_size=12, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, vmax=None)

#################################### Basic motility space ####################################
df.columns.get_loc('displ_autocorr_z_3')
motility_data = df.iloc[:, 136:228].drop(['speed_distribution', 'angle_distribution'], axis=1)

motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)

m = Morphodynamics(df, 'umap')
umap = m.get_umap(motility_data_scaled, 20, 0.5)

df = df.drop(['PC1', 'PC2'], axis=1)
df = pd.concat([df, umap], axis=1)


xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Figure1. Basic gc_analysis - supp/'
draw_umap_space(df, path, file_name='Basic motility space_tskmeans', condition_name='tskmeans', label_name='pseudo_Label', x_name='PC1', y_name='PC2', dot_size=12,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
draw_umap_space(df, path, file_name='Basic motility space_Type', condition_name='Type', label_name='pseudo_Label', x_name='PC1', y_name='PC2', dot_size=12,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

feature_list = df.columns[136:228].drop(['speed_distribution', 'angle_distribution'])
draw_space_feature_magnitude(df, path, feature_list, dot_size=12, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, vmax=None)

################# sanity check ####################
condition_name = 'Type'
cluster_type = 'tskmeans'
condition = 'Exp_group'

for group in list(pd.unique(df[condition])):
    aaa = df[df[condition] == group]
    group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0  # fill na with 0 (np.isnan() returns bool array, which is True whenever nan )

    plt.figure()
    sns.clustermap(group_clone.T, annot=False, cmap='rainbow')
    plt.xlabel('%s' % cluster_type)
    plt.ylabel('%s' % condition_name)
    plt.title('Distribution of phenotypes')

    plt.savefig(path + '(%s)%s_distribution_heatmap.png' % (group, cluster_type))
    colors = ('red', 'green','blue', 'gray', 'cyan', 'violet', 'coral', 'brown', 'powderblue', 'olive',
                      'hotpink', 'indigo', 'bisque', 'lawngreen', 'darksalmon', 'wheat', 'steelblue', 'black', 'lime', 'gold',
                      'orange', 'cornsilk', 'ivory', 'aliceblue', 'peru', 'mistyrose', 'chocolate', 'slategrey', 'cornflowerblue', 'silver')


    for i, cond in enumerate(list(group_clone.columns)):
        plt.figure()
        sns.barplot(x=list(group_clone[cond].index), y=list(group_clone[cond]), color=colors[i])
        plt.xlabel('%s' % cluster_type)
        plt.ylabel('%s distribution' % cond)
        #plt.ylim(0, max(group_clone.max())+1)
        plt.ylim(0,60)
        plt.savefig(path + '(%s)sanity_check_%s.png' % (group, cond))
        plt.close()
        plt.clf()


group_list = pd.unique(df[condition])
cell_types = ['T-cell', 'mt B-cell', 'wt B-cell']

for cell_type in cell_types:
    corrcoef = []
    for group in group_list:
        aaa = df[df[condition] == group]
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
        corrcoef.append(group_clone[cell_type].values)
    corrcoef = np.array([corrcoef]).reshape(3,11)

    plt.figure()
    sns.heatmap(np.corrcoef(corrcoef), annot=True, cmap='rainbow', xticklabels=group_list, yticklabels=group_list)
    plt.savefig(path + '%s_corrcoef.png' % (cell_type))
    plt.close()
    plt.clf()