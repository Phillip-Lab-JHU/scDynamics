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

"""Generates Data for Figure 4: Age Axis."""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast
from dnn.autoencoders import set_duration_for_autoencoder
from dnn.classification import Temporal_Conv1D_2D_classifier, Res_Conv1D_LSTM_classifier

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df = df[df['Age']!=55].reset_index(drop=True)
df_duration = df_duration[df_duration['Age']!=55].reset_index(drop=True)


path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure4. Age regression\age axis\\'

#################################### Construct Multidimensional Age Axis ####################################
df.columns.get_loc('morpho_displ_autocorr_3')
from sklearn.decomposition import PCA
motility_data = df.iloc[:, 2:115].drop(['speed_distribution_x', 'speed_distribution_y'], axis=1)
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_pc = pd.DataFrame(pcs, columns=['PC%s' % str(i) for i in range(0, pcs.shape[1])])
feature_list = list(df_pc.columns)
df_pc = pd.concat([df.drop(['PC1', 'PC2'], axis=1), df_pc], axis=1)


#ages_dict = {}
centroids = {}
for age_group in np.unique(df_pc['Type']):
    df_pc_part = df_pc[df_pc['Type']==age_group].reset_index(drop=True)
    #ages_temp = {}
    centroids_temp = {}
    for media in np.unique(df_pc_part['Condition']):
        df_pc_part_part = df_pc_part[df_pc_part['Condition']==media].reset_index(drop=True)
        ages = np.unique(df_pc_part_part['Age']) # ages[0] = min age, ages[-1] = max age
        #ages_temp[media] = np.array( [ages[0], ages[-1]] )

        min_centroid = np.mean( df_pc_part_part[df_pc_part_part['Age'] == ages[0]][feature_list], axis=0 )  # (n_pc, )
        max_centroid = np.mean(df_pc_part_part[df_pc_part_part['Age'] == ages[-1]][feature_list], axis=0) # (n_pc, )
        centroids_temp[media] = np.array([min_centroid, max_centroid]) # (2, n_pc)
    #ages_dict[age_group] = ages_temp
    centroids[age_group] = centroids_temp


datasets = {}
ages = []
coeffs = []
age_types = []
media_types = []

for batch in np.unique(df_pc['Patient']):
    df_part = df_pc[df_pc['Patient'] == batch].reset_index(drop=True)
    medias = np.unique(df_part['Condition'])

    for media in medias:
        df_part_part = df_part[df_part['Condition'] == media].reset_index(drop=True)
        age = np.unique( df_part_part['Age'] )[0]
        age_type = np.unique( df_part_part['Type'] )[0]
        target_data = np.array(df_part_part[feature_list])
        target_centroid = np.mean(target_data, axis=0)
        min_centroid, max_centroid = centroids[age_type][media]
        #print(batch, age_type, media, age, min_centroid[0], max_centroid[0], target_centroid[0], )

        proj, t = project_on_line(start=min_centroid, end=max_centroid, target=target_centroid, segment=False)

        ages.append(age)
        age_types.append(age_type)
        coeffs.append(t)
        media_types.append(media)

datasets['Age'] = np.array(ages)
datasets['Value'] = np.array(coeffs)
datasets['Type'] = np.array(age_types)
datasets['Condition'] = np.array(media_types)

df_ = pd.DataFrame(datasets)



linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = np.unique(df_pc['Type']).size
ncols = np.unique(df_pc['Condition']).size

from matplotlib.ticker import MaxNLocator
font = {'family': 'arial',
            'weight': 'normal',
            'size': 10}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = linewidth

fig,axes = plt.subplots(nrows, ncols, figsize=(20, 10), sharey='row')
for row, type in enumerate(['Young', 'Old', 'Frail']):
    for col, cond in enumerate(np.unique(df['Condition'])):
        ax = axes[row][col]
        df_part_= df_[ (df_['Type']==type) & (df_['Condition']==cond) ].reset_index(drop=True)
        sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

        r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
        if type == 'Young':
            plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            # plt.text(0.8, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
            #          fontsize=12, fontdict={'weight': 'bold'}, color="black")
        else:
            plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            # plt.text(0.1, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
            #          fontsize=12, fontdict={'weight': 'bold'}, color="black")

        ax.spines["left"].set_visible(True)
        ax.spines['left'].set_linewidth(linewidth)
        ax.spines['left'].set_color('0.2')

        ax.spines["bottom"].set_visible(True)
        ax.spines['bottom'].set_linewidth(linewidth)
        ax.spines['bottom'].set_color('0.2')

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=linewidth, color='0.2', labelsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('Age', fontsize=10, weight='bold', color='0.2', labelpad=5)
        ax.set_ylabel('', fontsize=10, weight='bold', color='0.2', labelpad=5)
        ax.set_xlim(16, 98)

plt.savefig(path + 'coeff regplot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/coeff regplot.svg', bbox_inches='tight')
plt.clf()
plt.close()


linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = 2
ncols = np.unique(df_pc['Condition']).size

from matplotlib.ticker import MaxNLocator
font = {'family': 'arial',
            'weight': 'normal',
            'size': 10}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = linewidth

fig,axes = plt.subplots(nrows, ncols, figsize=(20, 8), sharey='row')
for row, type in enumerate(['Old', 'Frail']):
    for col, cond in enumerate(np.unique(df['Condition'])):
        ax = axes[row][col]
        df_part_= df_[ (df_['Type']==type) & (df_['Condition']==cond) ].reset_index(drop=True)
        sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

        r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
        if type == 'Young':
            plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            # plt.text(0.8, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
            #          fontsize=12, fontdict={'weight': 'bold'}, color="black")
        else:
            plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            # plt.text(0.1, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
            #          fontsize=12, fontdict={'weight': 'bold'}, color="black")

        ax.spines["left"].set_visible(True)
        ax.spines['left'].set_linewidth(linewidth)
        ax.spines['left'].set_color('0.2')

        ax.spines["bottom"].set_visible(True)
        ax.spines['bottom'].set_linewidth(linewidth)
        ax.spines['bottom'].set_color('0.2')

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=linewidth, color='0.2', labelsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('Age', fontsize=10, weight='bold', color='0.2', labelpad=5)
        ax.set_ylabel('', fontsize=10, weight='bold', color='0.2', labelpad=5)
        ax.set_xlim(60, 98)

plt.savefig(path + 'no young coeff regplot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/no young coeff regplot.svg', bbox_inches='tight')
plt.clf()
plt.close()


#################################### Corr with PCs and Age ####################################


linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = np.unique(df_pc['Type']).size
ncols = np.unique(df_pc['Condition']).size


if not os.path.isdir(path + 'PCs regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'PCs regplot/')

df_pc.columns.get_loc('PC0')
df_pc.columns.get_loc('PC45')
feature_list = list( df_pc.columns[197:243] )

for feature in feature_list:
    datasets = {}
    ages = []
    mean_values = []
    age_types = []
    media_types = []

    for batch in np.unique(df_pc['Patient']):
        df_part = df_pc[df_pc['Patient'] == batch].reset_index(drop=True)
        medias = np.unique(df_part['Condition'])

        for media in medias:
            df_part_part = df_part[df_part['Condition'] == media].reset_index(drop=True)
            age = np.unique( df_part_part['Age'] )[0]
            age_type = np.unique( df_part_part['Type'] )[0]
            mean_value = np.mean(np.array(df_part_part[feature]))

            ages.append(age)
            age_types.append(age_type)
            mean_values.append(mean_value)
            media_types.append(media)

    datasets['Age'] = np.array(ages)
    datasets['Value'] = np.array(mean_values)
    datasets['Type'] = np.array(age_types)
    datasets['Condition'] = np.array(media_types)

    df_ = pd.DataFrame(datasets)

    from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig,axes = plt.subplots(nrows, ncols, figsize=(20, 10), sharey='row')
    for row, type in enumerate(['Young', 'Old', 'Frail']):
        for col, cond in enumerate(np.unique(df['Condition'])):
            ax = axes[row][col]
            df_part_= df_[ (df_['Type']==type) & (df_['Condition']==cond) ].reset_index(drop=True)
            sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

            r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
            if type == 'Young':
                plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
                plt.text(0.8, 0.85, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
            else:
                plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
                plt.text(0.1, 0.85, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")

            ax.spines["left"].set_visible(True)
            ax.spines['left'].set_linewidth(linewidth)
            ax.spines['left'].set_color('0.2')

            ax.spines["bottom"].set_visible(True)
            ax.spines['bottom'].set_linewidth(linewidth)
            ax.spines['bottom'].set_color('0.2')

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(width=linewidth, color='0.2', labelsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            ax.set_xlabel('Age', fontsize=10, weight='bold', color='0.2', labelpad=5)
            ax.set_ylabel('', fontsize=10, weight='bold', color='0.2', labelpad=5)
            ax.set_xlim(16, 98)

    plt.savefig(path + 'PCs regplot/%s regplot.png'%(feature), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/PCs regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/PCs regplot/')
    plt.savefig(path + 'svg/PCs regplot/%s regplot.svg'%(feature), bbox_inches='tight')
    plt.clf()
    plt.close()