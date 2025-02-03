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

"""Generates Data for Figure 4."""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast
from dnn.autoencoders import set_duration_for_autoencoder
from dnn.classification import Temporal_Conv1D_2D_classifier, Res_Conv1D_LSTM_classifier


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Anshika motility project\\'
df = pd.read_parquet(path+'motility_features_96.parquet')
df_duration = pd.read_parquet(path+'traj_duration_96.parquet')


df.loc[(df['Age'] <= 40), 'Type'] = 'Young'
df.loc[(df['Age'] > 40)&(df['Age'] < 65), 'Type'] = 'Mid'
df.loc[(df['Age'] >= 65), 'Type'] = 'Old'


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Anshika motility project\figure3. age correlation\\'


#################################### Continous age lm plot for motility features ####################################

linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = 1
ncols = np.unique(df['Senescence']).size

if not os.path.isdir(path + 'age continuous regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'age continuous regplot/')

feature_list = list( df.columns[2:103].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )

for feature in feature_list:
    datasets = {}
    ages = []
    mean_values = []
    age_types = []
    media_types = []

    for batch in np.unique(df['Cell Line']):
        df_part = df[df['Cell Line'] == batch].reset_index(drop=True)
        medias = np.unique(df_part['Senescence'])

        for media in medias:
            df_part_part = df_part[df_part['Senescence'] == media].reset_index(drop=True)
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
    datasets['Senescence'] = np.array(media_types)

    df_ = pd.DataFrame(datasets)

    from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5), sharey='row')
    for col, cond in enumerate(np.unique(df['Senescence'])):
        ax = axes[col]
        df_part_= df_[(df_['Senescence']==cond) ].reset_index(drop=True)
        sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

        r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
        if type == 'Non-frail':
            plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            plt.text(0.8, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
        else:
            plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            plt.text(0.1, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
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
        ax.set_xlim(20, 90)

    plt.savefig(path + 'age continuous regplot/%s regplot.png'%(feature), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/age continuous regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/age continuous regplot/')
    plt.savefig(path + 'svg/age continuous regplot/%s regplot.svg'%(feature), bbox_inches='tight')
    plt.clf()
    plt.close()

#################################### Continous age loess regression for motility features  ####################################

linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = 1
ncols = np.unique(df['Senescence']).size

import statsmodels.api as sm
lowess_sm = sm.nonparametric.lowess

def lowess_with_confidence_bounds(x, y, eval_x, N=200, conf_interval=0.95, lowess_kw=None):
    """
    Perform Lowess regression and determine a confidence interval by bootstrap resampling
    """
    # Lowess smoothing
    smoothed = sm.nonparametric.lowess(exog=x, endog=y, xvals=eval_x, **lowess_kw)

    # Perform bootstrap resamplings of the data and  evaluate the smoothing at a fixed set of points
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]

        smoothed_values[i] = sm.nonparametric.lowess(exog=sampled_x, endog=sampled_y, xvals=eval_x, **lowess_kw)

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return smoothed, bottom, top


if not os.path.isdir(path + 'age continuous loess/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'age continuous loess/')

feature_list = list( df.columns[2:103].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )
x_range = np.linspace(20, 90, 71)
lowess_kw = {'frac': 0.4, 'it': 50, 'return_sorted': False}

for feature in feature_list:
    datasets = {}
    ages = []
    mean_values = []
    age_types = []
    media_types = []

    for batch in np.unique(df['Cell Line']):
        df_part = df[df['Cell Line'] == batch].reset_index(drop=True)
        medias = np.unique(df_part['Senescence'])

        for media in medias:
            df_part_part = df_part[df_part['Senescence'] == media].reset_index(drop=True)
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
    datasets['Senescence'] = np.array(media_types)

    df_ = pd.DataFrame(datasets)

    from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5), sharey='row')
    for col, cond in enumerate(np.unique(df['Senescence'])):
        ax = axes[col]
        df_part_= df_[(df_['Senescence']==cond) ].reset_index(drop=True)
        yest_sm = lowess_sm(exog=df_part_['Age'].values, endog=df_part_['Value'].values, **lowess_kw)
        smoothed, bottom, top = lowess_with_confidence_bounds(df_part_['Age'].values, df_part_['Value'].values, x_range,
                                                              lowess_kw=lowess_kw)
        # sns.lineplot(x=df_part_['Age'].values, y=yest_sm, color='red', ax=ax)
        sns.lineplot(x=x_range, y=smoothed, color='red', lw=2.5, ax=ax)
        sns.scatterplot(x=df_part_['Age'].values, y=df_part_['Value'].values, color="darkblue", alpha=0.7, s=30, ax=ax)
        ax.fill_between(x_range, bottom, top, alpha=0.4, color='red')

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
        ax.set_xlim(20, 90)

    plt.savefig(path + 'age continuous loess/%s loess.png' % (feature), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/age continuous loess/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/age continuous loess/')
    plt.savefig(path + 'svg/age continuous loess/%s loess.svg' % (feature), bbox_inches='tight')
    plt.clf()
    plt.close()



#################################### Continous age lm plot for motility PCs####################################

from sklearn.decomposition import PCA
motility_data = df.iloc[:, 2:103].drop(['speed_distribution_x', 'speed_distribution_y'], axis=1)
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_pc = pd.DataFrame(pcs, columns=['motility_PC_%s' % str(i) for i in range(0, pcs.shape[1])])
feature_list = list(df_pc.columns)
df_pc = pd.concat([df, df_pc], axis=1)


linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = 1
ncols = np.unique(df['Senescence']).size

if not os.path.isdir(path + 'age continuous regplot PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'age continuous regplot PC/')

for feature in feature_list:
    datasets = {}
    ages = []
    mean_values = []
    age_types = []
    media_types = []

    for batch in np.unique(df_pc['Cell Line']):
        df_part = df_pc[df_pc['Cell Line'] == batch].reset_index(drop=True)
        medias = np.unique(df_part['Senescence'])

        for media in medias:
            df_part_part = df_part[df_part['Senescence'] == media].reset_index(drop=True)
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
    datasets['Senescence'] = np.array(media_types)

    df_ = pd.DataFrame(datasets)

    from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5), sharey='row')
    for col, cond in enumerate(np.unique(df['Senescence'])):
        ax = axes[col]
        df_part_= df_[(df_['Senescence']==cond) ].reset_index(drop=True)
        sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

        r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
        if type == 'Non-frail':
            plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            plt.text(0.8, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
        else:
            plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            plt.text(0.1, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
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
        ax.set_xlim(20, 90)

    plt.savefig(path + 'age continuous regplot PC/%s regplot.png'%(feature), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/age continuous regplot PC/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/age continuous regplot PC/')
    plt.savefig(path + 'svg/age continuous regplot PC/%s regplot.svg'%(feature), bbox_inches='tight')
    plt.clf()
    plt.close()


#################################### Piece-wise age lm plot ####################################

linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = np.unique(df['Type']).size
ncols = np.unique(df['Senescence']).size

if not os.path.isdir(path + 'age piecewise regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'age piecewise regplot/')

feature_list = list( df.columns[2:103].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )

for feature in feature_list:
    datasets = {}
    ages = []
    mean_values = []
    age_types = []
    media_types = []

    for batch in np.unique(df['Cell Line']):
        df_part = df[df['Cell Line'] == batch].reset_index(drop=True)
        medias = np.unique(df_part['Senescence'])

        for media in medias:
            df_part_part = df_part[df_part['Senescence'] == media].reset_index(drop=True)
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
    datasets['Senescence'] = np.array(media_types)

    df_ = pd.DataFrame(datasets)

    from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig,axes = plt.subplots(nrows, ncols, figsize=(8, 10), sharey='row')
    for row, type in enumerate(['Young', 'Mid', 'Old']):
        for col, cond in enumerate(np.unique(df['Senescence'])):
            ax = axes[row][col]
            df_part_= df_[ (df_['Type']==type) & (df_['Senescence']==cond) ].reset_index(drop=True)
            sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

            r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
            if type == 'Non-frail':
                plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
                plt.text(0.8, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
            else:
                plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
                plt.text(0.1, 0.88, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
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
            ax.set_xlim(20, 90)

    plt.savefig(path + 'age piecewise regplot/%s regplot.png'%(feature), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/age piecewise regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/age piecewise regplot/')
    plt.savefig(path + 'svg/age piecewise regplot/%s regplot.svg'%(feature), bbox_inches='tight')
    plt.clf()
    plt.close()



#################################### Correlation btw age and all motility features ####################################

feature_list = list( df.columns[2:103].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )

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

    for batch in np.unique(df['Cell Line']):
        df_part = df[df['Cell Line'] == batch].reset_index(drop=True)
        medias = np.unique(df_part['Senescence'])

        for media in medias:
            df_part_part = df_part[df_part['Senescence'] == media].reset_index(drop=True)
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
    datasets['Senescence'] = np.array(media_types)

    df_ = pd.DataFrame(datasets)

    for type in ['Young', 'Mid', 'Old']:
        condition_rs = []
        condition_ps = []

        for cond in np.unique(df['Senescence']):
            df_part_= df_[ (df_['Type']==type) & (df_['Senescence']==cond) ].reset_index(drop=True)
            #sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)
            r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
            #print(type, cond, r, p)
            condition_rs.append(r)
            condition_ps.append(p)
        each_rs = pd.DataFrame(condition_rs, columns=[feature], index='%s ' % type + np.unique(df['Senescence'].astype(str)))
        each_ps = pd.DataFrame(condition_ps, columns=[feature], index='%s ' % type + np.unique(df['Senescence'].astype(str)))
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
kws = dict(cbar_kws=dict(ticks=[0.9, 0, -0.9], orientation='horizontal'), vmin=-0.9, vmax=0.9 )

g=sns.clustermap(df_r_filtered, annot=False, cmap=cmc.bam, col_cluster=True, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (12, 3.5),
dendrogram_ratio=0.1,
cbar=True
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=10, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, rotation=0, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation Coefficient', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# for spine in g.ax_cbar.spines:
#     #g.ax_cbar.spines[spine].set_color('crimson')
#     g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'correlation with age for all features.png', dpi=300,bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/' + 'correlation with age for all features.svg', bbox_inches='tight')
plt.clf()
plt.close()



#################################### Correlation btw age and PC motility features ####################################

from sklearn.decomposition import PCA
motility_data = df.iloc[:, 2:103].drop(['speed_distribution_x', 'speed_distribution_y'], axis=1)
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_pc = pd.DataFrame(pcs, columns=['motility_PC_%s' % str(i) for i in range(0, pcs.shape[1])])
feature_list = list(df_pc.columns)
df_pc = pd.concat([df, df_pc], axis=1)

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

    for batch in np.unique(df_pc['Cell Line']):
        df_part = df_pc[df_pc['Cell Line'] == batch].reset_index(drop=True)
        medias = np.unique(df_part['Senescence'])

        for media in medias:
            df_part_part = df_part[df_part['Senescence'] == media].reset_index(drop=True)
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
    datasets['Senescence'] = np.array(media_types)

    df_ = pd.DataFrame(datasets)

    for type in ['Young', 'Mid', 'Old']:
        condition_rs = []
        condition_ps = []

        for cond in np.unique(df['Senescence']):
            df_part_= df_[ (df_['Type']==type) & (df_['Senescence']==cond) ].reset_index(drop=True)
            #sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)
            r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
            #print(type, cond, r, p)
            condition_rs.append(r)
            condition_ps.append(p)
        each_rs = pd.DataFrame(condition_rs, columns=[feature], index='%s ' % type + np.unique(df['Senescence'].astype(str)))
        each_ps = pd.DataFrame(condition_ps, columns=[feature], index='%s ' % type + np.unique(df['Senescence'].astype(str)))
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
kws = dict(cbar_kws=dict(ticks=[0.9, 0, -0.9], orientation='horizontal'), vmin=-0.9, vmax=0.9 )

g=sns.clustermap(df_r, annot=False, cmap=cmc.bam, col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (14, 3.5),
dendrogram_ratio=0.1,
cbar=True
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=10, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, rotation=0, va='center')

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


############# Plot continous linear regression btw age and cluster enrichment ###############

condition_name = 'Age'
cluster_type = 'kmeans'
for condition in np.unique(df['Senescence']):
    df_part = df[(df['Senescence']==condition)].reset_index(drop=True)
    group_clones=[]
    for group in np.unique(df_part[condition_name]):
    #for group in ['DZ', 'sLZ', 'dLZ']:
        aaa = df_part[df_part[condition_name] == group]

        group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
        group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
        group_clone = group_clone.unstack(level=0)
        group_clone[np.isnan(group_clone)] = 0
        group_clone_T = group_clone.T
        for cluster in list(np.unique(df_part[cluster_type])):
            if cluster in group_clone_T.columns:
                continue
            else:
                group_clone_T.insert(loc=int(cluster), column=cluster, value=[0])
        group_clone = group_clone_T.T
        group_clones.append(group_clone)


    enrichments = pd.DataFrame()
    for group_clone in group_clones:  # 'DZ', 'sLz', 'dLZ'
        enrichments = pd.concat( [enrichments, group_clone[list(group_clone.columns)[0]]], axis=1 )

    # for column in enrichments.columns:
    #     value = np.unique( df[df['Type'] == column]['Age'] )[0]
    #     enrichments.rename(columns={column:'%f'%value}, inplace=True)

    #enrichments.columns = enrichments.columns.astype(float)
    #enrichments = enrichments.sort_index(axis=1)

    n_colors = np.unique(df['kmeans']).shape[0]
    colors=cmc.batlow
    cmap = [colors(1. * i / n_colors) for i in range(n_colors)]

    font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    fig, ax = plt.subplots(figsize=(4,4))  # 2 inch by 2 inch
    # almost verbatim from question

    rs = []
    ps = []
    for idx, kmeans in enumerate(enrichments.index):
        x = list(enrichments.columns)
        y = enrichments.iloc[kmeans, :].values

        r, p = scipy.stats.pearsonr(x, y)
        if p>0.2:
            continue
        sns.regplot(x=x, y=y, ci=None, line_kws={'color':cmap[idx], 'linewidth':3},
                    label=kmeans, scatter=False, scatter_kws={'s': 10, 'color':cmap[idx]})
        rs.append(r)
        ps.append(p)
        # plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
        #          fontsize=20, fontdict={'weight': 'bold'}, color="blue")

    plt.title('r: %s, p: %s'%(rs, ps) , fontsize=4, fontdict={'weight': 'bold'})
    #plt.xticks(plt.xticks()[0], datasets['Exact_Age'], fontsize=12, fontdict={'weight': 'bold'})
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()

    ax.tick_params(width=2, color='0.2')

    ax.set_ylabel('fraction of cluster (%)', fontsize=12, weight='bold', color='0.2')
    # plt.xticks(np.arange(len(list(mt_enrichments.iloc[kmeans, :].index))), np.array(list(mt_enrichments.columns)), fontsize=16, rotation=35, # fontname = "Arial",
    #            rotation_mode='anchor', ha='right', color='0.2', weight='bold')
    plt.xticks(x, fontsize=7, rotation=35, # fontname = "Arial",
               rotation_mode='anchor', ha='right', color='0.2', weight='bold')

    plt.yticks(fontsize=7, color='0.2', weight='bold')

    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
               loc='best')

    plt.savefig(path + 'continous %s cluster fraction regplot.png'%(condition), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/continous %s cluster fraction regplot.svg'%(condition), bbox_inches='tight')
    plt.clf()
    plt.close()


############# Plot piecewise linear regression btw age and cluster enrichment ###############

condition_name = 'Age'
cluster_type = 'kmeans'
for age_group in np.unique(df['Type']):
    for condition in np.unique(df['Senescence']):
        df_part = df[(df['Type']==age_group)&(df['Senescence']==condition)].reset_index(drop=True)
        group_clones=[]
        for group in np.unique(df_part[condition_name]):
        #for group in ['DZ', 'sLZ', 'dLZ']:
            aaa = df_part[df_part[condition_name] == group]

            group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
            group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
            group_clone = group_clone.unstack(level=0)
            group_clone[np.isnan(group_clone)] = 0
            group_clone_T = group_clone.T
            for cluster in list(np.unique(df_part[cluster_type])):
                if cluster in group_clone_T.columns:
                    continue
                else:
                    group_clone_T.insert(loc=int(cluster), column=cluster, value=[0])
            group_clone = group_clone_T.T
            group_clones.append(group_clone)


        enrichments = pd.DataFrame()
        for group_clone in group_clones:  # 'DZ', 'sLz', 'dLZ'
            enrichments = pd.concat( [enrichments, group_clone[list(group_clone.columns)[0]]], axis=1 )

        # for column in enrichments.columns:
        #     value = np.unique( df[df['Type'] == column]['Age'] )[0]
        #     enrichments.rename(columns={column:'%f'%value}, inplace=True)

        #enrichments.columns = enrichments.columns.astype(float)
        #enrichments = enrichments.sort_index(axis=1)

        n_colors = np.unique(df['kmeans']).shape[0]
        colors=cmc.batlow
        cmap = [colors(1. * i / n_colors) for i in range(n_colors)]

        font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 8}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 1

        fig, ax = plt.subplots(figsize=(4,4))  # 2 inch by 2 inch
        # almost verbatim from question

        rs = []
        ps = []
        for idx, kmeans in enumerate(enrichments.index):
            x = list(enrichments.columns)
            y = enrichments.iloc[kmeans, :].values

            r, p = scipy.stats.pearsonr(x, y)
            if p>0.2:
                continue
            sns.regplot(x=x, y=y, ci=None, line_kws={'color':cmap[idx], 'linewidth':3},
                        label=kmeans, scatter=False, scatter_kws={'s': 10, 'color':cmap[idx]})
            rs.append(r)
            ps.append(p)
            # plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
            #          fontsize=20, fontdict={'weight': 'bold'}, color="blue")

        plt.title('r: %s, p: %s'%(rs, ps) , fontsize=4, fontdict={'weight': 'bold'})
        #plt.xticks(plt.xticks()[0], datasets['Exact_Age'], fontsize=12, fontdict={'weight': 'bold'})
        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_color('0.2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()

        ax.tick_params(width=2, color='0.2')

        ax.set_ylabel('fraction of cluster (%)', fontsize=12, weight='bold', color='0.2')
        # plt.xticks(np.arange(len(list(mt_enrichments.iloc[kmeans, :].index))), np.array(list(mt_enrichments.columns)), fontsize=16, rotation=35, # fontname = "Arial",
        #            rotation_mode='anchor', ha='right', color='0.2', weight='bold')
        plt.xticks(x, fontsize=7, rotation=35, # fontname = "Arial",
                   rotation_mode='anchor', ha='right', color='0.2', weight='bold')

        plt.yticks(fontsize=7, color='0.2', weight='bold')

        plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
                   loc='best')

        plt.savefig(path + '%s %s cluster fraction regplot.png'%(age_group, condition), dpi=300, bbox_inches='tight')
        if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/')
        plt.savefig(path + 'svg/%s %s cluster fraction regplot.svg'%(age_group, condition), bbox_inches='tight')
        plt.clf()
        plt.close()



############# Correlation btw age and cluster enrichment for all clusters and conditions###############

condition_name = 'Age'
cluster_type = 'kmeans'

df_enrichment_r = pd.DataFrame()
df_enrichment_p = pd.DataFrame()

for age_group in ['Young', 'Mid', 'Old']:
    df_r_temp = pd.DataFrame()
    df_p_temp = pd.DataFrame()

    for condition in np.unique(df['Senescence']):
        df_part = df[(df['Type']==age_group)&(df['Senescence']==condition)].reset_index(drop=True)
        group_clones=[]
        for group in np.unique(df_part[condition_name]):
        #for group in ['DZ', 'sLZ', 'dLZ']:
            aaa = df_part[df_part[condition_name] == group]

            group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
            group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
            group_clone = group_clone.unstack(level=0)
            group_clone[np.isnan(group_clone)] = 0
            group_clone_T = group_clone.T
            for cluster in list(np.unique(df_part[cluster_type])):
                if cluster in group_clone_T.columns:
                    continue
                else:
                    group_clone_T.insert(loc=int(cluster), column=cluster, value=[0])
            group_clone = group_clone_T.T
            group_clones.append(group_clone)


        enrichments = pd.DataFrame()
        for group_clone in group_clones:  # 'DZ', 'sLz', 'dLZ'
            enrichments = pd.concat( [enrichments, group_clone[list(group_clone.columns)[0]]], axis=1 )

        # for column in enrichments.columns:
        #     value = np.unique( df[df['Type'] == column]['Age'] )[0]
        #     enrichments.rename(columns={column:'%f'%value}, inplace=True)

        #enrichments.columns = enrichments.columns.astype(float)
        #enrichments = enrichments.sort_index(axis=1)

        n_colors = np.unique(df['kmeans']).shape[0]
        colors=cmc.batlow
        cmap = [colors(1. * i / n_colors) for i in range(n_colors)]


        rs = []
        ps = []
        for idx, kmeans in enumerate(enrichments.index):
            x = list(enrichments.columns)
            y = enrichments.iloc[kmeans, :].values

            r, p = scipy.stats.pearsonr(x, y)
            rs.append(r)
            ps.append(p)
            # plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
            #          fontsize=20, fontdict={'weight': 'bold'}, color="blue")
        each_rs = pd.DataFrame(rs, columns=['%s %s'%(age_group, condition)], index=enrichments.index)
        each_ps = pd.DataFrame(ps, columns=['%s %s'%(age_group, condition)], index=enrichments.index)
        df_r_temp = pd.concat([df_r_temp, each_rs], axis=1)
        df_p_temp = pd.concat([df_p_temp, each_ps], axis=1)

    df_enrichment_r = pd.concat([df_enrichment_r, df_r_temp], axis=1)
    df_enrichment_p = pd.concat([df_enrichment_p, df_p_temp], axis=1)

df_enrichment_r = df_enrichment_r.T
df_enrichment_p = df_enrichment_p.T

#df_enrichment_p = -np.log10(df_enrichment_p)


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
kws = dict(cbar_kws=dict(ticks=[0.9, 0, -0.9], orientation='horizontal'), vmin=-0.9, vmax=0.9 )

g=sns.clustermap(df_enrichment_r, annot=df_enrichment_p,  fmt=".3f", cmap=cmc.bam, col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (4, 4),
dendrogram_ratio=0.1,
cbar=True
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=12, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, rotation=0, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation Coefficient', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# for spine in g.ax_cbar.spines:
#     #g.ax_cbar.spines[spine].set_color('crimson')
#     g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'p value annotated Heatmap Age correlation for clusters.png', dpi=300,bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/' + 'p value annotated Heatmap Age correlation for clusters.svg', bbox_inches='tight')
plt.clf()
plt.close()

fig, ax = plt.subplots()
# if np.max(np.max(Z_avg_df)) >= abs(np.min(np.min(Z_avg_df))):
#     kws = dict(cbar_kws=dict(ticks=[-round(np.max(np.max(Z_avg_df)), 1), 0, round(np.max(np.max(Z_avg_df)), 1)], orientation='horizontal'),
#                vmin=-round(np.max(np.max(Z_avg_df)), 1))
# else:
#     kws = dict(cbar_kws=dict(ticks=[round(np.min(np.min(Z_avg_df)), 1), 0, -round(np.min(np.min(Z_avg_df)), 1)],orientation='horizontal'),
#                vmin=round(np.min(np.min(Z_avg_df)), 1) )
#
kws = dict(cbar_kws=dict(ticks=[0.6, 0, -0.6], orientation='horizontal'), vmin=-0.6, vmax=0.6 )

g=sns.clustermap(df_enrichment_r, annot=False, cmap=cmc.bam, col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (4, 4),
dendrogram_ratio=0.1,
cbar=True
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=12, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, rotation=0, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation Coefficient', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# for spine in g.ax_cbar.spines:
#     #g.ax_cbar.spines[spine].set_color('crimson')
#     g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'Heatmap Age correlation for clusters.png', dpi=300,bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/' + 'Heatmap Age correlation for clusters.svg', bbox_inches='tight')
plt.clf()
plt.close()

#################################### Construct Continous multidimensional age axis ####################################

from sklearn.decomposition import PCA
motility_data = df.iloc[:, 2:103].drop(['speed_distribution_x', 'speed_distribution_y'], axis=1)
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_pc = pd.DataFrame(pcs, columns=['PC%s' % str(i) for i in range(0, pcs.shape[1])])
feature_list = list(df_pc.columns)
df_pc = pd.concat([df.drop(['PC1', 'PC2'], axis=1), df_pc], axis=1)



centroids = {}
for media in np.unique(df_pc['Senescence']):
    df_pc_part_part = df_pc[df_pc['Senescence']==media].reset_index(drop=True)
    ages = np.unique(df_pc_part_part['Age']) # ages[0] = min age, ages[-1] = max age
    #ages_temp[media] = np.array( [ages[0], ages[-1]] )

    min_centroid = np.mean( df_pc_part_part[df_pc_part_part['Age'] == ages[0]][feature_list], axis=0 )  # (n_pc, )
    max_centroid = np.mean(df_pc_part_part[df_pc_part_part['Age'] == ages[-1]][feature_list], axis=0) # (n_pc, )
    centroids[media] = np.array([min_centroid, max_centroid]) # (2, n_pc)



datasets = {}
ages = []
coeffs = []
age_types = []
media_types = []

for batch in np.unique(df_pc['Cell Line']):
    df_part = df_pc[df_pc['Cell Line'] == batch].reset_index(drop=True)
    medias = np.unique(df_part['Senescence'])

    for media in medias:
        df_part_part = df_part[df_part['Senescence'] == media].reset_index(drop=True)
        age = np.unique( df_part_part['Age'] )[0]
        age_type = np.unique( df_part_part['Type'] )[0]
        target_data = np.array(df_part_part[feature_list])
        target_centroid = np.mean(target_data, axis=0)
        min_centroid, max_centroid = centroids[media]
        #print(batch, age_type, media, age, min_centroid[0], max_centroid[0], target_centroid[0], )

        proj, t = project_on_line(start=min_centroid, end=max_centroid, target=target_centroid, segment=False)

        ages.append(age)
        age_types.append(age_type)
        coeffs.append(t)
        media_types.append(media)

datasets['Age'] = np.array(ages)
datasets['Value'] = np.array(coeffs)
datasets['Type'] = np.array(age_types)
datasets['Senescence'] = np.array(media_types)

df_ = pd.DataFrame(datasets)

#################################### Continous AAC vs Age correlation ####################################
linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = 1
ncols = np.unique(df_pc['Senescence']).size

from matplotlib.ticker import MaxNLocator
font = {'family': 'arial',
            'weight': 'normal',
            'size': 10}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = linewidth

fig,axes = plt.subplots(nrows, ncols, figsize=(10, 5), sharey='row')
for col, cond in enumerate(np.unique(df['Senescence'])):
    ax = axes[col]
    df_part_= df_[(df_['Senescence']==cond) ].reset_index(drop=True)
    sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

    r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
    plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
             fontsize=12, fontdict={'weight': 'bold'}, color="black")
    plt.text(0.8, 0.88, "p = " + str(round(p, 3)), ha='left', va='top', transform=ax.transAxes,
             fontsize=12, fontdict={'weight': 'bold'}, color="black")
    # plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
    #          fontsize=12, fontdict={'weight': 'bold'}, color="black")
    # plt.text(0.1, 0.88, "p = " + str(round(p, 3)), ha='left', va='top', transform=ax.transAxes,
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
    ax.set_xlim(20, 90)

plt.savefig(path + 'continous coeff regplot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/continous coeff regplot.svg', bbox_inches='tight')
plt.clf()
plt.close()

#################################### Continous AAC LOESS regression  ####################################
import statsmodels.api as sm
lowess_sm = sm.nonparametric.lowess

def lowess_with_confidence_bounds(x, y, eval_x, N=200, conf_interval=0.95, lowess_kw=None):
    """
    Perform Lowess regression and determine a confidence interval by bootstrap resampling
    """
    # Lowess smoothing
    smoothed = sm.nonparametric.lowess(exog=x, endog=y, xvals=eval_x, **lowess_kw)

    # Perform bootstrap resamplings of the data
    # and  evaluate the smoothing at a fixed set of points
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]

        smoothed_values[i] = sm.nonparametric.lowess(exog=sampled_x, endog=sampled_y, xvals=eval_x, **lowess_kw)

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return smoothed, bottom, top



x_range = np.linspace(20, 90, 71)
lowess_kw = {'frac': 0.4, 'it': 50, 'return_sorted': False}
fig,axes = plt.subplots(nrows, ncols, figsize=(10, 5), sharey='row')
for col, cond in enumerate(np.unique(df['Senescence'])):
    ax = axes[col]
    df_part_= df_[(df_['Senescence']==cond) ].reset_index(drop=True)
    #yest_bell = lowess_bell_shape_kern(df_part_['Age'].values, df_part_['Value'].values, tau=0.005)
    yest_sm = lowess_sm(exog=df_part_['Age'].values, endog=df_part_['Value'].values, **lowess_kw)
    smoothed, bottom, top = lowess_with_confidence_bounds(df_part_['Age'].values, df_part_['Value'].values, x_range,
                                                          lowess_kw=lowess_kw)
    #sns.lineplot(x=df_part_['Age'].values, y=yest_sm, color='red', ax=ax)
    sns.lineplot(x=x_range, y=smoothed, color='red', lw=2.5, ax=ax)
    sns.scatterplot(x=df_part_['Age'].values, y=df_part_['Value'].values, color="darkblue", alpha=0.7, s=30, ax=ax)
    ax.fill_between(x_range, bottom, top, alpha=0.4, color='red')
    # plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
    #          fontsize=12, fontdict={'weight': 'bold'}, color="black")
    # plt.text(0.8, 0.88, "p = " + str(round(p, 3)), ha='left', va='top', transform=ax.transAxes,
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
    ax.set_xlim(20, 90)

plt.savefig(path + 'continous coeff loess_reg.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/continous coeff loess_reg.svg', bbox_inches='tight')
plt.clf()
plt.close()
#################################### Construct Piecewise multidimensional age axis ####################################

from sklearn.decomposition import PCA
motility_data = df.iloc[:, 2:103].drop(['speed_distribution_x', 'speed_distribution_y'], axis=1)
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
    for media in np.unique(df_pc_part['Senescence']):
        df_pc_part_part = df_pc_part[df_pc_part['Senescence']==media].reset_index(drop=True)
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

for batch in np.unique(df_pc['Cell Line']):
    df_part = df_pc[df_pc['Cell Line'] == batch].reset_index(drop=True)
    medias = np.unique(df_part['Senescence'])

    for media in medias:
        df_part_part = df_part[df_part['Senescence'] == media].reset_index(drop=True)
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
datasets['Senescence'] = np.array(media_types)

df_ = pd.DataFrame(datasets)

#################################### Piecewise AAC vs Age correlation ####################################
linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = np.unique(df_pc['Type']).size
ncols = np.unique(df_pc['Senescence']).size

from matplotlib.ticker import MaxNLocator
font = {'family': 'arial',
            'weight': 'normal',
            'size': 10}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = linewidth

fig,axes = plt.subplots(nrows, ncols, figsize=(8, 10), sharey='row')
for row, type in enumerate(['Young', 'Mid', 'Old']):
    for col, cond in enumerate(np.unique(df['Senescence'])):
        ax = axes[row][col]
        df_part_= df_[ (df_['Type']==type) & (df_['Senescence']==cond) ].reset_index(drop=True)
        sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

        r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
        if type == 'Young':
            plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            plt.text(0.8, 0.88, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
        else:
            plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                     fontsize=12, fontdict={'weight': 'bold'}, color="black")
            plt.text(0.1, 0.88, "p = " + str(round(p, 4)), ha='left', va='top', transform=ax.transAxes,
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
        ax.set_xlim(20, 90)

plt.savefig(path + 'coeff regplot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/coeff regplot.svg', bbox_inches='tight')
plt.clf()
plt.close()

