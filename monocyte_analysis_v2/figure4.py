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

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'
df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df = df[df['Age']!=55].reset_index(drop=True)
df_duration = df_duration[df_duration['Age']!=55].reset_index(drop=True)


path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure4. Age regression\\'

#################################### Piece-wise age lm plot for Frail vs Non frail and conditions ####################################
df['Type2'] = 'Frail'
df_duration['Type2'] = 'Frail'
df.loc[df['Type'] != 'Frail', 'Type2'] = 'Non-frail'
df_duration.loc[df_duration['Type'] != 'Frail', 'Type2'] = 'Non-frail'

linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = np.unique(df['Type2']).size
ncols = np.unique(df['Condition']).size



if not os.path.isdir(path + 'age continous regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'age continous regplot/')

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list( df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )

for feature in feature_list:
    datasets = {}
    ages = []
    mean_values = []
    age_types = []
    media_types = []

    for batch in np.unique(df['Patient']):
        df_part = df[df['Patient'] == batch].reset_index(drop=True)
        medias = np.unique(df_part['Condition'])

        for media in medias:
            df_part_part = df_part[df_part['Condition'] == media].reset_index(drop=True)
            age = np.unique( df_part_part['Age'] )[0]
            age_type = np.unique( df_part_part['Type2'] )[0]
            mean_value = np.mean(np.array(df_part_part[feature]))

            ages.append(age)
            age_types.append(age_type)
            mean_values.append(mean_value)
            media_types.append(media)

    datasets['Age'] = np.array(ages)
    datasets['Value'] = np.array(mean_values)
    datasets['Type2'] = np.array(age_types)
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
    for row, type in enumerate(['Non-frail', 'Frail']):
        for col, cond in enumerate(np.unique(df['Condition'])):
            ax = axes[row][col]
            df_part_= df_[ (df_['Type2']==type) & (df_['Condition']==cond) ].reset_index(drop=True)
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
            ax.set_xlim(16, 98)

    plt.savefig(path + 'age continous regplot/%s regplot.png'%(feature), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/age continous regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/age continous regplot/')
    plt.savefig(path + 'svg/age continous regplot/%s regplot.svg'%(feature), bbox_inches='tight')
    plt.clf()
    plt.close()



#################################### Piece-wise age lm plot for all age groups and conditions ####################################
linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = np.unique(df['Type']).size
ncols = np.unique(df['Condition']).size


if not os.path.isdir(path + 'age condition regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'age condition regplot/')

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list( df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )

for feature in feature_list:
    datasets = {}
    ages = []
    mean_values = []
    age_types = []
    media_types = []

    for batch in np.unique(df['Patient']):
        df_part = df[df['Patient'] == batch].reset_index(drop=True)
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
                # plt.text(0.8, 0.85, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                #          fontsize=12, fontdict={'weight': 'bold'}, color="black")
            else:
                plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
                # plt.text(0.1, 0.85, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
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

    plt.savefig(path + 'age condition regplot/%s regplot.png'%(feature), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/age condition regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/age condition regplot/')
    plt.savefig(path + 'svg/age condition regplot/%s regplot.svg'%(feature), bbox_inches='tight')
    plt.clf()
    plt.close()

#################################### Various patient info lm plot for young vs old vs frail and conditions ####################################


linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = len(['Old', 'Frail'])
ncols = np.unique(df['Condition']).size
y_feature = 'Frailty_score'


if not os.path.isdir(path + '%s regplot/'%y_feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + '%s regplot/'%y_feature)

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list( df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )
feature_list.append('Age')
for feature in feature_list:
    datasets = {}
    ages = []
    mean_values = []
    age_types = []
    media_types = []

    for batch in np.unique(df['Patient']):
        df_part = df[df['Patient'] == batch].reset_index(drop=True)
        medias = np.unique(df_part['Condition'])

        for media in medias:
            df_part_part = df_part[df_part['Condition'] == media].reset_index(drop=True)
            age = np.unique( df_part_part[y_feature] )[0]
            age_type = np.unique( df_part_part['Type'] )[0]
            mean_value = np.mean(np.array(df_part_part[feature]))

            ages.append(age)
            age_types.append(age_type)
            mean_values.append(mean_value)
            media_types.append(media)

    datasets[y_feature] = np.array(ages)
    datasets['Value'] = np.array(mean_values)
    datasets['Type'] = np.array(age_types)
    datasets['Condition'] = np.array(media_types)

    df_ = pd.DataFrame(datasets)

    df_ = df_[~df_[y_feature].isnull()].reset_index(drop=True)
    from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig,axes = plt.subplots(nrows, ncols, figsize=(20, 10), sharey='row')
    for row, type in enumerate(['Old', 'Frail']):
        for col, cond in enumerate(np.unique(df['Condition'])):
            ax = axes[row][col]
            df_part_= df_[ (df_['Type']==type) & (df_['Condition']==cond) ].reset_index(drop=True)
            sns.regplot(x=y_feature, y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

            r, p = scipy.stats.pearsonr(df_part_[y_feature], df_part_['Value'])
            if type == 'Non-frail':
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

            ax.set_xlabel('%s'%y_feature, fontsize=10, weight='bold', color='0.2', labelpad=5)
            ax.set_ylabel('', fontsize=10, weight='bold', color='0.2', labelpad=5)
            #ax.set_xlim(16, 98)

    plt.savefig(path + '%s regplot/%s regplot.png'%(y_feature, feature), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/%s regplot/'%y_feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/%s regplot/'%y_feature)
    plt.savefig(path + 'svg/%s regplot/%s regplot.svg'%(y_feature, feature), bbox_inches='tight')
    plt.clf()
    plt.close()

#################################### Correlation btw age and all motility features ####################################

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list( df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )

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

    for batch in np.unique(df['Patient']):
        df_part = df[df['Patient'] == batch].reset_index(drop=True)
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

    for type in ['Young', 'Old', 'Frail']:
        condition_rs = []
        condition_ps = []

        for cond in np.unique(df['Condition']):
            df_part_= df_[ (df_['Type']==type) & (df_['Condition']==cond) ].reset_index(drop=True)
            #sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)
            r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
            #print(type, cond, r, p)
            condition_rs.append(r)
            condition_ps.append(p)
        each_rs = pd.DataFrame(condition_rs, columns=[feature], index='%s ' % type + np.unique(df['Condition']))
        each_ps = pd.DataFrame(condition_ps, columns=[feature], index='%s ' % type + np.unique(df['Condition']))
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
kws = dict(cbar_kws=dict(ticks=[0.6, 0, -0.6], orientation='horizontal'), vmin=-0.6, vmax=0.6 )

g=sns.clustermap(df_r_filtered, annot=False, cmap=cmc.vik, col_cluster=True, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (12, 5),
dendrogram_ratio=0.1,
cbar=True
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=10, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

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


#################################### Frail vs Non-frail Correlation btw age and all motility features ####################################

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list( df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']) )

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

    for batch in np.unique(df['Patient']):
        df_part = df[df['Patient'] == batch].reset_index(drop=True)
        medias = np.unique(df_part['Condition'])

        for media in medias:
            df_part_part = df_part[df_part['Condition'] == media].reset_index(drop=True)
            age = np.unique( df_part_part['Age'] )[0]
            age_type = np.unique( df_part_part['Type2'] )[0]
            mean_value = np.mean(np.array(df_part_part[feature]))

            ages.append(age)
            age_types.append(age_type)
            mean_values.append(mean_value)
            media_types.append(media)

    datasets['Age'] = np.array(ages)
    datasets['Value'] = np.array(mean_values)
    datasets['Type2'] = np.array(age_types)
    datasets['Condition'] = np.array(media_types)

    df_ = pd.DataFrame(datasets)

    for type in ['Non-frail', 'Frail']:
        condition_rs = []
        condition_ps = []

        for cond in np.unique(df['Condition']):
            df_part_= df_[ (df_['Type2']==type) & (df_['Condition']==cond) ].reset_index(drop=True)
            #sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)
            r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
            #print(type, cond, r, p)
            condition_rs.append(r)
            condition_ps.append(p)
        each_rs = pd.DataFrame(condition_rs, columns=[feature], index='%s ' % type + np.unique(df['Condition']))
        each_ps = pd.DataFrame(condition_ps, columns=[feature], index='%s ' % type + np.unique(df['Condition']))
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
kws = dict(cbar_kws=dict(ticks=[0.5, 0, -0.5], orientation='horizontal'), vmin=-0.5, vmax=0.5 )

g=sns.clustermap(df_r_filtered, annot=False, cmap=cmc.vik, col_cluster=True, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (14, 4),
dendrogram_ratio=0.1,
cbar=True
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=8, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation Coefficient', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# for spine in g.ax_cbar.spines:
#     #g.ax_cbar.spines[spine].set_color('crimson')
#     g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'correlation with non-frail vs frail for all features.png', dpi=300,bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/' + 'correlation with non-frail vs frail for all features.svg', bbox_inches='tight')
plt.clf()
plt.close()




row_cluster=False
col_cluster=True

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
kws = dict(cbar_kws=dict(ticks=[0.6, 0, -0.6], orientation='horizontal'), vmin=-0.6, vmax=0.6 )

clustergrid=sns.clustermap(df_r, annot=False, cmap=cmc.vik, col_cluster=col_cluster, row_cluster=row_cluster,
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

row_order = clustergrid.dendrogram_row.reordered_ind if row_cluster else range(len(df_r.index))
col_order = clustergrid.dendrogram_col.reordered_ind if col_cluster else range(len(df_r.columns))
df_r = df_r.iloc[row_order, col_order]
df_p = df_p.iloc[row_order, col_order]
plt.clf()

fig, ax = plt.subplots(figsize=(24,7))

N, M = df_r.shape
ylabels = list(df_r.index)
xlabels = list(df_r.columns)

x, y = np.meshgrid(np.arange(M), np.arange(N))
s = df_p.values  # size column
c = df_r.values

R = s / s.max() / 2
circles = [plt.Circle((j, i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
from matplotlib.collections import PatchCollection
col = PatchCollection(circles, array=c.flatten(), cmap='OrRd',
                      norm=matplotlib.colors.Normalize(vmin=None, vmax=None))
ax.add_collection(col)

ax.set(xticks=np.arange(M), yticks=np.arange(N),
       xticklabels=xlabels, yticklabels=ylabels)
ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
# ax.grid(which='minor')

ax.set_facecolor((1, 1, 1, 0))
#ax.grid(b=None)
ax.grid(which='minor')
# fig.colorbar(col)
plt.gca().set_aspect('equal', adjustable='box')
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
plt.xticks(fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2', weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')

plt.savefig(path + 'Bubble plot correlation with age for all features.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/Bubble plot correlation with age for all features.svg', bbox_inches='tight')
plt.clf()
plt.close()

#################################### Correlation btw age and PC motility features ####################################

df.columns.get_loc('morpho_displ_autocorr_3')
from sklearn.decomposition import PCA
motility_data = df.iloc[:, 2:115].drop(['speed_distribution_x', 'speed_distribution_y'], axis=1)
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

    for type in ['Young', 'Old', 'Frail']:
        condition_rs = []
        condition_ps = []

        for cond in np.unique(df['Condition']):
            df_part_= df_[ (df_['Type']==type) & (df_['Condition']==cond) ].reset_index(drop=True)
            #sns.regplot(x='Age', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)
            r, p = scipy.stats.pearsonr(df_part_['Age'], df_part_['Value'])
            #print(type, cond, r, p)
            condition_rs.append(r)
            condition_ps.append(p)
        each_rs = pd.DataFrame(condition_rs, columns=[feature], index='%s ' % type + np.unique(df_pc['Condition']))
        each_ps = pd.DataFrame(condition_ps, columns=[feature], index='%s ' % type + np.unique(df_pc['Condition']))
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
kws = dict(cbar_kws=dict(ticks=[0.6, 0, -0.6], orientation='horizontal'), vmin=-0.6, vmax=0.6 )

g=sns.clustermap(df_r, annot=False, cmap=cmc.vik, col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (14, 5),
dendrogram_ratio=0.1,
cbar=True
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=10, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

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

############# Plot linear regression btw age and cluster enrichment ###############

condition_name = 'Age'
cluster_type = 'kmeans'
for age_group in np.unique(df['Type']):
    for condition in np.unique(df['Condition']):
        df_part = df[(df['Type']==age_group)&(df['Condition']==condition)].reset_index(drop=True)
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



############# Correlation btw agen and cluster enrichment for all clusters and conditions###############

condition_name = 'Age'
cluster_type = 'kmeans'

df_enrichment_r = pd.DataFrame()
df_enrichment_p = pd.DataFrame()

for age_group in ['Young', 'Old', 'Frail']:
    df_r_temp = pd.DataFrame()
    df_p_temp = pd.DataFrame()

    for condition in np.unique(df['Condition']):
        df_part = df[(df['Type']==age_group)&(df['Condition']==condition)].reset_index(drop=True)
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
kws = dict(cbar_kws=dict(ticks=[0.6, 0, -0.6], orientation='horizontal'), vmin=-0.6, vmax=0.6 )

g=sns.clustermap(df_enrichment_r, annot=df_enrichment_p,  fmt=".2f", cmap=cmc.vik, col_cluster=False, row_cluster=False,
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
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

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

g=sns.clustermap(df_enrichment_r, annot=False, cmap=cmc.vik, col_cluster=False, row_cluster=False,
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
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

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


############# Non-frail vs frail correlation btw agen and cluster enrichment for all clusters and conditions###############

condition_name = 'Age'
cluster_type = 'kmeans'

df_enrichment_r = pd.DataFrame()
df_enrichment_p = pd.DataFrame()

for age_group in ['Non-frail', 'Frail']:
    df_r_temp = pd.DataFrame()
    df_p_temp = pd.DataFrame()

    for condition in np.unique(df['Condition']):
        df_part = df[(df['Type2']==age_group)&(df['Condition']==condition)].reset_index(drop=True)
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
kws = dict(cbar_kws=dict(ticks=[0.6, 0, -0.6], orientation='horizontal'), vmin=-0.6, vmax=0.6 )

g=sns.clustermap(df_enrichment_r, annot=df_enrichment_p,  fmt=".2f", cmap=cmc.vik, col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (4, 3),
dendrogram_ratio=0.1,
cbar=True
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=12, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*3, 0.02])

g.ax_cbar.set_title('Correlation Coefficient', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# for spine in g.ax_cbar.spines:
#     #g.ax_cbar.spines[spine].set_color('crimson')
#     g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'Non-frail vs frail p value annotated Heatmap Age correlation for clusters.png', dpi=300,bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/' + 'Non-frail vs frail p value annotated Heatmap Age correlation for clusters.svg', bbox_inches='tight')
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

g=sns.clustermap(df_enrichment_r, annot=False, cmap=cmc.vik, col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (4, 3),
dendrogram_ratio=0.1,
cbar=True
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=12, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*3, 0.02])

g.ax_cbar.set_title('Correlation Coefficient', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# for spine in g.ax_cbar.spines:
#     #g.ax_cbar.spines[spine].set_color('crimson')
#     g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'Non-frail vs frail Heatmap Age correlation for clusters.png', dpi=300,bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/' + 'Non-frail vs frail Heatmap Age correlation for clusters.svg', bbox_inches='tight')
plt.clf()
plt.close()






row_cluster=False
col_cluster=False

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
kws = dict(cbar_kws=dict(ticks=[0.6, 0, -0.6], orientation='horizontal'), vmin=-0.6, vmax=0.6 )

clustergrid=sns.clustermap(df_enrichment_r, annot=False, cmap=cmc.vik, col_cluster=col_cluster, row_cluster=row_cluster,
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

row_order = clustergrid.dendrogram_row.reordered_ind if row_cluster else range(len(df_enrichment_r.index))
col_order = clustergrid.dendrogram_col.reordered_ind if col_cluster else range(len(df_enrichment_r.columns))
df_enrichment_r = df_enrichment_r.iloc[row_order, col_order]
df_enrichment_p = df_enrichment_p.iloc[row_order, col_order]
plt.clf()

fig, ax = plt.subplots(figsize=(7,7))

N, M = df_enrichment_r.shape
ylabels = list(df_enrichment_r.index)
xlabels = list(df_enrichment_r.columns)

x, y = np.meshgrid(np.arange(M), np.arange(N))
s = df_enrichment_p.values  # size column
c = df_enrichment_r.values

R = s / s.max() / 2
circles = [plt.Circle((j, i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
from matplotlib.collections import PatchCollection
col = PatchCollection(circles, array=c.flatten(), cmap='OrRd',
                      norm=matplotlib.colors.Normalize(vmin=None, vmax=None))
ax.add_collection(col)

ax.set(xticks=np.arange(M), yticks=np.arange(N),
       xticklabels=xlabels, yticklabels=ylabels)
ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
# ax.grid(which='minor')

ax.set_facecolor((1, 1, 1, 0))
#ax.grid(b=None)
ax.grid(which='minor')
# fig.colorbar(col)
plt.gca().set_aspect('equal', adjustable='box')
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
plt.xticks(fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2', weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')

plt.savefig(path + 'Bubble plot Age correlation for clusters.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/Bubble plot Age correlation for clusters.svg', bbox_inches='tight')
plt.clf()
plt.close()



