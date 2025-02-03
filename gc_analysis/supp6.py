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
"""Generates Data for Supp6. wt and mt GCB interaction with FDC NOI vs FDC PC"""

from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *

#################################### Pull out interaction features ####################################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/feature_csvs/'
df_lv = pd.read_parquet(path+'GCB_all_features_20.parquet')

interaction_features = []
overlapped_volume_features = []
shortest_distance_features = []
for column_name in df_lv.columns:
    if any(txt in column_name for txt in ('Overlapped',
                                          'Shortest_Distance')):  # Search whether 'Overlapped', or 'Shortest_Distance' word is in the column_name
        interaction_features.append(column_name)
    if ('Overlapped_Volume_Ratio' in column_name) and ('instant' not in column_name):
        overlapped_volume_features.append(column_name)

    if ('Shortest_Distance' in column_name) and ('instant' not in column_name):  # Search whether 'Overlapped', or 'Shortest_Distance' word is in the column_name
        shortest_distance_features.append(column_name)

df_interaction = pd.DataFrame()
tp_interactions = []
#df_interaction = pd.DataFrame()
for i in range(df_lv.shape[0]):
    each_row_interactions = pd.DataFrame()
    for feature_name in shortest_distance_features: # for each column
        each_row = df_lv.iloc[i, df_lv.columns.get_loc(feature_name)] # iloc[row_idx, column_idx]
        each_row_interactions = pd.concat([ each_row_interactions, pd.DataFrame(each_row, columns=[feature_name] )], axis=1)
        # each_row_interactions = (duration=20, Shortest Distance to Surface 1~n)
    tp_interaction = 0
    for t in range(0, each_row_interactions.shape[0]):
        each_time = each_row_interactions.iloc[t,:]
        if (each_time == 0).any() == True: # any returns True when at least one is True
            tp_interaction = tp_interaction + 1
    tp_interactions.append(tp_interaction)
df_interaction['tp_interaction_whole'] = tp_interactions
df = pd.concat([df_lv, df_interaction], axis = 1)

for feature_name in shortest_distance_features: # for each column
    tp_int_types = []
    for i in range(df_lv.shape[0]): # for each row
        each_row = df_lv.iloc[i, df_lv.columns.get_loc(feature_name)]
        tp_int_type = 0
        for t in range(each_row.shape[0]):
            each_time = each_row[t]
            if each_time == 0:
                tp_int_type = tp_int_type + 1
        tp_int_types.append(tp_int_type)

        #if each_row == 20:

    df_interaction['tp_interaction_' + feature_name[feature_name.find('=')+1:]] = tp_int_types

#for i in range(df_interaction.shape[0]):
int_list = []
for i in range(0,df_interaction.shape[0]):
    int_type = 'None'
    for feature_name in df_interaction.iloc[:,1:].columns:
        each_row = df_interaction.iloc[i, df_interaction.columns.get_loc(feature_name)]
        if each_row == 20:
            int_type = feature_name[feature_name.find('interaction')+12:]
    int_list.append(int_type)

df_interaction['interacted_type'] = int_list

df = pd.concat([df_lv, df_interaction], axis=1)

average_distances_to_FDC = []
average_distances_to_T = []
for i in range(df.shape[0]):
    avg_distance_FDC = np.mean(df[shortest_distance_features[0]][i])
    avg_distance_T = np.mean(df[shortest_distance_features[3]][i])
    average_distances_to_FDC.append(avg_distance_FDC)
    average_distances_to_T.append(avg_distance_T)
df['Average_distance_to_FDC'] = average_distances_to_FDC
df['Average_distance_to_T'] = average_distances_to_T

videos = np.unique(df['Video'])
df = df[(df['Video'] != videos[0])&(df['Video'] != videos[1])&(df['Video'] != videos[2])].reset_index(drop=True)

############# Plot NOI vs PC contour map for each cell type ###############

for cell_type in ['wt_B-cell', 'mt_B-cell']:

    fig, ax = plt.subplots(figsize=(2, 2))  # 2 inch by 2 inch

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    color_list = ['Greys', 'Reds', 'Greens', 'Blues', 'Oranges', 'Purples', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
                  'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'plasma']
    x_name='PC1'
    y_name='PC2'
    num_contours=6
    bin_num=50

    contours = []
    groups = ['NOI', 'PC']

    x0 = df[(df['Type']==cell_type) & (df['tp_interaction_whole'] == 0)][x_name]
    y0 = df[(df['Type']==cell_type) & (df['tp_interaction_whole'] == 0)][y_name]

    x20 = df[(df['Type']==cell_type) & (df['tp_interaction_whole'] == 20)][x_name]
    y20 = df[(df['Type']==cell_type) & (df['tp_interaction_whole'] == 20)][y_name]

    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    kde_coordinate0 = np.vstack([x0, y0])  # shape = (2(dimension), number of points)
    kde_coordinate20 = np.vstack([x20, y20])  # shape = (2(dimension), number of points)
    if (kde_coordinate0.shape[1] <= 2) or (kde_coordinate20.shape[1] <= 2):  # if there is only few points, it cannot calculate gaussian kde
        raise ValueError('Number of points should be greater than 2 to create contour')
    else:
        kde0 = scipy.stats.gaussian_kde(kde_coordinate0)  # Define kernel (bandwidth by Scott's Rule)
        kde20 = scipy.stats.gaussian_kde(kde_coordinate20)  # Define kernel (bandwidth by Scott's Rule)

        # evaluate on a regular grid
        xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
        ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        # Xgrid , Ygrid = (bin_num,bin_num) 2d array
        # Xgrid[i] = xgrid coordinate(divide by 100 from -x ~ +x` and repeat in row direction)
        # Ygrid[:,i] = ygrid coordinate(divide by 100 from -y ~ +y` and repeat in column direction)
        Z0 = kde0.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z20 = kde20.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        # Xgrid.ravel() = bin_num^2 shape 1d vector (vector that repeat -x ~ +x`, -x ~ +x` bin_num times)
        # np.vstack() = (2,bin_num^2) 2d array linspaced coordinates
        # Z = (10000,) 1d vector
        pdf0 = Z0.reshape(Xgrid.shape)
        pdf20 = Z20.reshape(Xgrid.shape)
        contour0 = ax.contour(Xgrid, Ygrid, pdf0,
                             # colors='red',
                             linewidths=1,
                             linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                             # label=group,
                             cmap=color_list[0],
                             origin='lower',
                             levels=num_contours,
                             )
        contour20 = ax.contour(Xgrid, Ygrid, pdf20,
                             # colors='red',
                             linewidths=1,
                             linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                             # label=group,
                             cmap=color_list[1],
                             origin='lower',
                             levels=num_contours,
                             )
        contours.append(contour0)
        contours.append(contour20)

    format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
    ax.legend(handles=[contour.legend_elements()[0][-1] for contour in contours], labels=groups,
              bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0, fontsize=3, frameon=False, markerscale=0.3)
    #plt.title('%s NOI vs PI' % cell_type, fontsize=25)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig(path + '%s_0 vs 20.png' % cell_type, dpi=300)
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s_0 vs 20.png.svg' % cell_type)
    plt.close()
    plt.clf()

############# Plot NOI vs PC FDC contour map for each cell type ###############

for cell_type in ['wt_B-cell', 'mt_B-cell']:

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    color_list = ['Greys', 'Reds']
    x_name='PC1'
    y_name='PC2'
    num_contours=6
    bin_num=50

    fig, ax = plt.subplots(figsize=(2, 2))  # 2 inch by 2 inch

    contours = []
    groups = ['NOI', 'PC']

    interaction_list = ['wt_B-cell', 'mt_B-cell', 'FDC', 'macrophage', 'T-cell']
    interaction_type = 'FDC'
    interaction_list.remove(cell_type)
    interaction_list.remove(interaction_type)

    x0 = df[(df['Type']==cell_type) & (df['tp_interaction_FDC'] == 0)][x_name]
    y0 = df[(df['Type']==cell_type) & (df['tp_interaction_FDC'] == 0)][y_name]

    # x20 = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] == 20) & (df['tp_interaction_%s' % interaction_list[0]] <= 5)
    #              & (df['tp_interaction_%s' % interaction_list[1]] <= 5) & (df['tp_interaction_%s' % interaction_list[2]] <= 5)][x_name]
    #
    # y20 = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] == 20) & (df['tp_interaction_%s' % interaction_list[0]] <= 5)
    #          & (df['tp_interaction_%s' % interaction_list[1]] <= 5) & (df['tp_interaction_%s' % interaction_list[2]] <= 5)][y_name]

    x20 = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] == 20)][x_name]
    y20 = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] == 20)][y_name]

    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    kde_coordinate0 = np.vstack([x0, y0])  # shape = (2(dimension), number of points)
    kde_coordinate20 = np.vstack([x20, y20])  # shape = (2(dimension), number of points)
    if (kde_coordinate0.shape[1] <= 2) or (kde_coordinate20.shape[1] <= 2):  # if there is only few points, it cannot calculate gaussian kde
        raise ValueError('Number of points should be greater than 2 to create contour')
    else:
        kde0 = scipy.stats.gaussian_kde(kde_coordinate0)  # Define kernel (bandwidth by Scott's Rule)
        kde20 = scipy.stats.gaussian_kde(kde_coordinate20)  # Define kernel (bandwidth by Scott's Rule)

        # evaluate on a regular grid
        xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
        ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        # Xgrid , Ygrid = (bin_num,bin_num) 2d array
        # Xgrid[i] = xgrid coordinate(divide by 100 from -x ~ +x` and repeat in row direction)
        # Ygrid[:,i] = ygrid coordinate(divide by 100 from -y ~ +y` and repeat in column direction)
        Z0 = kde0.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z20 = kde20.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        # Xgrid.ravel() = bin_num^2 shape 1d vector (vector that repeat -x ~ +x`, -x ~ +x` bin_num times)
        # np.vstack() = (2,bin_num^2) 2d array linspaced coordinates
        # Z = (10000,) 1d vector
        pdf0 = Z0.reshape(Xgrid.shape)
        pdf20 = Z20.reshape(Xgrid.shape)
        contour0 = ax.contour(Xgrid, Ygrid, pdf0,
                             # colors='red',
                             linewidths=1,
                             linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                             # label=group,
                             cmap=color_list[0],
                             origin='lower',
                             levels=num_contours,
                             )
        contour20 = ax.contour(Xgrid, Ygrid, pdf20,
                             # colors='red',
                             linewidths=1,
                             linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                             # label=group,
                             cmap=color_list[1],
                             origin='lower',
                             levels=num_contours,
                             )
        contours.append(contour0)
        contours.append(contour20)

    format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
    ax.legend(handles=[contour.legend_elements()[0][-1] for contour in contours], labels=groups,
              bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0, fontsize=3, frameon=False, markerscale=0.3)
    #plt.title('%s NOI vs PI' % cell_type, fontsize=25)
    plt.savefig(path + '%s_FDC_0 vs 20.png' % cell_type, dpi=300)
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s_FDC_0 vs 20.png.svg' % cell_type)
    plt.close()
    plt.clf()

############# Plot NOI vs PC cross correlation for each cell type ###############
font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

condition_name = 'Type'
cluster_type = 'tskmeans'
interaction_type = 'FDC'
#for cell_type in ['wt_B-cell', 'mt_B-cell']:
group_clones = []
df_corr_data = pd.DataFrame()

for cell_type in ['wt_B-cell', 'mt_B-cell']:
    df_corr_data_temp = pd.DataFrame()
    for group in [0, 20]:
        corrcoef = []
        if group == 0:
            aaa = df[(df['Type'] == cell_type)&(df['tp_interaction_FDC'] == 0)]
        if group == 20:
            interaction_list = ['wt_B-cell', 'mt_B-cell', 'FDC', 'macrophage', 'T-cell']
            interaction_list.remove(cell_type)
            interaction_list.remove(interaction_type)
            # aaa = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] == 20) & (df['tp_interaction_%s' % interaction_list[0]] <= 5)
            #          & (df['tp_interaction_%s' % interaction_list[1]] <= 5) & (df['tp_interaction_%s' % interaction_list[2]] <= 5)]
            aaa = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] == 20)]
        #print(cell_type, interaction_type, interaction_list, aaa.shape)
        group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
        group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
        group_clone = group_clone.unstack(level=0)
        group_clone[np.isnan(group_clone)] = 0
        group_clone_T = group_clone.T
        for cluster in sorted( list( pd.unique(df[cluster_type]) ) ):
            if cluster in group_clone_T.columns:
                continue
            else:
                group_clone_T.insert(loc=int(cluster), column=cluster, value=0)
                group_clone_T.sort_index(axis=1, inplace=True)

        group_clone = group_clone_T.T
        for column in group_clone.columns:
            group_clone.rename(columns={column:cell_type+'_%s'%group}, inplace=True)
        group_clones.append(group_clone)
        df_corr_data_temp = pd.concat([df_corr_data_temp, group_clone], axis=1)
    df_corr_data = pd.concat([df_corr_data, df_corr_data_temp], axis=1)

df_corr = df_corr_data.corr(method='spearman')

df_corr.columns.name = None # Remove name of column = 'Type'
df_corr.index.name = None # Remove name of index = 'Type'

rename_keys = {'FDC_0': 'NOI', 'FDC_20': '%s FDC PC'%(cell_type), 'T-cell_0': 'NOI', 'T-cell_20': '%s Tfh PC'%(cell_type),}
df_corr.rename(columns=rename_keys, inplace=True)
df_corr.rename(index=rename_keys, inplace=True)

mask = np.triu(df_corr) # Mask for only lower triangle

mask = mask[1:, :-1]  # mask that doesn't contain very top left and very bottom right block
corr = df_corr.iloc[1:,:-1].copy()  # df that doesn't contain very top left and very bottom right block

fig, ax = plt.subplots(figsize=(2, 2))

ax = sns.heatmap(corr, annot=True, mask=mask, cmap='OrRd', alpha=0.9, linewidths=1, linecolor='white', fmt=".2f", square= True,
                 annot_kws={'size': 4, 'weight':'bold'},
                 cbar_kws= {"shrink":0.7, 'label':'Correlation'})

cax = ax.figure.axes[-1]  # colorbar
cax.tick_params(labelsize=4)  # fontsize of tick label
cax.yaxis.label.set_size(6)  # fontsize of color bar y label

yticks = [i for i in corr.index]
xticks = [i for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=4, weight='bold')
plt.xticks(plt.xticks()[0], labels=xticks, fontsize=4, rotation=35, rotation_mode='anchor', ha='right', weight='bold')

plt.savefig(path+'NOI vs PC correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/NOI vs PC correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

############# Plot NOI vs PI fraction of cluster for each cell type ###############
vmax=40
#colors = ('#44AA99', '#6699CC', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')
if cell_type == 'mt_B-cell':
    color='#CC6677'
elif cell_type == 'wt_B-cell':
    color = '#888888'

for group_clone in group_clones:
    for i, cond in enumerate(list(group_clone.columns)):
        if 'mt_B-cell' in cond:
            color = '#CC6677'
        elif 'wt_B-cell' in cond:
            color = '#888888'
        fig, ax = plt.subplots(figsize=(2, 2))
        ax = sns.barplot(x=list(group_clone[cond].index), y=list(group_clone[cond]), capsize=0.5, edgecolor='0.2', lw=1, errwidth=1, color=color)

        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color('0.2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        #sns.barplot(x=list(group_clone[cond].index), y=list(group_clone[cond]), color=colors[i])
        #plt.xlabel('%s' % cluster_type)
        plt.ylabel('Occurence (%)')
        plt.ylim(0, vmax)
        plt.savefig(path + '%s_distribution.png' % (group_clone.columns.values[0]), dpi=300, bbox_inches='tight')
        if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/')
        plt.savefig(path + 'svg/%s_distribution.svg' % (group_clone.columns.values[0]), bbox_inches='tight')
        plt.clf()
        plt.close()

###################### Plot NOI vs PI feature bar graph for GCB cell ############################

if not os.path.isdir(path + 'NOI vs PC motility violin plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'NOI vs PC motility violin plot/')

feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    dataset={}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        ref_data = df[(df['Type'] == cell_type)&(df['tp_interaction_whole'] == 0)][feature_name]
        dataset['%s_NOI'%cell_type]=np.array(ref_data)
        interaction_type='FDC'
        interaction_list = ['wt_B-cell', 'mt_B-cell', 'FDC', 'macrophage', 'T-cell']
        interaction_list.remove(cell_type)
        interaction_list.remove(interaction_type)
        # data = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] == 20) & (df['tp_interaction_%s' % interaction_list[0]] <= 5)
        #          & (df['tp_interaction_%s' % interaction_list[1]] <= 5) & (df['tp_interaction_%s' % interaction_list[2]] <= 5)][feature_name]
        data = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] == 20)][feature_name]
        dataset[cell_type+'-'+interaction_type] = np.array(data)

    rename_keys = {'wt_B-cell_NOI': 'wt GCB NOI', 'wt_B-cell-FDC': 'wt GCB PC',
                   'mt_B-cell_NOI': 'mt GCB NOI','mt_B-cell-FDC': 'mt GCB PC'}
    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_violin_plot(dataset_renamed, path + 'NOI vs PC motility violin plot/', file_name=feature_name,
                            colors=('#888888', '#888888', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))

#################################### Volcano plot of NOI vs PC motility features ####################################
feature_list = df.columns[130:283].drop(['phi', 'speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    df_p = pd.DataFrame()
    for feature_name in feature_list:
        dataset = {}
        ref_data = df[(df['Type'] == cell_type) & (df['tp_interaction_whole'] == 0)][feature_name]
        dataset['%s_0' % cell_type] = np.array(ref_data)
        interaction_type = 'FDC'
        interaction_list = ['wt_B-cell', 'mt_B-cell', 'FDC', 'macrophage', 'T-cell']
        interaction_list.remove(cell_type)
        interaction_list.remove(interaction_type)
        # data = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] == 20) & (df['tp_interaction_%s' % interaction_list[0]] <= 5)
        #           & (df['tp_interaction_%s' % interaction_list[1]] <= 5) & (df['tp_interaction_%s' % interaction_list[2]] <= 5)][feature_name]
        data = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] == 20)][feature_name]
        dataset['%s_20' % cell_type] = np.array(data)

        pvalue = get_pvalue(dataset, test='mann-whitney')
        logp = -np.log10(pvalue)

        avgZ = get_avgZ(dataset, ref_name=cell_type+'_0', data_name=cell_type+'_20')

        row = pd.DataFrame()
        row['Feature'] = [feature_name]
        row['Pvalue'] = [pvalue]
        row['-Logp'] = [logp]
        row['AvgZ'] = [avgZ]
        df_p = pd.concat([df_p, row], axis=0)

    df_p['Adj_p'] = adjusted_pvalues(df_p['Pvalue'], correction_type='Benjamini-Hochberg')
    df_p['Adj_Logp'] = -np.log(df_p['Adj_p'])
    df_p = df_p[~df_p.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf


    draw_volcano_plot(df_p, path, file_name='%s NOI vs PC motility volcano plot'%cell_type, z_thresh=0.3, p_thresh=1, z_name='AvgZ', p_name='Adj_Logp',
                      feature_name='Feature', figsize=(6,6))

#################################### Box plot comparing all interaction features by cell type ####################################
df.columns.get_loc('FDC_total_distance')

feature_list = df.columns[324:].drop(['interacted_type'])
condition_name = 'Type'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
if not os.path.isdir(path + 'interaction_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'interaction_violin_plot_type/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    values = flatten_nested_dict(dataset)
    if np.isnan(values).any()==True:  # Check at least one nan
        continue
    elif np.isfinite(values).all()==False:  # Check everything is not inf
        continue
    else:
        new_order = ['wt_B-cell', 'mt_B-cell']
        ordered_dataset = change_dict_order(dataset, new_order)
        dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }
        draw_custom_violin_plot(dict_datasets, path + 'interaction_violin_plot_type/', file_name=feature_name,
        colors=('#888888', '#CC6677'),
        test='mann-whitney', pvalue=True, figsize=(1, 2))