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
"""Generates Data for Figure3. wt and mt GCB interaction with Tfh"""

from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import DistanceSignal

#################################### Pull out interaction features ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GCB_no_inhibit_all_features_40.parquet')
df_duration = pd.read_parquet(path+'GCB_no_inhibit_traj_duration_40.parquet')


_, _, Zone_series = to_timeseries_fast(df_duration, duration=40, feature_name='Zone')

feature_list = ['average']
FDC_dist = DistanceSignal(Zone_series)
df_distance = FDC_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'Zone_average'}, inplace=True)

df['Zone_average'] = df_distance


df.loc[(df['Zone_average'] < 0.4) & (df['Zone_average'] >= 0), 'Zone'] = 'DZ'
df.loc[(df['Zone_average'] < 0.8) & (df['Zone_average'] >= 0.4), 'Zone'] = 'DZ-sLZ'
df.loc[(df['Zone_average'] < 1.2) & (df['Zone_average'] >= 0.8), 'Zone'] = 'sLZ'
df.loc[(df['Zone_average'] < 1.6) & (df['Zone_average'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
df.loc[(df['Zone_average'] <= 2) & (df['Zone_average'] >= 1.6), 'Zone'] = 'dLZ'


print(df[df['Zone']=='DZ'].shape[0], df[df['Zone']=='DZ-sLZ'].shape[0], df[df['Zone']=='sLZ'].shape[0],
      df[df['Zone']=='sLZ-dLZ'].shape[0], df[df['Zone']=='dLZ'].shape[0])

duration=40
label_expanded = np.repeat(df['Zone'], duration).reset_index(drop=True)
df_duration['Zone_label'] = label_expanded

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure3. Interaction with Tfh of wt and mt GCB(long duration)\\'

#################################### all motility features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
df.columns.get_loc('quality_FDC_approach_times')
df.columns.get_loc('Core_diff_distance_autocorr_3')

motility_data = df.iloc[:,148:280].drop(['FDC_diff_distance_cov', 'T_diff_distance_cov',
                                         'Core_diff_distance_cov', 'LZ_diff_distance_cov', 'DZ_diff_distance_cov',
                                         'DZ_distance_autocorr_1', 'DZ_distance_autocorr_2', 'DZ_distance_autocorr_3',
                                         'DZ_diff_distance_autocorr_1', 'DZ_diff_distance_autocorr_2', 'DZ_diff_distance_autocorr_3',
                                         'LZ_diff_distance_autocorr_1', 'LZ_diff_distance_autocorr_2', 'LZ_diff_distance_autocorr_3',
                                         'Core_distance_autocorr_1', 'Core_distance_autocorr_2', 'Core_distance_autocorr_3',
                                         'Core_diff_distance_autocorr_1', 'Core_diff_distance_autocorr_2', 'Core_diff_distance_autocorr_3',
                                         'Core_distance_variance', 'Core_diff_distance_variance', 'DZ_distance_variance', 'DZ_diff_distance_variance',
                                         'FDC_diff_distance_variance', 'FDC_distance_variance', 'LZ_distance_variance', 'LZ_diff_distance_variance',
                                         'T_diff_distance_variance', 'T_distance_variance',
                                         ], axis=1)
columns_with_nan = motility_data.columns[motility_data.isna().any()].tolist()
feature_list = motility_data.drop(columns_with_nan, axis=1).columns

xlabel = 'Number of Tfh Contacts'
draw_lineplot_by_custom_ranges(df_, path, folder_name='int_feature_wrt_T_contacts', feature_list=feature_list,
                               condition_name='Type', custsom_range=(0, 30), stepsize=2, range_feature='T_contact_times',
                                   color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label=xlabel,
                               estimator='mean', error_type='ci_norm', replace_keys=None, pvalue=True, test='mann-whitney')

xlabel = 'Number of Tfh Persistent Contacts'
draw_lineplot_by_custom_ranges(df_, path, folder_name='int_feature_wrt_T_contact_persistence', feature_list=feature_list,
                               condition_name='Type', custsom_range=(0, 30), stepsize=2, range_feature='T_contact_persistences',
                                   color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label=xlabel,
                               estimator='mean', error_type='ci_norm', replace_keys=None, pvalue=True, test='mann-whitney')

####################################### Quantify Tfh interaction frequency #############################################
int_time = 30
test = 'mann-whitney'

persistent_int_freqs_datasets = {}
persistent_int_freq_per_cellnumbers_datasets = {}
total_n_contacts_per_cellnumbers_datasets = {}
persistent_int_freq_per_cellcontacts_datasets = {}
low_contact_freq_per_cellnumbers_datasets = {}

videos = np.unique(df['Video'])
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    df_part = df[df['Type'] == cell_type]
    persistent_int_freqs = []
    persistent_int_freq_per_cellnumbers = []
    total_n_contacts_per_cellnumbers = []
    persistent_int_freq_per_cellcontacts = []
    low_contact_freq_per_cellnumbers = []
    for video in videos:
        #if 'A' in video and cell_type == 'mt_B-cell':
        df_video = df_part[df_part['Video'] == video].reset_index(drop=True)
        if df_video.shape[0] == 0:
            continue
        data = df_video['T_contact_times']
        mask = ~np.isnan(data)
        data = data[mask]

        persistent_int_freq = sum(data >= int_time)
        total_n_contact = sum(data)
        n_trajs = df_video.shape[0]


        persistent_int_freq_per_cellnumber = persistent_int_freq / n_trajs
        total_n_contacts_per_cellnumber = total_n_contact / n_trajs
        persistent_int_freq_per_cellcontact = persistent_int_freq / total_n_contact
        low_contact_freq_per_cellnumber = sum((1<=data) & (data<=5)) / n_trajs

        if persistent_int_freq != 0:
            persistent_int_freqs.append(persistent_int_freq)
            persistent_int_freq_per_cellnumbers.append(persistent_int_freq_per_cellnumber)
            persistent_int_freq_per_cellcontacts.append(persistent_int_freq_per_cellcontact)

        total_n_contacts_per_cellnumbers.append(total_n_contacts_per_cellnumber)
        low_contact_freq_per_cellnumbers.append(low_contact_freq_per_cellnumber)

    persistent_int_freqs_datasets[cell_type] = persistent_int_freqs
    persistent_int_freq_per_cellnumbers_datasets[cell_type] = persistent_int_freq_per_cellnumbers
    total_n_contacts_per_cellnumbers_datasets[cell_type] = total_n_contacts_per_cellnumbers
    persistent_int_freq_per_cellcontacts_datasets[cell_type] = persistent_int_freq_per_cellcontacts
    low_contact_freq_per_cellnumbers_datasets[cell_type] = low_contact_freq_per_cellnumbers

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
new_order = ['wt_B-cell', 'mt_B-cell']
persistent_int_freqs_datasets = change_dict_order(persistent_int_freqs_datasets, new_order)
persistent_int_freqs_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freqs_datasets.items() }

persistent_int_freq_per_cellnumbers_datasets = change_dict_order(persistent_int_freq_per_cellnumbers_datasets, new_order)
persistent_int_freq_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freq_per_cellnumbers_datasets.items() }

total_n_contacts_per_cellnumbers_datasets = change_dict_order(total_n_contacts_per_cellnumbers_datasets, new_order)
total_n_contacts_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in total_n_contacts_per_cellnumbers_datasets.items() }

persistent_int_freq_per_cellcontacts_datasets = change_dict_order(persistent_int_freq_per_cellcontacts_datasets, new_order)
persistent_int_freq_per_cellcontacts_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freq_per_cellcontacts_datasets.items() }

low_contact_freq_per_cellnumbers_datasets = change_dict_order(low_contact_freq_per_cellnumbers_datasets, new_order)
low_contact_freq_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in low_contact_freq_per_cellnumbers_datasets.items() }

colors=('#888888', '#CC6677')
draw_custom_bar_plot(persistent_int_freqs_datasets, path, file_name='Total Tfh interaction frequency',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellnumbers_datasets, path, file_name='Tfh persistent interaction frequency per cell number',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))
draw_custom_bar_plot(total_n_contacts_per_cellnumbers_datasets, path, file_name='number of Tfh contacts per cell number',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellcontacts_datasets, path, file_name='Tfh persistent interaction frequency per number of contacts',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))
draw_custom_bar_plot(low_contact_freq_per_cellnumbers_datasets, path, file_name='Tfh low contact time frequency per cell number',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))

####################################### interaction frequency differential for B cell each interaction time #############################################
freq_datasets = {}
freq_per_cellnumber_datasets = {}
videos = np.unique(df['Video'])
for t in range(31):
    freq_per_cellnumber_datasets_temp = {}
    for interaction_type in ['FDC', 'T']:
        for cell_type in ['mt_B-cell', 'wt_B-cell']:
            df_part = df[df['Type'] == cell_type]
            int_freqs = []
            int_freq_per_cellnumbers = []
            for video in videos:
                #if 'A' in video and cell_type == 'mt_B-cell':
                if '-A' in video:
                    continue
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] == 0:
                    continue
                data = df_video['%s_contact_times' % interaction_type]
                mask = ~np.isnan(data)
                data = data[mask]
                #int_freq = sum(data <= 10)
                int_freq_per_cellnumber = sum(data == t) / df_video.shape[0]
                #int_freqs.append(int_freq)
                int_freq_per_cellnumbers.append(int_freq_per_cellnumber)
            #freq_datasets[cell_type + '-' + interaction_type] = int_freqs
            freq_per_cellnumber_datasets_temp[cell_type + '-' + interaction_type] = int_freq_per_cellnumbers
    freq_per_cellnumber_datasets[t] = freq_per_cellnumber_datasets_temp

B_FDC_diffs = {}
B_T_diffs = {}
for t in freq_per_cellnumber_datasets:
    freq_per_cellnumber_data = freq_per_cellnumber_datasets[t]
    B_FDC_diff_temp = []
    B_T_diff_temp = []
    for video_idx in range(11):
        B_FDC_diff = freq_per_cellnumber_data['mt_B-cell-FDC'][video_idx] - freq_per_cellnumber_data['wt_B-cell-FDC'][video_idx]
        B_T_diff = freq_per_cellnumber_data['wt_B-cell-T'][video_idx] - freq_per_cellnumber_data['mt_B-cell-T'][video_idx]
        B_FDC_diff_temp.append(B_FDC_diff)
        B_T_diff_temp.append(B_T_diff)
    #B_FDC_diffs[t] = np.mean(B_FDC_diff_temp)
    #B_T_diffs[t] = np.mean(B_T_diff_temp)
    B_FDC_diffs[t] = B_FDC_diff_temp
    B_T_diffs[t] = B_T_diff_temp

colors=('#888888', '#CC6677')
draw_custom_bar_plot(B_T_diffs, path, file_name='All interaction time for B-T',
                     strip_plot=False, colors=colors, test='mann-whitney', pvalue=True, figsize=(10,10))

l = [B_FDC_diffs[t] for t in B_FDC_diffs if 1<=t<=3]
flat_list = [item for sublist in l for item in sublist]
dataset = {'Low interaction':flat_list, 'High interaction': B_FDC_diffs[20]}

draw_custom_bar_plot(dataset, path, file_name='Low interaction vs High interaction for B-FDC',
                     strip_plot=False, colors=colors, test='mann-whitney', pvalue=True, figsize=(1,2))

l = [B_T_diffs[t] for t in B_T_diffs if 1<=t<=3]
flat_list = [item for sublist in l for item in sublist]
dataset = {'Low interaction':flat_list, 'High interaction': B_T_diffs[20]}

draw_custom_bar_plot(dataset, path, file_name='Low interaction vs High interaction for B-T',
                     strip_plot=False, colors=colors, test='mann-whitney', pvalue=True, figsize=(1,2))

############# Plot NOI vs PI contour map for each cell type ###############

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
    interaction_type = 'T-cell'
    interaction_list.remove(cell_type)
    interaction_list.remove(interaction_type)

    x0 = df[(df['Type']==cell_type) & (df['tp_interaction_whole'] == 0)][x_name]
    y0 = df[(df['Type']==cell_type) & (df['tp_interaction_whole'] == 0)][y_name]

    # x20 = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] >= 15) & (df['tp_interaction_%s' % interaction_list[0]] <= 5)
    #              & (df['tp_interaction_%s' % interaction_list[1]] <= 5) & (df['tp_interaction_%s' % interaction_list[2]] <= 5)][x_name]
    #
    # y20 = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] >= 15) & (df['tp_interaction_%s' % interaction_list[0]] <= 5)
    #          & (df['tp_interaction_%s' % interaction_list[1]] <= 5) & (df['tp_interaction_%s' % interaction_list[2]] <= 5)][y_name]

    x20 = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] >= 15)][x_name]
    y20 = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] >= 15)][y_name]

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
    plt.savefig(path + '%s_0 vs 20.png' % cell_type, dpi=300)
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s_0 vs 20.png.svg' % cell_type)
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
interaction_type = 'T-cell'
#for cell_type in ['wt_B-cell', 'mt_B-cell']:
group_clones = []
df_corr_data = pd.DataFrame()

for cell_type in ['wt_B-cell', 'mt_B-cell']:
    df_corr_data_temp = pd.DataFrame()
    for group in [0, 20]:
        corrcoef = []
        if group == 0:
            aaa = df[(df['Type'] == cell_type)&(df['tp_interaction_whole'] == 0)]
        if group == 20:
            interaction_list = ['wt_B-cell', 'mt_B-cell', 'FDC', 'macrophage', 'T-cell']
            interaction_list.remove(cell_type)
            interaction_list.remove(interaction_type)
            # aaa = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] >= 15) & (df['tp_interaction_%s' % interaction_list[0]] <= 5)
            #          & (df['tp_interaction_%s' % interaction_list[1]] <= 5) & (df['tp_interaction_%s' % interaction_list[2]] <= 5)]
            aaa = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] >= 15)]
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
        interaction_type='T-cell'
        interaction_list = ['wt_B-cell', 'mt_B-cell', 'FDC', 'macrophage', 'T-cell']
        interaction_list.remove(cell_type)
        interaction_list.remove(interaction_type)
        # data = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] >= 15) & (df['tp_interaction_%s' % interaction_list[0]] <= 5)
        #          & (df['tp_interaction_%s' % interaction_list[1]] <= 5) & (df['tp_interaction_%s' % interaction_list[2]] <= 5)][feature_name]
        data = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] >= 15)][feature_name]

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
        interaction_type = 'T-cell'
        interaction_list = ['wt_B-cell', 'mt_B-cell', 'FDC', 'macrophage', 'T-cell']
        interaction_list.remove(cell_type)
        interaction_list.remove(interaction_type)
        data = df[(df['Type'] == cell_type) & (df['tp_interaction_%s' % interaction_type] >= 15) & (df['tp_interaction_%s' % interaction_list[0]] <= 5)
                  & (df['tp_interaction_%s' % interaction_list[1]] <= 5) & (df['tp_interaction_%s' % interaction_list[2]] <= 5)][feature_name]
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


    draw_volcano_plot(df_p, path, file_name='%s NOI vs PC motility volcano plot'%cell_type, z_thresh=0.5, p_thresh=2, z_name='AvgZ', p_name='Adj_Logp',
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

####################################### Quantify Tfh interaction frequency #############################################
int_time = 20

persistent_int_freqs_datasets = {}
persistent_int_freq_per_cellnumbers_datasets = {}
total_n_contacts_per_cellnumbers_datasets = {}
persistent_int_freq_per_cellcontacts_datasets = {}
low_contact_freq_per_cellnumbers_datasets = {}

videos = np.unique(df['Video'])
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    df_part = df[df['Type'] == cell_type]
    persistent_int_freqs = []
    persistent_int_freq_per_cellnumbers = []
    total_n_contacts_per_cellnumbers = []
    persistent_int_freq_per_cellcontacts = []
    low_contact_freq_per_cellnumbers = []
    for video in videos:
        #if 'A' in video and cell_type == 'mt_B-cell':
        if '-A' in video:
            continue
        df_video = df_part[df_part['Video'] == video]
        if df_video.shape[0] == 0:
            continue
        data = df_video['tp_interaction_T-cell']
        mask = ~np.isnan(data)
        data = data[mask]
        persistent_int_freq = sum(data >= int_time)
        total_n_contact = sum(data)

        persistent_int_freq_per_cellnumber = persistent_int_freq / df_video.shape[0]
        total_n_contacts_per_cellnumber = total_n_contact / df_video.shape[0]
        persistent_int_freq_per_cellcontact = persistent_int_freq / sum(data)
        low_contact_freq_per_cellnumber = sum((1<=data) & (data<=3)) / df_video.shape[0]

        persistent_int_freqs.append(persistent_int_freq)
        persistent_int_freq_per_cellnumbers.append(persistent_int_freq_per_cellnumber)
        total_n_contacts_per_cellnumbers.append(total_n_contacts_per_cellnumber)
        persistent_int_freq_per_cellcontacts.append(persistent_int_freq_per_cellcontact)
        low_contact_freq_per_cellnumbers.append(low_contact_freq_per_cellnumber)

    persistent_int_freqs_datasets[cell_type] = persistent_int_freqs
    persistent_int_freq_per_cellnumbers_datasets[cell_type] = persistent_int_freq_per_cellnumbers
    total_n_contacts_per_cellnumbers_datasets[cell_type] = total_n_contacts_per_cellnumbers
    persistent_int_freq_per_cellcontacts_datasets[cell_type] = persistent_int_freq_per_cellcontacts
    low_contact_freq_per_cellnumbers_datasets[cell_type] = low_contact_freq_per_cellnumbers

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
new_order = ['wt_B-cell', 'mt_B-cell']
persistent_int_freqs_datasets = change_dict_order(persistent_int_freqs_datasets, new_order)
persistent_int_freqs_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freqs_datasets.items() }

persistent_int_freq_per_cellnumbers_datasets = change_dict_order(persistent_int_freq_per_cellnumbers_datasets, new_order)
persistent_int_freq_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freq_per_cellnumbers_datasets.items() }

total_n_contacts_per_cellnumbers_datasets = change_dict_order(total_n_contacts_per_cellnumbers_datasets, new_order)
total_n_contacts_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in total_n_contacts_per_cellnumbers_datasets.items() }

persistent_int_freq_per_cellcontacts_datasets = change_dict_order(persistent_int_freq_per_cellcontacts_datasets, new_order)
persistent_int_freq_per_cellcontacts_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freq_per_cellcontacts_datasets.items() }

low_contact_freq_per_cellnumbers_datasets = change_dict_order(low_contact_freq_per_cellnumbers_datasets, new_order)
low_contact_freq_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in low_contact_freq_per_cellnumbers_datasets.items() }

colors=('#888888', '#CC6677')
draw_custom_bar_plot(persistent_int_freqs_datasets, path, file_name='Total T interaction frequency',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellnumbers_datasets, path, file_name='T persistent interaction frequency per cell number',
                     strip_plot=True,colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(total_n_contacts_per_cellnumbers_datasets, path, file_name='number of T contacts per cell number',
                    strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellcontacts_datasets, path, file_name='T persistent interaction frequency per number of contacts',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(low_contact_freq_per_cellnumbers_datasets, path, file_name='T low contact time frequency per cell number',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))




####################################### T interaction frequency kde mt GCB vs wt GCB #############################################
replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}

dataset={}
for condition in np.unique(df['Type']):
    data = df[df['Type'] == condition]['tp_interaction_T-cell']
    dataset[condition] = np.array(data)
new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dataset, new_order)
dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

colors=('#888888', '#CC6677')
fig, ax = plt.subplots(figsize=(2,2))
for i, key in enumerate(dict_datasets):
    ax = sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=(0, 40), color=colors[i], label=key)

# ax = sns.kdeplot(data=dict_datasets['wt GCB'], fill=True, linewidth=1, clip=(0, 40), color='#888888', label='wt GCB')
# ax = sns.kdeplot(data=dict_datasets['mt GCB'], fill=True, linewidth=1, clip=(0, 40), color='#CC6677', label='mt GCB')

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('T interaction frequency', fontsize=8, weight='bold', color='0.2')
ax.set_ylabel('Density', fontsize=8, weight='bold', color='0.2')
plt.xticks(fontsize=8, color='0.2', weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')

plt.legend(frameon=False, prop = {'weight':'bold', 'size':8}, labelcolor='0.2')

plt.savefig(path+'cell-T interaction frequency.png', dpi=300, bbox_inches='tight')
plt.close()
plt.clf()

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/cell-T interaction frequency.svg', bbox_inches='tight')
plt.close()
plt.clf()

####################################### Average T distance kde mt GCB vs wt GCB #############################################
replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}

dataset={}
for condition in np.unique(df['Type']):
    data = df[df['Type'] == condition]['Average_distance_to_T']
    dataset[condition] = np.array(data)
new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dataset, new_order)
dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

colors=('#888888', '#CC6677')
fig, ax = plt.subplots(figsize=(2,2))
for i, key in enumerate(dict_datasets):
    ax = sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=(0, 40), color=colors[i], label=key)

# ax = sns.kdeplot(data=dict_datasets['wt GCB'], fill=True, linewidth=1, clip=(0, 40), color='#888888', label='wt GCB')
# ax = sns.kdeplot(data=dict_datasets['mt GCB'], fill=True, linewidth=1, clip=(0, 40), color='#CC6677', label='mt GCB')

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Distance to Tfh (μm)', fontsize=8, weight='bold', color='0.2')
ax.set_ylabel('Density', fontsize=8, weight='bold', color='0.2')
plt.xticks(fontsize=8, color='0.2', weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')

plt.legend(frameon=False, prop = {'weight':'bold', 'size':8}, labelcolor='0.2')

plt.savefig(path+'cell-T average distance.png', dpi=300, bbox_inches='tight')
plt.close()
plt.clf()

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/cell-T average distance.svg', bbox_inches='tight')
plt.close()
plt.clf()

############ Plot Close vs Far contour map for each cell type ###############
for cell_type in ['mt_B-cell', 'wt_B-cell']:

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
    groups = ['far', 'close']

    x_far = df[(df['Type']==cell_type) & (df['Average_distance_to_T'] >= 15)][x_name]
    y_far = df[(df['Type']==cell_type) & (df['Average_distance_to_T'] >= 15)][y_name]

    x_close = df[(df['Type']==cell_type) & (df['Average_distance_to_T'] <= 5)][x_name]
    y_close = df[(df['Type']==cell_type) & (df['Average_distance_to_T'] <= 5)][y_name]

    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    kde_coordinate_far = np.vstack([x_far, y_far])  # shape = (2(dimension), number of points)
    kde_coordinate_close = np.vstack([x_close, y_close])  # shape = (2(dimension), number of points)
    if (kde_coordinate_far.shape[1] <= 2) or (kde_coordinate_close.shape[1] <= 2):  # if there is only few points, it cannot calculate gaussian kde
        raise ValueError('Number of points should be greater than 2 to create contour')
    else:
        kde_far = scipy.stats.gaussian_kde(kde_coordinate_far)  # Define kernel (bandwidth by Scott's Rule)
        kde_close = scipy.stats.gaussian_kde(kde_coordinate_close)  # Define kernel (bandwidth by Scott's Rule)

        # evaluate on a regular grid
        xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
        ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        # Xgrid , Ygrid = (bin_num,bin_num) 2d array
        # Xgrid[i] = xgrid coordinate(divide by 100 from -x ~ +x` and repeat in row direction)
        # Ygrid[:,i] = ygrid coordinate(divide by 100 from -y ~ +y` and repeat in column direction)
        Z_far = kde_far.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z_close = kde_close.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        # Xgrid.ravel() = bin_num^2 shape 1d vector (vector that repeat -x ~ +x`, -x ~ +x` bin_num times)
        # np.vstack() = (2,bin_num^2) 2d array linspaced coordinates
        # Z = (10000,) 1d vector
        pdf_far = Z_far.reshape(Xgrid.shape)
        pdf_close = Z_close.reshape(Xgrid.shape)
        contour_far = ax.contour(Xgrid, Ygrid, pdf_far,
                             # colors='red',
                             linewidths=1,
                             linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                             # label=group,
                             cmap=color_list[0],
                             origin='lower',
                             levels=num_contours,
                             )
        contour_close = ax.contour(Xgrid, Ygrid, pdf_close,
                             # colors='red',
                             linewidths=1,
                             linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                             # label=group,
                             cmap=color_list[1],
                             origin='lower',
                             levels=num_contours,
                             )
        contours.append(contour_far)
        contours.append(contour_close)

    format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
    ax.legend(handles=[contour.legend_elements()[0][-1] for contour in contours], labels=groups,
              bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0, fontsize=3, frameon=False, markerscale=0.3)
    #plt.title('%s NOI vs PI' % cell_type, fontsize=25)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig(path + '%s_far vs close.png' % cell_type, dpi=300)
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s_far vs close.svg' % cell_type)
    plt.close()
    plt.clf()


############# Plot Close vs Far cross correlation for each cell type ###############
font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

condition_name = 'Type'
cluster_type = 'tskmeans'
condition = 'tp_interaction_FDC'
groups = ['NOI', 'PC']

df_corr_data = pd.DataFrame()
group_clones=[]
for group in ['far', 'close']:
    corrcoef = []
    #aaa = df[df[condition] == group]
    if group == 'far':
        aaa = df[(df['Average_distance_to_T'] >= 15)]
    elif group == 'close':
        aaa = df[(df['Average_distance_to_T'] <= 5)]

    group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    group_clone_T = group_clone.T
    for cluster in list(pd.unique(df[cluster_type])):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=[0, 0, 0])
    group_clone = group_clone_T.T

    for column in group_clone.columns:
        group_clone.rename(columns={column:column+'_%s'%group}, inplace=True)
    group_clones.append(group_clone)
    df_corr_data = pd.concat([df_corr_data, group_clone], axis=1)

df_corr = df_corr_data.corr()

df_corr.columns.name = None # Remove name of column = 'Type'
df_corr.index.name = None # Remove name of index = 'Type'


rename_keys = {'wt_B-cell_far': 'wt GCB far', 'wt_B-cell_close': 'wt GCB close',
               'mt_B-cell_far': 'mt GCB far', 'mt_B-cell_close': 'mt GCB close',}
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

plt.savefig(path+'far vs close correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/far vs close correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

############# Plot Close vs Far fraction of cluster for each cell type ###############
vmax=40
colors = ('#CC6677', '#888888')

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

for group_clone in group_clones:
    for i, cond in enumerate(list(group_clone.columns)):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax = sns.barplot(x=list(group_clone[cond].index), y=list(group_clone[cond]), capsize=0.5, edgecolor='0.2', lw=1, errwidth=1, color=colors[i])

        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color('0.2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        #sns.barplot(x=list(group_clone[cond].index), y=list(group_clone[cond]), color=colors[i])
        #plt.xlabel('%s' % cluster_type)
        plt.ylabel('Occurence (%)')
        plt.ylim(0, vmax)
        plt.savefig(path + '%s_distribution.png' % cond, dpi=300, bbox_inches='tight')
        if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/')
        plt.savefig(path + 'svg/%s_distribution.svg' % cond, bbox_inches='tight')
        plt.clf()
        plt.close()

############# Shannon entropy of Close vs Far for each video ###############
entropies_far = {'mt_B-cell': [], 'wt_B-cell':[]}
group_name = 'Video'
groups = np.unique(df[group_name])

for group in groups:
    df_part = df[(df[group_name]==group)&(df['Average_distance_to_T'] >= 15)]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='tskmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_far[type].append(entropy[type])

entropies_close = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Average_distance_to_T'] <= 5)]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='tskmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_close[type].append(entropy[type])

entropies = {}
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    entropies = {'%s_far'%cell_type:entropies_far[cell_type], '%s_close'%cell_type:entropies_close[cell_type]}
    rename_keys = {'wt_B-cell_far': 'wt GCB far',
                   'wt_B-cell_close': 'wt GCB close', 'mt_B-cell_far': 'mt GCB far', 'mt_B-cell_close': 'mt GCB close', }

    entropies_ = {rename_keys.get(k, k): v for (k, v) in entropies.items()}
    draw_custom_bar_plot(entropies_, path, file_name='entropy of far vs close for %s' %cell_type, colors=('#888888', '#CC6677'),
                         strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))

###################### Plot Close vs Far motility feature violin plot for mt GCB and wt GCB  ############################

if not os.path.isdir(path + 'close vs far motility box plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'close vs far motility box plot/')

feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for interaction in ['far','close']:
            if interaction == 'far':
                data = df[(df['Average_distance_to_T'] >= 15)&(df[condition_name] == cell_type)][feature_name]
            elif interaction == 'close':
                data = df[(df['Average_distance_to_T'] <= 5)&(df[condition_name] == cell_type)][feature_name]

            dataset[cell_type+'_'+str(interaction)] = np.array(data)

    rename_keys = {'wt_B-cell_far': 'wt GCB far', 'wt_B-cell_close': 'wt GCB close', 'mt_B-cell_far': 'mt GCB far', 'mt_B-cell_close': 'mt GCB close'}
    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_violin_plot(dataset_renamed, path + 'close vs far motility box plot/', file_name=feature_name,
                            colors=('#888888', '#888888', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))

#################################### Volcano plot of Close vs Far motility features ####################################
feature_list = df.columns[130:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z', 'phi'])
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    df_p = pd.DataFrame()
    for feature_name in feature_list:
        dataset = {}
        for interaction in ['far','close']:
            if interaction == 'far':
                data = df[(df['Average_distance_to_T'] >= 15)&(df[condition_name] == cell_type)][feature_name]
            elif interaction == 'close':
                data = df[(df['Average_distance_to_T'] <= 5)&(df[condition_name] == cell_type)][feature_name]
            dataset[cell_type + '_' + str(interaction)] = np.array(data)
        pvalue = get_pvalue(dataset, test='mann-whitney')
        logp = -np.log10(pvalue)

        avgZ = get_avgZ(dataset, ref_name=cell_type+'_far', data_name=cell_type+'_close')

        row = pd.DataFrame()
        row['Feature'] = [feature_name]
        row['Pvalue'] = [pvalue]
        row['-Logp'] = [logp]
        row['AvgZ'] = [avgZ]
        df_p = pd.concat([df_p, row], axis=0)

    df_p['Adj_p'] = adjusted_pvalues(df_p['Pvalue'], correction_type='Benjamini-Hochberg')
    df_p['Adj_Logp'] = -np.log(df_p['Adj_p'])
    df_p = df_p[~df_p.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf


    draw_volcano_plot(df_p, path, file_name='%s far vs close motility volcano plot'%cell_type, z_thresh=0.5, p_thresh=5, z_name='AvgZ', p_name='Adj_Logp',
                      feature_name='Feature', figsize=(6,6))

######################## Close vs Far way interaction features ###########################
if not os.path.isdir(path + 'close vs away int feature violin plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'close vs away int feature violin plot/')

df.columns.get_loc('FDC_total_distance')
df.columns.get_loc('tp_interaction_mt_B-cell')

k = df.iloc[:,324:417].isnull().any()
null_features = k.index[k==True]
feature_list = df.columns[324:417].drop(null_features).drop(['PC1', 'PC2', 'tskmeans'])
p_values_each_feature = {}
for feature_name in feature_list:
    datasets={}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for interaction_type in ['Away', 'Close']:
            if interaction_type == 'Away':
                data = df[(df['Type'] == cell_type) & (df['Average_distance_to_T'] >= 15) ][feature_name]
            elif interaction_type == 'Close':
                data = df[(df['Type'] == cell_type) & (df['Average_distance_to_T'] <= 5)][feature_name]
            datasets[cell_type+' '+interaction_type] = np.array(data)

    values = flatten_nested_dict(datasets)
    if np.isnan(values).any() == True:  # Check at least one nan
        continue
    elif np.isfinite(values).all() == False:  # Check everything is not inf
        continue

    draw_custom_violin_plot(datasets, path + 'close vs away int feature violin plot/', file_name=feature_name,
                            colors=('#888888', '#888888', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True,
                            figsize=(2, 2))

#################################### all motility features wrt avg T distance ####################################
FDC_dist_range = (0,20)

feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    mean_dataset = {}
    std_dataset = {}

    for cell_type in ['mt_B-cell', 'wt_B-cell']:
        df_part = df[df['Type'] == cell_type]
        means = []
        stds = []

        for i in np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1):
            if i == FDC_dist_range[1]:
                values = df_part[df_part['Average_distance_to_T'] >= i][feature_name].values
            else:
                values = df_part[(df_part['Average_distance_to_T'] >= i) & (df_part['Average_distance_to_T'] < i + 1)][feature_name].values

            #means.append(np.mean(values))
            means.append(np.median(values))
            stds.append(np.std(values))

        mean_dataset[cell_type] = np.array(means)
        std_dataset[cell_type] = np.array(stds)


    replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
    mean_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_dataset.items() }
    std_dataset = {replace_keys.get(k, k):v  for (k,v) in std_dataset.items() }

    font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(4,4))
    ax = sns.lineplot(data=mean_dataset, lw=2.5, markers=['o', '^'], dashes=False, markersize=8,
                 err_style='bars', palette=['#CC6677', '#888888'])
    ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['mt GCB'], 0.1*std_dataset['mt GCB'], color='#CC6677',
               capsize=3, capthick=1, elinewidth=1.5)
    ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['wt GCB'], 0.1*std_dataset['wt GCB'], color='#888888',
               capsize=3, capthick=1, elinewidth=1.5)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')

    ax.set_xlabel('Distance to Tfh (μm)', fontsize=16, weight='bold', color='0.2')
    #ax.set_ylabel('Average speed (μm/min)', fontsize=16, weight='bold', color='0.2')
    plt.xticks(fontsize=16, color='0.2',weight='bold', )
    plt.yticks(fontsize=16, color='0.2', weight='bold', )

    plt.legend(frameon=False, prop={'weight':'bold', 'size':12}, labelcolor='0.2')
    # plt.ylabel('%s' % feature_name, fontsize=4)

    if not os.path.isdir(path + 'feature_wrt_T_distance/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'feature_wrt_T_distance/')

    plt.savefig(path + 'feature_wrt_T_distance/%s.png'%feature_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'feature_wrt_T_distance/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'feature_wrt_T_distance/svg/')
    plt.savefig(path + 'feature_wrt_T_distance/svg/%s.svg'%feature_name, bbox_inches='tight')
    plt.clf()
    plt.close()

#################################### all motility features wrt T contact time ####################################
FDC_dist_range = (0,20)

feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    mean_dataset = {}
    std_dataset = {}

    for cell_type in ['mt_B-cell', 'wt_B-cell']:
        df_part = df[df['Type'] == cell_type]
        means = []
        stds = []

        for i in np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1):

            if i == FDC_dist_range[1]:
                values = df_part[df_part['tp_interaction_T-cell'] >= i][feature_name].values
            else:
                values = df_part[(df_part['tp_interaction_T-cell'] >= i) & (df_part['tp_interaction_T-cell'] < i + 1)][feature_name].values

            # interaction_type = 'FDC'
            # interaction_list = ['wt_B-cell', 'mt_B-cell', 'FDC', 'macrophage', 'T-cell']
            # interaction_list.remove(cell_type)
            # interaction_list.remove(interaction_type)

            # if i == FDC_dist_range[1]:
            #     values = df_part[(df_part['tp_interaction_FDC'] >= i) & (df_part['tp_interaction_%s' % interaction_list[0]] <= 5)
            #          & (df_part['tp_interaction_%s' % interaction_list[1]] <= 5) & (df_part['tp_interaction_%s' % interaction_list[2]] <= 5)][feature_name].values
            # else:
            #     values = df_part[(df_part['tp_interaction_FDC'] >= i) & (df_part['tp_interaction_FDC'] < i + 1) & (df_part['tp_interaction_%s' % interaction_list[0]] <= 5)
            #          & (df_part['tp_interaction_%s' % interaction_list[1]] <= 5) & (df_part['tp_interaction_%s' % interaction_list[2]] <= 5)][feature_name].values

            #means.append(np.mean(values))
            means.append(np.median(values))
            stds.append(np.std(values))

        mean_dataset[cell_type] = np.array(means)
        std_dataset[cell_type] = np.array(stds)


    replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
    mean_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_dataset.items() }
    std_dataset = {replace_keys.get(k, k):v  for (k,v) in std_dataset.items() }

    font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(4,4))
    ax = sns.lineplot(data=mean_dataset, lw=2.5, markers=['o', '^'], dashes=False, markersize=8,
                 err_style='bars', palette=['#CC6677', '#888888'])
    ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['mt GCB'], 0.1*std_dataset['mt GCB'], color='#CC6677',
               capsize=3, capthick=1, elinewidth=1.5)
    ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['wt GCB'], 0.1*std_dataset['wt GCB'], color='#888888',
               capsize=3, capthick=1, elinewidth=1.5)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')

    ax.set_xlabel('Contact time with Tfh', fontsize=16, weight='bold', color='0.2')
    #ax.set_ylabel('Average speed (μm/min)', fontsize=16, weight='bold', color='0.2')
    plt.xticks(fontsize=16, color='0.2',weight='bold', )
    plt.yticks(fontsize=16, color='0.2', weight='bold', )

    plt.legend(frameon=False, prop={'weight':'bold', 'size':12}, labelcolor='0.2')
    # plt.ylabel('%s' % feature_name, fontsize=4)

    if not os.path.isdir(path + 'feature_wrt_T_contact_time/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'feature_wrt_T_contact_time/')

    plt.savefig(path + 'feature_wrt_T_contact_time/%s.png'%feature_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'feature_wrt_T_contact_time/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'feature_wrt_T_contact_time/svg/')
    plt.savefig(path + 'feature_wrt_T_contact_time/svg/%s.svg'%feature_name, bbox_inches='tight')
    plt.clf()
    plt.close()

#################################### T cell, wt GCB all interaction features wrt T distance ####################################
FDC_dist_range = (0,20)
feature_list = df.columns[324:].drop(['interacted_type'])

for feature_name in feature_list:
    mean_dataset = {}
    std_dataset = {}

    for cell_type in ['mt_B-cell', 'wt_B-cell']:
        df_part = df[df['Type'] == cell_type]
        means = []
        stds = []

        for i in np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1):
            if i == FDC_dist_range[1]:
                values = df_part[df_part['Average_distance_to_T'] >= i][feature_name].values
            else:
                values = df_part[(df_part['Average_distance_to_T'] >= i) & (df_part['Average_distance_to_T'] < i + 1)][feature_name].values

            if np.isnan(values).any() == True:  # Check at least one nan
                means.append(np.nan)
                stds.append(np.nan)

            elif np.isfinite(values).all() == False:  # Check everything is not inf
                means.append(np.nan)
                stds.append(np.nan)

            else:
                means.append(np.mean(values))
                #means.append(np.median(values))
                stds.append(np.std(values))

        mean_dataset[cell_type] = np.array(means)
        std_dataset[cell_type] = np.array(stds)

    check_nan = flatten_nested_dict(mean_dataset)
    if np.isnan(check_nan).any() == True:  # Check at least one nan
        continue

    replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
    mean_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_dataset.items() }
    std_dataset = {replace_keys.get(k, k):v  for (k,v) in std_dataset.items() }

    font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(4,4))
    ax = sns.lineplot(data=mean_dataset, lw=2.5, markers=['o', '^'], dashes=False, markersize=8,
                 err_style='bars', palette=['#CC6677', '#888888'])
    ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['mt GCB'], 0.1*std_dataset['mt GCB'], color='#CC6677',
               capsize=3, capthick=1, elinewidth=1.5)
    ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['wt GCB'], 0.1*std_dataset['wt GCB'], color='#888888',
               capsize=3, capthick=1, elinewidth=1.5)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')

    ax.set_xlabel('Distance to Tfh (μm)', fontsize=16, weight='bold', color='0.2')
    #ax.set_ylabel('Average speed (μm/min)', fontsize=16, weight='bold', color='0.2')
    plt.xticks(fontsize=16, color='0.2',weight='bold', )
    plt.yticks(fontsize=16, color='0.2', weight='bold', )

    plt.legend(frameon=False, prop={'weight':'bold', 'size':12}, labelcolor='0.2')
    # plt.ylabel('%s' % feature_name, fontsize=4)

    if not os.path.isdir(path + 'int_feature_wrt_T_distance/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'int_feature_wrt_T_distance/')

    plt.savefig(path + 'int_feature_wrt_T_distance/%s.png'%feature_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'int_feature_wrt_T_distance/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'int_feature_wrt_T_distance/svg/')
    plt.savefig(path + 'int_feature_wrt_T_distance/svg/%s.svg'%feature_name, bbox_inches='tight')
    plt.clf()
    plt.close()


#################################### T cell, wt GCB all interaction features wrt Tfh contact time ####################################
FDC_dist_range = (0,20)
feature_list = df.columns[324:].drop(['interacted_type'])

for feature_name in feature_list:
    mean_dataset = {}
    std_dataset = {}

    for cell_type in ['mt_B-cell', 'wt_B-cell']:
        df_part = df[df['Type'] == cell_type]
        means = []
        stds = []

        for i in np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1):
            if i == FDC_dist_range[1]:
                values = df_part[df_part['tp_interaction_T-cell'] >= i][feature_name].values
            else:
                values = df_part[(df_part['tp_interaction_T-cell'] >= i) & (df_part['tp_interaction_T-cell'] < i + 1)][feature_name].values

            if np.isnan(values).any() == True:  # Check at least one nan
                means.append(np.nan)
                stds.append(np.nan)

            elif np.isfinite(values).all() == False:  # Check everything is not inf
                means.append(np.nan)
                stds.append(np.nan)

            else:
                means.append(np.mean(values))
                #means.append(np.median(values))
                stds.append(np.std(values))

        mean_dataset[cell_type] = np.array(means)
        std_dataset[cell_type] = np.array(stds)

    check_nan = flatten_nested_dict(mean_dataset)
    if np.isnan(check_nan).any() == True:  # Check at least one nan
        continue

    replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
    mean_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_dataset.items() }
    std_dataset = {replace_keys.get(k, k):v  for (k,v) in std_dataset.items() }

    font = {'family': 'arial',
                'weight': 'normal',
                'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(4,4))
    ax = sns.lineplot(data=mean_dataset, lw=2.5, markers=['o', '^'], dashes=False, markersize=8,
                 err_style='bars', palette=['#CC6677', '#888888'])
    ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['mt GCB'], 0.1*std_dataset['mt GCB'], color='#CC6677',
               capsize=3, capthick=1, elinewidth=1.5)
    ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['wt GCB'], 0.1*std_dataset['wt GCB'], color='#888888',
               capsize=3, capthick=1, elinewidth=1.5)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')

    ax.set_xlabel('Contact time with Tfh', fontsize=16, weight='bold', color='0.2')
    #ax.set_ylabel('Average speed (μm/min)', fontsize=16, weight='bold', color='0.2')
    plt.xticks(fontsize=16, color='0.2',weight='bold', )
    plt.yticks(fontsize=16, color='0.2', weight='bold', )

    plt.legend(frameon=False, prop={'weight':'bold', 'size':12}, labelcolor='0.2')
    # plt.ylabel('%s' % feature_name, fontsize=4)

    if not os.path.isdir(path + 'int_feature_wrt_T_contact_time/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'int_feature_wrt_T_contact_time/')

    plt.savefig(path + 'int_feature_wrt_T_contact_time/%s.png'%feature_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'int_feature_wrt_T_contact_time/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'int_feature_wrt_T_contact_time/svg/')
    plt.savefig(path + 'int_feature_wrt_T_contact_time/svg/%s.svg'%feature_name, bbox_inches='tight')
    plt.clf()
    plt.close()