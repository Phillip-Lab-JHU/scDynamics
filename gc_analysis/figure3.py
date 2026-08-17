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
"""Generates Data for Figure3. wt and MT interaction with Tfh"""

from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import DistanceSignal

#################################### Pull out interaction features ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
#path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis_test\feature_csvs\\'
df = pd.read_parquet(path+'GCB_no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_no_inhibit_traj_duration_20.parquet')


_, _, Zone_series = to_timeseries_fast(df_duration, duration=20, feature_name='Zone')

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

duration=20
label_expanded = np.repeat(df['Zone'], duration).reset_index(drop=True)
df_duration['Zone_label'] = label_expanded

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure3. Interaction with Tfh of wt and mt GCB\\'
#path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis_test\Figure3. Interaction with Tfh of wt and mt GCB\\'
#################################### T overlap volume WT vs MT ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

for condition in ['WT', 'MT']:
    n_cells = df_[(df_['Type'] == condition)&(df_['T_contact_persistences']>=10)].shape[0]
    print(condition, n_cells)



dict_datasets={}
for condition in ['WT', 'MT']:
    values = df_[df_['Type'] == condition]['T_avg_overlap'].values
    values = values[values!=0]
    dict_datasets[condition] = np.array(values)

draw_custom_violin_plot(dict_datasets, path, file_name='Overlapped T volume',
                        colors = ('#888888', '#CC6677') , test='mann-whitney', pvalue=True, figsize=(1, 2), vmax=0.1)
draw_custom_box_plot(dict_datasets, path, file_name='Overlapped T volume',
                        strip_plot=False, colors = ('#888888', '#CC6677') , test='mann-whitney', pvalue=True, figsize=(1, 2), vmax=0.1)

dict_datasets={}
for condition in ['WT', 'MT']:
    values = df_[(df_['Type'] == condition)&(df_['T_contact_times']>=19)]['T_avg_overlap'].values
    values = values[values!=0]
    dict_datasets[condition] = np.array(values)
    print(condition, values.shape[0])

draw_custom_bar_plot(dict_datasets, path, file_name='Overlapped T volume (for pesistent contacting cells)',
                     strip_plot=True, colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1,2))

draw_custom_violin_plot(dict_datasets, path, file_name='Overlapped T volume',
                        colors = ('#888888', '#CC6677') , test='mann-whitney', pvalue=True, figsize=(1, 2), vmax=0.1)
draw_custom_box_plot(dict_datasets, path, file_name='Overlapped T volume',
                        strip_plot=False, colors = ('#888888', '#CC6677') , test='mann-whitney', pvalue=True, figsize=(1, 2), vmax=0.1)


font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

colors=('#888888', '#CC6677')
fig, ax = plt.subplots(figsize=(2,2))
for i, key in enumerate(dict_datasets):
    ax = sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=(0, 0.2), color=colors[i], label=key)

# ax = sns.kdeplot(data=dict_datasets['WT'], fill=True, linewidth=1, clip=(0, 40), color='#888888', label='WT')
# ax = sns.kdeplot(data=dict_datasets['MT'], fill=True, linewidth=1, clip=(0, 40), color='#CC6677', label='MT')

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Tfh engaged fraction', fontsize=8, weight='normal', color='0.2')
ax.set_ylabel('Density', fontsize=8, weight='normal', color='0.2')
plt.xticks(fontsize=8, color='0.2', weight='normal')
plt.yticks(fontsize=8, color='0.2', weight='normal')

plt.legend(frameon=False, prop = {'weight':'normal', 'size':8}, labelcolor='0.2')

plt.savefig(path+'Overlapped T volume kde.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/Overlapped T volume kde.svg', bbox_inches='tight')
plt.close()
plt.clf()


font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

colors=('#888888', '#CC6677')
fig, ax = plt.subplots(figsize=(2,2))
for i, key in enumerate(dict_datasets):
    ax = sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=(0.05, 0.2), color=colors[i], label=key)

# ax = sns.kdeplot(data=dict_datasets['WT'], fill=True, linewidth=1, clip=(0, 40), color='#888888', label='WT')
# ax = sns.kdeplot(data=dict_datasets['MT'], fill=True, linewidth=1, clip=(0, 40), color='#CC6677', label='MT')

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Tfh engaged fraction', fontsize=8, weight='normal', color='0.2')
ax.set_ylabel('Density', fontsize=8, weight='normal', color='0.2')
plt.xticks(fontsize=8, color='0.2', weight='normal')
plt.yticks(fontsize=8, color='0.2', weight='normal')

plt.legend(frameon=False, prop = {'weight':'normal', 'size':8}, labelcolor='0.2')

plt.savefig(path+'Overlapped T volume kde_cropped.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/Overlapped T volume kde_cropped.svg', bbox_inches='tight')
plt.close()
plt.clf()


#################################### all motility features wrt avg T distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df_.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

coloc_features = ['T_distance_average']

for idx in [0]:
    coloc_feature = coloc_features[idx]
    if idx == 0:
        xlabel = 'Distance to Tfh cell (µm)'
    draw_lineplot_by_custom_ranges(df_, path, folder_name='motility_feature_wrt_%s' % coloc_feature,
                                   feature_list=feature_list,
                                   condition_name='Type', custsom_range=(4, 32), stepsize=4,
                                   range_feature=coloc_feature,
                                   color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4, 4),
                                   x_label=xlabel,
                                   estimator='mean', error_type='ci_norm', replace_keys=None, pvalue=True,
                                   test='mann-whitney')

#################################### all interaction features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
#df_['T_contact_times'] = (df_['T_contact_times'])/2

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
                               condition_name='Type', custsom_range=(0, 18), stepsize=2, range_feature='T_contact_times',
                                   color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label=xlabel,
                               estimator='mean', error_type='sem', replace_keys=None, pvalue=True, legend=False, test='mann-whitney')

xlabel = 'Number of Tfh Persistent Contacts'
draw_lineplot_by_custom_ranges(df_, path, folder_name='int_feature_wrt_T_contact_persistence', feature_list=feature_list,
                               condition_name='Type', custsom_range=(0, 20), stepsize=2, range_feature='T_contact_persistences',
                                   color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label=xlabel,
                               estimator='mean', error_type='sem', replace_keys=None, pvalue=True, legend=False, test='wilcoxon-ranksum')

xlabel = 'Number of FDC Contacts'
draw_lineplot_by_custom_ranges(df_, path, folder_name='int_feature_wrt_FDC_contacts', feature_list=feature_list,
                               condition_name='Type', custsom_range=(0, 20), stepsize=2, range_feature='FDC_contact_times',
                                   color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label=xlabel,
                               estimator='mean', error_type='sem', replace_keys=None, pvalue=True, legend=False, test='mann-whitney')

xlabel = 'Number of FDC Persistent Contacts'
draw_lineplot_by_custom_ranges(df_, path, folder_name='int_feature_wrt_FDC_persistence', feature_list=feature_list,
                               condition_name='Type', custsom_range=(0, 20), stepsize=2, range_feature='FDC_contact_persistences',
                                   color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label=xlabel,
                               estimator='mean', error_type='sem', replace_keys=None, pvalue=True, legend=False, test='mann-whitney')


feature_name = 'T_contact_persistences'
coloc_feature = 'FDC_contact_persistences'

condition_name = 'Type'
for cell_type in np.unique(df_[condition_name]):
    df_part = df_[df_[condition_name] == cell_type]
    custsom_range = (0, 16)
    stepsize = 2
    range_feature = coloc_feature
    for i in np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize):
        # if i == custsom_range[1]:
        #     values = df_part[df_part[range_feature] >= i][feature_name].values
        # else:
        values = df_part[(df_part[range_feature] >= i) & (df_part[range_feature] < i + stepsize)][
            feature_name].values
        print(coloc_feature, cell_type, i, values.shape[0])


####################################### Quantify Tfh interaction frequency #############################################
int_time = 19
test = 'mann-whitney'

persistent_int_freqs_datasets = {}
persistent_int_freq_per_cellnumbers_datasets = {}
total_n_contacts_per_cellnumbers_datasets = {}
persistent_int_freq_per_cellcontacts_datasets = {}
low_contact_freq_per_cellnumbers_datasets = {}
dLZ_persistent_int_freq_per_cellnumbers_datatsets = {}
sLZ_persistent_int_freq_per_cellnumbers_datatsets = {}
videos = np.unique(df['Video'])
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    df_part = df[df['Type'] == cell_type]
    persistent_int_freqs = []
    persistent_int_freq_per_cellnumbers = []
    total_n_contacts_per_cellnumbers = []
    persistent_int_freq_per_cellcontacts = []
    low_contact_freq_per_cellnumbers = []

    dLZ_persistent_int_freq_per_cellnumbers = []
    sLZ_persistent_int_freq_per_cellnumbers = []
    for video in videos:
        #if 'A' in video and cell_type == 'mt_B-cell':
        df_video = df_part[df_part['Video'] == video].reset_index(drop=True)
        if df_video.shape[0] == 0:
            continue
        data = df_video['T_contact_times']
        #data = df_video['T_contact_persistences']
        mask = ~np.isnan(data)
        data = data[mask]

        persistent_int_freq = sum(data == int_time)
        total_n_contact = sum(data)
        n_trajs = df_video.shape[0]
        # if n_trajs <=20:
        #     continue
        persistent_int_freq_per_cellnumber = persistent_int_freq / n_trajs
        total_n_contacts_per_cellnumber = total_n_contact / n_trajs
        persistent_int_freq_per_cellcontact = persistent_int_freq / total_n_contact
        low_contact_freq_per_cellnumber = sum((1<=data) & (data<=5)) / n_trajs

        df_dLZ = df_video[(df_video['Zone'] == 'dLZ')|(df_video['Zone'] == 'sLZ-dLZ')].reset_index(drop=True)
        df_sLZ = df_video[(df_video['Zone'] == 'sLZ')].reset_index(drop=True)
        dLZ_data = df_dLZ['T_contact_times']
        sLZ_data = df_sLZ['T_contact_times']
        dLZ_persistent_int_freq = sum(dLZ_data >= 15)
        sLZ_persistent_int_freq = sum(sLZ_data >= 15)
        dLZ_n_trajs = df_dLZ.shape[0]
        sLZ_n_trajs = df_sLZ.shape[0]
        dLZ_persistent_int_freq_per_cellnumber = dLZ_persistent_int_freq / dLZ_n_trajs
        sLZ_persistent_int_freq_per_cellnumber = sLZ_persistent_int_freq / sLZ_n_trajs

        #if dLZ_n_trajs >= 20:

        if persistent_int_freq != 0:
            persistent_int_freqs.append(persistent_int_freq)
            persistent_int_freq_per_cellnumbers.append(persistent_int_freq_per_cellnumber)
            persistent_int_freq_per_cellcontacts.append(persistent_int_freq_per_cellcontact)

            dLZ_persistent_int_freq_per_cellnumbers.append(dLZ_persistent_int_freq_per_cellnumber)
            sLZ_persistent_int_freq_per_cellnumbers.append(sLZ_persistent_int_freq_per_cellnumber)
            print(cell_type, video, dLZ_n_trajs)
        # else:
        #     print(video, persistent_int_freq)
        total_n_contacts_per_cellnumbers.append(total_n_contacts_per_cellnumber)
        low_contact_freq_per_cellnumbers.append(low_contact_freq_per_cellnumber)

    persistent_int_freqs_datasets[cell_type] = persistent_int_freqs
    persistent_int_freq_per_cellnumbers_datasets[cell_type] = persistent_int_freq_per_cellnumbers
    total_n_contacts_per_cellnumbers_datasets[cell_type] = total_n_contacts_per_cellnumbers
    persistent_int_freq_per_cellcontacts_datasets[cell_type] = persistent_int_freq_per_cellcontacts
    low_contact_freq_per_cellnumbers_datasets[cell_type] = low_contact_freq_per_cellnumbers

    dLZ_persistent_int_freq_per_cellnumbers_datatsets[cell_type] = dLZ_persistent_int_freq_per_cellnumbers
    sLZ_persistent_int_freq_per_cellnumbers_datatsets[cell_type] = sLZ_persistent_int_freq_per_cellnumbers

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
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


dLZ_persistent_int_freq_per_cellnumbers_datatsets = change_dict_order(dLZ_persistent_int_freq_per_cellnumbers_datatsets, new_order)
dLZ_persistent_int_freq_per_cellnumbers_datatsets = {replace_keys.get(k, k):v  for (k,v) in dLZ_persistent_int_freq_per_cellnumbers_datatsets.items() }

sLZ_persistent_int_freq_per_cellnumbers_datatsets = change_dict_order(sLZ_persistent_int_freq_per_cellnumbers_datatsets, new_order)
sLZ_persistent_int_freq_per_cellnumbers_datatsets = {replace_keys.get(k, k):v  for (k,v) in sLZ_persistent_int_freq_per_cellnumbers_datatsets.items() }


colors=('#888888', '#CC6677')
draw_custom_bar_plot(persistent_int_freqs_datasets, path, file_name='Total Tfh interaction frequency',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellnumbers_datasets, path, file_name='Tfh persistent interaction frequency per cell number_contact_%s'%int_time,
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(total_n_contacts_per_cellnumbers_datasets, path, file_name='number of Tfh contacts per cell number_contact_%s'%int_time,
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
# draw_custom_bar_plot(persistent_int_freq_per_cellcontacts_datasets, path, file_name='Tfh persistent interaction frequency per number of contacts',
#                      strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))
draw_custom_bar_plot(low_contact_freq_per_cellnumbers_datasets, path, file_name='Tfh low contact time frequency per cell number',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))


draw_custom_bar_plot(dLZ_persistent_int_freq_per_cellnumbers_datatsets, path, file_name='dLZ_Tfh persistent interaction frequency per cell number',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(sLZ_persistent_int_freq_per_cellnumbers_datatsets, path, file_name='sLZ_Tfh persistent interaction frequency per cell number',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))

####################################### interaction frequency differential for B cell each interaction time #############################################
freq_datasets = {}
freq_per_cellnumber_datasets = {}
videos = np.unique(df['Video'])
#videos[videos != '4-Good-D10-C1-ZT4-40-127-FOV230-256px_Statistics']

for t in np.arange(2, 10.5, 0.5):
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
                data = df_video['%s_contact_times' % interaction_type]
                #data = df_video['%s_contact_persistences' % interaction_type]
                #data = df_video['%s_avg_overlap' % interaction_type]
                mask = ~np.isnan(data)
                data = data[mask]
                #int_freq = sum(data <= 10)
                int_freq_per_cellnumber = sum(data == 2*t) / df_video.shape[0]
                #int_freqs.append(int_freq)
                int_freq_per_cellnumbers.append(int_freq_per_cellnumber)
            #freq_datasets[cell_type + '-' + interaction_type] = int_freqs
            freq_per_cellnumber_datasets_temp[cell_type + '-' + interaction_type] = int_freq_per_cellnumbers
    freq_per_cellnumber_datasets[t] = freq_per_cellnumber_datasets_temp

B_FDC_diffs = {}
B_T_diffs = {}
wt_B_Ts = {}
mt_B_Ts = {}
for t in freq_per_cellnumber_datasets:
    freq_per_cellnumber_data = freq_per_cellnumber_datasets[t]
    B_FDC_diff_temp = []
    B_T_diff_temp = []
    wt_B_T_temp = []
    mt_B_T_temp = []
    for video_idx in range(videos.size):
        B_FDC_diff = freq_per_cellnumber_data['wt_B-cell-FDC'][video_idx] - freq_per_cellnumber_data['mt_B-cell-FDC'][video_idx]
        B_T_diff = freq_per_cellnumber_data['wt_B-cell-T'][video_idx] - freq_per_cellnumber_data['mt_B-cell-T'][video_idx]
        wt_B_T = freq_per_cellnumber_data['wt_B-cell-T'][video_idx]
        mt_B_T = freq_per_cellnumber_data['mt_B-cell-T'][video_idx]

        B_FDC_diff_temp.append(B_FDC_diff)
        B_T_diff_temp.append(B_T_diff)
        wt_B_T_temp.append(wt_B_T)
        mt_B_T_temp.append(mt_B_T)

    #B_FDC_diffs[t] = np.mean(B_FDC_diff_temp)
    #B_T_diffs[t] = np.mean(B_T_diff_temp)
    B_FDC_diffs[t] = B_FDC_diff_temp
    B_T_diffs[t] = B_T_diff_temp
    wt_B_Ts[t] = wt_B_T_temp
    mt_B_Ts[t] = mt_B_T_temp

colors=('#888888', '#888888')

# l = [B_FDC_diffs[t] for t in B_FDC_diffs if 0<=t<=3]
# flat_list = [item for sublist in l for item in sublist]
# dataset = {'Low interaction':flat_list, 'High interaction': B_FDC_diffs[20]}
#
# draw_custom_bar_plot(dataset, path, file_name='Low interaction vs High interaction for B-FDC',
#                      strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, figsize=(1,2))
#
#
#
low = [B_T_diffs[t] for t in B_T_diffs if 2<=t<=7]
low_flat_list = [item for sublist in low for item in sublist]
high = [B_T_diffs[t] for t in B_T_diffs if 7.5<=t<=10]
high_flat_list = [item for sublist in high for item in sublist]
dataset = {'Low interaction':low_flat_list, 'High interaction': high_flat_list}
[print(key, np.array(value).size) for key, value in dataset.items()]

draw_custom_bar_plot(dataset, path, file_name='Low interaction vs High interaction for B-T',
                     strip_plot=False, colors=colors, test='mann-whitney', pvalue=True, figsize=(1,2))
draw_custom_violin_plot(dataset, path, file_name='Low interaction vs High interaction for B-T_violin',colors=colors,
                                test='mann-whitney', pvalue=True, figsize=(1, 2), )
draw_custom_box_plot(dataset, path, file_name='Low interaction vs High interaction for B-T_box',
                     strip_plot=False, colors=colors, test='mann-whitney', pvalue=True, figsize=(1,2))



low_minus = np.array(low_flat_list)[np.array(low_flat_list)<0]
low_plus = np.array(low_flat_list)[np.array(low_flat_list)>0]
high_minus = np.array(high_flat_list)[np.array(high_flat_list)<0]
high_plus = np.array(high_flat_list)[np.array(high_flat_list)>0]

dataset = {'Low minus':-low_minus, 'High minus':-high_minus}
[print(key, np.array(value).size) for key, value in dataset.items()]

draw_custom_bar_plot(dataset, path, file_name='Low interaction vs High interaction for B-T_minus',
                     strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, figsize=(1,2))

dataset = {'Low plus':low_plus, 'High plus':high_plus,}

draw_custom_bar_plot(dataset, path, file_name='Low interaction vs High interaction for B-T_plus',
                     strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, figsize=(1,2))


dataset = {'Low minus':low_minus, 'High minus':high_minus, 'Low plus':low_plus, 'High plus':high_plus}
draw_custom_bar_plot(dataset, path, file_name='Low interaction vs High interaction for B-T_all',
                     strip_plot=True, colors=('#888888', '#888888', '#888888', '#888888'), test='mann-whitney', pvalue=True, figsize=(1,2))
[print(key, np.array(value).size) for key, value in dataset.items()]



draw_custom_bar_plot(B_T_diffs, path, file_name='Tfh interaction diff',
                     strip_plot=False, colors=colors, test='mann-whitney', pvalue=False, figsize=(3,2))


df_part = B_T_diffs
xlabel = 'Tfh interaction'
file_name = 'Tfh interaction diff scatter cropped for time'

estimator = 'mean'
error_type = 'sem'

mean_dataset = {}
error_dataset = {}

means = []
errors = []
valuess = []

for i in df_part.keys():
    pass
    # if i == custsom_range[1]:
    #     values = df_part[df_part[range_feature] >= i][feature_name].values
    # else:
    values = df_part[i]

    if estimator == 'mean':
        means.append(np.mean(values))
    elif estimator == 'median':
        means.append(np.median(values))
    # means.append(np.mean(values))
    #means.append(np.mean(values))
    if error_type == 'std':
        error = np.std(values)
    elif error_type == 'sem':
        error = stats.sem(values)
    elif error_type == 'ci_norm':
        interval = stats.norm.interval(confidence=0.95, loc=np.mean(values), scale=stats.sem(values))
        error = np.mean(values) - interval[0]
    elif error_type == 'ci_t':
        interval = stats.t.interval(confidence=0.95, df=values.size-1, loc=np.mean(values), scale=stats.sem(values))
        error = np.mean(values) - interval[0]

    errors.append(error)
    valuess.append(values)

mean_dataset = np.array(means)
error_dataset = np.array(errors)


font = {'family': 'arial',
        'weight': 'normal',
        'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(5,4))

sns.scatterplot(x=list(df_part.keys()), y=mean_dataset, marker='o', color='#888888')
#sns.lineplot(x=list(df_part.keys()), y=mean_dataset, lw=2.5, marker='o', dashes=False, markersize=8, err_style='bars', color='#888888')

ax.errorbar(x=list(df_part.keys()), y=mean_dataset,
            yerr=error_dataset, color='#888888', capsize=3, capthick=1, linestyle='none', elinewidth=1.5)


# ax.errorbar(x=np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize), y=mean_dataset[key],
#             yerr=error_dataset[key], color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)
# ax.fill_between(x=df_part.keys(), y1=mean_dataset-error_dataset, y2=mean_dataset+error_dataset,
#              color='#888888', alpha=0.4)



ax.axhline(0, linestyle='--', linewidth=2, color='0.2')

#handles, labels = ax.get_legend_handles_labels()


for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_xlabel('%s'%xlabel, fontsize=16, weight='normal', color='0.2')

ax.set_xticks(list(df_part.keys()))  # Full tick range
ax.set_xticklabels([str(int(i)) if i.is_integer() else '' for i in list(df_part.keys())], fontsize=16, color='0.2', weight='normal',)  # Show only integer labels

#plt.xticks(list(df_part.keys()), fontsize=16, color='0.2', weight='normal', )
#plt.xticks(fontsize=12, color='0.2', weight='normal', )
plt.yticks(fontsize=16, color='0.2', weight='normal')

# plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
#            loc='best')

plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()








colors=('#888888', '#CC6677')

df_part = B_T_diffs
xlabel = 'Tfh interaction'
file_name = 'Tfh interaction scatter cropped for time'

estimator = 'mean'
error_type = 'sem'

mean_dataset = {}
error_dataset = {}

means = []
errors = []
valuess = []

for i in df_part.keys():
    pass
    # if i == custsom_range[1]:
    #     values = df_part[df_part[range_feature] >= i][feature_name].values
    # else:
    values = df_part[i]

    if estimator == 'mean':
        means.append(np.mean(values))
    elif estimator == 'median':
        means.append(np.median(values))
    # means.append(np.mean(values))
    #means.append(np.mean(values))
    if error_type == 'std':
        error = np.std(values)
    elif error_type == 'sem':
        error = stats.sem(values)
    elif error_type == 'ci_norm':
        interval = stats.norm.interval(confidence=0.95, loc=np.mean(values), scale=stats.sem(values))
        error = np.mean(values) - interval[0]
    elif error_type == 'ci_t':
        interval = stats.t.interval(confidence=0.95, df=values.size-1, loc=np.mean(values), scale=stats.sem(values))
        error = np.mean(values) - interval[0]

    errors.append(error)
    valuess.append(values)

mean_dataset = np.array(means)
error_dataset = np.array(errors)


font = {'family': 'arial',
        'weight': 'normal',
        'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(5,4))

#sns.scatterplot(x=list(df_part.keys()), y=mean_dataset, marker='o', color='#888888')
sns.lineplot(x=list(df_part.keys()), y=mean_dataset, lw=2.5, marker='o', dashes=False, markersize=8, err_style='bars', color='#888888')
# ax.errorbar(x=np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize), y=mean_dataset[key],
#             yerr=error_dataset[key], color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)
# ax.fill_between(x=df_part.keys(), y1=mean_dataset-error_dataset, y2=mean_dataset+error_dataset,
#              color='#888888', alpha=0.4)

ax.errorbar(x=list(df_part.keys()), y=mean_dataset,
            yerr=error_dataset, color='#888888', capsize=3, capthick=1, elinewidth=1.5)
ax.axhline(0, linestyle='--', linewidth=2, color='0.2')

#handles, labels = ax.get_legend_handles_labels()


for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_xlabel('%s'%xlabel, fontsize=16, weight='normal', color='0.2')

ax.set_xticks(list(df_part.keys()))  # Full tick range
ax.set_xticklabels([str(int(i)) if i.is_integer() else '' for i in list(df_part.keys())], fontsize=16, color='0.2', weight='normal',)  # Show only integer labels

#plt.xticks(list(df_part.keys()), fontsize=16, color='0.2', weight='normal', )
#plt.xticks(fontsize=12, color='0.2', weight='normal', )
plt.yticks(fontsize=16, color='0.2', weight='normal')

# plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
#            loc='best')

plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()





####################################### interaction frequency differential for B cell each interaction time #############################################
freq_datasets = {}
freq_per_cellnumber_datasets = {}
videos = np.unique(df['Video'])
interaction_type = 'T'

for video in videos:
    if '-A' in video:
        continue
    df_video = df[df['Video'] == video]
    df_video_MT = df_video[df_video['Type'] == 'mt_B-cell']['%s_contact_times' % interaction_type].values
    df_video_WT = df_video[df_video['Type'] == 'wt_B-cell']['%s_contact_times' % interaction_type].values
    diff = np.sum(df_video_WT)/df_video_WT.shape[0] - np.sum(df_video_MT)/df_video_MT.shape[0]
    print(video, diff)

for t in np.arange(0, 10.5, 0.5):
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
                data = df_video['%s_contact_times' % interaction_type]
                #data = df_video['%s_avg_overlap' % interaction_type]
                mask = ~np.isnan(data)
                data = data[mask]
                #int_freq = sum(data <= 10)
                int_freq_per_cellnumber = sum(data == 2*t) / df_video.shape[0]
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
    for video_idx in range(videos.size):
        B_FDC_diff = freq_per_cellnumber_data['wt_B-cell-FDC'][video_idx] - freq_per_cellnumber_data['mt_B-cell-FDC'][video_idx]
        B_T_diff = freq_per_cellnumber_data['wt_B-cell-T'][video_idx] - freq_per_cellnumber_data['mt_B-cell-T'][video_idx]
        B_FDC_diff_temp.append(B_FDC_diff)
        B_T_diff_temp.append(B_T_diff)
    #B_FDC_diffs[t] = np.mean(B_FDC_diff_temp)
    #B_T_diffs[t] = np.mean(B_T_diff_temp)
    B_FDC_diffs[t] = B_FDC_diff_temp
    B_T_diffs[t] = B_T_diff_temp

colors=('#888888', '#888888')

l = [B_FDC_diffs[t] for t in B_FDC_diffs if 0<=t<=3]
flat_list = [item for sublist in l for item in sublist]
dataset = {'Low interaction':flat_list, 'High interaction': B_FDC_diffs[20]}

draw_custom_bar_plot(dataset, path, file_name='Low interaction vs High interaction for B-FDC',
                     strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, figsize=(1,2))



low = [B_T_diffs[t] for t in B_T_diffs if t==0]
low_flat_list = [item for sublist in l for item in sublist]
high = [B_T_diffs[t] for t in B_T_diffs if t==20]
high_flat_list = [item for sublist in l for item in sublist]
dataset = {'Low interaction':low_flat_list, 'High interaction': high_flat_list}

draw_custom_bar_plot(dataset, path, file_name='Low interaction vs High interaction for B-T',
                     strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, figsize=(1,2))


df_part = B_T_diffs
xlabel = 'Tfh interaction'
file_name = 'Tfh interaction for time'

estimator = 'mean'
error_type = 'sem'

mean_dataset = {}
error_dataset = {}

means = []
errors = []
valuess = []

for i in df_part.keys():
    pass
    # if i == custsom_range[1]:
    #     values = df_part[df_part[range_feature] >= i][feature_name].values
    # else:
    values = df_part[i]

    if estimator == 'mean':
        means.append(np.mean(values))
    elif estimator == 'median':
        means.append(np.median(values))
    # means.append(np.mean(values))
    #means.append(np.mean(values))
    if error_type == 'std':
        error = np.std(values)
    elif error_type == 'sem':
        error = stats.sem(values)
    elif error_type == 'ci_norm':
        interval = stats.norm.interval(confidence=0.95, loc=np.mean(values), scale=stats.sem(values))
        error = np.mean(values) - interval[0]
    elif error_type == 'ci_t':
        interval = stats.t.interval(confidence=0.95, df=values.size-1, loc=np.mean(values), scale=stats.sem(values))
        error = np.mean(values) - interval[0]

    errors.append(error)
    valuess.append(values)

mean_dataset = np.array(means)
error_dataset = np.array(errors)


font = {'family': 'arial',
        'weight': 'normal',
        'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots(figsize=(5,4))

sns.lineplot(x=list(df_part.keys()), y=mean_dataset, lw=2.5, marker='o', dashes=False, markersize=8, err_style='bars', color='#888888')
# ax.errorbar(x=np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize), y=mean_dataset[key],
#             yerr=error_dataset[key], color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)
# ax.fill_between(x=df_part.keys(), y1=mean_dataset-error_dataset, y2=mean_dataset+error_dataset,
#              color='#888888', alpha=0.4)

ax.errorbar(x=list(df_part.keys()), y=mean_dataset,
            yerr=error_dataset, color='#888888', capsize=3, capthick=1, elinewidth=1.5)
ax.axhline(0, linestyle='--', linewidth=2, color='0.2')

#handles, labels = ax.get_legend_handles_labels()


for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_xlabel('%s'%xlabel, fontsize=16, weight='normal', color='0.2')

ax.set_xticks(list(df_part.keys()))  # Full tick range
ax.set_xticklabels([str(int(i)) if i.is_integer() else '' for i in list(df_part.keys())], fontsize=16, color='0.2', weight='normal',)  # Show only integer labels

#plt.xticks(list(df_part.keys()), fontsize=16, color='0.2', weight='normal', )
#plt.xticks(fontsize=12, color='0.2', weight='normal', )
plt.yticks(fontsize=16, color='0.2', weight='normal')

# plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
#            loc='best')

plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()




############################### For same GCB cell, how motility changes when before, during and after interacting ################################

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df_duration = pd.read_parquet(path+'long_traj_duration_20.parquet')

# Remove Group A, IgG , mLT and CD40L
df_duration = df_duration[(df_duration['Exp']=='Exp1')|(df_duration['Exp']=='Exp2')|(df_duration['Exp']=='Exp3')
                              |(df_duration['Exp']=='Exp5')].reset_index(drop=True)

videos = np.unique(df_duration['Video'])
df_duration = df_duration[(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])&(df_duration['Video'] != videos[4])
                          &(df_duration['Video'] != videos[-1])].reset_index(drop=True)
videos = np.unique(df_duration['Video'])

df_duration = df_duration[df_duration['Type']!='T-cell'].reset_index(drop=True)
df_duration = get_instant_movements_variable_duration(df_duration, frame_name='Time_span', time_unit=0.5,
                                                      feature_name=['Position X', 'Position Y', 'Position Z'])

df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=FDC')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=T-cell')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_DZ')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_LZ')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_FDC_core')

#df_duration = df_duration[(df_duration['pseudo_frame']!=0)&(df_duration['pseudo_frame']!=1)].reset_index(drop=True)

df_duration = df_duration.replace({'Type': {'wt B-cell': 'wt_B-cell', 'mt B-cell': 'mt_B-cell'}})

from itertools import groupby

threshold = 0
min_duration = 15
interaction_type = 'T-cell' #'FDC', 'T-cell'
test = 'wilcoxon-ranksum' # mann-whitney, wilcoxon-ranksum, t-test
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure3. Interaction with Tfh of wt and mt GCB\\'




folder_name = '%s median before vs during vs after'%interaction_type
features = ['instant_speed', 'instant_angle', 'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'Zone', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC',
            'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell', 'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC',
            'diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell', 'diff_Distance_to_DZ', 'diff_Distance_to_LZ',
            'diff_Distance_to_FDC_core', 'Area', 'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Sphericity', 'Volume']

#features = ['instant_speed', 'instant_angle']
for feature in tqdm(features):
    #traj = df_partial[(df_partial['TrackID']==1000008597)&(df_partial['Video']=='2-Good-D9-B-M1L-ZT1-070-157-FOV230-320px_Statistics')].reset_index(drop=True)
    df_final = pd.DataFrame()
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        df_partial = df_duration[df_duration['Type'] == cell_type].reset_index(drop=True)
        befores = []
        durings = []
        afters = []

        datas = []
        df_temporal = pd.DataFrame()

        for i in range(0, df_partial.shape[0]):
            if (i == 0) or (i == duration + i0):
                duration = df_partial['Time_span'][i]
                traj = df_partial[i: duration + i].reset_index(drop=True)
                i0 = i

                traj = traj[(traj['pseudo_frame'] != 0) & (traj['pseudo_frame'] != 1)].reset_index(drop=True)

                interaction_profile = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=%s' % interaction_type] > threshold
                interaction_profile = interaction_profile * 1  # Binary interaction profile

                elements = []
                indexes = []
                idx0 = 0
                for element, group in groupby(interaction_profile):  # Groups consistent values
                    # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
                    element_list = list(group)
                    idx1 = len(element_list) + idx0
                    idx_list = list(range(idx0, idx1))
                    idx0 = idx1
                    elements.append(element_list)
                    indexes.append(idx_list)

                check_minduration = np.array([np.sum(element) for element in elements]) >= min_duration
                # list of booleans whether each element have more than 20 persistent contact

                if any(check_minduration) == False:  # false -> No 20 persistent contact
                    continue

                elif np.sum(check_minduration)==1: # only one segment that is 20 persistent contact
                    n_groups = len(elements)
                    for group_idx, (index, element) in enumerate(zip(indexes, elements)):
                        if np.sum(element) >= min_duration:
                            if (group_idx != 0) and (group_idx != n_groups-1): # persistent segment should not be first nor last
                                #print(cell_type, interaction_type, traj['Exp'][0], traj['Video'][0], traj['TrackID'][0])
                                data = []
                                during= traj[feature][index].values
                                during_mean = np.median(during)
                                durings.append(during_mean)

                                before= traj[feature][indexes[group_idx - 1]].values
                                before_mean = np.median(before)
                                befores.append(before_mean)

                                after = traj[feature][indexes[group_idx + 1]].values
                                after_mean = np.median(after)
                                afters.append(after_mean)

                                data.append(before_mean)
                                data.append(during_mean)
                                data.append(after_mean)
                                datas.append(data)
        # befores = flatten_list(befores)
        # durings = flatten_list(durings)
        # afters = flatten_list(afters)

        for d_type in ['pre', 'during', 'post']:
            if d_type == 'pre':
                aaa = befores
            elif d_type == 'during':
                aaa = durings
            elif d_type == 'post':
                aaa = afters

            df_temp = pd.DataFrame()
            df_temp['value'] = aaa
            df_temp['data_type'] = d_type

            df_temporal = pd.concat([df_temporal, df_temp], axis=0)

        df_temporal = df_temporal.reset_index(drop=True)
        df_temporal['Type'] = cell_type

        # df_temporal['before'] = befores
        # df_temporal['during'] = durings
        # df_temporal['after'] = afters
        # df_temporal['Type'] = cell_type

        df_final = pd.concat([df_final, df_temporal], axis=0)

    df_final = df_final.reset_index(drop=True)

    df_final_ = df_final.copy()
    df_final_ = df_final_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
    df_final_['type_data_type'] = df_final_['Type'].astype(str) + ' ' + df_final_['data_type'].astype(str)

    dict_datasets = {}
    for typ in ['WT pre', 'WT during', 'WT post', 'MT pre', 'MT during', 'MT post']:
        values = df_final_[df_final_['type_data_type'] == typ]['value'].values
        dict_datasets[typ] = values

    colors = ('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677',)

    if not os.path.isdir(path + '%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s/' % folder_name)

    # draw_custom_bar_plot(dict_datasets, path+folder_name+'/', file_name=feature,
    #                      strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, return_sig=True,figsize=(2, 2))

    condition_name = 'Type'
    estimator = 'median'
    color_list = ('#CC6677', '#888888')
    marker_list=['o', '^', ]

    mean_dataset = {}
    error_dataset = {}
    p_value_data = {}
    for cell_type in np.unique(df_final[condition_name]):
        df_part = df_final[df_final[condition_name] == cell_type]
        means = []
        errors = []
        valuess = []

        for i in ['pre', 'during', 'post']:
            # if i == custsom_range[1]:
            #     values = df_part[df_part[range_feature] >= i][feature_name].values
            # else:
            values = df_part[(df_part['data_type'] == i)]['value'].values

            if estimator == 'mean':
                mean = np.mean(values)
            elif estimator == 'median':
                mean = np.median(values)

            # means.append(np.mean(values))
            #means.append(np.mean(values))
            interval = stats.norm.interval(confidence=0.95, loc=mean, scale=stats.sem(values))
            error = mean - interval[0]

            means.append(mean)
            errors.append(error)
            valuess.append(values)

            mean_dataset[cell_type] = np.array(means)
            error_dataset[cell_type] = np.array(errors)
            p_value_data[cell_type] = valuess


    font = {'family': 'arial',
            'weight': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(3,3))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    for idx, key in enumerate(mean_dataset):
        sns.lineplot(data=mean_dataset, x=np.arange(0, 3, 1), y=mean_dataset[key],
                     label=key, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars', color=color_list[idx])
        # ax.errorbar(x=np.arange(0, 3, 1), y=mean_dataset[key], yerr=error_dataset[key],
        #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)
        ax.fill_between(x=np.arange(0, 3, 1), y1=mean_dataset[key]-error_dataset[key], y2=mean_dataset[key]+error_dataset[key],
                                     color=color_list[idx], alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['MT'], 0.1*std_dataset['MT'], color='#CC6677',
    #            capsize=3, capthick=1, elinewidth=1.5)
    # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['WT'], 0.1*std_dataset['WT'], color='#888888',
    #            capsize=3, capthick=1, elinewidth=1.5)
    from scipy import stats
    import statsmodels.api as sm

    p_values = []
    pairs = []
    cohen_ds = []
    for idx, (mt_values, wt_values) in enumerate(zip(p_value_data[list(p_value_data.keys())[0]], p_value_data[list(p_value_data.keys())[1]])):
        if test == 'mann-whitney':
            stat_test = stats.mannwhitneyu(mt_values, wt_values)
        elif test == 't-test':
            stat_test = stats.ttest_ind(mt_values, wt_values)
        elif test == 'wilcoxon-ranksum':
            stat_test = stats.ranksums(mt_values, wt_values)
        # print(idx, stat_test.pvalue)
        p_values.append(stat_test.pvalue)
        pairs.append(idx)

        cohen_d = Cohen_d(mt_values, wt_values)
        cohen_ds.append(cohen_d)
    #_, p_values, _, _ = sm.stats.multipletests(p_values, alpha=0.05,method='fdr_bh')  # Benjamini/Hochberg adjustment

    plt.title('p: %s, d: %s' % (p_values, cohen_ds), fontsize=4)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')
    if feature in ['instant_speed', 'instant_angle', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core',]:
        plt.ylim(0, None)
    else:
        plt.ylim(None, None)
    #ax.set_xlabel('%s'%x_label, fontsize=16, weight='normal', color='0.2')
    plt.xticks(np.arange(0, 3, 1), labels=['before', 'during', 'after'], fontsize=12, color='0.2', weight='normal', )
    plt.yticks(fontsize=16, color='0.2', weight='normal')

    # plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
    #            loc='best')
    ax.get_legend().remove()
    if not os.path.isdir(path + '%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s/' % folder_name)
    plt.savefig(path + '%s/%s.png' % (folder_name, feature), dpi=300, bbox_inches='tight')


    if not os.path.isdir(path + 'svg/%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/%s/' % folder_name)
    plt.savefig(path + 'svg/%s/%s.svg' % (folder_name, feature), bbox_inches='tight')


############################### inst speed and angle before during after ################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure3. Interaction with Tfh of wt and mt GCB\\'
threshold = 0
min_duration = 15
interaction_type = 'T-cell' #'FDC', 'T-cell'

features = ['instant_speed', 'instant_angle', 'Zone',
            'Sphericity', 'Volume', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core',
            'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC','diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'diff_Distance_to_DZ', 'diff_Distance_to_LZ', 'diff_Distance_to_FDC_core',
            'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell']


df_bda_motility = pd.DataFrame()
for feature in tqdm(features):
    #traj = df_partial[(df_partial['TrackID']==1000008597)&(df_partial['Video']=='2-Good-D9-B-M1L-ZT1-070-157-FOV230-320px_Statistics')].reset_index(drop=True)
    df_final = pd.DataFrame()
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        df_partial = df_duration[df_duration['Type'] == cell_type].reset_index(drop=True)
        befores = []
        durings = []
        afters = []
        labels = []
        overlap_volumes = []
        dLZ_distances = []
        datas = []
        df_temporal = pd.DataFrame()
        count = 0
        for i in range(0, df_partial.shape[0]):
            if (i == 0) or (i == duration + i0):
                duration = df_partial['Time_span'][i]
                traj = df_partial[i: duration + i].reset_index(drop=True)
                i0 = i
                count = count + 1
                traj = traj[(traj['pseudo_frame'] != 0) & (traj['pseudo_frame'] != 1)].reset_index(drop=True)

                interaction_profile = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=%s' % interaction_type] > threshold
                interaction_profile = interaction_profile * 1  # Binary interaction profile

                elements = []
                indexes = []
                idx0 = 0
                for element, group in groupby(interaction_profile):  # Groups consistent values
                    # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
                    element_list = list(group)
                    idx1 = len(element_list) + idx0
                    idx_list = list(range(idx0, idx1))
                    idx0 = idx1
                    elements.append(element_list)
                    indexes.append(idx_list)

                check_minduration = np.array([np.sum(element) for element in elements]) >= min_duration
                # list of booleans whether each element have more than 20 persistent contact

                if any(check_minduration) == False:  # false -> No 20 persistent contact
                    continue

                elif np.sum(check_minduration)==1: # only one segment that is 20 persistent contact
                    n_groups = len(elements)
                    for group_idx, (index, element) in enumerate(zip(indexes, elements)):
                        if np.sum(element) >= min_duration:
                            if (group_idx != 0) and (group_idx != n_groups-1): # persistent segment should not be first nor last
                                #print(cell_type, interaction_type, traj['Exp'][0], traj['Video'][0], traj['TrackID'][0])
                                label = traj['Label'][0]
                                data = []
                                during= traj[feature][index].values
                                during_mean = np.median(during)
                                durings.append(during_mean)

                                before= traj[feature][indexes[group_idx - 1]].values
                                before_mean = np.median(before)
                                befores.append(before_mean)

                                after = traj[feature][indexes[group_idx + 1]].values
                                after_mean = np.median(after)
                                afters.append(after_mean)

                                overlap = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=%s' % interaction_type][index].values
                                overlap_mean = np.mean(overlap)

                                dLZ_distance = traj['Distance_to_FDC_core'][index].values
                                dLZ_distance_mean = np.mean(dLZ_distance)

                                data.append(before_mean)
                                data.append(during_mean)
                                data.append(after_mean)
                                labels.append(label)
                                datas.append(data)
                                overlap_volumes.append(overlap_mean)
                                dLZ_distances.append(dLZ_distance_mean)
        # befores = flatten_list(befores)
        # durings = flatten_list(durings)
        # afters = flatten_list(afters)
        print(cell_type, count)
        for d_type in ['before', 'during', 'after']:
            if d_type == 'before':
                aaa = befores
            elif d_type == 'during':
                aaa = durings
            elif d_type == 'after':
                aaa = afters

            df_temp = pd.DataFrame()
            df_temp['value'] = aaa
            df_temp['data_type'] = d_type
            df_temp['Label'] = labels
            df_temp['Overlap_volume'] = overlap_volumes
            df_temp['dLZ_distance'] = dLZ_distances
            df_temporal = pd.concat([df_temporal, df_temp], axis=0)

        df_temporal = df_temporal.reset_index(drop=True)
        df_temporal['Type'] = cell_type

        # df_temporal['before'] = befores
        # df_temporal['during'] = durings
        # df_temporal['after'] = afters
        # df_temporal['Type'] = cell_type

        df_final = pd.concat([df_final, df_temporal], axis=0)

    df_final = df_final.reset_index(drop=True)

    df_bda_motility[feature] = df_final['value']

df_bda_motility['data_type'] = df_final['data_type']
df_bda_motility['Type'] = df_final['Type']
df_bda_motility['Label'] = df_final['Label']
df_bda_motility['Overlap_volume'] = df_final['Overlap_volume']
df_bda_motility['dLZ_distance'] = df_final['dLZ_distance']
df_bda_motility['type_label'] = df_final['Type'].astype(str) + '_' +df_final['Label'].astype(str)



thresh = 0.07
df_bda_motility.loc[(df_bda_motility['Overlap_volume'] < thresh), 'Overlap_quality'] = 'Low'
df_bda_motility.loc[(df_bda_motility['Overlap_volume'] >= thresh), 'Overlap_quality'] = 'High'

for typ in ['wt_B-cell', 'mt_B-cell']:
    for zone in ['Low', 'High']:
        print(typ, zone, df_bda_motility[(df_bda_motility['Type']==typ)&(df_bda_motility['Overlap_quality']==zone)].shape[0]/3)

########
feature = 'instant_speed'

df_ts = pd.DataFrame()
time_series = []
for label in np.unique(df_bda_motility['type_label']):
    df_bda_motility_label = df_bda_motility[df_bda_motility['type_label'] == label]
    cell_type = df_bda_motility_label['Type'].values[0]

    before = df_bda_motility_label[df_bda_motility_label['data_type'] == 'before'][feature].values[0]
    during = df_bda_motility_label[df_bda_motility_label['data_type'] == 'during'][feature].values[0]
    after = df_bda_motility_label[df_bda_motility_label['data_type'] == 'after'][feature].values[0]

    ts = np.array([before, during, after])
    time_series.append(ts)

    df_ts_temp = pd.DataFrame()
    df_ts_temp['Type'] = [cell_type]
    df_ts_temp['Label'] = [label]
    df_ts_temp['before'] = [before]
    df_ts_temp['during'] = [during]
    df_ts_temp['after'] = [after]
    df_ts = pd.concat([df_ts, df_ts_temp], axis=0)

time_series = np.array(time_series)

time_series_dict = array_to_dict(time_series)
time_series_dict_scaled = normalize_timeseries(time_series_dict)
time_series_scaled = dict_to_array(time_series_dict_scaled)
from tslearn.clustering import TimeSeriesKMeans
tskm = TimeSeriesKMeans(n_clusters=4, metric='euclidean', random_state=0, verbose=True, max_iter=100)
tskmeans_predicted = tskm.fit_predict(time_series_scaled)

df_ts['tskmeans'] = tskmeans_predicted
df_ts, replace_map = order_cluster_by_feature(df_ts, cluster_name='tskmeans', feature_name='before')
tskm_cluster_centers_temp = tskm.cluster_centers_

tskm_cluster_centers= np.zeros_like(tskm_cluster_centers_temp)
# Apply the mapping to reorder rows
for old_idx, new_idx in replace_map.items():
    tskm_cluster_centers[new_idx] = tskm_cluster_centers_temp[old_idx]

df_ts['before_scaled'] = time_series_scaled[:, 0]
df_ts['during_scaled'] = time_series_scaled[:, 1]
df_ts['after_scaled'] = time_series_scaled[:, 2]


########## pre/post contact motility mode distribution ################
df_ts_ = df_ts.copy()
df_ts_ = df_ts_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
draw_cluster_distribution_heatmap(df_ts_, path, file_name='tskmeans_type_%s_heatmap'%feature, condition_name='Type', cluster_type='tskmeans',
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(3.5,2))


p_dict = permutation_test(df_ts_, group_name='Type', class_name='tskmeans', iteration=10000)

########## pre/post contact motility representatibe cluster ################
for cluster, tskm_cluster_center in enumerate(tskm_cluster_centers):
    ts = tskm_cluster_center.flatten()
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.lineplot(x=np.arange(0, 2 + 1, 1),
                 y=ts, lw=2.5,
                 dashes=False, markersize=8, err_style='bars',
                 color='#888888')
    if not os.path.isdir(path + 'tskmeans_cluster_%s/'%feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'tskmeans_cluster_%s/'%feature)
    plt.savefig(path + 'tskmeans_cluster_%s/cluster_%s.png' % (feature, cluster), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/tskmeans_cluster_%s/'%feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/tskmeans_cluster_%s/'%feature)
    plt.savefig(path + 'svg/tskmeans_cluster_%s/cluster_%s.svg' % (feature, cluster), dpi=300, bbox_inches='tight')

    plt.clf()
    plt.close()

########## pre/post contact motility mode actual time series per cluster ################
for cluster in np.unique(df_ts['tskmeans']):
    df_ts_part = df_ts[df_ts['tskmeans'] == cluster]
    for idx, each_cell in df_ts_part.iterrows():
        label = each_cell['Label']
        values = each_cell[['before', 'during', 'after']].values
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.lineplot(x=np.arange(0, 2 + 1, 1),
                     y=values, lw=2.5,
                     dashes=False, markersize=8, err_style='bars',
                     color='#888888')
        if not os.path.isdir(path + 'actual_ts_%s/cluster_%s/'%(feature, cluster)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'actual_ts_%s/cluster_%s/'%(feature, cluster))

        plt.savefig(path + 'actual_ts_%s/cluster_%s/%s.png' % (feature, cluster, label), dpi=300, bbox_inches='tight')

        if not os.path.isdir(path + 'svg/actual_ts_%s/cluster_%s/'%(feature, cluster)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/actual_ts_%s/cluster_%s/'%(feature, cluster))

        plt.savefig(path + 'svg/actual_ts_%s/cluster_%s/%s.svg' % (feature, cluster, label), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()


########## scaled (before-during) of WT vs MT ################
df_ts_ = df_ts.copy()
df_ts_ = df_ts_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

df_ts_['before-during_scaled'] = df_ts_['before_scaled'] - df_ts_['during_scaled']
df_ts_['after-during_scaled'] = df_ts_['after_scaled'] - df_ts_['during_scaled']


dict_datasets = {}
for typ in ['WT', 'MT']:
    values = df_ts_[(df_ts_['Type']==typ)]['before-during_scaled'].values
    #values = df_ts_[(df_ts_['Type'] == typ) & (df_ts_['tskmeans'] != 0)]['before-during_scaled'].values
    minus_values = values[values < 0]
    plus_values = values[values > 0]
    dict_datasets[typ+'_minus'] = minus_values
    dict_datasets[typ + '_plus'] = plus_values

new_order = ['WT_minus', 'MT_minus', 'WT_plus', 'MT_plus']
dict_datasets = change_dict_order(dict_datasets, new_order)
[print(key, np.array(value).size) for key, value in dict_datasets.items()]

colors = ('#888888', '#CC6677',)*2
draw_custom_bar_plot(dict_datasets, path, file_name='before-during_%s'%feature,
                     strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, return_sig=False, figsize=(1, 2))



dict_datasets = {}
for typ in ['WT', 'MT']:
    values = df_ts_[(df_ts_['Type']==typ)]['after-during_scaled'].values
    #values = df_ts_[(df_ts_['Type'] == typ) & (df_ts_['tskmeans'] != 0)]['before-during_scaled'].values
    minus_values = values[values < 0]
    plus_values = values[values > 0]
    dict_datasets[typ+'_minus'] = minus_values
    dict_datasets[typ + '_plus'] = plus_values

new_order = ['WT_minus', 'MT_minus', 'WT_plus', 'MT_plus']
dict_datasets = change_dict_order(dict_datasets, new_order)
[print(key, np.array(value).size) for key, value in dict_datasets.items()]
colors = ('#888888', '#CC6677',)*2
draw_custom_bar_plot(dict_datasets, path, file_name='after-during_%s'%feature,
                     strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, return_sig=False, figsize=(1, 2))




########## (before-during) of WT vs MT ################
df_ts_ = df_ts.copy()
df_ts_ = df_ts_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

df_ts_['before-during'] = (df_ts_['before'] - df_ts_['during'])/df_ts_['before']
df_ts_['before-after'] = (df_ts_['before'] - df_ts_['after'])/df_ts_['before']
df_ts_['after-during'] = (df_ts_['after'] - df_ts_['during'])/df_ts_['before']


df_ts_['before-during'] = (df_ts_['before'] - df_ts_['during'])
df_ts_['before-after'] = (df_ts_['before'] - df_ts_['after'])
df_ts_['after-during'] = (df_ts_['after'] - df_ts_['during'])

df_ts_['before-during'] = (df_ts_['before'] - df_ts_['during'])/np.mean(df_ts_[['before', 'during', 'after']], axis=1)
df_ts_['before-after'] = (df_ts_['before'] - df_ts_['after'])/np.mean(df_ts_[['before', 'during', 'after']], axis=1)
df_ts_['after-during'] = (df_ts_['after'] - df_ts_['during'])/np.mean(df_ts_[['before', 'during', 'after']], axis=1)

# df_ts_['before-during'] = (df_ts_['before'] - df_ts_['during'])/np.mean(df_ts_['after'])
# df_ts_['before-after'] = (df_ts_['before'] - df_ts_['after'])/np.mean(df_ts_['during'])
# df_ts_['after-during'] = (df_ts_['after'] - df_ts_['during'])/np.mean(df_ts_['before'])


#df_ts_['before-during'] = (df_ts_['before'] - df_ts_['during'])/(df_ts_['before'] - df_ts_['after'])

aa = df_ts_[df_ts_['Type'] =='WT'].reset_index(drop=True)
dict_datasets = {}
for typ in ['WT', 'MT']:
    #values = df_ts_[(df_ts_['Type']==typ)&(df_ts_['tskmeans']!=0)]['before-during'].values
    values = df_ts_[(df_ts_['Type'] == typ)]['before-during'].values
    #print(typ, np.mean(values))
    values = values[values < 0]
    dict_datasets[typ] = values

colors = ('#888888', '#CC6677',)
draw_custom_bar_plot(dict_datasets, path, file_name='before-during_%s'%feature,
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))

dict_datasets = {}
for typ in ['WT', 'MT']:
    values = df_ts_[df_ts_['Type']==typ]['before-after'].values
    values = values[values < 0]
    dict_datasets[typ] = values

colors = ('#888888', '#CC6677',)
draw_custom_bar_plot(dict_datasets, path, file_name='before-after_%s'%feature,
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))


dict_datasets = {}
for typ in ['WT', 'MT']:
    values = df_ts_[df_ts_['Type']==typ]['after-during'].values
    values = values[values < 0]
    dict_datasets[typ] = values

colors = ('#888888', '#CC6677',)
draw_custom_bar_plot(dict_datasets, path, file_name='after-during_%s'%feature,
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))







########## Speed vs angle joint plot ################

for typ in np.unique(df_bda_motility['Type']):
    for d_type in ['before', 'during', 'after']:
        print(typ, d_type, df_bda_motility[(df_bda_motility['Type']==typ)&(df_bda_motility['data_type']==d_type)].shape[0])

file_name = 'instant speed vs instant angle'

df_bda_motility_part = df_bda_motility[df_bda_motility['Type']=='wt_B-cell']

draw_jointplot(xs='instant_speed', y='instant_angle', df=df_bda_motility_part, path=path, file_name=file_name,
               hue="data_type", colors=('#1f77b4', '#888888', '#d62728'), hue_order=['before', 'during', 'after'],
               legend=False, fill=False, thresh=0.15, n_contours=3, alpha=1, height=4, ratio=5, space=0, xlabels='Speed', ylabel='Angle')


file_name = 'instant speed vs instant angle'
fig, ax = plt.subplots(figsize=(4,4))
# sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
#                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
#for idx, key in enumerate(dataset):
sns.scatterplot(data=df_bda_motility_part, x='instant_speed', y='instant_angle', hue='Type', lw=2.5,  s=20, hue_order=['wt_B-cell', 'mt_B-cell'],
                         palette=('#888888', '#CC6677'), style='data_type')
handles, labels = ax.get_legend_handles_labels()

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_xlabel('Instant speed (μm/min)', fontsize=16, weight='normal', color='0.2')
ax.set_ylabel('Instant turning angle (rad/min)', fontsize=16, weight='normal', color='0.2')

#custom_range = (70, 95)
# ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='normal')
# ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='normal')

# plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
#            weight='normal')
plt.xticks(fontsize=12, color='0.2', weight='normal')
plt.yticks(fontsize=12, color='0.2', weight='normal')
#plt.xlim(0, 13)
plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
           loc='best')
plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()


############################### interaction features before during after ################################
features = ['Sphericity', 'Volume', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core',
            'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'diff_Distance_to_FDC_core', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell'
            #'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC','diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            #'diff_Distance_to_DZ', 'diff_Distance_to_LZ',
            ]

p_dicts = {}
for feature in features:
    df_ts = pd.DataFrame()
    time_series = []
    cell_types = []
    labels = []
    for label in np.unique(df_bda_motility['type_label']):
        df_bda_motility_label = df_bda_motility[df_bda_motility['type_label'] == label]
        cell_type = df_bda_motility_label['Type'].values[0]

        before = df_bda_motility_label[df_bda_motility_label['data_type'] == 'before'][feature].values[0]
        during = df_bda_motility_label[df_bda_motility_label['data_type'] == 'during'][feature].values[0]
        after = df_bda_motility_label[df_bda_motility_label['data_type'] == 'after'][feature].values[0]

        ts = np.array([before, during, after])
        time_series.append(ts)
        cell_types.append(cell_type)
        labels.append(label)
        df_ts_temp = pd.DataFrame()
        df_ts_temp['Type'] = [cell_type]
        df_ts_temp['Label'] = [label]
        df_ts_temp['before'] = [before]
        df_ts_temp['during'] = [during]
        df_ts_temp['after'] = [after]
        df_ts = pd.concat([df_ts, df_ts_temp], axis=0)


    time_series = np.array(time_series)

    time_series_dict = array_to_dict(time_series)
    time_series_dict_scaled = normalize_timeseries(time_series_dict)
    time_series_scaled = dict_to_array(time_series_dict_scaled)
    from tslearn.clustering import TimeSeriesKMeans
    tskm = TimeSeriesKMeans(n_clusters=4, metric='euclidean', random_state=0, verbose=True, max_iter=100)
    tskmeans_predicted = tskm.fit_predict(time_series_scaled)

    df_ts['tskmeans'] = tskmeans_predicted
    df_ts, replace_map = order_cluster_by_feature(df_ts, cluster_name='tskmeans', feature_name='before')
    tskm_cluster_centers_temp = tskm.cluster_centers_

    tskm_cluster_centers= np.zeros_like(tskm_cluster_centers_temp)
    # Apply the mapping to reorder rows
    for old_idx, new_idx in replace_map.items():
        tskm_cluster_centers[new_idx] = tskm_cluster_centers_temp[old_idx]

    df_ts['before_scaled'] = time_series_scaled[:, 0]
    df_ts['during_scaled'] = time_series_scaled[:, 1]
    df_ts['after_scaled'] = time_series_scaled[:, 2]

    ########## pre/post contact motility mode distribution ################
    df_ts_ = df_ts.copy()
    df_ts_ = df_ts_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
    draw_cluster_distribution_heatmap(df_ts_, path+'before_during_after_int_features/', file_name='tskmeans_type_%s_heatmap' % feature,
                                      condition_name='Type', cluster_type='tskmeans',
                                      annot=True, col_cluster=False, row_cluster=False, figsize=(3.5, 2))

    p_dict = permutation_test(df_ts_, group_name='Type', class_name='tskmeans', iteration=10000)
    p_dicts[feature] = p_dict
    ########## pre/post contact motility representatibe cluster ################
    for cluster, tskm_cluster_center in enumerate(tskm_cluster_centers):
        ts = tskm_cluster_center.flatten()
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.lineplot(x=np.arange(0, 2 + 1, 1),
                     y=ts, lw=2.5,
                     dashes=False, markersize=8, err_style='bars',
                     color='#888888')
        if not os.path.isdir(
                path+'before_during_after_int_features/' + 'tskmeans_cluster_%s/' % feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path+'before_during_after_int_features/' + 'tskmeans_cluster_%s/' % feature)
        plt.savefig(path+'before_during_after_int_features/' + 'tskmeans_cluster_%s/cluster_%s.png' % (feature, cluster), dpi=300, bbox_inches='tight')

        if not os.path.isdir(
                path+'before_during_after_int_features/' + 'svg/tskmeans_cluster_%s/' % feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path+'before_during_after_int_features/' + 'svg/tskmeans_cluster_%s/' % feature)
        plt.savefig(path+'before_during_after_int_features/' + 'svg/tskmeans_cluster_%s/cluster_%s.svg' % (feature, cluster), dpi=300, bbox_inches='tight')

        plt.clf()
        plt.close()

    ########## pre/post contact motility mode actual time series per cluster ################
    for cluster in np.unique(df_ts['tskmeans']):
        df_ts_part = df_ts[df_ts['tskmeans'] == cluster]
        for idx, each_cell in df_ts_part.iterrows():
            label = each_cell['Label']
            values = each_cell[['before', 'during', 'after']].values
            fig, ax = plt.subplots(figsize=(2, 2))
            sns.lineplot(x=np.arange(0, 2 + 1, 1),
                         y=values, lw=2.5,
                         dashes=False, markersize=8, err_style='bars',
                         color='#888888')
            if not os.path.isdir(path+'before_during_after_int_features/' + 'actual_ts_%s/cluster_%s/' % (
            feature, cluster)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(path+'before_during_after_int_features/' + 'actual_ts_%s/cluster_%s/' % (feature, cluster))

            plt.savefig(path+'before_during_after_int_features/' + 'actual_ts_%s/cluster_%s/%s.png' % (feature, cluster, label[:9]+label[-11:]), dpi=300,
                        bbox_inches='tight')

            if not os.path.isdir(path+'before_during_after_int_features/' + 'svg/actual_ts_%s/cluster_%s/' % (
            feature, cluster)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(path+'before_during_after_int_features/' + 'svg/actual_ts_%s/cluster_%s/' % (feature, cluster))

            plt.savefig(path+'before_during_after_int_features/' + 'svg/actual_ts_%s/cluster_%s/%s.svg' % (feature, cluster, label[:9]+label[-11:]), dpi=300,
                        bbox_inches='tight')
            plt.clf()
            plt.close()

    ########## (before-during) of WT vs MT ################

    df_ts_ = df_ts.copy()
    df_ts_ = df_ts_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
    df_ts_['before-during'] = (df_ts_['before_scaled'] - df_ts_['during_scaled'])
    df_ts_['before-after'] = (df_ts_['before_scaled'] - df_ts_['after_scaled'])
    df_ts_['after-during'] = (df_ts_['after_scaled'] - df_ts_['during_scaled'])


    dict_datasets = {}
    for typ in ['WT', 'MT']:
        #values = df_ts_[(df_ts_['Type']==typ)&(df_ts_['tskmeans']!=0)]['before-during'].values
        values = df_ts_[(df_ts_['Type'] == typ)]['before-during'].values
        #print(typ, np.mean(values))
        #values = values[values < 0]
        dict_datasets[typ] = values

    colors = ('#888888', '#CC6677',)
    draw_custom_bar_plot(dict_datasets, path+'before_during_after_int_features/', file_name='%s_before-during'%feature,
                         strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))

    dict_datasets = {}
    for typ in ['WT', 'MT']:
        values = df_ts_[df_ts_['Type']==typ]['before-after'].values
        #values = values[values < 0]
        dict_datasets[typ] = values

    colors = ('#888888', '#CC6677',)
    draw_custom_bar_plot(dict_datasets, path+'before_during_after_int_features/', file_name='%s_before-after'%feature,
                         strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))


    dict_datasets = {}
    for typ in ['WT', 'MT']:
        values = df_ts_[df_ts_['Type']==typ]['after-during'].values
        #values = values[values < 0]
        dict_datasets[typ] = values

    colors = ('#888888', '#CC6677',)
    draw_custom_bar_plot(dict_datasets, path+'before_during_after_int_features/', file_name='%s_after-during'%feature,
                         strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))

    ########## (before-during minus and plus) of WT vs MT ################
    dict_datasets = {}
    for typ in ['WT', 'MT']:
        values = df_ts_[(df_ts_['Type'] == typ)]['before-during'].values
        # values = df_ts_[(df_ts_['Type'] == typ) & (df_ts_['tskmeans'] != 0)]['before-during_scaled'].values
        minus_values = values[values < 0]
        plus_values = values[values > 0]
        dict_datasets[typ + '_minus'] = minus_values
        dict_datasets[typ + '_plus'] = plus_values

    new_order = ['WT_minus', 'MT_minus', 'WT_plus', 'MT_plus']
    dict_datasets = change_dict_order(dict_datasets, new_order)
    [print(key, np.array(value).size) for key, value in dict_datasets.items()]
    try:
        colors = ('#888888', '#CC6677',) * 2
        draw_custom_bar_plot(dict_datasets, path + 'before_during_after_plus_minus_int_features/',
                             file_name='%s_before-during' % feature,
                             strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, return_sig=False,
                             figsize=(1, 2))
    except:
        print('cannot generate graph')
    dict_datasets = {}
    for typ in ['WT', 'MT']:
        values = df_ts_[(df_ts_['Type'] == typ)]['after-during'].values
        # values = df_ts_[(df_ts_['Type'] == typ) & (df_ts_['tskmeans'] != 0)]['before-during_scaled'].values
        minus_values = values[values < 0]
        plus_values = values[values > 0]
        dict_datasets[typ + '_minus'] = minus_values
        dict_datasets[typ + '_plus'] = plus_values

    new_order = ['WT_minus', 'MT_minus', 'WT_plus', 'MT_plus']
    dict_datasets = change_dict_order(dict_datasets, new_order)
    [print(key, np.array(value).size) for key, value in dict_datasets.items()]
    try:
        colors = ('#888888', '#CC6677',) * 2
        draw_custom_bar_plot(dict_datasets, path + 'before_during_after_plus_minus_int_features/',
                             file_name='%s_after-during' % feature,
                             strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, return_sig=False,
                             figsize=(1, 2))
    except:
        print('cannot generate graph')




############################### (classified based on overlapping volume) interaction features before during after ################################
features = ['Sphericity', 'Volume', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core',
            'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'diff_Distance_to_FDC_core', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC'
            #'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC','diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            #'diff_Distance_to_DZ', 'diff_Distance_to_LZ',
            ]
p_dicts = {}

feature = 'Distance_to_FDC_core'
for feature in features:
    df_ts = pd.DataFrame()
    time_series = []
    cell_types = []
    labels = []
    overlaps = []
    for label in np.unique(df_bda_motility['type_label']):
        df_bda_motility_label = df_bda_motility[df_bda_motility['type_label'] == label]
        cell_type = df_bda_motility_label['Type'].values[0]

        before = df_bda_motility_label[df_bda_motility_label['data_type'] == 'before'][feature].values[0]
        during = df_bda_motility_label[df_bda_motility_label['data_type'] == 'during'][feature].values[0]
        after = df_bda_motility_label[df_bda_motility_label['data_type'] == 'after'][feature].values[0]

        overlap = df_bda_motility_label['Overlap_volume'].values[0]
        dLZ_distance = df_bda_motility_label['dLZ_distance'].values[0]

        ts = np.array([before, during, after])
        time_series.append(ts)
        cell_types.append(cell_type)
        labels.append(label)
        overlaps.append(overlap)
        df_ts_temp = pd.DataFrame()
        df_ts_temp['Type'] = [cell_type]
        df_ts_temp['Label'] = [label]
        df_ts_temp['before'] = [before]
        df_ts_temp['during'] = [during]
        df_ts_temp['after'] = [after]
        df_ts_temp['Overlap'] = [overlap]
        df_ts_temp['dLZ_distance'] = [dLZ_distance]
        df_ts = pd.concat([df_ts, df_ts_temp], axis=0)

    thresh = 0.07
    df_ts.loc[(df_ts['Overlap'] < thresh), 'Overlap_quality'] = 'Low'
    df_ts.loc[(df_ts['Overlap'] >= thresh), 'Overlap_quality'] = 'High'

    thresh = 20
    df_ts.loc[(df_ts['dLZ_distance'] < thresh), 'dLZ'] = 'dLZ'
    df_ts.loc[(df_ts['dLZ_distance'] >= thresh), 'dLZ'] = 'non-dLZ'

    # if feature == 'Distance_to_FDC_core':
    #     df_ts['DZ_transition'] = (df_ts['after'] - df_ts['during'] > 10).map({True: 'DZ transition', False: 'no transition'})

    for typ in ['wt_B-cell', 'mt_B-cell']:
        for zone in ['Low', 'High']:
            print(typ, zone, df_ts[(df_ts['Type']==typ)&(df_ts['Overlap_quality']==zone)].shape[0])

    for typ in ['wt_B-cell', 'mt_B-cell']:
        for zone in ['non-dLZ', 'dLZ']:
            print(typ, zone, df_ts[(df_ts['Type']==typ)&(df_ts['dLZ']==zone)].shape[0])

    time_series = np.array(time_series)

    time_series_dict = array_to_dict(time_series)
    time_series_dict_scaled = normalize_timeseries(time_series_dict)
    time_series_scaled = dict_to_array(time_series_dict_scaled)
    from tslearn.clustering import TimeSeriesKMeans
    tskm = TimeSeriesKMeans(n_clusters=4, metric='euclidean', random_state=0, verbose=True, max_iter=100)
    tskmeans_predicted = tskm.fit_predict(time_series_scaled)

    df_ts['tskmeans'] = tskmeans_predicted
    df_ts, replace_map = order_cluster_by_feature(df_ts, cluster_name='tskmeans', feature_name='before')
    tskm_cluster_centers_temp = tskm.cluster_centers_

    tskm_cluster_centers= np.zeros_like(tskm_cluster_centers_temp)
    # Apply the mapping to reorder rows
    for old_idx, new_idx in replace_map.items():
        tskm_cluster_centers[new_idx] = tskm_cluster_centers_temp[old_idx]

    df_ts['before_scaled'] = time_series_scaled[:, 0]
    df_ts['during_scaled'] = time_series_scaled[:, 1]
    df_ts['after_scaled'] = time_series_scaled[:, 2]

    ########## pre/post contact motility mode distribution ################
    df_ts_ = df_ts.copy()
    df_ts_ = df_ts_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
    df_ts_['type_quality'] = df_ts_['Type'].astype(str) + ' ' + df_ts_['Overlap_quality']
    draw_cluster_distribution_heatmap(df_ts_, path+'before_during_after_int_features_overlap_quality/', file_name='tskmeans_type_quality_%s_heatmap' % feature,
                                      condition_name='type_quality', cluster_type='tskmeans',
                                      annot=True, col_cluster=False, row_cluster=False, figsize=(3.5, 3))

    df_ts_['type_dLZ'] = df_ts_['Type'].astype(str) + ' ' + df_ts_['dLZ']
    draw_cluster_distribution_heatmap(df_ts_, path + 'before_during_after_int_features_overlap_quality/',
                                      file_name='tskmeans_type_dLZ_%s_heatmap' % feature,
                                      condition_name='type_dLZ', cluster_type='tskmeans',
                                      annot=True, col_cluster=False, row_cluster=False, figsize=(3.5, 3))

    p_dict = permutation_test(df_ts_, group_name='type_quality', class_name='tskmeans', iteration=10000)
    p_dicts[feature] = p_dict
    ########## pre/post contact motility representatibe cluster ################
    for cluster, tskm_cluster_center in enumerate(tskm_cluster_centers):
        ts = tskm_cluster_center.flatten()
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.lineplot(x=np.arange(0, 2 + 1, 1),
                     y=ts, lw=2.5,
                     dashes=False, markersize=8, err_style='bars',
                     color='#888888')
        if not os.path.isdir(
                path+'before_during_after_int_features_overlap_quality/' + 'tskmeans_cluster_%s/' % feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path+'before_during_after_int_features_overlap_quality/' + 'tskmeans_cluster_%s/' % feature)
        plt.savefig(path+'before_during_after_int_features_overlap_quality/' + 'tskmeans_cluster_%s/cluster_%s.png' % (feature, cluster), dpi=300, bbox_inches='tight')

        if not os.path.isdir(
                path+'before_during_after_int_features_overlap_quality/' + 'svg/tskmeans_cluster_%s/' % feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path+'before_during_after_int_features_overlap_quality/' + 'svg/tskmeans_cluster_%s/' % feature)
        plt.savefig(path+'before_during_after_int_features_overlap_quality/' + 'svg/tskmeans_cluster_%s/cluster_%s.svg' % (feature, cluster), dpi=300, bbox_inches='tight')

        plt.clf()
        plt.close()


    ########## (before-during) of WT vs MT ################

    df_ts_ = df_ts.copy()
    df_ts_ = df_ts_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
    df_ts_['type_quality'] = df_ts_['Type'].astype(str) + ' ' + df_ts_['Overlap_quality']
    df_ts_['before-during'] = (df_ts_['before_scaled'] - df_ts_['during_scaled'])
    df_ts_['before-after'] = (df_ts_['before_scaled'] - df_ts_['after_scaled'])
    df_ts_['after-during'] = (df_ts_['after_scaled'] - df_ts_['during_scaled'])


    dict_datasets = {}
    for typ in ['WT Low', 'WT High', 'MT Low', 'MT High']:
        #values = df_ts_[(df_ts_['Type']==typ)&(df_ts_['tskmeans']!=0)]['before-during'].values
        values = df_ts_[(df_ts_['type_quality'] == typ)]['before-during'].values
        #print(typ, np.mean(values))
        #values = values[values < 0]
        dict_datasets[typ] = values

    colors = ('#888888', '#888888', '#CC6677','#CC6677',)
    draw_custom_bar_plot(dict_datasets, path+'before_during_after_int_features_overlap_quality/', file_name='%s_before-during'%feature,
                         strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))

    dict_datasets = {}
    for typ in ['WT Low', 'WT High', 'MT Low', 'MT High']:
        values = df_ts_[df_ts_['type_quality']==typ]['before-after'].values
        #values = values[values < 0]
        dict_datasets[typ] = values

    colors = ('#888888', '#888888', '#CC6677','#CC6677',)
    draw_custom_bar_plot(dict_datasets, path+'before_during_after_int_features_overlap_quality/', file_name='%s_before-after'%feature,
                         strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))


    dict_datasets = {}
    for typ in ['WT Low', 'WT High', 'MT Low', 'MT High']:
        values = df_ts_[df_ts_['type_quality']==typ]['after-during'].values
        #values = values[values < 0]
        dict_datasets[typ] = values

    colors = ('#888888', '#888888', '#CC6677','#CC6677',)
    draw_custom_bar_plot(dict_datasets, path+'before_during_after_int_features_overlap_quality/', file_name='%s_after-during'%feature,
                         strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))




    ########## (before-during minus and plus) of WT vs MT ################
    dict_datasets = {}
    for typ in ['WT Low', 'WT High', 'MT Low', 'MT High']:
        values = df_ts_[(df_ts_['type_quality'] == typ)]['before-during'].values
        # values = df_ts_[(df_ts_['Type'] == typ) & (df_ts_['tskmeans'] != 0)]['before-during_scaled'].values
        minus_values = values[values < 0]
        plus_values = values[values > 0]
        dict_datasets[typ + '_minus'] = minus_values
        dict_datasets[typ + '_plus'] = plus_values

    new_order = ['WT Low_minus', 'WT High_minus', 'MT Low_minus', 'MT High_minus', 'WT Low_plus', 'WT High_plus', 'MT Low_plus', 'MT High_plus']
    dict_datasets = change_dict_order(dict_datasets, new_order)
    [print(key, np.array(value).size) for key, value in dict_datasets.items()]
    try:
        colors = ('#888888', '#888888', '#CC6677', '#CC6677',) * 2
        draw_custom_bar_plot(dict_datasets, path + 'before_during_after_plus_minus_int_features_overlap_quality/',
                             file_name='%s_before-during' % feature,
                             strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, return_sig=True,
                             figsize=(2, 2))
    except:
        print('cannot generate graph')

    dict_datasets = {}
    for typ in ['WT Low', 'WT High', 'MT Low', 'MT High']:
        values = df_ts_[(df_ts_['type_quality'] == typ)]['after-during'].values
        # values = df_ts_[(df_ts_['Type'] == typ) & (df_ts_['tskmeans'] != 0)]['before-during_scaled'].values
        minus_values = values[values < 0]
        plus_values = values[values > 0]
        dict_datasets[typ + '_minus'] = minus_values
        dict_datasets[typ + '_plus'] = plus_values

    new_order = ['WT Low_minus', 'WT High_minus', 'MT Low_minus', 'MT High_minus', 'WT Low_plus', 'WT High_plus', 'MT Low_plus', 'MT High_plus']
    dict_datasets = change_dict_order(dict_datasets, new_order)
    [print(key, np.array(value).size) for key, value in dict_datasets.items()]
    try:
        colors = ('#888888', '#888888', '#CC6677', '#CC6677',) * 2
        draw_custom_bar_plot(dict_datasets, path + 'before_during_after_plus_minus_int_features_overlap_quality/',
                             file_name='%s_after-during' % feature,
                             strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, return_sig=True,
                             figsize=(2, 2))
    except:
        print('cannot generate graph')







############################### low interaction time before during after ################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure3. Interaction with Tfh of wt and mt GCB\\'
threshold = 0
min_duration = 5
#max_duration = 10
interaction_type = 'T-cell' #'FDC', 'T-cell'

features = ['instant_speed', 'instant_angle', 'Zone',
            'Sphericity', 'Volume', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core',
            'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC','diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'diff_Distance_to_DZ', 'diff_Distance_to_LZ', 'diff_Distance_to_FDC_core',
            'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell']


df_bda_motility = pd.DataFrame()
for feature in tqdm(features):
    #traj = df_partial[(df_partial['TrackID']==1000008597)&(df_partial['Video']=='2-Good-D9-B-M1L-ZT1-070-157-FOV230-320px_Statistics')].reset_index(drop=True)
    df_final = pd.DataFrame()
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        df_partial = df_duration[df_duration['Type'] == cell_type].reset_index(drop=True)
        befores = []
        durings = []
        afters = []
        labels = []
        overlap_volumes = []
        dLZ_distances = []
        contact_durations = []
        datas = []
        df_temporal = pd.DataFrame()
        count = 0
        for i in range(0, df_partial.shape[0]):
            if (i == 0) or (i == duration + i0):
                duration = df_partial['Time_span'][i]
                traj = df_partial[i: duration + i].reset_index(drop=True)
                i0 = i
                count = count + 1
                traj = traj[(traj['pseudo_frame'] != 0) & (traj['pseudo_frame'] != 1)].reset_index(drop=True)

                interaction_profile = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=%s' % interaction_type] > threshold
                interaction_profile = interaction_profile * 1  # Binary interaction profile

                elements = []
                indexes = []
                idx0 = 0
                for element, group in groupby(interaction_profile):  # Groups consistent values
                    # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
                    element_list = list(group)
                    idx1 = len(element_list) + idx0
                    idx_list = list(range(idx0, idx1))
                    idx0 = idx1
                    elements.append(element_list)
                    indexes.append(idx_list)

                check_minduration = np.array([np.sum(element) for element in elements]) >= min_duration
                # list of booleans whether each element have more than 20 persistent contact

                if any(check_minduration) == False:  # false -> No 20 persistent contact
                    continue

                elif np.sum(check_minduration)==1: # only one segment that is 20 persistent contact
                    n_groups = len(elements)
                    for group_idx, (index, element) in enumerate(zip(indexes, elements)):
                        if np.sum(element) >= min_duration:
                            if (group_idx != 0) and (group_idx != n_groups-1): # persistent segment should not be first nor last
                                #print(cell_type, interaction_type, traj['Exp'][0], traj['Video'][0], traj['TrackID'][0])
                                label = traj['Label'][0]
                                data = []
                                during= traj[feature][index].values
                                during_mean = np.median(during)
                                durings.append(during_mean)

                                before= traj[feature][indexes[group_idx - 1]].values
                                before_mean = np.median(before)
                                befores.append(before_mean)

                                after = traj[feature][indexes[group_idx + 1]].values
                                after_mean = np.median(after)
                                afters.append(after_mean)

                                overlap = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=%s' % interaction_type][index].values
                                overlap_mean = np.mean(overlap)

                                dLZ_distance = traj['Distance_to_FDC_core'][index].values
                                dLZ_distance_mean = np.mean(dLZ_distance)

                                contact_duration = np.sum((overlap > 0)*1)

                                data.append(before_mean)
                                data.append(during_mean)
                                data.append(after_mean)
                                labels.append(label)
                                datas.append(data)
                                overlap_volumes.append(overlap_mean)
                                contact_durations.append(contact_duration)
                                dLZ_distances.append(dLZ_distance_mean)
        # befores = flatten_list(befores)
        # durings = flatten_list(durings)
        # afters = flatten_list(afters)
        print(cell_type, count)
        for d_type in ['before', 'during', 'after']:
            if d_type == 'before':
                aaa = befores
            elif d_type == 'during':
                aaa = durings
            elif d_type == 'after':
                aaa = afters

            df_temp = pd.DataFrame()
            df_temp['value'] = aaa
            df_temp['data_type'] = d_type
            df_temp['Label'] = labels
            df_temp['Overlap_volume'] = overlap_volumes
            df_temp['dLZ_distance'] = dLZ_distances
            df_temp['Contact_duration'] = contact_durations
            df_temporal = pd.concat([df_temporal, df_temp], axis=0)

        df_temporal = df_temporal.reset_index(drop=True)
        df_temporal['Type'] = cell_type

        # df_temporal['before'] = befores
        # df_temporal['during'] = durings
        # df_temporal['after'] = afters
        # df_temporal['Type'] = cell_type

        df_final = pd.concat([df_final, df_temporal], axis=0)

    df_final = df_final.reset_index(drop=True)

    df_bda_motility[feature] = df_final['value']

df_bda_motility['data_type'] = df_final['data_type']
df_bda_motility['Type'] = df_final['Type']
df_bda_motility['Label'] = df_final['Label']
df_bda_motility['Overlap_volume'] = df_final['Overlap_volume']
df_bda_motility['Contact_duration'] = df_final['Contact_duration']
df_bda_motility['dLZ_distance'] = df_final['dLZ_distance']
df_bda_motility['type_label'] = df_final['Type'].astype(str) + '_' +df_final['Label'].astype(str)

df_bda_motility = df_bda_motility[df_bda_motility['Contact_duration']<15].reset_index(drop=True)

thresh = 0.07
df_bda_motility.loc[(df_bda_motility['Overlap_volume'] < thresh), 'Overlap_quality'] = 'Low'
df_bda_motility.loc[(df_bda_motility['Overlap_volume'] >= thresh), 'Overlap_quality'] = 'High'

for typ in ['wt_B-cell', 'mt_B-cell']:
    for zone in ['Low', 'High']:
        print(typ, zone, df_bda_motility[(df_bda_motility['Type']==typ)&(df_bda_motility['Overlap_quality']==zone)].shape[0]/3)

########
feature = 'Distance_to_FDC_core' # 'instant_angle', 'instant_speed',

df_ts = pd.DataFrame()
time_series = []
cell_types = []
labels = []
overlaps = []
for label in np.unique(df_bda_motility['type_label']):
    df_bda_motility_label = df_bda_motility[df_bda_motility['type_label'] == label]
    cell_type = df_bda_motility_label['Type'].values[0]

    before = df_bda_motility_label[df_bda_motility_label['data_type'] == 'before'][feature].values[0]
    during = df_bda_motility_label[df_bda_motility_label['data_type'] == 'during'][feature].values[0]
    after = df_bda_motility_label[df_bda_motility_label['data_type'] == 'after'][feature].values[0]

    overlap = df_bda_motility_label['Overlap_volume'].values[0]
    dLZ_distance = df_bda_motility_label['dLZ_distance'].values[0]

    ts = np.array([before, during, after])
    time_series.append(ts)
    cell_types.append(cell_type)
    labels.append(label)
    overlaps.append(overlap)
    df_ts_temp = pd.DataFrame()
    df_ts_temp['Type'] = [cell_type]
    df_ts_temp['Label'] = [label]
    df_ts_temp['before'] = [before]
    df_ts_temp['during'] = [during]
    df_ts_temp['after'] = [after]
    df_ts_temp['Overlap'] = [overlap]
    df_ts_temp['dLZ_distance'] = [dLZ_distance]
    df_ts = pd.concat([df_ts, df_ts_temp], axis=0)

thresh = 0.07
df_ts.loc[(df_ts['Overlap'] < thresh), 'Overlap_quality'] = 'Low'
df_ts.loc[(df_ts['Overlap'] >= thresh), 'Overlap_quality'] = 'High'

thresh = 20
df_ts.loc[(df_ts['dLZ_distance'] < thresh), 'dLZ'] = 'dLZ'
df_ts.loc[(df_ts['dLZ_distance'] >= thresh), 'dLZ'] = 'non-dLZ'

# if feature == 'Distance_to_FDC_core':
#     df_ts['DZ_transition'] = (df_ts['after'] - df_ts['during'] > 10).map({True: 'DZ transition', False: 'no transition'})

for typ in ['wt_B-cell', 'mt_B-cell']:
    for zone in ['Low', 'High']:
        print(typ, zone, df_ts[(df_ts['Type']==typ)&(df_ts['Overlap_quality']==zone)].shape[0])

for typ in ['wt_B-cell', 'mt_B-cell']:
    for zone in ['non-dLZ', 'dLZ']:
        print(typ, zone, df_ts[(df_ts['Type']==typ)&(df_ts['dLZ']==zone)].shape[0])

time_series = np.array(time_series)

time_series_dict = array_to_dict(time_series)
time_series_dict_scaled = normalize_timeseries(time_series_dict)
time_series_scaled = dict_to_array(time_series_dict_scaled)
from tslearn.clustering import TimeSeriesKMeans
tskm = TimeSeriesKMeans(n_clusters=4, metric='euclidean', random_state=0, verbose=True, max_iter=100)
tskmeans_predicted = tskm.fit_predict(time_series_scaled)

df_ts['tskmeans'] = tskmeans_predicted
df_ts, replace_map = order_cluster_by_feature(df_ts, cluster_name='tskmeans', feature_name='before')
tskm_cluster_centers_temp = tskm.cluster_centers_

tskm_cluster_centers= np.zeros_like(tskm_cluster_centers_temp)
# Apply the mapping to reorder rows
for old_idx, new_idx in replace_map.items():
    tskm_cluster_centers[new_idx] = tskm_cluster_centers_temp[old_idx]

df_ts['before_scaled'] = time_series_scaled[:, 0]
df_ts['during_scaled'] = time_series_scaled[:, 1]
df_ts['after_scaled'] = time_series_scaled[:, 2]

########## pre/post contact motility representatibe cluster ################

replace_map = {0:3, 1:2, 2:1, 3:0}
#replace_map = {0:2, 1:0, 2:1, 3:3}
df_ts = df_ts.replace({'tskmeans': replace_map})

tskm_cluster_centers_new= np.zeros_like(tskm_cluster_centers)
for old_idx, new_idx in replace_map.items():
    tskm_cluster_centers_new[new_idx] = tskm_cluster_centers[old_idx]


for cluster, tskm_cluster_center in enumerate(tskm_cluster_centers_new):
    ts = tskm_cluster_center.flatten()
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.lineplot(x=np.arange(0, 2 + 1, 1),
                 y=ts, lw=2.5,
                 dashes=False, markersize=8, err_style='bars',
                 color='#888888')
    if not os.path.isdir(
            path+'low interaction before_during_after_int_features_overlap_quality/' + 'tskmeans_cluster_%s/' % feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path+'low interaction before_during_after_int_features_overlap_quality/' + 'tskmeans_cluster_%s/' % feature)
    plt.savefig(path+'low interaction before_during_after_int_features_overlap_quality/' + 'tskmeans_cluster_%s/cluster_%s.png' % (feature, cluster), dpi=300, bbox_inches='tight')

    if not os.path.isdir(
            path+'low interaction before_during_after_int_features_overlap_quality/' + 'svg/tskmeans_cluster_%s/' % feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path+'low interaction before_during_after_int_features_overlap_quality/' + 'svg/tskmeans_cluster_%s/' % feature)
    plt.savefig(path+'low interaction before_during_after_int_features_overlap_quality/' + 'svg/tskmeans_cluster_%s/cluster_%s.svg' % (feature, cluster), dpi=300, bbox_inches='tight')

    plt.clf()
    plt.close()

########## pre/post contact motility mode distribution ################



df_ts_ = df_ts.copy()
df_ts_ = df_ts_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
draw_cluster_distribution_heatmap(df_ts_, path+'low interaction before_during_after_int_features_overlap_quality/', file_name='tskmeans_type_%s_heatmap' % feature,
                                  condition_name='Type', cluster_type='tskmeans',
                                  annot=True, col_cluster=False, row_cluster=False, figsize=(3.5, 2))

df_ts_['type_quality'] = df_ts_['Type'].astype(str) + ' ' + df_ts_['Overlap_quality']
draw_cluster_distribution_heatmap(df_ts_, path+'low interaction before_during_after_int_features_overlap_quality/', file_name='tskmeans_type_quality_%s_heatmap' % feature,
                                  condition_name='type_quality', cluster_type='tskmeans',
                                  annot=True, col_cluster=False, row_cluster=False, figsize=(3.5, 3))

df_ts_['type_dLZ'] = df_ts_['Type'].astype(str) + ' ' + df_ts_['dLZ']
draw_cluster_distribution_heatmap(df_ts_, path + 'low interaction before_during_after_int_features_overlap_quality/',
                                  file_name='tskmeans_type_dLZ_%s_heatmap' % feature,
                                  condition_name='type_dLZ', cluster_type='tskmeans',
                                  annot=True, col_cluster=False, row_cluster=False, figsize=(3.5, 3))

p_dict = permutation_test(df_ts_, group_name='Type', class_name='tskmeans', iteration=10000)
p_dicts[feature] = p_dict

for typ in ['wt_B-cell', 'mt_B-cell']:
    print(typ, df_ts[(df_ts['Type']==typ)].shape[0])







########## (before-during) of WT vs MT ################
for feature in features:
    print(feature)
    df_ts_ = df_ts.copy()
    df_ts_ = df_ts_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
    df_ts_['before-during'] = (df_ts_['%s_before'%feature] - df_ts_['%s_during'%feature])
    df_ts_['before-after'] = (df_ts_['%s_before'%feature] - df_ts_['%s_after'%feature])
    df_ts_['after-during'] = (df_ts_['%s_after'%feature] - df_ts_['%s_during'%feature])



    dict_datasets = {}
    for typ in ['WT', 'MT']:
        values = df_ts_[(df_ts_['Type'] == typ)]['before-during'].values
        # values = df_ts_[(df_ts_['Type'] == typ) & (df_ts_['tskmeans'] != 0)]['before-during_scaled'].values
        minus_values = values[values < 0]
        plus_values = values[values > 0]
        dict_datasets[typ + '_minus'] = minus_values
        dict_datasets[typ + '_plus'] = plus_values

    new_order = ['WT_minus', 'MT_minus', 'WT_plus', 'MT_plus']
    dict_datasets = change_dict_order(dict_datasets, new_order)
    [print(key, np.array(value).size) for key, value in dict_datasets.items()]
    try:
        colors = ('#888888', '#CC6677',) * 2
        draw_custom_bar_plot(dict_datasets, path+'before_during_after_plus_minus_int_features_overlap_quality/', file_name='%s_before-during' % feature,
                             strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, return_sig=True,
                             figsize=(1, 2))
    except:
        print('cannot generate graph')
    dict_datasets = {}
    for typ in ['WT', 'MT']:
        values = df_ts_[(df_ts_['Type'] == typ)]['after-during'].values
        # values = df_ts_[(df_ts_['Type'] == typ) & (df_ts_['tskmeans'] != 0)]['before-during_scaled'].values
        minus_values = values[values < 0]
        plus_values = values[values > 0]
        dict_datasets[typ + '_minus'] = minus_values
        dict_datasets[typ + '_plus'] = plus_values

    new_order = ['WT_minus', 'MT_minus', 'WT_plus', 'MT_plus']
    dict_datasets = change_dict_order(dict_datasets, new_order)
    [print(key, np.array(value).size) for key, value in dict_datasets.items()]
    try:
        colors = ('#888888', '#CC6677',) * 2
        draw_custom_bar_plot(dict_datasets, path+'before_during_after_plus_minus_int_features_overlap_quality/', file_name='%s_after-during' % feature,
                             strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, return_sig=True,
                             figsize=(1, 2))
    except:
        print('cannot generate graph')
############################### For T-cell, how motility changes when before, during and after interacting with GCB ################################

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df_duration = pd.read_parquet(path+'long_traj_duration_20.parquet')

# Remove Group A, IgG , mLT and CD40L
df_duration = df_duration[(df_duration['Exp']=='Exp1')|(df_duration['Exp']=='Exp2')|(df_duration['Exp']=='Exp3')
                              |(df_duration['Exp']=='Exp5')].reset_index(drop=True)

videos = np.unique(df_duration['Video'])
df_duration = df_duration[(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])&(df_duration['Video'] != videos[4])
                          &(df_duration['Video'] != videos[-1])].reset_index(drop=True)
videos = np.unique(df_duration['Video'])

df_duration = df_duration[df_duration['Type']=='T-cell'].reset_index(drop=True)
df_duration = get_instant_movements_variable_duration(df_duration, frame_name='Time_span', time_unit=0.5,
                                                      feature_name=['Position X', 'Position Y', 'Position Z'])

df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=FDC')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=T-cell')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_DZ')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_LZ')
df_duration = get_instant_direction_variable_duration(df_duration, frame_name='Time_span', thresh=0.1, feature_name='Distance_to_FDC_core')

#df_duration = df_duration[(df_duration['pseudo_frame']!=0)&(df_duration['pseudo_frame']!=1)].reset_index(drop=True)

from itertools import groupby

threshold = 0
min_duration = 12
test = 'mann-whitney' # mann-whitney, t-test

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure3. Interaction with Tfh of wt and mt GCB\\'
folder_name = 'T-cell motility before vs during vs after GCB interaction'
features = ['instant_speed', 'instant_angle', 'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'Shortest_Distance_to_Surfaces_Surfaces=wt_B-cell', 'Shortest_Distance_to_Surfaces_Surfaces=mt_B-cell',
            'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=wt_B-cell', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=mt_B-cell',
            'Zone', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC',
            'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell', 'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC',
            'diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell', 'diff_Distance_to_DZ', 'diff_Distance_to_LZ',
            'diff_Distance_to_FDC_core', 'Area', 'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Sphericity', 'Volume']

for feature in tqdm(features):
    #traj = df_partial[(df_partial['TrackID']==1000008597)&(df_partial['Video']=='2-Good-D9-B-M1L-ZT1-070-157-FOV230-320px_Statistics')].reset_index(drop=True)
    df_final = pd.DataFrame()
    for interaction_type in ['wt_B-cell', 'mt_B-cell']:
        befores = []
        durings = []
        afters = []
        labels = []
        datas = []
        df_temporal = pd.DataFrame()
        count = 0
        for i in range(0, df_duration.shape[0]):
            if (i == 0) or (i == duration + i0):
                duration = df_duration['Time_span'][i]
                traj = df_duration[i: duration + i].reset_index(drop=True)
                i0 = i
                count = count + 1
                traj = traj[(traj['pseudo_frame'] != 0) & (traj['pseudo_frame'] != 1)].reset_index(drop=True)

                interaction_profile = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=%s' % interaction_type] > threshold
                interaction_profile = interaction_profile * 1  # Binary interaction profile

                elements = []
                indexes = []
                idx0 = 0
                for element, group in groupby(interaction_profile):  # Groups consistent values
                    # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
                    element_list = list(group)
                    idx1 = len(element_list) + idx0
                    idx_list = list(range(idx0, idx1))
                    idx0 = idx1
                    elements.append(element_list)
                    indexes.append(idx_list)

                check_minduration = np.array([np.sum(element) for element in elements]) >= min_duration
                # list of booleans whether each element have more than 20 persistent contact

                if any(check_minduration) == False:  # false -> No 20 persistent contact
                    continue

                elif np.sum(check_minduration)==1: # only one segment that is 20 persistent contact
                    n_groups = len(elements)
                    for group_idx, (index, element) in enumerate(zip(indexes, elements)):
                        if np.sum(element) >= min_duration:
                            if (group_idx != 0) and (group_idx != n_groups-1): # persistent segment should not be first nor last
                                #print(cell_type, interaction_type, traj['Exp'][0], traj['Video'][0], traj['TrackID'][0])
                                label = traj['Label'][0]
                                data = []
                                during= traj[feature][index].values
                                during_mean = np.median(during)
                                durings.append(during_mean)

                                before= traj[feature][indexes[group_idx - 1]].values
                                before_mean = np.median(before)
                                befores.append(before_mean)

                                after = traj[feature][indexes[group_idx + 1]].values
                                after_mean = np.median(after)
                                afters.append(after_mean)

                                data.append(before_mean)
                                data.append(during_mean)
                                data.append(after_mean)
                                labels.append(label)
                                datas.append(data)
        # befores = flatten_list(befores)
        # durings = flatten_list(durings)
        # afters = flatten_list(afters)
        print(interaction_type, count)
        for d_type in ['before', 'during', 'after']:
            if d_type == 'before':
                aaa = befores
            elif d_type == 'during':
                aaa = durings
            elif d_type == 'after':
                aaa = afters

            df_temp = pd.DataFrame()
            df_temp['value'] = aaa
            df_temp['data_type'] = d_type

            df_temporal = pd.concat([df_temporal, df_temp], axis=0)

        df_temporal = df_temporal.reset_index(drop=True)
        df_temporal['Type'] = interaction_type

        # df_temporal['before'] = befores
        # df_temporal['during'] = durings
        # df_temporal['after'] = afters
        # df_temporal['Type'] = cell_type

        df_final = pd.concat([df_final, df_temporal], axis=0)

    df_final = df_final.reset_index(drop=True)


    condition_name = 'Type'
    estimator = 'mean'
    color_list = ('#CC6677', '#888888')
    marker_list=['o', '^', ]

    mean_dataset = {}
    error_dataset = {}
    p_value_data = {}
    for cell_type in np.unique(df_final[condition_name]):
        df_part = df_final[df_final[condition_name] == cell_type]
        means = []
        errors = []
        valuess = []

        for i in ['before', 'during', 'after']:
            # if i == custsom_range[1]:
            #     values = df_part[df_part[range_feature] >= i][feature_name].values
            # else:
            values = df_part[(df_part['data_type'] == i)]['value'].values

            if estimator == 'mean':
                mean = np.mean(values)
            elif estimator == 'median':
                mean = np.median(values)

            # means.append(np.mean(values))
            #means.append(np.mean(values))
            interval = stats.norm.interval(confidence=0.95, loc=mean, scale=stats.sem(values))
            error = mean - interval[0]

            means.append(mean)
            errors.append(error)
            valuess.append(values)

            mean_dataset[cell_type] = np.array(means)
            error_dataset[cell_type] = np.array(errors)
            p_value_data[cell_type] = valuess


    font = {'family': 'arial',
            'weight': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(3,3))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    for idx, key in enumerate(mean_dataset):
        sns.lineplot(data=mean_dataset, x=np.arange(0, 3, 1), y=mean_dataset[key],
                     label=key, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars', color=color_list[idx])
        # ax.errorbar(x=np.arange(0, 3, 1), y=mean_dataset[key], yerr=error_dataset[key],
        #             color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)
        ax.fill_between(x=np.arange(0, 3, 1), y1=mean_dataset[key]-error_dataset[key], y2=mean_dataset[key]+error_dataset[key],
                                     color=color_list[idx], alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['MT'], 0.1*std_dataset['MT'], color='#CC6677',
    #            capsize=3, capthick=1, elinewidth=1.5)
    # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['WT'], 0.1*std_dataset['WT'], color='#888888',
    #            capsize=3, capthick=1, elinewidth=1.5)
    from scipy import stats
    import statsmodels.api as sm

    p_values = []
    pairs = []
    cohen_ds = []
    for idx, (mt_values, wt_values) in enumerate(zip(p_value_data[list(p_value_data.keys())[0]], p_value_data[list(p_value_data.keys())[1]])):
        if test == 'mann-whitney':
            stat_test = stats.mannwhitneyu(mt_values, wt_values)
        elif test == 't-test':
            stat_test = stats.ttest_ind(mt_values, wt_values)
        elif test == 'wilcoxon-ranksum':
            stat_test = stats.ranksums(mt_values, wt_values)
        # print(idx, stat_test.pvalue)
        p_values.append(stat_test.pvalue)
        pairs.append(idx)

        cohen_d = Cohen_d(mt_values, wt_values)
        cohen_ds.append(cohen_d)
    #_, p_values, _, _ = sm.stats.multipletests(p_values, alpha=0.05,method='fdr_bh')  # Benjamini/Hochberg adjustment

    plt.title('p: %s, d: %s' % (p_values, cohen_ds), fontsize=4)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')
    if feature in ['instant_speed', 'instant_angle', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core',]:
        plt.ylim(0, None)
    else:
        plt.ylim(None, None)
    #ax.set_xlabel('%s'%x_label, fontsize=16, weight='normal', color='0.2')
    plt.xticks(np.arange(0, 3, 1), labels=['before', 'during', 'after'], fontsize=12, color='0.2', weight='normal', )
    plt.yticks(fontsize=16, color='0.2', weight='normal')

    # plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
    #            loc='best')
    ax.get_legend().remove()
    if not os.path.isdir(path + '%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s/' % folder_name)
    plt.savefig(path + '%s/%s.png' % (folder_name, feature), dpi=300, bbox_inches='tight')


    if not os.path.isdir(path + 'svg/%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/%s/' % folder_name)
    plt.savefig(path + 'svg/%s/%s.svg' % (folder_name, feature), bbox_inches='tight')



############################### T cell inst speed and angle before during after ################################

threshold = 0
min_duration = 12
cell_type = 'T-cell' #'FDC', 'T-cell'

features = ['instant_speed', 'instant_angle']

df_bda_motility = pd.DataFrame()
for feature in tqdm(features):
    #traj = df_partial[(df_partial['TrackID']==1000008597)&(df_partial['Video']=='2-Good-D9-B-M1L-ZT1-070-157-FOV230-320px_Statistics')].reset_index(drop=True)
    df_final = pd.DataFrame()
    for interaction_type in ['wt_B-cell', 'mt_B-cell']:
        df_partial = df_duration[df_duration['Type'] == cell_type].reset_index(drop=True)
        befores = []
        durings = []
        afters = []
        labels = []
        datas = []
        df_temporal = pd.DataFrame()
        count = 0
        for i in range(0, df_partial.shape[0]):
            if (i == 0) or (i == duration + i0):
                duration = df_partial['Time_span'][i]
                traj = df_partial[i: duration + i].reset_index(drop=True)
                i0 = i
                count = count + 1
                traj = traj[(traj['pseudo_frame'] != 0) & (traj['pseudo_frame'] != 1)].reset_index(drop=True)

                interaction_profile = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=%s' % interaction_type] > threshold
                interaction_profile = interaction_profile * 1  # Binary interaction profile

                elements = []
                indexes = []
                idx0 = 0
                for element, group in groupby(interaction_profile):  # Groups consistent values
                    # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
                    element_list = list(group)
                    idx1 = len(element_list) + idx0
                    idx_list = list(range(idx0, idx1))
                    idx0 = idx1
                    elements.append(element_list)
                    indexes.append(idx_list)

                check_minduration = np.array([np.sum(element) for element in elements]) >= min_duration
                # list of booleans whether each element have more than 20 persistent contact

                if any(check_minduration) == False:  # false -> No 20 persistent contact
                    continue

                elif np.sum(check_minduration)==1: # only one segment that is 20 persistent contact
                    n_groups = len(elements)
                    for group_idx, (index, element) in enumerate(zip(indexes, elements)):
                        if np.sum(element) >= min_duration:
                            if (group_idx != 0) and (group_idx != n_groups-1): # persistent segment should not be first nor last
                                #print(cell_type, interaction_type, traj['Exp'][0], traj['Video'][0], traj['TrackID'][0])
                                label = traj['Label'][0]
                                data = []
                                during= traj[feature][index].values
                                during_mean = np.mean(during)
                                durings.append(during_mean)

                                before= traj[feature][indexes[group_idx - 1]].values
                                before_mean = np.mean(before)
                                befores.append(before_mean)

                                after = traj[feature][indexes[group_idx + 1]].values
                                after_mean = np.mean(after)
                                afters.append(after_mean)

                                data.append(before_mean)
                                data.append(during_mean)
                                data.append(after_mean)
                                labels.append(label)
                                datas.append(data)
        # befores = flatten_list(befores)
        # durings = flatten_list(durings)
        # afters = flatten_list(afters)
        print(cell_type, count)
        for d_type in ['before', 'during', 'after']:
            if d_type == 'before':
                aaa = befores
            elif d_type == 'during':
                aaa = durings
            elif d_type == 'after':
                aaa = afters

            df_temp = pd.DataFrame()
            df_temp['value'] = aaa
            df_temp['data_type'] = d_type
            df_temp['Label'] = labels
            df_temporal = pd.concat([df_temporal, df_temp], axis=0)

        df_temporal = df_temporal.reset_index(drop=True)
        df_temporal['Interaction_type'] = interaction_type

        # df_temporal['before'] = befores
        # df_temporal['during'] = durings
        # df_temporal['after'] = afters
        # df_temporal['Type'] = cell_type

        df_final = pd.concat([df_final, df_temporal], axis=0)

    df_final = df_final.reset_index(drop=True)

    df_bda_motility[feature] = df_final['value']

df_bda_motility['data_type'] = df_final['data_type']
df_bda_motility['Interaction_type'] = df_final['Interaction_type']
df_bda_motility['Label'] = df_final['Label']
df_bda_motility['type_label'] = df_final['Interaction_type'].astype(str) + '_' +df_final['Label'].astype(str)


########## pre/post contact motility mode identification by tskmeans ################
feature = 'instant_speed'

df_ts = pd.DataFrame()
time_series = []
for label in np.unique(df_bda_motility['type_label']):
    df_bda_motility_label = df_bda_motility[df_bda_motility['type_label'] == label]
    cell_type = df_bda_motility_label['Interaction_type'].values[0]

    before = df_bda_motility_label[df_bda_motility_label['data_type'] == 'before'][feature].values[0]
    during = df_bda_motility_label[df_bda_motility_label['data_type'] == 'during'][feature].values[0]
    after = df_bda_motility_label[df_bda_motility_label['data_type'] == 'after'][feature].values[0]

    ts = np.array([before, during, after])
    time_series.append(ts)

    df_ts_temp = pd.DataFrame()
    df_ts_temp['Interaction_type'] = [cell_type]
    df_ts_temp['Label'] = [label]
    df_ts_temp['before'] = [before]
    df_ts_temp['during'] = [during]
    df_ts_temp['after'] = [after]
    df_ts = pd.concat([df_ts, df_ts_temp], axis=0)

time_series = np.array(time_series)

time_series_dict = array_to_dict(time_series)
time_series_dict_scaled = normalize_timeseries(time_series_dict)
time_series_scaled = dict_to_array(time_series_dict_scaled)
from tslearn.clustering import TimeSeriesKMeans
tskm = TimeSeriesKMeans(n_clusters=4, metric='euclidean', random_state=0, verbose=True, max_iter=100)
tskmeans_predicted = tskm.fit_predict(time_series_scaled)

df_ts['tskmeans'] = tskmeans_predicted
df_ts, replace_map = order_cluster_by_feature(df_ts, cluster_name='tskmeans', feature_name='before')
tskm_cluster_centers_temp = tskm.cluster_centers_

tskm_cluster_centers= np.zeros_like(tskm_cluster_centers_temp)
# Apply the mapping to reorder rows
for old_idx, new_idx in replace_map.items():
    tskm_cluster_centers[new_idx] = tskm_cluster_centers_temp[old_idx]

df_ts['before_scaled'] = time_series_scaled[:, 0]
df_ts['during_scaled'] = time_series_scaled[:, 1]
df_ts['after_scaled'] = time_series_scaled[:, 2]


########## pre/post contact motility mode distribution ################
df_ts_ = df_ts.copy()
df_ts_ = df_ts_.replace({'Interaction_type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
draw_cluster_distribution_heatmap(df_ts_, path+'Tfh motility/', file_name='tskmeans_type_%s_heatmap'%feature, condition_name='Interaction_type', cluster_type='tskmeans',
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(4,2))


p_dict = permutation_test(df_ts_, group_name='Interaction_type', class_name='tskmeans', iteration=10000)

########## pre/post contact motility representatibe cluster ################
for cluster, tskm_cluster_center in enumerate(tskm_cluster_centers):
    ts = tskm_cluster_center.flatten()
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.lineplot(x=np.arange(0, 2 + 1, 1),
                 y=ts, lw=2.5,
                 dashes=False, markersize=8, err_style='bars',
                 color='#888888')
    if not os.path.isdir(path+'Tfh motility/' + 'tskmeans_cluster_%s/'%feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path+'Tfh motility/' + 'tskmeans_cluster_%s/'%feature)
    plt.savefig(path+'Tfh motility/' + 'tskmeans_cluster_%s/cluster_%s.png' % (feature, cluster), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path+'Tfh motility/' + 'svg/tskmeans_cluster_%s/'%feature):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path+'Tfh motility/' + 'svg/tskmeans_cluster_%s/'%feature)
    plt.savefig(path+'Tfh motility/' + 'svg/tskmeans_cluster_%s/cluster_%s.svg' % (feature, cluster), dpi=300, bbox_inches='tight')

    plt.clf()
    plt.close()

########## pre/post contact motility mode actual time series per cluster ################
for cluster in np.unique(df_ts['tskmeans']):
    df_ts_part = df_ts[df_ts['tskmeans'] == cluster]
    for idx, each_cell in df_ts_part.iterrows():
        label = each_cell['Label']
        values = each_cell[['before', 'during', 'after']].values
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.lineplot(x=np.arange(0, 2 + 1, 1),
                     y=values, lw=2.5,
                     dashes=False, markersize=8, err_style='bars',
                     color='#888888')
        if not os.path.isdir(path+'Tfh motility/' + 'actual_ts_%s/cluster_%s/'%(feature, cluster)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path+'Tfh motility/' + 'actual_ts_%s/cluster_%s/'%(feature, cluster))

        plt.savefig(path+'Tfh motility/' + 'actual_ts_%s/cluster_%s/%s.png' % (feature, cluster, label), dpi=300, bbox_inches='tight')

        if not os.path.isdir(path+'Tfh motility/' + 'svg/actual_ts_%s/cluster_%s/'%(feature, cluster)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path+'Tfh motility/' + 'svg/actual_ts_%s/cluster_%s/'%(feature, cluster))

        plt.savefig(path+'Tfh motility/' + 'svg/actual_ts_%s/cluster_%s/%s.svg' % (feature, cluster, label), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()


########## scaled (before-during) of WT vs MT ################
df_ts_ = df_ts.copy()
df_ts_ = df_ts_.replace({'Interaction_type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

df_ts_['before-during_scaled'] = df_ts_['before_scaled'] - df_ts_['during_scaled']
df_ts_['after-during_scaled'] = df_ts_['after_scaled'] - df_ts_['during_scaled']


dict_datasets = {}
dict_datasets_all = {}
for typ in ['WT', 'MT']:
    values = df_ts_[(df_ts_['Interaction_type']==typ)]['before-during_scaled'].values
    #values = df_ts_[(df_ts_['Type'] == typ) & (df_ts_['tskmeans'] != 0)]['before-during_scaled'].values
    minus_values = values[values < 0]
    plus_values = values[values > 0]
    dict_datasets[typ+'_minus'] = minus_values
    dict_datasets[typ + '_plus'] = plus_values
    dict_datasets_all[typ] = values
new_order = ['WT_minus', 'MT_minus', 'WT_plus', 'MT_plus']
dict_datasets = change_dict_order(dict_datasets, new_order)
colors = ('#888888', '#CC6677',)*2
draw_custom_bar_plot(dict_datasets, path+'Tfh motility/', file_name='before-during_%s'%feature,
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, return_sig=False, figsize=(1, 2))

colors = ('#888888', '#CC6677',)
draw_custom_bar_plot(dict_datasets_all, path+'Tfh motility/', file_name='before-during_all_%s'%feature,
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, return_sig=False, figsize=(1, 2))


dict_datasets = {}
dict_datasets_all = {}
for typ in ['WT', 'MT']:
    values = df_ts_[(df_ts_['Interaction_type']==typ)]['after-during_scaled'].values
    #values = df_ts_[(df_ts_['Type'] == typ) & (df_ts_['tskmeans'] != 0)]['before-during_scaled'].values
    minus_values = values[values < 0]
    plus_values = values[values > 0]
    dict_datasets[typ+'_minus'] = minus_values
    dict_datasets[typ + '_plus'] = plus_values
    dict_datasets_all[typ] = values

new_order = ['WT_minus', 'MT_minus', 'WT_plus', 'MT_plus']
dict_datasets = change_dict_order(dict_datasets, new_order)
colors = ('#888888', '#CC6677',)*2
draw_custom_bar_plot(dict_datasets, path+'Tfh motility/', file_name='after-during_%s'%feature,
                     strip_plot=True, colors=colors, test='mann-whitney', pvalue=True, return_sig=False, figsize=(1, 2))
colors = ('#888888', '#CC6677',)
draw_custom_bar_plot(dict_datasets_all, path+'Tfh motility/', file_name='after-during_all_%s'%feature,
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, return_sig=False, figsize=(1, 2))






########## (before-during) of WT vs MT ################
df_ts_ = df_ts.copy()
df_ts_ = df_ts_.replace({'Interaction_type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

df_ts_['before-during'] = (df_ts_['before'] - df_ts_['during'])/df_ts_['before']
df_ts_['before-after'] = (df_ts_['before'] - df_ts_['after'])/df_ts_['before']
df_ts_['after-during'] = (df_ts_['after'] - df_ts_['during'])/df_ts_['before']


df_ts_['before-during'] = (df_ts_['before'] - df_ts_['during'])
df_ts_['before-after'] = (df_ts_['before'] - df_ts_['after'])
df_ts_['after-during'] = (df_ts_['after'] - df_ts_['during'])

df_ts_['before-during'] = (df_ts_['before'] - df_ts_['during'])/np.mean(df_ts_[['before', 'during', 'after']], axis=1)
df_ts_['before-after'] = (df_ts_['before'] - df_ts_['after'])/np.mean(df_ts_[['before', 'during', 'after']], axis=1)
df_ts_['after-during'] = (df_ts_['after'] - df_ts_['during'])/np.mean(df_ts_[['before', 'during', 'after']], axis=1)



aa = df_ts_[df_ts_['Interaction_type'] =='WT'].reset_index(drop=True)
dict_datasets = {}
for typ in ['WT', 'MT']:
    #values = df_ts_[(df_ts_['Type']==typ)&(df_ts_['tskmeans']!=0)]['before-during'].values
    values = df_ts_[(df_ts_['Interaction_type'] == typ)]['before-during'].values
    #print(typ, np.mean(values))
    #values = values[values < 0]
    dict_datasets[typ] = values

colors = ('#888888', '#CC6677',)
draw_custom_bar_plot(dict_datasets, path+'Tfh motility/', file_name='before-during_%s'%feature,
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))

dict_datasets = {}
for typ in ['WT', 'MT']:
    values = df_ts_[df_ts_['Interaction_type']==typ]['before-after'].values
    #values = values[values < 0]
    dict_datasets[typ] = values

colors = ('#888888', '#CC6677',)
draw_custom_bar_plot(dict_datasets, path+'Tfh motility/', file_name='before-after_%s'%feature,
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))


dict_datasets = {}
for typ in ['WT', 'MT']:
    values = df_ts_[df_ts_['Interaction_type']==typ]['after-during'].values
    #values = values[values < 0]
    dict_datasets[typ] = values

colors = ('#888888', '#CC6677',)
draw_custom_bar_plot(dict_datasets, path+'Tfh motility/', file_name='after-during_%s'%feature,
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1, 2))













file_name = 'T cell instant speed vs instant angle'
fig, ax = plt.subplots(figsize=(4,4))
# sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
#                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
#for idx, key in enumerate(dataset):
sns.scatterplot(data=df_bda_motility, x='instant_speed', y='instant_angle', hue='Interaction_type', lw=2.5,  s=20, hue_order=['wt_B-cell', 'mt_B-cell'],
                         palette=('#888888', '#CC6677'), style='data_type')
handles, labels = ax.get_legend_handles_labels()

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_xlabel('Instant speed (μm/min)', fontsize=16, weight='normal', color='0.2')
ax.set_ylabel('Instant turning angle (rad/min)', fontsize=16, weight='normal', color='0.2')

#custom_range = (70, 95)
# ax.set_xticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='normal')
# ax.set_yticklabels(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize),
#                    rotation=0, rotation_mode='anchor', ha='right', fontsize=12, weight='normal')

# plt.xticks(np.arange(custom_range[0], custom_range[1] + label_stepsize, label_stepsize), fontsize=12, color='0.2',
#            weight='normal')
plt.xticks(fontsize=12, color='0.2', weight='normal')
plt.yticks(fontsize=12, color='0.2', weight='normal')
#plt.xlim(0, 13)
plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
           loc='best')
plt.savefig(path + '%s.png' % (file_name), dpi=300,bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/' )
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
plt.clf()
plt.close()

############################### For same cell, zonal before vs during vs after ################################

from itertools import groupby

threshold = 0
min_duration = 15
interaction_type = 'T-cell' #'FDC', 'T-cell'
test = 'mann-whitney'

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure2-3. Transition dynamics of wt and mt GCB\\'
folder_name = 'zonal %s before vs during vs after'%interaction_type
features = ['instant_speed', 'instant_angle', 'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell',
            'Zone', 'Distance_to_DZ', 'Distance_to_LZ', 'Distance_to_FDC_core', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC',
            'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell', 'diff_Shortest_Distance_to_Surfaces_Surfaces=FDC',
            'diff_Shortest_Distance_to_Surfaces_Surfaces=T-cell', 'diff_Distance_to_DZ', 'diff_Distance_to_LZ',
            'diff_Distance_to_FDC_core', 'Area', 'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Sphericity', 'Volume']

for feature in tqdm(features):
    #traj = df_partial[(df_partial['TrackID']==1000008597)&(df_partial['Video']=='2-Good-D9-B-M1L-ZT1-070-157-FOV230-320px_Statistics')].reset_index(drop=True)
    df_final = pd.DataFrame()
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        df_partial = df_duration[df_duration['Type'] == cell_type].reset_index(drop=True)
        befores = []
        durings = []
        afters = []
        zones = []

        datas = []
        df_temporal = pd.DataFrame()

        for i in range(0, df_partial.shape[0]):
            if (i == 0) or (i == duration + i0):
                duration = df_partial['Time_span'][i]
                traj = df_partial[i: duration + i].reset_index(drop=True)
                i0 = i

                traj = traj[(traj['pseudo_frame'] != 0) & (traj['pseudo_frame'] != 1)].reset_index(drop=True)

                interaction_profile = traj['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=%s' % interaction_type] > threshold
                interaction_profile = interaction_profile * 1  # Binary interaction profile

                elements = []
                indexes = []
                idx0 = 0
                for element, group in groupby(interaction_profile):  # Groups consistent values
                    # Ex) [1,1,1,0,1,1] -> element1: [1,1,1], element2: [0], element3: [1,1]
                    element_list = list(group)
                    idx1 = len(element_list) + idx0
                    idx_list = list(range(idx0, idx1))
                    idx0 = idx1
                    elements.append(element_list)
                    indexes.append(idx_list)

                check_minduration = np.array([np.sum(element) for element in elements]) >= min_duration
                # list of booleans whether each element have more than 20 persistent contact

                if any(check_minduration) == False:  # false -> No 20 persistent contact
                    continue

                elif np.sum(check_minduration)==1: # only one segment that is 20 persistent contact
                    n_groups = len(elements)
                    for group_idx, (index, element) in enumerate(zip(indexes, elements)):
                        if np.sum(element) >= min_duration:
                            if (group_idx != 0) and (group_idx != n_groups-1): # persistent segment should not be first nor last
                                #print(cell_type, interaction_type, traj['Exp'][0], traj['Video'][0], traj['TrackID'][0])
                                data = []

                                zone_labels = traj['Zone'][index].values
                                zone_label = np.mean(zone_labels)
                                zones.append(zone_label)

                                during= traj[feature][index].values
                                during_mean = np.mean(during)
                                durings.append(during_mean)

                                before= traj[feature][indexes[group_idx - 1]].values
                                before_mean = np.mean(before)
                                befores.append(before_mean)

                                after = traj[feature][indexes[group_idx + 1]].values
                                after_mean = np.mean(after)
                                afters.append(after_mean)

                                data.append(before_mean)
                                data.append(during_mean)
                                data.append(after_mean)
                                datas.append(data)
        # befores = flatten_list(befores)
        # durings = flatten_list(durings)
        # afters = flatten_list(afters)

        for d_type in ['before', 'during', 'after']:
            if d_type == 'before':
                aaa = befores
            elif d_type == 'during':
                aaa = durings
            elif d_type == 'after':
                aaa = afters

            df_temp = pd.DataFrame()
            df_temp['value'] = aaa
            df_temp['data_type'] = d_type
            df_temp['zone'] = zones

            df_temporal = pd.concat([df_temporal, df_temp], axis=0)

        df_temporal = df_temporal.reset_index(drop=True)
        df_temporal['Type'] = cell_type

        # df_temporal['before'] = befores
        # df_temporal['during'] = durings
        # df_temporal['after'] = afters
        # df_temporal['Type'] = cell_type

        df_final = pd.concat([df_final, df_temporal], axis=0)

    df_final = df_final.reset_index(drop=True)

    df_final.loc[(df_final['zone'] < 0.4) & (df_final['zone'] >= 0), 'Zone'] = 'DZ'
    df_final.loc[(df_final['zone'] < 0.8) & (df_final['zone'] >= 0.4), 'Zone'] = 'DZ-sLZ'
    df_final.loc[(df_final['zone'] < 1.2) & (df_final['zone'] >= 0.8), 'Zone'] = 'sLZ'
    df_final.loc[(df_final['zone'] < 1.6) & (df_final['zone'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
    df_final.loc[(df_final['zone'] <= 2) & (df_final['zone'] >= 1.6), 'Zone'] = 'dLZ'

    condition_name = 'Type'
    estimator = 'mean'
    color_list = ('#CC6677', '#CC6677', '#CC6677', '#888888', '#888888', '#888888')
    marker_list=['o', '^', '*', 'o', '^', '*',]

    mean_dataset = {}
    std_dataset = {}
    p_value_data = {}
    for cell_type in np.unique(df_final[condition_name]):
        #df_part = df_final[df_final[condition_name] == cell_type]
        for group in ['DZ','sLZ', 'dLZ']:
            df_part = df_final[(df_final[condition_name] == cell_type)&(df_final['Zone'] == group)].reset_index(drop=True)

            means = []
            stds = []
            valuess = []

            for i in ['before', 'during', 'after']:
                # if i == custsom_range[1]:
                #     values = df_part[df_part[range_feature] >= i][feature_name].values
                # else:
                values = df_part[(df_part['data_type'] == i)]['value'].values

                if estimator == 'mean':
                    means.append(np.mean(values))
                elif estimator == 'median':
                    means.append(np.median(values))
                # means.append(np.mean(values))
                #means.append(np.mean(values))
                stds.append(np.std(values))
                valuess.append(values)

            mean_dataset[cell_type + '_' + str(group)] = np.array(means)
            std_dataset[cell_type + '_' + str(group)] = np.array(stds)
            p_value_data[cell_type + '_' + str(group)] = valuess



    font = {'family': 'arial',
            'weight': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=(3,3))
    # sns.lineplot(data=mean_dataset, x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+stepsize, stepsize), label=np.unique(df_ctrl['Type']),
    #                   lw=2.5, markers=['o', '^', '.'], dashes=False, markersize=8, err_style='bars', palette=color_list)
    for idx, key in enumerate(mean_dataset):
        sns.lineplot(data=mean_dataset, x=np.arange(0, 3, 1), y=mean_dataset[key],
                     label=key, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars', color=color_list[idx])
        ax.errorbar(np.arange(0, 3, 1), mean_dataset[key], 0.1 * std_dataset[key],
                    color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

    handles, labels = ax.get_legend_handles_labels()
    # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['MT'], 0.1*std_dataset['MT'], color='#CC6677',
    #            capsize=3, capthick=1, elinewidth=1.5)
    # ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['WT'], 0.1*std_dataset['WT'], color='#888888',
    #            capsize=3, capthick=1, elinewidth=1.5)
    from scipy import stats
    import statsmodels.api as sm

    p_values = []
    pairs = []
    cohen_ds = []
    for idx, (mt_values, wt_values) in enumerate(zip(p_value_data[list(p_value_data.keys())[0]], p_value_data[list(p_value_data.keys())[1]])):
        if test == 'mann-whitney':
            stat_test = stats.mannwhitneyu(mt_values, wt_values)
        elif test == 't-test':
            stat_test = stats.ttest_ind(mt_values, wt_values)
        elif test == 'wilcoxon-ranksum':
            stat_test = stats.ranksums(mt_values, wt_values)
        # print(idx, stat_test.pvalue)
        p_values.append(stat_test.pvalue)
        pairs.append(idx)

        cohen_d = Cohen_d(mt_values, wt_values)
        cohen_ds.append(cohen_d)
    #_, p_values, _, _ = sm.stats.multipletests(p_values, alpha=0.05,method='fdr_bh')  # Benjamini/Hochberg adjustment

    plt.title('p: %s, d: %s' % (p_values, cohen_ds), fontsize=4)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')

    #ax.set_xlabel('%s'%x_label, fontsize=16, weight='normal', color='0.2')
    plt.xticks(np.arange(0, 3, 1), labels=['before', 'during', 'after'], fontsize=12, color='0.2', weight='normal', )
    plt.yticks(fontsize=16, color='0.2', weight='normal')

    # plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
    #            loc='best')
    ax.get_legend().remove()

    if not os.path.isdir(path + '%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s/' % folder_name)
    plt.savefig(path + '%s/%s.png' % (folder_name, feature), dpi=300, bbox_inches='tight')


    if not os.path.isdir(path + 'svg/%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/%s/' % folder_name)
    plt.savefig(path + 'svg/%s/%s.svg' % (folder_name, feature), bbox_inches='tight')





