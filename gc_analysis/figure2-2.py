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
"""Generates Data for Figure2-2. FDC-dependent characterization of WT and MT GCB motility """
import scipy.stats
from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import ZoneSignal

#################################### Pull out interaction features ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GCB_no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_no_inhibit_traj_duration_20.parquet')


_, _, Zone_series = to_timeseries_fast(df_duration, duration=20, feature_name='Zone')

feature_list = ['avg_zone', 'dz_resident_times', 'dz_resident_persistences', 'slz_resident_times',
                'slz_resident_persistences', 'dlz_resident_times', 'dlz_resident_persistences']
Zone_func = ZoneSignal(Zone_series)
df_zone = Zone_func.extract_features(feature_list)

df = pd.concat([df, df_zone], axis=1)


df.loc[(df['avg_zone'] < 0.4) & (df['avg_zone'] >= 0), 'Zone'] = 'DZ'
df.loc[(df['avg_zone'] < 0.8) & (df['avg_zone'] >= 0.4), 'Zone'] = 'DZ-sLZ'
df.loc[(df['avg_zone'] < 1.2) & (df['avg_zone'] >= 0.8), 'Zone'] = 'sLZ'
df.loc[(df['avg_zone'] < 1.6) & (df['avg_zone'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
df.loc[(df['avg_zone'] <= 2) & (df['avg_zone'] >= 1.6), 'Zone'] = 'dLZ'

print(df[df['Zone']=='DZ'].shape[0], df[df['Zone']=='DZ-sLZ'].shape[0], df[df['Zone']=='sLZ'].shape[0],
      df[df['Zone']=='sLZ-dLZ'].shape[0], df[df['Zone']=='dLZ'].shape[0])

duration=20
label_expanded = np.repeat(df['Zone'], duration).reset_index(drop=True)
df_duration['Zone_label'] = label_expanded

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure2-2. Interaction with FDC of wt and mt GCB\\'

#################################### draw 3D trajectories by kmeans ####################################

for zone in np.unique(df_duration['Zone_label']):
    df_duration_part = df_duration[df_duration['Zone_label']==zone].reset_index(drop=True)
    draw_3D_trajectory_one_figure(df_duration_part, path, folder_name='zone trajectories/%s trajectory'%zone, duration=20, n_examples=20,
                                  label_name='Type', feature_name=['Position X', 'Position Y', 'Position Z'], lim=100)

####################################### Quantify FDC interaction frequency #############################################
int_time = 20

persistent_int_freqs_datasets = {}
persistent_int_freq_per_cellnumbers_datasets = {}
total_n_contacts_per_cellnumbers_datasets = {}
persistent_int_freq_per_cellcontacts_datasets = {}
low_contact_freq_per_cellnumbers_datasets = {}
overlap_datasets1 = {}
overlap_datasets2 = {}

videos = np.unique(df['Video'])
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    df_part = df[df['Type'] == cell_type].reset_index(drop=True)
    persistent_int_freqs = []
    persistent_int_freq_per_cellnumbers = []
    total_n_contacts_per_cellnumbers = []
    persistent_int_freq_per_cellcontacts = []
    low_contact_freq_per_cellnumbers = []
    overlap1 = []
    overlap2 = []
    for video in videos:
        #if 'A' in video and cell_type == 'mt_B-cell':
        if '-A' in video:
            continue
        df_video = df_part[df_part['Video'] == video].reset_index(drop=True)
        if df_video.shape[0] == 0:
            continue
        data = df_video['FDC_contact_times']
        overall_overlap_data = df_video['FDC_avg_overlap']
        overlap_data_PC = df_video[df_video['FDC_contact_times']>=int_time].reset_index(drop=True)['FDC_avg_overlap']
        mask = ~np.isnan(data)
        data = data[mask]
        persistent_int_freq = sum(data >= int_time)

        total_n_contact = sum(data)

        persistent_int_freq_per_cellnumber = persistent_int_freq / df_video.shape[0]
        total_n_contacts_per_cellnumber = total_n_contact / df_video.shape[0]
        persistent_int_freq_per_cellcontact = persistent_int_freq / sum(data)
        low_contact_freq_per_cellnumber = sum((1<=data) & (data<=3)) / df_video.shape[0]

        overlap_per_cellnumber = overall_overlap_data / df_video.shape[0]
        overlap_PC_per_cellnumber = overlap_data_PC / df_video.shape[0]

        persistent_int_freqs.append(persistent_int_freq)
        persistent_int_freq_per_cellnumbers.append(persistent_int_freq_per_cellnumber)
        total_n_contacts_per_cellnumbers.append(total_n_contacts_per_cellnumber)
        persistent_int_freq_per_cellcontacts.append(persistent_int_freq_per_cellcontact)
        low_contact_freq_per_cellnumbers.append(low_contact_freq_per_cellnumber)

        overlap1.append(overlap_per_cellnumber)
        overlap2.append(overlap_PC_per_cellnumber)

    persistent_int_freqs_datasets[cell_type] = persistent_int_freqs
    persistent_int_freq_per_cellnumbers_datasets[cell_type] = persistent_int_freq_per_cellnumbers
    total_n_contacts_per_cellnumbers_datasets[cell_type] = total_n_contacts_per_cellnumbers
    persistent_int_freq_per_cellcontacts_datasets[cell_type] = persistent_int_freq_per_cellcontacts
    low_contact_freq_per_cellnumbers_datasets[cell_type] = low_contact_freq_per_cellnumbers

    overlap_datasets1[cell_type] = overlap1
    overlap_datasets2[cell_type] = overlap2

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

overlap_datasets1 = change_dict_order(overlap_datasets1, new_order)
overlap_datasets1 = {replace_keys.get(k, k):v  for (k,v) in overlap_datasets1.items() }

overlap_datasets2 = change_dict_order(overlap_datasets2, new_order)
overlap_datasets2 = {replace_keys.get(k, k):v  for (k,v) in overlap_datasets2.items() }

colors=('#888888', '#CC6677')
draw_custom_bar_plot(persistent_int_freqs_datasets, path, file_name='Total FDC interaction frequency',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellnumbers_datasets, path, file_name='FDC persistent interaction frequency per cell number',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(total_n_contacts_per_cellnumbers_datasets, path, file_name='number of FDC contacts per cell number',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellcontacts_datasets, path, file_name='FDC persistent interaction frequency per number of contacts',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(low_contact_freq_per_cellnumbers_datasets, path, file_name='FDC low contact time frequency per cell number',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))

draw_custom_bar_plot(overlap_datasets1, path, file_name='FDC overlap per cell number',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(overlap_datasets2, path, file_name='FDC loverlap PC per cell number',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))

####################################### dLZ Quantify FDC interaction frequency within dLZ #############################################
int_time = 20

persistent_int_freqs_datasets = {}
persistent_int_freq_per_cellnumbers_datasets = {}
total_n_contacts_per_cellnumbers_datasets = {}
persistent_int_freq_per_cellcontacts_datasets = {}
low_contact_freq_per_cellnumbers_datasets = {}

videos = np.unique(df['Video'])
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    df_part = df[df['Type'] == cell_type].reset_index(drop=True)
    persistent_int_freqs = []
    persistent_int_freq_per_cellnumbers = []
    total_n_contacts_per_cellnumbers = []
    persistent_int_freq_per_cellcontacts = []
    low_contact_freq_per_cellnumbers = []
    for video in videos:
        # if 'A' in video and cell_type == 'mt_B-cell':
        if '-A' in video:
            continue
        df_video = df_part[(df_part['Video'] == video)&(df_part['Zone'] == 'dLZ')].reset_index(drop=True)
        if df_video.shape[0] == 0:
            continue
        data = df_video['FDC_contact_times'].reset_index(drop=True)

        mask = ~np.isnan(data)
        data = data[mask]
        persistent_int_freq = sum(data >= int_time)

        total_n_contact = sum(data)

        persistent_int_freq_per_cellnumber = persistent_int_freq / df_video.shape[0]
        total_n_contacts_per_cellnumber = total_n_contact / df_video.shape[0]
        persistent_int_freq_per_cellcontact = persistent_int_freq / sum(data)
        low_contact_freq_per_cellnumber = sum((1 <= data) & (data <= 3)) / df_video.shape[0]

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

replace_keys = {'T-cell': 'Tfh', 'mt_B-cell': 'mt GCB', 'wt_B-cell': 'wt GCB'}
new_order = ['wt_B-cell', 'mt_B-cell']
persistent_int_freqs_datasets = change_dict_order(persistent_int_freqs_datasets, new_order)
persistent_int_freqs_datasets = {replace_keys.get(k, k): v for (k, v) in persistent_int_freqs_datasets.items()}

persistent_int_freq_per_cellnumbers_datasets = change_dict_order(persistent_int_freq_per_cellnumbers_datasets,
                                                                 new_order)
persistent_int_freq_per_cellnumbers_datasets = {replace_keys.get(k, k): v for (k, v) in
                                                persistent_int_freq_per_cellnumbers_datasets.items()}

total_n_contacts_per_cellnumbers_datasets = change_dict_order(total_n_contacts_per_cellnumbers_datasets, new_order)
total_n_contacts_per_cellnumbers_datasets = {replace_keys.get(k, k): v for (k, v) in
                                             total_n_contacts_per_cellnumbers_datasets.items()}

persistent_int_freq_per_cellcontacts_datasets = change_dict_order(persistent_int_freq_per_cellcontacts_datasets,
                                                                  new_order)
persistent_int_freq_per_cellcontacts_datasets = {replace_keys.get(k, k): v for (k, v) in
                                                 persistent_int_freq_per_cellcontacts_datasets.items()}

low_contact_freq_per_cellnumbers_datasets = change_dict_order(low_contact_freq_per_cellnumbers_datasets, new_order)
low_contact_freq_per_cellnumbers_datasets = {replace_keys.get(k, k): v for (k, v) in
                                             low_contact_freq_per_cellnumbers_datasets.items()}

colors = ('#888888', '#CC6677')
draw_custom_bar_plot(persistent_int_freqs_datasets, path, file_name='dLZ Total FDC interaction frequency',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1, 2))
draw_custom_bar_plot(persistent_int_freq_per_cellnumbers_datasets, path,
                     file_name='dLZ FDC persistent interaction frequency per cell number',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1, 2))
draw_custom_bar_plot(total_n_contacts_per_cellnumbers_datasets, path,
                     file_name='dLZ number of FDC contacts per cell number',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1, 2))
draw_custom_bar_plot(persistent_int_freq_per_cellcontacts_datasets, path,
                     file_name='dLZ FDC persistent interaction frequency per number of contacts',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1, 2))
draw_custom_bar_plot(low_contact_freq_per_cellnumbers_datasets, path,
                     file_name='dLZ FDC low contact time frequency per cell number',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1, 2))


####################################### Normalized cell count for each zones #############################################
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})

df_.loc[(df_['Zone'] == 'DZ'), 'Zone1'] = 'DZ'
df_.loc[(df_['Zone'] == 'DZ-sLZ'), 'Zone1'] = 'DZ'
df_.loc[(df_['Zone'] == 'sLZ'), 'Zone1'] = 'sLZ'
df_.loc[(df_['Zone'] == 'sLZ-dLZ'), 'Zone1'] = 'dLZ'
df_.loc[(df_['Zone'] == 'dLZ'), 'Zone1'] = 'dLZ'

normalized_count_dataset = {'wt GCB DZ': [], 'mt GCB DZ': [], 'wt GCB sLZ': [], 'mt GCB sLZ': [], 'wt GCB dLZ': [], 'mt GCB dLZ': []}
count_dataset = {'wt GCB DZ': [], 'mt GCB DZ': [], 'wt GCB sLZ': [], 'mt GCB sLZ': [], 'wt GCB dLZ': [], 'mt GCB dLZ': []}
partial_normalized_count_dataset = {'wt GCB DZ': [], 'mt GCB DZ': [], 'wt GCB dLZ': [], 'mt GCB dLZ': []}

videos = np.unique(df_['Video'])
for cell_type in ['wt GCB', 'mt GCB']:
    df_part = df_[df_['Type']==cell_type].reset_index(drop=True)
    persistent_int_freqs = []
    for video in videos:
        # if 'A' in video and cell_type == 'mt_B-cell':
        if '-A' in video:
            continue
        df_video = df_part[(df_part['Video'] == video)].reset_index(drop=True)
        for zone in ['DZ', 'sLZ', 'dLZ']:
            df_part_video = df_video[(df_video['Zone'] == zone)].reset_index(drop=True)
            count = df_part_video.shape[0]
            total = df_video.shape[0]
            dz_count = df_video[(df_video['Zone'] == 'DZ')].shape[0]
            dlz_count = df_video[(df_video['Zone'] == 'dLZ')].shape[0]

            normalized_count = count / total
            partial_normalized_count = count / (dz_count + dlz_count)

            normalized_count_dataset[cell_type+' '+zone].append(normalized_count)
            count_dataset[cell_type + ' ' + zone].append(count)
            if zone != 'sLZ':
                partial_normalized_count_dataset[cell_type + ' ' + zone].append(partial_normalized_count)

colors = ('#888888', '#CC6677')*3
draw_custom_bar_plot(normalized_count_dataset, path, file_name='normalized count',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(2, 4))

colors = ('#888888', '#CC6677')*3
draw_custom_bar_plot(count_dataset, path, file_name='count',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(2, 4))

colors = ('#888888', '#CC6677')*3
draw_custom_bar_plot(partial_normalized_count_dataset, path, file_name='partial normalized count',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(2, 4))


####################################### Normalized cell count using other metirc #############################################
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})

df_.loc[(df_['Core_distance_average'] <= 80), 'dLZ_localization'] = 'dLZ'
df_.loc[(df_['Core_distance_average'] > 80), 'dLZ_localization'] = 'non-dLZ'

df_.loc[(df_['DZ_distance_average'] <= 80), 'DZ_localization'] = 'DZ'
df_.loc[(df_['DZ_distance_average'] > 80), 'DZ_localization'] = 'non-DZ'


normalized_count_dataset1 = {'wt GCB non dLZ': [], 'mt GCB non dLZ': [],  'wt GCB dLZ': [], 'mt GCB dLZ': []}
normalized_count_dataset2 = {'wt GCB DZ': [], 'mt GCB DZ': [],  'wt GCB non DZ': [], 'mt GCB non DZ': []}


videos = np.unique(df_['Video'])
for cell_type in ['wt GCB', 'mt GCB']:
    df_part = df_[df_['Type']==cell_type].reset_index(drop=True)
    persistent_int_freqs = []
    for video in videos:
        # if 'A' in video and cell_type == 'mt_B-cell':
        if '-A' in video:
            continue
        df_video = df_part[(df_part['Video'] == video)].reset_index(drop=True)

        total = df_video.shape[0]
        non_dlz_count = df_video[(df_video['dLZ_localization'] == 'non-dLZ')].shape[0]
        dlz_count = df_video[(df_video['dLZ_localization'] == 'dLZ')].shape[0]
        dz_count = df_video[(df_video['DZ_localization'] == 'DZ')].shape[0]
        non_dz_count = df_video[(df_video['DZ_localization'] == 'non-DZ')].shape[0]

        norm_non_dlz_count = non_dlz_count / total
        norm_dlz_count = dlz_count / total
        norm_dz_count = dz_count / total
        norm_non_dz_count = non_dz_count / total

        normalized_count_dataset1[cell_type+' '+'non dLZ'].append(norm_non_dlz_count)
        normalized_count_dataset1[cell_type + ' ' + 'dLZ'].append(norm_dlz_count)
        normalized_count_dataset2[cell_type + ' ' + 'DZ'].append(norm_dz_count)
        normalized_count_dataset2[cell_type + ' ' + 'non DZ'].append(norm_non_dz_count)

colors = ('#888888', '#CC6677')*2
draw_custom_bar_plot(normalized_count_dataset1, path, file_name='normalized dLZ localization count',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(2, 4))

colors = ('#888888', '#CC6677')*2
draw_custom_bar_plot(normalized_count_dataset2, path, file_name='normalized DZ localization count',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(2, 4))


####################################### FDC interaction kde mt GCB vs wt GCB #############################################
replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}

dataset={}
for condition in np.unique(df['Type']):
    data = df[df['Type'] == condition]['FDC_contact_times']
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

ax.set_xlabel('FDC interaction frequency', fontsize=8, weight='bold', color='0.2')
ax.set_ylabel('Density', fontsize=8, weight='bold', color='0.2')
plt.xticks(fontsize=8, color='0.2', weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')

plt.legend(frameon=False, prop = {'weight':'bold', 'size':8}, labelcolor='0.2')

plt.savefig(path+'cell-FDC interaction frequency.png', dpi=300, bbox_inches='tight')
plt.close()
plt.clf()

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/cell-FDC interaction frequency.svg', bbox_inches='tight')
plt.close()
plt.clf()

####################################### dLZ FDC interaction kde mt GCB vs wt GCB #############################################
replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}

dataset={}
for condition in np.unique(df['Type']):
    data = df[(df['Type'] == condition)&(df['Zone'] == 'dLZ')]['FDC_contact_persistences'] # 'FDC_contact_times', 'FDC_contact_persistences'
    dataset[condition] = np.array(data)
new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dataset, new_order)
dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

draw_custom_bar_plot(dict_datasets, path, file_name='dLZ cell-FDC interaction violinplot', colors = ('#888888', '#CC6677'), strip_plot=False,
                            test='mann-whitney', pvalue=True, figsize=(1,2))

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
    ax = sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=(15, 25), color=colors[i], label=key)

# ax = sns.kdeplot(data=dict_datasets['wt GCB'], fill=True, linewidth=1, clip=(0, 40), color='#888888', label='wt GCB')
# ax = sns.kdeplot(data=dict_datasets['mt GCB'], fill=True, linewidth=1, clip=(0, 40), color='#CC6677', label='mt GCB')

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('FDC interaction frequency', fontsize=8, weight='bold', color='0.2')
ax.set_ylabel('Density', fontsize=8, weight='bold', color='0.2')
plt.xticks(fontsize=8, color='0.2', weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')

plt.legend(frameon=False, prop = {'weight':'bold', 'size':8}, labelcolor='0.2')

plt.savefig(path+'dLZ cell-FDC interaction frequency.png', dpi=300, bbox_inches='tight')
plt.close()
plt.clf()

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/dLZ cell-FDC interaction frequency.svg', bbox_inches='tight')
plt.close()
plt.clf()


####################################### dLZ FDC interaction kde mt GCB vs wt GCB #############################################
replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}

dataset={}
for condition in np.unique(df['Type']):
    data = df[(df['Type'] == condition)&(df['Zone'] == 'dLZ')]['FDC_avg_overlap'] # ['FDC_contact_times']
    dataset[condition] = np.array(data)
new_order = ['wt_B-cell', 'mt_B-cell']
ordered_dataset = change_dict_order(dataset, new_order)
dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

draw_custom_violin_plot(dict_datasets, path, file_name='dLZ cell-FDC overlap violinplot', colors = ('#888888', '#CC6677'),
                            test='mann-whitney', pvalue=True, figsize=(1,2))

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
    ax = sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=(0, 0.5), color=colors[i], label=key)

# ax = sns.kdeplot(data=dict_datasets['wt GCB'], fill=True, linewidth=1, clip=(0, 40), color='#888888', label='wt GCB')
# ax = sns.kdeplot(data=dict_datasets['mt GCB'], fill=True, linewidth=1, clip=(0, 40), color='#CC6677', label='mt GCB')

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Engaged volume fraction', fontsize=8, weight='bold', color='0.2')
ax.set_ylabel('Density', fontsize=8, weight='bold', color='0.2')
plt.xticks(fontsize=8, color='0.2', weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')

plt.legend(frameon=False, prop = {'weight':'bold', 'size':8}, labelcolor='0.2')

plt.savefig(path+'dLZ cell-FDC overlap frequency.png', dpi=300, bbox_inches='tight')
plt.close()
plt.clf()

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/dLZ cell-FDC overlap frequency.svg', bbox_inches='tight')
plt.close()
plt.clf()

####################################### Average FDC distance kde mt GCB vs wt GCB #############################################
replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}

coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']
ranges = [(0, 60), (0,40), (0, 100)]
names = ['Distance to DZ', 'Distance to sLZ', 'Distance to dLZ']

replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}

for idx, coloc_feature in enumerate(coloc_features):
    dataset={}
    for condition in np.unique(df['Type']):
        data = df[(df['Type'] == condition)][coloc_feature] # ['FDC_contact_times']
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    draw_custom_violin_plot(dict_datasets, path, file_name='%s violinplot'%coloc_feature, colors = ('#888888', '#CC6677'),
                                test='mann-whitney', pvalue=True, figsize=(1,2))

for idx, coloc_feature in enumerate(coloc_features):

    dataset={}
    for condition in np.unique(df['Type']):
        data = df[df['Type'] == condition][coloc_feature]
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

    fig, ax = plt.subplots(figsize=(2,2))

    ax = sns.kdeplot(data=dict_datasets['wt GCB'], fill=True, linewidth=1, clip=ranges[idx], color='#888888', label='wt GCB')
    ax = sns.kdeplot(data=dict_datasets['mt GCB'], fill=True, linewidth=1, clip=ranges[idx], color='#CC6677', label='mt GCB')

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('%s (μm)'%names[idx], fontsize=8, weight='bold', color='0.2')
    ax.set_ylabel('Density', fontsize=8, weight='bold', color='0.2')
    plt.xticks(fontsize=8, color='0.2', weight='bold')
    plt.yticks(fontsize=8, color='0.2', weight='bold')

    plt.legend(frameon=False, prop = {'weight':'bold', 'size':8}, labelcolor='0.2')

    plt.savefig(path+'%s.png'%coloc_feature, dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%se.svg'%coloc_feature, bbox_inches='tight')
    plt.close()
    plt.clf()

############# Plot total_distance vs FDC_contact_persistences jointplot ###############
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})

x_name = 'total_distance'
y_name = 'FDC_contact_persistences'
xmin = math.floor(df_[x_name].min()) - 1
xmax = math.ceil(df_[x_name].max()) + 1
ymin = math.floor(df_[y_name].min()) - 1
ymax = math.ceil(df_[y_name].max()) + 1

draw_jointplot(xs=x_name, y=y_name, df=df_, path=path, file_name='speed vs FDC interaction jointplot',
               hue="Type", colors=('#CC6677', '#888888', ), hue_order=['mt GCB', 'wt GCB'],
               fill=False, legend=False, thresh=0.2, n_contours=5, alpha=1, height=4, ratio=5, space=0,
               xlabels='distance traveled', ylabel='FDC contact persistence', #xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
               )


############# Plot LZ vs DZ jointplot for each cell type ###############

for cell_type in ['mt_B-cell', 'wt_B-cell']:

    df_part = df[df['Type']==cell_type].reset_index(drop=True)

    df_part.loc[(df_part['Zone'] == 'DZ'), 'Zone1'] = 'DZ'
    #df_part.loc[(df_part['Zone'] == 'DZ-sLZ'), 'Zone1'] = 'DZ-sLZ'
    df_part.loc[( (df_part['Zone'] == 'sLZ') | (df_part['Zone'] == 'dLZ') | (df_part['Zone'] == 'sLZ-dLZ') ), 'Zone1'] = 'LZ'
    df_part['Zone1'] = df_part.Zone1.astype(str)

    print(cell_type, df_part[df_part['Zone1']=='DZ'].shape[0], df_part[df_part['Zone1']=='DZ-sLZ'].shape[0], df_part[df_part['Zone1']=='LZ'].shape[0])

    draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s_DZ vs LZ jointplot'%cell_type, hue="Zone1", hue_order=['LZ', 'DZ'],
                   colors=('#E69965', '#BAC8DA'), fill=False, legend=False, thresh=0.25, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')


############# Plot sLZ vs dLZ jointplot for each cell type ###############

for cell_type in ['mt_B-cell', 'wt_B-cell']:

    df_part = df[df['Type']==cell_type].reset_index(drop=True)

    df_part.loc[(df_part['Zone'] == 'sLZ'), 'Zone1'] = 'sLZ'
    #df_part.loc[(df_part['Zone'] == 'sLZ-dLZ'), 'Zone1'] = 'sLZ-dLZ'
    df_part.loc[(df_part['Zone'] == 'dLZ'), 'Zone1'] = 'dLZ'
    df_part['Zone1'] = df_part.Zone1.astype(str)

    print(cell_type, df_part[df_part['Zone1']=='sLZ'].shape[0], df_part[df_part['Zone1']=='sLZ-dLZ'].shape[0], df_part[df_part['Zone1']=='dLZ'].shape[0])

    draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s_sLZ vs dLZ jointplot'%cell_type, hue="Zone1", hue_order=['sLZ', 'dLZ'],
                   colors=('#4F609C', '#8A4F21'), fill=False, legend=False, thresh=0.25, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')


############# Plot Interzone kmeans cross correlation for each cell type ###############
font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

condition_name = 'Type'
cluster_type = 'kmeans'

df_corr_data = pd.DataFrame()
group_clones=[]
for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
#for group in ['DZ', 'sLZ', 'dLZ']:
    corrcoef = []
    aaa = df[df['Zone'] == group]

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

    for column in group_clone.columns:
        group_clone.rename(columns={column:column+'_%s'%group}, inplace=True)
    group_clones.append(group_clone)
    df_corr_data = pd.concat([df_corr_data, group_clone], axis=1)

df_corr = df_corr_data.corr()

df_corr.columns.name = None # Remove name of column = 'Type'
df_corr.index.name = None # Remove name of index = 'Type'


rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_DZ-sLZ': 'wt GCB DZ-sLZ', 'wt_B-cell_sLZ': 'wt GCB sLZ',
               'wt_B-cell_sLZ-dLZ': 'wt GCB sLZ-dLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
               'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_DZ-sLZ': 'mt GCB DZ-sLZ', 'mt_B-cell_sLZ': 'mt GCB sLZ',
               'mt_B-cell_sLZ-dLZ': 'mt GCB sLZ-dLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ',
               }
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

plt.savefig(path+'3 zone correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/3 zones correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

############# Plot Close vs Far fraction of cluster for each cell type ###############
vmax=30
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

############# All Kmeans distribution heatmap ###############
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
df_['Type_Zone'] = df_['Type'].astype(str) + ' ' + df_['Zone']
group_df = draw_cluster_distribution_heatmap(df_, path, file_name='all_kmeans_heatmap', condition_name='Type_Zone', annot=False,
                                  cluster_type='kmeans', col_cluster=False, figsize=(6,5))

p_dict = permutation_test(df_, group_name='Type_Zone', class_name='kmeans', iteration=50000)

draw_heatmap_with_circles(df_, path, file_name='all_kmeans_circleheatmap', condition_name='Type_Zone', cluster_type='kmeans',
                          vmax=None, transpose=False, row_cluster=True, col_cluster=False, figsize=(4,4))

for zone in np.unique(df_['Type_Zone']):
    print(zone, df_[df_['Type_Zone']==zone].shape[0])


for zone in ['DZ', 'sLZ', 'dLZ']:
    enrichment = {}
    for cluster in group_df.columns:
        temp = group_df.iloc[:, cluster]
        diff = temp.loc['mt GCB %s'%zone] - temp.loc['wt GCB %s'%zone]

        enrichment[cluster] = np.array(diff)

    cm = cmc.batlow
    n_colors = group_df.columns.size
    colors = [cm(1. * i / n_colors) for i in range(n_colors)]

    draw_custom_bar_plot(enrichment, path, file_name='MT vs WT MC enrichment %s'%zone,
                             strip_plot=False, colors=colors, vmax=10, vmin=-20,
                         test='mann-whitney', pvalue=False, figsize=(1, 2))

############# Plot linear regression btw FDC zones and cluster enrichment ###############


mt_enrichments = pd.DataFrame()
wt_enrichments = pd.DataFrame()
for group_clone in group_clones:  # 'DZ', 'sLz', 'dLZ'
    mt_enrichments = pd.concat( [mt_enrichments, group_clone[list(group_clone.columns)[0]]], axis=1 )
    wt_enrichments = pd.concat([wt_enrichments, group_clone[list(group_clone.columns)[1]]], axis=1)

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
for idx, kmeans in enumerate(mt_enrichments.index):
    r, p = scipy.stats.spearmanr(np.arange(len(list(mt_enrichments.iloc[kmeans, :].index))),
                                mt_enrichments.iloc[kmeans, :].values)
    if p>0.11:
        continue
    sns.regplot(x=np.arange(len(list(mt_enrichments.iloc[kmeans, :].index))), y=mt_enrichments.iloc[kmeans, :].values,
                ci=None, line_kws={'color':cmap[idx], 'linewidth':3}, label=kmeans, scatter=False, scatter_kws={'s': 10, 'color':cmap[idx]})
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

ax.set_ylabel('fraction of cluster (%)', fontsize=16, weight='bold', color='0.2')
plt.xticks(np.arange(len(list(mt_enrichments.iloc[kmeans, :].index))), ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ'], fontsize=16, rotation=35, # fontname = "Arial",
           rotation_mode='anchor', ha='right', color='0.2', weight='bold')

plt.yticks(fontsize=16, color='0.2', weight='bold')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
           loc='best')

plt.savefig(path + 'mt GCB cluster fraction regplot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/mt GCB cluster fraction regplot.svg', bbox_inches='tight')
plt.clf()
plt.close()



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
for idx, kmeans in enumerate(wt_enrichments.index):
    r, p = scipy.stats.spearmanr(np.arange(len(list(wt_enrichments.iloc[kmeans, :].index))),
                                wt_enrichments.iloc[kmeans, :].values)
    if p>0.11:
        continue
    sns.regplot(x=np.arange(len(list(wt_enrichments.iloc[kmeans, :].index))), y=wt_enrichments.iloc[kmeans, :].values,
                ci=None, line_kws={'color':cmap[idx], 'linewidth':3}, label=kmeans, scatter=False, scatter_kws={'s': 10, 'color':cmap[idx]})
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

ax.set_ylabel('fraction of cluster (%)', fontsize=16, weight='bold', color='0.2')
plt.xticks(np.arange(len(list(wt_enrichments.iloc[kmeans, :].index))), ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ'], fontsize=16, rotation=35, # fontname = "Arial",
           rotation_mode='anchor', ha='right', color='0.2', weight='bold')

plt.yticks(fontsize=16, color='0.2', weight='bold')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
           loc='best')

plt.savefig(path + 'wt GCB cluster fraction regplot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/wt GCB cluster fraction regplot.svg', bbox_inches='tight')
plt.clf()
plt.close()


############# Shannon entropy of DZ vs sLZ vs dLZ for each video ###############

group_name = 'Video'
groups = np.unique(df[group_name])

entropies_dz = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Zone'] == 'DZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_dz[type].append(entropy[type])

entropies_dz_slz = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Zone'] == 'DZ-sLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_dz_slz[type].append(entropy[type])

entropies_slz = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Zone'] == 'sLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_slz[type].append(entropy[type])

entropies_slz_dlz = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Zone'] == 'sLZ-dLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_slz_dlz[type].append(entropy[type])

entropies_dlz = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Zone'] == 'dLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_dlz[type].append(entropy[type])

entropies = {}
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    entropies = {'%s_DZ'%cell_type:entropies_dz[cell_type], '%s_DZ-sLZ'%cell_type:entropies_dz_slz[cell_type], '%s_sLZ'%cell_type:entropies_slz[cell_type],
                  '%s_sLZ-dLZ'%cell_type:entropies_slz_dlz[cell_type], '%s_dLZ'%cell_type:entropies_dlz[cell_type],}

    # entropies = {'%s_DZ' % cell_type: entropies_dz[cell_type], '%s_sLZ' % cell_type: entropies_slz[cell_type],
    #              '%s_dLZ' % cell_type: entropies_dlz[cell_type], }

    rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_DZ-sLZ': 'wt GCB DZ-sLZ', 'wt_B-cell_sLZ': 'wt GCB sLZ',
                   'wt_B-cell_sLZ-dLZ': 'wt GCB sLZ-dLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
                   'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_DZ-sLZ': 'mt GCB DZ-sLZ', 'mt_B-cell_sLZ': 'mt GCB sLZ',
                    'mt_B-cell_sLZ-dLZ': 'mt GCB sLZ-dLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ',}
    entropies_ = {rename_keys.get(k, k): v for (k, v) in entropies.items()}
    # draw_custom_bar_plot(entropies_, path, file_name='entropy of DZ vs sLZ vs dLZ for %s' %cell_type, colors=('#888888', '#CC6677', '#6699CC'),
    #                      strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))
    dict_datasets = entropies_
    file_name = 'entropy of DZ vs sLZ vs dLZ for %s' %cell_type
    test = 'mann-whitney'

    colors = ('#888888', '#CC6677', '#6699CC', '#44AA99', '#DDCC77')
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    fig, ax = plt.subplots(figsize=(1, 2))
    sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

    ax = sns.barplot(data=sorted_vals, capsize=0.5, edgecolor='0.2', lw=1, errwidth=1, palette=colors)
    plot_params = {'edgecolor': '0.2', 'linewidth': 1, 'fc': 'none'}
    ax = sns.stripplot(data=sorted_vals, marker='s', s=1.5, **plot_params)
    # marker='s'(square), s = marker size

    # format_figure(ax, title=None, xlabel=None, ylabel=None, despine=True, detick=True)
    ax.axhline(max_entropy, linestyle='--', linewidth=1, color='0.2')
    # plt.xticks(plt.xticks()[0], sorted_keys, fontsize=12, fontdict={'weight': 'bold'})
    # ax.set_xticks(plt.xticks()[0], sorted_keys, fontsize=4)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Shannon entropy', fontsize=8, weight='bold', color='0.2')
    plt.xticks(plt.xticks()[0], sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
               weight='bold')
    plt.yticks(fontsize=8, color='0.2', weight='bold')
    # plt.ylabel('%s' % feature_name, fontsize=4)
    # category labels
    plt.grid(False)

    from scipy import stats
    from itertools import combinations

    p_values = []
    pairs = []
    for pair in combinations(range(0, len(dict_datasets)), 2):  # 2 for pairs, 3 for triplets, etc
        if test == 'mann-whitney':
            stat_test = stats.mannwhitneyu(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
        elif test == 't-test':
            stat_test = stats.ttest_ind(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
        elif test == 'wilcoxon-ranksum':
            stat_test = stats.ranksums(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
        p_values.append(stat_test.pvalue)
        pairs.append(pair)
        #print(pair, stat_test.pvalue)
    plt.title('%s:%s' % (pairs, p_values), fontsize=4)
    plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()


############# Shannon entropy of DZ vs sLZ vs dLZ for each video ###############

df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})

group_name = 'Video'
groups = np.unique(df_[group_name])

entropies_dz = {'mt GCB': [], 'wt GCB':[]}
for group in groups:
    df_part = df_[(df_[group_name]==group)&(df_['Zone'] == 'DZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df_, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_dz[type].append(entropy[type])

entropies_dz_slz = {'mt GCB': [], 'wt GCB':[]}
for group in groups:
    df_part = df_[(df_[group_name]==group)&(df_['Zone'] == 'DZ-sLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df_, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_dz_slz[type].append(entropy[type])

entropies_slz = {'mt GCB': [], 'wt GCB':[]}
for group in groups:
    df_part = df_[(df_[group_name]==group)&(df_['Zone'] == 'sLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df_, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_slz[type].append(entropy[type])

entropies_slz_dlz = {'mt GCB': [], 'wt GCB':[]}
for group in groups:
    df_part = df_[(df_[group_name]==group)&(df_['Zone'] == 'sLZ-dLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df_, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_slz_dlz[type].append(entropy[type])

entropies_dlz = {'mt GCB': [], 'wt GCB':[]}
for group in groups:
    df_part = df_[(df_[group_name]==group)&(df_['Zone'] == 'dLZ')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df_, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_dlz[type].append(entropy[type])

test = 'mann-whitney'
color_list=['#CC6677', '#888888']
marker_list=['o', '^']
entropies = {}

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(4,4))

p_value_data = {}
for idx, cell_type in enumerate(['mt GCB', 'wt GCB']):
    entropies = {'%s_DZ'%cell_type:entropies_dz[cell_type], '%s_DZ-sLZ'%cell_type:entropies_dz_slz[cell_type], '%s_sLZ'%cell_type:entropies_slz[cell_type],
                  '%s_sLZ-dLZ'%cell_type:entropies_slz_dlz[cell_type], '%s_dLZ'%cell_type:entropies_dlz[cell_type],}

    # entropies = {'%s_DZ' % cell_type: entropies_dz[cell_type], '%s_sLZ' % cell_type: entropies_slz[cell_type],
    #              '%s_dLZ' % cell_type: entropies_dlz[cell_type], }

    rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_DZ-sLZ': 'wt GCB DZ-sLZ', 'wt_B-cell_sLZ': 'wt GCB sLZ',
                   'wt_B-cell_sLZ-dLZ': 'wt GCB sLZ-dLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
                   'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_DZ-sLZ': 'mt GCB DZ-sLZ', 'mt_B-cell_sLZ': 'mt GCB sLZ',
                    'mt_B-cell_sLZ-dLZ': 'mt GCB sLZ-dLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ',}
    entropies_ = {rename_keys.get(k, k): v for (k, v) in entropies.items()}

    p_value_data[cell_type] = entropies_
    mean_dataset = {}
    error_dataset = {}

    for key in entropies_:
        values = entropies_[key]
        mean = np.mean(values)
        mean_dataset[key] = mean
        # error = np.std(values)
        # interval = stats.norm.interval(confidence=0.95, loc=np.mean(values), scale=stats.sem(values))
        # error = np.mean(values) - interval[0]
        error = stats.sem(values)
        error_dataset[key] = error

    sns.lineplot(data=mean_dataset, x=np.arange(len(list(mean_dataset))), y=mean_dataset.values(),
                 label=cell_type, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars',
                 color=color_list[idx])
    ax.errorbar(np.arange(len(list(mean_dataset))), mean_dataset.values(), [x for x in error_dataset.values()],
                color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

handles, labels = ax.get_legend_handles_labels()
# ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['mt GCB'], 0.1*std_dataset['mt GCB'], color='#CC6677',
#            capsize=3, capthick=1, elinewidth=1.5)
# ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['wt GCB'], 0.1*std_dataset['wt GCB'], color='#888888',
#            capsize=3, capthick=1, elinewidth=1.5)

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_ylabel('Shannon entropy', fontsize=16, weight='bold', color='0.2')

#ax.set_xlabel('%s' % x_label, fontsize=16, weight='bold', color='0.2')

plt.xticks(np.arange(len(list(mean_dataset))), ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ'], fontsize=16, rotation=35, # fontname = "Arial",
           rotation_mode='anchor', ha='right', color='0.2', weight='bold')

plt.yticks(fontsize=16, color='0.2', weight='bold')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
           loc='best')

from scipy import stats
p_values = []
pairs = []
for (mt_key, mt_values), (wt_keys, wt_values) in zip(p_value_data['mt GCB'].items(), p_value_data['wt GCB'].items()):
    if test == 'mann-whitney':
        stat_test = stats.mannwhitneyu(mt_values, wt_values)
    elif test == 't-test':
        stat_test = stats.ttest_ind(mt_values, wt_values)
    elif test == 'wilcoxon-ranksum':
        stat_test = stats.ranksums(mt_values, wt_values)
    #print(mt_key, stat_test.pvalue)
    p_values.append(stat_test.pvalue)
    pairs.append(mt_key)

plt.title('%s:%s' % (pairs, p_values), fontsize=4)
plt.savefig(path + 'entropy of different zones.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/' ):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/entropy of different zones.svg', bbox_inches='tight')
plt.clf()
plt.close()


###################### Plot Zone motility feature violin plot for mt GCB and wt GCB  ############################

if not os.path.isdir(path + 'Zone motility box plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Zone motility box plot/')

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for group in ['DZ','sLZ', 'dLZ']:
        #for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
            data = df[(df[condition_name] == cell_type)&(df['Zone'] == group)][feature_name]

            dataset[cell_type+'_'+str(group)] = np.array(data)

    rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_sLZ': 'wt GCB sLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
                   'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_sLZ': 'mt GCB sLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ',}
    # rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_DZ-sLZ': 'wt GCB DZ-sLZ', 'wt_B-cell_sLZ': 'wt GCB sLZ',
    #                'wt_B-cell_sLZ-dLZ': 'wt GCB sLZ-dLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
    #                'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_DZ-sLZ': 'mt GCB DZ-sLZ', 'mt_B-cell_sLZ': 'mt GCB sLZ',
    #                 'mt_B-cell_sLZ-dLZ': 'mt GCB sLZ-dLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ',}

    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_violin_plot(dataset_renamed, path + 'Zone motility box plot/', file_name=feature_name,
                            colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))
    # draw_custom_violin_plot(dataset_renamed, path + 'Zone motility box plot/', file_name=feature_name,
    #                         colors=('#888888', '#888888', '#888888', '#888888', '#888888',
    #                                 '#CC6677', '#CC6677', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))


###################### Plot Experimental motility feature plots for Zones  ############################
videos = np.unique(df['Video'])

if not os.path.isdir(path + 'experimental_zone_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_zone_motility_feature/')

for feature_name in feature_list:
    dataset = {}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for group in ['DZ','sLZ', 'dLZ']:
            df_part = df[(df['Type'] == cell_type)&(df['Zone'] == group)].reset_index(drop=True)
            avgs = []
            for video in videos:
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] == 0:
                    continue
                data = df_video[feature_name]
                avg = np.mean(data)
                avgs.append(avg)
            dataset[cell_type + '_' + str(group)] = avgs
    #new_order = ['wt_B-cell', 'wt_B-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_sLZ': 'wt GCB sLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
                   'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_sLZ': 'mt GCB sLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ', }
    dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_bar_plot(dict_datasets, path+'experimental_zone_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))


#################################### Volcano plot of Close vs Far motility features ####################################
# feature_list = df.columns[130:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z', 'phi'])
# for cell_type in ['mt_B-cell', 'wt_B-cell']:
#     df_p = pd.DataFrame()
#     for feature_name in feature_list:
#         dataset = {}
#         for interaction in ['far','close']:
#             if interaction == 'far':
#                 data = df[(df['Average_distance_to_FDC'] >= 20)&(df[condition_name] == cell_type)][feature_name]
#             elif interaction == 'close':
#                 data = df[(df['Average_distance_to_FDC'] <= 5)&(df[condition_name] == cell_type)][feature_name]
#             dataset[cell_type + '_' + str(interaction)] = np.array(data)
#         pvalue = get_pvalue(dataset, test='mann-whitney')
#         logp = -np.log10(pvalue)
#
#         avgZ = get_avgZ(dataset, ref_name=cell_type+'_far', data_name=cell_type+'_close')
#
#         row = pd.DataFrame()
#         row['Feature'] = [feature_name]
#         row['Pvalue'] = [pvalue]
#         row['-Logp'] = [logp]
#         row['AvgZ'] = [avgZ]
#         df_p = pd.concat([df_p, row], axis=0)
#
#     df_p['Adj_p'] = adjusted_pvalues(df_p['Pvalue'], correction_type='Benjamini-Hochberg')
#     df_p['Adj_Logp'] = -np.log(df_p['Adj_p'])
#     df_p = df_p[~df_p.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf
#
#
#     draw_volcano_plot(df_p, path, file_name='%s far vs close motility volcano plot'%cell_type, z_thresh=0.5, p_thresh=5, z_name='AvgZ', p_name='Adj_Logp',
#                       feature_name='Feature', figsize=(6,6))


######################## Zone interaction features mt vs wt GCB box plot  ###########################
if not os.path.isdir(path + 'Zone int feature violin plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Zone int feature violin plot/')

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'Zone'])

# k = df.iloc[:,324:417].isnull().any()
# null_features = k.index[k==True]
# feature_list = df.columns[324:417].drop(null_features).drop(['PC1', 'PC2', 'tskmeans'])
for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for zone in ['DZ','sLZ', 'dLZ']:
            data = df[(df['Zone'] == zone) & (df[condition_name] == cell_type)][feature_name]
            dataset[cell_type + '_' + str(zone)] = np.array(data)

    values = flatten_nested_dict(dataset)
    if np.isnan(values).any() == True:  # Check at least one nan
        continue
    elif np.isfinite(values).all() == False:  # Check everything is not inf
        continue

    rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_sLZ': 'wt GCB sLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
                   'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_sLZ': 'mt GCB sLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ', }
    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    # draw_custom_violin_plot(dataset_renamed, path + 'Zone int feature violin plot/', file_name=feature_name,
    #                         colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))

    draw_custom_bar_plot(dataset_renamed, path + 'Zone int feature violin plot/', file_name=feature_name,
                         strip_plot=False, colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'),
                         test='mann-whitney', pvalue=True, figsize=(2, 2))

######################## DZ/sLZ/dLZ interaction features mt vs wt GCB box plot  ###########################

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'Zone'])

for zone in ['DZ','sLZ', 'dLZ']:
    df_part = df[(df['Zone'] == zone)].reset_index(drop=True)
    if not os.path.isdir(path + '%s int feature violin plot/'%zone):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s int feature violin plot/'%zone)
    for feature_name in feature_list:
        condition_name = 'Type'
        dataset={}
        for cell_type in ['wt_B-cell', 'mt_B-cell']:
            data = df_part[df_part[condition_name] == cell_type][feature_name]
            dataset[cell_type] = np.array(data)

        values = flatten_nested_dict(dataset)
        if np.isnan(values).any() == True:  # Check at least one nan
            continue
        elif np.isfinite(values).all() == False:  # Check everything is not inf
            continue

        rename_keys = {'wt_B-cell': 'wt GCBssss', 'mt_B-cell': 'mt GCB' }
        dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

        # draw_custom_bar_plot(dataset_renamed, path + '%s int feature violin plot/'%zone, file_name=feature_name,
        #                      strip_plot=False, colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))
        draw_custom_violin_plot(dataset_renamed, path + '%s int feature violin plot/'%zone, file_name=feature_name,
                                colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))

###################### Plot Experimental motility feature plots for Zones  ############################
df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone'])

videos = np.unique(df['Video'])

if not os.path.isdir(path + 'experimental_zone_int_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_zone_int_feature/')

for feature_name in feature_list:
    dataset = {}
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        for group in ['DZ','sLZ', 'dLZ']:
            df_part = df[(df['Type'] == cell_type)&(df['Zone'] == group)].reset_index(drop=True)
            avgs = []
            for video in videos:
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] == 0:
                    continue
                data = df_video[feature_name]
                avg = np.mean(data)
                avgs.append(avg)
            dataset[cell_type + '_' + str(group)] = avgs
    #new_order = ['wt_B-cell', 'wt_B-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_sLZ': 'wt GCB sLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
                   'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_sLZ': 'mt GCB sLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ', }
    dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_bar_plot(dict_datasets, path+'experimental_zone_int_feature/', file_name=feature_name,
                         strip_plot=True, colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))


#################################### all motility features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
df_.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df_.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']

for idx in [0,1,2]:
    coloc_feature = coloc_features[idx]
    if idx == 0:
        xlabel = 'Distance to DZ (µm)'
    elif idx == 1:
        xlabel = 'Distance to sLZ (µm)'
    elif idx == 2:
        xlabel = 'Distance to dLZ (µm)'

    draw_lineplot_by_custom_ranges(df_, path, folder_name='motility_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                                   condition_name='Type', custsom_range=(0, 40), stepsize=4, range_feature=coloc_feature,
                                       color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label=xlabel,
                                   estimator='mean', error_type='ci_norm',replace_keys=None, pvalue=True, test='mann-whitney', set_zero=False)

#################################### all approach / departure motility features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
df_.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df_.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']

coloc_feature = coloc_features[2]

df_approach = df_[df_['quality_Core_approach_times']>=12].reset_index(drop=True)
print(df_[df_['quality_Core_approach_times']>=12].shape[0])

draw_lineplot_by_custom_ranges(df_approach, path, folder_name='approach_motility_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                               condition_name='Type', custsom_range=(0, 40), stepsize=4, range_feature=coloc_feature,
                                   color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label='Distance to dLZ (µm)',
                               estimator='mean', error_type='ci_norm',replace_keys=None, pvalue=True, test='mann-whitney')


#################################### all interaction features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
df_.columns.get_loc('quality_FDC_approach_times')
feature_list = df_.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone'])

coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']

for idx in [0,1,2]:
    coloc_feature = coloc_features[idx]
    if idx == 0:
        xlabel = 'Distance to DZ (µm)'
    elif idx == 1:
        xlabel = 'Distance to sLZ (µm)'
    elif idx == 2:
        xlabel = 'Distance to dLZ (µm)'
    draw_lineplot_by_custom_ranges(df_, path, folder_name='interaction_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                                   condition_name='Type', custsom_range=(0, 40), stepsize=4, range_feature=coloc_feature,
                                       color_list=['#CC6677', '#888888'], marker_list=['o', '^', ], figsize=(4,4), x_label=xlabel,
                                   estimator='mean', error_type='ci_norm', replace_keys=None, pvalue=True, test='mann-whitney')


#################################### Correlation between morphodynamics and speed ####################################

linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = np.unique(df['Type']).size
ncols = 3
from matplotlib.ticker import MaxNLocator
font = {'family': 'arial',
            'weight': 'normal',
            'size': 10}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = linewidth

fig,axes = plt.subplots(nrows, ncols, figsize=(15, 10), sharey='row')
for row, type in enumerate(['wt_B-cell', 'mt_B-cell']):
    for col, zone in enumerate(['DZ', 'sLZ', 'dLZ']):
        ax = axes[row][col]
        df_part_= df[ (df['Type']==type)&(df['Zone']==zone) ].reset_index(drop=True)
        sns.regplot(x='avg_speed', y='morpho_avg_speed', data=df_part_, scatter_kws={"color":"black", "alpha":0.3, 's':5}, line_kws={"color":"black"}, ax=ax)

        r, p = scipy.stats.pearsonr(df_part_['avg_speed'], df_part_['morpho_avg_speed'])
        plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                 fontsize=14, fontdict={'weight': 'bold'}, color="black")
        # plt.text(0.1, 0.88, "p = " + str(p), ha='left', va='top', transform=ax.transAxes,
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

        ax.set_xlabel('Average speed', fontsize=12, weight='bold', color='0.2', labelpad=5)
        ax.set_ylabel('Shape Deformability', fontsize=12, weight='bold', color='0.2', labelpad=5)
    #ax.set_xlim(16, 98)

plt.savefig(path + 'corr btw speed and deformability.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/corr btw speed and deformability.svg', bbox_inches='tight')
plt.clf()
plt.close()

#################################### all motility features wrt FDC contact time ####################################
# FDC_dist_range = (0,20)
#
# feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
#
# for feature_name in feature_list:
#     mean_dataset = {}
#     std_dataset = {}
#
#     for cell_type in ['mt_B-cell', 'wt_B-cell']:
#         df_part = df[df['Type'] == cell_type]
#         means = []
#         stds = []
#
#         for i in np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1):
#
#             if i == FDC_dist_range[1]:
#                 values = df_part[df_part['tp_interaction_FDC'] >= i][feature_name].values
#             else:
#                 values = df_part[(df_part['tp_interaction_FDC'] >= i) & (df_part['tp_interaction_FDC'] < i + 1)][feature_name].values
#
#             # interaction_type = 'FDC'
#             # interaction_list = ['wt_B-cell', 'mt_B-cell', 'FDC', 'macrophage', 'T-cell']
#             # interaction_list.remove(cell_type)
#             # interaction_list.remove(interaction_type)
#
#             # if i == FDC_dist_range[1]:
#             #     values = df_part[(df_part['tp_interaction_FDC'] >= i) & (df_part['tp_interaction_%s' % interaction_list[0]] <= 5)
#             #          & (df_part['tp_interaction_%s' % interaction_list[1]] <= 5) & (df_part['tp_interaction_%s' % interaction_list[2]] <= 5)][feature_name].values
#             # else:
#             #     values = df_part[(df_part['tp_interaction_FDC'] >= i) & (df_part['tp_interaction_FDC'] < i + 1) & (df_part['tp_interaction_%s' % interaction_list[0]] <= 5)
#             #          & (df_part['tp_interaction_%s' % interaction_list[1]] <= 5) & (df_part['tp_interaction_%s' % interaction_list[2]] <= 5)][feature_name].values
#
#             #means.append(np.mean(values))
#             means.append(np.median(values))
#             stds.append(np.std(values))
#
#         mean_dataset[cell_type] = np.array(means)
#         std_dataset[cell_type] = np.array(stds)
#
#
#     replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
#     mean_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_dataset.items() }
#     std_dataset = {replace_keys.get(k, k):v  for (k,v) in std_dataset.items() }
#
#     font = {'family': 'arial',
#                 'weight': 'normal',
#                 'size': 16}
#     matplotlib.rc('font', **font)
#     matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
#     matplotlib.rcParams['lines.linewidth'] = 2
#
#     fig, ax = plt.subplots(figsize=(4,4))
#     ax = sns.lineplot(data=mean_dataset, lw=2.5, markers=['o', '^'], dashes=False, markersize=8,
#                  err_style='bars', palette=['#CC6677', '#888888'])
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['mt GCB'], 0.1*std_dataset['mt GCB'], color='#CC6677',
#                capsize=3, capthick=1, elinewidth=1.5)
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['wt GCB'], 0.1*std_dataset['wt GCB'], color='#888888',
#                capsize=3, capthick=1, elinewidth=1.5)
#
#     for axis in ['bottom', 'left']:
#         ax.spines[axis].set_linewidth(2)
#         ax.spines[axis].set_color('0.2')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     ax.tick_params(width=2, color='0.2')
#
#     ax.set_xlabel('Contact time with FDC', fontsize=16, weight='bold', color='0.2')
#     #ax.set_ylabel('Average speed (μm/min)', fontsize=16, weight='bold', color='0.2')
#     plt.xticks(fontsize=16, color='0.2',weight='bold', )
#     plt.yticks(fontsize=16, color='0.2', weight='bold', )
#
#     plt.legend(frameon=False, prop={'weight':'bold', 'size':12}, labelcolor='0.2')
#     # plt.ylabel('%s' % feature_name, fontsize=4)
#
#     if not os.path.isdir(path + 'feature_wrt_FDC_contact_time/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'feature_wrt_FDC_contact_time/')
#
#     plt.savefig(path + 'feature_wrt_FDC_contact_time/%s.png'%feature_name, dpi=300, bbox_inches='tight')
#
#     if not os.path.isdir(path + 'feature_wrt_FDC_contact_time/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'feature_wrt_FDC_contact_time/svg/')
#     plt.savefig(path + 'feature_wrt_FDC_contact_time/svg/%s.svg'%feature_name, bbox_inches='tight')
#     plt.clf()
#     plt.close()



#################################### all interaction features wrt FDC contact time ####################################
# FDC_dist_range = (0,20)
# feature_list = df.columns[324:].drop(['interacted_type'])
#
# for feature_name in feature_list:
#     mean_dataset = {}
#     std_dataset = {}
#
#     for cell_type in ['mt_B-cell', 'wt_B-cell']:
#         df_part = df[df['Type'] == cell_type]
#         means = []
#         stds = []
#
#         for i in np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1):
#             if i == FDC_dist_range[1]:
#                 values = df_part[df_part['tp_interaction_FDC'] >= i][feature_name].values
#             else:
#                 values = df_part[(df_part['tp_interaction_FDC'] >= i) & (df_part['tp_interaction_FDC'] < i + 1)][feature_name].values
#
#             if np.isnan(values).any() == True:  # Check at least one nan
#                 means.append(np.nan)
#                 stds.append(np.nan)
#
#             elif np.isfinite(values).all() == False:  # Check everything is not inf
#                 means.append(np.nan)
#                 stds.append(np.nan)
#
#             else:
#                 means.append(np.mean(values))
#                 #means.append(np.median(values))
#                 stds.append(np.std(values))
#
#         mean_dataset[cell_type] = np.array(means)
#         std_dataset[cell_type] = np.array(stds)
#
#     check_nan = flatten_nested_dict(mean_dataset)
#     if np.isnan(check_nan).any() == True:  # Check at least one nan
#         continue
#
#     replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
#     mean_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_dataset.items() }
#     std_dataset = {replace_keys.get(k, k):v  for (k,v) in std_dataset.items() }
#
#     font = {'family': 'arial',
#                 'weight': 'normal',
#                 'size': 16}
#     matplotlib.rc('font', **font)
#     matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
#     matplotlib.rcParams['lines.linewidth'] = 2
#
#     fig, ax = plt.subplots(figsize=(4,4))
#     ax = sns.lineplot(data=mean_dataset, lw=2.5, markers=['o', '^'], dashes=False, markersize=8,
#                  err_style='bars', palette=['#CC6677', '#888888'])
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['mt GCB'], 0.1*std_dataset['mt GCB'], color='#CC6677',
#                capsize=3, capthick=1, elinewidth=1.5)
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['wt GCB'], 0.1*std_dataset['wt GCB'], color='#888888',
#                capsize=3, capthick=1, elinewidth=1.5)
#
#     for axis in ['bottom', 'left']:
#         ax.spines[axis].set_linewidth(2)
#         ax.spines[axis].set_color('0.2')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     ax.tick_params(width=2, color='0.2')
#
#     ax.set_xlabel('Contact time with FDC', fontsize=16, weight='bold', color='0.2')
#     #ax.set_ylabel('Average speed (μm/min)', fontsize=16, weight='bold', color='0.2')
#     plt.xticks(fontsize=16, color='0.2',weight='bold', )
#     plt.yticks(fontsize=16, color='0.2', weight='bold', )
#
#     plt.legend(frameon=False, prop={'weight':'bold', 'size':12}, labelcolor='0.2')
#     # plt.ylabel('%s' % feature_name, fontsize=4)
#
#     if not os.path.isdir(path + 'int_feature_wrt_FDC_contact_time/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'int_feature_wrt_FDC_contact_time/')
#
#     plt.savefig(path + 'int_feature_wrt_FDC_contact_time/%s.png'%feature_name, dpi=300, bbox_inches='tight')
#
#     if not os.path.isdir(path + 'int_feature_wrt_FDC_contact_time/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'int_feature_wrt_FDC_contact_time/svg/')
#     plt.savefig(path + 'int_feature_wrt_FDC_contact_time/svg/%s.svg'%feature_name, bbox_inches='tight')
#     plt.clf()
#     plt.close()



