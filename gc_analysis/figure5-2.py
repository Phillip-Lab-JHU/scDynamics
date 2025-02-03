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
"""Generates Data for Figure5-2. Inhibition analysis with FDC and Tfh interaction"""

from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import DistanceSignal

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis_test\feature_csvs\\'
df = pd.read_parquet(path+'GCB_with_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_with_inhibit_traj_duration_20.parquet')

df_inhibit = df[(df['Exp']=='IgG')|(df['Exp']=='CD40L')|(df['Exp']=='mLT')].reset_index(drop=True)
df_duration_inhibit = df_duration[(df_duration['Exp']=='IgG')|(df_duration['Exp']=='CD40L')|(df_duration['Exp']=='mLT')].reset_index(drop=True)

df['Inhibition'] = df['Exp'].copy()
df['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)

df_duration['Inhibition'] = df_duration['Exp'].copy()
df_duration['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)

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

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis_test\Figure5. Inhibition\FDC and Tfh interaction\\'

####################################### Quantify Tfh and FDC interaction frequency #############################################
int_time = 15
features = ['T_contact_times', 'FDC_contact_times']
names = ['Tfh', 'FDC']

for typ in ['wt_B-cell', 'mt_B-cell']:
    for feature, name in zip(features, names):
        persistent_int_freqs_datasets = {}
        persistent_int_freq_per_cellnumbers_datasets = {}
        total_n_contacts_per_cellnumbers_datasets = {}
        persistent_int_freq_per_cellcontacts_datasets = {}
        low_contact_freq_per_cellnumbers_datasets = {}

        videos = np.unique(df['Video'])
        for cell_type in np.unique(df['Inhibition']):
            df_part = df[(df['Inhibition'] == cell_type)&(df['Type']==typ)]
            persistent_int_freqs = []
            persistent_int_freq_per_cellnumbers = []
            total_n_contacts_per_cellnumbers = []
            persistent_int_freq_per_cellcontacts = []
            low_contact_freq_per_cellnumbers = []
            videos = np.unique(df_part['Video'])
            for video in videos:
                #if 'A' in video and cell_type == 'mt_B-cell':
                # if '-A' in video:
                #     continue
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] == 0:
                    continue
                data = df_video[feature]
                mask = ~np.isnan(data)
                data = data[mask]
                persistent_int_freq = sum(data >= int_time)
                total_n_contact = sum(data)

                persistent_int_freq_per_cellnumber = persistent_int_freq / df_video.shape[0]
                total_n_contacts_per_cellnumber = total_n_contact / df_video.shape[0]
                if sum(data) != 0:
                    persistent_int_freq_per_cellcontact = persistent_int_freq / sum(data)
                elif sum(data) == 0:
                    persistent_int_freq_per_cellcontact = 0
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

        replace_keys = {'Control':'Control', 'IgG':'IgG', 'CD40L':'CD40L', 'mLT':'mLT'}
        #new_order = ['Control', 'ControlAb', 'CD40LAb']
        #new_order = ['Control', 'IgG', 'CD40L', 'mLT']
        new_order = ['IgG', 'CD40L', 'mLT']
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

        #colors=('#888888', '#6699CC', '#CC6677')
        colors = ('#888888', '#CC6677', '#6699CC', '#44AA99')

        draw_custom_bar_plot(persistent_int_freqs_datasets, path, file_name='Total %s %s interaction frequency'%(name, typ),
                             strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))
        draw_custom_bar_plot(persistent_int_freq_per_cellnumbers_datasets, path, file_name='%s %s persistent interaction frequency per cell number'%(name, typ),
                             strip_plot=True,colors=colors, test='t-test', pvalue=True, figsize=(1,2))
        draw_custom_bar_plot(total_n_contacts_per_cellnumbers_datasets, path, file_name='number of %s %s contacts per cell number'%(name, typ),
                            strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))
        draw_custom_bar_plot(persistent_int_freq_per_cellcontacts_datasets, path, file_name='%s %s persistent interaction frequency per number of contacts'%(name, typ),
                             strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
        draw_custom_bar_plot(low_contact_freq_per_cellnumbers_datasets, path, file_name='%s %s low contact time frequency per cell number'%(name, typ),
                             strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))


####################################### Average FDC and Tfh distance kde mt GCB vs wt GCB #############################################
replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}

coloc_features = ['FDC_distance_average', 'T_distance_average']
ranges = [(0, 20), (0,30)]
names = ['Distance to FDC', 'Distance to Tfh']

for idx, coloc_feature in enumerate(coloc_features):

    for condition in np.unique(df['Type']):
        dataset = {}
        for group in np.unique(df['Inhibition']):
            data = df[(df['Type'] == condition)&(df['Inhibition'] == group)][coloc_feature]
            dataset[group] = np.array(data)
        #new_order = ['Control', 'IgG', 'CD40L', 'mLT']
        new_order = ['IgG', 'CD40L', 'mLT']
        ordered_dataset = change_dict_order(dataset, new_order)
        dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

        sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

        font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 8}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 1

        colors = ('#888888', '#CC6677', '#6699CC', '#44AA99')
        fig, ax = plt.subplots(figsize=(2,2))
        for i, key in enumerate(dict_datasets):
            sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=ranges[idx], color=colors[i], label=key)

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

        plt.savefig(path+'%s %s.png'%(condition, coloc_feature), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

        if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/')
        plt.savefig(path + 'svg/%s %s.svg'%(condition, coloc_feature), bbox_inches='tight')
        plt.close()
        plt.clf()

####################################### Zone distance kde mt GCB vs wt GCB #############################################
replace_keys = {'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}

coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']
ranges = [(0, 60), (0, 30), (0, 100)]
names = ['Distance to DZ', 'Distance to sLZ', 'Distance to dLZ']

for idx, coloc_feature in enumerate(coloc_features):

    for condition in np.unique(df['Type']):
        dataset = {}
        for group in np.unique(df['Inhibition']):
            data = df[(df['Type'] == condition)&(df['Inhibition'] == group)][coloc_feature]
            dataset[group] = np.array(data)
        # new_order = ['Control', 'IgG', 'CD40L', 'mLT']
        new_order = ['IgG', 'CD40L', 'mLT']
        ordered_dataset = change_dict_order(dataset, new_order)
        dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

        sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

        font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 8}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 1

        colors = ('#888888', '#CC6677', '#6699CC', '#44AA99')
        fig, ax = plt.subplots(figsize=(2,2))
        for i, key in enumerate(dict_datasets):
            sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=ranges[idx], color=colors[i], label=key)

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

        plt.savefig(path+'%s %s.png'%(condition, coloc_feature), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

        if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/')
        plt.savefig(path + 'svg/%s %s.svg'%(condition, coloc_feature), bbox_inches='tight')
        plt.close()
        plt.clf()


############# Plot LZ vs DZ jointplot for each cell type ###############

for cell_type in ['mt_B-cell', 'wt_B-cell']:
    for group in ['Control', 'IgG', 'CD40L', 'mLT']:
        df_part = df[(df['Type']==cell_type)&(df['Inhibition']==group)].reset_index(drop=True)
        df_part = df_part[df_part['Zone']!='DZ-sLZ'].reset_index(drop=True)

        df_part.loc[(df_part['Zone'] == 'DZ'), 'Zone1'] = 'DZ'
        #df_part.loc[(df_part['Zone'] == 'DZ-sLZ'), 'Zone1'] = 'DZ-sLZ'
        df_part.loc[( (df_part['Zone'] == 'sLZ') | (df_part['Zone'] == 'dLZ') | (df_part['Zone'] == 'sLZ-dLZ') ), 'Zone1'] = 'LZ'
        df_part['Zone1'] = df_part.Zone1.astype(str)

        print(cell_type, group, df_part[df_part['Zone1']=='DZ'].shape[0], df_part[df_part['Zone1']=='DZ-sLZ'].shape[0], df_part[df_part['Zone1']=='LZ'].shape[0])

        draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s %s DZ vs LZ jointplot'%(cell_type, group), hue="Zone1", hue_order=['LZ', 'DZ'],
                       colors=('#E69965', '#BAC8DA'), fill=False, legend=False, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')



############# Plot sLZ vs dLZ jointplot for each cell type ###############

for cell_type in ['mt_B-cell', 'wt_B-cell']:
    for group in ['Control', 'IgG', 'CD40L', 'mLT']:
        df_part = df[(df['Type']==cell_type)&(df['Inhibition']==group)].reset_index(drop=True)

        df_part.loc[(df_part['Zone'] == 'sLZ'), 'Zone1'] = 'sLZ'
        #df_part.loc[(df_part['Zone'] == 'sLZ-dLZ'), 'Zone1'] = 'sLZ-dLZ'
        df_part.loc[(df_part['Zone'] == 'dLZ'), 'Zone1'] = 'dLZ'
        df_part['Zone1'] = df_part.Zone1.astype(str)

        print(cell_type, group, df_part[df_part['Zone1']=='sLZ'].shape[0], df_part[df_part['Zone1']=='sLZ-dLZ'].shape[0], df_part[df_part['Zone1']=='dLZ'].shape[0])

        draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s %s sLZ vs dLZ jointplot'%(cell_type, group), hue="Zone1", hue_order=['sLZ', 'dLZ'],
                       colors=('#4F609C', '#8A4F21'), fill=False, legend=False, thresh=0.25, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

############# All Kmeans distribution heatmap ###############

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
df_ = df_[(df_['Zone']!='DZ-sLZ')&(df_['Zone']!='sLZ-dLZ')].reset_index(drop=True)

df_['Type_Zone_Inhibition'] = df_['Type'].astype(str) + ' ' + df_['Zone'].astype(str) + ' ' + df_['Inhibition'].astype(str)
draw_cluster_distribution_heatmap(df_, path, file_name='all_kmeans_heatmap', condition_name='Type_Zone_Inhibition', annot=False,
                                  cluster_type='kmeans', row_cluster=True, col_cluster=False, vmax=30, cmap=cmc.bilbao_r, figsize=(5,8))

############# Inhibit only Kmeans distribution heatmap ###############

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
df_ = df_[(df_['Zone']!='DZ-sLZ')&(df_['Zone']!='sLZ-dLZ')].reset_index(drop=True)
df_ = df_[df_['Inhibition']!='Control'].reset_index(drop=True)

df_['Type_Zone_Inhibition'] = df_['Type'].astype(str) + ' ' + df_['Zone'].astype(str) + ' ' + df_['Inhibition'].astype(str)
draw_cluster_distribution_heatmap(df_, path, file_name='inhibit_kmeans_heatmap', condition_name='Type_Zone_Inhibition', annot=False,
                                  cluster_type='kmeans', row_cluster=True, col_cluster=False, vmax=30, cmap=cmc.bilbao_r, figsize=(5,6))

############# wt GCB Inhibit only Kmeans distribution heatmap ###############

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
df_ = df_[(df_['Zone']!='DZ-sLZ')&(df_['Zone']!='sLZ-dLZ')].reset_index(drop=True)
#df_ = df_[df_['Inhibition']!='Control'].reset_index(drop=True)
df_ = df_[df_['Type']=='mt GCB'].reset_index(drop=True)

df_['Type_Zone_Inhibition'] = df_['Type'].astype(str) + ' ' + df_['Zone'].astype(str) + ' ' + df_['Inhibition'].astype(str)
draw_cluster_distribution_heatmap(df_, path, file_name='mt GCB inhibit_kmeans_heatmap', condition_name='Type_Zone_Inhibition', annot=False,
                                  cluster_type='kmeans', row_cluster=True, col_cluster=False, vmax=30, cmap=cmc.bilbao_r, figsize=(5,5))


###################### Plot Zone motility feature violin plot for mt GCB and wt GCB  ############################

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for cell_type in ['wt_B-cell', 'mt_B-cell']:
    df_part = df[(df['Type'] == cell_type)].reset_index(drop=True)
    if not os.path.isdir(path + '%s Zone motility box plot/'%cell_type):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s Zone motility box plot/'%cell_type)

    for feature_name in feature_list:
        condition_name='Inhibition'
        print(feature_name)
        dataset={}
        for group in ['Control', 'IgG', 'CD40L']:
            for zone in ['DZ','sLZ', 'dLZ']:
        #for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
                data = df_part[(df_part[condition_name] == group)&(df_part['Zone'] == zone)][feature_name]

                dataset[group+' '+str(zone)] = np.array(data)

        # rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_sLZ': 'wt GCB sLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
        #                'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_sLZ': 'mt GCB sLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ',}


        #dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}
        draw_custom_bar_plot(dataset, path + '%s Zone motility box plot/'%cell_type, file_name=feature_name,
                                colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC', '#44AA99', '#44AA99', '#44AA99'),
                                strip_plot=False, test='mann-whitney', pvalue=True, figsize=(3, 3))
        # draw_custom_violin_plot(dataset_renamed, path + 'Zone motility box plot/', file_name=feature_name,
        #                         colors=('#888888', '#888888', '#888888', '#888888', '#888888',
        #                                 '#CC6677', '#CC6677', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))


###################### Plot Experimental motility feature plots for Zones  ############################
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

videos = np.unique(df['Video'])


for cell_type in ['wt_B-cell', 'mt_B-cell']:
    df_part = df[(df['Type'] == cell_type)].reset_index(drop=True)
    if not os.path.isdir(path + '%s experimental_zone_motility_feature/'%cell_type):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s experimental_zone_motility_feature/'%cell_type)

    for feature_name in feature_list:
        dataset = {}
        for group in ['Control', 'IgG', 'CD40L']:
            for zone in ['DZ','sLZ', 'dLZ']:
                df_part_temp = df_part[(df_part['Inhibition'] == group)&(df_part['Zone'] == zone)].reset_index(drop=True)
                avgs = []
                for video in videos:
                    df_video = df_part_temp[df_part_temp['Video'] == video]
                    if df_video.shape[0] == 0:
                        continue
                    data = df_video[feature_name]
                    avg = np.mean(data)
                    avgs.append(avg)
                dataset[group + ' ' + str(zone)] = avgs

        #new_order = ['wt_B-cell', 'wt_B-cell']
        #ordered_dataset = change_dict_order(dataset, new_order)
        # rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_sLZ': 'wt GCB sLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
        #                'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_sLZ': 'mt GCB sLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ', }
        # dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

        draw_custom_bar_plot(dataset, path+'%s experimental_zone_motility_feature/'%cell_type, file_name=feature_name,
                             colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC', '#44AA99', '#44AA99', '#44AA99'),
                             strip_plot=True,test='mann-whitney', pvalue=True, figsize=(3, 3))

###################### Plot Zone interaction feature bar plot for mt GCB and wt GCB  ############################

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone', 'Inhibition'])

for cell_type in ['wt_B-cell', 'mt_B-cell']:
    df_part = df[(df['Type'] == cell_type)].reset_index(drop=True)
    if not os.path.isdir(path + '%s Zone int box plot/'%cell_type):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s Zone int box plot/'%cell_type)

    for feature_name in feature_list:
        condition_name='Inhibition'
        dataset={}
        for group in ['IgG', 'CD40L', 'mLT']:
            for zone in ['DZ','sLZ', 'dLZ']:
        #for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
                data = df_part[(df_part[condition_name] == group)&(df_part['Zone'] == zone)][feature_name]

                dataset[group+' '+str(zone)] = np.array(data)

        # rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_sLZ': 'wt GCB sLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
        #                'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_sLZ': 'mt GCB sLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ',}


        #dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}
        draw_custom_bar_plot(dataset, path + '%s Zone int box plot/'%cell_type, file_name=feature_name,
                                colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC', '#44AA99', '#44AA99', '#44AA99'),
                                strip_plot=False, test='mann-whitney', pvalue=True, figsize=(3, 3))
        # draw_custom_violin_plot(dataset_renamed, path + 'Zone motility box plot/', file_name=feature_name,
        #                         colors=('#888888', '#888888', '#888888', '#888888', '#888888',
        #                                 '#CC6677', '#CC6677', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))


###################### Plot Experimental interaction feature plots for Zones  ############################
df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone', 'Inhibition'])

videos = np.unique(df['Video'])


for cell_type in ['wt_B-cell', 'mt_B-cell']:
    df_part = df[(df['Type'] == cell_type)].reset_index(drop=True)
    if not os.path.isdir(path + '%s experimental_zone_int_feature/'%cell_type):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s experimental_zone_int_feature/'%cell_type)

    for feature_name in feature_list:
        dataset = {}
        for group in ['Control', 'IgG', 'CD40L']:
            for zone in ['DZ','sLZ', 'dLZ']:
                df_part_temp = df_part[(df_part['Inhibition'] == group)&(df_part['Zone'] == zone)].reset_index(drop=True)
                avgs = []
                for video in videos:
                    df_video = df_part_temp[df_part_temp['Video'] == video]
                    if df_video.shape[0] == 0:
                        continue
                    data = df_video[feature_name]
                    avg = np.mean(data)
                    avgs.append(avg)
                dataset[group + ' ' + str(zone)] = avgs

        #new_order = ['wt_B-cell', 'wt_B-cell']
        #ordered_dataset = change_dict_order(dataset, new_order)
        # rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_sLZ': 'wt GCB sLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
        #                'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_sLZ': 'mt GCB sLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ', }
        # dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

        draw_custom_bar_plot(dataset, path+'%s experimental_zone_int_feature/'%cell_type, file_name=feature_name,
                             colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC', '#44AA99', '#44AA99', '#44AA99'),
                             strip_plot=True,test='mann-whitney', pvalue=True, figsize=(3, 3))


#################################### all motility features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
df_.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df_.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for typ in ['wt GCB', 'mt GCB']:
    df_part_ = df_[df_['Type']==typ].reset_index(drop=True)
    coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']

    for idx in [0, 1, 2]:
        coloc_feature = coloc_features[idx]
        if idx == 0:
            xlabel = 'Distance to DZ (µm)'
            custsom_range = (0, 40)
            stepsize = 4
        elif idx == 1:
            xlabel = 'Distance to sLZ (µm)'
            custsom_range = (0, 32)
            stepsize = 8
        elif idx == 2:
            xlabel = 'Distance to dLZ (µm)'
            custsom_range = (0, 40)
            stepsize = 4

        draw_lineplot_by_custom_ranges(df_part_, path, folder_name='%s motility_feature_wrt_%s'%(typ, coloc_feature), feature_list=feature_list,
                                       condition_name='Inhibition', custsom_range=custsom_range, stepsize=stepsize, range_feature=coloc_feature,
                                           color_list=['#44AA99', '#CC6677', '#6699CC'], marker_list=['o', '^', '.'], figsize=(4,4), x_label=xlabel,
                                       replace_keys=None, pvalue=True, test='mann-whitney')


#################################### all interaction features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB'}})
df_.columns.get_loc('quality_FDC_approach_times')
feature_list = df_.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone', 'Inhibition'])

for typ in ['wt GCB', 'mt GCB']:
    df_part_ = df_[df_['Type']==typ].reset_index(drop=True)
    coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']
    for idx in [0, 1, 2]:
        coloc_feature = coloc_features[idx]
        if idx == 0:
            xlabel = 'Distance to DZ (µm)'
            custsom_range = (0, 40)
            stepsize = 4
        elif idx == 1:
            xlabel = 'Distance to sLZ (µm)'
            custsom_range = (0, 32)
            stepsize = 8
        elif idx == 2:
            xlabel = 'Distance to dLZ (µm)'
            custsom_range = (0, 40)
            stepsize = 4

        draw_lineplot_by_custom_ranges(df_part_, path, folder_name='%s int_feature_wrt_%s'%(typ, coloc_feature), feature_list=feature_list,
                                       condition_name='Inhibition', custsom_range=custsom_range, stepsize=stepsize, range_feature=coloc_feature,
                                           color_list=['#44AA99', '#CC6677', '#6699CC'], marker_list=['o', '^', '.'], figsize=(4,4), x_label=xlabel,
                                       replace_keys=None, pvalue=True, test='mann-whitney')