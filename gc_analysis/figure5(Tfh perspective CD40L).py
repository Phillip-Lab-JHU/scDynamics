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
"""Generates Data for Figure5-1. Inhibition analysis"""

from utils.draw_utils import *
from utils.misc_utils import *
from utils.traj_utils import to_timeseries_fast
from features.interaction import DistanceSignal, OverlapSignal, ZoneSignal

duration=20
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis_test\\'
df_features_all = pd.read_parquet(path+'all_features_%s.parquet'%duration)
df_duration_all = pd.read_parquet(path+'traj_duration_%s.parquet'%duration)

for cell_type in ['wt_B-cell', 'mt_B-cell', 'T-cell']:
    volume = np.mean(df_duration_all[df_duration_all['Type'] == cell_type]['Volume'])
    r = ((3*volume)/(4*np.pi))**(1/3)
    print(cell_type, volume, r)


filtered_arr = get_filtered_string_list(arr=np.unique(df_features_all['Video']), keywords=['-A', 'DenseTfh'], filter_type='or')

df_features_all = df_features_all[~df_features_all['Video'].isin(filtered_arr)]
df_duration_all = df_duration_all[~df_duration_all['Video'].isin(filtered_arr)]


df = df_features_all[df_features_all['Type']=='T-cell'].reset_index(drop=True)
df_duration = df_duration_all[df_duration_all['Type']=='T-cell'].reset_index(drop=True)

df_duration = get_instant_direction(df_duration, duration=duration, thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=wt_B-cell')
df_duration = get_instant_direction(df_duration, duration=duration, thresh=0.1, feature_name='Shortest_Distance_to_Surfaces_Surfaces=mt_B-cell')

############################ Calculate WT B cell features ############################
_, _, WT_distances = to_timeseries_fast(df_duration, duration, feature_name='Shortest_Distance_to_Surfaces_Surfaces=wt_B-cell')
_, _, WT_overlap = to_timeseries_fast(df_duration, duration, feature_name='Overlapped_Volume_Ratio_to_Surfaces_Surfaces=wt_B-cell')

df_duration_diff = df_duration[(df_duration['pseudo_Time']!=0)].reset_index(drop=True)
_, _, WT_diff_distances = to_timeseries_fast(df_duration_diff, duration=duration-1, feature_name='diff_Shortest_Distance_to_Surfaces_Surfaces=wt_B-cell')

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average', 'autocorr',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr'
                ]
WT_dist = DistanceSignal(WT_distances)
df_distance = WT_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)

WT_diff_dist = DistanceSignal(WT_diff_distances)
df_diff_distance = WT_diff_dist.extract_features(feature_list, tau_limit=3)
for column in df_diff_distance.columns:
    df_diff_distance.rename(columns={column:'diff_distance_'+column}, inplace=True)

feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
WT_over = OverlapSignal(WT_overlap)
df_overlap = WT_over.extract_features(feature_list)

df_inter_WT = pd.concat([df_distance, df_diff_distance, df_overlap], axis=1)
for column in df_inter_WT.columns:
    df_inter_WT.rename(columns={column:'WT_'+column}, inplace=True)


############################ Calculate MT B cell features ############################
_, _, MT_distances = to_timeseries_fast(df_duration, duration, feature_name='Shortest_Distance_to_Surfaces_Surfaces=mt_B-cell')
_, _, MT_overlap = to_timeseries_fast(df_duration, duration, feature_name='Overlapped_Volume_Ratio_to_Surfaces_Surfaces=mt_B-cell')

df_duration_diff = df_duration[(df_duration['pseudo_Time']!=0)].reset_index(drop=True)
_, _, MT_diff_distances = to_timeseries_fast(df_duration_diff, duration=duration-1, feature_name='diff_Shortest_Distance_to_Surfaces_Surfaces=mt_B-cell')

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average', 'autocorr',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr'
                ]
MT_dist = DistanceSignal(MT_distances)
df_distance = MT_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)

MT_diff_dist = DistanceSignal(MT_diff_distances)
df_diff_distance = MT_diff_dist.extract_features(feature_list, tau_limit=3)
for column in df_diff_distance.columns:
    df_diff_distance.rename(columns={column:'diff_distance_'+column}, inplace=True)

feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
MT_over = OverlapSignal(MT_overlap)
df_overlap = MT_over.extract_features(feature_list)

df_inter_MT = pd.concat([df_distance, df_diff_distance, df_overlap], axis=1)
for column in df_inter_MT.columns:
    df_inter_MT.rename(columns={column:'MT_'+column}, inplace=True)


df = pd.concat([df, df_inter_WT, df_inter_MT], axis=1)

df_inhibit = df[(df['Exp']=='IgG')|(df['Exp']=='CD40L')].reset_index(drop=True)
df_duration_inhibit = df_duration[(df_duration['Exp']=='IgG')|(df_duration['Exp']=='CD40L')].reset_index(drop=True)
df_inhibit['Inhibition'] = df_inhibit['Exp'].copy()
df_duration_inhibit['Inhibition'] = df_duration_inhibit['Exp'].copy()


df['Inhibition'] = df['Exp'].copy()
df['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)
df = df[(df['Exp']!='mLT')].reset_index(drop=True)

df_duration['Inhibition'] = df_duration['Exp'].copy()
df_duration['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)
df_duration = df_duration[(df_duration['Exp']!='mLT')].reset_index(drop=True)

for i in np.unique(df['Type']):
    for j in np.unique(df['Inhibition']):
        print(i, j, df[(df['Type']==i)&(df['Inhibition']==j)].shape[0])

for i in np.unique(df['Type']):
    for j in np.unique(df['Inhibition']):
        print(i, j, np.unique(df[(df['Type']==i)&(df['Inhibition']==j)]['Video']).shape[0])

def remove_alphabets(s):
    return ''.join([char for char in s if not char.isalpha()])

def remove_LR(s):
    return ''.join([char for char in s if char not in 'LRlr'])

for video in np.unique(df['Video']):
    first = video.find('-')
    second = video.find('-', first + 1)
    third = video.find('-', second + 1)
    fourth = video.find('-', third + 1)
    mouse_name = remove_alphabets(video[:first]) + video[second:third] + remove_LR(video[third:fourth])
    print(video, mouse_name)

    df.loc[(df['Video'] == video), 'Mouse'] = mouse_name

for video in np.unique(df_inhibit['Video']):
    first = video.find('-')
    second = video.find('-', first + 1)
    third = video.find('-', second + 1)
    fourth = video.find('-', third + 1)
    mouse_name = remove_alphabets(video[:first]) + video[second:third] + remove_LR(video[third:fourth])
    print(video, mouse_name)

    df_inhibit.loc[(df_inhibit['Video'] == video), 'Mouse'] = mouse_name

#df = df[df['Mouse']!='20240128-D10-B2'].reset_index(drop=True)


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis_test\Figure5. Inhibition\Tfh perspective\\'

####################################### Quantify B cell interaction frequency #############################################
int_time = 15
test = 'mann-whitney'

df_int = pd.DataFrame()
videos = np.unique(df['Video'])
for video in videos:
    df_video = df[df['Video']==video].reset_index(drop=True)

    df_temp = pd.DataFrame()
    for interaction_type in ['WT', 'MT']:
        data = df_video['%s_contact_times'%interaction_type]
        mask = ~np.isnan(data)
        data = data[mask]
        persistent_int_freq = sum(data >= int_time)
        total_n_contact = sum(data)
        n_cells = df_video.shape[0]

        df_temp_temp = pd.DataFrame()
        df_temp_temp['Type'] = [interaction_type]
        df_temp_temp['Video'] = [video]
        df_temp_temp['Inhibition'] = df_video['Inhibition'][0]
        df_temp_temp['Mouse'] = df_video['Mouse'][0]
        df_temp_temp['persistent_int'] = [persistent_int_freq / n_cells]
        df_temp_temp['total_int'] = [total_n_contact / n_cells]
        df_temp_temp['low_int'] = [sum((1 <= data) & (data <= 5)) / n_cells]

        df_temp = pd.concat([df_temp, df_temp_temp], axis=0, ignore_index=True)

    df_int = pd.concat([df_int, df_temp], axis=0, ignore_index=True)


df_int = df_int[df_int['Mouse']!='20240128-D10-B2'].reset_index(drop=True)


feature = 'persistent_int'
file_name='%s_per experiment'%feature
draw_double_bar_plot(df_int, path, file_name=file_name, condition_name='Inhibition', conditions=['IgG', 'CD40L'],
                     category_name='Type', categories=['WT', 'MT'], y=feature, other_category=None,
                     other_category_colors=None, estimator='mean', error_type='std', condition_colors=('#888888', '#CC6677'),
                     test='t-test', figsize=(2,2))


####################################### Quantify B cell interaction frequency #############################################
test = 'mann-whitney'

df_int = pd.DataFrame()
videos = np.unique(df['Video'])
feature = 'contact_persistences' # 'contact_persistences', 'contact_times', 'avg_overlap'
for video in videos:
    df_video = df[df['Video']==video].reset_index(drop=True)

    data_WT = df_video['WT_%s'%feature]
    data_MT = df_video['MT_%s'%feature]
    diff_persistence = data_WT - data_MT
    WT_favor_cells = np.sum(diff_persistence>0)
    n_cells = np.sum(diff_persistence!=0)
    #n_cells = df_video.shape[0]
    df_temp = pd.DataFrame()
    df_temp['diff_%s'%feature] = [WT_favor_cells / n_cells]
    df_temp['Video'] = [video]
    df_temp['Inhibition'] = df_video['Inhibition'][0]
    df_temp['Mouse'] = df_video['Mouse'][0]
    print(video, df_video['Inhibition'][0], df_video.shape[0])
    df_int = pd.concat([df_int, df_temp], axis=0, ignore_index=True)

df_int = df_int[df_int['Mouse']!='20240128-D10-B2'].reset_index(drop=True)

feature_name = 'diff_%s'%feature
dataset={}
#for group in ['IgG', 'CD40L']:
for group in ['Control', 'IgG', 'CD40L']:
    data = df_int[(df_int['Inhibition'] == group)][feature_name]
    dataset[group] = np.array(data)


file_name='%s_per experiment'%feature_name
draw_custom_bar_plot(dataset, path, file_name='%s'%file_name,
                     strip_plot=True, colors=('#888888', '#888888','#888888'), test='mann-whitney', pvalue=True, figsize=(1,2))





######################## interaction features mt vs WT box plot  ###########################
if not os.path.isdir(path + 'Inhibit int feature violin plot_del_mouse/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Inhibit int feature violin plot_del_mouse/')

df.columns.get_loc('quality_FDC_approach_times')
feature_list = ['contact_persistences', 'contact_times', 'avg_overlap']

df_ = df.copy()

for feature_name in feature_list:
    dataset={}
    # for group in ['IgG', 'CD40L']:
    for group in ['Control', 'IgG', 'CD40L']:
        for cell_type in ['WT', 'MT']:
            data = df_[(df_['Inhibition'] == group)]['%s_%s'%(cell_type, feature_name)]
            dataset[str(group)+ ' ' + cell_type] = np.array(data)

    values = flatten_nested_dict(dataset)
    if np.isnan(values).any() == True:  # Check at least one nan
        continue
    elif np.isfinite(values).all() == False:  # Check everything is not inf
        continue

    # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
    #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ', }
    # dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}


    # draw_custom_violin_plot(dataset, path + 'Inhibit int feature violin plot_del_mouse/', file_name=feature_name,
    #                         #colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'),
    #                         colors=('#888888', '#888888', '#CC6677', '#CC6677',),
    #                         test='mann-whitney', pvalue=True, figsize=(2, 2))
    # draw_custom_box_plot(dataset, path + 'Inhibit int feature violin plot_del_mouse/', file_name=feature_name,
    #                      colors=('#888888', '#888888', '#CC6677', '#CC6677',),
    #                      strip_plot=False, test='mann-whitney', pvalue=True, figsize=(2, 2))

    draw_custom_bar_plot(dataset, path + 'Inhibit int feature violin plot_del_mouse/', file_name=feature_name,
                         strip_plot=False,
                         colors=('#888888', '#CC6677')*3,
                         #colors=('#888888', '#888888', '#CC6677', '#CC6677',),
                         test='mann-whitney', pvalue=True, figsize=(2, 2))

###################### Plot Experimental int feature plots for Exp  ############################
videos = np.unique(df['Video'])

if not os.path.isdir(path + 'experimental_inhibit_int_feature_del_mouse/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_inhibit_int_feature_del_mouse/')


df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT GC B cell', 'mt_B-cell': 'MT GC B cell'}})

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Inhibition', 'Mouse'])

for feature_name in feature_list:
    dataset = {}
    for cell_type in ['WT GC B cell', 'MT GC B cell']:
        #for group in ['Control','IgG', 'CD40L']:
        for group in ['IgG', 'CD40L']:
            df_part = df_[(df_['Type'] == cell_type)&(df_['Inhibition'] == group)].reset_index(drop=True)
            avgs = []
            for video in videos:
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] == 0:
                    continue
                data = df_video[feature_name]
                avg = np.mean(data)
                avgs.append(avg)
            dataset[cell_type + ' ' + str(group)] = avgs
    #new_order = ['wt_B-cell', 'wt_B-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    #dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_bar_plot(dataset, path+'experimental_inhibit_int_feature_del_mouse/', file_name=feature_name,
                         strip_plot=True,
                         #colors=('#888888', '#888888', '#888888',  '#CC6677', '#CC6677', '#CC6677'),
                         colors=('#888888', '#888888', '#CC6677', '#CC6677',),
                         test='mann-whitney', pvalue=True, figsize=(2, 2))







#################################### Motility ####################################
#################################### Plot whole state space ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_inhibit_ = df_inhibit.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')

xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

for typ in np.unique(df_['Type']):
    df_part = df_[df_['Type']==typ].reset_index(drop=True)
    df_part_inhibit = df_inhibit_[df_inhibit_['Type'] == typ].reset_index(drop=True)
    print(typ, df_part_inhibit.shape[0])
    # draw_umap_space(df_part, path, file_name='%s space_exp'%typ, condition_name='Inhibition', label_name='pseudo_Label', colors=color_list, x_name='PC1', y_name='PC2', dot_size=0.07)
    # draw_umap_space(df_part_inhibit, path, file_name='%s inhibit_space_exp'%typ, condition_name='Exp', label_name='pseudo_Label', colors=color_list, x_name='PC1', y_name='PC2', dot_size=0.07)

    draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s jointplot_exp'%typ, hue="Inhibition", colors = ('#888888', '#CC6677', '#6699CC', '#44AA99'),
                   hue_order=['Control', 'IgG', 'CD40L', 'mLT'], legend=True, fill=False, thresh=0.2, height=4, ratio=5, space=0, n_contours=3,
                   xlabels='UMAP1', ylabel='UMAP2', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    # draw_jointplot(xs='PC1', y='PC2', df=df_part_inhibit, path=path, file_name='%s inhibit_jointplot_exp'%typ, hue="Exp", colors=color_list,
    #                hue_order=['IgG', 'CD40L', 'mLT'], legend=True, fill=False, thresh=0.2, height=4, ratio=5, space=0,
    #                xlabels='UMAP1', ylabel='UMAP2', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)



df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_['type_exp'] = df_['Type'].astype(str) + ' ' + df_['Exp'].astype(str)
df_['type_inhibition'] = df_['Type'].astype(str) + ' ' + df_['Inhibition'].astype(str)

draw_cluster_distribution_heatmap(df_, path, file_name='exp_kmeans_heatmap', condition_name='type_exp', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,4))
draw_cluster_distribution_heatmap(df_, path, file_name='inhibition_kmeans_heatmap', condition_name='type_inhibition', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,3))

df_inhibit_ = df_inhibit.copy()
df_inhibit_ = df_inhibit_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_inhibit_['type_exp'] = df_inhibit_['Type'].astype(str) + ' ' + df_inhibit_['Exp'].astype(str)

draw_cluster_distribution_heatmap(df_inhibit_, path, file_name='no control inhibition_kmeans_heatmap', condition_name='type_exp', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,2))

draw_relative_cluster_distribution_heatmap(df_inhibit_, path, file_name='relative_no control inhibition_kmeans_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=40, cmap=cmc.oslo_r, transpose=False, condition_name='type_exp', cluster_type='kmeans', figsize=(4,2))


group_name = 'Video'
groups = np.unique(df[group_name])

entropies_control = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Inhibition'] == 'Control')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_control[type].append(entropy[type])

entropies_igg = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Inhibition'] == 'IgG')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_igg[type].append(entropy[type])

entropies_cd40l = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Inhibition'] == 'CD40L')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_cd40l[type].append(entropy[type])

entropies_mlt = {'mt_B-cell': [], 'wt_B-cell':[]}
for group in groups:
    df_part = df[(df[group_name]==group)&(df['Inhibition'] == 'mLT')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_mlt[type].append(entropy[type])


entropies = {}
for cell_type in ['mt_B-cell', 'wt_B-cell']:
    entropies = {
        '%s Control'%cell_type:entropies_control[cell_type],
                 '%s IgG'%cell_type:entropies_igg[cell_type], '%s CD40L'%cell_type:entropies_cd40l[cell_type],
        '%s mLT' % cell_type: entropies_mlt[cell_type],
                  }

    # entropies = {'%s_DZ' % cell_type: entropies_dz[cell_type], '%s_sLZ' % cell_type: entropies_slz[cell_type],
    #              '%s_dLZ' % cell_type: entropies_dlz[cell_type], }

    # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_DZ-sLZ': 'WT DZ-sLZ', 'wt_B-cell_sLZ': 'WT sLZ',
    #                'wt_B-cell_sLZ-dLZ': 'WT sLZ-dLZ', 'wt_B-cell_dLZ': 'WT dLZ',
    #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_DZ-sLZ': 'MT DZ-sLZ', 'mt_B-cell_sLZ': 'MT sLZ',
    #                 'mt_B-cell_sLZ-dLZ': 'MT sLZ-dLZ', 'mt_B-cell_dLZ': 'MT dLZ',}
    # entropies_ = {rename_keys.get(k, k): v for (k, v) in entropies.items()}
    # draw_custom_bar_plot(entropies_, path, file_name='entropy of DZ vs sLZ vs dLZ for %s' %cell_type, colors=('#888888', '#CC6677', '#6699CC'),
    #                      strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))
    dict_datasets = entropies
    file_name = 'entropy of IgG vs CD40L for %s' %cell_type
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
    # plt.xticks(plt.xticks()[0], sorted_keys, fontsize=12, fontdict={'weight': 'normal'})
    # ax.set_xticks(plt.xticks()[0], sorted_keys, fontsize=4)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Shannon entropy', fontsize=8, weight='normal', color='0.2')
    plt.xticks(plt.xticks()[0], sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
               weight='normal')
    plt.yticks(fontsize=8, color='0.2', weight='normal')
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

############# Plot type+day kmeans cross correlation for each cell type ###############
df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_['type_inhibition'] = df_['Type'].astype(str) + ' ' + df_['Inhibition'].astype(str)

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

condition_name = 'type_inhibition'
cluster_type = 'kmeans'

df_corr_data = pd.DataFrame()
group_clones=[]
for group in ['WT Control', 'WT IgG', 'WT CD40L', 'MT Control', 'MT IgG', 'MT CD40L']:
#for group in ['DZ', 'sLZ', 'dLZ']:
    corrcoef = []
    aaa = df_[df_['type_inhibition'] == group].reset_index(drop=True)

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
    group_clones.append(group_clone)
    df_corr_data = pd.concat([df_corr_data, group_clone], axis=1)

df_corr = df_corr_data.corr(method='spearman')

df_corr.columns.name = None # Remove name of column = 'Type'
df_corr.index.name = None # Remove name of index = 'Type'


mask = np.triu(df_corr) # Mask for only lower triangle

mask = mask[1:, :-1]  # mask that doesn't contain very top left and very bottom right block
corr = df_corr.iloc[1:,:-1].copy()  # df that doesn't contain very top left and very bottom right block

fig, ax = plt.subplots(figsize=(2, 2))

ax = sns.heatmap(corr, annot=True, mask=mask, cmap='OrRd', alpha=0.9, linewidths=1, linecolor='white', fmt=".2f", square= True,
                 annot_kws={'size': 4, 'weight':'normal'},
                 cbar_kws= {"shrink":0.7, 'label':'Correlation'})

cax = ax.figure.axes[-1]  # colorbar
cax.tick_params(labelsize=4)  # fontsize of tick label
cax.yaxis.label.set_size(6)  # fontsize of color bar y label

yticks = [i for i in corr.index]
xticks = [i for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=4, weight='normal')
plt.xticks(plt.xticks()[0], labels=xticks, fontsize=4, rotation=35, rotation_mode='anchor', ha='right', weight='normal')

plt.savefig(path+'inhibition correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/inhibition correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

#################################### Box plot comparing all motility features by cell types ####################################
if not os.path.isdir(path + 'feature_violin_plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot/')

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['WT', 'MT']:
        for group in ['Control','IgG', 'CD40L']:
        #for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
            data = df_[(df_[condition_name] == cell_type)&(df_['Inhibition'] == group)][feature_name]

            dataset[cell_type+' '+str(group)] = np.array(data)

    #dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_violin_plot(dataset, path + 'feature_violin_plot/', file_name=feature_name,
                            colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))

###################### Plot Experimental motility feature plots for Exp  ############################
videos = np.unique(df['Video'])

if not os.path.isdir(path + 'experimental_inhibit_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_inhibit_motility_feature/')

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    dataset = {}
    for cell_type in ['WT', 'MT']:
        for group in ['Control','IgG', 'CD40L', 'mLT']:
            df_part = df_[(df_['Type'] == cell_type)&(df_['Inhibition'] == group)].reset_index(drop=True)
            avgs = []
            for video in videos:
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] == 0:
                    continue
                data = df_video[feature_name]
                avg = np.mean(data)
                avgs.append(avg)
            dataset[cell_type + ' ' + str(group)] = avgs
    #new_order = ['wt_B-cell', 'wt_B-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    #dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_bar_plot(dataset, path+'experimental_inhibit_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=('#888888', '#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))



#################################### Volcano plot of all motility features ####################################
# feature_list = df_inhibit.columns[128:284].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
# cell_type = 'T-cell'
# df_part = df[df['Inhibition']!='Control'].reset_index(drop=True)
# df_part_ = df_part[df_part['Type']==cell_type]
#
#
# df_p = pd.DataFrame()
# for feature_name in feature_list:
#     dataset = {}
#     for condition in np.unique(df_part_['Inhibition']):
#         data = df_part_[df_part_['Inhibition'] == condition][feature_name]
#         dataset[condition] = np.array(data)
#
#     pvalue = get_pvalue(dataset, test='mann-whitney')
#     logp = -np.log10(pvalue)
#
#     avgZ = get_avgZ(dataset, ref_name='ControlAb', data_name='CD40LAb')
#
#     row = pd.DataFrame()
#     row['Feature'] = [feature_name]
#     row['Pvalue'] = [pvalue]
#     row['-Logp'] = [logp]
#     row['AvgZ'] = [avgZ]
#     df_p = pd.concat([df_p, row], axis=0)
#
# df_p['Adj_p'] = adjusted_pvalues(df_p['Pvalue'], correction_type='Benjamini-Hochberg')
# df_p['Adj_Logp'] = -np.log(df_p['Adj_p'])
# df_p = df_p[~df_p.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf
#
#
# draw_volcano_plot(df_p, path, file_name='motility volcano plot_%s'%cell_type, z_thresh=0.5, p_thresh=2, z_name='AvgZ', p_name='Adj_Logp',
#                   feature_name='Feature', figsize=(6,6))

