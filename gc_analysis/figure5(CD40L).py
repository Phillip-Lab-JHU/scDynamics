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
import matplotlib.pyplot as plt

from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis_test\feature_csvs\\'
df = pd.read_parquet(path+'GCB_with_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_with_inhibit_traj_duration_20.parquet')

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

df = df[df['Mouse']!='20240128-D10-B2'].reset_index(drop=True)

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis_test\Figure5. Inhibition\CD40L\\'

####################################### Quantify Tfh interaction frequency #############################################
int_time = 12
test = 'mann-whitney'

df_int = pd.DataFrame()
videos = np.unique(df_inhibit['Video'])
for video in videos:
    df_video = df_inhibit[df_inhibit['Video']==video].reset_index(drop=True)

    df_temp = pd.DataFrame()
    for cell_type in ['wt_B-cell', 'mt_B-cell']:
        df_part = df_video[df_video['Type'] == cell_type].reset_index(drop=True)
        if df_part.shape[0] <= 30:
            continue
        data = df_part['T_contact_times']
        mask = ~np.isnan(data)
        data = data[mask]
        persistent_int_freq = sum(data >= int_time)
        total_n_contact = sum(data)
        n_cells = df_part.shape[0]
        print(video, cell_type, n_cells)
        df_temp_temp = pd.DataFrame()
        df_temp_temp['Type'] = [cell_type]
        df_temp_temp['Video'] = [video]
        df_temp_temp['Inhibition'] = df_video['Inhibition'][0]
        df_temp_temp['Mouse'] = df_video['Mouse'][0]
        df_temp_temp['persistent_int'] = [persistent_int_freq / n_cells]
        df_temp_temp['total_int'] = [total_n_contact / n_cells]
        df_temp_temp['low_int'] = [sum((1 <= data) & (data <= 5)) / n_cells]

        df_temp = pd.concat([df_temp, df_temp_temp], axis=0, ignore_index=True)

    df_int = pd.concat([df_int, df_temp], axis=0, ignore_index=True)


df_int = df_int[df_int['Mouse']!='20240128-D10-B2'].reset_index(drop=True)


df_int_pairwise = pd.DataFrame()
for mouse in np.unique(df_int['Mouse']):
    df_part = df_int[df_int['Mouse']==mouse]
    bool_cond = np.all( np.isin( ['IgG', 'CD40L'], np.unique(df_part['Inhibition']) ) )
    if bool_cond == True:
        df_int_pairwise = pd.concat([df_int_pairwise, df_part], axis=0, ignore_index=True)



feature_name = 'persistent_int'
dataset={}
for cell_type in ['wt_B-cell', 'mt_B-cell']:
#cell_type = 'wt_B-cell'
#for group in ['Control', 'IgG', 'CD40L']:
    for group in ['IgG', 'CD40L']:
        data = df_int[(df_int['Inhibition'] == group)&(df_int['Type'] == cell_type)][feature_name]
        dataset[cell_type + ' ' + group] = np.array(data)

file_name='bar plot_%s_9min'%feature_name
draw_custom_bar_plot(dataset, path, file_name='%s'%file_name,
                     strip_plot=True,
                     #colors=('#888888', '#888888','#888888', '#CC6677', '#CC6677', '#CC6677'),
                     colors=('#888888', '#888888', '#CC6677', '#CC6677'),
                     test='mann-whitney', pvalue=True, figsize=(1,2))


feature = 'total_int'
file_name='pairwise_%s__6min_per experiment'%feature
draw_double_bar_plot(df_int_pairwise, path, file_name=file_name, condition_name='Inhibition', conditions=['IgG', 'CD40L'],
                     category_name='Type', categories=['wt_B-cell', 'mt_B-cell'], y=feature, other_category='Mouse',
                     other_category_colors=('red', 'green', 'blue', 'purple'), estimator='mean', error_type='std', condition_colors=('#888888', '#CC6677'),
                     test='t-test', figsize=(2,2))



####################################### Quantify Tfh interaction frequency #############################################
int_time = 19
test = 'mann-whitney'

persistent_int_freqs_datasets = {}
persistent_int_freq_per_cellnumbers_datasets = {}
total_n_contacts_per_cellnumbers_datasets = {}
persistent_int_freq_per_cellcontacts_datasets = {}
low_contact_freq_per_cellnumbers_datasets = {}

cell_type = 'wt_B-cell'
videos = np.unique(df['Video'])
for exp in ['Control', 'IgG', 'CD40L']:
    df_part = df[(df['Inhibition'] == exp)&(df['Type']==cell_type)]
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

        persistent_int_freq = sum(data == int_time)
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

    persistent_int_freqs_datasets[exp] = persistent_int_freqs
    persistent_int_freq_per_cellnumbers_datasets[exp] = persistent_int_freq_per_cellnumbers
    total_n_contacts_per_cellnumbers_datasets[exp] = total_n_contacts_per_cellnumbers
    persistent_int_freq_per_cellcontacts_datasets[exp] = persistent_int_freq_per_cellcontacts
    low_contact_freq_per_cellnumbers_datasets[exp] = low_contact_freq_per_cellnumbers

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
new_order = ['Control', 'IgG', 'CD40L']
persistent_int_freqs_datasets = change_dict_order(persistent_int_freqs_datasets, new_order)
#persistent_int_freqs_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freqs_datasets.items() }

persistent_int_freq_per_cellnumbers_datasets = change_dict_order(persistent_int_freq_per_cellnumbers_datasets, new_order)
#persistent_int_freq_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freq_per_cellnumbers_datasets.items() }

total_n_contacts_per_cellnumbers_datasets = change_dict_order(total_n_contacts_per_cellnumbers_datasets, new_order)
#total_n_contacts_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in total_n_contacts_per_cellnumbers_datasets.items() }

persistent_int_freq_per_cellcontacts_datasets = change_dict_order(persistent_int_freq_per_cellcontacts_datasets, new_order)
#persistent_int_freq_per_cellcontacts_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freq_per_cellcontacts_datasets.items() }

low_contact_freq_per_cellnumbers_datasets = change_dict_order(low_contact_freq_per_cellnumbers_datasets, new_order)
#low_contact_freq_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in low_contact_freq_per_cellnumbers_datasets.items() }

colors=('#888888', '#888888', '#888888')
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




df = df_int
file_name = 'test'
condition_name = 'Inhibition'
conditions = ['IgG', 'CD40L']
category_name = 'Type'
categories = ['wt_B-cell', 'mt_B-cell']
y = 'persistent_int'
other_category='Mouse'
estimator='mean'
error_type='std'
condition_colors=('#888888', '#CC6677')
test='mann-whitney'
figsize=(2,2)

bar_width = 0.3  # Width of bars

scatter1 = []
scatter2 = []
statistics1 = []
statistics2 = []
error1 = []
error2 = []

p_values = []
cohens_ds = []

other_categories1 = []
other_categories2 = []
for category in categories:
    df_part = df[df[category_name]==category].reset_index(drop=True)
    values1 = df_part[df_part[condition_name]==conditions[0]][y].values
    values2 = df_part[df_part[condition_name]==conditions[1]][y].values
    scatter1.append(values1)
    scatter2.append(values2)

    if other_category != None:
        other_category1 = df_part[df_part[condition_name]==conditions[0]][other_category].values
        other_category2 = df_part[df_part[condition_name] == conditions[1]][other_category].values
        other_categories1.append(other_category1)
        other_categories2.append(other_category2)

    stattest_dataset = {}
    stattest_dataset[conditions[0]] = values1
    stattest_dataset[conditions[1]] = values2
    _, p_value, cohens_d = get_various_statistics(stattest_dataset, test=test)
    p_values.append(p_value[0])
    cohens_ds.append(cohens_d[0])
    if estimator == 'mean':
        stats1, stats2 = np.mean(values1), np.mean(values2)
    elif estimator == 'median':
        stats1, stats2 = np.median(values1), np.median(values2)
    statistics1.append(stats1)
    statistics2.append(stats2)


    if error_type == 'std':
        err1, err2 = np.std(values1), np.std(values2)
    elif error_type == 'sem':
        err1, err2 = stats.sem(values1), stats.sem(values2)
    elif error_type == 'ci_norm':
        interval1 = stats.norm.interval(confidence=0.95, loc=np.mean(values1), scale=stats.sem(values1))
        interval2 = stats.norm.interval(confidence=0.95, loc=np.mean(values2), scale=stats.sem(values2))
        err1 = np.mean(values1) - interval1[0]
        err2 = np.mean(values2) - interval2[0]
    elif error_type == 'ci_t':
        interval1 = stats.t.interval(confidence=0.95, df=values1.size - 1, loc=np.mean(values1), scale=stats.sem(values1))
        interval2 = stats.t.interval(confidence=0.95, df=values2.size - 1, loc=np.mean(values2), scale=stats.sem(values2))
        err1 = np.mean(values1) - interval1[0]
        err2 = np.mean(values2) - interval2[0]
    error1.append(err1)
    error2.append(err2)

# X-axis positions
x = np.arange(len(categories))

font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=figsize)

bars1 = ax.bar(x - bar_width / 2, statistics1, yerr=error1, capsize=5, error_kw=dict(elinewidth=1, capthick=1,),
               edgecolor='0.2', lw=1, width=bar_width, label=conditions[0], color=condition_colors[0])
bars2 = ax.bar(x + bar_width / 2, statistics2, yerr=error2, capsize=5, error_kw=dict(elinewidth=1, capthick=1,),
               edgecolor='0.2', lw=1, width=bar_width, label=conditions[1], color=condition_colors[1])

plot_params={'edgecolor':'0.2', 'linewidth':0.5, 'fc': 'none'}

if other_category != None:

    #mapping = {label: idx for idx, label in enumerate(np.unique(df[other_category]))}

    for i in range(len(categories)):
        #unique_values1, unique_idxs1 = np.unique(other_categories1[i], return_inverse=True)
        transformed1 = np.array([mapping.get(x, np.nan) for x in other_categories1[i]]) # 's13' -> 0, 's14' -> 1

        x_pos1 = np.random.normal(x[i] - bar_width / 2, 1/4*bar_width/2, size = scatter1[i].shape[0])
        ax.scatter(x_pos1, scatter1[i], marker='s', s=6,
                   c=transformed1, **plot_params)
        # ax.scatter(np.full_like(scatter1[i], x[i] - bar_width / 2), scatter1[i], marker='s', s=6,
        #            c=transformed1, **plot_params, cmap=cmap)

        transformed2 = np.array([mapping.get(x, np.nan) for x in other_categories2[i]])  # 's13' -> 0, 's14' -> 1

        x_pos2 = np.random.normal(x[i] + bar_width / 2, 1/4*bar_width/2, size=scatter2[i].shape[0])
        ax.scatter(x_pos2, scatter2[i], marker='s', s=6,
                   c=transformed2, **plot_params)

        # ax.scatter(np.full_like(scatter2[i], x[i] + bar_width / 2), scatter2[i], marker='s', s=6,
        #            c=transformed2, **plot_params, cmap=cmap)

    import matplotlib.patches as mpatches
    color_mapping = {label: color for label, color in zip(mapping.keys(), other_category_colors)}
    # Create custom legend handles (list of patches)
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_mapping.items()]

else:
    for i in range(len(categories)):
        ax.scatter(np.full_like(scatter1[i], x[i] - bar_width / 2), scatter1[i], marker='s', s=6,
                   fc=None, **plot_params, cmap='Set1')
        ax.scatter(np.full_like(scatter2[i], x[i] + bar_width / 2), scatter2[i], marker='s', s=6,
                   fc=None, **plot_params, cmap='Set1')

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=1, color='0.2')

ax.set_xticks(x)
ax.set_xticklabels(categories)
plt.xticks(fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2', weight='normal')
plt.yticks(fontsize=8,  color='0.2', weight='normal')


ax.set_ylabel('%s'%y, fontsize=8, weight='normal', color='0.2')
legend1 = ax.legend(frameon=False, prop={'weight': 'normal', 'size': 8}, labelcolor='0.2',loc='best')
if other_category != None:
    fig.canvas.draw()  # Ensure the figure is updated
    bbox_legend1 = legend1.get_window_extent()  # Get bounding box in display coordinates
    bbox_legend1 = ax.transAxes.inverted().transform(bbox_legend1)  # Convert to axes coordinates
    x0, y0 = bbox_legend1[0]  # Lower-left corner of the first legend
    x1, y1 = bbox_legend1[1]  # High-right corner of the first legend

    ax.legend(handles=legend_patches, frameon=False, prop={'weight': 'normal', 'size': 8}, labelcolor='0.2',loc='best',
              bbox_to_anchor=(x1+0.5, y1), bbox_transform=ax.transAxes
              )
    ax.add_artist(legend1)

plt.title('%s: %s, %s' % (categories, p_values, cohens_ds), fontsize=4)

plt.grid(False)
plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()



######################## interaction features mt vs WT box plot  ###########################
if not os.path.isdir(path + 'Inhibit int feature box plot_del_mouse/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Inhibit int feature box plot_del_mouse/')

if not os.path.isdir(path + 'Inhibit int feature violin plot_del_mouse/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Inhibit int feature violin plot_del_mouse/')

if not os.path.isdir(path + 'Inhibit int feature bar plot_del_mouse/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Inhibit int feature bar plot_del_mouse/')

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Inhibition', 'Mouse'])

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT GC B cell', 'mt_B-cell': 'MT GC B cell'}})


# k = df.iloc[:,324:417].isnull().any()
# null_features = k.index[k==True]
# feature_list = df.columns[324:417].drop(null_features).drop(['PC1', 'PC2', 'tskmeans'])

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['WT GC B cell', 'MT GC B cell']:
        for group in ['IgG', 'CD40L']:
        #for group in ['IgG', 'CD40L', 'mLT']:
            data = df_[(df_['Inhibition'] == group) & (df_[condition_name] == cell_type)][feature_name]
            dataset[cell_type + ' ' + str(group)] = np.array(data)

    values = flatten_nested_dict(dataset)
    if np.isnan(values).any() == True:  # Check at least one nan
        continue
    elif np.isfinite(values).all() == False:  # Check everything is not inf
        continue

    # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
    #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ', }
    # dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}


    draw_custom_violin_plot(dataset, path + 'Inhibit int feature violin plot_del_mouse/', file_name=feature_name,
                            #colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'),
                            colors=('#888888', '#888888', '#CC6677', '#CC6677',),
                            test='kruskal-wallis_dunn', pvalue=True, figsize=(2, 2), vmax=4)


    draw_custom_bar_plot(dataset, path + 'Inhibit int feature bar plot_del_mouse/', file_name=feature_name,
                         strip_plot=True,
                         #colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'),
                         colors=('#888888', '#888888', '#CC6677', '#CC6677',),
                         test='kruskal-wallis_dunn', pvalue=True, figsize=(2, 2), vmax=5)

    draw_custom_box_plot(dataset, path + 'Inhibit int feature box plot_del_mouse/', file_name=feature_name,
                         strip_plot=False,
                         # colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677'),
                         colors=('#888888', '#888888', '#CC6677', '#CC6677',),
                         test='kruskal-wallis_dunn', pvalue=True, figsize=(2, 2), vmax=10)



df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT GC B cell', 'mt_B-cell': 'MT GC B cell'}})
feature_name = 'T_avg_overlap' #'T_contact_persistences', 'T_avg_overlap'
condition_name = 'Type'
dict_datasets={}
for cell_type in ['WT GC B cell', 'MT GC B cell']:
    for group in ['IgG', 'CD40L']:
    #for group in ['IgG', 'CD40L', 'mLT']:
        data = df_[(df_['Inhibition'] == group) & (df_[condition_name] == cell_type)][feature_name]
        data = data[data>0]
        dict_datasets[cell_type + ' ' + str(group)] = np.array(data)

[print(key, np.array(value).size) for key, value in dict_datasets.items()]


font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

colors=('#888888', '#888888', '#CC6677', '#CC6677',)
fig, ax = plt.subplots(figsize=(2,2))
for i, key in enumerate(dict_datasets):
    ax = sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=(0, 0.05), color=colors[i], label=key)

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

legend = plt.legend(frameon=False, prop = {'weight':'normal', 'size':8}, labelcolor='0.2')
legend.remove()
plt.savefig(path+'%s kde.png'%feature_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s kde.svg'%feature_name, bbox_inches='tight')
plt.close()
plt.clf()


# figsize=(2,2)
# test = 'kruskal-wallis_dunn'
# file_name = 'T_contact_persistences_all'
# font = {'family': 'arial',
#             'weight': 'normal',
#             'size': 8}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
# matplotlib.rcParams['lines.linewidth'] = 1
#
#
#
# sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())
# fig, ax = plt.subplots(figsize=figsize)
# # ax = sns.violinplot(data=sorted_vals, palette=colors, linewidth=1, linecolor="0.2", inner="box",
# #                     inner_kws=dict(box_width=10, whis_width=10, color="0.2"), cut=0)
# ax = sns.violinplot(data=sorted_vals, palette=colors, linewidth=1, linecolor="0.2", inner="stick",
#                     inner_kws=dict(box_width=10, whis_width=10, color="0.2"), cut=0)
# #plot_params={'edgecolor':'0.2', 'linewidth':0.001, 'fc':'none'}
# #ax=sns.stripplot(data=sorted_vals, marker='s', s=1.5, **plot_params)
# ax=sns.pointplot(data=sorted_vals, marker='o', scale=0.6, capsize=.05, join=False, color=".2", errorbar=('pi', 95),errwidth=1.5)
#
#
# for axis in ['bottom', 'left']:
#     ax.spines[axis].set_linewidth(1)
#     ax.spines[axis].set_color('0.2')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# ax.tick_params(width=1, color='0.2')
# #ax.set_ylabel(feature_name, fontsize=8, weight='normal')
# plt.xticks(plt.xticks()[0], sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
#            weight='normal')
# plt.yticks(fontsize=8, color='0.2', weight='normal')
# plt.ylim(0, 10)
# pairs, p_values, cohen_ds = get_various_statistics(dict_datasets, test=test, return_sig=True)
# plt.title('%s: %s, %s' % (pairs, p_values, cohen_ds), fontsize=4)
#
#
# plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
# plt.clf()
# plt.close()




from scipy import stats
figsize=(2,2)
test = 'kruskal-wallis_dunn'
error_type = 'ci_norm'
file_name = '%s_test'%feature_name
colors=('#888888', '#888888', '#CC6677', '#CC6677',)

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1


mean_dataset = {}
error_dataset = {}
for key, values in dict_datasets.items():

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

    mean_dataset[key] = np.mean(values)
    error_dataset[key] = error



sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())
fig, ax = plt.subplots(figsize=figsize)
ax = sns.violinplot(data=sorted_vals, palette=colors, linewidth=1, linecolor="0.2", inner=None,
                    inner_kws=dict(box_width=10, whis_width=10, color="0.2"), cut=0)

ax = sns.scatterplot(x=np.arange(0, len(sorted_keys), 1), y=mean_dataset.values(), color="0.2", s=8, zorder=3)

# for idx, key in enumerate(mean_dataset):
#     ax = sns.lineplot(data=mean_dataset, x=idx, y=mean_dataset[key], linestyle='',
#                                  label=key, lw=2.5,  dashes=False, markersize=8, err_style='bars', color=colors[idx])
ax.errorbar(x=np.arange(0, len(sorted_keys), 1), y=list(mean_dataset.values()),
                                yerr=list(error_dataset.values()), color='0.2', capsize=3, capthick=1, elinewidth=1.5, fmt='none', zorder=2)

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=1, color='0.2')
#ax.set_ylabel(feature_name, fontsize=8, weight='normal')
plt.xticks(plt.xticks()[0], sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
           weight='normal')
plt.yticks(fontsize=8, color='0.2', weight='normal')
#plt.ylim(0, 10)
plt.ylim(0, 0.1)
pairs, p_values, cohen_ds = get_various_statistics(dict_datasets, test=test, return_sig=True)
plt.title('%s: %s, %s' % (pairs, p_values, cohen_ds), fontsize=4)


plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()


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



###################### Plot Experimental int feature plots for Exp  ############################
videos = np.unique(df['Video'])

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT GC B cell', 'mt_B-cell': 'MT GC B cell'}})

feature_name = 'T_contact_persistences'
for thresh in range(0, 21):
    dataset = {}
    for cell_type in ['WT GC B cell', 'MT GC B cell']:
        #for group in ['Control','IgG', 'CD40L']:
        for group in ['IgG', 'CD40L']:
            df_part = df_[(df_['Type'] == cell_type)&(df_['Inhibition'] == group)].reset_index(drop=True)
            avgs = []
            for video in videos:
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] < 20:
                    continue
                print(cell_type, group, video, df_video.shape[0])

                data = df_video[feature_name]
                zero_count = np.sum(data<=thresh)
                nonzero_count = np.sum(data>thresh)
                avg = nonzero_count / data.size
                #avg = np.mean(data)
                avgs.append(avg)
            dataset[cell_type + ' ' + str(group)] = avgs
    #new_order = ['wt_B-cell', 'wt_B-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    #dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_bar_plot(dataset, path, file_name='experimental_T_contact_persistences_%s'%thresh,
                         strip_plot=True,
                         #colors=('#888888', '#888888', '#888888',  '#CC6677', '#CC6677', '#CC6677'),
                         colors=('#888888', '#888888', '#CC6677', '#CC6677',),
                         test='kruskal-wallis_dunn', pvalue=True, figsize=(2, 2))



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

