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
# ==============================================================================
"""Generates Data for Figure1-2. Behavior of T and WT wrt FDC & LZ vs DZ"""
import scipy.stats
from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import ZoneSignal

#################################### Pull out interaction features ####################################

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GroupA_no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GroupA_no_inhibit_traj_duration_20.parquet')

# thresh = 2
# df['Zone'] = 'None'
# df.loc[(df['LZ_distance_average'] <= thresh) & (df['Core_distance_average'] > thresh) & (df['DZ_distance_average'] > thresh), 'Zone'] = 'sLZ'
# df.loc[df['DZ_distance_average'] <= thresh, 'Zone'] = 'DZ'
# df.loc[df['Core_distance_average'] <= thresh, 'Zone'] = 'dLZ'


_, _, Zone_series = to_timeseries_fast(df_duration, duration=20, feature_name='Zone')

feature_list = ['avg_zone', 'dz_resident_times', 'dz_resident_persistences', 'slz_resident_times',
                'slz_resident_persistences', 'dlz_resident_times', 'dlz_resident_persistences']
Zone_func = ZoneSignal(Zone_series)
df_zone = Zone_func.extract_features(feature_list)

df = pd.concat([df, df_zone], axis=1)


# df.loc[(df['avg_zone'] < 0.4) & (df['avg_zone'] >= 0), 'Zone'] = 'DZ'
# df.loc[(df['avg_zone'] < 0.8) & (df['avg_zone'] >= 0.4), 'Zone'] = 'DZ-sLZ'
# df.loc[(df['avg_zone'] < 1.2) & (df['avg_zone'] >= 0.8), 'Zone'] = 'sLZ'
# df.loc[(df['avg_zone'] < 1.6) & (df['avg_zone'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
# df.loc[(df['avg_zone'] <= 2) & (df['avg_zone'] >= 1.6), 'Zone'] = 'dLZ'

# print(df[df['Zone']=='DZ'].shape[0], df[df['Zone']=='DZ_sLZ'].shape[0], df[df['Zone']=='sLZ'].shape[0],
#       df[df['Zone']=='sLZ_dLZ'].shape[0], df[df['Zone']=='dLZ'].shape[0])

df.loc[(df['avg_zone'] < 0.4) & (df['avg_zone'] >= 0), 'Zone'] = 'non FDC'
df.loc[(df['avg_zone'] <= 2) & (df['avg_zone'] >= 0.8), 'Zone'] = 'FDC'
df.loc[(df['avg_zone'] < 0.8) & (df['avg_zone'] >= 0.4), 'Zone'] = 'None'

print(df[df['Zone']=='non FDC'].shape[0], df[df['Zone']=='FDC'].shape[0])

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure1. Dynamic Behavior of T and WT\Zone dependent behavior\\'

####################################### Quantify FDC interaction frequency #############################################
int_time = 20

persistent_int_freqs_datasets = {}
persistent_int_freq_per_cellnumbers_datasets = {}
total_n_contacts_per_cellnumbers_datasets = {}
persistent_int_freq_per_cellcontacts_datasets = {}
low_contact_freq_per_cellnumbers_datasets = {}

videos = np.unique(df['Video'])
for cell_type in ['T-cell', 'wt_B-cell']:
    df_part = df[df['Type'] == cell_type]
    persistent_int_freqs = []
    persistent_int_freq_per_cellnumbers = []
    total_n_contacts_per_cellnumbers = []
    persistent_int_freq_per_cellcontacts = []
    low_contact_freq_per_cellnumbers = []
    for video in videos:
        #if 'A' in video and cell_type == 'mt_B-cell':
        # if '-A' in video:
        #     continue
        df_video = df_part[df_part['Video'] == video]
        if df_video.shape[0] == 0:
            continue
        data = df_video['FDC_contact_times']
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

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
new_order = ['wt_B-cell', 'T-cell']
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

colors=('#6699CC', '#CC6677')
draw_custom_bar_plot(persistent_int_freqs_datasets, path, file_name='Total FDC interaction frequency',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellnumbers_datasets, path, file_name='FDC persistent interaction frequency per cell number',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(total_n_contacts_per_cellnumbers_datasets, path, file_name='number of FDC contacts per cell number',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellcontacts_datasets, path, file_name='FDC persistent interaction frequency per number of contacts',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(low_contact_freq_per_cellnumbers_datasets, path, file_name='FDC low contact time frequency per cell number',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))

####################################### FDC interaction kde Tfh vs WT #############################################
replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'WT'}

dataset={}
for condition in np.unique(df['Type']):
    data = df[df['Type'] == condition]['FDC_contact_times']
    dataset[condition] = np.array(data)
new_order = ['wt_B-cell', 'T-cell']
ordered_dataset = change_dict_order(dataset, new_order)
dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

colors=('#6699CC', '#CC6677')
fig, ax = plt.subplots(figsize=(2,2))
for i, key in enumerate(dict_datasets):
    ax = sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=(0, 40), color=colors[i], label=key)

# ax = sns.kdeplot(data=dict_datasets['WT'], fill=True, linewidth=1, clip=(0, 40), color='#888888', label='WT')
# ax = sns.kdeplot(data=dict_datasets['MT'], fill=True, linewidth=1, clip=(0, 40), color='#CC6677', label='MT')

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('FDC interaction frequency', fontsize=8, weight='normal', color='0.2')
ax.set_ylabel('Density', fontsize=8, weight='normal', color='0.2')
plt.xticks(fontsize=8, color='0.2', weight='normal')
plt.yticks(fontsize=8, color='0.2', weight='normal')

#plt.legend(frameon=False, prop = {'weight':'normal', 'size':8}, labelcolor='0.2')

plt.savefig(path+'cell-FDC interaction frequency.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/cell-FDC interaction frequency.svg', bbox_inches='tight')
plt.close()
plt.clf()

####################################### Average FDC distance kde T cell vs WT #############################################
replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'WT'}

coloc_features = ['FDC_distance_average', 'DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']
ranges = [(0, 10), (0, 60), (0,20), (0, 100)]
names = ['Distance to FDC', 'Distance to DZ', 'Distance to sLZ', 'Distance to dLZ']

for idx, coloc_feature in enumerate(coloc_features):

    dataset={}
    for condition in np.unique(df['Type']):
        data = df[df['Type'] == condition][coloc_feature]
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell', 'T-cell']
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

    ax = sns.kdeplot(data=dict_datasets['WT'], fill=True, linewidth=1, clip=ranges[idx], color='#6699CC', label='WT')
    ax = sns.kdeplot(data=dict_datasets['Tfh'], fill=True, linewidth=1, clip=ranges[idx], color='#CC6677', label='Tfh')

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('%s (μm)'%names[idx], fontsize=8, weight='normal', color='0.2')
    ax.set_ylabel('Density', fontsize=8, weight='normal', color='0.2')
    plt.xticks(fontsize=8, color='0.2', weight='normal')
    plt.yticks(fontsize=8, color='0.2', weight='normal')

    plt.legend(frameon=False, prop = {'weight':'normal', 'size':8}, labelcolor='0.2')

    plt.savefig(path+'%s.png'%coloc_feature, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg'%coloc_feature, bbox_inches='tight')
    plt.close()
    plt.clf()


############# Plot non FDC vs FDC jointplot for each cell type ###############

for cell_type in ['T-cell', 'wt_B-cell']:

    df_part = df[df['Type']==cell_type].reset_index(drop=True)

    draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s_non FDC vs FDC jointplot'%cell_type, hue="Zone", hue_order=['non FDC', 'FDC'],
                   colors=('#888888', '#999933'), fill=True, legend=False, thresh=0.3, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')


############# Plot LZ vs DZ jointplot for each cell type ###############

for cell_type in ['T-cell', 'wt_B-cell']:

    df_part = df[df['Type']==cell_type].reset_index(drop=True)

    df_part.loc[(df_part['Zone'] == 'DZ'), 'Zone1'] = 'DZ'
    #df_part.loc[(df_part['Zone'] == 'DZ-sLZ'), 'Zone1'] = 'DZ-sLZ'
    df_part.loc[( (df_part['Zone'] == 'sLZ') | (df_part['Zone'] == 'dLZ') | (df_part['Zone'] == 'sLZ-dLZ') ), 'Zone1'] = 'LZ'


    print(cell_type, df_part[df_part['Zone1']=='DZ'].shape[0], df_part[df_part['Zone1']=='DZ-sLZ'].shape[0], df_part[df_part['Zone1']=='LZ'].shape[0])

    draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s_DZ vs LZ jointplot'%cell_type, hue="Zone1", hue_order=['DZ', 'LZ'],
                   colors=('#888888', '#CC6677'), fill=True, legend=False, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

############ Plot LZ vs DZ contour map for each cell type ###############

for cell_type in ['T-cell', 'wt_B-cell']:

    fig, ax = plt.subplots(figsize=(2, 2))  # 2 inch by 2 inch

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    color_list = ['Greys', 'Reds', 'Greens', 'Oranges', 'Purples', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
                  'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'plasma']
    x_name='PC1'
    y_name='PC2'
    num_contours=6
    bin_num=50

    contours = []
    groups = ['DZ', 'LZ']

    x_dz = df[(df['Type']==cell_type) & (df['Zone'] == 'DZ')][x_name]
    y_dz = df[(df['Type']==cell_type) & (df['Zone'] <= 'DZ')][y_name]

    x_lz = df[( (df['Type']==cell_type) & (df['Zone'] == 'sLZ') ) | ( (df['Type']==cell_type) & (df['Zone'] == 'dLZ') )][x_name]
    y_lz = df[( (df['Type']==cell_type) & (df['Zone'] == 'sLZ') ) | ( (df['Type']==cell_type) & (df['Zone'] == 'dLZ') )][y_name]


    print(cell_type, x_dz.shape, x_lz.shape)
    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    kde_coordinate_dz = np.vstack([x_dz, y_dz])  # shape = (2(dimension), number of points)
    kde_coordinate_lz = np.vstack([x_lz, y_lz])  # shape = (2(dimension), number of points)

    if (kde_coordinate_dz.shape[1] <= 2) or (kde_coordinate_lz.shape[1] <= 2):  # if there is only few points, it cannot calculate gaussian kde
        raise ValueError('Number of points should be greater than 2 to create contour')
    else:
        kde_dz = scipy.stats.gaussian_kde(kde_coordinate_dz)  # Define kernel (bandwidth by Scott's Rule)
        kde_lz = scipy.stats.gaussian_kde(kde_coordinate_lz)  # Define kernel (bandwidth by Scott's Rule)
        # evaluate on a regular grid
        xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
        ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        # Xgrid , Ygrid = (bin_num,bin_num) 2d array
        # Xgrid[i] = xgrid coordinate(divide by 100 from -x ~ +x` and repeat in row direction)
        # Ygrid[:,i] = ygrid coordinate(divide by 100 from -y ~ +y` and repeat in column direction)
        Z_dz = kde_dz.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z_lz = kde_lz.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        # Xgrid.ravel() = bin_num^2 shape 1d vector (vector that repeat -x ~ +x`, -x ~ +x` bin_num times)
        # np.vstack() = (2,bin_num^2) 2d array linspaced coordinates
        # Z = (10000,) 1d vector
        pdf_dz = Z_dz.reshape(Xgrid.shape)
        pdf_lz = Z_lz.reshape(Xgrid.shape)

        contour_dz = ax.contour(Xgrid, Ygrid, pdf_dz,
                             # colors='red',
                             linewidths=1,
                             linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                             # label=group,
                             cmap=color_list[0],
                             origin='lower',
                             levels=num_contours,
                             )
        contour_lz = ax.contour(Xgrid, Ygrid, pdf_lz,
                             # colors='red',
                             linewidths=1,
                             linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                             # label=group,
                             cmap=color_list[1],
                             origin='lower',
                             levels=num_contours,
                             )

        contours.append(contour_dz)
        contours.append(contour_lz)

    format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
    ax.legend(handles=[contour.legend_elements()[0][-1] for contour in contours], labels=groups,
              bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0, fontsize=3, frameon=False, markerscale=0.3)
    #plt.title('%s NOI vs PI' % cell_type, fontsize=25)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig(path + '%s_DZvsLZ.png' % cell_type, dpi=300)
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s_DZvsLZ.svg' % cell_type)
    plt.close()
    plt.clf()

############# Plot sLZ vs dLZ jointplot for each cell type ###############

for cell_type in ['T-cell', 'wt_B-cell']:

    df_part = df[df['Type']==cell_type].reset_index(drop=True)

    df_part.loc[(df_part['Zone'] == 'sLZ'), 'Zone1'] = 'sLZ'
    #df_part.loc[(df_part['Zone'] == 'sLZ-dLZ'), 'Zone1'] = 'sLZ-dLZ'
    df_part.loc[(df_part['Zone'] == 'dLZ'), 'Zone1'] = 'dLZ'

    print(cell_type, df_part[df_part['Zone1']=='sLZ'].shape[0], df_part[df_part['Zone1']=='sLZ-dLZ'].shape[0], df_part[df_part['Zone1']=='dLZ'].shape[0])

    draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s_sLZ vs dLZ jointplot'%cell_type, hue="Zone1", hue_order=['sLZ', 'dLZ'],
                   colors=('#6699CC', '#999933'), fill=True, legend=False, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

############# Plot DZ vs sLZ vs dLZ Zone contour map for each cell type ###############

for cell_type in ['T-cell', 'wt_B-cell']:

    fig, ax = plt.subplots(figsize=(2, 2))  # 2 inch by 2 inch

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    color_list = ['Greys', 'Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
                  'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'plasma']
    x_name='PC1'
    y_name='PC2'
    num_contours=6
    bin_num=50

    contours = []
    #groups = ['DZ', 'sLZ', 'dLZ']
    groups = ['sLZ', 'dLZ']

    x_dz = df[(df['Type']==cell_type) & (df['Zone'] == 'DZ')][x_name]
    y_dz = df[(df['Type']==cell_type) & (df['Zone'] == 'DZ')][y_name]

    x_slz = df[(df['Type']==cell_type) & (df['Zone'] == 'sLZ')][x_name]
    y_slz = df[(df['Type']==cell_type) & (df['Zone'] == 'sLZ')][y_name]

    x_dlz = df[(df['Type'] == cell_type) & (df['Zone'] == 'dLZ')][x_name]
    y_dlz = df[(df['Type'] == cell_type) & (df['Zone'] == 'dLZ')][y_name]

    print(cell_type, x_dz.shape, x_slz.shape, x_dlz.shape)
    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    kde_coordinate_dz = np.vstack([x_dz, y_dz])  # shape = (2(dimension), number of points)
    kde_coordinate_slz = np.vstack([x_slz, y_slz])  # shape = (2(dimension), number of points)
    kde_coordinate_dlz = np.vstack([x_dlz, y_dlz])  # shape = (2(dimension), number of points)

    if (kde_coordinate_dz.shape[1] <= 2) or (kde_coordinate_slz.shape[1] <= 2) or (kde_coordinate_dlz.shape[1] <= 2):  # if there is only few points, it cannot calculate gaussian kde
        raise ValueError('Number of points should be greater than 2 to create contour')
    else:
        kde_dz = scipy.stats.gaussian_kde(kde_coordinate_dz)  # Define kernel (bandwidth by Scott's Rule)
        kde_slz = scipy.stats.gaussian_kde(kde_coordinate_slz)  # Define kernel (bandwidth by Scott's Rule)
        kde_dlz = scipy.stats.gaussian_kde(kde_coordinate_dlz)  # Define kernel (bandwidth by Scott's Rule)
        # evaluate on a regular grid
        xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
        ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        # Xgrid , Ygrid = (bin_num,bin_num) 2d array
        # Xgrid[i] = xgrid coordinate(divide by 100 from -x ~ +x` and repeat in row direction)
        # Ygrid[:,i] = ygrid coordinate(divide by 100 from -y ~ +y` and repeat in column direction)
        Z_dz = kde_dz.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z_slz = kde_slz.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z_dlz = kde_dlz.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        # Xgrid.ravel() = bin_num^2 shape 1d vector (vector that repeat -x ~ +x`, -x ~ +x` bin_num times)
        # np.vstack() = (2,bin_num^2) 2d array linspaced coordinates
        # Z = (10000,) 1d vector
        pdf_dz = Z_dz.reshape(Xgrid.shape)
        pdf_slz = Z_slz.reshape(Xgrid.shape)
        pdf_dlz = Z_dlz.reshape(Xgrid.shape)

        # contour_dz = ax.contour(Xgrid, Ygrid, pdf_dz,
        #                      # colors='red',
        #                      linewidths=1,
        #                      linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
        #                      # label=group,
        #                      cmap=color_list[0],
        #                      origin='lower',
        #                      levels=num_contours,
        #                      )
        contour_slz = ax.contour(Xgrid, Ygrid, pdf_slz,
                             # colors='red',
                             linewidths=1,
                             linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                             # label=group,
                             cmap=color_list[1],
                             origin='lower',
                             levels=num_contours,
                             )
        contour_dlz = ax.contour(Xgrid, Ygrid, pdf_dlz,
                                # colors='red',
                                linewidths=1,
                                linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                                # label=group,
                                cmap=color_list[2],
                                origin='lower',
                                levels=num_contours,
                                )

        #contours.append(contour_dz)
        contours.append(contour_slz)
        contours.append(contour_dlz)

    format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
    ax.legend(handles=[contour.legend_elements()[0][-1] for contour in contours], labels=groups,
              bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0, fontsize=3, frameon=False, markerscale=0.3)
    #plt.title('%s NOI vs PI' % cell_type, fontsize=25)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig(path + '%s_zones.png' % cell_type, dpi=300)
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s_zones.svg' % cell_type)
    plt.close()
    plt.clf()

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


rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_DZ-sLZ': 'WT DZ-sLZ', 'wt_B-cell_sLZ': 'WT sLZ',
               'wt_B-cell_sLZ-dLZ': 'WT sLZ-dLZ', 'wt_B-cell_dLZ': 'WT dLZ',
               'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_DZ-sLZ': 'MT DZ-sLZ', 'mt_B-cell_sLZ': 'MT sLZ',
               'mt_B-cell_sLZ-dLZ': 'MT sLZ-dLZ', 'mt_B-cell_dLZ': 'MT dLZ',
               }
df_corr.rename(columns=rename_keys, inplace=True)
df_corr.rename(index=rename_keys, inplace=True)

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

plt.savefig(path+'DZ vs sLZ vs dLZ correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/DZ vs sLZ vs dLZ correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

############# fraction of cluster for each cell type ###############
vmax=35
colors = ('#CC6677', '#6699CC')

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
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'T-cell': 'Tfh'}})
df_ = df_[df_['Zone']!='None'].reset_index(drop=True)
df_['Type_Zone'] = df_['Type'].astype(str) + ' ' + df_['Zone']
draw_cluster_distribution_heatmap(df_, path, file_name='all_kmeans_heatmap', condition_name='Type_Zone', annot=False,
                                  cluster_type='kmeans', col_cluster=False, figsize=(6,3))

draw_heatmap_with_circles(df_, path, file_name='all_kmeans_circleheatmap', condition_name='Type_Zone', cluster_type='kmeans',
                          vmax=None, transpose=False, row_cluster=True, col_cluster=False, figsize=(4,4))

for zone in np.unique(df_['Type_Zone']):
    print(zone, df_[df_['Type_Zone']==zone].shape[0])


############# Plot linear regression btw FDC zones and cluster enrichment ###############


tfh_enrichments = pd.DataFrame()
wt_enrichments = pd.DataFrame()
for group_clone in group_clones:  # 'DZ', 'sLz', 'dLZ'
    tfh_enrichments = pd.concat( [tfh_enrichments, group_clone[list(group_clone.columns)[0]]], axis=1 )
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
for idx, kmeans in enumerate(tfh_enrichments.index):
    r, p = scipy.stats.spearmanr(np.arange(len(list(tfh_enrichments.iloc[kmeans, :].index))),
                                tfh_enrichments.iloc[kmeans, :].values)
    if p>0.1:
        continue
    sns.regplot(x=np.arange(len(list(tfh_enrichments.iloc[kmeans, :].index))), y=tfh_enrichments.iloc[kmeans, :].values,
                ci=None, line_kws={'color':cmap[idx], 'linewidth':3}, label=kmeans, scatter=False, scatter_kws={'s': 10, 'color':cmap[idx]})
    # sns.scatterplot(data=df_p, x=z_name, y=p_name, hue='color', hue_order=['NoChange', 'Change'],
    #                 palette=['gray', 'firebrick'])

    rs.append(r)
    ps.append(p)
    # plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
    #          fontsize=20, fontdict={'weight': 'normal'}, color="blue")

plt.title('r: %s, p: %s'%(rs, ps) , fontsize=4, fontdict={'weight': 'normal'})
#plt.xticks(plt.xticks()[0], datasets['Exact_Age'], fontsize=12, fontdict={'weight': 'normal'})
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

ax.tick_params(width=2, color='0.2')

ax.set_ylabel('fraction of cluster (%)', fontsize=16, weight='normal', color='0.2')
plt.xticks(np.arange(len(list(tfh_enrichments.iloc[kmeans, :].index))), ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ'], fontsize=16, rotation=35, # fontname = "Arial",
           rotation_mode='anchor', ha='right', color='0.2', weight='normal')

plt.yticks(fontsize=16, color='0.2', weight='normal')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
           loc='best')

plt.savefig(path + 'Tfh cluster fraction regplot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/Tfh cluster fraction regplot.svg', bbox_inches='tight')
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
    if p>0.1:
        continue
    sns.regplot(x=np.arange(len(list(wt_enrichments.iloc[kmeans, :].index))), y=wt_enrichments.iloc[kmeans, :].values,
                ci=None, line_kws={'color':cmap[idx], 'linewidth':3}, label=kmeans, scatter=False, scatter_kws={'s': 10, 'color':cmap[idx]})
    rs.append(r)
    ps.append(p)
    # plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
    #          fontsize=20, fontdict={'weight': 'normal'}, color="blue")

plt.title('r: %s, p: %s'%(rs, ps) , fontsize=4, fontdict={'weight': 'normal'})
#plt.xticks(plt.xticks()[0], datasets['Exact_Age'], fontsize=12, fontdict={'weight': 'normal'})
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

ax.tick_params(width=2, color='0.2')

ax.set_ylabel('fraction of cluster (%)', fontsize=16, weight='normal', color='0.2')
plt.xticks(np.arange(len(list(wt_enrichments.iloc[kmeans, :].index))), ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ'], fontsize=16, rotation=35, # fontname = "Arial",
           rotation_mode='anchor', ha='right', color='0.2', weight='normal')

plt.yticks(fontsize=16, color='0.2', weight='normal')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
           loc='best')

plt.savefig(path + 'WT cluster fraction regplot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/WT cluster fraction regplot.svg', bbox_inches='tight')
plt.clf()
plt.close()


############# Shannon entropy of DZ vs sLZ vs dLZ for each video ###############

group_name = 'Video'
groups = np.unique(df[group_name])

entropies_dz = {'T-cell': [], 'wt_B-cell':[]}
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

entropies_dz_slz = {'T-cell': [], 'wt_B-cell':[]}
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

entropies_slz = {'T-cell': [], 'wt_B-cell':[]}
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

entropies_slz_dlz = {'T-cell': [], 'wt_B-cell':[]}
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

entropies_dlz = {'T-cell': [], 'wt_B-cell':[]}
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
for cell_type in ['T-cell', 'wt_B-cell']:
    entropies = {'%s_DZ'%cell_type:entropies_dz[cell_type], '%s_DZ-sLZ'%cell_type:entropies_dz_slz[cell_type], '%s_sLZ'%cell_type:entropies_slz[cell_type],
                  '%s_sLZ-dLZ'%cell_type:entropies_slz_dlz[cell_type], '%s_dLZ'%cell_type:entropies_dlz[cell_type],}

    # entropies = {'%s_DZ' % cell_type: entropies_dz[cell_type], '%s_sLZ' % cell_type: entropies_slz[cell_type],
    #              '%s_dLZ' % cell_type: entropies_dlz[cell_type], }

    rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_DZ-sLZ': 'WT DZ-sLZ', 'wt_B-cell_sLZ': 'WT sLZ',
                   'wt_B-cell_sLZ-dLZ': 'WT sLZ-dLZ', 'wt_B-cell_dLZ': 'WT dLZ',
                   'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_DZ-sLZ': 'MT DZ-sLZ', 'mt_B-cell_sLZ': 'MT sLZ',
                    'mt_B-cell_sLZ-dLZ': 'MT sLZ-dLZ', 'mt_B-cell_dLZ': 'MT dLZ',}
    entropies_ = {rename_keys.get(k, k): v for (k, v) in entropies.items()}
    # draw_custom_bar_plot(entropies_, path, file_name='entropy of DZ vs sLZ vs dLZ for %s' %cell_type, colors=('#888888', '#CC6677', '#6699CC'),
    #                      strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))
    dict_datasets = entropies_
    file_name = 'entropy of DZ vs sLZ vs dLZ for %s' %cell_type
    test = 'mann-whitney'

    colors = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77')
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

############# Shannon entropy of DZ vs sLZ vs dLZ for each video ###############

df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'T-cell': 'Tfh'}})

group_name = 'Video'
groups = np.unique(df_[group_name])

entropies_dz = {'Tfh': [], 'WT':[]}
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

entropies_dz_slz = {'Tfh': [], 'WT':[]}
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

entropies_slz = {'Tfh': [], 'WT':[]}
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

entropies_slz_dlz = {'Tfh': [], 'WT':[]}
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

entropies_dlz = {'Tfh': [], 'WT':[]}
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
color_list=['#CC6677', '#6699CC']
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
for idx, cell_type in enumerate(['Tfh', 'WT']):
    entropies = {'%s_DZ'%cell_type:entropies_dz[cell_type], '%s_DZ-sLZ'%cell_type:entropies_dz_slz[cell_type], '%s_sLZ'%cell_type:entropies_slz[cell_type],
                  '%s_sLZ-dLZ'%cell_type:entropies_slz_dlz[cell_type], '%s_dLZ'%cell_type:entropies_dlz[cell_type],}

    # entropies = {'%s_DZ' % cell_type: entropies_dz[cell_type], '%s_sLZ' % cell_type: entropies_slz[cell_type],
    #              '%s_dLZ' % cell_type: entropies_dlz[cell_type], }

    rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_DZ-sLZ': 'WT DZ-sLZ', 'wt_B-cell_sLZ': 'WT sLZ',
                   'wt_B-cell_sLZ-dLZ': 'WT sLZ-dLZ', 'wt_B-cell_dLZ': 'WT dLZ',
                   'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_DZ-sLZ': 'MT DZ-sLZ', 'mt_B-cell_sLZ': 'MT sLZ',
                    'mt_B-cell_sLZ-dLZ': 'MT sLZ-dLZ', 'mt_B-cell_dLZ': 'MT dLZ',}
    entropies_ = {rename_keys.get(k, k): v for (k, v) in entropies.items()}

    p_value_data[cell_type] = entropies_
    mean_dataset = {}
    std_dataset = {}

    for key in entropies_:
        mean = np.mean(entropies_[key])
        std = np.std(entropies_[key])
        mean_dataset[key] = mean
        std_dataset[key] = std
        print(key, mean, std)

    sns.lineplot(data=mean_dataset, x=np.arange(len(list(mean_dataset))), y=mean_dataset.values(),
                 label=cell_type, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars',
                 color=color_list[idx])
    ax.errorbar(np.arange(len(list(mean_dataset))), mean_dataset.values(), [0.1*x for x in std_dataset.values()],
                color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

handles, labels = ax.get_legend_handles_labels()
# ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['MT'], 0.1*std_dataset['MT'], color='#CC6677',
#            capsize=3, capthick=1, elinewidth=1.5)
# ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['WT'], 0.1*std_dataset['WT'], color='#888888',
#            capsize=3, capthick=1, elinewidth=1.5)

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=2, color='0.2')

ax.set_ylabel('Shannon entropy', fontsize=16, weight='normal', color='0.2')

#ax.set_xlabel('%s' % x_label, fontsize=16, weight='normal', color='0.2')

plt.xticks(np.arange(len(list(mean_dataset))), ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ'], fontsize=16, rotation=35, # fontname = "Arial",
           rotation_mode='anchor', ha='right', color='0.2', weight='normal')

plt.yticks(fontsize=16, color='0.2', weight='normal')

plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
           loc='best')

from scipy import stats
p_values = []
pairs = []
for (mt_key, mt_values), (wt_keys, wt_values) in zip(p_value_data['Tfh'].items(), p_value_data['WT'].items()):
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

###################### Plot Zone motility feature violin plot  ############################

if not os.path.isdir(path + 'Zone motility box plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Zone motility box plot/')

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in np.unique(df[condition_name]):
        for group in ['DZ','sLZ', 'dLZ']:
        # for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
            data = df[(df[condition_name] == cell_type)&(df['Zone'] == group)][feature_name]

            dataset[cell_type+'_'+str(group)] = np.array(data)

    rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
                   'T-cell_DZ': 'Tfh DZ', 'T-cell_sLZ': 'Tfh sLZ', 'T-cell_dLZ': 'Tfh dLZ', }
    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_violin_plot(dataset_renamed, path + 'Zone motility box plot/', file_name=feature_name,
                            colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC'), test='mann-whitney', pvalue=True, figsize=(2, 2))


###################### Plot Experimental motility feature plots for Zones  ############################
videos = np.unique(df['Video'])

if not os.path.isdir(path + 'experimental_zone_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'experimental_zone_motility_feature/')

for feature_name in feature_list:
    dataset = {}
    for cell_type in np.unique(df['Type']):
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
    #new_order = ['wt_B-cell', 'T-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
                   'T-cell_DZ': 'Tfh DZ', 'T-cell_sLZ': 'Tfh sLZ', 'T-cell_dLZ': 'Tfh dLZ', }
    dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_bar_plot(dict_datasets, path+'experimental_zone_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC'), test='mann-whitney', pvalue=True, figsize=(2, 2))

# #################################### Volcano plot of NOI vs PC motility features ####################################
# df.columns.get_loc('morpho_displ_autocorr_3')
# feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
#
# for cell_type in ['T-cell', 'wt_B-cell']:
#     df_p = pd.DataFrame()
#     for feature_name in feature_list:
#         dataset = {}
#         for n_interaction in [0,20]:
#             data = df[(df['tp_interaction_FDC'] == n_interaction)&(df['Type'] == cell_type)][feature_name]
#             dataset[cell_type+'_'+str(n_interaction)] = np.array(data)
#
#         pvalue = get_pvalue(dataset, test='mann-whitney')
#         logp = -np.log10(pvalue)
#
#         avgZ = get_avgZ(dataset, ref_name=cell_type+'_0', data_name=cell_type+'_20')
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
#     draw_volcano_plot(df_p, path, file_name='%s NOI vs PC motility volcano plot'%cell_type, z_thresh=0.4, p_thresh=20, z_name='AvgZ', p_name='Adj_Logp',
#                       feature_name='Feature', figsize=(6,6))


######################## Zone interaction features Tfh vs WT box plot  ###########################
if not os.path.isdir(path + 'int feature violin plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'int feature violin plot/')

if not os.path.isdir(path + 'int feature box plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'int feature box plot/')

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone'])

condition_name = 'Type'
replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'GCB'}

for feature_name in feature_list:
    try:
        dataset={}
        for condition in np.unique(df[condition_name]):
            data = df[df[condition_name] == condition][feature_name]
            dataset[condition] = np.array(data)
        new_order = ['wt_B-cell', 'T-cell']
        ordered_dataset = change_dict_order(dataset, new_order)
        dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

        draw_custom_violin_plot(dict_datasets, path+'int feature violin plot/', file_name=feature_name, colors=('#6699CC', '#CC6677'),
                                test='mann-whitney', pvalue=True, figsize=(1,2))
        draw_custom_bar_plot(dict_datasets, path + 'int feature box plot/', file_name=feature_name,
                             colors=('#6699CC', '#CC6677'),
                             strip_plot=False, test='mann-whitney', pvalue=True, figsize=(1, 2))
    except:
        pass

    # draw_custom_box_plot(dict_datasets, path+ 'feature_box_plot_type/', file_name=feature_name, colors=('#6699CC', '#CC6677'),
    # strip_plot=False, test='mann-whitney', pvalue=True, figsize=(1,2))


######################## Zone interaction features Tfh vs WT box plot  ###########################
if not os.path.isdir(path + 'Zone int feature violin plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Zone int feature violin plot/')

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone'])

# k = df.iloc[:,324:417].isnull().any()
# null_features = k.index[k==True]
# feature_list = df.columns[324:417].drop(null_features).drop(['PC1', 'PC2', 'tskmeans'])
for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in np.unique(df[condition_name]):
        for zone in ['DZ','sLZ', 'dLZ']:
            data = df[(df['Zone'] == zone)&(df[condition_name] == cell_type)][feature_name]
            dataset[cell_type+'_'+str(zone)] = np.array(data)

    values = flatten_nested_dict(dataset)
    if np.isnan(values).any() == True:  # Check at least one nan
        continue
    elif np.isfinite(values).all() == False:  # Check everything is not inf
        continue

    rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
                   'T-cell_DZ': 'Tfh DZ', 'T-cell_sLZ': 'Tfh sLZ', 'T-cell_dLZ': 'Tfh dLZ', }
    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    # draw_custom_violin_plot(dataset_renamed, path + 'Zone int feature violin plot/', file_name=feature_name,
    #                         colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC'), test='mann-whitney', pvalue=True, figsize=(2, 2))

    draw_custom_bar_plot(dataset_renamed, path + 'Zone int feature violin plot/', file_name=feature_name,
                         strip_plot=False, colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC'),
                         test='mann-whitney', pvalue=True, figsize=(2, 2))

#################################### all motility features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'T-cell': 'Tfh'}})
df_.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df_.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']

for idx in [0,1,2]:
    coloc_feature = coloc_features[idx]
    if idx == 0:
        custom_range = (0, 40)
        xlabel = 'Distance to DZ (µm)'
    elif idx == 1:
        custom_range = (0,20)
        xlabel = 'Distance to sLZ (µm)'
    elif idx == 2:
        custom_range = (0,40)
        xlabel = 'Distance to dLZ (µm)'
    draw_lineplot_by_custom_ranges(df_, path, folder_name='motility_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                                   condition_name='Type', custsom_range=custom_range, stepsize=4, range_feature=coloc_feature,
                                       color_list=['#CC6677', '#6699CC'], marker_list=['o', '^', ], figsize=(4,4), x_label=xlabel,
                                   replace_keys=None, pvalue=True, test='mann-whitney')

#################################### all approach / departure motility features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'T-cell': 'Tfh'}})
df_.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df_.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']

coloc_feature = coloc_features[2]

df_approach = df_[df_['quality_Core_approach_times']>=12].reset_index(drop=True)
print(df_[df_['quality_Core_approach_times']>=12].shape[0])
draw_lineplot_by_custom_ranges(df_approach, path, folder_name='approach_motility_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                               condition_name='Type', custsom_range=(0, 20), stepsize=4, range_feature=coloc_feature,
                                   color_list=['#CC6677', '#6699CC'], marker_list=['o', '^', ], figsize=(4,4), x_label='Distance to dLZ (µm)',
                               replace_keys=None, pvalue=True, test='mann-whitney')

#################################### all interaction features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'T-cell': 'Tfh'}})
df_.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone'])

coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']

for idx in [0,1,2]:
    coloc_feature = coloc_features[idx]
    if idx == 0:
        custom_range = (0, 40)
        xlabel = 'Distance to DZ (µm)'
    elif idx == 1:
        custom_range = (0,20)
        xlabel = 'Distance to sLZ (µm)'
    elif idx == 2:
        custom_range = (0,40)
        xlabel = 'Distance to dLZ (µm)'

    draw_lineplot_by_custom_ranges(df, path, folder_name='interaction_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                                   condition_name='Type', custsom_range=custom_range, stepsize=4, range_feature=coloc_feature,
                                       color_list=['#CC6677', '#6699CC'], marker_list=['o', '^', ], figsize=(4,4), x_label=xlabel,
                                   replace_keys=None, pvalue=True, test='mann-whitney')





# #Todo
# ############# Plot Close vs Far contour map for each cell type ###############
# path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Cornell LN Spleen\Analysis\Figure1. Dynamic Behavior of T and WT\FDC dependent behavior\\'
#
# for cell_type in ['T-cell', 'wt_B-cell']:
#
#     fig, ax = plt.subplots(figsize=(2, 2))  # 2 inch by 2 inch
#
#     font = {'family': 'arial',
#             'weight': 'normal',
#             'size': 8}
#     matplotlib.rc('font', **font)
#     matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
#     matplotlib.rcParams['lines.linewidth'] = 1
#
#     color_list = ['Greys', 'Reds', 'Greens', 'Blues', 'Oranges', 'Purples', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
#                   'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'plasma']
#     x_name='PC1'
#     y_name='PC2'
#     num_contours=6
#     bin_num=50
#
#     contours = []
#     groups = ['far', 'close']
#
#     x_far = df[(df['Type']==cell_type) & (df['Average_distance_to_FDC'] >= 20)][x_name]
#     y_far = df[(df['Type']==cell_type) & (df['Average_distance_to_FDC'] >= 20)][y_name]
#
#     x_close = df[(df['Type']==cell_type) & (df['Average_distance_to_FDC'] <= 5)][x_name]
#     y_close = df[(df['Type']==cell_type) & (df['Average_distance_to_FDC'] <= 5)][y_name]
#
#     xmin = math.floor(df[x_name].min()) - 1
#     xmax = math.ceil(df[x_name].max()) + 1
#     ymin = math.floor(df[y_name].min()) - 1
#     ymax = math.ceil(df[y_name].max()) + 1
#
#     kde_coordinate_far = np.vstack([x_far, y_far])  # shape = (2(dimension), number of points)
#     kde_coordinate_close = np.vstack([x_close, y_close])  # shape = (2(dimension), number of points)
#     if (kde_coordinate_far.shape[1] <= 2) or (kde_coordinate_close.shape[1] <= 2):  # if there is only few points, it cannot calculate gaussian kde
#         raise ValueError('Number of points should be greater than 2 to create contour')
#     else:
#         kde_far = scipy.stats.gaussian_kde(kde_coordinate_far)  # Define kernel (bandwidth by Scott's Rule)
#         kde_close = scipy.stats.gaussian_kde(kde_coordinate_close)  # Define kernel (bandwidth by Scott's Rule)
#
#         # evaluate on a regular grid
#         xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
#         ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
#         Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
#         # Xgrid , Ygrid = (bin_num,bin_num) 2d array
#         # Xgrid[i] = xgrid coordinate(divide by 100 from -x ~ +x` and repeat in row direction)
#         # Ygrid[:,i] = ygrid coordinate(divide by 100 from -y ~ +y` and repeat in column direction)
#         Z_far = kde_far.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
#         Z_close = kde_close.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
#         # Xgrid.ravel() = bin_num^2 shape 1d vector (vector that repeat -x ~ +x`, -x ~ +x` bin_num times)
#         # np.vstack() = (2,bin_num^2) 2d array linspaced coordinates
#         # Z = (10000,) 1d vector
#         pdf_far = Z_far.reshape(Xgrid.shape)
#         pdf_close = Z_close.reshape(Xgrid.shape)
#         contour_far = ax.contour(Xgrid, Ygrid, pdf_far,
#                              # colors='red',
#                              linewidths=1,
#                              linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
#                              # label=group,
#                              cmap=color_list[0],
#                              origin='lower',
#                              levels=num_contours,
#                              )
#         contour_close = ax.contour(Xgrid, Ygrid, pdf_close,
#                              # colors='red',
#                              linewidths=1,
#                              linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
#                              # label=group,
#                              cmap=color_list[1],
#                              origin='lower',
#                              levels=num_contours,
#                              )
#         contours.append(contour_far)
#         contours.append(contour_close)
#
#     format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
#     ax.legend(handles=[contour.legend_elements()[0][-1] for contour in contours], labels=groups,
#               bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0, fontsize=3, frameon=False, markerscale=0.3)
#     #plt.title('%s NOI vs PI' % cell_type, fontsize=25)
#     plt.xlim(xmin, xmax)
#     plt.ylim(ymin, ymax)
#
#     plt.savefig(path + '%s_far vs close.png' % cell_type, dpi=300)
#     if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'svg/')
#     plt.savefig(path + 'svg/%s_far vs close.svg' % cell_type)
#     plt.close()
#     plt.clf()
#
#
# ############# Plot Close vs Far cross correlation for each cell type ###############
#
# df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT', 'T-cell': 'Tfh'}})
# draw_cluster_distribution_heatmap(df_, path, file_name='tskmeans_type_heatmap', condition_name='Type', cluster_type='tskmeans')
#
# ############# Plot Close vs Far cross correlation for each cell type ###############
# font = {'family': 'arial',
#             'weight': 'normal',
#             'size': 8}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
# matplotlib.rcParams['lines.linewidth'] = 1
#
# condition_name = 'Type'
# cluster_type = 'tskmeans'
# condition = 'tp_interaction_FDC'
# groups = ['NOI', 'PC']
#
# df_corr_data = pd.DataFrame()
# group_clones=[]
# for group in ['far', 'close']:
#     corrcoef = []
#     #aaa = df[df[condition] == group]
#     if group == 'far':
#         aaa = df[(df['Average_distance_to_FDC'] >= 20)]
#     elif group == 'close':
#         aaa = df[(df['Average_distance_to_FDC'] <= 5)]
#
#     group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
#     group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
#     group_clone = group_clone.unstack(level=0)
#     group_clone[np.isnan(group_clone)] = 0
#     group_clone_T = group_clone.T
#     for cluster in list(pd.unique(df[cluster_type])):
#         if cluster in group_clone_T.columns:
#             continue
#         else:
#             group_clone_T.insert(loc=int(cluster), column=cluster, value=[0, 0, 0])
#     group_clone = group_clone_T.T
#
#     for column in group_clone.columns:
#         group_clone.rename(columns={column:column+'_%s'%group}, inplace=True)
#     group_clones.append(group_clone)
#     df_corr_data = pd.concat([df_corr_data, group_clone], axis=1)
#
# df_corr = df_corr_data.corr()
#
# df_corr.columns.name = None # Remove name of column = 'Type'
# df_corr.index.name = None # Remove name of index = 'Type'
#
#
# rename_keys = {'wt_B-cell_far': 'WT far', 'wt_B-cell_close': 'WT close',
#                'T-cell_far': 'Tfh far', 'T-cell_close': 'Tfh close',}
# df_corr.rename(columns=rename_keys, inplace=True)
# df_corr.rename(index=rename_keys, inplace=True)
#
# mask = np.triu(df_corr) # Mask for only lower triangle
#
# mask = mask[1:, :-1]  # mask that doesn't contain very top left and very bottom right block
# corr = df_corr.iloc[1:,:-1].copy()  # df that doesn't contain very top left and very bottom right block
#
# fig, ax = plt.subplots(figsize=(2, 2))
#
# ax = sns.heatmap(corr, annot=True, mask=mask, cmap='OrRd', alpha=0.9, linewidths=1, linecolor='white', fmt=".2f", square= True,
#                  annot_kws={'size': 4, 'weight':'normal'},
#                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})
#
# cax = ax.figure.axes[-1]  # colorbar
# cax.tick_params(labelsize=4)  # fontsize of tick label
# cax.yaxis.label.set_size(6)  # fontsize of color bar y label
#
# yticks = [i for i in corr.index]
# xticks = [i for i in corr.columns]
# plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=4, weight='normal')
# plt.xticks(plt.xticks()[0], labels=xticks, fontsize=4, rotation=35, rotation_mode='anchor', ha='right', weight='normal')
#
# plt.savefig(path+'far vs close correlation.png', dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/far vs close correlation.svg', bbox_inches='tight')
#
# plt.close()
# plt.clf()
#
# ############# Plot Close vs Far fraction of cluster for each cell type ###############
# vmax=70
# colors = ('#CC6677', '#6699CC')
#
# font = {'family': 'arial',
#             'weight': 'normal',
#             'size': 8}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
# matplotlib.rcParams['lines.linewidth'] = 1
#
# for group_clone in group_clones:
#     for i, cond in enumerate(list(group_clone.columns)):
#         fig, ax = plt.subplots(figsize=(2, 2))
#         ax = sns.barplot(x=list(group_clone[cond].index), y=list(group_clone[cond]), capsize=0.5, edgecolor='0.2', lw=1, errwidth=1, color=colors[i])
#
#         for axis in ['bottom', 'left']:
#             ax.spines[axis].set_linewidth(1)
#             ax.spines[axis].set_color('0.2')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#
#         #sns.barplot(x=list(group_clone[cond].index), y=list(group_clone[cond]), color=colors[i])
#         #plt.xlabel('%s' % cluster_type)
#         plt.ylabel('Occurence (%)')
#         plt.ylim(0, vmax)
#         plt.savefig(path + '%s_distribution.png' % cond, dpi=300, bbox_inches='tight')
#         if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#             os.makedirs(path + 'svg/')
#         plt.savefig(path + 'svg/%s_distribution.svg' % cond, bbox_inches='tight')
#         plt.clf()
#         plt.close()
#
# ############# Shannon entropy of Close vs Far for each video ###############
# entropies_far = {'T-cell': [], 'wt_B-cell':[]}
# group_name = 'Video'
# groups = np.unique(df[group_name])
#
# for group in groups:
#     df_part = df[(df[group_name]==group)&(df['Average_distance_to_FDC'] >= 20)]
#     if df_part.shape[0] == 0:
#         continue
#     entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='tskmeans')
#     for type in entropy:
#         if entropy[type]==0:
#             continue
#         else:
#             entropies_far[type].append(entropy[type])
#
# entropies_close = {'T-cell': [], 'wt_B-cell':[]}
# for group in groups:
#     df_part = df[(df[group_name]==group)&(df['Average_distance_to_FDC'] <= 5)]
#     if df_part.shape[0] == 0:
#         continue
#     entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='tskmeans')
#     for type in entropy:
#         if entropy[type]==0:
#             continue
#         else:
#             entropies_close[type].append(entropy[type])
#
# entropies = {}
# for cell_type in ['T-cell', 'wt_B-cell']:
#     entropies = {'%s_far'%cell_type:entropies_far[cell_type], '%s_close'%cell_type:entropies_close[cell_type]}
#     rename_keys = {'wt_B-cell_far': 'WT far',
#                    'wt_B-cell_close': 'WT close', 'T-cell_far': 'Tfh far', 'T-cell_close': 'Tfh close', }
#
#     entropies_ = {rename_keys.get(k, k): v for (k, v) in entropies.items()}
#     draw_custom_bar_plot(entropies_, path, file_name='entropy of far vs close for %s' %cell_type, colors=('#888888', '#CC6677'),
#                          strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))
#
# ###################### Plot Close vs Far motility feature violin plot for T cell and WT  ############################
#
# if not os.path.isdir(path + 'close vs far motility box plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'close vs far motility box plot/')
#
# feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
#
# for feature_name in feature_list:
#     condition_name = 'Type'
#     dataset={}
#     for cell_type in ['wt_B-cell', 'T-cell']:
#         for interaction in ['far','close']:
#             if interaction == 'far':
#                 data = df[(df['Average_distance_to_FDC'] >= 20)&(df[condition_name] == cell_type)][feature_name]
#             elif interaction == 'close':
#                 data = df[(df['Average_distance_to_FDC'] <= 5)&(df[condition_name] == cell_type)][feature_name]
#
#             dataset[cell_type+'_'+str(interaction)] = np.array(data)
#
#     rename_keys = {'wt_B-cell_far': 'WT far', 'wt_B-cell_close': 'WT close', 'T-cell_far': 'Tfh far', 'T-cell_close': 'Tfh close'}
#     dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}
#
#     draw_custom_violin_plot(dataset_renamed, path + 'close vs far motility box plot/', file_name=feature_name,
#                             colors=('#6699CC', '#6699CC', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))
#
# #################################### Volcano plot of Close vs Far motility features ####################################
# feature_list = df.columns[130:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z', 'phi'])
# for cell_type in ['T-cell', 'wt_B-cell']:
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
#
# ######################## Close vs Far interaction features ###########################
# if not os.path.isdir(path + 'close vs away int feature violin plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'close vs away int feature violin plot/')
#
# df.columns.get_loc('FDC_total_distance')
# df.columns.get_loc('tp_interaction_mt_B-cell')
#
# k = df.iloc[:,324:417].isnull().any()
# null_features = k.index[k==True]
# feature_list = df.columns[324:417].drop(null_features).drop(['PC1', 'PC2', 'tskmeans'])
# p_values_each_feature = {}
# for feature_name in feature_list:
#     datasets={}
#     for cell_type in ['wt_B-cell', 'T-cell']:
#         for interaction_type in ['Away', 'Close']:
#             if interaction_type == 'Away':
#                 data = df[(df['Type'] == cell_type) & (df['Average_distance_to_FDC'] >= 20) & (df['tp_interaction_FDC'] <= 2)][feature_name]
#             elif interaction_type == 'Close':
#                 data = df[(df['Type'] == cell_type) & (df['Average_distance_to_FDC'] <= 5) & (df['tp_interaction_FDC'] <= 2)][feature_name]
#             datasets[cell_type+' '+interaction_type] = np.array(data)
#
#     values = flatten_nested_dict(datasets)
#     if np.isnan(values).any() == True:  # Check at least one nan
#         continue
#     elif np.isfinite(values).all() == False:  # Check everything is not inf
#         continue
#
#     draw_custom_violin_plot(datasets, path + 'close vs away int feature violin plot/', file_name=feature_name,
#                             colors=('#6699CC', '#6699CC', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True,
#                             figsize=(2, 2))
#
# #################################### T cell, WT avg speed wrt avg FDC distance ####################################
# FDC_dist_range = (0,20)
#
# mean_speed_dataset = {}
# std_speed_dataset = {}
# mean_angle_dataset = {}
# std_angle_dataset = {}
#
# for cell_type in ['T-cell', 'wt_B-cell']:
#     df_part = df[df['Type'] == cell_type]
#     mean_speeds = []
#     std_speeds = []
#     mean_angles = []
#     std_angles = []
#     for i in np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1):
#         if i == FDC_dist_range[1]:
#             speeds = df_part[df_part['Average_distance_to_FDC'] >= i]['avg_speed'].values
#             angles = df_part[df_part['Average_distance_to_FDC'] >= i]['avg_angle'].values
#         else:
#             speeds = df_part[(df_part['Average_distance_to_FDC'] >= i) & (df_part['Average_distance_to_FDC'] < i + 1)][
#                 'avg_speed'].values
#             angles = df_part[(df_part['Average_distance_to_FDC'] >= i) & (df_part['Average_distance_to_FDC'] < i + 1)][
#                 'avg_angle'].values
#         mean_speeds.append(np.mean(speeds))
#         if speeds.shape[0]==0:
#             print(i)
#         std_speeds.append(np.std(speeds))
#         mean_angles.append(np.mean(angles))
#         std_angles.append(np.std(angles))
#
#     mean_speed_dataset[cell_type] = np.array(mean_speeds)
#     std_speed_dataset[cell_type] = np.array(std_speeds)
#     mean_angle_dataset[cell_type] = np.array(mean_angles)
#     std_angle_dataset[cell_type] = np.array(std_angles)
#
# replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'WT'}
# mean_speed_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_speed_dataset.items() }
# std_speed_dataset = {replace_keys.get(k, k):v  for (k,v) in std_speed_dataset.items() }
# mean_angle_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_angle_dataset.items() }
# std_angle_dataset = {replace_keys.get(k, k):v  for (k,v) in std_angle_dataset.items() }
#
# font = {'family': 'arial',
#             'weight': 'normal',
#             'size': 16}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
# matplotlib.rcParams['lines.linewidth'] = 2
#
# fig, ax = plt.subplots(figsize=(4,4))
# ax = sns.lineplot(data=mean_speed_dataset, lw=2.5, markers=['o', '^'], dashes=False, markersize=8,
#              err_style='bars', palette=['#CC6677', '#6699CC'])
# ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_speed_dataset['Tfh'], 0.1*std_speed_dataset['Tfh'], color='#CC6677',
#            capsize=3, capthick=1, elinewidth=1.5)
# ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_speed_dataset['WT'], 0.1*std_speed_dataset['WT'], color='#6699CC',
#            capsize=3, capthick=1, elinewidth=1.5)
#
# for axis in ['bottom', 'left']:
#     ax.spines[axis].set_linewidth(2)
#     ax.spines[axis].set_color('0.2')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# ax.tick_params(width=2, color='0.2')
#
# ax.set_xlabel('Distance to FDC (μm)', fontsize=16, weight='normal', color='0.2')
# ax.set_ylabel('Average speed (μm/min)', fontsize=16, weight='normal', color='0.2')
# plt.xticks(fontsize=16, color='0.2',weight='normal', )
# plt.yticks(fontsize=16, color='0.2', weight='normal', )
#
# plt.legend(frameon=False, prop={'weight':'normal', 'size':12}, labelcolor='0.2')
# # plt.ylabel('%s' % feature_name, fontsize=4)
# plt.savefig(path + 'Avg speed wrt avg FDC distance.png', dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/Avg speed wrt avg FDC distance.svg', bbox_inches='tight')
# plt.clf()
# plt.close()
#
# #################################### T cell, WT avg angle wrt avg FDC distance ####################################
#
# font = {'family': 'arial',
#             'weight': 'normal',
#             'size': 16}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
# matplotlib.rcParams['lines.linewidth'] = 2
#
# fig, ax = plt.subplots(figsize=(4,4))
# ax = sns.lineplot(data=mean_angle_dataset, lw=2.5, markers=['o', '^'], dashes=False, markersize=8,
#              err_style='bars', palette=['#CC6677', '#6699CC'])
# ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_angle_dataset['Tfh'], 0.1*std_angle_dataset['Tfh'], color='#CC6677',
#            capsize=3, capthick=1, elinewidth=1.5)
# ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_angle_dataset['WT'], 0.1*std_angle_dataset['WT'], color='#6699CC',
#            capsize=3, capthick=1, elinewidth=1.5)
#
# for axis in ['bottom', 'left']:
#     ax.spines[axis].set_linewidth(2)
#     ax.spines[axis].set_color('0.2')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# ax.tick_params(width=2, color='0.2')
#
# ax.set_xlabel('Distance to FDC (μm)', fontsize=16, weight='normal', color='0.2')
# ax.set_ylabel('Average turning angle (rad/min)', fontsize=16, weight='normal', color='0.2')
# plt.xticks(fontsize=16, color='0.2',weight='normal', )
# plt.yticks(fontsize=16, color='0.2', weight='normal', )
#
# plt.legend(frameon=False, prop={'weight':'normal', 'size':12}, labelcolor='0.2')
# # plt.ylabel('%s' % feature_name, fontsize=4)
# plt.savefig(path + 'Avg angle wrt avg FDC distance.png', dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/Avg angle wrt avg FDC distance.svg', bbox_inches='tight')
# plt.clf()
# plt.close()
#
#
# #################################### T cell, WT all motility features wrt avg FDC distance ####################################
# FDC_dist_range = (0,20)
#
# feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
#
# for feature_name in feature_list:
#     mean_dataset = {}
#     std_dataset = {}
#
#     for cell_type in ['T-cell', 'wt_B-cell']:
#         df_part = df[df['Type'] == cell_type]
#         means = []
#         stds = []
#
#         for i in np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1):
#             if i == FDC_dist_range[1]:
#                 values = df_part[df_part['Average_distance_to_FDC'] >= i][feature_name].values
#             else:
#                 values = df_part[(df_part['Average_distance_to_FDC'] >= i) & (df_part['Average_distance_to_FDC'] < i + 1)][feature_name].values
#
#             #means.append(np.mean(values))
#             means.append(np.median(values))
#             stds.append(np.std(values))
#
#         mean_dataset[cell_type] = np.array(means)
#         std_dataset[cell_type] = np.array(stds)
#
#
#     replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'WT'}
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
#                  err_style='bars', palette=['#CC6677', '#6699CC'])
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['Tfh'], 0.1*std_dataset['Tfh'], color='#CC6677',
#                capsize=3, capthick=1, elinewidth=1.5)
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['WT'], 0.1*std_dataset['WT'], color='#6699CC',
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
#     ax.set_xlabel('Distance to FDC (μm)', fontsize=16, weight='normal', color='0.2')
#     #ax.set_ylabel('Average speed (μm/min)', fontsize=16, weight='normal', color='0.2')
#     plt.xticks(fontsize=16, color='0.2',weight='normal', )
#     plt.yticks(fontsize=16, color='0.2', weight='normal', )
#
#     plt.legend(frameon=False, prop={'weight':'normal', 'size':12}, labelcolor='0.2')
#     # plt.ylabel('%s' % feature_name, fontsize=4)
#
#     if not os.path.isdir(path + 'feature_wrt_FDC_distance/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'feature_wrt_FDC_distance/')
#
#     plt.savefig(path + 'feature_wrt_FDC_distance/%s.png'%feature_name, dpi=300, bbox_inches='tight')
#
#     if not os.path.isdir(path + 'feature_wrt_FDC_distance/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'feature_wrt_FDC_distance/svg/')
#     plt.savefig(path + 'feature_wrt_FDC_distance/svg/%s.svg'%feature_name, bbox_inches='tight')
#     plt.clf()
#     plt.close()
#
# #################################### T cell, WT all motility features wrt FDC contact time ####################################
# FDC_dist_range = (0,20)
#
# feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
#
# for feature_name in feature_list:
#     mean_dataset = {}
#     std_dataset = {}
#
#     for cell_type in ['T-cell', 'wt_B-cell']:
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
#             #means.append(np.mean(values))
#             means.append(np.median(values))
#             stds.append(np.std(values))
#
#         mean_dataset[cell_type] = np.array(means)
#         std_dataset[cell_type] = np.array(stds)
#
#
#     replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'WT'}
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
#                  err_style='bars', palette=['#CC6677', '#6699CC'])
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['Tfh'], 0.1*std_dataset['Tfh'], color='#CC6677',
#                capsize=3, capthick=1, elinewidth=1.5)
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['WT'], 0.1*std_dataset['WT'], color='#6699CC',
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
#     ax.set_xlabel('Contact time with FDC', fontsize=16, weight='normal', color='0.2')
#     #ax.set_ylabel('Average speed (μm/min)', fontsize=16, weight='normal', color='0.2')
#     plt.xticks(fontsize=16, color='0.2',weight='normal', )
#     plt.yticks(fontsize=16, color='0.2', weight='normal', )
#
#     plt.legend(frameon=False, prop={'weight':'normal', 'size':12}, labelcolor='0.2')
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
#
# #################################### T cell, WT all interaction features wrt FDC distance ####################################
# FDC_dist_range = (0,20)
# feature_list = df.columns[324:]
#
# for feature_name in feature_list:
#     mean_dataset = {}
#     std_dataset = {}
#
#     for cell_type in ['T-cell', 'wt_B-cell']:
#         df_part = df[df['Type'] == cell_type]
#         means = []
#         stds = []
#
#         for i in np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1):
#             if i == FDC_dist_range[1]:
#                 values = df_part[df_part['Average_distance_to_FDC'] >= i][feature_name].values
#             else:
#                 values = df_part[(df_part['Average_distance_to_FDC'] >= i) & (df_part['Average_distance_to_FDC'] < i + 1)][feature_name].values
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
#     replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'WT'}
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
#                  err_style='bars', palette=['#CC6677', '#6699CC'])
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['Tfh'], 0.1*std_dataset['Tfh'], color='#CC6677',
#                capsize=3, capthick=1, elinewidth=1.5)
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['WT'], 0.1*std_dataset['WT'], color='#6699CC',
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
#     ax.set_xlabel('Distance to FDC (μm)', fontsize=16, weight='normal', color='0.2')
#     #ax.set_ylabel('Average speed (μm/min)', fontsize=16, weight='normal', color='0.2')
#     plt.xticks(fontsize=16, color='0.2',weight='normal', )
#     plt.yticks(fontsize=16, color='0.2', weight='normal', )
#
#     plt.legend(frameon=False, prop={'weight':'normal', 'size':12}, labelcolor='0.2')
#     # plt.ylabel('%s' % feature_name, fontsize=4)
#
#     if not os.path.isdir(path + 'int_feature_wrt_FDC_distance/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'int_feature_wrt_FDC_distance/')
#
#     plt.savefig(path + 'int_feature_wrt_FDC_distance/%s.png'%feature_name, dpi=300, bbox_inches='tight')
#
#     if not os.path.isdir(path + 'int_feature_wrt_FDC_distance/svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'int_feature_wrt_FDC_distance/svg/')
#     plt.savefig(path + 'int_feature_wrt_FDC_distance/svg/%s.svg'%feature_name, bbox_inches='tight')
#     plt.clf()
#     plt.close()
#
#
# #################################### T cell, WT all interaction features wrt FDC contact time ####################################
# FDC_dist_range = (0,20)
# feature_list = df.columns[324:]
#
# for feature_name in feature_list:
#     mean_dataset = {}
#     std_dataset = {}
#
#     for cell_type in ['T-cell', 'wt_B-cell']:
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
#     replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'WT'}
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
#                  err_style='bars', palette=['#CC6677', '#6699CC'])
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['Tfh'], 0.1*std_dataset['Tfh'], color='#CC6677',
#                capsize=3, capthick=1, elinewidth=1.5)
#     ax.errorbar(range(FDC_dist_range[0], FDC_dist_range[1]+1), mean_dataset['WT'], 0.1*std_dataset['WT'], color='#6699CC',
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
#     ax.set_xlabel('Contact time with FDC', fontsize=16, weight='normal', color='0.2')
#     #ax.set_ylabel('Average speed (μm/min)', fontsize=16, weight='normal', color='0.2')
#     plt.xticks(fontsize=16, color='0.2',weight='normal', )
#     plt.yticks(fontsize=16, color='0.2', weight='normal', )
#
#     plt.legend(frameon=False, prop={'weight':'normal', 'size':12}, labelcolor='0.2')
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
#
#
#
#
# #draw_umap_space(df, path, file_name='distance_interaction_FDC', condition_name='Type', label_name='pseudo_Label', colors=('#CC6677', '#6699CC'), x_name='tp_interaction_FDC', y_name='Average_distance_to_FDC', dot_size=0.07)
#
#
#
#
#
#
#
# #################################### T cell, WT all features wrt avg FDC distance scatter plot ####################################
# from features.interaction import DistanceSignal, OverlapSignal
# duration=20
# _, _, B_distances = to_timeseries_fast(df_duration, duration, feature_name='Shortest_Distance_to_Surfaces_Surfaces=wt_B-cell')
# _, _, B_overlap = to_timeseries_fast(df_duration, duration, feature_name='Overlapped_Volume_Ratio_to_Surfaces_Surfaces=wt_B-cell')
#
# feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
# FDC_over = OverlapSignal(B_overlap)
# df_overlap = FDC_over.extract_features(feature_list)
#
#
# for column in df_overlap.columns:
#     df_overlap.rename(columns={column:'B_'+column}, inplace=True)
#
# #feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
# df_included = pd.concat([df, df_overlap], axis=1)
#
# df_part = df_included[df_included['Type']=='T-cell'].reset_index(drop=True)
#
#
# fig, ax = plt.subplots()
# plt.scatter(df_part['avg_angle'], df_part['Average_distance_to_FDC'], s=1, c=df['B_contact_times'], cmap=plt.cm.get_cmap('coolwarm'))
# #plt.scatter(df_part['avg_angle'], df_part['Average_distance_to_FDC'], s=1, c=df_part['B_contact_persistences'], cmap=plt.cm.get_cmap('coolwarm'))
#
#
# plt.show()
# plt.close()
# plt.clf()
#
#
# #################################### T cell, WT avg speed wrt avg FDC distance ####################################
# FDC_dist_range = (0.4, 2.1)
#
# mean_speed_dataset = {}
# std_speed_dataset = {}
# mean_angle_dataset = {}
# std_angle_dataset = {}
#
# for cell_type in ['far', 'close']:
#     if cell_type == 'far':
#         df_part = df_included[(df_included['Type'] == 'T-cell')&(df_included['Average_distance_to_FDC'] >= 20)]
#     elif cell_type == 'close':
#         df_part = df_included[(df_included['Type'] == 'T-cell')&(df_included['Average_distance_to_FDC'] <= 5)]
#
#     mean_speeds = []
#     std_speeds = []
#     mean_angles = []
#     std_angles = []
#     for i in np.arange(FDC_dist_range[0], FDC_dist_range[1]+0.1, 0.1):
#         if i == FDC_dist_range[1]:
#             speeds = df_part[df_part['avg_angle'] >= i]['B_contact_persistences'].values
#             angles = df_part[df_part['avg_angle'] >= i]['B_contact_times'].values
#         else:
#             speeds = df_part[(df_part['avg_angle'] >= i) & (df_part['avg_angle'] < i + 1)]['B_contact_persistences'].values
#             angles = df_part[(df_part['avg_angle'] >= i) & (df_part['avg_angle'] < i + 1)]['B_contact_times'].values
#         mean_speeds.append(np.mean(speeds))
#         if speeds.shape[0]==0:
#             print(i)
#         std_speeds.append(np.std(speeds))
#         mean_angles.append(np.mean(angles))
#         std_angles.append(np.std(angles))
#
#     mean_speed_dataset[cell_type] = np.array(mean_speeds)
#     std_speed_dataset[cell_type] = np.array(std_speeds)
#     mean_angle_dataset[cell_type] = np.array(mean_angles)
#     std_angle_dataset[cell_type] = np.array(std_angles)
#
# replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'WT'}
# mean_speed_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_speed_dataset.items() }
# std_speed_dataset = {replace_keys.get(k, k):v  for (k,v) in std_speed_dataset.items() }
# mean_angle_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_angle_dataset.items() }
# std_angle_dataset = {replace_keys.get(k, k):v  for (k,v) in std_angle_dataset.items() }
#
# font = {'family': 'arial',
#             'weight': 'normal',
#             'size': 16}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
# matplotlib.rcParams['lines.linewidth'] = 2
#
# fig, ax = plt.subplots(figsize=(4,4))
# ax = sns.lineplot(x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+0.1, 0.1), data=mean_speed_dataset, lw=2.5, markers=['o', '^'], dashes=False, markersize=8,
#              err_style='bars', palette=['#CC6677', '#6699CC'])
# ax.errorbar(np.arange(FDC_dist_range[0], FDC_dist_range[1]+0.1, 0.1), mean_speed_dataset['far'], 0.1*std_speed_dataset['far'], color='#CC6677',
#            capsize=3, capthick=1, elinewidth=1.5)
# ax.errorbar(np.arange(FDC_dist_range[0], FDC_dist_range[1]+0.1, 0.1), mean_speed_dataset['close'], 0.1*std_speed_dataset['close'], color='#6699CC',
#            capsize=3, capthick=1, elinewidth=1.5)
#
# for axis in ['bottom', 'left']:
#     ax.spines[axis].set_linewidth(2)
#     ax.spines[axis].set_color('0.2')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# ax.tick_params(width=2, color='0.2')
#
# ax.set_xlabel('Average turning angle (rad/min)', fontsize=16, weight='normal', color='0.2')
# ax.set_ylabel('B cell interaction score', fontsize=16, weight='normal', color='0.2')
# plt.xticks(fontsize=16, color='0.2',weight='normal', )
# plt.yticks(fontsize=16, color='0.2', weight='normal', )
#
# plt.legend(frameon=False, prop={'weight':'normal', 'size':12}, labelcolor='0.2')
# # plt.ylabel('%s' % feature_name, fontsize=4)
# plt.savefig(path + 'Avg angle wrt B cell contact.png', dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/Avg angle wrt B cell contact.svg', bbox_inches='tight')
# plt.clf()
# plt.close()
#
# #################################### T cell, WT avg speed wrt avg FDC distance ####################################
# FDC_dist_range = (1, 20)
#
# mean_speed_dataset = {}
# std_speed_dataset = {}
# mean_angle_dataset = {}
# std_angle_dataset = {}
#
# for cell_type in ['far', 'close']:
#     if cell_type == 'far':
#         df_part = df_included[(df_included['Type'] == 'T-cell')&(df_included['Average_distance_to_FDC'] >= 20)]
#     elif cell_type == 'close':
#         df_part = df_included[(df_included['Type'] == 'T-cell')&(df_included['Average_distance_to_FDC'] <= 5)]
#
#     mean_speeds = []
#     std_speeds = []
#     mean_angles = []
#     std_angles = []
#     for i in np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1):
#         if i == FDC_dist_range[1]:
#             speeds = df_part[df_part['avg_speed'] >= i]['B_contact_persistences'].values
#             angles = df_part[df_part['avg_speed'] >= i]['B_contact_times'].values
#         else:
#             speeds = df_part[(df_part['avg_speed'] >= i) & (df_part['avg_angle'] < i + 1)]['B_contact_persistences'].values
#             angles = df_part[(df_part['avg_speed'] >= i) & (df_part['avg_angle'] < i + 1)]['B_contact_times'].values
#         mean_speeds.append(np.mean(speeds))
#         if speeds.shape[0]==0:
#             print(i)
#         std_speeds.append(np.std(speeds))
#         mean_angles.append(np.mean(angles))
#         std_angles.append(np.std(angles))
#
#     mean_speed_dataset[cell_type] = np.array(mean_speeds)
#     std_speed_dataset[cell_type] = np.array(std_speeds)
#     mean_angle_dataset[cell_type] = np.array(mean_angles)
#     std_angle_dataset[cell_type] = np.array(std_angles)
#
# replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'WT'}
# mean_speed_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_speed_dataset.items() }
# std_speed_dataset = {replace_keys.get(k, k):v  for (k,v) in std_speed_dataset.items() }
# mean_angle_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_angle_dataset.items() }
# std_angle_dataset = {replace_keys.get(k, k):v  for (k,v) in std_angle_dataset.items() }
#
# font = {'family': 'arial',
#             'weight': 'normal',
#             'size': 16}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
# matplotlib.rcParams['lines.linewidth'] = 2
#
# fig, ax = plt.subplots(figsize=(4,4))
# ax = sns.lineplot(x=np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1), data=mean_speed_dataset, lw=2.5, markers=['o', '^'], dashes=False, markersize=8,
#              err_style='bars', palette=['#CC6677', '#6699CC'])
# ax.errorbar(np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1), mean_speed_dataset['far'], 0.1*std_speed_dataset['far'], color='#CC6677',
#            capsize=3, capthick=1, elinewidth=1.5)
# ax.errorbar(np.arange(FDC_dist_range[0], FDC_dist_range[1]+1, 1), mean_speed_dataset['close'], 0.1*std_speed_dataset['close'], color='#6699CC',
#            capsize=3, capthick=1, elinewidth=1.5)
#
# for axis in ['bottom', 'left']:
#     ax.spines[axis].set_linewidth(2)
#     ax.spines[axis].set_color('0.2')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# ax.tick_params(width=2, color='0.2')
#
# ax.set_xlabel('Average speed (um/min)', fontsize=16, weight='normal', color='0.2')
# ax.set_ylabel('B cell interaction score', fontsize=16, weight='normal', color='0.2')
# plt.xticks(fontsize=16, color='0.2',weight='normal', )
# plt.yticks(fontsize=16, color='0.2', weight='normal', )
#
# plt.legend(frameon=False, prop={'weight':'normal', 'size':12}, labelcolor='0.2')
# # plt.ylabel('%s' % feature_name, fontsize=4)
# plt.savefig(path + 'Avg speed wrt B cell contact.png', dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/Avg speed wrt B cell contact.svg', bbox_inches='tight')
# plt.clf()
# plt.close()