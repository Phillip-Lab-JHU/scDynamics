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

############################### Tfh + GCB space ################################
duration=20
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\\'
df_features_all = pd.read_parquet(path+'all_features_%s.parquet'%duration)
df_duration_all = pd.read_parquet(path+'traj_duration_%s.parquet'%duration)

#################################### Without inhibition GCB + inhibition (Figure 2, 3, 4) ####################################
df_features = df_features_all[(df_features_all['Exp']=='Exp1')|(df_features_all['Exp']=='Exp2')|(df_features_all['Exp']=='Exp3')].reset_index(drop=True)
df_duration = df_duration_all[(df_duration_all['Exp']=='Exp1')|(df_duration_all['Exp']=='Exp2')|(df_duration_all['Exp']=='Exp3')].reset_index(drop=True)

videos = np.unique(df_features['Video'])
df = df_features[(df_features['Video'] != videos[1])&(df_features['Video'] != videos[2])&(df_features['Video'] != videos[4])
                          &(df_features['Video'] != videos[-1])].reset_index(drop=True)

df_duration = df_duration[(df_duration['Video'] != videos[1])&(df_duration['Video'] != videos[2])&(df_duration['Video'] != videos[4])
                          &(df_duration['Video'] != videos[-1])].reset_index(drop=True)


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

df.columns.get_loc('inst_angle_cosine_similarity_entropies')
motility_data = df.iloc[:,8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)


from sklearn.decomposition import PCA
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)


from sklearn.cluster import KMeans
km = KMeans(n_clusters=9, random_state=0, init='k-means++')
# k-means++: Initialize centroids that are far away each other
kmeans_predicted = km.fit_predict(pcs)
cluster = pd.DataFrame(kmeans_predicted, columns=['kmeans'])


from umap import UMAP
__umap = UMAP(metric='euclidean', n_components=2, n_neighbors=40, min_dist=0.5, random_state=0)
pcs_array = __umap.fit_transform(pcs)
umap = pd.DataFrame(pcs_array, columns=['PC1', 'PC2'])


df = pd.concat([df, umap, cluster], axis=1)
df, replace_map = order_cluster_by_feature(df, cluster_name='kmeans', feature_name='avg_speed')

label_expanded = np.repeat(df['kmeans'], duration).reset_index(drop=True)
df_duration['kmeans'] = label_expanded

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure3-2. Tfh motility space\\'
df.to_parquet(path + 'no_inhibit_all_features_%s.parquet'%duration)
df.to_csv(path + 'no_inhibit_all_features_%s.csv'%duration, index=False)

df_duration.to_parquet(path + 'no_inhibit_all_traj_duration_%s.parquet'%duration)
df_duration.to_csv(path + 'no_inhibit_all_traj_duration_%s.csv'%duration, index=False)


####################################### Tfh cell interaction feature with WT and MT #############################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure3-2. Tfh motility space\\'
df = pd.read_parquet(path+'no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'no_inhibit_all_traj_duration_20.parquet')



for typ in ['T-cell', 'wt_B-cell', 'mt_B-cell']:
    print(typ, df[(df['Type']==typ)].shape[0])

df_T = df[df['Type']=='T-cell'].reset_index(drop=True)
df_duration_T = df_duration[df_duration['Type']=='T-cell'].reset_index(drop=True)

for typ in np.unique(df_T['Video']):
    print(typ, df_T[df_T['Video']==typ].shape[0])

duration=20
_, _, WT_overlap = to_timeseries_fast(df_duration_T, duration, feature_name='Overlapped_Volume_Ratio_to_Surfaces_Surfaces=wt_B-cell')
feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
from features.interaction import OverlapSignal
WT_over = OverlapSignal(WT_overlap)
df_overlap_WT = WT_over.extract_features(feature_list)
for column in df_overlap_WT.columns:
    df_overlap_WT.rename(columns={column:'WT_'+column}, inplace=True)

_, _, MT_overlap = to_timeseries_fast(df_duration_T, duration, feature_name='Overlapped_Volume_Ratio_to_Surfaces_Surfaces=mt_B-cell')
feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
from features.interaction import OverlapSignal
MT_over = OverlapSignal(MT_overlap)
df_overlap_MT = MT_over.extract_features(feature_list)
for column in df_overlap_MT.columns:
    df_overlap_MT.rename(columns={column:'MT_'+column}, inplace=True)

df_T = pd.concat([df_T, df_overlap_WT, df_overlap_MT], axis=1)


####################################### Quantify Tfh interaction frequency #############################################
int_time = 15
test = 'mann-whitney'

persistent_int_freqs_datasets = {}
persistent_int_freq_per_cellnumbers_datasets = {}
total_n_contacts_per_cellnumbers_datasets = {}
persistent_int_freq_per_cellcontacts_datasets = {}
low_contact_freq_per_cellnumbers_datasets = {}
# dLZ_persistent_int_freq_per_cellnumbers_datatsets = {}
# sLZ_persistent_int_freq_per_cellnumbers_datatsets = {}
videos = np.unique(df_T['Video'])
for cell_type in ['WT', 'MT']:
    persistent_int_freqs = []
    persistent_int_freq_per_cellnumbers = []
    total_n_contacts_per_cellnumbers = []
    persistent_int_freq_per_cellcontacts = []
    low_contact_freq_per_cellnumbers = []

    dLZ_persistent_int_freq_per_cellnumbers = []
    sLZ_persistent_int_freq_per_cellnumbers = []
    for video in videos:
        #if 'A' in video and cell_type == 'mt_B-cell':
        df_video = df_T[df_T['Video'] == video].reset_index(drop=True)
        if df_video.shape[0] == 0:
            continue
        data = df_video['%s_contact_times'%cell_type]
        mask = ~np.isnan(data)
        data = data[mask]

        persistent_int_freq = sum(data >= int_time)
        total_n_contact = sum(data)
        n_trajs = df_video.shape[0]

        persistent_int_freq_per_cellnumber = persistent_int_freq / n_trajs
        total_n_contacts_per_cellnumber = total_n_contact / n_trajs
        persistent_int_freq_per_cellcontact = persistent_int_freq / total_n_contact
        low_contact_freq_per_cellnumber = sum((1<=data) & (data<=5)) / n_trajs

        # df_dLZ = df_video[(df_video['Zone'] == 'dLZ')|(df_video['Zone'] == 'sLZ-dLZ')].reset_index(drop=True)
        # df_sLZ = df_video[(df_video['Zone'] == 'sLZ')].reset_index(drop=True)
        # dLZ_data = df_dLZ['T_contact_times']
        # sLZ_data = df_sLZ['T_contact_times']
        # dLZ_persistent_int_freq = sum(dLZ_data >= 15)
        # sLZ_persistent_int_freq = sum(sLZ_data >= 15)
        # dLZ_n_trajs = df_dLZ.shape[0]
        # sLZ_n_trajs = df_sLZ.shape[0]
        # dLZ_persistent_int_freq_per_cellnumber = dLZ_persistent_int_freq / dLZ_n_trajs
        # sLZ_persistent_int_freq_per_cellnumber = sLZ_persistent_int_freq / sLZ_n_trajs

        #if dLZ_n_trajs >= 20:

        if persistent_int_freq != 0:
            persistent_int_freqs.append(persistent_int_freq)
            persistent_int_freq_per_cellnumbers.append(persistent_int_freq_per_cellnumber)
            persistent_int_freq_per_cellcontacts.append(persistent_int_freq_per_cellcontact)
        #
        #     dLZ_persistent_int_freq_per_cellnumbers.append(dLZ_persistent_int_freq_per_cellnumber)
        #     sLZ_persistent_int_freq_per_cellnumbers.append(sLZ_persistent_int_freq_per_cellnumber)
        #     print(cell_type, video, dLZ_n_trajs)
        # else:
        #     print(video, persistent_int_freq)
        total_n_contacts_per_cellnumbers.append(total_n_contacts_per_cellnumber)
        low_contact_freq_per_cellnumbers.append(low_contact_freq_per_cellnumber)

    persistent_int_freqs_datasets[cell_type] = persistent_int_freqs
    persistent_int_freq_per_cellnumbers_datasets[cell_type] = persistent_int_freq_per_cellnumbers
    total_n_contacts_per_cellnumbers_datasets[cell_type] = total_n_contacts_per_cellnumbers
    persistent_int_freq_per_cellcontacts_datasets[cell_type] = persistent_int_freq_per_cellcontacts
    low_contact_freq_per_cellnumbers_datasets[cell_type] = low_contact_freq_per_cellnumbers

    # dLZ_persistent_int_freq_per_cellnumbers_datatsets[cell_type] = dLZ_persistent_int_freq_per_cellnumbers
    # sLZ_persistent_int_freq_per_cellnumbers_datatsets[cell_type] = sLZ_persistent_int_freq_per_cellnumbers



colors=('#888888', '#CC6677')
draw_custom_bar_plot(persistent_int_freqs_datasets, path, file_name='Total Tfh interaction frequency',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellnumbers_datasets, path, file_name='Tfh persistent interaction frequency per cell number',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(total_n_contacts_per_cellnumbers_datasets, path, file_name='number of Tfh contacts per cell number',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellcontacts_datasets, path, file_name='Tfh persistent interaction frequency per number of contacts',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))
draw_custom_bar_plot(low_contact_freq_per_cellnumbers_datasets, path, file_name='Tfh low contact time frequency per cell number',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))





####################################### Average Zone distance kde MT vs WT #############################################
for zone in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
    print(zone, df_T[(df_T['Zone']==zone)].shape[0])

df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT', 'T-cell': 'Tfh'}})
#df_ = df_[df_['Exp_group']=='C'].reset_index(drop=True)

replace_keys = {'wt_B-cell': 'WT', 'mt_B-cell': 'MT', 'T-cell': 'Tfh'}

coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']
ranges = [(0, 60), (0,40), (0, 100)]
names = ['Distance to DZ', 'Distance to sLZ', 'Distance to dLZ']


for idx, coloc_feature in enumerate(coloc_features):
    dataset={}
    for condition in np.unique(df_['Type']):
        data = df_[(df_['Type'] == condition)][coloc_feature] # ['FDC_contact_times']
        dataset[condition] = np.array(data)
    new_order = ['WT', 'MT', 'Tfh']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    draw_custom_violin_plot(dict_datasets, path, file_name='%s violinplot'%coloc_feature, colors = ('#888888', '#CC6677', '#6699CC'),
                                test='mann-whitney', pvalue=True, figsize=(1,2))

for idx, coloc_feature in enumerate(coloc_features):

    dataset={}
    for condition in np.unique(df_['Type']):
        data = df_[df_['Type'] == condition][coloc_feature]
        dataset[condition] = np.array(data)
    new_order = ['WT', 'MT', 'Tfh']
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

    ax = sns.kdeplot(data=dict_datasets['WT'], fill=True, linewidth=1, clip=ranges[idx], color='#888888', label='WT')
    ax = sns.kdeplot(data=dict_datasets['MT'], fill=True, linewidth=1, clip=ranges[idx], color='#CC6677', label='MT')
    ax = sns.kdeplot(data=dict_datasets['Tfh'], fill=True, linewidth=1, clip=ranges[idx], color='#6699CC', label='Tfh')

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('%s (μm)'%names[idx], fontsize=8, weight='normal', color='0.2')
    ax.set_ylabel('Density', fontsize=8, weight='normal', color='0.2')
    plt.xticks(fontsize=8, color='0.2', weight='normal')
    plt.yticks(fontsize=8, color='0.2', weight='normal')

    legend = plt.legend(frameon=False, prop = {'weight':'normal', 'size':8}, labelcolor='0.2')
    plt.savefig(path+'%s.png'%coloc_feature, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg'%coloc_feature, bbox_inches='tight')
    plt.close()
    plt.clf()


############# sLZ vs dLZ Tfh localization ###############
df_T = df[df['Type']=='T-cell'].reset_index(drop=True)
videos = np.unique(df_T['Video'])

DZs = []
DZ_sLZs = []
sLZs = []
sLZ_dLZs = []
dLZs = []
for video in videos:
    df_video = df_T[df_T['Video'] == video].reset_index(drop=True)
    if df_video.shape[0] < 30:
        continue
    DZ_portion = df_video[df_video['Zone']=='DZ'].shape[0]/df_video.shape[0]
    DZ_sLZ_portion = df_video[df_video['Zone']=='DZ-sLZ'].shape[0]/df_video.shape[0]
    sLZ_portion = df_video[df_video['Zone']=='sLZ'].shape[0]/df_video.shape[0]
    sLZ_dLZ_portion = df_video[df_video['Zone'] == 'sLZ-dLZ'].shape[0] / df_video.shape[0]
    dLZ_portion = df_video[df_video['Zone']=='dLZ'].shape[0]/df_video.shape[0]
    #persistent_int_freq_per_cellnumber = persistent_int_freq / df_video.shape[0]
    DZs.append(DZ_portion)
    DZ_sLZs.append(DZ_sLZ_portion)
    sLZs.append(sLZ_portion)
    sLZ_dLZs.append(sLZ_dLZ_portion)
    dLZs.append(dLZ_portion)

    print(df_video.shape[0])


#dataset = {'sLZ':sLZs, 'dLZ': dLZs}
dataset = {'DZ': DZs, 'sLZ':sLZs, 'dLZ': dLZs}
colors=('#888888', '#888888')
draw_custom_bar_plot(dataset, path, file_name='sLZ vs dLZ portion',
                     strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))




mean_dataset = {'DZ':np.mean(DZs)*100, 'DZ-sLZ':np.mean(DZ_sLZs)*100, 'sLZ':np.mean(sLZs)*100, 'sLZ-dLZ':np.mean(sLZ_dLZs)*100, 'dLZ':np.mean(dLZs)*100}
#std_dataset = {'DZ':np.std(DZs)*100, 'DZ-sLZ':np.std(DZ_sLZs)*100, 'sLZ':np.std(sLZs)*100, 'sLZ-dLZ':np.std(sLZ_dLZs)*100, 'dLZ':np.std(dLZs)*100}

def confidence_interval(values):
    interval = stats.norm.interval(confidence=0.95, loc=np.mean(values), scale=stats.sem(values))

    error = np.mean(values) - interval[0]
    return error

ci_dataset = {'DZ':confidence_interval(DZs)*100, 'DZ-sLZ':confidence_interval(DZ_sLZs)*100, 'sLZ':confidence_interval(sLZs)*100,
              'sLZ-dLZ':confidence_interval(sLZ_dLZs)*100, 'dLZ':confidence_interval(dLZs)*100}

zones = ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']

zone_colors = {
    'DZ':    '#85CBEB',  # blue
    'DZ-sLZ':'#BEDCB0',  # light blue
    'sLZ':   '#4F609C',  # green
    'sLZ-dLZ':'#E9C61D', # yellow
    'dLZ':   '#8A4F21',  # orange
}
colors = [zone_colors[z] for z in zones]

sizes = [mean_dataset[z] for z in zones]
labels = [f"{z}: {mean_dataset[z]:.1f}% ± {ci_dataset[z]:.1f}%" for z in zones]


fig, ax = plt.subplots(figsize=(6, 6))
wedges, _ = ax.pie(
    sizes,
    colors=colors,
    startangle=90,
    counterclock=False,
    normalize=True,
    textprops={'fontsize': 10},
    wedgeprops={'edgecolor':'white','linewidth':1}
)

total = sum(sizes)
threshold = 8.0  # percent: put label inside if slice >= threshold

for i, w in enumerate(wedges):
    mid_deg = 0.5*(w.theta1 + w.theta2)
    mid_rad = np.deg2rad(mid_deg)
    frac = 100.0 * sizes[i] / total
    r = getattr(w, "r", 1.0)

    # positions
    x_in,  y_in  = np.cos(mid_rad)*r*0.60, np.sin(mid_rad)*r*0.60
    x_out, y_out = np.cos(mid_rad)*r*1.15, np.sin(mid_rad)*r*1.15
    ha = 'left' if x_out >= 0 else 'right'

    if frac >= threshold:
        ax.text(x_in, y_in, labels[i], ha='center', va='center', fontsize=13)
    else:
        ax.annotate(
            labels[i],
            xy=(np.cos(mid_rad)*r*0.95, np.sin(mid_rad)*r*0.95),  # near wedge edge
            xytext=(x_out, y_out),
            ha=ha, va='center', fontsize=13,
            arrowprops=dict(arrowstyle='-', lw=1, connectionstyle="arc3,rad=0.2")
        )

#ax.set_title("GC Zone Composition (mean % ± 95% CI)", fontsize=16)
ax.axis('equal')
plt.tight_layout()
plt.savefig(path+"gc_zone_pie.png", dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/gc_zone_pie.svg', bbox_inches='tight')
plt.close()
plt.clf()

############# Plot DZ distance vs dLZ distance jointplot ###############
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT', 'T-cell': 'Tfh'}})
#df_ = df_[(df_['Zone']=='DZ')|(df_['Zone']=='dLZ')].reset_index(drop=True)
x_name = 'DZ_distance_average'
y_name = 'Core_distance_average'
xmin = math.floor(df_[x_name].min()) - 1
xmax = math.ceil(df_[x_name].max()) + 1
ymin = math.floor(df_[y_name].min()) - 1
ymax = math.ceil(df_[y_name].max()) + 1

draw_jointplot(xs=x_name, y=y_name, df=df_, path=path, file_name='DZ vs dLZ distance jointplot',
               hue="Type", colors=('#CC6677', '#6699CC', '#888888', ), hue_order=['MT', 'Tfh', 'WT'],
               fill=False, legend=False, thresh=0.2, n_contours=3, alpha=1, height=4, ratio=5, space=0,
               xlabels='DZ distance', ylabel='dLZ distance', #xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
               )


#################################### Plot whole state space ####################################
df_ = df.copy()
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT', 'T-cell': 'Tfh'}})
draw_umap_space(df_, path, file_name='space_kmeans', condition_name='kmeans', label_name='pseudo_Label', colors=cmc.batlow, x_name='PC1', y_name='PC2', dot_size=0.07)

draw_jointplot(xs='PC1', y='PC2', df=df_, path=path, file_name='jointplot_type', hue="Type", colors=('#CC6677', '#6699CC', '#888888', ), hue_order=['MT', 'Tfh', 'WT'],
               legend=False, fill=False, thresh=0.15, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

df.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df.columns[8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
draw_space_feature_magnitude(df, path, feature_list, dot_size=0.07, x_name='PC1', y_name='PC2', vmax=None)


draw_cluster_distribution_heatmap(df_, path, file_name='kmeans_type_heatmap', condition_name='Type', cluster_type='kmeans',
                                  annot=True, col_cluster=False, row_cluster=True,figsize=(6,2))
draw_relative_cluster_distribution_heatmap(df_, path, file_name='relative_kmeans_type_heatmap', condition_name='Type', cluster_type='kmeans',
                                  annot=True, transpose=False, col_cluster=False, row_cluster=True, vmax=80, cmap=cmc.oslo_r, figsize=(6,2))

entropies = {'mt_B-cell':[], 'wt_B-cell':[], 'T-cell':[]}
for video in np.unique(df['Video']):
    df_part = df[df['Video']==video]
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='kmeans')
    for type in entropy:
        entropies[type].append(entropy[type])

new_order = ['wt_B-cell', 'mt_B-cell', 'T-cell']
ordered_entropies = change_dict_order(entropies, new_order)

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
entropies_ = {replace_keys.get(k, k):v  for (k,v) in ordered_entropies.items() }

dict_datasets = entropies_
file_name='entropy'
test='mann-whitney'

colors = ('#888888', '#CC6677','#6699CC', )
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
    print(pair, stat_test.pvalue)
plt.title('%s:%s' % (pairs, p_values))
plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()

#################################### Box plot comparing all motility features by cell types ####################################
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
condition_name = 'Type'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
if not os.path.isdir(path + 'feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_type/')

if not os.path.isdir(path + 'feature_box_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_box_plot_type/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell', 'mt_B-cell', 'T-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    draw_custom_violin_plot(dict_datasets, path+'feature_violin_plot_type/', file_name=feature_name, colors = ('#888888', '#CC6677','#6699CC'),
                            test='kruskal-wallis_dunn', pvalue=True, figsize=(1,2))

    draw_custom_box_plot(dict_datasets, path+ 'feature_box_plot_type/', file_name=feature_name, colors = ('#888888', '#CC6677', '#6699CC'),
    strip_plot=False, test='kruskal-wallis_dunn', pvalue=True, figsize=(1,2))


#################################### draw 3D trajectories by kmeans ####################################
for i in np.unique(df['kmeans']):
    print('cluster: ', i, 'Cell number: ', df[df['kmeans']==i].shape[0])

np.mean( df[df['Type']=='wt_B-cell']['total_distance'] )


draw_3D_trajectory_one_figure(df_duration, path, folder_name='kmeans trajectory', duration=20,
                              n_examples=30, label_name='kmeans', feature_name=['Position X', 'Position Y', 'Position Z'], lim=150)

draw_3D_trajectory_one_figure(df_duration, path, folder_name='cell type trajectory', duration=20,
                              n_examples=30, label_name='Type', feature_name=['Position X', 'Position Y', 'Position Z'], lim=100)

############# Plot LZ vs DZ jointplot for each cell type ###############

for cell_type in ['mt_B-cell', 'wt_B-cell', 'T-cell']:

    df_part = df[df['Type']==cell_type].reset_index(drop=True)

    df_part.loc[(df_part['Zone'] == 'DZ'), 'Zone1'] = 'DZ'
    #df_part.loc[(df_part['Zone'] == 'DZ-sLZ'), 'Zone1'] = 'DZ-sLZ'
    df_part.loc[( (df_part['Zone'] == 'sLZ') | (df_part['Zone'] == 'dLZ') | (df_part['Zone'] == 'sLZ-dLZ') ), 'Zone1'] = 'LZ'
    df_part['Zone1'] = df_part.Zone1.astype(str)

    print(cell_type, df_part[df_part['Zone1']=='DZ'].shape[0], df_part[df_part['Zone1']=='DZ-sLZ'].shape[0], df_part[df_part['Zone1']=='LZ'].shape[0])

    draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s_DZ vs LZ jointplot'%cell_type, hue="Zone1", hue_order=['LZ', 'DZ'],
                   colors=('#E69965', '#BAC8DA'), fill=False, legend=False, thresh=0.25, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

############# Plot sLZ vs dLZ jointplot for each cell type ###############

for cell_type in ['mt_B-cell', 'wt_B-cell', 'T-cell']:

    df_part = df[df['Type']==cell_type].reset_index(drop=True)

    df_part.loc[(df_part['Zone'] == 'sLZ'), 'Zone1'] = 'sLZ'
    #df_part.loc[(df_part['Zone'] == 'sLZ-dLZ'), 'Zone1'] = 'sLZ-dLZ'
    df_part.loc[(df_part['Zone'] == 'dLZ'), 'Zone1'] = 'dLZ'
    df_part['Zone1'] = df_part.Zone1.astype(str)

    print(cell_type, df_part[df_part['Zone1']=='sLZ'].shape[0], df_part[df_part['Zone1']=='sLZ-dLZ'].shape[0], df_part[df_part['Zone1']=='dLZ'].shape[0])

    draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s_sLZ vs dLZ jointplot'%cell_type, hue="Zone1", hue_order=['sLZ', 'dLZ'],
                   colors=('#4F609C', '#8A4F21'), fill=False, legend=False, thresh=0.25, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')


############# All Kmeans distribution heatmap ###############
df_ = df.copy()
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT', 'T-cell': 'Tfh'}})
df_['Type_Zone'] = df_['Type'].astype(str) + ' ' + df_['Zone']
group_df = draw_cluster_distribution_heatmap(df_, path, file_name='all_kmeans_heatmap', condition_name='Type_Zone', annot=False,
                                  cluster_type='kmeans', col_cluster=False, figsize=(4,5))
group_df = draw_cluster_distribution_heatmap(df_[df_['Type']=='Tfh'], path, file_name='Tfh_kmeans_heatmap', condition_name='Type_Zone', annot=False,
                                  cluster_type='kmeans', col_cluster=False, figsize=(4,2))


###################### Plot Zone motility feature violin plot for MT and WT  ############################

if not os.path.isdir(path + 'Zone motility box plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Zone motility box plot/')

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for feature_name in feature_list:
    condition_name = 'Type'
    dataset={}
    for cell_type in ['wt_B-cell', 'mt_B-cell', 'T-cell']:
        for group in ['DZ','sLZ', 'dLZ']:
        #for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
            data = df[(df[condition_name] == cell_type)&(df['Zone'] == group)][feature_name]

            dataset[cell_type+'_'+str(group)] = np.array(data)

    rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
                   'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ',
                   'T-cell_DZ': 'Tfh DZ', 'T-cell_sLZ': 'Tfh sLZ', 'T-cell_dLZ': 'Tfh dLZ',}
    # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_DZ-sLZ': 'WT DZ-sLZ', 'wt_B-cell_sLZ': 'WT sLZ',
    #                'wt_B-cell_sLZ-dLZ': 'WT sLZ-dLZ', 'wt_B-cell_dLZ': 'WT dLZ',
    #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_DZ-sLZ': 'MT DZ-sLZ', 'mt_B-cell_sLZ': 'MT sLZ',
    #                 'mt_B-cell_sLZ-dLZ': 'MT sLZ-dLZ', 'mt_B-cell_dLZ': 'MT dLZ',}

    dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

    draw_custom_violin_plot(dataset_renamed, path + 'Zone motility box plot/', file_name=feature_name,
                            colors=('#888888', '#888888', '#888888', '#CC6677', '#CC6677', '#CC6677', '#6699CC','#6699CC','#6699CC'),
                            test='kruskal-wallis_dunn', pvalue=True, return_sig=True, figsize=(3, 2))


######################## DZ/sLZ/dLZ interaction features mt vs WT box plot  ###########################

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'Zone'])

for zone in ['DZ','sLZ', 'dLZ']:
    df_part = df[(df['Zone'] == zone)].reset_index(drop=True)
    if not os.path.isdir(path + '%s int feature violin plot/'%zone):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s int feature violin plot/'%zone)
    for feature_name in feature_list:
        condition_name = 'Type'
        dataset={}
        for cell_type in ['wt_B-cell', 'mt_B-cell', 'T-cell']:
            data = df_part[df_part[condition_name] == cell_type][feature_name]
            dataset[cell_type] = np.array(data)

        values = flatten_nested_dict(dataset)
        if np.isnan(values).any() == True:  # Check at least one nan
            continue
        elif np.isfinite(values).all() == False:  # Check everything is not inf
            continue

        rename_keys = {'wt_B-cell': 'WT', 'mt_B-cell': 'MT', 'T-cell':'Tfh'}
        dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

        # draw_custom_bar_plot(dataset_renamed, path + '%s int feature violin plot/'%zone, file_name=feature_name,
        #                      strip_plot=False, colors=('#888888', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(1, 2))
        draw_custom_violin_plot(dataset_renamed, path + '%s int feature violin plot/'%zone, file_name=feature_name,
                                colors=('#888888', '#CC6677', '#6699CC' ), test='mann-whitney', pvalue=True, figsize=(1, 2))


#################################### all motility features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT', 'T-cell': 'Tfh'}})
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
                                       color_list=['#CC6677', '#6699CC', '#888888', ], marker_list=['o', 'o', 'o',], figsize=(4,4), x_label=xlabel,
                                   estimator='mean', error_type='ci_norm', fill=False, replace_keys=None, pvalue=False, test='mann-whitney', set_zero=False)



####################################### Quantify Tfh interaction frequency (normalized by both GC B cells and T cells)  #############################################
int_time = 19
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
        df_T = df[(df['Type']=='T-cell')&(df['Video']==video)].reset_index(drop=True)
        data = df_video['T_contact_times']
        mask = ~np.isnan(data)
        data = data[mask]

        persistent_int_freq = sum(data == int_time)
        total_n_contact = sum(data)
        n_trajs = df_video.shape[0]
        n_Tfh = df_T.shape[0]

        persistent_int_freq_per_cellnumber = persistent_int_freq / (n_trajs*n_Tfh)
        total_n_contacts_per_cellnumber = total_n_contact / (n_trajs*n_Tfh)
        persistent_int_freq_per_cellcontact = persistent_int_freq / total_n_contact
        low_contact_freq_per_cellnumber = sum((1<=data) & (data<=5)) / (n_trajs*n_Tfh)

        if persistent_int_freq != 0:

            persistent_int_freqs.append(persistent_int_freq)
            persistent_int_freq_per_cellnumbers.append(persistent_int_freq_per_cellnumber)
            persistent_int_freq_per_cellcontacts.append(persistent_int_freq_per_cellcontact)
        else:
            print(video, persistent_int_freq)
        total_n_contacts_per_cellnumbers.append(total_n_contacts_per_cellnumber)
        low_contact_freq_per_cellnumbers.append(low_contact_freq_per_cellnumber)

    persistent_int_freqs_datasets[cell_type] = persistent_int_freqs
    persistent_int_freq_per_cellnumbers_datasets[cell_type] = persistent_int_freq_per_cellnumbers
    total_n_contacts_per_cellnumbers_datasets[cell_type] = total_n_contacts_per_cellnumbers
    persistent_int_freq_per_cellcontacts_datasets[cell_type] = persistent_int_freq_per_cellcontacts
    low_contact_freq_per_cellnumbers_datasets[cell_type] = low_contact_freq_per_cellnumbers

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
new_order = ['wt_B-cell', 'mt_B-cell']
persistent_int_freqs_datasets = change_dict_order(persistent_int_freqs_datasets, new_order)
persistent_int_freqs_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freqs_datasets.items() }

persistent_int_freq_per_cellnumbers_datasets = change_dict_order(persistent_int_freq_per_cellnumbers_datasets, new_order)
persistent_int_freq_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freq_per_cellnumbers_datasets.items() }

total_n_contacts_per_cellnumbers_datasets = change_dict_order(total_n_contacts_per_cellnumbers_datasets, new_order)
total_n_contacts_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in total_n_contacts_per_cellnumbers_datasets.items() }


low_contact_freq_per_cellnumbers_datasets = change_dict_order(low_contact_freq_per_cellnumbers_datasets, new_order)
low_contact_freq_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in low_contact_freq_per_cellnumbers_datasets.items() }

colors=('#888888', '#CC6677')
draw_custom_bar_plot(persistent_int_freqs_datasets, path, file_name='Normalized Total Tfh interaction frequency',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))
draw_custom_bar_plot(persistent_int_freq_per_cellnumbers_datasets, path, file_name='Normalized Tfh persistent interaction frequency per cell number',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(total_n_contacts_per_cellnumbers_datasets, path, file_name='Normalized number of Tfh contacts per cell number',
                     strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
draw_custom_bar_plot(low_contact_freq_per_cellnumbers_datasets, path, file_name='Normalized Tfh low contact time frequency per cell number',
                     strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1,2))