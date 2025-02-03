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
"""Generates Data for Figure 1."""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast


#################################### motility space ####################################
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
df = pd.read_parquet(path+'motility_features_nan_removed.parquet')
#df = pd.read_parquet(path+'latent_vectors_20_PC.parquet')
df_duration = pd.read_parquet(path + 'traj_duration_nan_removed.parquet')

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\analysis\\'
color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')

draw_umap_space(df, path, file_name='space_type', condition_name='type', label_name='label', colors=color_list,
                x_name='PC1', y_name='PC2', dot_size=0.07)
draw_umap_space(df, path, file_name='space_kmeans', condition_name='kmeans', label_name='label', colors=cmc.batlow,
                x_name='PC1', y_name='PC2', dot_size=0.07)
draw_umap_space(df, path, file_name='space_pancreatic', condition_name='pancreatic_phenotype', label_name='label', colors=('#888888', '#CC6677'),
                x_name='PC1', y_name='PC2', dot_size=0.07)
draw_umap_space(df, path, file_name='space_lung', condition_name='lung_phenotype', label_name='label', colors=('#888888', '#CC6677'),
                x_name='PC1', y_name='PC2', dot_size=0.07)

draw_jointplot(xs='PC1', y='PC2', df=df, path=path, file_name='jointplot_type', hue="type", colors=color_list, hue_order=None,
               legend=False, fill=False, thresh=0.12, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')
draw_jointplot(xs='PC1', y='PC2', df=df, path=path, file_name='jointplot_kmeans', hue="kmeans", colors=cmc.batlow,
               legend=False, fill=False, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

draw_jointplot(xs='PC1', y='PC2', df=df, path=path, file_name='jointplot_pancreas', hue="pancreatic_phenotype", colors=('#888888', '#CC6677'),
               legend=False, fill=False, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')
draw_jointplot(xs='PC1', y='PC2', df=df, path=path, file_name='jointplot_lung', hue="lung_phenotype", colors=('#888888', '#CC6677'),
               legend=False, fill=False, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')
draw_jointplot(xs='PC1', y='PC2', df=df, path=path, file_name='jointplot_ovary', hue="ovarian_phenotype", colors=('#888888', '#CC6677'),
               legend=False, fill=False, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

# draw_umap_space(df_ctrl, path, file_name='motility space_tskmeans', condition_name='tskmeans', label_name='pseudo_particle',
#                 colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')
df_ = df.copy()
df_['all_phenotype'] = 'p'+df['pancreatic_phenotype'].astype(str) + '_l' + df['lung_phenotype'].astype(str) + '_o' + df['ovarian_phenotype'].astype(str)
df_ = df_[(df_['all_phenotype']=='p0_l0_o0')|(df_['all_phenotype']=='p1_l1_o1')].reset_index(drop=True)

draw_jointplot(xs='PC1', y='PC2', df=df_, path=path, file_name='jointplot_non vs all effective', hue="all_phenotype", colors=('#888888', '#CC6677'),
               hue_order=['p0_l0_o0', 'p1_l1_o1'], legend=False, fill=False, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')


draw_contour(df, path, file_name='space_contour_type', condition_name='type', colors=color_list, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)

df.columns.get_loc('inst_angle_pulseindicator')
feature_list = df.columns[2:90].drop(['speed_distribution_x', 'speed_distribution_y'])
draw_space_feature_magnitude(df, path, feature_list, dot_size=0.07, x_name='PC1', y_name='PC2', vmax=None)

#################################### Heatmap of cluster enrichment and shannon entropy ####################################
#df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB', 'T-cell': 'Tfh'}})
draw_cluster_distribution_heatmap(df, path, file_name='kmeans_type_heatmap', condition_name='type', cluster_type='kmeans',
                                  annot=False, col_cluster=False, row_cluster=True,figsize=(4,3))

draw_cluster_distribution_heatmap(df, path, file_name='pancreatic_phenotype_heatmap', condition_name='pancreatic_phenotype', cluster_type='kmeans',
                                  annot=False, col_cluster=False, row_cluster=True,figsize=(3,1))
draw_cluster_distribution_heatmap(df, path, file_name='lung_phenotype_heatmap', condition_name='lung_phenotype', cluster_type='kmeans',
                                  annot=False, col_cluster=False, row_cluster=True,figsize=(3,1))
draw_cluster_distribution_heatmap(df, path, file_name='ovarian_phenotype_heatmap', condition_name='ovarian_phenotype', cluster_type='kmeans',
                                  annot=False, col_cluster=False, row_cluster=True,figsize=(3,1))

draw_relative_cluster_distribution_heatmap(df, path, file_name='relative_kmeans_type_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=30, cmap=cmc.oslo_r, transpose=False, condition_name='type', cluster_type='kmeans', figsize=(4,3))

draw_relative_cluster_distribution_heatmap(df, path, file_name='relative_pancreatic_phenotype_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=None, cmap=cmc.oslo_r, transpose=False, condition_name='pancreatic_phenotype', cluster_type='kmeans',
                                           figsize=(3, 1))

draw_relative_cluster_distribution_heatmap(df, path, file_name='relative_lung_phenotype_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=None, cmap=cmc.oslo_r, transpose=False, condition_name='lung_phenotype', cluster_type='kmeans',
                                           figsize=(3, 1))
draw_relative_cluster_distribution_heatmap(df, path, file_name='relative_ovarian_phenotype_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=None, cmap=cmc.oslo_r, transpose=False, condition_name='ovarian_phenotype', cluster_type='kmeans',
                                           figsize=(3, 1))


entropy, max_entropy = calculate_entropy(df, df, condition_name='type', cluster_type='kmeans')

# new_order = ['wt_B-cell', 'T-cell']
# ordered_entropies = change_dict_order(entropies, new_order)
#replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
#entropies_ = {replace_keys.get(k, k):v  for (k,v) in ordered_entropies.items() }

dict_datasets = entropy
file_name='entropy'
test='mann-whitney'

colors = color_list
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(1, 2))
sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

ax = sns.barplot(x=np.arange(len(list(sorted_keys))), y=sorted_vals, capsize=0.5, edgecolor='0.2', lw=1, errwidth=1, palette=colors)
plot_params = {'edgecolor': '0.2', 'linewidth': 1, 'fc': 'none'}
ax = sns.stripplot(x=np.arange(len(list(sorted_keys))), y=sorted_vals, marker='s', s=1.5, **plot_params)
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
plt.xticks(np.arange(len(list(sorted_keys))), sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
           weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')
# plt.ylabel('%s' % feature_name, fontsize=4)
# category labels
plt.grid(False)
plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()

#################################### draw 2D trajectories by kmeans ####################################
draw_2D_trajectories_one_figure(df_duration, df, path, duration=31, n_examples=17, label_name='kmeans', feature_name=['x', 'y'], lim=200)
for i in np.unique(df['kmeans']):
    print('cluster: ', i, 'Cell number: ', df[df['kmeans']==i].shape[0])

#################################### Box plot comparing all motility features by cell types ####################################
df.columns.get_loc('inst_angle_pulseindicator')
feature_list = df.columns[8:90].drop(['speed_distribution_x', 'speed_distribution_y'])
condition_name = 'type'
#replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'wt GCB'}
if not os.path.isdir(path + 'feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_type/')

if not os.path.isdir(path + 'feature_box_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_box_plot_type/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    # new_order = ['wt_B-cell', 'T-cell']
    # ordered_dataset = change_dict_order(dataset, new_order)
    # dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    draw_custom_violin_plot(dataset, path+'feature_violin_plot_type/', file_name=feature_name, colors = color_list,
                            test='mann-whitney', pvalue=True, figsize=(3,6))

    draw_custom_box_plot(dataset, path+ 'feature_box_plot_type/', file_name=feature_name, colors = color_list,
    strip_plot=False, test='mann-whitney', pvalue=True, figsize=(3,6))

#################################### Volcano plot of all motility features ####################################

condition_name = 'pancreatic_phenotype'
ref = 0
compare = 1
np.unique(df[condition_name])
df.columns.get_loc('inst_angle_pulseindicator')
feature_list = df.columns[8:90].drop(['speed_distribution_x', 'speed_distribution_y'])

df_part = df[(df[condition_name]==ref)|(df[condition_name]==compare)].reset_index(drop=True)

df_p = pd.DataFrame()
for feature_name in feature_list:
    dataset = {}
    for condition in np.unique(df_part[condition_name]):
        data = df_part[df_part[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)

    pvalue = get_pvalue(dataset, test='mann-whitney')
    logp = -np.log10(pvalue)

    avgZ = get_avgZ(dataset, ref_name=ref, data_name=compare)

    row = pd.DataFrame()
    row['Feature'] = [feature_name]
    row['Pvalue'] = [pvalue]
    row['-Logp'] = [logp]
    row['AvgZ'] = [avgZ]
    df_p = pd.concat([df_p, row], axis=0)

df_p['Adj_p'] = adjusted_pvalues(df_p['Pvalue'], correction_type='Benjamini-Hochberg')
df_p['Adj_Logp'] = -np.log(df_p['Adj_p'])
df_p = df_p[~df_p.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf


draw_volcano_plot(df_p, path, file_name='volcano plot with %s'%condition_name, z_thresh=0.3, p_thresh=10, z_name='AvgZ', p_name='Adj_Logp',
                  feature_name='Feature', figsize=(6,6))

#################################### Venn diagram for phenotypes ####################################
df.columns.get_loc('inst_angle_pulseindicator')
feature_list = df.columns[8:90].drop(['speed_distribution_x', 'speed_distribution_y'])

df_venn = pd.DataFrame()
for condition_name in ['pancreatic_phenotype', 'lung_phenotype', 'ovarian_phenotype']:
    ref = 0
    compare = 1
    np.unique(df[condition_name])
    df.columns.get_loc('inst_angle_pulseindicator')
    feature_list = df.columns[8:90].drop(['speed_distribution_x', 'speed_distribution_y'])

    df_part = df[(df[condition_name]==ref)|(df[condition_name]==compare)].reset_index(drop=True)

    df_p = pd.DataFrame()
    for feature_name in feature_list:
        dataset = {}
        for condition in np.unique(df_part[condition_name]):
            data = df_part[df_part[condition_name] == condition][feature_name]
            dataset[condition] = np.array(data)

        pvalue = get_pvalue(dataset, test='mann-whitney')
        logp = -np.log10(pvalue)

        avgZ = get_avgZ(dataset, ref_name=ref, data_name=compare)

        row = pd.DataFrame()
        row['Feature'] = [feature_name]
        row['Pvalue'] = [pvalue]
        row['-Logp'] = [logp]
        row['AvgZ'] = [avgZ]
        df_p = pd.concat([df_p, row], axis=0)

    df_p['Adj_p'] = adjusted_pvalues(df_p['Pvalue'], correction_type='Benjamini-Hochberg')
    df_p['Adj_Logp'] = -np.log(df_p['Adj_p'])
    df_p = df_p[~df_p.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf

    z_thresh=0.3
    p_thresh=10

    def map_color(a):
        AvgZ, Adj_Logp = a

        if abs(AvgZ) < z_thresh or Adj_Logp < p_thresh:
            return 'NoChange'
        else:
            if AvgZ<0:
                return 'Negative'
            elif AvgZ>=0:
                return 'Positive'


    df_p[condition_name] = df_p[['AvgZ', 'Adj_Logp']].apply(map_color, axis=1)
    df_venn = pd.concat([df_venn, df_p[condition_name]], axis=1)

df_venn['Feature'] = df_p['Feature']

#pancreas = set(df_venn.query('pancreatic_phenotype != "NoChange"')['Feature'])
lung = set(df_venn.query('lung_phenotype != "NoChange"')['Feature'])  # pancreas and lung is same
ovary = set(df_venn.query('ovarian_phenotype != "NoChange"')['Feature'])

onlylung = lung - ovary
onlyovary = ovary - lung
both = lung.intersection(ovary)



from matplotlib_venn import venn2_circles, venn2
fig, ax = plt.subplots(figsize=(4,4))
v2 = venn2(subsets = {
        '10': len(onlylung),  # Ab
        '01': len(onlyovary),  # aB
        '11': len(both)       # AB
        },
      set_labels = ('pancreas and lung','ovary'),
alpha = 0.6,
set_colors=('#CC6677', '#6699CC'),
     )

venn2_circles(subsets = {
        '10': len(onlylung),  # Ab
        '01': len(onlyovary),  # aB
        '11': len(both)       # AB
        },

linestyle='-',
      linewidth=3,
     )

for text in v2.set_labels:  # the text outside the circle
    text.set_fontsize(15);

for text in v2.subset_labels:  # the text inside the circle
    text.set_fontsize(7)

text = ''
for i in onlylung:
    text += f'{i}\n'
v2.get_label_by_id('10').set_text(text)  # Mac mini

text = ''
for i in onlyovary:
    text += f'{i}\n'
v2.get_label_by_id('01').set_text(text)  # Mac Studio

text = ''
for i in both:        # Mac mini and Mac Studio
    text += f'{i}\n'
v2.get_label_by_id('11').set_text(text)
v2.get_label_by_id('11').set_color('firebrick')

plt.savefig(path+'venn diagram of significant features.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/venn diagram of significant features.svg', bbox_inches='tight')
plt.clf()
plt.close()
#################################### All motility feature clustermap for conditions ####################################

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma

Z_avg_df = pd.DataFrame()
for feature_name in feature_list:
    Z_avg_df_temp = pd.DataFrame()
    Z_df = pd.DataFrame()
    pdf_temp = pd.DataFrame()
    cluster_list = []
    for cluster in np.unique(df['kmeans']):
        Z_matrix = pd.DataFrame()
        Z_avg_matrix = []
        data = df[(df['kmeans'] == cluster)][feature_name]
        avg = np.mean(data)
        cluster_list.append(avg)

    Z_avg_df = pd.concat([Z_avg_df,pd.DataFrame(cluster_list, columns=[feature_name])], axis=1)

Z_avg_df = Z_avg_df.replace([np.inf, -np.inf], np.nan)  # Convert inf to nan
Z_avg_df = Z_avg_df.dropna(axis=1, how='any')
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
Z_avg_df= pd.DataFrame(scaler.fit_transform( Z_avg_df ), columns=Z_avg_df.columns)

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)

if np.max(np.max(Z_avg_df)) >= abs(np.min(np.min(Z_avg_df))):
    kws = dict(cbar_kws=dict(ticks=[-round(np.max(np.max(Z_avg_df)), 1), 0, round(np.max(np.max(Z_avg_df)), 1)], orientation='horizontal'),
               vmin=-round(np.max(np.max(Z_avg_df)), 1))
else:
    kws = dict(cbar_kws=dict(ticks=[round(np.min(np.min(Z_avg_df)), 1), 0, -round(np.min(np.min(Z_avg_df)), 1)],orientation='horizontal'),
               vmin=round(np.min(np.min(Z_avg_df)), 1) )

kws = dict(cbar_kws=dict(ticks=[1.5, 0, -1.5], orientation='horizontal'), vmin=-1.5, vmax=1.5 )

g=sns.clustermap(Z_avg_df.T, annot=False, cmap=cmc.vik,
#cbar_pos=(1, 0.2, 0.03, 0.8),
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (6, 30),
dendrogram_ratio=0.05
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=16, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, g.ax_row_dendrogram.get_position().width*10, 0.02])

g.ax_cbar.set_title('Z score', fontsize=16)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=16)

for spine in g.ax_cbar.spines:
    #g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'kmeans Z score features_heatmap.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/kmeans Z score features_heatmap.svg', bbox_inches='tight')
plt.clf()
plt.close()


############# All Kmeans distribution heatmap ###############
df_ = df.copy()
df_['all_phenotype'] = 'pl'+df['pancreatic_phenotype'].astype(str) + ' o' + df['ovarian_phenotype'].astype(str)
draw_cluster_distribution_heatmap(df_, path, file_name='all_kmeans_heatmap', condition_name='all_phenotype', annot=False,
                                  cluster_type='kmeans', col_cluster=False, figsize=(4,2))

draw_heatmap_with_circles(df_, path, file_name='all_kmeans_circleheatmap', condition_name='all_phenotype', cluster_type='kmeans',
                          vmax=None, transpose=False, row_cluster=True, col_cluster=False, figsize=(4,4))

draw_relative_cluster_distribution_heatmap(df_, path, file_name='relative_all_kmeans_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=40, cmap=cmc.oslo_r, transpose=False, condition_name='all_phenotype', cluster_type='kmeans',
                                           figsize=(4, 2))

for zone in np.unique(df_['all_phenotype']):
    print(zone, df_[df_['all_phenotype']==zone].shape[0])


############# Plot type kmeans cross correlation  ###############
font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

condition_name = 'type'
cluster_type = 'kmeans'

df_corr_data = pd.DataFrame()
group_clones=[]
for group in np.unique(df[condition_name]):
#for group in ['DZ', 'sLZ', 'dLZ']:
    corrcoef = []
    aaa = df[df[condition_name] == group]

    group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    group_clone_T = group_clone.T
    for cluster in list(pd.unique(df[cluster_type])):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=[0])
    group_clone = group_clone_T.T
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

plt.savefig(path+'type cross correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/type cross correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

############# Plot linear regression btw tumor volume and cluster enrichment ###############


for feature in ['pancreatic_infiltration', 'pancreatic_tumor_volume', 'lung_infiltration', 'lung_tumor_volume', 'ovarian_infiltration', 'ovarian_tumor_volume']:

    mt_enrichments = pd.DataFrame()
    for group_clone in group_clones:  # 'DZ', 'sLz', 'dLZ'
        mt_enrichments = pd.concat( [mt_enrichments, group_clone[list(group_clone.columns)[0]]], axis=1 )

    for column in mt_enrichments.columns:
        value = np.unique( df[df['type'] == column][feature] )[0]
        mt_enrichments.rename(columns={column:'%f'%value}, inplace=True)

    mt_enrichments.columns = mt_enrichments.columns.astype(float)
    mt_enrichments = mt_enrichments.sort_index(axis=1)

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
        x = list(mt_enrichments.columns)
        y = mt_enrichments.iloc[kmeans, :].values

        r, p = scipy.stats.spearmanr(x, y)

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

    ax.tick_params(width=2, color='0.2', labelsize=12)

    ax.set_xlabel(feature, fontsize=16, weight='bold', color='0.2', labelpad=5)
    ax.set_ylabel('fraction of cluster (%)', fontsize=16, weight='bold', color='0.2')
    # plt.xticks(np.arange(len(list(mt_enrichments.iloc[kmeans, :].index))), np.array(list(mt_enrichments.columns)), fontsize=16, rotation=35, # fontname = "Arial",
    #            rotation_mode='anchor', ha='right', color='0.2', weight='bold')
    # plt.xticks(x, fontsize=8, rotation=35, # fontname = "Arial",
    #            rotation_mode='anchor', ha='right', color='0.2', weight='bold')
    #
    # plt.yticks(fontsize=16, color='0.2', weight='bold')


    plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'bold', 'size': 12}, labelcolor='0.2',
               loc='best')

    plt.savefig(path + '%s cluster fraction regplot.png'%feature, dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s cluster fraction regplot.svg'%feature, bbox_inches='tight')
    plt.clf()
    plt.close()

#################################### lung and pacnreatic infilatration lm plot for all motility features ####################################
linewidth = 1.5
fontsize = 16
width=10
ratio=5
space=0.2
nrows = 1
ncols = 6


if not os.path.isdir(path + 'motility regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'motility regplot/')

df.columns.get_loc('inst_angle_pulseindicator')
feature_list = df.columns[2:90].drop(['speed_distribution_x', 'speed_distribution_y'])

for feature in feature_list:
    datasets = {}
    mean_values = []
    pis = []
    pvs = []
    lis = []
    lvs = []
    ois = []
    ovs = []

    for batch in np.unique(df['type']):
        df_part = df[df['type'] == batch].reset_index(drop=True)
        pi = np.unique( df_part['pancreatic_infiltration'] )[0]
        pv = np.unique( df_part['pancreatic_tumor_volume'] )[0]
        li = np.unique( df_part['lung_infiltration'] )[0]
        lv = np.unique(df_part['lung_tumor_volume'])[0]
        oi = np.unique(df_part['ovarian_infiltration'])[0]
        ov = np.unique(df_part['ovarian_tumor_volume'])[0]
        mean_value = np.mean(np.array(df_part[feature]))

        mean_values.append(mean_value)
        pis.append(pi)
        pvs.append(pv)
        lis.append(li)
        lvs.append(lv)
        ois.append(oi)
        ovs.append(ov)

    datasets['Value'] = np.array(mean_values)
    datasets['pancreatic_infiltration'] = np.array(pis)
    datasets['pancreatic_tumor_volume'] = np.array(pvs)
    datasets['lung_infiltration'] = np.array(lis)
    datasets['lung_tumor_volume'] = np.array(lvs)
    datasets['ovarian_infiltration'] = np.array(ois)
    datasets['ovarian_tumor_volume'] = np.array(ovs)

    df_ = pd.DataFrame(datasets)

    #from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig,axes = plt.subplots(nrows, ncols, figsize=(16, 4), sharey='row')
    for col, typ in enumerate(['pancreatic_infiltration', 'pancreatic_tumor_volume', 'lung_infiltration', 'lung_tumor_volume', 'ovarian_infiltration', 'ovarian_tumor_volume']):
        ax = axes[col]
        sns.regplot(x=typ, y='Value', data=df_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

        #r, p = scipy.stats.pearsonr(df_[typ], df_['Value'])
        r, p = scipy.stats.spearmanr(df_[typ], df_['Value'])

        # if typ == 'Young':
        #     plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
        #              fontsize=12, fontdict={'weight': 'bold'}, color="black")
        #     plt.text(0.8, 0.85, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
        #              fontsize=12, fontdict={'weight': 'bold'}, color="black")
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
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel(typ, fontsize=10, weight='bold', color='0.2', labelpad=5)
        ax.set_ylabel('', fontsize=10, weight='bold', color='0.2', labelpad=5)
        #ax.set_xlim(16, 98)

    plt.savefig(path + 'motility regplot/%s regplot.png'%(feature), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/motility regplot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/motility regplot/')
    plt.savefig(path + 'svg/motility regplot/%s regplot.svg'%(feature), bbox_inches='tight')
    plt.clf()
    plt.close()




#################################### lung and pacnreatic infilatration corr heatmap for all motility features ####################################
df.columns.get_loc('inst_angle_pulseindicator')
feature_list = df.columns[2:90].drop(['speed_distribution_x', 'speed_distribution_y'])

df_r = pd.DataFrame()
df_p = pd.DataFrame()
for feature in tqdm(feature_list):
    datasets = {}
    mean_values = []
    pis = []
    pvs = []
    lis = []
    lvs = []

    for batch in np.unique(df['type']):
        df_part = df[df['type'] == batch].reset_index(drop=True)
        pi = np.unique(df_part['pancreatic_infiltration'])[0]
        pv = np.unique(df_part['pancreatic_tumor_volume'])[0]
        li = np.unique(df_part['lung_infiltration'])[0]
        lv = np.unique(df_part['lung_tumor_volume'])[0]
        oi = np.unique(df_part['ovarian_infiltration'])[0]
        ov = np.unique(df_part['ovarian_tumor_volume'])[0]
        mean_value = np.mean(np.array(df_part[feature]))

        mean_values.append(mean_value)
        pis.append(pi)
        pvs.append(pv)
        lis.append(li)
        lvs.append(lv)

    datasets['Value'] = np.array(mean_values)
    datasets['pancreatic_infiltration'] = np.array(pis)
    datasets['pancreatic_tumor_volume'] = np.array(pvs)
    datasets['lung_infiltration'] = np.array(lis)
    datasets['lung_tumor_volume'] = np.array(lvs)
    datasets['ovarian_infiltration'] = np.array(ois)
    datasets['ovarian_tumor_volume'] = np.array(ovs)

    df_ = pd.DataFrame(datasets)
    rs=[]
    ps=[]
    for typ in ['pancreatic_infiltration', 'pancreatic_tumor_volume', 'lung_infiltration', 'lung_tumor_volume', 'ovarian_infiltration', 'ovarian_tumor_volume']:

        #r, p = scipy.stats.pearsonr(df_[typ], df_['Value'])
        r, p = scipy.stats.spearmanr(df_[typ], df_['Value'])
        rs.append(r)
        ps.append(p)

    each_rs = pd.DataFrame(rs, columns=[feature], index=['pancreatic_infiltration', 'pancreatic_tumor_volume', 'lung_infiltration', 'lung_tumor_volume', 'ovarian_infiltration', 'ovarian_tumor_volume'])
    each_ps = pd.DataFrame(ps, columns=[feature],index=['pancreatic_infiltration', 'pancreatic_tumor_volume', 'lung_infiltration', 'lung_tumor_volume', 'ovarian_infiltration', 'ovarian_tumor_volume'])

    df_r = pd.concat([df_r, each_rs], axis=1)
    df_p = pd.concat([df_p, each_ps], axis=1)


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
kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

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

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=10, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('Correlation Coefficient', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# for spine in g.ax_cbar.spines:
#     #g.ax_cbar.spines[spine].set_color('crimson')
#     g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'correlation with aggressiveness for all features.png', dpi=300,bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/' + 'correlation with aggressiveness for all features.svg', bbox_inches='tight')
plt.clf()
plt.close()



#################################### Construct Multidimensional Age Axis ####################################
df.columns.get_loc('inst_angle_pulseindicator')
from sklearn.decomposition import PCA
motility_data = df.iloc[:, 2:90].drop(['speed_distribution_x', 'speed_distribution_y'], axis=1)
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
motility_data_scaled= pd.DataFrame(scaler.fit_transform( motility_data ), columns=motility_data.columns)
pca = PCA(0.95)
pcs = pca.fit_transform(motility_data_scaled)
df_pc = pd.DataFrame(pcs, columns=['PC%s' % str(i) for i in range(0, pcs.shape[1])])
feature_list = list(df_pc.columns)
df_pc = pd.concat([df.drop(['PC1', 'PC2'], axis=1), df_pc], axis=1)


centroids = {}
for feature in ['pancreatic_infiltration', 'pancreatic_tumor_volume', 'lung_infiltration', 'lung_tumor_volume', 'ovarian_infiltration', 'ovarian_tumor_volume']:
    values = np.unique(df_pc[feature]) # ages[0] = min age, ages[-1] = max age

    min_centroid = np.mean( df_pc[df_pc[feature] == values[0]][feature_list], axis=0 )  # (n_pc, )
    max_centroid = np.mean(df_pc[df_pc[feature] == values[-1]][feature_list], axis=0) # (n_pc, )
    centroids[feature] = np.array([min_centroid, max_centroid]) # (2, n_pc)


datasets = {}
coeffs = []
features = []
typs = []
phenos = []
for batch in np.unique(df_pc['type']):
    df_part = df_pc[df_pc['type'] == batch].reset_index(drop=True)
    for feature in ['pancreatic_infiltration', 'pancreatic_tumor_volume', 'lung_infiltration', 'lung_tumor_volume', 'ovarian_infiltration', 'ovarian_tumor_volume']:

        pheno = df_part[feature][0]
        target_data = np.array(df_part[feature_list])
        target_centroid = np.mean(target_data, axis=0)
        min_centroid, max_centroid = centroids[feature]
        #print(batch, age_type, media, age, min_centroid[0], max_centroid[0], target_centroid[0], )

        proj, t = project_on_line(start=min_centroid, end=max_centroid, target=target_centroid, segment=False)
        coeffs.append(t)
        features.append(feature)
        typs.append(batch)
        phenos.append(pheno)

datasets['Type'] = np.array(typs)
datasets['Feature'] = np.array(features)
datasets['Pheno'] = np.array(phenos)
datasets['Value'] = np.array(coeffs)

df_ = pd.DataFrame(datasets)



linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = 1
ncols = 6

from matplotlib.ticker import MaxNLocator
font = {'family': 'arial',
            'weight': 'normal',
            'size': 10}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = linewidth

fig,axes = plt.subplots(nrows, ncols, figsize=(20, 10), sharey='row')
#for row, type in enumerate(['Young', 'Old', 'Frail']):
for col, cond in enumerate( ['pancreatic_infiltration', 'pancreatic_tumor_volume', 'lung_infiltration', 'lung_tumor_volume', 'ovarian_infiltration', 'ovarian_tumor_volume'] ):
    ax = axes[col]
    df_part_= df_[ (df_['Feature']==cond) ].reset_index(drop=True)
    sns.regplot(x='Pheno', y='Value', data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

    r, p = scipy.stats.pearsonr(df_part_['Pheno'], df_part_['Value'])
    if type == 'Young':
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

    ax.set_xlabel('%s'%cond, fontsize=10, weight='bold', color='0.2', labelpad=5)
    ax.set_ylabel('', fontsize=10, weight='bold', color='0.2', labelpad=5)
    #ax.set_xlim(16, 98)

plt.savefig(path + 'coeff regplot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/coeff regplot.svg', bbox_inches='tight')
plt.clf()
plt.close()


############# Plot phenotype kmeans cross correlation ###############
font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

condition_name = 'all_phenotype'
cluster_type = 'kmeans'

df_corr_data = pd.DataFrame()
group_clones=[]
for group in np.unique(df_[condition_name]):
#for group in ['DZ', 'sLZ', 'dLZ']:
    corrcoef = []
    aaa = df_[df_[condition_name] == group]

    group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    group_clone_T = group_clone.T
    for cluster in list(pd.unique(df_[cluster_type])):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=[0])
    group_clone = group_clone_T.T

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

plt.savefig(path+'phenotypes cross correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/phenotypes cross correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()









#################################### latent space ####################################
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\\'
df = pd.read_parquet(path+'classifier_latent_vector.parquet')
#df = pd.read_parquet(path+'latent_vectors_20_PC.parquet')


path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CART\analysis\classification0_1_2\\'
color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')

draw_umap_space(df, path, file_name='latent space_type', condition_name='type', label_name='label',
                colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df, path, file_name='latent space_kmeans', condition_name='kmeans', label_name='label',
                colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df, path, file_name='latent space_killing_kmeans', condition_name='killing_kmeans', label_name='label',
                colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df, path, file_name='latent space_killing', condition_name='killing', label_name='label',
                colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')


# draw_umap_space(df_ctrl, path, file_name='motility space_tskmeans', condition_name='tskmeans', label_name='pseudo_particle',
#                 colors = color_list, dot_size=0.07, x_name='PC1', y_name='PC2')

draw_contour(df, path, file_name='space_contour_type', condition_name='type', colors=color_list, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df, path, file_name='space_contour_killing', condition_name='killing', colors=color_list, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)

df.columns.get_loc('displ_autocorr_y_3')
feature_list = df.columns[:106].drop(['phi'])
draw_space_feature_magnitude(df, path, feature_list, dot_size=0.07, x_name='PC1', y_name='PC2', vmax=None)

draw_cluster_distribution_heatmap(df, path, file_name='type_killingkmeans_heatmap', condition_name='type', cluster_type='killing_kmeans')
draw_cluster_distribution_heatmap(df, path, file_name='kmeans_killingkmeans_heatmap', condition_name='kmeans', cluster_type='killing_kmeans')