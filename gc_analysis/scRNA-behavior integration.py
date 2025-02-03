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
"""Generates Data for scRNA-seq behavior integration """
import scipy.stats
from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import ZoneSignal
import scanpy as sc

#################################### RNA data preprocessing ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'
adata_rna = sc.read_h5ad(path+'GCB_only_cluster.h5ad')

adata_rna.layers['counts'] = adata_rna.X # Save raw counts
sc.pp.normalize_total(adata_rna, target_sum=10000) # Changes adata.X
sc.pp.log1p(adata_rna) # change to log counts (Changes adata.X)
adata_rna.raw = adata_rna  # Save pre normalized counts (without this rank_genes_groups have nan logFC)

sc.pp.highly_variable_genes(adata_rna, flavor="seurat", n_top_genes=500)
adata_rna = adata_rna[:, adata_rna.var['highly_variable']]

sc.pp.scale(adata_rna, max_value=10)  # standard scale (mean=0, variance=1)  (Change adata.X)

sc.tl.pca(adata_rna, svd_solver='arpack')
sc.pp.neighbors(adata_rna, n_pcs=50)
sc.tl.umap(adata_rna)
#################################### Behavior data preprocessing ####################################
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
df.loc[(df['avg_zone'] < 0.8) & (df['avg_zone'] >= 0.4), 'Zone'] = 'DZ-LZ'
df.loc[(df['avg_zone'] < 1.2) & (df['avg_zone'] >= 0.8), 'Zone'] = 'LZ'
df.loc[(df['avg_zone'] < 1.6) & (df['avg_zone'] >= 1.2), 'Zone'] = 'LZ'
df.loc[(df['avg_zone'] <= 2) & (df['avg_zone'] >= 1.6), 'Zone'] = 'LZ'


df.loc[(df['avg_zone'] < 0.4) & (df['avg_zone'] >= 0), 'Zone1'] = 'DZ'
df.loc[(df['avg_zone'] < 0.8) & (df['avg_zone'] >= 0.4), 'Zone1'] = 'DZ-sLZ'
df.loc[(df['avg_zone'] < 1.2) & (df['avg_zone'] >= 0.8), 'Zone1'] = 'sLZ'
df.loc[(df['avg_zone'] < 1.6) & (df['avg_zone'] >= 1.2), 'Zone1'] = 'sLZ-dLZ'
df.loc[(df['avg_zone'] <= 2) & (df['avg_zone'] >= 1.6), 'Zone1'] = 'dLZ'

print(df[df['Zone1']=='DZ'].shape[0], df[df['Zone1']=='DZ-sLZ'].shape[0], df[df['Zone1']=='sLZ'].shape[0],
      df[df['Zone1']=='sLZ-dLZ'].shape[0], df[df['Zone1']=='dLZ'].shape[0])

# duration=20
# label_expanded = np.repeat(df['Zone'], duration).reset_index(drop=True)
# df_duration['Zone_label'] = label_expanded

df.columns.get_loc('inst_angle_cosine_similarity_entropies')
motility_data = df.iloc[:,8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)

adata_behavior = sc.AnnData(X=motility_data, var=motility_data.columns.to_frame(name='features'), obs = motility_data.index.to_frame(name='idx'))

sc.pp.scale(adata_behavior, max_value=10)  # standard scale (mean=0, variance=1)
sc.tl.pca(adata_behavior, svd_solver='arpack')
sc.pp.neighbors(adata_behavior, n_pcs=50)
sc.tl.umap(adata_behavior)

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\GCB space\RNA-behavior\\'

#################################### correlation btw Behavior and RNA features ####################################
n_samples = adata_behavior.shape[0]
n_samples_rna = adata_rna.shape[0]

np.random.seed(42)
random_traj_idxs = np.random.choice(range(0, n_samples_rna), size=n_samples, replace=False)

df_corr1 = pd.DataFrame(adata_rna.X[random_traj_idxs], columns=adata_rna.var.index)
df_corr2 = pd.DataFrame(adata_behavior.X, columns=adata_behavior.var.index)

#corr = df_corr1.corrwith(df_corr2, axis=0)

corr = pd.DataFrame() # Correlation matrix
df_p = pd.DataFrame()  # Matrix of p-values
for x in df_corr1.columns:
    for y in df_corr2.columns:
        r, p = scipy.stats.pearsonr(df_corr1[x], df_corr2[y])
        corr.loc[x,y] = r
        df_p.loc[x,y] = p

k = corr.isnull().any(axis=1)
null_genes = k.index[k==True]
corr = corr.drop(null_genes, axis=0)

# columns_to_drop = corr.columns[(abs(corr) < 0.4).all(axis=0)]
# corr_filtered1 = corr.drop(columns=columns_to_drop)
#
# index_to_drop = corr_filtered1.index[(abs(corr_filtered1) < 0.4).all(axis=1)]
# corr_filtered = corr_filtered1.drop(index=index_to_drop)

fig, ax = plt.subplots()
# ax = sns.heatmap(corr, annot=False, cmap=cmc.cork, alpha=0.9, linewidths=1, linecolor='white', square= True,
#                  annot_kws={'size': 4, 'weight':'bold'},
#                  cbar_kws= {"shrink":0.7, 'label':'Correlation'})

kws = dict(cbar_kws=dict(ticks=[0.1, 0, -0.1], orientation='horizontal'), vmin=-0.1, vmax=0.1 )

g=sns.clustermap(corr, annot=False, cmap=cmc.cork, col_cluster=False, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=1,
**kws,
figsize = (20, 60),
dendrogram_ratio=0.1,
cbar=True
)
# g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=4, rotation=35, rotation_mode='anchor', ha='right')
# g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=4, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1, 0.06*3, 0.02])

g.ax_cbar.set_title('Correlation', fontsize=7)
g.ax_cbar.tick_params(axis='x', length=5, labelsize=7)

# cax = ax.figure.axes[-1]  # colorbar
# cax.tick_params(labelsize=4)  # fontsize of tick label
# cax.yaxis.label.set_size(6)  # fontsize of color bar y label

# plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='bold')
# plt.yticks(fontsize=4, color='0.2', weight='bold')
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=7, rotation=35, rotation_mode='anchor',ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=7, va='center')

plt.savefig(path+'RNA vs behavior features correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/RNA vs behavior features correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()

#################################### Prepare Optimal Transport Domain Adaptation ####################################
import ot

df_xs = pd.DataFrame(adata_rna.X, columns=adata_rna.var.index)
df_xt = pd.DataFrame(adata_behavior.X, columns=adata_behavior.var.index)
df_xs['cluster'] = adata_rna.obs['cluster'].values
df_xt['cluster'] = df['Zone']
df_xs['Type'] = adata_rna.obs['Type'].values
df_xt['Type'] = df['Type']
mapping = {'mt_B-cell': 'EZH2', 'wt_B-cell': 'WT'}
df_xt['Type'].replace(mapping, inplace=True)
df_xt['Zone'] =  df['Zone1']

df_xs['UMAP1'] = adata_rna.obsm['X_umap'][:, 0]
df_xs['UMAP2'] = adata_rna.obsm['X_umap'][:, 1]
df_xt['UMAP1'] = adata_behavior.obsm['X_umap'][:, 0]
df_xt['UMAP2'] = adata_behavior.obsm['X_umap'][:, 1]

xs = adata_rna.obsm['X_pca']
xt = adata_behavior.obsm['X_pca']

ys_temp = adata_rna.obs['cluster'].values
yt_temp = df['Zone'].values
ys = np.array( [0 if label == 'DZ' else 1 if label == 'DZ-LZ' else 2 if label == 'LZ' else -1 for label in ys_temp] )
yt = np.array( [0 if label == 'DZ' else 1 if label == 'DZ-LZ' else 2 if label == 'LZ' else -1 for label in yt_temp] )

# df_X = pd.DataFrame(adata_rna.X, columns=adata_rna.var.index)
# df_Y = pd.DataFrame(adata_behavior.X, columns=adata_behavior.var.index)
# df_X['cluster'] = adata_rna.obs['cluster'].values
# df_Y['cluster'] = df['Zone']

Ns = xs.shape[0]
Nt = xt.shape[0]

########## Unsupervised Domain Adaptation ##########
ot_sinkhorn = ot.da.SinkhornTransport(reg_e=3)
ot_sinkhorn.fit(Xs=xs, Xt=xt)
trans_xs = ot_sinkhorn.transform(Xs=xs)
trans_xt = ot_sinkhorn.inverse_transform(Xt=xt)


# fig, ax = plt.subplots(figsize=(4, 4))
# plt.imshow(ot_sinkhorn.coupling_, interpolation='nearest')
# plt.show()

# ot_emd = ot.da.EMDTransport()
# ot_emd.fit(Xs=xs, Xt=xt)
# trans_xs = ot_emd.transform(Xs=xs)
#
# fig, ax = plt.subplots(figsize=(4, 4))
# plt.imshow(ot_emd.coupling_, interpolation='nearest')
# plt.show()

file_name = 'Unsupervised DA'
condition_name = 'cluster'
colors = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')
cmap_beh = ListedColormap(colors[:pd.unique(df_xt[condition_name]).shape[0]])
cmap_rna = cmc.batlow
font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(2,3, figsize=(18, 12))

ax[0, 0].scatter(xs[:, 0], xs[:, 1], marker='o',
           label='RNA', alpha=0.3, s=1)
ax[0, 0].scatter(xt[:, 0], xt[:, 1],
           marker='+', label='Behavior', s=1)
ax[0, 0].legend()

ax[0, 1].scatter(xs[:, 0], xs[:, 1], marker='o',
           label='RNA', alpha=0.3, s=1)
ax[0, 1].scatter(trans_xt[:, 0], trans_xt[:, 1],
           marker='+', label='Transported Behavior', s=1)
ax[0, 1].legend()

ax[0, 2].scatter(xt[:, 0], xt[:, 1], marker='o',
           label='Behavior', alpha=0.3, s=1)
ax[0, 2].scatter(trans_xs[:, 0], trans_xs[:, 1],
           marker='+', label='Transported RNA', s=1)
ax[0, 2].legend()


scatter1 = ax[1, 0].scatter(xs[:, 0], xs[:, 1], marker='o', alpha=0.3, s=1, label = df_xs[condition_name],
                 c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
                            [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), cmap='Set1')
scatter2 = ax[1, 0].scatter(xt[:, 0], xt[:, 1], marker='+', s=1, label = df_xt[condition_name],
                 c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
                            [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), cmap=cmap_beh)
# ax[1, 0].legend()
# handles, labels = scatter1.legend_elements(num=None)
# ax[1, 0].legend(handles=handles, labels=list(np.unique(df_xs[condition_name])),
#            bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
#            fontsize=3, frameon=False, markerscale=0.3)

# ax[1, 1].scatter(xs[:, 0], xs[:, 1], marker='o', alpha=0.3, s=1, label = df_xs[condition_name],
#                  c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
#                             [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), cmap=cmap_rna)

ax[1, 1].scatter(xs[:, 0], xs[:, 1], marker='o', alpha=0.2, s=1, label = df_xs[condition_name], color='gray')

ax[1, 1].scatter(trans_xt[:, 0], trans_xt[:, 1], marker='+', s=1, label = df_xt[condition_name],
                 c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
                            [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), cmap=cmap_beh)
# ax[1, 1].legend()

# ax[1, 2].scatter(xt[:, 0], xt[:, 1], marker='o', alpha=0.3, s=1, label = df_xt[condition_name],
#                  c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
#                             [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), cmap=cmap_beh)

ax[1, 2].scatter(xt[:, 0], xt[:, 1], marker='o', alpha=0.2, s=1, label = df_xt[condition_name], color='gray')

ax[1, 2].scatter(trans_xs[:, 0], trans_xs[:, 1], marker='+', s=1, label = df_xs[condition_name],
                 c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
                            [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), cmap=cmap_rna)
# ax[1, 2].legend()

plt.title('Unsupervised DA')

plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

plt.clf()
plt.close()


########## Semi-supervised DA (choose which algorithm) ##########

ot_sinkhorn = ot.da.SinkhornTransport(reg_e=3)
ot_sinkhorn.fit(Xs=xs, Xt=xt, ys=ys, yt=yt)
#trans_xs = ot_sinkhorn.transform(Xs=xs)
trans_xt = ot_sinkhorn.inverse_transform(Xt=xt)




ot_gw = ot.da.LinearGWTransport()
ot_gw.fit(Xs=xs, Xt=xt, ys=ys, yt=yt)
#trans_xs = ot_gw.transform(Xs=xs)
trans_xt = ot_gw.inverse_transform(Xt=xt)


ot_mt = ot.da.MappingTransport(mu=1, eta=0.001)
ot_mt.fit(Xs=xs, Xt=xt, ys=ys, yt=yt)
#trans_xs = ot_gw.transform(Xs=xs)
trans_xt = ot_mt.inverse_transform(Xt=xt)

adata_behavior = sc.AnnData(X=motility_data, var=motility_data.columns.to_frame(name='features'), obs = motility_data.index.to_frame(name='idx'))

ot.da.UnbalancedSinkhornTransport

ot.da.LinearTransport
Agw, bgw = ot.empirical_gaussian_gromov_wasserstein_mapping(xs, xt)

ot.da.LinearGWTransport
ot.gromov_wasserstein()
# fig, ax = plt.subplots(figsize=(4, 4))
# plt.imshow(ot_sinkhorn.coupling_, interpolation='nearest')
# plt.show()

# ot_emd = ot.da.EMDTransport()
# ot_emd.fit(Xs=xs, Xt=xt)
# trans_xs = ot_emd.transform(Xs=xs)
#
# fig, ax = plt.subplots(figsize=(4, 4))
# plt.imshow(ot_emd.coupling_, interpolation='nearest')
# plt.show()

file_name = 'Semi-supervised DA'
condition_name = 'cluster'
colors = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')
cmap_beh = ListedColormap(colors[:pd.unique(df_xt[condition_name]).shape[0]])
cmap_rna = cmc.batlow
font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(2,3, figsize=(18, 12))

ax[0, 0].scatter(xs[:, 0], xs[:, 1], marker='o',
           label='RNA', alpha=0.3, s=1)
ax[0, 0].scatter(xt[:, 0], xt[:, 1],
           marker='+', label='Behavior', s=1)
ax[0, 0].legend()

ax[0, 1].scatter(xs[:, 0], xs[:, 1], marker='o',
           label='RNA', alpha=0.3, s=1)
ax[0, 1].scatter(trans_xt[:, 0], trans_xt[:, 1],
           marker='+', label='Transported Behavior', s=1)
ax[0, 1].legend()

ax[0, 2].scatter(xt[:, 0], xt[:, 1], marker='o',
           label='Behavior', alpha=0.3, s=1)
ax[0, 2].scatter(trans_xs[:, 0], trans_xs[:, 1],
           marker='+', label='Transported RNA', s=1)
ax[0, 2].legend()


scatter1 = ax[1, 0].scatter(xs[:, 0], xs[:, 1], marker='o', s=1, label = df_xs[condition_name],
                 c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
                            [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), cmap='Set1')
scatter2 = ax[1, 0].scatter(xt[:, 0], xt[:, 1], marker='+', s=1, label = df_xt[condition_name],
                 c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
                            [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), cmap=cmap_beh)
#ax[1, 0].legend()
handles, labels = scatter1.legend_elements(num=None)
ax[1, 0].legend(handles=handles, labels=list(np.unique(df_xs[condition_name])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)

# ax[1, 1].scatter(xs[:, 0], xs[:, 1], marker='o', alpha=0.3, s=1, label = df_xs[condition_name],
#                  c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
#                             [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), cmap=cmap_rna)

ax[1, 1].scatter(xs[:, 0], xs[:, 1], marker='o', alpha=0.2, s=1, label = df_xs[condition_name], color='gray')

scatter = ax[1, 1].scatter(trans_xt[:, 0], trans_xt[:, 1], marker='+', s=1, label = df_xt[condition_name],
                 c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
                            [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), cmap=cmap_beh)

handles, labels = scatter.legend_elements(num=None)
ax[1, 1].legend(handles=handles, labels=list(np.unique(df_xt[condition_name])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)

# ax[1, 2].scatter(xt[:, 0], xt[:, 1], marker='o', alpha=0.3, s=1, label = df_xt[condition_name],
#                  c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
#                             [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), cmap=cmap_beh)

ax[1, 2].scatter(xt[:, 0], xt[:, 1], marker='o', alpha=0.2, s=1, label = df_xt[condition_name], color='gray')

scatter = ax[1, 2].scatter(trans_xs[:, 0], trans_xs[:, 1], marker='+', s=1, label = df_xs[condition_name],
                 c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
                            [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), cmap=cmap_beh)

handles, labels = scatter.legend_elements(num=None)
ax[1, 2].legend(handles=handles, labels=list(np.unique(df_xs[condition_name])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)


plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

plt.clf()
plt.close()



########## Semi-supervised Domain Adaptation (Reference as RNA-seq map) ##########
wasses_dict = {}
for reg_e in [2, 2.5, 3, 3.5, 4, 4.5, 5]:
    file_name = 'Sinkhorn DA_rege_%s' % reg_e
    condition_name = 'cluster'
    colors = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255','#661100', '#888888')
    cmap_beh = ListedColormap(colors[:pd.unique(df_xt[condition_name]).shape[0]])
    cmap_rna = cmc.batlow

    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=reg_e)
    ot_sinkhorn.fit(Xs=xs, Xt=xt, ys=ys, yt=yt)
    trans_xs = ot_sinkhorn.transform(Xs=xs)
    trans_xt = ot_sinkhorn.inverse_transform(Xt=xt)
    df_xt['trans_UMAP1'] = trans_xt[:, 0]
    df_xt['trans_UMAP2'] = trans_xt[:, 1]

    # fig, ax = plt.subplots(figsize=(4, 4))
    # plt.imshow(ot_sinkhorn.coupling_, interpolation='nearest')
    # plt.show()

    # ot_emd = ot.da.EMDTransport()
    # ot_emd.fit(Xs=xs, Xt=xt)
    # trans_xs = ot_emd.transform(Xs=xs)
    #
    # fig, ax = plt.subplots(figsize=(4, 4))
    # plt.imshow(ot_emd.coupling_, interpolation='nearest')
    # plt.show()


    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    def wasserstein_distance(p, q, xmin, xmax, ymin, ymax, bin_num):
        xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
        ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        coords = np.array([Xgrid.flatten(), Ygrid.flatten()]).T  # (-5, -5), (-4.8, -5), ...(15,15)
        coordsSqr = np.sum(coords ** 2, 1)  # array[50, 48.02, 46.12, ... 450], shape = (10000,)
        M = coordsSqr[:, None] + coordsSqr[None, :] - 2 * coords @ coords.T
        # coordsSqr[:, None] shape = (10000, 1), coordsSqr[None, :] shape = (1, 10000)
        M[M < 0] = 0
        M = np.sqrt(M)  # M is cost matrix, shape = (bin_num^2, bin_num^2)
        M = M / M.max()
        wass = ot.sinkhorn2(p.flatten(), q.flatten(), M, 1.0)
        return wass

    def get_2d_gaussian_pdf(x, y, xmin, xmax, ymin, ymax, bin_num):
        kde_coordinate = np.vstack([x, y]) # shape = (dimension (=2), number of dataset)
        kde = scipy.stats.gaussian_kde(kde_coordinate)
        xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
        ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        # from kde.evaluate(points), points = shape ((dimension (=2), number of coordinates))
        pdf = Z.reshape(Xgrid.shape) + 1e-20
        return pdf


    bin_num = 100
    xmin = math.floor(xs[:,0].min()) - 1
    xmax = math.ceil(xs[:,0].max()) + 1
    ymin = math.floor(xs[:,1].min()) - 1
    ymax = math.ceil(xs[:,1].max()) + 1

    wasses=[]
    for typ in ['EZH2', 'WT']:
        subset_idxs = df_xs['Type'] == typ
        x = xs[:, 0][subset_idxs]
        y = xs[:, 1][subset_idxs]

        pdf1 = get_2d_gaussian_pdf(x, y, xmin, xmax, ymin, ymax, bin_num)

        subset_idxs = df_xt['Type'] == typ
        x = trans_xt[:, 0][subset_idxs]
        y = trans_xt[:, 1][subset_idxs]
        pdf2 = get_2d_gaussian_pdf(x, y, xmin, xmax, ymin, ymax, bin_num)

        wass = wasserstein_distance(pdf1, pdf2, xmin, xmax, ymin, ymax, bin_num)

        wasses.append(wass)
    wasses_dict[reg_e] = wasses

    draw_jointplot(xs='UMAP1', y='UMAP2', df=df_xs, path=path, file_name='RNA_%s'%file_name, hue="Type", colors=('#CC6677', '#888888', ),
                   hue_order=['EZH2', 'WT'], legend=False, fill=False, thresh=0.15, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')
    draw_jointplot(xs='trans_UMAP1', y='trans_UMAP2', df=df_xt, path=path, file_name='trans_behavior_%s'%file_name, hue="Type", colors=('#CC6677', '#888888', ),
                   hue_order=['EZH2', 'WT'], legend=False, fill=False, thresh=0.15, n_contours=5, alpha=1, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')



    font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    fig, ax = plt.subplots(2,3, figsize=(18, 12))

    ax[0, 0].scatter(xs[:, 0], xs[:, 1], marker='o', label='RNA', alpha=0.3, s=1)
    ax[0, 0].scatter(xt[:, 0], xt[:, 1], marker='o', label='Behavior', s=1)
    ax[0, 0].legend()

    ax[0, 1].scatter(xs[:, 0], xs[:, 1], marker='o', label='RNA', alpha=0.3, s=1)
    ax[0, 1].scatter(trans_xt[:, 0], trans_xt[:, 1], marker='+', label='Transported Behavior', s=1)
    ax[0, 1].legend()

    ax[0, 2].scatter(xs[:, 0], xs[:, 1], marker='o', alpha=0.2, s=1,  color='gray')
    scatter1 = ax[0, 2].scatter(trans_xt[:, 0], trans_xt[:, 1], marker='o', s=1, label = df_xt['Type'], c=df_xt['Type'].replace(list(np.unique(df_xt['Type'])),
                                [i for i in range(np.unique(df_xt['Type']).shape[0])]), cmap=ListedColormap(('#6699CC', '#CC6677')[:pd.unique(df_xt['Type']).shape[0]]))
    handles, labels = scatter1.legend_elements(num=None)
    ax[0, 2].legend(handles=handles, labels=list(np.unique(df_xt['Type'])),
               #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
               fontsize=6, frameon=False, markerscale=0.6)


    scatter1 = ax[1, 0].scatter(xs[:, 0], xs[:, 1], marker='o', s=1, label = df_xs[condition_name],
                     c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
                                [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), cmap='Set1')

    #ax[1, 0].legend()
    handles, labels = scatter1.legend_elements(num=None)
    ax[1, 0].legend(handles=handles, labels=list(np.unique(df_xs[condition_name])),
               #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
               fontsize=6, frameon=False, markerscale=0.6)

    # ax[1, 1].scatter(xs[:, 0], xs[:, 1], marker='o', alpha=0.3, s=1, label = df_xs[condition_name],
    #                  c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
    #                             [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), cmap=cmap_rna)

    ax[1, 1].scatter(xs[:, 0], xs[:, 1], marker='o', alpha=0.2, s=1, label = df_xs[condition_name], color='gray')

    scatter = ax[1, 1].scatter(trans_xt[:, 0], trans_xt[:, 1], marker='o', s=1, label = df_xt[condition_name],
                     c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
                                [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), cmap=cmap_beh)

    handles, labels = scatter.legend_elements(num=None)
    ax[1, 1].legend(handles=handles, labels=list(np.unique(df_xt[condition_name])),
               #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
               fontsize=6, frameon=False, markerscale=0.6)

    # ax[1, 2].scatter(xt[:, 0], xt[:, 1], marker='o', alpha=0.3, s=1, label = df_xt[condition_name],
    #                  c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
    #                             [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), cmap=cmap_beh)

    ax[1, 2].scatter(xs[:, 0], xs[:, 1], marker='o', alpha=0.2, s=1, label = df_xs[condition_name], color='gray')

    scatter = ax[1, 2].scatter(trans_xt[:, 0], trans_xt[:, 1], marker='o', s=1, label = df_xt['Zone'],
                     c=df_xt['Zone'].replace(list(np.unique(df_xt['Zone'])),
                                [i for i in range(np.unique(df_xt['Zone']).shape[0])]), cmap=ListedColormap(colors[:pd.unique(df_xt['Zone']).shape[0]]))

    handles, labels = scatter.legend_elements(num=None)
    ax[1, 2].legend(handles=handles, labels=list(np.unique(df_xt['Zone'])),
               #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
               fontsize=6, frameon=False, markerscale=0.6)


    plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

    plt.clf()
    plt.close()


############################ Validation using Deformability ###############################
import gseapy as gp
gp.get_library_name(organism='Mouse')

reactome_pathway_sets = gp.get_library(name='Reactome_2022', organism='Mouse')  # Retrieve KEGG pathway gene sets
kegg_pathway_sets = gp.get_library(name='KEGG_2019_Mouse', organism='Mouse')  # Retrieve KEGG pathway gene sets
gocc_pathway_sets = gp.get_library(name='GO_Cellular_Component_2023',organism='Mouse')  # Retrieve KEGG pathway gene sets
gomf_pathway_sets = gp.get_library(name='GO_Molecular_Function_2023',organism='Mouse')  # Retrieve KEGG pathway gene sets

filtered_list = [pathway for pathway in reactome_pathway_sets.keys() if 'rho gtpases' in pathway.lower()]
filtered_list


sig_unconverted = {}
sig_unconverted['Signaling By Rho GTPases'] = reactome_pathway_sets['Signaling By Rho GTPases R-HSA-194315']
sig_unconverted['Leukocyte transendothelial migration'] = kegg_pathway_sets['Leukocyte transendothelial migration']
sig_unconverted['Microtubule Binding'] = gomf_pathway_sets['Microtubule Binding (GO:0008017)']
sig_unconverted['Microtubule'] = gocc_pathway_sets['Microtubule (GO:0005874)']
sig_unconverted['Cytoskeleton'] = gocc_pathway_sets['Cytoskeleton (GO:0005856)']
sig_unconverted

sig_unconverted = {}
sig_unconverted['Behav3d_MP'] = ['CCT3', 'PKIA', 'FAM3C', 'SQLE', 'SERPINE2', 'CHD4', 'IARS', 'DCAF13', 'BZW2', 'NCEH1', 'SNTB2', 'NTRK1', 'BYSL',
 'ARHGEF3', 'HEG1', 'EMP1', 'AFAP1L2', 'IGF2R', 'GPR18', 'POU2AF1', 'MYO1E', 'AMIGO2', 'ATP1B1', 'YBX1', 'YBX3',
 'PRKD3', 'CRTAM', 'XCL2', 'XCL1', 'CCL1', 'PGAM1']


####### Human gene to Mouse gene transformation #######
h2m = pd.read_csv(r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'+'h2m.csv')
h2m_dict = {}
for i, row in h2m.loc[:,["external_gene_name", "mmusculus_homolog_associated_gene_name"]].iterrows():
    if row.isna().any(): continue
    h2m_dict[row['external_gene_name']] = row["mmusculus_homolog_associated_gene_name"]

h2m_dict

sig = {}
for term, genes in sig_unconverted.items():
    count=0
    new_genes = []
    for gene in genes:
        if gene in h2m_dict:
            new_genes.append(h2m_dict[gene])
            count = count + 1
        else:
            new_genes.append(gene)
    print( term, ': ', '%s/%s genes converted'%(count, len(genes)) )
    sig[term] = new_genes

gene_list = flatten_nested_dict(sig)


####### DEG of EZH2 vs WT #######
sc.tl.rank_genes_groups(adata_rna, groupby="Type", method="wilcoxon", key_added="dea_type", reference='WT')
result = adata_rna.uns["dea_type"]
groups = result["names"].dtype.names

degs = pd.DataFrame(
    {group + '_' + key: result[key][group]
    for group in groups for key in ['names','scores', 'pvals','pvals_adj','logfoldchanges']})


for pathway in ['Signaling By Rho GTPases', 'Leukocyte transendothelial migration', 'Microtubule Binding', 'Microtubule', 'Cytoskeleton']:
    gene_list = sig[pathway]
    available_gene_list = [gene for gene in gene_list if gene in adata_rna.var_names]
    file_name = 'DEG_%s'%pathway
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    #sc.pl.umap(adata_rna, color=['deformability_score_%s'%pathway], frameon=False, ax=ax, cmap=plt.cm.get_cmap('coolwarm'))

    sc.pl.dotplot(adata_rna, var_names=available_gene_list, groupby="Type", standard_scale="var", swap_axes=True, cmap="coolwarm", ax=ax)

    fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):
        os.makedirs(path + 'svg/')

    fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')




####### Quantifying Deformability score in scRNA-seq map #######
sc.tl.score_genes(adata_rna, gene_list, score_name='deformability_score')

file_name = 'deformability score_behav3d'
fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
sc.pl.umap(adata_rna, color=['deformability_score'], frameon=False, ax=ax, cmap=plt.cm.get_cmap('coolwarm'))

fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')

fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')


for pathway in ['Signaling By Rho GTPases', 'Leukocyte transendothelial migration', 'Microtubule Binding', 'Microtubule', 'Cytoskeleton']:
    gene_list = sig[pathway]

    sc.tl.score_genes(adata_rna, gene_list, score_name='deformability_score_%s'%pathway)

    file_name = 'deformability score_%s'%pathway
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    sc.pl.umap(adata_rna, color=['deformability_score_%s'%pathway], frameon=False, ax=ax, cmap=plt.cm.get_cmap('coolwarm'))

    fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):
        os.makedirs(path + 'svg/')

    fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')



reg_e = 2.5
file_name = 'Deformability on transported behavior space_rege_%s' % reg_e
condition_name = 'cluster'
colors = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255','#661100', '#888888')
# cmap_beh = ListedColormap(colors[:pd.unique(df_xt[condition_name]).shape[0]])
# cmap_rna = cmc.batlow

ot_sinkhorn = ot.da.SinkhornTransport(reg_e=reg_e)
ot_sinkhorn.fit(Xs=xs, Xt=xt, ys=ys, yt=yt)
trans_xs = ot_sinkhorn.transform(Xs=xs)
trans_xt = ot_sinkhorn.inverse_transform(Xt=xt)
df_xt['trans_UMAP1'] = trans_xt[:, 0]
df_xt['trans_UMAP2'] = trans_xt[:, 1]


font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(xs[:, 0], xs[:, 1], marker='o', alpha=0.2, s=1, label=df_xs[condition_name], color='gray')

scatter = ax.scatter(trans_xt[:, 0], trans_xt[:, 1], marker='o', s=1, label=df_xt[condition_name],
                           c=df['morpho_avg_speed'], cmap=plt.cm.get_cmap('coolwarm'))



            #format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
            #plt.xlim(xmin, xmax)
            #plt.ylim(ymin, ymax)
cbar = fig.colorbar(scatter)
cbar.ax.tick_params(labelsize=6)

fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')

fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')




file_name = 'Kmeans on transported behavior space_rege_%s' % reg_e

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(xs[:, 0], xs[:, 1], marker='o', alpha=0.2, s=1, label=df_xs[condition_name], color='gray')

scatter = ax.scatter(trans_xt[:, 0], trans_xt[:, 1], marker='o', s=1, label=df['kmeans'],
                           c=df['kmeans'].replace(list(np.unique(df['kmeans'])),
                                [i for i in range(np.unique(df['kmeans']).shape[0])]), cmap=cmc.batlow)


handles, labels = scatter.legend_elements(num=None)
ax.legend(handles=handles, labels=list(np.unique(df['kmeans'])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)
        #format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
        #plt.xlim(xmin, xmax)
        #plt.ylim(ymin, ymax)


fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')

fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
# #################################### Canonical Correlation Analysis ####################################
#
#
# df_X = pd.DataFrame(adata_rna.X[random_traj_idxs], columns=adata_rna.var.index)
# df_Y = pd.DataFrame(adata_behavior.X, columns=adata_behavior.var.index)
#
# df_X['cluster'] = adata_rna.obs['cluster'].values[random_traj_idxs]
# df_Y['cluster'] = df['Zone']
#
# grouped_A = df_X.groupby('cluster').mean()
# grouped_B = df_Y.groupby('cluster').mean()
#
# aligned_A = grouped_A.loc[grouped_A.index.intersection(grouped_B.index)]
# aligned_B = grouped_B.loc[aligned_A.index]
#
#
# from sklearn.cross_decomposition import CCA
# cca = CCA(n_components=2)
# cca.fit(df_X, df_Y)
# X_c, Y_c = cca.transform(df_X, df_Y)
#
# score = cca.score(df_X, df_Y)
#
#
#
# x_input = X_c[:, 0]
# y_input = Y_c[:, 0]
# file_name='CCA0 space type'
# condition_name='Type'
# colors = ('#fdc086', '#beaed4', '#7fc97f')
# dot_size=0.07
#
# font = {'family': 'arial',
#         'weight': 'normal',
#         'size': 8}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
# matplotlib.rcParams['lines.linewidth'] = 1
#
# from collections.abc import Iterable
# if isinstance(colors, Iterable):
#     cmap = ListedColormap(colors[:pd.unique(df[condition_name]).shape[0]])
# else:
#     cmap=colors
# xmin = math.floor(x_input.min()) - 1
# xmax = math.ceil(x_input.max()) + 1
# ymin = math.floor(y_input.min()) - 1
# ymax = math.ceil(y_input.max()) + 1
#
# fig, ax = plt.subplots(figsize=(2, 2))
# #plt.figure(figsize=(15, 10))
# scatter = ax.scatter(x_input, y_input,
#                       c=df[condition_name].replace(list(np.unique(df[condition_name])),
#                         [i for i in range(np.unique(df[condition_name]).shape[0])]),
#                       # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
#                       s=dot_size, label=df[condition_name],
#                       cmap=cmap)
#
# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax)
#
# format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
# handles, labels = scatter.legend_elements(num=None)
# plt.legend(handles=handles, labels=list(np.unique(df[condition_name])),
#            bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
#            fontsize=3, frameon=False, markerscale=0.3)
#
# plt.savefig(path + '%s.png' % file_name, dpi=300)
#
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/%s.svg' % file_name)
# plt.clf()
# plt.close()
#
#
# plt.scatter(, )
# plt.xlabel('X_c1')
# plt.ylabel('Y_c1')
# plt.title('First pair of canonical variables')
# plt.show()
#
# plt.scatter(X_c[:, 1], Y_c[:, 1])
# plt.xlabel('X_c2')
# plt.ylabel('Y_c2')
# plt.title('Second pair of canonical variables')
# plt.show()
#
#
# correlation_matrix = np.corrcoef(X_c.T, Y_c.T)
#
# # Plot the correlation matrix as a heatmap
# plt.figure(figsize=(6,4))
# sns.heatmap(correlation_matrix, annot=True, cmap='Set2', xticklabels=[
#             'X_c1', 'X_c2'], yticklabels=['Y_c1', 'Y_c2'])
# plt.title('Canonical Variables Correlation Matrix')
# plt.show()