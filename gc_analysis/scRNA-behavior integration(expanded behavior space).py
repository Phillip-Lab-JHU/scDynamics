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
import os

# Avoid Numba cache serialization failures in pynndescent/umap on Windows/IPython.
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")

import scipy.stats
from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from utils.rna_utils import *
from features.interaction import ZoneSignal
import scanpy as sc
from umap import UMAP

import warnings
warnings.filterwarnings('ignore')
#################################### Visualize Full RNA space ####################################
# path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'
# adata_rna = sc.read_h5ad(path+'GCB_only_cluster.h5ad')
#
# adata_rna.layers['counts'] = adata_rna.X # Save raw counts
# sc.pp.normalize_total(adata_rna, target_sum=10000) # Changes adata.X
# sc.pp.log1p(adata_rna) # change to log counts (Changes adata.X)
# adata_rna.raw = adata_rna  # Save pre normalized counts (without this rank_genes_groups have nan logFC)
#
# sc.pp.highly_variable_genes(adata_rna, flavor="seurat", n_top_genes=3000)
# adata_rna = adata_rna[:, adata_rna.var['highly_variable']]
#
#
# sc.pp.scale(adata_rna, max_value=10)  # standard scale (mean=0, variance=1)  (Change adata.X)
#
# #sc.tl.pca(adata_rna, svd_solver='arpack')
# sc.tl.pca(adata_rna, n_comps=50, svd_solver='arpack')
# print('total captured variance:', np.sum(adata_rna.uns['pca']['variance_ratio']))
# #sc.pp.neighbors(adata_rna, n_pcs=50)
# sc.pp.neighbors(adata_rna, n_pcs=50)
# sc.tl.umap(adata_rna)
#
# path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\\'
#
# file_name = 'Full RNA space type'
# fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
# sc.pl.umap(adata_rna, color=['cluster'], frameon=False, ax=ax, cmap='Set1')
#
# fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):
#     os.makedirs(path + 'svg/')
#
# fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

#################################### RNA data preprocessing ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'
adata_rna = sc.read_h5ad(path+'GCB_only_cluster_final.h5ad')

# adata_rna = adata_rna[(adata_rna.obs['cluster']=='DZ') | (adata_rna.obs['cluster']=='DZ-LZ') | (adata_rna.obs['cluster']=='LZ'), :]
adata_rna = adata_rna[(adata_rna.obs['cluster']=='DZ') | (adata_rna.obs['cluster']=='DZ-LZ') | (adata_rna.obs['cluster']=='LZ')
                      |(adata_rna.obs['cluster']=='Recycle'), :]
mapping = {'Recycle': 'DZ-LZ'}
adata_rna.obs['cluster'].replace(mapping, inplace=True)

adata_rna.layers['counts'] = adata_rna.X # Save raw counts
sc.pp.normalize_total(adata_rna, target_sum=10000) # Changes adata.X
sc.pp.log1p(adata_rna) # change to log counts (Changes adata.X)
adata_rna.raw = adata_rna  # Save pre normalized counts (without this rank_genes_groups have nan logFC)

sc.pp.highly_variable_genes(adata_rna, flavor="seurat", n_top_genes=3000)
adata_rna = adata_rna[:, adata_rna.var['highly_variable']]


sc.pp.scale(adata_rna, max_value=10)  # standard scale (mean=0, variance=1)  (Change adata.X)

#sc.tl.pca(adata_rna, svd_solver='arpack')
sc.tl.pca(adata_rna, n_comps=330, svd_solver='arpack')
print('total captured variance:', np.sum(adata_rna.uns['pca']['variance_ratio']))
#sc.pp.neighbors(adata_rna, n_pcs=50)
import os
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")

sc.pp.neighbors(adata_rna, n_pcs=330)
sc.tl.umap(adata_rna)

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\\'
#color_list = ('#F06293', '#AF79C4', '#5B6BBF', '#FFC11C', '#D1917C', '#4DB6AC', '#00BCD4', '#A1877E', '#8F96A8', '#B02E8B', '#EB974D','#BCBCBC',)
color_list = [(0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0), (0.6196078431372549, 0.8549019607843137, 0.8980392156862745, 1.0),
              (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)]

df = pd.DataFrame(adata_rna.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
df['cluster'] = adata_rna.obs['cluster'].values
df['Type'] = adata_rna.obs['Type'].values

draw_umap_space(df, path, file_name='space_cluster', condition_name='cluster',
                colors=color_list, x_name='UMAP1', y_name='UMAP2', dot_size=0.7)
cell_counts = adata_rna.obs['cluster'].value_counts().sort_index()
print(cell_counts)

draw_umap_space(df, path, file_name='space_Type', condition_name='Type',
                colors=('orange', 'blue'), x_name='UMAP1', y_name='UMAP2', dot_size=0.7)
cell_counts = adata_rna.obs['Type'].value_counts().sort_index()
print(cell_counts)

# file_name = 'RNA space type'
# fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
# sc.pl.umap(adata_rna, color=['cluster'], frameon=False, ax=ax, cmap='Set1')
#
# fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):
#     os.makedirs(path + 'svg/')
#
# fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

#################################### Behavior data preprocessing ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure6. Expanded behavior\\'
df = pd.read_parquet(path+'Expanded_behavior.parquet')

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

df.columns.get_loc('morpho_displ_autocorr_3')
motility_data = df.iloc[:,8:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',],axis=1)


df.columns.get_loc('quality_FDC_approach_times')
df.columns.get_loc('dlz_resident_persistences')

colocalization_data = df.iloc[:,148:289].drop(['FDC_diff_distance_cov', 'T_diff_distance_cov',
                                         'Core_diff_distance_cov', 'LZ_diff_distance_cov', 'DZ_diff_distance_cov',
                                         'DZ_distance_autocorr_1', 'DZ_distance_autocorr_2', 'DZ_distance_autocorr_3',
                                         'DZ_diff_distance_autocorr_1', 'DZ_diff_distance_autocorr_2', 'DZ_diff_distance_autocorr_3',
                                         'LZ_diff_distance_autocorr_1', 'LZ_diff_distance_autocorr_2', 'LZ_diff_distance_autocorr_3',
                                         'Core_distance_autocorr_1', 'Core_distance_autocorr_2', 'Core_distance_autocorr_3',
                                         'Core_diff_distance_autocorr_1', 'Core_diff_distance_autocorr_2', 'Core_diff_distance_autocorr_3',
                                         'Core_distance_variance', 'Core_diff_distance_variance', 'DZ_distance_variance', 'DZ_diff_distance_variance',
                                         'FDC_diff_distance_variance', 'FDC_distance_variance', 'LZ_distance_variance', 'LZ_diff_distance_variance',
                                         'T_diff_distance_variance', 'T_distance_variance',
                                         'PC1', 'PC2', 'kmeans'
                                         ], axis=1)
columns_with_nan = colocalization_data.columns[colocalization_data.isna().any()].tolist()
colocalization_data = colocalization_data.drop(columns_with_nan, axis=1)

input_data = pd.concat([motility_data, colocalization_data], axis=1)


adata_behavior = sc.AnnData(X=input_data, var=input_data.columns.to_frame(name='features'), obs = input_data.index.to_frame(name='idx'))

sc.pp.scale(adata_behavior, max_value=10)  # standard scale (mean=0, variance=1)
#sc.tl.pca(adata_behavior, svd_solver='arpack')
sc.tl.pca(adata_behavior, n_comps=36, svd_solver='arpack')
print('total captured variance:', np.sum(adata_behavior.uns['pca']['variance_ratio']))
#sc.pp.neighbors(adata_behavior, n_pcs=50)
sc.pp.neighbors(adata_behavior, n_pcs=36)
sc.tl.umap(adata_behavior)

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\\'


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
# df_xs['ot_label'] = df_xs['Type'].astype(str) + '_' + df_xs['cluster'].astype(str)
# df_xt['ot_label'] = df_xt['Type'].astype(str) + '_' + df_xt['cluster'].astype(str)
df_xs['ot_label'] = df_xs['cluster'].astype(str)
df_xt['ot_label'] = df_xt['cluster'].astype(str)

print(np.unique(df_xs['ot_label']), np.unique(df_xt['ot_label']))

df_xt['Zone'] = df['Zone1']
df_xs['UMAP1'] = adata_rna.obsm['X_umap'][:, 0]
df_xs['UMAP2'] = adata_rna.obsm['X_umap'][:, 1]
df_xt['UMAP1'] = adata_behavior.obsm['X_umap'][:, 0]
df_xt['UMAP2'] = adata_behavior.obsm['X_umap'][:, 1]

xs = adata_rna.obsm['X_pca']
xt = adata_behavior.obsm['X_pca']
ys_temp = df_xs['ot_label'].values
yt_temp = df_xt['ot_label'].values

# label_mapping = {'WT_DZ': 0, 'WT_DZ-LZ': 1, 'WT_LZ': 2,
#                  'EZH2_DZ': 3, 'EZH2_DZ-LZ': 4, 'EZH2_LZ': 5}
label_mapping = {'DZ': 0, 'DZ-LZ': 1, 'LZ': 2}
# Apply the mapping to your labels
ys = np.array( [label_mapping[label] for label in ys_temp] )
yt = np.array( [label_mapping[label] for label in yt_temp] )
# ys = np.array( [0 if label == 'WT_DZ' else 1 if label == 'DZ-LZ' else 2 if label == 'LZ' else -1 for label in ys_temp] )
# yt = np.array( [0 if label == 'DZ' else 1 if label == 'DZ-LZ' else 2 if label == 'LZ' else -1 for label in yt_temp] )

# df_X = pd.DataFrame(adata_rna.X, columns=adata_rna.var.index)
# df_Y = pd.DataFrame(adata_behavior.X, columns=adata_behavior.var.index)
# df_X['cluster'] = adata_rna.obs['cluster'].values
# df_Y['cluster'] = df['Zone']

Ns = xs.shape[0]
Nt = xt.shape[0]
a, b = np.ones((Ns,)) / Ns, np.ones((Nt,)) / Nt  # assume uniform distribution on samples

C1 = scipy.spatial.distance.cdist(xs, xs)  # Distance matrix for source (RNA) samples
C2 = scipy.spatial.distance.cdist(xt, xt)  # Distance matrix for target (behavior) samples

labels = np.array(['source'] * Ns + ['target'] * Nt)

#################################### Compute RNA space deformability score ####################################

####### Human gene o Mouse gene transformation #######

import gseapy as gp
gp.get_library_name(organism='Mouse')

reactome_pathway_sets = gp.get_library(name='Reactome_2022', organism='Mouse')  # Retrieve KEGG pathway gene sets
kegg_pathway_sets = gp.get_library(name='KEGG_2019_Mouse', organism='Mouse')  # Retrieve KEGG pathway gene sets
gocc_pathway_sets = gp.get_library(name='GO_Cellular_Component_2023',organism='Mouse')  # Retrieve KEGG pathway gene sets
gomf_pathway_sets = gp.get_library(name='GO_Molecular_Function_2023',organism='Mouse')  # Retrieve KEGG pathway gene sets

filtered_list = [pathway for pathway in kegg_pathway_sets.keys() if 'antigen' in pathway.lower()]
filtered_list


sig_unconverted = {}
sig_unconverted['Signaling By Rho GTPases'] = reactome_pathway_sets['Signaling By Rho GTPases R-HSA-194315']
sig_unconverted['Leukocyte transendothelial migration'] = kegg_pathway_sets['Leukocyte transendothelial migration']
sig_unconverted['Microtubule Binding'] = gomf_pathway_sets['Microtubule Binding (GO:0008017)']
sig_unconverted['Microtubule'] = gocc_pathway_sets['Microtubule (GO:0005874)']
sig_unconverted['Cytoskeleton'] = gocc_pathway_sets['Cytoskeleton (GO:0005856)']
sig_unconverted['MHC class 2 antigen presentation'] = reactome_pathway_sets['MHC Class II Antigen Presentation R-HSA-2132295']
sig_unconverted['antigen processing and presentation'] = kegg_pathway_sets['Antigen processing and presentation']

sig_unconverted['Behav3d_MP'] = ['CCT3', 'PKIA', 'FAM3C', 'SQLE', 'SERPINE2', 'CHD4', 'IARS', 'DCAF13', 'BZW2', 'NCEH1', 'SNTB2', 'NTRK1', 'BYSL',
 'ARHGEF3', 'HEG1', 'EMP1', 'AFAP1L2', 'IGF2R', 'GPR18', 'POU2AF1', 'MYO1E', 'AMIGO2', 'ATP1B1', 'YBX1', 'YBX3',
 'PRKD3', 'CRTAM', 'XCL2', 'XCL1', 'CCL1', 'PGAM1']

sig_unconverted['Behav3d_MP_filtered'] = ['CCT3', 'SQLE', 'CHD4', 'DCAF13', 'BZW2', 'YBX1', 'YBX3', 'PGAM1']

sig_unconverted['Holmes_2020 BCR signaling'] = ['CD79A', 'CD79B', 'CD19', 'LYN', 'BLNK', 'BTK', 'CD72', 'CD22', 'PTPN6', 'SLA', 'FCRL2']
sig_unconverted['Holmes_2020 CD40 signaling'] = ['CD40', 'TRAF1', 'ICAM1', 'CD80', 'CD86', 'CFLAR', 'BCL2A1', 'BCL2L1', 'MIR155HG', 'EBI3', 'CD58', 'CCL22', 'STAT5A']
sig_unconverted['Holmes_2020 NF-kappaB'] = ['NFKB1', 'REL', 'RELB', 'NFKB2', 'NFKBIA', 'NFKBIE', 'MYD88', 'TNFAIP3']
sig_unconverted['Holmes_2020 MYC'] = ['MYC', 'BATF', 'GPR183', 'CD44']
sig_unconverted['Holmes_2020 DZ'] = ['CXCR4', 'FOXP1', 'CD27']
sig_unconverted['Holmes_2020 PreM'] = ['BACH2', 'BANK1', 'RASGRP2', 'CCR6', 'CELF2']
#sig_unconverted['Holmes_2020 IG'] = ['IGHM', 'IGHG1', 'IGHA1']


h2m = pd.read_csv(r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'+'h2m.csv')
h2m_dict = {}
for i, row in h2m.loc[:,["external_gene_name", "mmusculus_homolog_associated_gene_name"]].iterrows():
    if row.isna().any(): continue
    h2m_dict[row['external_gene_name']] = row["mmusculus_homolog_associated_gene_name"]


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


# from collections import Counter
# gene_counter = Counter(gene for genes in sig.values() for gene in genes)
# overlapping_genes = [gene for gene, count in gene_counter.items() if count >= 4]
# gene_list = overlapping_genes
# sig['Pathway overlapped genes'] = gene_list

sig['Beguelin_2020 FDC interaction'] = ['Bcr', 'Tnfrsf13c', 'Itgb2', 'Itgb4', 'Ighg1', 'Tnf', 'Lta', 'Ltb', 'Itga4']
sig['Beguelin_2020 Tfh interaction'] = ['Tnfrsf14', 'Icam1', 'Basp1', 'Egr2', 'Cd69', 'Itgam', 'Ptger4', 'Icosl', 'Socs3', 'Ciita', 'Cd40']
sig['Beguelin_2020 LZ and anti-apoptosis'] = ['Cd52', 'Mreg', 'Aldoc', 'Bcl2a1b', 'Cbx8']
sig['Beguelin_2020 DZ hallmark'] = ['Hmmr', 'Lgr5', 'Ptgr1', 'Pif1', 'Serinc5', 'Bcl2l11', 'Bcl2l14']
sig['Beguelin_2020 CC recycling'] = ['Pde3b', 'Klhl5', 'Ankrd28', 'Mycbpap', 'Bag3', 'Stag3', 'Tjp2', 'Tspan5', 'Kcna3', 'Abi2', 'Irak1bp1', 'Morn4']

custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\signatures\\'
df_mot_associated_genes = pd.read_csv(custom_sig_path+'Motility-Associated_Genes.csv')
sig['Motility associated signatures'] = df_mot_associated_genes["Gene"].values


# Bcr (B cell receptor): Binds to antigen of FDC (B-FDC interaction)
# Tnfrsf13c (BAFFR): Binds to BAFF of FDC (B-FDC interaction)
# Lta, Ltb (lymphotoxin alpha, beta) : Binds to Ltbr of FDC (B-FDC interaction)
# Itgb2 (LFA-1 / integrin) : Binds to ICAM-1 of FDC (B-FDC interaction)
# Itga4 (VLA-4) : Binds to VCAM-1 of FDC (B-FDC interaction)

# Ciita (MHC2): Binds to TCR of Tfh (B-Tfh interaction)
# Cd40: Binds to Cd40L of Tfh (B-Tfh interaction)
# Icam1: Binds to LFA-1 of Tfh (B-Tfh interaction)
# Icosl: Binds to Icos of Tfh (B-Tfh interaction)

for pathway in ['Signaling By Rho GTPases', 'Leukocyte transendothelial migration', 'Microtubule Binding', 'Microtubule', 'Cytoskeleton',
                'Behav3d_MP', 'Behav3d_MP_filtered', 'Motility associated signatures']:
    gene_list = sig[pathway]

    sc.tl.score_genes(adata_rna, gene_list, score_name='deformability_score_%s'%pathway)

    # file_name = 'deformability score_%s' % pathway
    # fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    # sc.pl.umap(adata_rna, color=['deformability_score_%s' % pathway], frameon=False, ax=ax, show=False,
    #            cmap=plt.cm.get_cmap('coolwarm'))
    #
    # fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
    # if not os.path.isdir(path + 'svg/'):
    #     os.makedirs(path + 'svg/')
    # fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')


for pathway in ['MHC class 2 antigen presentation', 'antigen processing and presentation', 'Beguelin_2020 FDC interaction', 'Beguelin_2020 Tfh interaction']:
    gene_list = sig[pathway]
    sc.tl.score_genes(adata_rna, gene_list, score_name='interaction_score_%s'%pathway)

#################### Pick shared feature (prior knowledge to link two dataset) for W-type OT #####################



####### Adjust cost matrix (M) for matching labels #######

########################### Hyperparameter exploration ############################

for label_coeff in [47, 100, 500, 1000, 2000, 3000, 4000, 5000, 10000]:
    for alpha in [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2]:
        for epsilon in [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 5]:
            pathway = 'Motility associated signatures'
            # 'Signaling By Rho GTPases', 'Leukocyte transendothelial migration', 'Microtubule Binding', 'Microtubule', 'Cytoskeleton',
            # 'Behav3d_MP', 'Behav3d_MP_filtered', 'Pathway overlapped genes'

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            shared_feature_s = scaler.fit_transform(np.array(adata_rna.obs['deformability_score_%s' % pathway])[:, np.newaxis])
            shared_feature_t = scaler.fit_transform(np.array(df['morpho_avg_speed'])[:, np.newaxis])
            # print(np.min(shared_feature_s), np.max(shared_feature_s))
            # print(np.min(shared_feature_t), np.max(shared_feature_t))

            # pathway = 'FDC interaction'
            # from sklearn.preprocessing import StandardScaler
            # scaler = StandardScaler()
            # shared_feature_s = scaler.fit_transform( np.array( adata_rna.obs['interaction_score_%s'%pathway] )[:, np.newaxis] )
            # shared_feature_t = scaler.fit_transform( np.array( df['FDC_contact_persistences'] )[:, np.newaxis] )

            M = ot.dist(shared_feature_s, shared_feature_t)  # (Ns, Nt)
            label_unmatch = (ys[:, np.newaxis] != yt[np.newaxis, :])  # (Ns, Nt) of True if label unmatched, and False otherwise
            cost_correction = label_unmatch * label_coeff
            #M = cost_correction
            M= np.maximum(M, cost_correction)  # Element-wise maximum operation

            ####### Solve FGW-type OT #######
            #alpha=0.001 # Increasing alpha increases GW-type (structural correspondence) and decrease W-type (shared space)
            #P = ot.gromov.fused_gromov_wasserstein(M, C1, C2, a, b, alpha=alpha) # 0<= alpha <=0.1 is good start
            #alpha = 0.3
            #epsilon = 3

            P = ot.gromov.entropic_fused_gromov_wasserstein(M, C1, C2, a, b, alpha=alpha, epsilon=epsilon, max_iter=1e4)
            P_barycentric = P.T / np.sum(P, axis=0)[:, np.newaxis] # For each row sum column-wise -> Dividie that number for each row
            # Without this we don't get accurate transformation
            P_barycentric = np.nan_to_num(P_barycentric, nan=0, posinf=0, neginf=0)

            trans_xt = P_barycentric@xs

            ##### Transform source labels to predicted target labels #####
            P_barycentric = P/np.sum(P, axis=0)[np.newaxis, :]
            P_barycentric = np.nan_to_num(P_barycentric, nan=0, posinf=0, neginf=0)
            labels_u, labels_idx = np.unique(ys, return_inverse=True)  # unique labels, list of label indexes (N, )
            n_labels = labels_u.shape[0]
            masks = np.eye(n_labels)[labels_idx]  # Hot encoded vector matrix (N, # of unique labels)
            trans_yt_prob = (masks.T@P_barycentric).T
            trans_yt = np.argmax(trans_yt_prob, axis=1)

            y_pred = trans_yt
            y_test = yt

            accuracy = np.sum(y_test == y_pred) / y_test.size
            print(pathway, label_coeff, alpha, epsilon, accuracy)

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            norm_cm = cm / np.sum(cm, axis=1)[:, np.newaxis]

            import seaborn as sns

            file_name = 'confusion matrix labelcoeff_%s_alpha_%s_eps_%s'%(label_coeff, alpha, epsilon)
            font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 8}
            matplotlib.rc('font', **font)

            fig, ax = plt.subplots(figsize=(4,4))
            ax = sns.heatmap(norm_cm, annot=True, annot_kws={'size': 16, 'weight': 'normal'}, linewidths=0.5, linecolor='black', alpha=0.8, cmap='Blues', vmax=0.9)
            #ax.set_xticklabels(pd.unique(yt), rotation=0, fontsize=16, weight='normal')
            #ax.set_yticklabels(pd.unique(y_names), rotation=0, fontsize=16, weight='normal')
            ax.set_xlabel('Predicted', fontsize=16, weight='normal', color='0.2')
            ax.set_ylabel('Truth', fontsize=16, weight='normal', color='0.2')
            plt.savefig(path + 'hyperparameter exploration/%s.png' % (file_name), dpi=300, bbox_inches='tight')

            if not os.path.isdir(path + 'svg/hyperparameter exploration/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(path + 'svg/hyperparameter exploration/')
            plt.savefig(path + 'svg/hyperparameter exploration/%s.svg' % (file_name), bbox_inches='tight')

            plt.close()
            plt.clf()



            concat = np.concatenate([xs, trans_xt], axis=0)

            __umap = UMAP(metric='euclidean', n_components=2, n_neighbors=30, min_dist=0.5, random_state=0)
            umap = __umap.fit_transform(concat)
            umap



            file_name = 'entropic FGW-type OT_labelcoeff_%s_alpha_%s_eps_%s'%(label_coeff, alpha, epsilon)

            fig, ax = plt.subplots(2, 4, figsize=(18, 8))
            ax[0, 0].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='gray', label='s', alpha=0.5, s=1)
            ax[0, 0].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], color='red', label='t', alpha=0.5, s=1)
            ax[0, 0].set_title('Integration')

            ax[0, 1].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=adata_rna.obs['deformability_score_%s'%pathway],
                             label='s', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=0.8*np.max(adata_rna.obs['deformability_score_%s'%pathway]))
            ax[0, 1].set_title('RNA deformability score')
            ax[0, 2].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df['morpho_avg_speed'], label='t', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'))
            ax[0, 2].set_title('Behavior deformability score')
            #ax[0, 3].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='gray', label='s', alpha=0.5, s=1)
            scatter1 = ax[0, 3].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df['kmeans'].replace(list(np.unique(df['kmeans'])),
                                            [i for i in range(np.unique(df['kmeans']).shape[0])]), cmap=cmc.batlow, label=df['kmeans'], alpha=0.5, s=1)
            handles, lab = scatter1.legend_elements(num=None)
            ax[0, 3].legend(handles=handles, labels=list(np.unique(df['kmeans'])),
                       #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
                       fontsize=6, frameon=False, markerscale=0.6)
            ax[0, 3].set_title('behavior cluster')

            condition_name='cluster'
            scatter1 = ax[1, 0].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
                                            [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), label = df_xs[condition_name], alpha=0.5, s=1,
                                        cmap=plt.cm.get_cmap('Set1'))
            handles, lab = scatter1.legend_elements(num=None)
            ax[1, 0].legend(handles=handles, labels=list(np.unique(df_xs[condition_name])),
                       #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
                       fontsize=6, frameon=False, markerscale=0.6)
            ax[1, 0].set_title('RNA labels')

            condition_name='cluster'
            scatter1 = ax[1, 1].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
                                            [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), label = df_xt[condition_name], alpha=0.5, s=1,
                                        cmap=plt.cm.get_cmap('Set1'))

            handles, lab = scatter1.legend_elements(num=None)
            ax[1, 1].legend(handles=handles, labels=list(np.unique(df_xt[condition_name])),
                       #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
                       fontsize=6, frameon=False, markerscale=0.6)

            ax[1, 1].set_title('Behavior labels')

            condition_name='Zone'
            scatter1 = ax[1, 2].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
                                            [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), label = df_xt[condition_name], alpha=0.5, s=1,
                                        cmap=plt.cm.get_cmap('Set1'))

            handles, lab = scatter1.legend_elements(num=None)
            ax[1, 2].legend(handles=handles, labels=list(np.unique(df_xt[condition_name])),
                       #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
                       fontsize=6, frameon=False, markerscale=0.6)

            ax[1, 2].set_title('Behavior zones')

            condition_name='Zone'
            ax[1, 3].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='gray', label='s', alpha=0.5, s=1)

            # scatter1 = ax[1, 3].scatter(umap[labels == 'target', 0][df_xt[condition_name]=='sLZ'], umap[labels == 'target', 1][df_xt[condition_name]=='sLZ'],
            #                             label = 'sLZ', alpha=0.5, s=1, color='blue')

            scatter1 = ax[1, 3].scatter(umap[labels == 'target', 0][df_xt[condition_name]=='dLZ'], umap[labels == 'target', 1][df_xt[condition_name]=='dLZ'],
                                        label = 'dLZ', alpha=0.5, s=1, color='orange')


            handles, lab = scatter1.legend_elements(num=None)
            ax[1, 3].legend(handles=handles, labels='dLZ',
                       #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
                       fontsize=6, frameon=False, markerscale=0.6)

            ax[1, 3].set_title('dLZ zone')


            plt.savefig(path + 'hyperparameter exploration/%s.png' % (file_name), dpi=300, bbox_inches='tight')

            if not os.path.isdir(path + 'svg/hyperparameter exploration/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(path + 'svg/hyperparameter exploration/')
            plt.savefig(path + 'svg/hyperparameter exploration/%s.svg' % (file_name), bbox_inches='tight')

            plt.clf()
            plt.close()


########################### For fixed parameter ############################

label_coeff = 100 # 100  # Higher, more punishment in getting labels wrong
alpha = 0.1 # 0.01
epsilon = 1.3 # 0.08


pathway = 'Motility associated signatures'
# 'Signaling By Rho GTPases', 'Leukocyte transendothelial migration', 'Microtubule Binding', 'Microtubule', 'Cytoskeleton',
# 'Behav3d_MP', 'Behav3d_MP_filtered', 'Pathway overlapped genes'

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
shared_feature_s = scaler.fit_transform(np.array(adata_rna.obs['deformability_score_%s' % pathway])[:, np.newaxis])
shared_feature_t = scaler.fit_transform(np.array(df['morpho_avg_speed'])[:, np.newaxis])
# print(np.min(shared_feature_s), np.max(shared_feature_s))
# print(np.min(shared_feature_t), np.max(shared_feature_t))

# pathway = 'FDC interaction'
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# shared_feature_s = scaler.fit_transform( np.array( adata_rna.obs['interaction_score_%s'%pathway] )[:, np.newaxis] )
# shared_feature_t = scaler.fit_transform( np.array( df['FDC_contact_persistences'] )[:, np.newaxis] )

M = ot.dist(shared_feature_s, shared_feature_t)  # (Ns, Nt)
label_unmatch = (ys[:, np.newaxis] != yt[np.newaxis, :])  # (Ns, Nt) of True if label unmatched, and False otherwise
cost_correction = label_unmatch * label_coeff
#M = cost_correction
M= np.maximum(M, cost_correction)  # Element-wise maximum operation

####### Solve FGW-type OT #######
#alpha=0.001 # Increasing alpha increases GW-type (structural correspondence) and decrease W-type (shared space)
#P = ot.gromov.fused_gromov_wasserstein(M, C1, C2, a, b, alpha=alpha) # 0<= alpha <=0.1 is good start
#alpha = 0.3
#epsilon = 3

P = ot.gromov.entropic_fused_gromov_wasserstein(M, C1, C2, a, b, alpha=alpha, epsilon=epsilon, max_iter=1e4)  # (Ns, Nt)
P_barycentric = P.T / np.sum(P, axis=0)[:, np.newaxis] # For each row sum column-wise -> Dividie that number for each row (Nt, Ns)
# Every column adds up to 1 (to make it probability)
# Without this we don't get accurate transformation

P_barycentric = np.nan_to_num(P_barycentric, nan=0, posinf=0, neginf=0) # (Nt, Ns)

trans_xt = P_barycentric@xs
##### Transform source labels to predicted target labels #####
P_barycentric = P/np.sum(P, axis=0)[np.newaxis, :] # (Ns, Nt)
P_barycentric = np.nan_to_num(P_barycentric, nan=0, posinf=0, neginf=0) # (Ns, Nt)
labels_u, labels_idx = np.unique(ys, return_inverse=True)  # unique labels, list of label indexes (N, )
n_labels = labels_u.shape[0]
masks = np.eye(n_labels)[labels_idx]  # Hot encoded vector matrix (N, # of unique labels)
trans_yt_prob = (masks.T@P_barycentric).T
trans_yt = np.argmax(trans_yt_prob, axis=1)

y_pred = trans_yt
y_test = yt

accuracy = np.sum(y_test == y_pred) / y_test.size
print(pathway, label_coeff, alpha, epsilon, accuracy)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# norm_cm = cm / np.sum(cm, axis=1)[:, np.newaxis]
#
# import seaborn as sns
#
# file_name = 'confusion matrix labelcoeff_%s_alpha_%s_eps_%s'%(label_coeff, alpha, epsilon)
# font = {'family': 'arial',
#         'weight': 'normal',
#         'size': 8}
# matplotlib.rc('font', **font)
#
# fig, ax = plt.subplots(figsize=(4,4))
# ax = sns.heatmap(norm_cm, annot=True, annot_kws={'size': 16, 'weight': 'normal'}, linewidths=0.5, linecolor='black', alpha=0.8, cmap='Blues', vmax=0.9)
# #ax.set_xticklabels(pd.unique(yt), rotation=0, fontsize=16, weight='normal')
# #ax.set_yticklabels(pd.unique(y_names), rotation=0, fontsize=16, weight='normal')
# ax.set_xlabel('Predicted', fontsize=16, weight='normal', color='0.2')
# ax.set_ylabel('Truth', fontsize=16, weight='normal', color='0.2')
# plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
#
# plt.close()
# plt.clf()



concat = np.concatenate([xs, trans_xt], axis=0)

__umap = UMAP(metric='euclidean', n_components=2, n_neighbors=30, min_dist=0.5, random_state=0)
umap = __umap.fit_transform(concat)
umap_rna = umap[labels == 'source', :]
umap_behavior = umap[labels == 'target', :]

####################### Visualize results #######################

file_name = 'entropic FGW-type OT_labelcoeff_%s_alpha_%s_eps_%s'%(label_coeff, alpha, epsilon)

fig, ax = plt.subplots(2, 4, figsize=(18, 8))
ax[0, 0].scatter(umap_rna[:, 0], umap_rna[:, 1], color='gray', label='s', alpha=0.5, s=1)
ax[0, 0].scatter(umap_behavior[:, 0], umap_behavior[:, 1], color='red', label='t', alpha=0.5, s=1)
ax[0, 0].set_title('Integration')

ax[0, 1].scatter(umap_rna[:, 0], umap_rna[:, 1], c=adata_rna.obs['deformability_score_%s'%pathway],
                 label='s', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=0.8*np.max(adata_rna.obs['deformability_score_%s'%pathway]))
ax[0, 1].set_title('RNA deformability score')
ax[0, 2].scatter(umap_behavior[:, 0], umap_behavior[:, 1], c=df['morpho_avg_speed'], label='t', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'))
ax[0, 2].set_title('Behavior deformability score')

#ax[0, 3].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='gray', label='s', alpha=0.5, s=1)
colors = ['#BAC8DA', '#4F609C', '#8A4F21', '#F06293', '#E9C61D', '#BCBCBC', '#BEDCB0', '#F5A9F5' ]
cmap = ListedColormap(colors[:np.unique(df['beh_kmeans']).shape[0]])
scatter1 = ax[0, 3].scatter(umap_behavior[:, 0], umap_behavior[:, 1], c=df['beh_kmeans'].replace(list(np.unique(df['beh_kmeans'])),
                                [i for i in range(np.unique(df['beh_kmeans']).shape[0])]), cmap=cmap, label=df['beh_kmeans'], alpha=0.5, s=1)
handles, lab = scatter1.legend_elements(num=None)
ax[0, 3].legend(handles=handles, labels=list(np.unique(df['beh_kmeans'])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)
ax[0, 3].set_title('behavior cluster')

condition_name='cluster'
colors = [(0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0), (0.6196078431372549, 0.8549019607843137, 0.8980392156862745, 1.0),
              (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)]
cmap = ListedColormap(colors[:np.unique(df_xs['cluster']).shape[0]])
scatter1 = ax[1, 0].scatter(umap_rna[:, 0], umap_rna[:, 1], c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
                                [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), label = df_xs[condition_name], alpha=0.5, s=1,
                            cmap=cmap)
handles, lab = scatter1.legend_elements(num=None)
ax[1, 0].legend(handles=handles, labels=list(np.unique(df_xs[condition_name])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)
ax[1, 0].set_title('RNA labels')

condition_name='cluster'
scatter1 = ax[1, 1].scatter(umap_behavior[:, 0], umap_behavior[:, 1], c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
                                [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), label = df_xt[condition_name], alpha=0.5, s=1,
                            cmap=cmap)

handles, lab = scatter1.legend_elements(num=None)
ax[1, 1].legend(handles=handles, labels=list(np.unique(df_xt[condition_name])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)

ax[1, 1].set_title('Behavior labels')

condition_name='Zone'
scatter1 = ax[1, 2].scatter(umap_behavior[:, 0], umap_behavior[:, 1], c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
                                [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), label = df_xt[condition_name], alpha=0.5, s=1,
                            cmap=plt.cm.get_cmap('Set1'))

handles, lab = scatter1.legend_elements(num=None)
ax[1, 2].legend(handles=handles, labels=list(np.unique(df_xt[condition_name])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)

ax[1, 2].set_title('Behavior zones')

condition_name='Zone'
ax[1, 3].scatter(umap_rna[:, 0], umap_rna[:, 1], color='gray', label='s', alpha=0.5, s=1)
# scatter1 = ax[1, 3].scatter(umap[labels == 'target', 0][df_xt[condition_name]=='sLZ'], umap[labels == 'target', 1][df_xt[condition_name]=='sLZ'],
#                             label = 'sLZ', alpha=0.5, s=1, color='blue')

scatter1 = ax[1, 3].scatter(umap_behavior[:, 0][df_xt[condition_name]=='dLZ'], umap_behavior[:, 1][df_xt[condition_name]=='dLZ'],
                            label = 'dLZ', alpha=0.5, s=1, color='orange')


handles, lab = scatter1.legend_elements(num=None)
ax[1, 3].legend(handles=handles, labels='dLZ',
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)

ax[1, 3].set_title('dLZ zone')


plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

plt.clf()
plt.close()
# file_name = 'entropic FGW-type OT global structure integrity'
#
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# ax[0].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='lightblue', label='s', alpha=0.7, s=2)
# ax[1].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], color='orange', label='t', alpha=0.7, s=2)
#
# plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
#
# plt.clf()
# plt.close()

####################### Visualize MC vs BC #######################

file_name = 'MC vs BC_labelcoeff_%s_alpha_%s_eps_%s'%(label_coeff, alpha, epsilon)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

#ax[0, 3].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='gray', label='s', alpha=0.5, s=1)
colors = ['#BAC8DA', '#4F609C', '#8A4F21', '#F06293', '#E9C61D', '#BCBCBC', '#BEDCB0', '#F5A9F5' ]
cmap = ListedColormap(colors[:np.unique(df['beh_kmeans']).shape[0]])
scatter1 = ax[0].scatter(umap_behavior[:, 0], umap_behavior[:, 1], c=df['beh_kmeans'].replace(list(np.unique(df['beh_kmeans'])),
                                [i for i in range(np.unique(df['beh_kmeans']).shape[0])]), cmap=cmap, label=df['beh_kmeans'], alpha=0.5, s=1)
handles, lab = scatter1.legend_elements(num=None)
ax[0].legend(handles=handles, labels=list(np.unique(df['beh_kmeans'])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)
ax[0].set_title('behavior cluster')


cmap = cmc.batlow
scatter1 = ax[1].scatter(umap_behavior[:, 0], umap_behavior[:, 1], c=df['kmeans'].replace(list(np.unique(df['kmeans'])),
                                [i for i in range(np.unique(df['kmeans']).shape[0])]), cmap=cmap, label=df['kmeans'], alpha=0.5, s=1)
handles, lab = scatter1.legend_elements(num=None)
ax[1].legend(handles=handles, labels=list(np.unique(df['kmeans'])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)
ax[1].set_title('motility cluster')

plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

plt.clf()
plt.close()


####################### Imputation of behavior cluster on RNA space #######################

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(trans_xt, df['beh_kmeans'])
pred_kmeans = knn.predict(xs)
adata_rna.obs['BC'] = pred_kmeans.astype('str')
adata_rna.obs['BC'] = adata_rna.obs['BC'].astype("category")

####################### Plot Imputed BC #######################

file_name = 'KNN classifier BC imputation_labelcoeff_%s_alpha_%s_eps_%s'%(label_coeff, alpha, epsilon)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='gray', label='s', alpha=0.2, s=1)
scatter1 = ax[0].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df['beh_kmeans'].replace(list(np.unique(df['beh_kmeans'])),
                                [i for i in range(np.unique(df['beh_kmeans']).shape[0])]), cmap=cmc.batlow, label=df['beh_kmeans'], alpha=0.5, s=1)
handles, lab = scatter1.legend_elements(num=None)
ax[0].legend(handles=handles, labels=list(np.unique(df['beh_kmeans'])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)

ax[0].set_title('behavior kmeans')


ax[1].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], color='gray', label='s', alpha=0.2, s=1)
scatter1 = ax[1].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=pred_kmeans, cmap=cmc.batlow, label=pred_kmeans, alpha=0.5, s=1)
handles, lab = scatter1.legend_elements(num=None)
ax[1].legend(handles=handles, labels=list(np.unique(pred_kmeans)),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)

ax[1].set_title('RNA kmeans imputed')


plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

plt.clf()
plt.close()


color_list = ['#BAC8DA', '#4F609C', '#8A4F21', '#F06293', '#E9C61D', '#BCBCBC', '#BEDCB0', '#F5A9F5']
for cluster in np.unique(pred_kmeans):
    file_name = 'BC%s KNN classifier BC imputation_labelcoeff_%s_alpha_%s_eps_%s'%(cluster, label_coeff, alpha, epsilon)

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    fig, ax = plt.subplots(figsize=(2, 2))

    ax.scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='gray', label='s', alpha=0.2, s=0.07)
    scatter1 = ax.scatter(umap[labels == 'source', 0][pred_kmeans==cluster], umap[labels == 'source', 1][pred_kmeans==cluster],
                                label = '%s'%cluster, alpha=1, s=0.07, color=color_list[cluster])

    format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)

    plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

    plt.clf()
    plt.close()


file_name = 'Leiden_Cluster_labelcoeff_%s_alpha_%s_eps_%s'%(label_coeff, alpha, epsilon)
df = pd.DataFrame(umap[labels == 'source'], columns=['UMAP1', 'UMAP2'])
df['leiden'] = adata_rna.obs['leiden'].values
color_list = ('#F06293', '#5B6BBF', '#FFC11C', '#D1917C', '#4DB6AC', '#00BCD4', '#A1877E', '#8F96A8', '#B02E8B', '#EB974D','#BCBCBC',)

draw_umap_space(df, path, file_name=file_name, condition_name='leiden', label_name='leiden',
                colors=color_list, x_name='UMAP1', y_name='UMAP2', dot_size=0.07)


##################### Gene painting on the space #####################

# Bcr (B cell receptor): Binds to antigen of FDC (B-FDC interaction)
# Tnfrsf13c (BAFFR): Binds to BAFF of FDC (B-FDC interaction)
# Lta, Ltb (lymphotoxin alpha, beta) : Binds to Ltbr of FDC (B-FDC interaction)
# Itgb2 (LFA-1 / integrin) : Binds to ICAM-1 of FDC (B-FDC interaction)
# Itga4 (VLA-4) : Binds to VCAM-1 of FDC (B-FDC interaction)

# Ciita (MHC2): Binds to TCR of Tfh (B-Tfh interaction)
# Cd40: Binds to Cd40L of Tfh (B-Tfh interaction)
# Icam1: Binds to LFA-1 of Tfh (B-Tfh interaction)
# Icosl: Binds to Icos of Tfh (B-Tfh interaction)


if not os.path.isdir(path + 'RNA space gene painting/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'RNA space gene painting/')

if not os.path.isdir(path + 'svg/RNA space gene painting/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/RNA space gene painting/')

gene = 'Cd40'
file_name = '%s'%gene
fig, ax = plt.subplots(1, 1, figsize=(4, 4))

#ax.scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], color='gray', label='s', alpha=0.2, s=1)
scatter1 = ax.scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=adata_rna[:, gene].X, cmap='coolwarm',
                      alpha=0.5, s=1, vmax=0.5*np.max(adata_rna[:, gene].X))

plt.savefig(path + 'RNA space gene painting/%s.png' % (file_name), dpi=300, bbox_inches='tight')

plt.savefig(path + 'svg/RNA space gene painting/%s.svg' % (file_name), bbox_inches='tight')

plt.clf()
plt.close()


##################### Module score for GCB cell subtype signatures #####################
folder_name = 'Module score for custom signatures'

for sig_name in list(sig.keys()):
    sc.tl.score_genes(adata_rna, sig[sig_name], score_name='%s_score' % sig_name)

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

    scatter1 = ax.scatter( umap_rna[:, 0], umap_rna[:, 1], c=adata_rna.obs['%s_score'%sig_name], cmap='coolwarm',
                          alpha=0.5, s=1, vmax=0.7 * np.max(adata_rna.obs['%s_score'%sig_name]) )

    if not os.path.isdir(path + '%s/' % folder_name):
        os.makedirs(path + '%s/' % folder_name)

    fig.savefig(path + '%s/%s.png' % (folder_name, sig_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/%s/' % folder_name):
        os.makedirs(path + 'svg/%s/' % folder_name)

    fig.savefig(path + 'svg/%s/%s.svg' % (folder_name, sig_name), bbox_inches='tight')

    plt.clf()
    plt.close()


#################################### Box plot Module score for each BC ####################################
test = 'kruskal-wallis_dunn'

if not os.path.isdir(path + 'Module score for custom signatures/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Module score for custom signatures/')

color_list = ['#BAC8DA', '#4F609C', '#8A4F21', '#F06293', '#E9C61D', '#BCBCBC', '#BEDCB0', '#F5A9F5' ]
for sig_name in list(sig.keys()):
    dataset={}
    for condition in np.unique(adata_rna.obs['BC']):
        data = adata_rna.obs[adata_rna.obs['BC'] == condition]['%s_score' % sig_name]
        dataset[condition] = np.array(data)

    draw_custom_violin_plot(dataset, path+'Module score for custom signatures/', file_name=sig_name, colors=color_list,
                            test=test, pvalue=True, return_sig=True, figsize=(5,3))

    draw_custom_box_plot(dataset, path+ 'Module score for custom signatures/', file_name=sig_name+'_box', colors=color_list,
    strip_plot=False, test=test, pvalue=True, return_sig=True, figsize=(5,3))


##################### Violin plot for single gene expression comparions #####################
folder_name = 'Single gene expression comparions for imputed BCs'

sig_names = ['Holmes_2020 BCR signaling', 'Holmes_2020 CD40 signaling', 'Holmes_2020 NF-kappaB', 'Holmes_2020 NF-kappaB',
             'Holmes_2020 MYC', 'Holmes_2020 DZ', 'Holmes_2020 PreM', 'Holmes_2020 IG', 'Tfh interaction', 'FDC interaction']

sig_names = ['MHC class 2 antigen presentation', 'antigen processing and presentation', 'Holmes_2020 BCR signaling',
             'Holmes_2020 CD40 signaling','Holmes_2020 NF-kappaB', 'Holmes_2020 MYC', 'Holmes_2020 DZ', 'Holmes_2020 PreM',
             'Beguelin_2020 FDC interaction', 'Beguelin_2020 Tfh interaction', 'Beguelin_2020 LZ and anti-apoptosis',
             'Beguelin_2020 DZ hallmark', 'Beguelin_2020 CC recycling', 'Motility associated signatures']

color_list = ['#BAC8DA', '#4F609C', '#8A4F21', '#F06293', '#E9C61D', '#BCBCBC', '#BEDCB0', '#F5A9F5' ]
for sig_name in sig_names:
    genes = sig[sig_name]
    dataset = {}
    for gene in genes:
        try:
            dataset = {}
            for condition in np.unique(adata_rna.obs['BC']):
                data = adata_rna[adata_rna.obs['BC'] == condition][:, gene].X
                dataset[condition] = np.array(data)

            if not os.path.isdir(path + '%s/'%folder_name + '%s/'%sig_name):
                os.makedirs(path + '%s/'%folder_name + '%s/'%sig_name)

            draw_custom_violin_plot(dataset, path + '%s/'%folder_name + '%s/'%sig_name, file_name=gene,
                                    colors=color_list,
                                    test=test, pvalue=True, return_sig=True, figsize=(5, 3))

            draw_custom_box_plot(dataset, path + '%s/'%folder_name + '%s/'%sig_name, file_name=gene + '_box',
                                 colors=color_list,
                                 strip_plot=False, test=test, pvalue=True, return_sig=True, figsize=(5, 3))
        except:
            print('%s: No %s gene'%(sig_name, gene))

##################### DE analysis on imputed BCs #####################

sc.tl.rank_genes_groups(adata_rna, groupby="BC", method="wilcoxon", key_added="dea_BC", reference='rest')
sc.tl.filter_rank_genes_groups(adata_rna, min_in_group_fraction=0.1, max_out_group_fraction=0.5, key="dea_BC",key_added="dea_BC_filtered")

result = adata_rna.uns["dea_BC"]
groups = result["names"].dtype.names
degs = pd.DataFrame(
    {group + '_' + key: result[key][group]
    for group in groups for key in ['names','scores', 'pvals','pvals_adj','logfoldchanges']})

##################### Gene expression heatmap #####################
fig, ax = plt.subplots(figsize=(20, 5), constrained_layout=True)
sc.pl.rank_genes_groups_dotplot(adata_rna, groupby="BC", standard_scale="var", n_genes=10, key="dea_BC", ax=ax, show=False)

fig.savefig(path + 'BC deg.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/BC deg.svg', bbox_inches='tight')
plt.close()
plt.clf()

fig, ax = plt.subplots(figsize=(20, 5), constrained_layout=True)
sc.pl.rank_genes_groups_dotplot(adata_rna,groupby="BC", standard_scale="var", n_genes=10,key="dea_BC_filtered", ax=ax,show=False)
fig.savefig(path + 'BC deg filtered.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/BC deg filtered.svg', bbox_inches='tight')
plt.close()
plt.clf()


##################### UpSet plot for gene overlap counts #####################
from upsetplot import UpSet, from_contents
bc_degs_sig_sets = {}  # Dictionary: {mc: set of significant genes}
for mc in np.unique(adata_rna.obs['BC']):  # MC0 to MC8
    name_col = f"{mc}_names"
    pval_col = f"{mc}_pvals_adj"
    sig_genes = degs[degs[pval_col] < 0.05][name_col].values
    bc_degs_sig_sets[f"BC{mc}"] = sig_genes
    print(mc, sig_genes.shape[0])
bc_degs_sig_sets = from_contents(bc_degs_sig_sets)
fig = plt.figure(figsize=(10, 6))
UpSet(bc_degs_sig_sets, show_counts=True, min_subset_size=10, sort_by='degree', orientation='horizontal').plot(fig=fig)
plt.savefig(path+'bc upset plot.png', dpi=300, bbox_inches='tight')
plt.close()
plt.clf()

##################### Calculate adjusted score (distinctiveness_score) #####################
n_top_genes = 500
bc_degs_sig_sets, degs = get_distinct_DEG_per_cluster(adata_rna, cluster_name='BC', n_top_genes=n_top_genes)

##################### UpSet plot for gene overlap counts #####################
from upsetplot import UpSet, from_contents, plot
bc_degs_sig_sets = from_contents(bc_degs_sig_sets)
fig = plt.figure(figsize=(13, 7))
UpSet(bc_degs_sig_sets, show_counts=True, min_subset_size=10, sort_by='degree', orientation='horizontal', element_size=None).plot(fig=fig)
#plot(UpSet(mc_degs_sig_sets, show_counts=True, min_subset_size=10, sort_by='degree', orientation='horizontal'), fig=fig)

plt.savefig(path+'upset plot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/upset plot.svg', bbox_inches='tight')
plt.close()
plt.clf()

##################### Pathway enrichment analysis for imputed BCs #####################
for cluster in tqdm(np.unique(pred_kmeans)):
    degs_sig = degs[degs['%s_pvals'%cluster] < 0.05]
    degs_up = degs_sig[degs_sig['%s_logfoldchanges'%cluster] > 0]
    degs_down = degs_sig[degs_sig['%s_logfoldchanges'%cluster] < 0]

    gsea_path = path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/'%(cluster, label_coeff, alpha, epsilon)

    if not os.path.isdir(gsea_path):
        os.makedirs(gsea_path)

    run_gsea(degs_up, degs_down, h2m_dict, gsea_path, col_name='%s_names'%cluster)


############# Plot curated pathways enrichment #############
cluster=1
curated_pathways = pd.read_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/curated pathway.xlsx'%(cluster, label_coeff, alpha, epsilon))


font = {'family': 'arial',
        'weight': 'normal', }
matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)

gp.barplot(curated_pathways,
           group='UP_DW',
           # title ="%s"%(library_name),
           ax=ax,
           color=['#6699CC', '#CC6677'])

fig.savefig(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/curated pathway.png'%(cluster, label_coeff, alpha, epsilon), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s'%(cluster, label_coeff, alpha, epsilon)):
    os.makedirs(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s'%(cluster, label_coeff, alpha, epsilon))

fig.savefig(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/curated pathway.svg'%(cluster, label_coeff, alpha, epsilon), bbox_inches='tight')
plt.clf()
plt.close()

############# Plot pathway graph networks #############
for cluster in [0, 1, 2, 3, 7]:
    df_sigs = pd.read_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/significant pathway.xlsx' % (cluster, label_coeff, alpha, epsilon))

    df_sig_graph = df_sigs[df_sigs['Gene_set'] != 'gs_ind_0'].reset_index(drop=True)
    df_sig_graph = df_sig_graph.dropna(subset=['Genes'])
    sig_temp = {
        row['Term']: np.array(genes)
        for _, row in df_sig_graph.iterrows()
        if len(genes := row['Genes'].split(';')) >= 5
    }

    regulation = {
        row['Term']: row['UP_DW']
        for _, row in df_sig_graph.iterrows()
        if len(genes := row['Genes'].split(';')) >= 5
    }


    custom_sig = {}
    for term, genes in sig_temp.items():
        count = 0
        new_genes = []
        for gene in genes:
            if gene in h2m_dict:
                new_genes.append(h2m_dict[gene])
                count = count + 1
            else:
                new_genes.append(gene)
        print(term, ': ', '%s/%s genes converted' % (count, len(genes)))
        custom_sig[term] = new_genes

    draw_graph_network(custom_sig, path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon),
                       file_name='gene overlap', sample_n=5, resolution=1.6, figsize=(10, 10), inter_spacing=2.5, intra_spacing=1.5)

    #### filtered pathway ####
    df_sigs = pd.read_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/filtered pathway.xlsx' % (cluster, label_coeff, alpha, epsilon))

    df_sig_graph = df_sigs
    df_sig_graph = df_sig_graph.dropna(subset=['Genes'])
    sig_temp = {
        row['Term']: np.array(genes)
        for _, row in df_sig_graph.iterrows()
        if len(genes := row['Genes'].split(';')) >= 3
    }

    regulation = {
        row['Term']: row['UP_DW']
        for _, row in df_sig_graph.iterrows()
        if len(genes := row['Genes'].split(';')) >= 3
    }


    custom_sig = {}
    for term, genes in sig_temp.items():
        count = 0
        new_genes = []
        for gene in genes:
            if gene in h2m_dict:
                new_genes.append(h2m_dict[gene])
                count = count + 1
            else:
                new_genes.append(gene)
        print(term, ': ', '%s/%s genes converted' % (count, len(genes)))
        custom_sig[term] = new_genes

    draw_graph_network(custom_sig, path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon),
                       file_name='gene overlap_filtered', sample_n=5, resolution=1.6, figsize=(10, 10), inter_spacing=2.5, intra_spacing=1.5)


    #### curated pathway ####
    df_sigs = pd.read_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/curated pathway.xlsx' % (cluster, label_coeff, alpha, epsilon))

    df_sig_graph = df_sigs
    df_sig_graph = df_sig_graph.dropna(subset=['Genes'])
    sig_temp = {
        row['Term']: np.array(genes)
        for _, row in df_sig_graph.iterrows()
        if len(genes := row['Genes'].split(';')) >= 3
    }

    regulation = {
        row['Term']: row['UP_DW']
        for _, row in df_sig_graph.iterrows()
        if len(genes := row['Genes'].split(';')) >= 3
    }


    custom_sig = {}
    for term, genes in sig_temp.items():
        count = 0
        new_genes = []
        for gene in genes:
            if gene in h2m_dict:
                new_genes.append(h2m_dict[gene])
                count = count + 1
            else:
                new_genes.append(gene)
        print(term, ': ', '%s/%s genes converted' % (count, len(genes)))
        custom_sig[term] = new_genes

    draw_graph_network(custom_sig, path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon),
                       file_name='gene overlap_curated', sample_n=5, resolution=1.4, figsize=(4, 4), inter_spacing=2.5, intra_spacing=1.5)


##################### EZH2 vs WT for each BC #####################
cluster = '1'
adata_rna_bc = adata_rna[(adata_rna.obs['BC']==cluster), :].copy()

#adata_rna_lz = adata_rna[(adata_rna.obs['cluster']=='LZ'), :]
sc.tl.rank_genes_groups(adata_rna_bc, groupby="Type", method="wilcoxon", key_added="dea_Type_%s"%cluster, reference='WT')
sc.tl.filter_rank_genes_groups(adata_rna_bc, min_in_group_fraction=0.1, max_out_group_fraction=0.5, key="dea_Type_%s"%cluster,key_added="dea_Type_%s_filtered"%cluster)


result = adata_rna_bc.uns["dea_Type_%s"%cluster]
groups = result["names"].dtype.names
degs = pd.DataFrame(
    {group + '_' + key: result[key][group]
    for group in groups for key in ['names','scores', 'pvals','pvals_adj','logfoldchanges']})


degs_sig = degs[degs['EZH2_pvals_adj'] < 0.05]


degs_up = degs_sig[degs_sig['EZH2_logfoldchanges'] > 0]
degs_down = degs_sig[degs_sig['EZH2_logfoldchanges'] < 0]

gsea_path = path + 'EZH2 vs WT/BC%s/'%(cluster)
if not os.path.isdir(gsea_path):
    os.makedirs(gsea_path)

degs.to_excel(gsea_path+'BC %s EZH2 DEG.xlsx'%cluster)

df_p = degs.copy()
df_p = draw_gene_rank_plot(df_p, gsea_path, file_name='gene_rank', gene_col='EZH2_names', p_col='EZH2_pvals_adj',
                           score_col='EZH2_scores', figsize=(4, 7), dot_size=7)


run_gsea(degs_up, degs_down, h2m_dict, gsea_path, col_name='EZH2_names')


df_sigs = pd.read_excel(gsea_path + 'significant pathway.xlsx')

df_sig_graph = df_sigs[df_sigs['Gene_set'] != 'gs_ind_0'].reset_index(drop=True)
#df_sig_graph = df_sig_graph[df_sig_graph['UP_DW'] == 'UP'].reset_index(drop=True)

df_sig_graph = df_sig_graph.dropna(subset=['Genes'])
sig_temp = {
    row['Term']: np.array(genes)
    for _, row in df_sig_graph.iterrows()
    if len(genes := row['Genes'].split(';')) >= 3
}

regulation = {
    row['Term']: row['UP_DW']
    for _, row in df_sig_graph.iterrows()
    if len(genes := row['Genes'].split(';')) >= 3
}

adj_p = {
    row['Term']: -np.log10(row['Adjusted P-value'])
    for _, row in df_sig_graph.iterrows()
    if len(genes := row['Genes'].split(';')) >= 3
}


custom_sig = {}
for term, genes in sig_temp.items():
    count = 0
    new_genes = []
    for gene in genes:
        if gene in h2m_dict:
            new_genes.append(h2m_dict[gene])
            count = count + 1
        else:
            new_genes.append(gene)
    print(term, ': ', '%s/%s genes converted' % (count, len(genes)))
    custom_sig[term] = new_genes

draw_graph_network(custom_sig, gsea_path, file_name='gene overlap', sample_n=10, resolution=1.06, regulation=regulation,
                   figsize=(10, 10), inter_spacing=2.5, intra_spacing=1.5)

df_communities = pd.read_excel(gsea_path + 'community_detected_pathways.xlsx')
df_communities['adj_p'] = df_communities['Pathway'].map(adj_p)
import re
# Define a cleaning function using regular expressions
def clean_pathway_name(name):
    # Remove anything in parentheses ()
    name = re.sub(r'\s*\(.*?\)', '', name)
    # Remove R-HSA-xxxxx pattern
    name = re.sub(r'R-HSA-\d+', '', name)
    # Remove WP xxxxx pattern
    name = re.sub(r'WP\s*\d+', '', name)
    return name

# Apply to your dataframe
df_communities['Pathway'] = df_communities['Pathway'].apply(clean_pathway_name)
df_communities = df_communities.drop_duplicates(subset='Pathway', keep='first').reset_index(drop=True)

df_communities_sorted = df_communities.sort_values(['Component_Community', 'adj_p'], ascending=[True, False])
counts = df_communities_sorted['Component_Community'].value_counts(dropna=False)

# Keep only rows whose value appears at least 5 times
df_communities_sorted = df_communities_sorted[df_communities_sorted['Component_Community'].map(counts).fillna(0) >= 5].copy()
df_communities_sorted.to_excel(gsea_path + 'community_detected_pathways_removed.xlsx', index=False)

# Generate the color palette used for Component_Community
unique_comms = df_communities['Component_Community'].unique()
palette = sns.color_palette('tab20', n_colors=len(unique_comms))
color_map = dict(zip(unique_comms, palette))

font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(20, 3))

ax = sns.barplot(
    data=df_communities_sorted,
    x='Pathway',
    y='adj_p',
    hue='Component_Community',
    dodge=False,
    palette=color_map,
    ci=None  # Disable error bars
)

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
    ax.spines[axis].set_color('0.2')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width=1, color='0.2')
# Custom ticks: label only every 3rd, color by Component_Community
ax = plt.gca()
ticks = np.arange(len(df_communities_sorted))
labels = []
label_colors = []

for i, (pathway, comm) in enumerate(zip(df_communities_sorted['Pathway'], df_communities_sorted['Component_Community'])):
    if i % 2 == 0:
        labels.append(pathway)
        label_colors.append(color_map[comm])
    else:
        labels.append('')
        label_colors.append('black')  # won't show, but safe fallback

# Set ticks
ax.set_xticks(ticks)
xtick_labels = ax.set_xticklabels(
    labels,
    fontsize=7,
    rotation=35,
    rotation_mode='anchor',
    ha='right',
    weight='normal'
)

# Apply color to the displayed ticks
for label, color in zip(xtick_labels, label_colors):
    if label.get_text() != '':
        label.set_color(color)

plt.ylabel('Adjusted P-value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Component_Community')
fig.savefig(gsea_path + 'pathway bar plot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(gsea_path + 'svg/'):
    os.makedirs(gsea_path + 'svg/')
fig.savefig(gsea_path + 'svg/pathway bar plot.svg', bbox_inches='tight')


# For each Component_Community, select the row with the highest -log10(adj_p)
best_pathways_log = df_communities_sorted.loc[df_communities_sorted.groupby('Component_Community')['adj_p'].idxmax()]

# Optional: clean and display
best_pathways_log = best_pathways_log[['Component_Community', 'Pathway', 'adj_p']]
print(best_pathways_log)


##################### DE analysis on imputed BCs vs specific BC #####################


for compare_cluster in np.unique(pred_kmeans):
    compare_cluster = 1
    sc.tl.rank_genes_groups(adata_rna, groupby="BC", method="wilcoxon", key_added="dea_BC", reference='%s'%compare_cluster)
    folder_name = 'vs BC%s/'%compare_cluster

    if not os.path.isdir(path + folder_name):
        os.makedirs(path + folder_name)


    file_name = 'Imputed LZ BC DEG_labelcoeff_%s_alpha_%s_eps_%s'%(label_coeff, alpha, epsilon)
    fig, ax = plt.subplots(figsize=(16, 4), constrained_layout=True)
    #sc.pl.umap(adata_rna, color=['deformability_score_%s'%pathway], frameon=False, ax=ax, cmap=plt.cm.get_cmap('coolwarm'))

    #sc.pl.dotplot(adata_rna, var_names=available_gene_list, groupby="Type", standard_scale="var", swap_axes=True, cmap="coolwarm", ax=ax)
    sc.pl.rank_genes_groups_dotplot(adata_rna, groupby="BC", standard_scale="var", n_genes=5, key="dea_BC", ax=ax)

    fig.savefig(path + folder_name + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + folder_name + 'svg/'):
        os.makedirs(path + folder_name + 'svg/')

    fig.savefig(path + folder_name + 'svg/%s.svg' % (file_name), bbox_inches='tight')



    ##################### Pathway enrichment analysis for imputed BCs #####################
    result = adata_rna.uns["dea_BC"]
    groups = result["names"].dtype.names
    degs = pd.DataFrame(
        {group + '_' + key: result[key][group]
        for group in groups for key in ['names','scores', 'pvals','pvals_adj','logfoldchanges']})


    for cluster in tqdm(np.unique(pred_kmeans)):
        #BC0, BC6: DZ
        if cluster == compare_cluster:
            continue

        degs_sig = degs[degs['%s_pvals' % cluster] < 0.05]
        degs_up = degs_sig[degs_sig['%s_logfoldchanges' % cluster] > 0]
        degs_down = degs_sig[degs_sig['%s_logfoldchanges' % cluster] < 0]

        gsea_path = path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon)

        if not os.path.isdir(gsea_path):
            os.makedirs(gsea_path)

        run_gsea(degs_up, degs_down, h2m_dict, gsea_path, col_name='%s_names' % cluster)



# ##################### DE analysis on imputed BCs (without DZ) #####################
# path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\without DZ\\'
#
# adata_rna.obs['BC'] = pred_kmeans.astype('str')
# adata_rna.obs['BC'] = adata_rna.obs['BC'].astype("category")
# adata_rna.obs
#
# adata_rna_lz = adata_rna[(adata_rna.obs['BC']!='0')&(adata_rna.obs['BC']!='6'), :]
# #adata_rna_lz = adata_rna[(adata_rna.obs['cluster']=='LZ'), :]
# sc.tl.rank_genes_groups(adata_rna_lz, groupby="BC", method="wilcoxon", key_added="dea_BC", reference='rest')
#
# file_name = 'Imputed LZ BC DEG_labelcoeff_%s_alpha_%s_eps_%s'%(label_coeff, alpha, epsilon)
# fig, ax = plt.subplots(figsize=(16, 4), constrained_layout=True)
# #sc.pl.umap(adata_rna, color=['deformability_score_%s'%pathway], frameon=False, ax=ax, cmap=plt.cm.get_cmap('coolwarm'))
#
# #sc.pl.dotplot(adata_rna, var_names=available_gene_list, groupby="Type", standard_scale="var", swap_axes=True, cmap="coolwarm", ax=ax)
# sc.pl.rank_genes_groups_dotplot(adata_rna_lz, groupby="BC", standard_scale="var", n_genes=5, key="dea_BC", ax=ax)
#
# fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):
#     os.makedirs(path + 'svg/')
#
# fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
#
#
# ##################### Pathway associated module scores painted on the RNA-behavior space #####################
#
#
#
#
# ##################### Pathway enrichment analysis for imputed BCs (without DZ) #####################
# result = adata_rna_lz.uns["dea_BC"]
# groups = result["names"].dtype.names
# degs = pd.DataFrame(
#     {group + '_' + key: result[key][group]
#     for group in groups for key in ['names','scores', 'pvals','pvals_adj','logfoldchanges']})
#
#
# for cluster in tqdm( np.unique(adata_rna_lz.obs['BC']) ):
#     cluster = int(cluster)
#     #BC0, BC6: DZ
#
#     #BC2: High FDC interaction (dLZ)
#     #BC3: High Tfh interaction
#     #BC7: Fastest speed
#
#     degs_sig = degs[degs['%s_pvals'%cluster] < 0.05]
#     degs_up = degs_sig[degs_sig['%s_logfoldchanges'%cluster] > 0]
#     degs_down = degs_sig[degs_sig['%s_logfoldchanges'%cluster] < 0]
#
#
#     if not os.path.isdir(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/'%(cluster, label_coeff, alpha, epsilon)):
#         os.makedirs(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/'%(cluster, label_coeff, alpha, epsilon))
#
#     ############################### Using online Enrichr method ###############################
#     df_sigs = pd.DataFrame()
#     for library_name in ['GO_Biological_Process_2023', 'GO_Cellular_Component_2023', 'GO_Molecular_Function_2023',
#                          'KEGG_2019_Mouse', 'Reactome_2022', 'WikiPathways_2024_Mouse', 'CORUM']:
#         file_name= 'GSEA_%s'%(library_name)
#         while True:
#             try:
#                 enr_up = gp.enrichr(degs_up['%s_names'%cluster].astype(str),gene_sets=library_name, outdir=None)
#                 enr_down = gp.enrichr(degs_down['%s_names'%cluster].astype(str), gene_sets=library_name, outdir=None)
#                 break
#             except Exception as e:
#                 print(f"Error: {e}. Retrying...")
#
#         enr_up.res2d['UP_DW'] = "UP"
#         enr_down.res2d['UP_DW'] = "DOWN"
#
#         enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
#         df_combined = enr_res.sort_values('Combined Score', ascending=False)
#         df_combined['Overlap'] = df_combined['Overlap'].astype(str)
#         df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]
#
#         df_sig.to_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/pathway list %s.xlsx' % (cluster, label_coeff, alpha, epsilon, file_name), index=False)
#         df_combined.to_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/entire pathway list %s.xlsx' % (cluster, label_coeff, alpha, epsilon, file_name), index=False)
#
#         fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
#
#         gp.barplot(enr_res, figsize=(6, 6),
#                    group='UP_DW',
#                    title="%s" % (library_name),
#                    ax=ax,
#                    color=['b', 'r'])
#
#         fig.savefig(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/%s.png' % (cluster, label_coeff, alpha, epsilon, file_name), dpi=300, bbox_inches='tight')
#
#         if not os.path.isdir(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/'%(cluster, label_coeff, alpha, epsilon)):
#             os.makedirs(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/'%(cluster, label_coeff, alpha, epsilon))
#
#         fig.savefig(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/%s.svg' % (cluster, label_coeff, alpha, epsilon, file_name), bbox_inches='tight')
#
#         df_sigs = pd.concat([df_sigs, df_sig], axis=0, ignore_index=True)
#
#     ############################### Using offline downloaded gmt files ###############################
#
#     ############################### MsigDB: C2-CGP ###############################
#     custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\without DZ\signatures\\'
#     sig_temp = gp.read_gmt(path=custom_sig_path + 'c2.cgp.v2024.1.Hs.symbols.gmt')
#     file_name = 'GSEA_MsigDB_CGP'
#
#     custom_sig = {}
#     for term, genes in sig_temp.items():
#         count = 0
#         new_genes = []
#         for gene in genes:
#             if gene in h2m_dict:
#                 new_genes.append(h2m_dict[gene])
#                 count = count + 1
#             else:
#                 new_genes.append(gene)
#         print(term, ': ', '%s/%s genes converted' % (count, len(genes)))
#         custom_sig[term] = new_genes
#
#     enr_up = gp.enrich(degs_up['%s_names' % cluster].astype(str), gene_sets=custom_sig, outdir=None)
#     enr_down = gp.enrich(degs_down['%s_names' % cluster].astype(str), gene_sets=custom_sig, outdir=None)
#
#     enr_up.res2d['UP_DW'] = "UP"
#     enr_down.res2d['UP_DW'] = "DOWN"
#
#     enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
#     df_combined = enr_res.sort_values('Combined Score', ascending=False)
#     df_combined['Overlap'] = df_combined['Overlap'].astype(str)
#     df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]
#
#     df_sig.to_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/pathway list %s.xlsx' % (cluster, label_coeff, alpha, epsilon, file_name), index=False)
#     df_combined.to_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/entire pathway list %s.xlsx' % (cluster, label_coeff, alpha, epsilon, file_name), index=False)
#
#     fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
#
#     gp.barplot(enr_res, figsize=(6, 6),
#                group='UP_DW',
#                title="%s" % (file_name),
#                ax=ax,
#                color=['b', 'r'])
#
#     fig.savefig(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/%s.png' % (cluster, label_coeff, alpha, epsilon, file_name),
#         dpi=300, bbox_inches='tight')
#
#     if not os.path.isdir(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon)):
#         os.makedirs(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon))
#
#     fig.savefig(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/%s.svg' % (cluster, label_coeff, alpha, epsilon, file_name),
#         bbox_inches='tight')
#
#
#     df_sigs = pd.concat([df_sigs, df_sig], axis=0, ignore_index=True)
#
#     ############################### MsigDB: C7-ImmuneSigDB ###############################
#     sig_temp = gp.read_gmt(path=custom_sig_path + 'c7.all.v2024.1.Hs.symbols.gmt')
#     file_name = 'GSEA_MsigDB_immunesigdb'
#
#     custom_sig = {}
#     for term, genes in sig_temp.items():
#         count = 0
#         new_genes = []
#         for gene in genes:
#             if gene in h2m_dict:
#                 new_genes.append(h2m_dict[gene])
#                 count = count + 1
#             else:
#                 new_genes.append(gene)
#         print(term, ': ', '%s/%s genes converted' % (count, len(genes)))
#         custom_sig[term] = new_genes
#
#     enr_up = gp.enrich(degs_up['%s_names' % cluster].astype(str), gene_sets=custom_sig, outdir=None)
#     enr_down = gp.enrich(degs_down['%s_names' % cluster].astype(str), gene_sets=custom_sig, outdir=None)
#
#     enr_up.res2d['UP_DW'] = "UP"
#     enr_down.res2d['UP_DW'] = "DOWN"
#
#     enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
#     df_combined = enr_res.sort_values('Combined Score', ascending=False)
#     df_combined['Overlap'] = df_combined['Overlap'].astype(str)
#     df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]
#
#     df_sig.to_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/pathway list %s.xlsx' % (cluster, label_coeff, alpha, epsilon, file_name), index=False)
#     df_combined.to_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/entire pathway list %s.xlsx' % (cluster, label_coeff, alpha, epsilon, file_name), index=False)
#
#     fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
#
#     gp.barplot(enr_res, figsize=(6, 6),
#                group='UP_DW',
#                title="%s" % (file_name),
#                ax=ax,
#                color=['b', 'r'])
#
#     fig.savefig(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/%s.png' % (cluster, label_coeff, alpha, epsilon, file_name),
#         dpi=300, bbox_inches='tight')
#
#     if not os.path.isdir(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon)):
#         os.makedirs(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon))
#
#     fig.savefig(path + 'svg/BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/%s.svg' % (cluster, label_coeff, alpha, epsilon, file_name),
#         bbox_inches='tight')
#
#
#     df_sigs = pd.concat([df_sigs, df_sig], axis=0, ignore_index=True)
#
#
#
#     df_sigs.to_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/significant pathway.xlsx' % (cluster, label_coeff, alpha, epsilon), index=False)
#
#     key_strings = 'migration|chemotaxis|motility|cytoskeleton|microtubu|b cell|atp|antigen|synapse|NF-kappa|myc|bcr|b cell receptor|cd40'
#     filtered_df = df_sigs[df_sigs['Term'].str.contains(key_strings, case=False, na=False)]
#     filtered_df.to_excel(path + 'BC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/filtered pathway.xlsx' % (cluster, label_coeff, alpha, epsilon), index=False)






############# Imputing deformability score and validate correlation btw behavior deformability and imputed deformability #############
df_ref = pd.DataFrame(trans_xt)
df_ref['morpho_avg_speed'] = df['morpho_avg_speed']

df_impute = pd.DataFrame(xs)
df_impute['morpho_avg_speed'] = np.nan

combined_df = pd.concat([df_ref, df_impute], ignore_index=True)

from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=8, weights="uniform")
imputed = knn_imputer.fit_transform(combined_df)

imputed_feature = imputed[df_ref.shape[0]:, -1]


file_name = 'deformability vs imputed deformability'

linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2

from matplotlib.ticker import MaxNLocator
font = {'family': 'arial',
            'weight': 'normal',
            'size': 10}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = linewidth

x = adata_rna.obs['deformability_score_%s'%pathway][adata_rna.obs['cluster']=='LZ']
y = imputed_feature[adata_rna.obs['cluster']=='LZ']

# x = adata_rna.obs['deformability_score_%s'%pathway][(adata_rna.obs['Type']=='WT')&(adata_rna.obs['cluster']=='LZ')]
# y = imputed_feature[(adata_rna.obs['Type']=='WT')&(adata_rna.obs['cluster']=='LZ')]

# x = adata_rna.obs['deformability_score_%s'%pathway]
# y = imputed_feature

fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharey='row')
sns.regplot(x=x, y=y, scatter_kws={"color":"black", "alpha":0.3, 's':4}, line_kws={"color":"black"}, ax=ax)
r, p = scipy.stats.spearmanr(x, y)
#print(r)
#r, p = scipy.stats.pearsonr(x, y)
plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
         fontsize=14, fontdict={'weight': 'normal'}, color="black")
plt.text(0.1, 0.88, "p = " + str('{:.2e}'.format(p)), ha='left', va='top', transform=ax.transAxes,
         fontsize=12, fontdict={'weight': 'normal'}, color="black")

ax.spines["left"].set_visible(True)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['left'].set_color('0.2')

ax.spines["bottom"].set_visible(True)
ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['bottom'].set_color('0.2')

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(width=linewidth, color='0.2', labelsize=12)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.set_xlabel('deformability score', fontsize=12, weight='normal', color='0.2', labelpad=5)
ax.set_ylabel('imputed score', fontsize=12, weight='normal', color='0.2', labelpad=5)

plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

plt.clf()
plt.close()

############# For fusing interaction scores #############
# file_name = 'entropic FGW-type OT'
#
# fig, ax = plt.subplots(2, 4, figsize=(18, 8))
# ax[0, 0].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='gray', label='s', alpha=0.5, s=1)
# ax[0, 0].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], color='red', label='t', alpha=0.3, s=1)
# ax[0, 0].set_title('Integration')
#
# ax[0, 1].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=adata_rna.obs['interaction_score_%s'%pathway],
#                  label='s', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=0.7*np.max(adata_rna.obs['interaction_score_%s'%pathway]))
# ax[0, 1].set_title('RNA FDC interaction score')
# ax[0, 2].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df['FDC_contact_persistences'], label='t', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'))
# ax[0, 2].set_title('Behavior interaction interaction score')
# #ax[0, 3].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='gray', label='s', alpha=0.5, s=1)
# scatter1 = ax[0, 3].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df['kmeans'].replace(list(np.unique(df['kmeans'])),
#                                 [i for i in range(np.unique(df['kmeans']).shape[0])]), cmap=cmc.batlow, label=df['kmeans'], alpha=0.5, s=1)
# handles, lab = scatter1.legend_elements(num=None)
# ax[0, 3].legend(handles=handles, labels=list(np.unique(df['kmeans'])),
#            #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
#            fontsize=6, frameon=False, markerscale=0.6)
# ax[0, 3].set_title('behavior cluster')
#
# condition_name='cluster'
# scatter1 = ax[1, 0].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
#                                 [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), label = df_xs[condition_name], alpha=0.5, s=1,
#                             cmap=plt.cm.get_cmap('Set1'))
# handles, lab = scatter1.legend_elements(num=None)
# ax[1, 0].legend(handles=handles, labels=list(np.unique(df_xs[condition_name])),
#            #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
#            fontsize=6, frameon=False, markerscale=0.6)
# ax[1, 0].set_title('RNA labels')
#
# condition_name='cluster'
# scatter1 = ax[1, 1].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
#                                 [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), label = df_xt[condition_name], alpha=0.5, s=1,
#                             cmap=plt.cm.get_cmap('Set1'))
#
# handles, lab = scatter1.legend_elements(num=None)
# ax[1, 1].legend(handles=handles, labels=list(np.unique(df_xt[condition_name])),
#            #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
#            fontsize=6, frameon=False, markerscale=0.6)
#
# ax[1, 1].set_title('Behavior labels')
#
# condition_name='Zone'
# scatter1 = ax[1, 2].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
#                                 [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), label = df_xt[condition_name], alpha=0.5, s=1,
#                             cmap=plt.cm.get_cmap('Set1'))
#
# handles, lab = scatter1.legend_elements(num=None)
# ax[1, 2].legend(handles=handles, labels=list(np.unique(df_xt[condition_name])),
#            #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
#            fontsize=6, frameon=False, markerscale=0.6)
#
# ax[1, 2].set_title('Behavior zones')
#
# condition_name='Zone'
# ax[1, 3].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], color='gray', label='s', alpha=0.3, s=1)
#
# # scatter1 = ax[1, 3].scatter(umap[labels == 'target', 0][df_xt[condition_name]=='sLZ'], umap[labels == 'target', 1][df_xt[condition_name]=='sLZ'],
# #                             label = 'sLZ', alpha=0.5, s=1, color='blue')
#
# scatter1 = ax[1, 3].scatter(umap[labels == 'target', 0][df_xt[condition_name]=='dLZ'], umap[labels == 'target', 1][df_xt[condition_name]=='dLZ'],
#                             label = 'dLZ', alpha=0.5, s=1, color='orange')
#
# handles, lab = scatter1.legend_elements(num=None)
# ax[1, 3].legend(handles=handles, labels='dLZ',
#            #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
#            fontsize=6, frameon=False, markerscale=0.6)
#
# ax[1, 3].set_title('dLZ zone')
#
#
# plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
#
# if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#     os.makedirs(path + 'svg/')
# plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
#
# plt.clf()
# plt.close()


####################### Imputation of interaction scores on RNA space #######################
feature = 'FDC_contact_persistences' #'FDC_contact_persistences', 'FDC_avg_overlap', 'T_contact_persistences', 'T_avg_overlap'
pathway = 'FDC interaction' #'MHC class 2 antigen presentation', 'Tfh interaction', 'antigen processing and presentation', 'FDC interaction'

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

for feature in feature_list:
    for pathway in ['FDC interaction']:
        df_ref = pd.DataFrame(trans_xt)
        df_ref[feature] = df[feature]

        df_impute = pd.DataFrame(xs)
        df_impute[feature] = np.nan

        combined_df = pd.concat([df_ref, df_impute], ignore_index=True)

        from sklearn.impute import KNNImputer
        knn_imputer = KNNImputer(n_neighbors=8, weights="uniform")
        imputed = knn_imputer.fit_transform(combined_df)

        imputed_feature = imputed[df_ref.shape[0]:, -1]
        #
        #
        #
        # file_name = 'interaction score comparison_%s_%s'%(feature, pathway)
        # fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        #
        # ax[0].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df[feature], label='t',
        #               alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1 * np.max(df[feature]))
        # ax[0].set_title('Behavior interaction score')
        #
        # ax[1].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=adata_rna.obs['interaction_score_%s'%pathway],
        #               label='s', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1*np.max(adata_rna.obs['interaction_score_%s'%pathway]))
        # ax[1].set_title('RNA interaction score')
        #
        #
        # ax[2].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=imputed_feature,
        #               label='s', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1*np.max(imputed_feature))
        # ax[2].set_title('RNA interaction imputed')
        #
        # plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
        #
        # if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        #     os.makedirs(path + 'svg/')
        # plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
        #
        # plt.clf()
        # plt.close()


        file_name = 'Corr between _%s_%s'%(feature, pathway)

        linewidth = 1.5
        fontsize = 16
        width=6
        ratio=5
        space=0.2

        from matplotlib.ticker import MaxNLocator
        font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 10}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = linewidth

        # x = adata_rna.obs['interaction_score_%s'%pathway]
        # y = imputed_feature
        x = adata_rna[(adata_rna.obs['cluster'] == 'LZ'), :].obs['interaction_score_%s'%pathway]
        y = imputed_feature[adata_rna.obs['cluster'] == 'LZ']

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharey='row')
        sns.regplot(x=x, y=y, scatter_kws={"color":"black", "alpha":0.3, 's':4}, line_kws={"color":"black"}, ax=ax)
        r, p = scipy.stats.spearmanr(x, y)
        #print(r)
        #r, p = scipy.stats.pearsonr(x, y)
        plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                 fontsize=14, fontdict={'weight': 'normal'}, color="black")
        plt.text(0.1, 0.88, "p = " + str('{:.2e}'.format(p)), ha='left', va='top', transform=ax.transAxes,
                 fontsize=12, fontdict={'weight': 'normal'}, color="black")

        ax.spines["left"].set_visible(True)
        ax.spines['left'].set_linewidth(linewidth)
        ax.spines['left'].set_color('0.2')

        ax.spines["bottom"].set_visible(True)
        ax.spines['bottom'].set_linewidth(linewidth)
        ax.spines['bottom'].set_color('0.2')

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=linewidth, color='0.2', labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('interaction_score_%s'%pathway, fontsize=12, weight='normal', color='0.2', labelpad=5)
        ax.set_ylabel('imputed score', fontsize=12, weight='normal', color='0.2', labelpad=5)

        plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

        if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/')
        plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

        plt.clf()
        plt.close()

# ####################### Imputation of interaction scores on RNA space #######################
# feature = 'FDC_contact_persistences' #'FDC_contact_persistences', 'FDC_avg_overlap', 'T_contact_persistences', 'T_avg_overlap'
# pathway = 'FDC interaction' #'MHC class 2 antigen presentation', 'Tfh interaction', 'antigen processing and presentation', 'FDC interaction'
#
# for n_neighbors in range(1, 20, 1):
#     df_ref = pd.DataFrame(trans_xt)
#     df_ref[feature] = df[feature]
#
#     df_impute = pd.DataFrame(xs)
#     df_impute[feature] = np.nan
#
#     combined_df = pd.concat([df_ref, df_impute], ignore_index=True)
#
#     from sklearn.impute import KNNImputer
#     knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
#     imputed = knn_imputer.fit_transform(combined_df)
#
#     imputed_feature = imputed[df_ref.shape[0]:, -1]
#
#
#
#     file_name = 'interaction score comparison_%s_%s_%s'%(feature, pathway, n_neighbors)
#     fig, ax = plt.subplots(1, 3, figsize=(12, 4))
#
#     ax[0].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df[feature], label='t',
#                   alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1 * np.max(df[feature]))
#     ax[0].set_title('Behavior interaction score')
#
#     ax[1].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=adata_rna.obs['interaction_score_%s'%pathway],
#                   label='s', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1*np.max(adata_rna.obs['interaction_score_%s'%pathway]))
#     ax[1].set_title('RNA interaction score')
#
#
#     ax[2].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=imputed_feature,
#                   label='s', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1*np.max(imputed_feature))
#     ax[2].set_title('RNA interaction imputed')
#
#     plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
#
#     if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'svg/')
#     plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
#
#     plt.clf()
#     plt.close()
#
#
#     file_name = 'RNA interaction score vs imputed score_%s_%s_%s'%(feature, pathway, n_neighbors)
#
#     linewidth = 1.5
#     fontsize = 16
#     width=6
#     ratio=5
#     space=0.2
#
#     from matplotlib.ticker import MaxNLocator
#     font = {'family': 'arial',
#                 'weight': 'normal',
#                 'size': 10}
#     matplotlib.rc('font', **font)
#     matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
#     matplotlib.rcParams['lines.linewidth'] = linewidth
#
#     x = adata_rna.obs['interaction_score_%s'%pathway]
#     y = imputed_feature
#     #x = adata_rna[(adata_rna.obs['cluster'] == 'DZ-LZ'), :].obs['interaction_score_%s'%pathway]
#     #y = imputed_feature[adata_rna.obs['cluster'] == 'DZ-LZ']
#
#     fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharey='row')
#     sns.regplot(x=x, y=y, scatter_kws={"color":"black", "alpha":0.3, 's':4}, line_kws={"color":"black"}, ax=ax)
#     r, p = scipy.stats.spearmanr(x, y)
#     #print(r)
#     #r, p = scipy.stats.pearsonr(x, y)
#     plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
#              fontsize=14, fontdict={'weight': 'normal'}, color="black")
#     plt.text(0.1, 0.88, "p = " + str('{:.2e}'.format(p)), ha='left', va='top', transform=ax.transAxes,
#              fontsize=12, fontdict={'weight': 'normal'}, color="black")
#
#     ax.spines["left"].set_visible(True)
#     ax.spines['left'].set_linewidth(linewidth)
#     ax.spines['left'].set_color('0.2')
#
#     ax.spines["bottom"].set_visible(True)
#     ax.spines['bottom'].set_linewidth(linewidth)
#     ax.spines['bottom'].set_color('0.2')
#
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.tick_params(width=linewidth, color='0.2', labelsize=12)
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#
#     ax.set_xlabel('interaction_score_%s'%pathway, fontsize=12, weight='normal', color='0.2', labelpad=5)
#     ax.set_ylabel('imputed score', fontsize=12, weight='normal', color='0.2', labelpad=5)
#
#     plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
#
#     if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'svg/')
#     plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
#
#     plt.clf()
#     plt.close()


####################### Imputation of interaction scores on RNA space and compare with single gene asssociate with FDC interaction #######################
feature = 'FDC_contact_persistences' #'FDC_contact_persistences', 'FDC_avg_overlap', 'T_contact_persistences', 'T_avg_overlap'
pathway = 'FDC interaction' #'MHC class 2 antigen presentation', 'Tfh interaction', 'antigen processing and presentation', 'FDC interaction'

df_ref = pd.DataFrame(trans_xt)
df_ref[feature] = df[feature]

df_impute = pd.DataFrame(xs)
df_impute[feature] = np.nan

combined_df = pd.concat([df_ref, df_impute], ignore_index=True)

from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=8, weights="uniform")
imputed = knn_imputer.fit_transform(combined_df)

imputed_feature = imputed[df_ref.shape[0]:, -1]


for gene in sig['FDC interaction']: # 'FDC interaction',

    try:
        gene_expression = adata_rna[:, gene].X.flatten()
    except:
        continue

    expressed = (gene_expression != np.min(gene_expression))


    file_name = '%s comparison'%(gene)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df[feature], label='t',
                  alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1 * np.max(df[feature]))
    ax[0].set_title('Behavior interaction score')

    ax[1].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=gene_expression,
                  label='s', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1*np.max(gene_expression))
    ax[1].set_title('RNA gene score')

    ax[2].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=imputed_feature,
                  label='s', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1*np.max(imputed_feature))
    ax[2].set_title('RNA interaction imputed')

    plt.savefig(path + 'FDC genes/%s.png' % (file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/FDC genes/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/FDC genes/')
    plt.savefig(path + 'svg/FDC genes/%s.svg' % (file_name), bbox_inches='tight')

    plt.clf()
    plt.close()


    file_name = '%s'%(gene)

    linewidth = 1.5
    fontsize = 16
    width=6
    ratio=5
    space=0.2

    from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    x = gene_expression[expressed]
    y = imputed_feature[expressed]
    #x = adata_rna[(adata_rna.obs['cluster'] == 'DZ-LZ'), :].obs['interaction_score_%s'%pathway]
    #y = imputed_feature[adata_rna.obs['cluster'] == 'DZ-LZ']

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharey='row')
    sns.regplot(x=x, y=y, scatter_kws={"color":"black", "alpha":0.3, 's':4}, line_kws={"color":"black"}, ax=ax)
    r, p = scipy.stats.spearmanr(x, y)
    #print(r)
    #r, p = scipy.stats.pearsonr(x, y)
    plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
             fontsize=14, fontdict={'weight': 'normal'}, color="black")
    plt.text(0.1, 0.88, "p = " + str('{:.2e}'.format(p)), ha='left', va='top', transform=ax.transAxes,
             fontsize=12, fontdict={'weight': 'normal'}, color="black")

    ax.spines["left"].set_visible(True)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['left'].set_color('0.2')

    ax.spines["bottom"].set_visible(True)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['bottom'].set_color('0.2')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=linewidth, color='0.2', labelsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel('%s'%gene, fontsize=12, weight='normal', color='0.2', labelpad=5)
    ax.set_ylabel('imputed score', fontsize=12, weight='normal', color='0.2', labelpad=5)

    plt.savefig(path + 'FDC genes/%s.png' % (file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/FDC genes/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/FDC genes/')
    plt.savefig(path + 'svg/FDC genes/%s.svg' % (file_name), bbox_inches='tight')

    plt.clf()
    plt.close()



# ####################### Imputation of deformability scores on RNA space #######################
# feature = 'morpho_avg_speed' #'FDC_contact_persistences', 'FDC_avg_overlap', 'T_contact_persistences', 'T_avg_overlap'
# pathway = 'FDC interaction' # 'Signaling By Rho GTPases', 'Leukocyte transendothelial migration', 'Microtubule Binding',
# # 'Microtubule', 'Cytoskeleton','Behav3d_MP', 'Behav3d_MP_filtered', 'Pathway overlapped genes'
#
# for pathway in ['Signaling By Rho GTPases', 'Leukocyte transendothelial migration', 'Microtubule Binding', 'Microtubule', 'Cytoskeleton',
#                 'Behav3d_MP', 'Behav3d_MP_filtered', 'Pathway overlapped genes']:
#     df_ref = pd.DataFrame(trans_xt)
#     df_ref[feature] = df[feature]
#
#     df_impute = pd.DataFrame(xs)
#     df_impute[feature] = np.nan
#
#     combined_df = pd.concat([df_ref, df_impute], ignore_index=True)
#
#     from sklearn.impute import KNNImputer
#     knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")
#     imputed = knn_imputer.fit_transform(combined_df)
#
#     imputed_feature = imputed[df_ref.shape[0]:, -1]
#
#
#
#     file_name = 'deformability score comparison__%s'%(pathway)
#     fig, ax = plt.subplots(1, 3, figsize=(12, 4))
#
#     ax[0].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df[feature], label='t',
#                   alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1 * np.max(df[feature]))
#     ax[0].set_title('Behavior deformability score')
#
#     ax[1].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=adata_rna.obs['deformability_score_%s'%pathway],
#                   label='s', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1*np.max(adata_rna.obs['deformability_score_%s'%pathway]))
#     ax[1].set_title('RNA deformability score')
#
#
#     ax[2].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], c=imputed_feature,
#                   label='s', alpha=0.5, s=1, cmap=plt.cm.get_cmap('coolwarm'), vmax=1*np.max(imputed_feature))
#     ax[2].set_title('RNA deformability imputed')
#
#     plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
#
#     if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'svg/')
#     plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
#
#     plt.clf()
#     plt.close()
#
#
#     file_name = 'RNA deformability score vs imputed score_%s'%(pathway)
#
#     linewidth = 1.5
#     fontsize = 16
#     width=6
#     ratio=5
#     space=0.2
#
#     from matplotlib.ticker import MaxNLocator
#     font = {'family': 'arial',
#                 'weight': 'normal',
#                 'size': 10}
#     matplotlib.rc('font', **font)
#     matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
#     matplotlib.rcParams['lines.linewidth'] = linewidth
#
#     x = adata_rna.obs['deformability_score_%s'%pathway]
#     y = imputed_feature
#     #x = adata_rna[(adata_rna.obs['cluster'] == 'DZ-LZ'), :].obs['interaction_score_%s'%pathway]
#     #y = imputed_feature[adata_rna.obs['cluster'] == 'DZ-LZ']
#
#     fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharey='row')
#     sns.regplot(x=x, y=y, scatter_kws={"color":"black", "alpha":0.3, 's':4}, line_kws={"color":"black"}, ax=ax)
#     r, p = scipy.stats.spearmanr(x, y)
#     #print(r)
#     #r, p = scipy.stats.pearsonr(x, y)
#     plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
#              fontsize=14, fontdict={'weight': 'normal'}, color="black")
#     plt.text(0.1, 0.88, "p = " + str('{:.2e}'.format(p)), ha='left', va='top', transform=ax.transAxes,
#              fontsize=12, fontdict={'weight': 'normal'}, color="black")
#
#     ax.spines["left"].set_visible(True)
#     ax.spines['left'].set_linewidth(linewidth)
#     ax.spines['left'].set_color('0.2')
#
#     ax.spines["bottom"].set_visible(True)
#     ax.spines['bottom'].set_linewidth(linewidth)
#     ax.spines['bottom'].set_color('0.2')
#
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.tick_params(width=linewidth, color='0.2', labelsize=12)
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#
#     ax.set_xlabel('deformability_score_%s'%pathway, fontsize=12, weight='normal', color='0.2', labelpad=5)
#     ax.set_ylabel('imputed score', fontsize=12, weight='normal', color='0.2', labelpad=5)
#
#     plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
#
#     if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
#         os.makedirs(path + 'svg/')
#     plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
#
#     plt.clf()
#     plt.close()



####################### Imputation of dLZ on RNA space #######################

x_train = trans_xt[df['Zone']=='LZ']
y_train = df['Zone1'][df['Zone']=='LZ']


n=2
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(trans_xt, df['Zone1'])
y_pred = knn.predict(xs)

print(n, np.sum(y_pred == 'dLZ'))

file_name = 'KNN classifier dLZ imputation for entropic FGW'
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], color='gray', label='s', alpha=0.2, s=1)
scatter1 = ax[0].scatter(umap[labels == 'target', 0][df['Zone1']=='dLZ'], umap[labels == 'target', 1][df['Zone1']=='dLZ'],
                         color='orange', label='dLZ', alpha=0.5, s=1)
handles, lab = scatter1.legend_elements(num=None)
ax[0].legend(handles=handles, labels='dLZ',
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)

ax[0].set_title('behavior dLZ')


ax[1].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='gray', label='s', alpha=0.2, s=1)
scatter1 = ax[1].scatter(umap[labels == 'source', 0][y_pred=='dLZ'], umap[labels == 'source', 1][y_pred=='dLZ'], color='orange',
                         label='dLZ', alpha=0.5, s=1)
handles, lab = scatter1.legend_elements(num=None)
ax[1].legend(handles=handles, labels='dLZ',
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)

ax[1].set_title('RNA dLZ imputed')


plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

plt.clf()
plt.close()





adata_rna.obs['Zone'] = y_pred.astype('str')
adata_rna.obs['Zone'] = adata_rna.obs['Zone'].astype("category")
adata_rna.obs

adata_rna_lz = adata_rna[(adata_rna.obs['cluster']=='LZ'), :]
adata_rna_lz = adata_rna

sc.tl.rank_genes_groups(adata_rna_lz, groupby="Zone", method="wilcoxon", key_added="dea_zone", reference='sLZ')

file_name = 'Imputed zone DEG'
fig, ax = plt.subplots(figsize=(16, 4), constrained_layout=True)
#sc.pl.umap(adata_rna, color=['deformability_score_%s'%pathway], frameon=False, ax=ax, cmap=plt.cm.get_cmap('coolwarm'))

#sc.pl.dotplot(adata_rna, var_names=available_gene_list, groupby="Type", standard_scale="var", swap_axes=True, cmap="coolwarm", ax=ax)
sc.pl.rank_genes_groups_dotplot(adata_rna_lz, groupby="Zone", standard_scale="var", n_genes=5, key="dea_zone", ax=ax)

fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')

fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')




result = adata_rna_lz.uns["dea_zone"]
groups = result["names"].dtype.names
degs = pd.DataFrame(
    {group + '_' + key: result[key][group]
    for group in groups for key in ['names','scores', 'pvals','pvals_adj','logfoldchanges']})

cluster='dLZ'
degs_sig = degs[degs['%s_pvals'%cluster] < 0.05]
degs_up = degs_sig[degs_sig['%s_logfoldchanges'%cluster] > 0]
degs_down = degs_sig[degs_sig['%s_logfoldchanges'%cluster] < 0]


if not os.path.isdir(path + 'Zone_%s GSEA/'%cluster):
    os.makedirs(path + 'Zone_%s GSEA/'%cluster)

for library_name in ['GO_Biological_Process_2023', 'GO_Cellular_Component_2023', 'GO_Molecular_Function_2023',
                     'KEGG_2019_Mouse', 'Reactome_2022', 'WikiPathways_2024_Mouse', 'CORUM']:
    file_name= 'GSEA_%s'%(library_name)
    enr_up = gp.enrichr(degs_up['%s_names'%cluster].astype(str),gene_sets=library_name, outdir=None)
    enr_down = gp.enrichr(degs_down['%s_names'%cluster].astype(str), gene_sets=library_name, outdir=None)

    enr_up.res2d['UP_DW'] = "UP"
    enr_down.res2d['UP_DW'] = "DOWN"

    enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
    df_combined = enr_res.sort_values('Combined Score', ascending=False)
    df_combined['Overlap'] = df_combined['Overlap'].astype(str)
    df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]

    df_sig.to_excel(path + 'Zone_%s GSEA/pathway list %s.xlsx' % (cluster, file_name), index=False)
    df_combined.to_excel(path + 'Zone_%s GSEA/entire pathway list %s.xlsx' % (cluster, file_name), index=False)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    gp.barplot(enr_res, figsize=(6, 6),
               group='UP_DW',
               title="%s" % (library_name),
               ax=ax,
               color=['b', 'r'])

    fig.savefig(path + 'Zone_%s GSEA/%s.png' % (cluster, file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/Zone_%s GSEA/'%cluster):
        os.makedirs(path + 'svg/Zone_%s GSEA/'%cluster)

    fig.savefig(path + 'svg/Zone_%s GSEA/%s.svg' % (cluster, file_name), bbox_inches='tight')





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
#                  annot_kws={'size': 4, 'weight':'normal'},
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

# plt.xticks(fontsize=4, color='0.2', rotation=35, rotation_mode='anchor', ha='right', weight='normal')
# plt.yticks(fontsize=4, color='0.2', weight='normal')
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=7, rotation=35, rotation_mode='anchor',ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=7, va='center')

plt.savefig(path+'RNA vs behavior features correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/RNA vs behavior features correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()
