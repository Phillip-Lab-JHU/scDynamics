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
from utils.rna_utils import *
from features.interaction import ZoneSignal
import scanpy as sc
from umap import UMAP

import warnings
warnings.filterwarnings('ignore')

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
sc.pp.neighbors(adata_rna, n_pcs=330)
sc.tl.umap(adata_rna)

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior(test)\\'
#color_list = ('#F06293', '#AF79C4', '#5B6BBF', '#FFC11C', '#D1917C', '#4DB6AC', '#00BCD4', '#A1877E', '#8F96A8', '#B02E8B', '#EB974D','#BCBCBC',)
# color_list = [(0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0), (0.6196078431372549, 0.8549019607843137, 0.8980392156862745, 1.0),
#               (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)]
#
# df = pd.DataFrame(adata_rna.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
# df['cluster'] = adata_rna.obs['cluster'].values
#
# draw_umap_space(df, path, file_name='space_cluster', condition_name='cluster', label_name='cluster',
#                 colors=color_list, x_name='UMAP1', y_name='UMAP2', dot_size=0.07)
# cell_counts = adata_rna.obs['cluster'].value_counts().sort_index()
# print(cell_counts)
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

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior(test)\\'


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



concat = np.concatenate([xs, trans_xt], axis=0)

__umap = UMAP(metric='euclidean', n_components=2, n_neighbors=30, min_dist=0.5, random_state=0)
umap = __umap.fit_transform(concat)
umap_rna = umap[labels == 'source', :]
umap_behavior = umap[labels == 'target', :]

####################### Visualize results #######################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior(test)\MC RNA profiles\\'
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
scatter1 = ax[1, 0].scatter(umap_rna[:, 0], umap_rna[:, 1], c=df_xs[condition_name].replace(list(np.unique(df_xs[condition_name])),
                                [i for i in range(np.unique(df_xs[condition_name]).shape[0])]), label = df_xs[condition_name], alpha=0.5, s=1,
                            cmap=plt.cm.get_cmap('Set1'))
handles, lab = scatter1.legend_elements(num=None)
ax[1, 0].legend(handles=handles, labels=list(np.unique(df_xs[condition_name])),
           #bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
           fontsize=6, frameon=False, markerscale=0.6)
ax[1, 0].set_title('RNA labels')

condition_name='cluster'
scatter1 = ax[1, 1].scatter(umap_behavior[:, 0], umap_behavior[:, 1], c=df_xt[condition_name].replace(list(np.unique(df_xt[condition_name])),
                                [i for i in range(np.unique(df_xt[condition_name]).shape[0])]), label = df_xt[condition_name], alpha=0.5, s=1,
                            cmap=plt.cm.get_cmap('Set1'))

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
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior(test)\MC RNA profiles\\'
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(trans_xt, df['kmeans'])
pred_kmeans = knn.predict(xs)
adata_rna.obs['MC'] = pred_kmeans.astype('str')
adata_rna.obs['MC'] = adata_rna.obs['MC'].astype("category")

####################### Plot Imputed MC #######################

file_name = 'KNN classifier MC imputation_labelcoeff_%s_alpha_%s_eps_%s'%(label_coeff, alpha, epsilon)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(umap[labels == 'source', 0], umap[labels == 'source', 1], color='gray', label='s', alpha=0.2, s=1)
scatter1 = ax[0].scatter(umap[labels == 'target', 0], umap[labels == 'target', 1], c=df['kmeans'].replace(list(np.unique(df['kmeans'])),
                                [i for i in range(np.unique(df['kmeans']).shape[0])]), cmap=cmc.batlow, label=df['kmeans'], alpha=0.5, s=1)
handles, lab = scatter1.legend_elements(num=None)
ax[0].legend(handles=handles, labels=list(np.unique(df['kmeans'])),
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


cmap = cmc.batlow
comm_list = np.unique(df['kmeans'])
color_list = [cmap(j / max(1, len(comm_list)-1)) for j, comm in enumerate(comm_list)]

for cluster in np.unique(pred_kmeans):
    file_name = 'MC%s KNN classifier BC imputation_labelcoeff_%s_alpha_%s_eps_%s'%(cluster, label_coeff, alpha, epsilon)

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


#################################### Box plot comparing all motility features by cell types ####################################
test = 'kruskal-wallis_dunn'

if not os.path.isdir(path + 'Module score for custom signatures/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'Module score for custom signatures/')

cmap = cmc.batlow
comm_list = np.unique(df['kmeans'])
color_list = [cmap(j / max(1, len(comm_list)-1)) for j, comm in enumerate(comm_list)]
for sig_name in list(sig.keys()):
    dataset={}
    for condition in np.unique(adata_rna.obs['MC']):
        data = adata_rna.obs[adata_rna.obs['MC'] == condition]['%s_score' % sig_name]
        dataset[condition] = np.array(data)

    draw_custom_violin_plot(dataset, path+'Module score for custom signatures/', file_name=sig_name, colors=color_list,
                            test=test, pvalue=True, return_sig=True, figsize=(5,3))

    draw_custom_box_plot(dataset, path+ 'Module score for custom signatures/', file_name=sig_name+'_box', colors=color_list,
    strip_plot=False, test=test, pvalue=True, return_sig=True, figsize=(5,3))


##################### Violin plot for single gene expression comparions #####################
folder_name = 'Single gene expression comparions for imputed BCs'

sig_names = ['Holmes_2020 BCR signaling', 'Holmes_2020 CD40 signaling', 'Holmes_2020 NF-kappaB', 'Holmes_2020 NF-kappaB',
             'Holmes_2020 MYC', 'Holmes_2020 DZ', 'Holmes_2020 PreM', 'Holmes_2020 IG', 'Tfh interaction', 'FDC interaction']

#'MHC class 2 antigen presentation', 'antigen processing and presentation',
for sig_name in sig_names:
    genes = sig[sig_name]
    for gene in genes:
        try:
            fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
            sc.pl.violin(adata_rna, gene, groupby="BC",
                stripplot=False,  # remove the internal dots
                inner="box",  # adds a boxplot inside violins
                cmap=cmc.batlow,
                ax = ax, show=False
            )

            if not os.path.isdir(path + '%s/' % folder_name):
                os.makedirs(path + '%s/' % folder_name)

            fig.savefig(path + '%s/%s_%s.png' % (folder_name, sig_name, gene), dpi=300, bbox_inches='tight')

            if not os.path.isdir(path + 'svg/%s/' % folder_name):
                os.makedirs(path + 'svg/%s/' % folder_name)

            fig.savefig(path + 'svg/%s/%s_%s.svg' % (folder_name, sig_name, gene), bbox_inches='tight')

            plt.clf()
            plt.close()
        except:
            print('%s: No %s gene'%(sig_name, gene))

##################### DE analysis on imputed MCs #####################

sc.tl.rank_genes_groups(adata_rna, groupby="MC", method="wilcoxon", key_added="dea_MC", reference='rest')
#sc.tl.rank_genes_groups(adata_rna, groupby="MC", method="t-test_overestim_var", key_added="dea_MC", reference='rest')
sc.tl.filter_rank_genes_groups(adata_rna, min_in_group_fraction=0.1, max_out_group_fraction=0.5, key="dea_MC",key_added="dea_MC_filtered")

result = adata_rna.uns["dea_MC"]
groups = result["names"].dtype.names
degs = pd.DataFrame(
    {group + '_' + key: result[key][group]
    for group in groups for key in ['names','scores', 'pvals','pvals_adj','logfoldchanges']})

##################### Gene expression heatmap #####################

fig, ax = plt.subplots(figsize=(20, 5), constrained_layout=True)
sc.pl.rank_genes_groups_dotplot(adata_rna, groupby="MC", standard_scale="var", n_genes=10, key="dea_MC", ax=ax, show=False)

fig.savefig(path + 'MC deg.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/MC deg.svg', bbox_inches='tight')
plt.close()
plt.clf()

fig, ax = plt.subplots(figsize=(20, 5), constrained_layout=True)
sc.pl.rank_genes_groups_dotplot(adata_rna,groupby="MC", standard_scale="var", n_genes=10,key="dea_MC_filtered", ax=ax,show=False)
fig.savefig(path + 'MC deg filtered.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/MC deg filtered.svg', bbox_inches='tight')
plt.close()
plt.clf()



##################### Calculate adjusted score (distinctiveness_score) #####################
n_top_genes = 300
mc_degs_sig_sets, degs = get_distinct_DEG_per_cluster(adata_rna, cluster_name='MC', n_top_genes=n_top_genes)

##################### UpSet plot for gene overlap counts #####################
from upsetplot import UpSet, from_contents, plot
mc_degs_sig_sets = from_contents(mc_degs_sig_sets)
fig = plt.figure(figsize=(10, 6))
UpSet(mc_degs_sig_sets, show_counts=True, min_subset_size=10, sort_by='degree', orientation='horizontal', element_size=None).plot(fig=fig)
#plot(UpSet(mc_degs_sig_sets, show_counts=True, min_subset_size=10, sort_by='degree', orientation='horizontal'), fig=fig)

plt.savefig(path+'upset plot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/upset plot.svg', bbox_inches='tight')
plt.close()
plt.clf()


##################### Pathway enrichment analysis for imputed MCs #####################

for cluster in tqdm(np.unique(adata_rna.obs['MC'])):
    # Extract these columns as a DataFrame
    name_col = f"{cluster}_names"
    score_col = f"{cluster}_adj_scores"
    logfc_col = f"{cluster}_logfoldchanges"

    temp_df = degs[[name_col, score_col, logfc_col]].copy()
    temp_df["abs_score"] = temp_df[score_col].abs()
    temp_df_sorted = temp_df.sort_values(by="abs_score", ascending=False).head(n_top_genes)

    # Split into UP and DOWN
    degs_up = temp_df_sorted[temp_df_sorted[logfc_col] > 0]
    degs_down = temp_df_sorted[temp_df_sorted[logfc_col] < 0]

    # degs_sig = degs[degs['%s_pvals_adj'%cluster] < 0.05]
    # degs_up = degs_sig[degs_sig['%s_logfoldchanges'%cluster] > 0]
    # degs_down = degs_sig[degs_sig['%s_logfoldchanges'%cluster] < 0]


    gsea_path = path + 'MC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon)

    if not os.path.isdir(gsea_path):
        os.makedirs(gsea_path)

    run_gsea(degs_up, degs_down, h2m_dict, gsea_path, col_name='%s_names' % cluster)


for cluster in tqdm(np.unique(adata_rna.obs['MC'])):
    name_col = f"{cluster}_names"
    score_col = f"{cluster}_adj_scores"
    logfc_col = f"{cluster}_logfoldchanges"

    temp_df = degs[[name_col, score_col, logfc_col]].copy()
    temp_df["abs_score"] = temp_df[score_col].abs()
    temp_df_sorted = temp_df.sort_values(by="abs_score", ascending=False).head(n_top_genes)

    # Split into UP and DOWN
    degs_up = temp_df_sorted[temp_df_sorted[logfc_col] > 0]
    degs_down = temp_df_sorted[temp_df_sorted[logfc_col] < 0]

    print(cluster, 'Upregulated: %s'%degs_up.shape[0],'Downregulated: %s'%degs_down.shape[0])

    gsea_path = path + 'MC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon)
    if not os.path.isdir(gsea_path):
        os.makedirs(gsea_path)

    temp_df_sorted.to_excel(gsea_path+'MC %s DEG.xlsx'%cluster)

    df_p = degs.copy()
    df_p = draw_gene_rank_plot(df_p, gsea_path, file_name='gene_rank_up_%s_down_%s'%(degs_up.shape[0], degs_down.shape[0]),
                               gene_col='%s_names'%cluster, p_col='%s_pvals_adj'%cluster, score_col='%s_scores'%cluster, figsize=(4, 7), dot_size=7)

############# Plot pathway graph networks #############
for cluster in np.unique(adata_rna.obs['MC']):
    gsea_path = path + 'MC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon)
    df_sigs = pd.read_excel(gsea_path + 'significant pathway.xlsx')

    df_sig_graph = df_sigs[df_sigs['Gene_set'] != 'gs_ind_0'].reset_index(drop=True)
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

    draw_graph_network(custom_sig, gsea_path, file_name='gene overlap', sample_n=8, resolution=1.06, regulation=regulation,
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

########################## Plot curated pathways enrichment ##########################
cluster=3
curated_pathways = pd.read_excel(path + 'MC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/curated pathway.xlsx'%(cluster, label_coeff, alpha, epsilon))


font = {'family': 'arial',
        'weight': 'normal', }
matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

gp.barplot(curated_pathways,
           group='UP_DW',
           # title ="%s"%(library_name),
           ax=ax,
           color=['#6699CC', '#CC6677'])

fig.savefig(path + 'MC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/curated pathway.png'%(cluster, label_coeff, alpha, epsilon), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/MC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s'%(cluster, label_coeff, alpha, epsilon)):
    os.makedirs(path + 'svg/MC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s'%(cluster, label_coeff, alpha, epsilon))

fig.savefig(path + 'svg/MC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/curated pathway.svg'%(cluster, label_coeff, alpha, epsilon), bbox_inches='tight')

# # Retrieve the PCA results
# X_pca = adata_rna.obsm['X_pca']  # shape: (n_cells, 330)
# pcs = adata_rna.varm['PCs']   # shape: (3000, 330)
# gene_means = adata_rna.var['mean']  # shape: (3000,)
# gene_stds = adata_rna.var['std']  # shape: (3000,)
#
# # Inverse transform: approximate scaled data matrix
# behavior_gene_scaled = trans_xt @ pcs.T  # shape: (n_cells, 3000)
#
# # Undo scaling: multiply by stddev and add mean
#
# behavior_gene = (behavior_gene_scaled * gene_stds.values) + np.array(gene_means)
##################### Multipartite graph from MC -> BC -> RNA profiles -> Pathway analysis (with comm detection) #####################

# === Step 0-1: DEG information for MC ===

_, mc_degs = get_distinct_DEG_per_cluster(adata_rna, cluster_name='MC', n_top_genes=n_top_genes)

upregulated_genes_per_mc = {}
downregulated_genes_per_mc = {}
deg_scores_per_mc = {}
for cluster in np.unique(adata_rna.obs['MC']):
    name_col = f"{cluster}_names"
    score_col = f"{cluster}_adj_scores"
    logfc_col = f"{cluster}_logfoldchanges"

    temp_df = mc_degs[[name_col, score_col, logfc_col]].copy()
    temp_df["abs_score"] = temp_df[score_col].abs()
    temp_df_sorted = temp_df.sort_values(by="abs_score", ascending=False).head(n_top_genes)

    degs_up = temp_df_sorted[temp_df_sorted[logfc_col] > 0]['%s_names'%cluster]
    degs_down = temp_df_sorted[temp_df_sorted[logfc_col] < 0]['%s_names'%cluster]

    # degs_sig = mc_degs[mc_degs['%s_pvals_adj'%cluster] < 0.01]
    # degs_up = degs_sig[degs_sig['%s_logfoldchanges'%cluster] > 0]['%s_names'%cluster]
    # degs_down = degs_sig[degs_sig['%s_logfoldchanges'%cluster] < 0]['%s_names'%cluster]
    upregulated_genes_per_mc[cluster] = np.array(degs_up)
    downregulated_genes_per_mc[cluster] = np.array(degs_down)

    scores = temp_df_sorted[[f'{cluster}_names', f'{cluster}_adj_scores']].rename(
        columns={f'{cluster}_names': 'gene', f'{cluster}_adj_scores': 'score'})

    deg_scores_per_mc[cluster] = scores.sort_values(by='score', ascending=False).reset_index(drop=True)
    print(cluster, temp_df_sorted.shape[0])

# === Step 0-2: DEG information for BC ===
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(trans_xt, df['beh_kmeans'])
pred_kmeans = knn.predict(xs)
adata_rna.obs['BC'] = pred_kmeans.astype('str')
adata_rna.obs['BC'] = adata_rna.obs['BC'].astype("category")
sc.tl.rank_genes_groups(adata_rna, groupby="BC", method="wilcoxon", key_added="dea_BC", reference='rest')

result = adata_rna.uns["dea_BC"]
groups = result["names"].dtype.names
bc_degs = pd.DataFrame(
    {group + '_' + key: result[key][group]
    for group in groups for key in ['names','scores', 'pvals','pvals_adj','logfoldchanges']})

upregulated_genes_per_bc = {}
downregulated_genes_per_bc = {}
for cluster in np.unique(adata_rna.obs['BC']):
    degs_sig = bc_degs[bc_degs['%s_pvals_adj'%cluster] < 0.05]
    degs_up = degs_sig[degs_sig['%s_logfoldchanges'%cluster] > 0]['%s_names'%cluster]
    degs_down = degs_sig[degs_sig['%s_logfoldchanges'%cluster] < 0]['%s_names'%cluster]
    upregulated_genes_per_bc[cluster] = np.array(degs_up)
    downregulated_genes_per_bc[cluster] = np.array(degs_down)

# === Step 0-3: MC -> BC transition ===

df_graph = df.copy()
df_graph['kmeans'] = df_graph['kmeans'].astype(str)
df_graph['beh_kmeans'] = df_graph['beh_kmeans'].astype(str)
transition_counts = df_graph.groupby(['kmeans', 'beh_kmeans']).size().reset_index(name='count')
transition_counts['fraction'] = transition_counts.groupby('kmeans')['count'].transform(lambda x: x / x.sum())


# === Step 0-4: curate pathway data ===

pathway_df = []
for cluster in np.unique(df['kmeans']):
    gsea_path = path + 'MC%s GSEA_labelcoeff_%s_alpha_%s_eps_%s/' % (cluster, label_coeff, alpha, epsilon)
    df_temp = pd.read_excel(gsea_path + 'community_detected_pathways.xlsx')
    pathway_df.append(df_temp)

pathway_df = pd.concat(pathway_df, axis=0, ignore_index=True)

# Split by regulation
df_up = pathway_df[pathway_df["Regulation"] == "UP"].copy()
df_down = pathway_df[pathway_df["Regulation"] == "DOWN"].copy()
df_up["Gene_Set"] = df_up["Genes"].apply(lambda x: set(x.split(";")))
df_down["Gene_Set"] = df_down["Genes"].apply(lambda x: set(x.split(";")))

# Remove duplicates within UP and DOWN separately
df_up = df_up.sort_values("Number_of_Genes", ascending=False).drop_duplicates(subset="Pathway", keep='first')
df_down = df_down.sort_values("Number_of_Genes", ascending=False).drop_duplicates(subset="Pathway", keep='first')


df_up_graph = df_up.dropna(subset=['Genes'])
df_down_graph = df_down.dropna(subset=['Genes'])

sig_up = {
    row['Pathway']: np.array(genes)
    for _, row in df_up_graph.iterrows()
    if len(genes := row['Genes'].split(';')) >= 3
}

sig_down = {
    row['Pathway']: np.array(genes)
    for _, row in df_down_graph.iterrows()
    if len(genes := row['Genes'].split(';')) >= 3
}


up_communities = community_detection_louvain(sig_up, resolution=1.05)
down_communities = community_detection_louvain(sig_up, resolution=1.05)
up_communities['Regulation'] = 'UP'
down_communities['Regulation'] = 'DOWN'
df_merged = pd.concat([up_communities, down_communities], ignore_index=True)
#df_merged.to_excel(path+'multipartite graph/pathway list.xlsx')

# ===== Step 1: Create Node MC → BC → DEG → Pathway =====
B = nx.Graph()

# === Create Node (MC and BC) ===
motility_clusters = sorted(df_graph['kmeans'].unique())
behavior_clusters = sorted(df_graph['beh_kmeans'].unique())

B.add_nodes_from([f'MC{mc}' for mc in motility_clusters], bipartite=0)
B.add_nodes_from([f'BC{bc}' for bc in behavior_clusters], bipartite=1)

# === Create Node (Genes) ===
cmap = cmc.batlow
comm_list = motility_clusters
mc_color_list = [cmap(j / max(1, len(comm_list)-1)) for j, comm in enumerate(comm_list)]
bc_color_list = ['#BAC8DA', '#4F609C', '#8A4F21', '#F06293', '#E9C61D', '#BCBCBC', '#BEDCB0', '#F5A9F5' ]

entire_gene_list = []
gene_nodes = []
gene_node_colors = []

for mc_idx, mc in enumerate(motility_clusters):
    mc_scores = deg_scores_per_mc[mc]
    for i, row in mc_scores.iterrows():
        gene = row['gene']
        score = row['score']
        if score >= 0:
            g_node = f'G_{gene}_MC{mc}_UP'
        else:
            g_node = f'G_{gene}_MC{mc}_DOWN'
        gene_nodes.append(g_node)
        entire_gene_list.append(gene)
        gene_node_colors.append(mc_color_list[mc_idx])
        B.add_node(g_node, bipartite=2)

entire_gene_list = list(set(entire_gene_list))

# === Create Node (Pathway) ===
cmap = plt.get_cmap("tab20")
unique_communities = sorted(df_merged["Component_Community"].dropna().unique())
pathway_color_list = {comm: cmap(j / max(1, len(unique_communities)-1)) for j, comm in enumerate(unique_communities)}


# Define layout index offset
pathway_nodes = []
pathway_node_colors = []
pathway_edges = []
gene_pathway_edge_colors = []

for i, row in up_communities.iterrows():
    p_node = f'P_UP_{row.Pathway}'
    B.add_node(p_node, bipartite=3)
    pathway_nodes.append(p_node)

    color = pathway_color_list[row["Component_Community"]]
    pathway_node_colors.append(color)

    # Connect to gene nodes
    for g in row['Genes'].split(";"):
        matching_gene_nodes = [n for n in gene_nodes if n.startswith(f'G_{g}_') and '_UP' in n]
        for g_node in matching_gene_nodes:
            B.add_edge(g_node, p_node, weight=1)
            pathway_edges.append((g_node, p_node))
            gene, mc_reg = g_node.replace("G_", "").rsplit("_MC", 1)  # ['gene_name', '0_UP']
            mc, reg = mc_reg.split('_')  # ['0', 'UP']
            gene_pathway_edge_colors.append(mc_color_list[int(mc)])

for i, row in down_communities.iterrows():
    p_node = f'P_DOWN_{row.Pathway}'
    B.add_node(p_node, bipartite=3)
    pathway_nodes.append(p_node)

    color = pathway_color_list[row["Component_Community"]]
    pathway_node_colors.append(color)

    # Connect to gene nodes
    for g in row['Genes'].split(";"):
        matching_gene_nodes = [n for n in gene_nodes if n.startswith(f'G_{g}_') and '_DOWN' in n]
        for g_node in matching_gene_nodes:
            B.add_edge(g_node, p_node, weight=1)
            pathway_edges.append((g_node, p_node))
            gene, mc_reg = g_node.replace("G_", "").rsplit("_MC", 1) # ['gene_name', '0_UP']
            mc, reg = mc_reg.split('_')  # ['0', 'UP']
            gene_pathway_edge_colors.append(mc_color_list[int(mc)])

# ===== Step 2: Adjust Node Layout =====
pos = {} # y-position from 1~0
for i, node in enumerate([f'MC{mc}' for mc in motility_clusters]):
    pos[node] = (0, 1 - i / max(len(motility_clusters) - 1, 1))
for i, node in enumerate([f'BC{bc}' for bc in behavior_clusters]):
    pos[node] = (1, 1 - i / max(len(behavior_clusters) - 1, 1))
for i, node in enumerate(gene_nodes):
    pos[node] = (2, 1 - i / max(len(gene_nodes) - 1, 1))

# Define y-scaling helper
def scaled_y(index, total, y_top, y_bottom):
    if total == 1:
        return (y_top + y_bottom) / 2
    return y_top - index * ((y_top - y_bottom) / (total - 1))

# Compute dynamic space allocation
n_up = len(up_communities)
n_down = len(down_communities)
total_height = 1.0
gap = 0.01

# Allocate space proportionally
h_up = (n_up / (n_up + n_down)) * (total_height - gap)
h_down = (n_down / (n_up + n_down)) * (total_height - gap)

# UP: from 1.0 → (1.0 - h_up)
y_up_top = 1.0
y_up_bottom = 1.0 - h_up

# DOWN: from (y_up_bottom - gap) → bottom
y_down_top = y_up_bottom - gap
y_down_bottom = 0.0

# Assign UP pathway positions
for i, row in up_communities.iterrows():
    node = f'P_UP_{row.Pathway}'
    pos[node] = (3, scaled_y(i, n_up, y_up_top, y_up_bottom))

# Assign DOWN pathway positions
for i, row in down_communities.iterrows():
    node = f'P_DOWN_{row.Pathway}'
    pos[node] = (3, scaled_y(i, n_down, y_down_top, y_down_bottom))

# ===== Step 3: Create Edges (MC → BC), (BC → DEG), (DEG → Pathway) =====

# === Create Edges (MC → BC) ===
mc_bc_edge_colors = []
for _, row in transition_counts.iterrows():
    B.add_edge(f'MC{row["kmeans"]}', f'BC{row["beh_kmeans"]}', weight=row['fraction'])
    mc_bc_edge_colors.append( mc_color_list[int(row["kmeans"])] )

# === Create Edges (BC -> Genes) ===
gene_edges = []
gene_edge_colors = []

for bc in upregulated_genes_per_bc:
    bc_node = f'BC{bc}'
    bc_up = set(upregulated_genes_per_bc[bc])
    for g_node in gene_nodes:
        gene, mc = g_node.replace("G_", "").rsplit("_MC", 1)
        if gene in bc_up:
            B.add_edge(bc_node, g_node, weight=1)
            gene_edges.append((bc_node, g_node))
            #gene_edge_colors.append('#D3D3D3')
            gene_edge_colors.append( bc_color_list[int(bc)] )

for bc in downregulated_genes_per_bc:
    bc_node = f'BC{bc}'
    bc_down = set(downregulated_genes_per_bc[bc])
    for g_node in gene_nodes:
        gene, mc = g_node.replace("G_", "").rsplit("_MC", 1)
        if gene in bc_down:
            B.add_edge(bc_node, g_node, weight=1)
            gene_edges.append((bc_node, g_node))
            #gene_edge_colors.append('#D3D3D3')
            gene_edge_colors.append(bc_color_list[int(bc)])

# ===== Step 4: Draw Full Graph  =====
# === Draw Nodes ===
fig, ax = plt.subplots(figsize=(8, 8))
nx.draw_networkx_nodes(B, pos, nodelist=[f'MC{mc}' for mc in motility_clusters], node_color=mc_color_list, node_size=1000) # 'lightblue'
nx.draw_networkx_nodes(B, pos, nodelist=[f'BC{bc}' for bc in behavior_clusters], node_color=bc_color_list, node_size=1000) # 'lightgreen'
#nx.draw_networkx_nodes(B, pos, nodelist=gene_nodes, node_color=gene_node_colors, node_size=0.1, node_shape='s')

from matplotlib.patches import Rectangle

box_width = 0.06
# Draw rectangular gene nodes manually
for node, color in zip(gene_nodes, gene_node_colors):
    if node in pos:
        x, y = pos[node]
        rect = Rectangle(
            (x, y),  # (x0, y0): bottom-left corner
            box_width,                     # width (along x-axis)
            0.002,                    # height (along y-axis)
            linewidth=0,
            edgecolor=None,
            facecolor=color
        )
        ax.add_patch(rect)

for node, color in zip(pathway_nodes, pathway_node_colors):
    if node in pos:
        x, y = pos[node]
        rect = Rectangle(
            (x, y),  # (x0, y0): bottom-left corner
            box_width,                     # width (along x-axis)
            0.002,                    # height (along y-axis)
            linewidth=0,
            edgecolor=None,
            facecolor=color
        )
        ax.add_patch(rect)


# === Draw Edges ===
mb_bc_edges = [(u, v) for u, v in B.edges() if u.startswith('MC') or v.startswith('MC')]
for (u, v), color in zip(mb_bc_edges, mc_bc_edge_colors):
    if u in pos and v in pos:
        draw_bezier_edge(ax, pos[u], pos[v], color=color, alpha=0.6, lw=2, ctrl_offset=0.6)

for (u, v), color in zip(gene_edges, gene_edge_colors):
    if u in pos and v in pos:
        draw_bezier_edge(ax, pos[u], pos[v], color=color, alpha=0.2, lw=0.1, ctrl_offset=0.6)

for (u, v), color in tqdm(zip(pathway_edges, gene_pathway_edge_colors)):
    if u in pos and v in pos:
        draw_bezier_edge(ax, tuple(np.add(pos[u], (box_width, 0))), pos[v], color=color, alpha=0.15, lw=0.1, ctrl_offset=0.6)


#nx.draw_networkx_edges(B, pos, edgelist=mb_bc_edges, width=2, edge_color=mc_bc_edge_colors, alpha=0.6 ) # '#D3D3D3'
#nx.draw_networkx_edges(B, pos, edgelist=gene_edges, width=0.01, edge_color=gene_edge_colors, alpha=0.2)

nx.draw_networkx_labels(B, pos, labels={n: n for n in B.nodes if n.startswith('MC') or n.startswith('BC')}, font_size=14)
plt.axis('off')
if not os.path.exists(path + 'multipartite graph/'):
    os.makedirs(path + 'multipartite graph/')
plt.savefig(path + 'multipartite graph/all_with_genes_with_pathway.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'multipartite graph/svg/'):
    os.makedirs(path + 'multipartite graph/svg/')
fig.savefig(path + 'multipartite graph/svg/all_with_genes_with_pathway.svg', bbox_inches='tight')
plt.close()
plt.clf()

# ===== Step 6: Save gene -> pathway edges  =====
from collections import defaultdict
pathway_to_mcs = defaultdict(set)

for g_node, p_node in pathway_edges:
    if p_node.startswith("P_"):
        pathway_name = p_node[2:]  # Remove 'P_' prefix
        try:
            parts = g_node.replace("G_", "").rsplit("_MC", 1)
            gene = parts[0]
            mc_part = parts[1]
            mc_id, regulation = mc_part.split("_")
            pathway_to_mcs[pathway_name].add(f"MC{mc_id}")
        except Exception as e:
            print(f"Skipping malformed entry: {g_node} → {p_node}")

# Step 2: Add 'Connected_MCs' column to df_merged
def lookup_mcs(pathway):
    return sorted(pathway_to_mcs.get(pathway, []))

df_temp = pd.DataFrame()
df_temp['look_up'] = df_merged["Regulation"].astype(str)+'_'+df_merged["Pathway"].astype(str)
df_merged["Connected_MCs"] = df_temp["look_up"].apply(lookup_mcs)
df_merged.to_excel(path+'multipartite graph/pathway list.xlsx')

# ===== Step 5: Draw per-MC Graph  =====
for mc in motility_clusters:
    mc_node = f'MC{mc}'
    fig, ax = plt.subplots(figsize=(8, 8))

    all_mc_nodes = [f'MC{mc_}' for mc_ in motility_clusters]
    all_bc_nodes = [f'BC{bc_}' for bc_ in behavior_clusters]

    # --- Step 1: Rank BC connections ---
    connected_bcs = [(v, B[mc_node][v]['weight']) for u, v in B.edges(mc_node) if v.startswith("BC")]
    connected_bcs.sort(key=lambda x: x[1], reverse=True)

    n_top = 1
    n_half = max(len(connected_bcs) // 2, 1)

    top_edges = connected_bcs[:n_top]
    top_half_edges = connected_bcs[n_top:n_half]
    remaining_edges = connected_bcs[n_half:]

    # --- Step 2: Assign BC node colors based on category ---
    bc_node_colors = {}
    bc_category = {}  # Will store for drawing gene edges later

    for bc, _ in top_edges:
        bc_node_colors[bc] = '#8B0000'
        bc_category[bc] = 'top'
    for bc, _ in top_half_edges:
        bc_node_colors[bc] = '#FA8072'
        bc_category[bc] = 'half'
    for bc, _ in remaining_edges:
        bc_node_colors[bc] = '#FFDAB9'
        bc_category[bc] = 'rest'
    for bc in behavior_clusters:
        bc_node = f'BC{bc}'
        if bc_node not in bc_node_colors:
            bc_node_colors[bc_node] = '#BCBCBC'
            bc_category[bc_node] = 'none'

    # --- Step 3: Draw all MCs ---
    mc_colors = ['#B02E8B' if node == mc_node else '#BCBCBC' for node in all_mc_nodes]
    nx.draw_networkx_nodes(B, pos, nodelist=all_mc_nodes, node_color=mc_colors, node_size=1500)

    # --- Step 4: Draw all BCs ---
    bc_colors = [bc_node_colors[bc_node] for bc_node in all_bc_nodes]
    nx.draw_networkx_nodes(B, pos, nodelist=all_bc_nodes, node_color=bc_colors, node_size=1500)

    # --- Step 5: Draw all gene nodes (unchanged set + color) ---
    #nx.draw_networkx_nodes(B, pos, nodelist=gene_nodes, node_color=gene_node_colors, node_size=20)
    for node, color in zip(gene_nodes, gene_node_colors):
        if node in pos:
            x, y = pos[node]
            rect = Rectangle(
                (x, y),  # (x0, y0): bottom-left corner
                box_width,  # width (along x-axis)
                0.002,  # height (along y-axis)
                linewidth=0,
                edgecolor=None,
                facecolor=color
            )
            ax.add_patch(rect)

    # --- Step 6: Draw all pathway nodes (unchanged set + color) ---
    for node, color in zip(pathway_nodes, pathway_node_colors):
        if node in pos:
            x, y = pos[node]
            rect = Rectangle(
                (x, y),  # (x0, y0): bottom-left corner
                box_width,  # width (along x-axis)
                0.002,  # height (along y-axis)
                linewidth=0,
                edgecolor=None,
                facecolor=color
            )
            ax.add_patch(rect)

    # --- Step 7: Draw MC → BC edges by strength ---
    edge_sets = {
        '#8B0000': [(mc_node, bc) for bc, _ in top_edges],
        '#FA8072': [(mc_node, bc) for bc, _ in top_half_edges],
        '#FFDAB9': [(mc_node, bc) for bc, _ in remaining_edges],
    }
    # for color, edges in edge_sets.items():
    #     widths = [B[u][v]['weight'] * 30 for u, v in edges]
    #     nx.draw_networkx_edges(B, pos, edgelist=edges, width=widths, edge_color=color)

    for color, edges in edge_sets.items():
        widths = [B[u][v]['weight'] * 30 for u, v in edges]
        for (u, v), width in zip(edges, widths):
            if u in pos and v in pos:
                draw_bezier_edge(ax, pos[u], pos[v], color=color, alpha=0.6, lw=width, ctrl_offset=0.6)

    # --- Step 8: Draw only BC → gene edges for this MC ---
    gene_edges_by_category = {'top': [], 'half': [], 'rest': []}
    gene_color_by_category = {'top': '#8B0000', 'half': '#FA8072', 'rest': '#FFDAB9'}

    for bc in bc_category:
        if bc_category[bc] == 'none':
            continue  # skip unconnected BCs
        bc_id = bc.replace("BC", "")

        # bc_genes = set(upregulated_genes_per_bc.get(bc_id, [])).union(downregulated_genes_per_bc.get(bc_id, []))
        #
        # for g in bc_genes:
        #     g_node = f'G_{g}_MC{mc}'
        #     if g_node in B.nodes:
        #         gene_edges_by_category[bc_category[bc]].append((bc, g_node))
        bc_genes_up = upregulated_genes_per_bc.get(bc_id, [])
        bc_genes_down = downregulated_genes_per_bc.get(bc_id, [])
        for g in bc_genes_up:
            g_node = f'G_{g}_MC{mc}_UP'
            if g_node in B.nodes:
                gene_edges_by_category[bc_category[bc]].append((bc, g_node))
        for g in bc_genes_down:
            g_node = f'G_{g}_MC{mc}_DOWN'
            if g_node in B.nodes:
                gene_edges_by_category[bc_category[bc]].append((bc, g_node))


    for cat in ['rest', 'half', 'top']:
        edges = gene_edges_by_category[cat]
        for (u, v) in edges:
            if u in pos and v in pos:
                draw_bezier_edge(ax, pos[u], pos[v], color=gene_color_by_category[cat], alpha=0.6, lw=0.1, ctrl_offset=0.6)

    # --- Step 9: Draw only gene → pathway edges for this MC ---
    pathway_edges_by_category = {'top': [], 'half': [], 'rest': []}
    pathway_color_by_category = {'top': '#8B0000', 'half': '#FA8072', 'rest': '#FFDAB9'}

    # For each gene node of MC{mc}
    mc_gene_nodes = [n for n in gene_nodes if f'_MC{mc}' in n]
    for g_node in mc_gene_nodes:
        gene = g_node.replace(f'_MC{mc}', '').replace('G_', '')

        # Find connected pathways for this gene
        connected_pathways = [n for n in B.neighbors(g_node) if n.startswith('P_')]
        for p_node in connected_pathways:
            if p_node in B[g_node]:  # confirm edge exists in graph
                #draw_bezier_edge(ax, pos[g_node], pos[p_node], color=mc_color_list[int(mc)], alpha=0.4, lw=0.01, ctrl_offset=0.6)
                draw_bezier_edge(ax, tuple(np.add(pos[g_node], (box_width, 0))), pos[p_node], color='#B02E8B', alpha=0.4, lw=0.1, ctrl_offset=0.6)

    # --- Step 10: Labels for MC and BC only ---
    label_nodes = all_mc_nodes + all_bc_nodes
    nx.draw_networkx_labels(B, pos, labels={n: n for n in label_nodes}, font_size=16)

    # --- Step 11: Save ---
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path + f'multipartite graph/MC_focused_{mc_node}_bc_gene_conditional_with_pathway.png', dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'multipartite graph/svg/'):
        os.makedirs(path + 'multipartite graph/svg/')
    fig.savefig(path + f'multipartite graph/svg/MC_focused_{mc_node}_bc_gene_conditional_with_pathway.svg', bbox_inches='tight')
    plt.close()
    plt.clf()























# === Step 2: Build bipartite graph ===
B = nx.Graph()

motility_clusters = sorted(df_graph['kmeans'].unique())
behavior_clusters = sorted(df_graph['beh_kmeans'].unique())

# Add nodes with bipartite attribute
B.add_nodes_from([f'MC{mc}' for mc in motility_clusters], bipartite=0)
B.add_nodes_from([f'BC{bc}' for bc in behavior_clusters], bipartite=1)

# Add edges with transition fractions
for _, row in transition_counts.iterrows():
    mc_node = f'MC{row["kmeans"]}'
    bc_node = f'BC{row["beh_kmeans"]}'
    B.add_edge(mc_node, bc_node, weight=row['fraction'])

# === Step 3: Layout ===
left_nodes = [f'MC{mc}' for mc in motility_clusters]
right_nodes = [f'BC{bc}' for bc in behavior_clusters]

# Top-to-bottom vertical ordering
left_y = [1 - (i / (len(left_nodes) - 1)) if len(left_nodes) > 1 else 0.5 for i in range(len(left_nodes))]
right_y = [1 - (i / (len(right_nodes) - 1)) if len(right_nodes) > 1 else 0.5 for i in range(len(right_nodes))]

pos = {}
for i, node in enumerate(left_nodes):
    pos[node] = (0, left_y[i])
for i, node in enumerate(right_nodes):
    pos[node] = (1, right_y[i])

# === Step 4: Draw Full Graph ===
fig, ax = plt.subplots(figsize=(15, 30))

edge_widths = [B[u][v]['weight'] * 50 for u, v in B.edges()]

nx.draw_networkx_nodes(B, pos, nodelist=left_nodes, node_color='lightblue', node_size=500, label='Motility Clusters')
nx.draw_networkx_nodes(B, pos, nodelist=right_nodes, node_color='lightgreen', node_size=500, label='Behavior Clusters')
nx.draw_networkx_edges(B, pos, width=edge_widths)
nx.draw_networkx_labels(B, pos, font_size=10)

plt.axis('off')
plt.savefig(path + 'multipartite graph/all.png', dpi=300, bbox_inches='tight')

# if not os.path.isdir(path + 'multipartite graph/svg/'):
#     os.makedirs(path + 'multipartite graph/svg/')
# fig.savefig(path + 'multipartite graph/svg/all.svg', bbox_inches='tight')
plt.close()
plt.clf()

# === Step 5: Draw Per-MC Highlighted Graphs ===
for cluster in left_nodes:
    fig, ax = plt.subplots(figsize=(15, 30))

    # Collect and rank BC connections by weight
    connected_edges = [(cluster, neighbor, B[cluster][neighbor]['weight']) for neighbor in B.neighbors(cluster)]
    sorted_edges = sorted(connected_edges, key=lambda x: x[2], reverse=True)

    # Top N
    n_top = 1
    top_edges = sorted_edges[:n_top]
    top_nodes = {v for u, v, _ in top_edges}

    # Top 50% (excluding top_n)
    n_top_half = max(len(sorted_edges) // 2, 1)
    top_half_edges = sorted_edges[n_top:n_top_half]
    top_half_nodes = {v for u, v, _ in top_half_edges}

    # Remaining connected
    remaining_nodes = set(v for u, v, _ in sorted_edges[n_top_half:])

    # Edge sets
    top_edge_set = set((u, v) if (u, v) in B.edges else (v, u) for u, v, _ in top_edges)
    top_half_edge_set = set((u, v) if (u, v) in B.edges else (v, u) for u, v, _ in top_half_edges)

    # Node coloring
    node_colors = []
    for node in B.nodes():
        if node == cluster:
            node_colors.append('#DC143C')  # Crimson
        elif node in top_nodes:
            node_colors.append('#8B0000')  # Dark red
        elif node in top_half_nodes:
            node_colors.append('#FA8072')  # Salmon
        elif node in remaining_nodes:
            node_colors.append('#FFDAB9')  # Peach
        else:
            node_colors.append('#D3D3D3')  # Gray

    # Edge coloring
    edge_colors = []
    edge_widths = []

    for u, v in B.edges():
        is_cluster_edge = (u == cluster or v == cluster)
        edge_key = (u, v) if (u, v) in B.edges else (v, u)

        if is_cluster_edge:
            if edge_key in top_edge_set:
                edge_colors.append('#8B0000')
                edge_widths.append(B[u][v]['weight'] * 40)
            elif edge_key in top_half_edge_set:
                edge_colors.append('#FA8072')
                edge_widths.append(B[u][v]['weight'] * 40)
            else:
                edge_colors.append('#FFDAB9')
                edge_widths.append(B[u][v]['weight'] * 40)
        else:
            edge_colors.append('#D3D3D3')
            edge_widths.append(1)

    # Draw per-MC graph
    nx.draw_networkx_nodes(B, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(B, pos, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_labels(B, pos, font_size=10)

    plt.axis('off')
    plt.savefig(path + f'multipartite graph/{cluster}.png', dpi=300, bbox_inches='tight')
    # fig.savefig(path + f'multipartite graph/svg/{cluster}.svg', bbox_inches='tight')
    plt.close()
    plt.clf()

adata_rna.raw.var_names




