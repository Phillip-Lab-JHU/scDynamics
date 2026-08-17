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
""" Generates Data for bulk RNA DEG """

import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
#import sc_toolbox
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gseapy as gp
from utils.draw_utils import *

#path = '/media/sf_RNAseq/scRNAseq_GCB-postGCB_EZH2-selected/'
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'
adata = sc.read_h5ad(path+'GCB_only.h5ad')

adata.layers['counts'] = adata.X # Save raw counts
sc.pp.normalize_total(adata, target_sum=10000)
sc.pp.log1p(adata) # change to log counts
adata.raw = adata  # Save pre normalized counts (without this rank_genes_groups have nan logFC)

sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
adata = adata[:, adata.var['highly_variable']]
sc.pp.scale(adata, max_value=10)  # standard scale (mean=0, variance=1)
sc.tl.pca(adata, n_comps=330, svd_solver='arpack')
print('total captured variance:', np.sum(adata.uns['pca']['variance_ratio']))
#sc.pl.pca_variance_ratio(adata, log=True, n_pcs=300)

sc.pp.neighbors(adata, n_pcs=330)
sc.tl.umap(adata)



#################################### Read custom signatures ####################################

import gseapy as gp

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'
sig_temp = gp.read_gmt(path=path + "GCsignatures.gmt")
sig_temp


def remove_nan(my_list):
    import math

    cleaned_list = [x for x in my_list if x != 'NA']  # Remove 'NA' values

    return cleaned_list


sig_unconverted = {}
for key in sig_temp:
    updated_key = key.replace('.', '_')  # Replace '.' with '_'
    sig_unconverted[updated_key] = remove_nan(sig_temp[key])

    print(key, updated_key, len(remove_nan(sig_temp[key])))
sig_unconverted

h2m = pd.read_csv(path+'h2m.csv')
h2m

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


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\GCB space\\'
#################################### Paint module scores in umap space ####################################
folder_name = 'Module score for custom signatures'

for sig_name in list(sig.keys()):
    sc.tl.score_genes(adata, sig[sig_name], score_name='%s_score' % sig_name)

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    sc.pl.umap(adata, color=['%s_score' % sig_name], frameon=False, ax=ax, cmap=plt.cm.get_cmap('viridis'), show=False, vmax=0.1)

    if not os.path.isdir(path + '%s/' % folder_name):
        os.makedirs(path + '%s/' % folder_name)

    fig.savefig(path + '%s/%s.png' % (folder_name, sig_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/%s/' % folder_name):
        os.makedirs(path + 'svg/%s/' % folder_name)

    fig.savefig(path + 'svg/%s/%s.svg' % (folder_name, sig_name), bbox_inches='tight')
    plt.close()
    plt.clf()


#################################### Paint module scores in umap space ####################################
custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\signatures\\'
df_genes1 = pd.read_excel(custom_sig_path+'morphological_plasticity_Tcells(Dekkers 2022).xlsx')
df_genes1 = df_genes1[df_genes1['Morpholocal plasticity-related']=='Yes'].reset_index(drop=True)

df_genes2 = pd.read_csv(custom_sig_path+'amoeboid_3d_genes_neutrophils(Belliveau 2023).csv')
df_genes2 = df_genes2[df_genes2['pvalue']<0.05].reset_index(drop=True)

df_genes3 = pd.read_csv(custom_sig_path+'chemotaxis_genes_neutrophils(Belliveau 2023).csv')
df_genes3 = df_genes3[df_genes3['pvalue']<0.05].reset_index(drop=True)

df_genes4 = pd.read_csv(custom_sig_path+'chemokinesis_genes_neutrophils(Belliveau 2023).csv')
df_genes4 = df_genes4[df_genes4['pvalue']<0.05].reset_index(drop=True)

sig_temp = {}
sig_temp['Morphological plasticity (Dekkers et. al., 2022)'] = df_genes1['Gene'].values
sig_temp['3D Amoeboidal (Belliveau et. al., 2023)'] = df_genes2['gene'].values
sig_temp['Chemotaxis (Belliveau et. al., 2023)'] = df_genes3['gene'].values
sig_temp['Chemokinesis (Belliveau et. al., 2023)'] = df_genes4['gene'].values

sig = {}
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
    sig[term] = new_genes


sig['FDC interaction'] = ['Bcr', 'Tnfrsf13c', 'Lta', 'Ltb', 'Itgb2', 'Itga4']
sig['Tfh interaction'] = ['Ciita', 'Cd40', 'Icam1', 'Icosl']

custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\signatures\\'
df_mot_associated_genes = pd.read_csv(custom_sig_path+'Motility-Associated_Genes.csv')
valid_genes = [g for g in df_mot_associated_genes["Gene"].values if g in adata.var_names]
sig['Motility associated genes'] = df_mot_associated_genes["Gene"].values

folder_name = 'Module score for other signatures'

for sig_name in list(sig.keys()):
    sc.tl.score_genes(adata, sig[sig_name], score_name='%s_score' % sig_name)

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    sc.pl.umap(adata, color=['%s_score' % sig_name], frameon=False, ax=ax, cmap=plt.cm.get_cmap('viridis'), show=False)

    if not os.path.isdir(path + '%s/' % folder_name):
        os.makedirs(path + '%s/' % folder_name)

    fig.savefig(path + '%s/%s.png' % (folder_name, sig_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/%s/' % folder_name):
        os.makedirs(path + 'svg/%s/' % folder_name)

    fig.savefig(path + 'svg/%s/%s.svg' % (folder_name, sig_name), bbox_inches='tight')
    plt.close()
    plt.clf()


#################################### RNA space with leiden clustering ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\GCB space\\'
for resolution in [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.35, 1.4, 1.5]:
    sc.tl.leiden(adata, resolution=resolution)

    color_list = ('#F06293', '#AF79C4', '#5B6BBF', '#FFC11C', '#D1917C', '#4DB6AC', '#00BCD4', '#A1877E', '#8F96A8', '#B02E8B', '#EB974D','#BCBCBC',)
    cmap=ListedColormap(color_list)
    file_name = 'RNA space type'

    df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
    df['leiden'] = adata.obs['leiden'].values

    draw_umap_space(df, path, file_name='leiden clusters/space_leiden_%s'%resolution, condition_name='leiden', label_name='leiden',
                    colors=color_list, x_name='UMAP1', y_name='UMAP2', dot_size=0.07)

sc.tl.leiden(adata, resolution=1.4)  # 1.5
cell_counts = adata.obs['leiden'].value_counts().sort_index()
print(cell_counts)

df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
df['leiden'] = adata.obs['leiden'].values
color_list = ('#F06293', '#AF79C4', '#5B6BBF', '#FFC11C', '#D1917C', '#4DB6AC', '#00BCD4', '#A1877E', '#8F96A8', '#B02E8B', '#EB974D','#BCBCBC',)
draw_umap_space(df, path, file_name='space_leiden', condition_name='leiden', label_name='leiden',
                colors=color_list, x_name='UMAP1', y_name='UMAP2', dot_size=0.07)
#################################### Manually annotate cluster by signatures ####################################
adata.obs['cluster'] = adata.obs['leiden']

sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", key_added="dea_leiden")
sc.tl.filter_rank_genes_groups(adata,min_in_group_fraction=0.1, max_out_group_fraction=0.1, key="dea_leiden",key_added="dea_leiden_filtered",)

fig, ax = plt.subplots(figsize=(20, 5), constrained_layout=True)
sc.pl.rank_genes_groups_dotplot(adata, groupby="leiden", standard_scale="var", n_genes=10, key="dea_leiden", ax=ax, show=False)

fig.savefig(path + 'leiden_dot_plot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/leiden_dot_plot.svg', bbox_inches='tight')
plt.close()
plt.clf()

fig, ax = plt.subplots(figsize=(20, 5), constrained_layout=True)
sc.pl.rank_genes_groups_dotplot(adata,groupby="leiden", standard_scale="var", n_genes=10,key="dea_leiden_filtered", ax=ax,show=False)
fig.savefig(path + 'leiden_dot_plot_filtered.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/leiden_dot_plot_filtered.svg', bbox_inches='tight')
plt.close()
plt.clf()

#################################### DEG of RNA clusters ####################################
result = adata.uns["dea_leiden"]
groups = result["names"].dtype.names
groups
degs = pd.DataFrame(
    {group + '_' + key: result[key][group]
    for group in groups for key in ['names','scores', 'pvals','pvals_adj','logfoldchanges']})

# cluster_signatures = { f"cluster_{group}": list(degs.loc[degs[f"{group}_pvals_adj"] < 0.05, f"{group}_names"])for group in groups }

top_n = 100
cluster_signatures = { f"cluster_{group}": list(degs[f"{group}_names"][:top_n]) for group in groups }

[print(key, len(cluster_signatures[key])) for key in cluster_signatures]


#################################### Manually annotate cluster by signatures (Holmes et. al., 2020) ####################################
custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\signatures\\'
lz_s_g2_m = pd.read_excel(custom_sig_path + 'LZ_subclusters(Holmes 2020).xlsx', sheet_name=1, skiprows=2)  # second tab
lz_g0_g1 = pd.read_excel(custom_sig_path + 'LZ_subclusters(Holmes 2020).xlsx', sheet_name=2, skiprows=2)  # third tab
dz_s_g2_m = pd.read_excel(custom_sig_path + 'DZ_subclusters(Holmes 2020).xlsx', sheet_name=1, skiprows=2)  # second tab
dz_g0_g1 = pd.read_excel(custom_sig_path + 'DZ_subclusters(Holmes 2020).xlsx', sheet_name=2, skiprows=2)  # third tab

# lz_gene_signatures_temp = {}
#
# for col in lz_s_g2_m.columns:
#     if 'log2 fold change' in col:
#         cluster = col.replace(' log2 fold change', '')
#         q_col = cluster + ' q'
#
#         if q_col in lz_s_g2_m.columns:
#             # Filter based on significance threshold
#             filtered = lz_s_g2_m[(lz_s_g2_m[col].abs() > 0) & (lz_s_g2_m[q_col] < 0.05)]
#
#             gene_list = filtered['Gene'].tolist()
#             lz_gene_signatures_temp[cluster] = gene_list
# for col in lz_g0_g1.columns:
#     if 'log2 fold change' in col:
#         cluster = col.replace(' log2 fold change', '')
#         q_col = cluster + ' q'
#
#         if q_col in lz_g0_g1.columns:
#             # Filter based on significance threshold
#             filtered = lz_g0_g1[(lz_g0_g1[col].abs() > 0) & (lz_g0_g1[q_col] < 0.05)]
#             gene_list = filtered['Gene'].tolist()
#             lz_gene_signatures_temp[cluster] = gene_list

# dz_gene_signatures_temp = {}
#
# for col in dz_s_g2_m.columns:
#     if 'log2 fold change' in col:
#         cluster = col.replace(' log2 fold change', '')
#         q_col = cluster + ' q'
#
#         if q_col in dz_s_g2_m.columns:
#             # Filter based on significance threshold
#             filtered = dz_s_g2_m[(dz_s_g2_m[col].abs() > 0) & (dz_s_g2_m[q_col] < 0.05)]
#             gene_list = filtered['Gene'].tolist()
#             dz_gene_signatures_temp[cluster] = gene_list
# for col in dz_g0_g1.columns:
#     if 'log2 fold change' in col:
#         cluster = col.replace(' log2 fold change', '')
#         q_col = cluster + ' q'
#
#         if q_col in dz_g0_g1.columns:
#             # Filter based on significance threshold
#             filtered = dz_g0_g1[(dz_g0_g1[col].abs() > 0) & (dz_g0_g1[q_col] < 0.05)]
#             gene_list = filtered['Gene'].tolist()
#             dz_gene_signatures_temp[cluster] = gene_list


lz_gene_signatures_temp = {}
for col in lz_s_g2_m.columns:
    if 'log2 fold change' in col:
        cluster = col.replace(' log2 fold change', '')
        avg_expr_col = f"{cluster} avg log2(tpm + 1)"
        logfc_col = f"{cluster} log2 fold change"
        q_col = f"{cluster} q"

        # Filter and compute proxy score
        sig_df = lz_s_g2_m[lz_s_g2_m[q_col] < 0.05].copy()
        sig_df["proxy_score"] = sig_df[logfc_col] * sig_df[avg_expr_col]

        # Sort and select top genes
        top_genes = sig_df.sort_values("proxy_score", ascending=False).head(top_n)["Gene"].tolist()
        lz_gene_signatures_temp[cluster] = top_genes
for col in lz_g0_g1.columns:
    if 'log2 fold change' in col:
        cluster = col.replace(' log2 fold change', '')
        avg_expr_col = f"{cluster} avg log2(tpm + 1)"
        logfc_col = f"{cluster} log2 fold change"
        q_col = f"{cluster} q"

        # Filter and compute proxy score
        sig_df = lz_g0_g1[lz_g0_g1[q_col] < 0.05].copy()
        sig_df["proxy_score"] = sig_df[logfc_col] * sig_df[avg_expr_col]
        # significantly upregulated genes (logFC) + abundantly expressed in the cluster (avg log2(tpm + 1)).

        # Sort and select top genes
        top_genes = sig_df.sort_values("proxy_score", ascending=False).head(top_n)["Gene"].tolist()
        lz_gene_signatures_temp[cluster] = top_genes

dz_gene_signatures_temp = {}
for col in dz_s_g2_m.columns:
    if 'log2 fold change' in col:
        cluster = col.replace(' log2 fold change', '')
        avg_expr_col = f"{cluster} avg log2(tpm + 1)"
        logfc_col = f"{cluster} log2 fold change"
        q_col = f"{cluster} q"

        # Filter and compute proxy score
        sig_df = dz_s_g2_m[dz_s_g2_m[q_col] < 0.05].copy()
        sig_df["proxy_score"] = sig_df[logfc_col] * sig_df[avg_expr_col]

        # Sort and select top genes
        top_genes = sig_df.sort_values("proxy_score", ascending=False).head(top_n)["Gene"].tolist()
        dz_gene_signatures_temp[cluster] = top_genes

for col in dz_g0_g1.columns:
    if 'log2 fold change' in col:
        cluster = col.replace(' log2 fold change', '')
        avg_expr_col = f"{cluster} avg log2(tpm + 1)"
        logfc_col = f"{cluster} log2 fold change"
        q_col = f"{cluster} q"

        # Filter and compute proxy score
        sig_df = dz_g0_g1[dz_g0_g1[q_col] < 0.05].copy()
        sig_df["proxy_score"] = sig_df[logfc_col] * sig_df[avg_expr_col]
        # significantly upregulated genes (logFC) + abundantly expressed in the cluster (avg log2(tpm + 1)).

        # Sort and select top genes
        top_genes = sig_df.sort_values("proxy_score", ascending=False).head(top_n)["Gene"].tolist()
        dz_gene_signatures_temp[cluster] = top_genes

[print(key, len(lz_gene_signatures_temp[key])) for key in lz_gene_signatures_temp]
[print(key, len(dz_gene_signatures_temp[key])) for key in dz_gene_signatures_temp]



h2m = pd.read_csv(r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'+'h2m.csv')
h2m_dict = {}
for i, row in h2m.loc[:,["external_gene_name", "mmusculus_homolog_associated_gene_name"]].iterrows():
    if row.isna().any(): continue
    h2m_dict[row['external_gene_name']] = row["mmusculus_homolog_associated_gene_name"]


lz_gene_signatures = {}
for term, genes in lz_gene_signatures_temp.items():
    count=0
    new_genes = []
    for gene in genes:
        if gene in h2m_dict:
            new_genes.append(h2m_dict[gene])
            count = count + 1
        # else:
        #     new_genes.append(gene)
    print( term, ': ', '%s/%s genes converted'%(count, len(genes)) )
    lz_gene_signatures['LZ_'+term] = new_genes


dz_gene_signatures = {}
for term, genes in dz_gene_signatures_temp.items():
    count=0
    new_genes = []
    for gene in genes:
        if gene in h2m_dict:
            new_genes.append(h2m_dict[gene])
            count = count + 1
        # else:
        #     new_genes.append(gene)
    print( term, ': ', '%s/%s genes converted'%(count, len(genes)) )
    dz_gene_signatures['DZ_'+term] = new_genes

custom_sig = {**cluster_signatures, **lz_gene_signatures, **dz_gene_signatures}

##################### Draw network graph for community detection #####################
draw_graph_network(custom_sig, path, file_name='gene overlap', sample_n=20, resolution=1.6, figsize=(10,10), inter_spacing=2.5, intra_spacing=1.5)

##################### Draw bipartite graph for semantic meaning of each cluster #####################
custom_signatures = {**lz_gene_signatures, **dz_gene_signatures}
import networkx as nx
fig, ax = plt.subplots(figsize=(15,30))
# Initialize the bipartite graph
B = nx.Graph()

# Add nodes with bipartite attribute
B.add_nodes_from(cluster_signatures.keys(), bipartite=0)
B.add_nodes_from(custom_signatures.keys(), bipartite=1)

# Compute IoU scores and store them temporarily
edge_iou = {}
for cluster, genes_cluster in cluster_signatures.items():
    for custom, genes_custom in custom_signatures.items():
        intersection = set(genes_cluster) & set(genes_custom)
        union = set(genes_cluster) | set(genes_custom)
        if union:
            iou = len(intersection) / len(union)
            if iou > 0:
                edge_iou[(cluster, custom)] = iou

# Normalize IoU scores per cluster
from collections import defaultdict

cluster_totals = defaultdict(float)
for (cluster, _), iou in edge_iou.items():
    cluster_totals[cluster] += iou

normalized_weights = {}
for (cluster, custom), iou in edge_iou.items():
    normalized_iou = iou / cluster_totals[cluster]
    normalized_weights[(cluster, custom)] = normalized_iou
    B.add_edge(cluster, custom, weight=normalized_iou)

# Define positions for a clear layout
left_nodes = list(cluster_signatures.keys())
right_nodes = list(custom_signatures.keys())

# Calculate y-positions for top-to-bottom ordering
left_y = [1 - (i / (len(left_nodes) - 1)) if len(left_nodes) > 1 else 0.5 for i in range(len(left_nodes))]
right_y = [1 - (i / (len(right_nodes) - 1)) if len(right_nodes) > 1 else 0.5 for i in range(len(right_nodes))]

# Assign positions: left nodes at x=0, right nodes at x=1
pos = {}
for i, node in enumerate(left_nodes):
    pos[node] = (0, left_y[i])
for i, node in enumerate(right_nodes):
    pos[node] = (1, right_y[i])

# Draw nodes
nx.draw_networkx_nodes(B, pos, nodelist=left_nodes, node_color='lightblue', node_size=500, label='Clusters')
nx.draw_networkx_nodes(B, pos, nodelist=right_nodes, node_color='lightgreen', node_size=500, label='Custom Signatures')

# Draw edges with widths proportional to normalized IoU
edges = B.edges(data=True)
edge_widths = [data['weight'] * 50 for _, _, data in edges]  # Scale for visibility
nx.draw_networkx_edges(B, pos, edgelist=edges, width=edge_widths)

# Draw labels
nx.draw_networkx_labels(B, pos, font_size=10)

# Display the plot
plt.axis('off')
#plt.legend()
#plt.title('Bipartite Graph: Clusters and Custom Gene Signatures with Normalized IoU Weights')
plt.savefig(path+'bipartite graph/all.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'bipartite graph/svg/'):
    os.makedirs(path + 'bipartite graph/svg/')
fig.savefig(path + 'bipartite graph/svg/all.svg', bbox_inches='tight')
plt.close()
plt.clf()


for cluster in left_nodes:
    fig, ax = plt.subplots(figsize=(15,30))

    # Get all edges from the current cluster
    connected_edges = [(cluster, neighbor, B[cluster][neighbor]['weight']) for neighbor in B.neighbors(cluster)]

    # Sort by IoU (descending)
    sorted_edges = sorted(connected_edges, key=lambda x: x[2], reverse=True)

    # Top 5
    n_top = 3
    top_edges = sorted_edges[:n_top]
    top_nodes = {v for u, v, _ in top_edges}

    # Top 50% (excluding top 5)
    n_top_half = max(len(sorted_edges) // 2, 1)
    top_half_edges = sorted_edges[n_top:n_top_half]
    top_half_nodes = {v for u, v, _ in top_half_edges}

    # All other connected nodes
    remaining_nodes = set(v for u, v, _ in sorted_edges[n_top_half:])

    # Set-based lookup
    top_edge_set = set((u, v) for u, v, _ in top_edges)
    top_half_edge_set = set((u, v) for u, v, _ in top_half_edges)

    # Node colors
    node_colors = []
    for node in B.nodes():
        if node == cluster:
            node_colors.append('#DC143C')  # Current cluster node
        elif node in top_nodes:
            node_colors.append('#8B0000') # Darkred
        elif node in top_half_nodes:
            node_colors.append('#FA8072') # Pink
        elif node in remaining_nodes:
            node_colors.append('#FFDAB9') # beige
        else:
            node_colors.append('#D3D3D3') # lightgray

    # Edge colors and widths
    edge_colors = []
    edge_widths = []

    for u, v in B.edges():
        is_cluster_edge = (u == cluster or v == cluster)
        key = (u, v) if (u, v) in top_edge_set or (u, v) in top_half_edge_set else (v, u)

        if is_cluster_edge:
            if key in top_edge_set:
                edge_colors.append('#8B0000')
                edge_widths.append(B[u][v]['weight'] * 40)
            elif key in top_half_edge_set:
                edge_colors.append('#FA8072')
                edge_widths.append(B[u][v]['weight'] * 40)
            else:
                edge_colors.append('#FFDAB9')
                edge_widths.append(B[u][v]['weight'] * 40)
        else:
            edge_colors.append('#D3D3D3')
            edge_widths.append(1)

    # Draw everything
    nx.draw_networkx_nodes(B, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(B, pos, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_labels(B, pos, font_size=10)

    # Save the figure
    plt.axis('off')
    plt.savefig(path + f'bipartite graph/{cluster}.png', dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'bipartite graph/svg/'):
        os.makedirs(path + 'bipartite graph/svg/')
    fig.savefig(path + f'bipartite graph/svg/{cluster}.svg', bbox_inches='tight')
    plt.close()
    plt.clf()



# from upsetplot import UpSet
#
# # Combine all sets
# all_gene_sets = {**cluster_signatures, **lz_gene_signatures, **dz_gene_signatures}
#
# # Restrict to top 20 sets by gene count
# top_sets = dict(sorted(all_gene_sets.items(), key=lambda x: len(x[1]), reverse=True))
# # n_genes = 30
# # top_sets = dict(sorted(all_gene_sets.items(), key=lambda x: len(x[1]), reverse=True)[:n_genes])
#
# # Get all genes from those sets
# all_genes = set().union(*top_sets.values())
#
# # Binary matrix
# binary_matrix = pd.DataFrame(index=sorted(all_genes))
# for name, genes in top_sets.items():
#     binary_matrix[name] = binary_matrix.index.isin(genes).astype(int)
#
# # Collapse to Series with MultiIndex
# grouped = binary_matrix.groupby(list(binary_matrix.columns)).size()
# upset_series = grouped.sort_values(ascending=False)
# #upset_series = grouped.sort_values(ascending=False).head(n_genes)
# # Plot with fixed figure size and orientation
# fig = plt.figure(figsize=(10, 6))
# UpSet(upset_series, orientation='horizontal').plot(fig=fig)
# plt.tight_layout()
# plt.savefig(path+'upset test.png', dpi=300, bbox_inches='tight')
# plt.close()
# plt.clf()
#################################### Manually annotate cluster by signatures ####################################
# sig_unconverted = {}
# sig_unconverted['Holmes_2020 BCR signaling'] = ['CD79A', 'CD79B', 'CD19', 'LYN', 'BLNK', 'BTK', 'CD72', 'CD22', 'PTPN6', 'SLA', 'FCRL2']
# sig_unconverted['Holmes_2020 CD40 signaling'] = ['CD40', 'TRAF1', 'ICAM1', 'CD80', 'CD86', 'CFLAR', 'BCL2A1', 'BCL2L1', 'MIR155HG', 'EBI3', 'CD58', 'CCL22', 'STAT5A']
# sig_unconverted['Holmes_2020 NF-kappaB'] = ['NFKB1', 'REL', 'RELB', 'NFKB2', 'NFKBIA', 'NFKBIE', 'MYD88', 'TNFAIP3']
# sig_unconverted['Holmes_2020 MYC'] = ['MYC', 'BATF', 'GPR183', 'CD44']
# sig_unconverted['Holmes_2020 DZ'] = ['CXCR4', 'FOXP1', 'CD27']
# sig_unconverted['Holmes_2020 PreM'] = ['BACH2', 'BANK1', 'RASGRP2', 'CCR6', 'CELF2']
#
# h2m = pd.read_csv(r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'+'h2m.csv')
# h2m_dict = {}
# for i, row in h2m.loc[:,["external_gene_name", "mmusculus_homolog_associated_gene_name"]].iterrows():
#     if row.isna().any(): continue
#     h2m_dict[row['external_gene_name']] = row["mmusculus_homolog_associated_gene_name"]
#
#
# sig = {}
# for term, genes in sig_unconverted.items():
#     count=0
#     new_genes = []
#     for gene in genes:
#         if gene in h2m_dict:
#             new_genes.append(h2m_dict[gene])
#             count = count + 1
#         else:
#             new_genes.append(gene)
#     print( term, ': ', '%s/%s genes converted'%(count, len(genes)) )
#     sig[term] = new_genes
#
# valid_genes = [g for g in flatten_nested_dict(sig) if g in adata.var_names]
# fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
# sc.pl.matrixplot(
#     adata,
#     var_names=valid_genes,
#     groupby='leiden',
#     use_raw=True,
#     standard_scale='var',
#     cmap='Reds',
#     swap_axes=True,
#     dendrogram=True,
#     layer=None,  # or e.g. 'log1p'
#     ax=ax,
#     show=False
# )
# fig.savefig(path + 'matrix plot.png', dpi=300, bbox_inches='tight')
# if not os.path.isdir(path + 'svg/'):
#     os.makedirs(path + 'svg/')
# fig.savefig(path + 'svg/matrix plot.svg', bbox_inches='tight')
# plt.close()
# plt.clf()





# Step 1: Subset adata to the genes you want
valid_genes = [g for g in valid_genes if g in adata.var_names]  # ensure genes exist
adata_subset = adata[:, valid_genes].copy()

# Step 2: Extract expression matrix from .raw if available, else from .X
if adata.raw is not None:
    X = adata_subset.raw[:, valid_genes].X.toarray()
else:
    X = adata_subset.X.toarray()

# Step 3: Z-score expression matrix across cells (axis=0)
X_z = (X - X.mean(axis=0)) / X.std(axis=0)

# Step 4: Assign z-scored matrix back to .X of the same shape
adata_subset.X = X_z

# Now plot without using layers
sc.pl.matrixplot(
    adata_subset,
    var_names=valid_genes,
    groupby='leiden',
    use_raw=False,
    cmap="bwr",
    vmin=-2, vmax=2,
    standard_scale=None,
    swap_axes=True,
    dendrogram=True,
    colorbar_title="Z-score",
)



#################################### Manually annotate cluster by signatures ####################################
custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\signatures\\'
df_mot_associated_genes = pd.read_csv(custom_sig_path+'Motility-Associated_Genes.csv')
valid_genes = [g for g in df_mot_associated_genes["Gene"].values if g in adata.var_names]

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
sc.pl.matrixplot(
    adata,
    var_names=valid_genes,
    groupby='leiden',
    use_raw=True,
    standard_scale='var',
    cmap='Reds',
    swap_axes=True,
    dendrogram=True,
    layer=None,  # or e.g. 'log1p'
    ax=ax,
    show=False
)
fig.savefig(path + 'matrix plot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/matrix plot.svg', bbox_inches='tight')
plt.close()
plt.clf()



fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
sc.pl.heatmap(
    adata,
    var_names=valid_genes,          # genes filtered as above
    groupby='leiden',
    use_raw=True,
    swap_axes=True,                 # so genes are clustered vertically
    standard_scale='var',           # normalize gene expression
    dendrogram=True,                # enables both row and column clustering
    show=False,
)
fig = plt.gcf()
fig.savefig(path + 'matrix plot.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/matrix plot.svg', bbox_inches='tight')
plt.close()
plt.clf()



#################################### Give leiden clusters names ####################################
cmap = plt.get_cmap("tab20")
comm_list = [0, 1, 2, 3, 4, 5]
color_map = {comm: cmap(j / max(1, len(comm_list)-1)) for j, comm in enumerate(comm_list)}

#mapping = {'0': 'Memory', '1': 'LZ', '2': 'DZ-LZ', '3': 'DZ', '4': 'DZ', '5': 'DZ-LZ', '6': 'LZ', '7': 'DZ', '8': 'DZ-LZ', '9':'Recycle', '10':'NA', '11':'NA'}
mapping = {'0': 'LZ', '1': 'Memory', '2': 'DZ-LZ', '3': 'DZ', '4': 'DZ-LZ', '5': 'LZ', '6': 'DZ', '7': 'LZ', '8': 'DZ-LZ', '9':'LZ'}

adata.obs['cluster'].replace(mapping, inplace=True)
cluster = adata.obs['cluster']
leiden = adata.obs['leiden']
#color_list = ('#F06293', '#AF79C4', '#5B6BBF', '#FFC11C', '#D1917C', '#4DB6AC', '#00BCD4', '#A1877E', '#8F96A8', '#B02E8B', '#EB974D','#BCBCBC',)
color_list = [(0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0), (0.6196078431372549, 0.8549019607843137, 0.8980392156862745, 1.0),
              (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)]
df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
df['cluster'] = adata.obs['cluster'].values

draw_umap_space(df, path, file_name='space_cluster', condition_name='cluster', label_name='cluster',
                colors=color_list, x_name='UMAP1', y_name='UMAP2', dot_size=0.07)

draw_umap_space(df[(df['cluster']!='Memory')], path, file_name='space_cluster_filtered', condition_name='cluster', label_name='cluster',
                colors=color_list, x_name='UMAP1', y_name='UMAP2', dot_size=0.07)

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'
adata_new = sc.read_h5ad(path+'GCB_only.h5ad')
adata_new.obs['cluster'] = cluster
adata_new.obs['leiden'] = leiden
adata_new.write_h5ad(path+'GCB_only_cluster_final.h5ad')


