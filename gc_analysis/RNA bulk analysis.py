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

df_cb = pd.read_excel(path+'Ezh2_DEG.xlsx', sheet_name='CB')
df_cc = pd.read_excel(path+'Ezh2_DEG.xlsx', sheet_name='CC')

df_cb['Adj_Logp'] = -np.log(df_cb['padj'])
df_cc['Adj_Logp'] = -np.log(df_cc['padj'])


df_cb_nan_rem = df_cb[~df_cb.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf
df_cc_nan_rem = df_cc[~df_cc.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf
df_cc_nan_rem

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\Bulk GCB\\'

#################################### Volcano plot ####################################

df_p_cb = draw_volcano_plot(df_cb_nan_rem, path, file_name='CB_volcano_texted', z_thresh=1, dot_size=5, p_thresh=-np.log(0.05), # -np.log(0.0001)
                  z_name='log2FoldChange', p_name='Adj_Logp', feature_name='Gene', text=True, figsize=(6,6))
df_p_cc = draw_volcano_plot(df_cc_nan_rem, path, file_name='CC_volcano_texted', z_thresh=1, dot_size=5, p_thresh=-np.log(0.05), # -np.log(0.0001)
                  z_name='log2FoldChange', p_name='Adj_Logp', feature_name='Gene', text=True, figsize=(6,6))

# cb_degs = df_p_cb[df_p_cb['color']=='Change'].reset_index(drop=True)
# cc_degs = df_p_cc[df_p_cc['color']=='Change'].reset_index(drop=True)



cb_degs = df_cb_nan_rem[(df_cb_nan_rem['padj'] < 0.05)&(abs(df_cb_nan_rem['log2FoldChange']) > 1)]
cc_degs = df_cc_nan_rem[(df_cc_nan_rem['padj'] < 0.05)&(abs(df_cc_nan_rem['log2FoldChange']) > 1)]

cb_degs_up = cb_degs[cb_degs['log2FoldChange'] > 0]
cb_degs_down = cb_degs[cb_degs['log2FoldChange'] < 0]

cc_degs_up = cc_degs[cc_degs['log2FoldChange'] > 0]
cc_degs_down = cc_degs[cc_degs['log2FoldChange'] < 0]

cb_degs.to_csv(path+'bulk_cb_deg.csv', index=False)
cc_degs.to_csv(path+'bulk_cc_deg.csv', index=False)

######### Draw venn diagram of upregulated genes ########
set_a = set(cb_degs_up['Gene'])
set_b = set(cc_degs_up['Gene'])

overlap = sorted(set_a & set_b)
overlap_text = '\n'.join(overlap)

from matplotlib_venn import venn2
fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
v = venn2([set_a, set_b], set_labels=('Centroblast', 'Centrocyte'), alpha=1)

v.get_patch_by_id('10').set_facecolor('white')
v.get_patch_by_id('01').set_facecolor('white')
v.get_patch_by_id('11').set_facecolor('white')

v.get_patch_by_id('10').set_edgecolor('#5B6BBF')
v.get_patch_by_id('01').set_edgecolor('#F06293')

v.get_patch_by_id('10').set_linewidth(3)
v.get_patch_by_id('01').set_linewidth(3)
v.get_patch_by_id('11').set_linewidth(3)

# Customize subset labels (counts)
for label in v.subset_labels:
    if label:
        label.set_fontname('Arial')  # Set font to Arial
        label.set_fontsize(14)       # Set font size

# Customize set labels ('Group A', 'Group B')
for label in v.set_labels:
    if label:
        label.set_fontname('Arial')  # Set font to Arial
        label.set_fontsize(14)       # Set font size



# intersection_label = v.get_label_by_id('11')
# if intersection_label and overlap:
#     x, y = intersection_label.get_position()
#     # Define position for the annotation text
#     text_x = x + 1.0
#     text_y = y
#
#     # Split the overlap list into two columns
#     mid = int(np.ceil(len(overlap) / 2))
#     col1 = overlap[:mid]
#     col2 = overlap[mid:]
#
#     # Determine the maximum number of rows
#     max_rows = max(len(col1), len(col2))
#
#     # Create the annotation text with two columns
#     annotation_text = ''
#     for i in range(max_rows):
#         gene1 = col1[i] if i < len(col1) else ''
#         gene2 = col2[i] if i < len(col2) else ''
#         annotation_text += f'{gene1:<40} {gene2}\n'
#
#     # Draw an arrow from the intersection to the annotation
#     ax.annotate(annotation_text.strip(), xy=(x, y), xytext=(text_x, text_y),
#                 #arrowprops=dict(arrowstyle='->', color='black'),
#                 fontsize=12, fontname='Arial', ha='left', va='center',
#                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=1))
intersection_label = v.get_label_by_id('11')
if intersection_label and overlap:
    x, y = intersection_label.get_position()
    # Define position for the annotation text
    text_x = x + 1.0
    text_y = y

    # Define the number of columns
    num_columns = 4  # You can change this value as needed

    # Calculate the number of rows needed
    num_rows = int(np.ceil(len(overlap) / num_columns))

    # Create the annotation text with multiple columns
    annotation_text = ''
    for i in range(num_rows):
        row_items = []
        for j in range(num_columns):
            index = i + j * num_rows
            if index < len(overlap):
                row_items.append(f'{overlap[index]:<15}')
            else:
                row_items.append(' ' * 15)
        annotation_text += ' '.join(row_items) + '\n'

    # Draw an arrow from the intersection to the annotation
    ax.annotate(annotation_text.strip(), xy=(x, y), xytext=(text_x, text_y),
                #arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=8, fontname='Arial', ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=1))

plt.savefig(path+'up_venn_diagram.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')

fig.savefig(path + 'svg/up_venn_diagram.svg', bbox_inches='tight')

######### Draw venn diagram of downregulated genes ########
set_a = set(cb_degs_down['Gene'])
set_b = set(cc_degs_down['Gene'])

overlap = sorted(set_a & set_b)
overlap_text = '\n'.join(overlap)


from matplotlib_venn import venn2
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
v = venn2([set_a, set_b], set_labels=('Centroblast', 'Centrocyte'), alpha=1)

v.get_patch_by_id('10').set_facecolor('white')
v.get_patch_by_id('01').set_facecolor('white')
v.get_patch_by_id('11').set_facecolor('white')

v.get_patch_by_id('10').set_edgecolor('#5B6BBF')
v.get_patch_by_id('01').set_edgecolor('#F06293')

v.get_patch_by_id('10').set_linewidth(3)
v.get_patch_by_id('01').set_linewidth(3)
v.get_patch_by_id('11').set_linewidth(3)

# Customize subset labels (counts)
for label in v.subset_labels:
    if label:
        label.set_fontname('Arial')  # Set font to Arial
        label.set_fontsize(14)       # Set font size

# Customize set labels ('Group A', 'Group B')
for label in v.set_labels:
    if label:
        label.set_fontname('Arial')  # Set font to Arial
        label.set_fontsize(14)       # Set font size


intersection_label = v.get_label_by_id('11')
if intersection_label and overlap:
    x, y = intersection_label.get_position()
    # Define position for the annotation text
    text_x = x + 1.0
    text_y = y

    # Define the number of columns
    num_columns = 10  # You can change this value as needed

    # Calculate the number of rows needed
    num_rows = int(np.ceil(len(overlap) / num_columns))

    # Create the annotation text with multiple columns
    annotation_text = ''
    for i in range(num_rows):
        row_items = []
        for j in range(num_columns):
            index = i + j * num_rows
            if index < len(overlap):
                row_items.append(f'{overlap[index]:<15}')
            else:
                row_items.append(' ' * 15)
        annotation_text += ' '.join(row_items) + '\n'

    # Draw an arrow from the intersection to the annotation
    ax.annotate(annotation_text.strip(), xy=(x, y), xytext=(text_x, text_y),
                #arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=6, fontname='Arial', ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=1))


plt.savefig(path+'down_venn_diagram.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/down_venn_diagram.svg', bbox_inches='tight')


#################################### log2FC for motility associated genes ####################################
custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\signatures\\'
df_mot_associated_genes = pd.read_csv(custom_sig_path+'Motility-Associated_Genes.csv')
motility_genes = set(df_mot_associated_genes["Gene"])

cb_degs = df_cb_nan_rem[(df_cb_nan_rem['padj'] < 0.05)&(abs(df_cb_nan_rem['log2FoldChange']) > 1)]
cc_degs = df_cc_nan_rem[(df_cc_nan_rem['padj'] < 0.05)&(abs(df_cc_nan_rem['log2FoldChange']) > 1)]

df = pd.concat([cb_degs, cc_degs], ignore_index=True)
df_filtered = df[df["Gene"].isin(motility_genes)]

df_filtered["mean_log2FC"] = df_filtered.groupby("Gene")["log2FoldChange"].transform("mean")
pivot = df_filtered.pivot_table(index="Gene", columns="CellType", values="log2FoldChange")

# STEP 3: Sort genes by average log2FoldChange
mean_fc = pivot.mean(axis=1)
pivot_sorted = pivot.loc[mean_fc.sort_values(ascending=False).index]

# STEP 4: Plot with square cells
num_genes = pivot_sorted.shape[0]
num_celltypes = pivot_sorted.shape[1]
cell_size = 0.6  # adjust for resolution
#figsize=(num_celltypes * cell_size + 2, num_genes * cell_size)

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)

figsize=(num_genes * cell_size, num_celltypes * cell_size + 2)
fig, ax = plt.subplots(figsize=figsize)
sns.heatmap(
    pivot_sorted.T,
    cmap=cmc.vik,
    annot=False,
    center=0,
    vmin=-3,
    vmax=3,
    linewidths=0.5,
    cbar_kws={"label": "log2 Fold Change"},
    square=True,
    ax=ax
)


plt.xticks(fontsize=14, rotation=35, rotation_mode='anchor', ha='right', color='0.2', weight='normal')
plt.yticks(fontsize=14, rotation=0, color='0.2', weight='normal')

plt.savefig(path+'motility_associated_genes.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/motility_associated_genes.svg', bbox_inches='tight')


#################################### log2FC for motility associated genes in other papers ####################################
custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\signatures\\'
df_genes1 = pd.read_excel(custom_sig_path+'morphological_plasticity_Tcells(Dekkers 2022).xlsx')
df_genes1 = df_genes1[df_genes1['Morpholocal plasticity-related']=='Yes'].reset_index(drop=True)

df_genes2 = pd.read_csv(custom_sig_path+'amoeboid_3d_genes_neutrophils(Belliveau 2023).csv')
df_genes2 = df_genes2[df_genes2['pvalue']<0.05].reset_index(drop=True)

df_genes3 = pd.read_csv(custom_sig_path+'chemotaxis_genes_neutrophils(Belliveau 2023).csv')
df_genes3 = df_genes3[df_genes3['pvalue']<0.05].reset_index(drop=True)

df_genes4 = pd.read_csv(custom_sig_path+'chemokinesis_genes_neutrophils(Belliveau 2023).csv')
df_genes4 = df_genes4[df_genes4['pvalue']<0.05].reset_index(drop=True)

sig_temp={}
sig_temp['Morphological plasticity (Dekkers et. al., 2022)'] = df_genes1['Gene'].values
sig_temp['3D Amoeboidal (Belliveau et. al., 2023)'] = df_genes2['gene'].values
sig_temp['Chemotaxis (Belliveau et. al., 2023)'] = df_genes3['gene'].values
sig_temp['Chemokinesis (Belliveau et. al., 2023)'] = df_genes4['gene'].values

h2m = pd.read_csv(r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'+'h2m.csv')
h2m_dict = {}
for i, row in h2m.loc[:,["external_gene_name", "mmusculus_homolog_associated_gene_name"]].iterrows():
    if row.isna().any(): continue
    h2m_dict[row['external_gene_name']] = row["mmusculus_homolog_associated_gene_name"]

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

gene_sets = {}
for term, genes in custom_sig.items():

    motility_genes = set(genes)
    if term == 'Morphological plasticity (Dekkers et. al., 2022)':
        z_thresh = 0.1
    elif term == '3D Amoeboidal (Belliveau et. al., 2023)':
        z_thresh = 0.5
    elif term == 'Chemotaxis (Belliveau et. al., 2023)':
        z_thresh = 1
    elif term == 'Chemokinesis (Belliveau et. al., 2023)':
        z_thresh = 0.7

    cb_degs = df_cb_nan_rem[(df_cb_nan_rem['padj'] < 0.05)&(abs(df_cb_nan_rem['log2FoldChange']) > z_thresh)]
    cc_degs = df_cc_nan_rem[(df_cc_nan_rem['padj'] < 0.05)&(abs(df_cc_nan_rem['log2FoldChange']) > z_thresh)]

    df = pd.concat([cb_degs, cc_degs], ignore_index=True)
    df_filtered = df[df["Gene"].isin(motility_genes)]

    df_filtered["mean_log2FC"] = df_filtered.groupby("Gene")["log2FoldChange"].transform("mean")
    pivot = df_filtered.pivot_table(index="Gene", columns="CellType", values="log2FoldChange")

    # STEP 3: Sort genes by average log2FoldChange
    mean_fc = pivot.mean(axis=1)
    pivot_sorted = pivot.loc[mean_fc.sort_values(ascending=False).index]
    pivot_sorted = pivot_sorted.dropna(axis=0, how='any')
    sig_genes = pivot_sorted.index.values
    gene_sets[term] = sig_genes
    # STEP 4: Plot with square cells
    num_genes = pivot_sorted.shape[0]
    num_celltypes = pivot_sorted.shape[1]
    cell_size = 0.6  # adjust for resolution
    #figsize=(num_celltypes * cell_size + 2, num_genes * cell_size)

    font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
    matplotlib.rc('font', **font)

    figsize=(num_genes * cell_size, num_celltypes * cell_size + 2)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_sorted.T,
        cmap=cmc.vik,
        annot=False,
        center=0,
        vmin=-3,
        vmax=3,
        linewidths=0.5,
        cbar_kws={"label": "log2 Fold Change"},
        square=True,
        ax=ax
    )


    plt.xticks(fontsize=14, rotation=35, rotation_mode='anchor', ha='right', color='0.2', weight='normal')
    plt.yticks(fontsize=14, rotation=0, color='0.2', weight='normal')

    plt.savefig(path+'%s_associated_genes.png'%term, dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/'):
        os.makedirs(path + 'svg/')
    fig.savefig(path + 'svg/%s_associated_genes.svg'%term, bbox_inches='tight')

#################################### Pathway association using graph networks ####################################
df = pd.read_excel(path+"both/significant pathway.xlsx")
df = df[df['Gene_set']!='gs_ind_0'].reset_index(drop=True)
df = df.dropna(subset=['Genes'])
sig_temp = {
    row['Term']: np.array(genes)
    for _, row in df.iterrows()
    if len(genes := row['Genes'].split(';')) >= 5
}
regulation = {
    row['Term']: row['UP_DW']
    for _, row in df.iterrows()
    if len(genes := row['Genes'].split(';')) >= 5
}

custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\signatures\\'
df_genes1 = pd.read_excel(custom_sig_path+'morphological_plasticity_Tcells(Dekkers 2022).xlsx')
df_genes1 = df_genes1[df_genes1['Morpholocal plasticity-related']=='Yes'].reset_index(drop=True)

df_genes2 = pd.read_csv(custom_sig_path+'amoeboid_3d_genes_neutrophils(Belliveau 2023).csv')
df_genes2 = df_genes2[df_genes2['pvalue']<0.05].reset_index(drop=True)

df_genes3 = pd.read_csv(custom_sig_path+'chemotaxis_genes_neutrophils(Belliveau 2023).csv')
df_genes3 = df_genes3[df_genes3['pvalue']<0.05].reset_index(drop=True)

df_genes4 = pd.read_csv(custom_sig_path+'chemokinesis_genes_neutrophils(Belliveau 2023).csv')
df_genes4 = df_genes4[df_genes4['pvalue']<0.05].reset_index(drop=True)

sig_temp['Morphological plasticity (Dekkers et. al., 2022)'] = df_genes1['Gene'].values
sig_temp['3D Amoeboidal (Belliveau et. al., 2023)'] = df_genes2['gene'].values
sig_temp['Chemotaxis (Belliveau et. al., 2023)'] = df_genes3['gene'].values
sig_temp['Chemokinesis (Belliveau et. al., 2023)'] = df_genes4['gene'].values


h2m = pd.read_csv(r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'+'h2m.csv')
h2m_dict = {}
for i, row in h2m.loc[:,["external_gene_name", "mmusculus_homolog_associated_gene_name"]].iterrows():
    if row.isna().any(): continue
    h2m_dict[row['external_gene_name']] = row["mmusculus_homolog_associated_gene_name"]

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


gene_sets = {}
for term, genes in custom_sig.items():

    motility_genes = set(genes)
    cb_degs = df_cb_nan_rem[(df_cb_nan_rem['padj'] < 0.05)&(abs(df_cb_nan_rem['log2FoldChange']) > 0)]
    cc_degs = df_cc_nan_rem[(df_cc_nan_rem['padj'] < 0.05)&(abs(df_cc_nan_rem['log2FoldChange']) > 0)]

    df = pd.concat([cb_degs, cc_degs], ignore_index=True)
    df_filtered = df[df["Gene"].isin(motility_genes)]

    df_filtered["mean_log2FC"] = df_filtered.groupby("Gene")["log2FoldChange"].transform("mean")
    pivot = df_filtered.pivot_table(index="Gene", columns="CellType", values="log2FoldChange")

    # STEP 3: Sort genes by average log2FoldChange
    mean_fc = pivot.mean(axis=1)
    pivot_sorted = pivot.loc[mean_fc.sort_values(ascending=False).index]
    pivot_sorted = pivot_sorted.dropna(axis=0, how='any')
    sig_genes = pivot_sorted.index.values
    gene_sets[term] = sig_genes


custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\signatures\\'
df_mot_associated_genes = pd.read_csv(custom_sig_path+'Motility-Associated_Genes.csv')
gene_sets['Motility associated signatures'] = df_mot_associated_genes["Gene"].values





import networkx as nx
from itertools import combinations
import community as community_louvain  # Louvain algorithm
import random
# Build the graph
G = nx.Graph()

for name, genes in gene_sets.items():
    G.add_node(name, size=len(genes))

for (p1, g1), (p2, g2) in combinations(gene_sets.items(), 2):
    intersection = len(set(g1) & set(g2))
    union = len(set(g1) | set(g2))
    if union > 0:
        jaccard = intersection / union
        if jaccard > 0:
            G.add_edge(p1, p2, weight=jaccard)

# Louvain clustering to detect communities
partition = community_louvain.best_partition(G, weight='weight')

# Assume G is your graph, pos is your layout
pos = nx.spring_layout(G, seed=42)
partition = community_louvain.best_partition(G, weight='weight')

# Group nodes by community
from collections import defaultdict

community_nodes = defaultdict(list)
for node, group in partition.items():
    community_nodes[group].append(node)

n_nodes_annotate = 20
labeled_nodes = []
for nodes in community_nodes.values():
    labeled_nodes += random.sample(nodes, min(n_nodes_annotate, len(nodes)))

# Plot graph
fig, ax = plt.subplots(figsize=(18, 18))
nx.draw_networkx_nodes(G, pos, node_color=[partition[n] for n in G.nodes()], cmap=plt.cm.tab20, node_size=100)
nx.draw_networkx_edges(G, pos, alpha=0.2)

# Use plt.text for labels
texts = []
for node in labeled_nodes:
    x, y = pos[node]
    texts.append( plt.text(x, y, node, fontsize=12, ha='center', va='center', weight='normal', color='0.2') )

    # Add regulation marker
    if node in regulation:
        if regulation[node] == 'UP':
            plt.scatter(x, y, marker='^', color='green', s=7, zorder=5)
        elif regulation[node] == 'DOWN':
            plt.scatter(x, y, marker='v', color='red', s=7, zorder=5)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='0.2'))
plt.axis("off")
plt.savefig(path+'graph network for both genes.png', dpi=300, bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):
    os.makedirs(path + 'svg/')
fig.savefig(path + 'svg/graph network for both genes.svg', bbox_inches='tight')


#################################### Gene Set Enrichment Analysis ####################################
h2m = pd.read_csv(r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'+'h2m.csv')
h2m_dict = {}
for i, row in h2m.loc[:,["external_gene_name", "mmusculus_homolog_associated_gene_name"]].iterrows():
    if row.isna().any(): continue
    h2m_dict[row['external_gene_name']] = row["mmusculus_homolog_associated_gene_name"]


cb_degs = df_cb_nan_rem[(df_cb_nan_rem['padj'] < 0.05)]
cc_degs = df_cc_nan_rem[(df_cc_nan_rem['padj'] < 0.05)]

for typ, degs_sig in zip(['CC', 'CB'], [cc_degs, cb_degs]):

    degs_up = degs_sig[degs_sig['log2FoldChange'] > 0]
    degs_down = degs_sig[degs_sig['log2FoldChange'] < 0]

    if not os.path.isdir(path + '%s/' %typ):
        os.makedirs(path + '%s/' %typ)

    ############################### Using online Enrichr method ###############################
    df_sigs = pd.DataFrame()
    for library_name in ['GO_Biological_Process_2023', 'GO_Cellular_Component_2023', 'GO_Molecular_Function_2023',
                         'KEGG_2019_Mouse', 'Reactome_2022', 'WikiPathways_2024_Mouse', 'CORUM']:
        file_name = 'GSEA_%s' % (library_name)
        while True:
            try:
                enr_up = gp.enrichr(degs_up['Gene'].astype(str), gene_sets=library_name, outdir=None)
                enr_down = gp.enrichr(degs_down['Gene'].astype(str), gene_sets=library_name, outdir=None)
                break
            except Exception as e:
                print(f"Error: {e}. Retrying...")

        enr_up.res2d['UP_DW'] = "UP"
        enr_down.res2d['UP_DW'] = "DOWN"

        enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
        df_combined = enr_res.sort_values('Combined Score', ascending=False)
        df_combined['Overlap'] = df_combined['Overlap'].astype(str)
        df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]

        df_sig.to_excel(path + '%s/pathway list %s.xlsx' % (typ, file_name), index=False)
        df_combined.to_excel(path + '%s/entire pathway list %s.xlsx' % (typ, file_name), index=False)
        try:
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

            gp.barplot(enr_res, figsize=(6, 6),
                       group='UP_DW',
                       title="%s" % (library_name),
                       ax=ax,
                       color=['b', 'r'])

            fig.savefig(path + '%s/%s.png' % (typ, file_name),dpi=300, bbox_inches='tight')
            if not os.path.isdir(path + 'svg/%s/' % (typ)):
                os.makedirs(path + 'svg/%s/' % (typ))

            fig.savefig(path + 'svg/%s/%s.svg' % (typ, file_name), bbox_inches='tight')

        except:
            pass
        df_sigs = pd.concat([df_sigs, df_sig], axis=0, ignore_index=True)

    ############################### Using offline downloaded gmt files ###############################

    ############################### MsigDB: C2-CGP ###############################
    custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\without DZ\signatures\\'
    sig_temp = gp.read_gmt(path=custom_sig_path + 'c2.cgp.v2024.1.Hs.symbols.gmt')
    file_name = 'GSEA_MsigDB_CGP'

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

    enr_up = gp.enrich(degs_up['Gene'].astype(str), gene_sets=custom_sig, outdir=None)
    enr_down = gp.enrich(degs_down['Gene'].astype(str), gene_sets=custom_sig, outdir=None)

    enr_up.res2d['UP_DW'] = "UP"
    enr_down.res2d['UP_DW'] = "DOWN"

    enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
    df_combined = enr_res.sort_values('Combined Score', ascending=False)
    df_combined['Overlap'] = df_combined['Overlap'].astype(str)
    df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]

    df_sig.to_excel(path + '%s/pathway list %s.xlsx' % (typ, file_name), index=False)
    df_combined.to_excel(path + '%s/entire pathway list %s.xlsx' % (typ, file_name), index=False)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    gp.barplot(enr_res, figsize=(6, 6),
               group='UP_DW',
               title="%s" % (file_name),
               ax=ax,
               color=['b', 'r'])

    fig.savefig(path + '%s/%s.png' % (typ, file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/%s/' % (typ)):
        os.makedirs(path + 'svg/%s/' % (typ))

    fig.savefig(path + 'svg/%s/%s.svg' % (typ, file_name), bbox_inches='tight')

    df_sigs = pd.concat([df_sigs, df_sig], axis=0, ignore_index=True)

    ############################### MsigDB: C7-ImmuneSigDB ###############################
    sig_temp = gp.read_gmt(path=custom_sig_path + 'c7.all.v2024.1.Hs.symbols.gmt')
    file_name = 'GSEA_MsigDB_immunesigdb'

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

    enr_up = gp.enrich(degs_up['Gene'].astype(str), gene_sets=custom_sig, outdir=None)
    enr_down = gp.enrich(degs_down['Gene'].astype(str), gene_sets=custom_sig, outdir=None)

    enr_up.res2d['UP_DW'] = "UP"
    enr_down.res2d['UP_DW'] = "DOWN"

    enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
    df_combined = enr_res.sort_values('Combined Score', ascending=False)
    df_combined['Overlap'] = df_combined['Overlap'].astype(str)
    df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]

    df_sig.to_excel(path + '%s/pathway list %s.xlsx' % (typ, file_name), index=False)
    df_combined.to_excel(path + '%s/entire pathway list %s.xlsx' % (typ, file_name), index=False)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    gp.barplot(enr_res, figsize=(6, 6),
               group='UP_DW',
               title="%s" % (file_name),
               ax=ax,
               color=['b', 'r'])

    fig.savefig(path + '%s/%s.png' % (typ, file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/%s/' % (typ)):
        os.makedirs(path + 'svg/%s/' % (typ))

    fig.savefig(path + 'svg/%s/%s.svg' % (typ, file_name), bbox_inches='tight')

    df_sigs = pd.concat([df_sigs, df_sig], axis=0, ignore_index=True)

    df_sigs.to_excel( path + '%s/significant pathway.xlsx' % (typ), index=False )

    key_strings = 'migration|chemotaxis|motility|cytoskeleton|microtubu|b cell|atp|antigen|synapse|NF-kappa|myc|bcr|b cell receptor|cd40|amoeboid|mtor|' \
                  'lamellipodi|integrin|depolymerization|cxcr|mapk|pi3k|leukocyte|ifn|adhesion|rho gtpase'
    filtered_df = df_sigs[df_sigs['Term'].str.contains(key_strings, case=False, na=False)]
    filtered_df.to_excel(path + '%s/filtered pathway.xlsx' % (typ), index=False)


############# Plot curated pathways enrichment #############
curated_pathways = pd.read_excel(path + 'CC/curated pathway.xlsx')


font = {'family': 'arial',
        'weight': 'normal', }
matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)

gp.barplot(curated_pathways,
           group='UP_DW',
           # title ="%s"%(library_name),
           ax=ax,
           color=['#6699CC', '#CC6677'])

fig.savefig(path + 'CC/curated pathway.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/CC'):
    os.makedirs(path + 'svg/CC')

fig.savefig(path + 'svg/CC/curated pathway.svg', bbox_inches='tight')


#################################### Gene Set Enrichment Analysis for overlapping gene btw CB and CC ####################################
h2m = pd.read_csv(r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'+'h2m.csv')
h2m_dict = {}
for i, row in h2m.loc[:,["external_gene_name", "mmusculus_homolog_associated_gene_name"]].iterrows():
    if row.isna().any(): continue
    h2m_dict[row['external_gene_name']] = row["mmusculus_homolog_associated_gene_name"]


typ = 'both'

cb_degs = df_cb_nan_rem[(df_cb_nan_rem['padj'] < 0.05)&(abs(df_cb_nan_rem['log2FoldChange']) > 1)]
cc_degs = df_cc_nan_rem[(df_cc_nan_rem['padj'] < 0.05)&(abs(df_cc_nan_rem['log2FoldChange']) > 1)]
cb_degs_up = cb_degs[cb_degs['log2FoldChange'] > 0]
cb_degs_down = cb_degs[cb_degs['log2FoldChange'] < 0]
cc_degs_up = cc_degs[cc_degs['log2FoldChange'] > 0]
cc_degs_down = cc_degs[cc_degs['log2FoldChange'] < 0]

set_a = set(cb_degs_up['Gene'])
set_b = set(cc_degs_up['Gene'])
overlap_up = set_a&set_b

set_a = set(cb_degs_down['Gene'])
set_b = set(cc_degs_down['Gene'])
overlap_down = set_a&set_b



if not os.path.isdir(path + '%s/' %typ):
    os.makedirs(path + '%s/' %typ)

############################### Using online Enrichr method ###############################
df_sigs = pd.DataFrame()
for library_name in ['GO_Biological_Process_2023', 'GO_Cellular_Component_2023', 'GO_Molecular_Function_2023',
                     'KEGG_2019_Mouse', 'Reactome_2022', 'WikiPathways_2024_Mouse', 'CORUM']:
    file_name = 'GSEA_%s' % (library_name)
    while True:
        try:
            enr_up = gp.enrichr(list(overlap_up), gene_sets=library_name, outdir=None)
            enr_down = gp.enrichr(list(overlap_down), gene_sets=library_name, outdir=None)
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")

    enr_up.res2d['UP_DW'] = "UP"
    enr_down.res2d['UP_DW'] = "DOWN"

    enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
    df_combined = enr_res.sort_values('Combined Score', ascending=False)
    df_combined['Overlap'] = df_combined['Overlap'].astype(str)
    df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]

    df_sig.to_excel(path + '%s/pathway list %s.xlsx' % (typ, file_name), index=False)
    df_combined.to_excel(path + '%s/entire pathway list %s.xlsx' % (typ, file_name), index=False)
    try:
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

        gp.barplot(enr_res, figsize=(6, 6),
                   group='UP_DW',
                   title="%s" % (library_name),
                   ax=ax,
                   color=['b', 'r'])

        fig.savefig(path + '%s/%s.png' % (typ, file_name),dpi=300, bbox_inches='tight')
        if not os.path.isdir(path + 'svg/%s/' % (typ)):
            os.makedirs(path + 'svg/%s/' % (typ))

        fig.savefig(path + 'svg/%s/%s.svg' % (typ, file_name), bbox_inches='tight')

    except:
        pass
    df_sigs = pd.concat([df_sigs, df_sig], axis=0, ignore_index=True)

############################### Using offline downloaded gmt files ###############################

############################### MsigDB: C2-CGP ###############################
custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\without DZ\signatures\\'
sig_temp = gp.read_gmt(path=custom_sig_path + 'c2.cgp.v2024.1.Hs.symbols.gmt')
file_name = 'GSEA_MsigDB_CGP'

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

enr_up = gp.enrich(list(overlap_up), gene_sets=custom_sig, outdir=None)
enr_down = gp.enrich(list(overlap_down), gene_sets=custom_sig, outdir=None)

enr_up.res2d['UP_DW'] = "UP"
enr_down.res2d['UP_DW'] = "DOWN"

enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
df_combined = enr_res.sort_values('Combined Score', ascending=False)
df_combined['Overlap'] = df_combined['Overlap'].astype(str)
df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]

df_sig.to_excel(path + '%s/pathway list %s.xlsx' % (typ, file_name), index=False)
df_combined.to_excel(path + '%s/entire pathway list %s.xlsx' % (typ, file_name), index=False)

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

gp.barplot(enr_res, figsize=(6, 6),
           group='UP_DW',
           title="%s" % (file_name),
           ax=ax,
           color=['b', 'r'])

fig.savefig(path + '%s/%s.png' % (typ, file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/%s/' % (typ)):
    os.makedirs(path + 'svg/%s/' % (typ))

fig.savefig(path + 'svg/%s/%s.svg' % (typ, file_name), bbox_inches='tight')

df_sigs = pd.concat([df_sigs, df_sig], axis=0, ignore_index=True)

############################### MsigDB: C7-ImmuneSigDB ###############################
sig_temp = gp.read_gmt(path=custom_sig_path + 'c7.all.v2024.1.Hs.symbols.gmt')
file_name = 'GSEA_MsigDB_immunesigdb'

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

enr_up = gp.enrich(list(overlap_up), gene_sets=custom_sig, outdir=None)
enr_down = gp.enrich(list(overlap_down), gene_sets=custom_sig, outdir=None)

enr_up.res2d['UP_DW'] = "UP"
enr_down.res2d['UP_DW'] = "DOWN"

enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
df_combined = enr_res.sort_values('Combined Score', ascending=False)
df_combined['Overlap'] = df_combined['Overlap'].astype(str)
df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]

df_sig.to_excel(path + '%s/pathway list %s.xlsx' % (typ, file_name), index=False)
df_combined.to_excel(path + '%s/entire pathway list %s.xlsx' % (typ, file_name), index=False)

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

gp.barplot(enr_res, figsize=(6, 6),
           group='UP_DW',
           title="%s" % (file_name),
           ax=ax,
           color=['b', 'r'])

fig.savefig(path + '%s/%s.png' % (typ, file_name), dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/%s/' % (typ)):
    os.makedirs(path + 'svg/%s/' % (typ))

fig.savefig(path + 'svg/%s/%s.svg' % (typ, file_name), bbox_inches='tight')

df_sigs = pd.concat([df_sigs, df_sig], axis=0, ignore_index=True)

df_sigs.to_excel( path + '%s/significant pathway.xlsx' % (typ), index=False )

key_strings = 'migration|chemotaxis|motility|cytoskeleton|microtubu|b cell|atp|antigen|synapse|NF-kappa|myc|bcr|b cell receptor|cd40|amoeboid|mtor|' \
              'lamellipodi|integrin|depolymerization|cxcr|mapk|pi3k|leukocyte|ifn|adhesion|rho gtpase'
filtered_df = df_sigs[df_sigs['Term'].str.contains(key_strings, case=False, na=False)]
filtered_df.to_excel(path + '%s/filtered pathway.xlsx' % (typ), index=False)


############# Plot curated pathways enrichment #############
curated_pathways = pd.read_excel(path + 'both/curated pathway.xlsx')


font = {'family': 'arial',
        'weight': 'normal', }
matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)

gp.barplot(curated_pathways,
           group='UP_DW',
           # title ="%s"%(library_name),
           ax=ax,
           color=['#6699CC', '#CC6677'])

fig.savefig(path + 'both/curated pathway.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/CC'):
    os.makedirs(path + 'svg/CC')

fig.savefig(path + 'svg/both/curated pathway.svg', bbox_inches='tight')



#################################### Pathway association using graph networks in CC ####################################
df = pd.read_excel(path+"CC/filtered pathway.xlsx")
df = df[df['Gene_set']!='gs_ind_0'].reset_index(drop=True)
df = df.dropna(subset=['Genes'])

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
df['Term'] = df['Term'].apply(clean_pathway_name)
df = df.drop_duplicates(subset='Term', keep='first').reset_index(drop=True)


sig_temp = {
    row['Term']: np.array(genes)
    for _, row in df.iterrows()
    if len(genes := row['Genes'].split(';')) >= 5
}

regulation = {
    row['Term']: row['UP_DW']
    for _, row in df.iterrows()
    if len(genes := row['Genes'].split(';')) >= 5
}
h2m = pd.read_csv(r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\\'+'h2m.csv')
h2m_dict = {}
for i, row in h2m.loc[:,["external_gene_name", "mmusculus_homolog_associated_gene_name"]].iterrows():
    if row.isna().any(): continue
    h2m_dict[row['external_gene_name']] = row["mmusculus_homolog_associated_gene_name"]

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

gsea_path = path + 'CC/'
draw_graph_network(custom_sig, gsea_path, file_name='gene overlap_curated', sample_n=8, resolution=1.06, regulation=regulation,
                   figsize=(6, 6), inter_spacing=2.5, intra_spacing=1.5)

# import networkx as nx
# from itertools import combinations
# import community as community_louvain  # Louvain algorithm
# import community
# import random
# # Build the graph
# G = nx.Graph()
#
# for name, genes in custom_sig.items():
#     G.add_node(name, size=len(genes))
#
# for (p1, g1), (p2, g2) in combinations(custom_sig.items(), 2):
#     intersection = len(set(g1) & set(g2))
#     union = len(set(g1) | set(g2))
#     if union > 0:
#         jaccard = intersection / union
#         if jaccard > 0:
#             G.add_edge(p1, p2, weight=jaccard)
#
#
# # Assume G is your main NetworkX graph
# components = list(nx.connected_components(G))
#
# sample_n = 10  # Number of nodes to sample per community
#
# for i, nodes in enumerate(components):
#     subG = G.subgraph(nodes).copy()
#
#     # Louvain community detection
#     partition = community.best_partition(subG)
#     communities = set(partition.values())
#
#     # Sample nodes from each community
#     sampled_nodes = []
#     for comm in communities:
#         comm_nodes = [n for n in subG.nodes if partition[n] == comm]
#         sampled = random.sample(comm_nodes, min(sample_n, len(comm_nodes)))
#         sampled_nodes.extend(sampled)
#
#     # Create a sampled subgraph
#     sampled_subG = subG.subgraph(sampled_nodes).copy()
#     pos = nx.spring_layout(sampled_subG, seed=42)
#
#     # Assign colors
#     node_colors = [partition[n] for n in sampled_subG.nodes]
#
#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 10))
#     nx.draw_networkx_edges(sampled_subG, pos, alpha=0.1)
#     nx.draw_networkx_nodes(sampled_subG, pos, node_color=node_colors, cmap=plt.cm.tab20, node_size=100)
#     #nx.draw_networkx_labels(sampled_subG, pos, font_size=6)
#     texts = []
#     for node in sampled_subG.nodes:
#         x, y = pos[node]
#         texts.append( plt.text(x, y, node, fontsize=7, ha='center', va='center') )
#
#         # Add regulation marker
#         if node in regulation:
#             if regulation[node] == 'UP':
#                 plt.scatter(x, y, marker='^', color='green', s=7, zorder=5)
#             elif regulation[node] == 'DOWN':
#                 plt.scatter(x, y, marker='v', color='red', s=7, zorder=5)
#
#
#     adjust_text(texts, arrowprops=dict(arrowstyle='-', color='0.2'))
#
#     plt.title(f"Component {i + 1}: Sampled Nodes from Each Community")
#     plt.axis('off')
#     plt.savefig(path + f'CC/graph network_curated {i + 1}.png', dpi=300, bbox_inches='tight')
#     if not os.path.isdir(path + 'svg/CC/'):
#         os.makedirs(path + 'svg/CC/')
#     fig.savefig(path + f'svg/CC/graph network_curated {i + 1}.svg', bbox_inches='tight')

