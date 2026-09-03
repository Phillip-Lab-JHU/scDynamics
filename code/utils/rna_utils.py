# Author: Chanhong Min <cmin11@jhmi.edu>

"""Functions for rna-seq"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import scipy
import os
import gseapy as gp
import networkx as nx
from itertools import combinations
import community as community_louvain  # Louvain algorithm
import scanpy as sc

def run_gsea(degs_up, degs_down, h2m_dict, path, col_name):

    df_sigs = pd.DataFrame()
    for library_name in ['GO_Biological_Process_2023', 'GO_Cellular_Component_2023', 'GO_Molecular_Function_2023',
                         'KEGG_2019_Mouse', 'Reactome_2022', 'WikiPathways_2024_Mouse', 'CORUM']:
        file_name= 'GSEA_%s'%(library_name)
        while True:
            try:
                enr_up = gp.enrichr(degs_up[col_name].astype(str), gene_sets=library_name, outdir=None)
                enr_down = gp.enrichr(degs_down[col_name].astype(str), gene_sets=library_name, outdir=None)
                break
            except Exception as e:
                print(f"Error: {e}. Retrying...")

        enr_up.res2d['UP_DW'] = "UP"
        enr_down.res2d['UP_DW'] = "DOWN"

        enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
        df_combined = enr_res.sort_values('Combined Score', ascending=False)
        df_combined['Overlap'] = df_combined['Overlap'].astype(str)
        df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]

        df_sig.to_excel(path + 'pathway list %s.xlsx' % (file_name), index=False)
        df_combined.to_excel(path + 'entire pathway list %s.xlsx' % (file_name), index=False)

        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

        gp.barplot(enr_res, figsize=(6, 6),
                   group='UP_DW',
                   title="%s" % (library_name),
                   ax=ax,
                   color=['b', 'r'])

        fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

        if not os.path.isdir(path + 'svg/'):
            os.makedirs(path + 'svg/')

        fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

        plt.close()
        plt.clf()

        df_sigs = pd.concat([df_sigs, df_sig], axis=0, ignore_index=True)

    ############################### Using offline downloaded gmt files ###############################

    ############################### MsigDB: C2-CGP ###############################
    custom_sig_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\RNAseq\scRNAseq_GCB-postGCB_EZH2-selected\analysis\RNA-behavior\signatures\\'
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

    enr_up = gp.enrich(degs_up[col_name].astype(str), gene_sets=custom_sig, outdir=None)
    enr_down = gp.enrich(degs_down[col_name].astype(str), gene_sets=custom_sig, outdir=None)

    enr_up.res2d['UP_DW'] = "UP"
    enr_down.res2d['UP_DW'] = "DOWN"

    enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
    df_combined = enr_res.sort_values('Combined Score', ascending=False)
    df_combined['Overlap'] = df_combined['Overlap'].astype(str)
    df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]

    df_sig.to_excel(path + 'pathway list %s.xlsx' % (file_name), index=False)
    df_combined.to_excel(path + 'entire pathway list %s.xlsx' % (file_name), index=False)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    gp.barplot(enr_res, figsize=(6, 6),
               group='UP_DW',
               title="%s" % (file_name),
               ax=ax,
               color=['b', 'r'])

    fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):
        os.makedirs(path + 'svg/')

    fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
    plt.close()
    plt.clf()

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

    enr_up = gp.enrich(degs_up[col_name].astype(str), gene_sets=custom_sig, outdir=None)
    enr_down = gp.enrich(degs_down[col_name].astype(str), gene_sets=custom_sig, outdir=None)

    enr_up.res2d['UP_DW'] = "UP"
    enr_down.res2d['UP_DW'] = "DOWN"

    enr_res = pd.concat([enr_up.res2d, enr_down.res2d])
    df_combined = enr_res.sort_values('Combined Score', ascending=False)
    df_combined['Overlap'] = df_combined['Overlap'].astype(str)
    df_sig = df_combined[df_combined['Adjusted P-value'] < 0.05]

    df_sig.to_excel(path + 'pathway list %s.xlsx' % (file_name), index=False)
    df_combined.to_excel(path + 'entire pathway list %s.xlsx' % (file_name), index=False)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    gp.barplot(enr_res, figsize=(6, 6),
               group='UP_DW',
               title="%s" % (file_name),
               ax=ax,
               color=['b', 'r'])

    fig.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):
        os.makedirs(path + 'svg/')

    fig.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
    plt.close()
    plt.clf()

    df_sigs = pd.concat([df_sigs, df_sig], axis=0, ignore_index=True)

    df_sigs.to_excel(path+'significant pathway.xlsx', index=False)

    key_strings = 'migration|chemotaxis|motility|cytoskeleton|microtubu|b cell|atp|antigen|synapse|NF-kappa|myc|bcr|b cell receptor|cd40|' \
                  'amoeboid|mtor|lamellipodi|integrin|depolymerization|cxcr|mapk|pi3k|leukocyte|adhesion|rho gtpase|ifn gamma'
    filtered_df = df_sigs[df_sigs['Term'].str.contains(key_strings, case=False, na=False)]
    filtered_df.to_excel(path+'filtered pathway.xlsx', index=False)


def community_detection_louvain(custom_sig: dict[str:list], resolution: float) -> pd.DataFrame:
    ''' Louvain Community detection based on graph
        Parameters:
        ----------
        custom_sig: dict[pathway_name:list of gene names]
            dictionary that contains list of associated genes of each pathway name
        resolution: float
            higher value more clusters

        Returns:
        -------
        df_communities: pd.DataFrame
            DataFrame that contains community ID for each pathway
        '''
    G = nx.Graph()
    for name, genes in custom_sig.items():
        G.add_node(name, size=len(genes))

    for (p1, g1), (p2, g2) in combinations(custom_sig.items(), 2):
        intersection = len(set(g1) & set(g2))
        union = len(set(g1) | set(g2))
        if union > 0:
            jaccard = intersection / union
            if jaccard > 0:
                G.add_edge(p1, p2, weight=jaccard)

    # Analyze each connected component
    components = list(nx.connected_components(G))
    all_records = []
    for i, nodes in enumerate(components):
        subG = G.subgraph(nodes).copy()

        # Louvain community detection
        partition = community_louvain.best_partition(subG, resolution=resolution, random_state=0)
        communities = set(partition.values())
        print(f'Component {i + 1} - Communities detected:', communities)
        for node in subG.nodes:
            comm = partition[node]
            genes = custom_sig[node]  # assuming custom_sig[node] is your gene list
            record = {
                'Component': f'Component_{i + 1}',
                'Pathway': node,
                'Community_ID': comm,
                'Number_of_Genes': len(genes),
                'Genes': ';'.join(genes)
            }
            all_records.append(record)

    df_communities = pd.DataFrame(all_records)
    df_communities['Component_Community'] = df_communities['Component'] + '_Community_' + df_communities[
        'Community_ID'].astype(str)
    df_communities = df_communities.sort_values(['Component_Community', 'Pathway']).reset_index(drop=True)

    return df_communities


def get_distinct_DEG_per_cluster(adata, cluster_name, n_top_genes=300):
    adata_rna = adata.copy()
    sc.tl.rank_genes_groups(adata_rna, groupby="%s" % cluster_name, method="wilcoxon",
                            key_added="dea_%s" % cluster_name, reference='rest')
    result = adata_rna.uns["dea_%s" % cluster_name]
    groups = result["names"].dtype.names
    degs = pd.DataFrame(
        {group + '_' + key: result[key][group]
         for group in groups for key in ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']})

    rows = []
    for mc in np.unique(adata_rna.obs['%s' % cluster_name]):
        names = degs[f'{mc}_names']
        logfc = degs[f'{mc}_logfoldchanges']
        pvals = degs[f'{mc}_pvals_adj']

        for gene, fc, p in zip(names, logfc, pvals):
            if pd.notna(gene):  # skip empty rows
                rows.append((gene, f"{cluster_name}{mc}", fc, p))

    df_long = pd.DataFrame(rows, columns=["gene", cluster_name, "logfc", "pvals_adj"])

    logfc_mat = df_long.pivot(index="gene", columns=cluster_name, values="logfc").fillna(0)  # shape [genes x MCs]
    pval_mat = df_long.pivot(index="gene", columns=cluster_name, values="pvals_adj").fillna(1)  # shape [genes x MCs]

    sig_mat = (pval_mat < 0.05)
    # Step 2: Count in how many MCs each gene is significant
    gene_overlap_counts = sig_mat.sum(axis=1)
    adjusted_scores = pd.DataFrame(index=logfc_mat.index, columns=logfc_mat.columns)

    for mc in logfc_mat.columns:
        sig_score = logfc_mat[mc] * -np.log10(pval_mat[mc])
        penalty = gene_overlap_counts - sig_mat[mc]  # do not count self-MC
        adjusted_scores[mc] = sig_score / (1 + penalty)
        # distinctiveness_score for each gene in each MC = logFC x -log10(adj_pvalue) / (1 + num of other MCs which this gene is also significant)
    adjusted_scores = adjusted_scores.fillna(0)  # fill any NaNs due to log(0) or div-by-zero

    degs_sig_sets = {}
    for mc in adjusted_scores.columns:
        top_genes = adjusted_scores[mc].abs().sort_values(ascending=False).head(n_top_genes).index.tolist()
        degs_sig_sets[mc] = top_genes

    for mc in np.unique(adata_rna.obs['%s' % cluster_name]):
        colname = f"{mc}_adj_scores"
        mc_label = f"{cluster_name}{mc}"
        gene_to_score = adjusted_scores[mc_label]

        # Map adjusted score to gene list row-wise
        degs[colname] = degs[f"{mc}_names"].map(gene_to_score).fillna(0)

    for mc in np.unique(adata_rna.obs['%s' % cluster_name]):
        name_col = f"{mc}_names"
        scores_col = f"{mc}_scores"
        adj_scores_col = f"{mc}_adj_scores"
        cols = degs.columns.tolist()
        insert_idx = cols.index(scores_col) + 1
        if adj_scores_col in cols:
            cols.remove(adj_scores_col)
        cols.insert(insert_idx, adj_scores_col)
        degs = degs[cols]

        # sort each MC's gene list based on adj scores
        temp_df = degs[[name_col, adj_scores_col]].copy()
        temp_df_sorted = temp_df.sort_values(by=adj_scores_col, ascending=False).reset_index(drop=True)
        # Replace original columns with sorted ones
        degs[name_col] = temp_df_sorted[name_col]
        degs[adj_scores_col] = temp_df_sorted[adj_scores_col]

    return degs_sig_sets, degs
