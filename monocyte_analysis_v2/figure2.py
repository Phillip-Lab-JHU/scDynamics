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
"""Generates Data for Figure 2."""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast

#################################### Motility space ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'

df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
#df = pd.read_parquet(path+'nonmoving_removed_all_features_30_PC.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df = df[df['Age']!=55].reset_index(drop=True)
df_duration = df_duration[df_duration['Age']!=55].reset_index(drop=True)

df_young = df[df['Type']=='Young'].reset_index(drop=True)
df_old = df[df['Type']=='Old'].reset_index(drop=True)
df_frail = df[df['Type']=='Frail'].reset_index(drop=True)

for group in np.unique(df['Type']):
    df_part = df[df['Type']==group].reset_index(drop=True)
    for cond in np.unique(df_part['Condition']):
        df_part_part = df_part[df_part['Condition']==cond].reset_index(drop=True)
        print(group, cond, df_part_part.shape[0], np.unique(df_part_part['Patient']).shape[0])


#p_dict = permutation_test(df, group_name='Age', class_name='kmeans', iteration=1000)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure2. Perturbation\\'
#path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis\Figure2-2. Only moving perturbation\\'


xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

young_path = path+'young/'
old_path = path+'old/'
frail_path = path+'frail/'

color_list = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')

draw_umap_space(df_young, young_path, file_name='motility space_media_young', condition_name='Condition', label_name='pseudo_particle',
                colors = ('#888888','#CC6677', '#44AA99', '#6699CC'), dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_young, young_path, file_name='motility space_exact_age_young', condition_name='Age', label_name='pseudo_particle',
                colors = cmc.turku, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_young, young_path, file_name='motility space_exact_patient_young', condition_name='Patient', label_name='pseudo_particle',
                colors = cmc.turku, dot_size=0.07, x_name='PC1', y_name='PC2')


draw_jointplot(xs='PC1', y='PC2', df=df_young, path=young_path, file_name='jointplot_condition', hue="Condition", colors=('#888888','#CC6677', '#44AA99', '#6699CC'),
               hue_order=['Control', 'DNA', 'IL6', 'LPS'], legend=False, fill=True, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2',
               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
draw_jointplot(xs='PC1', y='PC2', df=df_young, path=young_path, file_name='jointplot_age', hue="Age", colors=cmc.turku,
               legend=True, fill=True, thresh=0.6, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2',
               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

draw_contour(df_young, young_path, file_name='space_contour_media_young', condition_name='Condition', colors=('#888888','#CC6677', '#44AA99', '#6699CC'),
             x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df_young, young_path, file_name='space_contour_exact_age_young', condition_name='Age', colors=color_list*2, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df_young, young_path, file_name='space_contour_patient_young', condition_name='Patient', colors=color_list*2, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)

########################################################################

draw_umap_space(df_old, old_path, file_name='motility space_media_old', condition_name='Condition', label_name='pseudo_particle',
                colors = ('#888888','#CC6677', '#44AA99', '#6699CC'), dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_old, old_path, file_name='motility space_exact_age_old', condition_name='Age', label_name='pseudo_particle',
                colors = cmc.turku, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_old, old_path, file_name='motility space_exact_patient_old', condition_name='Patient', label_name='pseudo_particle',
                colors = cmc.turku, dot_size=0.07, x_name='PC1', y_name='PC2')

draw_jointplot(xs='PC1', y='PC2', df=df_old, path=old_path, file_name='jointplot_condition', hue="Condition", colors=('#888888','#CC6677', '#44AA99', '#6699CC'),
               hue_order=['Control', 'DNA', 'IL6', 'LPS'], legend=False, fill=True, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2',
               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
draw_jointplot(xs='PC1', y='PC2', df=df_old, path=old_path, file_name='jointplot_age', hue="Age", colors=cmc.turku,
               legend=True, fill=True, thresh=0.6, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2',
               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

draw_contour(df_old, old_path, file_name='space_contour_media_old', condition_name='Condition', colors=('#888888','#CC6677', '#44AA99', '#6699CC'),
             x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df_old, old_path, file_name='space_contour_exact_age_old', condition_name='Age', colors=color_list*2, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df_old, old_path, file_name='space_contour_patient_old', condition_name='Patient', colors=color_list*2, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)

########################################################################
draw_umap_space(df_frail, frail_path, file_name='motility space_media_frail', condition_name='Condition', label_name='pseudo_particle',
                colors = ('#888888','#CC6677', '#44AA99', '#6699CC'), dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_frail, frail_path, file_name='motility space_exact_age_frail', condition_name='Age', label_name='pseudo_particle',
                colors = cmc.turku, dot_size=0.07, x_name='PC1', y_name='PC2')
draw_umap_space(df_frail, frail_path, file_name='motility space_exact_patient_frail', condition_name='Patient', label_name='pseudo_particle',
                colors = cmc.turku, dot_size=0.07, x_name='PC1', y_name='PC2')

draw_jointplot(xs='PC1', y='PC2', df=df_frail, path=frail_path, file_name='jointplot_condition', hue="Condition", colors=('#888888','#CC6677', '#44AA99', '#6699CC'),
               hue_order=['Control', 'DNA', 'IL6', 'LPS'], legend=False, fill=True, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2',
               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
draw_jointplot(xs='PC1', y='PC2', df=df_frail, path=frail_path, file_name='jointplot_age', hue="Age", colors=cmc.turku,
               legend=True, fill=True, thresh=0.6, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2',
               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

draw_contour(df_frail, frail_path, file_name='space_contour_media_frail', condition_name='Condition', colors=('#888888','#CC6677', '#44AA99', '#6699CC'),
             x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df_frail, frail_path, file_name='space_contour_exact_age_frail', condition_name='Age', colors=color_list*2, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)
draw_contour(df_frail, frail_path, file_name='space_contour_patient_frail', condition_name='Patient', colors=color_list*2, x_name='PC1', y_name='PC2', bin_num=50, num_contours=5)

for cond in np.unique(df['Condition']):
    draw_umap_space(df[df['Condition']==cond].reset_index(drop=True), path, file_name='%s_space_type'%cond, condition_name='Type', label_name='pseudo_particle',
                    colors = ('#fdc086', '#beaed4', '#7fc97f'), dot_size=0.07, x_name='PC1', y_name='PC2')
    draw_jointplot(xs='PC1', y='PC2', df=df[df['Condition']==cond].reset_index(drop=True), path=path, file_name='%s_jointplot_type'%cond, hue="Type",
                   colors=('#fdc086', '#beaed4', '#7fc97f'), alpha=0.6, legend=False, fill=True,
                   thresh=0.3, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

#################################### Various Heatmaps for young, old, frail ####################################
draw_cluster_distribution_heatmap(df_young, young_path, file_name='condition_kmeans_young_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=30, cmap=cmc.bilbao_r, condition_name='Condition', cluster_type='kmeans', figsize=(4,2))
draw_cluster_distribution_heatmap(df_old, old_path, file_name='condition_kmeans_old_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=30, cmap=cmc.bilbao_r, condition_name='Condition', cluster_type='kmeans', figsize=(4,2))
draw_cluster_distribution_heatmap(df_frail, frail_path, file_name='condition_kmeans_frail_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=30, cmap=cmc.bilbao_r, condition_name='Condition', cluster_type='kmeans', figsize=(4,2))

draw_relative_cluster_distribution_heatmap(df_young, young_path, file_name='relative_condition_kmeans_young_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=40, cmap=cmc.oslo_r, transpose=False, condition_name='Condition', cluster_type='kmeans', figsize=(4,2))
draw_relative_cluster_distribution_heatmap(df_old, old_path, file_name='relative_condition_kmeans_old_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=40,cmap=cmc.oslo_r, transpose=False, condition_name='Condition', cluster_type='kmeans', figsize=(4,2))
draw_relative_cluster_distribution_heatmap(df_frail, frail_path, file_name='relative_condition_kmeans_frail_heatmap', col_cluster=False, row_cluster=False,
                                  annot=False, vmax=40, cmap=cmc.oslo_r, transpose=False, condition_name='Condition', cluster_type='kmeans', figsize=(4,2))


for cond in np.unique(df['Condition']):
    draw_cluster_distribution_heatmap(df_young[df_young['Condition']==cond].reset_index(drop=True), young_path, file_name='%s_age_kmeans_heatmap'%cond, condition_name='Age',
                                      cluster_type='kmeans', transpose=True, cmap=cmc.bilbao_r, row_cluster=False, col_cluster=False, figsize=(10,4))

for cond in np.unique(df['Condition']):
    draw_cluster_distribution_heatmap(df_young[df_young['Condition']==cond].reset_index(drop=True), young_path, file_name='%s_patient_kmeans_heatmap'%cond,
                                      condition_name='Patient',cluster_type='kmeans', transpose=False, row_cluster=True, col_cluster=False,
                                      cmap=cmc.bilbao_r, figsize=(4, 10))

for cond in np.unique(df['Condition']):
    draw_cluster_distribution_heatmap(df_old[df_old['Condition']==cond].reset_index(drop=True), old_path, file_name='%s_age_kmeans_heatmap'%cond, condition_name='Age',
                                      cluster_type='kmeans', transpose=True, cmap=cmc.bilbao_r, row_cluster=False, col_cluster=False, figsize=(4,4))

for cond in np.unique(df['Condition']):
    draw_cluster_distribution_heatmap(df_old[df_old['Condition']==cond].reset_index(drop=True), old_path, file_name='%s_patient_kmeans_heatmap'%cond,
                                      condition_name='Patient',cluster_type='kmeans', transpose=False, row_cluster=True, col_cluster=False,
                                      cmap=cmc.bilbao_r, figsize=(4, 6))

for cond in np.unique(df['Condition']):
    draw_cluster_distribution_heatmap(df_frail[df_frail['Condition']==cond].reset_index(drop=True), frail_path, file_name='%s_age_kmeans_heatmap'%cond, condition_name='Age',
                                      cluster_type='kmeans', transpose=True, cmap=cmc.bilbao_r, row_cluster=False, col_cluster=False, figsize=(5,4))

for cond in np.unique(df['Condition']):
    draw_cluster_distribution_heatmap(df_frail[df_frail['Condition']==cond].reset_index(drop=True), frail_path, file_name='%s_patient_kmeans_heatmap'%cond,
                                      condition_name='Patient',cluster_type='kmeans', transpose=False, row_cluster=True, col_cluster=False,
                                      cmap=cmc.bilbao_r, figsize=(4, 6))

#################################### Box plot comparing all motility features of young ####################################
test = 'one-way anova_dunnett'
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list(df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']))
feature_list.append('Time_span')

condition_name = 'Condition'
#replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'wt GCB'}
if not os.path.isdir(young_path + 'motility_feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(young_path + 'motility_feature_violin_plot_type/')

if not os.path.isdir(young_path + 'motility_feature_box_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(young_path + 'motility_feature_box_plot_type/')

for feature_name in feature_list:

    if feature_name=='total_distance':
        vmax=100
    elif feature_name=='progressivity':
        vmax=0.21
    else:
        vmax=None

    dataset={}
    for condition in np.unique(df_young[condition_name]):
        data = df_young[df_young[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    # new_order = ['wt_B-cell', 'T-cell']
    # ordered_dataset = change_dict_order(dataset, new_order)
    # dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    draw_custom_violin_plot(dataset, young_path+'motility_feature_violin_plot_type/', file_name=feature_name, colors = ('#888888','#CC6677', '#44AA99', '#6699CC'),
                            test=test, pvalue=True, figsize=(2,2))

    draw_custom_bar_plot(dataset, young_path + 'motility_feature_box_plot_type/', file_name=feature_name, vmax=vmax,
                         strip_plot=False, colors=('#888888','#CC6677', '#44AA99', '#6699CC'), test=test, pvalue=True, figsize=(1, 2))


batches = np.unique(df_young['Patient'])
condition_name = 'Condition'

colors = ('#888888','#CC6677', '#44AA99', '#6699CC')

if not os.path.isdir(young_path + 'experimental_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(young_path + 'experimental_motility_feature/')

for feature_name in feature_list:
    if feature_name=='total_distance':
        vmax=200
    elif feature_name=='progressivity':
        vmax=0.4
    else:
        vmax=None
    dataset = {}
    for type in np.unique(df[condition_name]):
        df_part = df_young[df_young[condition_name] == type]
        avgs = []
        for batch in batches:
            #if 'A' in video and cell_type == 'mt_B-cell':
            df_patient = df_part[df_part['Patient'] == batch]
            if df_patient.shape[0] == 0:
                continue
            data = df_patient[feature_name]
            avg = np.mean(data)
            avgs.append(avg)
        dataset[type] = avgs
    new_order = ['Control', 'DNA', 'IL6', 'LPS']
    dataset = change_dict_order(dataset, new_order)

    draw_custom_bar_plot(dataset, young_path+'experimental_motility_feature/', file_name=feature_name, vmax=vmax,
                         strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1, 2))



batches = np.unique(df_young['Patient'])
condition_name = 'Condition'
colors = ('#CC6677', '#44AA99', '#6699CC')

if not os.path.isdir(young_path + 'experimental_response_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(young_path + 'experimental_response_motility_feature/')

for feature_name in feature_list:
    dataset = {}
    for type in np.unique(df[condition_name]):
        if type == 'Control':
            continue
        df_part = df_young[df_young[condition_name] == type]
        df_ref = df_young[df_young[condition_name] == 'Control']
        avgs = []
        for batch in batches:
            #if 'A' in video and cell_type == 'mt_B-cell':
            df_patient = df_part[df_part['Patient'] == batch]
            df_patient_ref = df_ref[df_ref['Patient'] == batch]

            if (df_patient.shape[0] == 0) or (df_patient_ref.shape[0] == 0):
                continue
            data = df_patient[feature_name]
            ref_data = df_patient_ref[feature_name]
            avg = np.median(data)
            ref_avg = np.median(ref_data)
            diff = (avg - ref_avg)/ref_avg
            if feature_name == 'avg_speed':
                print(type, batch, diff, avg, ref_avg)
            avgs.append(diff)
        dataset[type] = avgs
    new_order = ['DNA', 'IL6', 'LPS']
    dataset = change_dict_order(dataset, new_order)

    draw_custom_bar_plot(dataset, young_path+'experimental_response_motility_feature/', file_name=feature_name,
                         strip_plot=True,  estimator='mean', colors=colors, test='kruskal-wallis_dunn', pvalue=True, figsize=(1, 2))

#################################### Box plot comparing all motility features of old ####################################
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list(df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']))
feature_list.append('Time_span')

condition_name = 'Condition'
#replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'wt GCB'}
if not os.path.isdir(old_path + 'motility_feature_violin_plot_type_old/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(old_path + 'motility_feature_violin_plot_type_old/')

if not os.path.isdir(old_path + 'motility_feature_box_plot_type_old/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(old_path + 'motility_feature_box_plot_type_old/')

for feature_name in feature_list:
    if feature_name=='total_distance':
        vmax=100
    elif feature_name=='progressivity':
        vmax=0.21
    else:
        vmax=None

    dataset={}
    for condition in np.unique(df_old[condition_name]):
        data = df_old[df_old[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    # new_order = ['wt_B-cell', 'T-cell']
    # ordered_dataset = change_dict_order(dataset, new_order)
    # dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    draw_custom_violin_plot(dataset, old_path+'motility_feature_violin_plot_type_old/', file_name=feature_name, colors = ('#888888','#CC6677', '#44AA99', '#6699CC'),
                            test=test, pvalue=True, figsize=(2,2))

    draw_custom_bar_plot(dataset, old_path + 'motility_feature_box_plot_type_old/', file_name=feature_name, vmax=vmax,
                         strip_plot=False, colors=('#888888', '#CC6677', '#44AA99', '#6699CC'), test=test, pvalue=True,
                         figsize=(1, 2))


batches = np.unique(df_old['Patient'])
condition_name = 'Condition'
colors = ('#888888','#CC6677', '#44AA99', '#6699CC')

if not os.path.isdir(old_path + 'experimental_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(old_path + 'experimental_motility_feature/')

for feature_name in feature_list:
    if feature_name=='total_distance':
        vmax=200
    elif feature_name=='progressivity':
        vmax=0.4
    else:
        vmax=None
    dataset = {}
    for type in np.unique(df[condition_name]):
        df_part = df_old[df_old[condition_name] == type]
        avgs = []
        for batch in batches:
            #if 'A' in video and cell_type == 'mt_B-cell':
            df_patient = df_part[df_part['Patient'] == batch]
            if df_patient.shape[0] == 0:
                continue
            data = df_patient[feature_name]
            avg = np.mean(data)
            avgs.append(avg)
        dataset[type] = avgs
    new_order = ['Control', 'DNA', 'IL6', 'LPS']
    dataset = change_dict_order(dataset, new_order)

    draw_custom_bar_plot(dataset, old_path+'experimental_motility_feature/', file_name=feature_name, vmax=vmax,
                         strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1, 2))


batches = np.unique(df_old['Patient'])
condition_name = 'Condition'
colors = ('#CC6677', '#44AA99', '#6699CC')

if not os.path.isdir(old_path + 'experimental_response_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(old_path + 'experimental_response_motility_feature/')

for feature_name in feature_list:
    dataset = {}
    for type in np.unique(df[condition_name]):
        if type == 'Control':
            continue
        df_part = df_old[df_old[condition_name] == type]
        df_ref = df_old[df_old[condition_name] == 'Control']
        avgs = []
        for batch in batches:
            #if 'A' in video and cell_type == 'mt_B-cell':
            df_patient = df_part[df_part['Patient'] == batch]
            df_patient_ref = df_ref[df_ref['Patient'] == batch]

            if (df_patient.shape[0] == 0) or (df_patient_ref.shape[0] == 0):
                continue
            data = df_patient[feature_name]
            ref_data = df_patient_ref[feature_name]
            avg = np.median(data)
            ref_avg = np.median(ref_data)
            diff = (avg - ref_avg)/ref_avg
            avgs.append(diff)
        dataset[type] = avgs
    new_order = ['DNA', 'IL6', 'LPS']
    dataset = change_dict_order(dataset, new_order)

    draw_custom_bar_plot(dataset, old_path+'experimental_response_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=colors, test='kruskal-wallis_dunn', pvalue=True, figsize=(1, 2))

#################################### Box plot comparing all motility features of frail ####################################
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list(df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']))
feature_list.append('Time_span')

condition_name = 'Condition'
#replace_keys = {'T-cell':'Tfh', 'wt_B-cell':'wt GCB'}
if not os.path.isdir(frail_path + 'motility_feature_violin_plot_type_frail/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(frail_path + 'motility_feature_violin_plot_type_frail/')

if not os.path.isdir(frail_path + 'motility_feature_box_plot_type_frail/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(frail_path + 'motility_feature_box_plot_type_frail/')

for feature_name in feature_list:
    if feature_name=='total_distance':
        vmax=100
    elif feature_name=='progressivity':
        vmax=0.21
    else:
        vmax=None
    dataset={}
    for condition in np.unique(df_frail[condition_name]):
        data = df_frail[df_frail[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    # new_order = ['wt_B-cell', 'T-cell']
    # ordered_dataset = change_dict_order(dataset, new_order)
    # dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    draw_custom_violin_plot(dataset, frail_path+'motility_feature_violin_plot_type_frail/', file_name=feature_name, colors = ('#888888','#CC6677', '#44AA99', '#6699CC'),
                            test=test, pvalue=True, figsize=(2,2))

    draw_custom_bar_plot(dataset, frail_path + 'motility_feature_box_plot_type_frail/', file_name=feature_name, vmax=vmax,
                         strip_plot=False, colors=('#888888', '#CC6677', '#44AA99', '#6699CC'), test=test, pvalue=True,
                         figsize=(1, 2))


batches = np.unique(df_frail['Patient'])
condition_name = 'Condition'
colors = ('#888888','#CC6677', '#44AA99', '#6699CC')

if not os.path.isdir(frail_path + 'experimental_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(frail_path + 'experimental_motility_feature/')

for feature_name in feature_list:
    if feature_name=='total_distance':
        vmax=200
    elif feature_name=='progressivity':
        vmax=0.4
    else:
        vmax=None
    dataset = {}
    for type in np.unique(df[condition_name]):
        df_part = df_frail[df_frail[condition_name] == type]
        avgs = []
        for batch in batches:
            #if 'A' in video and cell_type == 'mt_B-cell':
            df_patient = df_part[df_part['Patient'] == batch]
            if df_patient.shape[0] == 0:
                continue
            data = df_patient[feature_name]
            avg = np.mean(data)
            avgs.append(avg)
        dataset[type] = avgs
    new_order = ['Control', 'DNA', 'IL6', 'LPS']
    dataset = change_dict_order(dataset, new_order)

    draw_custom_bar_plot(dataset, frail_path+'experimental_motility_feature/', file_name=feature_name, vmax=vmax,
                         strip_plot=True, colors=colors, test=test, pvalue=True, figsize=(1, 2))


batches = np.unique(df_frail['Patient'])
condition_name = 'Condition'
colors = ('#CC6677', '#44AA99', '#6699CC')

if not os.path.isdir(frail_path + 'experimental_response_motility_feature/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(frail_path + 'experimental_response_motility_feature/')

for feature_name in feature_list:
    dataset = {}
    for type in np.unique(df[condition_name]):
        if type == 'Control':
            continue
        df_part = df_frail[df_frail[condition_name] == type]
        df_ref = df_frail[df_frail[condition_name] == 'Control']
        avgs = []
        for batch in batches:
            #if 'A' in video and cell_type == 'mt_B-cell':
            df_patient = df_part[df_part['Patient'] == batch]
            df_patient_ref = df_ref[df_ref['Patient'] == batch]

            if (df_patient.shape[0] == 0) or (df_patient_ref.shape[0] == 0):
                continue
            data = df_patient[feature_name]
            ref_data = df_patient_ref[feature_name]
            avg = np.median(data)
            ref_avg = np.median(ref_data)
            diff = (avg - ref_avg)/ref_avg
            avgs.append(diff)
        dataset[type] = avgs
    new_order = ['DNA', 'IL6', 'LPS']
    dataset = change_dict_order(dataset, new_order)

    draw_custom_bar_plot(dataset, frail_path+'experimental_response_motility_feature/', file_name=feature_name,
                         strip_plot=True, colors=colors, test='kruskal-wallis_dunn', pvalue=True, figsize=(1, 2))


#################################### Fold change of all motility features wrt each controls ####################################
df.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df.columns[8:103].drop(['speed_distribution_x', 'speed_distribution_y'])

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma

Z_avg_df = pd.DataFrame()
for feature_name in feature_list:
    Z_avg_df_temp = pd.DataFrame()
    group_list = []
    for age_group in np.unique(df['Type']):

        df_part = df[df['Type']==age_group].reset_index(drop=True)
        condition_list = []
        for group in np.unique(df_part['Condition']):
            if group == 'Control':
                continue

            data = df_part[(df_part['Condition'] == group)][feature_name]
            ref_data = df_part[(df_part['Condition'] == 'Control')][feature_name]
            avg = np.median(data)
            ref_avg = np.median(ref_data)
            diff = np.log2((avg)/(ref_avg))
            condition_list.append(diff)
        each_feature = pd.DataFrame( condition_list, columns=[feature_name], index='%s '%age_group + np.delete( np.unique(df_part['Condition']), 0 ) )

        Z_avg_df_temp = pd.concat([Z_avg_df_temp, each_feature], axis=0)

    Z_avg_df = pd.concat([Z_avg_df, Z_avg_df_temp], axis=1)

Z_avg_df = Z_avg_df.replace([np.inf, -np.inf], np.nan)  # Convert inf to nan
Z_avg_df = Z_avg_df.dropna(axis=1, how='any')
columns_to_drop = Z_avg_df.columns[(Z_avg_df < np.log2(1.3)).all()]
Z_avg_df.drop(columns=columns_to_drop, inplace=True)

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

g=sns.clustermap(Z_avg_df, annot=False, cmap=cmc.vik, col_cluster=True, row_cluster=False,
#cbar_pos=(1, 0.2, 0.03, 0.8),
metric='correlation', method='average',
linewidths=0.5, linecolor='black',
alpha=0.7,
**kws,
figsize = (12, 3.5),
dendrogram_ratio=0.1,
cbar=True
)

#g.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
#g.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=10, rotation=35, rotation_mode='anchor', ha='right')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=12, va='center')

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.8, 1.1, g.ax_row_dendrogram.get_position().width*1, 0.02])

g.ax_cbar.set_title('log2FC versus Control', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

# for spine in g.ax_cbar.spines:
#     #g.ax_cbar.spines[spine].set_color('crimson')
#     g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'filtered fold change wrt each control heatmap.png', dpi=300,bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/' + 'filtered fold change wrt each control heatmap.svg', bbox_inches='tight')
plt.clf()
plt.close()


#################################### Type+Condition kmeans heatmap ###################################

df_ = df.copy()
df_['type_condition'] = df['Type'].astype(str) + ' ' + df['Condition']
draw_cluster_distribution_heatmap(df_, path, file_name='all_kmeans_heatmap', condition_name='type_condition', vmax=None,
                                  cluster_type='kmeans', col_cluster=False, row_cluster=True, cmap=cmc.bilbao_r, figsize=(4,5))

#################################### Fold change of all motility features wrt Young control ####################################

df_ = df.copy()
df_['type_condition'] = df['Type'].astype(str) + ' ' + df['Condition']

df_.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df_.columns[8:103].drop(['speed_distribution_x', 'speed_distribution_y',
                                            'inst_speed_cosine_similarity_entropies', 'inst_angle_cosine_similarity_entropies',
                                        'angle_autocorr_2', 'angle_autocorr_3', 'displ_partial_autocorr_3'])

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma

Z_avg_df = pd.DataFrame()
for feature_name in feature_list:
    Z_avg_df_temp = pd.DataFrame()
    Z_df = pd.DataFrame()
    pdf_temp = pd.DataFrame()
    group_list = []
    for group in np.unique(df_['type_condition']):
        # if group == 'Young Control':
        #     continue
        Z_matrix = pd.DataFrame()
        Z_avg_matrix = []
        data = df_[(df_['type_condition'] == group)][feature_name]
        ref_data = df_[(df_['type_condition'] == 'Young Control')][feature_name]
        avg = np.median(data)
        ref_avg = np.median(ref_data)
        #diff = (avg - ref_avg) / ref_avg
        diff = np.log2((avg)/(ref_avg))

        group_list.append(diff)

    Z_avg_df = pd.concat([Z_avg_df,pd.DataFrame(group_list, columns=[feature_name], index=np.unique(df_['type_condition']) )], axis=1)

Z_avg_df = Z_avg_df.replace([np.inf, -np.inf], np.nan)  # Convert inf to nan
Z_avg_df = Z_avg_df.dropna(axis=1, how='any')
#scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
#Z_avg_df= pd.DataFrame(scaler.fit_transform( Z_avg_df ), columns=Z_avg_df.columns, index=Z_avg_df.index)

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

kws = dict(cbar_kws=dict(ticks=[1, 0, -1], orientation='horizontal'), vmin=-1, vmax=1 )

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

g.ax_cbar.set_title('log2FC versus Young Control', fontsize=16)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=16)

for spine in g.ax_cbar.spines:
    #g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(0.5)

plt.savefig(path+'fold change wrt young control heatmap.png', dpi=300,bbox_inches='tight')
if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/' + 'fold change wrt young control heatmap.svg', bbox_inches='tight')
plt.clf()
plt.close()


#################################### PCA by patient-wise (average motility features -> PCA) ###################################
df.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df.columns[8:103].drop(['speed_distribution_x', 'speed_distribution_y',
                                            'inst_speed_cosine_similarity_entropies', 'inst_angle_cosine_similarity_entropies'])

pca_inputs = pd.DataFrame()

for patient in np.unique(df['Patient']):
    each_patient = df[(df['Patient']==patient)].reset_index(drop=True)
    for cond in np.unique(each_patient['Condition']):
        df_part = each_patient[(each_patient['Condition']==cond)].reset_index(drop=True)
        motility_input = df_part[feature_list]
        pca_input = motility_input.median(axis=0).to_frame().T
        pca_input['Patient'] = np.unique(df_part['Patient'])[0]
        pca_input['Type'] = np.unique(df_part['Type'])[0]
        pca_input['Condition'] = np.unique(df_part['Condition'])[0]
        pca_input['Age'] = np.unique(df_part['Age'])[0]

        pca_input['kmeans'] = np.median(df_part['kmeans'])
        pca_input['Weakness'] = np.unique(df_part['Weakness'])[0]
        pca_input['Weight_loss'] = np.unique(df_part['Weight_loss'])[0]
        pca_input['Exhaustion'] = np.unique(df_part['Exhaustion'])[0]
        pca_input['Activity'] = np.unique(df_part['Activity'])[0]
        pca_input['Gait'] = np.unique(df_part['Gait'])[0]
        pca_input['Grip'] = np.unique(df_part['Grip'])[0]
        pca_input['Frailty_score'] = np.unique(df_part['Frailty_score'])[0]

        pca_inputs = pd.concat([pca_inputs, pca_input], axis=0)

pca_inputs = pca_inputs.reset_index(drop=True)

pca_inputs_only = pca_inputs.iloc[:, :-12]
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
pca_inputs_only= pd.DataFrame(scaler.fit_transform( pca_inputs_only ), columns=pca_inputs_only.columns)

from sklearn.decomposition import PCA
pca = PCA(n_components=4)

pcs_array = pca.fit_transform(pca_inputs_only)  # factor scores for non-rotated data
df_pcs = pd.DataFrame(pcs_array, columns=['PC1', 'PC2', 'PC3', 'PC4'])

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'])
#loadings = pd.concat([df_title, loadings], axis=1)

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC1', 'PC2', 'PC3', 'PC4'])

df_patient = pd.concat([pca_inputs, df_pcs], axis=1)

# pcs_list = ['PC1', 'PC2', 'PC3', 'PC4']
# from itertools import combinations
# for pair in list(combinations(pcs_list, 2)):
#     x_name = pair[0]
#     y_name = pair[1]

x_name='PC1'
y_name='PC2'
for cond in np.unique(df_patient['Condition']):
    draw_umap_space(df_patient[df_patient['Condition']==cond].reset_index(drop=True), path +'/patient space (average first)/', file_name='%s patient space %s %s'%(cond, x_name, y_name),
                    condition_name='Type', label_name='Patient', colors=('#fdc086', '#beaed4', '#7fc97f'), dot_size=5, x_name=x_name, y_name=y_name)

from itertools import combinations
for x_name, y_name in combinations(['PC1', 'PC2', 'PC3', 'PC4'], r=2):
    xlabel = '%s('%x_name + str(round(variance['%s'%x_name][1] * 100, ndigits=1)) + '%)'
    ylabel = '%s('%y_name + str(round(variance['%s'%y_name][1] * 100, ndigits=1)) + '%)'
    draw_diff_arrow_scatter(df_patient, path+'/patient space (average first)/', file_name='differential response patients %s %s'%(x_name, y_name), condition_name='Type',
                            diff_condition_name='Condition',ind_name='Patient', ref='Control', colors=('#fdc086', '#beaed4', '#7fc97f'),
                            dot_size=5, x_name=x_name, y_name=y_name, xlabel=xlabel, ylabel=ylabel)


#################################### dataframe for magnitude of change ###################################

x_name='PC1'
y_name='PC2'
for x_name, y_name in combinations(['PC1', 'PC2', 'PC3', 'PC4'], r=2):
    patient_response = pd.DataFrame()
    for patient in np.unique(df_patient['Patient']):  # For patients within condition
        df_part = df_patient[(df_patient['Patient'] == patient)].reset_index(drop=True)
        df_ref_ind = df_part[df_part['Condition'] == 'Control'].reset_index(drop=True)
        if df_ref_ind.shape[0] == 0:
            continue

        temp = pd.DataFrame()
        for cond in np.unique(df_part['Condition']):
            if cond == 'Control':
                continue
            df_cond_ind = df_part[df_part['Condition'] == cond].reset_index(drop=True)

            transition = variance['%s' % x_name][1] * (df_cond_ind[x_name] - df_ref_ind[x_name]) ** 2 + \
                         variance['%s' % y_name][1] * (df_cond_ind[y_name] - df_ref_ind[y_name]) ** 2
            # transition = (df_cond_ind[x_name] - df_ref_ind[x_name]) ** 2 + \
            #              (df_cond_ind[y_name] - df_ref_ind[y_name]) ** 2
            # transition = (df_cond_ind[x_name] - df_ref_ind[x_name]) ** 2 + \
            #              (df_cond_ind[y_name] - df_ref_ind[y_name]) ** 2 + \
            #              (df_cond_ind['PC3'] - df_ref_ind['PC3']) ** 2 + \
            #              (df_cond_ind['PC4'] - df_ref_ind['PC4']) ** 2
            # transition = variance['%s' % x_name][1] * (df_cond_ind[x_name] - df_ref_ind[x_name]) ** 2 + \
            #              variance['%s' % y_name][1] * (df_cond_ind[y_name] - df_ref_ind[y_name]) ** 2 + \
            #              variance['%s' % 'PC3'][1] * (df_cond_ind['PC3'] - df_ref_ind['PC3']) ** 2 + \
            #              variance['%s' % 'PC4'][1] * (df_cond_ind['PC4'] - df_ref_ind['PC4']) ** 2

            transition = np.sqrt(transition)
            temp = pd.concat([temp, pd.DataFrame(transition, columns=['%s response'%cond])], axis=1)
        temp['Type'] = np.unique(df_part['Type'])
        temp['Patient'] = np.unique(df_part['Patient'])

        temp['Age'] = np.unique(df_part['Age'])
        #temp['kmeans'] = np.unique(df_part['kmeans'])
        temp['Weakness'] = np.unique(df_part['Weakness'])
        temp['Weight_loss'] = np.unique(df_part['Weight_loss'])
        temp['Exhaustion'] = np.unique(df_part['Exhaustion'])
        temp['Activity'] = np.unique(df_part['Activity'])
        temp['Gait'] = np.unique(df_part['Gait'])
        temp['Grip'] = np.unique(df_part['Grip'])
        temp['Frailty_score'] = np.unique(df_part['Frailty_score'])

        patient_response = pd.concat([patient_response, temp], axis=0)
    patient_response = patient_response.reset_index(drop=True)



    #################################### perturbation map ###################################
    fig, ax = plt.subplots(figsize=(2,2))
    grid = sns.PairGrid(data=patient_response, vars=['DNA response', 'IL6 response', 'LPS response'],
                        height=4, aspect=1, hue='Type', hue_order=['Frail', 'Old', 'Young'], palette=('#fdc086', '#beaed4', '#7fc97f'),
                        diag_sharey=False, despine=True, corner=True)

    # grid.set(ylim=(-1, 12))
    # grid.set(xlim=(-1, 12))

    for ax in grid.axes.flat:
      if ax == None:
          continue
      ax.tick_params(axis='both',  width=1, color='0.2', size=12, labelsize=14)

      for axis in ['bottom', 'left']:
          ax.spines[axis].set_linewidth(1)
          ax.spines[axis].set_color('0.2')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)


    # grid.map_lower(sns.kdeplot,  alpha=0.8, linewidths=1.5,
    #                     fill=False, legend=False, common_norm=False, thresh=0.1)
    grid.map_offdiag(sns.scatterplot, alpha=0.8, linewidths=1.5, s=120,
                        legend=False)

    grid.map_diag(sns.kdeplot, alpha=0.6, linewidths=1.5,
                        fill=True, legend=False, common_norm=False, thresh=0.1);

    xlabels = [ax.xaxis.get_label_text() for ax in grid.axes[-1, :]]
    ylabels = [ax.yaxis.get_label_text() for ax in grid.axes[:, 0]]

    for i, xlabel in enumerate(xlabels):
        for j, ylabel in enumerate(ylabels):
            try:
                grid.axes[j, i].set_xlabel(xlabel, visible=True, fontsize=16, weight='bold', color='0.2')
                grid.axes[j, i].set_ylabel(ylabel, visible=True, fontsize=16, weight='bold', color='0.2')
                #grid.axes[j, i].set_xlabel(visible=True, fontsize=12, weight='bold', color='0.2')
                #grid.axes[j, i].set_ylabel(visible=True, fontsize=12, weight='bold', color='0.2')
                grid.axes[j, i].set_xlim(  xmin=-1, xmax=math.ceil(np.max(patient_response[xlabel])) + 1  )
                grid.axes[j, i].set_ylim(  ymin=-1, ymax=math.ceil(np.max(patient_response[ylabel])) + 1  )
            except:
                continue

    #plt.xticks(fontsize=8, color='0.2', weight='bold')
    #plt.yticks(fontsize=8, color='0.2', weight='bold')
    plt.tight_layout()

    plt.savefig(path+'/patient space (average first)/' + 'response map pair plot_%s_%s.png'%(x_name,y_name), dpi=300)

    if not os.path.isdir(path+'/patient space (average first)/' + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path+'/patient space (average first)/' + 'svg/')
    plt.savefig(path+'/patient space (average first)/' + 'svg/response map pair plot_%s_%s.svg'%(x_name,y_name))
    plt.clf()
    plt.close()



    for response in ['DNA response', 'IL6 response', 'LPS response']:
        dict_patient_response = {}
        for type in np.unique(patient_response['Type']):
            df_part = patient_response[patient_response['Type']==type][response]
            dict_patient_response[type] = df_part.values[~np.isnan(df_part.values)]  # Remove nan values
        new_order = ['Young', 'Old', 'Frail']
        dict_patient_response = change_dict_order(dict_patient_response, new_order)
        draw_custom_bar_plot(dict_patient_response, path+'/patient space (average first)/', file_name='%s bar plot_%s_%s'%(response, x_name, y_name),
                                 strip_plot=True, colors=('#7fc97f', '#beaed4', '#fdc086'), test='kruskal-wallis_dunn', pvalue=True, figsize=(1, 2))



#################################### differential PCA by patient-wise (average motility features -> PCA) ###################################
df.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df.columns[8:103].drop(['speed_distribution_x', 'speed_distribution_y',
                                            'inst_speed_cosine_similarity_entropies', 'inst_angle_cosine_similarity_entropies'])

diff_pca_inputs = pd.DataFrame()

for patient in np.unique(df['Patient']):
    df_patient = df[(df['Patient']==patient)].reset_index(drop=True)
    for cond in np.unique(df_patient['Condition']):
        if cond == 'Control':
            continue
        df_ref = df_patient[(df_patient['Condition']=='Control')].reset_index(drop=True)
        if df_ref.shape[0] == 0:
            continue
        df_part = df_patient[(df_patient['Condition']==cond)].reset_index(drop=True)

        motility_input = df_part[feature_list]
        motility_input_ref = df_ref[feature_list]

        pca_input = motility_input.median(axis=0).to_frame().T
        pca_input_ref = motility_input_ref.median(axis=0).to_frame().T

        diff_pca_input = pca_input - pca_input_ref
        diff_pca_input['Patient'] = np.unique(df_part['Patient'])[0]
        diff_pca_input['Type'] = np.unique(df_part['Type'])[0]
        diff_pca_input['Condition'] = np.unique(df_part['Condition'])[0]

        diff_pca_inputs = pd.concat([diff_pca_inputs, diff_pca_input], axis=0)

diff_pca_inputs = diff_pca_inputs.reset_index(drop=True)

diff_pca_inputs_only = diff_pca_inputs.iloc[:, :-3]
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
#aprw_data_scaled = pd.DataFrame(scaler.fit_transform( np.log10( aprw_data+abs(min(aprw_data.min()))+1e-10 ) ), columns=aprw_data.columns)
diff_pca_inputs_only= pd.DataFrame(scaler.fit_transform( diff_pca_inputs_only ), columns=diff_pca_inputs_only.columns)

from sklearn.decomposition import PCA
pca = PCA(n_components=4)

pcs_array = pca.fit_transform(diff_pca_inputs_only)  # factor scores for non-rotated data
df_pcs = pd.DataFrame(pcs_array, columns=['PC1', 'PC2', 'PC3', 'PC4'])

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'])
#loadings = pd.concat([df_title, loadings], axis=1)

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC1', 'PC2', 'PC3', 'PC4'])

patient_data = pd.concat([diff_pca_inputs, df_pcs], axis=1)

x_name='PC1'
y_name='PC2'
for cond in np.unique(patient_data['Condition']):
    draw_umap_space(patient_data[patient_data['Condition'] == cond].reset_index(drop=True), path+'/patient space (average first)/',
                    file_name='%s differential patient space %s %s' % (cond, x_name, y_name),
                    condition_name='Type', label_name=None, colors=('#fdc086', '#beaed4', '#7fc97f'), dot_size=5,
                    x_name=x_name, y_name=y_name)




#################################### PCA by patient-wise (PCA of motility features -> average PCs) ###################################
df.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df.columns[8:103].drop(['speed_distribution_x', 'speed_distribution_y',
                                            'inst_speed_cosine_similarity_entropies', 'inst_angle_cosine_similarity_entropies'])

pca_inputs_only = df[feature_list]
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
pca_inputs_only= pd.DataFrame(scaler.fit_transform( pca_inputs_only ), columns=pca_inputs_only.columns)

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pcs_array = pca.fit_transform(pca_inputs_only)  # factor scores for non-rotated data
df_pcs = pd.DataFrame(pcs_array, columns=['PC1', 'PC2', 'PC3', 'PC4'])

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'])
#loadings = pd.concat([df_title, loadings], axis=1)

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC1', 'PC2', 'PC3', 'PC4'])

df_preprocess = pd.concat([df_pcs, df[['Patient', 'Type', 'Condition', 'Age', 'kmeans', 'Weakness', 'Weight_loss', 'Exhaustion', 'Activity', 'Gait', 'Grip', 'Frailty_score']]], axis=1)


df_patient = pd.DataFrame()

for patient in np.unique(df_preprocess['Patient']):
    each_patient = df_preprocess[(df_preprocess['Patient']==patient)].reset_index(drop=True)
    for cond in np.unique(each_patient['Condition']):
        df_part = each_patient[(each_patient['Condition']==cond)].reset_index(drop=True)
        motility_input = df_part[['PC1', 'PC2', 'PC3', 'PC4']]
        df_patient_each = motility_input.median(axis=0).to_frame().T
        df_patient_each['Patient'] = np.unique(df_part['Patient'])[0]
        df_patient_each['Type'] = np.unique(df_part['Type'])[0]
        df_patient_each['Condition'] = np.unique(df_part['Condition'])[0]
        df_patient_each['Age'] = np.unique(df_part['Age'])[0]

        df_patient_each['kmeans'] = np.median(df_part['kmeans'])
        df_patient_each['Weakness'] = np.unique(df_part['Weakness'])[0]
        df_patient_each['Weight_loss'] = np.unique(df_part['Weight_loss'])[0]
        df_patient_each['Exhaustion'] = np.unique(df_part['Exhaustion'])[0]
        df_patient_each['Activity'] = np.unique(df_part['Activity'])[0]
        df_patient_each['Gait'] = np.unique(df_part['Gait'])[0]
        df_patient_each['Grip'] = np.unique(df_part['Grip'])[0]
        df_patient_each['Frailty_score'] = np.unique(df_part['Frailty_score'])[0]

        df_patient = pd.concat([df_patient, df_patient_each], axis=0)

df_patient = df_patient.reset_index(drop=True)


x_name='PC1'
y_name='PC2'
for cond in np.unique(df_patient['Condition']):
    draw_umap_space(df_patient[df_patient['Condition'] == cond].reset_index(drop=True), path,
                    file_name='%s PCA first patient space' % (cond),
                    condition_name='Type', label_name=None, colors=('#fdc086', '#beaed4', '#7fc97f'), dot_size=5,
                    x_name=x_name, y_name=y_name)


x_name='PC1'
y_name='PC2'
xlabel = '%s('%x_name + str(round(variance['%s'%x_name][1] * 100, ndigits=1)) + '%)'
ylabel = '%s('%y_name + str(round(variance['%s'%y_name][1] * 100, ndigits=1)) + '%)'
draw_diff_arrow_scatter(df_patient, path,
                        file_name='%s %s PCA first differential response patients'%(x_name, y_name), condition_name='Type',
                        diff_condition_name='Condition', ind_name='Patient', ref='Control', colors=('#fdc086', '#beaed4', '#7fc97f'),
                        dot_size=5, x_name=x_name, y_name=y_name, xlabel=xlabel, ylabel=ylabel)


x_name='PC1'
y_name='PC2'
diff_condition_name='Condition'
ind_name='Patient'
ref='Control'
condition_name='Type'
colors=('#fdc086', '#beaed4', '#7fc97f')
dot_size = 5
file_name='%s %s PCA first differential response patients'%(x_name, y_name)

#df_patient = df_patient[df_patient['Type']!='Young'].reset_index(drop=True)
df_ref = df_patient[df_patient[diff_condition_name] == ref].reset_index(drop=True)

n_colors = np.unique(df_patient[condition_name]).shape[0]
from collections.abc import Iterable
if isinstance(colors, Iterable):
    cmap = colors
else:
    cmap = [colors(1. * i / n_colors) for i in range(n_colors)]

xmin = math.floor(df_patient[x_name].min()) - 1
xmax = math.ceil(df_patient[x_name].max()) + 1
ymin = math.floor(df_patient[y_name].min()) - 1
ymax = math.ceil(df_patient[y_name].max()) + 1

font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

for cond in np.unique(df_patient[diff_condition_name]):
    if cond == ref:
        continue
    fig, ax = plt.subplots(figsize=(2, 2))
    df_part = df_patient[df_patient[diff_condition_name] == cond].reset_index(drop=True)

    for ind in np.unique(df_part[ind_name]):  # For patients within condition
        df_ref_ind = df_ref[df_ref[ind_name] == ind].reset_index(drop=True)
        df_cond_ind = df_part[df_part[ind_name] == ind].reset_index(drop=True)

        if (df_ref_ind.shape[0] == 0) or (df_cond_ind.shape[0] == 0):  # Test whether reference condition has this patient
            continue

        if np.unique(df_cond_ind['Type']) == 'Young':
            color=cmap[2]
        elif np.unique(df_cond_ind['Type']) == 'Old':
            color=cmap[1]
        elif np.unique(df_cond_ind['Type']) == 'Frail':
            color =cmap[0]
        scatter = ax.scatter(df_ref_ind[x_name], df_ref_ind[y_name],
                   # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
                   s=dot_size, label=df_ref_ind[condition_name], alpha=0.5,  # linestyle='dotted',
                   color=color)

        scatter = ax.scatter(df_cond_ind[x_name], df_cond_ind[y_name],
                   # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
                   s=dot_size, label=df_cond_ind[condition_name],
                   color=color)

        colors = np.array(colors)
        arrow_color = colors[np.isin(np.unique(df[condition_name]), df_cond_ind[condition_name])][0]
        ax.quiver(df_ref_ind[x_name], df_ref_ind[y_name], df_cond_ind[x_name] - df_ref_ind[x_name],
                  df_cond_ind[y_name] - df_ref_ind[y_name],
                  scale_units='xy', angles='xy', scale=1, color=arrow_color, headwidth=6, headlength=5)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    format_figure(ax, title=None, xlabel=xlabel, ylabel=ylabel, despine=True, detick=True)
    handles, labels = scatter.legend_elements(num=None)
    # plt.legend(handles=handles, labels=list(np.unique(df_part[condition_name])),
    #            bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
    #            fontsize=3, frameon=False, markerscale=0.3)

    # bbox_to_anchor is position of labels (x, y) (increasing x moves right, increasing y moves top)
    # frameon=False removes bounding box around label
    # font size adjust size of letter
    # markerscale adjust size of marker
    # plt.show()
    # plt.clf()
    # plt.close()

    plt.savefig(path + '%s %s.png' % (cond, file_name), dpi=300)

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s %s.svg' % (cond, file_name))
    plt.clf()
    plt.close()

#################################### x-axis: PC1 of motility, y-axis: magnitude of change by perturbation ###################################

diff_condition_name='Condition'
ind_name='Patient'
ref='Control'
condition_name='Type'
colors=('#fdc086', '#beaed4', '#7fc97f')
dot_size = 5
file_name='%s %s Motility over perturbation response'%(x_name, y_name)

df_ref = df_patient[df_patient[diff_condition_name] == ref].reset_index(drop=True)

n_colors = np.unique(df_patient[condition_name]).shape[0]
from collections.abc import Iterable
if isinstance(colors, Iterable):
    cmap = colors
else:
    cmap = [colors(1. * i / n_colors) for i in range(n_colors)]

xmin = math.floor(df_patient[x_name].min()) - 1
xmax = math.ceil(df_patient[x_name].max()) + 1
ymin = math.floor(df_patient[y_name].min()) - 1
ymax = math.ceil(df_patient[y_name].max()) + 1

font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

for cond in np.unique(df_patient[diff_condition_name]):
    if cond == ref:
        continue
    fig, ax = plt.subplots(figsize=(2, 2))
    df_part = df_patient[df_patient[diff_condition_name] == cond].reset_index(drop=True)

    for ind in np.unique(df_part[ind_name]):  # For patients within condition
        df_ref_ind = df_ref[df_ref[ind_name] == ind].reset_index(drop=True)
        df_cond_ind = df_part[df_part[ind_name] == ind].reset_index(drop=True)

        if (df_ref_ind.shape[0] == 0) or (df_cond_ind.shape[0] == 0):  # Test whether reference condition has this patient
            continue

        if np.unique(df_cond_ind['Type']) == 'Young':
            color=cmap[2]
        elif np.unique(df_cond_ind['Type']) == 'Old':
            color=cmap[1]
        elif np.unique(df_cond_ind['Type']) == 'Frail':
            color =cmap[0]

        transition = variance['%s'%x_name][1]*(df_cond_ind[x_name] - df_ref_ind[x_name])**2 + \
                     variance['%s'%y_name][1]*(df_cond_ind[y_name] - df_ref_ind[y_name])**2
        # transition = (df_cond_ind[x_name] - df_ref_ind[x_name]) ** 2 + \
        #              (df_cond_ind[y_name] - df_ref_ind[y_name]) ** 2
        transition = np.sqrt(transition)
        scatter = ax.scatter(df_ref_ind[x_name], transition,
                   # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
                   s=dot_size, label=df_ref_ind[condition_name], alpha=0.5,  # linestyle='dotted',
                   color=color)


    # plt.xlim(xmin, xmax)
    # plt.ylim(ymin, ymax)

    format_figure(ax, title=None, xlabel='Motility '+xlabel, ylabel='Response', despine=True, detick=True)
    handles, labels = scatter.legend_elements(num=None)
    # plt.legend(handles=handles, labels=list(np.unique(df_part[condition_name])),
    #            bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
    #            fontsize=3, frameon=False, markerscale=0.3)

    # bbox_to_anchor is position of labels (x, y) (increasing x moves right, increasing y moves top)
    # frameon=False removes bounding box around label
    # font size adjust size of letter
    # markerscale adjust size of marker
    # plt.show()
    # plt.clf()
    # plt.close()

    plt.savefig(path + '%s %s.png' % (cond, file_name), dpi=300)

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s %s.svg' % (cond, file_name))
    plt.clf()
    plt.close()


#################################### dataframe for magnitude of change ###################################

x_name='PC1'
y_name='PC2'
patient_response = pd.DataFrame()
for patient in np.unique(df_patient['Patient']):  # For patients within condition
    df_part = df_patient[(df_patient['Patient'] == patient)].reset_index(drop=True)
    df_ref_ind = df_part[df_part['Condition'] == 'Control'].reset_index(drop=True)
    if df_ref_ind.shape[0] == 0:
        continue

    temp = pd.DataFrame()
    for cond in np.unique(df_part['Condition']):
        if cond == 'Control':
            continue
        df_cond_ind = df_part[df_part['Condition'] == cond].reset_index(drop=True)

        transition = variance['%s' % x_name][1] * (df_cond_ind[x_name] - df_ref_ind[x_name]) ** 2 + \
                     variance['%s' % y_name][1] * (df_cond_ind[y_name] - df_ref_ind[y_name]) ** 2
        # transition = (df_cond_ind[x_name] - df_ref_ind[x_name]) ** 2 + \
        #              (df_cond_ind[y_name] - df_ref_ind[y_name]) ** 2
        # transition = (df_cond_ind[x_name] - df_ref_ind[x_name]) ** 2 + \
        #              (df_cond_ind[y_name] - df_ref_ind[y_name]) ** 2 + \
        #              (df_cond_ind['PC3'] - df_ref_ind['PC3']) ** 2 + \
        #              (df_cond_ind['PC4'] - df_ref_ind['PC4']) ** 2
        # transition = variance['%s' % x_name][1] * (df_cond_ind[x_name] - df_ref_ind[x_name]) ** 2 + \
        #              variance['%s' % y_name][1] * (df_cond_ind[y_name] - df_ref_ind[y_name]) ** 2 + \
        #              variance['%s' % 'PC3'][1] * (df_cond_ind['PC3'] - df_ref_ind['PC3']) ** 2 + \
        #              variance['%s' % 'PC4'][1] * (df_cond_ind['PC4'] - df_ref_ind['PC4']) ** 2

        transition = np.sqrt(transition)
        temp = pd.concat([temp, pd.DataFrame(transition, columns=['%s response'%cond])], axis=1)
    temp['Type'] = np.unique(df_part['Type'])
    temp['Patient'] = np.unique(df_part['Patient'])

    temp['Age'] = np.unique(df_part['Age'])
    #temp['kmeans'] = np.unique(df_part['kmeans'])
    temp['Weakness'] = np.unique(df_part['Weakness'])
    temp['Weight_loss'] = np.unique(df_part['Weight_loss'])
    temp['Exhaustion'] = np.unique(df_part['Exhaustion'])
    temp['Activity'] = np.unique(df_part['Activity'])
    temp['Gait'] = np.unique(df_part['Gait'])
    temp['Grip'] = np.unique(df_part['Grip'])
    temp['Frailty_score'] = np.unique(df_part['Frailty_score'])

    patient_response = pd.concat([patient_response, temp], axis=0)
patient_response = patient_response.reset_index(drop=True)



#################################### age vs response ###################################
linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = np.unique(df['Type']).size
ncols = 3

if not os.path.isdir(path + 'patient by patient/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'patient by patient/')

for feature in ['Age']:

    from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig,axes = plt.subplots(nrows, ncols, figsize=(15, 10), sharey='row')
    for row, type in enumerate(['Young', 'Old', 'Frail']):
        for col, cond in enumerate(['DNA response', 'IL6 response', 'LPS response']):
            ax = axes[row][col]
            df_part_= patient_response[ (patient_response['Type']==type)].reset_index(drop=True)
            df_part_ = df_part_[~df_part_[cond].isnull()].reset_index(drop=True)
            df_part_ = df_part_[~df_part_[feature].isnull()].reset_index(drop=True)
            sns.regplot(x=feature, y=cond, data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

            r, p = scipy.stats.pearsonr(df_part_[feature], df_part_[cond])
            if type == 'Young':
                plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
                plt.text(0.8, 0.85, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
            else:
                plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
                plt.text(0.1, 0.85, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
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

            ax.set_xlabel('%s'%feature, fontsize=10, weight='bold', color='0.2', labelpad=5)
            ax.set_ylabel('%s'%cond, fontsize=10, weight='bold', color='0.2', labelpad=5)
            ax.set_xlim(16, 98)

    plt.savefig(path + 'patient by patient/%s regplot.png'%(feature), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/patient by patient/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/patient by patient/')
    plt.savefig(path + 'svg/patient by patient/%s regplot.svg'%(feature), bbox_inches='tight')
    plt.clf()
    plt.close()


#################################### age vs response ###################################
linewidth = 1.5
fontsize = 16
width=6
ratio=5
space=0.2
nrows = 2
ncols = 3

if not os.path.isdir(path + 'patient by patient/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'patient by patient/')

for feature in ['Gait', 'Grip', 'Frailty_score']:

    from matplotlib.ticker import MaxNLocator
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig,axes = plt.subplots(nrows, ncols, figsize=(15, 10), sharey='row')
    for row, type in enumerate(['Old', 'Frail']):
        for col, cond in enumerate(['DNA response', 'IL6 response', 'LPS response']):
            ax = axes[row][col]
            df_part_= patient_response[ (patient_response['Type']==type)].reset_index(drop=True)
            df_part_ = df_part_[~df_part_[cond].isnull()].reset_index(drop=True)
            df_part_ = df_part_[~df_part_[feature].isnull()].reset_index(drop=True)
            sns.regplot(x=feature, y=cond, data=df_part_, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

            r, p = scipy.stats.pearsonr(df_part_[feature], df_part_[cond])
            if type == 'Young':
                plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
                plt.text(0.8, 0.85, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
            else:
                plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                         fontsize=12, fontdict={'weight': 'bold'}, color="black")
                plt.text(0.1, 0.85, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
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

            ax.set_xlabel('%s'%feature, fontsize=10, weight='bold', color='0.2', labelpad=5)
            ax.set_ylabel('%s'%cond, fontsize=10, weight='bold', color='0.2', labelpad=5)
            if feature == 'Gait':
                ax.set_xlim(2.8, 9.5)
            elif feature == 'Grip':
                ax.set_xlim(9, 47)
            if feature == 'Frailty_score':
                ax.set_xlim(-1, 6)

    plt.savefig(path + 'patient by patient/%s regplot.png'%(feature), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/patient by patient/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/patient by patient/')
    plt.savefig(path + 'svg/patient by patient/%s regplot.svg'%(feature), bbox_inches='tight')
    plt.clf()
    plt.close()


#################################### perturbation map ###################################
fig, ax = plt.subplots(figsize=(2,2))
grid = sns.PairGrid(data=patient_response, vars=['DNA response', 'IL6 response', 'LPS response'],
                    height=4, aspect=1, hue='Type', hue_order=['Frail', 'Old', 'Young'], palette=('#fdc086', '#beaed4', '#7fc97f'),
                    diag_sharey=False, despine=True, corner=True)
grid.set(ylim=(-1, 12))
grid.set(xlim=(-1, 12))

for ax in grid.axes.flat:
  if ax == None:
      continue
  ax.tick_params(axis='both',  width=1, color='0.2', size=6, labelsize=10)

  for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(1)
      ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)


# grid.map_lower(sns.kdeplot,  alpha=0.8, linewidths=1.5,
#                     fill=False, legend=False, common_norm=False, thresh=0.1)
grid.map_offdiag(sns.scatterplot, alpha=0.8, linewidths=1.5, s=120,
                    legend=False)

grid.map_diag(sns.kdeplot, alpha=0.8, linewidths=1.5,
                    fill=True, legend=False, common_norm=False, thresh=0.1);

xlabels = [ax.xaxis.get_label_text() for ax in grid.axes[-1, :]]
ylabels = [ax.yaxis.get_label_text() for ax in grid.axes[:, 0]]

for i, xlabel in enumerate(xlabels):
    for j, ylabel in enumerate(ylabels):
        try:
            grid.axes[j, i].set_xlabel(xlabel, visible=True, fontsize=12, weight='bold', color='0.2')
            grid.axes[j, i].set_ylabel(ylabel, visible=True, fontsize=12, weight='bold', color='0.2')
            #grid.axes[j, i].set_xlabel(visible=True, fontsize=12, weight='bold', color='0.2')
            #grid.axes[j, i].set_ylabel(visible=True, fontsize=12, weight='bold', color='0.2')
        except:
            continue

plt.tight_layout()

plt.savefig(path + 'response map pair plot.png', dpi=300)

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/response map pair plot.svg')
plt.clf()
plt.close()



for response in ['DNA response', 'IL6 response', 'LPS response']:
    dict_patient_response = {}
    for type in np.unique(patient_response['Type']):
        df_part = patient_response[patient_response['Type']==type][response]
        dict_patient_response[type] = df_part.values[~np.isnan(df_part.values)]  # Remove nan values
    new_order = ['Young', 'Old', 'Frail']
    dict_patient_response = change_dict_order(dict_patient_response, new_order)
    draw_custom_bar_plot(dict_patient_response, path, file_name='%s bar plot'%response,
                             strip_plot=True, colors=('#7fc97f', '#beaed4', '#fdc086'), test='kruskal-wallis_dunn', pvalue=True, figsize=(1, 2))

#################################### PCA by patient-wise (PCA of cluster distribution) ###################################
group_name = 'Patient'
batches = np.unique(df[group_name])

pca_inputs = pd.DataFrame()
for group in batches:
    df_part = df[(df[group_name]==group)].reset_index(drop=True)
    group_clone = pd.DataFrame(df_part.groupby(['Condition', 'kmeans']).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    group_clone_T = group_clone.T
    for cluster in list(np.unique(df['kmeans'])):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=[0] * np.unique(df_part['Condition']).size)
    pca_input = group_clone_T.reset_index()
    pca_input['Patient'] = np.unique(df_part['Patient'])[0]
    pca_input['Type'] = np.unique(df_part['Type'])[0]
    pca_inputs = pd.concat([pca_inputs, pca_input])


pca_inputs = pca_inputs.reset_index(drop=True)
pca_inputs_only = pca_inputs.iloc[:, 1:10]
scaler = StandardScaler()  # if not normalize, UMAP space is completely different
pca_inputs_only= pd.DataFrame(scaler.fit_transform( pca_inputs_only ), columns=pca_inputs_only.columns)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pcs_array = pca.fit_transform(pca_inputs_only)  # factor scores for non-rotated data
df_pcs = pd.DataFrame(pcs_array, columns=['PC1', 'PC2'])

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'])

#loadings = pd.concat([df_title, loadings], axis=1)

variance = pd.DataFrame(np.array(
            [pca.explained_variance_, pca.explained_variance_ratio_,
             np.cumsum(pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC1', 'PC2'])

df_patient = pd.concat([df_pcs, pca_inputs[['Patient', 'Type', 'Condition']]], axis=1)

x_name='PC1'
y_name='PC2'
for cond in np.unique(df_patient['Condition']):
    draw_umap_space(df_patient[df_patient['Condition'] == cond].reset_index(drop=True), path,
                    file_name='%s Cluster enrichment based patient space' % (cond),
                    condition_name='Type', label_name=None, colors=('#fdc086', '#beaed4', '#7fc97f'), dot_size=5,
                    x_name=x_name, y_name=y_name)


xlabel = '%s('%x_name + str(round(variance['%s'%x_name][1] * 100, ndigits=1)) + '%)'
ylabel = '%s('%y_name + str(round(variance['%s'%y_name][1] * 100, ndigits=1)) + '%)'
draw_diff_arrow_scatter(df_patient, path, file_name='Cluster enrichment based differential response patients', condition_name='Type',
                        diff_condition_name='Condition',ind_name='Patient', ref='Control', colors=('#fdc086', '#beaed4', '#7fc97f'),
                        dot_size=5, x_name=x_name, y_name=y_name, xlabel=xlabel, ylabel=ylabel)










df_patient = pd.DataFrame()

for patient in np.unique(df_preprocess['Patient']):
    each_patient = df_preprocess[(df_preprocess['Patient']==patient)].reset_index(drop=True)
    if 'Control' not in np.unique(each_patient['Condition']):
        continue
    group_clone = pd.DataFrame(each_patient.groupby(['Condition', 'kmeans']).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    group_clone_T = group_clone.T
    for cluster in list(pd.unique(df_preprocess['kmeans'])):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=[0] * np.unique(each_patient['Condition']).size)
    group_clone = group_clone_T.T

    df_corr = group_clone.corr()

    df_corr.columns.name = None  # Remove name of column = 'Type'
    df_corr.index.name = None  # Remove name of index = 'Type'

    for cond in np.unique(each_patient['Condition']):
        df_part = each_patient[(each_patient['Condition']==cond)].reset_index(drop=True)
        motility_input = df_part[['PC1', 'PC2', 'PC3', 'PC4']]
        df_patient_each = motility_input.median(axis=0).to_frame().T
        df_patient_each['Patient'] = np.unique(df_part['Patient'])[0]
        df_patient_each['Type'] = np.unique(df_part['Type'])[0]
        df_patient_each['Condition'] = np.unique(df_part['Condition'])[0]
        df_patient_each['Corr'] = df_corr['Control'][cond]
        df_patient = pd.concat([df_patient, df_patient_each], axis=0)

df_patient = df_patient.reset_index(drop=True)


x_name='PC1'
y_name='PC2'
for cond in np.unique(df_patient['Condition']):
    draw_umap_space(df_patient[df_patient['Condition'] == cond].reset_index(drop=True), path,
                    file_name='%s PCA first patient space' % (cond),
                    condition_name='Type', label_name=None, colors=('#fdc086', '#beaed4', '#7fc97f'), dot_size=5,
                    x_name=x_name, y_name=y_name)


#################################### Cross correlation of young_cond, old_cond, frail_cond ###################################

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

condition_name = 'Condition'
cluster_type = 'kmeans'

df_corr_data = pd.DataFrame()
group_clones=[]
for group in np.unique(df['Type']):
    corrcoef = []
    aaa = df[df['Type'] == group]

    group_clone = pd.DataFrame(aaa.groupby([condition_name, cluster_type]).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    group_clone_T = group_clone.T
    for cluster in list(pd.unique(df[cluster_type])):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=[0]*np.unique(df[condition_name]).size)
    group_clone = group_clone_T.T

    for column in group_clone.columns:
        group_clone.rename(columns={column:'%s '%group+column}, inplace=True)
    group_clones.append(group_clone)
    df_corr_data = pd.concat([df_corr_data, group_clone], axis=1)

df_corr = df_corr_data.corr()

df_corr.columns.name = None # Remove name of column = 'Type'
df_corr.index.name = None # Remove name of index = 'Type'


# rename_keys = {'wt_B-cell_far': 'wt GCB far', 'wt_B-cell_close': 'wt GCB close',
#                'mt_B-cell_far': 'mt GCB far', 'mt_B-cell_close': 'mt GCB close',}
# df_corr.rename(columns=rename_keys, inplace=True)
# df_corr.rename(index=rename_keys, inplace=True)

mask = np.triu(df_corr) # Mask for only lower triangle

mask = mask[1:, :-1]  # mask that doesn't contain very top left and very bottom right block
corr = df_corr.iloc[1:,:-1].copy()  # df that doesn't contain very top left and very bottom right block

fig, ax = plt.subplots(figsize=(4, 4))

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

plt.savefig(path+'Type_Perturbation correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/Type_Perturbation correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()


############# Plot Type_perturbation fraction of cluster for each cell type ###############
vmax=35
colors = ('#888888', '#CC6677', '#6699CC', '#44AA99',)

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

############# Total Shannon entropy of young, old, frail ###############

df_ = df.copy()
df_['type_condition'] = df['Type'].astype(str) + ' ' + df['Condition']

entropy, max_entropy = calculate_entropy(df_, df_, condition_name='type_condition', cluster_type='kmeans')

dict_datasets={}
for key in entropy:
    dict_datasets[key] = np.array([entropy[key]])

new_order = ['Young Control', 'Young DNA', 'Young IL6', 'Young LPS',
             'Old Control', 'Old DNA', 'Old IL6', 'Old LPS',
             'Frail Control', 'Frail DNA', 'Frail IL6', 'Frail LPS']
dict_datasets = change_dict_order(dict_datasets, new_order)

file_name='total entropy for all conditions'
test='mann-whitney'

colors = ('#7fc97f', '#7fc97f', '#7fc97f', '#7fc97f',
          '#beaed4', '#beaed4', '#beaed4', '#beaed4',
          '#fdc086', '#fdc086', '#fdc086', '#fdc086')


font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(4, 2))
sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

ax = sns.barplot(data=sorted_vals, capsize=0.5, edgecolor='0.2', lw=1, errwidth=1, palette=colors)
plot_params = {'edgecolor': '0.2', 'linewidth': 1, 'fc': 'none'}
#ax = sns.stripplot(data=sorted_vals, marker='s', s=1.5, **plot_params)
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
plt.xticks(np.arange(len(entropy.keys())), sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
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

############# Patient wise Shannon entropy of young, old, frail ###############

group_name = 'Patient'
batches = np.unique(df[group_name])

entropies_young = {'Control': [], 'DNA':[], 'IL6':[], 'LPS':[]}
for group in batches:
    df_part = df[(df[group_name]==group)&(df['Type'] == 'Young')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Condition', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_young[type].append(entropy[type])

entropies_old = {'Control': [], 'DNA':[], 'IL6':[], 'LPS':[]}
for group in batches:
    df_part = df[(df[group_name]==group)&(df['Type'] == 'Old')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Condition', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_old[type].append(entropy[type])

entropies_frail = {'Control': [], 'DNA':[], 'IL6':[], 'LPS':[]}
for group in batches:
    df_part = df[(df[group_name]==group)&(df['Type'] == 'Frail')]
    if df_part.shape[0] == 0:
        continue
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Condition', cluster_type='kmeans')
    for type in entropy:
        if entropy[type]==0:
            continue
        else:
            entropies_frail[type].append(entropy[type])

entropies = {}
for cell_type in ['Control', 'DNA', 'IL6', 'LPS']:
    entropies = {'Young %s'%cell_type:entropies_young[cell_type], 'Old %s'%cell_type:entropies_old[cell_type], 'Frail %s'%cell_type:entropies_frail[cell_type]}

    # rename_keys = {'wt_B-cell_DZ': 'wt GCB DZ', 'wt_B-cell_sLZ': 'wt GCB sLZ', 'wt_B-cell_dLZ': 'wt GCB dLZ',
    #                'mt_B-cell_DZ': 'mt GCB DZ', 'mt_B-cell_sLZ': 'mt GCB sLZ', 'mt_B-cell_dLZ': 'mt GCB dLZ', }
    #
    # entropies_ = {rename_keys.get(k, k): v for (k, v) in entropies.items()}
    draw_custom_bar_plot(entropies, path, file_name='entropy of Type perturbation for %s' %cell_type, colors=('#7fc97f', '#beaed4', '#fdc086'),
                         strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))

#################################### Perturbation effect on each patients ####################################

if not os.path.isdir(path + 'old patient wise perturbation umap/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'old patient wise perturbation umap/')

# df.columns.get_loc('inst_angle_symbolic_dynamic_entropies')
color_list = ('#888888', '#CC6677', '#44AA99', '#6699CC')
patients = np.unique(df_old['Patient'])
for patient in patients:
    df_part = df_old[df_old['Patient']==patient]
    draw_contour(df_part, path+'old patient wise perturbation umap/', file_name='%s'%patient, condition_name='Media', colors=color_list, x_name='PC1', y_name='PC2', bin_num=50, num_contours=3)

#################################### Perturbation effect on each patients ####################################

if not os.path.isdir(path + 'young patient wise perturbation umap/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'young patient wise perturbation umap/')

# df.columns.get_loc('inst_angle_symbolic_dynamic_entropies')
color_list = ('#888888', '#CC6677', '#44AA99', '#6699CC')
patients = np.unique(df_young['Patient'])
for patient in patients:
    df_part = df_young[df_young['Patient']==patient]
    draw_contour(df_part, path+'young patient wise perturbation umap/', file_name='%s'%patient, condition_name='Media', colors=color_list, x_name='PC1', y_name='PC2', bin_num=50, num_contours=3)


