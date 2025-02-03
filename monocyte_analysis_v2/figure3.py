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
"""Generates Data for Figure3. Spatiotemporal patterns for monocytes """

from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *

#################################### Load csv file ####################################
path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\\'

df = pd.read_parquet(path+'cleaned_all_features_30.parquet')
df_duration = pd.read_parquet(path+'cleaned_traj_duration_30.parquet')

df = df[df['Type']!='Prefrail'].reset_index(drop=True)
df_duration = df_duration[df_duration['Type']!='Prefrail'].reset_index(drop=True)

df = df[df['Age']!=55].reset_index(drop=True)
df_duration = df_duration[df_duration['Age']!=55].reset_index(drop=True)

df_ctrl = df[df['Condition']=='Control'].reset_index(drop=True)

df_young = df[df['Type']=='Young'].reset_index(drop=True)
df_old = df[df['Type']=='Old'].reset_index(drop=True)
df_frail = df[df['Type']=='Frail'].reset_index(drop=True)

path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis_v2\Figure3. Spatiotemporal pattern\\'

xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

young_path = path+'young/'
old_path = path+'old/'
frail_path = path+'frail/'


df.loc[(df['n_neighbors_average'] >= 9), 'Zone'] = 'close'
#df_part.loc[(df_part['n_neighbors_average'] < 6) & (df_part['n_neighbors_average'] > 0.5), 'Zone'] = 'medium'
df.loc[(df['n_neighbors_average'] <= 0.6), 'Zone'] = 'isolated'
for group in np.unique(df['Type']):
    df_part = df[df['Type']==group].reset_index(drop=True)
    for cond in np.unique(df_part['Condition']):
        for zone_typ in ['isolated', 'close']:
            df_part_part = df_part[(df_part['Condition']==cond)&(df_part['Zone']==zone_typ)].reset_index(drop=True)
            print(group, cond, zone_typ, df_part_part.shape[0], np.unique(df_part_part['Patient']).shape[0])

####################################### interaction features for all age groups+perturbation #############################################
if not os.path.isdir(path + 'int feature violin plot/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'int feature violin plot/')

df.columns.get_loc('nearest_approach_times')
df.columns.get_loc('diff_group_distance_autocorr_3')
feature_list = list(df.columns[133:187])

for feature_name in feature_list:
    dataset={}
    for age_group in np.unique(df['Type']):
        for condition in np.unique(df['Condition']):
            data = df[(df['Type'] == age_group)&(df['Condition']==condition)][feature_name]
            dataset[age_group+' '+condition] = np.array(data)

    values = flatten_nested_dict(dataset)
    if np.isnan(values).any() == True:  # Check at least one nan
        continue
    elif np.isfinite(values).all() == False:  # Check everything is not inf
        continue

    new_order = ['Young Control', 'Young DNA', 'Young IL6', 'Young LPS',
                 'Old Control', 'Old DNA', 'Old IL6', 'Old LPS',
                 'Frail Control', 'Frail DNA', 'Frail IL6', 'Frail LPS']
    dataset = change_dict_order(dataset, new_order)

    draw_custom_violin_plot(dataset, path + 'int feature violin plot/', file_name=feature_name,
                            colors=('#7fc97f', '#7fc97f', '#7fc97f', '#7fc97f', '#beaed4', '#beaed4', '#beaed4', '#beaed4',
                                    '#fdc086', '#fdc086', '#fdc086', '#fdc086',), test='mann-whitney', pvalue=False, figsize=(4, 4))

############# Plot isolated vs close jointplot for each cell type ###############
xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

for age_group in np.unique(df['Type']):
    for cond in np.unique(df['Condition']):
        df_part = df[(df['Type']==age_group)&(df['Condition']==cond)].reset_index(drop=True)

        df_part.loc[(df_part['n_neighbors_average'] >= 9), 'Zone'] = 'close'
        #df_part.loc[(df_part['n_neighbors_average'] < 6) & (df_part['n_neighbors_average'] > 0.5), 'Zone'] = 'medium'
        df_part.loc[(df_part['n_neighbors_average'] <= 0.6), 'Zone'] = 'isolated'
        df_part['Zone'] = df_part.Zone.astype(str)
        print(age_group, cond, df_part[df_part['Zone'] == 'close'].shape[0], df_part[df_part['Zone'] == 'isolated'].shape[0])

        draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s_%s_close vs far jointplot' % (age_group, cond), hue="Zone",
                       hue_order=['close', 'isolated'], colors=('#8cc5e3', '#f55f74'), fill=True, legend=False, thresh=0.2, height=4, ratio=5, space=0,
                       xlabels='UMAP1', ylabel='UMAP2', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


############# Plot young vs old vs frail in each isolated or close ###############
xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

df_ = df.copy()
df_['cell_density'] = 'medium'
df_.loc[df_['n_neighbors_average'] >= 9, 'cell_density'] = 'close'
df_.loc[df_['n_neighbors_average'] <= 0.6, 'cell_density'] = 'isolated'
df_ = df_[df_['cell_density']!='medium']

for cond in np.unique(df_['Condition']):
    for density in np.unique(df_['cell_density']):
        df_part = df_[(df_['Condition']==cond)&(df_['cell_density']==density)].reset_index(drop=True)

        print(cond, density, df_part[df_part['Type'] == 'Young'].shape[0], df_part[df_part['Type'] == 'Old'].shape[0],
              df_part[df_part['Type'] == 'Frail'].shape[0], )

        draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s_%s_age groups jointplot' % (cond, density), hue="Type",
                       hue_order=['Young', 'Old', 'Frail'], colors=('#7fc97f', '#beaed4', '#fdc086'), fill=True, legend=False, thresh=0.3, height=4, ratio=5, space=0,
                       xlabels='UMAP1', ylabel='UMAP2', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)



############# Plot isolated vs close Zone contour map for each cell type ###############
#features = ['n_neighbors_average', 'nearest_distance_average']

for age_group in np.unique(df_ctrl['Type']):
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    fig, ax = plt.subplots(figsize=(2, 2))  # 2 inch by 2 inch



    color_list = ['Greys', 'Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
                  'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'plasma']
    x_name='PC1'
    y_name='PC2'
    num_contours=6
    bin_num=50

    contours = []
    #groups = ['DZ', 'sLZ', 'dLZ']
    groups = ['close', 'far']

    # x_close = df_ctrl[(df_ctrl['Type']==age_group) & (df_ctrl['nearest_distance_average'] <= 30)][x_name]
    # y_close = df_ctrl[(df_ctrl['Type']==age_group) & (df_ctrl['nearest_distance_average'] <= 30)][y_name]
    #
    # x_medium = df_ctrl[(df_ctrl['Type'] == age_group) & (df_ctrl['nearest_distance_average'] < 130) & (df_ctrl['nearest_distance_average'] > 30)][x_name]
    # y_medium = df_ctrl[(df_ctrl['Type'] == age_group) & (df_ctrl['nearest_distance_average'] < 130) & (df_ctrl['nearest_distance_average'] > 30)][y_name]
    #
    # x_far = df_ctrl[(df_ctrl['Type'] == age_group) & (df_ctrl['nearest_distance_average'] >= 130)][x_name]
    # y_far = df_ctrl[(df_ctrl['Type'] == age_group) & (df_ctrl['nearest_distance_average'] >= 130)][y_name]

    x_close = df_ctrl[(df_ctrl['Type'] == age_group) & (df_ctrl['n_neighbors_average'] >= 9)][x_name]
    y_close = df_ctrl[(df_ctrl['Type'] == age_group) & (df_ctrl['n_neighbors_average'] >= 9)][y_name]

    x_medium = df_ctrl[(df_ctrl['Type'] == age_group) & (df_ctrl['n_neighbors_average'] < 9) & (df_ctrl['n_neighbors_average'] > 0.6)][x_name]
    y_medium = df_ctrl[(df_ctrl['Type'] == age_group) & (df_ctrl['n_neighbors_average'] < 9) & (df_ctrl['n_neighbors_average'] > 0.6)][y_name]

    x_far = df_ctrl[(df_ctrl['Type'] == age_group) & (df_ctrl['n_neighbors_average'] <= 0.6)][x_name]
    y_far = df_ctrl[(df_ctrl['Type'] == age_group) & (df_ctrl['n_neighbors_average'] <= 0.6)][y_name]


    print(age_group, x_close.shape, x_far.shape)
    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    kde_coordinate_close = np.vstack([x_close, y_close])  # shape = (2(dimension), number of points)
    kde_coordinate_medium = np.vstack([x_medium, y_medium])  # shape = (2(dimension), number of points)
    kde_coordinate_far = np.vstack([x_far, y_far])  # shape = (2(dimension), number of points)

    if (kde_coordinate_close.shape[1] <= 2) or (kde_coordinate_far.shape[1] <= 2):  # if there is only few points, it cannot calculate gaussian kde
        raise ValueError('Number of points should be greater than 2 to create contour')
    else:
        kde_close = scipy.stats.gaussian_kde(kde_coordinate_close)  # Define kernel (bandwidth by Scott's Rule)
        kde_medium = scipy.stats.gaussian_kde(kde_coordinate_medium)  # Define kernel (bandwidth by Scott's Rule)
        kde_far = scipy.stats.gaussian_kde(kde_coordinate_far)  # Define kernel (bandwidth by Scott's Rule)
        # evaluate on a regular grid
        xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
        ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        # Xgrid , Ygrid = (bin_num,bin_num) 2d array
        # Xgrid[i] = xgrid coordinate(divide by 100 from -x ~ +x` and repeat in row direction)
        # Ygrid[:,i] = ygrid coordinate(divide by 100 from -y ~ +y` and repeat in column direction)
        Z_close = kde_close.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z_medium = kde_medium.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z_far = kde_far.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        # Xgrid.ravel() = bin_num^2 shape 1d vector (vector that repeat -x ~ +x`, -x ~ +x` bin_num times)
        # np.vstack() = (2,bin_num^2) 2d array linspaced coordinates
        # Z = (10000,) 1d vector
        pdf_close = Z_close.reshape(Xgrid.shape)
        pdf_medium = Z_medium.reshape(Xgrid.shape)
        pdf_far = Z_far.reshape(Xgrid.shape)

        # contour_dz = ax.contour(Xgrid, Ygrid, pdf_dz,
        #                      # colors='red',
        #                      linewidths=1,
        #                      linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
        #                      # label=group,
        #                      cmap=color_list[0],
        #                      origin='lower',
        #                      levels=num_contours,
        #                      )
        contour_close = ax.contour(Xgrid, Ygrid, pdf_close,
                             # colors='red',
                             linewidths=1,
                             linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                             # label=group,
                             cmap=color_list[2],
                             origin='lower',
                             levels=num_contours,
                             )
        # contour_medium = ax.contour(Xgrid, Ygrid, pdf_medium,
        #                          # colors='red',
        #                          linewidths=1,
        #                          linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
        #                          # label=group,
        #                          cmap=color_list[1],
        #                          origin='lower',
        #                          levels=num_contours,
        #                          )
        contour_far = ax.contour(Xgrid, Ygrid, pdf_far,
                                # colors='red',
                                linewidths=1,
                                linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                                # label=group,
                                cmap=color_list[1],
                                origin='lower',
                                levels=num_contours,
                                )

        contours.append(contour_close)
        #contours.append(contour_medium)
        contours.append(contour_far)

    format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
    ax.legend(handles=[contour.legend_elements()[0][-1] for contour in contours], labels=groups,
              bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0, fontsize=3, frameon=False, markerscale=0.3)
    #plt.title('%s NOI vs PI' % cell_type, fontsize=25)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig(path + '%s_zones.png' % age_group, dpi=300)
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s_zones.svg' % age_group)
    plt.close()
    plt.clf()

############# Plot local cell density + age group kmeans distribution heatmap###############
for condition in np.unique(df['Condition']):
    df_part = df[df['Condition']==condition].reset_index(drop=True)

    df_ = df_part.copy()
    df_['cell_density'] = 'medium'
    df_.loc[df_['n_neighbors_average'] >= 9, 'cell_density'] = 'close'
    df_.loc[df_['n_neighbors_average'] <= 0.6, 'cell_density'] = 'isolated'
    df_ = df_[df_['cell_density']!='medium']

    df_['type_density'] = df_['Type'].astype(str) + ' ' + df_['cell_density'].astype(str)

    draw_cluster_distribution_heatmap(df_, path, file_name='%s_kmeans_heatmap'%condition, condition_name='type_density',
                                      metric='correlation', cmap=cmc.bilbao_r, vmax=35,
                                      cluster_type='kmeans', col_cluster=False, row_cluster=True, figsize=(4,3))

    draw_relative_cluster_distribution_heatmap(df_, path, file_name='relative_%s_kmeans_heatmap'%condition, col_cluster=False, row_cluster=True, metric='correlation',
                                      annot=False, vmax=40, cmap=cmc.oslo_r, transpose=False, condition_name='type_density', cluster_type='kmeans', figsize=(4,3))


for txt in ['young', 'old', 'frail']:
    if txt=='young':
        df_ = df_young.copy()
        path_ = young_path
    elif txt=='old':
        df_ = df_old.copy()
        path_ = old_path
    elif txt=='frail':
        df_ = df_frail.copy()
        path_ = frail_path

    for cond in np.unique(df_['Condition']):
        df_part = df_[df_['Condition']==cond].reset_index(drop=True)
        df_part['cell_density'] = 'medium'
        df_part.loc[df_part['n_neighbors_average'] >= 9, 'cell_density'] = 'close'
        df_part.loc[df_part['n_neighbors_average'] <= 0.6, 'cell_density'] = 'isolated'
        df_part = df_part[df_part['cell_density']!='medium']
        df_part['type_density'] = df_part['Type'].astype(str) + ' ' + df_part['cell_density'].astype(str)
        draw_relative_cluster_distribution_heatmap(df_part, path_, file_name='relative_%s_%s_kmeans_heatmap'%(cond, txt), col_cluster=False, row_cluster=False,
                                          annot=False, vmax=80, cmap=cmc.oslo_r, transpose=False, condition_name='type_density', cluster_type='kmeans', figsize=(4,3))

############# Plot local cell density + age group + perturbation kmeans distribution heatmap###############
df_ = df.copy()
df_['cell_density'] = 'medium'
#df_[df_['n_neighbors_average'] >= 6]['cell_density'] = 'close'
#df_[df_['n_neighbors_average'] == 0]['cell_density'] = 'isolated'
df_.loc[df_['n_neighbors_average'] >= 9, 'cell_density'] = 'close'
df_.loc[df_['n_neighbors_average'] <= 0.6, 'cell_density'] = 'isolated'
df_ = df_[df_['cell_density']!='medium']

df_['type_density'] = df_['Type'].astype(str) + '_' + df_['cell_density'].astype(str)
df_['type_condition_density'] = df_['Type'].astype(str) + ' ' + df_['Condition'].astype(str)+ ' ' + df_['cell_density'].astype(str)

draw_cluster_distribution_heatmap(df_, path, file_name='all_kmeans_heatmap', condition_name='type_condition_density', cmap=cmc.bilbao_r,
                                  cluster_type='kmeans', col_cluster=False, figsize=(4,8), vmax=40)

#################################### Cross correlation  ###################################



for cond in np.unique(df['Condition']):
    df_ = df[df['Condition']==cond].reset_index(drop=True)
    df_['cell_density'] = 'medium'
    df_.loc[df_['n_neighbors_average'] >= 9, 'cell_density'] = 'close'
    df_.loc[df_['n_neighbors_average'] <= 0.6, 'cell_density'] = 'isolated'
    df_ = df_[df_['cell_density']!='medium']
    df_['type_density'] = df_['Type'].astype(str) + ' ' + df_['cell_density'].astype(str)

    condition_name = 'type_density'
    cluster_type = 'kmeans'

    df_corr_data = pd.DataFrame()
    group_clones=[]

    group_clone = pd.DataFrame(df_.groupby([condition_name, cluster_type]).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    group_clone_T = group_clone.T
    for cluster in list(pd.unique(df[cluster_type])):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=[0]*np.unique(df_[condition_name]).size)
    group_clone = group_clone_T.T
    df_corr = group_clone.corr()

    df_corr.columns.name = None # Remove name of column = 'Type'
    df_corr.index.name = None # Remove name of index = 'Type'


    # rename_keys = {'wt_B-cell_far': 'wt GCB far', 'wt_B-cell_close': 'wt GCB close',
    #                'mt_B-cell_far': 'mt GCB far', 'mt_B-cell_close': 'mt GCB close',}
    # df_corr.rename(columns=rename_keys, inplace=True)
    # df_corr.rename(index=rename_keys, inplace=True)

    mask = np.triu(df_corr) # Mask for only lower triangle

    mask = mask[1:, :-1]  # mask that doesn't contain very top left and very bottom right block
    corr = df_corr.iloc[1:,:-1].copy()  # df that doesn't contain very top left and very bottom right block

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

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

    plt.savefig(path+'%s_Type_Colocalization correlation.png'%cond, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s_Type_Colocalization correlation.svg'%cond, bbox_inches='tight')

    plt.close()
    plt.clf()

#################################### For each perturbation motility features wrt Colocalization features ####################################

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list(df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']))
feature_list.append('Time_span')

coloc_features = ['n_neighbors_average', 'nearest_distance_average', 'group_distance_average']

for cond in np.unique(df['Condition']):
    df_cond = df[df['Condition']==cond].reset_index(drop=True)
    coloc_feature = coloc_features[0]
    draw_lineplot_by_custom_ranges(df_cond, path, folder_name='%s_motility_feature_wrt_%s'%(cond, coloc_feature), feature_list=feature_list,
                                   condition_name='Type', custsom_range=(0, 10), stepsize=1, range_feature=coloc_feature,
                                       color_list=['#fdc086', '#beaed4', '#7fc97f'], marker_list=['o', '^', '*'], figsize=(4,4), x_label='Number of neighbors',
                                   error_type='ci_norm', pvalue=False, replace_keys=None, set_zero=True)

    coloc_feature = coloc_features[0]
    draw_lineplot_by_custom_ranges(df_cond[df_cond['Type']!='Young'].reset_index(drop=True), path,
                                   folder_name='%s_no_young_motility_feature_wrt_%s'%(cond,coloc_feature), feature_list=feature_list,
                                   condition_name='Type', custsom_range=(0, 10), stepsize=1, range_feature=coloc_feature,
                                       color_list=['#fdc086', '#beaed4', '#7fc97f'], marker_list=['o', '^', '*'], figsize=(4,4), x_label='Number of neighbors',
                                   error_type='ci_norm', pvalue=False, replace_keys=None, set_zero=True)

    coloc_feature = coloc_features[1]
    draw_lineplot_by_custom_ranges(df_cond, path, folder_name='%s_motility_feature_wrt_%s'%(cond, coloc_feature), feature_list=feature_list,
                                   condition_name='Type', custsom_range=(10, 100), stepsize=10, range_feature=coloc_feature,
                                       color_list=['#fdc086', '#beaed4', '#7fc97f'], marker_list=['o', '^', '*'], figsize=(4,4), x_label='Distance to nearest (um)',
                                   error_type='ci_norm', pvalue=False, replace_keys=None, set_zero=True)

    coloc_feature = coloc_features[2]
    draw_lineplot_by_custom_ranges(df_cond, path, folder_name='%s_motility_feature_wrt_%s'%(cond, coloc_feature), feature_list=feature_list,
                                   condition_name='Type', custsom_range=(30, 200), stepsize=10, range_feature=coloc_feature,
                                       color_list=['#fdc086', '#beaed4', '#7fc97f'], marker_list=['o', '^', '*'], figsize=(4,4), x_label='Distance to groups (um)',
                                   error_type='ci_norm', pvalue=False, replace_keys=None, set_zero=True)


#################################### Correlation btw motility and Colocalization for each perturbation ####################################

#for feature_name in feature_list:
feature_name='avg_speed'
condition_name='Type'
custsom_range=(0, 10)
stepsize=1
range_feature='n_neighbors_average'
cond='Control'
df_cond = df[df['Condition']==cond].reset_index(drop=True)


for cell_type in np.unique(df_cond[condition_name]):
    df_part = df_cond[df_cond[condition_name] == cell_type].reset_index(drop=True)
    rs = []
    ps = []

    means=[]
    stds = []
    valuess = []

    for i in np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize):
        values = df_part[(df_part[range_feature] >= i) & (df_part[range_feature] < i + stepsize)][feature_name].values
        # if values.size==0:
        #     continue
        means.append(np.mean(values))
        stds.append(np.std(values))
        valuess.append(values)

    if np.isnan(means).any(): # Detect at least one nan value
        continue
    x = np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize)
    y = np.array(means)
    r, p = scipy.stats.spearmanr(x, y)
    # print(r, p)
    # rs.append(r)
    # ps.append(p)






#################################### Patient to Patient Correlation btw motility and Colocalization for each perturbation ####################################

#for feature_name in feature_list:
feature_name='progressivity'
condition_name='Type'
custsom_range=(0, 7)
stepsize=1
range_feature='n_neighbors_average'
cond='Control'
df_cond = df[df['Condition']==cond].reset_index(drop=True)

r_dataset = {}
p_dataset = {}
std_dataset = {}
for cell_type in np.unique(df_cond[condition_name]):
    df_part = df_cond[df_cond[condition_name] == cell_type].reset_index(drop=True)
    rs = []
    ps = []
    stds = []
    valuess = []

    for patient in np.unique(df_part['Patient']):
        df_patient = df_part[df_part['Patient']==patient].reset_index(drop=True)


        patient_means = []
        patient_stds = []
        patient_valuess = []

        for i in np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize):
            values = df_patient[(df_patient[range_feature] >= i) & (df_patient[range_feature] < i + stepsize)][feature_name].values
            # if values.size==0:
            #     continue
            patient_means.append(np.mean(values))
            patient_stds.append(np.std(values))
            patient_valuess.append(values)

        if np.isnan(patient_means).any(): # Detect at least one nan value
            continue
        x = np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize)
        y = np.array(patient_means)
        r, p = scipy.stats.spearmanr(x, y)
        print(patient, r, p)
        rs.append(r)
        ps.append(p)
        stds.append( np.mean(patient_stds) )

    r_dataset[cell_type] = rs
    p_dataset[cell_type] = ps
    std_dataset[cell_type] = stds

new_order = ['Young', 'Old', 'Frail']
r_dataset = change_dict_order(r_dataset, new_order)
std_dataset = change_dict_order(std_dataset, new_order)
# draw_custom_bar_plot(r_dataset, path, file_name='%s %s corr with n_neighbors bar plot'%(cond, feature_name),
#                          strip_plot=True, colors=('#7fc97f', '#beaed4', '#fdc086'), test='mann-whitney', pvalue=True, figsize=(1, 2))
draw_custom_bar_plot(std_dataset, path, file_name='%s %s std with n_neighbors bar plot'%(cond, feature_name),
                         strip_plot=True, colors=('#7fc97f', '#beaed4', '#fdc086'), test='mann-whitney', pvalue=True, figsize=(1, 2))

#################################### Young motility features wrt Colocalization features for each perturbation ####################################

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list(df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']))
feature_list.append('Time_span')

coloc_features = ['n_neighbors_average', 'nearest_distance_average', 'group_distance_average']

coloc_feature = coloc_features[0]
draw_lineplot_by_custom_ranges(df_young, young_path, folder_name='motility_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                               condition_name='Condition', custsom_range=(0, 9), stepsize=1, range_feature=coloc_feature,
                                   color_list=['#888888', '#CC6677', '#6699CC', '#44AA99',], marker_list=['o', '^', '*', 'x'], figsize=(4,4), x_label='Number of neighbors',
                               error_type='ci_norm', replace_keys=None, set_zero=True)

coloc_feature = coloc_features[1]
draw_lineplot_by_custom_ranges(df_young, young_path, folder_name='motility_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                               condition_name='Condition', custsom_range=(10, 100), stepsize=10, range_feature=coloc_feature,
                                   color_list=['#888888', '#CC6677', '#6699CC', '#44AA99',], marker_list=['o', '^', '*', 'x'], figsize=(4,4), x_label='Distance to closest (um)',
                               error_type='ci_norm', replace_keys=None, set_zero=True)

#################################### Old motility features wrt Colocalization features for each perturbation ####################################

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list(df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']))
feature_list.append('Time_span')

coloc_features = ['n_neighbors_average', 'nearest_distance_average', 'group_distance_average']

coloc_feature = coloc_features[0]
draw_lineplot_by_custom_ranges(df_old, old_path, folder_name='motility_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                               condition_name='Condition', custsom_range=(0, 9), stepsize=1, range_feature=coloc_feature,
                                   color_list=['#888888', '#CC6677', '#6699CC', '#44AA99',], marker_list=['o', '^', '*', 'x'], figsize=(4,4), x_label='Number of neighbors',
                               error_type='ci_norm', replace_keys=None, set_zero=True)

coloc_feature = coloc_features[1]
draw_lineplot_by_custom_ranges(df_old, old_path, folder_name='motility_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                               condition_name='Condition', custsom_range=(10, 100), stepsize=10, range_feature=coloc_feature,
                                   color_list=['#888888', '#CC6677', '#6699CC', '#44AA99',], marker_list=['o', '^', '*', 'x'], figsize=(4,4), x_label='Distance to closest (um)',
                               error_type='ci_norm', replace_keys=None, set_zero=True)

#################################### frail motility features wrt Colocalization features for each perturbation ####################################

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = list(df.columns[2:115].drop(['phi', 'speed_distribution_x', 'speed_distribution_y']))
feature_list.append('Time_span')

coloc_features = ['n_neighbors_average', 'nearest_distance_average', 'group_distance_average']

coloc_feature = coloc_features[0]
draw_lineplot_by_custom_ranges(df_frail, frail_path, folder_name='motility_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                               condition_name='Condition', custsom_range=(0, 9), stepsize=1, range_feature=coloc_feature,
                                   color_list=['#888888', '#CC6677', '#6699CC', '#44AA99',], marker_list=['o', '^', '*', 'x'], figsize=(4,4), x_label='Number of neighbors',
                               error_type='ci_norm', replace_keys=None, set_zero=True)

coloc_feature = coloc_features[1]
draw_lineplot_by_custom_ranges(df_frail, frail_path, folder_name='motility_feature_wrt_%s'%coloc_feature, feature_list=feature_list,
                               condition_name='Condition', custsom_range=(10, 100), stepsize=10, range_feature=coloc_feature,
                                   color_list=['#888888', '#CC6677', '#6699CC', '#44AA99',], marker_list=['o', '^', '*', 'x'], figsize=(4,4), x_label='Distance to closest (um)',
                               error_type='ci_norm', replace_keys=None, set_zero=True)