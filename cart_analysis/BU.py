"""Generates Data"""

from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast
from features.interaction import ZoneSignal

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\NiaLab_TcellTimelapse\\'
df = pd.read_csv(path+'motility_features.csv')
df_duration = pd.read_parquet(path + 'traj_duration_11.parquet')

path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\NiaLab_TcellTimelapse\analysis\\'
draw_umap_space(df, path, file_name='space_Type', condition_name='Type', label_name='pseudo_TrackID', colors=('#CC6677', '#888888'), x_name='PC1', y_name='PC2', dot_size=0.6)
draw_umap_space(df, path, file_name='space_kmeans', condition_name='kmeans', label_name='pseudo_particle', colors=cmc.batlow, x_name='PC1', y_name='PC2', dot_size=0.6)
draw_umap_space(df, path, file_name='space_exp', condition_name='Exp', label_name='pseudo_TrackID', colors=cmc.batlow, x_name='PC1', y_name='PC2', dot_size=0.6)


df.columns.get_loc('inst_angle_pulseindicator')
feature_list = df.columns[2:90].drop(['speed_distribution_x', 'speed_distribution_y'])
draw_space_feature_magnitude(df, path, feature_list, dot_size=0.6, x_name='PC1', y_name='PC2', vmax=None)

#################################### Heatmap of cluster enrichment and shannon entropy ####################################

draw_cluster_distribution_heatmap(df, path, file_name='kmeans_type_heatmap', condition_name='Type', cluster_type='kmeans',
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(4,2))

draw_cluster_distribution_heatmap(df, path, file_name='kmeans_video_heatmap', condition_name='Video', cluster_type='kmeans',
                                  annot=True, col_cluster=False, row_cluster=False,figsize=(4,4))

#################################### draw 3D trajectories by kmeans ####################################
for cluster in np.unique(df['kmeans']):
    print(cluster, df[df['kmeans']==cluster].shape[0])

draw_2D_trajectories_one_figure(df_duration, df, path, duration=11, n_examples=20, label_name='kmeans', feature_name=['x', 'y'], lim=100)

#################################### Box plot comparing all motility features by cell types ####################################
df.columns.get_loc('inst_angle_pulseindicator')
feature_list = df.columns[8:90].drop(['speed_distribution_x', 'speed_distribution_y'])
condition_name = 'kmeans'
#replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
if not os.path.isdir(path + 'feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_type/')

if not os.path.isdir(path + 'feature_box_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_box_plot_type/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    #new_order = ['wt_B-cell', 'mt_B-cell']
    #ordered_dataset = change_dict_order(dataset, new_order)
    #dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    draw_custom_violin_plot(dataset, path+'feature_violin_plot_type/', file_name=feature_name, colors = ('#888888', '#888888', '#888888', '#888888'),
                            test='mann-whitney', pvalue=True, figsize=(1,2))

    draw_custom_box_plot(dataset, path+ 'feature_box_plot_type/', file_name=feature_name, colors = ('#888888', '#888888', '#888888', '#888888'),
    strip_plot=True, test='mann-whitney', pvalue=True, figsize=(1,2))