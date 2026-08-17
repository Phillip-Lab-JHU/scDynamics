import scipy.stats
from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import ZoneSignal

#################################### Pull out interaction features ####################################
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
df.loc[(df['avg_zone'] < 0.8) & (df['avg_zone'] >= 0.4), 'Zone'] = 'DZ-sLZ'
df.loc[(df['avg_zone'] < 1.2) & (df['avg_zone'] >= 0.8), 'Zone'] = 'sLZ'
df.loc[(df['avg_zone'] < 1.6) & (df['avg_zone'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
df.loc[(df['avg_zone'] <= 2) & (df['avg_zone'] >= 1.6), 'Zone'] = 'dLZ'

for typ in ['wt_B-cell', 'mt_B-cell']:
    for zone in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
        print(typ, zone, df[(df['Type']==typ)&(df['Zone']==zone)].shape[0])


duration=20
label_expanded = np.repeat(df['Zone'], duration).reset_index(drop=True)
df_duration['Zone_label'] = label_expanded

df['type_zone'] = df['Type'].astype(str) + ' ' + df['Zone'].astype(str)
df['type_zone'].value_counts(normalize=True)
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\rebuttal\video3-4\\'


#################################### Box plot comparing all motility features by cell types ####################################
df = df[(df['Zone']=='dLZ')|(df['Zone']=='sLZ')|(df['Zone']=='sLZ-dLZ')].reset_index(drop=True)
df['Type'].value_counts()
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
condition_name = 'Type'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}

if not os.path.isdir(path + 'feature_box_plot_type_fdc/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_box_plot_type_fdc/')
if not os.path.isdir(path + 'feature_violin_plot_type_fdc/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_type_fdc/')
feature_list = ['FDC_avg_overlap', 'FDC_contact_times', 'FDC_contact_persistences']
for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }
    #
    draw_custom_violin_plot(dict_datasets, path+'feature_violin_plot_type_fdc/', file_name=feature_name, colors = ('#888888', '#CC6677'),
                            test='mann-whitney', pvalue=True, figsize=(1,2))

    draw_custom_box_plot(dict_datasets, path+ 'feature_box_plot_type_fdc/', file_name=feature_name, colors = ('#888888', '#CC6677'),
    strip_plot=False, test='t-test', pvalue=True, figsize=(1,2))



#################################### Box plot comparing all motility features by cell types ####################################
df = df[df['Video']=='4-Good-D10-B1-ZT1-45-132-FOV230-256px_Statistics'].reset_index(drop=True)

df['Type'].value_counts()
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
condition_name = 'Type'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'MT', 'wt_B-cell':'WT'}
if not os.path.isdir(path + 'feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_type/')

if not os.path.isdir(path + 'feature_box_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_box_plot_type/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell', 'mt_B-cell']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }
    #
    # draw_custom_violin_plot(dict_datasets, path+'feature_violin_plot_type/', file_name=feature_name, colors = ('#888888', '#CC6677'),
    #                         test='mann-whitney', pvalue=True, figsize=(1,2))

    draw_custom_box_plot(dict_datasets, path+ 'feature_box_plot_type/', file_name=feature_name, colors = ('#888888', '#CC6677'),
    strip_plot=False, test='t-test', pvalue=True, figsize=(1,2))



