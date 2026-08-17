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


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\rebuttal\day\\'
video_comp = (
    df[df['Type'].isin(['wt_B-cell', 'mt_B-cell'])]
    .groupby(['Video', 'Type'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)


video_comp['wt_count'] = video_comp.get('wt_B-cell', 0)
video_comp['mt_count'] = video_comp.get('mt_B-cell', 0)
video_comp['total_count'] = video_comp['wt_count'] + video_comp['mt_count']

video_comp['mt_fraction'] = video_comp['mt_count'] / video_comp['total_count']
video_comp['Ratio'] = video_comp['wt_count'] / video_comp['mt_count']

video_comp['Dominance'] = np.select(
    [
        video_comp['mt_fraction'] < 0.4,
        video_comp['mt_fraction'] > 0.6
    ],
    [
        'WT-dominant',
        'MT-dominant'
    ],
    default='Comparable'
)

df2 = df.merge(
    video_comp[['Video', 'wt_count', 'mt_count', 'mt_fraction', 'Ratio', 'Dominance']],
    on='Video',
    how='left'
)

video_counts_by_dominance = df2.groupby('Dominance')['Video'].nunique().reset_index(name='Video_count')

video_counts_by_day = df2.groupby('Day')['Video'].nunique().reset_index(name='Video_count')
video_counts = (
    df2.groupby(['Day', 'Dominance'])['Video']
    .nunique()
    .reset_index(name='Video_count')
    .sort_values(['Day', 'Dominance'])
)


df.columns.get_loc('inst_angle_cosine_similarity_entropies')
feature_list = df.columns[8:126].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z',])

video_type_features = (
    df2[df2['Type'].isin(['wt_B-cell', 'mt_B-cell'])]
    .groupby(['Video', 'Type', 'Dominance', 'Day', 'mt_fraction'])[feature_list]
    .median()
    .reset_index()
)


# ----------------------------------------------------------------------
df2['type_day'] = df2['Type'].astype(str) + ' ' + df2['Day'].astype(str)
df2['type_day'].value_counts()
replace_keys = {'wt_B-cell D9':'WT D9', 'mt_B-cell D9':'MT D9',
                'wt_B-cell D10':'WT D10', 'mt_B-cell D10':'MT D10',
                'wt_B-cell D11':'WT D11', 'mt_B-cell D11':'MT D11', }
for feature in feature_list:
    dataset={}
    for condition in np.unique(df2['type_day']):
        data = df2[(df2['type_day'] == condition)][feature] # 'FDC_contact_times', 'FDC_contact_persistences'
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell D9', 'mt_B-cell D9',
                 'wt_B-cell D10', 'mt_B-cell D10',
                 'wt_B-cell D11', 'mt_B-cell D11']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    os.makedirs(path+'/motility_feature_type_day_sc/', exist_ok=True)
    draw_custom_box_plot(dict_datasets, path+'motility_feature_type_day_sc/',
                         file_name='%s'%feature,
                         colors = ('#888888', '#CC6677')*3, strip_plot=False,
                                test='one-way anova_tukey', pvalue=True, figsize=(2,3))


# ----------------------------------------------------------------------
df2['type_day_dominance'] = df2['Type'].astype(str) + ' ' + df2['Day'].astype(str) + ' ' + df2['Dominance'].astype(str)
df2['type_day_dominance'].value_counts()

df2['type_day'] = df2['Type'].astype(str) + ' ' + df2['Day'].astype(str)

for dominance in np.unique(df2['Dominance']):
    df_part = df2[df2['Dominance']==dominance]
    replace_keys = {'wt_B-cell D9':'WT D9', 'mt_B-cell D9':'MT D9',
                    'wt_B-cell D10':'WT D10', 'mt_B-cell D10':'MT D10',
                    'wt_B-cell D11':'WT D11', 'mt_B-cell D11':'MT D11', }
    print(dominance, df_part['type_day'].value_counts())
    for feature in feature_list:
        dataset={}
        for condition in np.unique(df_part['type_day']):
            data = df_part[(df_part['type_day'] == condition)][feature] # 'FDC_contact_times', 'FDC_contact_persistences'
            dataset[condition] = np.array(data)
        new_order = ['wt_B-cell D9', 'mt_B-cell D9',
                     'wt_B-cell D10', 'mt_B-cell D10',
                     'wt_B-cell D11', 'mt_B-cell D11']
        ordered_dataset = change_dict_order(dataset, new_order)
        dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

        os.makedirs(path+'/motility_feature_type_day_sc_%s/'%dominance, exist_ok=True)
        draw_custom_box_plot(dict_datasets, path+'motility_feature_type_day_sc_%s/'%dominance,
                             file_name='%s'%feature,
                             colors = ('#888888', '#CC6677')*3, strip_plot=False,
                                    test='one-way anova_tukey', pvalue=True, figsize=(2,3))

# ----------------------------------------------------------------------

df2['type_dominance'] = df2['Type'].astype(str) + ' ' + df2['Dominance'].astype(str)
day = 'D11'
# for day in np.unique(df2['Day']):
df_part = df2[df2['Day']==day]
replace_keys = {'wt_B-cell WT-dominant': 'WT / WT Dominant', 'mt_B-cell WT-dominant': 'MT / WT Dominant',
                'wt_B-cell Comparable': 'WT / Comparable', 'mt_B-cell Comparable': 'MT / Comparable',
                'wt_B-cell MT-dominant': 'WT / MT Dominant', 'mt_B-cell MT-dominant': 'MT / MT Dominant', }
print(day, df_part['type_dominance'].value_counts())
for feature in feature_list:
    dataset={}
    for condition in np.unique(df_part['type_dominance']):
        data = df_part[(df_part['type_dominance'] == condition)][feature] # 'FDC_contact_times', 'FDC_contact_persistences'
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell WT-dominant', 'mt_B-cell WT-dominant',
                 'wt_B-cell Comparable', 'mt_B-cell Comparable',
                 'wt_B-cell MT-dominant', 'mt_B-cell MT-dominant']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    os.makedirs(path+'/motility_feature_type_dominance_sc_%s/'%day, exist_ok=True)
    draw_custom_box_plot(dict_datasets, path+'motility_feature_type_dominance_sc_%s/'%day,
                         file_name='%s'%feature,
                         colors = ('#888888', '#CC6677')*3, strip_plot=False,
                                test='one-way anova_tukey', pvalue=True, figsize=(2,3))

# ----------------------------------------------------------------------
video_type_features['type_day'] = video_type_features['Type'].astype(str) + ' ' + video_type_features['Day'].astype(str)
video_type_features['type_day'].value_counts()
replace_keys = {'wt_B-cell D9':'WT D9', 'mt_B-cell D9':'MT D9',
                'wt_B-cell D10':'WT D10', 'mt_B-cell D10':'MT D10',
                'wt_B-cell D11':'WT D11', 'mt_B-cell D11':'MT D11', }
for feature in feature_list:
    dataset={}
    for condition in np.unique(video_type_features['type_day']):
        data = video_type_features[(video_type_features['type_day'] == condition)][feature] # 'FDC_contact_times', 'FDC_contact_persistences'
        dataset[condition] = np.array(data)
    new_order = ['wt_B-cell D9', 'mt_B-cell D9',
                 'wt_B-cell D10', 'mt_B-cell D10',
                 'wt_B-cell D11', 'mt_B-cell D11']
    ordered_dataset = change_dict_order(dataset, new_order)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

    os.makedirs(path+'/motility_feature_type_day/', exist_ok=True)
    draw_custom_bar_plot(dict_datasets, path+'motility_feature_type_day/',
                         file_name='%s'%feature,
                         colors = ('#888888', '#CC6677')*3, strip_plot=True,
                                test='mann-whitney', pvalue=True, figsize=(2,3))



# ----------------------------------------------------------------------
for feature in feature_list:
    feature_diff = (
        video_type_features
        .pivot_table(
            index=['Video', 'Dominance', 'mt_fraction'],
            columns='Type',
            values=feature
        )
        .reset_index()
    )

    feature_diff['Difference'] = (
        feature_diff['mt_B-cell'] - feature_diff['wt_B-cell']
    )

    dict_datasets = {}

    for dom in ['WT-dominant', 'Comparable', 'MT-dominant']:
        dict_datasets[dom] = feature_diff.loc[
            feature_diff['Dominance'] == dom, 'Difference'
        ].dropna().tolist()

    os.makedirs(path+'/motility_feature_difference/', exist_ok=True)
    draw_custom_bar_plot(dict_datasets, path+'motility_feature_difference/',
                             file_name='%s'%feature,
                             colors = ('#888888', '#888888', '#888888'), strip_plot=True,
                                    test='mann-whitney', pvalue=True, figsize=(1,2))


    fig, ax = plt.subplots(figsize=(3, 3))

    sns.regplot(x='mt_fraction', y='Difference', data=feature_diff, scatter_kws={"color":"black", "alpha":0.7, 's':20}, line_kws={"color":"black"}, ax=ax)

    r, p = scipy.stats.pearsonr(feature_diff['mt_fraction'], feature_diff['Difference'])

    if r < 0:
        plt.text(0.8, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                 fontsize=12, fontdict={'weight': 'normal'}, color="black")
        plt.text(0.8, 0.85, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                 fontsize=12, fontdict={'weight': 'normal'}, color="black")
    elif r >=0 :
        plt.text(0.1, 0.95, "r = " + str(round(r, 2)), ha='left', va='top', transform=ax.transAxes,
                 fontsize=12, fontdict={'weight': 'normal'}, color="black")
        plt.text(0.1, 0.85, "p = " + str(round(p, 2)), ha='left', va='top', transform=ax.transAxes,
                 fontsize=12, fontdict={'weight': 'normal'}, color="black")

    linewidth = 1.5
    fontsize = 16
    width=6
    ratio=5
    space=0.2
    ax.spines["left"].set_visible(True)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['left'].set_color('0.2')

    ax.spines["bottom"].set_visible(True)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['bottom'].set_color('0.2')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=linewidth, color='0.2', labelsize=10)
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.set_xlim(16, 98)
    ax.set_ylabel('', fontsize=10, weight='normal', color='0.2', labelpad=5)

    plt.savefig(path+'motility_feature_difference/' + 'diff %s regplot.png'%feature, dpi=300, bbox_inches='tight')
    os.makedirs(path+'motility_feature_difference/' + 'svg/', exist_ok=True)
    plt.savefig(path+'motility_feature_difference/' + 'svg/diff %s regplot.svg'%feature, bbox_inches='tight')
    plt.clf()
    plt.close()
# ----------------------------------------------------------------------
for feature in feature_list:
    fig, ax = plt.subplots(figsize=(2,2))
    sns.lineplot(
        data=video_type_features,
        x='mt_fraction',
        y=feature,
        hue='Type',
        marker='o',
        errorbar='se'
    )
    plt.xlabel('mt_fraction')
    plt.ylabel(feature)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    os.makedirs(path + '/motility_feature_mt_fraction/', exist_ok=True)
    plt.savefig(path + 'motility_feature_mt_fraction/%s.png' % (feature), dpi=300, bbox_inches='tight')
    os.makedirs(path + 'svg/motility_feature_mt_fraction/', exist_ok=True)
    plt.savefig(path + 'svg/motility_feature_mt_fraction/%s.svg' % (feature), bbox_inches='tight')
    plt.clf()
    plt.close()


# ----------------------------------------------------------------------
all_mcs = list(np.unique(df2['kmeans']))

mc_dist = (
    df2[df2['Type'].isin(['wt_B-cell', 'mt_B-cell'])]
    .groupby(['Video', 'Type', 'Dominance', 'mt_fraction', 'kmeans'])
    .size()
    .rename('n')
    .reset_index()
)

mc_dist['cluster_fraction'] = (
    mc_dist['n'] /
    mc_dist.groupby(['Video', 'Type'])['n'].transform('sum')
)

mc_dist = (
    mc_dist.groupby(['Video', 'Type', 'Dominance', 'mt_fraction'], group_keys=False)
    .apply(
        lambda g: g.set_index('kmeans')
        .reindex(all_mcs)
        .rename_axis('kmeans')
        .reset_index()
        .assign(
            Video=g['Video'].iloc[0],
            Type=g['Type'].iloc[0],
            Dominance=g['Dominance'].iloc[0],
            mt_fraction=g['mt_fraction'].iloc[0]
        )
    )
    .reset_index(drop=True)
)

mc_dist['n'] = mc_dist['n'].fillna(0)
mc_dist['cluster_fraction'] = mc_dist['cluster_fraction'].fillna(0)



mc_profile = (
    mc_dist
    .groupby(['Type', 'Dominance', 'kmeans'])['cluster_fraction']
    .mean()
    .reset_index()
)

mc_profile['Type_Dominance'] = mc_profile['Type'] + ' ' + mc_profile['Dominance']
mc_profile_wide = (
    mc_profile
    .pivot(index='Type_Dominance', columns='kmeans', values='cluster_fraction')
    .reindex(columns=range(9), fill_value=0)
)

file_name = 'mt_dominance mc distribution'
vmax = 0.33
kws = dict(cbar_kws=dict(ticks=[0, vmax], orientation='horizontal'), vmin=0)
g = sns.clustermap(mc_profile_wide, annot=True, fmt='.2f', annot_kws={"fontsize":12},  # .1f
                       cmap='OrRd', method='ward', col_cluster=False, row_cluster=True, vmax=vmax,
                       # cbar_pos=(1, 0.2, 0.03, 0.8),
                       linewidths=0.5, linecolor='black', alpha=0.7,
                       #dendrogram_ratio=(0.12, 0.03),

                       **kws,
                       figsize=(6,3.6),
                       )

# cbar_pos = (left, bottom, width, height)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=16, va='center', rotation=0)
# Set tick label size = 16, and center y tick labels (centering x tick labels make it strange)
g.ax_heatmap.set_yticks(np.arange(mc_profile_wide.shape[0]) + 0.5)
g.ax_heatmap.set_yticklabels(mc_profile_wide.index, fontsize=16, rotation=0)

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0, 0.96, g.ax_row_dendrogram.get_position().width, 0.02])
# Set position of colorbar

g.ax_cbar.set_title('Occurrence', fontsize=12)
g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)
# Set colorbar title and font size

for spine in g.ax_cbar.spines:
    #g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(0.5)
# Set bounding box line width of colorbar

#plt.xlabel('%s' % cluster_type)
#plt.ylabel('%s' % condition_name)

plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
os.makedirs(path + 'svg/', exist_ok=True)
plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

plt.clf()
plt.close()


df_corr = mc_profile_wide.T.corr()

desired_order = [
    'wt_B-cell WT-dominant',
    'wt_B-cell Comparable',
    'wt_B-cell MT-dominant',
    'mt_B-cell WT-dominant',
    'mt_B-cell Comparable',
    'mt_B-cell MT-dominant'
]

df_corr = df_corr.reindex(index=desired_order, columns=desired_order)



mask = np.triu(df_corr) # Mask for only lower triangle

mask = mask[1:, :-1]  # mask that doesn't contain very top left and very bottom right block
corr = df_corr.iloc[1:,:-1].copy()  # df that doesn't contain very top left and very bottom right block

fig, ax = plt.subplots(figsize=(2, 2))
kws = dict(cbar_kws=dict(ticks=[0, 0.5, 1], orientation='horizontal'), vmin=0)

ax = sns.heatmap(abs(corr), annot=True, mask=mask, cmap='OrRd', alpha=0.9, linewidths=1, linecolor='white', fmt=".2f", square= True,
                 annot_kws={'size': 4, 'weight':'normal'}, vmin=0, vmax=np.max(np.max(abs(corr))),
                 cbar_kws= {"shrink":0.7, 'label':'Correlation', 'ticks':[0, np.max(np.max(abs(corr)))/2, np.max(np.max(abs(corr)))]})

cax = ax.figure.axes[-1]  # colorbar
cax.tick_params(labelsize=4)  # fontsize of tick label
cax.yaxis.label.set_size(6)  # fontsize of color bar y label

yticks = [i for i in corr.index]
xticks = [i for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=4, weight='normal')
plt.xticks(plt.yticks()[0], labels=xticks, fontsize=4, rotation=35, rotation_mode='anchor', ha='right', weight='normal')

plt.savefig(path+'MC correlation.png', dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/MC correlation.svg', bbox_inches='tight')

plt.close()
plt.clf()