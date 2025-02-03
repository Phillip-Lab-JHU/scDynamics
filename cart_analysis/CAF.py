from utils.draw_utils import *

#################################### Plot whole state space ####################################
path = r'C:\Users\ChanhongMin\OneDrive - Johns Hopkins\Desktop\CAF\\'
df_features = pd.read_csv(path+'features_total_20221022_invitroBreastCAF.csv')
df_pcs = pd.read_csv(path+'df_PCs.csv').drop(['Unnamed: 0', 'time_point', 'cell_label', 'condition'], axis=1)

df = pd.concat([df_features, df_pcs], axis=1)
xmin = math.floor(df['PC1'].min()) - 1
xmax = math.ceil(df['PC1'].max()) + 1
ymin = math.floor(df['PC2'].min()) - 1
ymax = math.ceil(df['PC2'].max()) + 1

fig, ax = plt.subplots(figsize=(2, 2))
# plt.subplot(len(n_neighbors_list), len(min_dist_list), i)

font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

scatter = ax.scatter(df['PC1'], df['PC2'], s=0.05,color='grey')
format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.savefig(path + '%s.png' % 'plain_space', dpi=300)
plt.clf()
plt.close()

df_ = df.replace({'condition': {'control': 'Control', 'with7': '+ MCF-7', 'with231': '+ MDA-MB-231'}})
draw_umap_space(df_, path, file_name='space_Type', condition_name='condition', label_name='cell_label', colors = ('#888888', '#6699CC', '#CC6677'), x_name='PC1', y_name='PC2', dot_size=0.07)
draw_umap_space(df_, path, file_name='space_tskmeans', condition_name='tskmeans', label_name='pseudo_Label', x_name='PC1', y_name='PC2', dot_size=0.07)
draw_contour(df_, path, file_name='space_contour', condition_name='condition', colors = ('#888888', '#6699CC', '#CC6677'), x_name='PC1', y_name='PC2', bin_num=50, num_contours=4)
draw_contour(df_, path, file_name='space_tskmeans_contour', condition_name='tskmeans', x_name='PC1', y_name='PC2', bin_num=50, num_contours=1)

df.columns.get_loc('extent')
df.columns.get_loc('zernike_moments_29')
feature_list = df.columns[5:61].drop(['bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2'])
draw_space_feature_magnitude(df, path, feature_list, dot_size=0.07, x_name='PC1', y_name='PC2', vmax=None)

#################################### Heatmap of cluster enrichment and shannon entropy ####################################
path = 'C:/Users/ChanhongMin/OneDrive - Johns Hopkins/Cornell LN Spleen/Analysis/Short term/Figure2. Overall behavior of T and wt GCB cell/'
df_ = df.replace({'Type': {'wt_B-cell': 'wt GCB', 'mt_B-cell': 'mt GCB', 'T-cell': 'Tfh'}})
draw_cluster_distribution_heatmap(df_, path, condition_name='Type', cluster_type='tskmeans')

entropies = {'T-cell':[], 'wt_B-cell':[]}
for video in np.unique(df['Video']):
    df_part = df[df['Video']==video]
    entropy, max_entropy = calculate_entropy(df_part, df, condition_name='Type', cluster_type='tskmeans')
    for type in entropy:
        entropies[type].append(entropy[type])

new_order = ['wt_B-cell', 'T-cell']
ordered_entropies = change_dict_order(entropies, new_order)

replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
entropies_ = {replace_keys.get(k, k):v  for (k,v) in ordered_entropies.items() }

dict_datasets = entropies_
file_name='entropy'
test='mann-whitney'

colors = ('#6699CC', '#CC6677')
font = {'family': 'arial',
        'weight': 'normal',
        'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
matplotlib.rcParams['lines.linewidth'] = 1

fig, ax = plt.subplots(figsize=(1, 2))
sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

ax = sns.barplot(data=sorted_vals, capsize=0.5, edgecolor='0.2', lw=1, errwidth=1, palette=colors)
plot_params = {'edgecolor': '0.2', 'linewidth': 1, 'fc': 'none'}
ax = sns.stripplot(data=sorted_vals, marker='s', s=1.5, **plot_params)
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
plt.xticks(plt.xticks()[0], sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
           weight='bold')
plt.yticks(fontsize=8, color='0.2', weight='bold')
# plt.ylabel('%s' % feature_name, fontsize=4)
# category labels
plt.grid(False)

from scipy import stats
from itertools import combinations

p_values = []
pairs = []
for pair in combinations(range(0, len(dict_datasets)), 2):  # 2 for pairs, 3 for triplets, etc
    if test == 'mann-whitney':
        stat_test = stats.mannwhitneyu(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
    elif test == 't-test':
        stat_test = stats.ttest_ind(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
    elif test == 'wilcoxon-ranksum':
        stat_test = stats.ranksums(dict_datasets[sorted_keys[pair[0]]], dict_datasets[sorted_keys[pair[1]]])
    p_values.append(stat_test.pvalue)
    pairs.append(pair)
    print(pair, stat_test.pvalue)
# plt.title('%s:%s' % (pairs, p_values))
plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'svg/')
plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
plt.clf()
plt.close()

#draw_custom_box_plot(entropies, path, feature_name='entropy', test='mann-whitney')

#################################### Box plot comparing all motility features by cell types ####################################
feature_list = df.columns[2:61].drop(['x_c', 'y_c', 'bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2'])

condition_name = 'condition'
replace_keys = {'control': 'Control', 'with7': '+ MCF-7', 'with231': '+ MDA-MB-231'}
if not os.path.isdir(path + 'feature_violin_plot_type/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_type/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    new_order = ['control', 'with7', 'with231']
    dataset = change_dict_order(dataset, new_order)
    dataset = {replace_keys.get(k, k):v  for (k,v) in dataset.items() }

    draw_custom_violin_plot(dataset, path+'feature_violin_plot_type/', file_name=feature_name, colors=('#44AA99', '#6699CC', '#CC6677'),
                            test='mann-whitney', pvalue=True, figsize=(2,2))

#################################### Box plot comparing all motility features by tskmeans ####################################
feature_list = df.columns[128:283].drop(['speed_distribution', 'angle_distribution', 'speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])
condition_name = 'tskmeans'
replace_keys = {'T-cell':'Tfh', 'mt_B-cell':'mt GCB', 'wt_B-cell':'wt GCB'}
if not os.path.isdir(path + 'feature_violin_plot_tskmeans/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(path + 'feature_violin_plot_tskmeans/')

for feature_name in feature_list:
    dataset={}
    for condition in np.unique(df[condition_name]):
        data = df[df[condition_name] == condition][feature_name]
        dataset[condition] = np.array(data)
    dict_datasets = {replace_keys.get(k, k):v  for (k,v) in dataset.items() }
    draw_custom_violin_plot(dict_datasets, path + 'feature_violin_plot_tskmeans/', file_name=feature_name,
    colors=('#888888', '#CC6677', '#44AA99', '#6699CC', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', ),
    test='mann-whitney', pvalue=False, figsize=(2, 2))

#################################### Volcano plot of all motility features ####################################
feature_list = df.columns[2:61].drop(['x_c', 'y_c', 'bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2', 'zernike_moments_0'])
df_part = df[df['condition']!='with7'].reset_index(drop=True)

df_p = pd.DataFrame()
for feature_name in feature_list:
    dataset = {}
    for condition in np.unique(df_part['condition']):
        data = df_part[df_part['condition'] == condition][feature_name]
        dataset[condition] = np.array(data)

    pvalue = get_pvalue(dataset, test='mann-whitney')
    logp = -np.log10(pvalue)

    avgZ = get_avgZ(dataset, ref_name='control', data_name='with231')

    row = pd.DataFrame()
    row['Feature'] = [feature_name]
    row['Pvalue'] = [pvalue]
    row['-Logp'] = [logp]
    row['AvgZ'] = [avgZ]
    df_p = pd.concat([df_p, row], axis=0)

df_p['Adj_p'] = adjusted_pvalues(df_p['Pvalue'], correction_type='Benjamini-Hochberg')
df_p['Adj_Logp'] = -np.log(df_p['Adj_p'])
df_p = df_p[~df_p.isin([np.nan, np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)  # Remove rows that have nan / inf


draw_volcano_plot(df_p, path, file_name='volcano plot with 231', z_thresh=0.1, p_thresh=100, z_name='AvgZ', p_name='Adj_Logp',
                  feature_name='Feature', figsize=(6,6))

