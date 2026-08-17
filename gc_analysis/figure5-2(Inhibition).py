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
"""Generates Data for Figure5-2. Inhibition analysis with FDC and Tfh interaction"""

from scipy import stats
from utils.draw_utils import *
from utils.misc_utils import *
from features.interaction import DistanceSignal

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GCB_with_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_with_inhibit_traj_duration_20.parquet')

df_inhibit = df[(df['Exp']=='IgG')|(df['Exp']=='CD40L')|(df['Exp']=='mLT')].reset_index(drop=True)
df_duration_inhibit = df_duration[(df_duration['Exp']=='IgG')|(df_duration['Exp']=='CD40L')|(df_duration['Exp']=='mLT')].reset_index(drop=True)

df['Inhibition'] = df['Exp'].copy()
df['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)

df_duration['Inhibition'] = df_duration['Exp'].copy()
df_duration['Inhibition'].replace({'Exp1': 'Control', 'Exp2': 'Control', 'Exp3': 'Control', 'Exp5': 'Control'}, inplace=True)

_, _, Zone_series = to_timeseries_fast(df_duration, duration=20, feature_name='Zone')

feature_list = ['average']
FDC_dist = DistanceSignal(Zone_series)
df_distance = FDC_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'Zone_average'}, inplace=True)

df['Zone_average'] = df_distance


df['Zone'] = np.nan
df.loc[(df['Zone_average'] < 0.4) & (df['Zone_average'] >= 0), 'Zone'] = 'DZ'
df.loc[(df['Zone_average'] < 0.8) & (df['Zone_average'] >= 0.4), 'Zone'] = 'DZ-sLZ'
df.loc[(df['Zone_average'] < 1.2) & (df['Zone_average'] >= 0.8), 'Zone'] = 'sLZ'
df.loc[(df['Zone_average'] < 1.6) & (df['Zone_average'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
df.loc[(df['Zone_average'] <= 2) & (df['Zone_average'] >= 1.6), 'Zone'] = 'dLZ'

df_unassigned_zone = df[df['Zone'].isna()].copy()
print('number of cells without assigned Zone:', df_unassigned_zone.shape[0])
print(df_unassigned_zone[[column for column in ['traj_Label', 'Video', 'Type', 'Inhibition', 'Zone_average'] if column in df_unassigned_zone.columns]].head())

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure5. Inhibition\FDC and Tfh interaction\\'

####################################### Quantify FDC interaction frequency #############################################
df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT GC B cell', 'mt_B-cell': 'MT GC B cell'}})
df_ = df_.replace({'Inhibition': {'mLT': 'mLTβR'}})

feature_name = 'FDC_avg_overlap' #'T_contact_persistences', 'T_avg_overlap'
int_feature_list = ['FDC_avg_overlap', 'FDC_contact_persistences', 'FDC_contact_times',
                    'Core_distance_average', 'DZ_distance_average', 'LZ_distance_average',
                    'T_distance_average', 'T_contact_persistences', 'T_contact_times', 'T_avg_overlap']
condition_name = 'Type'

for feature_name in int_feature_list:
    dict_datasets={}
    for cell_type in ['WT GC B cell', 'MT GC B cell']:
        for group in ['CD40L', 'mLTβR']:
        #for group in ['IgG', 'CD40L', 'mLT']:
            data = df_[(df_['Inhibition'] == group) & (df_[condition_name] == cell_type)][feature_name]
            data = data[data>0]
            dict_datasets[cell_type + ' ' + str(group)] = np.array(data)

    [print(key, np.array(value).size) for key, value in dict_datasets.items()]

    # font = {'family': 'arial',
    #         'weight': 'normal',
    #         'size': 8}
    # matplotlib.rc('font', **font)
    # matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    # matplotlib.rcParams['lines.linewidth'] = 1
    #
    # colors = ('#888888', '#888888', '#CC6677', '#CC6677',)
    # colors = ('#888888', '#CC6677',)
    # fig, ax = plt.subplots(figsize=(2, 2))
    # if feature_name == 'Core_distance_average':
    #     clip = (0, 100)
    # elif feature_name == 'DZ_distance_average':
    #     clip = (0, 60)
    # elif feature_name == 'LZ_distance_average':
    #     clip = (0, 40)
    # else:
    #     clip = None
    # for i, key in enumerate(dict_datasets):
    #     ax = sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=clip, color=colors[i], label=key)
    #
    # # ax = sns.kdeplot(data=dict_datasets['WT'], fill=True, linewidth=1, clip=(0, 40), color='#888888', label='WT')
    # # ax = sns.kdeplot(data=dict_datasets['MT'], fill=True, linewidth=1, clip=(0, 40), color='#CC6677', label='MT')
    #
    # for axis in ['bottom', 'left']:
    #     ax.spines[axis].set_linewidth(1)
    #     ax.spines[axis].set_color('0.2')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    #
    # ax.set_xlabel('%s'%feature_name, fontsize=8, weight='normal', color='0.2')
    # ax.set_ylabel('Density', fontsize=8, weight='normal', color='0.2')
    # plt.xticks(fontsize=8, color='0.2', weight='normal')
    # plt.yticks(fontsize=8, color='0.2', weight='normal')
    #
    # legend = plt.legend(frameon=False, prop={'weight': 'normal', 'size': 8}, labelcolor='0.2')
    # legend.remove()
    # os.makedirs(path + 'IgG vs mLT/', exist_ok=True)
    # plt.savefig(path + 'IgG vs mLT/%s kde.png' % feature_name, dpi=300, bbox_inches='tight')
    #
    # os.makedirs(path + 'IgG vs mLT/svg/', exist_ok=True)
    # plt.savefig(path + 'IgG vs mLT/svg/%s kde.svg' % feature_name, bbox_inches='tight')
    # plt.clf()
    # plt.close()


    from scipy import stats
    figsize=(2,2)
    test = 'kruskal-wallis_dunn'
    error_type = 'ci_norm'
    file_name = '%s'%feature_name
    colors=('#888888', '#888888', '#CC6677', '#CC6677',)

    font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1


    mean_dataset = {}
    error_dataset = {}
    for key, values in dict_datasets.items():

        if error_type == 'std':
            error = np.std(values)
        elif error_type == 'sem':
            error = stats.sem(values)
        elif error_type == 'ci_norm':
            interval = stats.norm.interval(confidence=0.95, loc=np.mean(values), scale=stats.sem(values))
            error = np.mean(values) - interval[0]
        elif error_type == 'ci_t':
            interval = stats.t.interval(confidence=0.95, df=values.size-1, loc=np.mean(values), scale=stats.sem(values))
            error = np.mean(values) - interval[0]

        mean_dataset[key] = np.mean(values)
        error_dataset[key] = error



    sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.violinplot(data=sorted_vals, palette=colors, linewidth=1, linecolor="0.2", inner=None,
                        inner_kws=dict(box_width=10, whis_width=10, color="0.2"), cut=0)

    ax = sns.scatterplot(x=np.arange(0, len(sorted_keys), 1), y=mean_dataset.values(), color="0.2", s=8, zorder=3)

    # for idx, key in enumerate(mean_dataset):
    #     ax = sns.lineplot(data=mean_dataset, x=idx, y=mean_dataset[key], linestyle='',
    #                                  label=key, lw=2.5,  dashes=False, markersize=8, err_style='bars', color=colors[idx])
    ax.errorbar(x=np.arange(0, len(sorted_keys), 1), y=list(mean_dataset.values()),
                                    yerr=list(error_dataset.values()), color='0.2', capsize=3, capthick=1, elinewidth=1.5, fmt='none', zorder=2)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=1, color='0.2')
    #ax.set_ylabel(feature_name, fontsize=8, weight='normal')
    plt.xticks(plt.xticks()[0], sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2',
               weight='normal')
    plt.yticks(fontsize=8, color='0.2', weight='normal')
    #plt.ylim(0, 10)
    #plt.ylim(0, 0.1)
    pairs, p_values, cohen_ds = get_various_statistics(dict_datasets, test=test, return_sig=True)
    plt.title('%s: %s, %s' % (pairs, p_values, cohen_ds), fontsize=4)

    os.makedirs(path + 'IgG vs mLT/', exist_ok=True)
    plt.savefig(path + 'IgG vs mLT/%s.png' % file_name, dpi=300, bbox_inches='tight')

    os.makedirs(path + 'IgG vs mLT/svg/', exist_ok=True)
    plt.savefig(path + 'IgG vs mLT/svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()


################################ IgG vs mLTβR: video-level two-part analysis ################################
# Each video contains both WT and MT cells. Treat the video as the experimental
# replicate and test the Type x Inhibition interaction by comparing the
# within-video (MT - WT) difference between IgG and mLTβR videos.
#
# For overlap/contact endpoints, zeros are biologically informative and are
# analyzed separately as:
#   1) fraction of trajectories with any interaction (> 0)
#   2) median magnitude among trajectories with a positive interaction
#
# Distance endpoints are summarized across all trajectories; a distance of zero
# is retained because it can represent direct contact.
video_analysis_path = path + 'IgG vs mLT/video_level_two_part/'
os.makedirs(video_analysis_path, exist_ok=True)
os.makedirs(video_analysis_path + 'svg/', exist_ok=True)

treatment_order = ['IgG', 'mLTβR']
type_order = ['WT GC B cell', 'MT GC B cell']
group_order = [
    'WT GC B cell IgG',
    'MT GC B cell IgG',
    'WT GC B cell mLTβR',
    'MT GC B cell mLTβR',
]
group_palette = {
    'WT GC B cell IgG': '#888888',
    'MT GC B cell IgG': '#CC6677',
    'WT GC B cell mLTβR': '#888888',
    'MT GC B cell mLTβR': '#CC6677',
}


def bootstrap_interaction_ci(video_contrasts, n_boot=10000, confidence=0.95, seed=0):
    """Bootstrap videos within treatment to estimate the interaction CI."""
    igg = video_contrasts.loc[
        video_contrasts['Inhibition'] == 'IgG', 'MT_minus_WT'
    ].dropna().to_numpy()
    mlt = video_contrasts.loc[
        video_contrasts['Inhibition'] == 'mLTβR', 'MT_minus_WT'
    ].dropna().to_numpy()

    if igg.size == 0 or mlt.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = (
            rng.choice(mlt, size=mlt.size, replace=True).mean()
            - rng.choice(igg, size=igg.size, replace=True).mean()
        )

    alpha = 1 - confidence
    estimate = mlt.mean() - igg.mean()
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return estimate, low, high


def get_video_contrasts(video_summary, value_name):
    """Return one paired MT-WT contrast per video."""
    contrast = (
        video_summary
        .pivot_table(
            index=['Video', 'Inhibition'],
            columns='Type',
            values=value_name,
            aggfunc='first'
        )
        .reset_index()
    )

    missing_types = [typ for typ in type_order if typ not in contrast.columns]
    if missing_types:
        raise ValueError(
            f'Cannot calculate paired video contrasts; missing Type columns: {missing_types}'
        )

    contrast['MT_minus_WT'] = (
        contrast['MT GC B cell'] - contrast['WT GC B cell']
    )
    return contrast


def interaction_statistics(video_contrasts, feature, endpoint):
    """Summarize the video-level Type x Inhibition interaction."""
    igg = video_contrasts.loc[
        video_contrasts['Inhibition'] == 'IgG', 'MT_minus_WT'
    ].dropna().to_numpy()
    mlt = video_contrasts.loc[
        video_contrasts['Inhibition'] == 'mLTβR', 'MT_minus_WT'
    ].dropna().to_numpy()

    estimate, ci_low, ci_high = bootstrap_interaction_ci(video_contrasts)

    if igg.size >= 2 and mlt.size >= 2:
        welch = stats.ttest_ind(mlt, igg, equal_var=False, nan_policy='omit')
        mann_whitney = stats.mannwhitneyu(mlt, igg, alternative='two-sided')
        welch_p = welch.pvalue
        mann_whitney_p = mann_whitney.pvalue
    else:
        welch_p = np.nan
        mann_whitney_p = np.nan

    return {
        'feature': feature,
        'endpoint': endpoint,
        'n_IgG_videos': igg.size,
        'n_mLTbR_videos': mlt.size,
        'IgG_mean_MT_minus_WT': np.mean(igg) if igg.size else np.nan,
        'mLTbR_mean_MT_minus_WT': np.mean(mlt) if mlt.size else np.nan,
        'interaction_mLTbR_minus_IgG': estimate,
        'bootstrap_CI_low': ci_low,
        'bootstrap_CI_high': ci_high,
        'welch_p': welch_p,
        'mann_whitney_p': mann_whitney_p,
    }


def add_video_summaries_and_pairwise_ttests(
        ax, plot_df, value_name, ordered_groups, alpha=0.05):
    """Add median markers and significant unadjusted pairwise Welch t-tests."""
    from itertools import combinations

    datasets = []
    valid_group_positions = []

    for position, group in enumerate(ordered_groups):
        values = plot_df.loc[
            plot_df['Group'] == group, value_name
        ].dropna().to_numpy()
        if values.size:
            datasets.append(values)
            valid_group_positions.append(position)

            # Median: horizontal black line.
            ax.plot(
                [position - 0.20, position + 0.20],
                [np.median(values), np.median(values)],
                color='0.05',
                linewidth=2,
                solid_capstyle='butt',
                zorder=4
            )

    if len(datasets) < 2:
        return

    pairs = list(combinations(range(len(datasets)), 2))
    p_values = np.array([
        stats.ttest_ind(
            datasets[first],
            datasets[second],
            equal_var=False,
            nan_policy='omit'
        ).pvalue
        for first, second in pairs
    ])

    significant_pairs = []
    for (first, second), p_value in zip(pairs, p_values):
        if np.isfinite(p_value) and p_value < alpha:
            significant_pairs.append(
                (
                    valid_group_positions[first],
                    valid_group_positions[second],
                    p_value
                )
            )

    if not significant_pairs:
        return

    plotted_values = plot_df[value_name].dropna().to_numpy()
    y_min = np.min(plotted_values)
    y_max = np.max(plotted_values)
    y_range = y_max - y_min
    if not np.isfinite(y_range) or y_range == 0:
        y_range = max(abs(y_max), 1.0)

    base = y_max + 0.08 * y_range
    step = 0.12 * y_range
    for level, (first, second, p_value) in enumerate(significant_pairs):
        y = base + level * step
        bracket_height = 0.025 * y_range
        ax.plot(
            [first, first, second, second],
            [y, y + bracket_height, y + bracket_height, y],
            color='0.2',
            linewidth=0.7,
            clip_on=False
        )
        if p_value < 0.0001:
            significance = '****'
        elif p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        else:
            significance = '*'
        ax.text(
            (first + second) / 2,
            y + bracket_height,
            f'{significance}  p={p_value:.3g}',
            ha='center',
            va='bottom',
            fontsize=6,
            color='0.2'
        )

    current_bottom, _ = ax.get_ylim()
    annotation_top = base + len(significant_pairs) * step
    ax.set_ylim(current_bottom, annotation_top)


def plot_video_values(ax, video_summary, value_name, ylabel):
    """Plot video-level values and connect WT/MT measurements from each video."""
    plot_df = video_summary.copy()
    plot_df['Group'] = (
        plot_df['Type'].astype(str) + ' ' + plot_df['Inhibition'].astype(str)
    )
    x_lookup = {group: idx for idx, group in enumerate(group_order)}

    for (_, treatment), video_part in plot_df.groupby(['Video', 'Inhibition']):
        video_part = video_part[video_part['Group'].isin(x_lookup)]
        if video_part.shape[0] == 2:
            xs = [x_lookup[group] for group in video_part['Group']]
            ax.plot(xs, video_part[value_name], color='0.75', linewidth=0.6, zorder=1)

    sns.stripplot(
        data=plot_df,
        x='Group',
        y=value_name,
        order=group_order,
        palette=group_palette,
        size=4,
        jitter=0.08,
        edgecolor='0.2',
        linewidth=0.3,
        ax=ax,
        zorder=2
    )
    add_video_summaries_and_pairwise_ttests(
        ax,
        plot_df,
        value_name,
        group_order
    )
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticklabels(
        ['WT\nIgG', 'MT\nIgG', 'WT\nmLTβR', 'MT\nmLTβR'],
        fontsize=7
    )
    ax.tick_params(axis='y', labelsize=7, width=0.8)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(0.8)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_video_interaction(ax, video_contrasts, ylabel):
    """Plot the paired Type contrast for each video by treatment."""
    sns.stripplot(
        data=video_contrasts,
        x='Inhibition',
        y='MT_minus_WT',
        order=treatment_order,
        color='0.25',
        size=4,
        jitter=0.08,
        ax=ax
    )
    ax.axhline(0, color='0.65', linestyle='--', linewidth=0.8)

    for x, treatment in enumerate(treatment_order):
        values = video_contrasts.loc[
            video_contrasts['Inhibition'] == treatment, 'MT_minus_WT'
        ].dropna().to_numpy()
        if values.size:
            median = np.median(values)
            ax.plot(
                [x - 0.12, x + 0.12],
                [median, median],
                color='0.05',
                linewidth=2,
                solid_capstyle='butt',
                zorder=3
            )

    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticklabels(['IgG', 'mLTβR'], fontsize=7)
    ax.tick_params(axis='y', labelsize=7, width=0.8)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(0.8)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


analysis_df = df_[
    df_['Inhibition'].isin(treatment_order)
    & df_['Type'].isin(type_order)
].copy()

if analysis_df.empty:
    raise ValueError(
        'No IgG/mLTβR rows were found. Check the Inhibition replacement labels.'
    )

video_type_counts = (
    analysis_df.groupby(['Inhibition', 'Video'])['Type'].nunique()
)
if (video_type_counts != len(type_order)).any():
    incomplete_videos = video_type_counts[video_type_counts != len(type_order)]
    raise ValueError(
        'Every video must contain both WT and MT for the paired interaction '
        f'analysis. Incomplete videos: {incomplete_videos.index.tolist()}'
    )

zero_inflated_features = [
    'FDC_avg_overlap',
    'FDC_contact_persistences',
    'FDC_contact_times',
    'T_contact_persistences',
    'T_contact_times',
    'T_avg_overlap',
]
distance_features = ['Core_distance_average', 'T_distance_average']

all_interaction_stats = []
all_video_summaries = []
all_video_contrasts = []

for feature_name in zero_inflated_features:
    feature_df = analysis_df[
        ['Video', 'Inhibition', 'Type', feature_name]
    ].dropna(subset=[feature_name]).copy()
    feature_df['has_interaction'] = feature_df[feature_name] > 0
    feature_df['positive_value'] = feature_df[feature_name].where(
        feature_df['has_interaction']
    )

    video_summary = (
        feature_df
        .groupby(['Video', 'Inhibition', 'Type'], observed=True)
        .agg(
            n_trajectories=(feature_name, 'size'),
            interaction_fraction=('has_interaction', 'mean'),
            overall_median=(feature_name, 'median'),
            positive_median=('positive_value', 'median'),
        )
        .reset_index()
    )
    video_summary['feature'] = feature_name
    all_video_summaries.append(video_summary)

    fraction_contrast = get_video_contrasts(video_summary, 'interaction_fraction')
    fraction_contrast['feature'] = feature_name
    fraction_contrast['endpoint'] = 'fraction_positive'
    all_video_contrasts.append(fraction_contrast)
    fraction_stats = interaction_statistics(
        fraction_contrast, feature_name, 'fraction_positive'
    )
    all_interaction_stats.append(fraction_stats)

    magnitude_contrast = get_video_contrasts(video_summary, 'positive_median')
    magnitude_contrast['feature'] = feature_name
    magnitude_contrast['endpoint'] = 'positive_median'
    all_video_contrasts.append(magnitude_contrast)
    magnitude_stats = interaction_statistics(
        magnitude_contrast, feature_name, 'positive_median'
    )
    all_interaction_stats.append(magnitude_stats)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    plot_video_values(
        axes[0],
        video_summary,
        'interaction_fraction',
        'Fraction > 0'
    )
    plot_video_values(
        axes[1],
        video_summary,
        'positive_median',
        'Median among > 0'
    )
    plot_video_interaction(
        axes[2],
        fraction_contrast,
        'MT − WT\nfraction > 0'
    )
    axes[2].set_title(
        'Interaction: '
        f"{fraction_stats['interaction_mLTbR_minus_IgG']:.3g}\n"
        f"95% CI [{fraction_stats['bootstrap_CI_low']:.3g}, "
        f"{fraction_stats['bootstrap_CI_high']:.3g}]",
        fontsize=7
    )
    fig.suptitle(feature_name, fontsize=9)
    fig.tight_layout()
    fig.savefig(
        video_analysis_path + feature_name + '_two_part.png',
        dpi=300,
        bbox_inches='tight'
    )
    fig.savefig(
        video_analysis_path + 'svg/' + feature_name + '_two_part.svg',
        bbox_inches='tight'
    )
    plt.close(fig)

for feature_name in distance_features:
    feature_df = analysis_df[
        ['Video', 'Inhibition', 'Type', feature_name]
    ].dropna(subset=[feature_name]).copy()

    video_summary = (
        feature_df
        .groupby(['Video', 'Inhibition', 'Type'], observed=True)
        .agg(
            n_trajectories=(feature_name, 'size'),
            overall_median=(feature_name, 'median')
        )
        .reset_index()
    )
    video_summary['feature'] = feature_name
    all_video_summaries.append(video_summary)

    distance_contrast = get_video_contrasts(video_summary, 'overall_median')
    distance_contrast['feature'] = feature_name
    distance_contrast['endpoint'] = 'overall_median'
    all_video_contrasts.append(distance_contrast)
    distance_stats = interaction_statistics(
        distance_contrast, feature_name, 'overall_median'
    )
    all_interaction_stats.append(distance_stats)

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.2))
    plot_video_values(
        axes[0],
        video_summary,
        'overall_median',
        'Video median'
    )
    plot_video_interaction(
        axes[1],
        distance_contrast,
        'MT − WT\nvideo median'
    )
    axes[1].set_title(
        'Interaction: '
        f"{distance_stats['interaction_mLTbR_minus_IgG']:.3g}\n"
        f"95% CI [{distance_stats['bootstrap_CI_low']:.3g}, "
        f"{distance_stats['bootstrap_CI_high']:.3g}]",
        fontsize=7
    )
    fig.suptitle(feature_name, fontsize=9)
    fig.tight_layout()
    fig.savefig(
        video_analysis_path + feature_name + '_video_level.png',
        dpi=300,
        bbox_inches='tight'
    )
    fig.savefig(
        video_analysis_path + 'svg/' + feature_name + '_video_level.svg',
        bbox_inches='tight'
    )
    plt.close(fig)

interaction_stats_df = pd.DataFrame(all_interaction_stats)

# Benjamini-Hochberg correction across the tested feature/end-point
# combinations. Keep both the raw p-values and FDR-adjusted q-values.
for p_column in ['welch_p', 'mann_whitney_p']:
    p_values = interaction_stats_df[p_column].to_numpy(dtype=float)
    valid = np.isfinite(p_values)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    if valid.any():
        valid_p = p_values[valid]
        order = np.argsort(valid_p)
        ranked_p = valid_p[order]
        ranked_q = ranked_p * valid_p.size / np.arange(1, valid_p.size + 1)
        ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
        valid_q = np.empty_like(ranked_q)
        valid_q[order] = np.clip(ranked_q, 0, 1)
        adjusted[valid] = valid_q
    interaction_stats_df[p_column.replace('_p', '_FDR_q')] = adjusted


################################ Focused dLZ/FDC-core localization analysis ################################
# Core_distance_average is distance to the dense FDC region (dLZ):
# smaller distance = closer to dLZ. Use only trajectories with a valid core
# distance for every endpoint so missing zone measurements are not interpreted
# as zero dLZ residence.
dlz_analysis_path = path + 'IgG vs mLT/dLZ_localization/'
os.makedirs(dlz_analysis_path, exist_ok=True)
os.makedirs(dlz_analysis_path + 'svg/', exist_ok=True)

dlz_source = analysis_df.copy()
dlz_source['Acquisition_year'] = pd.to_numeric(
    dlz_source['Video'].astype(str).str.extract(r'^(\d{4})', expand=False),
    errors='coerce'
)
dlz_source['valid_core_distance'] = dlz_source['Core_distance_average'].notna()

dlz_qc = (
    dlz_source
    .groupby(
        ['Inhibition', 'Acquisition_year', 'Video', 'Type'],
        dropna=False,
        observed=True
    )
    .agg(
        n_trajectories=('Core_distance_average', 'size'),
        n_valid_core_distance=('Core_distance_average', 'count'),
        n_valid_avg_zone=('avg_zone', 'count')
    )
    .reset_index()
)
dlz_qc['core_distance_valid_fraction'] = (
    dlz_qc['n_valid_core_distance'] / dlz_qc['n_trajectories']
)
dlz_qc.to_csv(
    dlz_analysis_path + 'dLZ_localization_missingness_by_video.csv',
    index=False
)

dlz_valid = dlz_source[dlz_source['valid_core_distance']].copy()
dlz_valid['any_dLZ_residence'] = dlz_valid['dlz_resident_times'] > 0
dlz_valid['average_position_in_dLZ'] = dlz_valid['avg_zone'] >= 1.6

if dlz_valid.empty:
    raise ValueError('No valid Core_distance_average measurements were found')

dlz_endpoint_specs = {
    'Core_distance_average': {
        'summary': 'median',
        'ylabel': 'Median distance to dLZ',
        'direction': 'Lower = closer to dLZ'
    },
    'avg_zone': {
        'summary': 'median',
        'ylabel': 'Median average zone',
        'direction': 'Higher = more dLZ-localized'
    },
    'any_dLZ_residence': {
        'summary': 'mean',
        'ylabel': 'Fraction with any dLZ residence',
        'direction': 'Higher = more dLZ residence'
    },
    'average_position_in_dLZ': {
        'summary': 'mean',
        'ylabel': 'Fraction averaging in dLZ',
        'direction': 'Higher = more dLZ-localized'
    },
}

dlz_video_summaries = []
dlz_video_contrasts = []
dlz_interaction_stats = []

for endpoint, spec in dlz_endpoint_specs.items():
    value_column = endpoint + '_video_summary'
    video_summary = (
        dlz_valid
        .groupby(
            ['Video', 'Inhibition', 'Acquisition_year', 'Type'],
            observed=True
        )
        .agg(
            n_valid=(endpoint, 'size'),
            **{value_column: (endpoint, spec['summary'])}
        )
        .reset_index()
    )
    video_summary['endpoint'] = endpoint
    video_summary['direction'] = spec['direction']
    dlz_video_summaries.append(video_summary)

    contrast = get_video_contrasts(video_summary, value_column)
    contrast = contrast.merge(
        video_summary[['Video', 'Acquisition_year']].drop_duplicates(),
        on='Video',
        how='left',
        validate='one_to_one'
    )
    contrast['endpoint'] = endpoint
    contrast['direction'] = spec['direction']
    dlz_video_contrasts.append(contrast)

    stats_all = interaction_statistics(
        contrast, endpoint, 'all_available_videos'
    )
    stats_all['direction'] = spec['direction']
    dlz_interaction_stats.append(stats_all)

    # All usable IgG videos are from 2024, while mLTβR spans 2024 and 2025.
    # Report a 2024-only sensitivity analysis alongside the full comparison.
    contrast_2024 = contrast[contrast['Acquisition_year'] == 2024].copy()
    stats_2024 = interaction_statistics(
        contrast_2024, endpoint, '2024_only_sensitivity'
    )
    stats_2024['direction'] = spec['direction']
    dlz_interaction_stats.append(stats_2024)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    plot_video_values(
        axes[0], video_summary, value_column, spec['ylabel']
    )
    plot_video_interaction(
        axes[1], contrast, 'MT − WT\nall videos'
    )
    plot_video_interaction(
        axes[2], contrast_2024, 'MT − WT\n2024 only'
    )
    axes[1].set_title(
        'All: '
        f"{stats_all['interaction_mLTbR_minus_IgG']:.3g}\n"
        f"95% CI [{stats_all['bootstrap_CI_low']:.3g}, "
        f"{stats_all['bootstrap_CI_high']:.3g}]",
        fontsize=7
    )
    axes[2].set_title(
        '2024: '
        f"{stats_2024['interaction_mLTbR_minus_IgG']:.3g}\n"
        f"95% CI [{stats_2024['bootstrap_CI_low']:.3g}, "
        f"{stats_2024['bootstrap_CI_high']:.3g}]",
        fontsize=7
    )
    fig.suptitle(endpoint + '\n' + spec['direction'], fontsize=9)
    fig.tight_layout()
    fig.savefig(
        dlz_analysis_path + endpoint + '_focused_dLZ.png',
        dpi=300,
        bbox_inches='tight'
    )
    fig.savefig(
        dlz_analysis_path + 'svg/' + endpoint + '_focused_dLZ.svg',
        bbox_inches='tight'
    )
    plt.close(fig)

dlz_stats_df = pd.DataFrame(dlz_interaction_stats)
for p_column in ['welch_p', 'mann_whitney_p']:
    p_values = dlz_stats_df[p_column].to_numpy(dtype=float)
    valid = np.isfinite(p_values)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    if valid.any():
        valid_p = p_values[valid]
        order = np.argsort(valid_p)
        ranked_p = valid_p[order]
        ranked_q = ranked_p * valid_p.size / np.arange(1, valid_p.size + 1)
        ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
        valid_q = np.empty_like(ranked_q)
        valid_q[order] = np.clip(ranked_q, 0, 1)
        adjusted[valid] = valid_q
    dlz_stats_df[p_column.replace('_p', '_FDR_q')] = adjusted

dlz_stats_df.to_csv(
    dlz_analysis_path + 'dLZ_type_by_treatment_interaction_statistics.csv',
    index=False
)
pd.concat(dlz_video_summaries, ignore_index=True, sort=False).to_csv(
    dlz_analysis_path + 'dLZ_video_level_summaries.csv',
    index=False
)
pd.concat(dlz_video_contrasts, ignore_index=True, sort=False).to_csv(
    dlz_analysis_path + 'dLZ_video_level_MT_minus_WT_contrasts.csv',
    index=False
)


######################## Focused dLZ localization: CD40L versus mLTβR ########################
# This is saved separately from the IgG comparison. CD40L data are all from
# 2024, so the 2024-only analysis is the acquisition-year-matched comparison.
cd40_mlt_path = path + 'CD40L vs mLT/dLZ_localization/'
os.makedirs(cd40_mlt_path, exist_ok=True)
os.makedirs(cd40_mlt_path + 'svg/', exist_ok=True)

cd40_mlt_source = df_[
    df_['Inhibition'].isin(['CD40L', 'mLTβR'])
    & df_['Type'].isin(type_order)
].copy()
cd40_mlt_source['Acquisition_year'] = pd.to_numeric(
    cd40_mlt_source['Video'].astype(str).str.extract(r'^(\d{4})', expand=False),
    errors='coerce'
)
cd40_mlt_source['valid_core_distance'] = (
    cd40_mlt_source['Core_distance_average'].notna()
)

cd40_mlt_qc = (
    cd40_mlt_source
    .groupby(
        ['Inhibition', 'Acquisition_year', 'Video', 'Type'],
        dropna=False,
        observed=True
    )
    .agg(
        n_trajectories=('Core_distance_average', 'size'),
        n_valid_core_distance=('Core_distance_average', 'count'),
        n_valid_avg_zone=('avg_zone', 'count')
    )
    .reset_index()
)
cd40_mlt_qc['core_distance_valid_fraction'] = (
    cd40_mlt_qc['n_valid_core_distance']
    / cd40_mlt_qc['n_trajectories']
)
cd40_mlt_qc.to_csv(
    cd40_mlt_path + 'dLZ_localization_missingness_by_video.csv',
    index=False
)

cd40_mlt_valid = cd40_mlt_source[
    cd40_mlt_source['valid_core_distance']
].copy()
cd40_mlt_valid['any_dLZ_residence'] = (
    cd40_mlt_valid['dlz_resident_times'] > 0
)
cd40_mlt_valid['average_position_in_dLZ'] = (
    cd40_mlt_valid['avg_zone'] >= 1.6
)


def two_treatment_interaction_statistics(
        video_contrasts, reference, comparison, feature, endpoint,
        n_boot=10000, seed=0):
    """Compare paired MT-WT video contrasts between two treatments."""
    reference_values = video_contrasts.loc[
        video_contrasts['Inhibition'] == reference, 'MT_minus_WT'
    ].dropna().to_numpy()
    comparison_values = video_contrasts.loc[
        video_contrasts['Inhibition'] == comparison, 'MT_minus_WT'
    ].dropna().to_numpy()

    if reference_values.size and comparison_values.size:
        estimate = comparison_values.mean() - reference_values.mean()
        rng = np.random.default_rng(seed)
        boot = np.empty(n_boot)
        for i in range(n_boot):
            boot[i] = (
                rng.choice(
                    comparison_values,
                    size=comparison_values.size,
                    replace=True
                ).mean()
                - rng.choice(
                    reference_values,
                    size=reference_values.size,
                    replace=True
                ).mean()
            )
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    else:
        estimate = ci_low = ci_high = np.nan

    if reference_values.size >= 2 and comparison_values.size >= 2:
        welch_p = stats.ttest_ind(
            comparison_values,
            reference_values,
            equal_var=False,
            nan_policy='omit'
        ).pvalue
        mann_whitney_p = stats.mannwhitneyu(
            comparison_values,
            reference_values,
            alternative='two-sided'
        ).pvalue
    else:
        welch_p = mann_whitney_p = np.nan

    return {
        'feature': feature,
        'endpoint': endpoint,
        'reference_treatment': reference,
        'comparison_treatment': comparison,
        'n_reference_videos': reference_values.size,
        'n_comparison_videos': comparison_values.size,
        'reference_mean_MT_minus_WT': (
            reference_values.mean() if reference_values.size else np.nan
        ),
        'comparison_mean_MT_minus_WT': (
            comparison_values.mean() if comparison_values.size else np.nan
        ),
        'interaction_comparison_minus_reference': estimate,
        'bootstrap_CI_low': ci_low,
        'bootstrap_CI_high': ci_high,
        'welch_p': welch_p,
        'mann_whitney_p': mann_whitney_p,
    }


def add_cd40_mlt_pairwise_ttests(
        ax, plot_df, value_name, ordered_groups, alpha=0.05):
    """Add median markers and unadjusted pairwise Welch t-tests.

    This helper is intentionally defined inside the CD40L-versus-mLT section
    so that running only this section in IPython does not reuse an older helper
    left in the interactive namespace.
    """
    from itertools import combinations

    datasets = []
    positions = []

    for position, group in enumerate(ordered_groups):
        values = plot_df.loc[
            plot_df['Group'] == group, value_name
        ].dropna().to_numpy()
        if values.size:
            median_value = np.median(values)
            datasets.append(values)
            positions.append(position)

            # Median: horizontal black line.
            ax.plot(
                [position - 0.20, position + 0.20],
                [median_value, median_value],
                color='0.05',
                linewidth=2,
                solid_capstyle='butt',
                zorder=4
            )

    if len(datasets) < 2:
        return

    pairs = list(combinations(range(len(datasets)), 2))
    p_values = np.array([
        stats.ttest_ind(
            datasets[first],
            datasets[second],
            equal_var=False,
            nan_policy='omit'
        ).pvalue
        for first, second in pairs
    ])

    significant_pairs = []
    for (first, second), p_value in zip(pairs, p_values):
        if np.isfinite(p_value) and p_value < alpha:
            significant_pairs.append(
                (positions[first], positions[second], p_value)
            )

    if not significant_pairs:
        return

    all_values = plot_df[value_name].dropna().to_numpy()
    y_min = np.min(all_values)
    y_max = np.max(all_values)
    y_range = y_max - y_min
    if not np.isfinite(y_range) or y_range == 0:
        y_range = max(abs(y_max), 1.0)

    bracket_base = y_max + 0.08 * y_range
    bracket_step = 0.13 * y_range
    for level, (first, second, p_value) in enumerate(significant_pairs):
        y = bracket_base + level * bracket_step
        bracket_height = 0.025 * y_range
        ax.plot(
            [first, first, second, second],
            [y, y + bracket_height, y + bracket_height, y],
            color='0.2',
            linewidth=0.7,
            clip_on=False
        )
        if p_value < 0.0001:
            significance = '****'
        elif p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        else:
            significance = '*'
        ax.text(
            (first + second) / 2,
            y + bracket_height,
            f'{significance}  p={p_value:.3g}',
            ha='center',
            va='bottom',
            fontsize=6,
            color='0.2'
        )

    required_top = bracket_base + len(significant_pairs) * bracket_step
    ax.set_ylim(ax.get_ylim()[0], required_top)


def plot_two_treatment_video_values(
        ax, video_summary, value_name, ylabel, treatment_order_local):
    """Plot paired WT/MT values for an arbitrary treatment pair."""
    plot_df = video_summary.copy()
    plot_df['Group'] = (
        plot_df['Type'].astype(str) + ' ' + plot_df['Inhibition'].astype(str)
    )
    local_group_order = [
        typ + ' ' + treatment
        for treatment in treatment_order_local
        for typ in type_order
    ]
    x_lookup = {
        group: idx for idx, group in enumerate(local_group_order)
    }

    for (_, treatment), video_part in plot_df.groupby(
            ['Video', 'Inhibition']):
        video_part = video_part[
            video_part['Group'].isin(local_group_order)
        ]
        if video_part.shape[0] == 2:
            xs = [x_lookup[group] for group in video_part['Group']]
            ax.plot(
                xs,
                video_part[value_name],
                color='0.75',
                linewidth=0.6,
                zorder=1
            )

    local_palette = {
        group: ('#888888' if group.startswith('WT') else '#CC6677')
        for group in local_group_order
    }
    sns.stripplot(
        data=plot_df,
        x='Group',
        y=value_name,
        order=local_group_order,
        palette=local_palette,
        size=4,
        jitter=0.08,
        edgecolor='0.2',
        linewidth=0.3,
        ax=ax,
        zorder=2
    )
    add_cd40_mlt_pairwise_ttests(
        ax,
        plot_df,
        value_name,
        local_group_order
    )
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticklabels(
        [
            ('WT' if group.startswith('WT') else 'MT')
            + '\n'
            + group.rsplit(' ', 1)[-1]
            for group in local_group_order
        ],
        fontsize=7
    )
    ax.tick_params(axis='y', labelsize=7, width=0.8)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(0.8)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_two_treatment_contrast(
        ax, video_contrasts, ylabel, treatment_order_local):
    """Plot MT-WT video contrasts for an arbitrary treatment pair."""
    sns.stripplot(
        data=video_contrasts,
        x='Inhibition',
        y='MT_minus_WT',
        order=treatment_order_local,
        color='0.25',
        size=4,
        jitter=0.08,
        ax=ax
    )
    ax.axhline(0, color='0.65', linestyle='--', linewidth=0.8)
    for x, treatment in enumerate(treatment_order_local):
        values = video_contrasts.loc[
            video_contrasts['Inhibition'] == treatment, 'MT_minus_WT'
        ].dropna().to_numpy()
        if values.size:
            median = np.median(values)
            ax.plot(
                [x - 0.12, x + 0.12],
                [median, median],
                color='0.05',
                linewidth=2,
                solid_capstyle='butt',
                zorder=3
            )
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticklabels(treatment_order_local, fontsize=7)
    ax.tick_params(axis='y', labelsize=7, width=0.8)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(0.8)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


cd40_mlt_video_summaries = []
cd40_mlt_video_contrasts = []
cd40_mlt_stats = []
cd40_mlt_treatment_order = ['CD40L', 'mLTβR']

for endpoint, spec in dlz_endpoint_specs.items():
    value_column = endpoint + '_video_summary'
    video_summary = (
        cd40_mlt_valid
        .groupby(
            ['Video', 'Inhibition', 'Acquisition_year', 'Type'],
            observed=True
        )
        .agg(
            n_valid=(endpoint, 'size'),
            **{value_column: (endpoint, spec['summary'])}
        )
        .reset_index()
    )
    video_summary['endpoint'] = endpoint
    video_summary['direction'] = spec['direction']
    cd40_mlt_video_summaries.append(video_summary)

    contrast = get_video_contrasts(video_summary, value_column)
    contrast = contrast.merge(
        video_summary[['Video', 'Acquisition_year']].drop_duplicates(),
        on='Video',
        how='left',
        validate='one_to_one'
    )
    contrast['endpoint'] = endpoint
    contrast['direction'] = spec['direction']
    cd40_mlt_video_contrasts.append(contrast)

    stats_all = two_treatment_interaction_statistics(
        contrast,
        reference='CD40L',
        comparison='mLTβR',
        feature=endpoint,
        endpoint='all_available_videos'
    )
    stats_all['direction'] = spec['direction']
    cd40_mlt_stats.append(stats_all)

    contrast_2024 = contrast[contrast['Acquisition_year'] == 2024].copy()
    stats_2024 = two_treatment_interaction_statistics(
        contrast_2024,
        reference='CD40L',
        comparison='mLTβR',
        feature=endpoint,
        endpoint='2024_only_primary'
    )
    stats_2024['direction'] = spec['direction']
    cd40_mlt_stats.append(stats_2024)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    plot_two_treatment_video_values(
        axes[0],
        video_summary,
        value_column,
        spec['ylabel'],
        cd40_mlt_treatment_order
    )
    plot_two_treatment_contrast(
        axes[1],
        contrast,
        'MT − WT\nall videos',
        cd40_mlt_treatment_order
    )
    plot_two_treatment_contrast(
        axes[2],
        contrast_2024,
        'MT − WT\n2024 only',
        cd40_mlt_treatment_order
    )
    axes[1].set_title(
        'All: '
        f"{stats_all['interaction_comparison_minus_reference']:.3g}\n"
        f"95% CI [{stats_all['bootstrap_CI_low']:.3g}, "
        f"{stats_all['bootstrap_CI_high']:.3g}]",
        fontsize=7
    )
    axes[2].set_title(
        '2024: '
        f"{stats_2024['interaction_comparison_minus_reference']:.3g}\n"
        f"95% CI [{stats_2024['bootstrap_CI_low']:.3g}, "
        f"{stats_2024['bootstrap_CI_high']:.3g}]",
        fontsize=7
    )
    fig.suptitle(endpoint + '\n' + spec['direction'], fontsize=9)
    fig.tight_layout()
    fig.savefig(
        cd40_mlt_path + endpoint + '_CD40L_vs_mLT_focused_dLZ.png',
        dpi=300,
        bbox_inches='tight'
    )
    fig.savefig(
        cd40_mlt_path + 'svg/'
        + endpoint
        + '_CD40L_vs_mLT_focused_dLZ.svg',
        bbox_inches='tight'
    )
    plt.close(fig)

cd40_mlt_stats_df = pd.DataFrame(cd40_mlt_stats)
for p_column in ['welch_p', 'mann_whitney_p']:
    p_values = cd40_mlt_stats_df[p_column].to_numpy(dtype=float)
    valid = np.isfinite(p_values)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    if valid.any():
        valid_p = p_values[valid]
        order = np.argsort(valid_p)
        ranked_p = valid_p[order]
        ranked_q = ranked_p * valid_p.size / np.arange(1, valid_p.size + 1)
        ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
        valid_q = np.empty_like(ranked_q)
        valid_q[order] = np.clip(ranked_q, 0, 1)
        adjusted[valid] = valid_q
    cd40_mlt_stats_df[p_column.replace('_p', '_FDR_q')] = adjusted

cd40_mlt_stats_df.to_csv(
    cd40_mlt_path + 'dLZ_CD40L_vs_mLT_interaction_statistics.csv',
    index=False
)
pd.concat(
    cd40_mlt_video_summaries,
    ignore_index=True,
    sort=False
).to_csv(
    cd40_mlt_path + 'dLZ_CD40L_vs_mLT_video_level_summaries.csv',
    index=False
)
pd.concat(
    cd40_mlt_video_contrasts,
    ignore_index=True,
    sort=False
).to_csv(
    cd40_mlt_path + 'dLZ_CD40L_vs_mLT_MT_minus_WT_contrasts.csv',
    index=False
)


####################################### Quantify Tfh and FDC interaction frequency #############################################
int_time = 15
features = ['T_contact_times', 'FDC_contact_times']
names = ['Tfh', 'FDC']

for typ in ['wt_B-cell', 'mt_B-cell']:
    for feature, name in zip(features, names):
        persistent_int_freqs_datasets = {}
        persistent_int_freq_per_cellnumbers_datasets = {}
        total_n_contacts_per_cellnumbers_datasets = {}
        persistent_int_freq_per_cellcontacts_datasets = {}
        low_contact_freq_per_cellnumbers_datasets = {}

        videos = np.unique(df['Video'])
        for cell_type in np.unique(df['Inhibition']):
            df_part = df[(df['Inhibition'] == cell_type)&(df['Type']==typ)]
            persistent_int_freqs = []
            persistent_int_freq_per_cellnumbers = []
            total_n_contacts_per_cellnumbers = []
            persistent_int_freq_per_cellcontacts = []
            low_contact_freq_per_cellnumbers = []
            videos = np.unique(df_part['Video'])
            for video in videos:
                #if 'A' in video and cell_type == 'mt_B-cell':
                # if '-A' in video:
                #     continue
                df_video = df_part[df_part['Video'] == video]
                if df_video.shape[0] == 0:
                    continue
                data = df_video[feature]
                mask = ~np.isnan(data)
                data = data[mask]
                persistent_int_freq = sum(data >= int_time)
                total_n_contact = sum(data)

                persistent_int_freq_per_cellnumber = persistent_int_freq / df_video.shape[0]
                total_n_contacts_per_cellnumber = total_n_contact / df_video.shape[0]
                if sum(data) != 0:
                    persistent_int_freq_per_cellcontact = persistent_int_freq / sum(data)
                elif sum(data) == 0:
                    persistent_int_freq_per_cellcontact = 0
                low_contact_freq_per_cellnumber = sum((1<=data) & (data<=3)) / df_video.shape[0]

                persistent_int_freqs.append(persistent_int_freq)
                persistent_int_freq_per_cellnumbers.append(persistent_int_freq_per_cellnumber)
                total_n_contacts_per_cellnumbers.append(total_n_contacts_per_cellnumber)
                persistent_int_freq_per_cellcontacts.append(persistent_int_freq_per_cellcontact)
                low_contact_freq_per_cellnumbers.append(low_contact_freq_per_cellnumber)

            persistent_int_freqs_datasets[cell_type] = persistent_int_freqs
            persistent_int_freq_per_cellnumbers_datasets[cell_type] = persistent_int_freq_per_cellnumbers
            total_n_contacts_per_cellnumbers_datasets[cell_type] = total_n_contacts_per_cellnumbers
            persistent_int_freq_per_cellcontacts_datasets[cell_type] = persistent_int_freq_per_cellcontacts
            low_contact_freq_per_cellnumbers_datasets[cell_type] = low_contact_freq_per_cellnumbers

        replace_keys = {'Control':'Control', 'IgG':'IgG', 'CD40L':'CD40L', 'mLT':'mLT'}
        #new_order = ['Control', 'ControlAb', 'CD40LAb']
        #new_order = ['Control', 'IgG', 'CD40L', 'mLT']
        new_order = ['Control', 'IgG', 'CD40L', 'mLT']
        persistent_int_freqs_datasets = change_dict_order(persistent_int_freqs_datasets, new_order)
        persistent_int_freqs_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freqs_datasets.items() }

        persistent_int_freq_per_cellnumbers_datasets = change_dict_order(persistent_int_freq_per_cellnumbers_datasets, new_order)
        persistent_int_freq_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freq_per_cellnumbers_datasets.items() }

        total_n_contacts_per_cellnumbers_datasets = change_dict_order(total_n_contacts_per_cellnumbers_datasets, new_order)
        total_n_contacts_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in total_n_contacts_per_cellnumbers_datasets.items() }

        persistent_int_freq_per_cellcontacts_datasets = change_dict_order(persistent_int_freq_per_cellcontacts_datasets, new_order)
        persistent_int_freq_per_cellcontacts_datasets = {replace_keys.get(k, k):v  for (k,v) in persistent_int_freq_per_cellcontacts_datasets.items() }

        low_contact_freq_per_cellnumbers_datasets = change_dict_order(low_contact_freq_per_cellnumbers_datasets, new_order)
        low_contact_freq_per_cellnumbers_datasets = {replace_keys.get(k, k):v  for (k,v) in low_contact_freq_per_cellnumbers_datasets.items() }

        #colors=('#888888', '#6699CC', '#CC6677')
        colors = ('#888888', '#CC6677', '#6699CC', '#44AA99')

        draw_custom_bar_plot(persistent_int_freqs_datasets, path, file_name='Total %s %s interaction frequency'%(name, typ),
                             strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))
        draw_custom_bar_plot(persistent_int_freq_per_cellnumbers_datasets, path, file_name='%s %s persistent interaction frequency per cell number'%(name, typ),
                             strip_plot=True,colors=colors, test='t-test', pvalue=True, figsize=(1,2))
        draw_custom_bar_plot(total_n_contacts_per_cellnumbers_datasets, path, file_name='number of %s %s contacts per cell number'%(name, typ),
                            strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))
        draw_custom_bar_plot(persistent_int_freq_per_cellcontacts_datasets, path, file_name='%s %s persistent interaction frequency per number of contacts'%(name, typ),
                             strip_plot=True, colors=colors, test='t-test', pvalue=True, figsize=(1,2))
        draw_custom_bar_plot(low_contact_freq_per_cellnumbers_datasets, path, file_name='%s %s low contact time frequency per cell number'%(name, typ),
                             strip_plot=True, colors=colors, test='wilcoxon-ranksum', pvalue=True, figsize=(1,2))


####################################### Average FDC and Tfh distance kde MT vs WT #############################################
replace_keys = {'mt_B-cell':'MT', 'wt_B-cell':'WT'}

coloc_features = ['FDC_distance_average', 'T_distance_average']
ranges = [(0, 20), (0,30)]
names = ['Distance to FDC', 'Distance to Tfh']

for idx, coloc_feature in enumerate(coloc_features):

    for condition in np.unique(df['Type']):
        dataset = {}
        for group in np.unique(df['Inhibition']):
            data = df[(df['Type'] == condition)&(df['Inhibition'] == group)][coloc_feature]
            dataset[group] = np.array(data)
        #new_order = ['Control', 'IgG', 'CD40L', 'mLT']
        new_order = ['IgG', 'CD40L', 'mLT']
        ordered_dataset = change_dict_order(dataset, new_order)
        dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

        sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

        font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 8}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 1

        colors = ('#888888', '#CC6677', '#6699CC', '#44AA99')
        fig, ax = plt.subplots(figsize=(2,2))
        for i, key in enumerate(dict_datasets):
            sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=ranges[idx], color=colors[i], label=key)

        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color('0.2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('%s (μm)'%names[idx], fontsize=8, weight='normal', color='0.2')
        ax.set_ylabel('Density', fontsize=8, weight='normal', color='0.2')
        plt.xticks(fontsize=8, color='0.2', weight='normal')
        plt.yticks(fontsize=8, color='0.2', weight='normal')

        plt.legend(frameon=False, prop = {'weight':'normal', 'size':8}, labelcolor='0.2')

        plt.savefig(path+'%s %s.png'%(condition, coloc_feature), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

        if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/')
        plt.savefig(path + 'svg/%s %s.svg'%(condition, coloc_feature), bbox_inches='tight')
        plt.close()
        plt.clf()

####################################### Zone distance kde MT vs WT #############################################
replace_keys = {'mt_B-cell':'MT', 'wt_B-cell':'WT'}

coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']
ranges = [(0, 60), (0, 30), (0, 100)]
names = ['Distance to DZ', 'Distance to sLZ', 'Distance to dLZ']

for idx, coloc_feature in enumerate(coloc_features):

    for condition in np.unique(df['Type']):
        dataset = {}
        for group in np.unique(df['Inhibition']):
            data = df[(df['Type'] == condition)&(df['Inhibition'] == group)][coloc_feature]
            dataset[group] = np.array(data)
        # new_order = ['Control', 'IgG', 'CD40L', 'mLT']
        new_order = ['IgG', 'CD40L', 'mLT']
        ordered_dataset = change_dict_order(dataset, new_order)
        dict_datasets = {replace_keys.get(k, k):v  for (k,v) in ordered_dataset.items() }

        sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

        font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 8}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 1

        colors = ('#888888', '#CC6677', '#6699CC', '#44AA99')
        fig, ax = plt.subplots(figsize=(2,2))
        for i, key in enumerate(dict_datasets):
            sns.kdeplot(data=dict_datasets[key], fill=True, linewidth=1, clip=ranges[idx], color=colors[i], label=key)

        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color('0.2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('%s (μm)'%names[idx], fontsize=8, weight='normal', color='0.2')
        ax.set_ylabel('Density', fontsize=8, weight='normal', color='0.2')
        plt.xticks(fontsize=8, color='0.2', weight='normal')
        plt.yticks(fontsize=8, color='0.2', weight='normal')

        plt.legend(frameon=False, prop = {'weight':'normal', 'size':8}, labelcolor='0.2')

        plt.savefig(path+'%s %s.png'%(condition, coloc_feature), dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()

        if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/')
        plt.savefig(path + 'svg/%s %s.svg'%(condition, coloc_feature), bbox_inches='tight')
        plt.close()
        plt.clf()


############# Plot LZ vs DZ jointplot for each cell type ###############

for cell_type in ['mt_B-cell', 'wt_B-cell']:
    for group in ['Control', 'IgG', 'CD40L', 'mLT']:
        df_part = df[(df['Type']==cell_type)&(df['Inhibition']==group)].reset_index(drop=True)
        df_part = df_part[df_part['Zone']!='DZ-sLZ'].reset_index(drop=True)

        df_part.loc[(df_part['Zone'] == 'DZ'), 'Zone1'] = 'DZ'
        #df_part.loc[(df_part['Zone'] == 'DZ-sLZ'), 'Zone1'] = 'DZ-sLZ'
        df_part.loc[( (df_part['Zone'] == 'sLZ') | (df_part['Zone'] == 'dLZ') | (df_part['Zone'] == 'sLZ-dLZ') ), 'Zone1'] = 'LZ'
        df_part['Zone1'] = df_part.Zone1.astype(str)

        print(cell_type, group, df_part[df_part['Zone1']=='DZ'].shape[0], df_part[df_part['Zone1']=='DZ-sLZ'].shape[0], df_part[df_part['Zone1']=='LZ'].shape[0])

        draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s %s DZ vs LZ jointplot'%(cell_type, group), hue="Zone1", hue_order=['LZ', 'DZ'],
                       colors=('#E69965', '#BAC8DA'), fill=False, legend=False, thresh=0.2, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')



############# Plot sLZ vs dLZ jointplot for each cell type ###############

for cell_type in ['mt_B-cell', 'wt_B-cell']:
    for group in ['Control', 'IgG', 'CD40L', 'mLT']:
        df_part = df[(df['Type']==cell_type)&(df['Inhibition']==group)].reset_index(drop=True)

        df_part.loc[(df_part['Zone'] == 'sLZ'), 'Zone1'] = 'sLZ'
        #df_part.loc[(df_part['Zone'] == 'sLZ-dLZ'), 'Zone1'] = 'sLZ-dLZ'
        df_part.loc[(df_part['Zone'] == 'dLZ'), 'Zone1'] = 'dLZ'
        df_part['Zone1'] = df_part.Zone1.astype(str)

        print(cell_type, group, df_part[df_part['Zone1']=='sLZ'].shape[0], df_part[df_part['Zone1']=='sLZ-dLZ'].shape[0], df_part[df_part['Zone1']=='dLZ'].shape[0])

        draw_jointplot(xs='PC1', y='PC2', df=df_part, path=path, file_name='%s %s sLZ vs dLZ jointplot'%(cell_type, group), hue="Zone1", hue_order=['sLZ', 'dLZ'],
                       colors=('#4F609C', '#8A4F21'), fill=False, legend=False, thresh=0.25, height=4, ratio=5, space=0, xlabels='UMAP1', ylabel='UMAP2')

############# All Kmeans distribution heatmap ###############

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_ = df_[(df_['Zone']!='DZ-sLZ')&(df_['Zone']!='sLZ-dLZ')&(df_['Zone'].notna())].reset_index(drop=True)

df_['Type_Zone_Inhibition'] = df_['Type'].astype(str) + ' ' + df_['Zone'].astype(str) + ' ' + df_['Inhibition'].astype(str)
min_cells = 28
condition_counts = df_['Type_Zone_Inhibition'].value_counts()
valid_conditions = condition_counts[condition_counts >= min_cells].index
df_ = df_[df_['Type_Zone_Inhibition'].isin(valid_conditions)].reset_index(drop=True)
draw_cluster_distribution_heatmap(df_, path, file_name='all_kmeans_heatmap', condition_name='Type_Zone_Inhibition', annot=False,
                                  cluster_type='kmeans', row_cluster=True, col_cluster=False, vmax=30, cmap=cmc.bilbao_r, figsize=(5,8))

############# Inhibit only Kmeans distribution heatmap ###############

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_ = df_[(df_['Zone']!='DZ-sLZ')&(df_['Zone']!='sLZ-dLZ')&(df_['Zone'].notna())].reset_index(drop=True)
df_ = df_[df_['Inhibition']!='Control'].reset_index(drop=True)

df_['Type_Zone_Inhibition'] = df_['Type'].astype(str) + ' ' + df_['Zone'].astype(str) + ' ' + df_['Inhibition'].astype(str)
condition_counts = df_['Type_Zone_Inhibition'].value_counts()
valid_conditions = condition_counts[condition_counts >= min_cells].index
df_ = df_[df_['Type_Zone_Inhibition'].isin(valid_conditions)].reset_index(drop=True)
draw_cluster_distribution_heatmap(df_, path, file_name='inhibit_kmeans_heatmap', condition_name='Type_Zone_Inhibition', annot=False,
                                  cluster_type='kmeans', row_cluster=True, col_cluster=False, vmax=30, cmap=cmc.bilbao_r, figsize=(5,6))

############# WT Inhibit only Kmeans distribution heatmap ###############

df_ = df.copy()
df_ = df_.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_ = df_[(df_['Zone']!='DZ-sLZ')&(df_['Zone']!='sLZ-dLZ')&(df_['Zone'].notna())].reset_index(drop=True)

#df_ = df_[df_['Inhibition']!='Control'].reset_index(drop=True)
df_ = df_[df_['Type']=='MT'].reset_index(drop=True)

df_['Type_Zone_Inhibition'] = df_['Type'].astype(str) + ' ' + df_['Zone'].astype(str) + ' ' + df_['Inhibition'].astype(str)
condition_counts = df_['Type_Zone_Inhibition'].value_counts()
valid_conditions = condition_counts[condition_counts >= min_cells].index
df_ = df_[df_['Type_Zone_Inhibition'].isin(valid_conditions)].reset_index(drop=True)
draw_cluster_distribution_heatmap(df_, path, file_name='MT inhibit_kmeans_heatmap', condition_name='Type_Zone_Inhibition', annot=False,
                                  cluster_type='kmeans', row_cluster=True, col_cluster=False, vmax=30, cmap=cmc.bilbao_r, figsize=(5,5))


###################### Plot Zone motility feature violin plot for MT and WT  ############################

df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for cell_type in ['wt_B-cell', 'mt_B-cell']:
    df_part = df[(df['Type'] == cell_type)].reset_index(drop=True)
    if not os.path.isdir(path + '%s Zone motility box plot/'%cell_type):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s Zone motility box plot/'%cell_type)

    for feature_name in feature_list:
        condition_name='Inhibition'
        print(feature_name)
        dataset={}
        for group in ['Control', 'IgG', 'CD40L', 'mLT']:
            for zone in ['DZ','sLZ', 'dLZ']:
        #for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
                data = df_part[(df_part[condition_name] == group)&(df_part['Zone'] == zone)][feature_name]

                dataset[group+' '+str(zone)] = np.array(data)

        # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
        #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ',}


        #dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}
        draw_custom_bar_plot(dataset, path + '%s Zone motility box plot/'%cell_type, file_name=feature_name,
                                colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC', '#44AA99', '#44AA99', '#44AA99'),
                                strip_plot=False, test='mann-whitney', pvalue=True, figsize=(3, 3))
        # draw_custom_violin_plot(dataset_renamed, path + 'Zone motility box plot/', file_name=feature_name,
        #                         colors=('#888888', '#888888', '#888888', '#888888', '#888888',
        #                                 '#CC6677', '#CC6677', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))


###################### Plot Experimental motility feature plots for Zones  ############################
df.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

videos = np.unique(df['Video'])


for cell_type in ['wt_B-cell', 'mt_B-cell']:
    df_part = df[(df['Type'] == cell_type)].reset_index(drop=True)
    if not os.path.isdir(path + '%s experimental_zone_motility_feature/'%cell_type):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s experimental_zone_motility_feature/'%cell_type)

    for feature_name in feature_list:
        dataset = {}
        for group in ['Control', 'IgG', 'CD40L']:
            for zone in ['DZ','sLZ', 'dLZ']:
                df_part_temp = df_part[(df_part['Inhibition'] == group)&(df_part['Zone'] == zone)].reset_index(drop=True)
                avgs = []
                for video in videos:
                    df_video = df_part_temp[df_part_temp['Video'] == video]
                    if df_video.shape[0] == 0:
                        continue
                    data = df_video[feature_name]
                    avg = np.mean(data)
                    avgs.append(avg)
                dataset[group + ' ' + str(zone)] = avgs

        #new_order = ['wt_B-cell', 'wt_B-cell']
        #ordered_dataset = change_dict_order(dataset, new_order)
        # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
        #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ', }
        # dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

        draw_custom_bar_plot(dataset, path+'%s experimental_zone_motility_feature/'%cell_type, file_name=feature_name,
                             colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC', '#44AA99', '#44AA99', '#44AA99'),
                             strip_plot=True,test='mann-whitney', pvalue=True, figsize=(3, 3))

###################### Plot Zone interaction feature bar plot for MT and WT  ############################

df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone', 'Inhibition'])

for cell_type in ['wt_B-cell', 'mt_B-cell']:
    df_part = df[(df['Type'] == cell_type)].reset_index(drop=True)
    if not os.path.isdir(path + '%s Zone int box plot/'%cell_type):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s Zone int box plot/'%cell_type)

    for feature_name in feature_list:
        condition_name='Inhibition'
        dataset={}
        for group in ['IgG', 'CD40L', 'mLT']:
            for zone in ['DZ','sLZ', 'dLZ']:
        #for group in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
                data = df_part[(df_part[condition_name] == group)&(df_part['Zone'] == zone)][feature_name]

                dataset[group+' '+str(zone)] = np.array(data)

        # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
        #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ',}


        #dataset_renamed = {rename_keys.get(k, k): v for (k, v) in dataset.items()}
        draw_custom_bar_plot(dataset, path + '%s Zone int box plot/'%cell_type, file_name=feature_name,
                                colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC', '#44AA99', '#44AA99', '#44AA99'),
                                strip_plot=False, test='mann-whitney', pvalue=True, figsize=(3, 3))
        # draw_custom_violin_plot(dataset_renamed, path + 'Zone motility box plot/', file_name=feature_name,
        #                         colors=('#888888', '#888888', '#888888', '#888888', '#888888',
        #                                 '#CC6677', '#CC6677', '#CC6677', '#CC6677', '#CC6677'), test='mann-whitney', pvalue=True, figsize=(2, 2))


###################### Plot Experimental interaction feature plots for Zones  ############################
df.columns.get_loc('quality_FDC_approach_times')
feature_list = df.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone', 'Inhibition'])

videos = np.unique(df['Video'])


for cell_type in ['wt_B-cell', 'mt_B-cell']:
    df_part = df[(df['Type'] == cell_type)].reset_index(drop=True)
    if not os.path.isdir(path + '%s experimental_zone_int_feature/'%cell_type):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + '%s experimental_zone_int_feature/'%cell_type)

    for feature_name in feature_list:
        dataset = {}
        for group in ['Control', 'IgG', 'CD40L']:
            for zone in ['DZ','sLZ', 'dLZ']:
                df_part_temp = df_part[(df_part['Inhibition'] == group)&(df_part['Zone'] == zone)].reset_index(drop=True)
                avgs = []
                for video in videos:
                    df_video = df_part_temp[df_part_temp['Video'] == video]
                    if df_video.shape[0] == 0:
                        continue
                    data = df_video[feature_name]
                    avg = np.mean(data)
                    avgs.append(avg)
                dataset[group + ' ' + str(zone)] = avgs

        #new_order = ['wt_B-cell', 'wt_B-cell']
        #ordered_dataset = change_dict_order(dataset, new_order)
        # rename_keys = {'wt_B-cell_DZ': 'WT DZ', 'wt_B-cell_sLZ': 'WT sLZ', 'wt_B-cell_dLZ': 'WT dLZ',
        #                'mt_B-cell_DZ': 'MT DZ', 'mt_B-cell_sLZ': 'MT sLZ', 'mt_B-cell_dLZ': 'MT dLZ', }
        # dict_datasets = {rename_keys.get(k, k): v for (k, v) in dataset.items()}

        draw_custom_bar_plot(dataset, path+'%s experimental_zone_int_feature/'%cell_type, file_name=feature_name,
                             colors=('#CC6677', '#CC6677', '#CC6677', '#6699CC', '#6699CC', '#6699CC', '#44AA99', '#44AA99', '#44AA99'),
                             strip_plot=True,test='mann-whitney', pvalue=True, figsize=(3, 3))


#################################### all motility features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_.columns.get_loc('morpho_displ_autocorr_3')
feature_list = df_.columns[:134].drop(['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z'])

for typ in ['WT', 'MT']:
    df_part_ = df_[df_['Type']==typ].reset_index(drop=True)
    #df_part_ = df_part_[df_part_['Inhibition'].isin(['Control', 'mLT'])].reset_index(drop=True)
    coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']

    for idx in [0, 1, 2]:
        coloc_feature = coloc_features[idx]
        if idx == 0:
            xlabel = 'Distance to DZ (µm)'
            custsom_range = (0, 40)
            stepsize = 4
        elif idx == 1:
            xlabel = 'Distance to sLZ (µm)'
            custsom_range = (0, 32)
            stepsize = 8
        elif idx == 2:
            xlabel = 'Distance to dLZ (µm)'
            custsom_range = (0, 40)
            stepsize = 4

        valid_features = [feature for feature in feature_list if feature in df_part_.columns]
        for feature_name in valid_features:
            df_plot = df_part_.copy()
            df_plot[coloc_feature] = pd.to_numeric(df_plot[coloc_feature], errors='coerce')
            df_plot[feature_name] = pd.to_numeric(df_plot[feature_name], errors='coerce')
            df_plot = df_plot[np.isfinite(df_plot[coloc_feature]) & np.isfinite(df_plot[feature_name])].reset_index(drop=True)

            draw_lineplot_by_custom_ranges(df_plot, path, folder_name='%s motility_feature_wrt_%s'%(typ, coloc_feature), feature_list=[feature_name],
                                           condition_name='Inhibition', custsom_range=custsom_range, stepsize=stepsize, range_feature=coloc_feature,
                                               color_list=['#888888', '#44AA99', '#CC6677', '#6699CC'], marker_list=['o', '^', '.', 's'], figsize=(4,4), x_label=xlabel,
                                           replace_keys=None, pvalue=True, test='mann-whitney')


#################################### all interaction features wrt avg Zone distance ####################################
df_ = df.replace({'Type': {'wt_B-cell': 'WT', 'mt_B-cell': 'MT'}})
df_.columns.get_loc('quality_FDC_approach_times')
feature_list = df_.columns[148:].drop(['PC1', 'PC2', 'kmeans', 'Zone', 'Inhibition'])

for typ in ['WT', 'MT']:
    df_part_ = df_[df_['Type']==typ].reset_index(drop=True)
    coloc_features = ['DZ_distance_average', 'LZ_distance_average', 'Core_distance_average']
    for idx in [0, 1, 2]:
        coloc_feature = coloc_features[idx]
        if idx == 0:
            xlabel = 'Distance to DZ (µm)'
            custsom_range = (0, 40)
            stepsize = 4
        elif idx == 1:
            xlabel = 'Distance to sLZ (µm)'
            custsom_range = (0, 32)
            stepsize = 8
        elif idx == 2:
            xlabel = 'Distance to dLZ (µm)'
            custsom_range = (0, 40)
            stepsize = 4

        valid_features = [feature for feature in feature_list if feature in df_part_.columns]
        for feature_name in valid_features:
            df_plot = df_part_.copy()
            df_plot[coloc_feature] = pd.to_numeric(df_plot[coloc_feature], errors='coerce')
            df_plot[feature_name] = pd.to_numeric(df_plot[feature_name], errors='coerce')
            df_plot = df_plot[np.isfinite(df_plot[coloc_feature]) & np.isfinite(df_plot[feature_name])].reset_index(drop=True)

            draw_lineplot_by_custom_ranges(df_plot, path, folder_name='%s int_feature_wrt_%s'%(typ, coloc_feature), feature_list=[feature_name],
                                           condition_name='Inhibition', custsom_range=custsom_range, stepsize=stepsize, range_feature=coloc_feature,
                                               color_list=['#888888', '#44AA99', '#CC6677', '#6699CC'], marker_list=['o', '^', '.', 's'], figsize=(4,4), x_label=xlabel,
                                           replace_keys=None, pvalue=True, test='mann-whitney')
