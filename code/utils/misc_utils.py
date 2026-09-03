# Author: Chanhong Min <cmin11@jhmi.edu>

"""Functions for general purposes"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import scipy
import os

def reduced_labels(df, duration):
    """ Generate reduced features where each row is a trajectory
        Parameters:
        ----------
        df: pandas dataframe
            raw df where each row is one cell state at time t
        duration: int
            Number of time frames for each cell trajectory (all trajectories should have same duration)
        Returns:
        -------
        other_features_data: pandas dataframe
            dataframe with reduced label features
        """
    other_features_data = {}
    for feature_name in df.columns:
        row_values = []
        for traj_idx in range(int(df.shape[0] / duration)):  # For each cell trajectory
            traj_data_temp = df[duration * traj_idx:duration * (traj_idx + 1)]
            row_value = pd.unique(traj_data_temp[feature_name])
            if row_value.shape[0] > 1: # ex) Overlapped Volume = multiple values per trajectory
                row_values.append(traj_data_temp[feature_name].values)
            elif row_value.shape[0] == 1: # ex) cell_id = 1 value per trajectory
                row_values.append(row_value[0])

        other_features_data[feature_name] = row_values
    return pd.DataFrame(other_features_data)


def reduced_label_for_overlapped_volume(df, duration):
    """ Generate reduced features where each row is a trajectory for overlapped volume data
        Parameters:
        ----------
        df: pandas dataframe
            raw df where each row is one cell state at time t
        duration: int
            Number of time frames for each cell trajectory (all trajectories should have same duration)
        Returns:
        -------
        other_features_data: pandas dataframe
            dataframe with reduced label features
        """
    other_features_data = {}
    for feature_name in df.columns:
        aa = []
        for traj_idx in range(int(df.shape[0] / duration)):
            traj_data_temp = df[duration * traj_idx:duration * (traj_idx + 1)]
            row_values = pd.unique(traj_data_temp[feature_name])
            if any(txt in feature_name for txt in ('Overlapped', 'Shortest_Distance')):
                aa.append(traj_data_temp[feature_name].values)
            else:
                if row_values.shape[0] == 1:
                    aa.append(row_values[0])
                else:
                    aa.append(traj_data_temp[feature_name].values)
        other_features_data[feature_name] = aa
    return pd.DataFrame(other_features_data)


def convert_df_indices_to_df_duration(indices, duration):
    indices = list(indices)
    df_duration_indices = []
    for i in indices:
        df_duration_indices.extend(range(i * duration, i * duration + duration))

    df_duration_indices = np.array(df_duration_indices)
    return df_duration_indices

def dict_to_array(trajectories):
    a = []
    for traj_idx in trajectories:
        traj = trajectories[traj_idx]
        a.append(traj)
    trajectories_array = np.array(a)
    return trajectories_array

def array_to_dict(trajectories):
    trajectories_dict = {}
    for traj_idx in range(trajectories.shape[0]):
        traj = trajectories[traj_idx]
        trajectories_dict[traj_idx]=traj
    return trajectories_dict

def change_dict_order(orig_dict, new_order):
    ''' Change the order of dictionary based on the specified keys
    Parameters:
    ----------
    orig_dict: dict
        original data in dictinary form
    new_order: list
        list of keys of desired order

    Returns:
    -------
    ordered_dict: dict
        new ordered data in dictionary form
    '''
    from collections import OrderedDict
    ordered_dict = OrderedDict(orig_dict)
    ordered_dict = OrderedDict((key, ordered_dict[key]) for key in new_order)

    return ordered_dict

def flatten_list_of_list(l):
    return [item for sublist in l for item in sublist]


def flatten_nested_dict(d):
    ''' Get values from nested dictionary
    Parameters:
    ----------
    d: dict
        nested dictionary

    Returns:
    -------
    list(get_all_values(d)): list
        list of all value elements
    '''

    from typing import Iterable

    def get_all_values(d):
        if isinstance(d, dict):
            for v in d.values():
                yield from get_all_values(v)
        elif isinstance(d, Iterable) and not isinstance(d, str):  # or list, set, array ... only
            for v in d:
                yield from get_all_values(v)
        else:
            yield d

    return list(get_all_values(d))

def get_avgZ(dict_dataset, ref_name, data_name):
    ref_data = dict_dataset[ref_name]
    data = dict_dataset[data_name]
    Z = (data - np.mean(ref_data))/np.std(ref_data)
    return np.mean(Z)


def get_pvalue(dict_dataset, test='mann-whitney'):
    sorted_keys, sorted_vals = list(dict_dataset.keys()), list(dict_dataset.values())

    from scipy import stats
    from itertools import combinations

    for pair in combinations(range(0, len(dict_dataset)), 2):  # 2 for pairs, 3 for triplets, etc
        if test == 'mann-whitney':
            stat_test = stats.mannwhitneyu(dict_dataset[sorted_keys[pair[0]]], dict_dataset[sorted_keys[pair[1]]])
        elif test == 't-test':
            stat_test = stats.ttest_ind(dict_dataset[sorted_keys[pair[0]]], dict_dataset[sorted_keys[pair[1]]])
        elif test == 'wilcoxon-ranksum':
            stat_test = stats.ranksums(dict_dataset[sorted_keys[pair[0]]], dict_dataset[sorted_keys[pair[1]]])
        pvalue = stat_test.pvalue

    return pvalue

def adjusted_pvalues(pvalues, correction_type):
    pvalues = np.array(pvalues)
    n = float(pvalues.shape[0])
    new_pvalues = np.empty(pvalues.shape[0])
    if correction_type == "Bonferroni":
        new_pvalues = n * pvalues
        new_pvalues[new_pvalues>1] = 1

    elif correction_type == "Benjamini-Hochberg":
        values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = n - i
            pvalue, index = vals
            new_values.append((n/rank) * pvalue)
        for i in range(0, int(n)-1):
            if new_values[i] < new_values[i+1]:
                new_values[i+1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]
    return new_pvalues

def calculate_entropy(df, df_big, condition_name, cluster_type):
    group_clone = pd.DataFrame(df.groupby([condition_name, cluster_type]).size())
    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0

    ###### Find missing cluster and put 0 ######
    group_clone_T = group_clone.T
    for cluster in sorted(list(pd.unique(df_big[cluster_type]))):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=0)
            group_clone_T.sort_index(axis=1, inplace=True)
    group_clone = group_clone_T.T
    #############################################

    shannon_entropy_dict ={}
    for cond in list(group_clone.columns): # cond = 'T-cell', 'wt B-cell', 'mt B-cell'
        shannon_entropy = 0
        for i in range(0, group_clone[cond].shape[0]):
            if group_clone[cond][i] == 0:
                continue
            else:
                shannon_entropy = shannon_entropy + -group_clone[cond][i] / 100 * np.log(group_clone[cond][i] / 100)
        shannon_entropy_dict[cond] = shannon_entropy

    max_entropy = np.log(group_clone.shape[0])

    return shannon_entropy_dict, max_entropy


def shannon_entropy(probabilities):
    """
    Compute Shannon's entropy given a list of probabilities.

    Parameters:
    - probabilities (list or numpy array): A list of probabilities summing to 1.

    Returns:
    - float: Shannon entropy value.
    """
    probabilities = np.array(probabilities)

    # Compute entropy
    entropy = -np.sum( probabilities * np.log( probabilities + 1e-10) )  # Add 1e-10 to remove zero probabilities to avoid log(0)
    return entropy

def calculate_gini(x):
    ''' Calculate Gini coefficient, where 0 is perfect equality, 1 is perfect inequality
    Parameters:
    ----------
    x: np.array()
        1D array with all elements positive
    Returns:
    -------
    Gini coefficient: float
        Gini coefficient
    '''
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))


def compute_distance_matrix(X, Y):  # Compute the euclidean distance matrix

    ''' Compute euclidean distance matrix
    Parameters:
    ----------
    X: np.array()
        shape of (M = num of data points, D = num of dimensions)
    Y: np.array()
        shape of (N = num of data points, D = num of dimensions)

    Returns:
    -------
    distance_matrix: np.array()
        shape of (M, N)
    '''

    x_2 = np.sum(X ** 2, axis=1, keepdims=True)  # shape = (M, 1)

    y_2 = np.sum(Y ** 2, axis=1)
    y_2 = y_2[np.newaxis, :]  # shape = (1, N)

    xy = X @ Y.T  # shape = (M, N)

    distance_matrix = x_2 - 2 * xy + y_2
    distance_matrix[distance_matrix < 0] = 0
    distance_matrix = np.sqrt(distance_matrix)

    return distance_matrix


def compute_transition_matrix(transition:np.array, n_states:int) -> np.array:
    ''' Compute transition matrix
    Parameters:
    ----------
    transition: np.array()
        sequence of transitions
    n_states: int
        number of states

    Returns:
    -------
    trans_matrix: np.array()
        shape of (n_states, n_states), where row is current state and column is future state
    '''

    trans_matrix = np.zeros(shape=(n_states, n_states))

    for (current, future) in zip(transition, transition[1:]):
        if (np.isnan(current) == True) or (np.isnan(future) == True):
            pass
        else:
            trans_matrix[int(current)][int(future)] += 1

    #now convert to probabilities:
    # for row in M:
    #     s = sum(row)
    #     if s > 0:
    #         row[:] = [f/s for f in row]
    return trans_matrix


def order_cluster_by_feature(df:pd.DataFrame, cluster_name:str, feature_name:str) -> pd.DataFrame:
    ''' order cluster by feature
    Parameters:
    ----------
    df: pd.DataFrame
        dataframe
    cluster_name: str
        column name of cluster
    feature_name: str
        reference feature to be ordered
    Returns:
    -------
    df_replaced: pd.DataFrame
        dataframe with ordered cluster
    replace: dict
        dictionary with replace mapping information
    '''

    refs = []
    for i in np.unique(df[cluster_name]):
        subset = df[df[cluster_name] == i].reset_index(drop=True)
        ref = np.median(subset[feature_name])
        print('cluster', i, ref)
        refs.append(ref)

    refs = np.array(refs)
    sorted_refs = np.unique(df[cluster_name])[np.argsort(refs)]

    replace_map = {}
    for i, j in zip(sorted_refs, np.unique(df[cluster_name])):
        replace_map[i] = j
    df_replaced = df.replace({cluster_name: replace_map})

    return df_replaced, replace_map

def Cohen_d(group1, group2):
    ''' Cohen's d: magnitude of difference (effect size)
        - While p value only tells whether difference exists, cohen's d quantifies the magnitude of the difference.
        - Also, it is independent of sample size.
        - |d|<0.2: small difference, |d|<0.5: medium difference, |d|<0.8: large difference
    '''
    group1, group2 = np.array(group1), np.array(group2)
    diff = np.mean(group2) - np.mean(group1)
    n1, n2 = group1.size, group2.size
    s1, s2 = np.var(group1), np.var(group2)
    pooled_var = ( (n1-1)*s1 + (n2-1)*s2 ) / (n1+n2-2)
    d = diff/np.sqrt(pooled_var)
    return d

def get_various_statistics(dict_datasets:dict, test:str, return_sig=False) -> tuple:
    ''' Get various statistical tests

        one-way anova -> Parametric (normal distribution), more than 2 groups
        kruskal-wallis -> Non-parametric (no assumption), more than 2 groups

        Tukey's test -> Parametric (normal assumption), post hoc test after one-way anova, compare all pairs
        Dunn's test -> Non-parametric (no assumption), post hoc test after kruskal-wallis, compare all pairs

        Dunnett's test -> Parametric (normal distribution), post hoc test after one-way anova, compare only control with other experiment groups
        -> Applied when the experiment have 'Control'

        Parameters:
        ----------
        dict_datasets: dict
            data
        test: str
            type of statistical tests
        Returns:
        -------
        pairs: list
            list of group indexes
        p_values: list
            list of corresponding p values
        '''

    # one-way anova -> Parametric (normal distribution), more than 2 groups
    # kruskal-wallis -> Non-parametric (no assumption), more than 2 groups

    # Dunnett's test -> Parametric (normal distribution), post hoc test after one-way anova, compare only control with other experiment groups
    # -> Applied when the experiment have 'Control'
    # Tukey's test -> Parametric (normal assumption), post hoc test after one-way anova, compare all pairs,
    # Dunn's test -> Non-parametric (no assumption), post hoc test after kruskal-wallis, compare all pairs,

    sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

    from scipy import stats
    from itertools import combinations
    import scikit_posthocs
    import statsmodels.api as sm

    p_values = []
    pairs = []
    cohen_ds = []
    if test == 'kruskal-wallis_dunn':
        stat_test = stats.kruskal(*dict_datasets.values())
        if stat_test.pvalue <= 0.05:
            adj_ps = np.array(scikit_posthocs.posthoc_dunn(list(dict_datasets.values()), p_adjust='fdr_bh'))  # Benjamini/Hochberg adjustment
            for pair in combinations(range(0, len(dict_datasets)), 2):
                p_values.append(adj_ps[pair])
                pairs.append(pair)

                group1 = dict_datasets[sorted_keys[pair[0]]]
                group2 = dict_datasets[sorted_keys[pair[1]]]
                cohen_d = Cohen_d(group1, group2)
                cohen_ds.append(cohen_d)

    elif test == 'one-way anova_tukey':
        stat_test = stats.f_oneway(*dict_datasets.values())
        if stat_test.pvalue <= 0.05:
            adj_ps = np.array(scikit_posthocs.posthoc_tukey(list(dict_datasets.values())))
            for pair in combinations(range(0, len(dict_datasets)), 2):
                p_values.append(adj_ps[pair])
                pairs.append(pair)

                group1 = dict_datasets[sorted_keys[pair[0]]]
                group2 = dict_datasets[sorted_keys[pair[1]]]
                cohen_d = Cohen_d(group1, group2)
                cohen_ds.append(cohen_d)

    elif test == 'one-way anova_dunnett':
        stat_test = stats.f_oneway(*dict_datasets.values())
        if stat_test.pvalue <= 0.05:
            iter_keys = iter(dict_datasets)
            control_key = next(iter_keys)  # Treat first key as control
            rest_dict = {k: v for k, v in dict_datasets.items() if k not in control_key}
            stat_test = scipy.stats.dunnett(*rest_dict.values(), control=dict_datasets[control_key])

            for pair in range(1, len(dict_datasets)):
                pairs.append(pair)

                group1 = dict_datasets[control_key]
                group2 = dict_datasets[sorted_keys[pair]]
                cohen_d = Cohen_d(group1, group2)
                cohen_ds.append(cohen_d)

            p_values = list(stat_test.pvalue)


    else:
        for pair in combinations(range(0, len(dict_datasets)), 2):  # 2 for pairs, 3 for triplets, etc
            group1 = dict_datasets[sorted_keys[pair[0]]]
            group2 = dict_datasets[sorted_keys[pair[1]]]

            if test == 'mann-whitney':
                stat_test = stats.mannwhitneyu(group1, group2)
            elif test == 't-test':
                stat_test = stats.ttest_ind(group1, group2)
            elif test == 'wilcoxon-ranksum':
                stat_test = stats.ranksums(group1, group2)
            p_values.append(stat_test.pvalue)
            pairs.append(pair)

            cohen_d = Cohen_d(group1, group2)
            cohen_ds.append(cohen_d)
        #_, p_values, _, _ = sm.stats.multipletests(p_values, alpha=0.05,method='fdr_bh')  # Benjamini/Hochberg adjustment

    if return_sig == True:
        sig = np.array(p_values)<=0.05
        pairs = np.array(pairs)[sig]
        p_values = np.array(p_values)[sig]
        cohen_ds = np.array(cohen_ds)[sig]

    return pairs, p_values, cohen_ds


def project_on_line(start, end, target, segment=False):
    ''' Project target point onto the line segment defined by start and end points
    line segment = start + t*(end - start) where 0<=t<=1
    Parameters:
    ----------
    start: np.array()
        1D array for defining line segment
    end: np.array()
        1D array for defining line segment
    target: np.array()
        target point that will be projected
    segment: Boolean
        If True, line is only limited between ref1 and ref2, otherwise infinite
    Returns:
    -------
    proj: np.array()
        projected point coordinate
    '''
    start = np.array(start)
    end = np.array(end)

    assert (start.ndim == 1) and (end.ndim == 1), 'ref must be 1-dimensional vector'
    l2 = np.sum((start - end) ** 2)
    assert l2 != 0, 'ref1 and ref2 should not be same'

    # line segment =  p1 + t (p2 - p1).
    # The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2
    t = np.sum((target - start) * (end - start)) / l2

    # if you need the point to project on line segment between p1 and p2 or closest point of the line segment
    if segment == True:
        t = max(0, min(1, t))

    proj = start + t * (end - start)

    return proj, t


def permutation_test(df, group_name, class_name, iteration=10000)-> dict[dict[tuple]]:
    ''' Perform randomized permutation test for enrichment & depletion
        Parameters:
        ----------
        df: pd.DataFrame()
            dataframe
        group_name: str
            name of the condition to separate
        class_name: str
            name of the class that you want to calculate the fraction
        iteration: int
            Number of iterations for permutation test
        Returns:
        -------
        p_dict: dict[dict[tuple]]
            { group A: {class 1: (p_enrich, p_deplete), class 2: (p_enrich, p_deplete), ...}, group B: {...} }
        '''

    p_dict = {}
    for group in tqdm(np.unique(df[group_name])):
        p_dict_temp = {}
        for c in np.unique(df[class_name]):  # Kmeans
            total_data = df[class_name].values
            group_data = df[df[group_name] == group][class_name].values
            data_to_shuffle = total_data.copy()

            n_total = group_data.size
            n_target = group_data[group_data == c].size
            obs_statistics = n_target / n_total  # Observed statistics

            k_enrich = 0
            k_deplete = 0

            for _ in range(iteration):
                np.random.shuffle(data_to_shuffle)
                shuffled = data_to_shuffle[:n_total]
                n_shuffled = shuffled[shuffled == c].size
                simul_statistics = n_shuffled / n_total  # simulated statistics
                k_enrich += obs_statistics > simul_statistics  # Count how many times it is enrichment
                k_deplete += obs_statistics < simul_statistics  # Count how many times it is depletion

            p_enrich = 1 - k_enrich / iteration
            p_deplete = 1 - k_deplete / iteration

            p_dict_temp[c] = (p_enrich, p_deplete)
            if p_enrich < 0.05:
                print('Group %s, cluster %s is enriched'%(group, c))
            if p_deplete < 0.05:
                print('Group %s, cluster %s is depleted'%(group, c))

        p_dict[group] = p_dict_temp
    return p_dict

def extract_n_substrings(s:str, n:int, symbol='_') -> list:
    """
    Extracts the first `n` substrings that are between symbol like '_'.

    Parameters:
    - s (str): Input string.
    - n (int): Number of substrings to extract.
    - symbol (str): String used to split substrings.

    Returns:
    - List of extracted substrings.
    """
    parts = s.split(symbol)  # Split by underscores
    return parts[:n]  # Return the first `n` parts

def get_filtered_string_list(arr, keywords, filter_type='or') -> np.array:
    """
    Extracts the first `n` substrings that are between symbol like '_'.

    Parameters:
    - arr (np.array): Input array that contain strings
    - keywords (list of strings): Keywords that should be included in the filtered strings
    - filter_type (str): Applying 'or' operation or 'and' operation.

    Returns:
    - List of filtered strings that contain keywords.
    """
    keywords = np.array(keywords)

    if filter_type == 'or':
        filtered_arr = np.array([s for s in arr if any(term in s for term in keywords)])
    elif filter_type == 'and':
        filtered_arr = np.array([s for s in arr if all(term in s for term in keywords)])
    return filtered_arr


def outlier_detection_by_discontinuity(df, feature):
    df_sorted = df.sort_values(by='%s' % feature).reset_index()
    df_sorted['%s_diff' % feature] = df_sorted['%s' % feature].diff() # Calculate differences between consecutive feature values
    max_diff_index = df_sorted['%s_diff' % feature].idxmax()  # Identify the index with the maximum gap

    left_idxs = df_sorted.loc[:max_diff_index, 'index'].tolist()
    right_idxs = df_sorted.loc[max_diff_index:, 'index'].tolist()

    if len(left_idxs) >= len(right_idxs):
        outlier_idxs = right_idxs
    else:
        outlier_idxs = left_idxs

    df.loc[outlier_idxs, 'outlier'] = 'outlier'
    df['outlier'] = df['outlier'].fillna('data')

    df_removed = df[df['outlier'] == 'data'].reset_index(drop=True)
    df_removed = df_removed.drop(columns=['outlier'])

    return df, df_removed

def project_feature_onto_FDC(df, df_duration, feature, duration, img_shape, offsets, um_per_zslice, um_per_pixel, count_norm=True):
    n_trajs = df_duration.shape[0] // duration
    z, r, w = img_shape
    feature_map = np.zeros(img_shape, dtype=np.float32)
    count = np.zeros(img_shape, dtype=np.int32)

    for traj_idx in tqdm(range(n_trajs)):
        # Extract the 20-frame segment for the current cell
        traj = df_duration.iloc[traj_idx * duration: (traj_idx + 1) * duration]

        # Convert positions from micrometers to pixels
        positions_z = (traj['Position Z'].values / um_per_zslice).round().astype(int)
        positions_y = (traj['Position Y'].values / um_per_pixel).round().astype(int)
        positions_x = (traj['Position X'].values / um_per_pixel).round().astype(int)

        # Stack positions into a single array of shape (duration, 3)
        positions = np.stack((positions_z, positions_y, positions_x), axis=1)

        # Apply offsets to get neighboring voxel indices
        neighbors = positions[:, np.newaxis, :] + offsets[np.newaxis, :, :]  # Shape: (duration, num_offsets, 3)
        neighbors = neighbors.reshape(-1, 3)  # Flatten to (duration * num_offsets, 3)

        # Filter out-of-bounds indices
        valid_mask = (
                (neighbors[:, 0] >= 0) & (neighbors[:, 0] < z) &
                (neighbors[:, 1] >= 0) & (neighbors[:, 1] < r) &
                (neighbors[:, 2] >= 0) & (neighbors[:, 2] < w)
        )
        valid_neighbors = neighbors[valid_mask]

        # Retrieve the feature value for this trajectory
        feature_value = df.iloc[traj_idx][feature]

        # Convert 3D indices to linear indices
        linear_indices = np.ravel_multi_index(
            (valid_neighbors[:, 0], valid_neighbors[:, 1], valid_neighbors[:, 2]),
            dims=(z, r, w)
        )

        # Accumulate feature values and counts
        np.add.at(feature_map.ravel(), linear_indices, feature_value)
        np.add.at(count.ravel(), linear_indices, 1)

    if count_norm == True:
        nonzero_mask = count > 0
        feature_map[nonzero_mask] /= count[nonzero_mask]
    else:
        nonzero_mask = count == 1
        feature_map = np.where(nonzero_mask, feature_map, 0)
    return feature_map

def project_feature_onto_FDC_by_majority_vote(df, df_duration, feature, duration, img_shape, offsets, um_per_zslice, um_per_pixel):
    from collections import defaultdict, Counter
    n_trajs = df_duration.shape[0] // duration
    z, r, w = img_shape
    feature_map = np.zeros(img_shape, dtype=np.float32)
    count = np.zeros(img_shape, dtype=np.int32)
    label_votes = defaultdict(Counter)  # voxel_linear_index -> Counter({label: hits})

    for traj_idx in tqdm(range(n_trajs)):
        # Extract the 20-frame segment for the current cell
        traj = df_duration.iloc[traj_idx * duration: (traj_idx + 1) * duration]

        # Convert positions from micrometers to pixels
        positions_z = (traj['Position Z'].values / um_per_zslice).round().astype(int)
        positions_y = (traj['Position Y'].values / um_per_pixel).round().astype(int)
        positions_x = (traj['Position X'].values / um_per_pixel).round().astype(int)

        # Stack positions into a single array of shape (duration, 3)
        positions = np.stack((positions_z, positions_y, positions_x), axis=1)

        # Apply offsets to get neighboring voxel indices
        neighbors = positions[:, np.newaxis, :] + offsets[np.newaxis, :, :]  # Shape: (duration, num_offsets, 3)
        neighbors = neighbors.reshape(-1, 3)  # Flatten to (duration * num_offsets, 3)

        # Filter out-of-bounds indices
        valid_mask = (
                (neighbors[:, 0] >= 0) & (neighbors[:, 0] < z) &
                (neighbors[:, 1] >= 0) & (neighbors[:, 1] < r) &
                (neighbors[:, 2] >= 0) & (neighbors[:, 2] < w)
        )
        valid_neighbors = neighbors[valid_mask]

        # Retrieve the feature value for this trajectory
        feature_value = df.iloc[traj_idx][feature]

        # Convert 3D indices to linear indices
        linear_indices = np.ravel_multi_index(
            (valid_neighbors[:, 0], valid_neighbors[:, 1], valid_neighbors[:, 2]),
            dims=(z, r, w)
        )

        u, c = np.unique(linear_indices, return_counts=True)
        # accumulate votes for this label at all voxels hit by this trajectory
        for idx, hits in zip(u, c):
            label_votes[int(idx)][int(feature_value)] += int(hits)

    fm = feature_map.ravel()
    for lin_idx, ctr in label_votes.items():
        mode_label, _ = ctr.most_common(1)[0]
        fm[lin_idx] = float(mode_label)
    feature_map = feature_map  # already filled

    return feature_map
