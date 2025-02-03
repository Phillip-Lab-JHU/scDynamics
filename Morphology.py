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
"""Morphodynamic gc_analysis"""

from umap import UMAP
from utils.traj_utils import *

import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import ListedColormap


class Morphodynamics(object):
    from factor_analyzer import FactorAnalyzer
    __fa = FactorAnalyzer(n_factors=2, rotation=None, method='principal')
    from sklearn.decomposition import PCA
    __pca = PCA(n_components=2)

    def __init__(self, df, method='umap'):
        """
        Parameters:
        ----------
        df: pd.DataFrame
            dataframe with all parameters including labels
        method: str
            type of dimension reduction
            'pfa' for principal factor gc_analysis, 'pca' for principal component gc_analysis, 'umap' for UMAP
        """
        if 'PC1' in df.columns and 'PC2' in df.columns:
            self.xmin = math.floor(df['PC1'].min()) - 1
            self.xmax = math.ceil(df['PC1'].max()) + 1
            self.ymin = math.floor(df['PC2'].min()) - 1
            self.ymax = math.ceil(df['PC2'].max()) + 1

        self.method = method

        self.min_dist = None
        self.n_neighbors = None
        self.n_clusters = None
        self.time_series = None

    def evaluate_pfa(self, df_input):
        df_title = pd.DataFrame(df_input.T.index)
        df_title.columns = ['parameter']

        Morphodynamics.__fa.fit(df_input)  # fit to calculate corr_, loadings_

        ################# Calculate KMO  #################
        from factor_analyzer import calculate_kmo  # Appropriateness of the input data
        kmo_all, kmo_model = calculate_kmo(df_input)
        # (All kmo(Appropriateness of each features), model of kmo(Appropriateness as a whole model)) -> model of kmo should be > 0.6
        print('\n******KMO test*******\n')
        print('kmo: (must be above 0.6)', kmo_model)

        ################# Calculate chi-square and p-value  #################
        from factor_analyzer import calculate_bartlett_sphericity  # how different with identity matrix
        chi_square, p_value = calculate_bartlett_sphericity(df_input)  # (chi-sqaure , p) -> p should be less than 0.05
        print('\n******Chi-square test*******\n')
        print('chi-sqaure: ', chi_square, '\np_value: (must be less than 0.05)', p_value)

        ################# Calculate communalities of each features  #################
        df_communalities = pd.DataFrame(Morphodynamics.__fa.get_communalities(), columns=['communalities'])
        communality = pd.concat([df_title, df_communalities], axis=1)
        self.communality = communality
        print('\n******Communality test*******\n')
        print(communality)

        ################# Calculate correlation coefficients  #################
        correlation = pd.DataFrame(Morphodynamics.__fa.corr_, columns=df_title['parameter'], index=df_title['parameter'])
        self.correlation = correlation
        plt.figure(figsize=(20, 15))
        heatmap = sns.heatmap(correlation, annot=False,  yticklabels=True, xticklabels=True,
                              # yticklabels = ['clone 1-1','clone 1-2','clone 1-3','clone 3-3'],
                              cmap='RdBu_r'
                              )
        plt.title('Correlation Matrix')
        plt.show()
        print('\n******Correlation test*******\n')
        print(correlation)

        ################# Calculate variance of each PCs  #################
        eigenvalues = pd.DataFrame(Morphodynamics.__fa.get_eigenvalues(), columns=range(1, df_input.shape[1] + 1),
                                   index=['eigen value', 'Not used'])
        eigenvalue = eigenvalues[range(1, df_input.shape[1] + 1)][0:1].T
        variance = pd.DataFrame(Morphodynamics.__fa.get_factor_variance(),
                                index=['eigen value', 'variance', 'cumulative variance'],
                                columns=['PC1', 'PC2'])
        self.variance = variance
        print('\n******Variance test*******\n')
        print(variance)
        plt.figure()
        plt.plot(eigenvalue.index, eigenvalue['eigen value'], marker='o')
        plt.title('Scree plot', fontsize=20)
        plt.xlabel('Principal Component', fontsize=15)
        plt.ylabel('Variance', fontsize=15)
        plt.show()

        return communality, correlation, variance

    def get_pfa(self, df_input):
        df_title = pd.DataFrame(df_input.T.index)
        df_title.columns = ['parameter']

        loadings = pd.DataFrame(Morphodynamics.__fa.loadings_, columns=['PC1', 'PC2'])
        loadings = pd.concat([df_title, loadings], axis=1)
        self.loadings = loadings

        pcs_array = Morphodynamics.__fa.fit_transform(df_input)  # factor scores for non-rotated data
        df_pcs = pd.DataFrame(pcs_array, columns=['PC1', 'PC2'])
        self.pcs = df_pcs

        self.xmin = math.floor(df_pcs['PC1'].min()) - 1
        self.xmax = math.ceil(df_pcs['PC1'].max()) + 1
        self.ymin = math.floor(df_pcs['PC2'].min()) - 1
        self.ymax = math.ceil(df_pcs['PC2'].max()) + 1

        #pcs = pd.concat([df_pcs, df_raw], axis=1)

        return df_pcs

    def evaluate_pca(self, df_input):
        Morphodynamics.__pca.fit(df_input)
        correlation = df_input.corr()
        self.correlation = correlation
        variance = pd.DataFrame(np.array(
            [Morphodynamics.__pca.explained_variance_, Morphodynamics.__pca.explained_variance_ratio_,
             np.cumsum(Morphodynamics.__pca.explained_variance_ratio_)]),
            index=['eigen value', 'variance', 'cumulative variance'], columns=['PC1', 'PC2'])
        self.variance = variance

        heatmap = sns.heatmap(correlation, annot=False,
                              # yticklabels = ['clone 1-1','clone 1-2','clone 1-3','clone 3-3'],
                              cmap='RdBu_r'
                              )


        plt.title('Correlation Matrix')
        plt.show()
        print('\n******Correlation test*******\n')
        print(correlation)

        print('\n******Variance test*******\n')
        print(variance)

        return correlation, variance

    def get_pca(self, df_input):
        df_title = pd.DataFrame(df_input.T.index)
        df_title.columns = ['parameter']

        pcs_array = Morphodynamics.__pca.fit_transform(df_input)  # factor scores for non-rotated data
        df_pcs = pd.DataFrame(pcs_array, columns=['PC1', 'PC2'])
        self.pcs = df_pcs

        loadings = pd.DataFrame(Morphodynamics.__pca.components_.T, columns=['PC1', 'PC2'])
        loadings = pd.concat([df_title, loadings], axis=1)
        self.loadings = loadings

        self.xmin = math.floor(df_pcs['PC1'].min()) - 1
        self.xmax = math.ceil(df_pcs['PC1'].max()) + 1
        self.ymin = math.floor(df_pcs['PC2'].min()) - 1
        self.ymax = math.ceil(df_pcs['PC2'].max()) + 1

        #pcs = pd.concat([df_pcs, self.df_raw], axis=1)

        return df_pcs

    def evaluate_umap(self, df_raw, df_input, directory, n_neighbors_list=[10, 20, 30, 40, 50, 60],
                      min_dist_list=[0.005, 0.01, 0.05, 0.1, 0.3, 0.5], condition_name ='Type'):
        colors = ('red', 'green', 'blue', 'gray', 'cyan', 'violet', 'coral', 'brown', 'powderblue', 'olive', 'hotpink',
                  'indigo', 'bisque', 'lawngreen', 'darksalmon', 'wheat', 'steelblue')
        cmap = ListedColormap(colors[:pd.unique(df_raw[condition_name]).shape[0]])
        # cmap = plt.cm.get_cmap('Set3')
        # n_neighbors_list = [10, 20, 30, 40, 50, 60]  # Range: 5~50
        # min_dist_list = [0.005, 0.01, 0.05, 0.1, 0.3, 0.5]  # Range: 0.001 to 0.5
        i = 1
        for n_neighbors_value in tqdm(n_neighbors_list):
            for min_dist_value in min_dist_list:
                umap = UMAP(metric='euclidean', n_components=2, n_neighbors=n_neighbors_value, min_dist=min_dist_value,
                            random_state=0)
                umap_data = umap.fit_transform(df_input)

                plt.figure(figsize=(2, 2))
                # plt.subplot(len(n_neighbors_list), len(min_dist_list), i)
                scatter = plt.scatter(umap_data[:, 0], umap_data[:, 1],
                                      c=df_raw[condition_name].replace(list(pd.unique(df_raw[condition_name])),
                                                                   [j for j in range(pd.unique(df_raw[condition_name]).shape[0])]),
                                      s=0.07,
                                      # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
                                      label=df_raw[condition_name], cmap=cmap)
                plt.title('n_neighbors:' + str(n_neighbors_value) + ' ' + 'mid_dist' + str(min_dist_value), fontsize=4)
                plt.legend(handles=scatter.legend_elements()[0], labels=list(pd.unique(df_raw[condition_name])),
                           bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0, fontsize=3, frameon=False, markerscale=0.3)

                if not os.path.isdir(directory + 'UMAP hyperparameters/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                    os.makedirs(directory + 'UMAP hyperparameters/')
                plt.savefig(directory + 'UMAP hyperparameters/%s.png' % i, dpi=300, bbox_inches='tight')
                plt.clf()
                plt.close()
                i = i + 1

    def get_umap(self, df_input, n_neighbors, min_dist):

        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

        __umap = UMAP(metric='euclidean', n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
        pcs_array = __umap.fit_transform(df_input)
        df_pcs = pd.DataFrame(pcs_array, columns=['PC1', 'PC2'])

        self.xmin = math.floor(df_pcs['PC1'].min()) - 1
        self.xmax = math.ceil(df_pcs['PC1'].max()) + 1
        self.ymin = math.floor(df_pcs['PC2'].min()) - 1
        self.ymax = math.ceil(df_pcs['PC2'].max()) + 1

        return df_pcs

    def evaluate_cluster(self, df_input, directory: str, cluster_type='kmeans', k_max=50):
        ''' evaluate number of clusters:
            ----------
            directory: str
                directory to save files
            cluster_type: str
                type of clustering algorithm
                e.g.: 'kmeans', 'gmm', 'tskmeans'
            k_max: int
                maximum number of clusters to be evaluated
            condition_name: str, only for tskmeans
                column name of condition
            label_name: str, only for tskmeans
                column name of label
            feature_name: list of str, only for tskmeans
                column name of features
            Returns:
            -------
            optimal number of clusters: int
                recommended number of clusters calculated by inertia or silhouette
            '''
        k_range = range(2, k_max)
        if cluster_type == 'kmeans':
            from sklearn.cluster import KMeans
            ##################### Deciding Number of clusters by sum of squared errors ###########################
            # find point of maximum curvature
            inertia = []
            for k in tqdm(k_range):
                km = KMeans(n_clusters=k)
                km.fit(df_input)
                inertia.append(km.inertia_)  # inertia_ 자체가 sum of squared error 계산식을 포함

            plt.figure()
            plt.xlabel('K')
            plt.xticks(k_range, fontsize=7)
            plt.ylabel('Inertia (Sum of squared error)')
            plt.plot(k_range, inertia)

            plt.savefig(directory + 'kmeans_inertia.png')

            from kneed import KneeLocator  # conda install -c conda-forge kneed
            kl = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
            print('number of cluster by inertia: ', kl.elbow)  # find point of maximum curvature

            ##################### Deciding Number of clusters by silhouette score ###########################

            from sklearn.metrics import silhouette_score
            silhouette_coefficients = []
            for k in tqdm(k_range):
                km = KMeans(n_clusters=k)
                km.fit(df_input)
                score = silhouette_score(df_input, km.labels_)
                silhouette_coefficients.append(score)

            plt.figure()
            plt.xlabel('K')
            plt.xticks(k_range, fontsize=7)
            plt.ylabel('Silhouette Coefficient')
            plt.plot(k_range, silhouette_coefficients)
            # choose the maximum value
            plt.savefig(directory + 'kmeans_silhouette.png')

            print('number of cluster by silhouette coefficient: ', silhouette_coefficients.index(max(silhouette_coefficients)) + 2)

            ##################### Deciding Number of clusters by inertia / silhouette score ###########################
            inertia_over_silhouette = [a/b for a, b in zip(inertia, silhouette_coefficients)]

            plt.figure()
            plt.xlabel('K')
            plt.xticks(k_range, fontsize=7)
            plt.ylabel('Inertia over silhouette')
            plt.plot(k_range, inertia_over_silhouette)

            plt.savefig(directory + 'kmeans_inertia_over_silhouette.png')

            from kneed import KneeLocator  # conda install -c conda-forge kneed
            kl = KneeLocator(k_range, inertia_over_silhouette, curve='convex', direction='decreasing')
            print('number of cluster by inertia/silhouette: ', kl.elbow)  # find point of maximum curvature

        ##################### Deciding Number of clusters by bic ###########################
        elif cluster_type == 'gmm':
            from sklearn.mixture import GaussianMixture
            gmm_bic = []
            for k in tqdm(k_range):
                GMM = GaussianMixture(n_components=k, covariance_type='tied')
                GMM.fit(df_input)
                gmm_bic.append(GMM.bic(df_input))

            plt.figure()
            plt.xlabel('K')
            plt.xticks(k_range, fontsize=7)
            plt.ylabel('BIC')
            plt.plot(k_range, gmm_bic)
            plt.savefig(directory + 'gmm_bic.png')

            from kneed import KneeLocator  # conda install -c conda-forge kneed
            kl = KneeLocator(k_range, gmm_bic, curve='convex', direction='decreasing')
            print('number of cluster by Bayesian Information Criterion(bic): ', kl.elbow)  # find point of maximum curvature

            ##################### Deciding Number of clusters by aic ###########################
            gmm_aic = []
            for k in tqdm(k_range):
                GMM = GaussianMixture(n_components=k, covariance_type='tied')
                GMM.fit(df_input)
                gmm_aic.append(GMM.aic(df_input))  # inertia_ 자체가 sum of squared error 계산식을 포함

            plt.figure()
            plt.xlabel('K')
            plt.xticks(np.arange(1, 15), fontsize=7)
            plt.ylabel('BIC')
            plt.plot(k_range, gmm_aic)

            plt.savefig(directory + 'gmm_aic.png')

            from kneed import KneeLocator  # conda install -c conda-forge kneed
            kl = KneeLocator(k_range, gmm_aic, curve='convex', direction='decreasing')
            print('number of cluster by Akaike Information Criterion(aic): ', kl.elbow)  # find point of maximum curvature


    def evaluate_ts_cluster(self, df_input, directory: str, k_max=50, duration=20, feature_name = ['PC1', 'PC2']):
        k_range = range(2, k_max)
        import tslearn
        from tslearn.clustering import TimeSeriesKMeans
        traj_list, time_series, _ = to_timeseries_fast(df_input, duration=duration, feature_name=feature_name)
        silhouette_coefficients = []
        for k in tqdm(k_range):
            tskm = TimeSeriesKMeans(n_clusters=k, metric='softdtw', random_state=0, verbose=True, max_iter=50)
            tskm.fit(time_series)
            score = tslearn.clustering.silhouette_score(time_series, tskm.labels_)
            silhouette_coefficients.append(score)

        plt.figure()
        plt.xlabel('K')
        plt.xticks(k_range, fontsize=7)
        plt.ylabel('Silhouette Coefficient')
        plt.plot(k_range, silhouette_coefficients)

        plt.savefig(directory + 'tskmeans_silhouette.png')

        print('number of cluster by silhouette coefficient: ',
              silhouette_coefficients.index(max(silhouette_coefficients)) + 2)


    def get_cluster(self, df_input, n_clusters, cluster_type='kmeans'):
        df = pd.DataFrame()
        if cluster_type == 'kmeans':
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++')
            # k-means++: Initialize centroids that are far away each other
            kmeans_predicted = km.fit_predict(df_input)

            df['kmeans'] = kmeans_predicted
            self.n_clusters = n_clusters

        elif cluster_type == 'gmm':
            from sklearn.mixture import GaussianMixture
            GMM = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=0)
            GMM_predicted = GMM.fit_predict(df_input)

            df['gmm'] = GMM_predicted
            self.n_clusters = n_clusters

        elif cluster_type == 'affinity_propagation':
            # Bottom-Up clustering
            from sklearn.cluster import AffinityPropagation
            af = AffinityPropagation(preference=None, random_state=0, affinity='euclidean', verbose=True)
            # High preference (positive / less negative) -> More points become 'exemplar' (centroid in kmeans) -> More clusters
            # Low preference (more negative) -> Less points become 'exemplar' (centroid in kmeans) -> Less clusters
            # Preference = None use median of the input similarities
            # Preference = -50 -> 3000 clusters, preference = None -> 775 clusters
            af_predicted = af.fit_predict(df_input)

            df['af'] = af_predicted
            self.n_clusters = len(af.cluster_centers_indices_)
            print('estimated number of clusters: %s' %self.n_clusters)

        return df

    def get_ts_cluster(self, df, n_clusters, duration=20, metric='softdtw', normalize=False, feature_name=['PC1', 'PC2']):
        from tslearn.clustering import TimeSeriesKMeans
        tskm = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=0, verbose=True, max_iter=100)
        # if dtw: strange center trajectory
        traj_list, time_series, time_series_dict = to_timeseries_fast(df, duration=duration, feature_name=feature_name)
        if normalize ==True:
            time_series_dict = normalize_timeseries(time_series_dict)
            time_series = dict_to_array(time_series_dict)
        tskmeans_predicted = tskm.fit_predict(time_series)
        labels = tskmeans_predicted # row = traj

        df_temp = pd.DataFrame()
        for traj_idx, traj_df in enumerate(traj_list):  # traj_list = list of volume, label, area, ... dataframe for each cell trajectory
            traj_df['tskmeans'] = tskmeans_predicted[traj_idx]
            df_temp = pd.concat([df_temp, traj_df])

        labels_expanded = df_temp['tskmeans'].values # row = cell state at time t

        self.tskm_cluster_center = tskm.cluster_centers_

        return labels, labels_expanded, self.tskm_cluster_center




