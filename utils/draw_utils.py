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
"""Functions for drawing publication quality figures"""
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from matplotlib.colors import ListedColormap
import matplotlib
from statannot import add_stat_annotation
from utils.misc_utils import *
from utils.traj_utils import *
from math import ceil, sqrt
import networkx as nx
from itertools import combinations
import community as community_louvain  # Louvain algorithm
import random
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from adjustText import adjust_text
import cmcrameri.cm as cmc
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from collections.abc import Iterable

color_list = ('#F06293', '#AF79C4', '#5B6BBF', '#FFC11C', '#D1917C', '#4DB6AC', '#00BCD4', '#A1877E', '#BCBCBC', '#8F96A8', '#B02E8B', '#EB974D')
def format_figure(ax,title=None,xlabel=None,ylabel=None,despine=True,detick=False):
    # Nikita's function to format figures visually appealing
    if title != None:
        ax.set_title(title, pad=5)
    if xlabel != None:
        ax.set_xlabel(xlabel, labelpad=5)
    if ylabel != None:
        ax.set_ylabel(ylabel, labelpad=5)
    if despine:
        sns.despine()
    if detick:
        plt.tick_params(left=False, right=False, labelleft=False,
            labelbottom=False, bottom=False)

def add_p_value_annotation(fig, array_columns, test, subplot=None,
                           _format=dict(interline=0.03, text_height=1.045, color='black')):
    ''' Adds notations giving the p-value between two box plot data (t-test two-sided comparison)
    Parameters:
    ----------
    fig: figure
        plotly boxplot figure
    array_columns: np.array
        array of which columns to compare
        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
    test: str
        'Mann-Whitney', 't-test_ind', 't-test_welch', 'kruskal'
    subplot: None or int
        specifies if the figures has subplots and what subplot to add the notation to
    _format: dict
        format characteristics for the lines
    Returns:
    -------
    fig: figure
        figure with the added notation
    '''
    from scipy import stats
    # Specify in what y_range to plot for each pair of columns
    y_range = np.zeros([len(array_columns), 2])
    for i in range(len(array_columns)):
        y_range[i] = [1.01 + i * _format['interline'], 1.02 + i * _format['interline']]
    # Get values from figure
    fig_dict = fig.to_dict()
    # Get indices if working with subplots
    if subplot:
        if subplot == 1:
            subplot_str = ''
        else:
            subplot_str = str(subplot)
        indices = []  # Change the box index to the indices of the data for that subplot
        for index, data in enumerate(fig_dict['data']):
            # print(index, data['xaxis'], 'x' + subplot_str)
            if data['xaxis'] == 'x' + subplot_str:
                indices = np.append(indices, index)
        indices = [int(i) for i in indices]
        print((indices))
    else:
        subplot_str = ''
    # Print the p-values
    for index, column_pair in enumerate(array_columns):
        if subplot:
            data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
        else:
            data_pair = column_pair
        # Mare sure it is selecting the data and subplot you want
        # print('0:', fig_dict['data'][data_pair[0]]['name'], fig_dict['data'][data_pair[0]]['xaxis'])
        # print('1:', fig_dict['data'][data_pair[1]]['name'], fig_dict['data'][data_pair[1]]['xaxis'])

        # Get the p-value
        if test == 't-test_welch':
            pvalue = stats.ttest_ind(
                fig_dict['data'][data_pair[0]]['y'],
                fig_dict['data'][data_pair[1]]['y'],
                equal_var=False, # Welch’s t-test (equal_var = True: standard t-test, assuming equal population variance)
                alternative='two-sided',
            )[1]

        elif test == 't-test_ind':
            pvalue = stats.ttest_ind(
                fig_dict['data'][data_pair[0]]['y'],
                fig_dict['data'][data_pair[1]]['y'],
                equal_var=True, # (equal_var = True: standard t-test, assuming equal population variance)
                alternative='two-sided',
            )[1]

        elif test == 'Mann-Whitney':
            pvalue = stats.mannwhitneyu(
                fig_dict['data'][data_pair[0]]['y'],
                fig_dict['data'][data_pair[1]]['y'],
                alternative='two-sided'
            )[1]

        elif test == 'kruskal':
            pvalue = stats.kruskal(
                fig_dict['data'][data_pair[0]]['y'],
                fig_dict['data'][data_pair[1]]['y'],
            )[1]

        if pvalue >= 0.05:
            symbol = 'ns'
        elif pvalue >= 0.01:
            symbol = '*'
        elif pvalue >= 0.001:
            symbol = '**'
        else:
            symbol = '***'
        # Vertical line
        fig.add_shape(type="line",
                      xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=column_pair[0], y0=y_range[index][0],
                      x1=column_pair[0], y1=y_range[index][1],
                      line=dict(color=_format['color'], width=2, )
                      )
        # Horizontal line
        fig.add_shape(type="line",
                      xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=column_pair[0], y0=y_range[index][1],
                      x1=column_pair[1], y1=y_range[index][1],
                      line=dict(color=_format['color'], width=2, )
                      )
        # Vertical line
        fig.add_shape(type="line",
                      xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=column_pair[1], y0=y_range[index][0],
                      x1=column_pair[1], y1=y_range[index][1],
                      line=dict(color=_format['color'], width=2, )
                      )
        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        fig.add_annotation(dict(font=dict(color=_format['color'], size=14),
                                x=(column_pair[0] + column_pair[1]) / 2,
                                y=y_range[index][1] * _format['text_height'],
                                showarrow=False,
                                text=symbol,
                                textangle=0,
                                xref="x" + subplot_str,
                                yref="y" + subplot_str + " domain"
                                ))
    return fig

def draw_pca_space(df, path, file_name, condition_name, x_name, y_name, xmin, xmax, ymin, ymax, variance, loadings):
    colors = ('red', 'green', 'blue', 'gray', 'cyan', 'violet', 'coral', 'brown', 'powderblue', 'olive',
              'hotpink', 'indigo', 'bisque', 'lawngreen', 'darksalmon', 'wheat', 'steelblue', 'black', 'lime', 'gold',
              'orange', 'cornsilk', 'ivory', 'aliceblue', 'peru', 'mistyrose', 'chocolate', 'slategrey',
              'cornflowerblue', 'silver')
    cmap = ListedColormap(colors[:pd.unique(df[condition_name]).shape[0]])
    # cmap = plt.cm.get_cmap('Set1')

    ################## Draw interactive version of space #######################
    fig = px.scatter(
        data_frame=df,
        x=x_name,
        y=y_name,
        color=condition_name,
        # color_discrete_sequence = ['red','green','blue','yellow'], # label이 숫자나 bool 형태이면 color 적용이 안되는 버그가 있음
        opacity=0.9,
        template='plotly_white',
        # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
        # symbol = 'TrackID',
        # symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
        # title='space',
        labels={x_name: 'PC1(' + str(round(variance['PC1'][1] * 100, ndigits=1)) + '%)',
                y_name: 'PC2(' + str(round(variance['PC2'][1] * 100, ndigits=1)) + '%)',
                },
        hover_data={'pseudo_Label': True, 'Time': True, 'Exp': True},
        hover_name=df.index,

        range_x=[xmin, xmax],
        range_y=[ymin, ymax],

        height=1000,
        width=2000,
    )

    fig.update_traces(marker=dict(size=3),
                      # line = dict(width=1, color='DarkSlateGrey')) ,
                      # selector=dict(mode='markers')
                      )

    for index, value in loadings.iterrows():
        fig.add_shape(type='line', x0=0, y0=0, x1=3 * value[x_name], y1=3 * value[y_name], opacity=0.7,
                      line=dict(color='black', width=1, dash='dot'))
        fig.add_annotation(x=3 * value[x_name], y=3 * value[y_name], ax=0, ay=0, xanchor='center',
                           yanchor='bottom',
                           text=value['parameter'], font=dict(size=8, color='black'), opacity=0.7)

    fig.write_html(path + '%s.html' % file_name)

    ################## Draw figure version of state space #######################
    plt.figure(figsize=(15, 10))
    # plt.subplot(len(n_neighbors_list), len(min_dist_list), i)
    scatter = plt.scatter(df[x_name], df[y_name],
                          c=df[condition_name].replace(list(pd.unique(df[condition_name])),
                                                       [i for i in range(pd.unique(df[condition_name]).shape[0])]),
                          # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
                          s=5, label=df[condition_name],
                          cmap=cmap)
    plt.title('State space', fontsize=20)
    plt.xlabel('PC1(' + str(round(variance['PC1'][1] * 100, ndigits=1)) + '%)', fontsize=15)
    plt.ylabel('PC2(' + str(round(variance['PC2'][1] * 100, ndigits=1)) + '%)', fontsize=15)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    handles, labels = scatter.legend_elements(num=None)
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1),
               loc=2, borderaxespad=0.0, fontsize=10, frameon=False, markerscale=2)


    plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

def draw_umap_space(
    df, path, file_name, condition_name, colors, dot_size, x_name, y_name,
    umap_size=(2.4, 2.4),      # fixed size of UMAP panel (inches)
    legend_width=1.6,          # width of legend panel (inches)
    legend_fontsize=5
):
    # colormap
    cats = pd.Categorical(df[condition_name])
    n_cat = len(cats.categories)
    if isinstance(colors, Iterable) and not isinstance(colors, str):
        cmap = ListedColormap(list(colors)[:n_cat])
    else:
        cmap = colors

    # fixed axis ranges
    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    # style
    font = {"family": "arial", "weight": "normal", "size": 8}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["axes.linewidth"] = 0.25
    matplotlib.rcParams["lines.linewidth"] = 1

    # figure with 2 panels: left UMAP (fixed), right legend
    fig = plt.figure(figsize=(umap_size[0] + legend_width, umap_size[1]))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[umap_size[0], legend_width], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    scatter = ax.scatter(
        df[x_name], df[y_name],
        c=cats.codes,
        s=dot_size,
        edgecolor="none",
        cmap=cmap
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    format_figure(ax, title=None, xlabel="UMAP1", ylabel="UMAP2", despine=True, detick=True)

    handles, _ = scatter.legend_elements(num=None)
    labels = [str(x) for x in cats.categories]
    ax_leg.legend(
        handles, labels,
        loc="upper left",
        fontsize=legend_fontsize,
        frameon=False,
        markerscale=0.8,
        borderaxespad=0.0
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)

    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f"{file_name}.png"), dpi=300, bbox_inches="tight")
    os.makedirs(os.path.join(path, "svg"), exist_ok=True)
    plt.savefig(os.path.join(path, "svg", f"{file_name}.svg"), bbox_inches="tight")
    plt.clf()
    plt.close()

def draw_scatter_plot(df, path, file_name, figsize, condition_name, colors, dot_size, x_name, y_name, texts=None, invert_y=False):
    ################## Draw interactive version of state space #######################

    #colors = ('#CC6677', '#6699CC', '#44AA99', '#DDCC77', '#88CCEE', '#117733', '#332288', '#AA4499', '#999933', '#882255', '#661100', '#888888')
    from collections.abc import Iterable
    if isinstance(colors, Iterable):
        cmap = ListedColormap(colors[:np.unique(df[condition_name]).shape[0]])
    else:
        cmap=colors
    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    ################## Draw figure version of state space #######################

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    fig, ax = plt.subplots(figsize=figsize)
    #plt.figure(figsize=(15, 10))
    scatter = ax.scatter(df[x_name], df[y_name],
                          c=df[condition_name].replace(list(np.unique(df[condition_name])),
                            [i for i in range(np.unique(df[condition_name]).shape[0])]),
                          # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
                          s=dot_size, label=df[condition_name],
                          cmap=cmap)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)


    if np.all(texts) != None:
        for idx, text in enumerate(texts):
            plt.text(x=text[0], y=text[1], s=idx, fontsize=8, weight='normal', ha='center', va='center', color='0.2')
    else:
        handles, labels = scatter.legend_elements(num=None)
        plt.legend(handles=handles, labels=list(np.unique(df[condition_name])),
                   bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
                   fontsize=3, frameon=False, markerscale=0.3)

    if invert_y == True:
        ax.invert_yaxis()

    # bbox_to_anchor is position of labels (x, y) (increasing x moves right, increasing y moves top)
    # frameon=False removes bounding box around label
    # font size adjust size of letter
    # markerscale adjust size of marker

    plt.savefig(path + '%s.png' % file_name, dpi=300)

    # if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    #     os.makedirs(path + 'svg/')
    # plt.savefig(path + 'svg/%s.svg' % file_name)
    # plt.clf()
    # plt.close()

def draw_contour(df, path, file_name, condition_name, colors, x_name='PC1', y_name='PC2', bin_num=50, num_contours=6):
    # color_list = ['Reds', 'Greens', 'Blues', 'Greys', 'Oranges', 'Purples', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
    #               'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'plasma']

    n_colors = np.unique(df[condition_name]).shape[0]
    from collections.abc import Iterable
    if isinstance(colors, Iterable):
        cmap = colors
    else:
        cmap = [colors(1. * i / n_colors) for i in range(n_colors)]

    fig, ax = plt.subplots(figsize=(2, 2))  # 2 inch by 2 inch

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    #ax = fig.add_subplot(111)
    contours = []
    groups = []

    if condition_name == None:
        x = df[x_name]
        y = df[y_name]
        kde_coordinate = np.vstack([x, y])  # shape = (2(dimension), number of points)
        if kde_coordinate.shape[1] <= 2:  # if there is only few points, it cannot calculate gaussian kde
            raise ValueError('Number of points should be greater than 2 to create contour')
        else:
            kde = scipy.stats.gaussian_kde(kde_coordinate)  # Define kernel (bandwidth by Scott's Rule)

            # evaluate on a regular grid
            xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
            ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            # Xgrid , Ygrid = (bin_num,bin_num) 2d array
            # Xgrid[i] = xgrid coordinate(divide by 100 from -x ~ +x` and repeat in row direction)
            # Ygrid[:,i] = ygrid coordinate(divide by 100 from -y ~ +y` and repeat in column direction)
            Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
            # Xgrid.ravel() = bin_num^2 shape 1d vector (vector that repeat -x ~ +x`, -x ~ +x` bin_num times)
            # np.vstack() = (2,bin_num^2) 2d array linspaced coordinates
            # Z = (10000,) 1d vector
            pdf = Z.reshape(Xgrid.shape)
            contour = ax.contour(Xgrid, Ygrid, pdf,
                                 # colors='red',
                                 linewidths=1,
                                 linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                                 # label=group,
                                 cmap='Reds',
                                 origin='lower',
                                 levels=num_contours,
                                 )
            contours.append(contour)

        format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(path + '/%s.png' % (file_name), dpi=300)

        if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/')
        plt.savefig(path + 'svg/%s.svg' % (file_name))
        plt.close()
        plt.clf()

    else:
        for i, group in enumerate(list(np.unique(df[condition_name]))):
            x = df[df[condition_name] == group][x_name]
            y = df[df[condition_name] == group][y_name]

            kde_coordinate = np.vstack([x, y])  # shape = (2(dimension), number of points)
            if kde_coordinate.shape[1] <= 2: # if there is only few points, it cannot calculate gaussian kde
                continue
            else:
                kde = scipy.stats.gaussian_kde(kde_coordinate)  # Define kernel (bandwidth by Scott's Rule)

                # evaluate on a regular grid
                xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
                ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
                Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
                # Xgrid , Ygrid = (bin_num,bin_num) 2d array
                # Xgrid[i] = xgrid coordinate(divide by 100 from -x ~ +x` and repeat in row direction)
                # Ygrid[:,i] = ygrid coordinate(divide by 100 from -y ~ +y` and repeat in column direction)
                Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
                # Xgrid.ravel() = bin_num^2 shape 1d vector (vector that repeat -x ~ +x`, -x ~ +x` bin_num times)
                # np.vstack() = (2,bin_num^2) 2d array linspaced coordinates
                # Z = (10000,) 1d vector
                pdf = Z.reshape(Xgrid.shape)
                contour = ax.contour(Xgrid, Ygrid, pdf,
                            # colors='red',
                            linewidths=1,
                            linestyles='solid',  # 'solid', 'dashed', 'dashdot', 'dotted'
                            #label=group,
                            #cmap=colors[i],
                            colors=cmap[i],
                            origin='lower',
                            levels=num_contours,
                            )
                contours.append(contour)
                groups.append(group)

        format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
        ax.legend(handles=[contour.legend_elements()[0][-1] for contour in contours], labels=groups,
                  bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0, fontsize=3, frameon=False, markerscale=0.3)
        #ax.legend(handles=[contour.legend_elements()[0][-1] for contour in contours], labels=groups, fontsize=3, markerscale=0.3)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(path + '/%s.png' % (file_name), dpi=300)

        if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/')
        plt.savefig(path + 'svg/%s.svg' % (file_name))
        plt.close()
        plt.clf()

def draw_clustermap(data, path, file_name, vmax=None, annot=False, metric='euclidean', transpose=False,
                    row_cluster=True, col_cluster=True, cmap='OrRd', figsize=(4,4)):

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    # matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    # matplotlib.rcParams['lines.linewidth'] = 1
    if transpose == True:
        data = data.T

    kws = dict(
        cbar_kws=dict(ticks=[0, math.floor(np.max(np.max(data)) / 2), math.floor(np.max(np.max(data)))],
                      orientation='horizontal'), vmin=0)
    if vmax !=None:
        kws = dict(
            cbar_kws=dict(ticks=[0, vmax], orientation='horizontal'), vmin=0)

    # Tick number = [0, max/2, max] (you have to set vmin=0 otherwise 0 tick will not show up)
    if metric == 'euclidean':
        method = 'ward'
    elif metric == 'correlation':
        method = 'average'

    if annot == True:
        fmt = '.1f'
    if annot == False:
        fmt = None

    g = sns.clustermap(data, annot=annot, fmt=fmt, annot_kws={"fontsize": 12},  # .1f
                       cmap=cmap, method='ward', col_cluster=col_cluster, row_cluster=row_cluster, vmax=vmax,
                       # cbar_pos=(1, 0.2, 0.03, 0.8),
                       linewidths=0.5, linecolor='black', alpha=0.7,
                       **kws,
                       figsize=figsize,
                       )

    if data.shape[0] <= 2:
        g.ax_row_dendrogram.set_visible(False)
    # cbar_pos = (left, bottom, width, height)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=16, va='center', rotation=0)
    # Set tick label size = 16, and center y tick labels (centering x tick labels make it strange)

    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([0, 0.9, g.ax_row_dendrogram.get_position().width, 0.02])
    # Set position of colorbar

    g.ax_cbar.set_title('Probability (%)', fontsize=12)
    g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)
    # Set colorbar title and font size

    for spine in g.ax_cbar.spines:
        # g.ax_cbar.spines[spine].set_color('crimson')
        g.ax_cbar.spines[spine].set_linewidth(0.5)
    # Set bounding box line width of colorbar

    # plt.xlabel('%s' % cluster_type)
    # plt.ylabel('%s' % condition_name)

    plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

    plt.clf()
    plt.close()

def draw_cluster_distribution_heatmap(
    df,
    path,
    file_name,
    condition_name,
    cluster_type,
    vmax=None,
    annot=False,
    metric='euclidean',
    transpose=False,
    row_cluster=True,
    col_cluster=True,
    cmap='OrRd',
    figsize=(4, 4),
):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    counts = df.groupby([condition_name, cluster_type]).size()

    group_clone_size = counts.unstack(level=0).fillna(0)
    group_clone = counts.groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0).fillna(0)

    clusters = sorted(pd.unique(df[cluster_type]))

    group_clone = group_clone.T
    group_clone_size = group_clone_size.T

    for cluster in clusters:
        if cluster not in group_clone.columns:
            group_clone[cluster] = 0
        if cluster not in group_clone_size.columns:
            group_clone_size[cluster] = 0

    group_clone = group_clone.reindex(sorted(group_clone.columns), axis=1)
    group_clone_size = group_clone_size.reindex(sorted(group_clone_size.columns), axis=1)

    group_clone.columns.name = None
    group_clone.index.name = None
    group_clone_size.columns.name = None
    group_clone_size.index.name = None

    if transpose:
        group_clone = group_clone.T
        group_clone_size = group_clone_size.T

    font = {
        'family': 'arial',
        'weight': 'normal',
        'size': 8,
    }
    matplotlib.rc('font', **font)

    data_max = float(np.nanmax(group_clone.values)) if group_clone.size else 0
    if vmax is None:
        ticks = [0, math.floor(data_max / 2), math.floor(data_max)]
    else:
        ticks = [0, vmax]

    kws = dict(
        cbar_kws=dict(ticks=ticks, orientation='horizontal'),
        vmin=0,
    )

    if metric == 'euclidean':
        method = 'ward'
    elif metric == 'correlation':
        method = 'average'
    else:
        method = 'average'

    row_linkage = None
    col_linkage = None

    if row_cluster and group_clone.shape[0] > 1:
        row_dist = pdist(group_clone.values, metric=metric)
        row_linkage = linkage(row_dist, method=method)

    if col_cluster and group_clone.shape[1] > 1:
        col_dist = pdist(group_clone.values.T, metric=metric)
        col_linkage = linkage(col_dist, method=method)

    if annot == 'size':
        annot_data = group_clone_size
        fmt = 'g'
    elif annot is True:
        annot_data = True
        fmt = '.1f'
    else:
        annot_data = False
        fmt = ''

    g = sns.clustermap(
        group_clone,
        annot=annot_data,
        fmt=fmt,
        annot_kws={"fontsize": 12},
        cmap=cmap,
        metric=metric,
        method=method,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        vmax=vmax,
        linewidths=0.5,
        linecolor='black',
        alpha=0.7,
        **kws,
        figsize=figsize,
    )

    if group_clone.shape[0] <= 2:
        g.ax_row_dendrogram.set_visible(False)

    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=16, va='center', rotation=0)
    #g.ax_heatmap.set_yticks(np.arange(group_clone.shape[0]) + 0.5)
    #g.ax_heatmap.set_yticklabels(group_clone.index, fontsize=16, rotation=0)

    g.ax_cbar.set_position([0, 0.96, g.ax_row_dendrogram.get_position().width, 0.02])
    g.ax_cbar.set_title('Occurrence (%)', fontsize=12)
    g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)

    for spine in g.ax_cbar.spines.values():
        spine.set_linewidth(0.5)

    plt.savefig(path + f'{file_name}.png', dpi=300, bbox_inches='tight')

    svg_dir = path + 'svg/'
    if not os.path.isdir(svg_dir):
        os.makedirs(svg_dir)
    plt.savefig(svg_dir + f'{file_name}.svg', bbox_inches='tight')

    plt.clf()
    plt.close()

    return group_clone


def draw_relative_cluster_distribution_heatmap(df, path, file_name, condition_name, cluster_type, vmax=None, annot=False, metric='euclidean',
                                      transpose=False, row_cluster=True, col_cluster=True, cmap='OrRd', figsize=(4, 4)):
    if vmax != None:
        vmax = vmax
    group_clone = pd.DataFrame(df.groupby([condition_name, cluster_type]).size())
    # group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    # group_clone = group_clone.unstack(level=0)
    # group_clone[np.isnan(group_clone)] = 0  # fill na with 0 (np.isnan() returns bool array, which is True whenever nan )

    group_clone_size = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: x)
    group_clone_size = group_clone_size.unstack(level=0)
    group_clone_size[np.isnan(group_clone_size)] = 0
    group_clone_size_T = group_clone_size.T
    for cluster in sorted(list(pd.unique(df[cluster_type]))):
        if cluster in group_clone_size_T.columns:
            continue
        else:
            group_clone_size_T.insert(loc=int(cluster), column=cluster, value=0)
            group_clone_size_T.sort_index(axis=1, inplace=True)

    group_clone_size = group_clone_size_T.T

    group_clone_size.columns.name = None  # Remove name of column = 'Type'
    group_clone_size.index.name = None  # Remove name of index = 'tskmeans'

    group_clone_size = group_clone_size.T

    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    group_clone_T = group_clone.T
    for cluster in sorted(list(pd.unique(df[cluster_type]))):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=0)
            group_clone_T.sort_index(axis=1, inplace=True)

    group_clone = group_clone_T.T

    group_clone.columns.name = None  # Remove name of column = 'Type'
    group_clone.index.name = None  # Remove name of index = 'tskmeans'

    group_clone = group_clone.T

    column_sums = group_clone.sum(axis=0)
    group_clone_relative = group_clone.div(column_sums, axis=1)*100

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    # matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    # matplotlib.rcParams['lines.linewidth'] = 1
    if transpose == True:
        group_clone = group_clone.T
        group_clone_relative = group_clone_relative.T
        group_clone_size = group_clone_size.T

    if metric =='euclidean':
        method = 'ward'
    elif metric=='correlation':
        method = 'average'

    kws = dict(
        cbar_kws=dict(ticks=[0, math.floor(np.max(np.max(group_clone_relative)) / 2), math.floor(np.max(np.max(group_clone_relative)))],
                      orientation='horizontal'), vmin=0)
    # Tick number = [0, max/2, max] (you have to set vmin=0 otherwise 0 tick will not show up)
    g = sns.clustermap(group_clone_relative, annot=False, cmap='OrRd', col_cluster=col_cluster,
                       metric=metric, method=method,
                       row_cluster=row_cluster, vmax=vmax,
                       # cbar_pos=(1, 0.2, 0.03, 0.8),
                       linewidths=0.5, linecolor='black',
                       alpha=0.7,
                       **kws,
                       figsize=figsize,
                       )

    row_order = g.dendrogram_row.reordered_ind if row_cluster else range(len(group_clone.index))
    col_order = g.dendrogram_col.reordered_ind if col_cluster else range(len(group_clone.columns))
    group_clone_size = group_clone_size.iloc[row_order, col_order]

    plt.clf()

    if annot == 'size':
        annot = group_clone_size
        fmt = 'G'
    if annot == True:
        fmt = '.1f'
    if annot == False:
        fmt = None

    g = sns.clustermap(group_clone_relative, annot=annot, fmt=fmt, annot_kws={"fontsize": 12},  # .1f
                       cmap=cmap, method='ward', col_cluster=col_cluster, row_cluster=row_cluster, vmax=vmax,
                       # cbar_pos=(1, 0.2, 0.03, 0.8),
                       linewidths=0.5, linecolor='black', alpha=0.7,
                       **kws,
                       figsize=figsize,
                       )

    if group_clone_relative.shape[0] <= 2:
        g.ax_row_dendrogram.set_visible(False)
    # cbar_pos = (left, bottom, width, height)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=16, va='center', rotation=0)
    # Set tick label size = 16, and center y tick labels (centering x tick labels make it strange)

    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([0, 0.9, g.ax_row_dendrogram.get_position().width, 0.02])
    # Set position of colorbar

    g.ax_cbar.set_title('Occurrence (%)', fontsize=12)
    g.ax_cbar.tick_params(axis='x', length=10, labelsize=12)
    # Set colorbar title and font size

    for spine in g.ax_cbar.spines:
        # g.ax_cbar.spines[spine].set_color('crimson')
        g.ax_cbar.spines[spine].set_linewidth(0.5)
    # Set bounding box line width of colorbar

    # plt.xlabel('%s' % cluster_type)
    # plt.ylabel('%s' % condition_name)

    plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')

    plt.clf()
    plt.close()

def draw_heatmap_with_circles(df, path, file_name, condition_name, cluster_type, vmax=None, transpose=False, row_cluster=True, col_cluster=True, figsize=(4,4)):
    from matplotlib.collections import PatchCollection
    group_clone = pd.DataFrame(df.groupby([condition_name, cluster_type]).size())
    group_clone_size = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: x)
    group_clone_size = group_clone_size.unstack(level=0)
    group_clone_size[np.isnan(group_clone_size)] = 0
    group_clone_size_T = group_clone_size.T
    for cluster in sorted(list(pd.unique(df[cluster_type]))):
        if cluster in group_clone_size_T.columns:
            continue
        else:
            group_clone_size_T.insert(loc=int(cluster), column=cluster, value=0)
            group_clone_size_T.sort_index(axis=1, inplace=True)

    group_clone_size = group_clone_size_T.T

    group_clone_size.columns.name = None  # Remove name of column = 'Type'
    group_clone_size.index.name = None  # Remove name of index = 'tskmeans'

    # group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    # group_clone = group_clone.unstack(level=0)
    # group_clone[np.isnan(group_clone)] = 0  # fill na with 0 (np.isnan() returns bool array, which is True whenever nan )

    group_clone = group_clone[0].groupby(level=0, group_keys=False).apply(lambda x: 100 * x / x.sum())
    group_clone = group_clone.unstack(level=0)
    group_clone[np.isnan(group_clone)] = 0
    group_clone_T = group_clone.T
    for cluster in sorted(list(pd.unique(df[cluster_type]))):
        if cluster in group_clone_T.columns:
            continue
        else:
            group_clone_T.insert(loc=int(cluster), column=cluster, value=0)
            group_clone_T.sort_index(axis=1, inplace=True)

    group_clone = group_clone_T.T

    group_clone.columns.name = None  # Remove name of column = 'Type'
    group_clone.index.name = None  # Remove name of index = 'tskmeans'

    input_colors = group_clone
    input_sizes = group_clone_size

    if input_colors.shape != input_sizes.shape:
        raise ValueError("Input matrices must be the same shape.")
    if all(input_colors.index == input_sizes.index) == False:
        raise ValueError("Input indexes and order must be equal.")
    if all(input_colors.columns == input_sizes.columns) == False:
        raise ValueError("Input columns and order must be equal.")

    # clustergrid = sns.clustermap(input_colors, cmap="OrRd",
    #                             row_cluster=row_cluster, col_cluster=col_cluster, method=method)
    input_colors = input_colors.T
    input_sizes = input_sizes.T

    if transpose == True:
        input_colors = input_colors.T
    # fig, ax = plt.subplots(figsize=(4, 4))

    clustergrid = sns.clustermap(input_colors, annot=False, cmap='OrRd', method='ward', col_cluster=col_cluster,
                                 row_cluster=row_cluster, vmax=vmax,
                                 # cbar_pos=(1, 0.2, 0.03, 0.8),
                                 linewidths=0.5, linecolor='black',
                                 alpha=0.7,
                                 # figsize=(4,4),
                                 )

    row_order = clustergrid.dendrogram_row.reordered_ind if row_cluster else range(len(input_colors.index))
    col_order = clustergrid.dendrogram_col.reordered_ind if col_cluster else range(len(input_colors.columns))
    input_colors = input_colors.iloc[row_order, col_order]
    input_sizes = input_sizes.iloc[row_order, col_order]
    plt.clf()

    fig, ax = plt.subplots(figsize=figsize)

    N, M = input_colors.shape
    ylabels = list(input_colors.index)
    xlabels = list(input_colors.columns)

    x, y = np.meshgrid(np.arange(M), np.arange(N))
    s = input_sizes.values  # size column
    c = input_colors.values

    R = s / s.max() / 2
    circles = [plt.Circle((j, i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
    col = PatchCollection(circles, array=c.flatten(), cmap='OrRd',
                          norm=matplotlib.colors.Normalize(vmin=None, vmax=vmax))
    ax.add_collection(col)

    ax.set(xticks=np.arange(M), yticks=np.arange(N),
           xticklabels=xlabels, yticklabels=ylabels)
    ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
    # ax.grid(which='minor')

    ax.set_facecolor((1, 1, 1, 0))
    #ax.grid(b=None)
    ax.grid(which='minor')
    # fig.colorbar(col)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    plt.savefig(path + '%s.png' % (file_name), dpi=300, bbox_inches='tight')
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % (file_name), bbox_inches='tight')
    plt.clf()
    plt.close()

def draw_state_occupancy(df, path, bin_num, mode, condition_name, x_name='PC1', y_name='PC2'):

    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    from scipy.stats import gaussian_kde
    if mode == 'all':
        x = df[x_name]  # (23177, )
        y = df[y_name]  # (23177, )
        xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
        ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing='xy')
        # Xgrid , Ygrid = each (100,100) 2d array
        # Xgrid[i] = xgrid 좌표(xmin~ xmax 100분할한게 row방향으로 반복)
        # Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)

        kde_data = np.vstack([x, y])  # (2, 23177) 2d array
        kde = gaussian_kde(kde_data)  # kernel 정의(bandwidth by Scott's Rule)
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        pdf = Z.reshape(Xgrid.shape)
        pdf = pdf / np.sum(pdf)

        plt.figure(figsize=(20, 15))
        plt.imshow(pdf, origin='lower', aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='jet')
        cb = plt.colorbar()
        cb.set_label("density")
        plt.savefig(path + 'gaussian_kde_all.png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

        count = np.zeros((bin_num, bin_num))
        for coor in zip(x, y):
            residual = (coor[0] - Xgrid) ** 2 + (coor[1] - Ygrid) ** 2  # residual = (100,100) 2d array
            min_coordinate = np.unravel_index(np.argmin(residual), residual.shape)  # residual이 minimum인 array index 반환, (x,y)꼴
            count[min_coordinate] = count[min_coordinate] + 1
            # count = count/x.shape[0]
        plt.figure(figsize=(20, 15))
        plt.imshow(count, origin='lower', aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='jet')
        cb = plt.colorbar()
        cb.set_label("counts")
        plt.savefig(path + 'state_occupancy_all.png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

    elif mode =='condition':
        for idx, condition_name in enumerate(list(pd.unique(df[condition_name]))):
            x = df[df['Type'] == condition_name][x_name]  # (data 수,) 1d vector
            y = df[df['Type'] == condition_name][y_name]  # (data 수,) 1d vector
            xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
            ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing='xy')

            kde_data = np.vstack([x, y])  # (2, 23177) 2d array
            kde = gaussian_kde(kde_data)  # kernel 정의(bandwidth by Scott's Rule)
            Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
            pdf = Z.reshape(Xgrid.shape)
            pdf = pdf / np.sum(pdf)

            plt.figure(figsize=(20, 15))
            plt.imshow(pdf, origin='lower', aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='jet')
            cb = plt.colorbar()
            cb.set_label("density")

            plt.savefig(path + 'gaussian_kde_%s.png' % (condition_name), dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()

            count = np.zeros((bin_num, bin_num))
            for coor in zip(x, y):
                residual = (coor[0] - Xgrid) ** 2 + (coor[1] - Ygrid) ** 2  # residual = (100,100) 2d array
                min_coordinate = np.unravel_index(np.argmin(residual),
                                                  residual.shape)  # residual이 minimum인 array index 반환, (x,y)꼴
                count[min_coordinate] = count[min_coordinate] + 1

            plt.figure(figsize=(20, 15))
            plt.imshow(count, origin='lower', aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='jet')
            plt.title(condition_name)
            cb = plt.colorbar()
            cb.set_label("counts")

            plt.savefig(path + 'state_occupancy_%s.png' % (condition_name), dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()

def draw_transition_field(df, path, bin_num, mode, condition_name, label_name, x_name='PC1', y_name='PC2'):

    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    from math import sqrt
    from statistics import mean
    if mode == 'all':
        xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
        ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing='xy')
        # Xgrid , Ygrid = 각각 (100,100) 2d array
        # Xgrid[i] = xgrid 좌표(-2에서 5를 100분할한게 row방향으로 반복)
        # Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)

        ########### 각 element가 list인 100x100 array 형성 ##########
        transition_mag_array_temp = np.empty((bin_num, bin_num), dtype='object')
        transition_vec_array_temp = np.empty((bin_num, bin_num), dtype='object')
        for row in range(0, transition_mag_array_temp.shape[0]):
            for col in range(0, transition_mag_array_temp.shape[1]):
                transition_mag_array_temp[row, col] = [0]
                transition_vec_array_temp[row, col] = [(0, 0)]

                ######### 각 세포, 시간마다 transition magnitude 계산하며 list로 append ##########
        label_data = df.groupby([condition_name, label_name]).apply(lambda x: x.name)
        for traj_idx in range(0, label_data.shape[0]):  # 각 세포마다
            traj_data = df.groupby([condition_name, label_name]).get_group(label_data[traj_idx]).copy().reset_index()
            # 한 세포에 time span에 대한 PC1, PC2, GMM_cluster, Kmeans_cluster 정보

            transition_mag = 0
            for t in range(0, traj_data.shape[0]-1):  # 한 세포 안에서 각 time frame 마다
                x = traj_data[x_name]
                y = traj_data[y_name]
                residual = (x[t] - Xgrid) ** 2 + (y[t] - Ygrid) ** 2  # residual = (100,100) 2d array
                min_coordinate = np.unravel_index(np.argmin(residual), residual.shape)  # residual이 minimum인 array index 반환, (x,y)꼴
                transition_mag = sqrt((x[t] - x[t + 1]) ** 2 + (y[t] - y[t + 1]) ** 2)
                transition_vec = (x[t + 1] - x[t], y[t + 1] - y[t])
                transition_mag_array_temp[min_coordinate].append(transition_mag)
                transition_vec_array_temp[min_coordinate].append(transition_vec)

        ########### transition magnitude의 list의 element 개수가 2 이상이면 0을 제외(평균 낼 때 평균값을 작게 만듬) #########
        for row in range(0, transition_mag_array_temp.shape[0]):
            for col in range(0, transition_mag_array_temp.shape[1]):
                if len(transition_mag_array_temp[row, col]) > 1:
                    transition_mag_array_temp[row, col].remove(0)
                if len(transition_vec_array_temp[row, col]) > 1:
                    transition_vec_array_temp[row, col].remove((0, 0))

        ########### 각 element가 transition magnitude의 list인 100x100 array -> 각 list의 평균이 element인 100x100 array #########
        transition_mag_array = np.empty((bin_num, bin_num))
        transition_vec_x_array = np.empty((bin_num, bin_num))
        transition_vec_y_array = np.empty((bin_num, bin_num))

        for row in range(0, transition_mag_array.shape[0]):
            for col in range(0, transition_mag_array.shape[1]):
                transition_mag_array[row, col] = mean(transition_mag_array_temp[row, col])
                x_temp = []
                y_temp = []
                for x, y in transition_vec_array_temp[row, col]:
                    x_temp.append(x)
                    y_temp.append(y)
                transition_vec_x_array[row, col] = mean(x_temp)
                transition_vec_y_array[row, col] = mean(y_temp)

        plt.figure(figsize=(20, 15))
        plt.imshow(transition_mag_array, origin='lower', aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='jet')
        cb = plt.colorbar()
        cb.set_label("counts")
        plt.savefig(path + 'transition_magnitude_field_all.png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

        dot_c1 = np.arange(bin_num * bin_num)
        plt.figure(figsize=(20, 15))
        plt.quiver(Xgrid, Ygrid, transition_vec_x_array, transition_vec_y_array, dot_c1,
                   scale_units='xy', angles='xy', scale=1, cmap=plt.cm.get_cmap('flag'))
        plt.savefig(path + 'transition_vector_field_all.png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

    if mode == 'condition':
        xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
        ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing='xy')
        # Xgrid , Ygrid = 각각 (100,100) 2d array
        # Xgrid[i] = xgrid 좌표(-2에서 5를 100분할한게 row방향으로 반복)
        # Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)

        for condition in list(pd.unique(df[condition_name])):
            label_data = df.groupby([condition_name, label_name]).apply(lambda x: x.name).reset_index() # reset_index produce separate 'Type' and 'ID' column
            label_data = label_data[label_data[condition_name] == condition].reset_index()
            label_data = label_data.drop('index', axis=1)

            ########### 각 element가 list인 100x100 array 형성 ##########
            transition_mag_array_temp = np.empty((bin_num, bin_num), dtype='object')
            transition_vec_array_temp = np.empty((bin_num, bin_num), dtype='object')
            for row in range(0, transition_mag_array_temp.shape[0]):
                for col in range(0, transition_mag_array_temp.shape[1]):
                    transition_mag_array_temp[row, col] = [0]
                    transition_vec_array_temp[row, col] = [(0, 0)]

            ######### 각 세포, 시간마다 transition magnitude 계산하며 list로 append ##########
            for traj_idx in range(0, label_data.shape[0]):  # 각 세포마다
                cell_data = df.groupby([condition_name, label_name]).get_group(label_data[0][traj_idx]).copy().reset_index()
                # 한 세포에 time span에 대한 PC1, PC2, GMM_cluster, Kmeans_cluster 정보
                transition_mag = 0
                for t in range(0, cell_data.shape[0] - 1):  # 한 세포 안에서 각 time frame 마다
                    x = cell_data[x_name]
                    y = cell_data[y_name]
                    residual = (x[t] - Xgrid) ** 2 + (y[t] - Ygrid) ** 2  # residual = (100,100) 2d array
                    min_coordinate = np.unravel_index(np.argmin(residual), residual.shape)  # residual이 minimum인 array index 반환, (x,y)꼴
                    transition_mag = sqrt((x[t] - x[t + 1]) ** 2 + (y[t] - y[t + 1]) ** 2)
                    transition_vec = (x[t + 1] - x[t], y[t + 1] - y[t])
                    transition_mag_array_temp[min_coordinate].append(transition_mag)
                    transition_vec_array_temp[min_coordinate].append(transition_vec)

            ########### transition magnitude의 list의 element 개수가 2 이상이면 0을 제외(평균 낼 때 평균값을 작게 만듬) #########
            for row in range(0, transition_mag_array_temp.shape[0]):
                for col in range(0, transition_mag_array_temp.shape[1]):
                    if len(transition_mag_array_temp[row, col]) > 1:
                        transition_mag_array_temp[row, col].remove(0)
                    if len(transition_vec_array_temp[row, col]) > 1:
                        transition_vec_array_temp[row, col].remove((0, 0))

            ########### 각 element가 transition magnitude의 list인 100x100 array -> 각 list의 평균이 element인 100x100 array #########
            transition_mag_array = np.empty((bin_num, bin_num))
            transition_vec_x_array = np.empty((bin_num, bin_num))
            transition_vec_y_array = np.empty((bin_num, bin_num))

            for row in range(0, transition_mag_array.shape[0]):
                for col in range(0, transition_mag_array.shape[1]):
                    transition_mag_array[row, col] = mean(transition_mag_array_temp[row, col])
                    x_temp = []
                    y_temp = []
                    for x, y in transition_vec_array_temp[row, col]:
                        x_temp.append(x)
                        y_temp.append(y)
                    transition_vec_x_array[row, col] = mean(x_temp)
                    transition_vec_y_array[row, col] = mean(y_temp)

            plt.figure(figsize=(20, 15))
            plt.imshow(transition_mag_array, origin='lower', aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='jet')
            cb = plt.colorbar()
            cb.set_label("counts")

            plt.savefig(path + 'transition_magnitude_field_%s.png' % (condition), dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()

            dot_c1 = np.arange(bin_num * bin_num)
            plt.figure(figsize=(20, 15))
            plt.quiver(Xgrid, Ygrid, transition_vec_x_array, transition_vec_y_array, dot_c1,
                       scale_units='xy', angles='xy', scale=1, cmap=plt.cm.get_cmap('flag'))

            plt.savefig(path + 'transition_vector_field_%s.png' % (condition), dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()

def draw_1d_trajectory(path, df, duration, feature_name):
    traj_list, _, _ = to_timeseries_fast(df, duration=duration, feature_name=feature_name)

    for traj_idx, traj_df in enumerate(traj_list):
        series = traj_df[feature_name].values.flatten()
        time_range = range(0,duration)
        fig, ax = plt.subplots(figsize=(15, 6))

        sns.lineplot(x=time_range, y=series, linewidth = 3)
        #slope, intercept, r_val, p_val, SE = scipy.stats.linregress(time_range, series)
        #sns.lineplot(x=time_range, y=[slope * t for t in time_range] + intercept, color='red')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(feature_name, fontsize=12)
        plt.tick_params(axis='x', labelsize=10)
        plt.tick_params(axis='y', labelsize=10)
        ax.set_xticks(ticks=time_range)
        # ax.set_yticks(ticks=np.linspace(min(a), max(a), 5))
        # ax.grid(True)
        plt.xlim(0, duration-1)
        #plt.ylim(, )

        if not os.path.isdir(path + '%s_series/'%feature_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + '%s_series/'%feature_name)
        plt.savefig(path + '%s_series/%s.png' % (feature_name, traj_idx), dpi=300, bbox_inches='tight')

        if not os.path.isdir(path + 'svg/%s_series/'%feature_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/%s_series/'%feature_name)
        plt.savefig(path + 'svg/%s_series/%s.svg' % (feature_name, traj_idx), bbox_inches='tight')
        plt.clf()
        plt.close()


def draw_2d_trajectory_arrow(path, df, duration, feature_name, label_name='pseudo_Label'):
    traj_list, time_series, _ = to_timeseries_fast(df, duration=duration, feature_name=feature_name)
    x_name = feature_name[0]
    y_name = feature_name[1]
    for traj_idx, traj_df in enumerate(traj_list):
        plt.quiver(np.array(traj_df[x_name].iloc[:-1]), np.array(traj_df[y_name].iloc[:-1]),
                   np.array(traj_df[x_name].iloc[1:]) - np.array(traj_df[x_name].iloc[:-1]),
                   np.array(traj_df[y_name].iloc[1:]) - np.array(traj_df[y_name].iloc[:-1]),
                   np.arange(traj_df.shape[0] - 1), scale_units='xy', angles='xy', scale=1,
                   cmap=plt.cm.get_cmap('jet'))
        plt.title( 'idx: ' + str(traj_idx) + '  ' +'  label: ' + str(traj_df[label_name].values[0]))
        plt.xlabel(x_name)
        plt.ylabel(y_name)

        if not os.path.isdir(path + '%s_trajectory/'%feature_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + '%s_trajectory/'%feature_name)
        plt.savefig(path + '%s_trajectory/%s.png' % (feature_name, traj_idx), dpi=300, bbox_inches='tight')

        if not os.path.isdir(path + 'svg/%s_trajectory/'%feature_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/%s_trajectory/'%feature_name)
        plt.savefig(path + 'svg/%s_trajectory/%s.svg' % (feature_name, traj_idx), bbox_inches='tight')
        plt.clf()
        plt.close()

def add_scale_bar(ax, length=50, loc='lower right',
                  pad=0.08, bar_height=0.03, color='black', linewidth=2,
                  fontsize=8):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    x_range = x1 - x0
    y_range = y1 - y0

    if loc == 'lower right':
        x_start = x1 - pad * x_range - length
        x_end = x1 - pad * x_range
        y = y0 + pad * y_range
        va = 'top'
        text_y = y - bar_height * y_range

    elif loc == 'lower left':
        x_start = x0 + pad * x_range
        x_end = x_start + length
        y = y0 + pad * y_range
        va = 'top'
        text_y = y - bar_height * y_range

    else:
        raise ValueError("loc must be 'lower right' or 'lower left'")

    label = '%s µm'%length
    ax.plot([x_start, x_end], [y, y], color=color, linewidth=linewidth, solid_capstyle='butt')
    ax.text((x_start + x_end) / 2, text_y, label, ha='center', va=va,
            color=color, fontsize=fontsize)

def draw_2D_trajectories_one_figure(df_duration, df, path, duration=20, n_examples=30,
                                    label_name='kmeans', feature_name=['Rotated_X', 'Rotated_Y'],
                                    scale_bar_um=50, lim=300):
    ''' Plots example trajectories per condition
        Parameters:
        ----------
        df_duration: pandas dataframe
            position X and Y, where each row is one cell state at time t
        df: pandas dataframe
            that has labels, where each row is one cell trajectory
        duration: int
            Number of time frames for each cell trajectory (all trajectories should have same duration)
        n_examples: int
            Number of example trajectories to extract per figure
        label_name: str
            Name of condition column in df
        feature_name: list of str
            Name of column that has x, y coordinates in df
        Returns:
        -------
        fig: figure
            figure with the example trajectories per condition
        '''
    label = df[label_name]
    label_expanded = np.repeat(label, duration).reset_index(drop=True)
    df_duration[label_name] = label_expanded

    n_colors = n_examples
    # cm = plt.cm.get_cmap(name='jet')
    cm = cmc.romaO
    currentColors = [cm(1. * i / n_colors) for i in range(n_colors)]

    for cluster in np.unique(df_duration[label_name]):

        font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 1

        fig, ax = plt.subplots()
        df_part = df_duration[df_duration[label_name]==cluster].reset_index(drop=True)
        n_trajs = df_part.shape[0]//duration

        if n_trajs >= n_examples:
            random_traj_idxs = np.random.choice(range(0, n_trajs), size=n_examples, replace=False)  # Replace=False -> No redundant value

        elif n_trajs < n_examples:
            raise ValueError('%s has less than %s trajectories'%(cluster, n_examples))

        for i, traj_idx in enumerate(random_traj_idxs):
            traj = df_part[duration * traj_idx:duration * (traj_idx + 1)][feature_name].values
            plt.plot(traj[:, 0] - traj[0][0], traj[:, 1] - traj[0][1], '-', color=currentColors[i], linewidth=2, )
            # plt.title( 'idx: ' + str(traj_idx) + '  ' +'  label: ' + str(traj_df[label_name].values[0]))
            plt.xlim(-lim, lim)
            plt.ylim(-lim, lim)
            #plt.axvline(0, color='black', linewidth=1)
            #plt.axhline(0, color='black', linewidth=1)

        #plt.title('%s: %s' % (label_name, cluster))
        #plt.xlabel(feature_name[0])
        #plt.ylabel(feature_name[1])

        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        if scale_bar_um is not None:
            add_scale_bar(
                ax,
                length=scale_bar_um,
                loc='lower right',
                color='black',
                linewidth=2,
                fontsize=8
            )

        plt.axis('off')
        if not os.path.isdir(path + '%s_trajectory/'%feature_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + '%s_trajectory/'%feature_name)
        plt.savefig(path + '%s_trajectory/%s.png' % (feature_name, cluster), dpi=300, bbox_inches='tight')

        if not os.path.isdir(path + 'svg/%s_trajectory/'%feature_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/%s_trajectory/'%feature_name)
        plt.savefig(path + 'svg/%s_trajectory/%s.svg' % (feature_name, cluster), bbox_inches='tight')

        plt.clf()
        plt.close()

        # for traj_idx in range(0, len(traj_list)):
        #     if traj_list[traj_idx][label_name].iloc[0] == cluster:
        #         traj = time_series[traj_idx]
        #         plt.plot(traj[:,0]-traj[0][0], traj[:,1]-traj[0][1], '-', color= currentColors[i], linewidth = 2,)
        #         #plt.title( 'idx: ' + str(traj_idx) + '  ' +'  label: ' + str(traj_df[label_name].values[0]))
        #         plt.xlim(-lim, lim)
        #         plt.ylim(-lim, lim)
        #         plt.axvline(0, color='black', linewidth=1)
        #         plt.axhline(0, color='black', linewidth=1)
        #         i=i+1
        #     if i >= number_of_ex:
        #         break
        # plt.title('%s: %s' %(label_name, cluster))
        # plt.xlabel(feature_name[0])
        # plt.ylabel(feature_name[1])
        #
        # if not os.path.isdir(path + '%s_trajectory/'%feature_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        #     os.makedirs(path + '%s_trajectory/'%feature_name)
        # plt.savefig(path + '%s_trajectory/%s.png' % (feature_name, cluster), dpi=300, bbox_inches='tight')
        # plt.clf()
        # plt.close()


def draw_2D_trajectory(trajectories, path, folder_name, lim, df=None, condition_name1=None):

    from matplotlib.collections import LineCollection
    if any(df != None) & (condition_name1 != None):
        labels = reduced_labels(df, duration=trajectories[0].shape[0])
    for traj_idx in tqdm(trajectories):
        traj = trajectories[traj_idx]
        if any(df != None) & (condition_name1 != None):
            condition1 = labels[condition_name1][traj_idx]
        x = traj[:, 0]
        y = traj[:, 1]
        colors = np.arange(traj.shape[0]) / traj.shape[0]
        color_list = []

        for i in range(0, traj.shape[0]):
            color_list.append(plt.cm.jet(colors[i]))
        xy = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])  # (x0, y0), (x1, y1), ... (xt, yt) form
        fig, ax = plt.subplots()
        lc = LineCollection(segments, colors=color_list)

        plt.scatter(x, y, c=color_list, s=5)
        ax.add_collection(lc)
        ax.set_aspect('equal', adjustable='datalim') # adjustable = 'box', 'datalim'

        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        # ax.set_title('A multi-color plot')

        if not os.path.isdir(path + '%s/'%folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + '%s/'%folder_name)
        if any(df != None) & (condition_name1 != None):
            plt.savefig(path + '%s/%s_%s.png' % (folder_name, condition_name1,traj_idx), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(path + '%s/%s.png' % (folder_name, traj_idx), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

def draw_3D_trajectory(path, trajectories, folder_name, idx_range=None, matplotlib_plot=False):

    if idx_range != None:
        trajectories = {i: trajectories[i] for i in idx_range}

    for traj_idx in trajectories:
        # if N != None:
        #     numb_list = np.random.choice(list(trajectories.keys()), N, replace=False)
        #     if traj_idx not in numb_list:
        #         continue
        traj = trajectories[traj_idx]
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]
        colors = np.arange(traj.shape[0]) / traj.shape[0]
        color_list = []

        for i in range(0, traj.shape[0]):
            color_list.append(plt.cm.jet(colors[i]))

        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            marker=dict(
                size=4,
                color=color_list,
            ),
            line=dict(
                color=color_list,
                width=2
            )
        ))

        fig.update_layout(
            # width=1024,
            # height=1024,
            scene=dict(
                aspectmode='data',  # this string can be 'data', 'cube', 'auto', 'manual'
                # if 'manual', set a custom aspectratio as below
                # aspectratio=dict(x=1, y=1, z=0.95,)
            ))
        if not os.path.isdir(path + '%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + '%s/' % folder_name)
        fig.write_html(path + '%s/interactive_%s.html' % (folder_name, traj_idx))

    if matplotlib_plot == True:
        min_ = []
        max_ = []
        for traj_idx in trajectories:
            min_b = []
            max_b = []
            for i in trajectories[traj_idx]:
                min_b.append(min(i))
                max_b.append(max(i))
            min_.append(min(min_b))
            max_.append(max(max_b))
        minimum = min(min_)
        maximum = max(max_)

        for traj_idx in trajectories:
            plt.figure()
            traj = trajectories[traj_idx]
            traj_origin = traj - np.tile(traj[0], (traj.shape[0], 1))

            ax = plt.axes(projection='3d')
            x = traj_origin[:, 0]
            y = traj_origin[:, 1]
            z = traj_origin[:, 2]
            c = range(traj.shape[0])
            colors = np.arange(traj.shape[0]) / traj.shape[0]

            ax.scatter(x, y, z, c=c, cmap='jet', s=4)

            for i in range(1, traj.shape[0]):
                ax.plot(x[i - 1:i + 1], y[i - 1:i + 1], z[i - 1:i + 1], c=plt.cm.jet(colors[i]))

            ax.grid(False)
            #         ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide YZ Plane
            #         ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide XZ Plane
            #         ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide XY Plane
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.set_ticklabels([])
                axis._axinfo['axisline']['linewidth'] = 1
                axis._axinfo['axisline']['color'] = "b"
                axis._axinfo['grid']['linewidth'] = 0.5
                axis._axinfo['grid']['linestyle'] = "--"
                axis._axinfo['grid']['color'] = "#d1d1d1"
                axis._axinfo['tick']['inward_factor'] = 0.0
                axis._axinfo['tick']['outward_factor'] = 0.0
                axis.set_pane_color((1, 1, 1))
            ax.plot([minimum/2, maximum/2], [0, 0], [0, 0], color='black')
            ax.plot([0, 0], [minimum/2, maximum/2], [0, 0], color='black')
            ax.plot([0, 0], [0, 0], [minimum/2, maximum/2], color='black')
            ax.axis('off')
            ax.set_xlim3d(minimum/2, maximum/2)
            ax.set_ylim3d(minimum/2, maximum/2)
            ax.set_zlim3d(minimum/2, maximum/2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('idx: %s' % (traj_idx))

            if not os.path.isdir(path + 'matplotlib_%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(path + 'matplotlib_%s/' % folder_name)
            plt.savefig(path + 'matplotlib_%s/%s.png' % (folder_name, traj_idx), dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()

def draw_3D_trajectory_labels(path, trajectories, folder_name, label, idx_range=None):

    if idx_range != None:
        trajectories = {i: trajectories[i] for i in idx_range}

    min_ = []
    max_ = []
    for traj_idx in trajectories:
        min_b = []
        max_b = []
        for i in trajectories[traj_idx]:
            min_b.append(min(i))
            max_b.append(max(i))
        min_.append(min(min_b))
        max_.append(max(max_b))
    minimum = min(min_)
    maximum = max(max_)

    for traj_idx in trajectories:
        plt.figure()
        traj = trajectories[traj_idx]
        traj_origin = traj - np.tile(traj[0], (traj.shape[0], 1))

        ax = plt.axes(projection='3d')
        x = traj_origin[:, 0]
        y = traj_origin[:, 1]
        z = traj_origin[:, 2]
        c = range(traj.shape[0])
        colors = np.arange(traj.shape[0]) / traj.shape[0]

        ax.scatter(x, y, z, c=c, cmap='jet', s=4)

        for i in range(1, traj.shape[0]):
            ax.plot(x[i - 1:i + 1], y[i - 1:i + 1], z[i - 1:i + 1], c=plt.cm.jet(colors[i]))

        ax.grid(False)
        #         ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide YZ Plane
        #         ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide XZ Plane
        #         ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide XY Plane
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            axis._axinfo['axisline']['linewidth'] = 1
            axis._axinfo['axisline']['color'] = "b"
            axis._axinfo['grid']['linewidth'] = 0.5
            axis._axinfo['grid']['linestyle'] = "--"
            axis._axinfo['grid']['color'] = "#d1d1d1"
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0
            axis.set_pane_color((1, 1, 1))
        ax.plot([minimum/2, maximum/2], [0, 0], [0, 0], color='black')
        ax.plot([0, 0], [minimum/2, maximum/2], [0, 0], color='black')
        ax.plot([0, 0], [0, 0], [minimum/2, maximum/2], color='black')
        ax.axis('off')
        ax.set_xlim3d(minimum/2, maximum/2)
        ax.set_ylim3d(minimum/2, maximum/2)
        ax.set_zlim3d(minimum/2, maximum/2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('%s, idx: %s' % (label[traj_idx],traj_idx))

        if not os.path.isdir(path + '%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + '%s/' % folder_name)
        plt.savefig(path + '%s/%s_%s.png' % (folder_name, label[traj_idx],traj_idx), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()


def draw_3D_trajectory_one_figure(df_duration, path, folder_name, duration=20, n_examples=30, label_name='kmeans',
                                  feature_name=['Position X', 'Position Y', 'Position Z'], lim=300):


    # min_ = []
    # max_ = []
    # for traj_idx in trajectories:
    #     min_b = []
    #     max_b = []
    #     for i in trajectories[traj_idx]:
    #         min_b.append(min(i))
    #         max_b.append(max(i))
    #     min_.append(min(min_b))
    #     max_.append(max(max_b))
    # minimum = min(min_)
    # maximum = max(max_)
    maximum=lim
    minimum=0

    n_colors = n_examples
    #cm = plt.cm.get_cmap(name='jet')
    cm = cmc.romaO
    currentColors = [cm(1. * i / n_colors) for i in range(n_colors)]

    for cluster in np.unique(df_duration[label_name]):
        df_part = df_duration[df_duration[label_name]==cluster].reset_index(drop=True)
        n_trajs = df_part.shape[0]//duration
        print('cluster %s: %s trajectories'%(cluster, n_trajs))
        font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 1.5

        #fig, ax = plt.subplots()
        ax = plt.figure().add_subplot(projection='3d')
        #ax = plt.axes(projection='3d')
        ax.set_aspect('equal')
        if n_trajs >= n_examples:
            random_traj_idxs = np.random.choice(range(0, n_trajs), size=n_examples, replace=False)  # Replace=False -> No redundant value

        elif n_trajs < n_examples:
            raise ValueError('%s has less than %s trajectories'%(cluster, n_examples))

        for i, traj_idx in enumerate(random_traj_idxs):
            traj = df_part[duration*traj_idx: duration*(traj_idx+1)][feature_name].values
            traj_center_to_origin = traj - np.tile(traj[0], (traj.shape[0], 1))
            #ax.scatter(x, y, z, c=c, cmap='jet', s=4)
            ax.plot(traj_center_to_origin[:, 0], traj_center_to_origin[:, 1], traj_center_to_origin[:, 2], color=currentColors[i])
            #ax.plot(x[i - 1:i + 1], y[i - 1:i + 1], z[i - 1:i + 1], c=plt.cm.jet(colors[i]))
        ax.grid(False)
        #         ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide YZ Plane
        #         ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide XZ Plane
        #         ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # Hide XY Plane
        # for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        #     axis.set_ticklabels([])
        #     axis._axinfo['axisline']['linewidth'] = 1
        #     axis._axinfo['axisline']['color'] = "b"
        #     axis._axinfo['grid']['linewidth'] = 0.5
        #     axis._axinfo['grid']['linestyle'] = "--"
        #     axis._axinfo['grid']['color'] = "#d1d1d1"
        #     axis._axinfo['tick']['inward_factor'] = 0.0
        #     axis._axinfo['tick']['outward_factor'] = 0.0
        #     axis.set_pane_color((1, 1, 1))
        ax.plot([minimum/2, maximum/2], [0, 0], [0, 0], color='gray', alpha=0.4)
        ax.plot([0, 0], [minimum/2, maximum/2], [0, 0], color='gray', alpha=0.4)
        ax.plot([0, 0], [0, 0], [minimum/2, maximum/2], color='gray', alpha=0.4)

        ax.set_xlim3d(minimum/2, maximum/2)
        ax.set_ylim3d(minimum/2, maximum/2)
        ax.set_zlim3d(minimum/2, maximum/2)

        ax.text(1.15*maximum/2, 0, 0, 'x', fontsize=15, color='gray', alpha=0.4)
        ax.text(0, 1.1 * maximum / 2, 0, 'y', fontsize=15, color='gray', alpha=0.4)
        ax.text(0, 0, 1.1 * maximum / 2, 'z', fontsize=15, color='gray', alpha=0.4)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        ax.view_init(elev=20., azim=32)
        #ax.set_title('%s, idx: %s' % (label[traj_idx],traj_idx))
        ax.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        ax.axis('off')
        if not os.path.isdir(path + '%s/' % (folder_name)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + '%s/' % (folder_name))
        plt.savefig(path + '%s/%s.png' % (folder_name, cluster), dpi=300, bbox_inches='tight')

        if not os.path.isdir(path + 'svg/%s/'%(folder_name)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/%s/'%(folder_name))
        plt.savefig(path + 'svg/%s/%s.svg' % (folder_name, cluster), bbox_inches='tight')
        plt.clf()
        plt.close()

def draw_3D_trajectory_one_figure_GC(df_duration, path, folder_name, duration, n_examples=30, label_name='kmeans',
                                  feature_name=['Position X', 'Position Y', 'Position Z'], lim=300):

    # min_ = []
    # max_ = []
    # for traj_idx in trajectories:
    #     min_b = []
    #     max_b = []
    #     for i in trajectories[traj_idx]:
    #         min_b.append(min(i))
    #         max_b.append(max(i))
    #     min_.append(min(min_b))
    #     max_.append(max(max_b))
    # minimum = min(min_)
    # maximum = max(max_)
    maximum=lim
    minimum=0

    n_colors = np.unique(df_duration[label_name]).size
    #cm = plt.cm.get_cmap(name='jet')
    cm = cmc.batlow
    currentColors = [cm(1. * i / (n_colors-1)) for i in range(n_colors)]

    for idx, cluster in enumerate(np.unique(df_duration[label_name])):
        df_part = df_duration[df_duration[label_name]==cluster].reset_index(drop=True)
        n_trajs = df_part.shape[0]//duration
        print('cluster %s: %s trajectories'%(cluster, n_trajs))
        font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 1.5

        #fig, ax = plt.subplots()
        ax = plt.figure().add_subplot(projection='3d')
        #ax = plt.axes(projection='3d')
        ax.set_aspect('equal')
        if n_trajs >= n_examples:
            random_traj_idxs = np.random.choice(range(0, n_trajs), size=n_examples, replace=False)  # Replace=False -> No redundant value

        elif n_trajs < n_examples:
            raise ValueError('%s has less than %s trajectories'%(cluster, n_examples))

        color = currentColors[idx]
        for i, traj_idx in enumerate(random_traj_idxs):
            traj = df_part[duration*traj_idx: duration*(traj_idx+1)][feature_name].values
            traj_center_to_origin = traj - np.tile(traj[0], (traj.shape[0], 1))
            ax.plot(traj_center_to_origin[:, 0], traj_center_to_origin[:, 1], traj_center_to_origin[:, 2], color=color, linewidth=5)
        ax.grid(False)
        ax.plot([minimum/2, maximum/2], [0, 0], [0, 0], color='gray', alpha=0.3)
        ax.plot([0, 0], [minimum/2, maximum/2], [0, 0], color='gray', alpha=0.3)
        ax.plot([0, 0], [0, 0], [minimum/2, maximum/2], color='gray', alpha=0.3)

        ax.set_xlim3d(minimum/2, maximum/2)
        ax.set_ylim3d(minimum/2, maximum/2)
        ax.set_zlim3d(minimum/2, maximum/2)

        ax.text(1.15*maximum/2, 0, 0, 'x', fontsize=15, color='gray', alpha=0.3)
        ax.text(0, 1.1 * maximum / 2, 0, 'y', fontsize=15, color='gray', alpha=0.3)
        ax.text(0, 0, 1.1 * maximum / 2, 'z', fontsize=15, color='gray', alpha=0.3)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        ax.view_init(elev=20., azim=32)
        #ax.set_title('%s, idx: %s' % (label[traj_idx],traj_idx))
        ax.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        ax.axis('off')
        if not os.path.isdir(path + '%s/' % (folder_name)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + '%s/' % (folder_name))
        plt.savefig(path + '%s/%s.png' % (folder_name, cluster), dpi=300, bbox_inches='tight')

        if not os.path.isdir(path + 'svg/%s/'%(folder_name)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/%s/'%(folder_name))
        plt.savefig(path + 'svg/%s/%s.svg' % (folder_name, cluster), bbox_inches='tight')
        plt.clf()
        plt.close()

        # traj_idxs = {0: 201, 1: 155, 2: 723, 3: 287, 4: 208, 5: 144, 6: 74, 7: 161, 8: 122}
        # traj_idx = traj_idxs[cluster]
        # color = currentColors[idx]
        # traj = df_part.loc[duration*traj_idx: duration*(traj_idx+1)][feature_name].values
        # traj_center_to_origin = traj - np.tile(traj[0], (traj.shape[0], 1))
        # ax.plot(traj_center_to_origin[:, 0], traj_center_to_origin[:, 1], traj_center_to_origin[:, 2], color=color, linewidth=3)
        # ax.grid(False)
        # ax.plot([minimum/2, maximum/2], [0, 0], [0, 0], color='gray', alpha=0.3)
        # ax.plot([0, 0], [minimum/2, maximum/2], [0, 0], color='gray', alpha=0.3)
        # ax.plot([0, 0], [0, 0], [minimum/2, maximum/2], color='gray', alpha=0.3)
        #
        # ax.set_xlim3d(minimum/2, maximum/2)
        # ax.set_ylim3d(minimum/2, maximum/2)
        # ax.set_zlim3d(minimum/2, maximum/2)
        #
        # ax.text(1.15*maximum/2, 0, 0, 'x', fontsize=15, color='gray', alpha=0.3)
        # ax.text(0, 1.1 * maximum / 2, 0, 'y', fontsize=15, color='gray', alpha=0.3)
        # ax.text(0, 0, 1.1 * maximum / 2, 'z', fontsize=15, color='gray', alpha=0.3)
        # # ax.set_xlabel('X')
        # # ax.set_ylabel('Y')
        # # ax.set_zlabel('Z')
        #
        # ax.view_init(elev=20., azim=32)
        # #ax.set_title('%s, idx: %s' % (label[traj_idx],traj_idx))
        # ax.tick_params(left=False, right=False, labelleft=False,
        #                 labelbottom=False, bottom=False)
        # ax.axis('off')
        # if not os.path.isdir(path + '%s/' % (folder_name)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        #     os.makedirs(path + '%s/' % (folder_name))
        # plt.savefig(path + '%s/%s.png' % (folder_name, cluster), dpi=300, bbox_inches='tight')
        #
        # if not os.path.isdir(path + 'svg/%s/'%(folder_name)):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        #     os.makedirs(path + 'svg/%s/'%(folder_name))
        # plt.savefig(path + 'svg/%s/%s.svg' % (folder_name, cluster), bbox_inches='tight')
        # plt.clf()
        # plt.close()

def draw_feature_bar_graph_interactive(df, path, feature_list, condition_name, label_name, test, box_pairs= [ [0,1], [1,2], [0,2] ]):
    for feature_name in feature_list:
        fig = px.box(
            data_frame=df,
            x=condition_name,
            y=feature_name,
            color=condition_name,
            points='all',
            color_discrete_sequence=px.colors.qualitative.Light24,
            template='plotly_white',
            # ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none
            # symbol = 'label',
            # symbol_map = {'Control':0,'Clone A':1,'Clone B':2, 'Clone C':3},
            # title = 'Clone 1-1 Morphology Space',
            # labels = {'Type':'cell type',
            #           'interaction_sum':'sum of interaction',
            #           },
            hover_data={condition_name: True, label_name: True},
            hover_name=df.index,

            height=500,
            width=700,
        )

        fig.update_traces(marker=dict(size=2),
                          # line = dict(width=1, color='DarkSlateGrey')) ,
                          # selector=dict(mode='markers')
                          )

        add_p_value_annotation(fig, box_pairs, test = test)


        if not os.path.isdir(path + 'feature_bar_graph/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'feature_bar_graph/')
        fig.write_html(path + 'feature_bar_graph/%s.html' % feature_name)

def draw_feature_bar_graph(df, path, feature_list, condition_name, test_type, box_pairs):
    #test_type = 'Mann-Whitney', 't-test_ind', 't-test_welch', 't-test_paired', 'Wilcoxon', 'Kruskal'
    for feature_name in feature_list:
        plt.figure(figsize=(15, 10))
        ax = sns.boxplot(data=df, x=condition_name, y=feature_name)

        add_stat_annotation(ax, data=df, x=condition_name, y=feature_name, box_pairs=box_pairs,
                            test=test_type, text_format='star', loc='inside', verbose=0)

        if not os.path.isdir(path + 'feature_bar_graph/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'feature_bar_graph/')
        plt.savefig(path + 'feature_bar_graph/%s.png' % feature_name, dpi=300, bbox_inches='tight')

        if not os.path.isdir(path + 'svg/feature_bar_graph/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/feature_bar_graph/')
        plt.savefig(path + 'svg/feature_bar_graph/%s.svg' % feature_name, bbox_inches='tight')

        plt.clf()
        plt.close()

def draw_custom_box_plot(dict_datasets, path, file_name, colors, strip_plot, test, pvalue=True, return_sig=False, figsize=(2,2), vmax=None, vmin=None):
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(data=sorted_vals, palette=colors)

    if strip_plot == True:
        plot_params = {'edgecolor': '0.2', 'linewidth': 1, 'fc': 'none'}
        ax = sns.stripplot(data=sorted_vals, marker='s', s=1.5, **plot_params)
    # marker='s'(square), s = marker size

    # format_figure(ax, title=None, xlabel=None, ylabel=None, despine=True, detick=True)
    # ax.axhline(max_entropy, linestyle='--', linewidth=1, color='red')
    # plt.xticks(plt.xticks()[0], sorted_keys, fontsize=12, fontdict={'weight': 'normal'})
    # ax.set_xticks(plt.xticks()[0], sorted_keys, fontsize=4)
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

    if (vmax != None) & (vmin == None):
        plt.ylim(0, vmax)
    if (vmin != None):
        plt.ylim(vmin, vmax)

    if pvalue==True:
        pairs, p_values, cohen_ds = get_various_statistics(dict_datasets, test=test, return_sig=return_sig)
        plt.title('%s: %s, %s' % (pairs, p_values, cohen_ds), fontsize=4)

    elif pvalue==False:
        pass

    plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()

def draw_custom_bar_plot(dict_datasets, path, file_name, colors, strip_plot, estimator='mean', vmax=None, vmin=None, pvalue=True, return_sig=False, test='mann-whitney', figsize=(2,2)):
    font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    fig, ax = plt.subplots(figsize=figsize)
    sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())

    if estimator=='mean':
        estimator=np.mean
    elif estimator=='median':
        estimator=np.median

    ax=sns.barplot(data=sorted_vals, capsize=0.5, edgecolor='0.2', lw=1, errwidth=1, palette=colors, estimator=estimator)
    if strip_plot == True:
        plot_params={'edgecolor':'0.2', 'linewidth':1, 'fc':'none'}
        ax=sns.stripplot(data=sorted_vals, marker='s', s=1.5, **plot_params)
    # marker='s'(square), s = marker size

    #format_figure(ax, title=None, xlabel=None, ylabel=None, despine=True, detick=True)
    #ax.axhline(max_entropy, linestyle='--', linewidth=1, color='red')
    #plt.xticks(plt.xticks()[0], sorted_keys, fontsize=12, fontdict={'weight': 'normal'})
    #ax.set_xticks(plt.xticks()[0], sorted_keys, fontsize=4)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=1, color='0.2')
    plt.xticks(plt.xticks()[0], sorted_keys, fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2', weight='normal')
    plt.yticks(fontsize=8,  color='0.2', weight='normal')
    #plt.ylabel('%s' % feature_name, fontsize=4)
    # category labels
    if (vmax!=None)&(vmin==None):
        plt.ylim(0, vmax)
    if (vmin!=None):
        plt.ylim(vmin, vmax)

    plt.grid(False)

    if pvalue == True:
        pairs, p_values, cohen_ds = get_various_statistics(dict_datasets, test=test, return_sig=return_sig)
        plt.title('%s: %s, %s' % (pairs, p_values, cohen_ds), fontsize=4)

    elif pvalue == False:
        pass

    plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()

def draw_double_bar_plot(df, path, file_name, condition_name, conditions, category_name, categories, y, other_category=None,
                         other_category_colors=None, estimator='mean', error_type='std', condition_colors=('#888888', '#CC6677'),
                         test='mann-whitney', return_sig=False, figsize=(2,2)):
    bar_width = 0.3  # Width of bars

    scatter1 = []
    scatter2 = []
    statistics1 = []
    statistics2 = []
    error1 = []
    error2 = []

    p_values = []
    cohens_ds = []

    other_categories1 = []
    other_categories2 = []
    for category in categories:
        df_part = df[df[category_name]==category].reset_index(drop=True)
        values1 = df_part[df_part[condition_name]==conditions[0]][y].values
        values2 = df_part[df_part[condition_name]==conditions[1]][y].values
        scatter1.append(values1)
        scatter2.append(values2)

        if other_category != None:
            other_category1 = df_part[df_part[condition_name]==conditions[0]][other_category].values
            other_category2 = df_part[df_part[condition_name] == conditions[1]][other_category].values
            other_categories1.append(other_category1)
            other_categories2.append(other_category2)

        stattest_dataset = {}
        stattest_dataset[conditions[0]] = values1
        stattest_dataset[conditions[1]] = values2
        _, p_value, cohens_d = get_various_statistics(stattest_dataset, test=test, return_sig=return_sig)
        p_values.append(p_value[0])
        cohens_ds.append(cohens_d[0])
        if estimator == 'mean':
            stats1, stats2 = np.mean(values1), np.mean(values2)
        elif estimator == 'median':
            stats1, stats2 = np.median(values1), np.median(values2)
        statistics1.append(stats1)
        statistics2.append(stats2)


        if error_type == 'std':
            err1, err2 = np.std(values1), np.std(values2)
        elif error_type == 'sem':
            err1, err2 = stats.sem(values1), stats.sem(values2)
        elif error_type == 'ci_norm':
            interval1 = stats.norm.interval(confidence=0.95, loc=np.mean(values1), scale=stats.sem(values1))
            interval2 = stats.norm.interval(confidence=0.95, loc=np.mean(values2), scale=stats.sem(values2))
            err1 = np.mean(values1) - interval1[0]
            err2 = np.mean(values2) - interval2[0]
        elif error_type == 'ci_t':
            interval1 = stats.t.interval(confidence=0.95, df=values1.size - 1, loc=np.mean(values1), scale=stats.sem(values1))
            interval2 = stats.t.interval(confidence=0.95, df=values2.size - 1, loc=np.mean(values2), scale=stats.sem(values2))
            err1 = np.mean(values1) - interval1[0]
            err2 = np.mean(values2) - interval2[0]
        error1.append(err1)
        error2.append(err2)

    # X-axis positions
    x = np.arange(len(categories))

    font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - bar_width / 2, statistics1, yerr=error1, capsize=5, error_kw=dict(elinewidth=1, capthick=1,),
                   edgecolor='0.2', lw=1, width=bar_width, label=conditions[0], color=condition_colors[0])
    bars2 = ax.bar(x + bar_width / 2, statistics2, yerr=error2, capsize=5, error_kw=dict(elinewidth=1, capthick=1,),
                   edgecolor='0.2', lw=1, width=bar_width, label=conditions[1], color=condition_colors[1])

    plot_params={'edgecolor':'0.2', 'linewidth':0.5,}

    if other_category != None:

        mapping = {label: idx for idx, label in enumerate(np.unique(df[other_category]))}

        for i in range(len(categories)):
            #unique_values1, unique_idxs1 = np.unique(other_categories1[i], return_inverse=True)
            transformed1 = np.array([mapping.get(x, np.nan) for x in other_categories1[i]]) # 's13' -> 0, 's14' -> 1
            try:
                cmap = ListedColormap(np.array(other_category_colors)[transformed1])
            except:
                cmap=None
            x_pos1 = np.random.normal(x[i] - bar_width / 2, 1/4*bar_width/2, size = scatter1[i].shape[0])
            ax.scatter(x_pos1, scatter1[i], marker='s', s=6,
                       c=transformed1, **plot_params, cmap=cmap)
            # ax.scatter(np.full_like(scatter1[i], x[i] - bar_width / 2), scatter1[i], marker='s', s=6,
            #            c=transformed1, **plot_params, cmap=cmap)

            transformed2 = np.array([mapping.get(x, np.nan) for x in other_categories2[i]])  # 's13' -> 0, 's14' -> 1
            try:
                cmap = ListedColormap(np.array(other_category_colors)[transformed2])
            except:
                cmap=None
            x_pos2 = np.random.normal(x[i] + bar_width / 2, 1/4*bar_width/2, size=scatter2[i].shape[0])
            ax.scatter(x_pos2, scatter2[i], marker='s', s=6,
                       c=transformed2, **plot_params, cmap=cmap)

            # ax.scatter(np.full_like(scatter2[i], x[i] + bar_width / 2), scatter2[i], marker='s', s=6,
            #            c=transformed2, **plot_params, cmap=cmap)

        import matplotlib.patches as mpatches
        color_mapping = {label: color for label, color in zip(mapping.keys(), other_category_colors)}
        # Create custom legend handles (list of patches)
        legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_mapping.items()]

    else:
        for i in range(len(categories)):
            ax.scatter(np.full_like(scatter1[i], x[i] - bar_width / 2), scatter1[i], marker='s', s=6,
                       fc=None, **plot_params, cmap='Set1')
            ax.scatter(np.full_like(scatter2[i], x[i] + bar_width / 2), scatter2[i], marker='s', s=6,
                       fc=None, **plot_params, cmap='Set1')

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=1, color='0.2')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    plt.xticks(fontsize=8, rotation=35, rotation_mode='anchor', ha='right', color='0.2', weight='normal')
    plt.yticks(fontsize=8,  color='0.2', weight='normal')


    ax.set_ylabel('%s'%y, fontsize=8, weight='normal', color='0.2')
    legend1 = ax.legend(frameon=False, prop={'weight': 'normal', 'size': 8}, labelcolor='0.2',loc='best')
    if other_category != None:
        fig.canvas.draw()  # Ensure the figure is updated
        bbox_legend1 = legend1.get_window_extent()  # Get bounding box in display coordinates
        bbox_legend1 = ax.transAxes.inverted().transform(bbox_legend1)  # Convert to axes coordinates
        x0, y0 = bbox_legend1[0]  # Lower-left corner of the first legend
        x1, y1 = bbox_legend1[1]  # High-right corner of the first legend

        ax.legend(handles=legend_patches, frameon=False, prop={'weight': 'normal', 'size': 8}, labelcolor='0.2',loc='best',
                  bbox_to_anchor=(x1+0.5, y1), bbox_transform=ax.transAxes
                  )
        ax.add_artist(legend1)

    plt.title('%s: %s, %s' % (categories, p_values, cohens_ds), fontsize=4)

    plt.grid(False)
    plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()

def draw_custom_violin_plot(dict_datasets, path, file_name, colors, test, pvalue=True, return_sig=False, figsize=(2,2), vmax=None, vmin=None):

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    sorted_keys, sorted_vals = list(dict_datasets.keys()), list(dict_datasets.values())
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.violinplot(data=sorted_vals, palette=colors, linewidth=1, linecolor="0.2", inner="box",
                        inner_kws=dict(box_width=10, whis_width=10, color="0.2"))

    # marker='s'(square), s = marker size

    # format_figure(ax, title=None, xlabel=None, ylabel=None, despine=True, detick=True)
    # ax.axhline(max_entropy, linestyle='--', linewidth=1, color='red')
    # plt.xticks(plt.xticks()[0], sorted_keys, fontsize=12, fontdict={'weight': 'normal'})
    # ax.set_xticks(plt.xticks()[0], sorted_keys, fontsize=4)

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

    if (vmax!=None)&(vmin==None):
        plt.ylim(0, vmax)
    if (vmin!=None):
        plt.ylim(vmin, vmax)

    if pvalue == True:
        pairs, p_values, cohen_ds = get_various_statistics(dict_datasets, test=test, return_sig=return_sig)
        plt.title('%s: %s, %s' % (pairs, p_values, cohen_ds), fontsize=4)

    elif pvalue == False:
        pass

    plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()

def draw_volcano_plot(df_p, path, file_name, z_thresh, dot_size, p_thresh=-np.log(0.05), z_name='AvgZ', p_name='Adj_Logp', feature_name='Feature',
                      text=True, text_z_thresh=3, text_p_thresh=40, figsize=(2,2)):

    def map_color(a):
        AvgZ, Adj_Logp = a

        if abs(AvgZ) < z_thresh or Adj_Logp < p_thresh:
            return 'NoChange'
        return 'Change'

    df_p['color'] = df_p[[z_name, p_name]].apply(map_color, axis=1)

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.scatterplot(data=df_p, x=z_name, y=p_name, hue='color', hue_order=['NoChange', 'Change'], s=dot_size,
                         palette=['gray', 'firebrick'])

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')
    plt.xticks(fontsize=16, color='0.2', weight='normal')
    plt.yticks(fontsize=16, color='0.2', weight='normal')

    ax.set_xlabel('Average Z score', fontsize=16, weight='normal', color='0.2')
    ax.set_ylabel('Adjusted log 10 p-value', fontsize=16, weight='normal', color='0.2')

    ax.axhline(p_thresh, zorder=0, color='0.2', lw=1, ls='--')
    ax.axvline(-z_thresh, zorder=0, color='0.2', lw=1, ls='--')
    ax.axvline(z_thresh, zorder=0, color='0.2', lw=1, ls='--')

    ax.legend().set_visible(False)

    if text == True:
        texts = []

        for i in range(df_p.shape[0]):
            if (abs(df_p[z_name].iloc[i]) > text_z_thresh and df_p[p_name].iloc[i] > p_thresh) or (abs(df_p[z_name].iloc[i]) > z_thresh and df_p[p_name].iloc[i] > text_p_thresh):
                texts.append(
                    plt.text(x=df_p[z_name].iloc[i], y=df_p[p_name].iloc[i], s=df_p[feature_name].iloc[i], fontsize=8,
                             weight='normal', color='0.2'))

        if (df_p['color'] == 'Change').any():  # At least one feature that 'changed'
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='0.2'))

    plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()

    return df_p



def draw_gene_rank_plot(df_p, path, file_name, gene_col, p_col, score_col, figsize=(4, 7), dot_size=7):
    top_10_genes = df_p.head(10)[gene_col].tolist()
    bottom_10_genes = df_p.tail(10)[gene_col].tolist()

    df_p['color'] = 'NoChange'
    df_p.loc[df_p[p_col] <= 0.05, 'color'] = 'Change'
    df_p.loc[df_p[gene_col].isin(top_10_genes + bottom_10_genes), 'color'] = 'Top'

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(data=df_p, x=df_p.index, y=df_p[score_col], hue='color',
                         hue_order=['NoChange', 'Change', 'Top'], s=dot_size,
                         palette=['#000000', '#00BCD4', '#F06293'], alpha=1, edgecolor=None)
    # x_values = np.linspace(len(df_p), 0, len(df_p))
    # ax.scatter(x_values, df_p['EZH2_scores'], alpha=0.7)

    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('0.2')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(width=2, color='0.2')
    plt.xticks(fontsize=16, color='0.2', weight='normal')
    plt.yticks(fontsize=16, color='0.2', weight='normal')
    plt.gca().invert_xaxis()
    ax.set_xlabel('Gene rank', fontsize=16, weight='normal', color='0.2')
    ax.set_ylabel('Z-score', fontsize=16, weight='normal', color='0.2')

    # ax.axvline(-z_thresh, zorder=0, color='0.2', lw=1, ls='--')
    # ax.axvline(z_thresh, zorder=0, color='0.2', lw=1, ls='--')

    ax.legend().set_visible(False)

    text_df = df_p[df_p['color'] == 'Top']
    texts = []
    for row in text_df.iterrows():
        idx, values = row
        texts.append(plt.text(x=idx, y=values[score_col], s=values[gene_col],
                              fontsize=16, weight='normal', color='0.2'))

    adjust_text(texts, expand_text=(1.5, 1.5), arrowprops=dict(arrowstyle='-', color='0.2'))

    plt.savefig(path + '%s.png' % file_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()

    return df_p

def draw_space_feature_magnitude(df, path, feature_list, dot_size, x_name='PC1', y_name='PC2', vmax=None):

    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    if len(feature_list) > 1:
        for feature_name in feature_list:
            cmap = plt.cm.get_cmap('coolwarm')
            fig, ax = plt.subplots(figsize=(2, 2))
            font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 8}
            matplotlib.rc('font', **font)
            matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
            matplotlib.rcParams['lines.linewidth'] = 1

            scatter = ax.scatter(df[x_name], df[y_name], c=df['%s' % feature_name], s=dot_size, cmap=cmap)

            format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            cbar = fig.colorbar(scatter)
            cbar.ax.tick_params(labelsize=6)

            if not os.path.isdir(path + 'feature_magnitude_space/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(path + 'feature_magnitude_space/')
            plt.savefig(path + 'feature_magnitude_space/%s.png' % (feature_name), dpi=300)

            if not os.path.isdir(path + 'svg/feature_magnitude_space/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(path + 'svg/feature_magnitude_space')
            plt.savefig(path + 'svg/feature_magnitude_space/%s.svg' % (feature_name))

            plt.clf()
            plt.close()

    elif len(feature_list) == 1:
        cmap = plt.cm.get_cmap('coolwarm')
        fig, ax = plt.subplots(figsize=(2, 2))
        font = {'family': 'arial',
                'weight': 'normal',
                'size': 8}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
        matplotlib.rcParams['lines.linewidth'] = 1

        scatter = ax.scatter(df[x_name], df[y_name], c=df['%s' % feature_list[0]], s=dot_size, cmap=cmap, vmax=vmax)

        format_figure(ax, title=None, xlabel='UMAP1', ylabel='UMAP2', despine=True, detick=True)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        cbar = fig.colorbar(scatter)
        cbar.ax.tick_params(labelsize=6)

        if not os.path.isdir(path + 'feature_magnitude_space/'):  # Returns Boolean (if folder doesn't exist, False)
            os.makedirs(path + 'feature_magnitude_space/')
        plt.savefig(path + 'feature_magnitude_space/%s.png' % (feature_list[0]), dpi=300)

        if not os.path.isdir(
                path + 'svg/feature_magnitude_space/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(path + 'svg/feature_magnitude_space')
        plt.savefig(path + 'svg/feature_magnitude_space/%s.svg' % (feature_list[0]))

        plt.clf()
        plt.close()


def draw_confusion_matrix(y_pred, y_test, y_names, path, figsize, file_name, vmax, thresh=0.5):

    if y_pred.shape[1] >=3:  # Multi-class classification
        y_class = np.argmax(y_pred, axis=1)
    else:  # Binary classification
        y_class = np.array([1 if prob >= thresh else 0 for prob in np.ravel(y_pred)])
    accuracy = np.sum(y_test == y_class) / y_test.size

    print('average accuracy', accuracy)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_class)
    norm_cm = cm / np.sum(cm, axis=1)[:, np.newaxis]

    import seaborn as sns

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(norm_cm, annot=True, annot_kws={'size': 16, 'weight': 'normal'}, linewidths=0.5, linecolor='black', alpha=0.8, cmap='Blues', vmax=vmax)
    ax.set_xticklabels(pd.unique(y_names), rotation=0, fontsize=16, weight='normal')
    ax.set_yticklabels(pd.unique(y_names), rotation=0, fontsize=16, weight='normal')
    ax.set_xlabel('Predicted', fontsize=16, weight='normal', color='0.2')
    ax.set_ylabel('Truth', fontsize=16, weight='normal', color='0.2')

    plt.savefig(path + '%s.png'%file_name, dpi=300, bbox_inches='tight')

    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % file_name, bbox_inches='tight')
    plt.clf()
    plt.close()

def draw_lineplot_by_custom_ranges(df, path, folder_name, feature_list, condition_name, custsom_range, range_feature, stepsize,
                                   color_list, marker_list, figsize, x_label, estimator='mean', error_type='ci_norm', fill=True,
                                   replace_keys=None, pvalue=False, test='mann-whitney', legend=True, set_zero=False,
                                   side_note=None):
    '''
    error_type:
    standard deviation: measure dispersion of the data
    confidence interval: measure uncertainty of the mean (or other statistics)
        CI from t distribution: for n<=30 sample size
        CI from normal distribution: for large sample size
    '''
    from scipy import stats
    for feature_name in feature_list:
        try:
            mean_dataset = {}
            error_dataset = {}
            p_value_data = {}
            for cell_type in np.unique(df[condition_name]):
                df_part = df[df[condition_name] == cell_type]
                means = []
                errors = []
                valuess = []

                for i in np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize):
                    # if i == custsom_range[1]:
                    #     values = df_part[df_part[range_feature] >= i][feature_name].values
                    # else:
                    values = df_part[(df_part[range_feature] >= i) & (df_part[range_feature] < i + stepsize)][feature_name].values

                    if estimator == 'mean':
                        means.append(np.mean(values))
                    elif estimator == 'median':
                        means.append(np.median(values))
                    # means.append(np.mean(values))
                    #means.append(np.mean(values))
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

                    errors.append(error)
                    valuess.append(values)

                mean_dataset[cell_type] = np.array(means)
                error_dataset[cell_type] = np.array(errors)
                p_value_data[cell_type] = valuess

            if replace_keys != None:
                mean_dataset = {replace_keys.get(k, k):v  for (k,v) in mean_dataset.items() }
                error_dataset = {replace_keys.get(k, k):v  for (k,v) in error_dataset.items() }

            font = {'family': 'arial',
                    'weight': 'normal',
                    'size': 16}
            matplotlib.rc('font', **font)
            matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
            matplotlib.rcParams['lines.linewidth'] = 2

            fig, ax = plt.subplots(figsize=figsize)

            for idx, key in enumerate(mean_dataset):
                sns.lineplot(data=mean_dataset, x=np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize), y=mean_dataset[key],
                             label=key, lw=2.5, marker=marker_list[idx], dashes=False, markersize=8, err_style='bars', color=color_list[idx])

                if fill == True:
                    ax.fill_between(x=np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize),
                                y1=mean_dataset[key]-error_dataset[key], y2=mean_dataset[key]+error_dataset[key],
                                 color=color_list[idx], alpha=0.4)
                else:
                    ax.errorbar(x=np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize), y=mean_dataset[key],
                                yerr=error_dataset[key], color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

                # else:
                #     ax.errorbar(x=np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize), y=mean_dataset[key],
                #                 yerr=error_dataset[key].T, color=color_list[idx], capsize=3, capthick=1, elinewidth=1.5)

            handles, labels = ax.get_legend_handles_labels()


            for axis in ['bottom', 'left']:
                ax.spines[axis].set_linewidth(2)
                ax.spines[axis].set_color('0.2')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.tick_params(width=2, color='0.2')

            ax.set_xlabel('%s'%x_label, fontsize=16, weight='normal', color='0.2')
            if set_zero == True:
                ax.set_ylim(0, )
            plt.xticks(np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize), fontsize=12, color='0.2',
                       weight='normal', )
            plt.yticks(fontsize=16, color='0.2', weight='normal')

            plt.legend(handles=handles, labels=labels, frameon=False, prop={'weight': 'normal', 'size': 12}, labelcolor='0.2',
                       loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
            if legend == False:
                ax.get_legend().remove()

            if side_note is not None and side_note != '':
                ax.text(
                    1.02,
                    0.02,
                    side_note,
                    transform=ax.transAxes,
                    ha='left',
                    va='bottom',
                    fontsize=8,
                    color='0.2',
                )

            if pvalue == True:
                from scipy import stats
                p_values = []
                pairs = []
                for idx, (mt_values, wt_values) in enumerate(zip(p_value_data[list(p_value_data.keys())[0]], p_value_data[list(p_value_data.keys())[1]])):
                    if test == 'mann-whitney':
                        stat_test = stats.mannwhitneyu(mt_values, wt_values)
                    elif test == 't-test':
                        stat_test = stats.ttest_ind(mt_values, wt_values)
                    elif test == 'wilcoxon-ranksum':
                        stat_test = stats.ranksums(mt_values, wt_values)
                    #print(idx, stat_test.pvalue)
                    p_values.append(stat_test.pvalue)
                    pairs.append(idx)

                plt.title('%s' % (p_values), fontsize=4)

            elif pvalue == False:
                pass
            if not os.path.isdir(path + '%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(path + '%s/' % folder_name)
            plt.savefig(path + '%s/%s.png' % (folder_name, feature_name), dpi=300,bbox_inches='tight')

            if not os.path.isdir(path + 'svg/%s/' % folder_name):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(path + 'svg/%s/' % folder_name)
            plt.savefig(path + 'svg/%s/%s.svg' % (folder_name, feature_name), bbox_inches='tight')
            plt.clf()
            plt.close()
        except:
            print('feature %s graph cannot be generated'%feature_name)
            plt.clf()
            plt.close()

def _nice_step(raw_step):
    if not np.isfinite(raw_step) or raw_step <= 0:
        return None

    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10 ** exponent)
    nice_fractions = np.array([1, 2, 2.5, 5, 10])
    nice_fraction = nice_fractions[np.argmin(np.abs(nice_fractions - fraction))]
    return nice_fraction * (10 ** exponent)


def _get_auto_valid_custom_range(df, condition_name, range_feature, feature_name, min_bins=3, max_bins=10,
                                 min_bin_count=2):
    data = df[[condition_name, range_feature, feature_name]].copy()
    data[range_feature] = pd.to_numeric(data[range_feature], errors='coerce')
    data[feature_name] = pd.to_numeric(data[feature_name], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[condition_name, range_feature, feature_name])

    if data.empty:
        return None

    conditions = [c for c in np.unique(data[condition_name]) if pd.notna(c)]
    if len(conditions) == 0:
        return None

    x_min = data[range_feature].min()
    x_max = data[range_feature].max()
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        return None

    candidates = []
    raw_range = x_max - x_min
    for n_bins in range(max_bins, min_bins - 1, -1):
        step = _nice_step(raw_range / n_bins)
        if step is None:
            continue

        start = np.floor(x_min / step) * step
        end = np.ceil(x_max / step) * step
        bin_starts = np.arange(start, end, step)

        valid_bins = []
        for bin_start in bin_starts:
            bin_end = bin_start + step
            bin_valid = True
            for condition in conditions:
                values = data[
                    (data[condition_name] == condition) &
                    (data[range_feature] >= bin_start) &
                    (data[range_feature] < bin_end)
                ][feature_name].values
                if values.size < min_bin_count:
                    bin_valid = False
                    break
            valid_bins.append(bin_valid)

        run_start = None
        for idx, is_valid in enumerate(valid_bins + [False]):
            if is_valid and run_start is None:
                run_start = idx
            elif not is_valid and run_start is not None:
                run_end = idx
                run_len = run_end - run_start
                if run_len >= min_bins:
                    range_start = bin_starts[run_start]
                    range_end = bin_starts[run_end - 1]
                    candidates.append(
                        {
                            'range': (range_start, range_end),
                            'step': step,
                            'n_bins': run_len,
                            'width': range_end - range_start + step,
                        }
                    )
                run_start = None

    if len(candidates) == 0:
        return None

    candidates = sorted(candidates, key=lambda x: (x['n_bins'], x['width']), reverse=True)
    return candidates[0]['range'], candidates[0]['step']


def draw_lineplot_by_auto_ranges(df, path, folder_name, feature_list, condition_name, range_feature,
                                 color_list, marker_list, figsize, x_label, estimator='mean',
                                 error_type='ci_norm', fill=True, replace_keys=None, pvalue=False,
                                 test='mann-whitney', legend=True, set_zero=False, min_bins=3,
                                 max_bins=10, min_bin_count=None):
    '''
    Like draw_lineplot_by_custom_ranges, but chooses an equal-width range and step size per feature.
    The selected bins are the longest contiguous set where every condition has enough finite values;
    range width is used as a tie-breaker. This prevents empty-bin NaNs in the mean, error,
    and p-value calculations.
    '''
    if min_bin_count is None:
        if error_type in ['sem', 'ci_norm', 'ci_t']:
            min_bin_count = 2
        else:
            min_bin_count = 1

    for feature_name in feature_list:
        auto_range = _get_auto_valid_custom_range(
            df=df,
            condition_name=condition_name,
            range_feature=range_feature,
            feature_name=feature_name,
            min_bins=min_bins,
            max_bins=max_bins,
            min_bin_count=min_bin_count,
        )
        if auto_range is None:
            print('feature %s graph cannot be generated: no valid auto range for %s' % (feature_name, range_feature))
            continue

        custsom_range, stepsize = auto_range
        n_conditions = len([c for c in np.unique(df[condition_name].dropna())])
        plot_color_list = list(color_list)
        plot_marker_list = list(marker_list)
        if len(plot_color_list) < n_conditions:
            repeats = int(np.ceil(n_conditions / len(plot_color_list)))
            plot_color_list = (plot_color_list * repeats)[:n_conditions]
        if len(plot_marker_list) < n_conditions:
            repeats = int(np.ceil(n_conditions / len(plot_marker_list)))
            plot_marker_list = (plot_marker_list * repeats)[:n_conditions]

        draw_lineplot_by_custom_ranges(
            df=df,
            path=path,
            folder_name=folder_name,
            feature_list=[feature_name],
            condition_name=condition_name,
            custsom_range=custsom_range,
            range_feature=range_feature,
            stepsize=stepsize,
            color_list=plot_color_list,
            marker_list=plot_marker_list,
            figsize=figsize,
            x_label=x_label,
            estimator=estimator,
            error_type=error_type,
            fill=fill,
            replace_keys=replace_keys,
            pvalue=pvalue,
            test=test,
            legend=legend,
            set_zero=set_zero,
        )


def draw_lineplot_by_auto_ranges_per_video(df, path, folder_name, feature_list, condition_name, range_feature,
                                           color_list, marker_list, figsize, x_label, group_col='video_id',
                                           estimator='mean', error_type='sem', fill=True, replace_keys=None,
                                           pvalue=False, test='mann-whitney', legend=True, set_zero=False,
                                           min_bins=3, max_bins=10, min_bin_count=None,
                                           min_group_bin_count=1):
    '''
    Draw line plots using the same auto-binning logic as draw_lineplot_by_auto_ranges,
    but first collapse cells within each condition/bin/video to one mean. The plotted
    mean and error are therefore across videos/ROIs instead of across individual cells.
    '''
    if group_col not in df.columns:
        print('per-video lineplot skipped: %s column is missing' % group_col)
        return

    if min_bin_count is None:
        min_bin_count = 2

    for feature_name in feature_list:
        required_columns = [condition_name, group_col, range_feature, feature_name]
        missing_columns = [column for column in required_columns if column not in df.columns]
        if len(missing_columns) > 0:
            print('feature %s graph cannot be generated: missing columns %s' % (feature_name, missing_columns))
            continue

        auto_range = _get_auto_valid_custom_range(
            df=df,
            condition_name=condition_name,
            range_feature=range_feature,
            feature_name=feature_name,
            min_bins=min_bins,
            max_bins=max_bins,
            min_bin_count=min_bin_count,
        )
        if auto_range is None:
            print('feature %s graph cannot be generated: no valid auto range for %s' % (feature_name, range_feature))
            continue

        custsom_range, stepsize = auto_range
        data = df[required_columns].copy()
        data[range_feature] = pd.to_numeric(data[range_feature], errors='coerce')
        data[feature_name] = pd.to_numeric(data[feature_name], errors='coerce')
        data = data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[condition_name, group_col, range_feature, feature_name]
        )

        bin_starts = np.arange(custsom_range[0], custsom_range[1] + stepsize, stepsize)
        per_bin_tables = []
        for bin_start in bin_starts:
            bin_end = bin_start + stepsize
            bin_data = data[
                (data[range_feature] >= bin_start) &
                (data[range_feature] < bin_end)
            ].copy()
            if bin_data.empty:
                continue

            group_sizes = (
                bin_data
                .groupby([condition_name, group_col], dropna=False)
                .size()
                .rename('__n_cells')
                .reset_index()
            )
            bin_summary = (
                bin_data
                .groupby([condition_name, group_col], dropna=False)
                .agg(
                    **{
                        range_feature: (range_feature, 'mean'),
                        feature_name: (feature_name, estimator),
                    }
                )
                .reset_index()
                .merge(group_sizes, on=[condition_name, group_col], how='left')
            )
            bin_summary = bin_summary[bin_summary['__n_cells'] >= min_group_bin_count]
            if bin_summary.empty:
                continue

            bin_summary['__bin_start'] = bin_start
            per_bin_tables.append(bin_summary.drop(columns='__n_cells'))

        if len(per_bin_tables) == 0:
            print('feature %s graph cannot be generated: no per-video bins for %s' % (feature_name, range_feature))
            continue

        per_video_df = pd.concat(per_bin_tables, axis=0, ignore_index=True)

        video_count_lines = ['n videos/point']
        for condition in np.unique(per_video_df[condition_name].dropna()):
            condition_counts = []
            for bin_start in bin_starts:
                n_videos = per_video_df[
                    (per_video_df[condition_name] == condition) &
                    (per_video_df['__bin_start'] == bin_start)
                ][group_col].nunique()
                condition_counts.append(str(int(n_videos)))

            condition_label = replace_keys.get(condition, condition) if replace_keys is not None else condition
            video_count_lines.append('%s: %s' % (condition_label, ', '.join(condition_counts)))
        video_count_note = '\n'.join(video_count_lines)

        n_conditions = len([c for c in np.unique(per_video_df[condition_name].dropna())])
        plot_color_list = list(color_list)
        plot_marker_list = list(marker_list)
        if len(plot_color_list) < n_conditions:
            repeats = int(np.ceil(n_conditions / len(plot_color_list)))
            plot_color_list = (plot_color_list * repeats)[:n_conditions]
        if len(plot_marker_list) < n_conditions:
            repeats = int(np.ceil(n_conditions / len(plot_marker_list)))
            plot_marker_list = (plot_marker_list * repeats)[:n_conditions]

        draw_lineplot_by_custom_ranges(
            df=per_video_df,
            path=path,
            folder_name=folder_name,
            feature_list=[feature_name],
            condition_name=condition_name,
            custsom_range=custsom_range,
            range_feature=range_feature,
            stepsize=stepsize,
            color_list=plot_color_list,
            marker_list=plot_marker_list,
            figsize=figsize,
            x_label=x_label,
            estimator=estimator,
            error_type=error_type,
            fill=fill,
            replace_keys=replace_keys,
            pvalue=pvalue,
            test=test,
            legend=legend,
            set_zero=set_zero,
            side_note=video_count_note,
        )

def draw_jointplot(xs, y, df, path, file_name, colors, hue=None, hue_order=None, alpha=0.8, height=6, ratio=5, space=0.2, xlabels=None, ylabel=None, legend=True,
                   fill=True, thresh=0.05, n_contours=5, margin_norm=False, xmin=None, xmax=None, ymin=None, ymax=None):
    """
    -------------------
    Input Parameters
    -------------------
    xs      : (list or str) feature name(s) of data
    y       : (str) feature name of dataframe
    df    : (pandas.DataFrame)
    hue     : (str) semantic variable that is mapped to determine the color of plot elements. Semantic variable that is mapped to determine the color of plot elements.

    height  : (float) size of the figure
    ratio   : (float) ratio of the joint axes height to marginal axes height.
    space   : (float) space between the joint and marginal axes

    xlabels : (list or str) xlabels
    ylabel  : (str) ylabel
    margin_norm : (boolean) if True, kdeplots at marginal axes have same scale.
    """
    ### 1. input check
    linewidth = 1.5
    fontsize = 16

    n_colors = np.unique(df[hue]).shape[0]
    from collections.abc import Iterable
    if isinstance(colors, Iterable):
        cmap = colors
    else:
        cmap = [colors(1. * i / n_colors) for i in range(n_colors)]

    # input type
    assert isinstance(xs, list) or isinstance(xs, str)
    if isinstance(xs, list):
        assert all([isinstance(x, str) for x in xs])
    else:
        xs = [xs]

    if xlabels != None:
        assert isinstance(xlabels, list) or isinstance(xlabels, str)
        if isinstance(xlabels, list):
            assert all([isinstance(xlabel, str) for xlabel in xlabels])
        else:
            xlabels = [xlabels]

    if ylabel != None:
        assert isinstance(ylabel, str)

    if hue != None:
        assert isinstance(hue, str)

    # input data
    assert all([x in df.columns for x in xs])
    assert y in df.columns
    if hue != None:
        assert hue in df.columns

    ### 2. figure
    h_margin = height / (ratio + 1)
    h_joint = height - h_margin

    if isinstance(xs, list):
        n_x = len(xs)
    else:
        n_x = 1

    widths = [h_joint] * n_x + [h_margin]
    heights = [h_margin, h_joint]
    ncols = len(widths)
    nrows = len(heights)

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.5  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig = plt.figure(figsize=(sum(widths), sum(heights)))
    ### 3. gridspec preparation
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows,
                            width_ratios=widths, height_ratios=heights,
                            wspace=space, hspace=space
                            )

    ### 4. setting axes
    axs = {}
    for i in range(ncols * nrows):
        axs[i] = fig.add_subplot(spec[i // ncols, i % ncols])

    ### 5. Main jointplots (scatterplot + kdeplot)
    for i, x in enumerate(xs, ncols):
        sns.kdeplot(x=x, y=y, data=df, hue=hue, alpha=alpha, ax=axs[i], zorder=2, linewidths=linewidth, palette=cmap, hue_order=hue_order,
                    fill=fill, legend=legend, common_norm=False, thresh=thresh, levels=n_contours+1)
        #sns.scatterplot(x=x, y=y, data=df, hue=hue, alpha=0.8, ax=axs[i], zorder=3, legend=legend)
        #plt.xlim(xmin, xmax)
        #plt.ylim(ymin, ymax)
        axs[i].set_xlim(xmin, xmax)
        axs[i].set_ylim(ymin, ymax)

        for axis in ['top', 'bottom', 'left', 'right']:
            axs[i].spines[axis].set_linewidth(linewidth)
            axs[i].spines[axis].set_color('0.2')

        axs[i].tick_params(left=False, right=False, labelleft=False,
                           labelbottom=False, bottom=False)

        axs[i].get_legend_handles_labels()
        # axs[i].legend(frameon=False, prop={'weight': 'normal', 'size': 8}, labelcolor='0.2')

    ### 6. kdeplots at marginal axes
    axs[ncols - 1].axis("off")

    axes_mx = list(range(ncols - 1))
    axes_my = 2 * ncols - 1

    for i, x in zip(axes_mx, xs):
        sns.kdeplot(x=x, data=df, hue=hue, fill=True, ax=axs[i], zorder=2, linewidth=linewidth, palette=cmap, legend=False, hue_order=hue_order,
                    common_norm=False, multiple='layer')
        axs[i].set_xlim(axs[i + ncols].get_xlim())
        axs[i].set_xlabel("")
        axs[i].set_xticklabels([])
        axs[i].spines["left"].set_visible(True)
        axs[i].spines['left'].set_linewidth(linewidth)
        axs[i].spines['left'].set_color('0.2')
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)

        axs[i].tick_params(width=linewidth, color='0.2')
        # axs[i].set_yticklabels(labels=axs[i].get_yticks()[:-1], fontsize=16, color='0.2', weight='normal')

        axs[i].tick_params(left=True, right=False, labelleft=True,
                           labelbottom=False, bottom=False)

    sns.kdeplot(y=y, data=df, hue=hue, fill=True, ax=axs[axes_my], zorder=2, linewidth=linewidth, palette=cmap, legend=False, hue_order=hue_order,
                common_norm=False, multiple='layer')

    axs[axes_my].set_ylim(axs[ncols].get_ylim())
    axs[axes_my].set_ylabel("")
    axs[axes_my].set_yticklabels([])
    axs[axes_my].spines["bottom"].set_visible(True)
    axs[axes_my].spines['bottom'].set_linewidth(linewidth)
    axs[axes_my].spines['bottom'].set_color('0.2')

    axs[axes_my].spines["left"].set_visible(True)
    axs[axes_my].spines['left'].set_linewidth(linewidth)
    axs[axes_my].spines['left'].set_color('0.2')

    axs[axes_my].spines["top"].set_visible(False)
    axs[axes_my].spines["right"].set_visible(False)

    axs[axes_my].tick_params(width=linewidth, color='0.2')
    # axs[axes_my].set_xticklabels(labels=axs[axes_my].get_xticks()[:-1], fontsize=16, color='0.2', weight='normal')
    axs[axes_my].tick_params(left=False, right=False, labelleft=True,
                             labelbottom=True, bottom=True)

    if margin_norm == True:
        hist_range_max = max([axs[m].get_ylim()[-1] for m in axes_mx] + [axs[axes_my].get_xlim()[-1]])
        for i in axes_mx:
            axs[i].set_ylim(0, hist_range_max)
        axs[axes_my].set_xlim(0, hist_range_max)

    ### 7. labels

    axes_j = list(range(ncols, 2 * ncols - 1))

    for i, x in zip(axes_j, xlabels):
        axs[i].set_xlabel(x, fontsize=fontsize, weight='normal', color='0.2', labelpad=5)
        if i == ncols:
            axs[i].set_ylabel(ylabel, fontsize=fontsize, weight='normal', color='0.2', labelpad=5)

    axs[0].set_ylabel("")
    axs[2 * ncols - 1].set_xlabel("")

    # fig.align_ylabels([axs[0], axs[ncols]])
    # fig.align_xlabels([axs[x] for x in range(ncols, 2 * ncols)])
    plt.tight_layout()

    plt.savefig(path + '/%s.png' % (file_name), dpi=300)
    if not os.path.isdir(path + 'svg/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'svg/')
    plt.savefig(path + 'svg/%s.svg' % (file_name))
    plt.clf()
    plt.close()

    return fig, axs


def draw_diff_arrow_scatter(df, path, file_name, condition_name, diff_condition_name, ind_name, ref, colors, dot_size, x_name, y_name, xlabel, ylabel):

    df_ref = df[df[diff_condition_name] == ref].reset_index(drop=True)

    n_colors = np.unique(df[condition_name]).shape[0]
    from collections.abc import Iterable
    if isinstance(colors, Iterable):
        cmap = ListedColormap(colors[:pd.unique(df[condition_name]).shape[0]])
    else:
        cmap = [colors(1. * i / n_colors) for i in range(n_colors)]

    # from collections.abc import Iterable
    # if isinstance(colors, Iterable):
    #     cmap = ListedColormap(colors[:pd.unique(df[condition_name]).shape[0]])
    # else:
    #     cmap = colors

    xmin = math.floor(df[x_name].min()) - 1
    xmax = math.ceil(df[x_name].max()) + 1
    ymin = math.floor(df[y_name].min()) - 1
    ymax = math.ceil(df[y_name].max()) + 1

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 8}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 0.25  # Visually good to have font size : line width = 8 : 0.25
    matplotlib.rcParams['lines.linewidth'] = 1

    for cond in np.unique(df[diff_condition_name]):
        if cond == ref:
            continue
        fig, ax = plt.subplots(figsize=(2, 2))

        ax.scatter(df_ref[x_name], df_ref[y_name],
                   c=df_ref[condition_name].replace(list(np.unique(df_ref[condition_name])),
                                                    [i for i in range(np.unique(df_ref[condition_name]).shape[0])]),
                   # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
                   s=dot_size, label=df_ref[condition_name], alpha=0.5,  # linestyle='dotted',
                   cmap=cmap)

        df_part = df[df[diff_condition_name] == cond].reset_index(drop=True)
        scatter = ax.scatter(df_part[x_name], df_part[y_name],
                             c=df_part[condition_name].replace(list(np.unique(df_part[condition_name])),
                                                               [i for i in
                                                                range(np.unique(df_part[condition_name]).shape[0])]),
                             # replace 'wt B-cell', 'mt B-cell', 'T-cell' with 0, 1, 2 respectively
                             s=dot_size, label=df_part[condition_name],
                             cmap=cmap)

        for ind in np.unique(df_part[ind_name]):  # For patients within condition
            df_ref_ind = df_ref[df_ref[ind_name] == ind].reset_index(drop=True)
            df_cond_ind = df_part[df_part[ind_name] == ind].reset_index(drop=True)

            if df_ref_ind.shape[0] == 0:  # Test whether reference condition has this patient
                continue

            colors = np.array(colors)
            arrow_color = colors[np.isin(np.unique(df[condition_name]), df_cond_ind[condition_name])][0]
            ax.quiver(df_ref_ind[x_name], df_ref_ind[y_name], df_cond_ind[x_name] - df_ref_ind[x_name],
                      df_cond_ind[y_name] - df_ref_ind[y_name],
                      scale_units='xy', angles='xy', scale=1, color=arrow_color)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        texts = []

        for i in range(df.shape[0]):
            texts.append(
                plt.text(x=df[x_name].iloc[i], y=df[y_name].iloc[i], s=df[ind_name].iloc[i],
                         fontsize=1.5, weight='normal', color='0.2'))
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='0.2'))

        format_figure(ax, title=None, xlabel=xlabel, ylabel=ylabel, despine=True, detick=True)
        handles, labels = scatter.legend_elements(num=None)
        plt.legend(handles=handles, labels=list(np.unique(df_part[condition_name])),
                   bbox_to_anchor=(0.9, 1.1), loc=2, borderaxespad=0.0,
                   fontsize=3, frameon=False, markerscale=0.3)

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


def draw_graph_network(custom_sig, path, file_name, sample_n, resolution, regulation=None, figsize=(10,10), inter_spacing=3, intra_spacing=0.5):
    font = {'family': 'arial',
            'weight': 'normal',}
    matplotlib.rc('font', **font)

    cmap = plt.get_cmap("tab20")
    # cmap = cmc.batlow
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
        print(f'Component {i+1} - Communities detected:', communities)

        # Normalize community labels for consistent RGB coloring
        comm_list = sorted(list(communities))

        color_map = {comm: cmap(j / max(1, len(comm_list)-1)) for j, comm in enumerate(comm_list)}

        # Sample nodes per community
        sampled_nodes = []
        for comm in communities:
            comm_nodes = [n for n in subG.nodes if partition[n] == comm]
            sampled = random.sample(comm_nodes, min(sample_n, len(comm_nodes)))
            sampled_nodes.extend(sampled)

        sampled_subG = subG.subgraph(sampled_nodes).copy()

        # COMMUNITY-AWARE LAYOUT (new)
        n_comm = len(comm_list)
        grid_size = ceil(sqrt(n_comm))  # Layout in grid


        # Assign grid locations
        community_positions = {}
        for idx, comm in enumerate(comm_list):
            row = idx // grid_size
            col = idx % grid_size
            community_positions[comm] = (col * inter_spacing, -row * inter_spacing)

        # Build layout with community offsets
        pos = {}
        for comm in comm_list:
            comm_nodes = [n for n in sampled_subG.nodes if partition[n] == comm]
            sub_pos = nx.spring_layout(sampled_subG.subgraph(comm_nodes), seed=42, k=intra_spacing)
            cx, cy = community_positions[comm]
            for node in comm_nodes:
                x, y = sub_pos[node]
                pos[node] = (x + cx, y + cy)

        # Plotting
        fig, ax = plt.subplots(figsize=figsize)

        # Draw polygons (convex hulls) around each community
        for comm in communities:
            comm_nodes = [n for n in sampled_subG.nodes if partition[n] == comm]
            points = np.array([pos[n] for n in comm_nodes])
            if len(points) >= 3:
                hull = ConvexHull(points)
                polygon = Polygon(points[hull.vertices], closed=True, alpha=0.2,
                                  color=color_map[comm], zorder=0, edgecolor='none')
                ax.add_patch(polygon)

        #nx.draw_networkx_edges(sampled_subG, pos, alpha=0.1)
        # Draw edges with thickness = IoU score
        edges = sampled_subG.edges(data=True)
        for u, v, d in edges:
            weight = d["weight"]
            nx.draw_networkx_edges(sampled_subG, pos, edgelist=[(u, v)],
                                   width=1 + 5 * weight, alpha=0.1)

        # Draw nodes with consistent community-based coloring
        node_colors = [color_map[partition[n]] for n in sampled_subG.nodes]
        nx.draw_networkx_nodes(sampled_subG, pos, node_color=node_colors, node_size=100)

        # Draw node labels
        texts = []
        for node in sampled_subG.nodes:
            x, y = pos[node]
            texts.append( plt.text(x, y, node, fontsize=7, ha='center', va='center') )

            if regulation != None:
                if node in regulation:
                    if regulation[node] == 'UP':
                        plt.scatter(x, y, marker='^', color='green', s=7, zorder=5)
                    elif regulation[node] == 'DOWN':
                        plt.scatter(x, y, marker='v', color='red', s=7, zorder=5)

        #plt.title(f"Component {i + 1}: Sampled Nodes from Each Community")
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='0.2'))
        # Community color legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=color_map[comm], label=f"Community {comm}") for comm in comm_list]

        # Edge thickness legend
        edge_legend_scores = [0.1, 0.5, 0.9]
        edge_lines = [
            Line2D([0], [0], color='gray', lw=1 + 5 * s, alpha=0.5, label=f"IoU = {s:.1f}")
            for s in edge_legend_scores
        ]

        # Combine legends and show
        combined_handles = handles + edge_lines
        ax.legend(handles=combined_handles, bbox_to_anchor=(0.9, 1.1), loc=2, fontsize=5, frameon=False)

        plt.axis('off')
        plt.savefig(path + f'%s_{i + 1}.png'%file_name, dpi=300, bbox_inches='tight')
        if not os.path.isdir(path + 'svg/'):
            os.makedirs(path + 'svg/')
        fig.savefig(path + f'svg/%s_{i + 1}.svg'%file_name, bbox_inches='tight')
        plt.close()
        plt.clf()

        for node in subG.nodes:
            comm = partition[node]
            genes = custom_sig[node]  # assuming custom_sig[node] is your gene list
            if regulation !=None:
                reg = regulation[node]
                record = {
                    'Component': f'Component_{i + 1}',
                    'Pathway': node,
                    'Community_ID': comm,
                    'Number_of_Genes': len(genes),
                    'Genes': ';'.join(genes),
                    'Regulation': reg
                }
            else:
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

    output_excel = os.path.join(path, 'community_detected_pathways.xlsx')
    df_communities.to_excel(output_excel, index=False)

    return df_communities



def draw_bezier_edge(ax, src, dst, color, alpha=0.3, lw=1.0, ctrl_offset=0.6):
    """Draw a cubic Bezier curve between src and dst positions."""
    # Higher ctrl_offset curves more
    x0, y0 = src
    x1, y1 = dst

    # Control points: create smooth horizontal bend
    ctrl1 = (x0 + ctrl_offset, y0)
    ctrl2 = (x1 - ctrl_offset, y1)

    # Separate vertices and codes
    vertices = [src, ctrl1, ctrl2, dst]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    path = Path(vertices, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color, alpha=alpha, lw=lw)
    ax.add_patch(patch)
