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
"""Visualize images and tracks"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from skimage import io
from skimage import io, img_as_ubyte, img_as_float32, segmentation, color, measure, morphology, feature, filters
from scipy.ndimage import distance_transform_edt
from extract_features.tracking import linking
import trackpy as tp
from matplotlib.collections import LineCollection
import cmcrameri.cm as cmc
import scipy

############################### Tracking ###############################

path = r"\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\images\Young\Y244\Control\\"
raw_path = path+'/Tiff Files/'
mask_path = path+'/CellProfiler Segmented Masks/'

save_path = r'\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\analysis\Visualization\Y244-Control\\'

files = next(os.walk(raw_path))[2]
bool_list = ['tif' in ele for ele in files]
raw_files = np.array(files)[np.array(bool_list)]
raw_files.sort()

files = next(os.walk(mask_path))[2]
bool_list = ['tif' in ele for ele in files]
mask_files = np.array(files)[np.array(bool_list)]
mask_files.sort()

binary = True

df_cell = pd.DataFrame()
for t, (raw_file, mask_file) in tqdm(enumerate( zip(raw_files, mask_files) )):
    #raw_img = io.imread(raw_path + raw_file)
    mask_img = io.imread(mask_path + mask_file)[500:, :]

    if binary == True:  # Apply watershed to the binary mask
        distance = distance_transform_edt(mask_img)  # Distance transformation by euclidean distance
        # (Compute shortest distance from non-zero(foreground) to zero(background)
        local_max_coords = feature.peak_local_max(distance, min_distance=8)  # Coords for maximum distance
        local_max_mask = np.zeros(distance.shape, dtype=bool)
        local_max_mask[tuple(local_max_coords.T)] = True
        markers = measure.label(local_max_mask)
        segmented = segmentation.watershed(-distance, markers, mask=mask_img)

    elif binary == False:
        segmented = mask_img

    property_list = ['label', 'area', 'perimeter', 'convex_area', 'solidity', 'eccentricity', 'equivalent_diameter',
                     'extent', 'major_axis_length', 'minor_axis_length', 'orientation', 'centroid']

    properties = measure.regionprops_table(segmented, properties=['label', 'centroid'])
    df_cell_temp = pd.DataFrame(properties)
    df_cell_temp['x'] = df_cell_temp['centroid-1']
    df_cell_temp['y'] = df_cell_temp['centroid-0']

    df_cell_temp['frame'] = t
    df_cell_temp = df_cell_temp.drop(['centroid-1', 'centroid-0'], axis=1)
    df_cell = pd.concat([df_cell, df_cell_temp], axis=0)

df_cell = df_cell.reset_index(drop=True)

pred = tp.predict.NearestVelocityPredict()
df_linked = pred.link_df(df_cell, search_range=50, memory=0, adaptive_stop=12, adaptive_step=0.9)

############################### Visualize Trajectory + Mask ###############################
if not os.path.isdir(save_path + 'raw image/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(save_path + 'raw image/')

if not os.path.isdir(save_path + 'trajectory mask overlay/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(save_path + 'trajectory mask overlay/')

df_cumulative = pd.DataFrame()
for t in tqdm( range(0, raw_files.shape[0]) ):
    mask_file = mask_files[t]
    raw_file = raw_files[t]

    raw_img = io.imread(raw_path + raw_file)[500:, :]
    mask_img = io.imread(mask_path + mask_file)[500:, :]

    row, col = mask_img.shape
    cell_temp_t0 = df_linked.groupby(['frame']).get_group(t).reset_index(drop=True)
    df_cumulative = pd.concat([df_cumulative, cell_temp_t0], axis=0).reset_index(drop=True)

    mask = df_cumulative['particle'].isin(cell_temp_t0['particle'])  # Get the particles that exist in this frame t
    df_cumulative_partial = df_cumulative[mask]

    label_data = df_cumulative_partial.groupby(['particle']).apply(lambda x: x.name)

    fig, ax = plt.subplots()
    plt.imshow(raw_img, extent=[0, col, 0, row], origin='lower')
    plt.axis('off')
    plt.savefig(save_path + 'raw image/%s.png' % t, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    plt.imshow(mask_img, cmap='gray', extent=[0, col, 0, row], origin='lower')
    for traj_idx in range(0, label_data.shape[0]):  # For each cell trajectory(time 1~t)
        #traj_idx = 300
        traj_data_temp = df_cumulative_partial.groupby(['particle']).get_group(label_data.iloc[traj_idx]).copy().reset_index(drop=True)
        if traj_data_temp.shape[0] >= 2:

            x = traj_data_temp['x'].values
            y = traj_data_temp['y'].values

            n_colors = x.shape[0]
            cm = cmc.romaO_r
            currentColors = [cm(1. * i / n_colors) for i in range(n_colors)]

            #ax.plot(x, y,  '-', color=currentColors[t], linewidth=0.5,)

            xy = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.hstack([xy[:-1], xy[1:]])  # (x0, y0), (x1, y1), ... (xt, yt) form
            lc = LineCollection(segments, colors=currentColors, linewidths=0.5)
            ax.add_collection(lc)
            ax.set_aspect('equal', adjustable='datalim')  # adjustable = 'box', 'datalim'

    plt.axis('off')
    if t!=0:
        plt.savefig(save_path+'trajectory mask overlay/%s.png'%t, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.clf()
        plt.close()


############################### Visualize Local Density Map ###############################
if not os.path.isdir(save_path + 'density map/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(save_path + 'density map/')

kernel = np.full(shape=(32, 32), fill_value=1)

for t in tqdm( range(0, mask_files.shape[0]) ):
    mask_file = mask_files[t]

    mask_img = io.imread(mask_path + mask_file)[500:, :]
    row, col = mask_img.shape

    conv_img = scipy.ndimage.convolve(mask_img, kernel)
    g_img = filters.gaussian(conv_img, sigma=100, preserve_range=True)

    fig, ax = plt.subplots()
    plt.imshow(g_img, cmap='turbo', extent=[0, col, 0, row], origin='lower')
    plt.imshow(mask_img, cmap='gray', extent=[0, col, 0, row], origin='lower', alpha=0.25)
    plt.axis('off')
    # plt.show()

    plt.savefig(save_path + 'density map/%s.png' % t, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.clf()
    plt.close()

############################### Visualize Density + Trajectory + Mask ###############################
if not os.path.isdir(save_path + 'trajectory density map overlay/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(save_path + 'trajectory density map overlay/')

kernel = np.full(shape=(32, 32), fill_value=1)

df_cumulative = pd.DataFrame()
for t in tqdm( range(0, raw_files.shape[0]) ):
    mask_file = mask_files[t]
    #raw_file = raw_files[t]
    #raw_img = io.imread(raw_path + raw_file)
    mask_img = io.imread(mask_path + mask_file)[500:, :]

    row, col = mask_img.shape
    cell_temp_t0 = df_linked.groupby(['frame']).get_group(t).reset_index(drop=True)
    df_cumulative = pd.concat([df_cumulative, cell_temp_t0], axis=0).reset_index(drop=True)

    mask = df_cumulative['particle'].isin(cell_temp_t0['particle'])  # Get the particles that exist in this frame t
    df_cumulative_partial = df_cumulative[mask]

    label_data = df_cumulative_partial.groupby(['particle']).apply(lambda x: x.name)


    conv_img = scipy.ndimage.convolve(mask_img, kernel)
    g_img = filters.gaussian(conv_img, sigma=100, preserve_range=True)

    fig, ax = plt.subplots()
    plt.imshow(g_img, cmap='turbo', extent=[0, col, 0, row], origin='lower')
    plt.imshow(mask_img, cmap='gray', extent=[0, col, 0, row], origin='lower', alpha=0.25)

    for traj_idx in range(0, label_data.shape[0]):  # For each cell trajectory(time 1~t)
        #traj_idx = 300
        traj_data_temp = df_cumulative_partial.groupby(['particle']).get_group(label_data.iloc[traj_idx]).copy().reset_index(drop=True)
        if traj_data_temp.shape[0] >= 2:

            x = traj_data_temp['x'].values
            y = traj_data_temp['y'].values

            n_colors = x.shape[0]
            cm = cmc.grayC
            currentColors = [cm(1. * i / n_colors) for i in range(n_colors)]

            #ax.plot(x, y,  '-', color=currentColors[t], linewidth=0.5,)

            xy = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.hstack([xy[:-1], xy[1:]])  # (x0, y0), (x1, y1), ... (xt, yt) form
            lc = LineCollection(segments, colors=currentColors, linewidths=0.5)
            ax.add_collection(lc)
            ax.set_aspect('equal', adjustable='datalim')  # adjustable = 'box', 'datalim'

    plt.axis('off')
    if t!=0:
        plt.savefig(save_path+'trajectory density map overlay/%s.png'%t, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.clf()
        plt.close()