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
"""Preprocess FDC mask for GC dynamics"""

import os
import numpy as np
from skimage import io
import napari
import skimage
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
from utils.img_utils import *
import scipy
import seaborn as sns
import pyclesperanto_prototype as cle

#um_per_pixel = 230.9/320  # For Exp1
um_per_pixel = 230.9/256
um_per_zsice = 3


############################### Read FDC masks ###############################

#path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Raw images\Exp3\\'
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Raw images\mLT\norm_img\20240727f-FDC_masks\\'

files = next(os.walk(path))[2]
bool_list = ['tif' in ele for ele in files]
files = np.array(files)[np.array(bool_list)]
files.sort()
files[-1]

imgs = get_hyperstack(path=path, files=files, order='zt', n_zslices=30, n_frames=81)

# viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(imgs,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap='gray', opacity=1)

############################### Set FDC zones (DZ, LZ, FDC core) ###############################

kernel = np.full(shape=(6, 6, 3), fill_value=1)

density_map = []
for t, img in tqdm(enumerate(imgs)):
    conv_img = scipy.ndimage.convolve(img, kernel)
    g_img = skimage.filters.gaussian(conv_img, sigma=10, preserve_range=True)
    #g_img = g_img/np.max(g_img)
    density_map.append(g_img)

    # z, r, w = g_img.shape
    # zone = np.empty(shape=(z, r, w))
    # vmin, vmax = np.quantile(g_img, q=(0.7, 0.98))  # Lower 70% are DZ, Upper 5% are dLZ

    #vmin = 3350  # Exp1-1
    #vmin = 4000  # Exp1-2
    #vmin = 6000  # Exp1-7, Exp2-2
    #vmin = 8000  # Exp2-4, Exp5-20240719b
    #vmin = 9000  # Exp2-6, Exp2-8
    #vmin = 11000  # Exp2-9 , All Exp3, All CD40L, IgG-20240128, IgG-20240212b

    #vmax = 19000  # Exp2-8
    #vmax = 20982  # All Exp1, Exp2-6, Exp3-3, Exp3-4, Exp5-20240719b
    #vmax = 22000  # Exp2-2, Exp2-9, Exp3-1, Exp3-7, Exp3-8, CD40L-20240204b, CD40L-20240211a, CD40L-20240211b, CD40L-20240212, IgG-20240128, IgG-20240212b
    #vmax = 24000 # Exp2-4, Exp3-6, Exp3-9, CD40L-20240128a, CD40L-20240128b,

    # dark_zone = g_img <= vmin
    # light_zone = (g_img > vmin) & (g_img < vmax)
    # FDC_core = g_img >= vmax
    #
    # zone[dark_zone] = 0
    # zone[light_zone] = 1
    # zone[FDC_core] = 2
    # zones.append(zone)

density_map = np.array(density_map)
density_map_smoothed = skimage.filters.gaussian(density_map, sigma=10, preserve_range=True)

# t, z, r, w = density_map_smoothed.shape
# zones = np.empty(shape=(t, z, r, w))
#
# vmin, vmax = np.quantile(density_map_smoothed, q=(1-sLZ_portion, 1-0.1*sLZ_portion))  # Lower 50% are DZ, Upper 5% are dLZ
# dark_zone = density_map_smoothed <= vmin
# light_zone = (density_map_smoothed > vmin) & (density_map_smoothed < vmax)
# FDC_core = density_map_smoothed >= vmax
# zones[dark_zone] = 0
# zones[light_zone] = 1
# zones[FDC_core] = 2

sLZ_portion = 0.3
# Exp2-6, 2-9, 3-1, 3-7, 5-20240719b, IgG-20240809a, IgG-20240809b, CD40L-20240204b, mLT-20240727d, mLT-20240727f: 0.3
# CD40L-20240128b: 0.4,
# Exp3-9: 0.6
# otherwise 0.5
zones = []
for t, density_map_smoothed_t in tqdm(enumerate(density_map_smoothed)):
    z, r, w = density_map_smoothed_t.shape
    zone = np.empty(shape=(z, r, w))
    vmin, vmax = np.quantile(density_map_smoothed_t, q=(1-sLZ_portion, 1-0.1*sLZ_portion))  # Lower 50% are DZ, Upper 5% are dLZ for every snapshot
    dark_zone = density_map_smoothed_t <= vmin
    light_zone = (density_map_smoothed_t > vmin) & (density_map_smoothed_t < vmax)
    FDC_core = density_map_smoothed_t >= vmax
    print(vmin, vmax)
    zone[dark_zone] = 0
    zone[light_zone] = 1
    zone[FDC_core] = 2
    zones.append(zone)
zones = np.array(zones)

print('DZ Volume: ', np.sum(zones==0)/(imgs.shape[0]*imgs.shape[1]*imgs.shape[2]*imgs.shape[3])*100, '%')
print('LZ Volume:', np.sum(zones==1)/(imgs.shape[0]*imgs.shape[1]*imgs.shape[2]*imgs.shape[3])*100, '%')
print('dLZ Volume:', np.sum(zones==2)/(imgs.shape[0]*imgs.shape[1]*imgs.shape[2]*imgs.shape[3])*100, '%')
#pts_coordinates = positions[['Position Z', 'Position Y', 'Position X']]  # (50000, 3)
#pts_values = da.random.random((50000, 4000), chunks=(50000, 1))  # (50000, 4000)

#pos = np.round(positions[['Position Z', 'Position Y', 'Position X']]).astype(int).values
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(imgs,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap='gray', opacity=1)
viewer.add_image(density_map_smoothed,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap='turbo', opacity=1)
viewer.add_image(zones, scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='Zone',  colormap='gray', opacity=1)
# pts_layer = viewer.add_points(pts_coordinates, scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), features={'value': np.asarray(positions['TrackID'])},
#                               face_color='value', size=2, face_colormap='jet',)

# plt.figure(figsize=(14,7)) # Make it 14x7 inch
# plt.hist(density_map.flatten(), bins=256, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
# plt.show()
# plt.clf()
# plt.close()


############################### Extract association of cell positions to Zones ###############################

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Imaris csvs\mLT\20240727f-DenseTfh-D10-B5L-ZT4-30-117-fov230-256px-mLT_Statistics\\'
types = next(os.walk(path))[1]

for type in types:
    files = next(os.walk(path+type))[2]
    bool_list = ['Position' in ele for ele in files]
    position_file = np.array(files)[np.array(bool_list)][0]

    csv = pd.read_csv(path+type+'/'+position_file, skiprows=3)

    positions = csv[['Position Z']] * 1 / um_per_zsice  # Change um -> pix
    positions[['Position Y', 'Position X']] = csv[['Position Y', 'Position X']] * 1/um_per_pixel  # Change um -> pix
    positions['Time'] = csv['Time']
    positions['TrackID'] = csv['TrackID']

    df = pd.DataFrame()
    duration = np.max(positions['Time'])
    for t, zone in tqdm( zip(range(1, duration+1), zones) ):
        df_temp=pd.DataFrame()
        position_t = positions[positions['Time']==t][['Position Z', 'Position Y', 'Position X']].values  # (n_cell, 3)

        ################# Locate Zone (DZ:0, LZ:1, FDC core:2) #################
        int_positions = np.round(position_t).astype(int)
        label = zone[int_positions[:,0], int_positions[:,1], int_positions[:,2]]  # Data for DZ:0, LZ:1, FDC core:2
        df_temp['Zone'] = label

        ################# Distance to FDC core #################
        DZ_positions = np.argwhere(zone == 0)  # (n_points, 3)
        LZ_positions = np.argwhere(zone == 1)  # (n_points, 3)
        FDC_core_positions = np.argwhere(zone == 2)  # (n_points, 3)

        if FDC_core_positions.shape[0] == 0:  # No FDC Core region
            FDC_core_positions = previous_FDC_core_positions  # If first frame doesn't have FDC core, this code doesn't work

        previous_FDC_core_positions = FDC_core_positions

        DZ_min_distances=[]
        LZ_min_distances = []
        FDC_core_min_distances=[]
        for position_t_cell in position_t:

            displacements = DZ_positions - position_t_cell
            displacements[:, 0] = displacements[:, 0]*um_per_zsice  # pix -> um for z
            displacements[:, 1:] = displacements[:, 1:] * um_per_pixel  # pix -> um for x&y
            distances = np.linalg.norm(displacements, axis=1)
            DZ_min_distance = np.min(distances)
            DZ_min_distances.append(DZ_min_distance)

            displacements = LZ_positions - position_t_cell
            displacements[:, 0] = displacements[:, 0] * um_per_zsice  # pix -> um for z
            displacements[:, 1:] = displacements[:, 1:] * um_per_pixel  # pix -> um for x&y
            distances = np.linalg.norm(displacements, axis=1)
            LZ_min_distance = np.min(distances)
            LZ_min_distances.append(LZ_min_distance)

            displacements = FDC_core_positions - position_t_cell
            displacements[:, 0] = displacements[:, 0] * um_per_zsice  # pix -> um for z
            displacements[:, 1:] = displacements[:, 1:] * um_per_pixel  # pix -> um for x&y
            distances = np.linalg.norm(displacements, axis=1)
            FDC_core_min_distance = np.min(distances)
            FDC_core_min_distances.append(FDC_core_min_distance)
            #min_distance_idx = np.argmin(distances)
            #min_distance_FDC_core_position = FDC_core_positions[min_distance_idx]

        df_temp['Distance_to_DZ'] = DZ_min_distances
        df_temp['Distance_to_LZ'] = LZ_min_distances
        df_temp['Distance_to_FDC_core'] = FDC_core_min_distances
        df = pd.concat([df, df_temp], axis=0)

    df = df.reset_index(drop=True)
    df['Time'] = csv['Time']
    df['TrackID'] = csv['TrackID']

    df.to_csv(path+type+'/'+'_FDCfeatures.csv')







