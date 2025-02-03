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
from utils.traj_utils import *

############################### Read FDC masks ###############################

#path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\FDC feature projection\Exp3-7-Good-D11-B2-ZT2-30-117-FOV230-256px\\'

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\FDC feature projection\\'
folders = next(os.walk(path))[1]

folder = folders[-2]

first_ = folder.find('-')
second_ = folder.find('-', first_+1)
third_ = folder.find('-', second_+1)
forth_ = folder.find('-', third_+1)
fifth_ = folder.find('-', forth_+1)

exp = folder[:first_]
group = folder[forth_+1:fifth_]
video = folder[first_+1:]

if exp == 'Exp1':
    um_per_pixel = 230.9 / 320  # For Exp1
    um_per_zsice = 3
else:
    um_per_pixel = 230.9 / 256
    um_per_zsice = 3

if 'A' in group:
    FDC_channel = 'C2'
else:
    FDC_channel = 'C3'


files = next(os.walk(path+folder))[2]
bool_list = [FDC_channel in ele for ele in files]
files = np.array(files)[np.array(bool_list)]
files.sort()


first_ = files[-1].find('_')
second_ = files[-1].find('_', first_+1)
third_ = files[-1].find('_', second_+1)
end = files[-1].find('.tif')

n_frames = int( files[-1][first_+2:second_] ) + 1
n_zslices = int( files[-1][third_+2:end] ) + 1

#imgs = get_5d_stack(path=path, files=files, order='zct', n_zslices=30, n_frames=181, n_channels=4)
imgs = get_hyperstack(path=path+folder+'/', files=files, order='zt', n_zslices=n_zslices, n_frames=n_frames)

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(imgs,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap='gray', opacity=1)
#viewer.add_image(imgs,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap=["blue", "green",'red','gray'], opacity=1, channel_axis=1)

############################### Visualize FDC zones (DZ, LZ, FDC core) ###############################

kernel = np.full(shape=(6, 6, 3), fill_value=1)

density_map = []
for t, img in tqdm(enumerate(imgs)):
    conv_img = scipy.ndimage.convolve(img, kernel)
    g_img = skimage.filters.gaussian(conv_img, sigma=10, preserve_range=True)
    density_map.append(g_img)

density_map = np.array(density_map)
density_map_smoothed = skimage.filters.gaussian(density_map, sigma=10, preserve_range=True)

sLZ_portion = 0.5 # Exp2-6, 2-9, 3-1: 0.3,
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


viewer = napari.Viewer(ndisplay=3)
viewer.axes.visible = True
viewer.scale_bar.visible=True
viewer.scale_bar.unit = 'μm'
t, z, y, x = imgs.shape
lines = bbox_3d(z*um_per_zsice, y*um_per_pixel, x*um_per_pixel)
viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=0.1,
        edge_color="white",
    )
#viewer.add_image(imgs,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap='gray', opacity=1)
viewer.add_image(density_map_smoothed*imgs,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap='twilight_shifted', opacity=1)
viewer.add_image(zones*imgs, scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='Zone',  colormap='gray', opacity=1)

viewer.camera.angles = (-1, 25, 85)
viewer.dims.axis_labels = ['t', 'z', 'y', 'x']

############################### Extract association of cell positions to Zones ###############################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df_duration = pd.read_parquet(path+'long_traj_duration_20.parquet')
df_duration = df_duration[df_duration['Video']==video].reset_index(drop=True)

# int_features = ['Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC', 'Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell',
#                 'Shortest_Distance_to_Surfaces_Surfaces=FDC', 'Shortest_Distance_to_Surfaces_Surfaces=T-cell', 'Zone', 'Distance_to_DZ', 'Distance_to_LZ',
#        'Distance_to_FDC_core',]
for t in range(n_frames+1)[20:]:
    print(t)


trajectories = {}
traj_idx = 0
i0 = 0
length_name='Time_span'
coord_name='Time'

t = 40
for i in range(0, df_duration.shape[0]):
    if (i == 0) or (i == duration + i0):
        duration = df_duration[length_name][i]
        traj_data_temp = df_duration[i: duration + i]
        last_tp = np.array(traj_data_temp['Time'])[-1]
        if last_tp >= t:
            traj = traj_data_temp[traj_data_temp['Time']<=t].reset_index(drop=True)
            trajectories[traj_idx] = traj
        else:
            trajectories[traj_idx] = 0
        traj_idx = traj_idx + 1
        i0 = i
    else:
        continue



FDC_overlap = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Overlapped_Volume_Ratio_to_Surfaces_Surfaces=FDC',
                                              equal_length=False, frame_name='Time')
FDC_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Shortest_Distance_to_Surfaces_Surfaces=FDC',
                                              equal_length=False, frame_name='Time')
T_overlap = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Overlapped_Volume_Ratio_to_Surfaces_Surfaces=T-cell',
                                              equal_length=False, frame_name='Time')
T_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Shortest_Distance_to_Surfaces_Surfaces=T-cell',
                                              equal_length=False, frame_name='Time')
DZ_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Distance_to_DZ',
                                              equal_length=False, frame_name='Time')
LZ_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Distance_to_LZ',
                                              equal_length=False, frame_name='Time')
Core_distances = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name='Distance_to_FDC_core',
                                              equal_length=False, frame_name='Time')

label_series = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['Type', 'Time_span', 'Zone', 'Label'],
                                               equal_length=False, frame_name='Time')
trajectories = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['Position X', 'Position Y', 'Position Z'],
                                               equal_length=False, frame_name='Time')

from features.interaction import DistanceSignal, OverlapSignal
feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr', 'autocorr',
                ]
FDC_dist = DistanceSignal(FDC_distances)
df_distance = FDC_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)

feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
FDC_over = OverlapSignal(FDC_overlap)
df_overlap = FDC_over.extract_features(feature_list)

df_inter_FDC = pd.concat([df_distance, df_overlap], axis=1)
for column in df_inter_FDC.columns:
    df_inter_FDC.rename(columns={column:'FDC_'+column}, inplace=True)

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr', 'autocorr',
                ]

T_dist = DistanceSignal(T_distances)
df_distance = T_dist.extract_features(feature_list, tau_limit=3)
for column in df_distance.columns:
    df_distance.rename(columns={column:'distance_'+column}, inplace=True)

feature_list = ['avg_overlap', 'overlap_slopes', 'contact_times', 'contact_persistences', 'noncontact_times', 'noncontact_persistences']
T_over = OverlapSignal(T_overlap)
df_overlap = T_over.extract_features(feature_list)

df_inter_T = pd.concat([df_distance, df_overlap], axis=1)
for column in df_inter_T.columns:
    df_inter_T.rename(columns={column:'T_'+column}, inplace=True)


feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr', 'autocorr',
                ]
DZ_dist = DistanceSignal(DZ_distances)
df_inter_DZ = DZ_dist.extract_features(feature_list, tau_limit=3)
for column in df_inter_DZ.columns:
    df_inter_DZ.rename(columns={column:'DZ_distance_'+column}, inplace=True)

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr', 'autocorr',
                ]
LZ_dist = DistanceSignal(LZ_distances)
df_inter_LZ = LZ_dist.extract_features(feature_list, tau_limit=3)
for column in df_inter_LZ.columns:
    df_inter_LZ.rename(columns={column:'LZ_distance_'+column}, inplace=True)

feature_list = ['total', 'maximum',  'distance_slopes', 'variance', 'cov', 'average',
        #'RMSs', 'crestfactor', 'formfactor', 'pulseindicator',  'skewness', 'kurtosis', 'ngaussalpha', 'peak_to_peak',
        #'partial_autocorr', 'autocorr',
                ]
Core_dist = DistanceSignal(Core_distances)
df_inter_Core = Core_dist.extract_features(feature_list, tau_limit=3)
for column in df_inter_Core.columns:
    df_inter_Core.rename(columns={column:'Core_distance_'+column}, inplace=True)

df_inter = pd.concat([df_inter_FDC, df_inter_T, df_inter_DZ, df_inter_LZ, df_inter_Core], axis=1)


label_list = []
for idx, typs in label_series.items():
    label_list_temp = []
    n_columns = typs.shape[1]
    for col in range(n_columns):
        col_data = typs[:, col][0]
        label_list_temp.append(col_data)
    label_list.append(label_list_temp)

label_list = np.array(label_list)

from features.basic_motility import BasicMotility
feature_list = ['total_distance', 'avg_speed', 'max_speed', 'min_speed', 'net_distance',  'progressivity', 'alphas',
                'total_angle', 'avg_angle', 'max_angle', 'min_angle',
                'displ_variance', 'displ_cov','displ_skewness', 'displ_kurtosis', 'displ_ngaussalpha', 'displ_gini',
                'angle_variance', 'angle_cov', 'angle_skewness', 'angle_kurtosis', 'angle_ngaussalpha', 'angle_gini',
                'msds', 'displ_autocorr', 'displ_partial_autocorr', 'angle_autocorr', 'angle_partial_autocorr',
                #'displ_hurst_RS', 'angle_hurst_RS',
                ]
basic_motil = BasicMotility(trajectories, time_unit=0.5, feature_list=feature_list)
df_basic = basic_motil.extract_features(tau_limit=3)


df_long = pd.concat([df_inter, pd.DataFrame(label_list, columns=['Type', 'Time_span', 'Zone', 'Label'])], axis=1)









path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\\'
df = pd.read_parquet(path+'all_features_20.parquet')
df_duration = pd.read_parquet(path+'traj_duration_20.parquet')

df = df[df['Video']==video].reset_index(drop=True)
df_duration = df_duration[df_duration['Video']==video].reset_index(drop=True)


feature = 'FDC_contact_persistences'


positions = df_duration[['Position Z']] * 1 / um_per_zsice  # Change um -> pix
positions[['Position Y', 'Position X']]  = df_duration[['Position Y', 'Position X']] * 1/um_per_pixel  # Change um -> pix
positions['Time'] = df_duration['Time']
positions['TrackID'] = df_duration['TrackID']

duration=20
feature_maps = []
t, z, r, w = imgs.shape

for t in range(n_frames):
    print(t)

for idx, row in df.iterrows():

    cell_position = positions[duration * idx:duration * (idx + 1)][['Position Z', 'Position Y', 'Position X']].values
    cell_last_time = positions[duration * idx:duration * (idx + 1)]['Time'].values[-1]
    avg_position = np.mean(cell_position, axis=0)
    int_avg_position = np.round(avg_position).astype(int)

    typ = row['Type']
    row[feature]
    feature_map = np.empty(shape=(z, r, w))
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

#df.to_csv(path+type+'/'+'_FDCfeatures.csv')


############################### Visualize FDC zones (DZ, LZ, FDC core) ###############################

def lissajous(t):
    a = np.random.random(size=(3,)) * 80.0 - 40.0
    b = np.random.random(size=(3,)) * 0.05
    c = np.random.random(size=(3,)) * 0.1
    return (a[i] * np.cos(b[i] * t + c[i]) for i in range(3))


def tracks_3d(num_tracks=10):
    """ create 3d+t track data """
    tracks = []

    for track_id in range(num_tracks):

        # space to store the track data and features
        track = np.zeros((200, 10), dtype=np.float32)

        # time
        timestamps = np.arange(track.shape[0])  # n_frames
        x, y, z = lissajous(timestamps)

        track[:, 0] = track_id
        track[:, 1] = timestamps
        track[:, 2] = 50.0 + z
        track[:, 3] = 50.0 + y
        track[:, 4] = 50.0 + x

        # calculate the speed as a feature
        gz = np.gradient(track[:, 2])
        gy = np.gradient(track[:, 3])
        gx = np.gradient(track[:, 4])

        speed = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        track[:, 5] = gz
        track[:, 6] = gy
        track[:, 7] = gx
        track[:, 8] = speed
        track[:, 9] = distance

        tracks.append(track)

    tracks = np.concatenate(tracks, axis=0)
    data = tracks[:, :5]  # just the coordinate data

    features = {
        'time': tracks[:, 1],
        'gradient_z': tracks[:, 5],
        'gradient_y': tracks[:, 6],
        'gradient_x': tracks[:, 7],
        'speed': tracks[:, 8],
        'distance': tracks[:, 9],
    }

    graph = {}
    return data, features, graph


tracks, features, graph = tracks_3d(num_tracks=10)
vertices = tracks[:, 1:]

viewer = napari.Viewer(ndisplay=3)
viewer.add_points(vertices, size=1, name='points', opacity=0.3)
viewer.add_tracks(tracks, features=features, name='tracks')














