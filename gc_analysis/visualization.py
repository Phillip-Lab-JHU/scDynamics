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

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\FDC feature projection\FDC images\\'
folders = next(os.walk(path))[1]
folders.sort()
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
    um_per_zslice = 3
else:
    um_per_pixel = 230.9 / 256
    um_per_zslice = 3

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

# viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(imgs,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='gray', opacity=1)

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
    #vmin, vmax = 11000/10000, 22000/10000 # Exp 3-7
    dark_zone = density_map_smoothed_t <= vmin
    light_zone = (density_map_smoothed_t > vmin) & (density_map_smoothed_t < vmax)
    FDC_core = density_map_smoothed_t >= vmax
    #print(vmin, vmax)
    zone[dark_zone] = 0
    zone[light_zone] = 1
    zone[FDC_core] = 2
    zones.append(zone)
zones = np.array(zones)

print('DZ Volume: ', np.sum(zones==0)/(imgs.shape[0]*imgs.shape[1]*imgs.shape[2]*imgs.shape[3])*100, '%')
print('LZ Volume:', np.sum(zones==1)/(imgs.shape[0]*imgs.shape[1]*imgs.shape[2]*imgs.shape[3])*100, '%')
print('dLZ Volume:', np.sum(zones==2)/(imgs.shape[0]*imgs.shape[1]*imgs.shape[2]*imgs.shape[3])*100, '%')


print(np.min(density_map_smoothed), np.max(density_map_smoothed))
thresh=0.8  # 0.8 for Exp3-7 / 1.5 for Exp3-5 / 1 for Exp3-4, Exp2-6 / 8000 for Exp2-9
density_map_smoothed_refined = np.where(density_map_smoothed < thresh, 0, density_map_smoothed)
ref_FDC_img = np.where(density_map_smoothed_refined*imgs>0, 1, 0)

save_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\FDC feature projection\projections\%s\\' % video
if not os.path.isdir(save_path):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
    os.makedirs(save_path)



viewer = napari.Viewer(ndisplay=3)
viewer.axes.visible = True
viewer.scale_bar.visible=True
viewer.scale_bar.unit = 'μm'
t, z, y, x = imgs.shape
lines = bbox_3d(z*um_per_zslice, y*um_per_pixel, x*um_per_pixel)
viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=0.1,
        edge_color="white",
    )
#viewer.add_image(density_map_smoothed*imgs,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='turbo', opacity=1)
viewer.add_image(density_map_smoothed_refined*imgs[50],  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='turbo', opacity=1)
#viewer.add_image(zones*imgs, scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='Zone',  colormap='gray', opacity=1)

viewer.camera.angles = (-1, 25, 85)
viewer.dims.axis_labels = ['t', 'z', 'y', 'x']
viewer.screenshot(path=save_path+'_density_map.png', canvas_only=True, scale=2)
viewer.close()


viewer = napari.Viewer(ndisplay=3)
viewer.axes.visible = True
viewer.scale_bar.visible=True
viewer.scale_bar.unit = 'μm'
t, z, y, x = imgs.shape
lines = bbox_3d(z*um_per_zslice, y*um_per_pixel, x*um_per_pixel)
viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=0.1,
        edge_color="white",
    )
#viewer.add_image(density_map_smoothed*imgs,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='turbo', opacity=1)
viewer.add_image(zones*imgs[50],  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='gray', opacity=1)
#viewer.add_image(zones*imgs, scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='Zone',  colormap='gray', opacity=1)

viewer.camera.angles = (-1, 25, 85)
viewer.dims.axis_labels = ['t', 'z', 'y', 'x']
viewer.screenshot(path=save_path+'_zone_map.png', canvas_only=True, scale=2)
viewer.close()
############################### Extract association of cell positions to Zones ###############################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df = pd.read_parquet(path+'GCB_no_inhibit_all_features_20.parquet')
df_duration = pd.read_parquet(path+'GCB_no_inhibit_traj_duration_20.parquet')

for vid in np.unique(df['Video']):
    df_video = df[df['Video']==vid].reset_index(drop=True)
    print(df_video['Exp'][0], vid, df_video[df_video['Type']=='wt_B-cell'].shape[0], df_video[df_video['Type']=='mt_B-cell'].shape[0])

_, _, Zone_series = to_timeseries_fast(df_duration, duration=20, feature_name='Zone')

feature_list = ['avg_zone', 'dz_resident_times', 'dz_resident_persistences', 'slz_resident_times',
                'slz_resident_persistences', 'dlz_resident_times', 'dlz_resident_persistences']
from features.interaction import ZoneSignal
Zone_func = ZoneSignal(Zone_series)
df_zone = Zone_func.extract_features(feature_list)

df = pd.concat([df, df_zone], axis=1)


df.loc[(df['avg_zone'] < 0.4) & (df['avg_zone'] >= 0), 'Zone'] = 'DZ'
df.loc[(df['avg_zone'] < 0.8) & (df['avg_zone'] >= 0.4), 'Zone'] = 'DZ-sLZ'
df.loc[(df['avg_zone'] < 1.2) & (df['avg_zone'] >= 0.8), 'Zone'] = 'sLZ'
df.loc[(df['avg_zone'] < 1.6) & (df['avg_zone'] >= 1.2), 'Zone'] = 'sLZ-dLZ'
df.loc[(df['avg_zone'] <= 2) & (df['avg_zone'] >= 1.6), 'Zone'] = 'dLZ'

duration=20
label_expanded = np.repeat(df['Zone'], duration).reset_index(drop=True)
df_duration['Zone_label'] = label_expanded

df = df[df['Video']==video].reset_index(drop=True)
df_duration = df_duration[df_duration['Video']==video].reset_index(drop=True)


for typ in ['wt_B-cell', 'mt_B-cell']:
    for zone in ['DZ', 'DZ-sLZ', 'sLZ', 'sLZ-dLZ', 'dLZ']:
        print(typ, zone, df[(df['Type']==typ)&(df['Zone']==zone)].shape[0])

####### Paint motility features onto FDC mask #######
feature_list = ['avg_speed', 'progressivity', 'avg_angle', 'displ_cov', 'angle_cov', 'displ_autocorr_1',
                'avg_speed_x', 'avg_speed_y', 'avg_speed_z','exy_total', 'phi_total', 'morpho_avg_speed',
                'quality_FDC_approach_persistences', 'quality_DZ_approach_persistences', 'quality_Core_approach_persistences',
                'FDC_distance_average', 'FDC_diff_distance_average', 'FDC_avg_overlap', 'FDC_contact_persistences',
                'T_distance_average', 'T_diff_distance_average', 'T_avg_overlap', 'T_contact_persistences',
                'DZ_distance_average', 'DZ_diff_distance_average', 'LZ_distance_average', 'LZ_diff_distance_average',
                'Core_distance_average', 'Core_diff_distance_average', 'slz_resident_times', 'dlz_resident_times']


for feature in feature_list:
    duration=20
    from itertools import product
    suggested_radius = (3*np.mean(df_duration['Volume'])/(4*np.pi))**(1/3)
    z_radius = 2 # 2
    radius = 5 # 5
    # offsets = np.array([offset for offset in product(range(-radius, radius + 1), repeat=3)])  # Generate neighborhood offsets
    offsets = np.array([[dz, dy, dx] for dz in range(-z_radius, z_radius+1)
                                      for dy in range(-radius, radius+1)
                                      for dx in range(-radius, radius+1)])
    # Total number of segments (cells)
    t, z, r, w = imgs.shape

    df_part = df[df['Type'] == 'wt_B-cell'].reset_index(drop=True)
    df_duration_part = df_duration[df_duration['Type'] == 'wt_B-cell'].reset_index(drop=True)

    feature_map_wt = project_feature_onto_FDC(df_part, df_duration_part, feature, duration=duration, img_shape=(z,r,w),
                                           offsets=offsets, um_per_zslice=um_per_zslice, um_per_pixel=um_per_pixel)
    df_part = df[df['Type'] == 'mt_B-cell'].reset_index(drop=True)
    df_duration_part = df_duration[df_duration['Type'] == 'mt_B-cell'].reset_index(drop=True)

    feature_map_mt = project_feature_onto_FDC(df_part, df_duration_part, feature, duration=duration, img_shape=(z,r,w),
                                           offsets=offsets, um_per_zslice=um_per_zslice, um_per_pixel=um_per_pixel)
    print(np.min(feature_map_wt[feature_map_wt!=0]), np.min(feature_map_mt[feature_map_mt!=0]))
    if feature == 'T_contact_persistences':
        feature_map_wt[feature_map_wt > 15] = 15
        feature_map_mt[feature_map_mt > 15] = 15

    # elif feature == 'FDC_contact_persistences':
    #     feature_map_wt[feature_map_wt > 15] = 15
    #     feature_map_mt[feature_map_mt > 15] = 15

    elif feature == 'T_avg_overlap':
        feature_map_wt[feature_map_wt > 0.06] = 0.06
        feature_map_mt[feature_map_mt > 0.06] = 0.06

    elif feature == 'FDC_avg_overlap':
        feature_map_wt[feature_map_wt > 0.2] = 0.2
        feature_map_mt[feature_map_mt > 0.2] = 0.2

    # elif feature == 'avg_speed':
    #     feature_map_wt[feature_map_wt < 3] = 3
    #     feature_map_mt[feature_map_mt < 3] = 3

    elif feature == 'avg_speed':
        feature_map_wt[feature_map_wt > 10] = 10
        feature_map_mt[feature_map_mt > 10] = 10

    # elif feature == 'morpho_avg_speed':
    #     feature_map_wt[feature_map_wt < 2] = 2
    #     feature_map_mt[feature_map_mt < 2] = 2

    elif feature == 'morpho_avg_speed':
        feature_map_wt[feature_map_wt > 4] = 4
        feature_map_mt[feature_map_mt > 4] = 4

    elif feature == 'morpho_avg_speed':
        feature_map_wt[feature_map_wt > 4] = 4
        feature_map_mt[feature_map_mt > 4] = 4

    elif feature == 'progressivity':
        feature_map_wt[feature_map_wt > 0.8] = 0.8
        feature_map_mt[feature_map_mt > 0.8] = 0.8

    elif feature == 'displ_cov':
        feature_map_wt[feature_map_wt > 1.2] = 1.2
        feature_map_mt[feature_map_mt > 1.2] = 1.2

    feature_map_wt[feature_map_wt < 0] = 0
    feature_map_mt[feature_map_mt < 0] = 0


    viewer = napari.Viewer(ndisplay=3)
    viewer.axes.visible = True
    viewer.scale_bar.visible=True
    viewer.scale_bar.unit = 'μm'

    viewer.add_image(ref_FDC_img[50], scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='gray',blending='additive',opacity=1.0)
    viewer.add_image(ref_FDC_img[50]*feature_map_wt,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='turbo', blending='translucent',opacity=1, name='wt')

    t, z, y, x = imgs.shape
    lines = bbox_3d(z*um_per_zslice, y*um_per_pixel, x*um_per_pixel)
    viewer.add_shapes(
            lines,
            shape_type="line",
            edge_width=0.1,
            edge_color="white",
        )

    viewer.camera.angles = (-1, 25, 85)
    viewer.dims.axis_labels = ['t', 'z', 'y', 'x']
    viewer.screenshot(path=save_path+'%s_wt_range(%s, %s).png'%(feature, np.min(feature_map_wt), np.max(feature_map_wt)), canvas_only=True, scale=2)
    viewer.close()

    # viewer.layers['mt'].visible = False
    # viewer.layers['wt'].visible = True

    viewer = napari.Viewer(ndisplay=3)
    viewer.axes.visible = True
    viewer.scale_bar.visible=True
    viewer.scale_bar.unit = 'μm'

    viewer.add_image(ref_FDC_img[50], scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='gray',blending='additive',opacity=1.0)
    viewer.add_image(ref_FDC_img[50]*feature_map_mt,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='turbo', blending='translucent',opacity=1, name='mt')

    t, z, y, x = imgs.shape
    lines = bbox_3d(z*um_per_zslice, y*um_per_pixel, x*um_per_pixel)
    viewer.add_shapes(
            lines,
            shape_type="line",
            edge_width=0.1,
            edge_color="white",
        )

    viewer.camera.angles = (-1, 25, 85)
    viewer.dims.axis_labels = ['z', 'y', 'x']

    viewer.screenshot(path=save_path+'%s_mt_range(%s, %s).png'%(feature, np.min(feature_map_mt), np.max(feature_map_mt)), canvas_only=True, scale=2)
    viewer.close()




############################### Extract association of cell positions to Zones ###############################
# path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
# df = pd.read_parquet(path+'Genes in behavior.parquet')
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\Figure6. Expanded behavior\\'
df = pd.read_parquet(path+'Expanded_behavior.parquet')

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Analysis\feature_csvs\\'
df_duration = pd.read_parquet(path+'GCB_no_inhibit_traj_duration_20.parquet')

#df.iloc[:, :3000] = df.iloc[:, :3000].apply(lambda x: x - x.min() + 1e-6)  # shift values more than 0

for vid in np.unique(df['Video']):
    df_video = df[df['Video']==vid].reset_index(drop=True)
    print(df_video['Exp'][0], vid, df_video[df_video['Type']=='wt_B-cell'].shape[0], df_video[df_video['Type']=='mt_B-cell'].shape[0])

df = df[df['Video']==video].reset_index(drop=True)
df_duration = df_duration[df_duration['Video']==video].reset_index(drop=True)

####### Paint predicted gene expression onto FDC mask #######


feature_dict = {}
feature_dict['Beguelin_2020 FDC interaction'] = ['Bcr', 'Tnfrsf13c', 'Itgb2', 'Itgb4', 'Ighg1', 'Tnf', 'Lta', 'Ltb', 'Itga4']
feature_dict['Beguelin_2020 Tfh interaction'] = ['Tnfrsf14', 'Icam1', 'Basp1', 'Egr2', 'Cd69', 'Itgam', 'Ptger4', 'Icosl', 'Socs3', 'Ciita', 'Cd40']
feature_dict['Beguelin_2020 LZ and anti-apoptosis'] = ['Cd52', 'Mreg', 'Aldoc', 'Bcl2a1b', 'Cbx8']
feature_dict['Beguelin_2020 DZ hallmark'] = ['Hmmr', 'Lgr5', 'Ptgr1', 'Pif1', 'Serinc5', 'Bcl2l11', 'Bcl2l14']
feature_dict['Beguelin_2020 CC recycling'] = ['Pde3b', 'Klhl5', 'Ankrd28', 'Mycbpap', 'Bag3', 'Stag3', 'Tjp2', 'Tspan5', 'Kcna3', 'Abi2', 'Irak1bp1', 'Morn4']


for gene_module, gene_list in feature_dict.items():
    #print(gene_module, gene_list)

    gene_save_path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\FDC feature projection\projections\%s\gene\%s\\' % (video, gene_module)
    if not os.path.isdir(gene_save_path):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(gene_save_path)

    for feature in gene_list:
        if feature not in df.columns[:3000]:
            print(gene_module, feature, 'does not exist in dataframe')
            continue
        duration=20
        from itertools import product
        suggested_radius = (3*np.mean(df_duration['Volume'])/(4*np.pi))**(1/3)
        z_radius = 2 # 2
        radius = 5 # 5
        # offsets = np.array([offset for offset in product(range(-radius, radius + 1), repeat=3)])  # Generate neighborhood offsets
        offsets = np.array([[dz, dy, dx] for dz in range(-z_radius, z_radius+1)
                                          for dy in range(-radius, radius+1)
                                          for dx in range(-radius, radius+1)])
        # Total number of segments (cells)
        t, z, r, w = imgs.shape


        df_part = df[df['Type'] == 'wt_B-cell'].reset_index(drop=True)
        df_duration_part = df_duration[df_duration['Type'] == 'wt_B-cell'].reset_index(drop=True)

        feature_map_wt = project_feature_onto_FDC(df_part, df_duration_part, feature, duration=duration, img_shape=(z,r,w),
                                               offsets=offsets, um_per_zslice=um_per_zslice, um_per_pixel=um_per_pixel)
        df_part = df[df['Type'] == 'mt_B-cell'].reset_index(drop=True)
        df_duration_part = df_duration[df_duration['Type'] == 'mt_B-cell'].reset_index(drop=True)

        feature_map_mt = project_feature_onto_FDC(df_part, df_duration_part, feature, duration=duration, img_shape=(z,r,w),
                                               offsets=offsets, um_per_zslice=um_per_zslice, um_per_pixel=um_per_pixel)
        print(np.min(feature_map_wt[feature_map_wt!=0]), np.min(feature_map_mt[feature_map_mt!=0]))

        # if feature == 'T_contact_persistences':
        #     feature_map_wt[feature_map_wt > 15] = 15
        #     feature_map_mt[feature_map_mt > 15] = 15
        #feature_map_wt[feature_map_wt < 0] = 0
        #feature_map_mt[feature_map_mt < 0] = 0


        viewer = napari.Viewer(ndisplay=3)
        viewer.axes.visible = True
        viewer.scale_bar.visible=True
        viewer.scale_bar.unit = 'μm'

        viewer.add_image(ref_FDC_img[50], scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='gray',blending='additive',opacity=1.0)
        viewer.add_image(ref_FDC_img[50]*feature_map_wt,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='turbo', blending='translucent',opacity=1, name='wt')

        t, z, y, x = imgs.shape
        lines = bbox_3d(z*um_per_zslice, y*um_per_pixel, x*um_per_pixel)
        viewer.add_shapes(
                lines,
                shape_type="line",
                edge_width=0.1,
                edge_color="white",
            )

        viewer.camera.angles = (-1, 25, 85)
        viewer.dims.axis_labels = ['t', 'z', 'y', 'x']
        viewer.screenshot(path=gene_save_path+'%s_wt_range(%s, %s).png'%(feature, np.min(feature_map_wt), np.max(feature_map_wt)), canvas_only=True, scale=2)
        viewer.close()

        # viewer.layers['mt'].visible = False
        # viewer.layers['wt'].visible = True

        viewer = napari.Viewer(ndisplay=3)
        viewer.axes.visible = True
        viewer.scale_bar.visible=True
        viewer.scale_bar.unit = 'μm'

        viewer.add_image(ref_FDC_img[50], scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='gray',blending='additive',opacity=1.0)
        viewer.add_image(ref_FDC_img[50]*feature_map_mt,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='turbo', blending='translucent',opacity=1, name='mt')

        t, z, y, x = imgs.shape
        lines = bbox_3d(z*um_per_zslice, y*um_per_pixel, x*um_per_pixel)
        viewer.add_shapes(
                lines,
                shape_type="line",
                edge_width=0.1,
                edge_color="white",
            )

        viewer.camera.angles = (-1, 25, 85)
        viewer.dims.axis_labels = ['z', 'y', 'x']

        viewer.screenshot(path=gene_save_path+'%s_mt_range(%s, %s).png'%(feature, np.min(feature_map_mt), np.max(feature_map_mt)), canvas_only=True, scale=2)
        viewer.close()


####### Convert positions to pixel and locate them in FDC #######
feature = 'beh_kmeans'

duration=20
from itertools import product
suggested_radius = (3*np.mean(df_duration['Volume'])/(4*np.pi))**(1/3)
z_radius = 2 # 2
radius = 5 # 5
# offsets = np.array([offset for offset in product(range(-radius, radius + 1), repeat=3)])  # Generate neighborhood offsets
offsets = np.array([[dz, dy, dx] for dz in range(-z_radius, z_radius+1)
                                  for dy in range(-radius, radius+1)
                                  for dx in range(-radius, radius+1)])
# Total number of segments (cells)
t, z, r, w = imgs.shape

df_part = df[df['Type'] == 'wt_B-cell'].reset_index(drop=True)
df_part[feature] = df_part[feature]+1
df_duration_part = df_duration[df_duration['Type'] == 'wt_B-cell'].reset_index(drop=True)

# feature_map_wt = project_feature_onto_FDC(df_part, df_duration_part, feature, duration=duration, img_shape=(z,r,w),
#                                        offsets=offsets, um_per_zslice=um_per_zslice, um_per_pixel=um_per_pixel, count_norm=False)

feature_map_wt = project_feature_onto_FDC_by_majority_vote(df_part, df_duration_part, feature, duration=duration, img_shape=(z,r,w),
                                       offsets=offsets, um_per_zslice=um_per_zslice, um_per_pixel=um_per_pixel)
df_part = df[df['Type'] == 'mt_B-cell'].reset_index(drop=True)
df_part[feature] = df_part[feature]+1
df_duration_part = df_duration[df_duration['Type'] == 'mt_B-cell'].reset_index(drop=True)

# feature_map_mt = project_feature_onto_FDC(df_part, df_duration_part, feature, duration=duration, img_shape=(z,r,w),
#                                        offsets=offsets, um_per_zslice=um_per_zslice, um_per_pixel=um_per_pixel, count_norm=False)
feature_map_mt = project_feature_onto_FDC_by_majority_vote(df_part, df_duration_part, feature, duration=duration, img_shape=(z,r,w),
                                       offsets=offsets, um_per_zslice=um_per_zslice, um_per_pixel=um_per_pixel)
df_all = df.copy()
df_duration_all = df_duration.copy()
df_all[feature] = df_all[feature]+1

# feature_map_all = project_feature_onto_FDC(df_all, df_duration_all, feature, duration=duration, img_shape=(z,r,w),
#                                        offsets=offsets, um_per_zslice=um_per_zslice, um_per_pixel=um_per_pixel, count_norm=False)
feature_map_all = project_feature_onto_FDC_by_majority_vote(df_all, df_duration_all, feature, duration=duration, img_shape=(z,r,w),
                                       offsets=offsets, um_per_zslice=um_per_zslice, um_per_pixel=um_per_pixel)
import cmcrameri.cm as cmc
from vispy.color import Colormap
batlow_colormap = Colormap(cmc.batlow.colors)
color_list = ('white', '#BAC8DA', '#4F609C', '#8A4F21', '#F06293', '#E9C61D', '#BCBCBC', '#BEDCB0', '#F5A9F5', )
custom_colormap = Colormap(color_list)
viewer = napari.Viewer(ndisplay=3)
viewer.axes.visible = True
viewer.scale_bar.visible=True
viewer.scale_bar.unit = 'μm'

viewer.add_image(ref_FDC_img[50], scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='gray',blending='additive',opacity=1.0)
viewer.add_image(ref_FDC_img[50]*feature_map_wt,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap=('batlow', custom_colormap), blending='translucent',opacity=1, name='wt')

t, z, y, x = imgs.shape
lines = bbox_3d(z*um_per_zslice, y*um_per_pixel, x*um_per_pixel)
viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=0.1,
        edge_color="white",
    )

viewer.camera.angles = (-1, 25, 85)
viewer.dims.axis_labels = ['t', 'z', 'y', 'x']
viewer.screenshot(path=save_path+'%s_wt_range(%s, %s).png'%(feature, np.min(feature_map_wt), np.max(feature_map_wt)), canvas_only=True, scale=2)
viewer.close()

# viewer.layers['mt'].visible = False
# viewer.layers['wt'].visible = True

viewer = napari.Viewer(ndisplay=3)
viewer.axes.visible = True
viewer.scale_bar.visible=True
viewer.scale_bar.unit = 'μm'

viewer.add_image(ref_FDC_img[50], scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='gray',blending='additive',opacity=1.0)
viewer.add_image(ref_FDC_img[50]*feature_map_mt,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap=('batlow', custom_colormap), blending='translucent',opacity=1, name='mt')

t, z, y, x = imgs.shape
lines = bbox_3d(z*um_per_zslice, y*um_per_pixel, x*um_per_pixel)
viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=0.1,
        edge_color="white",
    )

viewer.camera.angles = (-1, 25, 85)
viewer.dims.axis_labels = ['z', 'y', 'x']

viewer.screenshot(path=save_path+'%s_mt_range(%s, %s).png'%(feature, np.min(feature_map_mt), np.max(feature_map_mt)), canvas_only=True, scale=2)
viewer.close()


viewer = napari.Viewer(ndisplay=3)
viewer.axes.visible = True
viewer.scale_bar.visible=True
viewer.scale_bar.unit = 'μm'

viewer.add_image(ref_FDC_img[50], scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='gray',blending='additive',opacity=1.0)
viewer.add_image(ref_FDC_img[50]*feature_map_all,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap=('batlow', custom_colormap), blending='translucent',opacity=1, name='mt')

t, z, y, x = imgs.shape
lines = bbox_3d(z*um_per_zslice, y*um_per_pixel, x*um_per_pixel)
viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=0.1,
        edge_color="white",
    )

viewer.camera.angles = (-1, 25, 85)
viewer.dims.axis_labels = ['z', 'y', 'x']

viewer.screenshot(path=save_path+'%s_all_range(%s, %s).png'%(feature, np.min(feature_map_all), np.max(feature_map_all)), canvas_only=True, scale=2)
viewer.close()

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from qtpy.QtWidgets import QVBoxLayout, QWidget
#
# # Create a napari viewer
# viewer = napari.Viewer()
#
# # Add the image layer with a colormap
# layer = viewer.add_image(ref_FDC_img[0]*feature_map_wt,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='viridis', opacity=1, name='wt')
#
# # Create a matplotlib figure and axis for the colorbar
# fig, ax = plt.subplots(figsize=(1, 4))
# fig.subplots_adjust(left=0.5, right=0.6)
#
# # Create a colorbar
# norm = plt.Normalize(vmin=feature_map_wt.min(), vmax=feature_map_wt.max())
# cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=ax)
# cbar.set_label('Feature Value')
#
# # Create a QWidget to house the matplotlib figure
# canvas = FigureCanvas(fig)
# widget = QWidget()
# layout = QVBoxLayout()
# layout.addWidget(canvas)
# widget.setLayout(layout)
#
# # Add the QWidget to the napari viewer
# viewer.window.add_dock_widget(widget, area='right', name='Colorbar')











####### Convert positions to pixel and locate them in FDC (use final point only) #######

df['final_time'] = df['Time'].apply(lambda x: x[-1])
# Initialize the image and a count array for averaging overlapping positions
feature='avg_speed'
duration=20

t, z, r, w = imgs.shape
feature_map = np.zeros((z, r, w), dtype=np.float32)
count = np.zeros((z, r, w), dtype=np.int32)

# Total number of segments (cells)
n_trajs = df_duration.shape[0] // duration

for traj_idx in range(n_trajs):
    # Extract the 20-frame segment for the current cell
    traj = df_duration.iloc[traj_idx*duration : (traj_idx+1)*duration]

    positions = traj[['Position Z']] * 1 / um_per_zslice  # Change um -> pix
    positions[['Position Y', 'Position X']] = traj[['Position Y', 'Position X']] * 1 / um_per_pixel  # Change um -> pix
    positions = np.round(positions).astype(int)

    z, y, x = positions[['Position Z', 'Position Y', 'Position X']].iloc[-1]  # Final position of cell traj

    # Ensure indices are within bounds
    if 0 <= x < 256 and 0 <= y < 256 and 0 <= z < 30:
        # Retrieve the feature value for this segment
        feature_value = df.iloc[traj_idx][feature]

        # Accumulate the feature value and increment the count
        feature_map[z, y, x] += feature_value
        count[z, y, x] += 1

nonzero_mask = count > 0 # Avoid division by zero
feature_map[nonzero_mask] /= count[nonzero_mask]  # Compute the average feature value at each position



viewer = napari.Viewer(ndisplay=3)
viewer.axes.visible = True
viewer.scale_bar.visible=True
viewer.scale_bar.unit = 'μm'
t, z, y, x = imgs.shape
lines = bbox_3d(z*um_per_zslice, y*um_per_pixel, x*um_per_pixel)
viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=0.1,
        edge_color="white",
    )
#viewer.add_image(imgs,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap='gray', opacity=1)
viewer.add_image(ref_FDC_img[40]*feature_map,  scale=np.array([um_per_zslice, um_per_pixel, um_per_pixel]), colormap='twilight_shifted', opacity=1)
#viewer.add_image(feature_map,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap='twilight_shifted', opacity=1)
#viewer.add_image(zones*imgs, scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='Zone',  colormap='gray', opacity=1)

viewer.camera.angles = (-1, 25, 85)
viewer.dims.axis_labels = ['t', 'z', 'y', 'x']




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














