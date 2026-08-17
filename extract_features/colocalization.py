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

############################### Read imgs and create hyperstack ###############################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Raw images\Exp1\norm_img\1-FDC_masks\\'
#files = next(os.walk(path))[2]

c_list = ['C4', 'C5', 'C6', 'C7', 'C8']

imgs = []
for c in c_list:
    files = next(os.walk(path))[2]
    c_bool = [c in ele for ele in files]
    files = np.array(files)[np.array(c_bool)]
    files.sort()

    img = get_hyperstack(path=path, files=files, order='zt', n_zslices=30, n_frames=170)
    imgs.append(img)

imgs = np.stack(imgs, axis=-1)  # list of (180, 30, 256, 256) -> (180, 30, 256, 256, c)

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(imgs,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), channel_axis=-1,
                 name=['wtGCB', 'mtGCB', 'Tfh', 'FDC', 'macrophage'],
                 colormap=['blue', 'green', 'red', 'gray', 'cyan'], opacity=1)


wtGCB_temp = imgs[:,:,:,:,0]
mtGCB_temp = imgs[:,:,:,:,1]
Tfh_temp = imgs[:,:,:,:,2]
FDC_temp = imgs[:,:,:,:,3]
macrophage_temp = imgs[:,:,:,:,4]

### Set Foreground as 1 ###
wtGCB_temp[wtGCB_temp != 0] = 1
mtGCB_temp[mtGCB_temp != 0] = 1
Tfh_temp[Tfh_temp != 0] = 1
FDC_temp[FDC_temp != 0] = 1
macrophage_temp[macrophage_temp != 0] = 1

############################### boolean operations ###############################

fdc_or_macrophage = FDC_temp | macrophage_temp  # element-wise OR
Tfh_or_macrophage = Tfh_temp | macrophage_temp  # element-wise OR

wtGCB = wtGCB_temp & ~fdc_or_macrophage  # element-wise A not B (A - B)
mtGCB = mtGCB_temp & ~fdc_or_macrophage  # element-wise A not B (A - B)
Tfh = Tfh_temp & ~fdc_or_macrophage  # element-wise A not B (A - B)
FDC = FDC_temp & ~Tfh_or_macrophage  # element-wise A not B (A - B)


# viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(FDC,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]),
#                  colormap='gray', opacity=1)

############################### segmentation ###############################
spot_sigma = 2.8
outline_sigma = 0.01

cle.select_device('RTX')

wtGCB_seg_ = []
for each_frame in tqdm(wtGCB):
    img_gpu = cle.push(each_frame)
    # backgrund_subtracted = cle.top_hat_box(img_gpu, radius_x=5, radius_y=5, radius_z=2)
    #segmented_gpu = cle.voronoi_otsu_labeling(img_gpu, spot_sigma=2.7, outline_sigma=0.01)

    blurred = cle.gaussian_blur(img_gpu, sigma_x=spot_sigma, sigma_y=spot_sigma,
                                sigma_z=spot_sigma*(um_per_pixel / um_per_zsice) )
    detected_spots = cle.detect_maxima_box(blurred, radius_x=0, radius_y=0, radius_z=0)

    blurred = cle.gaussian_blur(img_gpu, sigma_x=outline_sigma, sigma_y=outline_sigma,
                                sigma_z=outline_sigma*(um_per_pixel / um_per_zsice) )
    binary = cle.threshold_otsu(blurred)

    selected_spots = cle.binary_and(binary, detected_spots)
    segmented_gpu = cle.masked_voronoi_labeling(selected_spots, binary)

    seg_each_frame = cle.pull(segmented_gpu)
    wtGCB_seg_.append(seg_each_frame)

wtGCB_seg_ = np.array(wtGCB_seg_)

mtGCB_seg_ = []
for each_frame in tqdm(mtGCB):
    img_gpu = cle.push(each_frame)
    # backgrund_subtracted = cle.top_hat_box(img_gpu, radius_x=5, radius_y=5, radius_z=2)
    #segmented_gpu = cle.voronoi_otsu_labeling(img_gpu, spot_sigma=2.7, outline_sigma=0.01)

    blurred = cle.gaussian_blur(img_gpu, sigma_x=spot_sigma, sigma_y=spot_sigma,
                                sigma_z=spot_sigma * (um_per_pixel / um_per_zsice))
    detected_spots = cle.detect_maxima_box(blurred, radius_x=spot_sigma, radius_y=spot_sigma,
                                           radius_z=spot_sigma * (um_per_pixel / um_per_zsice))

    blurred = cle.gaussian_blur(img_gpu, sigma_x=outline_sigma, sigma_y=outline_sigma,
                                sigma_z=outline_sigma * (um_per_pixel / um_per_zsice))
    binary = cle.threshold_otsu(blurred)

    selected_spots = cle.binary_and(binary, detected_spots)
    segmented_gpu = cle.masked_voronoi_labeling(selected_spots, binary)

    seg_each_frame = cle.pull(segmented_gpu)
    mtGCB_seg_.append(seg_each_frame)

mtGCB_seg_ = np.array(mtGCB_seg_)

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(mtGCB,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), colormap='blue', opacity=1)
viewer.add_labels(mtGCB_seg, scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='mt')
viewer.add_labels(mtGCB_seg_, scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), name='mt_')