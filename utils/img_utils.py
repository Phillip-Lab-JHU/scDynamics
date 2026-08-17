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
"""Functions for image processing"""

import numpy as np
import skimage
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from utils.misc_utils import compute_distance_matrix
import os
from tqdm import tqdm
import tifffile

def normalize_zstack(z_stack):
    if z_stack.dtype == 'uint8':
        normalized_imgs = (z_stack - np.min(z_stack)) / np.max(z_stack) * 255
        normalized_imgs = normalized_imgs.astype(np.float64)
        normalized_imgs = normalized_imgs / 255.0

    elif z_stack.dtype == 'uint16':
        normalized_imgs = (z_stack - np.min(z_stack)) / np.max(z_stack) * 65535
        normalized_imgs = normalized_imgs.astype(np.float64)
        normalized_imgs = normalized_imgs / 65535.0

    elif z_stack.dtype == 'float':
        normalized_imgs = (z_stack - np.min(z_stack)) / np.max(z_stack)
        normalized_imgs = normalized_imgs.astype(np.float64)

    return normalized_imgs


def interpolate_zstacks(z_stack, add=1, method='linear'):
    z, y, x = z_stack.shape
    xgrid = np.linspace(0, (x - 1), x)  # [0, 1, 2, ... 1431]
    ygrid = np.linspace(0, (y - 1), y)  # [0, 1, 2, ... 1431]
    zgrid = np.linspace(0, (z - 1), z)  # [0, 1, 2, ... 25]

    interpolator = RegularGridInterpolator((zgrid, ygrid, xgrid), z_stack, method=method, bounds_error=False,
                                           fill_value=None)

    new_z_grid = np.linspace(0, z - 1, z * (add + 1) - add)  # For add=1, [0, 0.5, 1, 1.5, ... 25]
    Zgrid, Ygrid, Xgrid = np.meshgrid(new_z_grid, ygrid, xgrid, indexing='ij')

    interpolated = interpolator((Zgrid, Ygrid, Xgrid))

    return interpolated


def postprocess_segmented_zstack(segmented, volume_thresh):
    ''' Apply volume filter (to remove small objects) + clear objects touching borders for segmented Z stack
        Parameters:
        ----------
        segmented: np.array()
            3D array that represents segmented Z stack
        volume_thresh: float
            Volume threshold that removes the object that has volume less than this value

        Returns:
        -------
        relabeled: np.array()
            3D array that represents relabeled segmented Z stack
        '''

    volume_array = np.array([len(segmented[segmented == each]) for each in np.unique(segmented)])
    # for label 0, 1, 2, 3, ... quantify corresponding volumes [104402869, 876, 16, 28, ...]
    removing_idxs = np.where(volume_array < volume_thresh)[0]  # Return labels that are less than threshold [2, 3, ...]
    mask = np.isin(segmented, removing_idxs)  # boolean matrix with same shape as segmented
    import copy
    new_segmented = copy.copy(segmented)
    new_segmented[mask] = 0

    segmented_cleared = skimage.segmentation.clear_border(new_segmented)  # Remove objects that touch the border

    relabeled, fw_map, inv_map = skimage.segmentation.relabel_sequential(segmented_cleared)
    # np.array([1, 1, 5, 5, 8, 99, 42]) -> np.array([1, 1, 2, 2, 3, 5, 4]): assign elements in order starting from 1

    return relabeled


def remove_nearby_objects(segmented, thresh=None):
    ''' Remove objects that are nearby each other
    Parameters:
    ----------
    segmented: np.array()
        3D array that represents segmented Z stack
    thresh: float
        Distance threshold that removes the object that has pairwise distance less than this value

    Returns:
    -------
    relabeled: np.array()
        3D array that represents relabeled segmented Z stack
    '''

    df = pd.DataFrame(skimage.measure.regionprops_table(segmented, properties=['label', 'centroid', 'area']))
    df.rename(columns={'centroid-0': 'z', 'centroid-1': 'y', 'centroid-2': 'x', 'area': 'volume', }, inplace=True)

    if thresh == None:
        vol = np.array(df['volume'])  # shape = (N, )

        R = (3 * vol / (4 * np.pi)) ** (1 / 3)  # Equivalent R of Sphere
        R = R[:, np.newaxis]  # shape = (N, 1)
        R_matrix = R + R.T  # shape = (N , N)

        R_matrix = np.tril(R_matrix, k=-1)  # Get only lower triangle (k=-1, remove diagonal elements), shape = (N , N)

        thresh = np.max(R_matrix)

    X = np.array(df[['z', 'y', 'x']])  # N = 128 data
    dist_matrix = compute_distance_matrix(X, X)  # shape = (N, N)
    dist_matrix_low_triangle = np.tril(dist_matrix,
                                       k=-1)  # Get only lower triangle (k=-1, remove diagonal elements), shape = (N , N)

    inf_triu_matrix = np.triu(np.full(dist_matrix_low_triangle.shape, np.inf))
    # Upper triangle matrix including diagonal with np.inf value, shape = (N, N)

    remove_idx_row_col = np.where((dist_matrix_low_triangle + inf_triu_matrix) < thresh)
    # [ [row_idx1, row_idx2, ...]. [col_idx1, col_idx2, ...] ]

    removing_idxs = np.unique(np.concatenate([remove_idx_row_col[0], remove_idx_row_col[1]]))
    # [ unique_idx1, unique_idx2, ... ]

    mask = np.isin(segmented, removing_idxs)  # boolean matrix with same shape as segmented
    import copy
    new_segmented = copy.copy(segmented)
    new_segmented[mask] = 0

    relabeled, fw_map, inv_map = skimage.segmentation.relabel_sequential(new_segmented)
    # np.array([1, 1, 5, 5, 8, 99, 42]) -> np.array([1, 1, 2, 2, 3, 5, 4]): assign elements in order starting from 1

    return relabeled


def find_plane_in_focus(z_stack):
    ''' Find z-plane that are in focus
    Parameters:
    ----------
    z_stack: np.array()
        3D array that represents cropped z stack ex) shape = (20, 32, 32)

    Returns:
    -------
    z_disp: np.array()
        1D array that has position relative to plane in focus
        ex) z_disp = np.array([-3, -2, -1, 0(plane in focus), 1, 2])
    '''

    covs = []
    for each_z in z_stack:
        cov = np.var(each_z) / np.mean(each_z)
        covs.append(cov)

    ref_z = np.argmin(covs)  # Plane in focus = minimum variance of intensity

    z_slices = np.arange(0, z_stack.shape[0])
    z_disp = z_slices - ref_z

    return z_disp


def croppedstacks_and_zdisps(img, seg, size):
    ''' Remove objects that are nearby each other
    Parameters:
    ----------
    img: np.array()
        3D array that represents original z stack, shape = (z, y, x)
    seg: np.array()
        3D array that represents segmented z stack ex) shape = (z, y, x)
    size: array-like obj
        size of cropped stacks, size = (z_size, y_size, x_size)

    Returns:
    -------
    cropped_imgs: np.array()
        array that cropped original z_stacks, shape = (N, z_size, y_size, x_size)
    cropped_segs: np.array()
        array that cropped segmented z_stacks, shape = (N, z_size, y_size, x_size)
    z_disps: np.array()
        array that has position relative to plane in focus,. shape = (N, z_size)

    '''

    z_size, y_size, x_size = size
    z_max, y_max, x_max = img.shape

    df = pd.DataFrame(skimage.measure.regionprops_table(seg, img, properties=['label', 'centroid']))
    df.rename(columns={'centroid-0': 'z', 'centroid-1': 'y', 'centroid-2': 'x'}, inplace=True)

    cropped_imgs = []
    cropped_segs = []
    z_disps = []

    for idx in range(df.shape[0]):
        each_cell = df.iloc[idx, :]
        z_center = int(np.round(each_cell['z']))
        y_center = int(np.round(each_cell['y']))
        x_center = int(np.round(each_cell['x']))

        if (z_center - z_size // 2 <= 0) or (z_center + z_size // 2 >= z_max) or (y_center - y_size // 2 <= 0) or (
                y_center + y_size // 2 >= y_max) or (x_center - x_size // 2 <= 0) or (x_center + x_size // 2 >= x_max):
            # Remove the cropped images that the cropped size go beyond the actual image size

            # print(idx, each_cell['label'])
            pass
        else:
            cropped_img = img[z_center - z_size // 2:z_center + z_size // 2,
                          y_center - y_size // 2:y_center + y_size // 2, x_center - x_size // 2:x_center + x_size // 2]
            cropped_seg = seg[z_center - z_size // 2:z_center + z_size // 2,
                          y_center - y_size // 2:y_center + y_size // 2, x_center - x_size // 2:x_center + x_size // 2]
            z_disp = find_plane_in_focus(cropped_img)

            cropped_imgs.append(cropped_img)
            cropped_segs.append(cropped_seg)
            z_disps.append(z_disp)

    cropped_imgs = np.array(cropped_imgs)
    cropped_segs = np.array(cropped_segs)
    z_disps = np.array(z_disps)

    return cropped_imgs, cropped_segs, z_disps


def get_hyperstack(path:str, files:str, order, n_zslices, n_frames):

    assert files.size == n_zslices * n_frames, 'number of files and number of zslices * number of frames do not match'

    if order == 'zt':
        change_first = n_zslices
    elif order == 'tz':
        change_first = n_frames

    imgs = []
    for i in tqdm(range(0, len(files), change_first)):
        file_group = files[i:i + change_first]
        imgs_temp = []
        for file in file_group:
            tif = tifffile.TiffFile(path + file)  # Image(np.array) + Metadata
            img = tif.asarray()  # Image(np.array)
            imgs_temp.append(img)

        img_temp_array = np.array(imgs_temp)
        imgs.append(img_temp_array)
    imgs = np.array(imgs)
    # metadata = tif.imagej_metadata

    return imgs


def get_5d_stack(path:str, files:str, order, n_zslices, n_frames, n_channels):

    assert files.size == n_zslices * n_frames * n_channels, 'number of files != number of zslices * number of frames * number of channels'

    if order == 'ztc':
        change_first = n_zslices
        change_second = n_frames

    elif order == 'zct':
        change_first = n_zslices
        change_second = n_channels

    elif order == 'tzc':
        change_first = n_frames
        change_second = n_zslices

    elif order == 'tcz':
        change_first = n_frames
        change_second = n_channels

    elif order == 'czt':
        change_first = n_channels
        change_second = n_zslices

    elif order == 'ctz':
        change_first = n_channels
        change_second = n_frames

    imgs = []
    for t in tqdm(range(0, len(files), change_first*change_second)):
        file_group1 = files[t:t + change_first*change_second]
        imgs_temp1 = []
        for z in range(0, len(file_group1), change_first):
            file_group2 = file_group1[z:z + change_first]
            imgs_temp2 = []
            for file in file_group2:
                tif = tifffile.TiffFile(path + file)  # Image(np.array) + Metadata
                img = tif.asarray()  # Image(np.array)
                imgs_temp2.append(img)
            imgs_temp1.append(imgs_temp2)
        imgs.append(imgs_temp1)
    imgs = np.array(imgs)
    # metadata = tif.imagej_metadata

    return imgs


def bbox_3d(zmax, ymax, xmax):
    zaxis, yaxis, xaxis = np.array([zmax, 0, 0]), np.array([0, ymax, 0]), np.array([0, 0, xmax])
    farpoint = zaxis + yaxis + xaxis

    lines = [
        np.array([[0, 0, 0], zaxis]),
        np.array([[0, 0, 0], yaxis]),
        np.array([[0, 0, 0], xaxis]),
        np.array([zaxis, zaxis + yaxis]),
        np.array([zaxis, zaxis + xaxis]),
        np.array([yaxis, yaxis + zaxis]),
        np.array([yaxis, yaxis + xaxis]),
        np.array([xaxis, xaxis + zaxis]),
        np.array([xaxis, xaxis + yaxis]),
        np.array([farpoint, farpoint - zaxis]),
        np.array([farpoint, farpoint - yaxis]),
        np.array([farpoint, farpoint - xaxis]),
    ]
    return lines


def bbox_2d(ymax, xmax):
    yaxis, xaxis = np.array([ymax, 0]), np.array([0, xmax])
    farpoint = yaxis + xaxis

    lines = [
        np.array([[0, 0], yaxis]),
        np.array([[0, 0], xaxis]),
        np.array([yaxis, yaxis + xaxis]),
        np.array([xaxis, xaxis + yaxis]),
        np.array([farpoint, farpoint - yaxis]),
        np.array([farpoint, farpoint - xaxis]),
    ]
    return lines