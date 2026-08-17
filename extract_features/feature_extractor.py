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
"""Single-cell feature extraction"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte, img_as_float32, segmentation, color, measure, morphology, feature
import cv2
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm # loop에서 progress 보게 해줌
import imutils
import mahotas
from utils.misc_utils import compute_distance_matrix


def Cell_Feature_Extractor(mask_path:str, mask_files:list, binary:bool=True, area_thresh:int=50, buffer_size:int=30,
                           um_per_pix:float=0.568, n_cells:int=5, neighbor_distance:float=20*10, change_to_binary=False) -> pd.DataFrame:
    '''
    Accepts time series of 2D image (Binary) and calculate single-cell morphology and positions
    Parameters
    ----------
    mask_path : str.
        path for masks.
    mask_files : list.
        list of mask file names to read.
    mask_files : bool.
        if True, cells are all 1, if False, each cell has label.
    area_thresh : float.
        remove cells that are below this value (pix**2).
    buffer_size : float.
        distance from the boundary to clear border (pix).
    n_cells: int
        number of nearest cells to calculate pairwise distances
    neighbor_distance: float
        radius to create neighbor boundary in pixels(within this radius is neighbor)
    um_per_pix: float
        pixel to um conversion constant

    Returns
    -------
    df_cell : pandas DataFrame
        containing morphology and position for single-cells.
    '''

    assert n_cells >= 2, 'n_cells should be greater or equal to 2'
    property_list = ['label', 'area', 'perimeter', 'convex_area', 'solidity', 'eccentricity', 'equivalent_diameter',
                     'extent', 'major_axis_length', 'minor_axis_length', 'orientation', 'centroid']

    df_cell = pd.DataFrame()
    for t, mask_file in tqdm(enumerate(mask_files)):

        mask_img = io.imread(mask_path + mask_file)
        mask_img_cleared = segmentation.clear_border(mask_img, buffer_size = buffer_size)


        if change_to_binary == True:
            mask_img_cleared = np.where(mask_img_cleared >= 1, 1, 0)

        if binary == True:  # Apply watershed to the binary mask
            distance = distance_transform_edt(mask_img_cleared)  # Distance transformation by euclidean distance
            # (Compute shortest distance from non-zero(foreground) to zero(background)
            local_max_coords = feature.peak_local_max(distance, min_distance=8)  # Coords for maximum distance
            local_max_mask = np.zeros(distance.shape, dtype=bool)
            local_max_mask[tuple(local_max_coords.T)] = True
            markers = measure.label(local_max_mask)
            segmented = segmentation.watershed(-distance, markers, mask=mask_img_cleared)

        elif binary == False:
            segmented = mask_img_cleared

        properties = measure.regionprops_table(segmented, properties=property_list)
        df_cell_temp = pd.DataFrame(properties)



        df_cell_temp['aspect_ratio'] = df_cell_temp['major_axis_length'] / df_cell_temp['minor_axis_length']
        df_cell_temp['elongation'] = 1 - df_cell_temp['minor_axis_length'] / df_cell_temp['major_axis_length']
        df_cell_temp['compactness'] = df_cell_temp['perimeter']**2 / df_cell_temp['area']
        df_cell_temp['roundness'] = 4*df_cell_temp['area'] / (np.pi*df_cell_temp['major_axis_length']**2)
        df_cell_temp['circularity'] = 4*np.pi*df_cell_temp['area'] / (df_cell_temp['perimeter']**2)
        df_cell_temp['rectangularity'] = df_cell_temp['area'] / (df_cell_temp['major_axis_length']*df_cell_temp['minor_axis_length'])

        df_cell_temp = df_cell_temp[df_cell_temp['area']>=area_thresh].reset_index(drop=True)

        positions = df_cell_temp[['centroid-1', 'centroid-0']].values
        distance_matrix = compute_distance_matrix(positions, positions)

        ##### Number of closest neighbors #####
        neighbor_matrix = distance_matrix <= neighbor_distance  # About 10 x cell_length  # [False, True, ...]
        n_neighbors = np.sum(neighbor_matrix, axis=1) - 1  # Remove self
        df_cell_temp['n_neighbors'] = n_neighbors

        #n_neighbor_idxs = [np.where(row)[0] for row in neighbor_matrix]

        ##### Shortest centroid distance #####
        nearby_cellidx_order = np.argsort(distance_matrix, axis=1)  # nth row: order of cell idx that are close to cell idx n

        closest_centroid_idxs = nearby_cellidx_order[:, :n_cells+1]  # [:, 0] = self, # [:, 1] = 1st nearest cell, ...

        closest_centroid_distances_all = []
        for i in range(1, n_cells+1):
            closest_centroid_distances = distance_matrix[closest_centroid_idxs[:,0], closest_centroid_idxs[:,i]]
            closest_centroid_distances_all.append(closest_centroid_distances)
        closest_centroid_distances_all = np.array(closest_centroid_distances_all).T  # nth row: order of distance that are close to cell idx n
        #  ex. [:, 0] = 1st closest centroid distance, [:, 1] = 2nd closest centroid distance, ...
        df_cell_temp['shortest_distance'] = closest_centroid_distances_all[:,0]
        df_cell_temp['avg_shortest_distance'] = np.mean(closest_centroid_distances_all, axis=1)

        ##### Shortest surface distance #####
        # filtered_idxs = df_cell_temp['label'].values
        # filtered_segmented = segmented * np.isin(segmented, filtered_idxs)

        #shortest_distances = []
        #zernike_moments_list = []
        #haralick_features_list = []
        # for idx, label in enumerate(np.unique(df_cell_temp['label'])):
        #
        #     min_idxs = np.argsort(distance_matrix[idx])[1:6]
        #     min_labels = min_idxs + 1
        #
        #     cell_mask = np.uint8(segmented == label)  # Binarize current segmented mask
        #     other_cells_mask = np.uint8( np.isin(segmented, min_labels) )  # # Binarize neighbor segmented mask
        #
        #     #other_cells = np.uint8( (segmented != label) & (segmented != 0) )
        #
        #     cell_mask_positions = np.argwhere(cell_mask == 1)  # (n_points, 2)
        #     other_cells_positions = np.argwhere(other_cells_mask == 1)  # (n_points, 2)
        #
        #     each_distance_matrix = compute_distance_matrix(cell_mask_positions, other_cells_positions)
        #
        #     if np.min(each_distance_matrix) <= np.sqrt(2)+0.001:  # Touching cells
        #         shortest_distance = 0
        #     else:
        #         shortest_distance = np.min(each_distance_matrix) * um_per_pix
        #
        #     shortest_distances.append(shortest_distance)
        #
        #     #n_neighbor_idx = other_labels[neighbor_cells]
        #     #n_neighbor_idxs.append(n_neighbor_idx)
        #
        #     # #################### Zernike moments & Haralick Features ####################
        #     # cell_contour = cv2.findContours(cell_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #     # cell_contour = imutils.grab_contours(cell_contour) # opencv 버젼에 따라 findContour 함수 달라지는거 호환해줌
        #     #
        #     # for c in cell_contour:
        #     #     mask = np.zeros((segmented.shape), dtype="uint8")
        #     #     # empty mask 생성
        #     #     cv2.drawContours(mask, [c], -1, 255, -1) # 꼭 필요한 line, 빈 mask에 cell_contour 그리기
        #     #     # cell_mask와 동일한 모양이지만 cell_mask는 array이고 mask는 contour임
        #     #     (x, y, w, h) = cv2.boundingRect(c)
        #     #
        #     #     if (w < 7) or (h < 7):   #  Sometimes if mask has discontinuity, it creates one big contour and one small contour
        #     #         # Remove small contour
        #     #         print(label)
        #     #         continue
        #     #
        #     #     else:
        #     #         cropped_segmented = mask[y:y + h, x:x + w] # bounding rectangle을 모서리로 하여 이미지 crop
        #     #         center, radius = cv2.minEnclosingCircle(c) # center = (x,y) = (column, row), 가장 작은 bounding circle
        #     #         zernike_moments = mahotas.features.zernike_moments(cropped_segmented, radius, degree=9)
        #     #         # By default use center of mass of the cropped image
        #     #         # use radius from smallest bounding circle
        #     #         # degree 9 -> 30 zernike_moments, degree 26 -> 195 zernike_moments, degree 50 -> 676 moments
        #     #         # 0th zernike moment(A00) = 1/pi
        #     #
        #     #         #cropped_original = (bf_img*cell_mask)[y:y + h, x:x + w]
        #     #         #haralick_features = np.mean(mahotas.features.haralick(cropped_original, ignore_zeros=False),axis=0)
        #     #         zernike_moments_list.append(zernike_moments)
        #     #         #haralick_features_list.append(haralick_features)
        #
        # shortest_distances = np.array(shortest_distances)

        # zernike_moments_list = np.array(zernike_moments_list)
        #haralick_features_list = np.array(haralick_features_list)

        #concat_zernike = pd.DataFrame()
        # concat_haralick = pd.DataFrame()
        #
        # haralick_features_name = ['angular_second_moment', 'contrast','correlation','variance','inverse_difference_moment',
        #                           'sum_average','sum_variance','sum_entropy','entropy','difference_variance',
        #                           'difference_entropy','information_measures_of_correlation_1','information_measures_of_correlation_2']
        # Haralick, et. al., 1973, "Textural Features for Image Classification,"" IEEE Transactions on Systems

        # when concatenating empty dataframe there is dimension difference -> error -> try/except
        # try:
        #
        #     for i in range(0, zernike_moments_list.shape[1]):
        #         df_zernike_moments = pd.DataFrame(zernike_moments_list[:,i], columns=['zernike_moments_%s' % str(i)])
        #         concat_zernike = pd.concat([concat_zernike, df_zernike_moments], axis = 1)
        #
        #     # for j in range(0, haralick_features_list.shape[1]):
        #     #     df_haralick_features = pd.DataFrame(haralick_features_list[:,j], columns=[haralick_features_name[j]])
        #     #     concat_haralick = pd.concat([concat_haralick, df_haralick_features], axis = 1)
        #
        # except:
        #     pass

        # df_cell_temp = pd.concat([df_cell_temp, concat_zernike], axis = 1)

        df_cell_temp['shortest_distance'] = df_cell_temp['shortest_distance'] * (um_per_pix)
        df_cell_temp['avg_shortest_distance'] = df_cell_temp['avg_shortest_distance'] * (um_per_pix)
        df_cell_temp['area'] = df_cell_temp['area']*(um_per_pix**2)
        df_cell_temp['perimeter'] = df_cell_temp['perimeter'] * (um_per_pix)
        df_cell_temp['convex_area'] = df_cell_temp['convex_area'] * (um_per_pix ** 2)
        df_cell_temp['equivalent_diameter'] = df_cell_temp['equivalent_diameter'] * (um_per_pix)
        df_cell_temp['major_axis_length'] = df_cell_temp['major_axis_length'] * (um_per_pix)
        df_cell_temp['minor_axis_length'] = df_cell_temp['minor_axis_length'] * (um_per_pix)

        df_cell_temp['x_pix'] = df_cell_temp['centroid-1']
        df_cell_temp['y_pix'] = df_cell_temp['centroid-0']
        df_cell_temp['x'] = df_cell_temp['centroid-1'] * (um_per_pix)
        df_cell_temp['y'] = df_cell_temp['centroid-0'] * (um_per_pix)
        df_cell_temp['frame'] = t
        df_cell_temp['file_name'] = mask_file
        #df_cell_temp['UCI'] = df_cell_temp['file_name'].str.split('_Merging').str[0]
        df_cell_temp = df_cell_temp.drop(['centroid-1'], axis=1)
        df_cell_temp = df_cell_temp.drop(['centroid-0'], axis=1)

        df_cell = pd.concat([df_cell, df_cell_temp], axis=0)

    df_cell = df_cell.reset_index(drop=True)

    return df_cell




# label = 9
# cell_mask = np.uint8(segmented == label) # Binarize instant segmented mask
# other_cells = np.uint8( (segmented != label) & (segmented != 0) )
#
# plt.figure(figsize=(20,20))
# plt.imshow(filtered_segmented, cmap='gray')
# #plt.imshow(color.label2rgb(segmented, bg_label=0))
# plt.show()
# plt.close()
# plt.clf()
#
# plt.figure(figsize=(20,20))
# plt.imshow(other_cells_mask, cmap='gray')
# #plt.imshow(color.label2rgb(segmented, bg_label=0))
# plt.show()
# plt.close()
# plt.clf()