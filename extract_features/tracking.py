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
"""single-cell tracking using trackpy"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte, img_as_float32
import pandas as pd
from tqdm import tqdm
import imageio
from tqdm import tqdm
import trackpy as tp
import cv2


def linking(df, um_per_pix):

    #excel['Time'] = excel['ImageNumber'] - 1
    #df = excel.rename(columns={'AreaShape_Center_X': 'x', 'AreaShape_Center_Y': 'y', 'Time': 'frame'}).copy()
    pred = tp.predict.NearestVelocityPredict()
    # NearestVelocityPredict method use previous velocity to predict position
    # If newly appeared position, use nearby velocity to predict position

    df_linked = pred.link_df(df, search_range=50*um_per_pix, memory=0, adaptive_stop=12, adaptive_step=0.9)
    # Adaptive search: range changes from search_range(30) -> adaptive_stop(12), decrease by 10% (adaptive_step=0.9)
    # df_linked.to_csv(reindexed_excel_path, index=False)

    return df_linked


def verify_linking(df_linked, mask_path, image_num=10, img_format='tif'):
    from PIL import Image

    label_files = next(os.walk(mask_path))[2]
    bool_list = [img_format in ele for ele in label_files]
    filtered_label_files = np.array(label_files)[np.array(bool_list)]
    filtered_label_files.sort()

    for t, file_name in tqdm(enumerate(filtered_label_files[:image_num]), total=len(filtered_label_files[:image_num])):
        #print(file_name)
        image = io.imread(mask_path + file_name)
        image = image.astype(np.uint8)
        cell_temp = df_linked.groupby(['frame']).get_group(t)  # instantaneous cells at time = t
        cell_temp.reset_index(inplace=True, drop=True)

        for index in range(0, cell_temp.shape[0]):
            cv2.putText(image, '%s' % str(cell_temp['particle'][index]),
                        (round(cell_temp['x_pix'][index] + 10), round(cell_temp['y_pix'][index] - 10)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(200, 200, 100), thickness=2)
        image = Image.fromarray(image)
        if not os.path.isdir(mask_path + 'Reindexed/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(mask_path + 'Reindexed/')
        image.save(mask_path + 'Reindexed/%s.%s' % (t, img_format))