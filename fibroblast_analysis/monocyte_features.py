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
"""Implementing feature extraction for monocyte data"""

from extract_features.feature_extractor import Cell_Feature_Extractor
from extract_features.tracking import linking, verify_linking
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

###################### Get training dataset #########################
path = r"\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\images\\"
csv_path = r"\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\csv_files\\"

patient_info = pd.read_excel(path+'Patient_info.xlsx')
age_groups = next(os.walk(path))[1] # ['Young', 'Old', 'Frail']

excel_file = pd.DataFrame()
for age_group in age_groups:
    patient_folders = next(os.walk(path + age_group))[1]  # ['F659', 'F1044', ...]

    df3 = pd.DataFrame()
    for patient_folder in patient_folders:
        condition_folders = next(os.walk(path + age_group + '/' + patient_folder))[1]  # ['Control', 'DNA', ...]

        df2 = pd.DataFrame()
        for k, condition_folder in enumerate(condition_folders):
            if 'Rep' in condition_folder:
                continue
            mask_path = path + age_group + '/' + patient_folder + '/' + condition_folder + '/CellProfiler Segmented Masks/'
            print(age_group, patient_folder, condition_folder)
            files = next(os.walk(mask_path))[2]
            bool_list = ['tif' in ele for ele in files]
            mask_files = np.array(files)[np.array(bool_list)]
            mask_files.sort()

            df1 = Cell_Feature_Extractor(mask_path=mask_path, mask_files=mask_files, binary=True, area_thresh=50, buffer_size=30,
                                         um_per_pix=0.568, n_cells=5, neighbor_distance=20*10)
            df1_linked = linking(df1, um_per_pix=0.568)
            verify_linking(df1_linked, mask_path=mask_path, image_num=10, img_format='tif')


            df1_linked['Condition'] = condition_folder
            df1_linked['Patient'] = patient_folder
            df1_linked['Type'] = age_group

            df1_linked['Age'] = patient_info[patient_info['ID'] == patient_folder]['Age'].values[0]
            df1_linked['Sex'] = patient_info[patient_info['ID'] == patient_folder]['Sex'].values[0]

            df1_linked.to_csv(csv_path+'%s_%s_%s.csv'%(age_group, patient_folder, condition_folder))

            df2 = pd.concat([df2, df1_linked], axis=0)
        df3 = pd.concat([df3, df2], axis=0)

    excel_file = pd.concat([excel_file, df3], axis=0)

excel_file = excel_file.reset_index(drop=True)

#excel_file.to_csv(path + 'monocyte_df.csv', index=False)
#excel_file.to_parquet(path + 'monocyte_df.parquet')

###################### Get testing dataset (replicates) #########################

path = r"\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\images\\"
csv_path = r"\\philliplab-server.wse.jhu.edu\data\Charles\Motility\Low Density (25k cells)\0.5 Gel\rep_csv_files\\"

patient_info = pd.read_excel(path+'Patient_info.xlsx')
age_groups = next(os.walk(path))[1] # ['Young', 'Old', 'Frail']

excel_file = pd.DataFrame()
for age_group in age_groups:
    patient_folders = next(os.walk(path + age_group))[1]  # ['F659', 'F1044', ...]

    df3 = pd.DataFrame()
    for patient_folder in patient_folders:
        condition_folders = next(os.walk(path + age_group + '/' + patient_folder))[1]  # ['Control', 'DNA', ...]

        df2 = pd.DataFrame()
        for k, condition_folder in enumerate(condition_folders):
            if 'Rep' in condition_folder:
                rep_condition_folders = next(os.walk(path + age_group + '/' + patient_folder + '/Replicates/'))[1]  # ['Control', 'DNA', ...]
                for rep_condition_folder in rep_condition_folders:
                    mask_path = path + age_group + '/' + patient_folder + '/Replicates/' + rep_condition_folder + '/CellProfiler Segmented Masks/'

                    files = next(os.walk(mask_path))[2]
                    bool_list = ['tif' in ele for ele in files]
                    mask_files = np.array(files)[np.array(bool_list)]
                    mask_files.sort()
                    print(age_group, patient_folder, rep_condition_folder, ',frames: ', mask_files.size)

                    df1 = Cell_Feature_Extractor(mask_path=mask_path, mask_files=mask_files, binary=True, area_thresh=50, buffer_size=30,
                                                 um_per_pix=0.568, n_cells=5, neighbor_distance=20*10)
                    df1_linked = linking(df1)
                    verify_linking(df1_linked, mask_path=mask_path, image_num=10, img_format='tif')


                    df1_linked['Condition'] = rep_condition_folder
                    df1_linked['Patient'] = patient_folder
                    df1_linked['Type'] = age_group

                    df1_linked['Age'] = patient_info[patient_info['ID'] == patient_folder]['Age'].values[0]
                    df1_linked['Sex'] = patient_info[patient_info['ID'] == patient_folder]['Sex'].values[0]

                    df1_linked.to_csv(csv_path+'rep_%s_%s_%s.csv'%(age_group, patient_folder, rep_condition_folder))