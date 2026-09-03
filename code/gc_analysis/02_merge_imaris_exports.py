# Author: Chanhong Min <cmin11@jhmi.edu>

"""Combines the data from Imaris Software"""

path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Imaris csvs\\'
excel_file_name = 'Intravital Data_all'


def imaris_excel_merge(path: str, excel_file_name: str):
    """ Merges multiple Imaris produced Excel sheets into one and save it to designated path

    Parameters:
    ----------
    path: str
        path you want to save the merged file
    excel_file_name: str
        name of the merge Excel/csv file
        e.g.: 'Imaris file.xlsx', 'Imaris file.csv'
    type_list: list
        list of cell types to label(Order is determined by the order of folder)

    Returns:
    -------
    excel_file: excel_file
        merged Excel file
    """
    import os
    import pandas as pd
    from tqdm import tqdm

    exp_groups = next(os.walk(path))[1] #['Exp1', 'Exp2', 'Exp3']
    excel_file = pd.DataFrame()
    for exp_group in tqdm(exp_groups):  # each Exp1, Exp2, Exp3
        exp_folders = next(os.walk(path + exp_group))[1]  # ['2-Bad-D09-A1-ZT1-73-160-FOV230-256px_Statistics', '4-Good-D10-B1-ZT1-45-132-FOV230-256px_Statistics', ...]

        df3 = pd.DataFrame()
        for exp_folder in exp_folders: # each '2-Bad-D09-A1-ZT1-73-160-FOV230-256px_Statistics', ...
            condition_folders = next(os.walk(path + exp_group + '/' + exp_folder))[1] # ['macrophage', 'mt B-cell', 'T-cell', 'wt B-cell']
            #drift_file_name = next(os.walk(path + exp_group + '/' + exp_folder))[2][0]
            #drift_file = pd.read_csv(path + exp_group + '/' + exp_folder + '/' + drift_file_name, skiprows=3)
            print(exp_folder)
            df2 = pd.DataFrame()
            for k, condition_folder in enumerate(condition_folders): # each macrophage, mt B-cell, T-cell', wt B-cell
                file_names = next(os.walk(path + exp_group + '/' + exp_folder + '/' + condition_folder))[2]
                df1 = pd.DataFrame()
                for l, file_name in enumerate(file_names):  # for each feature Excel file
                    if 'FDCfeatures' in file_name:
                        df_temp = pd.read_csv(path + exp_group + '/' + exp_folder + '/' + condition_folder + '/' + file_name)
                        df1 = pd.concat([df1, df_temp[df_temp.columns[1]], df_temp[df_temp.columns[2]], df_temp[df_temp.columns[3]], df_temp[df_temp.columns[4]]], axis=1)
                    else:
                        df_temp = pd.read_csv(path + exp_group + '/' + exp_folder + '/' + condition_folder + '/' + file_name, skiprows=3)

                    if df_temp.columns[0] == 'Position X':
                        df1 = pd.concat([df1, df_temp[df_temp.columns[0]], df_temp[df_temp.columns[1]], df_temp[df_temp.columns[2]]], axis=1)

                    elif any(txt in file_name for txt in ('Overlapped', 'Shortest_Distance')):  # Search whether 'Overlapped', or 'Shortest_Distance' word is in the file_name
                        df_temp.rename(
                            columns={df_temp.columns[0]: '%s' % (file_name[file_name.find('_') + 1: file_name.find('.')])},
                            inplace=True)  # rename column from name of first column to string between _ and .
                        # ex) if file_name = (B-cells)4-Good-D10-B1-ZT1-45-132-FOV230-256px_Overlapped_Volume_Ratio_to_Surfaces_Surfaces=Surfaces_1_T_cells.csv
                        # column name change from Overlapped_Volume_Ratio_to_Surfaces_Surfaces -> Overlapped_Volume_Ratio_to_Surfaces_Surfaces=Surfaces_1_T_cells
                        df1 = pd.concat([df1, df_temp[df_temp.columns[0]]], axis=1)

                    else:
                        df1 = pd.concat([df1, df_temp[df_temp.columns[0]]], axis=1)

                df1 = pd.concat([df_temp['TrackID'], df_temp['Time'], df1], axis=1)
                df1['Label'] = exp_folder + '_' + pd.DataFrame(df_temp['TrackID'].values, dtype='string')
                df1['Type'] = condition_folder
                df1['Video'] = exp_folder
                df1['Exp'] = exp_group
                second_ = exp_folder.find('-', exp_folder.find('-') + 1)
                third_ = exp_folder.find('-', exp_folder.find('-', exp_folder.find('-')+1)+1)
                day = exp_folder[second_+1:third_]
                if day == 'D09':
                    day = 'D9'
                df1['Day'] = day
                df1['Exp_group'] = exp_folder[third_+1:third_+2]

                if exp_folder == '20240211b-0to50-D10-B6L-ZT1-70-147-fov230-256px':
                    df1 = df1[df1['Time']<=50].reset_index(drop=True)
                df2 = pd.concat([df2, df1])
            df3 = pd.concat([df3, df2])
        excel_file = pd.concat([excel_file, df3])

    ######### Replace N/A with 0 for overlapped volumes #########
    dict_overlap = {}
    for column_name in excel_file.columns:
        if 'Overlapped' in column_name:
            dict_overlap[column_name] = 0  # dictionary that has 'Overlapped_sdfsdf' as keys and values as 0
    excel_file = excel_file.fillna(dict_overlap)  # fill N/A with 0 for columns that are overlapping volumes

    excel_file = excel_file.reset_index(drop=True)

    # ######### Replace N/A with 10000 for shortest distance between cells #########
    # dict_shortest = {}
    # for column_name in excel_file.columns:
    #     if 'Shortest_Distance' in column_name:
    #         dict_shortest[column_name] = 10000  # dictionary that has 'Overlapped_sdfsdf' as keys and values as 0
    # excel_file = excel_file.fillna(dict_shortest)  # fill N/A with 0 for columns that are overlapping volumes

    #excel_file_final = excel_file.loc[excel_file['Type'] != 'macrophage']  # Delete the macrophage data, because we only need them for cell-cell interaction
    #macrophage_file = excel_file.loc[excel_file['Type'] == 'macrophage']

    excel_file.to_csv(path + excel_file_name + '.csv', index=False)
    excel_file.to_parquet(path + excel_file_name + '.parquet')

#with pd.ExcelWriter(path + 'Macrophage_file.xlsx') as writer:  # call ExcelWriter function as writer
#    macrophage_file.to_excel(writer, index=False)

    return excel_file


imaris_excel_merge(path, excel_file_name)
