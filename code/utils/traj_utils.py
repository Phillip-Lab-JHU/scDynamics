# Author: Chanhong Min <cmin11@jhmi.edu>

"""Functions for processing trajectory data"""

from utils.misc_utils import *

def to_trajectory_duration(df, duration=10, condition_name='Type', frame_name='Time', label_name='Label', verbose=True):
    ''' Generate trajectories that are spliced by specific duration (applied to trajectories with varied duration)
    Parameters:
    ----------
    df: pd.DataFrame()
        raw dataframe
    duration: int
        number of time frames to generate cell trajectories with consistent frames
    condition_name: str
        name of the column to be grouped

    Returns:
    -------
    traj_data: pd.DataFrame()
        generate spliced trajectory dataframe with additional Time_span, pseudo_TrackID, pseudo_Time column
    '''

    label_data = df.groupby([condition_name, label_name]).apply(lambda x: x.name)  # contain (cell type, TrackID) tuple

    traj_data = pd.DataFrame()
    time_list = []

    for traj_idx in tqdm(range(0, label_data.shape[0])):  # For each cell trajectory(time 1~t)

        traj_data_temp = df.groupby([condition_name, label_name]).get_group(label_data.iloc[traj_idx]).copy()
        # traj_data_temp= PC1, PC2, feature1, feature 2, ... data for each cell trajectory(time 1~t)
        traj_data_temp.reset_index(inplace=True, drop=True)
        traj_data_temp['Time_span'] = traj_data_temp.shape[0]
        time_list.append(traj_data_temp.shape[0])

        if traj_data_temp.shape[0] < duration:  # discard cell that are not tracked long enough
            continue
        if traj_data_temp.shape[0] >= duration:
            for j in range(0, df[frame_name].max() // duration):  # 181 // 20 = 9 (quotient) -> j = 0~8,
                if traj_data_temp.shape[0] // duration > j:  # ex) time_span = 31, then 31//10 = 3, so loop only from j = 0~2   time_span = 181, 181//20 = 9, so j = 0~8
                    new_traj_data = traj_data_temp[:][duration * j:duration * (j+1)]  # df[:][0~10] = row 0~9,df[:][10~20] = row 10~19, ...
                    new_traj_data['pseudo_%s' % label_name] = np.array(pd.DataFrame(traj_data_temp[:][duration * j:duration * (j+1)][label_name].values, dtype='string') + '_%s' % j).flatten()
                    new_traj_data['pseudo_%s' % frame_name] = np.array(range(0, duration))
                    traj_data = pd.concat([traj_data, new_traj_data])

    traj_data.reset_index(inplace=True, drop=True)

    if verbose:
        plt.figure()
        plt.hist(np.array(time_list), bins=80)
        plt.title("Time Histogram")
        plt.show()

        print('total number of cells trajectories: ', label_data.shape[0])
        print('number of cell trajectories more than %s frames: ' % duration, len(traj_data[traj_data['Time_span'] >= duration].groupby([condition_name, label_name]).apply(lambda x: x.name)))
        print('number of generated cell trajectories by %s frames duration: ' % duration, traj_data.groupby([condition_name, 'pseudo_%s' % label_name]).apply(lambda x: x.name).shape[0])

    return traj_data

def to_trajectory_variable_duration(df, min_duration=40, condition_name='Type', label_name='Label'):
    label_data = df.groupby([condition_name, label_name]).apply(lambda x: x.name)  # contain (cell type, TrackID) tuple
    traj_data = pd.DataFrame()

    for traj_idx in tqdm(range(0, label_data.shape[0])):  # For each cell trajectory(time 1~t)
        traj_data_temp = df.groupby([condition_name, label_name]).get_group(label_data[traj_idx]).copy()
        # traj_data_temp= PC1, PC2, feature1, feature 2, ... data for each cell trajectory(time 1~t)
        traj_data_temp.reset_index(inplace=True, drop=True)
        traj_length = traj_data_temp.shape[0]
        traj_data_temp['Time_span'] = traj_length
        if traj_length >= min_duration:
            traj_data = pd.concat([traj_data, traj_data_temp])

        elif traj_length < min_duration:  # discard cell that are not tracked long enough
            continue

    traj_data.reset_index(inplace=True, drop=True)

    return traj_data

def to_trajectory_duration_fast(df, duration, cut_duration, frame_name='Time', label_name='Label'):
    ''' Generate trajectories that are spliced by specific cut_duration (only for trajectories with same duration)
    Parameters:
    ----------
    df: pd.DataFrame()
        raw dataframe
    duration: int
        number of time frames that original trajectories have
    cut_duration: int
        number of time frames to splice trajectories by 'cut_duration'
    frame_name: str
        name of the column for time frame
    label_name: str
        name of the column for trajectory id

    Returns:
    -------
    traj_data: pd.DataFrame()
        generate spliced trajectory dataframe with additional pseudo_Label, pseudo_Time column
    '''
    traj_data = pd.DataFrame()
    for traj_idx in tqdm(range(0, int(df.shape[0]/duration))):  # For each cell trajectory(time 1~t)
        traj_data_temp = df[duration * traj_idx:duration * (traj_idx + 1)]
        if traj_data_temp.shape[0] < cut_duration:  # discard cell that are not tracked long enough
            continue
        if traj_data_temp.shape[0] >= cut_duration:
            for j in range(1, duration // cut_duration+1):  # 49 // 24 = 2 (quotient) -> j = 1~2
                new_traj_data = traj_data_temp[:][cut_duration * (j - 1):cut_duration * j]  # df[:][0~10] = row 0~9,df[:][10~20] = row 10~19, ...
                new_traj_data['pseudo_%s' % label_name] = np.array(
                    pd.DataFrame(traj_data_temp[:][cut_duration * (j - 1):cut_duration * j][label_name].values,dtype='string') + '_%s' % j)
                new_traj_data['pseudo_%s' % frame_name] = np.array(range(1, cut_duration + 1))
                traj_data = pd.concat([traj_data, new_traj_data])
    traj_data = traj_data.reset_index(drop=True)
    return traj_data

def to_timeseries(df, condition_name='Type', label_name='pseudo_Label', feature_name=['PC1', 'PC2']):
    label_data = df.groupby([condition_name, label_name]).apply(lambda x: x.name)  # contain (cell type, TrackID) tuple
    traj_list = []
    time_series = []
    time_series_dict = {}
    for traj_idx in tqdm(range(0, label_data.shape[0])):  # For each cell trajectory
        traj_data_temp = df.groupby([condition_name, label_name]).get_group(label_data[traj_idx]).copy()
        traj_list.append(traj_data_temp)
        time_series.append(traj_data_temp[feature_name].values)  # np.array with [PC1, PC2] at t=0, [PC1, PC2] at t=1, ... [PC1, PC2] at t= T (shape = (time frames, 2) )
        time_series_dict[traj_idx] = traj_data_temp[feature_name].values
    time_series = np.array(time_series) # time_series = np.array with shape (number of traj,number of frames, dimension = 2 or 3)
    return traj_list, time_series, time_series_dict

def to_timeseries_fast(df, duration, feature_name=['PC1', 'PC2']):
    traj_list = []
    time_series_list = []
    time_series_dict = {}
    for traj_idx in range(int(df.shape[0]/duration)):  # For each cell trajectory
        traj_data_temp = df[duration*traj_idx:duration*(traj_idx+1)]
        traj_list.append(traj_data_temp)
        time_series_list.append(traj_data_temp[feature_name].values)  # np.array with [PC1, PC2] at t=0, [PC1, PC2] at t=1, ... [PC1, PC2] at t= T (shape = (time frames, 2) )
        time_series_dict[traj_idx] = traj_data_temp[feature_name].values
    time_series_array = np.array(time_series_list) # time_series = np.array with shape (number of traj,number of frames, dimension = 2 or 3)
    return traj_list, time_series_array, time_series_dict

def to_timeseries_variable_duration(df:pd.DataFrame, length_name:str, coord_name:list=['x', 'y', 'z'], equal_length:bool=False, frame_name:str=None) -> dict:
    trajectories = {}
    traj_idx = 0
    i0 = 0

    if equal_length == False:

        for i in range(0, df.shape[0]):
            if (i == 0) or (i == duration + i0):
                duration = df[length_name][i]
                traj_data_temp = df[i: duration + i]
                trajectories[traj_idx] = traj_data_temp[coord_name].values
                traj_idx = traj_idx + 1
                i0 = i
            else:
                continue

    if equal_length==True:  # If you want the output traj length to be equal filled with np.nan

        dim = len(coord_name)
        max_length = np.max(df[length_name])

        for i in range(0, df.shape[0]):
            if (i == 0) or (i == duration + i0):
                duration = df[length_name][i]
                traj_data_temp = df[i: duration + i]
                n_frames = traj_data_temp.shape[0]

                if n_frames == max_length:
                    trajectories[traj_idx] = traj_data_temp[coord_name].values
                else:
                    traj = np.full(shape=(max_length, dim), fill_value=np.nan)
                    time_idxs = traj_data_temp[frame_name].values.astype(np.uint8)
                    traj[time_idxs, :] = traj_data_temp[coord_name].values
                    trajectories[traj_idx] = traj

                traj_idx = traj_idx + 1
                i0 = i
            else:
                continue

    return trajectories


def register_traj_freq(trajectories):
    registered_trajectories = {}
    for traj_idx in trajectories:
        traj = trajectories[traj_idx]
        traj_avg = np.mean(traj, axis=0)  # shape = (2, )
        traj_centered = traj - np.tile(traj_avg, (traj.shape[0], 1))  # np.tile -> repeats X_avg by number given(shape = (10000,2)) so Transpose = (2,10000)
        U, S, VT = np.linalg.svd(traj_centered / np.sqrt(traj.shape[0]))

        #print('first singular value: ', S[0])
        #print('second singular value: ', S[1])
        #print('third singular value: ', S[2])

        #print('principal axis 1: ', VT[0, :])  # orthogonal, normal vector (size = 1)
        #print('principal axis 2: ', VT[1, :])  # orthogonal, normal vector (size = 1)
        #print('principal axis 3: ', VT[2, :])  # orthogonal, normal vector (size = 1)

        C = []
        for i in range(traj.shape[0]):
            C.append(np.linalg.inv(VT.T) @ traj[i, :]) # apply rotation matrix by primary axis
        C = np.array(C) # trajectory coordinates that are in primary axis space, but not yet centered to origin
        registered_traj = C - np.tile(C[0], (traj.shape[0], 1)) # starting point(C[0]) to origin
        registered_trajectories[traj_idx] = registered_traj
    return registered_trajectories

def register_traj_disp(trajectories):
    def calc_max_distance(traj):
        all_distance_list = []
        for t in range(1, traj.shape[0]):
            distance = traj[t:] - traj[:-t]
            all_distance_list.append(max(abs(distance)))
        return max(all_distance_list)

    dim = trajectories[0].shape[1]
    registered_trajectories = {}
    for traj_idx in trajectories:
        traj = trajectories[traj_idx]

        max_dist, arg, tlag_max = -1, -1, -1
        for tlag in range(1, traj.shape[0]):
            dxyz = traj[tlag:] - traj[:-tlag]  # Displacement between two nearby points
            avg = np.ones((len(traj[:, 0]), 1)) * np.mean(traj, axis=0)
            xyr = traj - avg

            # determine the rotational matrix
            u, s, rotational_matrix = np.linalg.svd(dxyz)
            rotational_matrix = rotational_matrix.T

            # project major axis of trajectories onto rotational matrix
            xyr_r = xyr @ rotational_matrix
            if dim == 3:
                x = xyr_r[:, 0]
                y = xyr_r[:, 1]
                z = xyr_r[:, 2]
                list_dist = [calc_max_distance(x), calc_max_distance(y), calc_max_distance(z)]
                dist = max(list_dist)
                arg = np.argmax(list_dist)
                # print(tlag, calc_max_distance(x), calc_max_distance(y), calc_max_distance(z))
                if dist > max_dist:
                    max_dist = dist
                    max_arg = arg
                    tlag_max = tlag
                    list_dist[max_arg] = -1
                    max_arg2 = np.argmax(list_dist)
                    list_dist[max_arg2] = -1
                    max_arg3 = np.argmax(list_dist)
            elif dim == 2:
                x = xyr_r[:, 0]
                y = xyr_r[:, 1]
                list_dist = [calc_max_distance(x), calc_max_distance(y)]
                dist = max(list_dist)
                arg = np.argmax(list_dist)
                # print(tlag, calc_max_distance(x), calc_max_distance(y), calc_max_distance(z))
                if dist > max_dist:
                    max_dist = dist
                    max_arg = arg
                    tlag_max = tlag
                    list_dist[max_arg] = -1
                    max_arg2 = np.argmax(list_dist)
            # print(tlag, max_dist, max_arg, max_arg2, max_arg3)

        # print(tlag_max, max_dist, max_arg, max_arg2, max_arg3)

        dxyz = traj[tlag_max:] - traj[:-tlag_max]  # Displacement between two nearby points
        avg = np.ones((len(traj[:, 0]), 1)) * np.mean(traj, axis=0)
        xyr = traj - avg

        # determine the rotational matrix
        u, s, rotational_matrix = np.linalg.svd(dxyz)
        rotational_matrix = rotational_matrix.T

        # project major axis of trajectories onto rotational matrix
        rotated_traj = xyr @ rotational_matrix
        if dim == 3:
            pc1 = rotated_traj[:, max_arg]
            pc2 = rotated_traj[:, max_arg2]
            pc3 = rotated_traj[:, max_arg3]

            rotated_traj = np.vstack((pc1, pc2, pc3)).T

        elif dim == 2:
            pc1 = rotated_traj[:, max_arg]
            pc2 = rotated_traj[:, max_arg2]

            rotated_traj = np.vstack((pc1, pc2)).T

        #rotated_traj_origin = rotated_traj - np.tile(rotated_traj[0], (traj.shape[0], 1))
        registered_trajectories[traj_idx] = rotated_traj

    return registered_trajectories


def register_traj_disp_reflection(trajectories):
    def calc_max_distance(traj):
        all_distance_list = []
        for t in range(1, traj.shape[0]):
            distance = traj[t:] - traj[:-t]
            all_distance_list.append(max(abs(distance)))
        return max(all_distance_list)

    dim = trajectories[0].shape[1]
    registered_trajectories = {}
    for traj_idx in trajectories:
        traj = trajectories[traj_idx]

        max_dist, arg, tlag_max = -1, -1, -1
        for tlag in range(1, traj.shape[0]):
            dxyz = traj[tlag:] - traj[:-tlag]  # Displacement between two nearby points
            avg = np.ones((len(traj[:, 0]), 1)) * np.mean(traj, axis=0)
            xyr = traj - avg

            # determine the rotational matrix
            u, s, rotational_matrix = np.linalg.svd(dxyz)
            rotational_matrix = rotational_matrix.T

            # project major axis of trajectories onto rotational matrix
            xyr_r = xyr @ rotational_matrix
            if dim == 3:
                x = xyr_r[:, 0]
                y = xyr_r[:, 1]
                z = xyr_r[:, 2]
                list_dist = [calc_max_distance(x), calc_max_distance(y), calc_max_distance(z)]
                dist = max(list_dist)
                arg = np.argmax(list_dist)
                # print(tlag, calc_max_distance(x), calc_max_distance(y), calc_max_distance(z))
                if dist > max_dist:
                    max_dist = dist
                    max_arg = arg
                    tlag_max = tlag
                    list_dist[max_arg] = -1
                    max_arg2 = np.argmax(list_dist)
                    list_dist[max_arg2] = -1
                    max_arg3 = np.argmax(list_dist)
            elif dim == 2:
                x = xyr_r[:, 0]
                y = xyr_r[:, 1]
                list_dist = [calc_max_distance(x), calc_max_distance(y)]
                dist = max(list_dist)
                arg = np.argmax(list_dist)
                # print(tlag, calc_max_distance(x), calc_max_distance(y), calc_max_distance(z))
                if dist > max_dist:
                    max_dist = dist
                    max_arg = arg
                    tlag_max = tlag
                    list_dist[max_arg] = -1
                    max_arg2 = np.argmax(list_dist)
            # print(tlag, max_dist, max_arg, max_arg2, max_arg3)

        # print(tlag_max, max_dist, max_arg, max_arg2, max_arg3)

        dxyz = traj[tlag_max:] - traj[:-tlag_max]  # Displacement between two nearby points
        avg = np.ones((len(traj[:, 0]), 1)) * np.mean(traj, axis=0)
        xyr = traj - avg

        # determine the rotational matrix
        u, s, rotational_matrix = np.linalg.svd(dxyz)
        rotational_matrix = rotational_matrix.T

        # project major axis of trajectories onto rotational matrix
        rotated_traj = xyr @ rotational_matrix
        if dim == 3:
            pc1 = rotated_traj[:, max_arg]
            pc2 = rotated_traj[:, max_arg2]
            pc3 = rotated_traj[:, max_arg3]

            rotated_traj = np.vstack((pc1, pc2, pc3)).T

        elif dim == 2:
            pc1 = rotated_traj[:, max_arg]
            pc2 = rotated_traj[:, max_arg2]

            rotated_traj = np.vstack((pc1, pc2)).T

        rotated_traj_origin = rotated_traj - np.tile(rotated_traj[0], (traj.shape[0], 1))

        x_minus_list = [x for x in rotated_traj_origin[:, 0] if x < 0]
        x_plus_list = [x for x in rotated_traj_origin[:, 0] if x > 0]

        y_minus_list = [y for y in rotated_traj_origin[:, 1] if y < 0]
        y_plus_list = [y for y in rotated_traj_origin[:, 1] if y > 0]

        x_minus_list.append(0)  # for case where list is empty
        x_plus_list.append(0)  # for case where list is empty
        y_minus_list.append(0)  # for case where list is empty
        y_plus_list.append(0)  # for case where list is empty

        if dim ==3 :
            z_minus_list = [z for z in rotated_traj_origin[:, 2] if z < 0]
            z_plus_list = [z for z in rotated_traj_origin[:, 2] if z > 0]

            z_minus_list.append(0)  # for case where list is empty
            z_plus_list.append(0)  # for case where list is empty

        if dim == 3:
            if (abs(min(x_minus_list)) > max(x_plus_list)) and (abs(min(y_minus_list)) > max(y_plus_list)) and (abs(min(z_minus_list)) < max(z_plus_list)):
                rotated_traj_reflected = rotated_traj_origin * [-1, -1, 1]

            elif (abs(min(x_minus_list)) > max(x_plus_list)) and (abs(min(y_minus_list)) < max(y_plus_list)) and (abs(min(z_minus_list)) < max(z_plus_list)):
                rotated_traj_reflected = rotated_traj_origin * [-1, 1, 1]

            elif (abs(min(x_minus_list)) < max(x_plus_list)) and (abs(min(y_minus_list)) > max(y_plus_list)) and (abs(min(z_minus_list)) < max(z_plus_list)):
                rotated_traj_reflected = rotated_traj_origin * [1, -1, 1]

            elif (abs(min(x_minus_list)) < max(x_plus_list)) and (abs(min(y_minus_list)) < max(y_plus_list)) and (abs(min(z_minus_list)) < max(z_plus_list)):
                rotated_traj_reflected = rotated_traj_origin * [1, 1, 1]

            elif (abs(min(x_minus_list)) > max(x_plus_list)) and (abs(min(y_minus_list)) > max(y_plus_list)) and (abs(min(z_minus_list)) > max(z_plus_list)):
                rotated_traj_reflected = rotated_traj_origin * [-1, -1, -1]

            elif (abs(min(x_minus_list)) > max(x_plus_list)) and (abs(min(y_minus_list)) < max(y_plus_list)) and (abs(min(z_minus_list)) > max(z_plus_list)):
                rotated_traj_reflected = rotated_traj_origin * [-1, 1, -1]

            elif (abs(min(x_minus_list)) < max(x_plus_list)) and (abs(min(y_minus_list)) > max(y_plus_list)) and (abs(min(z_minus_list)) > max(z_plus_list)):
                rotated_traj_reflected = rotated_traj_origin * [1, -1, -1]

            elif (abs(min(x_minus_list)) < max(x_plus_list)) and (abs(min(y_minus_list)) < max(y_plus_list)) and (abs(min(z_minus_list)) > max(z_plus_list)):
                rotated_traj_reflected = rotated_traj_origin * [1, 1, -1]

        elif dim == 2:
            if (abs(min(x_minus_list)) > max(x_plus_list)) and (abs(min(y_minus_list)) > max(y_plus_list)):
                rotated_traj_reflected = rotated_traj_origin * [-1, -1]
                # Reflection by origin when maximum displacement in x-axis and y-axis is minus
            elif (abs(min(x_minus_list)) > max(x_plus_list)) and (abs(min(y_minus_list)) < max(y_plus_list)):
                rotated_traj_reflected = rotated_traj_origin * [-1, 1]
                # Reflection by y-axis when maximum displacement in x-axis is minus
            elif (abs(min(x_minus_list)) < max(x_plus_list)) and (abs(min(y_minus_list)) > max(y_plus_list)):
                rotated_traj_reflected = rotated_traj_origin * [1, -1]
                # Reflection by x-axis when maximum displacement in y-axis is minus
            else:
                rotated_traj_reflected = rotated_traj_origin

        registered_trajectories[traj_idx] = rotated_traj_reflected
    return registered_trajectories


def remove_stationary_trajs(df, duration, feature_name=['x', 'y']):
    ''' Remove trajectories that are not moving (only for trajectories with same duration)
        (to remove nan for angle features)
    Parameters:
    ----------
    df: pd.DataFrame()
        dataframe with each row is a time point
    duration: int
        number of time frames of a trajectory
    feature_name: list of str
        name of the column for trajectory coordinates

    Returns:
    -------
    removed_df: pd.DataFrame()
        dataframe that stationary trajectories are removed
    removed_index_list: list
        removed index list
    '''

    removed_df = df
    removed_index_list = []
    for traj_idx in range(int(df.shape[0] / duration)):  # For each cell trajectory
        traj_data_temp = df[duration * traj_idx:duration * (traj_idx + 1)]
        traj = traj_data_temp[feature_name].values

        def calc_angle(start, middle, end):  # Start = [x0, y0, z0], middle = [x1, y1, z1], end = [x2, y2, z2],
            '''Angle in radians, range from 0 to pi'''

            ba = start - middle
            bc = end - middle

            if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:  # cell didn't move
                angle = 0

            else:
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                if cosine_angle > 1:  # because of limited number of digits, sometimes cosine_angle = 1.00000001
                    cosine_angle = 1
                elif cosine_angle < -1:  # because of limited number of digits, sometimes cosine_angle = -1.00000001
                    cosine_angle = -1
                angle = np.arccos(cosine_angle)

            return angle

        start = traj[0]
        middle = traj[1]
        angle_list = []
        for coor in traj[2:]:
            angle = calc_angle(start, middle, coor)
            angle_list.append(angle)
            start = middle
            middle = coor

        if all(np.array(angle_list) == 0):
            removed_df = removed_df.drop(traj_data_temp.index, axis=0)
            removed_index_list.append(traj_data_temp.index)
    removed_df = removed_df.reset_index(drop=True)
    print('removed %s trajectories' % len(removed_index_list))

    return removed_df, removed_index_list

def remove_trajs_condition(df_duration:pd.DataFrame, duration:int, remove_traj_idxs:list[int]) -> pd.DataFrame:
    ''' Remove trajectory dataframe by list of indexes
        Parameters:
        ----------
        df_duration: pd.DataFrame()
            raw dataframe with trajectory coordinates (x, y)
        duration: int
            number of time frames that original trajectories have
        remove_traj_idxs: list
            list of trajectory indexes to be removed (indexes for each cell trajectories, not cell at one time point)
        Returns:
        -------
        df_removed: pd.DataFrame()
            removed dataframe with trajectory coordinates
        '''
    total_idxs = list(range(int(df_duration.shape[0] / duration)))

    set1 = set(total_idxs)
    set2 = set(remove_traj_idxs)

    set3 = set1 - set2
    result_list = list(set3)

    removed_idxs = [list(range(duration * traj_idx, duration * (traj_idx + 1))) for traj_idx in result_list]
    removed_idxs_flattened = [item for sublist in removed_idxs for item in sublist]
    df_duration_removed = df_duration.iloc[removed_idxs_flattened, :].reset_index(drop=True)

    return df_duration_removed

def fit_pmc(df, path, duration, interval, desired_interval, pmcf=2.01, draw_figure=True):
    ''' Generate trajectories that are spliced by specific cut_duration (only for trajectories with same duration)
    Parameters:
    ----------
    df: pd.DataFrame()
        raw dataframe with trajectory coordinates (x, y)
    path: str
        path you want to save fitted trajectory figures
    duration: int
        number of time frames that original trajectories have
    interval: int
        interval in minutes that original trajectories have
    desired_interval: int
        interval in minutes that you want the newly constructed trajectories have
    pmcf: str
        coefficient for pseudo Monte Carlo

    Returns:
    -------
    interval_trajs: pd.DataFrame()
        generate trajectory dataframe with desired interval only
    new_trajs: pd.DataFrame()
        generate trajectory dataframe that has both original and desired interval
    '''
    interval_trajs = pd.DataFrame()
    new_trajs = pd.DataFrame()
    for traj_idx in (range(0, int(df.shape[0] / duration))):
        traj = df[duration * traj_idx:duration * (traj_idx + 1)]
        n = 1
        new_dict = {}
        cell_ids = []
        times = []
        xs = []
        ys = []

        interval_dict = {}
        interval_cell_ids = []
        interval_times = []
        interval_xs = []
        interval_ys = []

        for i in range(1, traj.shape[0]):
            t_i2 = i * interval
            t_i0 = (i - 1) * interval
            t_i1 = desired_interval * n

            x_i2 = traj.iloc[i, :]['x']
            x_i0 = traj.iloc[i - 1, :]['x']

            y_i2 = traj.iloc[i, :]['y']
            y_i0 = traj.iloc[i - 1, :]['y']

            time_i2 = traj.iloc[i, :]['time']
            time_i0 = traj.iloc[i - 1, :]['time']
            if (t_i2 > t_i1):
                x_intercept_i1 = x_i0 + (t_i1 - t_i0) * (x_i2 - x_i0) / (t_i2 - t_i0) # linear interpolation
                y_intercept_i1 = y_i0 + (t_i1 - t_i0) * (y_i2 - y_i0) / (t_i2 - t_i0) # linear interpolation
                time_intercept_i1 = time_i0 + (t_i1 - t_i0) * (time_i2 - time_i0) / (t_i2 - t_i0) # linear interpolation

                if abs(x_i2 - x_i0) >= abs(y_i2 - y_i0):
                    x_i1 = x_intercept_i1 + (x_i2 - x_i0) / pmcf # linear interpolation + pMC correction (directional persistence)
                    y_i1 = y_intercept_i1
                else:
                    x_i1 = x_intercept_i1
                    y_i1 = y_intercept_i1 + (y_i2 - y_i0) / pmcf # linear interpolation + pMC correction (directional persistence)

                n = n + 1

                cell_ids.append(np.unique(traj['cell_id'])[0])
                times.append(time_intercept_i1)
                xs.append(x_i1)
                ys.append(y_i1)

                interval_cell_ids.append(np.unique(traj['cell_id'])[0])
                interval_times.append(time_intercept_i1)
                interval_xs.append(x_i1)
                interval_ys.append(y_i1)

            elif (t_i2 == t_i1):
                n = n + 1
                interval_cell_ids.append(np.unique(traj['cell_id'])[0])
                interval_times.append(time_i2)
                interval_xs.append(x_i2)
                interval_ys.append(y_i2)

        new_dict['cell_id'] = cell_ids
        new_dict['time'] = times
        new_dict['x'] = xs
        new_dict['y'] = ys
        new_df = pd.DataFrame(new_dict)
        new_traj = pd.concat([traj, new_df], axis=0)
        new_traj = new_traj.sort_values(by='time').reset_index(drop=True)

        interval_dict['cell_id'] = interval_cell_ids
        interval_dict['time'] = interval_times
        interval_dict['x'] = interval_xs
        interval_dict['y'] = interval_ys
        interval_traj = pd.DataFrame(interval_dict)
        interval_traj = interval_traj.sort_values(by='time').reset_index(drop=True)

        if draw_figure == True:
            _, orig, _ = to_timeseries_fast(traj, duration=traj.shape[0], feature_name=['x', 'y'])
            _, new, _ = to_timeseries_fast(new_traj, duration=new_traj.shape[0], feature_name=['x', 'y'])
            _, inter, _ = to_timeseries_fast(interval_traj, duration=interval_traj.shape[0], feature_name=['x', 'y'])
            orig = orig[0]
            new = new[0]
            inter = inter[0]
            plt.figure()
            plt.plot(orig[:, 0], orig[:, 1], '-', color='red', linewidth=2, alpha=1, zorder=1)
            plt.scatter(orig[:, 0], orig[:, 1], color='red', alpha=0.7, s=15, zorder=1)
            # plt.plot(new[:, 0], new[:, 1] , '-', color='blue', linewidth=2, alpha=0.4, zorder=2)
            # plt.scatter(new[:, 0], new[:, 1], color='blue', alpha=0.5, s=6, zorder=2)
            plt.plot(inter[:, 0], inter[:, 1], '-', color='blue', linewidth=2, alpha=0.4, zorder=2)
            plt.scatter(inter[:, 0], inter[:, 1], color='blue', alpha=0.5, s=6, zorder=2)
            plt.title(
                'original : every %s min for %s frames' % (interval, duration) + ' pMC : every %s min for %s frames' % (
                desired_interval, interval_traj.shape[0]))
            if not os.path.isdir(
                    path + 'pMC fitted trajectory/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(path + 'pMC fitted trajectory/')
            plt.savefig(path + 'pMC fitted trajectory/%s.png' % traj_idx)
            plt.clf()
            plt.close()
        else:
            pass

        new_trajs = pd.concat([new_trajs, new_traj], axis=0).reset_index(drop=True)
        interval_trajs = pd.concat([interval_trajs, interval_traj], axis=0).reset_index(drop=True)

        interval_traj_duration = interval_traj.shape[0]
        new_traj_duration = new_traj.shape[0]

    return interval_trajs, new_trajs, interval_traj_duration, new_traj_duration

def morpho_trajectory(df, features, duration, dim):
    """ Generate trajectories that are spliced by specific cut_duration (only for trajectories with same duration)
    Parameters:
    ----------
    df: pandas dataframe
        raw dataframe where each row is one cell state at time t
    features: list
        list of morphology feature names
    dim: int
        dimension of Morpho-trajectory

    Returns:
    -------
    trajectories: dict
        generate morpho_trajectories
    """
    from sklearn.preprocessing import StandardScaler
    morphology = df[features]
    scaler = StandardScaler()
    morphology_data = pd.DataFrame(scaler.fit_transform(morphology), columns=morphology.columns)  # (x-mu)/sigma

    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim)
    pcs = pca.fit_transform(morphology_data)
    if dim==3:
        df_pcs = pd.DataFrame(pcs, columns=['Morpho_X', 'Morpho_Y', 'Morpho_Z'])
        traj_list, trajectories_array, trajectories = to_timeseries_fast(df_pcs, duration=duration, feature_name=['Morpho_X', 'Morpho_Y', 'Morpho_Z'])
    elif dim==2:
        df_pcs = pd.DataFrame(pcs, columns=['Morpho_X', 'Morpho_Y'])
        traj_list, trajectories_array, trajectories = to_timeseries_fast(df_pcs, duration=duration, feature_name=['Morpho_X', 'Morpho_Y'])

    return trajectories

def normalize_timeseries(timeseries):
    """ Generate normalized timeseries
        Parameters:
        ----------
        timeseries: dict
            timeseries that each key is trajectory index and value has shape (duration, dimension of timeseries) or (duration, )

        Returns:
        -------
        norm_signals: dict
            normalized timeseries
        """
    import logging
    norm_signals = {}
    for traj_idx in timeseries:
        signals = timeseries[traj_idx]
        norm_signals_each = []

        if len(signals.shape) == 2:
            traj_len, col_len = signals.shape
            for col in range(col_len):
                signal = signals[:, col]
                if any(np.isnan(signal)):  # at least one point is nan in one time series
                    logging.warning('trajectory %s with column number %s contains nan and removed' % (traj_idx, col))
                    continue
                elif all(signal == 0):
                    norm_signals_each.append(signal)
                else:
                    mean = np.mean(signal)
                    std = np.std(signal)
                    if std == 0:
                        logging.warning('trajectory %s with column number %s has std of 0' % (traj_idx, col))
                    norm_signal = (signal - mean) / std
                    norm_signals_each.append(norm_signal)
            norm_signals[traj_idx] = np.array(norm_signals_each).T

        elif len(signals.shape) == 1:
            if any(np.isnan(signals)):
                logging.warning('trajectory %s contains nan and removed' % (traj_idx))
                continue
            elif all(signals == 0):
                norm_signals_each.append(signals)
            else:
                mean = np.mean(signals)
                std = np.std(signals)
                if std == 0:
                    logging.warning('trajectory %s has std of 0' % (traj_idx))
                norm_signal = (signals - mean) / std
                norm_signals_each.append(norm_signal)
            norm_signals[traj_idx] = np.array(norm_signals_each).flatten()
    return norm_signals

def difference_timeseries(timeseries):
    """ Generate first difference of timeseries
        Parameters:
        ----------
        timeseries: dict
            timeseries that each key is trajectory index and value has shape (duration, number of different timeseries) or (duration, )

        Returns:
        -------
        diff_signals: dict
            first difference of original timeseries
        """

    diff_signals = {}
    for traj_idx in timeseries:
        signals = timeseries[traj_idx]
        diff_signals_each = []

        if len(signals.shape) == 2:
            traj_len, col_len = signals.shape
            for col in range(col_len):
                signal = signals[:, col]
                diff_signals_each.append(np.diff(signal))
            diff_signals[traj_idx] = np.array(diff_signals_each).T

        elif len(signals.shape) == 1:
            diff_signals_each.append(np.diff(signals))
            diff_signals[traj_idx] = np.array(diff_signals_each).flatten()
    return diff_signals

def get_instant_movements(df, duration, time_unit, feature_name=['PC1', 'PC2']):
    def calc_distance(coor1, coor2):
        '''Euclidean distance'''
        if coor1.shape[0] == 2:
            x1, y1 = coor1
            x2, y2 = coor2
            distance = math.sqrt((x2 - x1) ** 2.00 + (y2 - y1) ** 2.00)

        elif coor1.shape[0] == 3:
            x1, y1, z1 = coor1
            x2, y2, z2 = coor2
            distance = math.sqrt((x2 - x1) ** 2.00 + (y2 - y1) ** 2.00 + (z2 - z1) ** 2.00)

        return distance

    def calc_angle(start, middle, end):  # Start = [x0, y0, z0], middle = [x1, y1, z1], end = [x2, y2, z2],
        '''Angle in radians, range from 0 to pi'''

        ba = middle - start
        bc = end - middle

        if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:  # cell didn't move
            angle = 0

        else:
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            if cosine_angle > 1:  # because of limited number of digits, sometimes cosine_angle = 1.00000001
                cosine_angle = 1
            elif cosine_angle < -1:  # because of limited number of digits, sometimes cosine_angle = -1.00000001
                cosine_angle = -1
            angle = np.arccos(cosine_angle)

        return angle

    new_df = pd.DataFrame()
    for traj_idx in range(int(df.shape[0]/duration)):  # For each cell trajectory
        traj_data_temp = df[duration*traj_idx:duration*(traj_idx+1)].copy()
        traj = traj_data_temp[feature_name].values
        start = traj[0]
        distance_list = [0]
        for coor in traj[1:]:
            distance = calc_distance(start, coor)
            distance_list.append(distance / time_unit)
            start = coor

        angle_list = [0, 0]
        start_angle = traj[0]
        middle_angle = traj[1]
        for coor in traj[2:]:
            angle = calc_angle(start_angle, middle_angle, coor)
            angle_list.append(angle / time_unit)
            start_angle = middle_angle
            middle_angle = coor

        traj_data_temp.loc[:, 'instant_speed'] = distance_list
        traj_data_temp.loc[:, 'instant_angle'] = angle_list
        new_df = pd.concat([new_df, traj_data_temp])
    new_df = new_df.reset_index(drop=True)
    return new_df

def get_instant_movements_variable_duration(df, frame_name, time_unit, feature_name=['PC1', 'PC2']):
    def calc_distance(coor1, coor2):
        '''Euclidean distance'''
        if coor1.shape[0] == 2:
            x1, y1 = coor1
            x2, y2 = coor2
            distance = math.sqrt((x2 - x1) ** 2.00 + (y2 - y1) ** 2.00)

        elif coor1.shape[0] == 3:
            x1, y1, z1 = coor1
            x2, y2, z2 = coor2
            distance = math.sqrt((x2 - x1) ** 2.00 + (y2 - y1) ** 2.00 + (z2 - z1) ** 2.00)

        return distance

    def calc_angle(start, middle, end):  # Start = [x0, y0, z0], middle = [x1, y1, z1], end = [x2, y2, z2],
        '''Angle in radians, range from 0 to pi'''

        ba = middle - start
        bc = end - middle

        if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:  # cell didn't move
            angle = 0

        else:
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            if cosine_angle > 1:  # because of limited number of digits, sometimes cosine_angle = 1.00000001
                cosine_angle = 1
            elif cosine_angle < -1:  # because of limited number of digits, sometimes cosine_angle = -1.00000001
                cosine_angle = -1
            angle = np.arccos(cosine_angle)

        return angle

    new_df = pd.DataFrame()
    trajectories = {}
    traj_idx = 0
    i0 = 0
    for i in tqdm(range(0, df.shape[0])):
        if (i == 0) or (i == duration + i0):
            duration = df[frame_name][i]
            traj_data_temp = df[i: duration + i].copy()
            i0 = i
            traj = traj_data_temp[feature_name].values
            trajectories[traj_idx] = traj

            start = traj[0]
            distance_list = [0]
            for coor in traj[1:]:
                distance = calc_distance(start, coor)
                distance_list.append(distance / time_unit)
                start = coor

            angle_list = [0, 0]
            start_angle = traj[0]
            middle_angle = traj[1]
            for coor in traj[2:]:
                angle = calc_angle(start_angle, middle_angle, coor)
                angle_list.append(angle)
                start_angle = middle_angle
                middle_angle = coor

            traj_data_temp.loc[:, 'instant_speed'] = distance_list
            traj_data_temp.loc[:, 'instant_angle'] = angle_list
            traj_data_temp.loc[:, 'pseudo_frame'] = np.arange(0, traj_data_temp.shape[0], 1)

            new_df = pd.concat([new_df, traj_data_temp])
            traj_idx = traj_idx + 1

        else:
            continue

    new_df = new_df.reset_index(drop=True)

    return new_df



def get_instant_direction(df, duration, thresh, feature_name='Shortest_Distance_to_Surfaces_Surfaces=FDC'):

    new_df = pd.DataFrame()

    type_lists=[]
    diff_lists=[]
    for traj_idx in range(int(df.shape[0]/duration)):  # For each cell trajectory
        traj_data_temp = df[duration*traj_idx:duration*(traj_idx+1)].copy()
        traj = traj_data_temp[feature_name].values
        start = traj[0]
        type_list = ['None']
        diff_list = [np.nan]
        for next in traj[1:]:
            sign = next - start
            typ='None'
            if sign > thresh:
                typ = 'departure'
            elif sign < -thresh:
                typ = 'approach'
            elif (sign <= thresh) & (sign >= -thresh):
                typ = 'stay'
            type_list.append(typ)
            diff_list.append(sign)
            start = next
        type_lists.append(type_list)
        diff_lists.append(diff_list)

    type_lists = np.array(flatten_nested_dict(type_lists))
    diff_lists = np.array(flatten_nested_dict(diff_lists))

    df.loc[:, 'quality_%s'%feature_name] = type_lists
    df.loc[:, 'diff_%s' % feature_name] = diff_lists

    return df


def get_instant_direction_variable_duration(df, frame_name, thresh, feature_name=['PC1', 'PC2']):

    new_df = pd.DataFrame()
    trajectories = {}
    traj_idx = 0
    i0 = 0
    type_lists = []
    diff_lists = []
    for i in range(0, df.shape[0]):
        if (i == 0) or (i == duration + i0):
            duration = df[frame_name][i]
            traj_data_temp = df[i: duration + i].copy()
            i0 = i
            traj = traj_data_temp[feature_name].values
            trajectories[traj_idx] = traj

            start = traj[0]
            type_list = ['None']
            diff_list = [np.nan]
            for next in traj[1:]:
                sign = next - start
                typ = 'None'
                if sign > thresh:
                    typ = 'departure'
                elif sign < -thresh:
                    typ = 'approach'
                elif (sign <= thresh) & (sign >= -thresh):
                    typ = 'stay'
                type_list.append(typ)
                diff_list.append(sign)
                start = next
            type_lists.append(type_list)
            diff_lists.append(diff_list)

        else:
            continue
    type_lists = np.array(flatten_nested_dict(type_lists))
    diff_lists = np.array(flatten_nested_dict(diff_lists))

    df.loc[:, 'quality_%s' % feature_name] = type_lists
    df.loc[:, 'diff_%s' % feature_name] = diff_lists

    return df


def get_consecutive_frame_df(df_duration, label_name, frame_name):
    consecutive_labels = []
    for label, group in df_duration.groupby(label_name):
        time_diff = group[frame_name].sort_values().diff().dropna()  # Drop the first time point by dropna
        if np.all(time_diff == 1):  # Check whether all time difference is 1
            consecutive_labels.append(label)
        # else:
        #     print(label, group[frame_name])
    df_duration_filtered = df_duration[df_duration[label_name].isin(consecutive_labels)]
    return df_duration_filtered.reset_index(drop=True)


def harmonize_pmc(
    traj: pd.DataFrame,
    orig_dt_min: float,
    new_dt_min: float,
    x_name: str,
    y_name: str,
    frame_name: str,
    pmc: float = 2.01,
):
    """
    traj : dataframe with x, y, frame of a trajectory
    orig_dt_min : original frame interval in minutes (e.g., 2.0)
    new_dt_min  : new desired interval in minutes (e.g., 0.5 for 30 seconds)
    """

    # Time in minutes (NOT seconds)
    t = traj[frame_name].to_numpy(dtype=float) * float(orig_dt_min)
    x = traj[x_name].to_numpy(dtype=float)
    y = traj[y_name].to_numpy(dtype=float)

    # new uniform time grid
    t_new = np.arange(t.min(), t.max() + 1e-9, float(new_dt_min))

    x_new = np.zeros_like(t_new)
    y_new = np.zeros_like(t_new)

    j = 0
    for i, tn in enumerate(t_new):
        while j < len(t) - 2 and tn > t[j + 1]:
            j += 1

        t0, t1 = t[j], t[j + 1]
        x0, x1 = x[j], x[j + 1]
        y0, y1 = y[j], y[j + 1]

        dt = t1 - t0
        if dt == 0:
            x_new[i] = x0
            y_new[i] = y0
            continue

        alpha = (tn - t0) / dt  # 0..1

        dx = x1 - x0
        dy = y1 - y0

        # pure linear
        x_linear = x0 + alpha * dx
        y_linear = y0 + alpha * dy

        # correction weight that is ZERO at endpoints
        w = alpha * (1.0 - alpha)

        # PMC correction
        if abs(dx) >= abs(dy):
            x_interp = x_linear + w * (dx / pmc)
            y_interp = y_linear
        else:
            x_interp = x_linear
            y_interp = y_linear + w * (dy / pmc)

        x_new[i] = x_interp
        y_new[i] = y_interp

    df_traj = pd.DataFrame({
        "%s"%frame_name: t_new / float(orig_dt_min),
        "%s"%x_name: x_new,
        "%s"%y_name: y_new
    })
    return df_traj
