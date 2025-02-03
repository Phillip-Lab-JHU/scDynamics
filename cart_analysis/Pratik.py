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
"""Pratik data"""

from sklearn.preprocessing import StandardScaler  # (x-mu)/sigma
from Morphology import Morphodynamics
from utils.draw_utils import *
from utils.traj_utils import to_timeseries_fast

#################################### tskmeans ####################################

path = r'\\philliplab-server.wse.jhu.edu\data\Pratik\Functional subtypes senescence\Senotherapies\12_21_23 Dox Senotherapy Response\Chanhong EXCELS\4clusters\\'
files = next(os.walk(path))[2]
bool_list = ['csv' in ele for ele in files]
files = np.array(files)[np.array(bool_list)]

df = pd.DataFrame()
tskm_cluster_centers=[]
for file in files:
    excel = pd.read_csv(path + file)
    df_duration = to_trajectory_variable_duration(excel, min_duration=18, condition_name='Drug Treatment', label_name='particle_well')

    trajectories = to_timeseries_variable_duration(df_duration, length_name='Time_span', coord_name=['UMAP_1', 'UMAP_2'], equal_length=True, frame_name='frame')
    trajectories_arr = dict_to_array(trajectories)

    from tslearn.clustering import TimeSeriesKMeans
    tskm = TimeSeriesKMeans(n_clusters=4, metric='softdtw', random_state=0, verbose=True, max_iter=100)
    tskmeans_predicted = tskm.fit_predict(trajectories_arr)
    tskm_cluster_center = tskm.cluster_centers_
    tskm_cluster_centers.append(tskm_cluster_center)
    traj_idx = 0
    i0 = 0
    df_duration_tskmeans = pd.DataFrame()
    for i in range(0, df_duration.shape[0]):
        if (i == 0) or (i == duration + i0):
            duration = df_duration['Time_span'][i]
            traj_data_temp = df_duration[i: duration + i].copy()
            traj_data_temp.loc[:, 'tskmeans'] = tskmeans_predicted[traj_idx]
            df_duration_tskmeans = pd.concat([df_duration_tskmeans, traj_data_temp], axis=0)
            traj_idx = traj_idx + 1
            i0 = i
        else:
            continue
    df_duration_tskmeans = df_duration_tskmeans.reset_index(drop=True)
    df= pd.concat([df, df_duration_tskmeans], axis=0)

df = df.reset_index(drop=True)

df.to_csv(path+'all_points.csv')


df_tskmeans = pd.DataFrame()
start_kmeans_list = [10, 11, 7]
for i, tskm_cluster_center in enumerate(tskm_cluster_centers):
    start_kmeans =  start_kmeans_list[i]
    df_tskmeans_tem = pd.DataFrame()
    for tskmeans, each_tskmeans_cluster_center in enumerate(tskm_cluster_center):
        df_tskmeans_temp = pd.DataFrame()
        df_tskmeans_temp['UMAP_1'] = each_tskmeans_cluster_center[:, 0]
        df_tskmeans_temp['UMAP_2'] = each_tskmeans_cluster_center[:, 1]
        df_tskmeans_temp['frame'] = range(0, each_tskmeans_cluster_center.shape[0])
        df_tskmeans_temp['tskmeans'] = tskmeans
        df_tskmeans_temp['Starting KMEANS'] = start_kmeans
        df_tskmeans_tem = pd.concat([df_tskmeans_tem, df_tskmeans_temp], axis=0)

    df_tskmeans = pd.concat([df_tskmeans, df_tskmeans_tem], axis=0)

df_tskmeans = df_tskmeans.reset_index(drop=True)

df_tskmeans.to_csv(path+'representative_trajs.csv')
#################################### plot representative trajectory  ####################################

xmin = math.floor(df['UMAP_1'].min()) - 1
xmax = math.ceil(df['UMAP_1'].max()) + 1
ymin = math.floor(df['UMAP_2'].min()) - 1
ymax = math.ceil(df['UMAP_2'].max()) + 1


tskm_cluster_center = tskm_cluster_centers[0]
plt.figure(figsize=(15,12))

color = ['red','green','blue','darkorange', 'magenta','purple']

#plt.scatter(df_fulltime_cell['PC1'],df_fulltime_cell['PC2'], marker=',',s=10, alpha = 0.1,c='cornflowerblue')
plt.scatter(df['UMAP_1'],df['UMAP_2'], marker=',',s=10, alpha = 0.2,
            c=df['Current KMEANS'],cmap = plt.cm.get_cmap('Set3'))
dot_c1=np.arange(tskm_cluster_center.shape[1]-1) # tskm.cluster_centers_ = (cluster 수, time point 수, dimension)

for cluster in range(0,tskm_cluster_center.shape[0]):
    plt.quiver(tskm_cluster_center[cluster][:-1,0],tskm_cluster_center[cluster][:-1,1],
               tskm_cluster_center[cluster][1:,0]-tskm_cluster_center[cluster][:-1,0],
               tskm_cluster_center[cluster][1:,1]-tskm_cluster_center[cluster][:-1,1],
               scale_units='xy', angles='xy', scale=1, color = color[cluster], label = cluster)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
plt.legend()
plt.savefig(path + 'tskmeans cluster10.png')
#plt.show()
plt.close()
plt.clf()


draw_cluster_distribution_heatmap(df[df['Starting KMEANS']==10], path, file_name='kmeans_c10_heatmap', condition_name='Drug Treatment', cluster_type='tskmeans')
draw_cluster_distribution_heatmap(df[df['Starting KMEANS']==11], path, file_name='kmeans_c11_heatmap', condition_name='Drug Treatment', cluster_type='tskmeans')
draw_cluster_distribution_heatmap(df[df['Starting KMEANS']==7], path, file_name='kmeans_c7_heatmap', condition_name='Drug Treatment', cluster_type='tskmeans')



df_part = df[df['Starting KMEANS']==7]

tskm_cluster_center = tskm_cluster_centers[2]
dot_c1=np.arange(tskm_cluster_center.shape[1]-1) # tskm.cluster_centers_ = (n_clusters, n_time points, dimension)

plt.figure(figsize=(20,5))
for cluster in range(0,tskm_cluster_center.shape[0]):
    colors = {0:'cornflowerblue', 1:'cornflowerblue', 2:'cornflowerblue', 3:'cornflowerblue', 4:'cornflowerblue', 5:'cornflowerblue', cluster:'indianred'}
    plt.subplot(1,4,cluster+1)
    plt.scatter(df['UMAP_1'], df['UMAP_2'], marker=',', s=10, alpha=0.1,
                color='cornflowerblue')
    plt.scatter(df_part['UMAP_1'],df_part['UMAP_2'], marker=',',s=10, alpha = 0.1,
                c=df_part['tskmeans'].map(colors))
    plt.quiver(tskm_cluster_center[cluster][:-1,0],tskm_cluster_center[cluster][:-1,1],
               tskm_cluster_center[cluster][1:,0]-tskm_cluster_center[cluster][:-1,0],
               tskm_cluster_center[cluster][1:,1]-tskm_cluster_center[cluster][:-1,1],
               dot_c1, scale_units='xy', angles='xy', scale=1, cmap = plt.cm.get_cmap('jet'), label = cluster)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
plt.savefig(path+'tskmeans cluster7 time evolution.png')
plt.close()
plt.clf()


#################################### plot vector field  ####################################

bin_num=30
condition_name='Drug Treatment'
label_name='particle_well'
x_name='UMAP_1'
y_name='UMAP_2'

df_part = df[df['Starting KMEANS']==11].reset_index(drop=True)

xmin = math.floor(df['UMAP_1'].min()) - 1
xmax = math.ceil(df['UMAP_1'].max()) + 1
ymin = math.floor(df['UMAP_2'].min()) - 1
ymax = math.ceil(df['UMAP_2'].max()) + 1

from math import sqrt
from statistics import mean
xgrid = np.linspace(xmin, xmax, bin_num)  # (100, ) 1d x coordinate
ygrid = np.linspace(ymin, ymax, bin_num)  # (100, ) 1d y coordinate
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid, indexing='xy')
# Xgrid , Ygrid = 각각 (100,100) 2d array
# Xgrid[i] = xgrid 좌표(-2에서 5를 100분할한게 row방향으로 반복)
# Ygrid[:,i] = ygrid 좌표(-6에서 4를 100분할한게 column방향으로 반복)

for condition in list(pd.unique(df_part[condition_name])):
    label_data = df_part.groupby([condition_name, label_name]).apply(lambda x: x.name).reset_index() # reset_index produce separate 'Type' and 'ID' column
    label_data = label_data[label_data[condition_name] == condition].reset_index()
    label_data = label_data.drop('index', axis=1)

    ########### 각 element가 list인 100x100 array 형성 ##########
    transition_mag_array_temp = np.empty((bin_num, bin_num), dtype='object')
    transition_vec_array_temp = np.empty((bin_num, bin_num), dtype='object')
    for row in range(0, transition_mag_array_temp.shape[0]):
        for col in range(0, transition_mag_array_temp.shape[1]):
            transition_mag_array_temp[row, col] = [0]
            transition_vec_array_temp[row, col] = [(0, 0)]

    ######### 각 세포, 시간마다 transition magnitude 계산하며 list로 append ##########
    for traj_idx in range(0, label_data.shape[0]):  # 각 세포마다
        cell_data = df_part.groupby([condition_name, label_name]).get_group(label_data[0][traj_idx]).copy().reset_index()
        # 한 세포에 time span에 대한 PC1, PC2, GMM_cluster, Kmeans_cluster 정보
        transition_mag = 0
        for t in range(0, cell_data.shape[0] - 1):  # 한 세포 안에서 각 time frame 마다
            x = cell_data[x_name]
            y = cell_data[y_name]
            residual = (x[t] - Xgrid) ** 2 + (y[t] - Ygrid) ** 2  # residual = (100,100) 2d array
            min_coordinate = np.unravel_index(np.argmin(residual), residual.shape)  # residual이 minimum인 array index 반환, (x,y)꼴
            transition_mag = sqrt((x[t] - x[t + 1]) ** 2 + (y[t] - y[t + 1]) ** 2)
            transition_vec = (x[t + 1] - x[t], y[t + 1] - y[t])
            transition_mag_array_temp[min_coordinate].append(transition_mag)
            transition_vec_array_temp[min_coordinate].append(transition_vec)

    ########### transition magnitude의 list의 element 개수가 2 이상이면 0을 제외(평균 낼 때 평균값을 작게 만듬) #########
    for row in range(0, transition_mag_array_temp.shape[0]):
        for col in range(0, transition_mag_array_temp.shape[1]):
            if len(transition_mag_array_temp[row, col]) > 1:
                transition_mag_array_temp[row, col].remove(0)
            if len(transition_vec_array_temp[row, col]) > 1:
                transition_vec_array_temp[row, col].remove((0, 0))

    ########### 각 element가 transition magnitude의 list인 100x100 array -> 각 list의 평균이 element인 100x100 array #########
    transition_mag_array = np.empty((bin_num, bin_num))
    transition_vec_x_array = np.empty((bin_num, bin_num))
    transition_vec_y_array = np.empty((bin_num, bin_num))

    for row in range(0, transition_mag_array.shape[0]):
        for col in range(0, transition_mag_array.shape[1]):
            transition_mag_array[row, col] = mean(transition_mag_array_temp[row, col])
            x_temp = []
            y_temp = []
            for x, y in transition_vec_array_temp[row, col]:
                x_temp.append(x)
                y_temp.append(y)
            transition_vec_x_array[row, col] = mean(x_temp)
            transition_vec_y_array[row, col] = mean(y_temp)

    plt.figure(figsize=(20, 15))
    plt.imshow(transition_mag_array, origin='lower', aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='jet')
    cb = plt.colorbar()
    cb.set_label("counts")

    plt.savefig(path + 'transition_magnitude_field_%s_c11.png' % (condition), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

    dot_c1 = np.arange(bin_num * bin_num)
    plt.figure(figsize=(20, 15))
    plt.quiver(Xgrid, Ygrid, transition_vec_x_array, transition_vec_y_array, dot_c1,
               scale_units='xy', angles='xy', scale=1, cmap=plt.cm.get_cmap('flag'))

    plt.savefig(path + 'transition_vector_field_%s_c11.png' % (condition), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

#################################### pseudotime heatmap?  ####################################