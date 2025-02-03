# Nikita Sivakumar <nsivaku3@jhmi.edu>, Chanhong Min <cmin11@jhmi.edu>

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
"""Calculates APRW features"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

class APRW3D:
    @staticmethod
    def get_HO(traj, dt, max_P, max_S):

        [popt_pc1, popt_pc2, popt_pc3] = APRW3D.fit_ARPW(traj, dt, max_P, max_S)

        msd = APRW3D.msd_allD(traj)

        MSD10 = msd[1][0]
        MSD100 = msd[-1][0]

        P_pc1 = popt_pc1[0]
        S_pc1 = popt_pc1[1]
        D_pc1 = ((S_pc1 ** 2) * P_pc1) / 4  # might need to be 8
        se_pc1 = popt_pc1[2]

        P_pc2 = popt_pc2[0]
        S_pc2 = popt_pc2[1]
        D_pc2 = ((S_pc2 ** 2) * P_pc2) / 4  # might need to be 8
        se_pc2 = popt_pc2[2]

        P_pc3 = popt_pc3[0]
        S_pc3 = popt_pc3[1]
        D_pc3 = ((S_pc3 ** 2) * P_pc3) / 4  # might need to be 8

        Dtot = D_pc1 + D_pc2
        phi = D_pc1 / (D_pc2 + 1 )

        return [MSD10, MSD100, P_pc1, S_pc1, D_pc1, P_pc2, S_pc2, D_pc2, P_pc3, S_pc3, D_pc3, Dtot, phi]

    @staticmethod
    def calc_max_distance(traj):
        all_distance_list = []
        for t in range(1, traj.shape[0]):
            distance = traj[t:] - traj[:-t]
            all_distance_list.append(max(abs(distance)))
        return max(all_distance_list)

    @staticmethod
    def register_traj_disp(traj):
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
            x = xyr_r[:, 0]
            y = xyr_r[:, 1]
            z = xyr_r[:, 2]
            list_dist = [APRW3D.calc_max_distance(x), APRW3D.calc_max_distance(y), APRW3D.calc_max_distance(z)]
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
        pc1 = rotated_traj[:, max_arg]
        pc2 = rotated_traj[:, max_arg2]
        pc3 = rotated_traj[:, max_arg3]

        registered_traj = np.vstack((pc1, pc2, pc3)).T

        return pc1, pc2, pc3

    @staticmethod
    def fit_ARPW(traj, dt, max_P, max_S):

        pc1, pc2, pc3 = APRW3D.register_traj_disp(traj)

        msdpc1 = APRW3D.msd_1D(pc1) # MSD of x only
        msdpc2 = APRW3D.msd_1D(pc2) # MSD of y only
        msdpc3 = APRW3D.msd_1D(pc3) # MSD of z only

        # if dim3 == True:
        #     msdnp1 = msdnp1 + APRW.ezmsd0_1D(xyr_r[:, 1])

        popt_pc1, prw_pc1 = APRW3D.fit_PRW(msdpc1 + 1e-30, dt, max_P, max_S)
        popt_pc2, prw_pc2 = APRW3D.fit_PRW(msdpc2 + 1e-30, dt, max_P, max_S)
        popt_pc3, prw_pc3 = APRW3D.fit_PRW(msdpc3 + 1e-30, dt, max_P, max_S)

        return popt_pc1, popt_pc2, popt_pc3

    @staticmethod
    def msd_1D(traj):
        fn = traj.shape[0] # Number of time frames
        msdr = np.zeros((fn-1,1))
        for dt in range(1,fn):
            dxyz = traj[dt:] - traj[:-dt] # displacement btw two adjacent points
            msdr[dt-1] = np.mean(dxyz**2,axis=0)
        return msdr

    @staticmethod
    def msd_allD(traj):
        fn = traj.shape[0] # Number of time frames
        msdr = np.zeros((fn-1,1))
        for dt in range(1,fn):
            dxyz = traj[dt:] - traj[:-dt]
            msdr[dt-1] = np.mean(np.sum(dxyz**2,axis=1))
        return msdr

    @staticmethod
    def PRW_model(t, P, S, se):
        return 1 * (S ** 2) * P * (t - P * (1 - np.exp(-t / P))) + 2 * se

    @staticmethod
    def fit_PRW(msd, dt, max_P, max_S):
        toi = np.arange(0, (msd.shape[0] / 3) - 1).astype(int)
        ydata = msd[toi].reshape(msd[toi].shape[0])

        ti = np.arange(1, msd.shape[0] / 3)

        Nt = len(ti)
        wif = (2 * ti ** 2 + 1) / 3 / ti / (Nt - ti + 1)
        wt = 1 / wif ** 2 / ydata

        xdata = ti * dt

        wt = np.diag(1 / wt)
        popt, pcov = curve_fit(APRW3D.PRW_model, xdata, ydata, p0=[10, 1, 1], # po: initial guess for the parameters (P, S, se)
                               method='trf', # Use Trust Region Reflective if bound is provided (Constrained optimization)
                               bounds=([0, 0, 0], [max_P, max_S, 100]),
                               sigma=wt, maxfev=1e10 # sigma: Determines the uncertainty in ydata
                               ) # maxfev: maximum number of iteration to find optimal value of parameters (P, S, se)
        return popt, APRW3D.PRW_model(xdata, *popt)


    @staticmethod
    def get_APRW(trajectories, dt, max_speed=20):
        HO_all = []
        for traj_idx in trajectories:
            traj = trajectories[traj_idx]
            HO = APRW3D.get_HO(traj, dt=dt, max_P=traj.shape[0]*dt, max_S=max_speed)
            HO_all.append(HO)

        df_HO = pd.DataFrame(HO_all,columns=['MSD1','MSD20','Pp','Sp','Dp','Psp','Ssp','Dsp','Pnp','Snp','Dnp','Dtot','phi'])

        return df_HO


class APRW:
    @staticmethod
    def get_HO(traj, dt, max_P, max_S):

        [popt_pc1, popt_pc2] = APRW.fit_ARPW(traj, dt, max_P, max_S)

        msd = APRW.msd_allD(traj)

        MSD10 = msd[1][0]
        #MSD100 = msd[19][0]
        MSD100 = msd[-1][0]

        P_pc1 = popt_pc1[0]
        S_pc1 = popt_pc1[1]
        D_pc1 = ((S_pc1 ** 2) * P_pc1) / 4
        se_pc1 = popt_pc1[2]

        P_pc2 = popt_pc2[0]
        S_pc2 = popt_pc2[1]
        D_pc2 = ((S_pc2 ** 2) * P_pc2) / 4
        se_pc2 = popt_pc2[2]


        Dtot = D_pc1 + D_pc2
        phi = D_pc1 / (D_pc2)

        return [MSD10, MSD100, P_pc1, P_pc2, D_pc1, D_pc2, Dtot, phi]

    @staticmethod
    def calc_max_distance(traj):
        all_distance_list = []
        for t in range(1, traj.shape[0]):
            distance = traj[t:] - traj[:-t]
            all_distance_list.append(max(abs(distance)))
        return max(all_distance_list)

    @staticmethod
    def register_traj_disp(traj):
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
            x = xyr_r[:, 0]
            y = xyr_r[:, 1]
            list_dist = [APRW.calc_max_distance(x), APRW.calc_max_distance(y)]
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
        pc1 = rotated_traj[:, max_arg]
        pc2 = rotated_traj[:, max_arg2]

        registered_traj = np.vstack((pc1, pc2)).T

        return pc1, pc2

    @staticmethod
    def fit_ARPW(traj, dt, max_P, max_S):

        pc1, pc2 = APRW.register_traj_disp(traj)

        msdpc1 = APRW.msd_1D(pc1) # MSD of x only
        msdpc2 = APRW.msd_1D(pc2) # MSD of y only

        # if dim3 == True:
        #     msdnp1 = msdnp1 + APRW.ezmsd0_1D(xyr_r[:, 1])

        popt_pc1, prw_pc1 = APRW.fit_PRW(msdpc1 + 1e-30, dt, max_P, max_S)
        popt_pc2, prw_pc2 = APRW.fit_PRW(msdpc2 + 1e-30, dt, max_P, max_S)

        return popt_pc1, popt_pc2

    @staticmethod
    def msd_1D(traj):
        fn = traj.shape[0] # Number of time frames
        msdr = np.zeros((fn-1,1))
        for dt in range(1,fn):
            dxyz = traj[dt:] - traj[:-dt] # displacement btw two adjacent points
            msdr[dt-1] = np.mean(dxyz**2,axis=0)
        return msdr

    @staticmethod
    def msd_allD(traj):
        fn = traj.shape[0] # Number of time frames
        msdr = np.zeros((fn-1,1))
        for dt in range(1,fn):
            dxyz = traj[dt:] - traj[:-dt]
            msdr[dt-1] = np.mean(np.sum(dxyz**2,axis=1))
        return msdr

    @staticmethod
    def PRW_model(t, P, S, se):
        return 1 * (S ** 2) * P * (t - P * (1 - np.exp(-t / P))) + 2 * se

    @staticmethod
    def fit_PRW(msd, dt, max_P, max_S):
        toi = np.arange(0, (msd.shape[0] / 2) - 1).astype(int)
        ydata = msd[toi].reshape(msd[toi].shape[0])

        ti = np.arange(1, msd.shape[0] / 2)

        Nt = len(ti)
        wif = (2 * ti ** 2 + 1) / 3 / ti / (Nt - ti + 1)
        wt = 1 / wif ** 2 / ydata

        xdata = ti * dt

        wt = np.diag(1 / wt)
        popt, pcov = curve_fit(APRW.PRW_model, xdata, ydata,
                               p0=[10, 1, 1], # po: initial guess for the parameters (P, S, se)
                               method='trf', # Use Trust Region Reflective if bound is provided (Constrained optimization)
                               bounds=([0, 0, 0], [max_P, max_S, 1000]),
                               sigma=wt, maxfev=1e10 # sigma: Determines the uncertainty in ydata
                               ) # maxfev: maximum number of iteration to find optimal value of parameters (P, S, se)
        return popt, APRW.PRW_model(xdata, *popt)


    @staticmethod
    def get_APRW(trajectories, dt, max_speed):
        HO_all = []
        for traj_idx in tqdm(trajectories):
            traj = trajectories[traj_idx]
            HO = APRW.get_HO(traj, dt=dt, max_P=traj.shape[0]*dt, max_S=max_speed)
            HO_all.append(HO)

        df_HO = pd.DataFrame(HO_all,columns=['MSD10','MSD100','Pp','Pnp','Dp','Dnp','Dtot','phi'])
        return df_HO
