# Author: Chanhong Min <cmin11@jhmi.edu>

"""Calculates Directional features wrt FDC and T cell"""

from utils.draw_utils import *

class Directionality(object):
    def __init__(self, signals):
        self.signals = signals
        self.approach_times, self.approach_persistences, self.departure_times, self.departure_persistences, self.stay_times, self.stay_persistences\
            = self.get_properties(self.signals)

    def get_properties(self, signals):
        from itertools import groupby

        approach_times = {}
        approach_persistences = {}
        departure_times = {}
        departure_persistences = {}
        stay_times = {}
        stay_persistences = {}

        for idx in signals:  # u는 세포 하나의 index
            signal = signals[idx]  # cell은 t=0 ~ t=T 까지

            approach_profile = signal =='approach'  # ['True', 'False', ...]
            departure_profile = signal =='departure'
            stay_profile = signal == 'stay'
            approach_profile = approach_profile * 1  # [1, 0, ...]
            departure_profile = departure_profile * 1
            stay_profile = stay_profile * 1

            if np.mean(approach_profile) == 0:  # profile = [0, 0, 0, 0, ... 0]
                approach_times[idx] = 0
                approach_persistences[idx] = 0

            else:
                approach_time = np.sum(approach_profile)
                approach_persistence = max([sum(group) for element, group in groupby(approach_profile) if element == 1])
                # if profile = [0, 0, 0, 0, ... 0], max(empty list) -> error
                approach_times[idx] = approach_time
                approach_persistences[idx] = approach_persistence

            if np.mean(departure_profile) == 0:  # interaction_profile = [0, 0, 0, 0, ... 0]
                departure_times[idx] = 0
                departure_persistences[idx] = 0

            else:
                departure_time = np.sum(departure_profile)
                departure_persistence = max([sum(group) for element, group in groupby(departure_profile) if element == 1])
                # if profile = [0, 0, 0, 0, ... 0], max(empty list) -> error
                departure_times[idx] = departure_time
                departure_persistences[idx] = departure_persistence

            if np.mean(stay_profile) == 0:  # interaction_profile = [0, 0, 0, 0, ... 0]
                stay_times[idx] = 0
                stay_persistences[idx] = 0

            else:
                stay_time = np.sum(stay_profile)
                stay_persistence = max([sum(group) for element, group in groupby(stay_profile) if element == 1])
                # if profile = [0, 0, 0, 0, ... 0], max(empty list) -> error
                stay_times[idx] = stay_time
                stay_persistences[idx] = stay_persistence


        return approach_times, approach_persistences, departure_times, departure_persistences, stay_times, stay_persistences

    def extract_features(self, feature_list):
        motility_data_basic = pd.DataFrame()
        for feature_name in feature_list:
            feature_dict = getattr(self, feature_name)
            df_temp = pd.DataFrame(feature_dict.values(), columns=[feature_name])
            motility_data_basic = pd.concat([motility_data_basic, df_temp], axis=1)

        return motility_data_basic
