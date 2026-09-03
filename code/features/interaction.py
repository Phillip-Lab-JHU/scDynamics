# Author: Chanhong Min <cmin11@jhmi.edu>

"""Calculates Interaction features"""

from scipy import stats
from utils.draw_utils import *


class DistanceSignal(object):
    def __init__(self, signals):

        self.signals = signals

        self.total, self.maximum, self.peak_to_peak, self.distance_slopes, \
        self.RMSs, self.crestfactor, self.formfactor, self.pulseindicator = self.get_timedomain_properties(self.signals)

        self.average, self.variance, self.cov, self.skewness, self.kurtosis, self.ngaussalpha = self.get_distribution_props(self.signals)

        self.autocorr, self.partial_autocorr = self.get_autocorr(self.signals)


    def get_timedomain_properties(self, signals):
        total = {}
        maximum = {}
        peak_to_peak = {}
        distance_slopes = {}

        RMSs = {}
        crestfactor = {}
        formfactor = {}
        pulseindicator = {}

        for idx in signals:  # u는 세포 하나의 index
            signal = signals[idx]  # cell은 t=0 ~ t=T 까지
            signal = np.array(signal)
            total[idx] = sum(signal)
            maximum[idx] = np.max(signal)
            peak_to_peak[idx] = np.max(signal) - np.min(signal)

            time_range = range(signal.size)
            slope, intercept, r_val, p_val, SE = scipy.stats.linregress(time_range, signal.flatten())
            distance_slopes[idx] = slope

            if all(signal == 0):  # cell with persistent contact
                RMSs[idx] = 0
                crestfactor[idx] = 0
                formfactor[idx] = 0
                pulseindicator[idx] = 0
            else:
                RMS = np.sqrt(np.mean(signal ** 2))
                RMSs[idx] = RMS
                crestfactor[idx] = np.max(signal) / RMS
                formfactor[idx] = RMS / np.mean(signal)
                pulseindicator[idx] = np.max(np.abs(signal)) / np.mean(signal)

        return total, maximum, peak_to_peak, distance_slopes, RMSs, crestfactor, formfactor, pulseindicator

    def get_distribution_props(self, signals):
        '''
        Calculates displacement distribution properties (variance, skewness, kurtosis, non gaussian parameter
        for each cell in trajectories
        Returns
        -------
        variance : dict keyed by cell_id with variance of displacement distribution
        skewness : dict keyed by trajectories with skewness of displacement distribution
        kurtosis : dict keyed by trajectories with kurtosis of displacement distribution
        ngaussalpha : dict keyed by trajectories with non gaussian parameter of displacement distribution
        '''
        average = {}
        variance = {}
        cov = {}  # coefficient of variation
        skewness = {}
        kurtosis = {}
        ngaussalpha = {}

        for idx in signals:
            signal = signals[idx]
            if all(signal == 0):  # cell with persistent contact
                average[idx] = 0
                variance[idx] = 0
                cov[idx] = 0
                skewness[idx] = 0
                kurtosis[idx] = 0
                ngaussalpha[idx] = 0
            else:
                average[idx] = np.mean(signal)
                variance[idx] = np.var(signal)
                cov[idx] = np.std(signal) / np.mean(signal)
                skewness[idx] = scipy.stats.skew(signal)
                kurtosis[idx] = scipy.stats.kurtosis(signal)
                ngaussalpha[idx] = np.mean(signal ** 4) / (3 * np.mean(signal ** 2) ** 2) - 1

        return average, variance, cov, skewness, kurtosis, ngaussalpha


    def get_autocorr(self, signals):
        '''
        Estimates the autocorrelation coefficient for each series of cell
        displacements over a range of time lags.
        Parameters
        ----------
        trajectories : dict of lists keyed by cell_id
        ea. list represents a cell. lists contain sequential tuples
        containing XY coordinates of a cell at a given timepoint
        Returns
        -------
        autocorr : dict of lists, containing autocorrelation coeffs for
        sequential time lags
        qstats : dict of lists containing Q-Statistics (Ljung-Box)
        pvals : dict of lists containing p-vals, as calculated from Q-Statistics
        Notes
        -----
        Estimation method:
        https://en.wikipedia.org/wiki/Autocorrelation#Estimation
        R(tau) = 1/(n-tau)*sigma**2 [sum(X_t - mu)*(X_t+tau - mu)] | t = [1,n-tau]
        X as a time series, mu as the mean of X, sigma**2 as variance of X
        tau as a given time lag (sometimes referred to as k in literature)
        Implementation uses statsmodels.tsa.stattools.acf()
        n.b. truncated to taus [1,10], to expand to more time lags, simply
        alter the indexing being loaded into the return dicts
        '''
        from statsmodels.tsa.stattools import acf, pacf
        autocorr = {}
        partial_autocorr = {}

        for idx in signals:
            signal = signals[idx]
            signal = np.array(signal)

            traj_autocorr = {}
            traj_partial_autocorr = {}

            # Perform Ljung-Box Q-statistic calculation to determine if autocorrelations detected are significant or random
            if np.unique(signal).size == 1:  # Non-motile ( mostly [0, 0, 0, 0, 0, 0, ...])
                nlags = int(min(10 * np.log10(signal.size), signal.size - 1)) + 1
                # Default number of time lags for statsmodels.tsa.stattools.acf
                for i in range(1, nlags):
                    traj_autocorr[i] = np.nan
                nlags = int(min(10 * np.log10(signal.size), signal.size // 2 - 1)) + 1
                # Default number of time lags for statsmodels.tsa.stattools.pacf
                for i in range(1, nlags):
                    traj_partial_autocorr[i] = np.nan
                autocorr[idx] = traj_autocorr
                partial_autocorr[idx] = traj_partial_autocorr

            else:
                ac = acf(signal)  # time lag 0 ~ max_tau 까지 autocorrelation 계산
                pac = pacf(signal)  # time lag 0 ~ max_tau 까지 partial autocorrelation 계산

                for i in range(1, ac.shape[0]): # always autocorr[0] = 1
                    traj_autocorr[i] = ac[i]
                for i in range(1, pac.shape[0]): # always partial_autocorr[0] = 1
                    traj_partial_autocorr[i] = pac[i]
                autocorr[idx] = traj_autocorr
                partial_autocorr[idx] = traj_partial_autocorr

        return autocorr, partial_autocorr

    def extract_features(self, feature_list, tau_limit):
        motility_data_basic = pd.DataFrame()
        for feature_name in feature_list:
            feature_dict = getattr(self, feature_name)
            if feature_name == 'msd' or 'autocorr' in feature_name:
                for tau in feature_dict[0].keys():
                    if tau <= tau_limit:
                        temp = {keys: feature_dict[keys][tau] for keys in feature_dict}
                        df_temp = pd.DataFrame(temp.values(), columns=[feature_name + '_%s' % str(tau)])
                        motility_data_basic = pd.concat([motility_data_basic, df_temp], axis=1)
            else:
                df_temp = pd.DataFrame(feature_dict.values(), columns=[feature_name])
                motility_data_basic = pd.concat([motility_data_basic, df_temp], axis=1)


        return motility_data_basic


class OverlapSignal(object):
    def __init__(self, signals):
        self.signals = signals

        self.avg_overlap, self.overlap_slopes = self.get_timedomain_properties(self.signals)
        self.contact_times, self.contact_persistences, self.noncontact_times, self.noncontact_persistences = self.get_quality_properties(self.signals)

    def get_timedomain_properties(self, signals):
        avg_overlap = {}
        overlap_slopes = {}

        for idx in signals:  # u는 세포 하나의 index
            signal = signals[idx]  # cell은 t=0 ~ t=T 까지
            signal = np.array(signal)
            avg_overlap[idx] = np.mean(signal)
            time_range = range(signal.size)
            slope, intercept, r_val, p_val, SE = scipy.stats.linregress(time_range, signal.flatten())
            overlap_slopes[idx] = slope

        return avg_overlap, overlap_slopes

    def get_quality_properties(self, signals):
        from itertools import groupby
        contact_times = {}
        contact_persistences = {}
        noncontact_times = {}
        noncontact_persistences = {}

        for idx in signals:  # u는 세포 하나의 index
            signal = signals[idx]  # cell은 t=0 ~ t=T 까지
            contact_profile = signal > 0
            contact_profile = contact_profile * 1
            if np.mean(contact_profile) == 0:  # interaction_profile = [0, 0, 0, 0, ... 0]
                contact_times[idx] = 0
                contact_persistences[idx] = 0
            else:
                contact_time = np.sum(contact_profile)
                contact_persistence = np.max([sum(group) for element, group in groupby(contact_profile) if element == 1])
                contact_times[idx] = contact_time
                contact_persistences[idx] = contact_persistence

            noncontact_profile = signal == 0
            noncontact_profile = noncontact_profile * 1
            if np.mean(noncontact_profile) == 0:  # interaction_profile = [0, 0, 0, 0, ... 0]
                noncontact_times[idx] = 0
                noncontact_persistences[idx] = 0
            else:
                noncontact_time = np.sum(noncontact_profile)
                noncontact_persistence = np.max([sum(group) for element, group in groupby(noncontact_profile) if element == 1])
                noncontact_times[idx] = noncontact_time
                noncontact_persistences[idx] = noncontact_persistence

        return contact_times, contact_persistences, noncontact_times, noncontact_persistences

    def extract_features(self, feature_list):
        motility_data_basic = pd.DataFrame()
        for feature_name in feature_list:
            feature_dict = getattr(self, feature_name)
            df_temp = pd.DataFrame(feature_dict.values(), columns=[feature_name])
            motility_data_basic = pd.concat([motility_data_basic, df_temp], axis=1)

        return motility_data_basic


class ZoneSignal(object):
    def __init__(self, signals):
        self.signals = signals

        self.avg_zone = self.get_timedomain_properties(self.signals)
        self.dz_resident_times, self.dz_resident_persistences, self.slz_resident_times, self.slz_resident_persistences, \
        self.dlz_resident_times, self.dlz_resident_persistences = self.get_quality_properties(self.signals)

    def get_timedomain_properties(self, signals):
        avg_zone = {}

        for idx in signals:  # u는 세포 하나의 index
            signal = signals[idx]  # cell은 t=0 ~ t=T 까지
            avg_zone[idx] = np.mean(signal)

        return avg_zone

    def get_quality_properties(self, signals):
        from itertools import groupby
        dz_resident_times = {}
        dz_resident_persistences = {}
        slz_resident_times = {}
        slz_resident_persistences = {}
        dlz_resident_times = {}
        dlz_resident_persistences = {}

        for idx in signals:  # u는 세포 하나의 index
            signal = signals[idx]  # cell은 t=0 ~ t=T 까지
            dz_profile = signal == 0
            slz_profile = signal == 1
            dlz_profile = signal == 2

            dz_profile = dz_profile * 1
            slz_profile = slz_profile * 1
            dlz_profile = dlz_profile * 1

            if np.mean(dz_profile) == 0:  # dz_profile = [0, 0, 0, 0, ... 0]
                dz_resident_times[idx] = 0
                dz_resident_persistences[idx] = 0

            else:
                dz_time = np.sum(dz_profile)
                dz_resident_persistence = np.max([sum(group) for element, group in groupby(dz_profile) if element == 1])
                dz_resident_times[idx] = dz_time
                dz_resident_persistences[idx] = dz_resident_persistence

            if np.mean(slz_profile) == 0:  # slz_profile = [0, 0, 0, 0, ... 0]
                slz_resident_times[idx] = 0
                slz_resident_persistences[idx] = 0

            else:
                slz_time = np.sum(slz_profile)
                slz_resident_persistence = np.max([sum(group) for element, group in groupby(slz_profile) if element == 1])
                slz_resident_times[idx] = slz_time
                slz_resident_persistences[idx] = slz_resident_persistence

            if np.mean(dlz_profile) == 0:  # dlz_profile = [0, 0, 0, 0, ... 0]
                dlz_resident_times[idx] = 0
                dlz_resident_persistences[idx] = 0

            else:
                dlz_time = np.sum(dlz_profile)
                dlz_resident_persistence = np.max([sum(group) for element, group in groupby(dlz_profile) if element == 1])
                dlz_resident_times[idx] = dlz_time
                dlz_resident_persistences[idx] = dlz_resident_persistence


        return dz_resident_times, dz_resident_persistences, slz_resident_times, slz_resident_persistences, dlz_resident_times, dlz_resident_persistences

    def extract_features(self, feature_list):
        motility_data_basic = pd.DataFrame()
        for feature_name in feature_list:
            feature_dict = getattr(self, feature_name)
            df_temp = pd.DataFrame(feature_dict.values(), columns=[feature_name])
            motility_data_basic = pd.concat([motility_data_basic, df_temp], axis=1)

        return motility_data_basic
