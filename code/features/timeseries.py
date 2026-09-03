# Author: Chanhong Min <cmin11@jhmi.edu>

"""Calculate timeseries features"""

from scipy import stats
from utils.draw_utils import *
import EntropyHub as EH

class Timeseries(object):

    def __init__(self, timeseries):

        self.timeseries = timeseries

        self.average, self.variance, self.cov, self.skewness, self.kurtosis, self.ngaussalpha = self.get_distribution_props()

        self.total_distance, self.max_distance, self.peak_to_peak, self.slopes, \
        self.RMSs, self.crestfactor, self.formfactor, self.pulseindicator = self.get_timedomain_properties()

        self.approximate_entropies, self.attention_entropies, self.permutation_entropies, self.bubble_entropies, self.cosine_similarity_entropies, \
        self.corrected_conditional_entropies, self.dispersion_entropies, self.distribution_entropies, self.entropy_of_entropies, self.fuzzy_entropies, \
        self.gridded_distribution_entropies, self.increment_entropies, self.kolmogorov_entropies, self.phase_entropies, self.sample_entropies, \
        self.slope_entropies, self.spectral_entropies, self.symbolic_dynamic_entropies = self.get_various_entropies()


    def get_timedomain_properties(self):
        total_distance = {}
        max_distance = {}
        peak_to_peak = {}
        slopes = {}

        RMSs = {}
        crestfactor = {}
        formfactor = {}
        pulseindicator = {}

        for idx in self.timeseries:
            signal = self.timeseries[idx]

            total_distance[idx] = sum(signal)
            max_distance[idx] = max(signal)
            peak_to_peak[idx] = max(signal) - min(signal)

            time_range = range(signal.size)
            slope, intercept, r_val, p_val, SE = scipy.stats.linregress(time_range, signal)
            slopes[idx] = slope

            if all(signal == 0):  # cell with persistent contact
                RMSs[idx] = 0
                crestfactor[idx] = 0
                formfactor[idx] = 0
                pulseindicator[idx] = 0
            else:
                RMS = np.sqrt(np.mean(signal ** 2))
                RMSs[idx] = RMS
                crestfactor[idx] = max(signal) / RMS
                formfactor[idx] = RMS / np.mean(signal)
                pulseindicator[idx] = max(np.abs(signal)) / np.mean(signal)

        return total_distance, max_distance, peak_to_peak, slopes, RMSs, crestfactor, formfactor, pulseindicator


    def get_distribution_props(self):
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

        for idx in self.timeseries:
            ts = self.timeseries[idx]
            if all(ts == 0):  # cell with persistent contact
                variance[idx] = 0
                cov[idx] = 0
                skewness[idx] = 0
                kurtosis[idx] = 0
                ngaussalpha[idx] = 0
            else:
                average[idx] = np.mean(ts)
                variance[idx] = np.var(ts)
                cov[idx] = np.std(ts) / np.mean(ts)
                skewness[idx] = scipy.stats.skew(ts)
                kurtosis[idx] = scipy.stats.kurtosis(ts)
                ngaussalpha[idx] = np.mean(ts ** 4) / (3 * np.mean(ts ** 2) ** 2) - 1

        return average, variance, cov, skewness, kurtosis, ngaussalpha

    def get_various_entropies(self):

        approximate_entropies = {}
        attention_entropies = {}
        permutation_entropies = {}
        bubble_entropies = {}
        cosine_similarity_entropies = {}
        corrected_conditional_entropies = {}
        dispersion_entropies = {}
        distribution_entropies = {}
        entropy_of_entropies = {}
        fuzzy_entropies = {}
        gridded_distribution_entropies = {}
        increment_entropies = {}
        kolmogorov_entropies = {}
        phase_entropies = {}
        sample_entropies = {}
        slope_entropies = {}
        spectral_entropies = {}
        symbolic_dynamic_entropies = {}

        for idx in self.timeseries:
            ts = self.timeseries[idx]

            try:
                Ap, Phi = EH.ApEn(ts, m=3, Logx=np.e)  # ApEn(Sig, m=2, tau=1, r=None, Logx=numpy.exp)
                approximate_entropy = Ap[-1]

                Attn, (Hxx, Hnn, Hxn, Hnx) = EH.AttnEn(ts, Logx=np.e)
                attention_entropy = Attn

                Perm, Pnorm, cPE = EH.PermEn(ts, m=3, Logx=np.e)
                permutation_entropy = Perm[-1]

                Bubb, H = EH.BubbEn(ts, m=3, Logx=np.e)
                bubble_entropy = Bubb[-1]

                CoSi, Bm = EH.CoSiEn(ts, m=3, Logx=np.e)
                cosine_similarity_entropy = CoSi

                Cond, SEw, SEz = EH.CondEn(ts, m=3, Logx=np.e)
                corrected_conditional_entropy = Cond[-1]

                Dispx, Ppi = EH.DispEn(ts, m=3, Logx=np.e)
                dispersion_entropy = Dispx

                Dist, Ppi = EH.DistEn(ts, m=3, Logx=np.e)
                distribution_entropy = Dist

                EoE, AvEn, S2 = EH.EnofEn(ts, Logx=np.e, tau=5, S=5)
                entropy_of_entropy = EoE

                Fuzz, Ps1, Ps2 = EH.FuzzEn(ts, m=3, Logx=np.e)
                fuzzy_entropy = Fuzz[-1]

                GDE, GDR, _ = EH.GridEn(ts, m=4, Logx=np.e)
                gridded_distribution_entropy = GDE

                Incr = EH.IncrEn(ts, m=3, R=4, Logx=np.e)
                increment_entropy = Incr

                K2, Ci = EH.K2En(ts, m=3, Logx=np.e)
                kolmogorov_entropy = K2[-1]

                Phas = EH.PhasEn(ts, K=4, Logx=np.e)
                phase_entropy = Phas

                Samp, A, B = EH.SampEn(ts, m=3, Logx=np.e)
                sample_entropy = Samp[-1]

                Slop = EH.SlopEn(ts, m=3, tau=1, Lvls=(5, 45), Logx=np.e)
                slope_entropy = Slop[-1]

                Spec, BandEn = EH.SpecEn(ts, Freqs=(0, 1), Logx=np.e)
                spectral_entropy = Spec

                SyDy, Zt = EH.SyDyEn(ts, m=3, c=3, Logx=np.e)
                symbolic_dynamic_entropy = SyDy

            except:
                approximate_entropy = np.nan
                attention_entropy = np.nan # When no local maxima can be found
                permutation_entropy = np.nan
                bubble_entropy = np.nan
                cosine_similarity_entropy = np.nan
                corrected_conditional_entropy = np.nan
                dispersion_entropy = np.nan
                distribution_entropy = np.nan
                entropy_of_entropy = np.nan
                fuzzy_entropy = np.nan
                gridded_distribution_entropy = np.nan
                increment_entropy = np.nan
                kolmogorov_entropy = np.nan
                phase_entropy = np.nan
                sample_entropy = np.nan
                slope_entropy = np.nan
                spectral_entropy = np.nan
                symbolic_dynamic_entropy = np.nan

            approximate_entropies[idx] = approximate_entropy
            attention_entropies[idx] = attention_entropy
            permutation_entropies[idx] = permutation_entropy
            bubble_entropies[idx] = bubble_entropy
            cosine_similarity_entropies[idx] = cosine_similarity_entropy
            corrected_conditional_entropies[idx] = corrected_conditional_entropy
            dispersion_entropies[idx] = dispersion_entropy
            distribution_entropies[idx] = distribution_entropy
            entropy_of_entropies[idx] = entropy_of_entropy
            fuzzy_entropies[idx] = fuzzy_entropy
            gridded_distribution_entropies[idx] = gridded_distribution_entropy
            increment_entropies[idx] = increment_entropy
            kolmogorov_entropies[idx] = kolmogorov_entropy
            phase_entropies[idx] = phase_entropy
            sample_entropies[idx] = sample_entropy
            slope_entropies[idx] = slope_entropy
            spectral_entropies[idx] = spectral_entropy
            symbolic_dynamic_entropies[idx] = symbolic_dynamic_entropy

        return approximate_entropies, attention_entropies, permutation_entropies, bubble_entropies, cosine_similarity_entropies, \
               corrected_conditional_entropies, dispersion_entropies, distribution_entropies, entropy_of_entropies, fuzzy_entropies, \
               gridded_distribution_entropies, increment_entropies, kolmogorov_entropies, phase_entropies, sample_entropies, slope_entropies, \
               spectral_entropies, symbolic_dynamic_entropies

    def extract_features(self, feature_list):
        df = pd.DataFrame()
        for feature_name in feature_list:
            feature_dict = getattr(self, feature_name)
            df_temp = pd.DataFrame(feature_dict.values(), columns=[feature_name])
            df = pd.concat([df, df_temp], axis=1)

        return df





