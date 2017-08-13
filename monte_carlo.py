import numpy as np
import scipy.stats as scsts

#1.  Sampling student t_distribution using Method of Composition algorithm.
class StudentTDistribuiton:
    def __init__(self, mean, scale, freedom, samp_size=100):
        '''
            mean: int, mean of t distribution
            scale: int, scale parameter of t distribution
            freedom: int, freedom number of t distribution
            samp_size: int, sampling size
        '''
        self.m = mean
        self.scl = scale
        self.v = freedom
        self.s_size = samp_size
        self.stud_t_dist = None

    @property
    def distribution(self):
        self.do_sampling()
        return self.stud_t_dist

    def do_sampling(self):
        # sampling by method of composition
        z = self._sampling_from_gamma()
        self.stud_t_dist = self._sampling_from_conditional_normal(z)

    def _sampling_from_gamma(self):
        return np.random.gamma(self.v/2, scale=1.0, size=self.s_size)
    
    def _sampling_from_conditional_normal(self , z):
        std_t_dist = np.empty(self.s_size)
        for i in range(self.s_size):
            scale = np.dot(1/z[i], self.scl)
            std_t_dist[i] = np.random.normal(self.m, scale, size=1)
        return std_t_dist

#2.  Sampling Truncated Noraml distribution using Probablity Integral Transformation algorithm.
class TruncatedNormal:
    def __init__(self, lower, upper, samp_size=100):
        self.interval_lower = lower
        self.interval_upper = upper
        self.s_size = samp_size
        self._distribution = None

    @property
    def distribution(self):
        self.do_sampling()
        return self._distribution

    def do_sampling(self):
        uniform = self.sampling_from_uniform()
        self._distribution = self.calculate_from_inverse_cumulative_distribution(uniform) 

    def sampling_from_uniform(self):
        return np.random.uniform(size=self.s_size)     

    def calculate_from_inverse_cumulative_distribution(self, uniform):
        inv_cumul = np.empty(self.s_size)
        for i in range(self.s_size):
            value = ststs.norm.cdf(self.interval_lower) \
                + uniform[i]*(ststs.norm.cdf(self.interval_upper) \
                              - ststs.norm.cdf(self.interval_lower))

            inv_cumul[i] = ststs.norm.ppf(value ,loc=0, scale=1)
        return inv_cumul

# stdtd = StudentTDistribuiton(0, 5, 10, samp_size=15)
# std_t_distributrion = stdtd.distribution

if __name__ == "__main__":
    #test
    stdtd = StudentTDistribuiton(0, 5, 10, samp_size=15)
    std_t_dsist = stdtd.distribution

    tc_normal_dist = TruncatedNormal(-2, -3, 10).distribution
