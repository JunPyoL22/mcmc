
import numpy as np

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


# stdtd = StudentTDistribuiton(0, 5, 10, samp_size=15)
# std_t_distributrion = stdtd.distribution

if __name__ == "__main__":
    stdtd = StudentTDistribuiton(0, 5, 10, samp_size=15)
    std_t_distributrion = stdtd.distribution