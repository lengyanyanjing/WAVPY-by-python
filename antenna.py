import const
import math
import numpy as np
from numpy.linalg import norm


angles_default = np.deg2rad([0.0, 20.0, 40.0, 45.0, 50.0, 65.0, 80.0, 90.0])
pattern_default = np.array([12.8, 8.26, -6.3, -15.0, -7.44, -3.3, -5.7, -15.0])


class Antenna:
    def __init__(self):
        self.freq = const.FREQ1    # singal frequency in Hz
        self.sys = const.SYS_GPS
        self.isotropic_flg = True  # iostropic antenna flag
        self.antenna_gain = 3.0    # antenna gain in dBw
        self.antenna_temp = 200.0  # antenna temperature in K
        self.effective_area = 0.00575  # antenna effective area
        self.noise_figure = 3.0      # noise figure in dB
        self.noise_temp = 488.626    # noise temperature in K
        self.filter_bb_bw = 5.0e6    # filter baseband bandwidth
        self.noise_power = -134.719  # noise power in dBW

        """
        define antenna frame should be unit vector, Antenna frame's X-axis in
        the receiver body frame, defines the orientation of the antenna
        in the receiver frame
        """
        self.antenna_vec_BF_E = np.array([1.0, 0.0, 0.0])
        self.antenna_vec_BF_H = np.array([0.0, 0.0, 0.0])
        self.antenna_vec_BF_k = np.cross(self.antenna_vec_BF_E, self.antenna_vec_BF_H)

    def __compute_effective_area__(self):
        """ compute antenna effective area """
        tmp = pow(const.C_LIGHT/self.freq, 2)*pow(10.0, self.antenna_gain/10.0)
        self.effective_area = tmp/4.0/const.PI

    def __compute_noise_temp__(self):
        """ compute noise temperature """
        self.noise_temp = 290.0*(pow(10.0, self.noise_figure/10.0)-1.0)
        self.noise_temp += self.antenna_temp

    def __compute_noise_power__(self):
        """ compute noise power """
        tmp = const.K_BOLTZMANN*self.noise_temp*self.filter_bb_bw
        self.noise_power = 10.0*math.log10(tmp)

    def set_antenna_orientation(self, ant_BF_E, ant_BF_H):
        """
        set antenna orientation,
        equal to the perpendicularly towards the propagation direction
        """
        self.antenna_vec_BF_E = ant_BF_E/norm(ant_BF_E)
        self.antenna_vec_BF_H = ant_BF_H/norm(ant_BF_H)
        self.antenna_vec_BF_k = np.cross(self.antenna_vec_BF_E,
                                         self.antenna_vec_BF_H)

    def set_receiver(self, gain, temp, figure, fbb_bw, antenna_flg=False):
        self.antenna_gain = gain
        self.antenna_temp = temp
        self.noise_figure = figure
        self.filter_bb_bw = fbb_bw
        self.isotropic_flg = antenna_flg
        self.__compute_effective_area__()
        self.__compute_noise_temp__()
        self.__compute_noise_power__()

    def get_antenna_bf(self):
        return (self.antenna_vec_BF_E, self.antenna_vec_BF_H,
                self.antenna_vec_BF_k)

    def get_receiver_gain(self, paras):
        """
        return the receiver antenna gain at offboresight angles and
        around-boresight angles, in dB
        """
        theta, phi = paras
        return 14.205198  # -11.3341

