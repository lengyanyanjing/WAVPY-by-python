import numpy as np
import const
from math import sin, cos, tan, atan2, exp, sqrt
import matplotlib.pyplot as plt
from antenna import Antenna
from gsignal import Signal
from waveform import Waveform
from geometry import Geometry
from interface import Interface
from numpy.linalg import norm


class Simulator(object):
    def __init__(self):
        self.wind_speed = 5.0          # default wind speed
        self.coherent_pow_flg = True   # add coherent power
        self.coh_integrate_time = 0.001  # coherent integrateed time
        self.num_angles = 360          # integrated angle resolution
        self.interface_flg = 1         # curvature surface
        self.ddm_cov_factor = 5        # cov mode factor for ddm
        # # atmosphere loss for signal propagation 0.5 dB
        self.atmospheric_loss = pow(10.0, (0.5/10.0))
        # # members
        self.geometry = Geometry()
        self.nadir_RF = Antenna()
        self.zenith_RF = Antenna()
        self.signal = Signal()
        self.power_waveform = Waveform(True, 1000)
        self.interface = Interface()

    def plot_delay_waveform(self, flg=False):
        """ plot delay simulated waveform """
        if flg:
            waveform = self.integrate_waveform
            title = "Integrated waveform"
        else:
            waveform = self.waveform
            title = "Delay waveform"
        noise_level = waveform[0]
        plt.figure()
        plt.plot(self.wave_range/1000.0,
                 10.0*np.log10(waveform/noise_level), '*-')
        plt.grid()
        plt.title(title)
        plt.xlabel("Range from specular [km]")
        plt.ylabel("SNR [dB]")
        plt.ylim(0, 5)
        plt.xlim(-1.0, 5.0)
        plt.tight_layout()
        plt.show()

    def plot_power_ddm(self):
        """ plot scattered power DDM """
        plt.figure(figsize=(4, 3))
        noise_level = self.ddm.min()
        extent = [self.wave_range[0]/1000.0,
                  self.wave_range[-1]/1000.0,
                  -self.dopp_bin*self.center_dopp/1000.0,
                  self.dopp_bin*self.center_dopp/1000.0]
        plt.imshow(self.ddm, extent=extent, vmin=noise_level,
                   vmax=max(self.ddm[self.center_dopp, :]),
                   cmap='jet', aspect='auto')
        plt.title("Scattered power DDM")
        plt.xlabel("Range from specular [km]")
        plt.ylabel("Doppler [kHz]")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def plot_scattered_area(self, flg=True):
        """ plot scattered area """
        if flg:
            sca_area = self.eff_area
            title = "Effective scatter area"
        else:
            sca_area = self.phy_area
            title = "Physical scatter area"
        plt.figure(figsize=(4, 3))
        noise_level = sca_area.min()
        max_pow = max(sca_area[self.center_dopp, :])
        extent = [self.wave_range[0]/1000.0,
                  self.wave_range[-1]/1000.0,
                  -self.dopp_bin*self.center_dopp/1000.0,
                  self.dopp_bin*self.center_dopp/1000.0]
        plt.imshow(sca_area, extent=extent, vmin=noise_level,
                   vmax=max_pow, cmap='jet', aspect='auto')
        plt.title(title)
        plt.xlabel("Range from specular [km]")
        plt.ylabel("Doppler [kHz]")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    ###########################################################################
    def compute_direct_power(self, transmit_pow):
        """ compute direct signal power """
        # # receiver body frame
        bf_e, bf_h, bf_k = self.zenith_RF.get_antenna_bf()
        # # receiver body frame to specular
        self.geometry.BF2spfs(bf_e, bf_h, bf_k)
        scat_vec, rang = self.geometry.compute_r2t_vector()
        angles = self.geometry.compute_antenna_gain_pos(scat_vec)
        directivity = pow(10.0, (self.nadir_RF.get_receiver_gain(angles))/10.0)
        tmp = const.C_LIGHT/(self.signal.freq*4.0*const.PI*rang)
        self.direct_pow = pow(tmp, 2)
        self.direct_pow *= transmit_pow*directivity/self.atmospheric_loss

    def correlate_direct_signal(self):
        sin_arg = 2*self.zenith_RF.filter_bb_bw/self.sampling_rate
        sin_arg *= np.arange(0, self.range_samples_len)
        sin_arg[sin_arg == 0.0] = 1.0
        sin_arg = np.sinc(sin_arg)
        self.nd_nr_cov = abs(sin_arg)
        self.sd_nr_cov = np.convolve(sin_arg, self.signal.lambda_fun,
                                     mode="same")
        self.sd_nr_cov = np.abs(self.sd_nr_cov)
        max_value = self.sd_nr_cov.max()
        if max_value > 0.0:
            self.sd_nr_cov = self.sd_nr_cov/max_value
        for i in range(self.dopps):
            power = self.power[:, i]
            pass

    ###########################################################################
    def compute_coherent_power(self, scat_vec):
        """
        compute coherent power at scat_vec diretion, commonly coherent
        relection accur at specular point
        """
        # # compute receiver antenna gain at scat_vec diretion
        # # receiver body frame
        bf_e, bf_h, bf_k = self.nadir_RF.get_antenna_bf()
        # # receiver body frame to specular
        self.geometry.BF2spfs(bf_e, bf_h, bf_k)
        angles = self.geometry.compute_antenna_gain_pos(scat_vec)
        directivity = pow(10.0, (self.nadir_RF.get_receiver_gain(angles))/10.0)
        # # specular point sinc function
        self.cosi = cos(self.geometry.tx_inc)
        self.tani = tan(self.geometry.tx_inc)
        self.sinc_dopps = np.zeros(self.dopps)
        sinc_arg = self.dopp_bin*self.coh_integrate_time
        if sinc_arg == 0.0:
            self.sinc_dopps[self.center_dopp] = 1.0
        else:
            self.sinc_dopps[self.center_dopp] = pow(np.sinc(sinc_arg), 2)
        # # compute fresnel coefficients
        self.interface.compute_fresnel(self.geometry.tx_inc)
        # # get corherent power
        tmp = 4.0*const.PI*self.interface.sigma_z*self.cosi
        coherent_pow = self.signal.trans_pow*self.interface.fresnel
        coherent_pow *= exp(-1.0*pow(tmp/self.signal.wavelen, 2))
        tmp = const.C_LIGHT*self.coh_integrate_time
        tmp /= (4.0*const.PI*self.signal.freq)
        tmp /= (norm(self.geometry.tx_spf)+norm(self.geometry.rx_spf))
        coherent_pow *= directivity*self.sinc_dopps[self.center_dopp]
        coherent_pow *= pow(tmp, 2)/self.atmospheric_loss
        self.coherent_pow = coherent_pow

    def set_pow_waveform(self, init_range, sampling_rate):
        """ set power waveform for delays computation """
        self.power_waveform.sampling_rate = sampling_rate
        self.power_waveform.set_init_range(init_range)

    def set_nadir_antenna(self, gain, temp, figure,
                          filter_bb_bw, antenna_flg=True):
        """
        set antenna attitude information for receiver antenna gain
        computation
        """
        self.nadir_RF.set_receiver(gain, temp, figure, filter_bb_bw,
                                   antenna_flg)

    def set_radar_signal(self, sampling_rate, filter_bb_bw, exponent):
        """ initailze the bistatic radar signal """
        self.signal.set_radar_signal(sampling_rate, filter_bb_bw)
        # # compute corelation function of WAF
        self.signal.compute_lambda()
        self.isotropic_factor = (self.signal.wavelen**2)/(4.0*const.PI)**3
        self.dt = const.C_LIGHT/sampling_rate  # just for computation later
        # # compute the transmit power
        ele = const.PI/2-self.geometry.tx_inc
        self.signal.compute_transmit_power(self.signal, ele)

    def set_interface(self, wind_speed):
        self.interface.ws = wind_speed
        self.interface.set_polarization(self.polar_mode)

    def configure_radar_geometry(self, tx, tv, rx, rv, undulation_flg=True):
        """
        set bistatic radar configuration, need the ecef postion and velocity
        of transimiter and receiver, compute specular point postion. function
        can also account for the undualtion of geoid
        """
        self.geometry.tx_pos = np.asarray(tx)
        self.geometry.tx_vel = np.asarray(tv)
        self.geometry.rx_pos = np.asarray(rx)
        self.geometry.rx_vel = np.asarray(rv)
        # # compute the specular point
        self.geometry.compute_sp_pos(undulation_flg)

    def earth_curvature_appro(self, tau, x):
        """ modified surface glisten zone coordinations for earth curvature """
        rad = norm(x[:2])
        az = atan2(x[1], x[0])
        x[2] = sqrt(const.RE_WGS84**2-rad**2)-const.RE_WGS84
        rr = norm(self.geometry.rx_spf-x)
        rt = norm(self.geometry.tx_spf-x)
        delay = rr+rt-self.geometry.rrt-self.sp_delay
        rad *= sqrt(tau/delay)
        x[0] = rad*cos(az)
        x[1] = rad*sin(az)
        x[2] = sqrt(const.RE_WGS84**2-rad**2)-const.RE_WGS84

    def compute_sinc(self, dopp):
        """ compute doppler sinc function """
        sinc_arg = (np.arange(self.dopps)-self.center_dopp)*self.dopp_bin
        sinc_arg = (dopp-sinc_arg)*self.coh_integrate_time
        ind = sinc_arg != 0.0
        self.sinc_dopps[ind] = pow(np.sinc(sinc_arg[ind]), 2)
        self.sinc_dopps[~ind] = 1.0

    def delay_integration(self, tau, a, c, delta):
        """
        integration points over the surface ellipse for each range sample
        """
        x = np.zeros(3)
        pow_tau = np.zeros(self.dopps)
        phy_area = np.zeros(self.dopps)
        eff_area = np.zeros(self.dopps)
        left_side = -1.0*(self.center_dopp+0.5)*self.dopp_bin
        for i in range(self.num_angles):
            # # surface point calculation
            theta = i*delta
            x[0] = a*self.cosi*cos(theta)
            x[1] = a*sin(theta)+c
            # # surface point earth curvature modified
            if self.interface_flg == 1:
                self.earth_curvature_appro(tau, x)
            # # surface point scatter vector and scatter area
            inc_vec, sca_vec, jacob, coeff = self.geometry.compute_scattering_vector(x)
            # # surface point relative doppler shift to the specular point
            dopp = self.geometry.doppler_shift(inc_vec, sca_vec,
                                               self.signal.freq)-self.sp_dopp
            # if self.dopps % 2 == 0:
            #     dopp -= self.dopp_bin
            # # sinc function
            self.compute_sinc(dopp)
            # # reflected coeffcient
            simga0 = self.interface.compute_scattered_coeff(inc_vec, sca_vec,
                                                            self.geometry.tx_az)
            # # receicer antenna gain at the surface point direction
            angles = self.geometry.compute_antenna_gain_pos(inc_vec)
            rev_gain_db = self.nadir_RF.get_receiver_gain(angles)
            directivity = pow(10.0, rev_gain_db/10.0)
            # # a factor for correlation calculation
            factor = directivity*self.isotropic_factor*simga0*jacob*coeff
            if i == 0:
                # # restore the first surface point
                fx = np.copy(x[:2])
                px = np.copy(x[:2])  # the former point relative to current one
                fst_dopp = pre_dopp = dopp
                fst_jac = pre_jac = jacob
                fst_ft = factor*self.sinc_dopps
                pre_ft = fst_ft.copy()
                fst_area = jacob*self.sinc_dopps
                pre_area = fst_area.copy()
                continue
            diff_ang = abs(atan2(x[1], x[0])-atan2(px[1], px[0]))
            new_theta = min(diff_ang, 2.0*const.PI-diff_ang)
            px = np.copy(x[:2])
            tmp = factor*self.sinc_dopps
            area = jacob*self.sinc_dopps
            pow_tau += new_theta*(tmp+pre_ft)/2.0    # accumulate the power
            ind = int(((dopp+pre_dopp)/2.0-left_side)//self.dopp_bin)
            if (ind >= 0) and (ind < self.dopps):
                phy_area[ind] += (jacob+pre_jac)/2.0*new_theta
            eff_area += new_theta*(area+pre_area)/2.0
            pre_dopp = dopp
            pre_jac = jacob
            pre_ft = tmp.copy()
            pre_area = area.copy()
            if i == self.num_angles-1:
                # # intergration to finish the whole ellipse, connect
                # # the last point to the first point
                diff_ang = abs(atan2(fx[1], fx[0])-atan2(x[1], x[0]))
                new_theta = min(diff_ang, 2.0*const.PI-diff_ang)
                pow_tau += new_theta*(fst_ft+tmp)/2.0   # accumulate the power
                ind = int(((dopp+fst_dopp)/2.0-left_side)//self.dopp_bin)
                if (ind >= 0) and (ind < self.dopps):
                    phy_area[ind] += (jacob+fst_jac)/2.0*new_theta
                eff_area += new_theta*(fst_area+area)/2.0
        return pow_tau, phy_area, eff_area

    def compute_noise_floor(self):
        """ compute noise floor """
        eta_factor = 1.0
        self.noise_floor = eta_factor*pow(10.0, self.nadir_RF.noise_power/10.0)
        self.noise_floor /= self.nadir_RF.filter_bb_bw*self.coh_integrate_time

    def compute_power_waveform(self, ind):
        """ get lambda function """
        lam_len = self.signal.lambda_size
        half_lam = lam_len//2
        waveform_conv = np.convolve(self.power[:, ind],
                                    self.signal.lambda_fun, mode="same")
        area_conv = np.convolve(self.dis_eff_area[:, ind],
                                self.signal.lambda_fun, mode="same")
        # # compute delay power waveform
        tmp = waveform_conv[half_lam:half_lam+self.delays]
        tmp *= self.signal.trans_pow*self.dt/self.atmospheric_loss
        tmp += self.noise_floor
        if lam_len > self.delays:
            lam_len = self.delays
        tmp[:lam_len] += self.coherent_pow*self.signal.lambda_fun[:lam_len]
        return abs(tmp), area_conv[half_lam:half_lam+self.delays]

    def compute_ddm(self):
        """ compute ddm of scattered surface """
        half_lam = self.signal.lambda_size//2
        self.ddm = np.zeros((self.dopps, self.delays))
        self.eff_area = np.zeros((self.dopps, self.delays))
        self.phy_area = self.dis_phy_area[half_lam:half_lam+self.delays,
                                          range(self.dopps)].T
        # # zero doppler shift delay waveform
        for i in range(self.dopps):
            sca_pow, eff_area = self.compute_power_waveform(i)
            self.ddm[i, :] = sca_pow
            self.eff_area[i, :] = eff_area
        # # integrated waveform
        self.integrate_waveform = self.ddm.sum(axis=0)
        if not hasattr(self, "wave_range"):
            self.power_waveform.set_waveform(self.ddm[self.center_dopp, :])
            self.power_waveform.compute_waveform_delays()
            self.wave_range = self.power_waveform.get_range_waveform()
            self.wave_range -= self.geometry.geometric_delay

    def compute_center_waveform(self):
        """ compute delay power waveform  """
        self.compute_noise_floor()
        self.waveform, _ = self.compute_power_waveform(self.center_dopp)
        # # compute spatical delays of delay waveform in meters
        self.power_waveform.set_waveform(self.waveform)
        self.power_waveform.compute_waveform_delays()
        self.wave_range = self.power_waveform.get_range_waveform()
        self.wave_range -= self.geometry.geometric_delay

    def compute_power_distribution(self):
        """
        Computation power distribution over reflecting surface
        origin located at sample = gnss_signal.lambda_size
        """
        # # signal correlation starting postion
        lam_len = self.signal.lambda_size
        end = self.range_samples_len-lam_len
        self.power = np.zeros((self.range_samples_len, self.dopps))
        self.dis_phy_area = np.zeros((self.range_samples_len, self.dopps))
        self.dis_eff_area = np.zeros((self.range_samples_len, self.dopps))
        for i in range(lam_len, end):
            # # compute relative delay
            tau = (i-lam_len)*self.dt
            tau = 1.0e-6 if i == lam_len else tau
            # # compute absolute delay
            tau_abs = tau+self.sp_delay
            a = tau_abs/self.cosi**2*sqrt(1.0-self.sp_delay/tau_abs)
            c = self.tani/self.cosi*tau
            delta = 2.0*const.PI/self.num_angles
            sca_pow, phy_area, eff_area = self.delay_integration(tau, a, c, delta)
            self.power[i, :] = sca_pow
            self.dis_phy_area[i, :] = phy_area
            self.dis_eff_area[i, :] = eff_area

    def compute_sp_info(self):
        """
        compute the delay/dopper on the specular point, also calculate the
        coherent power on the specular point for the coherent reflection
        """
        # # compute scattering vector
        inc_vec = -1.0*self.geometry.tx_spf/norm(self.geometry.tx_spf)
        scat_vec = self.geometry.rx_spf/norm(self.geometry.rx_spf)
        # # delay and dopper at specular point
        self.sp_dopp = self.geometry.doppler_shift(inc_vec, scat_vec,
                                                   self.signal.freq)
        self.sp_delay = self.geometry.geometric_delay
        # # coherent power
        if self.coherent_pow_flg:
            self.compute_coherent_power(scat_vec)

    def set_model(self, rx_pos, rx_vel, tx_pos, tx_vel, cov_mode=False):
        """ set model for simulator initalization"""
        self.ddm_cov_mode = cov_mode
        # # equal 244ns( 1/(1/1023000/4)), it is ddm delay sampling rate,
        self.sampling_rate = 4091750.0    # 4092000
        self.range_len_exponent = 8
        self.delays = 17                # DDM delay chips
        self.dopps = 11                 # DDM doppler bins
        self.dopp_bin = 500.0           # doppler resolution unit in Hz
        self.filter_bb_bw = 5000000.0   # receiver baseband bandwidth in Hz
        self.polar_mode = "RL"          # poliariztion of reflected signal
        # # variable initialize
        if self.ddm_cov_mode:
            self.dopps = (self.dopps-1)*2*self.ddm_cov_factor
        self.center_dopp = self.dopps//2
        self.range_samples_len = 2**self.range_len_exponent
        # # set bistatic radar geometry
        self.configure_radar_geometry(tx_pos, tx_vel, rx_pos, rx_vel, True)
        # # set interface
        self.set_interface(self.wind_speed)
        # # set radar signal
        self.set_radar_signal(self.sampling_rate, self.filter_bb_bw,
                              self.range_len_exponent)
        # # set intrument information
        self.gain = 0.0
        self.antenna_temperature = 200
        self.noise_figure = 3.0
        self.set_nadir_antenna(self.gain, self.antenna_temperature,
                               self.noise_figure, self.filter_bb_bw, False)
        # # set power waveform information
        init_range = self.geometry.geometric_delay
        init_range -= (self.signal.lambda_size//2+1)*self.dt
        self.set_pow_waveform(init_range, self.sampling_rate)

    def simulate(self, rx_pos, rx_vel, tx_pos, tx_vel):
        self.set_model(rx_pos, rx_vel, tx_pos, tx_vel)
        self.compute_sp_info()
        self.compute_power_distribution()
        self.compute_center_waveform()
        self.compute_ddm()
        # self.output_subddm()
        return self.waveform, self.integrate_waveform, self.wave_range


###############################################################################
def plot_incidences_waveform(rx_pos, rx_vel, tx_pos, tx_vel,
                             wind_list=np.arange(5, 40, 5.0)):
    sim = Simulator()
    wave_power = {}
    for i in wind_list:
        sim.wind_speed = i
        wave_power[i] = sim.simulate(rx_pos, rx_vel, tx_pos, tx_vel)
    # # zeros dopplers delay waveform
    plt.figure(figsize=(4.5, 4))
    for i in sorted(list(wave_power.keys())):
        power = wave_power[i][0]
        rang = wave_power[i][2]
        noise_level = power[0]
        plt.plot(rang/1000.0, 10.0*np.log10(power/noise_level), '-',
                 label="{} m/s".format(i))
    plt.grid()
    plt.title("Delay waveform")
    plt.xlabel("Range from specular [km]")
    plt.ylabel("SNR [dB]")
    plt.ylim(0, 0.05)
    plt.xlim(-1.0, 5.0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # # integrated delay doppler
    plt.figure(figsize=(4.5, 4))
    for i in sorted(list(wave_power.keys())):
        power = wave_power[i][1]
        rang = wave_power[i][2]
        noise_level = power[0]
        plt.plot(rang/1000.0, 10.0*np.log10(power/noise_level), '-',
                 label="{} m/s".format(i))
    plt.grid()
    plt.title("Integrated delay waveform")
    plt.xlabel("Range from specular [km]")
    plt.ylabel("SNR [dB]")
    plt.ylim(0, 0.02)
    plt.xlim(-1.0, 5.0)
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.show()


###############################################################################
if __name__ == "__main__":
    # plot_incidences_waveform()

    # rx_pos = [-3147581.506, 3434454.649, -5238650.750]
    # rx_vel = [5181.727, -2703.680, -4885.340]
    # tx_pos = [-14849239.465, 15882877.420, -14561494.119]
    # tx_vel = [398.198, -1872.789, -2476.810]

    rx_pos = [-6451331.0, 2469841.0, -162858.0]
    rx_vel = [-1962.0, -5374.0, -4350.0]
    tx_pos = [-21185116.0, 14082214.0, -6987244.0]
    tx_vel = [518.0, -815.0, -3063.0]

    sim = Simulator()
    sim.simulate(rx_pos, rx_vel, tx_pos, tx_vel)
    sim.plot_power_ddm()
    sim.plot_delay_waveform()
    sim.plot_scattered_area()
    sim.plot_scattered_area(False)
