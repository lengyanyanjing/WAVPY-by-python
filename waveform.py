import const
import numpy as np
from scipy import signal


class Waveform:
    def __init__(self, specular_weight_flg=True, incoherent_num=0):
        # # normalize the input power waveform
        self.sampling_rate = 2.0e7  # sampling rate of the waveform in samples/sec
        self.init_range = 0.0       # initial range value of the stored waveform in meters
        self.relative_factor = 0.5  # relative delay located at the leading edge with a power
                                    # equal to the peak of normalized waveform
        self.noise_floor = 0.0      # signal noise floor
        self.min_fft_interp = 0.15  # minimum range resolution of the fft 
                                    # interpolation for delays computation
        self.specular_weight_flg = specular_weight_flg
        self.incoherent_num = incoherent_num
        self.fit_len = 10.0
        self.limit_flg = False
        self.limit_center = 0.0
        self.limit_width = 0.0
        # # waveform information
        self.pos_max = 0.0            # maximum power location
        self.pos_der = 0.0            # maximum 1st derivative location
        self.pos_sample_max = 0.0    # maximum samples corresponding to peak value location
        self.pos_rel = 0.0            # relaive power value location to power peak location
        self.pos_sample_der = 0.0     # maximum samples corresponding to maximum derivative location
        self.pos_sample_rel = 0.0     # maximum samples corresponding to relative power location
        self.pow_max = 0.0            # maximum power
        self.pow_pos_der = 0.0        # maximum 1st derivative
        self.pow_der_pos_der = 0.0
        self.sigma_pos_max = 999999.999999   # sigma of maximum power position
        self.sigma_pos_der = 999999.999999   # sigma of derivative position
        self.sigma_pos_rel = 999999.999999   # sigma of relative postion
        self.sigma_slope_norm_tail = 999999.999999  # sigmal of tailing edge slope
        self.slope_norm_tail = 0.0           # tailing edge slope

        self.prio_scatter_del = 0.0
        self.norm_tial_len = 50.0

    def __compute_tail_slope__(self, dt, ind_pow_max):
        fit_samples = int(self.norm_tial_len/dt)
        if (ind_pow_max+fit_samples) < len(self.ext_waveform):
            X = self.tau[ind_pow_max:ind_pow_max+fit_samples]
            Y = self.ext_waveform[ind_pow_max:ind_pow_max+fit_samples]/self.pow_max
            if self.specular_weight_flg:
                sigma = self.ext_waveform[ind_pow_max:ind_pow_max+fit_samples]
                sigma /= (np.sqrt(self.incoherent_num)*self.pow_max)
                W = 1.0/sigma**2
                linear_coef, cov = np.polyfit(X, Y, 1, w=W, cov=True)
            else:
                linear_coef, cov = np.polyfit(X, Y, 1, cov=True)
            self.slope_norm_tail = linear_coef[0]
            self.sigma_slope_norm_tail = cov[0, 0]
        else:
            self.slope_norm_tail = 0.0
            self.sigma_slope_norm_tail = 999999.999999

    def __compute_rel_max__(self, dt, ind_pow_max):
        fit_samples = int(self.fit_len/dt)
        self.pos_rel = -1.0
        self.sigma_pos_rel = -1.0
        self.pos_sample_der = 1.0
        # # search location equal to tmp power
        tmp = self.relative_factor*self.pow_max
        if tmp > self.noise_floor:
            # # search relative location for fitting
            ind_rel = ind_pow_max
            while (self.ext_waveform[ind_rel] > tmp) and (ind_rel > 0):
                ind_rel -= 1
            ind_min = ind_rel-fit_samples//2
            if ind_min < 0:
                ind_min = 0
            ind = len(self.ext_waveform)-fit_samples-1
            if ind_min > ind:
                ind_min = ind
            X = self.tau[ind_min:ind_min+fit_samples]
            Y = self.ext_waveform[ind_min:ind_min+fit_samples]-tmp
            if self.specular_weight_flg:
                sigma = self.ext_waveform[ind_min:ind_min+fit_samples]
                W = 1.0/sigma**2
                linear_coef, cov = np.polyfit(X, Y, 1, w=W, cov=True)
            else:
                linear_coef, cov = np.polyfit(X, Y, 1, cov=True)
            self.pos_der = -linear_coef[1]/linear_coef[0]
            self.pos_sample_rel = self.tau[ind_rel]
            tmp = cov[1, 1]/linear_coef[0]**2
            tmp += (cov[0, 0]*linear_coef[1]**2/linear_coef[0]**4)
            tmp -= (2.0*cov[0, 1]*linear_coef[1]/linear_coef[0]**3)
            self.sigma_pos_rel = np.sqrt(tmp)
        else:
            self.pos_der = 0.0
            self.pos_sample_rel = 0.0
            self.sigma_pos_rel = 999999.999999

    def __compute_der_max__(self, dt, ind_pow_max):
        if not self.limit_flg:
            tmp = self.noise_floor+0.3*(self.pow_max-self.noise_floor)
            # # find power equal tmp postion of leading edge of signal
            ind_min = 0
            while self.ext_waveform[ind_min] < tmp:
                ind_min += 1
            # # get 1st derivative maximum and location
            self.pow_der_pos_der = 0.0
            max_der1 = max(self.derivative_1st[ind_min:ind_pow_max])
            if self.pow_der_pos_der < max_der1:
                self.pow_der_pos_der = max_der1
                ind_max_der1 = np.argmax(self.derivative_1st[ind_min:ind_pow_max])+ind_min
        fit_samples = int(self.fit_len/dt)
        # # corresponding power of 1st derivative maximum location
        self.pow_pos_der = self.ext_waveform[ind_max_der1]
        # # start and end posiztion around sigal peak to fitting
        ind_min = int(ind_max_der1-fit_samples/2)
        if ind_min < 2:
            ind_min = 2
        tmp = len(self.ext_waveform)-fit_samples-4
        if ind_min > tmp:
            ind_min = tmp
        X = self.tau[ind_min:ind_min+fit_samples]
        Y = self.derivative_2nd[ind_min:ind_min+fit_samples]
        if self.specular_weight_flg:
            # sigma = np.array([abs(self.ext_waveform[ind_min+i+2]-self.ext_waveform[ind_min+i])
            #                   for i in range(fit_samples)])
            # sigma -= np.array([abs(self.ext_waveform[ind_min+i]-self.ext_waveform[ind_min+i-2])
            #                    for i in range(fit_samples)])
            sigma = np.abs(self.ext_waveform[ind_min+2:ind_min+2+fit_samples]
                           - self.ext_waveform[ind_min:ind_min+fit_samples])
            sigma -= np.abs(self.ext_waveform[ind_min:ind_min+fit_samples]
                            - self.ext_waveform[ind_min-2:ind_min-2+fit_samples])
            sigma = np.abs(sigma)/np.sqrt(self.incoherent_num*4.0*dt*dt)
            W = 1.0/sigma**2
            linear_coef, cov = np.polyfit(X, Y, 1, w=W, cov=True)
        else:
            linear_coef, cov = np.polyfit(X, Y, 1, cov=True)
        self.pos_der = -linear_coef[1]/linear_coef[0]
        self.pos_sample_der = self.tau[ind_max_der1]
        tmp = cov[1, 1]/linear_coef[0]**2
        tmp += (cov[0, 0]*linear_coef[1]**2/linear_coef[0]**4)
        tmp -= (2.0*cov[0, 1]*linear_coef[1]/linear_coef[0]**3)
        self.sigma_pos_der = np.sqrt(tmp)

    def __compute_pos_max__(self, dt):
        """ get power peak value and location """
        fit_samples = int(self.fit_len/dt)  # fit samples
        ext_len = len(self.ext_waveform)
        self.pos_max = 0.0
        # # get peak value and locatin of signal
        if self.limit_flg:
            self.pow_der_pos_der = 0.0
            start = int(round((self.limit_center-self.limit_width/2)/dt))
            end = int(round((self.limit_center+self.limit_width/2)/dt))
            if start < 0:
                start = 0
            if end >= ext_len:
                end = ext_len
            self.pow_der_pos_der = max(self.derivative_1st[start:end])
            ind_max_der = np.argmax(self.derivative_1st[start:end])+start
            start = int(round((dt*ind_max_der+self.prio_scatter_del-self.limit_width/4)/dt))
            end = int(round((dt*ind_max_der+self.prio_scatter_del+self.limit_width/4)/dt))
            if start < 0:
                start = 0
            if end >= ext_len:
                end = ext_len
            self.pow_max = max(self.ext_waveform[start:end])
            ind_pow_max = np.argmax(self.ext_waveform[start:end])+start
        else:
            self.pow_max = max(self.ext_waveform)
            ind_pow_max = self.ext_waveform.argmax()
        # # start and end posiztion around sigal peak to fitting
        ind_min = int(ind_pow_max-fit_samples/2)
        if ind_min < 1:
            ind_min = 1
        tmp = ext_len-fit_samples-3
        if ind_min > tmp:
            ind_min = tmp
        X = self.tau[ind_min:ind_min+fit_samples]
        Y = self.derivative_1st[ind_min:ind_min+fit_samples]
        if self.specular_weight_flg:
            sigma = [abs(self.ext_waveform[ind_min+i+1]-self.ext_waveform[ind_min+i-1])
                     for i in range(fit_samples)]
            sigma /= (np.sqrt(self.incoherent_num)*2*dt)
            W = 1.0/sigma/sigma
            linear_coef, cov = np.polyfit(X, Y, 1, w=W, cov=True)
        else:
            linear_coef, cov = np.polyfit(X, Y, 1, cov=True)
        self.pos_max = -linear_coef[1]/linear_coef[0]
        self.pos_sample_max = self.tau[ind_pow_max]
        tmp = cov[1, 1]/linear_coef[0]**2
        tmp += (cov[0, 0]*linear_coef[1]**2/linear_coef[0]**4)
        tmp -= (2.0*cov[0, 1]*linear_coef[1]/linear_coef[0]**3)
        self.sigma_pos_max = np.sqrt(tmp)
        return ind_pow_max

    def __compute_interp_waveform__(self, interp_factor, dt):
        """ compute interpolate waveform and derivative """
        # # the interval of chips interploate the number of interp_factor points
        wave_len = len(self.waveform)
        sizes = int(interp_factor*wave_len)
        self.ext_waveform = signal.resample(self.waveform, sizes)
        # # signal first derivative, second derivative
        self.derivative_1st = np.gradient(self.ext_waveform)
        self.derivative_2nd = np.gradient(-1.0*self.derivative_1st)

    def __compute_interp_factor__(self):
        """ compute interpolate fft """
        interp_factor = 1
        dt = const.C_LIGHT/self.sampling_rate
        dt_ext = dt/interp_factor
        while dt_ext > self.min_fft_interp:
            interp_factor *= 2
            dt_ext = dt/interp_factor
        self.tau = np.arange(0, len(self.waveform)*interp_factor)*dt_ext
        return interp_factor, dt_ext

    def __compute_noise_floor__(self):
        """ compute noise floor and re-initialze the waveform with noise"""
        noise_lags = 6   # chips compute noise floor
        extra_lags = 6   # extrapolate waveform tails chips
        # # compute noise floor of waveform and which impact the first
        # # 6 waveform chips, noise floor effect the head of waveform most
        self.noise_floor = np.mean(self.waveform[:noise_lags])
        l_coef = np.linspace(noise_lags, 1, noise_lags)/noise_lags
        r_coef = np.arange(0, noise_lags)/noise_lags
        self.waveform[:noise_lags] += self.waveform[:noise_lags]*r_coef
        self.waveform[:noise_lags] += self.noise_floor*l_coef
        # # extrapolate waveform
        X = np.linspace(self.init_len-extra_lags, self.init_len-1, extra_lags)
        Y = self.waveform[-extra_lags:]
        linear_coef = np.polyfit(X, Y, 1)
        pred = linear_coef[1]+linear_coef[0]*(len(self.waveform)-1)
        if (linear_coef[0] < 0.0) and (pred <= self.noise_floor):
            ind_chips = np.arange(self.init_len, self.waveform)
            power = linear_coef[1]+linear_coef[0]*ind_chips
            power[power < self.noise_floor] = self.noise_floor
            self.waveform[self.init_len:] = power
        else:
            len_diff = len(self.waveform)-self.init_len
            l_coef = np.linspace(len_diff-1, 0, len_diff)/len_diff
            r_coef = np.arange(1, len_diff+1)/len_diff
            self.waveform[self.init_len:] = self.waveform[self.init_len-1]*l_coef
            self.waveform[self.init_len:] += self.noise_floor*r_coef

    def __init__waveform__(self):
        """ initialize waveform """
        self.init_len = len(self.waveform)
        if len(self.waveform) == 0:
            return
        # # normalize the waveform
        tail_len = 2
        while tail_len <= self.init_len:
            tail_len *= 2
        # # fill zero for signal interpolate in frequency domain
        self.waveform = np.hstack([self.waveform,
                                   np.zeros(tail_len-self.init_len)])

    def update_delays(self):
        """ re-compute waveform information """
        if self.scale_factor != 0.0:
            self.pos_max *= self.scale_factor
            self.pow_der_pos_der *= self.scale_factor
            self.pow_pos_der *= self.scale_factor
            self.noise_floor *= self.scale_factor
        self.pos_max += self.init_range
        self.pos_sample_max += self.init_range
        self.pos_der += self.init_range
        self.pos_sample_der += self.init_range
        self.pos_rel += self.init_range
        self.pos_sample_rel += self.init_range

    def compute_waveform_delays(self):
        """ compute waveform delays """
        # # signal zero fill for fft
        self.__init__waveform__()
        if self.init_len <= 0:
            return
        # # compute noise floor add to original clean signal
        self.__compute_noise_floor__()
        # # compute time/frequency domain interpolate factor
        interp_factor, dt = self.__compute_interp_factor__()
        # # get interpolated waveform
        self.__compute_interp_waveform__(interp_factor, dt)
        # # get power peak value and locatin
        ind_pow_max = self.__compute_pos_max__(dt)
        # # get derivative maximum and location
        self.__compute_der_max__(dt, ind_pow_max)
        # # get specific power location of relative peak value
        self.__compute_rel_max__(dt, ind_pow_max)
        # # compute tailing edge slope
        self.__compute_tail_slope__(dt, ind_pow_max)
        self.update_delays()

    def set_min_fft_interp(self, value=0.15):
        """ set minimum fft interpolate parameters """
        if (0.0 < value) and (value <= 0.15):
            self.min_fft_interp = value

    def get_original_waveform(self):
        """ return original input waveform """
        if self.init_len > 0:
            return self.ext_waveform[:self.init_len]*self.scale_factor

    def get_range_waveform(self):
        if self.init_len > 0:
            dt = const.C_LIGHT/self.sampling_rate
            return self.init_range+np.arange(self.init_len)*dt

    def set_init_range(self, init_range):
        diff = init_range - self.init_range
        self.pos_max += diff
        self.pos_der += diff
        self.pos_rel += diff
        self.pos_sample_max += diff
        self.pos_sample_der += diff
        self.pos_sample_rel += diff
        self.init_range = init_range

    def set_waveform(self, wave):
        self.scale_factor = max(abs(wave))
        if self.scale_factor == 0.0:
            self.scale_factor = 1.0
        self.waveform = wave/self.scale_factor


if __name__ == "__main__":
    print("success!")
    