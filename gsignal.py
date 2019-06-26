import const
import numpy as np


class Signal(object):
    def __init__(self):
        """ constructor """
        self.freq = const.FREQ1  # Sampling rate of the receiver in samples/sec
        self.wavelen = const.C_LIGHT/self.freq
        self.sampling_rate = 8*1.0e7  # Base-band bandwidth of the lter applied in Hz
        self.filter_BB_BW = 12*1.0e6  # Weight for the different code
        self.weight_CA = 1.0          # CA default 1.0
        self.chip_rate = const.GPS_CA_CHIP_RATE      # signal chip rate
        self.transmit_pow = const.POW_CA_Trans_dBW   # transmit power in dBw
        self.chip_rate = const.GPS_CA_CHIP_RATE
        # # number of samples of the autocorrelation function
        self.lambda_size = 2*int(round(self.sampling_rate/self.chip_rate))+1

    def __filter_samples__(self, samples):
        """ compute filter size """
        return 2*int(round(samples*self.chip_rate/(2.0*self.filter_BB_BW), 0)/2)+1

    def __lambda_index__(self, samples):
        """ compure index """
        ind = []
        half_lambda_size = int(self.lambda_size/2)
        factor = samples*self.chip_rate/self.sampling_rate
        for i in range(self.lambda_size):
            ind.append(int(round((i-half_lambda_size)*factor, 0))+samples*2)
        return ind

    def compute_lambda(self):
        """ compute correlation function for the incoming signal """
        if self.weight_CA <= 0.0:
            return
        # # compute the correlation function
        samples_CA = 2**14
        sequence = np.zeros(samples_CA*4)
        sequence[:samples_CA] = 1.0 / np.sqrt(samples_CA)
        # # correlation
        aux = np.correlate(sequence, sequence, mode="same")
        amplitude = np.sqrt(pow(10.0, self.transmit_pow/10.0))*self.weight_CA
        # # number of filter samples has to be an odd integer
        samples_filter = self.__filter_samples__(samples_CA)
        filter_seq = np.ones(samples_filter)/samples_filter
        # # convlution with filter
        conv = np.convolve(aux*amplitude, filter_seq, mode="same")
        conv = conv[self.__lambda_index__(samples_CA)]/conv.max()
        self.lambda_fun = pow(conv, 2)
        return self.lambda_fun

    def set_radar_signal(self, sampling_rate, filter_bb_bw):
        """ set signal parameters """
        self.sampling_rate = sampling_rate
        self.lambda_size = 2*int(round(self.sampling_rate/self.chip_rate))+1
        self.filter_BB_BW = filter_bb_bw

    def compute_transmit_power(self, signal, elevation):
        self.trans_pow = 1225.172
        # if self.freq == const.FREQ1:
        #     gain_coef = np.array([11.805, 0.063546, -0.00090420])
        #     ele = np.array([1.0, elevation, elevation**2])
        #     trans_gain = ele.dot(gain_coef)+const.POW_CA_Trans_dBW
        #     trans_gain = pow(10.0, trans_gain/20.0)
        #     self.trans_pow = pow(signal.weight_CA*trans_gain, 2.0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    signal = Signal()
    lambda_fun = signal.compute_lambda()
    print(lambda_fun.shape)
    plt.figure()
    plt.plot(lambda_fun)
    plt.xlim(1, 160)
    plt.ylim(0, 1)
    plt.xlabel("GNSS code phase (chips)") 
    plt.ylabel("Corrrelation power")
    plt.show()
    