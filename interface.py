import math
import const
import numpy as np
from numpy.linalg import norm


class Interface:
    def __init__(self):
        self.freq = const.FREQ1/1.0e9    # incoming signal frequency in GHz
        self.polar = "RL"  # the polarization of incoming signal
        self.ws = 5.0      # ocean surface wind speed
        self.wd = 0.0      # wind direction from north
        self.mss_x = 0.0   # mean square slope x component
        self.mss_y = 0.0   # mean square slope y component
        self.sigma_z = 0.069  # standard deviation of surface height in meters

    def compute_epsilon_sea_water(self, salinity, temp):
        """ compute  permittivity of sea water """
        E0 = 8.854e-12
        Eswinf = 4.9
        Esw00 = 87.174-0.1949*temp-0.01279*temp**2+0.0002491*temp**3
        a = 1.0+1.613e-5*temp*salinity-3.656e-3*salinity+(
            3.21e-5*salinity**2-4.232e-7*salinity**3)
        Esw0 = Esw00*a
        Tausw0 = 1.1109e-10-3.824e-12*temp+6.238e-14*temp**2-5.096e-16*temp**3
        b = 1.0+2.282e-5*temp*salinity-7.638e-4*salinity-7.760e-6*salinity**2+(
            1.105e-8*salinity**3)
        Tausw = Tausw0 / 2.0 / const.PI * b
        si1 = (0.18252-0.0014619*salinity+2.093e-5*salinity**2)*salinity
        si1 -= 1.282e-7*salinity**4
        inc = 25.0 - temp
        phi = 0.02033 + 1.266e-4*inc + 2.464e-6*inc**2
        phi -= salinity*(1.849e-5 - 2.551e-7*inc + 2.551e-8*inc**2)
        phi *= inc
        si = si1*math.exp(-phi)
        Epimn = 2.0*const.PI*self.freq*Tausw*(Esw0 - Eswinf)
        Epimd = 1.0 + pow((2.0*const.PI*self.freq*Tausw), 2)
        c = si / (2 * const.PI*E0*self.freq)
        real = Eswinf+((Esw0-Eswinf)/(1.0+pow((2.0*const.PI*self.freq*Tausw), 2)))
        imag = c+Epimn/Epimd
        self.epsilon = complex(real, imag)
        return self.epsilon

    def __compute_linear_fresnel__(self, incidence, epsilon_up):
        """ compute linear ploarization parameters """
        tmp1 = np.asarray(epsilon_up)*np.cos(incidence)
        tmp2 = np.asarray(epsilon_up)*self.epsilon
        tmp2 -= (np.asarray(epsilon_up)*(np.cos(incidence))**2)
        tmp3 = self.epsilon*np.cos(incidence)
        self.rhh = (tmp1-np.sqrt(tmp2))/(tmp1+np.sqrt(tmp2))
        self.rvv = (tmp3-np.sqrt(tmp2))/(tmp3+np.sqrt(tmp2))

    def compute_fresnel(self, incidence, epsilon_up=complex(1.0, 0.0),
                        salinity=35, temp=15):
        """ compute the reflection coefficients for interface """
        if not hasattr(self, "epsilon"):
            self.compute_epsilon_sea_water(salinity, temp)
        self.__compute_linear_fresnel__(incidence, epsilon_up)
        if self.polar == "RL":
            self.rcross = 0.5*(self.rvv-self.rhh)
            tmp = abs(self.rcross)
        if self.polar == "RR":
            self.rco = 0.5*(self.rvv+self.rhh)
            tmp = abs(self.rco)
        self.fresnel = tmp*tmp

    def compute_mss_from_wind(self):
        """ calculate the mss from Karzberg model """
        if self.ws < 0.0:
            return
        if self.ws < 3.40:
            fu = self.ws
        elif self.ws < 46.0:
            fu = 6.0*math.log(self.ws)-4.0
        else:
            fu = 0.411*self.ws
        self.mss_x = 0.45*0.00316*fu
        self.mss_y = 0.45*(0.003+0.00192*fu)
        self.sigma_z = 0.00669*math.exp(156.1*(self.mss_x+self.mss_y))

    def GC_nonGaussian_coeff(self, q, az):
        """
        Gram-Charlier C21 and C03 coefficients for non-Guassian
        process in sea surface
        """
        up = -q[0]/q[2]
        cross = -q[1]/q[2]
        phi = const.PI/2+az-self.wd
        cosi = math.cos(phi)
        sini = math.sin(phi)
        up_sigma = (up*cosi+cross*sini)/math.sqrt(self.mss_x)
        cross_sigma = (up*cosi-cross*sini)/math.sqrt(self.mss_y)
        c21 = 0.01-0.0085*self.ws
        c03 = 0.04-0.033*self.ws
        gc_coeff = 1.0+c21/2*(cross_sigma**2-1.0)*up_sigma
        gc_coeff += c03/6.0*(up_sigma**3-3.0*up_sigma)
        return up_sigma, cross_sigma, gc_coeff

    def CM_nonGaussian_coeff(self, q, az):
        """ non-Guassian Cox-Munk coeffcient """
        sigma = q[0]/self.mss_x/q[2]
        eta = q[1]/self.mss_y/q[2]
        c21 = 0.01-0.0085*self.ws
        c03 = 0.04-0.033*self.ws
        c40 = 0.4
        c22 = 0.12
        c04 = 0.23
        cm_coeff = 1-c21/2*(sigma**2-1)*eta
        cm_coeff -= c03/6*(eta**3-3.0*eta)
        cm_coeff += c40/24*(sigma**4-6*sigma**2+3)
        cm_coeff += c22/4*(sigma**2-1)*(eta**2-1)
        cm_coeff += c04/24*(eta**4-6*eta**2+3)
        return sigma, eta, cm_coeff

    def compute_mss_pdf(self, q, az):
        """ compute the mss of ocean surface """
        up_sigma, cross_sigma, gc_coeff = self.GC_nonGaussian_coeff(q, az)
        # up_sigma, cross_sigma, gc_coeff = self.CM_nonGaussian_coeff(q, az)
        # # Gaussian random rough surface
        self.mss_pdf = (1.0/(2.0*const.PI*math.sqrt(self.mss_x*self.mss_y)))
        self.mss_pdf *= math.exp(-(up_sigma**2+cross_sigma**2)/2.0)
        self.mss_pdf *= gc_coeff
        return self.mss_pdf

    def compute_scattered_coeff(self, scat_vec, inc_vec, az):
        """ compute radar scattered coefficent sigma0 """
        q = scat_vec-inc_vec
        rq = norm(q)
        if rq > 0.0:
            theta = scat_vec.dot(q)/rq
        self.compute_fresnel(math.acos(theta))
        self.compute_mss_from_wind()
        self.compute_mss_pdf(q, az)
        self.sigma0 = const.PI*self.fresnel*pow(rq/q[2], 4)*self.mss_pdf
        return self.sigma0

    def set_polarization(self, polar):
        if (polar == "RR") or (polar == "RL"):
            self.polar == polar


if __name__ == "__main__":
    interface = Interface()
    interface.compute_mss_from_wind()
    interface.compute_fresnel(0)

    print(interface.epsilon)
