import const
import numpy as np
import pandas as pd
from math import sin, cos, asin, acos, atan2
from numpy.linalg import norm
from scipy.optimize import fsolve
from scipy.interpolate import interp2d
from utility import (ecef2pos, ecef2enu, yaw_rotate, earth_radius,
                     rotate_rpy, enu2ecef, spf2ecef)


UNDULATION_PATH = r".\res\p85und_qrtdeg_egm96_to360.mean_tide"


class Geometry:
    def __init__(self, path=UNDULATION_PATH):
        self.tx_pos = np.zeros(3)  # ecef coordinates of transmitor
        self.tx_vel = np.zeros(3)  # velocity of transimitor
        self.rx_pos = np.zeros(3)  # ecef coordinates of receiver
        self.rx_vel = np.zeros(3)  # velocity of receiver
        # vertical height offset of the specular point with respect to WGS84 ellipsoid
        self.undulation = 0.0
        self.get_geiod_undulation(path)
        
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
    def get_geiod_undulation(self, path):
        rows = 681
        cols = 1441
        undulation = np.zeros((rows, cols))
        chuncks = pd.read_csv(path, delim_whitespace=True,
                              header=-1, iterator=True)
        for i in range(rows):
            undulation[i] = chuncks.get_chunk(145).values.flatten()[:1441]
            lats = np.linspace(85, -85, rows)
            lons = np.linspace(-180, 180, cols)
            undulation = np.roll(undulation, cols//2, axis=1)
            self.interp = interp2d(lats, lons, undulation.T, kind="cubic",
                                   bounds_error=False, fill_value=0.0)
        
    def __sp_jacobian__(self, x):
        """ compute Jacobian of function equations """
        x = np.asarray(x)
        ellip = np.array([self.a, self.a, self.b])
        ellip2 = ellip**2
        normal_vec = x/ellip2
        tr_vec = self.rx_pos-self.tx_pos
        st_vec = self.tx_pos-x
        sr_vec = self.rx_pos-x
        cross_vec = np.cross(st_vec, sr_vec)/ellip2
        tmp1 = np.zeros(3)
        tmp1[0] = normal_vec[1]*tr_vec[2]-normal_vec[2]*tr_vec[1]
        tmp1[1] = normal_vec[2]*tr_vec[0]-normal_vec[0]*tr_vec[2]
        tmp1[2] = normal_vec[0]*tr_vec[1]-normal_vec[1]*tr_vec[0]
        st_range = norm(st_vec)
        sr_range = norm(sr_vec)
        unit_st_vec = st_vec/st_range
        unit_sr_vec = sr_vec/sr_range
        nst = (self.tx_pos-2.0*x)/ellip2
        nsr = (self.rx_pos-2.0*x)/ellip2
        tmp2 = unit_st_vec*sr_vec.dot(normal_vec)+st_range*nsr
        tmp2 -= (unit_sr_vec*st_vec.dot(normal_vec)+sr_range*nst)
        return np.vstack((2*x/ellip2, cross_vec+tmp1, tmp2))

    def __sp_contrains_func__(self, x):
        """
        compute left hands of multiple nonlinear algebraic equations
        (1) specular point on the ellipsoid
        (2) vector of st,sr,n coplanar
        (3) ncidence angle equal to scattering angle
        """
        x = np.asarray(x)
        ellip = np.array([self.a, self.a, self.b])
        ellip2 = ellip**2
        normal_vec = x/ellip2
        st_vec = self.tx_pos-x
        sr_vec = self.rx_pos-x
        st_range = norm(st_vec)
        sr_range = norm(sr_vec)
        return [sum(x**2/ellip2)-1.0,
                normal_vec.dot(np.cross(st_vec, sr_vec)),
                st_range*sr_vec.dot(normal_vec)-sr_range*st_vec.dot(normal_vec)]

    def __itera_sp__(self):
        """
        iterate computing specular point postion with the methered of iteration
        multiple nonlinear algebraic equations, like newton iterate
        """
        radius = earth_radius(self.rx_pos)
        # # receiver ecef corrdinates on geoid (ellipsoid+ondul)
        tmp = (1.0 + self.undulation / radius)
        self.a = const.RE_WGS84*tmp
        self.b = const.RE_WGS84*(1.0 - const.FE_WGS84)*tmp
        rt = norm(self.tx_pos)
        rr = norm(self.rx_pos)
        unit_vec_t = self.tx_pos/rt  # unit vector of transmiter
        unit_vec_r = self.rx_pos/rr  # unit vector of receiver
        # # calculate approximate postion of specular
        ratio = rt/(rt+rr)
        sp_approx = self.a*(ratio*unit_vec_r+(1-ratio)*unit_vec_t)
        # # newton iterate
        # sol = fsolve(self.__sp_contrains_func__, sp_approx,
        #              fprime=self.__sp_jacobian__, full_output=1,
        #              xtol=1.e-11)
        sol = fsolve(self.__sp_contrains_func__, sp_approx, col_deriv=True,
                     full_output=1, xtol=1.e-11, epsfcn=1.e-25)
        if sol[2] == 1:
            self.sp_pos = sol[0]

    def __enu2azel__(self, pos_enu):
        """
        compute azimuth and incidence angle
        para:
            pos_enu: local tangental coordinate
        """
        enu = pos_enu/norm(pos_enu)
        tmp = np.dot(enu[:2], enu[:2]) < 1.0e-12
        az = 0.0 if tmp else atan2(enu[0], enu[1])
        if az < 0.0:
            az += 2*const.PI
        inc = const.PI/2.0-asin(enu[2])
        return az, inc

    def set_inertial_rotation(self, roll, pitch, yaw):
        """
        roll: rotate around x aixs, pitch:  rotate around y aixs
        yaw: rotate around z aixs
        """
        self.roll = roll*const.D2R
        self.pitch = pitch*const.D2R
        self.yaw = yaw*const.D2R

    def __BFvec2spf__(self, vec_BF):
        # # receiver body frame to ecef
        BF_rotate = rotate_rpy([self.roll, self.pitch, self.yaw], vec_BF)
        pos = ecef2pos(self.rx_pos)
        BF_pos = enu2ecef(pos, BF_rotate)
        pos = ecef2pos(self.sp_pos, self.a, (self.a-self.b)/self.a)
        BF_enu = ecef2enu(pos, BF_pos)
        BF_spf = yaw_rotate(BF_enu, self.tx_az)
        ele = const.PI/2-self.tx_inc
        inertial_delay = -BF_spf[1]*cos(ele) + BF_spf[2]*sin(ele)
        return BF_spf, inertial_delay

    def BF2spfs(self, bf_e, bf_h, bf_k):
        self.bf_e_spf, _ = self.__BFvec2spf__(bf_e)
        self.bf_h_spf, _ = self.__BFvec2spf__(bf_h)
        self.bf_k_spf, _ = self.__BFvec2spf__(bf_k)

    def compute_antenna_gain_pos(self,  scattered_vec):
        scatter_vec = -scattered_vec  # minus unit scattered vector
        # # compute intersection of signal propagation
        # # direction scattered direction
        phi = acos(self.bf_k_spf.dot(scatter_vec))
        vec_E = self.bf_e_spf.dot(scatter_vec)
        vec_H = self.bf_h_spf.dot(scatter_vec)
        theta = atan2(vec_H, vec_E)
        return theta, phi

    def doppler_shift(self, inc_vec, scat_vec, freq):
        tmp = freq/const.C_LIGHT
        vel = self.tv_spf.dot(inc_vec)-self.rv_spf.dot(scat_vec)
        return vel*tmp

    def compute_spf_pos(self):
        """ transform local tangental coordinate to specular frame """
        self.rx_spf = yaw_rotate(self.rx_enu, self.tx_az)
        self.tx_spf = yaw_rotate(self.tx_enu, self.tx_az)
        self.rv_spf = yaw_rotate(self.rv_enu, self.tx_az)
        self.tv_spf = yaw_rotate(self.tv_enu, self.tx_az)

    def compute_enu_pos(self):
        """
        compute radar system transmitor and receiver position
        at local tangental coordinate azimuth and elevation,
        0.0<=azimuth<2*pi,  -pi/2<=elevation<=pi/2
        """
        if hasattr(self, "sp_pos"):
            pos = ecef2pos(self.sp_pos)
            self.rx_enu = ecef2enu(pos, self.rx_pos-self.sp_pos)
            self.tx_enu = ecef2enu(pos, self.tx_pos-self.sp_pos)
            self.rv_enu = ecef2enu(pos, self.rx_vel)
            self.tv_enu = ecef2enu(pos, self.tx_vel)
            self.rx_az, self.rx_inc = self.__enu2azel__(self.rx_enu)
            self.tx_az, self.tx_inc = self.__enu2azel__(self.tx_enu)

    def compute_geometric_delay(self):
        """ compute geometric delay """
        if hasattr(self, "sp_pos"):
            rst = norm(self.tx_pos-self.sp_pos)
            rsr = norm(self.rx_pos-self.sp_pos)
            rrt = norm(self.tx_pos-self.rx_pos)
            self.rrt = rrt
            self.geometric_delay = rst+rsr-rrt

    def compute_sp_pos(self, undulation_flg):
        """ compute radar geometric configuration """
        if (norm(self.tx_pos) <= 0.0) or (norm(self.rx_pos) <= 0.0):
            return
        self.__itera_sp__()
        if undulation_flg:
            pos = np.rad2deg(ecef2pos(self.sp_pos))
            tmp = self.interp(pos[0], pos[1])
            self.undulation = tmp[0]
            self.__itera_sp__()
            self.undulation = 0.0
        self.compute_geometric_delay()
        self.compute_enu_pos()
        self.compute_spf_pos()

    def compute_scattering_vector(self, x):
        """ compute scattering vector surface reflection point """
        xr = self.rx_spf-x
        xt = self.tx_spf-x
        rr = norm(xr)
        rt = norm(xt)
        sca_vec = xr/rr
        inc_vec = -xt/rt
        eta_dx = x[0]*(1.0/rr+1.0/rt)
        eta_dy = (x[1]-self.rx_spf[1])/rr
        eta_dy += (x[1]-self.tx_spf[1])/rt
        tmp = x[0]**2+x[1]**2
        theta_dx = x[1]/tmp
        theta_dy = -x[0]/tmp
        jacob = abs(1.0/(eta_dx*theta_dy-eta_dy*theta_dx))
        return inc_vec, sca_vec, jacob, 1/(rr*rr*rt*rt)

    def compute_r2t_vector(self):
        """
        compute the vector of receiver to transmiter, and the range of them
        """
        r2t = self.rx_spf-self.tx_spf
        rang = norm(r2t)
        return r2t/rang, rang

    def plot_ios_lines(self, xlim, ylim):
        """ plot gliston zone ios-delay and ios-doppler line  """
        import matplotlib.pyplot as plt
        sampling_rate = 4091750.0       # delay frequency resolution
        dt = const.C_LIGHT/sampling_rate
        x = np.linspace(-xlim, xlim, int(xlim/100*2+1))
        y = np.linspace(-xlim, xlim, int(xlim/100*2+1))
        X, Y = np.meshgrid(x, y)
        delay = np.zeros(X.shape)
        dopp = np.zeros(X.shape)
        _, rang = self.compute_r2t_vector()  # direct signal range
        inc_vec = -1.0*self.tx_spf/norm(self.tx_spf)
        sca_vec = self.rx_spf/norm(self.rx_spf)
        # # specular point delay and doppler shift
        sp_dopp = self.doppler_shift(inc_vec, sca_vec, const.FREQ1)
        sp_delay = self.geometric_delay
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                pos = np.array([xi, yi, 0.0])
                # # specular point
                if norm(pos) == 0:
                    delay[i, j] = 0.0
                    dopp[i, j] = 0.0
                    continue
                rr = norm(self.rx_spf-pos)
                rt = norm(self.tx_spf-pos)
                # # compute delay in unit chip
                delay[i, j] = (rr+rt-rang-sp_delay)/dt
                vec = self.compute_scattering_vector(pos)
                # # compute doppler shift in unit Hz
                dopp[i, j] = self.doppler_shift(vec[0], vec[1], const.FREQ1)-sp_dopp
        # # contour label
        level1 = np.arange(0, xlim/dt, 0.25)
        level2 = np.arange(-4, 4.5, 0.5)
        plt.figure(figsize=(4.2, 4))
        ax = plt.subplot(111)
        cs1 = ax.contour(X, Y, delay/4.0, level1)
        cs2 = ax.contour(X, Y, dopp/1000.0, level2)
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.clabel(cs1, fontsize=10, fmt='%.2f')
        ax.clabel(cs2, fontsize=10, fmt='%.1f')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        # ax.axis('equal')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    geometry = Geometry()
    # geometry.tx_pos = np.asarray([-14849239.465, 15882877.420, -14561494.119])
    # geometry.rx_pos = np.asarray([-3147581.506, 3434454.649, -5238650.750])
    # geometry.rx_pos = np.asarray([-4386241.0, 4887114.0, 2122148.0])
    # geometry.tx_pos = np.asarray([-6515486.0, 19664460.0, -16749344.0])
    # geometry.compute_sp_pos(True)
    # print(geometry.sp_pos)
    # -2953617.1903865258, 3214122.1975928270, -4634805.4556144867
    Rt = np.array([14880, -17126, 13763])*1000.
    Vt = np.array([-0.258, 1.747, 2.456])*1000.
    Rr = np.array([3473, -5796, 1284])*1000.
    Vr = np.array([3.786, 4.346, 4.326])*1000.
    geometry.tx_pos = Rt
    geometry.rx_pos = Rr
    geometry.tx_vel = Vt
    geometry.rx_vel = Vr
    geometry.compute_sp_pos(True)
    geometry.plot_ios_lines(xlim=3.0e4, ylim=3.0e4)
    print(np.rad2deg(geometry.tx_inc))
