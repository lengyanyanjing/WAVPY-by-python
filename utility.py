import const
import math
import numpy as np
import numpy.linalg as npl


# -----------------------------------------------------------------------------
def ecef2pos(xyz_pos, a=const.RE_WGS84, f=const.FE_WGS84):
    """ transform ecef to geodetic postion """
    llh = np.zeros(3)
    e2 = f*(2.0-f)  # second eccentricity
    r2 = np.dot(xyz_pos[:2], xyz_pos[:2])
    z = xyz_pos[2]
    zk = 0.0
    while (abs(z - zk) >= 1E-4):
        zk = z
        sinp = z / math.sqrt(r2 + z*z)
        v = a / math.sqrt(1.0-e2*sinp*sinp)
        z = xyz_pos[2] + v * e2 * sinp
    llh[0] = math.atan(z/math.sqrt(r2)) if r2 > 1e-12 else \
        (const.PI / 2.0 if xyz_pos[2] > 0.0 else -const.PI / 2.0)
    llh[1] = math.atan2(xyz_pos[1], xyz_pos[0]) if r2 > 1e-12 else 0.0
    llh[2] = np.sqrt(r2+z*z)-v
    return llh


# ------------------------------------------------------------------------------
def pos2ecef(llh_ref, a=const.RE_WGS84, f=const.FE_WGS84):
    """ transform geodetic to ecef position """
    xyz_pos = np.zeros(3)
    sinp = math.sin(llh_ref[0])
    cosp = math.cos(llh_ref[0])
    sinl = math.sin(llh_ref[1])
    cosl = math.cos(llh_ref[1])
    e2 = f*(2.0-f)      # second eccentricity
    v = a/math.sqrt(1.0-e2*sinp*sinp)

    xyz_pos[0] = (v+llh_ref[2])*cosp*cosl
    xyz_pos[1] = (v+llh_ref[2])*cosp*sinl
    xyz_pos[2] = (v*(1.0-e2)+llh_ref[2])*sinp
    return xyz_pos


# ------------------------------------------------------------------------------
def xyz2enu(llh_ref):
    """ ecef to local coordinate transfromation matrix """
    E = np.zeros((3, 3))
    sinp = math.sin(llh_ref[0])
    cosp = math.cos(llh_ref[0])
    sinl = math.sin(llh_ref[1])
    cosl = math.cos(llh_ref[1])

    E[0][0] = -sinl
    E[0][1] = -sinp * cosl
    E[0][2] = cosp * cosl
    E[1][0] = cosl
    E[1][1] = -sinp * sinl
    E[1][2] = cosp * sinl
    E[2][0] = 0.0
    E[2][1] = cosp
    E[2][2] = sinp
    return E


# ------------------------------------------------------------------------------
def ecef2enu(llh, xyz_pos):
    """ transform ecef vector to local tangental coordinate """
    E = xyz2enu(llh)
    return np.dot(xyz_pos, E)


# ------------------------------------------------------------------------------
def enu2ecef(llh, enu_pos):
    """ transform local vector to ecef coordinate """
    E = xyz2enu(llh)
    return np.dot(enu_pos, npl.inv(E))


# ------------------------------------------------------------------------------
def yaw_rotate(enu_pos, angle):
    R = np.zeros((2, 2))
    R[0, 0] = math.cos(angle)
    R[0, 1] = -math.sin(angle)
    R[1, 0] = math.sin(angle)
    R[1, 1] = math.cos(angle)
    tmp = np.dot(enu_pos[[0, 1]], R)
    return np.hstack((tmp, enu_pos[2]))


# ------------------------------------------------------------------------------
def roll_rotate(enu_pos, angle):
    R = np.zeros((2, 2))
    R[0, 0] = math.cos(angle)
    R[0, 1] = -math.sin(angle)
    R[1, 0] = math.sin(angle)
    R[1, 1] = math.cos(angle)
    tmp = np.dot(enu_pos[1:], R)
    return np.hstack((enu_pos[0], tmp))


# ------------------------------------------------------------------------------
def pitch_rotate(enu_pos, angle):
    R = np.zeros((2, 2))
    R[0, 0] = math.cos(angle)
    R[0, 1] = math.sin(angle)
    R[1, 0] = -math.sin(angle)
    R[1, 1] = math.cos(angle)
    tmp = np.dot(enu_pos[[0, 2]], R)
    return np.hstack((tmp[0], enu_pos[1], tmp[1]))


# ------------------------------------------------------------------------------
def rotate_rpy(rotate_angle, pos):
    """ transform receiver body frame to ecef """
    r_pos = roll_rotate(pos, rotate_angle[0])
    rp_pos = pitch_rotate(r_pos, rotate_angle[1])
    rpy_pos = yaw_rotate(rp_pos, rotate_angle[2])
    return np.asarray([rpy_pos[1], rpy_pos[0], rpy_pos[2]])


# ------------------------------------------------------------------------------
def spf2ecef(sp_pos, spf_pos, az, a=const.RE_WGS84, f=const.FE_WGS84):
    enu_pos = yaw_rotate(spf_pos, -az)
    pos = ecef2pos(sp_pos, a, f)
    ecef_pos = enu2ecef(pos, enu_pos)
    return ecef_pos+sp_pos


# ------------------------------------------------------------------------------
def earth_radius(xyz_pos):
    # # latitude
    lat = math.atan2(xyz_pos[2], npl.norm(xyz_pos[:2], 2))
    # # geopysical latitude
    B = math.atan(math.tan(lat)*pow((1.0-const.FE_WGS84), 2))
    e2 = const.FE_WGS84*(2.0-const.FE_WGS84)  # second eccentricity
    return const.RE_WGS84*(1.0-e2)/(1.0+math.sqrt(e2)*math.cos(B))


if __name__ == "__main__":
    pos = np.array([-14738.498707, - 5798.484533, - 21362.115024])
    print(earth_radius(pos))
