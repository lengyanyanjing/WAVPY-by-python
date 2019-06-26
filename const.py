import scipy.constants as const


C_LIGHT = const.c
PI = const.pi

K_BOLTZMANN = 1.3806488e-23

D2R = (PI/180.0)      # deg to rad
R2D = (180.0/PI)      # rad to deg
RE_WGS84 = 6378137.0  # earth semimajor axis (WGS84) (m)
FE_WGS84 = 1.0/298.257223563  # earth flattening (WGS84)

GPS_CA_CHIP_RATE = 1023000.0

FREQ1 = 1575420000
FREQ2_BEIDOU = 1561098000

SYS_GPS = 0x01

POW_CA_Trans_dBW = 14.3
POW_L1C_Trans_dBW = 15.8
POW_PY_Trans_dBW = 13.05
POW_M_Trans_dBW = 15.27
POW_IM_Trans_dBW = 14.3

POW_B1I_RecEarth_dBW = -163.0
POW_CA_RecEarth_dBW = -158.5
POW_L1C_RecEarth_dBW = -157.0
POW_E1A_RecEarth_dBW = -157.0
POW_E1B_RecEarth_dBW = -160.0
POW_E1C_RecEarth_dBW = -160.0


if __name__ == "__main__":
    print("{:.40f}".format(PI))
    print("success")