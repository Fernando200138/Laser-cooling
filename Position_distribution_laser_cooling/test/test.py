import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# Properties for BaF (or SrF)
# mass = (88 + 19) *amu
c = 299792458  # m/s, speed of light
kB = 1.380649e-23  # J/K. Boltzmann constant
amu = 1.66053904e-27  # kg, molar mass conversion constant
h = 6.62607015e-34  # J s,  Planck constant
hbar = h / (2 * np.pi)
g = 9.81  # m/s**2, constant of acceleration due to Earth's gravity
mass = (138 + 19) * amu
B = 0.21594802  # rotational constant of BaF in cm-1 (Master Thesis Jeroen Maat)
mu = 3.170 * 0.0168  # dipole moment in cm-1/(kv/cm)  (debye * conversionfactor)

Starkfit = [[0 for i in range(7)] for j in range(5)]
Starkfit[0] = [1.99832523, 2.78038613, 1.0962849, 11.2629089, 1.6618291, 0., 0., 0.,
               0.]  # parameters for N=1,M=0 state of BaF. Which paramters?
Starkfit[1] = [1.99297730e+00, -2.30557413e-06, 6.22563075e-04, 1.66160290e+00, 2.42295526e-01, 2.69430173e+00,
               4.12591610e-01]  # parametersfor N=1,M=1 state of BaF

delta = 1.e-6  # used for taking derivative of field. > 1.e-4 the results start to deviate.
dt = 1.e-4  # timestep in the guiding() function; depends on the acceleration.
# 1.e-4 is sufficient for an error less than 10 micron after 5 meters of flight.

J2wavenr = 1 / (100 * h * c)  # factor to convert energy in cm-1 to SI

r0hex = 6.e-3  # inner radius hexapole
phi0hex = 4.501e3  # 4.5e3          # voltage applied to hexapole rods (plus and minus -> the voltage difference between adjacent rods is 2*phi0hex)

wavevector = 2 * np.pi / 860e-9
gamma = 1 / 60e-9

x_laser = 10  # 10.e-3      # size laserbeam
'''
size laserbeam? like what, the width?
'''
v_Doppler = 0.1  # minimum velocity corresponding to the Doppler temperature in m/s

s0 = 4.  # 12.  # I/I_s
detuning = 3.2 * gamma  # detuning in Hz

xx0 = [0.66e-3, 0, 0.e-3, 0., 0., 0., 184.]
dxx = [0.33e-3, 4.5e-3, 4.5e-3, 0.e-3, 20., 20., 27.]

nn = 1000  # number of molecules !100000
nnsim = 10000  # number of molecules in simulation !100000
ni = 100  # number of voltage steps !200
pi = np.pi

def ascat(vx,fudgefactor,s0,detuning):
    ascat = fudgefactor*((gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma + wavevector*vx)/gamma)**2))*hbar*wavevector  \
                        -(gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma - wavevector*vx)/gamma)**2))*hbar*wavevector)/mass
    return ascat

