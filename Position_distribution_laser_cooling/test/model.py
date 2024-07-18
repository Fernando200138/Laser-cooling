#In this file I'm going to use import all the functions I'm going to use
import statistics

#Import standard libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.stats as st
from scipy.optimize import curve_fit
import pandas as pd
import random
from scipy.optimize import leastsq
from matplotlib import cm as CM
from scipy.stats import norm

###############################
# Global constants
c = 299792458  # m/s, speed of light
kB = 1.380649e-23  # J/K. Boltzmann constant
amu = 1.66053904e-27  # kg, molar mass conversion constant
h = 6.62607015e-34  # J s,  Planck constant
hbar = h / (2 * np.pi)
g = 9.81  # m/s**2, constant of acceleration due to Earth's gravity

# properties of setup
Labs = 0.005
Lfront = 0.197
Lhex = 0.390
Lback = 0.127 - 0.04
Llc = 0.15
Ldetect = 0.60
Lsp = 3.50 - Ldetect - Llc - Lback - Lhex - Lfront

L = [Lfront - Labs, Lhex, Lback, Llc, Ldetect, Lsp]

# Properties for BaF (or SrF)
# mass = (88 + 19) *amu
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

#############################
def ascat(vx,fudgefactor,s0,detuning):
    ascat = fudgefactor*((gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma + wavevector*vx)/gamma)**2))*hbar*wavevector  \
                        -(gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma - wavevector*vx)/gamma)**2))*hbar*wavevector)/mass
    return ascat
def fitfunction(vx_list,fudgefactor,s0,detuning):
    fit = np.array([0.0 for j in range(len(vx_list))])
    for j in range(len(vx_list)):
        fit[j] = fudgefactor*((gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma + wavevector*vx_list[j])/gamma)**2))*hbar*wavevector  \
                             -(gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma - wavevector*vx_list[j])/gamma)**2))*hbar*wavevector)/mass
    return fit
def lasercooling(finish,fudgefactor,s0,detuning,xxs,hit): #xxs represents an array of t,x,y,z,vx,vy,vz
    xxp=xxs
    ti,xi,yi,zi,vxi,vyi,vzi=xxs
    tf,xf,yf,zf,vxf,vyf,vzf=xxp
    nsteps=0
    while((vzi<=vzf<finish) and hit==True):
        if ((abs(xf)<=0.5*x_laser) and (abs(vxf)>0.5*v_Doppler)):
            vxf+=dt*ascat(vxf,fudgefactor,s0,detuning)
        if ((abs(yf)<=0.5*x_laser) and (abs(vyf)>0.5*v_Doppler)):
            vyf +=dt*ascat(vyf,fudgefactor,s0,detuning)
        vzf+=0
        xf+=dt*vxf
        yf+=dt*vyf
        vzf+=dt*vzf
        tf+=dt
        nsteps+=1
    return [np.array([tf,xf,yf,zf,vxf,vyf,vzf]),hit]


def WStark(E, s):
    muoverB = mu / B
    WStark = B * (Starkfit[s][0] + (np.sqrt(Starkfit[s][1] ** 2 + (Starkfit[s][2] * muoverB * E) ** 2) - Starkfit[s][1]) \
                  - (np.sqrt(Starkfit[s][3] ** 2 + (Starkfit[s][4] * muoverB * E) ** 2) - Starkfit[s][3]) \
                  - (np.sqrt(Starkfit[s][5] ** 2 + (Starkfit[s][6] * muoverB * E) ** 2) - Starkfit[s][5]) \
                  )
    return WStark


def Phihex(x, y, phi0hex, r0hex):
    r = (x ** 2 + y ** 2) ** (1 / 2)
    if r > 0:
        if y > 0.:
            theta = np.arccos(x / r) - (10 / 180) * np.pi
        else:
            theta = -np.arccos(x / r) - (10 / 180) * np.pi

    phihex = (r / r0hex) ** 3 * (phi0hex * np.cos(3 * theta))  # with a3=3
    '''They are using equation 26'''

    return phihex


def Ehex(x, y, phi0hex, r0hex):
    Ehex = 1.e-5 * np.sqrt(
        ((Phihex(x + delta, y, phi0hex, r0hex) - Phihex(x - delta, y, phi0hex, r0hex)) / (2. * delta)) ** 2 \
        + ((Phihex(x, y + delta, phi0hex, r0hex) - Phihex(x, y - delta, phi0hex, r0hex)) / (2. * delta)) ** 2)

    # Gives electric field in cm-1/(kV/cm) (hence the factor 1.e-5)
    return Ehex


def Whex(x, y, phi0hex, r0hex, s):
    E = Ehex(x, y, phi0hex, r0hex)
    Whex = WStark(E, s)
    return Whex


def axhex(x, y, phi0hex, r0hex, s):
    axhex = -(((Whex(x + delta, y, phi0hex, r0hex, s) - Whex(x - delta, y, phi0hex, r0hex, s)) / J2wavenr) / (
                2 * delta * mass))
    return axhex


def ayhex(x, y, phi0hex, r0hex, s):
    ayhex = -(((Whex(x, y + delta, phi0hex, r0hex, s) - Whex(x, y - delta, phi0hex, r0hex, s)) / J2wavenr) / (
                2 * delta * mass))
    return ayhex
def residual(vars, vx_list,ascat_sim, data, eps_data):
   afit = vars
   chi  = 0.
   fit  = fitfunction(vx_list,afit[0],afit[1],afit[2])
   for j in range(len(vx_list)):
       chi +=    ((ascat_sim - fit[j])/1.)**2/float(len(vx_list))
#   print(np.sqrt(chi))
#   print(vars)
   return ((ascat_sim-fit) / 1.)


def phasespaceellipse2D(xx0, dxx):
    #
    itry = 0

    hit = False
    while hit == False and itry < 100:
        xx = np.zeros(7)
        itry += 1
        #        print("itry:",itry)
        if itry > 99:
            print("itry exceeded 100!")
        xr = np.sqrt(np.random.uniform())
        theta = np.random.uniform() * 2 * np.pi
        '''
        This are the coordinates of the random walk that are been used to simulate the trajectory of the particle.
        Why do we need a random walk to simulate it? Since particles are emitting photons in random directions
        '''
        xx[1] = xx0[1] + 0.5 * dxx[1] * xr * np.sin(theta)  # Only x-coordinate. This is the position of the particle
        xx[4] = xx0[4] + 0.5 * dxx[4] * xr * np.cos(theta)  # This is the velocity of the particle.

        xr = np.sqrt(np.random.uniform())
        theta = np.random.uniform() * 2 * np.pi
        '''
        This lines may seem redudant but they're not. They are used to defined the coordinates for the y-axis. If they weren't redefined then x==y would always happpend.
        '''
        xx[2] = xx0[2] + 0.5 * dxx[2] * xr * np.sin(theta)  #
        xx[5] = xx0[5] + 0.5 * dxx[5] * xr * np.cos(theta)

        if ((xx[1] - xx0[1]) ** 2 + (xx[2] - xx0[2]) ** 2 < (0.5 * dxx[1]) ** 2):  # wouldn't we need dxx[2] here two?
            hit = True
    xx[0] = np.random.normal(xx0[0], dxx[0])
    xx[3] = xx0[3]
    xx[6] = np.random.normal(xx0[6], dxx[6])
    if (np.random.uniform() > 0.):  # in N=1; 3/5 in m=1 and 2/5 in m=0; (if > 0.4 then s=0)
        s = 0
    else:
        s = 1
    #        xr = np.sqrt(np.random.uniform())
    #        theta = np.random.uniform()*2*np.pi
    #
    #        xx[3] = xx0[3] + 0.5*dxx[3]*xr*np.sin(theta)  # Only z-coordinate
    #        xx[6] = xx0[6] + 0.5*dxx[6]*xr*np.cos(theta)
    return [xx, s, hit]

def freeflight(endpoint,xxs,hit):
    nteps=0
    xxp=xxs
    ti,xi,yi,zi,vxi,vyi,vzi=xxs
    tf,xf,yf,zf,vxf,vyf,vzf=xxp
    while (zi<=zf<endpoint and hit ==True):
        xf+=vxf*(endpoint-zi)/vzf
        yf+=vyf*(endpoint-zi)/vzf
        zf=endpoint
        #This function will run for only one iteration. There might be a better way to do it
        tf+=(endpoint-zi)/vzf
        nteps+=1
    return [np.array([tf,xf,yf,zf,vxf,vyf,vzf]),hit]



def hexapole(endpoint,phi0hex,r0hex,xxs,s,hit): #xxs is a 1-D arrray with 7 elements.
    nteps=0
    xxp=xxs
    ti,xi,yi,zi,vxi,vyi,vzi=xxs
    tf,xf,yf,zf,vxf,vyf,vzf=xxp
    while((zi<=zf<endpoint) and hit==True):
        if -phi0hex<Phihex(xf,yf,phi0hex,r0hex)<phi0hex:
            vxf+=dt*axhex(xf,yf,phi0hex,r0hex,s)
            vyf+=dt*ayhex(xf,yf,phi0hex,r0hex,s)
            vzf+=0
            xf+=dt*vxf
            yf+=dt*vyf
            zf+=dt*vzf
            tf+=dt
            nteps+=1
        else:
            hit=False
            break
    return [np.array([tf,xf,yf,zf,vxf,vyf,vzf]),hit]


def trajectory_simulation(nn, nj,fudgefactor=1,s0=s0,detuning=detuning):
    z_position = []
    x_positiion = []
    y_position = []
    vx = []
    vy = []
    vz = []
    ax = []
    ay = []
    xx = np.zeros(7)

    # nn number of molecules
    # nj number of points in each stage of the trajectory (counting only hexapole and laser cooling)
    time = datetime.datetime.now().time()
    print("started at:", time)
    for i in range(0, nn):
        tempz = []
        tempx = []
        tempy = []
        tempvx = []
        tempvy = []
        tempax = []
        tempay = []
        [xx, s, hit] = phasespaceellipse2D(xx0, dxx)  # How are xx0 and dxx really defined?
        tempz.append(xx[3])
        tempx.append(xx[1])
        tempy.append(xx[2])
        tempvx.append(xx[4])
        tempvy.append(xx[5])
        tempax.append(0.)
        tempay.append(0.)
        [xx, hit] = freeflight(L[0], xx, hit)
        if hit:
            tempz.append(xx[3])
            tempx.append(xx[1])
            tempy.append(xx[2])
            tempvx.append(xx[4])
            tempvy.append(xx[5])
            tempax.append(0.)
            tempay.append(0.)
        if L[1] > 0:
            for j in range(nj):
                [xx, hit] = hexapole(L[0] + L[1] * float(j) / float(nj - 1), phi0hex, r0hex, xx, s, hit)
                if hit:
                    tempz.append(xx[3])
                    tempx.append(xx[1])
                    tempy.append(xx[2])
                    tempvx.append(xx[4])
                    tempvy.append(xx[5])
                    tempax.append(axhex(xx[4], xx[5], phi0hex, r0hex, s))
                    tempay.append((ayhex(xx[4], xx[5], phi0hex, r0hex, s)))
        tempz.append(xx[3])
        tempx.append(xx[1])
        tempy.append(xx[2])
        tempvx.append(xx[4])
        tempvy.append(xx[5])
        tempax.append(0.)
        tempay.append(0.)
        [xx, hit] = freeflight(L[0] + L[1] + L[2], xx, hit)
        if hit:
            tempz.append(xx[3])
            tempx.append(xx[1])
            tempy.append(xx[2])
            tempvx.append(xx[4])
            tempvy.append(xx[5])
            tempax.append(0.)
            tempay.append(0.)
        if L[1] > 0:
            for j in range(nj):
                #[xx, hit] = lasercooling(L[0] + L[1] + L[3] * float(j) / float(nj - 1), fudgefactor, s0, detuning, xx,hit)
                [xx,hit]=freeflight(L[0] + L[1] + L[3] * float(j) / float(nj - 1),xx,hit)
                if hit:
                    tempz.append(xx[3])
                    tempx.append(xx[1])
                    tempy.append(xx[2])
                    tempvx.append(xx[4])
                    tempvy.append(xx[5])
                    #tempax.append(ascat(xx[4], fudgefactor, s0, detuning))
                    #tempay.append(ascat(xx[5], fudgefactor, s0, detuning))
                    tempax.append(0)
                    tempay.append(0)

        tempz.append(xx[3])
        tempx.append(xx[1])
        tempy.append(xx[2])
        tempvx.append(xx[4])
        tempvy.append(xx[5])
        tempax.append(0.)
        tempay.append(0.)
        [xx, hit] = freeflight(L[0] + L[1] + L[2] + L[3] + L[4], xx, hit)
        if hit:
            tempz.append(xx[3])
            tempx.append(xx[1])
            tempy.append(xx[2])
            tempvx.append(xx[4])
            tempvy.append(xx[5])
            tempax.append(0.)
            tempay.append(0.)
        [xx, hit] = freeflight(L[0] + L[1] + L[2] + L[3] + L[4] + L[5], xx, hit)
        if hit:
            tempz.append(xx[3])
            tempx.append(xx[1])
            tempy.append(xx[2])
            tempvx.append(xx[4])
            tempvy.append(xx[5])
            tempax.append(0.)
            tempay.append(0.)
        z_position.append(tempz)
        x_positiion.append(tempx)
        y_position.append(tempy)
        vx.append(tempvx)
        vy.append(tempvy)
        ax.append(tempax)
        ay.append(tempay)
    time = datetime.datetime.now().time()
    print("finished at:", time)

    return [x_positiion, y_position, z_position, vx, vy, ax, ay]















