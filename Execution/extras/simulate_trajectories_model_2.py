# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.stats as st
from scipy.optimize import curve_fit
# Global constants
c    = 299792458         # m/s, speed of light
kB   = 1.380649e-23      # J/K. Boltzmann constant
amu  = 1.66053904e-27    # kg, molar mass conversion constant
h    = 6.62607015e-34    # J s,  Planck constant
g    = 9.81              # m/s**2, constant of acceleration due to Earth's gravity
hbar = h/(2*np.pi)
# properties of setup
Labs     =  0.005
Lfront   =  0.197
Lhex     =  0.390
Lback    =  0.127 - 0.04
Llc      =  0.15
Ldetect  =  0.60
Lsp      =  3.50 - Ldetect - Llc - Lback - Lhex - Lfront
xx0      = [0.66e-3,   0,    0.e-3,  0.,   0.,  0., 184.]
dxx      = [0.33e-3,  4.5e-3, 4.5e-3, 0.e-3, 20., 20., 27.]
L        = [Lfront-Labs, Lhex, Lback, Llc, Ldetect, Lsp]
r0hex    = 6.e-3
x_acc    = 2.e-3         # one dimensional acceptance is a circle with radii x_acc and vx_acc 2.e-3
vx_acc   = 2.35     #4.7           # acceptance = pi*x_acc*v_acc
#vx_acc   = 4.7*np.sqrt(107/157)
#vx_acc = 2.
omegadec = vx_acc/x_acc
pi =np.pi
# Properties for BaF (or SrF)
#mass = (88 + 19) *amu
mass = (138 + 19) *amu
B    = 0.21594802         # rotational constant of BaF in cm-1 (Master Thesis Jeroen Maat)
mu   = 3.170*0.0168       # dipole moment in cm-1/(kv/cm)  (debye * conversionfactor)
def Gauss(x, a, sigma):
    return a*np.exp(-(x)**2/(2*sigma**2))
xbeam    =  10.    # size laserbeam
X=[-0.5*xbeam,0,0.5*xbeam]
Y=[1/np.e**2,1,1/np.e**2]
par,cov=curve_fit(Gauss, X, Y)
X2=[-0.5*4,0,0.5*4]
Y2=[1/np.e**2,1,1/np.e**2]
par2,cov2=curve_fit(Gauss, X2, Y2)
Starkfit = [[[0 for i in range(10)] for j in range(3)] for k in range(3)]
Starkfit[1][0] = [2.013269511042715, -0.43415903768574665, 16.558716659221304, -48.099818499336806, 62.91106815725683, -49.33438615121195, 24.350644986688938, -7.420789800737273, 1.2774041721099232, -0.09509274108833095]
Starkfit[1][1] = [2.0084181028908383, 0.015314368146711785, -5.132508349094792, 0.3071790830246961, 4.265880399067137, -5.389647723159123, 3.4486758017655452, -1.268392427196492, 0.2543771744286416, -0.021587823445842334]
Starkfit[2][0] = [6.012488561733084, 0.1145038279727921, 0.5508587369389193, 7.587327046808108, -9.376499846282153, 1.1961272670732497, 3.843154508305001, -2.7173886101310623, 0.7475408486279619, -0.07676056282921148]
Starkfit[2][1] = [6.015800312103941, -0.07153364471446283, 1.5094022648057377, -0.5214080788036027, -4.359948930972086, 6.085292001519875, -3.959907428237688, 1.4402946234915248, -0.28288088678567674, 0.023454418875353256]
Starkfit[2][2] = [6.00990920680478, 0.014733311476906936, -2.450181491564421, 0.1549836432602054, 0.25599195269143415, 0.029661068573406606, -0.16602273558649208, 0.09683477704576915, -0.024392745487986325, 0.002375962624931891]

delta    = 1.e-6          # used for taking derivative of field. > 1.e-4 the results start to deviate.
dt       = 1.e-4          # timestep in the guiding() function; depends on the acceleration.
                          # 1.e-4 is sufficient for an error less than 10 micron after 5 meters of flight.

J2wavenr = 1/(100*h*c)    # factor to convert energy in cm-1 to SI
wavevector =  2*np.pi/860e-9
gamma      =  1/57e-9 #decay rate from A2pi lifetime of 57.1 ns
x_laser    =  10.e-3      # size laserbeam
v_Doppler  =  0.1
def WStark(E, N, MN):
    muoverB = mu / B
    WStark = B * (Starkfit[N][MN][0] + Starkfit[N][MN][1] * (muoverB * E / 10) ** 1 + Starkfit[N][MN][2] * (
                muoverB * E / 10) ** 2 + Starkfit[N][MN][3] * (muoverB * E / 10) ** 3 \
                  + Starkfit[N][MN][4] * (muoverB * E / 10) ** 4 + Starkfit[N][MN][5] * (muoverB * E / 10) ** 5 +
                  Starkfit[N][MN][6] * (muoverB * E / 10) ** 6 \
                  + Starkfit[N][MN][7] * (muoverB * E / 10) ** 7 + Starkfit[N][MN][8] * (muoverB * E / 10) ** 8 +
                  Starkfit[N][MN][9] * (muoverB * E / 10) ** 9 \
                  )
    return WStark


def Phihex(x, y, phi0hex, r0hex, angle=10):
    r = (x ** 2 + y ** 2) ** (1 / 2)
    if r > 0:
        if y > 0.:
            theta = np.arccos(x / r) - (angle / 180) * np.pi
        else:
            theta = -np.arccos(x / r) - (angle / 180) * np.pi
    else:
        theta = 0
    phihex = (r / r0hex) ** 3 * (np.abs(phi0hex) * np.cos(3 * theta))  # with a3=3

    return phihex


def Ehex(x, y, phi0hex, r0hex, angle=10):
    Ehex = 1.e-5 * np.sqrt(((Phihex(x + delta, y, phi0hex, r0hex, angle) - Phihex(x - delta, y, phi0hex, r0hex,
                                                                                  angle)) / (2. * delta)) ** 2 \
                           + ((Phihex(x, y + delta, phi0hex, r0hex, angle) - Phihex(x, y - delta, phi0hex, r0hex,
                                                                                    angle)) / (2. * delta)) ** 2)

    # Gives electric field in cm-1/(kV/cm) (hence the factor 1.e-5)
    return Ehex


def Whex(x, y, phi0hex, r0hex, N, MN, angle=10):
    E = Ehex(x, y, phi0hex, r0hex, angle)
    Whex = WStark(E, N, MN)
    return Whex


def axhex(x, y, phi0hex, r0hex, N, MN, angle=10):
    axhex = -(((Whex(x + delta, y, phi0hex, r0hex, N, MN, angle) - Whex(x - delta, y, phi0hex, r0hex, N,
                                                                        MN)) / J2wavenr) / (2 * delta * mass))
    return axhex


def ayhex(x, y, phi0hex, r0hex, N, MN):
    ayhex = -(((Whex(x, y + delta, phi0hex, r0hex, N, MN) - Whex(x, y - delta, phi0hex, r0hex, N,
                                                                        MN)) / J2wavenr) / (2 * delta * mass))
    return ayhex


def phasespaceellipse2D(xx0, dxx, rand=None):
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

        xx[1] = xx0[1] + 0.5 * dxx[1] * xr * np.sin(theta)  # Only x-coordinate
        xx[4] = xx0[4] + 0.5 * dxx[4] * xr * np.cos(theta)

        xr = np.sqrt(np.random.uniform())
        theta = np.random.uniform() * 2 * np.pi

        xx[2] = xx0[2] + 0.5 * dxx[2] * xr * np.sin(theta)
        xx[5] = xx0[5] + 0.5 * dxx[5] * xr * np.cos(theta)

        if ((xx[1] - xx0[1]) ** 2 + (xx[2] - xx0[2]) ** 2 <= (0.5 * dxx[1]) ** 2):
            hit = True

    xx[0] = np.random.normal(xx0[0], dxx[0])
    xx[3] = xx0[3]
    xx[6] = np.random.normal(xx0[6], dxx[6])

    if rand == None:
        rand = np.random.uniform()
    if (rand > 2 / 5):
        N = 1
        MN = 0
    else:
        N = 1
        MN = 1

    return [xx, N, MN, hit]


def phasespaceellipse2Dgauss(xx0, dxx, rand=None):
    itry = 0

    hit = False
    while hit == False and itry < 100:
        xx = np.zeros(7)
        itry += 1
        #        print("itry:",itry)
        if itry > 99:
            print("itry exceeded 100!")

        xr = np.sqrt(np.abs(np.random.normal(0, 1 / 2.355)))
        theta = np.random.uniform() * 2 * np.pi

        xx[1] = xx0[1] + dxx[1] * xr * np.sin(theta)  # Only x-coordinate
        xx[2] = xx0[2] + dxx[2] * xr * np.cos(theta)

        xr = np.sqrt(np.random.uniform())
        theta = np.random.uniform() * 2 * np.pi

        xx[4] = xx0[4] + 0.5 * dxx[4] * xr * np.sin(theta)
        xx[5] = xx0[5] + 0.5 * dxx[5] * xr * np.cos(theta)

        # if ((xx[1]-xx0[1])**2 + (xx[2]-xx0[2])**2 <= (0.5*dxx[1])**2):
        hit = True

    xx[0] = np.random.normal(xx0[0], dxx[0])
    xx[3] = xx0[3]
    xx[6] = np.random.normal(xx0[6], dxx[6])

    N = 1
    MN = 0

    return [xx, N, MN, hit]
def freeflight(endpoint, xxs, hit):  # here, xxs is assumed to be 1-D arrays with 7 elems; t,x,y,z,vx,vy,vz

    xxp = np.zeros(7)
    for i in range(0, 7): xxp[i] = xxs[i]
    nsteps = 0
    while (xxs[3] <= xxp[3] < endpoint and hit == True):
        xxp[1] += xxp[4] * (endpoint - xxs[3]) / xxp[6]
        xxp[2] += xxp[5] * (endpoint - xxs[3]) / xxp[6]
        xxp[3] = endpoint

        xxp[0] += (endpoint - xxs[3]) / xxp[6]

        nsteps += 1

    return [xxp, hit]


def hexapole(endpoint, phi0hex, r0hex, xxs, N, NM,
             hit):  # here, xxs is assumed to be 1-D arrays with 7 elems; t,x,y,z,vx,vy,vz

    xxp = np.zeros(7)
    for i in range(0, 7): xxp[i] = xxs[i]
    nsteps = 0
    while ((xxs[3] <= xxp[3] < endpoint) and hit == True):

        #        if (xxp[1]**2 + xxp[2]**2) <= r0hex**2:
        if -np.abs(phi0hex) < Phihex(xxp[1], xxp[2], phi0hex, r0hex) < np.abs(phi0hex) and \
                (xxp[1] ** 2 + xxp[2] ** 2) <= (1.5 * r0hex) ** 2:

            xxp[4] += dt * axhex(xxp[1], xxp[2], phi0hex, r0hex, N, NM)
            xxp[5] += dt * ayhex(xxp[1], xxp[2], phi0hex, r0hex, N, NM)
            xxp[6] += 0

            xxp[1] += dt * xxp[4]
            xxp[2] += dt * xxp[5]
            xxp[3] += dt * xxp[6]

            xxp[0] += dt

            nsteps += 1

        else:  # molecule has hit an electrode. Stop calculation.
            hit = False
            break

    return [xxp, hit]
def ascat(vx,fudgefactor,s0,detuning):
    ascat = fudgefactor*((gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma + wavevector*vx)/gamma)**2))*hbar*wavevector  \
                        -(gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma - wavevector*vx)/gamma)**2))*hbar*wavevector)/mass
    return ascat
def ascat_sub_doppler(vx,ff1,ff2,s1,s2,det1,det2):
    ascat=ff1*((gamma/2)*(s1/(1 + s1 + (2*(det1*gamma + wavevector*vx)/gamma)**2))*hbar*wavevector  \
                        -(gamma/2)*(s1/(1 + s1 + (2*(det1*gamma - wavevector*vx)/gamma)**2))*hbar*wavevector)/mass \
            +ff2*((gamma/2)*(s2/(1 + s2 + (2*(det2*gamma + wavevector*vx)/gamma)**2))*hbar*wavevector  \
                        -(gamma/2)*(s2/(1 + s2 + (2*(det2*gamma - wavevector*vx)/gamma)**2))*hbar*wavevector)/mass
    return ascat
def lasercooling(endpoint,fudgefactor,s0,detuning,xxs,hit
                 ,xcooling = True ,ycooling = True , n_reflections = 35):
    #here, xxs is assumed to be 1-D arrays with 7 elems; t,x,y,z,vx,vy,vz

    centers=[L[0]+L[1]+L[2]+L[3]*float(i+1)/float(n_reflections) for i in range(n_reflections)]
    z=[]
    intensity=[]
    xxp = np.zeros(7)
    for i in range(0,7): xxp[i] = xxs[i]
    nsteps = 0
    #print(endpoint)
    while((xxs[3] <= xxp[3] < endpoint) and hit == True):
        distance_center=abs(xxp[3]- min(centers, key=lambda x:abs(x-xxp[3])))
        alpha0=Gauss(distance_center*1e3,*par2)#*0.08794/(xxp[3])**6.1454
        z.append(xxp[3])
        intensity.append(alpha0)
        #print(distance_center,alpha0)
        if xcooling:
            if((abs(xxp[1]) <= 0.5*x_laser) and (abs(xxp[4]) > 0.5*v_Doppler)) :
                alpha=Gauss(xxp[2]*1e3,*par)
                xxp[4] += dt*ascat(xxp[4],fudgefactor,s0*alpha*alpha0,detuning)
        if ycooling == True:
            if((abs(xxp[2]) <= 0.5*x_laser) and (abs(xxp[5]) > 0.5*v_Doppler)) :
                alpha=Gauss(xxp[1]*1e3,*par)
                xxp[5] += dt*ascat(xxp[5],fudgefactor,s0*alpha*alpha0,detuning)
        xxp[6] += 0
        xxp[1] += dt*xxp[4]
        xxp[2] += dt*xxp[5]
        xxp[3] += dt*xxp[6]
        xxp[0] += dt
        nsteps += 1
    return [xxp, hit]
def lasercooling_sub_doppler(endpoint, ff1, ff2, s1, s2, det1, det2, xxs, hit,
                             xcooling = True ,ycooling = True , n_reflections = 35):
    centers = [L[0] + L[1] + L[2] + L[3] * float(i + 1) / float(n_reflections) for i in range(n_reflections)]
    z = []
    intensity = []
    xxp = np.zeros(7)
    for i in range(0, 7): xxp[i] = xxs[i]
    nsteps = 0
    # print(endpoint)
    while ((xxs[3] <= xxp[3] < endpoint) and hit == True):
        distance_center = abs(xxp[3] - min(centers, key=lambda x: abs(x - xxp[3])))
        alpha0 = Gauss(distance_center * 1e3, *par2) * 0.08794 / (xxp[3]) ** 6.1454
        z.append(xxp[3])
        intensity.append(alpha0)
        # print(distance_center,alpha0)
        if xcooling:
            if ((abs(xxp[1]) <= 0.5 * x_laser) and (abs(xxp[4]) > 0.5 * v_Doppler)):
                alpha = Gauss(xxp[2] * 1e3, *par)
                xxp[4] += dt * ascat_sub_doppler(xxp[4],ff1, ff2, s1*alpha*alpha0, s2*alpha*alpha0, det1, det2)
        if ycooling == True:
            if ((abs(xxp[2]) <= 0.5 * x_laser) and (abs(xxp[5]) > 0.5 * v_Doppler)):
                alpha = Gauss(xxp[1] * 1e3, *par)
                xxp[5] += dt * ascat_sub_doppler(xxp[5], ff1, ff2, s1 * alpha * alpha0, s2 * alpha * alpha0, det1, det2)
        xxp[6] += 0
        xxp[1] += dt * xxp[4]
        xxp[2] += dt * xxp[5]
        xxp[3] += dt * xxp[6]
        xxp[0] += dt
        nsteps += 1
    return [xxp, hit]

def trajectory_simulation(initial_pos, nn, nj, ff, s0, detun, phi0hex,
                          xcooling=True, ycooling=True, lc=True, hex=True,
                          sub_doppler=False, par_sub_doppler=np.zeros(6)):
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
    # print("started at:",time)
    for i in range(0, nn):
        tempz = []
        tempx = []
        tempy = []
        tempvx = []
        tempvy = []
        tempax = []
        tempay = []
        [xx, N,MN, hit] = initial_pos[i]  # How are xx0 and dxx really defined?
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
                if hex:
                    [xx, hit] = hexapole(L[0] + L[1] * float(j) / float(nj - 1), phi0hex, r0hex, xx, N,MN, hit)
                    if hit:
                        tempz.append(xx[3])
                        tempx.append(xx[1])
                        tempy.append(xx[2])
                        tempvx.append(xx[4])
                        tempvy.append(xx[5])
                        #axhex(x, y, phi0hex, r0hex, N, MN
                        tempax.append(axhex(xx[4], xx[5], phi0hex, r0hex, N,MN))
                        tempay.append(ayhex(xx[4], xx[5], phi0hex, r0hex, N, MN))
                else:
                    [xx, hit] = freeflight(L[0] + L[1] * float(j) / float(nj - 1), xx, hit)
                    # print('not cooling')
                    if hit:
                        tempz.append(xx[3])
                        tempx.append(xx[1])
                        tempy.append(xx[2])
                        tempvx.append(xx[4])
                        tempvy.append(xx[5])
                        tempax.append(0)
                        tempay.append(0)

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
                if lc:
                    if sub_doppler == False:
                        [xx, hit] = lasercooling(L[0] + L[1] + L[2] + L[3] * float(j) / float(nj - 1),
                                                 ff, s0, detun, xx, hit, xcooling=xcooling, ycooling=ycooling)
                    else:
                        [xx, hit] = lasercooling_sub_doppler(L[0] + L[1] + L[2] + L[3] * float(j) / float(nj - 1),
                                                             par_sub_doppler[0], par_sub_doppler[1], par_sub_doppler[2],
                                                             par_sub_doppler[3], par_sub_doppler[4], par_sub_doppler[5],
                                                             xx, hit)
                    if hit:
                        tempz.append(xx[3])
                        tempx.append(xx[1])
                        tempy.append(xx[2])
                        tempvx.append(xx[4])
                        tempvy.append(xx[5])
                        tempax.append(ascat(xx[4], ff, s0, detun))
                        tempay.append(ascat(xx[5], ff, s0, detun))

                else:
                    [xx, hit] = freeflight(L[0] + L[1] + L[2] + L[3] * float(j) / float(nj - 1), xx, hit)
                    # print('not cooling')
                    if hit:
                        tempz.append(xx[3])
                        tempx.append(xx[1])
                        tempy.append(xx[2])
                        tempvx.append(xx[4])
                        tempvy.append(xx[5])
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
    return [x_positiion, y_position, z_position, vx, vy, ax, ay]


'''
xy_distribution finds all the molecules at a certain distance and gives back their
x and y coordinates (also their velocities).
We use this one to simulate the heatmaps.
'''


def xy_distribution(xpoints, ypoints, zpoints, vx, vy, Ldetection):
    # First we want the x and y points that are at a distances z. This
    # distance z is Ldetec
    # Lcooling=L[0]+L[1]+L[2]+L[3]+L[4]
    n = len(zpoints)
    x = []
    y = []
    z = []
    vx1 = []
    vy1 = []
    for i in range(n):
        if Ldetection in zpoints[i]:
            index = zpoints[i].index(Ldetection)
            x.append(xpoints[i][index])
            y.append(ypoints[i][index])
            z.append(zpoints[i][index])
            vx1.append(vx[i][index])
            vy1.append(vy[i][index])
    return x, y, z, vx1, vy1
def points_in_circle(x,y,r):
    n=len(x)
    counter=0
    for i in range(n):
        if np.sqrt(x[i]**2+y[i]**2)<=r:
            counter+=1
    return counter
