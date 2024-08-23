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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tol_colors import *
import random
plt.colormaps.register(tol_cmap('sunset'))

# Global constants
c    = 299792458         # m/s, speed of light
kB   = 1.380649e-23      # J/K. Boltzmann constant
amu  = 1.66053904e-27    # kg, molar mass conversion constant
h    = 6.62607015e-34    # J s,  Planck constant
hbar = h/(2*np.pi)
g    = 9.81              # m/s**2, constant of acceleration due to Earth's gravity

# properties of setup
Labs     =  0.005
Lfront   =  0.197
Lhex     =  0.390
Lback    =  0.127 - 0.04           
Llc      =  0.15
Ldetect  =  0.60
Lsp      =  3.50 - Ldetect - Llc - Lback - Lhex - Lfront

L        = [Lfront-Labs, Lhex, Lback, Llc, Ldetect, Lsp]

# Properties for BaF (or SrF)
#mass = (88 + 19) *amu   
mass = (138 + 19) *amu   
B    = 0.21594802         # rotational constant of BaF in cm-1 (Master Thesis Jeroen Maat)
mu   = 3.170*0.0168       # dipole moment in cm-1/(kv/cm)  (debye * conversionfactor)

Starkfit = [[0 for i in range(7)] for j in range(5)]
Starkfit[0] = [1.99832523,  2.78038613,  1.0962849,  11.2629089,   1.6618291,   0.,  0.,  0.,  0.]
# parameters for N=1,M=0 state of BaF. Which parameters?
Starkfit[1] = [1.99297730e+00, -2.30557413e-06,  6.22563075e-04,  1.66160290e+00,  2.42295526e-01,  2.69430173e+00,
               4.12591610e-01]    # parametersfor N=1,M=1 state of BaF
#Starkfit[0] = [1.9,  2.7,  1.09,  11.2,   1.6,   0.,  0.,  0.,  0.]
#Starkfit[0]=[i+1.5 for i in Starkfit[0]]
# parameters for N=1,M=0 state of BaF. Which parameters?
#Starkfit[1] = [1.9e+00, -2.3e-06,  6.2e-04,  1.6,  2.4e-01,  2.6, 4.1e-01]
#Starkfit[1] = [i+1.5 for i in Starkfit[1]]

delta    = 1.e-6          # used for taking derivative of field. > 1.e-4 the results start to deviate.
dt       = 1.e-4          # timestep in the guiding() function; depends on the acceleration.
                          # 1.e-4 is sufficient for an error less than 10 micron after 5 meters of flight.
J2wavenr = 1/(100*h*c)    # factor to convert energy in cm-1 to SI

r0hex    = 6.e-3          # inner radius hexapole
wavevector =  2*np.pi/860e-9
gamma      =  1/57e-9 #decay rate from A2pi lifetime of 57.1 ns
x_laser    =  10.e-3      # size laserbeam
v_Doppler  =  0.1         # minimum velocity corresponding to the Doppler temperature in m/s
#xx0      = [0.66e-3,   0.5e-3,    0.5e-3,  0.,   -1.5,  -1.5, 184.]
#dxx      = [0.33e-3,  4.5e-3, 4.5e-3, 0.e-3, 20., 20., 27.]
xx0      = [0.66e-3,   0,    0.e-3,  0.,   0.,  0., 184.]
dxx      = [0.33e-3,  4.5e-3, 4.5e-3, 0.e-3, 20., 20., 27.]
ni = 100      # number of voltage steps !200
pi=np.pi
'''
The Gauss function is used to simulate the laser beam.
In the first iteration of the simulation the beam was uniform, meanining that as long as you were
inside it you would feel the full force of the laser beam. Now you only feel the full force of the laser beam
and outside you have a Gaussian distribution.
'''
def Gauss(x, a, sigma):
    return a*np.exp(-(x)**2/(2*sigma**2))
xbeam    =  10.    # size laserbeam
X=[-0.5*xbeam,0,0.5*xbeam]
Y=[1/np.e**2,1,1/np.e**2]
par,cov=curve_fit(Gauss, X, Y)
X2=[-0.5*4,0,0.5*4]
Y2=[1/np.e**2,1,1/np.e**2]
par2,cov2=curve_fit(Gauss, X2, Y2)
'''
Can't talk too much about the WStark function. I just know it has to something to do
with the Stark shift.
Based on numerical approximation of derivative and on full expression of the electrostatic potential.

The expressions for the electrostatic potential for a quadrupole and hexapole
are take from Eq. 24 and 26 of van de Meerakker Chem. Rev. 112 4826 (2012), with a2=2 and a3=3, respectively.
'''
def WStark(E,s):
   muoverB  = mu/B     
   WStark   = B*(Starkfit[s][0] + (np.sqrt(Starkfit[s][1]**2 + (Starkfit[s][2]*muoverB*E)**2) - Starkfit[s][1]) \
                                - (np.sqrt(Starkfit[s][3]**2 + (Starkfit[s][4]*muoverB*E)**2) - Starkfit[s][3]) \
                                - (np.sqrt(Starkfit[s][5]**2 + (Starkfit[s][6]*muoverB*E)**2) - Starkfit[s][5]) \
                 )
   return WStark 
def Phihex(x,y,phi0hex,r0hex): 
    r = (x**2+y**2)**(1/2)
    if r > 0:
        if y > 0. :
            theta = np.arccos(x/r) - (10/180)*np.pi
        else :
            theta = -np.arccos(x/r) - (10/180)*np.pi
    phihex = (r/r0hex)**3 * (phi0hex * np.cos(3*theta))   # with a3=3
    '''They are using equation 26'''
    return phihex
def Ehex(x,y,phi0hex,r0hex):    
    Ehex = 1.e-5*np.sqrt(((Phihex(x+delta,y,phi0hex,r0hex)-Phihex(x-delta,y,phi0hex,r0hex))/(2.*delta))**2 \
                        +((Phihex(x,y+delta,phi0hex,r0hex)-Phihex(x,y-delta,phi0hex,r0hex))/(2.*delta))**2)
                                  # Gives electric field in cm-1/(kV/cm) (hence the factor 1.e-5)
    return Ehex
def Whex(x,y,phi0hex,r0hex,s):
   E        =  Ehex(x,y,phi0hex,r0hex)  
   Whex     =  WStark(E,s)
   return Whex 
'''
Acceleration in the x and y axis
'''
def axhex(x,y,phi0hex,r0hex,s):
    axhex = -(((Whex(x+delta,y,phi0hex,r0hex,s)-Whex(x-delta,y,phi0hex,r0hex,s))/J2wavenr)/(2*delta*mass))
    return axhex
def ayhex(x,y,phi0hex,r0hex,s):
    ayhex = -(((Whex(x,y+delta,phi0hex,r0hex,s)-Whex(x,y-delta,phi0hex,r0hex,s))/J2wavenr)/(2*delta*mass))
    return ayhex      
'''
The ascat function is a fit for the acceleration as a function of velocities

'''
# see eq. 7.1 Metcalf and van der Straten with factor 4500/15000 to get proper acceleration
# according to simulations of Roman
def ascat(vx,fudgefactor,s0,detuning):
    detuning=-detuning
    ascat = fudgefactor*((gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma + wavevector*vx)/gamma)**2))*hbar*wavevector  \
                        -(gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma - wavevector*vx)/gamma)**2))*hbar*wavevector)/mass  
    return ascat
def ascat_sub_doppler(vx,ff1,ff2,s1,s2,det1,det2):
    ascat=ff1*((gamma/2)*(s1/(1 + s1 + (2*(det1*gamma + wavevector*vx)/gamma)**2))*hbar*wavevector  \
                        -(gamma/2)*(s1/(1 + s1 + (2*(det1*gamma - wavevector*vx)/gamma)**2))*hbar*wavevector)/mass \
            +ff2*((gamma/2)*(s2/(1 + s2 + (2*(det2*gamma + wavevector*vx)/gamma)**2))*hbar*wavevector  \
                        -(gamma/2)*(s2/(1 + s2 + (2*(det2*gamma - wavevector*vx)/gamma)**2))*hbar*wavevector)/mass
    return ascat

def fitfunction(vx_list,ff,s0,detuning):
    fit = np.array([0.0 for j in range(len(vx_list))])
    for j in range(len(vx_list)):
        fit[j] = ff*((gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma + wavevector*vx_list[j])/gamma)**2))*hbar*wavevector  \
                -(gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma - wavevector*vx_list[j])/gamma)**2))*hbar*wavevector)/mass
    return fit

#i.e., ascat and fitfunction are the same, but fitfunction returns a list of accelerations rather than one acceleration
'''
phasespaceellipse2D gives the initial trajectory of the molecules. All molecules start in xx0 but phasespaceelipse2D 
gives and initial velocity and a direction.
hfs_percentage is the percentage of high field seekers in the simulation
'''
def phasespaceellipse2D(xx0,dxx,lfs_percentage=0.666):
    itry = 0
    hit = False
    while hit == False and itry < 100:
        xx = np.zeros(7)
        itry += 1
        if itry > 99 :
            print("itry exceeded 100!")
        xr = np.sqrt(np.random.uniform())
        theta = np.random.uniform()*2*pi
        '''
        This are the coordinates of the random walk that are been used to simulate the trajectory of the particle.
        Why do we need a random walk to simulate it? Since particles are emitting photons in random directions
        '''
        xx[1] = xx0[1] + 0.5*dxx[1]*xr*np.sin(theta)  # Only x-coordinate. This is the position of the particle
        xx[4] = xx0[4] + 0.5*dxx[4]*xr*np.cos(theta)  #This is the velocity of the particle.

        xr = np.sqrt(np.random.uniform())
        theta = np.random.uniform()*2*pi
        '''
        This lines may seem redudant but they're not. They are used to defined the coordinates for the y-axis. 
        If they weren't redefined then x==y would always happpend.
        '''
        xx[2] = xx0[2] + 0.5*dxx[2]*xr*np.sin(theta)
        xx[5] = xx0[5] + 0.5*dxx[5]*xr*np.cos(theta)
        if ((xx[1]-xx0[1])**2 + (xx[2]-xx0[2])**2 < (0.5*dxx[1])**2):
            hit = True
    xx[0] = np.random.normal(xx0[0],dxx[0])    
    xx[3] = xx0[3]
    xx[6] = np.random.normal(xx0[6],dxx[6])     

    if(np.random.uniform() >lfs_percentage):              # in N=1; 3/5 in m=1 and 2/5 in m=0; (if > 0.4 then s=0)
       s = 0            #High field seekers
    else:
       s = 1            #Low field seekers
               #High field seekers
#        xr = np.sqrt(np.random.uniform())            
#        theta = np.random.uniform()*2*np.pi
#        xx[3] = xx0[3] + 0.5*dxx[3]*xr*np.sin(theta)  # Only z-coordinate
#        xx[6] = xx0[6] + 0.5*dxx[6]*xr*np.cos(theta)
    return [xx,s,hit]

'''
This funcition runs until it finds a point it can work with i.e it keeps redefining xx
'''
'''
freeflight just let's particles evolve without any external force
'''
def freeflight(endpoint,xxs,hit):
    #here, xxs is assumed to be 1-D arrays with 7 elems; t,x,y,z,vx,vy,vz
    xxp = np.zeros(7)
    nsteps=0
    for i in range(0,7): xxp[i] = xxs[i]
    while(xxs[3] <= xxp[3] < endpoint and hit == True):
        xxp[1] += xxp[4]*(endpoint-xxs[3])/xxp[6]
        xxp[2] += xxp[5]*(endpoint-xxs[3])/xxp[6]
        xxp[3]  = endpoint
        #Wouldn't this only run for one iteration?
        xxp[0] += (endpoint-xxs[3])/xxp[6]
        nsteps     += 1
    return [xxp,hit]     
'''
freeflight will always return a hit==True because it never changes the variable.
'''
def hexapole(endpoint,phi0hex,r0hex,xxs,s,hit):
    #here, xxs is assumed to be 1-D arrays with 7 elems; t,x,y,z,vx,vy,vz
    xxp = np.zeros(7)
    for i in range(0,7): xxp[i] = xxs[i]
    nsteps = 0   
    while((xxs[3] <= xxp[3] < endpoint) and hit == True):
        if -phi0hex < Phihex(xxp[1],xxp[2],phi0hex,r0hex) < phi0hex :
            xxp[4] += dt*axhex(xxp[1],xxp[2],phi0hex,r0hex,s)  
            '''
            Here I think that axhex is the function that changes the trajectory of the particle (an also ayhex)
            '''
            xxp[5] += dt*ayhex(xxp[1],xxp[2],phi0hex,r0hex,s) 
            xxp[6] += 0
            xxp[1] += dt*xxp[4]
            xxp[2] += dt*xxp[5]
            xxp[3] += dt*xxp[6]
            xxp[0] += dt
            nsteps += 1
        else:                                               # molecule has hit an electrode. Stop calculation.
            hit = False
            break
    return [xxp,hit]     
'''
lasercooling will calculate the acceleration experience by the particles.
endpoint is the distance the particles will move through the laser cooling
fudgefactor,s0, detuning are parameters adjustable to laser cooling.
n_reflections are the number of reflections will have inside the laser cooling region 
'''
Rp_736, R_s736 = 0.9992, 0.9998
Rp_860, R_s860 = 0.9815, 0.9916
Vc_window736, Vc_window860 = 0.992, 0.9932
R_736 = (Rp_736+Rp_736)/2
R_860 = (Rp_860+Rp_860)/2
R=(R_736+R_860)/2
Vc_window = (Vc_window860+Vc_window736)/2

def lasercooling(endpoint,fudgefactor,s0,detuning,xxs,hit
                 ,xcooling = True ,ycooling = True , n_reflections = 35):
    #here, xxs is assumed to be 1-D arrays with 7 elems; t,x,y,z,vx,vy,vz
    centers = [L[0] + L[1] + L[2] + L[3] * float(i + 1) / float(n_reflections) for i in range(n_reflections)]
    centers2= np.zeros(n_reflections)
    for i in range(n_reflections): centers2[i] = centers[i]
    z=[]
    intensity=[]
    xxp = np.zeros(7)
    for i in range(0,7): xxp[i] = xxs[i]
    nsteps = 0
    #print(endpoint)
    iter_reflection = 0
    alpha00 = 1

    while((xxs[3] <= xxp[3] < endpoint) and hit == True):
        distance_center=abs(xxp[3]- min(centers, key=lambda x:abs(x-xxp[3])))
        #if abs(xxp[3]- min(centers2, key=lambda x:abs(x-xxp[3]))) < 0.0001:
        #    alpha00 = ((R+Vc_window)/2)**iter_reflection
        #    centers2[iter_reflection] =0.
        #    iter_reflection += 1


        alpha0=Gauss(distance_center*1e3,*par2)#*alpha00#*0.08794/(xxp[3])**6.1454
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

    #plt.plot(np.array(z)*1e3,intensity)
    #plt.plot(np.array(z) * 1e3, [1 / np.e ** 2 for i in z])
    #plt.xlabel('z(mm)')
    #plt.ylabel('intensity (%)')
    #plt.legend(['num_reflections = '+str(n_reflections),'1/e^2'])
    #plt.show()
    #peaks=np.ones(len(centers))
    #return peaks, centers
    return [xxp,hit]
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


'''
trajectory_simulation will simulate the trajectory of nn particles (pretty self explanatory)
initial is the initial distribution of all particles. It's an array of size nn.
nj is the number of steps will take in the hexapole and laser cooling region. In reality we only need two. 
If we simulate only two we'll have just a line (instead of a curve) in the hexapole and laser cooling region,
but the forces in the last point of this stages are still the exact same.
If we are simulating the distribution of x-y molecules it really doesn't matter nj. But if we are simulating the 
trajectory and want to see how the molecules behave in those regions then it those matter.

xcooling,ycooling,lc and hex all have True assign to them. If we want to turn off say the laser cooling we just assign 
lc=False.
hex is for hexapole 
'''

def trajectory_simulation(initial_pos ,nn ,nj ,ff ,s0 ,detun ,phi0hex ,
                          xcooling = True ,ycooling = True ,lc = True, hex = True,
                          sub_doppler = False,par_sub_doppler =np.zeros(6)):
    z_position=[]
    x_position=[]
    y_position=[]
    vx=[]
    vy=[]
    vz=[]
    ax=[]
    ay=[]
    xx=np.zeros(7)

#nn number of molecules
#nj number of points in each stage of the trajectory (counting only hexapole and laser cooling)
    time = datetime.datetime.now().time()
   # print("started at:",time)
    for i in range(0,nn):
        tempz=[]
        tempx=[]
        tempy=[]
        tempvx=[]
        tempvy=[]
        tempax=[]
        tempay=[]
        [xx,s,hit]=initial_pos[i]#How are xx0 and dxx really defined?
        tempz.append(xx[3])
        tempx.append(xx[1])
        tempy.append(xx[2])
        tempvx.append(xx[4])
        tempvy.append(xx[5])
        tempax.append(0.)
        tempay.append(0.)
        [xx,hit]=freeflight(L[0],xx,hit)
        if hit:
            tempz.append(xx[3])
            tempx.append(xx[1])
            tempy.append(xx[2])
            tempvx.append(xx[4])
            tempvy.append(xx[5])
            tempax.append(0.)
            tempay.append(0.)
        if L[1]>0:
            for j in range(nj):
                if hex:
                    [xx,hit]=hexapole(L[0]+L[1]*float(j)/float(nj-1),phi0hex,r0hex,xx,s,hit)
                    if hit:
                        tempz.append(xx[3])
                        tempx.append(xx[1])
                        tempy.append(xx[2])
                        tempvx.append(xx[4])
                        tempvy.append(xx[5])
                        tempax.append(axhex(xx[4],xx[5],phi0hex,r0hex,s))
                        tempay.append(ayhex(xx[4],xx[5],phi0hex,r0hex,s))
                else:
                    [xx,hit]=freeflight(L[0]+L[1]*float(j)/float(nj-1),xx,hit)
                    #print('not cooling')
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
        [xx,hit]=freeflight(L[0]+L[1]+L[2],xx,hit)
        if hit:
            tempz.append(xx[3])
            tempx.append(xx[1])
            tempy.append(xx[2])
            tempvx.append(xx[4])
            tempvy.append(xx[5])
            tempax.append(0.)
            tempay.append(0.)
        if L[1]>0:
            for j in range(nj):
                if lc:
                    if sub_doppler==False:
                        [xx,hit]=lasercooling(L[0]+L[1]+L[2]+L[3]*float(j)/float(nj-1),
                                              ff ,s0 ,detun ,xx ,hit ,xcooling = xcooling , ycooling = ycooling)
                    else:
                        [xx,hit] = lasercooling_sub_doppler(L[0]+L[1]+L[2]+L[3]*float(j)/float(nj-1),
                                                            par_sub_doppler[0], par_sub_doppler[1], par_sub_doppler[2],
                                                            par_sub_doppler[3], par_sub_doppler[4], par_sub_doppler[5],
                                                            xx, hit)
                    if hit:
                        tempz.append(xx[3])
                        tempx.append(xx[1])
                        tempy.append(xx[2])
                        tempvx.append(xx[4])
                        tempvy.append(xx[5])
                        tempax.append(ascat(xx[4],ff,s0,detun))
                        tempay.append(ascat(xx[5],ff,s0,detun))

                else:
                    [xx,hit]=freeflight(L[0]+L[1]+L[2]+L[3]*float(j)/float(nj-1),xx,hit)
                    #print('not cooling')
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
        [xx,hit]=freeflight(L[0]+L[1]+L[2]+L[3]+L[4],xx,hit)
        if hit:
            tempz.append(xx[3])
            tempx.append(xx[1])
            tempy.append(xx[2])
            tempvx.append(xx[4])
            tempvy.append(xx[5])
            tempax.append(0.)
            tempay.append(0.)
        [xx,hit] = freeflight(L[0] + L[1] + L[2] + L[3] + L[4] + L[5], xx, hit) 
        if hit:            
            tempz.append(xx[3])
            tempx.append(xx[1])
            tempy.append(xx[2])
            tempvx.append(xx[4])
            tempvy.append(xx[5])
            tempax.append(0.)
            tempay.append(0.)
        z_position.append(tempz)
        x_position.append(tempx)
        y_position.append(tempy)
        vx.append(tempvx)
        vy.append(tempvy)
        ax.append(tempax)
        ay.append(tempay)
    return [x_position,y_position,z_position,vx,vy,ax,ay]

'''
xy_distribution finds all the molecules at a certain distance and gives back their
x and y coordinates (also their velocities).
We use this one to simulate the heatmaps.
'''
def xy_distribution(xpoints,ypoints,zpoints,vx,vy,Ldetection):
    #First we want the x and y points that are at a distances z. This
    #distance z is Ldetec
    #Lcooling=L[0]+L[1]+L[2]+L[3]+L[4]
    n=len(zpoints)
    x=[]
    y=[]
    z=[]
    vx1=[]
    vy1=[]
    for i in range(n):
        if Ldetection in zpoints[i]:
            index=zpoints[i].index(Ldetection)
            x.append(xpoints[i][index])
            y.append(ypoints[i][index])
            z.append(zpoints[i][index])
            vx1.append(vx[i][index])
            vy1.append(vy[i][index])
    return x,y,z,vx1,vy1
'''
Gives the number of molecules insisde a circle of radius r
'''
def points_in_circle(x,y,r):
    n=len(x)
    counter=0
    for i in range(n):
        if np.sqrt(x[i]**2+y[i]**2)<=r:
            counter+=1
    return counter

import pickle

def export_phase_space(_filename, _t = [], _x = [], _y = [], _z = [], _vx = [], _vy = [], _vz = []):
    if (_t != []):
        file = open(_filename+"_t.simbin", "wb")
        pickle.dump(_t, file)
        file.close()
    if (_x != []):
        file = open(_filename+"_x.simbin", "wb")
        pickle.dump(_x, file)
        file.close()
    if (_y != []):
        file = open(_filename+"_y.simbin", "wb")
        pickle.dump(_y, file)
        file.close()
    if (_z != []):
        file = open(_filename+"_z.simbin", "wb")
        pickle.dump(_z, file)
        file.close()
    if (_vx != []):
        file = open(_filename+"_vx.simbin", "wb")
        pickle.dump(_vx, file)
        file.close()
    if (_vy != []):
        file = open(_filename+"_vy.simbin", "wb")
        pickle.dump(_vz, file)
        file.close()
    if (_vz != []):
        file = open(_filename+"_vz.simbin", "wb")
        pickle.dump(_x, file)
        file.close()