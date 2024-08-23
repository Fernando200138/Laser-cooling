from model.molecules import BaF
from model.helpers import *
from model.save_and_load_forces import *
from matplotlib import pyplot as plt
from scipy.optimize import leastsq

import datetime
import scipy.stats as st
from scipy.optimize import curve_fit

c    = 299792458         # m/s, speed of light
kB   = 1.380649e-23      # J/K. Boltzmann constant
amu  = 1.66053904e-27    # kg, molar mass conversion constant
h    = 6.62607015e-34    # J s,  Planck constant
hbar = h/(2*np.pi)
g    = 9.81              # m/s**2, constant of acceleration due to Earth's gravity

wavevector =  2*np.pi/860e-9
gamma      =  1/60e-9

mass = BaF.mass
molecule = BaF

omega = 2 * np.pi * (cts.c / molecule.wave_length)
Isat = cts.hbar * omega ** 3 * (2 * np.pi * molecule.line_width_in_MHz * 1e6) / (12 * np.pi * cts.c ** 2)
def fitfunction(vx_list,fudgefactor,s0,detuning):
    fit = np.array([0.0 for j in range(len(vx_list))])
    for j in range(len(vx_list)):
        fit[j] = fudgefactor*((gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma + wavevector*vx_list[j])/gamma)**2))*hbar*wavevector \
                              -(gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma - wavevector*vx_list[j])/gamma)**2))*hbar*wavevector)/mass
    return fit
def ascat(vx,fudgefactor,s0,detuning):
    ascat = fudgefactor*((gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma + wavevector*vx)/gamma)**2))*hbar*wavevector  \
                        -(gamma/2)*(s0/(1 + s0 + (2*(detuning*gamma - wavevector*vx)/gamma)**2))*hbar*wavevector)/mass
    return ascat

def residual(vars, vx_list,ascat_sim, eps_data):
   afit = vars
   chi  = 0.
   fit  = fitfunction(vx_list,afit[0],afit[1],afit[2])
   for j in range(len(vx_list)):
       chi +=    ((ascat_sim - fit[j])/1.)**2/float(len(vx_list))
#   print(np.sqrt(chi))
#   print(vars)
   return ((ascat_sim-fit) / 1.)


def curve_fit2(vel,acc,saturation,detuning,fudgefactor=1):
    vary = np.array(vel.shape) + 1.0

    afit = [fudgefactor, saturation, detuning]
    vars = afit
    out = leastsq(residual, vars, args=(vel, acc, np.sqrt(vary)))

    fudgefactor = out[0][0]
    s0 = out[0][1]
    detuning = out[0][2]
    return fudgefactor, s0, detuning

def get_R2(vel,acc,parameters):
    resid=acc-ascat(vel,*parameters)
    ss_res=np.sum(resid**2)
    ss_tot=np.sum((acc-np.mean(acc))**2)
    return 1-(ss_res/ss_tot)



def get_parameters_from_force_curves(det):
    saturation = np.arange(0.5, 15.5, 0.5)
    det_temp = []
    sat_temp= []
    ff_temp = []
    sat_temp2=[]
    R2_temp=[]
    print(det)
    for sat in saturation:
        try:
            velocity_sim, ascat_sim = load_forces('obe', det, sat, molecule, additional_title='2.0_45',
                                                  velocity_in_y=False, directory='data_grid')
            parameters,cov=curve_fit(ascat,velocity_sim,ascat_sim)
            #parameters = curve_fit2(velocity_sim, ascat_sim, det, sat)
            R2_temp.append(get_R2(velocity_sim,ascat_sim,parameters))
            sat_temp.append(parameters[1])
            ff_temp.append(parameters[0])
            det_temp.append(parameters[2])
            sat_temp2.append(sat)
        except RuntimeError:
            continue
    return ff_temp,sat_temp,det_temp,sat_temp2,R2_temp


def plot_sat_parameter(sat2,fit,parameter):
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(8, 4)
    a,par1=fit_polynomial(sat2[0],fit[0],1)
    a,par2=fit_polynomial(sat2[1],fit[1],1)
    a,par3=fit_polynomial(sat2[2],fit[2],1)
    a,par4=fit_polynomial(sat2[3],fit[3],1)

    x1, y1 = connect_dots(sat2[0], fit[0])
    x2, y2 = connect_dots(sat2[1], fit[1])
    x3, y3 = connect_dots(sat2[2], fit[2])
    x4, y4 = connect_dots(sat2[3], fit[3])

    fig, axs = plt.subplots(2, 2)
    axs[0,0].plot(sat2[0],pol_to_rule_them_all(np.array(sat2[0]),*par1),color='red')
    axs[0,1].plot(sat2[0],pol_to_rule_them_all(np.array(sat2[0]),*par2),color='red')
    axs[1,0].plot(sat2[0],pol_to_rule_them_all(np.array(sat2[0]),*par3),color='red')
    axs[1,1].plot(sat2[0],pol_to_rule_them_all(np.array(sat2[0]),*par4),color='red')

    axs[0,0].scatter(sat2[0],fit[0],color='blue')
    axs[0,1].scatter(sat2[1],fit[1],color='blue')
    axs[1, 0].scatter(sat2[2], fit[2], color='blue')
    axs[1,1].scatter(sat2[3],fit[3],color='blue')

    axs[0, 0].plot(x1, y1, color='black')
    axs[0, 1].plot(x2, y2, color='black')
    axs[1, 0].plot(x3, y3, color='black')
    axs[1, 1].plot(x4, y4, color='black')

    axs[0, 0].set_title('$\\Delta = -0.5 \\Gamma$ ')
    axs[0, 1].set_title('$\\Delta = -1 \\Gamma$')
    axs[1, 0].set_title('$\\Delta = -1.5 \\Gamma$')
    axs[1, 1].set_title('$ \\Delta = -2 \\Gamma$')

    ylabel=parameter+' two level system'
    acceleration_par=parameter+'fitted'
    fontsize=12
    ticks_font=10
    for ax in axs.flat:
        #ax.set(xlabel='sat pylcp', ylabel=ylabel,fontsize=fontsize)
        ax.set_xlabel('sat pylcp',fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)

        ax.legend(['Grade 11 polynomial fitting',acceleration_par,'connecting dots'])
        ax.tick_params(axis='x', labelsize=ticks_font)
        ax.tick_params(axis='y', labelsize=ticks_font)
    #for ax in axs.flat:
     #   ax.label_outer()
    fig.tight_layout()
    plt.show()
    plt.clf()


def pol_to_rule_them_all(x,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11):
    return (a0+a1*x+a2*x**2+a3*x**3+a4*x**4+a5*x**5+a6*x**6+a7*x**7+a8*x**8+a9*x**9+a10*x**10+a11*x**11)

def connect_dots(x,y):
    step=10
    points=np.linspace(0,10,step)
    x_final=[]
    num_points=len(x)
    y_final=[]
    for i in range(1,len(y)):
        X_initial=np.array((x[i-1],y[i-1]))
        X_final=np.array((x[i],y[i]))
        for t in list(np.linspace(0,1,step)):

            v=X_initial+t*(X_final-X_initial)
            y_final.append(v[1])
            x_final.append(v[0])
    return np.array(x_final),np.array(y_final)

def fit_polynomial(sat,y,sat_fit):
    par,cov=curve_fit(pol_to_rule_them_all,sat,y)
    return pol_to_rule_them_all(sat_fit,*par),par

def get_ff_sat_det(detun_py,sat_py):
    ff_fit,sat_fit,det_fit,sat2,R2=get_parameters_from_force_curves(detun_py)
    ff_final,par=fit_polynomial(sat2,ff_fit,sat_py)
    sat_final,par=fit_polynomial(sat2,sat_fit,sat_py)
    det_final,par=fit_polynomial(sat2,det_fit,sat_py)
    return ff_final,sat_final,det_final

def get_ff_sat_det_cd_method(detun_py,sat_py):
    ff_fit,sat_fit,det_fit,sat2,R2=get_parameters_from_force_curves(detun_py)
    ff_final=find_parameter_connecting_dots(sat2,ff_fit,sat_py)
    sat_final=find_parameter_connecting_dots(sat2,sat_fit,sat_py)
    det_final=find_parameter_connecting_dots(sat2,det_fit,sat_py)
    return ff_final,sat_final,det_final


def find_parameter_connecting_dots(sat,y,sat_py):
    sat,y=connect_dots(sat,y)
    optimal=min(sat, key=lambda x:abs(x-sat_py))
    index=list(sat).index(optimal)
    return y[index]