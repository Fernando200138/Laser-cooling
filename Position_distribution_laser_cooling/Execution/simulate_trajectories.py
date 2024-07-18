#from simulate_trajectories_model import *
from execution import *
from model_fit_max_data  import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import curve_fit
a = datetime.datetime.now()
print(a)
nn = 12000
nj = 150
Ldet1 = L[0]+L[1]+L[2]+L[3]+L[4]
Ldet2 = L[0]+L[1]+L[2]+L[3]+L[4]+L[5]
gamma = 1/(2*np.pi*57e-9) #2.78 MHz

#fudgefactor,s00,detuning=0.26,5.74,1

Voltagehex = [2.5e3,2.5e3,2.5e3,2.5e3]
fudgefactor = [0.3,0.3,0.3,0.3]
s00 = [5,5,5,5]
detuning = [1,-1,1,-1]
lc_bool = [True for _ in range(len(Voltagehex))]
hex_bool = [True for _ in range(len(Voltagehex))]
hfs_bool = [True if i < len(Voltagehex) / 2 else False for i in range(len(Voltagehex))]

#Voltagehex = list(range(0,2000,500))
#fudgefactor = np.ones(len(Voltagehex))
#s00 = np.zeros(len(Voltagehex))
#detuning = np.zeros(len(Voltagehex))
#lc_bool = [False for _ in range(len(Voltagehex))]
#hex_bool = [True for _ in range(len(Voltagehex))]
#hfs_bool = [True for _ in range(len(Voltagehex))]

simulate_lc_2_distance_hex_off(nn,nj,0.3,7,1,2e3,Ldet1, Ldet2,hfs_bool=True,plot_circle=True,plot_safe=True,extra_title='hfs = True ,beam_profile = Gaussian')

#simulate_lc_hex_off_histogram(nn, nj, 0.3, 7, 1, 2e3, Ldet1, Ldet2,hfs_bool=True, plot_circle=False,plot_safe=False,extra_title='')
#simulate_vr_hist(nn,nj,0.3,7,1,2e3,Ldet1,Ldet2,save_plot=False)

#simulate_trajectories_111(nn,nj,0.3,7,1,2e3,Ldet1, Ldet2)
#simulate_n_lc(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2,lc_bool,hex_bool,hfs_bool,plot_titles=True, plot_circle=False,plot_safe=False)

#simulate_lc_2_distance_hex_off(nn, nj, 0.27, 8.06, 1.04, 2.5e3, Ldet1, Ldet2,plot_safe=False)


b = datetime.datetime.now()
print(b-a)