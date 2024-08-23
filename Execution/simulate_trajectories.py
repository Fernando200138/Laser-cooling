import numpy as np

from execution import *
import datetime
a = datetime.datetime.now()
print(a)
nn = int(5e4) #number of molecules
nj = 2 #steps in laser cooling stage, freeflight and hexapole
Ldet1 = L[0]+L[1]+L[2]+L[3]+L[4]  #laser cooling detection distance, defined in simulate_trajectories_model.py
Ldet2 = L[0]+L[1]+L[2]+L[3]+L[4]+L[5] #spin-precession distance, defined in simulate_trajectories_model.py
fudgefactor=0.3 #scattering rate
s00=7 #saturation
detuning=-1 #detuning
Voltagehex=list(np.linspace(0,5.5e3,12))
fudgefactor=[0.3 for i in range(12)]
s00=[7 for i in range(12)]
detuning=[-1 for i in range(12)]
lc_bool=[False for i in range(12)]
simulate_number_mol_vs_hexvoltage(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, lc_bool, lfs=0.666)
b = datetime.datetime.now()
print(b-a)