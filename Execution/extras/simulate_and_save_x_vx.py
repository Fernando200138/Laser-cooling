
import numpy as np
import datetime
import scipy.stats as st
from scipy.optimize import curve_fit
import pandas as pd
#from pathlib import Path

from model import *

gamma=1/60e-9

fudgefactor,s0,detuning=0.03722520507936374,1.7648565663543367,0.6716348438729361
nn=1000000
nj=50

x,y,z,vx,vy,ax,ay=trajectory_simulation(nn,nj, fudgefactor, s0, detuning)
dic={'x':x,'y':y,'z':z,'vx':vx,'vy':vy,'ax':ax,'ay':ay}

df=pd.DataFrame(dic)

path='Position_distribution_laser_cooling/data'
name='nn1e6_nj50_s1.7_ff037_d0_67_nolc.csv'
#np.savetxt(name,np.asarray((x,y,z,vx,vy,ax,ay)),delimiter=',')
df.to_csv(name)