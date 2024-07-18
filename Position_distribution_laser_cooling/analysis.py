import matplotlib.pyplot as plt
import numpy as np
from model_fit_max_data import *
import numpy as np
from collections import Counter
from itertools import chain

#detun = np.array([-0.5,-1,-1.5,-2])
#saturation=np.arange(0.5,15.5,0.5)
#mag_fields=2

detun=-1.5
sat_py=6.522

ff,sat,det=get_ff_sat_det(detun,sat_py)

#velocity_sim, ascat_sim = load_forces('obe', detun, sat_py, molecule, additional_title='2.0_45',
#                                                  velocity_in_y=False, directory='data_grid')


vel=np.linspace(0,10,101)
y_mine=ascat(vel,ff,sat,det)

#plt.plot(velocity_sim,ascat_sim,color='blue')
plt.plot(vel,y_mine,color='red')
plt.show()

'''
sat_fit,det_fit,ff_fit,sat2,R2=get_parameters_from_force_curves(detun)
y,par=fit_polynomial(sat2[0],det_fit[0])



freq=Counter(chain.from_iterable(sat2))
res = [idx for idx in freq if freq[idx] == 1]
print(res)

set0=set(sat2[0])
set1=set(sat2[1])
set2=set(sat2[2])
set3=set(sat2[3])
sd0=set3.symmetric_difference(set0)
sd1=set3.symmetric_difference(set1)
sd2=set3.symmetric_difference(set2)

SAT0=[i for i,item in enumerate(sat2[0]) if item not in res]
SAT1=[i for i,item in enumerate(sat2[1]) if item not in res]
SAT2=[i for i,item in enumerate(sat2[2]) if item not in res] 
SAT3=[i for i,item in enumerate(sat2[3]) if item not in res]


print(len(SAT0),len(SAT1),len(SAT2),len(SAT3))
#optimal_par,cov=return_optimal_paramters(sat2[0],det_fit[0],det_fit[1],det_fit[2],det_fit[3])
print([len(sat2[i]) for i in range(0,len(sat2))])
'''
'''
yy=pol_to_rule_them_all(sat2[1],*par)
plt.plot(sat2[1],3.2*yy,'r')
plt.plot(sat2[3],det_fit[3],'b')
plt.show()
#print(sat_fit)
#plot_sat_R2(sat2, R2)
#plot_sat_detfit(sat2,det_fit)
'''















