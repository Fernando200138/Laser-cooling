from model_fit_max_data import *
import numpy as np
detun = np.array([-0.5,-1,-1.5,-2])
saturation = np.arange(0.5, 15.5, 0.5)
mag_fields=2
ss=[]
dd=[]
ff=[]
ss2=[]
for det in detun:
    ff_fit,sat_fit,det_fit,sat2,R2=get_parameters_from_force_curves(det,saturation)
    ss.append(sat_fit)
    dd.append(det_fit)
    ff.append(ff_fit)
    ss2.append(sat2)
plot_fitted_parameter(ss2,dd,'detuning')












