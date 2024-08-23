import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from simulate_trajectories_model import *
from model.helpers import *
from matplotlib.widgets import Slider, Button, RadioButtons
def ascat_sub_doppler(vx,ff1,ff2,s1,s2,det1,det2):
    ascat=ff1*((gamma/2)*(s1/(1 + s1 + (2*(det1*gamma + wavevector*vx)/gamma)**2))*hbar*wavevector  \
                        -(gamma/2)*(s1/(1 + s1 + (2*(det1*gamma - wavevector*vx)/gamma)**2))*hbar*wavevector)/mass \
            +ff2*((gamma/2)*(s2/(1 + s2 + (2*(det2*gamma + wavevector*vx)/gamma)**2))*hbar*wavevector  \
                        -(gamma/2)*(s2/(1 + s2 + (2*(det2*gamma - wavevector*vx)/gamma)**2))*hbar*wavevector)/mass
    return ascat
vx=np.linspace(-30,30,200)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.35)

axis_color = 'lightgoldenrodyellow'

[line] = ax.plot(vx,ascat_sub_doppler(vx,0.4,0.3,7,7,2,-1))
ax.set_ylabel('acceleration (m/s^2)')
ax.set_xlabel('velocity (m/s)')
ax.set_xlim([-30, 30])
ax.set_ylim([-6e3, 6e3])
ff1_slider_ax = fig.add_axes([0.25, 0.25, 0.65, 0.03])
ff2_slider_ax = fig.add_axes([0.25, 0.2, 0.65, 0.03])
s1_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
s2_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
det1_slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
det2_slider_ax = fig.add_axes([0.25, 0., 0.65, 0.03])
ff1_slider=Slider(ff1_slider_ax,'ff1',0.,1)
ff2_slider=Slider(ff2_slider_ax,'ff2',0.,1)
s1_slider=Slider(s1_slider_ax,'s1',0.,10)
s2_slider=Slider(s2_slider_ax,'s2',0.,10)
det1_slider=Slider(det1_slider_ax,'det1',-4,4)
det2_slider=Slider(det2_slider_ax,'det2',-4.,4)
def sliders_on_changed(val):
    line.set_ydata(ascat_sub_doppler(vx,ff1_slider.val,ff2_slider.val,
                                     s1_slider.val,s2_slider.val,det1_slider.val
                                     ,det2_slider.val))
    fig.canvas.draw_idle()
ff1_slider.on_changed(sliders_on_changed)
ff2_slider.on_changed(sliders_on_changed)
s1_slider.on_changed(sliders_on_changed)
s2_slider.on_changed(sliders_on_changed)
det1_slider.on_changed(sliders_on_changed)
det2_slider.on_changed(sliders_on_changed)


#reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
#reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    ff1_slider.reset()
    ff2_slider.reset()
    s1_slider.reset()
    s2_slider.reset()
    det1_slider.reset()
    det2_slider.reset()

#reset_button.on_clicked(reset_button_on_clicked)
plt.tight_layout
plt.show()