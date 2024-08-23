import sys
import os
sys.path.append(os.path.dirname(__file__).split('Analysis')[0]+'Analysis') # Locate analysis scripts

from PlotManager import *
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica']}) # Set font
plt.rc('axes', prop_cycle=plt.cycler('color', [tol_cmap('sunset')(c%1) for c in np.arange(0, 100, 6*np.pi)]))
plt.rc('image', cmap='sunset')
from matplotlib import transforms

pmg = PlotManager(_verbose = True)

# use integrate camera data to find camera counts in detection cube in a certain area for several hexapole voltages
camera_countrange = (4, 20)

pico_countrange = (0, 3)
pico_timerange = (2,8)
pico_timerange_ns = (pico_timerange[0]*1e6, pico_timerange[1]*1e6)
pico_backrange_ns = (0*1e6, 2*1e6)

width_factor = 2.355 # Translate standard deviation to FWHM

abs_timerange = (0,3)
abs_backrange = (2,10)
abs_range = (0,3)


camera_h_countrange = (-100, 900)
pos_range = (-12.5/0.57,12.5/0.57) # mm



mask_center = [-1.2, 1.5] # roughly determined using fits 1567
# center = (2.6, -2.4)

colors = [plt.colormaps['sunset'](0.95),plt.colormaps['sunset'](0.05),plt.colormaps['sunset'](0.85),plt.colormaps['sunset'](0.15)]
marker_colors = [plt.colormaps['sunset'](0.85),plt.colormaps['sunset'](0.15),plt.colormaps['sunset'](0.75),plt.colormaps['sunset'](0.25)]


# Voltage scan
voltages = np.linspace(0, 5.5, 12)
labels = [r"0.0 kV", r"0.5 kV", r"1.0 kV",r"1.5 kV", r"2.0 kV",r"2.5 kV", r"3.0 kV",r"3.5 kV", r"4.0 kV",r"4.5 kV", r"5.0 kV",r"5.5 kV"]
#camera filenames 
camera_filenames = ['3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1771','3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1776','3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1785',\
                    '3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1783','3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1780','3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1782',\
                    '3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1778','3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1786','3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1781',\
                    '3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1779','3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1784','3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1777'
                    ]

#camera_filenames = ['3_camangle-55,0_imageheight-27_Mon Apr 15 2024_1567','3_camangle-55,0_imageheight-27_Mon Apr 15 2024_1568','3_camangle-55,0_imageheight-27_Mon Apr 15 2024_1569',\
#                    '3_camangle-55,0_imageheight-27_Mon Apr 15 2024_1566','3_camangle-55,0_imageheight-27_Mon Apr 15 2024_1565']

camera_counts_all_mean_r5xmm = []
camera_counts_all_mean_r2_5xmm = []
camera_counts_all_std_r5xmm = []
camera_counts_all_std_r2_5xmm = []

integration_mask_size = (2.5,2.5) # circle, radius in mm
integration_mask = pmg.dataManager.create_camera_mask(camera_filenames[0], _type = 'ellipse',  _position = mask_center,\
                                                      _dimension = [integration_mask_size[0], integration_mask_size[1]] , \
                                                        _unit = "mm", _invert = False)
#integration_mask_size = (5,2.5) # circle, radius in mm
#integration_mask = pmg.dataManager.create_camera_mask(camera_filenames[0], _type = 'ellipse',  _position = mask_center,\
#                                                      _dimension = [integration_mask_size, integration_mask_size] , \
#                                                        _unit = "mm", _invert = False)


for vi,v in enumerate(camera_filenames):
    pmg.dataManager.integrate_camera_data(_filename = camera_filenames[vi], _per_shot =True, _mask = integration_mask, _overwrite = True)
    camera_counts_all_mean_r2_5xmm.append(pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signal'])
    #print("[camera_filenames[vi]]['Camera']['integrated_signal'] = ", pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signal'] )
    #print("np.mean([camera_filenames[vi]]['Camera']['integrated_signals']) = ", np.mean(pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signal']))
    #print("np.std([camera_filenames[vi]]['Camera']['integrated_signals'])/np.sqrt(N) = ", np.std(pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signals'])/np.sqrt(len(pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signals'])), len(pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signals']))
    camera_counts_all_std_r2_5xmm.append(np.std(pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signals'])/np.sqrt(len(pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signals'])))

integration_mask_size = (5,5) # circle, radius in mm
integration_mask = pmg.dataManager.create_camera_mask(camera_filenames[0], _type = 'ellipse',  _position = mask_center,\
                                                      _dimension = [integration_mask_size[0], integration_mask_size[1]] , \
                                                        _unit = "mm", _invert = False)


for vi,v in enumerate(camera_filenames):
    pmg.dataManager.integrate_camera_data(_filename = camera_filenames[vi], _mask = integration_mask, _overwrite = True)
    camera_counts_all_mean_r5xmm.append(pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signal'])
    #print("[camera_filenames[vi]]['Camera']['integrated_signal'] = ", pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signal'] )
    camera_counts_all_std_r5xmm.append(np.std(pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signals'])/np.sqrt(len(pmg.dataManager.data[camera_filenames[vi]]['Camera']['integrated_signals'])))


pmg.create_figure(_subplots = [1,1],_size_inches = (10,6), _fontsize=16)
pmg.ax = np.ndarray((1,2), dtype=object, buffer = np.array(((pmg.ax[0,0], pmg.ax[0,0].twinx())))) # Create two axes object with same x-axis, but different y-axis
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.9)

# normalized signal gain    
pmg.ax[0,0].errorbar(voltages, np.array(camera_counts_all_mean_r5xmm/camera_counts_all_mean_r5xmm[0]), np.array(camera_counts_all_std_r5xmm/camera_counts_all_mean_r5xmm[0]), [0.01 for v in voltages],\
    linestyle = 'None', label = 'r = 5 mm', color=colors[0], mec = colors[0], marker='o', ms=8, mew =2, mfc = marker_colors[0])
pmg.ax[0,1].errorbar(voltages, np.array(camera_counts_all_mean_r2_5xmm/camera_counts_all_mean_r2_5xmm[0]), np.array(camera_counts_all_std_r2_5xmm/camera_counts_all_mean_r2_5xmm[0]), [0.01 for v in voltages],\
    linestyle = 'None', label = 'r = 2.5 mm', color=colors[1], mec = colors[1], marker='o', ms=8, mew =2, mfc = marker_colors[1])

# just plot counts in area
#pmg.ax[0,0].errorbar(voltages, np.array(camera_counts_all_mean_r5xmm), np.array(camera_counts_all_std_r5xmm), [0.01 for v in voltages],\
#    linestyle = 'None', label = 'r = 5 mm', color=colors[0], mec = colors[0], marker='o', ms=8, mew =2, mfc = marker_colors[0])
#pmg.ax[0,1].errorbar(voltages, np.array(camera_counts_all_mean_r2_5xmm), np.array(camera_counts_all_std_r2_5xmm), [0.01 for v in voltages],\
#    linestyle = 'None', label = 'r = 2.5 mm', color=colors[1], mec = colors[1], marker='o', ms=8, mew =2, mfc = marker_colors[1])

pmg.ax[0,0].set_xticks(np.linspace(0, 5.5, 12))
#pmg.ax[0,0].set_ylim((8000, 11000))
#pmg.ax[0,1].set_ylim((2000, 3000))
pmg.ax[0,0].set_xlabel("Voltage [kV]", fontsize=16)
pmg.ax[0,0].set_ylabel("Normalized signal r = 5 mm [arb. units]", fontsize=16, color = colors[0])
pmg.ax[0,1].set_ylabel("Normalized signal r = 2.5 mm [arb. units]", fontsize=16, color = colors[1])

pmg.save_figure(os.path.basename(__file__)[:-3]+"_circle_v2")

print("voltages =", voltages)
print("camera_counts_all_mean_r5mm =", camera_counts_all_mean_r5xmm)
print("Normalized camera_counts_all_mean_r5mm =", camera_counts_all_mean_r5xmm/camera_counts_all_mean_r5xmm[0])
print("camera_counts_all_std_r5mm =", camera_counts_all_std_r5xmm)

print("camera_counts_all_mean_r2_5mm =", camera_counts_all_mean_r2_5xmm)
print("Normalized camera_counts_all_mean_r2_5mm =", camera_counts_all_mean_r2_5xmm/camera_counts_all_mean_r2_5xmm[0])
print("camera_counts_all_std_r2_5mm =", camera_counts_all_std_r2_5xmm)
pmg.flush()