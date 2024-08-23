import sys
import os
sys.path.append(os.path.dirname(__file__).split('Analysis')[0]+'Analysis') # Locate analysis scripts

from PlotManager import *
from matplotlib import rc

#rc('font',**{'family':'serif','serif':['Helvetica']}) # Set font
plt.rc('axes', prop_cycle=plt.cycler('color', [tol_cmap('sunset')(c%1) for c in np.arange(0, 100, 6*np.pi)]))
plt.rc('image', cmap='sunset')
from matplotlib import transforms

### Plot several camera images side by side. In this case for L00 and L00 + L10
pmg = PlotManager(_verbose = True)


angle_correct = 0.2  # 0.57, because of 55 deg angle of camera w.r.t. z. 
camera_countrange = (4, 20)
_binfactor = 1
#img acquired with 4x4 binning, so 512/4 = 128 bins.
_rescale_x = 1/(128/_binfactor)/angle_correct/1.2

# make long and short axis of 2D gaussian fit align with x and y axis respectively
fix_rotation = False

#integrate_timestep = 1.6e4
camera_h_countrange = (0,370)
camera_h_countrange_vert = (0,400)
pos_range = (-12.5/angle_correct,12.5/angle_correct) # mm

camera_h_countrange_zoom = (0,30)
camera_h_countrange_vert_zoom = (0,100)
pos_range_zoomx = (-7,3) # mm
pos_range_zoomy = (-4,6) # mm

center = (2.6, -2.4)
#center_hexon = (7.5, -1.5)

integration_mask_size = [15, 3] # rectangle, z,x

ticks = [-20, -10, 0, 10, 20]
ticksy = [-10, 0, 10]
ticks_zoom = [-5, 0, 5]
ticks_empty = []
labels = [r"0.0 kV", r"0.5 kV", r"1.0 kV",r"1.5 kV", r"2.0 kV",r"2.5 kV", r"3.0 kV",r"3.5 kV", r"4.0 kV",r"4.5 kV", r"5.0 kV",r"5.5 kV"]
#camera filenames ordered 5 rows by 3 columns: 3 hex voltages vs 5 L00 freqs
camera_filenames = ['experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1771','experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1776','experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1785',\
                    'experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1783','experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1780','experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1782',\
                    'experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1778','experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1786','experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1781',\
                    'experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1779','experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1784','experimental_data/3_camangle-55,0_imageheight-27_Fri Jul 12 2024_1777'
                    ]

#titles = ["$V_{hex} =$ 3.5 kV, \n $\mathcal{B}_{app} =$ 1.5 G"]

_linewidth = 2
_mask_linewidth = 1.5
_cross_linewidth = 0.75
fit_colors = np.array([plt.colormaps['PRGn'](0.9),plt.colormaps['PRGn'](0.1),plt.colormaps['PRGn'](0.8),plt.colormaps['PRGn'](0.15),plt.colormaps['PRGn'](0.0)])

### Make a grid of camera images for 3 hexapole voltages vs 5 L00 frequencies

gridspec_kw = dict(width_ratios=[0.7,0.2,0.3,0.7,0.2], height_ratios=[0.3,1,0.3,0.3,1,0.3,0.3,1,0.3,0.3,1,0.3,0.3,1,0.3,0.3,1])
pmg.create_figure(_subplots = [17,5],_size_inches = (7.66,10), _fontsize=10, _gridspec_kw = gridspec_kw)#, _sharex='col',  _sharey='row'


plt.subplots_adjust(left=0.1, bottom=0.08, right=0.9, top=0.95, hspace=0, wspace=0)

integration_mask = pmg.dataManager.create_camera_mask(camera_filenames[0], _type = 'rectangle', \
                                                      _dimension = [integration_mask_size[0],integration_mask_size[1]] , \
                                                        _unit = "mm", _invert = False)




iter = 0
for idx in range(12):
  jdx = 0
  print('iter = ', iter)
  
  if idx < 6: #plot column on the left: 0.0 - 2.5 kV
    popt, pcov, contour_2D_gauss_fit, unfitted_noise = pmg.add_camera_2D_gauss_contour(camera_filenames[iter], _subplot = [3*idx+1,3*jdx], colors='k',\
      levels=[], linewidths = .75, fix_rotation = fix_rotation) #levels=[2, 4, 6, 8, 10, 12, 14]
  
    # Add camera images
    if fix_rotation:
        tr = transforms.Affine2D().rotate(popt[5])
    else:
        tr = transforms.Affine2D().rotate(0)

    pmg.add_camera_image(camera_filenames[iter], _subplot = [3*idx+1,3*jdx], _vmin = camera_countrange[0], _vmax = camera_countrange[1], _binfactor = _binfactor, transform=tr + pmg.ax[3*idx+1,3*jdx].transData)
    print('shape subplots '+str(3*idx+1)+'  '+str(3*jdx))
    print('##########################################################################################')
    # x, # gauss fit input: t0, A, dt, base_level
    #pmg.dataManager.integrate_camera_data(_filename = camera_filenames[iter], _axis = 0, _overwrite = True)
    #pmg.ax[3*idx,3*jdx+1].plot(pmg.dataManager.data[camera_filenames[iter]]['Camera']['metadata']['horizontal_mm']-center[0], \
    #                    np.array(pmg.dataManager.data[camera_filenames[iter]]['Camera']['integrated_signal'])*pmg.dataManager.data[camera_filenames[iter]]['Camera']['metadata']['aspect']*_rescale_x, \
    #                    linewidth = _linewidth, color = plt.colormaps['PRGn'](0.85)) 
    
    pmg.add_gauss(np.linspace(pos_range[0], pos_range[1], 100), [popt[1], popt[0], popt[3], popt[6]], _subplot = [3*idx,3*jdx], _flip_axes = False, _color='k', linestyle = '--') # t0, A, dt, base_level
    pmg.set_xlim(pos_range, _subplot = [3*idx,3*jdx])
    pmg.set_ylim([0,22], _subplot = [3*idx,3*jdx])
    x_FWHM = '{:.1f}'.format(popt[3]*2.355) #2.355 = prefactor for standard deviation to FWHM
    pmg.add_text(f"w(x) = {x_FWHM} mm", _subplot = [3*idx,3*jdx],_color= 'k', _position= (0, -0.5), _fontsize = 10)
    
    # y
    #pmg.dataManager.integrate_camera_data(_filename = camera_filenames[iter], _axis = 1, _overwrite = True)
    #pmg.ax[3*idx+1,3*jdx+2].plot(pmg.dataManager.data[camera_filenames[iter]]['Camera']['integrated_signal']*_rescale_x, \
    #                    pmg.dataManager.data[camera_filenames[iter]]['Camera']['metadata']['vertical_mm']-center[1],\
    #                    linewidth = _linewidth, color=plt.colormaps['PRGn'](0.15))
    
    pmg.add_gauss(np.linspace(pos_range[0], pos_range[1], 100), [-1*popt[2], popt[0], popt[4], popt[6]], _subplot = [3*idx+1,3*jdx+1], _flip_axes = True, _color='k', linestyle = '--') # t0, A, dt, base_level
    pmg.set_ylim(np.array(pos_range)*angle_correct, _subplot = [3*idx+1,3*jdx+1])
    pmg.set_xlim([0,22], _subplot = [3*idx+1,3*jdx+1])
    y_FWHM = '{:.1f}'.format(popt[4]*2.355) #2.355 = prefactor for standard deviation to FWHM
    pmg.add_text(f"w(y) = {y_FWHM} mm", _subplot = [3*idx+1,3*jdx+1], _color= 'k', _position= (-0.35, 0), _fontsize = 10, _rotation = 270)
      
    # x sum
    pmg.ax[3*idx,3*jdx].tick_params(axis='both',which='both',direction='in', left = False, right=False, top=True, bottom=True, labelleft=False, labelbottom=False)
    
    # camera image
    pmg.ax[3*idx+1,3*jdx].tick_params(axis='both',which='both',direction='out', left = True, right=False, top=False, bottom=True)  
      
    # y sum
    pmg.ax[3*idx+1,3*jdx+1].tick_params(axis='both',which='both',direction='in', left = True, right=True, top=False, bottom=False,labelleft=False, labelbottom=False)
    
  
    # idx is rows, jdx+1 is columns    
    pmg.ax[3*idx,3*jdx+1].set_axis_off()

    
    if jdx != 0:
      pmg.ax[3*idx+1,3*jdx].tick_params(labelleft=False)  
      
    if idx != pmg.ax.shape[0]-1:
      pmg.ax[3*idx+1,3*jdx].tick_params(labelbottom=False)  

    #pmg.ax[1,3*jdx].set_title(titles[jdx], fontsize = 14, position= (0.35, 0.0))  
    
    pmg.set_xlabel('x (mm)', _subplot = [pmg.ax.shape[0]-1,3*jdx])
    
    iter += 1
    pmg.set_ylabel('y (mm)', _subplot = [3*idx+1,0])
    pmg.add_text(labels[idx], _subplot = [3*idx+1,0],_color= 'w', _position= (0.025, 0.05), _fontsize = 10, _rotation = 0)
    
    
  else: #plot column on the right: 3.0 - 5.5 kV
    idx -= 6
    popt, pcov, contour_2D_gauss_fit, unfitted_noise = pmg.add_camera_2D_gauss_contour(camera_filenames[iter], _subplot = [3*idx+1,3*jdx+3], colors='k',\
      levels=[], linewidths = .75, fix_rotation = fix_rotation) #levels=[2, 4, 6, 8, 10, 12, 14]
    
    # Add camera images
    if fix_rotation:
        tr = transforms.Affine2D().rotate(popt[5])
    else:
        tr = transforms.Affine2D().rotate(0)

    # x, # gauss fit input: t0, A, dt, base_level
    #pmg.dataManager.integrate_camera_data(_filename = camera_filenames[iter], _axis = 0, _overwrite = True)
    #pmg.ax[3*idx,3*jdx+1].plot(pmg.dataManager.data[camera_filenames[iter]]['Camera']['metadata']['horizontal_mm']-center[0], \
    #                    np.array(pmg.dataManager.data[camera_filenames[iter]]['Camera']['integrated_signal'])*pmg.dataManager.data[camera_filenames[iter]]['Camera']['metadata']['aspect']*_rescale_x, \
    #                    linewidth = _linewidth, color = plt.colormaps['PRGn'](0.85)) 
    
    pmg.add_gauss(np.linspace(pos_range[0], pos_range[1], 100), [popt[1], popt[0], popt[3], popt[6]], _subplot = [3*idx,3*jdx+3], _flip_axes = False, _color='k', linestyle = '--') # t0, A, dt, base_level
    pmg.set_xlim(pos_range, _subplot = [3*idx,3*jdx+3])
    pmg.set_ylim([0,22], _subplot = [3*idx,3*jdx+3])
    x_FWHM = '{:.1f}'.format(popt[3]*2.355) #2.355 = prefactor for standard deviation to FWHM
    pmg.add_text(f"w(x) = {x_FWHM} mm", _subplot = [3*idx,3*jdx+3],_color= 'k', _position= (0, -0.5), _fontsize = 10)
    
    # y
    #pmg.dataManager.integrate_camera_data(_filename = camera_filenames[iter], _axis = 1, _overwrite = True)
    #pmg.ax[3*idx+1,3*jdx+2].plot(pmg.dataManager.data[camera_filenames[iter]]['Camera']['integrated_signal']*_rescale_x, \
    #                    pmg.dataManager.data[camera_filenames[iter]]['Camera']['metadata']['vertical_mm']-center[1],\
    #                    linewidth = _linewidth, color=plt.colormaps['PRGn'](0.15))
    
    pmg.add_gauss(np.linspace(pos_range[0], pos_range[1], 100), [-1*popt[2], popt[0], popt[4], popt[6]], _subplot = [3*idx+1,3*jdx+4], _flip_axes = True, _color='k', linestyle = '--') # t0, A, dt, base_level
    pmg.set_ylim(np.array(pos_range)*angle_correct, _subplot = [3*idx+1,3*jdx+4])
    pmg.set_xlim([0,22], _subplot = [3*idx+1,3*jdx+4])
    y_FWHM = '{:.1f}'.format(popt[4]*2.355) #2.355 = prefactor for standard deviation to FWHM
    pmg.add_text(f"w(y) = {y_FWHM} mm", _subplot = [3*idx+1,3*jdx+4], _color= 'k', _position= (-0.35, 0), _fontsize = 10, _rotation = 270)
      
    # x sum
    pmg.ax[3*idx,3*jdx+3].tick_params(axis='both',which='both',direction='in', left = False, right=False, top=True, bottom=True, labelleft=False, labelbottom=False)
    # camera image
    pmg.ax[3*idx+1,3*jdx+3].tick_params(axis='both',which='both',direction='out', left = True, right=False, top=False, bottom=True, labelleft=True, labelbottom=False)  
    # y sum
    pmg.ax[3*idx+1,3*jdx+4].tick_params(axis='both',which='both',direction='in', left = True, right=True, top=False, bottom=False, labelleft=False, labelbottom=False)
    
    pmg.add_camera_image(camera_filenames[iter], _subplot = [3*idx+1,3*jdx+3], _vmin = camera_countrange[0], _vmax = camera_countrange[1], _binfactor = _binfactor, transform=tr + pmg.ax[3*idx+1,3*jdx+3].transData)
    pmg.add_text(labels[idx+6], _subplot = [3*idx+1,3*jdx+3],_color= 'w', _position= (0.025, 0.05), _fontsize = 10, _rotation = 0)
    
    pmg.ax[3*idx,3*jdx+4].set_axis_off()
    
    if iter == len(labels)-1:   
      pmg.set_xlabel('x (mm)', _subplot = [pmg.ax.shape[0]-1,3*jdx+3]) 
      pmg.ax[3*idx+1,3*jdx+3].tick_params(labelbottom=True)
      
      for i in range(pmg.ax.shape[0]): #iterate over all rows, 14 total
        pmg.ax[i,2].set_axis_off()
        
        
      for i in range(pmg.ax.shape[1]): # iterate over all columns, 8 total
        pmg.ax[2,i].set_axis_off()
        pmg.ax[5,i].set_axis_off()
        pmg.ax[8,i].set_axis_off()
        pmg.ax[11,i].set_axis_off()
        pmg.ax[14,i].set_axis_off()  
        
    if jdx != 0:
      pmg.ax[3*idx+1,3*jdx+3].tick_params(labelleft=False)  
      
    if 3*idx+1 == pmg.ax.shape[0]-1:
      pmg.ax[3*idx+1,3*jdx+3].tick_params(labelbottom=True)
    
        
    iter += 1
    

    

  
   
    
pmg.add_colorbar(_label = 'Intensity (arb. units)')
pmg.save_figure(os.path.basename(__file__)[:-3]+"_v2")
pmg.flush()


'''### Now compare the horizontal component of the 2D gaussian fit of selected camera images

labels_subset = [r"$\Delta = $-6 MHz", r"$\Delta = $-2 MHz", r"$\Delta = $ +2 MHz",\
          r"$\Delta = $+6 MHz", r"$\Delta = $+10 MHz", r"No LCx"]
#camera filenames ordered 5 rows by 3 columns: 3 hex voltages vs 5 L00 freqs
camera_filenames_subset = ['3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1690','3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1692','3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1689',\
                    '3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1693','3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1691','3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1694']

titles = ['(a)', '(b)', 'c', 'd','e','f']
linestyles = ['-','-','-','-','-','--']
colors_subset = np.array([plt.colormaps['sunset'](0.95),plt.colormaps['sunset'](0.85), plt.colormaps['sunset'](0.65), \
                          plt.colormaps['sunset'](0.25),plt.colormaps['sunset'](0.05),plt.colormaps['PRGn'](0.05)])


pos_range = (-12.5/0.57,12.5/0.57) # mm 
popt_vals = np.zeros([len(camera_filenames_subset), 7])
pcov_vals = np.zeros([len(camera_filenames_subset), 7])
print(popt_vals)
pmg.create_figure(_subplots = [1,1],_size_inches = (10, 6), _fontsize=20)

iter = 0
for idx,label in enumerate(labels_subset):
  
  # x sum
  popt, pcov, contour_2D_gauss_fit, unfitted_noise = pmg.add_camera_2D_gauss_contour(camera_filenames_subset[idx], _subplot = [0,0], colors='k',\
        levels=[], linewidths = .75, fix_rotation = fix_rotation) #levels=[2, 4, 6, 8, 10, 12, 14]
  
  #for vi,v in enumerate(np.array(popt)):
  #  popt_vals[idx,vi] = popt[vi]
  #  #pcov_vals[idx,vi] = pcov[vi]
  
  
  pmg.add_gauss(np.linspace(pos_range[0], pos_range[1], 100), [popt[1], popt[0], popt[3], popt[6]], _subplot = [0,0], _flip_axes = False, \
    _color=colors_subset[idx], linestyle = linestyles[idx], linewidth = 2) # t0, A, dt, base_level

pmg.ax[0,0].set_xlabel('x (mm)', fontsize=20)
pmg.ax[0,0].set_xlim(pos_range)
pmg.ax[0,0].set_ylim(1,13.5)
pmg.ax[0,0].set_ylabel('Intensity (arb. units)', fontsize=20)
pmg.ax[0,0].legend(labels = labels_subset, fontsize=20, labelcolor=colors_subset, frameon = False)

pmg.save_figure(os.path.basename(__file__)[:-3]+"_fitonly_v1")

pmg.flush()
'''

'''# Plot horizontal histograms together
labels_subset = [r"$\Delta = $-6 MHz", r"$\Delta = $-2 MHz", r"$\Delta = $ +2 MHz",\
          r"$\Delta = $+6 MHz", r"$\Delta = $+10 MHz", r"No LCx, $\Delta= $ +2 MHz"]

camera_filenames_subset = ['3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1690','3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1692','3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1689',\
                    '3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1693','3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1691','3_camangle-55,0_imageheight-27_Tue Apr 30 2024_1694']

titles = ['(a)', '(b)', 'c', 'd','e','f']
linestyles = ['-','-','-','-','-','--']
colors_subset = np.array([plt.colormaps['sunset'](0.95),plt.colormaps['PRGn'](0.9), plt.colormaps['sunset'](0.7), \
                          plt.colormaps['sunset'](0.25),plt.colormaps['sunset'](0.05),plt.colormaps['PRGn'](0.1)])


pos_range = (-12.5/0.57,12.5/0.57) # mm 
popt_vals = np.zeros([len(camera_filenames_subset), 7])
pcov_vals = np.zeros([len(camera_filenames_subset), 7])
print(popt_vals)
pmg.create_figure(_subplots = [1,1],_size_inches = (10, 6), _fontsize=20)

iter = 0
for idx,label in enumerate(labels_subset):
  
  # x sum
  popt, pcov, contour_2D_gauss_fit, unfitted_noise = pmg.add_camera_2D_gauss_contour(camera_filenames_subset[idx], _subplot = [0,0], colors='k',\
        levels=[], linewidths = .75, fix_rotation = fix_rotation) #levels=[2, 4, 6, 8, 10, 12, 14]
  
  #for vi,v in enumerate(np.array(popt)):
  #  popt_vals[idx,vi] = popt[vi]
  #  #pcov_vals[idx,vi] = pcov[vi]
  
  pmg.dataManager.integrate_camera_data(_filename = camera_filenames[iter], _axis = 0, _overwrite = True)
  pmg.ax[0,0].plot(pmg.dataManager.data[camera_filenames[iter]]['Camera']['metadata']['horizontal_mm']-center[0], \
                      np.array(pmg.dataManager.data[camera_filenames[iter]]['Camera']['integrated_signal'])*pmg.dataManager.data[camera_filenames[iter]]['Camera']['metadata']['aspect']*_rescale_x, \
                      linewidth = _linewidth, color=colors_subset[idx]) 
  iter +=1
  #pmg.add_gauss(np.linspace(pos_range[0], pos_range[1], 100), [popt[1], popt[0], popt[3], popt[6]], _subplot = [0,0], _flip_axes = False, \
  #  _color=colors_subset[idx], linestyle = linestyles[idx], linewidth = 2) # t0, A, dt, base_level

pmg.ax[0,0].set_xlabel('x (mm)', fontsize=20)
pmg.ax[0,0].set_xlim(pos_range)
pmg.ax[0,0].set_ylim(0,15)
pmg.ax[0,0].set_ylabel('Intensity (arb. units)', fontsize=20)
pmg.ax[0,0].legend(labels = labels_subset, fontsize=14, labelcolor=colors_subset, frameon = False)

pmg.save_figure(os.path.basename(__file__)[:-3]+"_xhist_v2")

pmg.flush()
'''

