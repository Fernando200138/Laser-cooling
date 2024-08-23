import matplotlib.pyplot as plt
import numpy as np
from DataManager import *
import pickle
import os
from pint import UnitRegistry
ur = UnitRegistry()
ur.define("micro- = 1e-6 = µ- = u-")
import csv
from tol_colors import *
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D
from matplotlib import rc, rcParams, transforms

#plt.rc('axes', prop_cycle=plt.cycler('color', list(tol_cset('muted'))))
plt.cm.register_cmap('sunset', tol_cmap('sunset'))
#plt.rc('image', cmap='sunset')
from math import floor

def t2v(x, _distance):
    # 1/x with special treatment of x == -1e-6 (shifted all by 1e-6 to avoid the pole at zero for t = 0)
    x = np.array(x).astype(float)
    near_zero = np.isclose(x, -1e-6)
    x[near_zero] = np.inf
    x[~near_zero] = _distance / (x[~near_zero]+1e-6)
    return x

class PlotManager():
    def __init__(self, _verbose = False):
        self.dataManager = DataManager(_verbose)
        self.latest_filename = "No_latest_filename"
        self.size_inches = (10,7.5)
        self.set_fontsize(10)
        self.verbose = _verbose
        self.overlay_cindex = None
        rc('font',**{'size'   : self.fontsize})

    #
    # Create the figure and axes, with _size_inches the size of the figure (inch)
    #
    # _sharex and _sharey determine sharing of axes
    #   True or 'all': x- or y-axis will be shared among all subplots.
    #   False or 'none': each subplot x- or y-axis will be independent.
    #   'row': each subplot row will share an x- or y-axis.
    #   'col': each subplot column will share an x- or y-axis.


    def create_figure(self, _size_inches = (4.5, 4.5), _fontsize=None, _subplots = (1,1), _sharex = 'none', _sharey = 'none', _gridspec_kw=None):
        self.fig, self.ax = plt.subplots(_subplots[0], _subplots[1], squeeze = False, sharex=_sharex, sharey=_sharey, gridspec_kw=_gridspec_kw) 
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9,wspace=0.4, hspace=0.4)
        self.size_inches = _size_inches
        self.fig.set_size_inches(self.size_inches)
        self.set_fontsize(_fontsize)
        self.subplots = _subplots
        for i in range(_subplots[0]):
            for j in range(_subplots[1]):
                self.ax[i,j].tick_params(gridOn=False, direction='in', labelsize=self.fontsize) 
                self.ax[i,j].xaxis.get_offset_text().set_fontsize(_fontsize)
                self.ax[i,j].yaxis.get_offset_text().set_fontsize(_fontsize)
        if self.verbose:
            print("create_figure: created figure.")
            
    def set_fontsize(self, _fontsize = None):
        if _fontsize != None:
            self.fontsize = _fontsize
            rc('font',**{'size'   : self.fontsize})

    #
    # Set the figure and axes to the predefined objects
    #
    def set_figure(self, _fig, _ax):
        self.fig, self.ax = _fig, _ax
        for axi, axx in enumerate(self.ax):
            for axj, axs in enumerate(axx):
                if isinstance(axs, Axes):
                    axs.tick_params(gridOn=False, direction='in', labelsize=self.fontsize) 
                    axs.xaxis.get_offset_text().set_fontsize(self.fontsize)
                    axs.yaxis.get_offset_text().set_fontsize(self.fontsize)
        if self.verbose:
            print("set_figure: set figure to given objects.")
        self.subplots_adjust()
    
    #
    # Subplots_adjust to better defaults
    # 
    def subplots_adjust(self, _left=0.15, _bottom=0.15, _right=0.9, _top=0.9,_wspace=0.4, _hspace=0.4):
        plt.subplots_adjust(left=_left, bottom=_bottom, right=_right, top=_top,wspace=_wspace, hspace=_hspace)

    #
    # Create standard time of flight figure
    #
    def create_standard_figure(self):
        self.create_figure()
        self.set_xlabel("Time (ms)")
        self.set_ylabel("Fluorescence (photon counts/shot)")

    # 
    # Get dimensions of _subplot. Returns left, top, width, height
    #
    def get_subplot_dimensions(self, _subplot = [0,0]):
        pos = self.ax[_subplot[0], _subplot[1]].get_position().get_points().flatten()
        top = max(pos[1], pos[3])
        left = min(pos[0], pos[2])
        width = abs(pos[0] - pos[2])
        height = abs(pos[1] - pos[3])
        return left, top, width, height

    #
    # Add overlay to subplot _subplot. The overlay subplots are added to a column of self.ax, the cooridinate of the subplot are returned
    # _left, _top, _width and _height are fractions relative to the hosting _subplot. Can be excended beyond the _subplot with values above one or below zero. Note these might not rescale with adjustment to the figure.
    #
    def add_overlay_subplot(self, _left = 0.1, _top = 0.1, _width = 0.4, _height = 0.4, _subplot = [0,0], _show_zoombox = False, _zoombox_color = 'k', _zoombox_x = [0,1], _zoombox_y = [0,1], **kwargs):
        left, top, width, height = self.get_subplot_dimensions(_subplot = _subplot)
        rect = (left + _left*width, top - (_top+_height)*height, _width*width, _height*height)       
        ax = self.fig.add_axes(rect, zorder = 2) #, facecolor = '#FFFFFF00'
        if self.overlay_cindex == None or not any([self.ax[j,self.overlay_cindex] == None for j in range(self.ax.shape[0])]):
            self.overlay_cindex = self.ax.shape[1]
            self.ax = np.append(self.ax, np.array([[None]]*self.ax.shape[0]), 1)
            self.ax[0, self.overlay_cindex] = ax
            j = 0
        else:
            for j in range(self.ax.shape[0]):
                if self.ax[j, self.overlay_cindex] == None:
                    break
            self.ax[j, self.overlay_cindex] = ax
        if _show_zoombox:
            axLeftTop = self.fig.transFigure.inverted().transform(ax.transAxes.transform([0,1]))
            axRightBottom = self.fig.transFigure.inverted().transform(ax.transAxes.transform([1,0]))
            
            zoomLeftTop = self.fig.transFigure.inverted().transform(self.ax[_subplot[0], _subplot[1]].transData.transform([_zoombox_x[0], _zoombox_y[1]]))
            zoomRightBottom = self.fig.transFigure.inverted().transform(self.ax[_subplot[0], _subplot[1]].transData.transform([_zoombox_x[1], _zoombox_y[0]]))
          
            self.fig.add_artist(Line2D([axLeftTop[0], axRightBottom[0]], [axLeftTop[1], axLeftTop[1]], transform = self.fig.transFigure, color = _zoombox_color, zorder=3, **kwargs))
            self.fig.add_artist(Line2D([axLeftTop[0], axRightBottom[0]], [axRightBottom[1], axRightBottom[1]], transform = self.fig.transFigure, color = _zoombox_color, zorder=3, **kwargs))
            self.fig.add_artist(Line2D([axLeftTop[0], axLeftTop[0]], [axLeftTop[1], axRightBottom[1]], transform = self.fig.transFigure, color = _zoombox_color, zorder=3, **kwargs))
            self.fig.add_artist(Line2D([axRightBottom[0], axRightBottom[0]], [axLeftTop[1], axRightBottom[1]], transform = self.fig.transFigure, color = _zoombox_color, zorder=3, **kwargs))

            self.ax[_subplot[0], _subplot[1]].add_artist(Line2D([_zoombox_x[0], _zoombox_x[1]], [_zoombox_y[0], _zoombox_y[0]], transform = self.ax[_subplot[0], _subplot[1]].transData, color = _zoombox_color, zorder=1, **kwargs))
            self.ax[_subplot[0], _subplot[1]].add_artist(Line2D([_zoombox_x[0], _zoombox_x[1]], [_zoombox_y[1], _zoombox_y[1]], transform = self.ax[_subplot[0], _subplot[1]].transData, color = _zoombox_color, zorder=1, **kwargs))
            self.ax[_subplot[0], _subplot[1]].add_artist(Line2D([_zoombox_x[0], _zoombox_x[0]], [_zoombox_y[0], _zoombox_y[1]], transform = self.ax[_subplot[0], _subplot[1]].transData, color = _zoombox_color, zorder=1, **kwargs))
            self.ax[_subplot[0], _subplot[1]].add_artist(Line2D([_zoombox_x[1], _zoombox_x[1]], [_zoombox_y[0], _zoombox_y[1]], transform = self.ax[_subplot[0], _subplot[1]].transData, color = _zoombox_color, zorder=1, **kwargs))

            '''
            ax.add_artist(Line2D([1,1], [0,1], transform = ax.transData, color = _zoombox_color), **kwargs)
            ax.add_artist(Line2D([0,1], [0,0], transform = ax.transData, color = _zoombox_color), **kwargs)
            ax.add_artist(Line2D([0,1], [1,1], transform = ax.transData, color = _zoombox_color), **kwargs)
            '''

        return [j, self.overlay_cindex]

    #
    # Save the figure to _filename, as png, pdf and as matplotlib object, if * is in _filename it is replaced by the latest data filename
    #
    def save_figure(self, _filename = "*", _directory = '', _clear = False, **kwargs):
        _filename = _filename.replace("*", self.latest_filename)
        _filename = _filename.replace("/", "_")
        _filename = os.path.join(_directory, _filename)
        self.fig.savefig(_filename+'.png', **kwargs)#,bbox_inches='tight')
        self.fig.savefig(_filename+'.pdf', **kwargs)#,bbox_inches='tight')
        try:
            pickle.dump(self.fig, open(_filename+'.pyfig', 'wb'))
        except:
            pass
        if _clear:
            self.clear()
        if self.verbose:
            print("save_figure: saved figure as " + str(_filename) + ".")
            
    #
    # Load figure from .pyfig file as matplotlib object
    #
    def load_figure(self, _filename, _directory = ''):
        _filename = os.path.join(_directory, _filename)
        self.fig = pickle.load(open(_filename, 'rb'))
        if self.verbose:
            print("load_figure: loaded figure " + str(_filename) + ".")
            
    #
    # Show the figure
    #
    def flush(self, _clear=False):
        plt.show()
        if _clear:
            self.clear()

    #
    # Clear figure
    #
    def clear(self):
        plt.close()
        plt.cla()
        plt.clf()
        self.overlay_cindex = None

        #self.create_figure(self.size_inches, self.fontsize)

    #
    # Set axis labels
    #
    def set_xlabel(self, _label, _subplot = [0,0], _position='bottom'):
        self.ax[_subplot[0], _subplot[1]].set_xlabel(_label, fontsize=self.fontsize)
        self.ax[_subplot[0], _subplot[1]].xaxis.set_label_position(_position)
        self.ax[_subplot[0], _subplot[1]].xaxis.set_ticks_position(_position)
        

    def set_ylabel(self, _label, _subplot = [0,0], _position='left'):
        self.ax[_subplot[0], _subplot[1]].set_ylabel(_label, fontsize=self.fontsize)
        self.ax[_subplot[0], _subplot[1]].yaxis.set_label_position(_position)
        self.ax[_subplot[0], _subplot[1]].yaxis.set_ticks_position(_position)


    def set_zlabel(self, _label, _subplot = [0,0]):
        self.ax[_subplot[0], _subplot[1]].set_zlabel(_label, fontsize=self.fontsize)

    #
    # Set axis ranges
    #
    def set_xlim(self, _lim, _subplot = [0,0]):
        self.ax[_subplot[0], _subplot[1]].set_xlim(_lim)

    def set_ylim(self, _lim, _subplot = [0,0]):
        self.ax[_subplot[0], _subplot[1]].set_ylim(_lim)

    def set_zlim(self, _lim, _subplot = [0,0]):
        self.ax[_subplot[0], _subplot[1]].set_zlim(_lim)

    #
    # Add legend
    #
    def legend(self, _subplot = [0,0], _frameon = False):
        self.ax[_subplot[0], _subplot[1]].legend(fontsize = self.fontsize, frameon = _frameon)


    #
    # Add _text to the _subplot, in _color at fractional _positon relative to the top-right corner
    # Add _text to the _subplot, in _color at fractional _positon relative to the top-right corner
    #
    def add_text(self, _text, _subplot = [0,0], _color = 'k', _position = (0.05,0.05), _fontsize = 14, _rotation = 0, **kwargs):
        self.ax[_subplot[0], _subplot[1]].text(1-_position[0], 1-_position[1], _text, transform=self.ax[_subplot[0], _subplot[1]].transAxes, color = _color, fontsize=_fontsize, rotation = _rotation, horizontalalignment='right', verticalalignment='top', **kwargs)

    #
    # Add time of flight histogram to figure
    # Note _binSize and _range are in ns
    # To add a velocity axis, assuming instantaneous emmision at ablation trigger, set _add_velocity_axis to True and specify the detection point _distance in mm. Also provide the _velocity_axislabel and _velocity_ticks if required
    # 
    def add_TOF(self, _filename, _binSize, _range = None, _channel = 'A', _background=False, _rescale=1, _offset=0, _label=None, _color=None, _subplot = [0,0], _export_data_filename='', _add_velocity_axis = False, _distance = 5268, _velocity_axislabel = 'Velocity (m/s)', _velocity_ticks = None, **kwargs):
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        if _filename not in self.dataManager.data or "Peak"+str(_channel) not in self.dataManager.data[_filename]:                           # Read missing data
            self.dataManager.read_peakdata(_filename)

        self.dataManager.bin_timestamps(_filename, _binSize, _channels = [_channel], _normalise = True)

        bins = []
        counts = []
        if _background:
            channel = _channel
        else:
            channel = _channel+'nb'
            self.dataManager.subtract_background(_filename, _binSize, _channels = [_channel])
        for k in range(len(self.dataManager.data[_filename]['Binned'][_binSize]['Bins'])-1):
            if isinstance(_range, type(None)) or _range[0] < self.dataManager.data[_filename]['Binned'][_binSize]['Bins'][k] < _range[1]:
                bins.append(self.dataManager.data[_filename]['Binned'][_binSize]['Bins'][k]*1e-6)                                # Bin edge in ms
                counts.append(self.dataManager.data[_filename]['Binned'][_binSize][channel][k] + _offset)                        # counts + _offset

        plot = self.ax[_subplot[0], _subplot[1]].plot(bins, np.array(counts)*_rescale, drawstyle = 'steps-mid', label = _label, color = _color, **kwargs)
        
        if _add_velocity_axis:
            secax = self.ax[_subplot[0], _subplot[1]].secondary_xaxis('top', functions=(lambda x: t2v(x, _distance), lambda x: t2v(x, _distance)))
            secax.set_xlabel(_velocity_axislabel)
            if _velocity_ticks != None:
                secax.xaxis.set_ticks(_velocity_ticks)
                if len(_velocity_ticks) > 3:
                    _velocity_ticks[::2] = ['' for ct in _velocity_ticks[::2]]
                secax.xaxis.set_ticklabels(_velocity_ticks)

        # print("add_TOF: total counts per shot   ", np.sum(counts))
        if _export_data_filename != '':                                                                                         # Export data to file
            if _export_data_filename == None:
                _export_data_filename = _filename+"_TOF_binsize_"+str(_binSize)+"ns"

            file = open(_export_data_filename+"_x", "w")
            for i in bins:
                file.write(str(i) + "\n")
            file.close()
            
            file = open(_export_data_filename+"_y", "w")
            for i in counts:
                file.write(str(i) + "\n")
            file.close()
            if self.verbose:
                print("add_TOF: exported Time of Flight data with binned data from " + str(_filename) + " with binsize " + str(_binSize) + " ns to " + str(_export_data_filename) + "+_x/y.")
                      
        self.latest_filename = _filename
        if self.verbose:
            print("add_TOF: added Time of Flight to subplot " + str(_subplot) + " with binned data from " + str(_filename) + " with binsize " + str(_binSize) + " ns.")
        return plot

    #
    # Add scope vs time plot to figure
    # If _shots = None, plot the mean of the shots, if _shots = [] plot all shots, otherwise plot the selected _shots
    #
    def add_scope(self, _filename, _channels = ['B'], _shots = None, _range = None, _rescale = 1, _offset=0, _x_rescale = 1, _x_offset = 0, _label=None, _color=None, _subplot = [0,0], **kwargs):     
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        
        if _filename not in self.dataManager.data or "Scopedata" not in self.dataManager.data[_filename]:                       # Read missing data
            self.dataManager.read_scopedata(_filename)
        
        if "Metadata" not in self.dataManager.data[_filename]:
            self.dataManager.read_metadata(_filename)
            
        if _shots == None:                                                                                                      # Plot mean
            _channels = [c+"_mean" for c in _channels]
            
            time = []
            scope = [[]*len(_channels)]
        
            for k in range(len(self.dataManager.data[_filename]['Scopedata']["Time"])):
                if _range == None or _range[0] < _x_rescale*self.dataManager.data[_filename]['Scopedata']["Time"][k]+_x_offset < _range[1]:
                    time.append(_x_rescale*self.dataManager.data[_filename]['Scopedata']["Time"][k]+_x_offset)
                    for c in range(len(_channels)):
                        scope[c].append(self.dataManager.data[_filename]['Scopedata'][_channels[c]][k]*_rescale+_offset)

            for c in range(len(_channels)):
                self.ax[_subplot[0], _subplot[1]].plot(time, scope[c], label = _label, color = _color, **kwargs)
            self.latest_filename = _filename
            if self.verbose:
                print("add_scope: added mean scope plot to subplot " + str(_subplot) + " with data from " + str(_filename) + " with channels " + str(_channels) + ".")
            return    
            
            
        elif _shots == []:                                                                                                       # Plot all shots
            _shots = range(self.dataManager.data[_filename]['Metadata']['Analyse']['Scanpoints'])
        
        for ci, c in enumerate(_channels):
            if self.dataManager.data[_filename]['Scopedata'][c] == []:
                self.dataManager.read_scopedata(_filename, _save_each_shot = True, _overwrite=True)
        
        for s in _shots:                                                                                                        # Plot selected shots
            time = []
            scope = [[]*len(_channels)]
        
            for k in range(len(self.dataManager.data[_filename]['Scopedata']["Time"])):
                if _range == None or _range[0] < _x_rescale*self.dataManager.data[_filename]['Scopedata']["Time"][k]+_x_offset < _range[1]:
                    time.append(_x_rescale*self.dataManager.data[_filename]['Scopedata']["Time"][k]+_x_offset)
                    for ci, c in enumerate(_channels):
                        scope[ci].append(self.dataManager.data[_filename]['Scopedata'][c][s][k]*_rescale+_offset)

            for c in range(len(_channels)):
                self.ax[_subplot[0], _subplot[1]].plot(time, scope[c], label = _label, color = _color, **kwargs)

        self.latest_filename = _filename
        if self.verbose:
            print("add_scope: added scope plot for shots " +str(_shots)+" to subplot " + str(_subplot) + " with data from " + str(_filename) + " with channels " + str(_channels) + ".")

    '''
    #
    # OLD, overwritten by next
    # Add mean absorption vs time plot to figure
    # Absorption is calculated with a background level as the average over a region _background in ms. 
    #
    def add_absorption(self, _filename, _channel = "B_mean", _range = None, _background = (2,10), _rescale = 1, _offset=0, _label=None, _color=None, _subplot = [0,0], _export_data_filename=''):     
        print("add_absorption: WARNING: this a method is no longer in use and has been replaced. If this message shows up something went wrong!")
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        if _filename not in self.dataManager.data or "Scopedata" not in self.dataManager.data[_filename] or len(self.dataManager.data[_filename]['Scopedata'][channel]) == 0:                                                     # Read missing data
            self.read_scopedata(_filename, _save_each_shot = True, _overwrite=True)  
        
        time = []
        scope = []
        
        background = 0
        background_counter = 0
        for ti, t in enumerate(self.dataManager.data[_filename]['Scopedata']["Time"]):
            if _background[0] < t < _background[1]:
                background += self.dataManager.data[_filename]['Scopedata'][_channel][ti]
                background_counter += 1
        background /= background_counter

    
        for k in range(len(self.dataManager.data[_filename]['Scopedata']["Time"])):
            if _range == None or _range[0] < self.dataManager.data[_filename]['Scopedata']["Time"][k] < _range[1]:
                time.append(self.dataManager.data[_filename]['Scopedata']["Time"][k])
                scope.append((self.dataManager.data[_filename]['Scopedata'][_channel][k]-background)/background*100*_rescale+_offset)

        self.ax[_subplot[0], _subplot[1]].plot(time, scope, label = _label, color = _color)
        
        if _export_data_filename != '':                                                                                         # Export data to file
            if _export_data_filename == None:
                _export_data_filename = _filename+"_abs"

            file = open(_export_data_filename+"_x", "w")
            for i in time:
                file.write(str(i) + "\n")
            file.close()
            
            file = open(_export_data_filename+"_y", "w")
            for i in scope:
                file.write(str(i) + "\n")
            file.close()
        self.latest_filename = _filename
        if self.verbose:
            print("add_absorption: added absorption plot to subplot " + str(_subplot) + " with data from " + str(_filename) + ".")
    '''
    
    #
    # Add mean absorption vs time plot to figure
    # Absorption is calculated with a background level as the average over a region _background in ms. 
    # Take the average over _average consecutive samples
    #
    def add_absorption(self, _filename, _channel = "B", _range = None, _background = (2,10), _rescale = 1, _offset=0, _label=None, _color=None, _average = 1, _subplot = [0,0], _export_data_filename='',  _overwrite = True, **kwargs):     
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        
        self.dataManager.calculate_absorption(_filename, _range, _background, [_channel.replace("_mean", "")], _overwrite = _overwrite)

        self.ax[_subplot[0], _subplot[1]].plot(self.dataManager.average_consecutive(self.dataManager.data[_filename]["Absorption"][_channel+"_time"], _average), self.dataManager.average_consecutive(_rescale*self.dataManager.data[_filename]["Absorption"][_channel+"_percentage_mean"]+_offset, _average), label = _label, color = _color, **kwargs)
        
        if _export_data_filename != '':                                                                                         # Export data to file
            if _export_data_filename == None:
                _export_data_filename = _filename+"_abs"

            file = open(_export_data_filename+"_x", "w")
            for i in self.dataManager.data[_filename]["Absorption"][_channel+"_time"]:
                file.write(str(i) + "\n")
            file.close()
            
            file = open(_export_data_filename+"_y", "w")
            for i in self.dataManager.data[_filename]["Absorption"][_channel+"_percentage_mean"]:
                file.write(str(i) + "\n")
            file.close()
        self.latest_filename = _filename
        if self.verbose:
            print("add_absorption: added absorption plot to subplot " + str(_subplot) + " with data from " + str(_filename) + ".")

    #
    # Add scatter to figure based on scandata
    #
    def add_scatter_scan(self, _filename, _x, _y, _range = None, _rescale = 1, _offset = 0, _x_rescale = 1, _x_offset = 0, _label=None, _color=None, _pointsize=3 ,_alpha=0.2, _subplot = [0,0], **kwargs):           
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        
        if _filename not in self.dataManager.data:                                                                              # Read missing data
            self.dataManager.read_metadata(_filename)
            self.dataManager.read_scandata(_filename)
        else:
            if "Metadata" not in self.dataManager.data[_filename]:
                self.dataManager.read_metadata(_filename)
            if "Scandata" not in self.dataManager.data[_filename]:
                self.dataManager.read_scandata(_filename)
       
        calculators = self.dataManager.data[_filename]["Metadata"]["Analyse"]["Calculators"]      
        if _x not in self.dataManager.data[_filename]['Scandata']:                                                              # Replace py calculator index if Calculator
            for c in calculators:
                if calculators[c]['Name'] == _x:
                    _x = c
            
        if _y not in self.dataManager.data[_filename]['Scandata']:
            for c in calculators:
                if calculators[c]['Name'] == _y:
                    _y = c
        
        '''
        x = []
        y = []
        
        for i in range(len(self.dataManager.data[_filename]['Scandata'][_x])):                                                  # Read data in correct range and rescaling, offset
            if _range == None or _range[0] < self.dataManager.data[_filename]['Scandata'][_x][i] < _range[1]:
                x.append(_x_rescale*self.dataManager.data[_filename]['Scandata'][_x][i] + _x_offset)
                y.append(_rescale*self.dataManager.data[_filename]['Scandata'][_y][i] + _offset)            

        self.ax[_subplot[0], _subplot[1]].scatter(x, y, s=_pointsize, alpha=_alpha, label = _label, color = _color)
        '''

        self.add_scatter(self.dataManager.data[_filename]['Scandata'][_x], self.dataManager.data[_filename]['Scandata'][_y], _range, _rescale, _offset, _x_rescale, _x_offset, _label, _color, _pointsize, _alpha, _subplot, **kwargs)

        self.latest_filename = _filename

    #
    # Add scatter to figure of _x and _y array
    #
    def add_scatter(self, _x, _y, _range = None, _rescale = 1, _offset = 0, _x_rescale = 1, _x_offset = 0, _label=None, _color=None, _pointsize=3 ,_alpha=0.2, _subplot = [0,0], **kwargs):             
        x = []
        y = []
        
        for xi, xe in enumerate(_x):                                                  # Read data in correct range and rescaling, offset
            if _range == None or _range[0] < xe < _range[1]:
                x.append(_x_rescale*xe + _x_offset)
                y.append(_rescale*_y[xi] + _offset)            

        self.ax[_subplot[0], _subplot[1]].scatter(x, y, s=_pointsize, alpha=_alpha, label = _label, color = _color, **kwargs)

    #
    # Add profile to figure (Binned data). Take scandata from _filename (or index), bin data in _x_scanlabel over _nBins in the _range. Calculate mean and standarddeviation in _scanlabel.
    #
    def add_profile_scan(self, _filename, _x_scanlabel = "Frequency Castor", _scanlabel = "LIF", _nBins = None, _range = None, _rescale = 1, _offset = 0, _x_rescale = 1, _x_offset = 0, _label=None, _color=None, _linestyle = 'None', _marker = ',', _subplot = [0,0], **kwargs):      
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
            
        if _filename not in self.dataManager.data:
            self.dataManager.data[_filename] = {}
            
        if "Scandata" not in self.dataManager.data[_filename]:
            self.dataManager.read_scandata(_filename)

        if "Metadata" not in self.dataManager.data[_filename]:
            self.dataManager.read_metadata(_filename)
            
        # Check whether the _scanlabel is a calculator 
        calculators = self.dataManager.data[_filename]["Metadata"]["Analyse"]["Calculators"]      
        if _scanlabel not in self.dataManager.data[_filename]['Scandata']:
            for c in calculators:
                if calculators[c]['Name'] == _scanlabel:
                    _scanlabel = c
                if calculators[c]['Name'] == _x_scanlabel:
                    _x_scanlabel = c
                    
        self.dataManager.bin_data_scan(_filenames = [_filename], _x_scanlabel = _x_scanlabel, _scanlabels = [_scanlabel], _nBins = _nBins, _range = _range, _overwrite = False)     

        self.ax[_subplot[0], _subplot[1]].errorbar(x=_x_rescale*self.dataManager.data[_filename]['ScandataBinned'][_x_scanlabel][0] + _x_offset, xerr=_x_rescale*self.dataManager.data[_filename]['ScandataBinned'][_x_scanlabel][1]/2, y=_rescale*self.dataManager.data[_filename]['ScandataBinned'][_scanlabel][0]+_offset, yerr=_rescale*self.dataManager.data[_filename]['ScandataBinned'][_scanlabel][1], linestyle=_linestyle, marker=_marker , label = _label, color = _color, **kwargs)
        self.latest_filename = _filename

    #
    # Add profile to figure (Binned data). Bin data in _x over _nBins in the _range. Calculate mean and standarddeviation/sqrt(N_i) in _y.
    #
    def add_profile(self, _x, _y, _nBins = None, _range = None, _rescale = 1, _offset = 0, _x_rescale = 1, _x_offset = 0, _label=None, _color=None, _linestyle = 'None', _elinewidth = None, _marker = ',', _subplot = [0,0], **kwargs):
        y, yerr, x, xerr = self.dataManager.bin_data([_x_rescale*_xi + _x_offset for _xi in _x], [_rescale*_yi+_offset for _yi in _y], _nBins = _nBins, _range = _range)     
        self.ax[_subplot[0], _subplot[1]].errorbar(x=x, xerr=xerr/2, y=y, yerr=yerr, linestyle=_linestyle, marker=_marker , label = _label, color = _color, elinewidth = _elinewidth, **kwargs)
        return y, yerr, x, xerr
        
    def add_scandata_fit(self, _filename, _x_scanlabel = "Frequency Castor", _scanlabel = "LIF", _nBins = 1e3, _alpha = [10], _gamma = [10], _I = [1e2], _x0 = [348e6], _show_statistics = True, _rescale = 1, _offset = 0, _x_rescale = 1, _x_offset = 0, _label=None, _color=None, _linestyle = '-', _subplot = [0,0], **kwargs):
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
              
        self.dataManager.fit_voigt(_filenames = [_filename], _x_scanlabel = _x_scanlabel, _nBins = _nBins, _alpha = _alpha, _gamma = _gamma, _I = _I, _x0 = _x0)

        # Check whether the _x_scanlabel or _scanlabel is a calculator 
        if _x_scanlabel not in self.dataManager.data[_filename]['Scandata'] or _scanlabel not in self.dataManager.data[_filename]['Scandata']:                                                              # Replace py calculator index if Calculator
            calculators = self.dataManager.data[_filename]["Metadata"]["Analyse"]["Calculators"]
            for c in calculators:
                if calculators[c]['Name'] == _x_scanlabel:
                    _x_scanlabel = c
                if calculators[c]['Name'] == _scanlabel:
                    _scanlabel = c

        self.ax[_subplot[0], _subplot[1]].plot(_x_rescale*self.dataManager.data[_filename]['ScandataFitted'][_x_scanlabel] + _x_offset, _rescale*self.dataManager.data[_filename]['ScandataFitted'][_scanlabel]+_offset, linestyle=_linestyle, label = _label, color = _color, **kwargs)
        self.latest_filename = _filename    

    #
    # Add voltages in _waveform of _channels to figure. If * in _label it will be replaced by _waveform.
    #
    def add_waveform(self, _waveform, _channels = [4], _range = None,_offset=0, _label=None, _color=None, _rescale=1, _subplot = [0,0], **kwargs):
        if _label != None:
            _label = _label.replace('*', _waveform)                                                                                 # Replace * by _waveform in _label
        
        if _waveform not in self.dataManager.waveforms:                                                                         # Read missing data
            self.dataManager.read_waveform(_waveform, _channels)
        for channel in _channels:
            if channel not in self.dataManager.waveforms[_waveform]:
                self.dataManager.read_waveform(_waveform, [channel])
        
        time = []
        voltages = {}
        for channel in _channels:
            voltages[channel] = []
        for i in range(len(self.dataManager.waveforms[_waveform]['Time_ms'])):                                                  # Get data from dataManager with correct _range and _offset
            if _range == None or _range[0] < self.dataManager.waveforms[_waveform]['Time_ms'][i]*1e6 < _range[1]:
                time.append(self.dataManager.waveforms[_waveform]['Time_ms'][i])
                for channel in _channels:
                    voltages[channel].append(_rescale*self.dataManager.waveforms[_waveform][channel][i] + _offset)
        
        for channel in _channels:                                                                                               # Add waveform to figure
            try:
                if len(_color) == len(_channels):
                    color = _color[channel]
                else:
                    color = _color
            except:
                color = _color
            self.ax[_subplot[0], _subplot[1]].plot(time, voltages[channel], label = _label, color = color, **kwargs)

    #
    # Add a fit to the buckets in the time of flight profile. 
    # Note this function can do a fit of _N peaks itself, but as fitting is non-trivial it is beter to do the plotting manual using the fit_buckets function in the DataManager.
    #
    def add_TOF_fit(self, _filename, _binSize, _channel = 'A', _N = 0, _range = None, _t0 = None, _A = None, _dt = None, _bounds = None, _offset=0, _rescale=1, _label=None, _color=None, _linestyle = 'dashed', _subplot = [0,0], _export_data_filename='', **kwargs):
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        if _filename not in self.dataManager.data or "Peak"+_channel not in self.dataManager.data[_filename]:                           # Read missing data
            self.dataManager.read_peakdata(_filename)

        self.dataManager.bin_timestamps(_filename, _binSize)

        bins = []
        counts = []
        channel = _channel+"nb"
        self.dataManager.subtract_background(_filename, _binSize, _channels = [_channel])

        max_count = np.max(self.dataManager.data[_filename]['Binned'][_binSize][channel])

        if _range != None:
            peak_width = .25*(_range[1] - _range[0])/_N
            fit_range = _range 
        else:
            max_count_i = self.dataManager.data[_filename]['Binned'][_binSize][channel].index(max_count)
            count_i = max_count_i
            while self.dataManager.data[_filename]['Binned'][_binSize][channel][count_i] > max_count/2:
                count_i += 1
            peak_width = _binSize*(count_i - max_count_i) * 4
            fit_range = (_binSize*max_count_i - peak_width * _N*3/4, _binSize*max_count_i + peak_width * _N*3/4)

        if isinstance(_t0, type(None)):
            _t0 = np.linspace(fit_range[0], fit_range[1], _N)
            
        if isinstance(_A, type(None)):
            _A = [max_count]*_N 

        if isinstance(_dt, type(None)):
            _dt = [peak_width]*_N 

        if isinstance(_bounds, type(None)):
            _bounds = ([x for xs in [[fit_range[0]]*_N, [0]*_N, [peak_width/10]*_N] for x in xs], [x for xs in [[fit_range[1]]*_N, [np.inf]*_N, [peak_width*5]*_N] for x in xs])       
        
        popt, pcov = self.dataManager.fit_buckets(_filenames = [_filename], _binSize=_binSize, _range = _range, _channels = [_channel+'nb'], _overwrite = False, _N = _N, _t0 = _t0, _A = _A, _dt = _dt, _bounds = _bounds)


        for k in range(len(self.dataManager.data[_filename]['Binned'][_binSize]['Bins'])-1):
            if _range == None or _range[0] < self.dataManager.data[_filename]['Binned'][_binSize]['Bins'][k] < _range[1]:
                bins.append(self.dataManager.data[_filename]['Binned'][_binSize]['Bins'][k]*1e-6)                                # Bin edge in ms
                counts.append(_rescale*self.dataManager.data[_filename]['Binned'][_binSize][channel+'_fit'][k] + _offset)        # _rescale*counts + _offset

        self.ax[_subplot[0], _subplot[1]].plot(bins, counts, drawstyle = 'steps-mid', label = _label, color = _color, linestyle = _linestyle, **kwargs)
        
        if _export_data_filename != '':                                                                                         # Export data to file
            if _export_data_filename == None:
                _export_data_filename = _filename+"_TOF_fit_binsize_"+str(_binSize)+"ns"

            file = open(_export_data_filename+"_x", "w")
            for i in bins:
                file.write(str(i) + "\n")
            file.close()
            
            file = open(_export_data_filename+"_y", "w")
            for i in counts:
                file.write(str(i) + "\n")
            file.close()
            
        self.latest_filename = _filename
        return popt, pcov
        
    #
    # Add fit parameter t0 (Mean arrival time) per bucket
    #
    def add_t0_fit_parameter(self, _filename, _binSize =  1.2e3, _N = 0, _channel = 'Anb', _rescale=1e-6, _label=None, _color=None, _subplot = [0,0], _velocity_fit = True, **kwargs):
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        
        if _N == 0:
            _N = int(len(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'])/3)

        self.ax[_subplot[0], _subplot[1]].errorbar(range(_N), self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'][0:_N]*_rescale, np.sqrt(np.diag(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_pcov']))[0:_N]*_rescale, fmt='.', marker = '.', label = _label, color = _color)
        
        if _velocity_fit:
            self.dataManager.fit_velocity_t0(_filenames = [_filename], _binSize=_binSize, _channel = _channel, _N = _N)
            self.ax[_subplot[0], _subplot[1]].plot(range(_N), self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_velocity_fit']*_rescale, color = _color, **kwargs)
            self.ax[_subplot[0], _subplot[1]].text(_N, self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'][0]*_rescale, 'v = ' + "{:.2f}".format(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_velocity_popt'][0])+ u"\u00B1" + "{:.2f}".format(np.sqrt(np.diag(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_velocity_pcov']))[0])+' m/s', horizontalalignment = 'right', fontsize = self.fontsize)
        self.latest_filename = _filename


    #
    # Add fit parameter A (Amplitude) per bucket
    #
    def add_A_fit_parameter(self, _filename, _binSize, _N = 0, _channel = 'Anb', _rescale=1, _label=None, _color=None, _subplot = [0,0], **kwargs):
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        N = int(len(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'])/3)
        if _N == 0:
            _N = N

        self.ax[_subplot[0], _subplot[1]].errorbar(range(_N), self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'][N:N+_N]*_rescale, np.sqrt(np.diag(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_pcov']))[N:N+_N]*_rescale, fmt='.', marker = '.', label = _label, color = _color, **kwargs)
            
        self.latest_filename = _filename
        

    #
    # Add fit parameter dt (Time peak width) per bucket
    #
    def add_dt_fit_parameter(self, _filename, _binSize, _N = 0, _channel = 'Anb', _rescale=1e-3, _label=None, _color=None, _subplot = [0,0], _width_fit = True, **kwargs):
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        N = int(len(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'])/3)
        if _N == 0:
            _N = N
        
        self.ax[_subplot[0], _subplot[1]].errorbar(range(_N), self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'][2*N:2*N+_N]*_rescale, np.sqrt(np.diag(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_pcov']))[2*N:2*N+_N]*_rescale, fmt='.', marker = '.', label = _label, color = _color)

        if _width_fit:
            self.dataManager.fit_time_width_dt(_filenames = [_filename], _binSize=_binSize, _channel = _channel, _N = _N)
            self.ax[_subplot[0], _subplot[1]].plot(range(_N), self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_fit']*_rescale, color = _color, **kwargs)
            self.ax[_subplot[0], _subplot[1]].text(_N, self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'][-1]*_rescale, 'dv = ' + "{:.2f}".format(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_popt'][0])+ u"\u00B1" + "{:.2f}".format(np.sqrt(np.diag(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_pcov']))[0])+' m/s\n'+'dx = '+ "{:.2f}".format(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_popt'][1]*1e-6)+ u"\u00B1" + "{:.2f}".format(np.sqrt(np.diag(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_pcov']))[1]*1e-6)+' mm\n'+'d0 = ' + "{:.2f}".format(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_popt'][2]*1e-6)+ u"\u00B1" + "{:.2f}".format(np.sqrt(np.diag(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_pcov']))[2]*1e-6)+' mm', horizontalalignment = 'right', fontsize = self.fontsize)


        self.latest_filename = _filename


    #
    # Add fit parameter dt (Time peak width) vs mean arrival time
    #
    def add_dt_fit_parameter_Parul(self, _filename, _binSize, _N = 0, _channel = 'Anb', _rescale=1e-3, _label=None, _color=None, _subplot = [0,0], _width_fit = True, **kwargs):
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        N = int(len(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'])/3)
        if _N == 0:
            _N = N
        
        self.ax[_subplot[0], _subplot[1]].errorbar(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'][0:_N]*1e-6, self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'][2*N:2*N+_N]*_rescale, np.sqrt(np.diag(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_pcov']))[2*N:2*N+_N]*_rescale, fmt='.', marker = '.', label = _label, color = _color, **kwargs)

        if _width_fit:
            self.dataManager.fit_time_width_dt_Parul(_filenames = [_filename], _binSize=_binSize, _channel = _channel, _N = _N)
            self.ax[_subplot[0], _subplot[1]].plot(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'][0:_N]*1e-6, self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_fit_Parul']*_rescale, color = _color, **kwargs)
            self.ax[_subplot[0], _subplot[1]].text(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'][0]*1e-6, self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_popt'][-1]*_rescale, 'dv = ' + "{:.2f}".format(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_popt_Parul'][0])+ u"\u00B1" + "{:.2f}".format(np.sqrt(np.diag(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_pcov_Parul']))[0])+' m/s\n'+'dx = '+ "{:.2f}".format(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_popt_Parul'][1]*1e-6)+ u"\u00B1" + "{:.2f}".format(np.sqrt(np.diag(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_pcov_Parul']))[1]*1e-6)+' mm\n'+'d0 = ' + "{:.2f}".format(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_popt_Parul'][2]*1e-6)+ u"\u00B1" + "{:.2f}".format(np.sqrt(np.diag(self.dataManager.data[_filename]['Binned'][_binSize][_channel+'_width_pcov_Parul']))[2]*1e-6)+' mm', horizontalalignment = 'right', fontsize = self.fontsize)


        self.latest_filename = _filename

    # 
    # Plot fit result of the fit_buckets procedure
    #

    def plot_fit(self, _filenames = None, _binSize = 1.2e3, _N = 0, _channel = 'Anb', _range = None, _offset_fit=0, _rescale_fit=1, _directory = '', _waveform = None, _waveform_offset = 0.5, _waveform_rescale = 0.05, **kwargs):
        self.create_figure(_subplots = (5,1), _size_inches = (10, 7.5*4))
        if _filenames == None:
            self.dataManager.read_all(_directory = _directory, _verbose = True)
            _filenames = self.dataManager.data
        for data_filename in _filenames:
            self.dataManager.subtract_background(_filename = data_filename, _binSize = _binSize, _channels = [_channel.replace('nb', '')])
            self.set_xlabel("Time [ms]", _subplot = [0,0])
            self.set_ylabel("Signal [arb.]", _subplot = [0,0])
            self.add_TOF(data_filename, _binSize, _subplot = [0,0], **kwargs)
            if _waveform != None:
                minimum = 1e21
                maximum = -1e21
                minimum = np.min(self.dataManager.data[data_filename]['Binned'][_binSize][_channel])
                maximum = np.max(self.dataManager.data[data_filename]['Binned'][_binSize][_channel])
                _offset = (maximum-minimum)*_waveform_offset
                self.add_waveform(_waveform, _offset = _offset, _rescale = _waveform_rescale*_offset, _subplot = [0,0], **kwargs)
            self.add_TOF_fit(data_filename, _binSize, _N, _range, _offset_fit, _rescale_fit, _subplot = [0,0], **kwargs)
            if _range != None:
                self.set_xlim((_range[0]/1e6, _range[1]/1e6))

            self.set_xlabel("Bucket", _subplot = [1,0])
            self.set_ylabel("Mean arrival time [ms]", _subplot = [1,0])
            self.add_t0_fit_parameter(data_filename, _binSize, _N, _channel, _subplot = [1,0])            

            self.set_xlabel("Bucket", _subplot = [2,0])
            self.set_ylabel("Amplitude [arb.]", _subplot = [2,0])
            self.add_A_fit_parameter(data_filename, _binSize, _N, _channel, _subplot = [2,0])     

            self.set_xlabel("Bucket", _subplot = [3,0])
            self.set_ylabel("Peak width [µs]", _subplot = [3,0])
            self.add_dt_fit_parameter(data_filename, _binSize, _N, _channel, _subplot = [3,0])

            self.set_xlabel("Bucket", _subplot = [4,0])
            self.set_ylabel("Peak width [µs]", _subplot = [4,0])
            self.add_dt_fit_parameter_Parul(data_filename, _binSize, _N, _channel, _subplot = [4,0])

            self.save_figure('*_fit_binSize_'+str(_binSize)+'us', _directory, _clear = True)

    #
    # Add a gaussian plot for values _x and and gaussian parameters popt (t0, A, dt, base_level) to the _subplot with and _color.
    #
    def add_gauss(self, _x, popt, _subplot = [0,0], _flip_axes = False, _color='k', **kwargs):
        if _flip_axes:
            self.ax[_subplot[0], _subplot[1]].plot(t_single_Gaussian(_x, *popt), _x, color=_color, **kwargs)
        else:
            self.ax[_subplot[0], _subplot[1]].plot(_x, t_single_Gaussian(_x, *popt), color=_color, **kwargs)
        return _x, popt

    #
    # Add a gaussian fit to the data in _x and _y (with optinal standard deviation in _y as _std_y) to the _subplot with _linestyle and _color. One can provide a _guess and _bounds to improve the fit.
    #
    def add_gauss_fit(self, _x, _y, _std_y = None, _guess = None, _bounds = None , _color="k", _linestyle = '--', _subplot = [0,0], _flip_axes = False, **kwargs):
        popt, pcov = self.dataManager.fit_gauss(_x, _y, _std_y = _std_y, _guess = _guess, _bounds = _bounds)
        if _flip_axes:
            self.ax[_subplot[0], _subplot[1]].plot(t_single_Gaussian(_x, *popt), _x, color=_color, linestyle = _linestyle, **kwargs)
        else:
            self.ax[_subplot[0], _subplot[1]].plot(_x, t_single_Gaussian(_x, *popt), color=_color, linestyle = _linestyle, **kwargs)
        return popt, pcov

    #
    # Add a 2D gaussian fit as a contour to the camera data in _filename to the _subplot. One can provide a _guess and _bounds to improve the fit. If _overwrite = False take the previously fitted values instead. If fix_rotation, the fit is rotated to an angle of 0 degrees.
    #
    def add_camera_2D_gauss_contour(self, _filename, _overwrite = True, _guess = None, _bounds = None, _subplot = [0,0], levels=8, colors='w', _print=True, fix_rotation = False, **kwargs):
        found_filename = self.dataManager.find_filename(_filename)                                                  # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        
        if _filename not in self.dataManager.data:
            self.dataManager.read_camera_data(_filename)
            
        _x, _y = np.meshgrid(self.dataManager.data[_filename]['Camera']['metadata']['horizontal_mm'], self.dataManager.data[_filename]['Camera']['metadata']['vertical_mm'])
        #print('The x shape is ',self.dataManager.data[_filename]['Camera']['metadata']['horizontal_mm'].shape)
        #print('the y shape is ',self.dataManager.data[_filename]['Camera']['metadata']['vertical_mm'].shape)
        if '2D gauss' in self.dataManager.data[_filename]['Camera'] and not _overwrite:
            popt, pcov, contour_2D_gauss_fit, unfitted_noise = self.add_2D_gauss_contour(_x, _y, self.dataManager.data[_filename]['Camera']['2D gauss']['popt'],
                 self.dataManager.data[_filename]['Camera']['2D gauss']['pcov'], self.dataManager.data[_filename]['Camera']['2D gauss']['unfitted_noise'], _subplot = _subplot, levels=levels, colors=colors, _print = _print, fix_rotation = fix_rotation, **kwargs)
            print('popt', self.dataManager.data[_filename]['Camera']['2D gauss']['popt'].shape)
        else:
            popt, pcov, contour_2D_gauss_fit, unfitted_noise = self.fit_2D_gauss_contour(_x, _y, self.dataManager.data[_filename]['Camera']['corrected_array'], _guess = _guess, _bounds = _bounds, _subplot = _subplot, levels=levels, colors=colors, _print = _print, fix_rotation = fix_rotation, **kwargs)
            self.dataManager.data[_filename]['Camera']['2D gauss'] = {}
            self.dataManager.data[_filename]['Camera']['2D gauss']['popt'] = popt
            self.dataManager.data[_filename]['Camera']['2D gauss']['pcov'] = pcov
            self.dataManager.data[_filename]['Camera']['2D gauss']['unfitted_noise'] = unfitted_noise
            print('z shape', self.dataManager.data[_filename]['Camera']['corrected_array'].shape)

        return popt, pcov, contour_2D_gauss_fit, unfitted_noise

    def add_simulation_2D_gauss_contour(self, _filename,x_edges=[-25,25],y_edges=[-12.5,12.5],bins=129, **kwargs):
        self.dataManager.read_simulated_data(filename=_filename)
        x = np.linspace(x_edges[0],x_edges[1],bins)
        y = np.linspace(y_edges[0],y_edges[1],bins)
        X = self.dataManager.data[_filename]['Simulation']['x']*1e3
        Y = self.dataManager.data[_filename]['Simulation']['y']*1e3
        H, xedges, yedges = np.histogram2d(X,Y,bins=[x,y])
        H_shape=list(H.shape)
        vmin=H.min()
        vmax=H.max()
        for k in range(H_shape[0]):
            for l in range(H_shape[1]):
                if H[k,l]>0.98*vmax:
                    H[k,l]=vmax*0.98
                #if H[k,1]<4:
                #    H[k,l]=4
        x=np.linspace(x_edges[0],x_edges[1],bins-1)
        y=np.linspace(y_edges[0],y_edges[1],bins-1)
        xx,yy = np.meshgrid(x,y)
        popt,cov= self.dataManager.fit_2D_gauss(xx, yy, H.T)
        return popt, cov
    

        

    

    # Fit a 2D gaussian fit as a contour to the data in _z (2D array) over the coordinate meshgrids _x and _y to the _subplot. One can provide a _guess and _bounds to improve the fit. If fix_rotation, the fit is rotated to an angle of 0 degrees.
    #
    def fit_2D_gauss_contour(self, _x, _y, _z, _guess = None, _bounds = None, _subplot = [0,0], levels=8, colors='w', _print = True, fix_rotation = False,simulation=False, **kwargs):
        try:
            popt, pcov = self.dataManager.fit_2D_gauss(_x, _y, _z, _guess = _guess, _bounds = _bounds)
        except:
            self.print("add_2D_gauss_contour: Warning: Gaussian Fit did not Converge")
            if isinstance(_guess, type(None)):
                popt = (3,0,0,5,5,0,0)
            else:
                popt = _guess
            pcov = np.zeros((len(popt), len(popt)))
        data_fitted = t_2D_Gaussian((_x, _y), *popt)
        unfitted_noise = np.mean(np.sqrt((data_fitted-_z)**2))
        popt, pcov, contour_2D_gauss_fit, unfitted_noise = self.add_2D_gauss_contour(_x, _y, popt, pcov, unfitted_noise, _subplot = _subplot, levels=levels, colors=colors, _print = _print, fix_rotation = fix_rotation, **kwargs)
        return popt, pcov, contour_2D_gauss_fit, unfitted_noise
       
    #def fit_2D_gauss_countour_simulation(self,_x,_y,_guess = None, _bounds = None, _subplot = [0,0], levels=8, colors='w', _print = True, fix_rotation = False, **kwargs):

    #
    # Add a 2D gaussian fit as a contour to the data in _z (2D array) over the coordinate meshgrids _x and _y to the _subplot. One can provide a _guess and _bounds to improve the fit. If fix_rotation, the fit is rotated to an angle of 0 degrees.
    #
    def add_2D_gauss_contour(self, _x, _y, popt, pcov, unfitted_noise, _subplot = [0,0], levels=8, colors='w', _print = True, fix_rotation = False, **kwargs):
        if fix_rotation:
            tr = transforms.Affine2D().rotate(popt[5])
            transform=tr + self.ax[_subplot[0], _subplot[1]].transData
        else: 
            transform = None
        data_fitted = t_2D_Gaussian((_x, _y), *popt)
        contour_2D_gauss_fit = self.ax[_subplot[0], _subplot[1]].contour(_x, _y, data_fitted, levels=levels, colors=colors, transform=transform, **kwargs)
        if _print:
            self.print("add_2D_gauss_contour: Gaussian fit: \n A: {:.2f} ± {:.3f}".format(popt[0], np.sqrt(pcov[0][0]))
                       +"\n x₀: {:.2f} ± {:.3f} mm".format(popt[1], np.sqrt(pcov[1][1]))
                       +"\n y₀: {:.2f} ± {:.3f} mm".format(popt[2], np.sqrt(pcov[2][2]))
                       +"\n σx: {:.2f} ± {:.3f} mm".format(popt[3], np.sqrt(pcov[3][3]))
                       +"\n σy: {:.2f} ± {:.3f} mm".format(popt[4], np.sqrt(pcov[4][4]))
                       +"\n θ: {:.2f} ± {:.3f} degree".format(popt[5]/np.pi*180, np.sqrt(pcov[5][5])/np.pi*180)
                       +"\n offset: {:.2f} ± {:.3f}".format(popt[6], np.sqrt(pcov[6][6]))
                       +"\n RMSE: {:.2f}".format(unfitted_noise))
        
        return popt, pcov, contour_2D_gauss_fit, unfitted_noise

    #
    # Add signal in _pgopher vs _x ("Frequency" in THz or "Wavelength" in nm) to figure. If _pgopher == None take only know pgopher file in the dataManager. _range selects a range for _x. 
    # If _pgopher is a list of pgopher entries these are combined. Only works when the energy axis is the same for files to be combined. If so, _offset, _rescale, _x_offset and _x_rescale can all be arrays of the same length, or one 
    # _offset and _rescale are used to manipulate the positioning of the pgopher plot. If * in _label it will be replaced by _pgopher.
    # _x_offset shifts the x-axis. If None the default is taken.
    #
    def add_pgopher(self, _pgopher = None, _x = "Frequency", _range = None,_offset=0, _label=None, _color="grey", _rescale=-1, _x_offset = 0, _x_rescale = 1, _subplot = [0,0], _linewidth = 1, **kwargs):
        if _x_offset == None:                                                                                                    # Take default x-axis # Old bit of code to compensate for PGopher mismatch, should not be relevant anymore
            if _x == "Frequency":
                _x_offset = -0.0004 # THz
            elif _x == "Wavelength":
                _x_offset = 0 # nm (To be filled)
        
        x = []
        signal = []
        
        if isinstance(_pgopher, list):      # List of PGopher files
            try:
                if len(_offset) != len(_pgopher):
                    self.print("WARNING add_pgopher: Number of _offset entries ("+str(len(_offset))+") does not match number of pgopher entries to be added ("+str(len(_pgopher)))
            except TypeError:
                _offset = [_offset]*len(_pgopher)

            try:
                if len(_rescale) != len(_pgopher):
                    self.print("WARNING add_pgopher: Number of _rescale entries ("+str(len(_rescale))+") does not match number of pgopher entries to be added ("+str(len(_pgopher)))
            except TypeError:
                _rescale = [_rescale]*len(_pgopher)
                    
            try:
                if len(_x_offset) != len(_pgopher):
                    self.print("WARNING add_pgopher: Number of _x_offset entries ("+str(len(_x_offset))+") does not match number of pgopher entries to be added ("+str(len(_pgopher)))
            except TypeError:
                _x_offset = [_x_offset]*len(_pgopher)
                    
            try:
                if len(_x_rescale) != len(_pgopher):
                    self.print("WARNING add_pgopher: Number of _x_rescale entries ("+str(len(_x_rescale))+") does not match number of pgopher entries to be added ("+str(len(_pgopher)))
            except TypeError:
                _x_rescale = [_x_rescale]*len(_pgopher)
                                        
            for pi, p in enumerate(_pgopher):
                if p not in self.dataManager.pgopher:                                                                        # Read missing data
                    self.dataManager.read_pgopher(p)
            
                if x == []:
                    for i in range(len(self.dataManager.pgopher[p]['Signal'])):                                                      # Get data from dataManager with correct _range and _offset
                        if _range == None or _range[0] < self.dataManager.pgopher[p][_x][i] < _range[1]:
                            x.append(self.dataManager.pgopher[p][_x][i]*_x_rescale[pi] + _x_offset[pi])
                            signal.append(_rescale[pi]*self.dataManager.pgopher[p]["Signal"][i] + _offset[pi])
                else:
                    counter = 0
                    for i in range(len(self.dataManager.pgopher[p]['Signal'])):                                                      # Get data from dataManager with correct _range and _offset
                        if _range == None or _range[0] < self.dataManager.pgopher[p][_x][i] < _range[1]:
                            if x[counter] == (self.dataManager.pgopher[p][_x][i]*_x_rescale[pi] + _x_offset[pi]):
                                signal[counter] += _rescale[pi]*self.dataManager.pgopher[p]["Signal"][i] + _offset[pi]
                                counter += 1
                            else:
                                self.print("ERROR add_pgopher: Can not combine PGopher files in "+str(p)+", probably mismatch between energy axes")
                                return
                        
        else:      # Only one PGopher file  
            if _pgopher == None:                                                                                                    # Select pgopher file if None is given
                if len(list(self.dataManager.pgopher)) < 1:
                    print("No pgopher file in dataManager")
                    return
                elif len(list(self.dataManager.pgopher)) > 1:
                    print("Multiple pgopher files loaded, ambiguous! Plotted " + list(self.dataManager.pgopher)[0])
                _pgopher = list(self.dataManager.pgopher)[0]
                
            if _pgopher not in self.dataManager.pgopher:                                                                            # Read missing data
                self.dataManager.read_pgopher(_pgopher)

            if _label != None:
                _label = _label.replace('*', _pgopher)                                                                              # Replace * by _pgopher in _label

            for i in range(len(self.dataManager.pgopher[_pgopher]['Signal'])):                                                      # Get data from dataManager with correct _range and _offset
                if _range == None or _range[0] < self.dataManager.pgopher[_pgopher][_x][i] < _range[1]:
                    x.append(self.dataManager.pgopher[_pgopher][_x][i]*_x_rescale + _x_offset)
                    signal.append(_rescale*self.dataManager.pgopher[_pgopher]["Signal"][i] + _offset)
        
        self.ax[_subplot[0], _subplot[1]].plot(x, signal, label = _label, color = _color, linewidth = _linewidth, **kwargs)                                        # Add pgopher to figure
        
    #
    # Add camera image to figure. 
    #
    def add_camera_image(self, _filename, _mask = None, _offset = (0,0), _cmap=None, _vmin=None, _vmax=None, _binfactor = 1, _colorbar_orientation=None, _subplot = [0,0], **kwargs):      
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        if _filename not in self.dataManager.data:                                                                              # Read missing data
            self.dataManager.read_camera_data(_filename)
        
        try:
            if _mask == None:
                _mask = np.full_like(self.dataManager.data[_filename]['Camera']['corrected_array'], 1)
        except ValueError:
            pass

        if _mask.shape != self.dataManager.data[_filename]['Camera']['corrected_array'].shape:
            self.print("Shape of the _mask", _mask.shape, "does not match the shape of the camera data", self.dataManager.data[_filename]['Camera']['corrected_array'].shape, ", can not integrate_camera_data.")
        
        self.latest_image = self.ax[_subplot[0], _subplot[1]].imshow(self.dataManager.bin_camera_array(np.multiply(self.dataManager.data[_filename]['Camera']['corrected_array'], _mask), _binfactor), cmap = _cmap, vmin = _vmin, vmax = _vmax, extent=[-0.5*self.dataManager.data[_filename]['Camera']['metadata']['width_mm']+_offset[0],0.5*self.dataManager.data[_filename]['Camera']['metadata']['width_mm']+_offset[0],0.5*self.dataManager.data[_filename]['Camera']['metadata']['height_mm']+_offset[1],-0.5*self.dataManager.data[_filename]['Camera']['metadata']['height_mm']+_offset[1]], **kwargs) #, aspect=self.dataManager.data[_filename]['Camera']['metadata']['aspect'],
        
        print("data, min", np.min(self.dataManager.bin_camera_array(np.multiply(self.dataManager.data[_filename]['Camera']['corrected_array'], _mask), _binfactor)), ", max", np.max(self.dataManager.bin_camera_array(np.multiply(self.dataManager.data[_filename]['Camera']['corrected_array'], _mask), _binfactor)), ", mean", np.mean(self.dataManager.bin_camera_array(np.multiply(self.dataManager.data[_filename]['Camera']['corrected_array'], _mask), _binfactor)), ", std", np.std(self.dataManager.bin_camera_array(np.multiply(self.dataManager.data[_filename]['Camera']['corrected_array'], _mask), _binfactor)), "with range", _vmin, _vmax)
        self.latest_filename = _filename
        
        if _colorbar_orientation != None:
            self.add_colorbar(_orientation=_colorbar_orientation)
        
        if self.verbose:
            print("add_camera_image: added camera image to subplot " + str(_subplot) + " with data from " + str(_filename) + ".")

    def add_simulation_image(self, _filename, _colorbar_orientation=None, _subplot = [0,0],bins=129,x_edges=[-25,25],y_edges=[-12.5,12-5],**kwargs):
        if _filename not in self.dataManager.data:                                                                              # Read missing data
            self.dataManager.read_simulated_data(_filename)
        _x = np.linspace(x_edges[0],x_edges[1],bins)
        _y = np.linspace(y_edges[0],y_edges[1],bins)
        X = self.dataManager.data[_filename]['Simulation']['x']*1e3
        Y = self.dataManager.data[_filename]['Simulation']['y']*1e3
        H, xedges, yedges = np.histogram2d(X,Y,bins=[_x,_y])
        _x = np.linspace(x_edges[0],x_edges[1],bins-1)
        _y = np.linspace(y_edges[0], y_edges[1], bins-1)
        H_shape=list(H.shape)
        vmin=H.min()
        vmax=H.max()
        for k in range(H_shape[0]):
            for l in range(H_shape[1]):
                if H[k,l]>0.98*vmax:
                    H[k,l]=vmax*0.98
               # if H[k,l]<4:
               #     H[k,l]=4
        self.latest_image = self.ax[_subplot[0],_subplot[1]].imshow(H.T,extent = [_x[0],_x[-1],_y[0],_y[-1]])
        #self.ax[_subplot[0],_subplot[1]].set_box_aspect(1/2)
        if _colorbar_orientation != None:
            self.add_colorbar(_orientation=_colorbar_orientation)

    #
    # Add camera image to figure from the hexapole simulation data. 
    # If _bins = 'c' the bins are taken from the data with the latest_filename, which is assumed to be camera data. Bins are combined over a _binfactor.
    #
    def add_camera_simulation(self, _filename, _offset = (0,0), _cmap=None, _cmin=None, _cmax=None, _vmin=None, _vmax=None, _rescale_x = 1, _rescale_y = 1, _bins = 'c', _binfactor = 1, _colorbar_orientation=None, _subplot = [0,0], **kwargs):      
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        if _filename not in self.dataManager.data:                                                                              # Read missing data
            self.dataManager.read_simulation_phasespace(_filename, _channels = ['x', 'y'])
            
        if _bins == 'c':
            try:
                _bins = [self.dataManager.data[self.latest_filename]['Camera']['metadata']['horizontal_mm'],  self.dataManager.data[self.latest_filename]['Camera']['metadata']['vertical_mm']]
            except:
                _bins = None
                self.print("WARNING: add_camera_simulation: Tried to detect bins from latest data file as _bins = 'c' has been used, but could not. Are you sure "+str(self.latest_filename)+ " contains camera data?\nIf you did not intend to use camera data to detect the bins use another _bins argument instead. Continuing with _bins = None for now.")
        
        if _binfactor != 1:
            _bins = [_bins[0][::_binfactor], _bins[1][::_binfactor]]
        
        h,x,y,p = self.ax[_subplot[0], _subplot[1]].hist2d(_rescale_x*np.array(self.dataManager.data[_filename]['x'])+_offset[0], _rescale_y*np.array(self.dataManager.data[_filename]['y'])+_offset[1], bins=_bins, cmin=_cmin, cmax = _cmax, cmap=_cmap, vmin=_vmin, vmax=_vmax, rasterized=True, **kwargs)
        
        #h,x,y = np.histogram2d(_rescale_x*np.array(self.dataManager.data[_filename]['x'])+_offset[0], _rescale_y*np.array(self.dataManager.data[_filename]['y'])+_offset[1], bins=(128, 128))

        #print("sim , vmin, vmax", p.get_clim())
        #self.ax[_subplot[0], _subplot[1]].imshow(h, cmap = plt.cm.jet, vmin = _vmin, vmax = _vmax, extent = [x[0], x[-1], y[0], y[-1]])
        
        print("sim, min", np.min(h), "max", np.max(h), "mean", np.mean(h), "std", np.std(h), "with range", _vmin, _vmax)
        print("Simulated molecules", len(self.dataManager.data[_filename]['x']))
        #print("Simulation bins", _bins)

        self.ax[_subplot[0], _subplot[1]].set_aspect('equal')

        self.latest_filename = _filename
        
        if _colorbar_orientation != None:
            self.add_colorbar(_orientation=_colorbar_orientation)
        
        if self.verbose:
            print("add_camera_simulation: added camera image from hexapole simulation to subplot " + str(_subplot) + " with data from " + str(_filename) + ".")

    def add_colorbar(self, _orientation = 'vertical', _rescale = 1, _label = '', **kwargs):
        bar = plt.colorbar(self.latest_image, orientation = _orientation, ax = self.ax.ravel().tolist(), shrink=_rescale, **kwargs)
        bar.set_label(label=_label, fontsize=self.fontsize)
        bar.ax.tick_params(labelsize=self.fontsize)
        if self.verbose:
            print("add_colorbar: added color bar " + str(_label)+".")

    #
    # Plot time of flight histograms and zoomed time of flight histograms of all datafiles in the directory and save them there
    # Note _binsize is only for the full TOF and in ns
    #
    def plot_all_hist(self, _directory = '', _binSize = 2e5, **kwargs):
        self.dataManager.read_all(_directory = _directory, _verbose = True)
        self.create_figure()
        for data_filename in self.dataManager.data:
            self.dataManager.subtract_background(_filename = data_filename, _binSize = _binSize)
            self.dataManager.subtract_background(_filename = data_filename, _binSize = self.dataManager.get_zoom_binsize(data_filename))
            self.set_xlabel("Time [ms]")
            self.set_ylabel("Signal [arb.]")
            self.add_TOF(data_filename, _binSize, **kwargs)
            self.save_figure('*_TOF', _directory, _clear = True)
            self.set_xlabel("Time [ms]")
            self.set_ylabel("Signal [arb.]")
            self.add_TOF(data_filename, self.dataManager.get_zoom_binsize(data_filename), _range = self.dataManager.get_zoom_range(data_filename), **kwargs)
            self.save_figure('*_ZOOM', _directory, _clear = True)

    #
    # Plot time of flight histograms with an offset, labeled by their _labels, if None labeled by their _filenames instead. Histograms are ordered as in _filenames from bottom to top, where _filenames can also be a list of scannumbers. 
    # The _offset is the step between two histograms, if None the _offset is set to the maximum y-range of the histograms. 
    # To add _waveforms to the figure, add a list of _waveforms, which _channels of these to plot, and the _waveform_offset which will be taken relative to the _offset (every waveform will be translated (_waveform_offset+index)*_offset).
    # The _waveform_rescale sets the amplitude relative to the _offset (_waveform_rescale*_offset corresponds to 10 kV)
    #
    def plot_offset_hists(self, _filenames, _labels=None, _binSize = 2e5, _range = None, _background=False, _offset=None, _save_filename = "offset_hist", _directory = '', _waveforms = [], _waveform_channels = [4], _waveform_offset = 0.5, _waveform_color = 'grey', _waveform_rescale = 0.05, **kwargs):
        self.dataManager.read_all(_directory = _directory, _verbose = True)                                                     # Read all datafiles

        filenames = _filenames                                                                                                  # Find filenames if scannumbers are given instead
        for i in range(len(filenames)):
            returned_filename = self.dataManager.find_filename(filenames[i])
            if returned_filename != "":
                filenames[i] = returned_filename

        if _labels == None:                                                                                                     # If no labels, label by _filenames
            _labels = _filenames

        for i in range(len(filenames)):
            self.dataManager.bin_timestamps(filenames[i], _binSize = _binSize)
        
        if _background:                                                                                                         # Make sure data is there
            channel = "A"
        else:
            channel = "Anb"
            for i in range(len(filenames)):
                self.dataManager.subtract_background(filenames[i], _binSize = _binSize, _channels = ['A'])

        if _offset == None:                                                                                                     # Find _offset as maximum y-range of the histograms if None is given
            minimum = 1e21
            maximum = -1e21
            for i in range(len(filenames)):
                minimum = min(minimum, np.min(self.dataManager.data[filenames[i]]['Binned'][_binSize][channel]))
                maximum = max(maximum, np.max(self.dataManager.data[filenames[i]]['Binned'][_binSize][channel]))
            _offset = maximum-minimum

        self.create_figure(_size_inches = (7.5, 1.5*len(filenames)))                                                            # Create figure and axes
        self.set_xlabel("Time [ms]")
        self.set_ylabel("Fluorescence signal [arb. units, offset]")

        for w in range(len(_waveforms)):                                                                                        # Add _waveforms if any
            self.add_waveform(_waveforms[w], _channels = _waveform_channels, _range = _range,_offset=(w+_waveform_offset)*_offset, _color=_waveform_color, _rescale = _waveform_rescale*_offset, **kwargs)

        for i in range(len(filenames)):                                                                                         # Add histograms
            self.add_TOF(filenames[i], _binSize, _range = _range, _offset = _offset*i, _label = _labels[i], **kwargs)

        if _range==None:                                                                                                        # Add labels
            for i in range(len(filenames)):
                self.ax[0, 0].text(self.dataManager.data[filenames[i]]['Binned'][_binSize]['Bins'][-1]*1e-6, _offset*(0.3+i), _labels[i], horizontalalignment='right', fontsize=self.fontsize)  
        else:
            for i in range(len(filenames)):
                self.ax[0, 0].text(_range[1]*1e-6, _offset*(0.3+i), _labels[i], horizontalalignment='right', fontsize=self.fontsize)  

        self.save_figure(_save_filename, _directory, _clear = False)                                                            # Save figure

    #
    # Plot absorption signal and time of flight histograms with an offset, labeled by their _labels, if None labeled by their _filenames instead. Histograms are ordered as in _filenames from bottom to top, where _filenames can also be a list of scannumbers. 
    # The _offset is the step between two histograms, if None the _offset is set to the maximum y-range of the histograms. 
    # To add _waveforms to the figure, add a list of _waveforms, which _channels of these to plot, and the _waveform_offset which will be taken relative to the _offset (every waveform will be translated (_waveform_offset+index)*_offset).
    # The _waveform_rescale sets the amplitude relative to the _offset (_waveform_rescale*_offset corresponds to 10 kV)
    #
    def plot_offset_hists_and_abs(self, _filenames, _labels=None, _binSize = 2e5, _range = None, _background=False, _offset=None, _save_filename = "offset_hist", _directory = '', _waveforms = [], _waveform_channels = [4], _waveform_offset = 0.5, _waveform_color = 'grey', _waveform_rescale = 0.05, **kwargs):
        self.dataManager.read_all(_directory = _directory, _verbose = True)                                                     # Read all datafiles

        filenames = _filenames                                                                                                  # Find filenames if scannumbers are given instead
        for i in range(len(filenames)):
            returned_filename = self.dataManager.find_filename(filenames[i])
            if returned_filename != "":
                filenames[i] = returned_filename

        if _labels == None:                                                                                                     # If no labels, label by _filenames
            _labels = _filenames
        
        if _background:                                                                                                         # Make sure data is there
            channel = "A"
            for i in range(len(filenames)):
                self.dataManager.bin_timestamps(filenames[i], _binSize = _binSize)
        else:
            channel = "Anb"
            for i in range(len(filenames)):
                self.dataManager.subtract_background(filenames[i], _binSize = _binSize, _channels = ['A'])

        if _offset == None:                                                                                                     # Find _offset as maximum y-range of the histograms if None is given
            minimum = 1e21
            maximum = -1e21
            for i in range(len(filenames)):
                minimum = min(minimum, np.min(self.dataManager.data[filenames[i]]['Binned'][_binSize][channel]))
                maximum = max(maximum, np.max(self.dataManager.data[filenames[i]]['Binned'][_binSize][channel]))
            _offset = maximum-minimum

        self.create_figure(_size_inches = (7.5, 1.5*len(filenames)))                                                            # Create figure and axes
        self.set_xlabel("Time [ms]")
        self.set_ylabel("Fluorescence signal [arb. units, offset]")

        for w in range(len(_waveforms)):                                                                                        # Add _waveforms if any
            self.add_waveform(_waveforms[w], _channels = _waveform_channels, _range = _range,_offset=(w+_waveform_offset)*_offset, _color=_waveform_color, _rescale = _waveform_rescale*_offset, **kwargs)

        for i in range(len(filenames)):                                                                                         # Add histograms
            self.add_TOF(filenames[i], _binSize, _range = _range, _offset = _offset*i, _label = _labels[i], **kwargs)

        if _range==None:                                                                                                        # Add labels
            for i in range(len(filenames)):
                self.ax[0,0].text(self.dataManager.data[filenames[i]]['Binned'][_binSize]['Bins'][-1]*1e-6, _offset*(0.3+i), _labels[i], horizontalalignment='right', fontsize=self.fontsize)  
        else:
            for i in range(len(filenames)):
                self.ax[0,0].text(_range[1]*1e-6, _offset*(0.3+i), _labels[i], horizontalalignment='right', fontsize=self.fontsize)  

        self.save_figure(_save_filename, _directory, _clear = False)                                                            # Save figure

    #
    # Plot time of flight profiles for absorption, fluorescence or both for a range of _scan_values, aswell as a scanplot with the integrated signals over these _scan_values
    # Parameters:
    #   _filenames:                 list of filenames or fileindices for the scan data
    #   _scan_values:               list of scanvalues for each of the _filenames
    #   _filenames_selection:       list of the filenames for which the TOF is plotted, if None all in _filenames are used

    #   _absorption_TOF:            bool to set whether TOF is plotted for absorption signal
    #   _fluorescence_TOF:          bool to set whether TOF is plotted for fluorescence signal
    #   _absorption_scan:           bool to set whether scan is plotted for absorption signal
    #   _fluorescence_scan:         bool to set whether scan is plotted for fluorescence signal

    #   _scan_unit:                 string with the unit of values in the scan, used in the labels of the TOFs
    #   _scan_axislabel:            string used as axislabel in the scanplot

    #   _fluorescence_label:        axis label used in fluorescence TOFs
    #   _fluorescence_rescales:     list of rescale factors for fluorescence TOFs and scan. If None set all to 1
    #   _fluorescence_colors:       list of colors in used in the fluorescence TOFs. If None different shades of orange/ red are used
    #   _fluorescence_scan_color:   color used in fluorescence scanplot. If None the middle value in _fluorescence_colors is used
    #   _fluorescence_scan_label:   axis label used in fluorescence scanplot
    #   _fluorescence_scan_labels:  plot labels used in fluorescence scanplot to be put in legend for each of the _fluorescence_integration_time_ranges. Legend not shown by default
    #   _fluorescence_scan_rescale: rescale value of fluorescence scanplot. If a float or int it is set to all, (default 1), if an array it is set for each of the _fluorescence_integration_time_ranges
    #   _fluorescence_range:        fluorescence signal range used for TOFs
    #   _fluorescence_time_range:   fluorescence time range in ns for TOFs 
    #   _fluorescence_integration_time_ranges:      fluorescence time range in ns for integration range for the scanvalues, if None _fluorescence_time_range is also used
    #   _fluorescence_integration_time_ranges_show: boolean whether to show the integration time ranges in TOFs
    #   _binsize:                   binsize for fluorescence TOFs in ns
    #   _fluorescence_subtract_background:  Subtracht background in fluorescence TOF
    #   _fluorescence_scan_range:   fluorescence signal range used for scanplot

    #   _fluorescence_zoom_box:     bool to set zoom box active
    #   _fluorescence_zoom_box_ltwh [_left, _top, _width, _height] values for zoombox position and size relative to the plot it is put in
    #   _fluorescence_zoom_ranges:  list of fluorescence signal ranges used for TOF zoomboxs. If None use estimate based on _fluorescence_range and binsize ratio
    #   _fluorescence_zoom_time_ranges: list of time ranges for fluorescence TOF zoomboxs in ns
    #   _fluorescence_zoom_binsizes: list of binsizes for fluorescence TOF zoomboxs in ns. If None the _binsize is used

    #   _add_velocity_axis:         add velocity axis to fluorescence TOFs assuming all molecules emmited at ablation
    #   _distance:                  distance of detection point downstream from cell exit in mm
    #   _velocity_ticks             tick postions to add velocity labels. None for automatic

    #   _absorption_label:          axis label used in absorption TOFs
    #   _absorption_colors:         list of colors in used in the absorption TOFs. If None different shades of blue are used
    #   _absorption_scan_color:     color used in absorption scanplot. If None the middle value in _absorption_colors is used
    #   _absorption_scan_label:     axis label used in absorption scanplot
    #   _absorption_scan_labels:    plot labels used in absorption scanplot to be put in legend. Legend not shown by default
    #   _absorption_range:          absorption signal range used for TOFs
    #   _absorption_time_range:     absorption time range in ns for TOFs
    #   _absorption_scan_range:     absorption signal range used for scanplot

    #   _figsize:                   figure size in inches
    #   _label_position:            position of labels in TOFs relative to topright corner. If None the TOF labels are not shown
    #   _flush:                     bool whether to flush the figure, otherwise only saved
    #   _time_label:                time axis label used in TOFs

    #   _left, _bottom, _right, _top,_wspace, _hspace   subplots_adjust values

    def plot_scan_and_TOFs(self, _filenames, _scan_values = None, _filenames_selection = None, _absorption_TOF = False, _fluorescence_TOF = True, _absorption_scan = True, _fluorescence_scan = True, _scan_unit = '', _scan_axislabel = 'Scanvalue', _fluorescence_label = 'Photon counts/shot/bin', _fluorescence_rescales = None, _fluorescence_colors = None, _fluorescence_scan_color = None, _fluorescence_scan_label = 'Photon counts/shot', _fluorescence_scan_rescale = 1, _fluorescence_range = None, _fluorescence_time_range = None, _fluorescence_integration_time_ranges = None, _fluorescence_scan_labels = 'Fluorescence', _fluorescence_integration_time_ranges_show = False, _binsize = 6e4, _fluorescence_subtract_background = True, _fluorescence_scan_range = None, _fluorescence_zoom_box = False, _fluorescence_zoom_box_ltwh = [0.35, 0.1, 0.6, 0.5], _fluorescence_zoom_ranges = None, _fluorescence_zoom_time_ranges = None, _fluorescence_zoom_binsizes = None, _add_velocity_axis = False, _distance = 5268, _velocity_ticks = None, _absorption_colors = None, _absorption_label = 'Absorption (%)', _absorption_scan_color = None, _absorption_scan_label = 'Absorption (%)', _absorption_scan_labels = 'Absorption', _absorption_range = None, _absorption_time_range = None, _absorption_scan_range = None, _figsize = (7, 4), _label_position = (0.05, 0.15), _flush = True, _time_label = 'Time (ms)',_left=0.15, _bottom=0.15, _right=0.9, _top=0.9,_wspace=0.2, _hspace=0.4, **kwargs):
        self.overlay_cindex = None
        if isinstance(_scan_values, type(None)):
            _scan_values = range(len(_filenames))

        if isinstance(_filenames_selection, type(None)):
            _filenames_selection = _filenames
            _scan_values_selection = _scan_values
        else:
            _scan_values_selection = []
            for f in _filenames_selection:
                for ie, e in enumerate(_filenames):
                    if f == e:
                        _scan_values_selection.append(_scan_values[ie])
                        break
                           
        fig = plt.figure(figsize=_figsize)

        # Find number of subplots to be created
        num_cols = 1
        fcol_index = 0
        fscan_index = 1

        num_rows = len(_filenames_selection)

        if _absorption_TOF:
            num_cols += 1
            fcol_index += 1
        if _fluorescence_TOF:
            num_cols += 1
            
        if _absorption_scan:
            fscan_index += 1

        if isinstance(_absorption_colors, type(None)):
            _absorption_colors = [plt.colormaps['sunset'](n) for n in np.linspace(0., 0.3, num_rows)]

        if isinstance(_fluorescence_rescales, type(None)):
            _fluorescence_rescales = np.array([1 for n in range(len(_filenames))])
        else:
            _fluorescence_rescales = np.array(_fluorescence_rescales)

        if isinstance(_fluorescence_colors, type(None)):
            _fluorescence_colors = [plt.colormaps['sunset'](n) for n in np.linspace(0.7, 0.99, num_rows)]

        if isinstance(_absorption_scan_color, type(None)):
            _absorption_scan_color = _absorption_colors[floor(num_rows/2)]

        if isinstance(_fluorescence_scan_color, type(None)):
            if isinstance(_fluorescence_integration_time_ranges, type(None)) or len(_filenames) == len(_fluorescence_integration_time_ranges):
                _fluorescence_scan_color = _fluorescence_colors[floor(num_rows/2)]
            else:
                _fluorescence_scan_color = [plt.colormaps['sunset'](n) for n in np.linspace(0.7, 0.99, len(_fluorescence_integration_time_ranges))]
                
        if isinstance(_fluorescence_scan_rescale, float) or isinstance(_fluorescence_scan_rescale, int):
            if isinstance(_fluorescence_integration_time_ranges,  type(None)):
                _fluorescence_scan_rescale = [_fluorescence_scan_rescale]
            else:
                _fluorescence_scan_rescale = [_fluorescence_scan_rescale]*len(_fluorescence_integration_time_ranges)

        if isinstance(_fluorescence_scan_labels, str):
            if isinstance(_fluorescence_integration_time_ranges,  type(None)):
                _fluorescence_scan_labels = [_fluorescence_scan_labels]
            else:
                _fluorescence_scan_labels = [_fluorescence_scan_labels]*len(_fluorescence_integration_time_ranges)

        # Define axes objects/ subplots
        ax = []
        if _absorption_TOF:
            ax.append([plt.subplot2grid((num_rows, num_cols), (fi, 0)) for fi in range(num_rows)])
        if _fluorescence_TOF:
            ax.append([plt.subplot2grid((num_rows, num_cols), (fi, fcol_index)) for fi in range(num_rows)])

        ax.append([None for fi in range(num_rows)])
        ax[-1][0] = plt.subplot2grid((num_rows, num_cols), (0, fcol_index+1), rowspan=num_rows)
        if _absorption_scan:
            ax[-1][1] = ax[-1][0].twinx() # Create axes object with same x-axis, but different y-axis
        ax = np.array(ax)
        ax = np.transpose(ax)
        self.set_figure(fig, ax)
        
        self.subplots_adjust(_left=_left, _bottom=_bottom, _right=_right, _top=_top,_wspace=_wspace, _hspace=_hspace)

        labels = []
        for v in _scan_values_selection:
            labels.append(str(v) + " " + _scan_unit)

        if _absorption_TOF:
            for vi, v in enumerate(_filenames_selection):
                self.add_absorption(v, _subplot = [vi,0], _label=labels[vi], _color = _absorption_colors[vi], **kwargs)
                if _label_position != None:
                    self.add_text(labels[vi], _subplot = [vi,0], _color = 'k', _position = _label_position)
                if vi == len(_filenames_selection) - 1:
                    self.set_xlabel(_time_label, _subplot = [vi,0])
                else:
                    self.ax[vi,0].xaxis.set_ticklabels([])
                if vi == int(len(_filenames_selection)/2):
                    self.set_ylabel(_absorption_label, _subplot = [vi,0])
                    
                if _absorption_time_range != None:
                    self.set_xlim(np.array(_absorption_time_range)/1e6, _subplot = [vi,0])
                if _absorption_range != None:
                    self.set_ylim(_absorption_range, _subplot = [vi,0])

        if _fluorescence_TOF:
            for vi, v in enumerate(_filenames_selection):
                if vi == 0:
                    self.add_TOF(v, _binsize, _subplot = [vi,fcol_index], _label=labels[vi], _rescale = _fluorescence_rescales[_filenames.index(v)], _color = _fluorescence_colors[vi], _background=not _fluorescence_subtract_background, _add_velocity_axis = _add_velocity_axis, _distance = _distance, _velocity_axislabel = 'Velocity (m/s)', _velocity_ticks = _velocity_ticks, **kwargs)
                else:
                    self.add_TOF(v, _binsize, _subplot = [vi,fcol_index], _label=labels[vi], _rescale = _fluorescence_rescales[_filenames.index(v)], _color = _fluorescence_colors[vi], _background=not _fluorescence_subtract_background, _add_velocity_axis = _add_velocity_axis, _distance = _distance, _velocity_axislabel = '', _velocity_ticks = [], **kwargs)
                if vi == len(_filenames_selection) - 1:
                    self.set_xlabel(_time_label, _subplot = [vi,fcol_index])
                else:
                    self.ax[vi,fcol_index].xaxis.set_ticklabels([])
                if vi == int(len(_filenames_selection)/2):
                    self.set_ylabel(_fluorescence_label, _subplot = [vi,fcol_index])
                if _label_position != None:
                    self.add_text(labels[vi], _subplot = [vi,fcol_index], _color = 'k', _position = _label_position)
                if _fluorescence_time_range != None:
                    self.set_xlim(np.array(_fluorescence_time_range)/1e6, _subplot = [vi,fcol_index])
                    
                if not isinstance(_fluorescence_range, type(None)):
                    self.set_ylim(_fluorescence_range, _subplot = [vi,fcol_index])

                if _fluorescence_zoom_box:                
                    if isinstance(_fluorescence_zoom_binsizes, type(None)):
                        _fluorescence_zoom_binsizes = [_binsize]*len(_filenames_selection)

                    if isinstance(_fluorescence_zoom_ranges, type(None)):
                        _fluorescence_zoom_ranges = [np.array(_fluorescence_range)*_binsize/zbs for zbs in _fluorescence_zoom_binsizes]

                    if isinstance(_fluorescence_zoom_time_ranges, type(None)):
                        _x_zoom_range = np.array(_fluorescence_time_range)/1e6
                    else:
                        _x_zoom_range = np.array(_fluorescence_zoom_time_ranges[vi])/1e6

                    self.add_overlay_subplot(_left = _fluorescence_zoom_box_ltwh[0], _top = _fluorescence_zoom_box_ltwh[1], _width = _fluorescence_zoom_box_ltwh[2], _height = _fluorescence_zoom_box_ltwh[3], _subplot = [vi,fcol_index], _show_zoombox = False, _zoombox_color = None, _zoombox_x = _x_zoom_range, _zoombox_y = _fluorescence_zoom_ranges[vi], linewidth = 0.5)
                    self.add_TOF(v, _fluorescence_zoom_binsizes[vi], _rescale = _fluorescence_rescales[_filenames.index(v)], _background=not _fluorescence_subtract_background, _color=_fluorescence_colors[vi], _subplot = [vi,fscan_index+1], **kwargs)
                    self.set_ylim(_fluorescence_zoom_ranges[vi], _subplot = [vi,fscan_index+1])
                    self.set_xlim(_x_zoom_range, _subplot = [vi,fscan_index+1])


        if _absorption_scan:
            # Absorption
            mean, std, N = self.dataManager.mean_std_scan(_filenames, _scanlabels = [self.dataManager.find_calculator_number('Absorption')])
            mean = np.reshape(mean, (len(mean)), 'C')*100
            std = np.reshape(std, (len(std)), 'C')/np.sqrt(np.reshape(N, (len(N)), 'C'))*100
            
            self.ax[1,fscan_index].set_ylabel(_absorption_scan_label)
            self.ax[1,fscan_index].set_ylim(_absorption_scan_range)

            self.ax[1,fscan_index].errorbar(x=_scan_values, xerr=None, y=mean, yerr=std, color = _absorption_scan_color, marker = '.', linestyle = 'None', label = _absorption_scan_labels)

        if _fluorescence_scan:
            # Fluorescence over timewindow
            if isinstance(_fluorescence_integration_time_ranges, type(None)):
                if isinstance(_fluorescence_time_range, type(None)):
                    mean, std, N = self.dataManager.integrate_timestamps_unbinned(_filenames, _range=_fluorescence_time_range, _overwrite = True)
                else:
                    mean, std, N = self.dataManager.integrate_timestamps_unbinned(_filenames, _range=_fluorescence_time_range, _background_range=np.array((0, _fluorescence_time_range[0]/2)), _overwrite = True)
                mean = np.reshape(mean, (len(mean)), 'C')
                std = np.reshape(std, (len(std)), 'C')/np.sqrt(np.reshape(N, (len(N)), 'C'))

                self.ax[0,fscan_index].errorbar(x=_scan_values, xerr=None, y=mean*_fluorescence_rescales*_fluorescence_scan_rescale[0], yerr=std*_fluorescence_rescales*_fluorescence_scan_rescale[0], color = _fluorescence_scan_color, marker = '.', linestyle = 'None', label = _fluorescence_scan_labels[0])

            else:
                if len(_fluorescence_integration_time_ranges) == len(_filenames):
                    output = np.concatenate([self.dataManager.integrate_timestamps_unbinned([_filename], _range=_fluorescence_integration_time_ranges[_filenamei], _background_range=np.array((0, _fluorescence_integration_time_ranges[0][0]/2)), _overwrite = True) for _filenamei, _filename in enumerate(_filenames)], axis = 2)[:,0]
                    #mean, std, N = 
                    mean = output[0] #np.reshape(mean, (len(mean)), 'C')
                    std = output[1]/np.sqrt(output[2]) #np.reshape(std, (len(std)), 'C')/np.sqrt(np.reshape(N, (len(N)), 'C'))

                    self.ax[0,fscan_index].errorbar(x=_scan_values, xerr=None, y=mean*_fluorescence_rescales*_fluorescence_scan_rescale[0], yerr=std*_fluorescence_rescales*_fluorescence_scan_rescale[0], color = _fluorescence_scan_color, marker = '.', linestyle = 'None', label = _fluorescence_scan_labels[0])
                    
                    if self.verbose:
                        self.print("Integration time " + str(_fluorescence_integration_time_ranges) + " ns: scanvalues = " + str(list(_scan_values)) + " " + str(_scan_unit) + ", counts = " + str(list(mean)) + ", std/sqrt(N) = " + str(list(std)))
                    if _fluorescence_integration_time_ranges_show:
                        for vi, v in enumerate(_filenames_selection):
                            self.ax[vi,fcol_index].axvline(x=_fluorescence_integration_time_ranges[_filenames.index(v)][0]/1e6, ymin=-1e6, ymax=1e6, linewidth = 0.5, linestyle = '--', color = _fluorescence_scan_color)
                            self.ax[vi,fcol_index].axvline(x=_fluorescence_integration_time_ranges[_filenames.index(v)][1]/1e6, ymin=-1e6, ymax=1e6, linewidth = 0.5, linestyle = '--', color = _fluorescence_scan_color)

                
                else:
                    for _fluorescence_integration_time_rangei, _fluorescence_integration_time_range in enumerate(_fluorescence_integration_time_ranges):
                        mean, std, N = self.dataManager.integrate_timestamps_unbinned(_filenames, _range=_fluorescence_integration_time_range, _background_range=np.array((0, _fluorescence_integration_time_range[0]/2)), _overwrite = True)
                        mean = np.reshape(mean, (len(mean)), 'C')
                        std = np.reshape(std, (len(std)), 'C')/np.sqrt(np.reshape(N, (len(N)), 'C'))

                        self.ax[0,fscan_index].errorbar(x=_scan_values, xerr=None, y=mean*_fluorescence_rescales*_fluorescence_scan_rescale[_fluorescence_integration_time_rangei], yerr=std*_fluorescence_rescales*_fluorescence_scan_rescale[_fluorescence_integration_time_rangei], color = _fluorescence_scan_color[_fluorescence_integration_time_rangei], marker = '.', linestyle = 'None', label =_fluorescence_scan_labels[_fluorescence_integration_time_rangei])
                        
                        if self.verbose:
                            self.print("Integration time " + str(_fluorescence_integration_time_range) + " ns: scanvalues = " + str(list(_scan_values)) + " " + str(_scan_unit) + ", counts = " + str(list(mean)) + ", std/sqrt(N) = " + str(list(std)))
                        if _fluorescence_integration_time_ranges_show:
                            for vi, v in enumerate(_filenames_selection):
                                self.ax[vi,fcol_index].axvline(x=_fluorescence_integration_time_range[0]/1e6, ymin=-1e6, ymax=1e6, linewidth = 0.5, linestyle = '--', color = _fluorescence_scan_color[_fluorescence_integration_time_rangei])
                                self.ax[vi,fcol_index].axvline(x=_fluorescence_integration_time_range[1]/1e6, ymin=-1e6, ymax=1e6, linewidth = 0.5, linestyle = '--', color = _fluorescence_scan_color[_fluorescence_integration_time_rangei])
            self.set_ylabel(_fluorescence_scan_label, _subplot = [0,fscan_index])
            self.set_ylim(_fluorescence_scan_range, _subplot = [0,fscan_index])

            if _absorption_scan:
                ax[0,fscan_index].yaxis.set_label_position("right")
                ax[0,fscan_index].yaxis.tick_right()
                ax[1,fscan_index].yaxis.set_label_position("left")
                ax[1,fscan_index].yaxis.tick_left()
                
        self.set_xlabel(_scan_axislabel, _subplot = [0,fscan_index])


        self.save_figure(_scan_axislabel.replace(" ", '_')+"_scan_with_TOFs")
        if _flush:
            self.flush()

        return _scan_values, mean, std

    #
    # Print statement to be overwritten for implementation of UI
    #
    def print(self, _message):
        self.dataManager.print(_message)

# Default function to test the program
if __name__ == '__main__':
    dataManager = DataManager()
    plotManager = PlotManager(dataManager)
    plotManager.plot_all_hist()
    plotManager.plot_all_hist()

#pmg=PlotManager()
#pmg.add_simulation_2D_gauss_contour('test_data_07_08.npy')