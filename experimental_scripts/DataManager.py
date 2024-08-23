import csv
import matplotlib.pyplot as plt
import numpy as np
import yaml ## Install as pyyaml
import pickle
from glob import glob
import os
from pint import UnitRegistry
ur = UnitRegistry()
ur.define("micro- = 1e-6 = µ- = u-")
import scipy.stats
from scipy.special import wofz  
from scipy.optimize import curve_fit

from astropy.io import fits
import gc
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__).split('Analysis')[0]+'Analysis', 'Anno_fis')) # Locate analysis scripts
#from cryosource_tools import cryosource_fis 

def t_single_Gaussian(t, t0, A, dt, base_level = 0):
    f = A* np.exp(-0.5*( (t - t0)/dt )**2 ) + base_level
    return f

# Mutiple gaussians for buckets
def t_multiple_Gaussians(t, N, *args):
    t0 = args[0:N]
    A = args[N:2*N]                                         # I = A*dt*(2*np.pi)**.5
    dt = args[2*N:3*N]                                      # FWHM = 2*(2*np.log(2))**.5*dt
    

    f=0
    for n in range(N):
        f += t_single_Gaussian(t, t0[n], A[n], dt[n])
    
    return f

# N gaussian profiles
# link is the link to the relevant input parameter in the *args array. With two link entries the same index of the *args entry a parameter can be used for multiple peaks to do a correlated fit. If the link entry is array like the sum of all args pointed to is used.
# For fitting: Fix parameters by putting their index in fix. When parameters are fixed the corresponding index in fixed_args is used instead of the value in args. 
def t_multiple_gauss(x, N, link = None, fix = [], fixed_args = [], *args):  
    reply = 0
            
    if link == None:
        link = np.arange(4*N)
        for l in range(N):
            link[3*N + l] = 4*N
    
    if len(link) != 4*N:
        print("Error t_multiple_gauss: No match between number of links to args ("+str(len(link))+") and number of links needed (" +str(4*N)+") for fitting N (" +str(N)+") peaks. To use independent args use link = range(4*N), to make the base level shared use link = None.")
        return reply
        
    alpha = np.zeros(N)     # Gaussian width
    I = np.zeros(N)         # Amplitude
    pos = np.zeros(N)       # Position
    base = np.zeros(N)      # Base level

    for i in range(N):
        try:
            if int(link[i]) in fix:
                alpha[i] = fixed_args[int(link[i])]
            else:
                alpha[i] = args[int(link[i])]               # Gaussian component FWHM
        except TypeError:
            for j in link[i]:            
                if int(j) in fix:
                    alpha[i] += fixed_args[int(j)]
                else:
                    alpha[i] += args[int(j)]
        
        try:
            if int(link[i + N]) in fix:
                I[i] = fixed_args[int(link[i + N])]
            else:
                I[i] = args[int(link[i + N])]               # Amplitude
        except TypeError:
            for j in link[i + N]:            
                if int(j) in fix:
                    I[i] += fixed_args[int(j)]
                else:
                    I[i] += args[int(j)]            
        
        try:
            if int(link[i + 2*N]) in fix:
                pos[i] = fixed_args[int(link[i + 2*N])]
            else:
                pos[i] = args[int(link[i + 2*N])]
        except TypeError:
            for j in link[i + 2*N]:            
                if int(j) in fix:
                    pos[i] += fixed_args[int(j)]
                else:
                    pos[i] += args[int(j)]   
        try:
            if int(link[i + 3*N]) in fix:
                base[i] = fixed_args[int(link[i + 3*N])]
            else:
                base[i] = args[int(link[i + 3*N])]
        except TypeError:
            for j in link[i + 3*N]:
                if int(j) in fix:
                    base[i] += fixed_args[int(j)]
                else:
                    base[i] += args[int(j)]  
                

        reply += t_single_Gaussian(x, pos[i], I[i], alpha[i], base[i])
    #print("alpha", alpha, "I", I, "pos", pos, "base", base)

    return reply
    
# N voigt profiles
# link is the link to the relevant input parameter in the *args array. With two link entries the same index of the *args entry a parameter can be used for multiple peaks to do a correlated fit. If the link entry is array like the sum of all args pointed to is used.
# For fitting: Fix parameters by putting their index in fix. When parameters are fixed the corresponding index in fixed_args is used instead of the value in args. 
def t_multiple_voigts(x, N, link = None, fix = [], fixed_args = [], *args):  
    reply = 0
            
    if link == None:
        link = np.arange(5*N)
        for l in range(N):
            link[4*N + l] = 5*N
    
    if len(link) != 5*N:
        print("Error t_multiple_voigts: No match between number of links to args ("+str(len(link))+") and number of links needed (" +str(5*N)+") for fitting N (" +str(N)+") peaks. To use independent args use link = range(5*N), to make the base level shared use link = None.")
        return reply
        
    alpha = np.zeros(N)     # Gaussian width
    gamma = np.zeros(N)     # Lorentzian width
    I = np.zeros(N)         # Amplitude
    pos = np.zeros(N)       # Position
    base = np.zeros(N)      # Base level
       
    for i in range(N):
        try:
            if int(link[i]) in fix:
                alpha[i] = fixed_args[int(link[i])]
            else:
                alpha[i] = args[int(link[i])]               # Gaussian component FWHM
        except TypeError:
            for j in link[i]:            
                if int(j) in fix:
                    alpha[i] += fixed_args[int(j)]
                else:
                    alpha[i] += args[int(j)]
        
        try:
            if int(link[i + N]) in fix:
                gamma[i] = fixed_args[int(link[i + N])]
            else:
                gamma[i] = args[int(link[i + N])]           # Lorentzian component FWHM
        except TypeError:
            for j in link[i + N]:            
                if int(j) in fix:
                    gamma[i] += fixed_args[int(j)]
                else:
                    gamma[i] += args[int(j)]

        try:
            if int(link[i + 2*N]) in fix:
                I[i] = fixed_args[int(link[i + 2*N])]
            else:
                I[i] = args[int(link[i + 2*N])]               # Intensity
        except TypeError:
            for j in link[i + 2*N]:            
                if int(j) in fix:
                    I[i] += fixed_args[int(j)]
                else:
                    I[i] += args[int(j)]            
        
        try:
            if int(link[i + 3*N]) in fix:
                pos[i] = fixed_args[int(link[i + 3*N])]
            else:
                pos[i] = args[int(link[i + 3*N])]
        except TypeError:
            for j in link[i + 3*N]:            
                if int(j) in fix:
                    pos[i] += fixed_args[int(j)]
                else:
                    pos[i] += args[int(j)]   
        try:
            if int(link[i + 4*N]) in fix:
                base[i] = fixed_args[int(link[i + 4*N])]
            else:
                base[i] = args[int(link[i + 4*N])]
        except TypeError:
            for j in link[i + 4*N]:
                if int(j) in fix:
                    base[i] += fixed_args[int(j)]
                else:
                    base[i] += args[int(j)]  
                

        reply += t_voigt(x, alpha[i], gamma[i], I[i], pos[i], base[i])
    #print("alpha", alpha, "I", I, "pos", pos, "base", base)

    return reply

def t_2D_Gaussian_flat(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    return t_2D_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset).ravel()


def t_2D_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g

'''
# N gaussian profiles
# link is the link to the relevant input parameter in the *args array. With two link entries the same index of the *args entry a parameter can be used for multiple peaks to do a correlated fit. If the link entry is array like the sum of all args pointed to is used.
# For fitting: Fix parameters by putting their index in fix. When parameters are fixed the corresponding index in fixed_args is used instead of the value in args. 
def t_multiple_gauss(x, N, link = None, fix = [], fixed_args = [], *args):  
    reply = 0
            
    if link == None:
        link = np.arange(4*N)
        for l in range(N):
            link[3*N + l] = 4*N
    
    if len(link) != 4*N:
        print("Error t_multiple_gauss: No match between number of links to args ("+str(len(link))+") and number of links needed (" +str(4*N)+") for fitting N (" +str(N)+") peaks. To use independent args use link = range(4*N), to make the base level shared use link = None.")
        return reply
        
    alpha = np.zeros(N)     # Gaussian width
    I = np.zeros(N)         # Amplitude
    pos = np.zeros(N)       # Position
    base = np.zeros(N)      # Base level

    for i in range(N):
        try:
            if int(link[i]) in fix:
                alpha[i] = fixed_args[int(link[i])]
            else:
                alpha[i] = args[int(link[i])]               # Gaussian component FWHM
        except TypeError:
            for j in link[i]:            
                if int(j) in fix:
                    alpha[i] += fixed_args[int(j)]
                else:
                    alpha[i] += args[int(j)]
        
        try:
            if int(link[i + N]) in fix:
                I[i] = fixed_args[int(link[i + N])]
            else:
                I[i] = args[int(link[i + N])]               # Amplitude
        except TypeError:
            for j in link[i + N]:            
                if int(j) in fix:
                    I[i] += fixed_args[int(j)]
                else:
                    I[i] += args[int(j)]            
        
        try:
            if int(link[i + 2*N]) in fix:
                pos[i] = fixed_args[int(link[i + 2*N])]
            else:
                pos[i] = args[int(link[i + 2*N])]
        except TypeError:
            for j in link[i + 2*N]:            
                if int(j) in fix:
                    pos[i] += fixed_args[int(j)]
                else:
                    pos[i] += args[int(j)]   
        try:
            if int(link[i + 3*N]) in fix:
                base[i] = fixed_args[int(link[i + 3*N])]
            else:
                base[i] = args[int(link[i + 3*N])]
        except TypeError:
            for j in link[i + 3*N]:
                if int(j) in fix:
                    base[i] += fixed_args[int(j)]
                else:
                    base[i] += args[int(j)]  
                
        #print("alpha", alpha, "gamma", gamma, "I", I, "pos", pos, "base", base)

        reply += t_single_Gaussian(x, pos[i], I[i], alpha[i], base[i])
    return reply
'''    

# Get longitudinal velocity from arrival times of different buckets
def t_velocity_t0(n, *args):
    v = args[0]         # longitudinal velocity (m/s)
    offset = args[1]    # arrival time offset (ns)
    d = 6e6             # distance between adjacent buckets (nm)
    return offset + n * d / v

# Get longitudinal velocity spread, longitudinal position spread and free flight offset from widths of different buckets. v is the mean longitudinal velocity (in m/s) which is assumed to be the same for all buckets and can be found from the fit_velocity_t0
def t_time_width_dt(n, v, *args):
    dv = args[0]        # longitudinal velocity spread (m/s)
    dx = args[1]        # longitudinal position spread (nm)
    offset = args[2]    # free fligh distance offset offset (nm)
    d = 6e6             # distance between adjacent buckets (nm)
    
    return ((dx/v)**2 + ((offset + d*n)*dv/v**2)**2)**0.5

# Get longitudinal velocity spread, longitudinal position spread and time of switching off the decelerator from widths of different buckets. v is the mean longitudinal velocity (in m/s) which is assumed to be the same for all buckets and can be found from the fit_velocity_t0, d0 is the distance end of decelerator to detector (nm)
# If t0_fix = None it will be fitted, otherwise it will be kept fixed to the given value
def t_time_width_dt_Parul(t, v, d0, t0_fix = None, *args):
    dv = args[0]        # longitudinal velocity spread (m/s)
    dx = args[1]        # longitudinal position spread (nm)
    if isinstance(t0_fix, type(None)):
        t0 = args[2]    # Time of switching off the decelerator (ns)
    else:
        t0 = t0_fix     # Time of switching off the decelerator (ns) (Fixed)
    return np.array([((dx/v)**2 + ((d0/v+_t-t0)*dv/v)**2)**0.5 if _t > t0 else ((dx/v)**2 + ((d0/v)*dv/v)**2)**0.5 for _t in t])


# Voigt profile
def t_voigt(x, alpha, gamma, I, pos, base = 0):
    #alpha     # Gaussian component FWHM
    #gamma     # Lorentzian component FWHM
    #I         # Amplitude
    #pos        # Position
    #base
    #print(alpha, gamma, I, pos, base)
    if alpha < 1e-9:
        alpha = 1e-9
    
    sigma = alpha / (2 * np.sqrt(2 * np.log(2)))
    return I*(np.real(wofz(((x-pos) + 1j*gamma/2)/sigma/np.sqrt(2))))/ (sigma/np.sqrt(2*np.pi)) + base
'''
# N Voigt profiles
# link is the link to the relevant input parameter in the *args array. With two link entries the same index of the *args entry a parameter can be used for multiple peaks to do a correlated fit. If the link entry is array like the sum of all args pointed to is used.
def t_multiple_voigts(x, N, link = None, *args):  
    reply = 0
            
    if link == None:
        link = np.arange(5*N)
        for l in range(N):
            link[4*N + l] = 4*N
    
    if len(link) != 5*N:
        print("Error t_multiple_voigts: No match between number of links to args ("+str(len(link))+") and number of links needed (" +str(5*N)+") for fitting N (" +str(N)+") peaks. To use independent args use link = range(5*N), to make the base level shared use link = None.")
        return reply
        
    alpha = np.zeros(N)
    gamma = np.zeros(N)
    I = np.zeros(N)
    pos = np.zeros(N)
    base = np.zeros(N)
    

    for i in range(N):
        try:
            alpha[i] = args[int(link[i])]           # Gaussian component FWHM
        except TypeError:
            for j in link[i]:
                alpha[i] += args[int(j)]
        try:
            gamma[i] = args[int(link[i + N])]       # Lorentzian component FWHM 
        except TypeError:
            for j in link[i + N]:
                gamma[i] += args[int(j)]
        try:
            I[i] = args[int(link[i + 2*N])]         # Amplitude
        except TypeError:
            for j in link[i + 2*N]:
                I[i] += args[int(j)]                
        try:
            pos[i] = args[int(link[i + 3*N])]        # Position
        except TypeError:
            for j in link[i + 3*N]:
                pos[i] += args[int(j)]   
        try:
            base[i] = args[int(link[i + 4*N])]      # Base level
        except TypeError:
            for j in link[i + 4*N]:
                base[i] += args[int(j)]
                
        #print("alpha", alpha, "gamma", gamma, "I", I, "pos", pos, "base", base)

        reply += t_voigt(x, alpha[i], gamma[i], I[i], pos[i], base[i])
    return reply
'''
# N Voigt profiles
def t_voigt_N(x, *args):
    N = int(len(args)/5)
    
    reply = 0
    print(args)
    
    for n in range(N):
        reply += t_voigt(x, args[n + 2*N], args[n + 3*N], args[n + 1*N], args[n], args[n + 4*N])
    return reply    
    '''
    for n in range(N):
        nargs = [args[n], args[n + N], args[n + 2*N], args[n + 3*N]]
        reply += t_voigt(x, *nargs)
    return reply
    '''
def fit_spectrum(_x, _y, _sigma = None, _N = 1, _link = None, _fix = [], _guess = None, _bounds = (0, np.inf)):
    if _link == None:
        _link = range(4*_N)
    try:
        popt, pcov = curve_fit(lambda t, *args : t_multiple_gauss(t, _N, _link, _fix, _guess, *args), _x, _y, sigma = _sigma, p0 = _guess, bounds = _bounds)
        for i in _fix:
            popt[i] = _guess[i]
            pcov[i] = 0
            pcov[:, i] = 0

    except RuntimeError:
        return _guess, None

    return popt, pcov

def fit_spectrum_voigt(_x, _y, _sigma = None, _N = 1, _link = None, _fix = [], _guess = None, _bounds = (0, np.inf)):
    if _link == None:
        _link = range(5*_N)
    try:
        popt, pcov = curve_fit(lambda t, *args : t_multiple_voigts(t, _N, _link, _fix, _guess, *args), _x, _y, sigma = _sigma, p0 = _guess, bounds = _bounds)
        for i in _fix:
            popt[i] = _guess[i]
            pcov[i] = 0
            pcov[:, i] = 0
    except RuntimeError:
        return _guess, None

    return popt, pcov
    
class DataManager():
    def __init__(self, _verbose = False):
        self.data = {}
        self.waveforms = {}
        self.pgopher = {}
        self.verbose = _verbose
  
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #
    # Reading files
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    #
    # Read signal predicted by PGopher and the corresponding frequencies
    #
    def read_pgopher(self, _file, _overwrite = False, _directory = ''):
        if _file in self.pgopher and not _overwrite:                                                    # Check for existance
            return self.pgopher[_file]

        self.pgopher[_file] = {}                                                                        # Add phopher file to data structure
        
        pgopher_filename = os.path.join(_directory, _file + '_pgopher.txt')                             # Filename signal and frequencies
        
        pgopher_reader = list(csv.reader(open(pgopher_filename, 'r'), delimiter='\t'))
        
        self.pgopher[_file]["Signal"] = []
        self.pgopher[_file]["Frequency"] = []
        self.pgopher[_file]["Wavelength"] = []
        
        for p in range(len(pgopher_reader)):                                                            # Read file
            if pgopher_reader[p] == []:                                                                 # Skip empty lines
                continue
            
            self.pgopher[_file]["Signal"].append(float(pgopher_reader[p][1]))
            self.pgopher[_file]["Frequency"].append(float(pgopher_reader[p][0])/10**6)
            self.pgopher[_file]["Wavelength"].append(2.99792458*10**11/float(pgopher_reader[p][0]))
        
        max_signal = np.max(self.pgopher[_file]["Signal"])
        for p in range(len(self.pgopher[_file]["Signal"])):                                             # Normalise signal
            self.pgopher[_file]["Signal"][p] = self.pgopher[_file]["Signal"][p]/max_signal
        
        return self.pgopher[_file]

    #
    # Read voltages on _channels of a _waveform from file and store in waveforms dictionary (along with a time axis)
    #
    def read_waveform(self, _waveform, _channels = [4], _overwrite = False, _directory = ''):
        if _waveform in self.waveforms and not _overwrite:                                              # Check for existance
            for channel in _channels:                                                                   # Check for existing channels
                if channel in self.waveforms[_waveform]:
                    _channels.pop(_channels.index(channel))

            if len(_channels) == 0:
                self.print("Waveform " + _waveform + " already loaded. If you want to overwrite, call read_waveform() with _overwrite = True as an argument.")
                return self.waveforms[_waveform]
        else:
            self.waveforms[_waveform] = {}                                                              # Add waveform to data structure
        
        for electrode in _channels:
            self.waveforms[_waveform][electrode] = []

        waveform_filename = os.path.join(_directory, _waveform + '.txt')                                # Filename for waveform voltages
            
        line_counter  = 0

        with open(waveform_filename, newline = '') as waveform_file_line:                               # Open waveform file                                       
            waveform_file_reader = csv.reader(waveform_file_line, delimiter='\t')
            for waveform_file_line in waveform_file_reader:
                line_counter += 1
                if line_counter < 3:                                                                    # Skip first lines which do not contain voltage information
                    continue
                for electrode in _channels:
                    self.waveforms[_waveform][electrode].append(10*float(waveform_file_line[electrode]))  # Append voltage on the electrode in kV
        
        self.waveforms[_waveform]['Time_ms'] = np.arange(0, len(self.waveforms[_waveform][_channels[0]])*2e-4-1e-5, 2e-4)  # Times for the waveform file in ms

        return self.waveforms[_waveform]

    #
    # Read metadata and store it in the data dictionary. Note _filename is without _metadata and extension
    #
    def read_metadata(self, _filename, _overwrite = False, _directory = ''):
        found_filename = self.find_filename(_filename)                                                  # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        if _filename not in self.data:                                                                  # If it does not yet exist, add this _filename to the data dictionary
            self.data[_filename] = {}
        elif 'Metadata' in self.data[_filename] and not _overwrite:
            self.print("Metadata for " + _filename + " already loaded. If you want to overwrite, call read_metadata() with _overwrite = True as an argument.")
            return self.data[_filename]['Metadata']

        if _directory == '':                                                                            # Filename for loading
            full_filename = os.path.join(_directory, _filename + '_metadata.yml')
        else:
            full_filename = _filename + '_metadata.yml'
        metadata = self.read_from_yml(full_filename)
        metadata = metadata[list(metadata)[0]]                                                          # Skip User
        metadata = metadata[list(metadata)[0]]                                                          # Skip Project

        self.data[_filename]['Metadata'] = metadata                                                     # Read data to dictionary
        return self.data[_filename]['Metadata']

    #
    # Read scandata and store it in the data dictionary. Note _filename is without extension
    #
    def read_scandata(self, _filename, _overwrite = False, _directory = ''):
        found_filename = self.find_filename(_filename)                                          # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        if _filename not in self.data:                                                                  # If it does not yet exist, add this _filename to the data dictionary
            self.data[_filename] = {}
        elif 'Scandata' in self.data[_filename] and not _overwrite:
            self.print("Scandata for " + _filename + " already loaded. If you want to overwrite, call read_scandata() with _overwrite = True as an argument.")
            return self.data[_filename]['Scandata']

        full_filename = os.path.join(_directory, _filename + '.yml')                                    # Filename for loading

        self.data[_filename]['Scandata'] = self.read_from_yml(full_filename)                            # Read data to dictionary

        if 'peakA' in self.data[_filename]['Scandata'] and self.data[_filename]['Scandata']['peakA'] != []:         # Used to save peakdata with scandata
            self.data[_filename]['PeakA'] = [i for j in self.data[_filename]['Scandata']['peakA'] for i in j]
            self.data[_filename]['Scandata'].pop('peakA')
            file = open(os.path.join(_directory, _filename + '_peaks_A'), "wb")
            pickle.dump(self.data[_filename]['PeakA'], file)
            file.close()
        return self.data[_filename]['Scandata']

    #
    # Read peakdata and store it in the data dictionary. Note _filename is without _peak_<Channel> and extension
    #
    def read_peakdata(self, _filename, _channels = ['A'], _overwrite = False, _directory = ''):
        found_filename = self.find_filename(_filename)                                                      # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        if _filename not in self.data:                                                                  # If it does not yet exist, add this _filename to the data dictionary
            self.data[_filename] = {}
        if 'Metadata' not in self.data[_filename]:
            self.read_metadata(_filename)
        for channel in _channels:
            if 'scope' in channel:                                                                          # Instead of reading from the peak data file one can also detect peaks again from the raw scope data file
                self.detect_peak_scope(_filename, _channels = [channel.replace('scope','')])
                continue
            if 'Peak'+channel in self.data[_filename] and not _overwrite:
                self.print("Peak" + channel + " for " + _filename + " already loaded. If you want to overwrite, call read_peakdata() with _overwrite = True as an argument.")
                continue


            full_filename = os.path.join(_directory, _filename+'_peaks_'+channel)                           # Filename for loading

            self.data[_filename]['Peak'+channel], self.data[_filename]['Peak'+channel+'shots'], self.data[_filename]['Peak'+channel+'perShot'] = self.read_from_bin(full_filename)                        # Read data to dictionary
            if len(self.data[_filename]['Peak' + channel + 'perShot']) != self.data[_filename]['Metadata']['Analyse']['Scanpoints']:
                self.print("!!!WARNING!!! read_peakdata: "+str(_filename)+" is missing shots in peakdata channel "+str(channel)+" ("+ str(len(self.data[_filename]['Peak' + channel + 'perShot']))+"/"+str(self.data[_filename]['Metadata']['Analyse']['Scanpoints'])+" shots found). \nConsider using scope data by read_peakdata("+str(_filename)+", _channels = ['"+str(channel)+"scope']) instead.")
        return self.data[_filename]

    #
    # Combine peakdata from multiple _filenames and store it in the data dictionary under the new _new_filename
    #
    def combine_peakdata(self, _new_filename, _filenames = None, _channels = ['A'], _overwrite = False):
        if _new_filename in self.data and not _overwrite:
            self.print("Data with filename", _new_filename, " already exists. If you want to overwrite, call combine_peakdata() with _overwrite = True as an argument. Otherwise pick another _new_filename")
            return
        if isinstance(_filenames, type(None)):                                                              # If None are specified combine all datafiles
            _filenames = list(self.data)
        
        for i, _filename in enumerate(_filenames):
            found_filename = self.find_filename(_filename)                                                  # If index is given instead of _filename find the corresponding _filename
            if found_filename != "":
                _filenames[i] = found_filename
        
        if _new_filename not in self.data:                                                                  # If it does not yet exist, add this _filename to the data dictionary
            self.data[_new_filename] = {}
        if 'Metadata' not in self.data[_new_filename]:
            self.data[_new_filename]['Metadata'] = self.data[_filenames[0]]['Metadata']
            
        for channel in _channels:
            self.data[_new_filename]['Peak'+channel] = []
            self.data[_new_filename]['Peak'+channel+'shots'] = np.sum([self.data[_filename]['Peak'+channel+'shots'] for _filename in _filenames])
            self.data[_new_filename]['Peak'+channel+'perShot']  = []
            for _filename in _filenames:
                [self.data[_new_filename]['Peak'+channel].append(timestamp) for timestamp in self.data[_filename]['Peak'+channel]]
                for shot in self.data[_filename]['Peak'+channel+'perShot']:
                    self.data[_new_filename]['Peak'+channel+'perShot'].append(shot)

        return self.data[_new_filename]

    #
    # Read peakdata from Signadyne system as in the new filestructure. _filename is without trig.pt or pmt.pt
    # Stored as channel in _channels
    #
    def read_peakdata_signadyne(self, _filename, _channels = ['A'], _period = 1e11, _overwrite = False, _directory = ''):
        if _filename not in self.data:                                                                  # If it does not yet exist, add this _filename to the data dictionary
            self.data[_filename] = {}

        file = open(os.path.join(_directory, _filename+'trig.pt'),'r') 
        trig = np.fromfile(file, dtype=np.uint64)
        file.close()

        # procedure to open .pmt.pt file
        file = open(os.path.join(_directory, _filename+'pmt.pt'),'r') 
        data = np.fromfile(file, dtype=np.uint64)
        file.close()
      
        trig = [x for x in trig if x != 0]             # removing zeroes from the trigger file

        # calculation of number of shots - length of trig
        nShots=len(trig)
        # now we loop the data around the trigger and convert to ns by dividing by 1e3
        dataLooped=((data-trig[0])%_period)/1e3
        
        dataPerShot = [[]]
        
        for di, d in enumerate(dataLooped):
            if di != 0 and  d < dataLooped[di-1]:
                dataPerShot.append([])
            dataPerShot[-1].append(d)
        
        for channel in _channels:
            self.data[_filename]['Peak'+channel] = dataLooped
            self.data[_filename]['Peak'+channel+'shots'] = nShots
            self.data[_filename]['Peak'+channel+'perShot'] = dataPerShot                        # Read data to dictionary
        
        return self.data[_filename]
        
    #
    # Read peakdata from scandata file system (where it was stored untill 2022-01-12)
    #
    def read_peakdata_scandata(self, _filename, _channels = ['A'], _overwrite = False, _directory = ''):
        found_filename = self.find_filename(_filename)                                                      # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        if _filename not in self.data:                                                                  # If it does not yet exist, add this _filename to the data dictionary
            self.data[_filename] = {}
        for channel in _channels:
            if 'Peak'+channel in self.data[_filename] and not _overwrite:
                self.print("Peak" + channel + " for " + _filename + " already loaded. If you want to overwrite, call read_peakdata_scandata() with _overwrite = True as an argument.")
                continue
                        
        for channel in _channels:
            self.data[_filename]['Peak'+channel+'perShot'] = self.read_from_yml(os.path.join(_directory, _filename+'.yml'))['peak'+channel]     
            self.data[_filename]['Peak'+channel] = [i for j in self.data[_filename]['Peak'+channel+'perShot'] for i in j]
            self.data[_filename]['Peak'+channel+'shots'] = len(self.data[_filename]['Peak'+channel+'perShot'])
        return self.data[_filename]
    #
    # Read scopedata and store it in the data dictionary. Note _filename is without _peak_<Channel> and extension
    # If _save_each_shot == False only the mean of the scope shots is saved in the data dictionary
    #
    def read_scopedata(self, _filename, _overwrite = False, _directory = '', _save_each_shot = False):
        found_filename = self.find_filename(_filename)                                          # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        if _filename not in self.data:                                                                      # If it does not yet exist, add this _filename to the data dictionary
            self.data[_filename] = {}
        if "Metadata" not in self.data[_filename]:
            self.read_metadata(_filename)
        if "Scopedata" in self.data[_filename] and not _overwrite:
            self.print("Scopedata for " + _filename + " already loaded. If you want to overwrite, call read_scopedata() with _overwrite = True as an argument.")
            return self.data[_filename]['Scopedata']
        self.data[_filename]["Scopedata"] = {}
        
        samples = self.data[_filename]["Metadata"]['Time']['Samples']                                       # Read relevant metadata
        timestep = ur(str(self.data[_filename]["Metadata"]['Time']['Timestep']).replace(' ', '')).m_as('ms')
        scanpoints = self.data[_filename]["Metadata"]['Analyse']['Scanpoints']
        scans = self.data[_filename]["Metadata"]['Analyse']['Scans']

        maxADC = self.data[_filename]["Metadata"]['Time']['maxADC']
        channels = ['A', 'B', 'C', 'D']
        channels = [i for i in channels if self.data[_filename]["Metadata"]['Channels'][i]['Active'] == 2]
        ranges = [float(ur(str(self.data[_filename]["Metadata"]['Channels'][i]['Range']).replace(' ', '')).m_as('V')) for i in channels]
        self.data[_filename]["Scopedata"]["Time"] = np.linspace(0, (samples-1) * timestep, samples)         # Create time axis
        
        datadir = os.path.join(_directory, _filename+"_scope")                                              # Find data folder
        
        for channel in channels:                                                                            # Prepare datastructure for channels
            self.data[_filename]["Scopedata"][channel] = []
        
        for s in range(scans):                                                                              # Read datafiles
            for sp in range(scanpoints):
                datafile = os.path.join(datadir, _filename+'_'+str(s+1)+'_'+str(sp+1)+'.bin')
                
                f = open(datafile, 'rb')
                for channel in channels:
                    self.data[_filename]["Scopedata"][channel].append(np.fromfile(f, np.short, samples)/maxADC*ranges[channels.index(channel)])
                f.close
                

        for channel in channels:                                                                            # Calculate mean of signal for every channel
            self.data[_filename]["Scopedata"][channel+"_mean"] = np.zeros(samples)
            for i in range(len(self.data[_filename]["Scopedata"][channel])):
                self.data[_filename]["Scopedata"][channel+"_mean"] = np.sum([self.data[_filename]["Scopedata"][channel+"_mean"], self.data[_filename]["Scopedata"][channel][i]], axis=0)
            self.data[_filename]["Scopedata"][channel+"_mean"] = self.data[_filename]["Scopedata"][channel+"_mean"]/len(self.data[_filename]["Scopedata"][channel])
            
            if not _save_each_shot:
                self.data[_filename]["Scopedata"][channel] = []                                                 # Delete scope shots to avoid memory issues
            '''
            for i in range(samples):
                for s in range(len(self.data[_filename]["Scopedata"][channel])):
                    self.data[_filename]["Scopedata"][channel+"_mean"][i] += self.data[_filename]["Scopedata"][channel][s][i]
                self.data[_filename]["Scopedata"][channel+"_mean"][i] = self.data[_filename]["Scopedata"][channel+"_mean"][i]/len(self.data[_filename]["Scopedata"][channel])
            '''
        
        return self.data[_filename]["Scopedata"]


    #
    # Read absorption data from fisfile and store it in the data dictionary. Note _filename is without .fis extension
    # Absorption is calculated with a background level as the average over a region _background in ms. 
    # Many data formats are returned to the data[_filename]["Absorption"] dictionary
    # - ["B_time"]               : [time array]                  : Time entries in the Time of Flight in ms
    # - ["B_percentage_mean"]    : [time array]                  : Absorption percentages for each time averaged over all shots
    def read_absorption_fisfile(self, _filename, _overwrite = False, _directory = '', _range = (0.04,2.), _background_start = 2.):
        found_filename = self.find_filename(_filename)                                                      # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        if _filename not in self.data:                                                                      # If it does not yet exist, add this _filename to the data dictionary
            self.data[_filename] = {}

        if "Absorption" in self.data[_filename] and not _overwrite:
            self.print("Absorption for " + _filename + " already loaded. If you want to overwrite, call read_absorption_fisfile() with _overwrite = True as an argument.")
            return self.data[_filename]['Absorption']
        self.data[_filename]["Absorption"] = {}

        csobj = cryosource_fis(os.path.join(_directory, _filename+".fis"))
        csobj.auto_analysis1(head = 'absorption', signal_start = _range[0]*1e-3, signal_end = _range[1]*1e-3, bg_start = _background_start*1e-3, use_monitor_vals = False)

        pd = csobj.fisfile.absorption_tof_histogram
        self.data[_filename]["Absorption"]["B_time"] = np.array(pd.x[:-1])*1e3
        self.data[_filename]["Absorption"]["B_raw"] = np.array(pd.y)

        return self.data[_filename]["Absorption"]


    #
    # Read molecule arrival times from a (Coldsim) simulation file and store it in the data dictionary. Data structure is mimicking the stucture of peak data from the picoscope
    #

    def read_simulation_peakdata(self, _filename, _channels = ['A'], _binSizes = [1.2e3, 2e5], _range=(0,6e7), _encoding = 'latin1', _overwrite = False, _directory = ''):
        for channel in _channels:
            if _filename not in self.data:                                                                  # If it does not yet exist, add this _filename to the data dictionary
                self.data[_filename] = {}
            elif 'Peak'+channel in self.data[_filename] and not _overwrite:
                self.print("Peak" + channel + " for simulation " + _filename + " already loaded. If you want to overwrite, call read_simulation_peakdata() with _overwrite = True as an argument.")
                continue

            full_filename = os.path.join(_directory, _filename+'.sim')                                      # Filename for loading

            self.data[_filename]['Peak'+channel], self.data[_filename]['Peak'+channel+'shots'], self.data[_filename]['Peak'+channel+'perShot'] = self.read_from_bin(full_filename, _encoding = _encoding)                        # Read data to dictionary

            self.data[_filename]['Binned'] = {}
            self.data[_filename]['Metadata'] = {}
            self.data[_filename]['Metadata']['Zoom'] = {}
            for _binSize in _binSizes:
                nBins = int(round((_range[1]-_range[0])/_binSize))

                self.data[_filename]['Binned'][_binSize] = {}
                self.data[_filename]['Binned'][_binSize][channel], self.data[_filename]['Binned'][_binSize]['Bins'] = np.histogram(self.data[_filename]['Peak'+channel], bins=nBins, range=_range)
                self.data[_filename]['Binned'][_binSize][channel+"nb"] = self.data[_filename]['Binned'][_binSize][channel]
                self.data[_filename]['Metadata']['Zoom']['Binsize'] = str(np.min(_binSizes))+" ns"
                self.data[_filename]['Metadata']['Zoom']['Start'] = str(_range[0])+" ns"
                self.data[_filename]['Metadata']['Zoom']['Length'] = str(_range[1])+" ns"
        return self.data[_filename]
        
    #
    # Read phase space data from hexapole simulation
    #
    def read_simulation_phasespace(self, _filename, _channels = ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'], _overwrite = False, _directory = ''):
        if _filename in self.data and not _overwrite:
            return self.data[_filename]
        self.data[_filename] = {}
        
        for c in _channels:
            full_filename = os.path.join(_directory, _filename + '_'+ c + '.simbin')                                      # Filename for loading
            try:
                file = open(full_filename, "rb")
            except:
                self.print('WARNING: read_simulation_phasespace: No data for coordinate ' + str(c) + ' in simulation phase space data, ' + str(full_filename) + ' not found.')
                continue
            if 'v' in c:
                self.data[_filename][c] = np.array(pickle.load(file))
            else:
                self.data[_filename][c] = 1e3*np.array(pickle.load(file))
            file.close()
        return self.data[_filename]
        
    #
    # Read hexapole simulation data as camera data. 
    # Take metadata from _camera_filename. If _camera_filename = None take the first _filename in the data dictionary that contains camera data
    #
    def read_camera_simulation(self, _filename, _camera_filename = None, _overwrite = False, _directory = '', _offset = (0,0)):      
        found_filename = self.dataManager.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        if _filename in self.data:
            if 'Camera' in self.data[_filename] and not _overwrite:
                self.print("Camera simulation " + _filename + " already loaded. If you want to overwrite, call read_camera_simulation() with _overwrite = True as an argument.")
                return self.data[_filename]
        else:
            self.data[_filename] = {}
            self.data[_filename]['Camera'] = {}
        
        self.read_simulation_phasespace(_filename, _channels = ['x', 'y'], _overwrite = _overwrite, _directory = _directory)
        
        if _camera_filename == None:
            for f in self.data:
                if 'Camera' in self.data[f]:
                    _camera_filename = f 
                    break
        
        self.data[_filename]['Camera']['metadata'] = self.data[_camera_filename]['Camera']['metadata']
        
        self.data[_filename]['Camera']['corrected_array']= np.histogram2d(np.array(self.dataManager.data[_filename]['x'])+_offset[0],np.array(self.dataManager.data[_filename]['y'])+_offset[1], bins=(self.data[_filename]['Camera']['metadata']['horizontal_mm'],self.data[_filename]['Camera']['metadata']['vertical_mm']))

        return self.data[_filename]
    
    #
    # Read raw data from _word_spacer separated values file to dicionary structure and save to file. Entries for the dictionary read from line _header_line, first _skipped_lines_after_header are skipped while reading file (default data before measurement).
    #
    def read_to_dictionary(self, _filename, _header_line = 0, _skipped_lines_after_header = 1, _word_spacer = ' ', _copy_to_yml = False):
        readData = {}
        column_names = []

        line_counter  = 0

        with open(_filename) as file_data_line:                                                                                          
            file_reader = csv.reader(file_data_line, delimiter=_word_spacer)
            for file_data_line in file_reader:
                if line_counter < _header_line: # Skip anything before the header
                    line_counter+=1
                    continue                
                elif line_counter == _header_line: # Read header parameters
                    for column_counter in range(len(file_data_line)):
                        if str(file_data_line[column_counter]) != '':
                            column_names.append(str(file_data_line[column_counter]))
                            readData[column_names[column_counter]] = []
                    line_counter+=1
                elif line_counter <= _header_line + _skipped_lines_after_header: # Skip anything after the header, within the _skipped_lines_after_header
                    line_counter+=1
                    continue
                else: # Read data
                    for column_counter in range(len(file_data_line)):
                        readData[column_names[column_counter]].append(float(file_data_line[column_counter]))
                    line_counter+=1
        if _copy_to_yml:
            self.save_to_yml(_filename.replace('.txt', '.yml'), readData)
        return readData

    #
    # Save _data dictionary to file
    #
    def save_to_yml(self, _filename, _data):
        f = open(_filename, 'w')
        yaml.safe_dump(_data, f)
        f.close()
        return _data

    #
    # Read data dictionary from file
    #
    def read_from_yml(self, _filename):
        f = open(_filename, 'r')
        data = yaml.safe_load(f)
        f.close() 
        return data

    #
    # Read binary from file
    # 
    def read_from_bin(self, _filename, _encoding = 'ascii'):
        try:
            file = open(_filename, "rb")
        except:
            self.print("Could not read binary file " + _filename)
            return
        loading = True
        readDataShot = []
        while loading: 
            try: 
                readDataShot.append(pickle.load(file, encoding=_encoding)) 
            except EOFError: 
                loading = False
        file.close()    
        readData = [i for j in readDataShot for i in j]
        shots = len(readDataShot)
        return readData, shots, readDataShot

    #
    # Read all recognized files in _directory
    #
    def read_all(self, _directory = '', _verbose = True):
        filenames = [i for i in glob(os.path.join(_directory,"*"))]
        for filename in filenames:
            if "_metadata.yml" in filename:
                self.read_metadata(filename.replace("_metadata.yml", ""), _directory = _directory)
                if _verbose:
                    self.print("Read " + filename + " as Metadata")
                #except:
                #    if _verbose:
                #        self.print("Could not read " + filename + " as Metadata")
            elif ".yml" in filename:
                #try:
                self.read_scandata(filename.replace(".yml", ""), _directory = _directory)
                if _verbose:
                    self.print("Read " + filename + " as Scandata")
                #except:
                #    if _verbose:
                #        self.print("Could not read " + filename + " as Scandata")
            elif "_peaks_" in filename:
                #try:
                channel = filename[-1]
                self.read_peakdata(filename[:-8], _channels = [channel], _directory = _directory)
                if _verbose:
                    self.print("Read " + filename + " as Peakdata")
                #except:
                #    if _verbose:
                #        self.print("Could not read " + filename + " as Peakdata")
            elif "_pgopher.txt" in filename:
                self.read_pgopher(filename.replace("_pgopher.txt", ""))
                if _verbose:
                    self.print("Read " + filename + " as PGopher file")
            elif ".txt" in filename:
                #try:
                self.read_waveform(filename[:-4], _directory = _directory)
                if _verbose:
                    self.print("Read " + filename + " as Waveform")
                #except:
                #    if _verbose:
                #        self.print("Could not read " + filename + " as Waveform")
            elif "_scope" in filename:
                self.read_scopedata(filename.replace("_scope", ""))
                if _verbose:
                    self.print("Read " + filename + " as Scopedata")
            elif ".simbin" in filename:
                self.read_simulation_phasespace('_'.join(filename.split("_")[:-1]), _directory = _directory)
                if _verbose:
                    self.print("Read " + filename + " as Simulation phasespace data")
            elif ".sim" in filename:
                channel = "A"
                self.read_simulation_peakdata(filename[:-4], _channels = [channel], _directory = _directory)
                if _verbose:
                    self.print("Read " + filename + " as Simulated Peakdata")
            elif ".fits" in filename:
                self.read_camera_data(filename[:-5], _directory = _directory)
                if _verbose:
                    self.print("Read " + filename + " as Camera data (fits)")
            elif ".fis" in filename:
                self.read_absorption_fisfile(filename[:-4], _directory = _directory)
                if _verbose:
                    self.print("Read " + filename + " as Fisfile absorption data (fis)")
            else:
                if _verbose:
                    self.print("Could not read " + filename)                

    #
    # Print statement to be overwritten for implementation of UI
    #
    def print(self, _message):
        print(_message)

    #
    # Find the filename corresponding to the _scannumber
    #
    def find_filename(self, _scannumber):
        # Find in previously loaded filenames
        for filename in list(self.data):
            if ("scan_" + str(_scannumber)) in filename:
                return filename
            if (("_"+str(_scannumber)) == filename[(-1*len(str(_scannumber))-1):]):
                return filename
        # Find in filenames in this directory
        for filename in [i for i in glob("*")]:
            if ("scan_" + str(_scannumber)) + ".yml" in filename:
                return filename[:-4]
            if ("_" + str(_scannumber)) + ".fits" in filename:
                return filename[:-5]
        return ""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #
    # Timestamps
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
     
    #
    # Detect number of shots in peak data from checking 
    # Peak detected when passing _threshold through an _edge = +1 for rising, -1 for falling
    #
    def detect_peak_scope(self, _filename, _threshold = 0.1, _edge = +1,  _channels = ['A'], _overwrite = False, _directory = ''):
        found_filename = self.find_filename(_filename)                                                      # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        if _filename not in self.data:                                                                  # If it does not yet exist, add this _filename to the data dictionary
            self.data[_filename] = {}
        for channel in _channels:
            if "Scopedata" not in self.data[_filename] or len(self.data[_filename]['Scopedata'][channel]) == 0:                                                     # Read missing data
                self.read_scopedata(_filename, _save_each_shot = True, _overwrite=True)       
            if 'Peak'+channel+'scope' in self.data[_filename] and not _overwrite:
                self.print("Peak" + channel + "scope for " + _filename + " already detected. If you want to overwrite, call detect_peak_scope() with _overwrite = True as an argument.")
                continue
            self.data[_filename]['Peak'+channel+'scopeperShot'] = [[] for i in range(len(self.data[_filename]['Scopedata'][channel]))]
            for shot in range(len(self.data[_filename]['Scopedata'][channel])):
                above_threshold = False
                for ei, e in enumerate(self.data[_filename]['Scopedata'][channel][shot]):
                    if _edge*e > _edge*_threshold and not above_threshold:
                        above_threshold = True
                        self.data[_filename]['Peak'+channel+'scopeperShot'][shot].append(self.data[_filename]['Scopedata']['Time'][ei]*1e6) # append timestamp in ns
                    else:
                        above_threshold = False
            self.data[_filename]['Peak'+channel+'scopeshots'] = len(self.data[_filename]['Scopedata'][channel])
            self.data[_filename]['Peak'+channel+'scope'] = [i for j in self.data[_filename]['Peak'+channel+'scopeperShot'] for i in j]
        if self.verbose:
            self.print("detect_peak_scope: Detected peaks from scope data " + str(_filename) + " in channels " + str(_channels) + ".")
        return self.data[_filename]

    #
    # Bin timestamps in _channels (ns) for _filename into bins of _binSize (ns), in the _range (ns). If _normalise the data will be normalised to the number of shots
    #
    def bin_timestamps(self, _filename, _binSize = 6e3, _range = None, _channels = ['A'], _overwrite = False, _normalise = True):
        if _filename not in self.data:
            self.data[_filename] = {}
        #if "Metadata" not in self.data[_filename]:
        #    self.read_metadata(_filename)
        for channel in _channels:
            if 'Peak' + channel not in self.data[_filename]:
                self.read_peakdata(_filename, [channel])
            if 'Binned' not in self.data[_filename]:
                self.data[_filename]['Binned'] = {}
            if not _binSize in self.data[_filename]['Binned']:
                self.data[_filename]['Binned'][_binSize] = {}
            if channel in self.data[_filename]['Binned'][_binSize] and not _overwrite:
                continue
            
            if _range == None:
                _range = (np.min(self.data[_filename]['Peak'+channel]), np.max(self.data[_filename]['Peak'+channel]))
            nBins = int(round((_range[1]-_range[0])/_binSize))

            self.data[_filename]['Binned'][_binSize][channel], self.data[_filename]['Binned'][_binSize]['Bins'] = np.histogram(self.data[_filename]['Peak'+channel], bins=nBins, range=_range)
            if _normalise:
                self.data[_filename]['Binned'][_binSize][channel] = self.data[_filename]['Binned'][_binSize][channel]/self.data[_filename]['Peak'+channel+'shots']

        return self.data[_filename]['Binned'][_binSize]

    #
    # Subtract background in binned data of _channels (ns) for _filename in bins of _binSize, with background region _background_range (ns). If _background_range = None take last quarter of the bins
    #
    def subtract_background(self, _filename, _binSize = 6e3, _background_range = None, _channels = ['A'], _overwrite = False):
        found_filename = self.find_filename(_filename)                                                      # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        if _filename not in self.data:
            self.data[_filename] = {}
        for channel in _channels:
            if not 'Binned' in self.data[_filename] or not _binSize in self.data[_filename]['Binned'] or not channel in self.data[_filename]['Binned'][_binSize]:
                self.bin_timestamps(_filename, _binSize, _range = None, _channels = [channel], _overwrite = False)
            if channel + 'nb'  in self.data[_filename]['Binned'][_binSize] and not _overwrite:
                continue
        
            # Find background counts per bin
            background = 0
            NBackBins = 0

            # For first half of time range
            if _background_range == None:
                NBackBins = int(round(len(self.data[_filename]['Binned'][_binSize][channel])/4))
                for k in range(NBackBins):
                    background += self.data[_filename]['Binned'][_binSize][channel][-1*k] # Remove -1* for first quarter of the bins
                background = background/NBackBins
            # or for specified range
            else:
                for k in range(len(self.data[_filename]['Binned'][_binSize][channel])):
                    if _background_range[0] < self.data[_filename]['Binned'][_binSize]['Bins'][k] < _background_range[1]:
                        NBackBins += 1
                        background += self.data[_filename]['Binned'][_binSize][channel][k]
                background = background/NBackBins

            # Subtract background
            self.data[_filename]['Binned'][_binSize][channel+'nb'] = [k - background for k in self.data[_filename]['Binned'][_binSize][channel]]

        return self.data[_filename]['Binned'][_binSize]

    #
    # Integrate over _range for binned data of _binSize for _channels in _filename
    # Add bin content if the startingtime of the bin is in the _range
    #
    def integrate_timestamps(self, _filenames = None, _binSize=6e3, _range=None, _background_range=None, _channels = ['Anb'], _overwrite = False):
        # Change to all filenames if None are given
        if _filenames == None:
            _filenames = list(self.data)

        for _filename in _filenames:
            # Make sure all relevant data is available
            if _filename not in self.data:
                self.data[_filename] = {}
            for channel in _channels:
                if not 'Binned' in self.data[_filename] or not _binSize in self.data[_filename]['Binned'] or not channel in self.data[_filename]['Binned'][_binSize]:
                    self.bin_timestamps(_filename, _binSize, _range = None, _channels = [channel.replace('nb', '')], _overwrite = False)
                    if 'nb' in channel:
                        self.subtract_background(_filename, _binSize, _channels = [channel.replace('nb', '')],_background_range=_background_range, _overwrite = False)
            
            # Update _range if None to full range
            if _range == None:
                _range = (self.data[_filename]['Binned'][_binSize]['Bins'][0], self.data[_filename]['Binned'][_binSize]['Bins'][-1])

            # Check whether data is already there, otherwise integrate
            for channel in _channels:
                if _range in self.data[_filename]['Binned'][_binSize] and channel in self.data[_filename]['Binned'][_binSize][_range] and not _overwrite:
                    continue
                if _range not in self.data[_filename]['Binned'][_binSize]:
                    self.data[_filename]['Binned'][_binSize][_range] = {}
                self.data[_filename]['Binned'][_binSize][_range][channel] = 0
                for i in range(len(self.data[_filename]['Binned'][_binSize]['Bins'])):
                    if _range[0] < self.data[_filename]['Binned'][_binSize]['Bins'][i] < _range[1]:
                        self.data[_filename]['Binned'][_binSize][_range][channel] += self.data[_filename]['Binned'][_binSize][channel][i]
                        
        if self.verbose:
            self.print("integrate_timestamps: Integrated timestamps in binned data " + str(_filenames) + " in channels " + str(_channels) + " over range " + str(_range) + " ns.")
        return [[self.data[_filename]['Binned'][_binSize][_range][channel] for channel in _channels] for _filename in _filenames]

    #
    # Integrate over _range for unbinned data for _channels in _filename
    # Add counts in _range, subtract counts in _background_range weigthed to the length ratio of the _background_range to signal _range
    # Note: Subtract background per shot. Does this introduce noise? Is there a better way?
    #
    def integrate_timestamps_unbinned(self, _filenames = None, _range=None, _background_range=None, _channels = ['A'], _overwrite = True):
        # Change to all filenames if None are given
        if isinstance(_filenames, type(None)):
            _filenames = list(self.data)

        for _filenamei, _filename in enumerate(_filenames):
            found_filename = self.find_filename(_filename)                                                      # If index is given instead of _filename find the corresponding _filename
            if found_filename != "":
                _filename = found_filename
                _filenames[_filenamei] = found_filename
            # Make sure all relevant data is available
            if _filename not in self.data:
                self.data[_filename] = {}

            if 'Integrated timestamps' not in self.data[_filename]:
                self.data[_filename]['Integrated timestamps'] = {}
            for channel in _channels:
                if 'Peak' + channel not in self.data[_filename]:
                    self.read_peakdata(_filename, [channel])
            
            # Update _range if None to full range
            if isinstance(_range, type(None)):
                if "Metadata" not in self.data[_filename]:
                    self.read_metadata(_filename)
                _range = (0, float(ur(str(self.data[_filename]['Metadata']['Time']['Blocklength'])).m_as('ns')))
            
            # Update _background_range if None to first quarter of full range
            if isinstance(_background_range, type(None)):
                if "Metadata" not in self.data[_filename]:
                    self.read_metadata(_filename)
                _background_range = (0, float(ur(str(self.data[_filename]['Metadata']['Time']['Blocklength'])).m_as('ns'))/4)
            
            back_range_weight = (_range[1]-_range[0])/(_background_range[1]-_background_range[0])
            
            # Check whether data is already there, otherwise integrate
            for channel in _channels:
                if channel in self.data[_filename]['Integrated timestamps'] and not _overwrite:
                    continue

                self.data[_filename]['Integrated timestamps'][channel] = np.zeros(len(self.data[_filename]['Peak' + channel + 'perShot']))
                for shoti, shot in enumerate(self.data[_filename]['Peak' + channel + 'perShot']):
                    for i in shot:
                        # Check for signal range
                        if _range[0] < i < _range[1]:
                            self.data[_filename]['Integrated timestamps'][channel][shoti] += 1
                        # Check for background range
                        if _background_range[0] < i < _background_range[1]:
                            self.data[_filename]['Integrated timestamps'][channel][shoti] -= back_range_weight
                                                
                self.data[_filename]['Integrated timestamps'][channel+'_mean'] = np.mean(self.data[_filename]['Integrated timestamps'][channel])
                self.data[_filename]['Integrated timestamps'][channel+'_std'] = np.std(self.data[_filename]['Integrated timestamps'][channel])
            if self.verbose:
                self.print("integrate_timestamps_unbinned: Integrated timestamps in unbinned data " + str(_filename) + " in channels " + str(_channels) + " over range " + str(_range) + " ns, subtracting background in " + str(_background_range) + " ns.")        
        return [[self.data[_filename]['Integrated timestamps'][channel+'_mean'] for channel in _channels] for _filename in _filenames], [[self.data[_filename]['Integrated timestamps'][channel+'_std'] for channel in _channels] for _filename in _filenames], [[len(self.data[_filename]['Integrated timestamps'][channel]) for channel in _channels] for _filename in _filenames]

    #
    # Fit _N gaussians with initial guesses for the means _t0, amplitudes _A and stardard deviations _dt to the time of flight in the _range for binned data of _binSize for _channels in _filename
    # Note _t0, _A and _dt should be lists of length _N.
    #
    def fit_buckets(self, _filenames = None, _binSize=1.2e3, _range = None, _channels = ['Anb'], _overwrite = False, _N = 1, _t0 = [25e6], _A = [400], _dt = [2e3], _bounds = (0, 0, 2e3, np.inf, np.inf, 4e4)):
        # Change to all filenames if None are given
        if isinstance(_filenames, type(None)):
            _filenames = list(self.data)

        for _filename in _filenames:
            # Make sure all relevant data is available
            if _filename not in self.data:
                self.data[_filename] = {}
            for channel in _channels:
                if not 'Binned' in self.data[_filename] or not _binSize in self.data[_filename]['Binned'] or not channel in self.data[_filename]['Binned'][_binSize]:
                    self.bin_timestamps(_filename, _binSize, _range = None, _channels = [channel.replace('nb', '')], _overwrite = False)
                    if 'nb' in channel:
                        self.subtract_background(_filename, _binSize, _channels = [channel.replace('nb', '')], _overwrite = False)
            
            for channel in _channels:
                if not _overwrite and channel+'_popt' in self.data[_filename]['Binned'][_binSize]:
                    continue

                if _range == None:
                    time = self.data[_filename]['Binned'][_binSize]['Bins'][:-1]
                    signal = self.data[_filename]['Binned'][_binSize][channel]
                else:
                    time = []
                    signal = []
                    for i in range(len(self.data[_filename]['Binned'][_binSize]['Bins'][:-1])):
                        if  self.data[_filename]['Binned'][_binSize]['Bins'][i] > _range[0] and self.data[_filename]['Binned'][_binSize]['Bins'][i] < _range[1]:
                            time.append(self.data[_filename]['Binned'][_binSize]['Bins'][i])
                            signal.append(self.data[_filename]['Binned'][_binSize][channel][i])

                self.data[_filename]['Binned'][_binSize][channel+'_popt'], self.data[_filename]['Binned'][_binSize][channel+'_pcov'] = curve_fit(lambda t, *args : t_multiple_Gaussians(t, _N, *args), time, signal, p0 = [p for pp in [_t0, _A, _dt] for p in pp], bounds = _bounds)

                self.data[_filename]['Binned'][_binSize][channel+'_fit'] = np.array([t_multiple_Gaussians(t, _N, *(self.data[_filename]['Binned'][_binSize][channel+'_popt'])) for t in self.data[_filename]['Binned'][_binSize]['Bins'][:-1]]) #

        return self.data[_filename]['Binned'][_binSize][channel+'_popt'], self.data[_filename]['Binned'][_binSize][channel+'_pcov']


    #
    # Fit straight line through mean arrival times t0 to find the mean longitudinal velocity 
    # Note only a straight line is expected if the mean longitudinal velocity is constant for all buckets
    #
    def fit_velocity_t0(self, _filenames = None, _binSize=1.2e3, _channel = 'Anb', _N = 0, _velocity = 200, _offset = 2e7, _bounds = None, _overwrite = False):
        # Change to all filenames if None are given
        if isinstance(_filenames, type(None)):
            _filenames = list(self.data)

        for _filename in _filenames:
            if _bounds == None:
                _bounds = ((0, 0), (np.inf, np.inf))

            if _channel+'_velocity_popt' in self.data[_filename]['Binned'][_binSize] and not _overwrite:
                continue

            if _N == 0:
                _N = int(len(self.data[_filename]['Binned'][_binSize][_channel+'_popt'])/3)


            self.data[_filename]['Binned'][_binSize][_channel+'_velocity_popt'], self.data[_filename]['Binned'][_binSize][_channel+'_velocity_pcov'] = curve_fit(t_velocity_t0, list(range(_N)), self.data[_filename]['Binned'][_binSize][_channel+'_popt'][0:_N], p0 = [_velocity, _offset], sigma=np.sqrt(np.diag(self.data[_filename]['Binned'][_binSize][_channel+'_pcov']))[0:_N], bounds = _bounds)
            self.data[_filename]['Binned'][_binSize][_channel+'_velocity_fit'] = np.array([t_velocity_t0(n, *(self.data[_filename]['Binned'][_binSize][_channel+'_velocity_popt'])) for n in range(_N)])

        return self.data[_filename]['Binned'][_binSize][_channel+'_velocity_popt'], self.data[_filename]['Binned'][_binSize][_channel+'_velocity_pcov']


    #
    # Fit to time width of the buckets dt to find the longitudinal velocity spread, longitudinal position spread and free flight offset
    # dt = sqrt((dx/v)^2 + ((d0+ 6 mm * n)/dv)^2) is fitted (see t_time_width_dt)
    #
    def fit_time_width_dt(self, _filenames = None, _binSize=1.2e3, _channel = 'Anb', _N = 0, _velocity = None, _dv = 1, _dx = 2e6, _d0 = 10e6, _bounds = None, _overwrite = False):
        # Change to all filenames if None are given
        if _filenames == None:
            _filenames = list(self.data)

        for _filename in _filenames:
            if _bounds == None:
                _bounds = ((0, 0, 0), (np.inf, np.inf, np.inf))

            if _channel+'_width_popt' in self.data[_filename]['Binned'][_binSize] and not _overwrite:
                continue

            N = int(len(self.data[_filename]['Binned'][_binSize][_channel+'_popt'])/3)
            if _N == 0:
                _N = N

            if _velocity == None:
                self.fit_velocity_t0([_filename], _binSize = _binSize, _channel = _channel, _N = _N)
                _velocity = self.data[_filename]['Binned'][_binSize][_channel+'_velocity_popt'][0]

            self.data[_filename]['Binned'][_binSize][_channel+'_width_popt'], self.data[_filename]['Binned'][_binSize][_channel+'_width_pcov'] = curve_fit(lambda n, *args : t_time_width_dt(n, _velocity, *args), list(range(_N)), self.data[_filename]['Binned'][_binSize][_channel+'_popt'][2*N:2*N+_N]*2*(2*np.log(2))**.5, p0 = [_dv, _dx, _d0], sigma=np.sqrt(np.diag(self.data[_filename]['Binned'][_binSize][_channel+'_pcov']))[2*N:2*N+_N]*2*(2*np.log(2))**.5, bounds = _bounds)
            self.data[_filename]['Binned'][_binSize][_channel+'_width_fit'] = np.array([t_time_width_dt(n, _velocity, *(self.data[_filename]['Binned'][_binSize][_channel+'_width_popt'])) for n in range(_N)])

        return True


    #
    # Can only be used after fit_buckets
    # Fit to time width of the buckets dt to find the longitudinal velocity spread, longitudinal position spread and free flight offset
    # dt = sqrt((_dx/_velocity)^2 + (_d0/_velocity+ t-_t0)^2*(_dv/_velocity)^2) is fitted (see t_time_width_dt_Parul)
    #
    def fit_time_width_dt_Parul(self, _filenames = None, _binSize=1.2e3, _channel = 'Anb', _N = 0, _velocity = None, _dv = 5, _dx = 2e6, _d0 = 10e6, _t0 = 28e6, _fix_t0 = False, _bounds = None, _overwrite = False):
        # Change to all filenames if None are given
        if _filenames == None:
            _filenames = list(self.data)

        for _filename in _filenames:
            if _bounds == None:
                _bounds = ((0, 0, 0), (np.inf, np.inf, np.inf))

            if _channel+'_width_popt_Parul' in self.data[_filename]['Binned'][_binSize] and not _overwrite:
                continue

            N = int(len(self.data[_filename]['Binned'][_binSize][_channel+'_popt'])/3)
            if _N == 0:
                _N = N

            if _velocity == None:
                self.fit_velocity_t0([_filename], _binSize = _binSize, _channel = _channel, _N = _N)
                _velocity = self.data[_filename]['Binned'][_binSize][_channel+'_velocity_popt'][0]
                
            if _fix_t0:
                t0_fix = _t0
            else:
                t0_fix = None

            self.data[_filename]['Binned'][_binSize][_channel+'_width_popt_Parul'], self.data[_filename]['Binned'][_binSize][_channel+'_width_pcov_Parul'] = curve_fit(lambda _t, *args : t_time_width_dt_Parul(_t, _velocity, _d0, t0_fix, *args), self.data[_filename]['Binned'][_binSize][_channel+'_popt'][:_N], self.data[_filename]['Binned'][_binSize][_channel+'_popt'][2*N:2*N+_N], p0 = [_dv, _dx, _t0], sigma=np.sqrt(np.diag(self.data[_filename]['Binned'][_binSize][_channel+'_pcov']))[2*N:2*N+_N], bounds = _bounds)
            self.data[_filename]['Binned'][_binSize][_channel+'_width_fit_Parul'] = t_time_width_dt_Parul(self.data[_filename]['Binned'][_binSize][_channel+'_popt'][:_N], _velocity, _d0, t0_fix, *(self.data[_filename]['Binned'][_binSize][_channel+'_width_popt_Parul']))

        return self.data[_filename]['Binned'][_binSize][_channel+'_width_popt_Parul'], self.data[_filename]['Binned'][_binSize][_channel+'_width_pcov_Parul']

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #
    # Scandata
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    # 
    # Find the entry of the 
    # If _filename is specified the mapping in the corresponding metadatafile is used, if _filename = None the first available metadatafile is used
    #
    def find_calculator_number(self, _calculator_name, _filename = None):
        if _filename == None:
            _filename = list(self.data)[0]
        
        if 'Metadata' not in self.data[_filename]:
            self.read_metadata(_filename)     
            
        for j in self.data[_filename]['Metadata']['Analyse']['Calculators']:
            if self.data[_filename]['Metadata']['Analyse']['Calculators'][j]['Name'] == _calculator_name:
                return int(j)
        
        self.print("!!! Error find_calculator_number: Could not find calculator with name " + _calculator_name + " in metadata for " + _filename)
        
        return None
        

    #
    # Calculate mean and standard deviation of scandata. If _filenames is None look at all known files. If _scanlabels is None calculate for all scanlabels
    #
    def mean_std_scan(self, _filenames = None, _scanlabels = None, _overwrite = False):
        # Change to all filenames if None are given
        if _filenames == None:
            _filenames = list(self.data)
        
        for _filenamei, filename in enumerate(_filenames):
            found_filename = self.find_filename(filename)                                                      # If index is given instead of _filename find the corresponding _filename
            if found_filename != "":
                filename = found_filename
                _filenames[_filenamei] = found_filename
            # Make sure all relevant data is available
            if filename not in self.data:
                self.data[filename] = {}
            if 'Metadata' not in self.data[filename]:
                self.read_metadata(filename)            
            if 'Scandata' not in self.data[filename]:
                self.read_scandata(filename)
            if 'ScandataMean' not in self.data[filename]:
                self.data[filename]['ScandataMean'] = {}
            if 'ScandataStd' not in self.data[filename]:
                self.data[filename]['ScandataStd'] = {}

            # Change to all _scanlabels if None are given
            if _scanlabels == None:
                _scanlabels = list(self.data[filename]['Scandata'])


            # Calculate mean and standard deviation
            for scanlabel in _scanlabels:
                if scanlabel == 'not averaged':
                    continue
                if not scanlabel in self.data[filename]['ScandataMean'] or _overwrite:
                    self.data[filename]['ScandataMean'][scanlabel] = np.mean(self.data[filename]['Scandata'][scanlabel])
                if not scanlabel in self.data[filename]['ScandataStd'] or _overwrite:
                    self.data[filename]['ScandataStd'][scanlabel] = np.std(self.data[filename]['Scandata'][scanlabel])
        return [[self.data[filename]['ScandataMean'][scanlabel] for scanlabel in _scanlabels] for filename in _filenames], [[self.data[filename]['ScandataStd'][scanlabel] for scanlabel in _scanlabels] for filename in _filenames], [[len(self.data[filename]['Scandata'][scanlabel]) for scanlabel in _scanlabels] for filename in _filenames]

    #
    # Bin scandata based on the values in the _x_scanlabel for all _scanlabels, where the mean and standard deviation of the _scanlabels values are calculated corresponding to _x_scanlabel values that fall is a bin, where the bins are made by splitting the _range in _nBins
    #
    def bin_data_scan(self, _filenames = None, _x_scanlabel = "Frequency Castor", _scanlabels = None, _nBins = None, _range = None, _overwrite = False):
        # Change to all filenames if None are given
        if _filenames == None:
            _filenames = list(self.data)

        for filename in _filenames:
            # Make sure all relevant data is available
            if filename not in self.data:
                self.data[filename] = {}
            if 'Scandata' not in self.data[filename]:
                self.read_scandata(filename)
            if "Metadata" not in self.data[filename]:
                self.read_metadata(filename)
            if 'ScandataBinned' not in self.data[filename] or _overwrite:
                self.data[filename]['ScandataBinned'] = {}
            else:
                continue
                
            # Check whether the _x_scanlabel is a calculator 
            if _x_scanlabel not in self.data[filename]['Scandata']:                                                              # Replace py calculator index if Calculator
                calculators = self.data[filename]["Metadata"]["Analyse"]["Calculators"]
                for c in calculators:
                    if calculators[c]['Name'] == _x_scanlabel:
                        _x_scanlabel = c
                    
            # Change to all _scanlabels if None are given
            if _scanlabels == None:
                _scanlabels = list(self.data[filename]['Scandata']).pop(_x_scanlabel)
            
            # Calculate mean and standard deviation
            for scanlabel in _scanlabels:
                if scanlabel == 'not averaged':
                    continue
                if not scanlabel in self.data[filename]['ScandataBinned'] or _overwrite:
                    self.data[filename]['ScandataBinned'][_x_scanlabel] = [[],[]]
                    self.data[filename]['ScandataBinned'][scanlabel] = [[],[]]
                    
                    self.data[filename]['ScandataBinned'][scanlabel][0], self.data[filename]['ScandataBinned'][scanlabel][1], self.data[filename]['ScandataBinned'][_x_scanlabel][0],self.data[filename]['ScandataBinned'][_x_scanlabel][1] = self.bin_data(self.data[filename]['Scandata'][_x_scanlabel], self.data[filename]['Scandata'][scanlabel], _nBins, _range)
                
            '''
            # If no number of bins is given set it to the squareroot of the number of scanpoints
            if _nBins == None:
                _nBins = int(np.sqrt(len(self.data[filename]['Scandata'][_x_scanlabel])))

            # Calculate mean and standard deviation
            for scanlabel in _scanlabels:
                if scanlabel == 'not averaged':
                    continue
                if not scanlabel in self.data[filename]['ScandataBinned'] or _overwrite:
                    # Bin data
                    y2 = [yi**2 for yi in self.data[filename]['Scandata'][scanlabel]]

                    means_result = scipy.stats.binned_statistic(self.data[filename]['Scandata'][_x_scanlabel], [self.data[filename]['Scandata'][scanlabel], y2], bins=_nBins, range=_range, statistic='mean')
                    means, means2 = means_result.statistic
                    N = np.histogram(means_result.binnumber, np.arange(1, _nBins*2, 1))[0][:len(means)]
                    standard_deviations = np.sqrt(means2 - means**2)/np.sqrt(N)
                    
                    self.data[filename]['ScandataBinned'][scanlabel] = [means, standard_deviations]
            
            # Set bins
            bin_edges = means_result.bin_edges
            bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
            bin_widths = bin_edges[1:] - bin_edges[:-1]
            
            self.data[filename]['ScandataBinned'][_x_scanlabel] = [bin_centers, bin_widths]
            '''
            
        return True
        
    #
    # Bin data in _x and _y (arrays of length N), where the mean and standard deviation/sqrt(N_i) of the _y values are calculated corresponding to _x values that fall in a bin, where the bins are made by splitting the _range in _nBins
    # If _nBins = None it will be taken as sqrt(N).
    # If _range = None the full span of _x is taken.
    #
    def bin_data(self, _x, _y, _nBins = None, _range = None):
        # If no number of bins is given set it to the squareroot of the number of scanpoints
        if _nBins == None:
            _nBins = int(np.sqrt(len(_x)))

        # Calculate mean and standard deviation
        # Bin data
        y2 = [yi**2 for yi in _y]

        means_result = scipy.stats.binned_statistic(_x, [_y, y2], bins=_nBins, range=_range, statistic='mean')
        means, means2 = means_result.statistic
        N = np.histogram(means_result.binnumber, np.arange(1, _nBins*2, 1))[0][:len(means)]
        standard_deviations = np.sqrt(means2 - means**2)/np.sqrt(N)
                        
        # Set bins
        bin_edges = means_result.bin_edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
        bin_widths = bin_edges[1:] - bin_edges[:-1]
                
        return means, standard_deviations, bin_centers, bin_widths

    def fit_gauss(self, _x, _y, _std_y = None, _guess = None, _bounds = None):
        if _guess == None:
            _guess = [_x[int(len(_x)/2)], np.max(_y) - np.min(_y), _x[int(len(_x)/4)]-_x[0], np.min(_y)]
        if _bounds == None:
            return curve_fit(lambda f, *args : t_single_Gaussian(f, *args), _x, _y, sigma=_std_y, p0 = _guess)
        else:
            return curve_fit(lambda f, *args : t_single_Gaussian(f, *args), _x, _y, sigma=_std_y, p0 = _guess, bounds = _bounds)

    def fit_voigt(self, _filenames = None, _x_scanlabel = "Frequency Castor", _scanlabel = "LIF", _alpha = [10], _gamma = [10], _I = [1e2], _x0 = [348e6], _base = [0], _bounds = None, _nBins = 1e3, _overwrite = False):
        # Change to all filenames if None are given
        if _filenames == None:
            _filenames = list(self.data)

        for filename in _filenames:
            # Make sure all relevant data is available
            if filename not in self.data:
                self.data[filename] = {}
            if 'Scandata' not in self.data[filename]:
                self.read_scandata(filename)
            if 'Metadata' not in self.data[filename]:
                self.read_metadata(filename)
            if 'ScandataFitted' not in self.data[filename]:
                self.data[filename]['ScandataFitted'] = {}
                
            # Check whether the _x_scanlabel or _scanlabel is a calculator 
            if _x_scanlabel not in self.data[filename]['Scandata'] or _scanlabel not in self.data[filename]['Scandata']:                                                              # Replace py calculator index if Calculator
                calculators = self.data[filename]["Metadata"]["Analyse"]["Calculators"]
                for c in calculators:
                    if calculators[c]['Name'] == _x_scanlabel:
                        _x_scanlabel = c
                    if calculators[c]['Name'] == _scanlabel:
                        _scanlabel = c

            if (_scanlabel in self.data[filename]['ScandataFitted'] or _x_scanlabel in self.data[filename]['ScandataFitted']) and not _overwrite:
                continue
            
            # Check all consistancy in number of peaks to be fitted
            if not len(_alpha) == len(_gamma) == len(_I) == len(_x0) == len(_base):
                self.print("Could not fit voigt as number of peaks according to the number of initial guesses is not consistent (_alpha: "+str(len(_alpha))+", _gamma: "+str(len(_gamma))+"_I: "+str(len(_I))+", _x0: "+str(len(_x0))+"_base: "+str(len(_base))+")")
                continue
            p0 = []

            for a in _x0:
                p0.append(a)  
            for a in _I:
                p0.append(a)
            for a in _alpha:
                p0.append(a)           
            for a in _gamma:
                p0.append(a) 
            for a in _base:
                p0.append(a) 
                
            if _bounds == None:
                _bounds = ((0)*len(_alpha), (np.inf)*len(_alpha))
            self.data[filename]['ScandataFitted'][str(_scanlabel)+'_popt'], self.data[filename]['ScandataFitted'][str(_scanlabel)+'_pcov'] = p0, p0#curve_fit(lambda f, *args : t_voigt_N(f, *args), self.data[filename]['Scandata'][_x_scanlabel], self.data[filename]['Scandata'][_scanlabel], p0 = p0, bounds = _bounds)
            placeholder, self.data[filename]['ScandataFitted'][_x_scanlabel] = np.histogram(self.data[filename]['Scandata'][_x_scanlabel], int(_nBins-1))
            self.print(self.data[filename]['ScandataFitted'][str(_scanlabel)+'_popt'])
            self.data[filename]['ScandataFitted'][_scanlabel] = np.array([t_voigt_N(f, *(self.data[filename]['ScandataFitted'][str(_scanlabel)+'_popt'])) for f in self.data[filename]['ScandataFitted'][_x_scanlabel]])
        return True
    
    def fit_2D_gauss(self, _x, _y, _z, _guess = None, _bounds = None):
        if isinstance(_guess, type(None)):
            _guess = (3,0,0,5,5,0,0)
        if isinstance(_bounds, type(None)):
            return curve_fit(t_2D_Gaussian_flat, (_x, _y), _z.ravel(), p0 = _guess)
        else:
            return curve_fit(t_2D_Gaussian_flat, (_x, _y), _z.ravel(), p0 = _guess, bounds = _bounds)
    #def fit_2D_gauss(self,_x,_y)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #
    # Absorption
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         

    #
    # Calculate absorption percentages per shot from scope data
    # Absorption is calculated with a background level as the average over a region _background in ms. 
    # Many data formats are returned to the data[_filename]["Absorption"] dictionary for each channel in the list of _channels
    # - [channel+"_time"]               : [time array]                  : Time entries in the Time of Flight in ms
    # - [channel+"_percentage"]         : [shot array][time array]      : Absorption percentages for each time for each shot                                only if _save_each_shot == True
    # - [channel+"_percentage_mean"]    : [time array]                  : Absorption percentages for each time averaged over all shots
    # - [channel+"_integrated"]         : [shot array]                  : Absorption percentages for each shot averaged over _range
    # - [channel+"_mean"]               : number                        : Absorption percentages averaged over _range averaged over all shots
    # - [channel+"_std"]                : number                        : Absorption percentages averaged over _range standard deviation over all shots
    # - [channel+"_background"]         : [shot array]                  : Background level averaged over _background_range for each shot 
    # - [channel+"_background_mean"]    : [shot array]                  : Background level averaged over _background_range average over all shots
    # - [channel+"_background_std"]     : [shot array]                  : Background level averaged over _background_range standard deviation over all shots


    def calculate_absorption(self, _filename, _range = (0.1,2.1), _background_range = (2.1,10), _channels = ['B'], _overwrite = False, _save_each_shot = False, _save_background_level = False):
        found_filename = self.find_filename(_filename)                                                      # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        if _filename not in self.data:                                                                      # If it does not yet exist, add this _filename to the data dictionary
            self.data[_filename] = {}
        for channel in _channels:
            if "Absorption" in self.data[_filename] and (channel+"_time" in self.data[_filename]["Absorption"] or channel+"_mean" in self.data[_filename]["Absorption"]) and not _overwrite:
                continue 

            if "Scopedata" not in self.data[_filename] or len(self.data[_filename]['Scopedata'][channel]) == 0:                                                     # Read missing data
                self.read_scopedata(_filename, _save_each_shot = True, _overwrite=True)     
            
            if "Absorption" not in self.data[_filename]:
                self.data[_filename]["Absorption"] = {}
            
            self.data[_filename]["Absorption"][channel+"_time"]= np.array(self.data[_filename]['Scopedata']['Time'])
            self.data[_filename]["Absorption"][channel+"_mean"]= np.zeros(len(self.data[_filename]['Scopedata']['Time']))
            self.data[_filename]["Absorption"][channel+"_percentage"]= np.empty((len(self.data[_filename]['Scopedata'][channel]), len(self.data[_filename]['Scopedata']['Time'])))
            self.data[_filename]["Absorption"][channel+"_integrated"] = np.zeros(len(self.data[_filename]['Scopedata'][channel]))
            
            if _save_background_level:
                self.data[_filename]["Absorption"][channel+"_background"] = np.empty(len(self.data[_filename]['Scopedata'][channel]))

            for shoti, shot in enumerate(self.data[_filename]['Scopedata'][channel]):
                base = 0
                base_counter = 0
                for ti, t in enumerate(self.data[_filename]["Absorption"][channel+"_time"]):
                    if _background_range[0] < t < _background_range[1]:
                        base += self.data[_filename]['Scopedata'][channel][shoti][ti]
                        base_counter += 1
                base = base/base_counter
                if _save_background_level:
                    self.data[_filename]["Absorption"][channel+"_background"][shoti] = base
                
                range_counter = 0            
                for ti, t in enumerate(self.data[_filename]["Absorption"][channel+"_time"]):
                    absorption = (base - self.data[_filename]['Scopedata'][channel][shoti][ti])/base*100
                    self.data[_filename]["Absorption"][channel+"_percentage"][shoti][ti] = absorption
                    if _range == None or _range[0] < t < _range[1]:
                        range_counter += 1
                        self.data[_filename]["Absorption"][channel+"_integrated"][shoti] += absorption
                self.data[_filename]["Absorption"][channel+"_integrated"][shoti] /= range_counter
                
            self.data[_filename]["Absorption"][channel+"_percentage_mean"] = np.mean(self.data[_filename]["Absorption"][channel+"_percentage"], axis=0)
            
            if not _save_each_shot:
                self.data[_filename]["Absorption"][channel+"_percentage"] = []

            
            self.data[_filename]["Absorption"][channel+"_mean"] = np.mean(self.data[_filename]["Absorption"][channel+"_integrated"])
            self.data[_filename]["Absorption"][channel+"_std"] = np.std(self.data[_filename]["Absorption"][channel+"_integrated"])
            
            if _save_background_level:
                self.data[_filename]["Absorption"][channel+"_background_mean"] = np.mean(self.data[_filename]["Absorption"][channel+"_background"])
                self.data[_filename]["Absorption"][channel+"_background_std"] = np.std(self.data[_filename]["Absorption"][channel+"_background"])
                
        if self.verbose:
            self.print("calculate_absorption: Calculated absorption percentages for " + str(_filename) + " in channels " + str(_channels) + ".")
        return self.data[_filename]["Absorption"]   

    #
    # Calculate absorption percentages per shot from scope data
    # Absorption is calculated with a background level as the average over a region _background in ms. 
    # Many data formats are returned to the data[_filename]["Absorption"] dictionary for each channel in the list of _channels
    # - [channel+"_time"]               : [time array]                  : Time entries in the Time of Flight in ms
    # - [channel+"_percentage_mean"]    : [time array]                  : Absorption percentages for each time averaged over all shots
    # - [channel+"_mean"]               : number                        : Absorption percentages averaged over _range averaged over all shots

    def calculate_absorption_fisfile(self, _filename, _range = (0.04,2.), _background_range = (2.,10), _overwrite = False):
        channel = 'B'
        found_filename = self.find_filename(_filename)                                                      # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename
        if _filename not in self.data:                                                                      # If it does not yet exist, add this _filename to the data dictionary
            self.data[_filename] = {}
        if "Absorption" in self.data[_filename]:
            if channel+"_percentage_mean" in self.data[_filename]["Absorption"] and not _overwrite:
                return self.data[_filename]["Absorption"]   
        else:                                                     # Read missing data
            self.read_absorption_fisfile(_filename, _range = _range, _background_start = _background_range[0])

        self.data[_filename]["Absorption"][channel+"_percentage_mean"] = np.zeros_like(self.data[_filename]["Absorption"][channel+"_time"])
        self.data[_filename]["Absorption"][channel+"_mean"] = 0

        base = 0
        base_counter = 0
        for ti, t in enumerate(self.data[_filename]["Absorption"][channel+"_time"]):
            if _background_range[0] < t < _background_range[1]:
                base += self.data[_filename]['Absorption'][channel+"_raw"][ti]
                base_counter += 1
        base = base/base_counter
            
        range_counter = 0            
        for ti, t in enumerate(self.data[_filename]["Absorption"][channel+"_time"]):
            absorption = (base - self.data[_filename]['Absorption'][channel+"_raw"][ti])/base*100
            self.data[_filename]["Absorption"][channel+"_percentage_mean"][ti] = absorption
            if _range == None or _range[0] < t < _range[1]:
                range_counter += 1
                self.data[_filename]["Absorption"][channel+"_mean"] += absorption
        self.data[_filename]["Absorption"][channel+"_mean"] /= range_counter    
                
        if self.verbose:
            self.print("calculate_absorption_fisfile: Calculated absorption percentages for " + str(_filename) + ".")
        return self.data[_filename]["Absorption"]   

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #
    # Metadata
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    #
    # Get range of Zoom plot from metadata
    #
    def get_zoom_range(self, _filename):
        if _filename not in self.data:
            self.data[_filename] = {}
        if 'Metadata' not in self.data[_filename]:
            self.read_metadata(_filename)
        start = round(ur(str(self.data[_filename]['Metadata']['Zoom']['Start'])).m_as('ns'), 3)
        end = start+round(ur(str(self.data[_filename]['Metadata']['Zoom']['Length'])).m_as('ns'),3)
        return (start, end)

    #
    # Get binsize of Zoom plot from metadata
    #
    def get_zoom_binsize(self, _filename):
        if _filename not in self.data:
            self.data[_filename] = {}
        if 'Metadata' not in self.data[_filename]:
            self.read_metadata(_filename)
        return round(ur(str(self.data[_filename]['Metadata']['Zoom']['Binsize'])).m_as('ns'))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #
    # Camera
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    #
    # Read camera data file and store it in the data dictionary. The _filename is without .fits, _overwrite if was already loaded, file in _directory. 
    # The _order determines the order of signal (s) calibration (c) and background (b) window.
    # Set _per_shot to True if shots should be read separately.
    # Set _read_calibration if calibration data should also be read.
    # Set _aspect to read with this aspect ratio instead of determining the aspect ratio from the angle in the filename
    # Set _height_mm to read with this height for each frame in mm instead of determining it from the filename
    # Set _background_filename to a camera filename of index to subtract the camera background. Note the background image is smeared by MinderViezeOorSmear( ,_background_smear,_background_smear)  to reduce noise impact in this image. Set _background_smear = 0 to prevent this.
    # Add _offset to the axis to move zero point
    # Rescale intensities by _intensity_rescale
    # Add _intensity_offset to all intensities
    # _flip axes (x, y) included in this string

    def read_camera_data(self, _filename, _overwrite = False, _directory = '', _order = 'scb', _per_shot = False, _read_calibration = False, _aspect = None, _height_mm = None, _background_filename = None, _background_smear = 8, _offset = (0,0), _intensity_rescale = 1, _intensity_offset = 0, _flip = ''):
        if _background_filename != None:
            found_filename = self.find_filename(_background_filename)                                                  # If index is given instead of _filename find the corresponding _filename
            if found_filename != "":
                _background_filename = found_filename
            self.read_camera_data(_background_filename, _directory = _directory, _order = _order,  _per_shot=False, _aspect = _aspect, _height_mm = None)
            
        single_image = False    
        if _filename[0] != '3':
            if _filename[0] != '0':
                self.print("Warning: "+ str(_filename)+ " does not start with 3, and therefore might not be a tripple trigger file. Will process with order "+str(_order)+ " anyway.")
            else:
                single_image = True
        found_filename = self.find_filename(_filename)                                                  # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        if _filename not in self.data:                                                                  # If it does not yet exist, add this _filename to the data dictionary
            self.data[_filename] = {}
        full_filename = os.path.join(_directory, _filename+'.fits')                                     # Filename for loading

        if 'Camera' in self.data[_filename] and not _overwrite:
            self.print("Camera data " + _filename + " already loaded. If you want to overwrite, call read_camera_data() with _overwrite = True as an argument.")
            return self.data[_filename]

        self.data[_filename]['Camera'] = {}

        # read data and metadata
        with  fits.open(full_filename,mode='denywrite') as fits_file:
            fits_data = fits.getdata(full_filename, ext=0)
            fits_header = fits_file[0].header
            fits_file.close()
            gc.collect()

        self.data[_filename]['Camera']['metadata'] = {}                                                         # Read metadata

        self.data[_filename]['Camera']['metadata']['shape'] = (fits_header["NAXIS1"],fits_header["NAXIS2"])
        self.data[_filename]['Camera']['metadata']['frames'] = fits_data.shape[0]
        if single_image:
            self.data[_filename]['Camera']['metadata']['shots'] = int(self.data[_filename]['Camera']['metadata']['frames'])
        else:
            self.data[_filename]['Camera']['metadata']['shots'] = int(self.data[_filename]['Camera']['metadata']['frames']/3)

        if _aspect == None:
            angle_list = [s[9:] for s in _filename.split("_") if s.startswith('camangle-')]                         # Camera angle from filename
            if len(angle_list):
                try:
                    self.data[_filename]['Camera']['metadata']['angle'] = float(angle_list[0].replace(",","."))
                except:
                    self.print("Warning: found 'camangle-' keyword, but unable to get camera angle from: {}".format(_filename))

                                                                                                                    # Calculates the factor with which the y-direction of the camera should be multiplied to get the correct aspect ratio
                                                                                                                    # Stretching happens with 1/cos(angle), so the function should return the cos(angle)
            self.data[_filename]['Camera']['metadata']['aspect'] = np.cos(self.data[_filename]['Camera']['metadata']['angle']/180*np.pi)
        else:
            self.data[_filename]['Camera']['metadata']['aspect'] = _aspect
            self.data[_filename]['Camera']['metadata']['angle'] = np.arccos(_aspect)/np.pi*180
            
        print("Camera data " + _filename + " aspect ratio " + str(self.data[_filename]['Camera']['metadata']['aspect']) + ".")
        
        self.data[_filename]['Camera']['metadata']['Vertical Shift speed'] = fits_header["VSHIFT"]*1e6
        self.data[_filename]['Camera']['metadata']['Vertical Clock Voltage Amplitude'] = fits_header["VCLKAMP"]
        self.data[_filename]['Camera']['metadata']['Readout Rate'] = 1e-6/fits_header["READTIME"]
        self.data[_filename]['Camera']['metadata']['Pre-Amplifier Gain'] = fits_header["HIERARCH PREAMPGAINTEXT"]
        self.data[_filename]['Camera']['metadata']['Output Amplifier Type'] = fits_header["OUTPTAMP"]
        self.data[_filename]['Camera']['metadata']['EM Gain'] = fits_header["GAIN"]
        region_of_interest = eval('['+fits_header["SUBRECT"]+']')
        self.data[_filename]['Camera']['metadata']['ROI shape'] = (region_of_interest[0],region_of_interest[3],region_of_interest[1],region_of_interest[2]) #Do some string manupuation to format ('1, xxx, xxx,1' -> 'xxx, xxx') 
        self.data[_filename]['Camera']['metadata']['ROI size'] = (region_of_interest[1]-region_of_interest[0]+1,region_of_interest[2]-region_of_interest[3]+1)
        self.data[_filename]['Camera']['metadata']['Binning'] = (fits_header["HBIN"], fits_header["VBIN"])
        self.data[_filename]['Camera']['metadata']['Flip'] = (fits_header["FLIPX"], fits_header["FLIPY"])
               
        if _height_mm == None:
            height_list = [s[12:] for s in _filename.split("_") if s.startswith('imageheight-')]                    # Image height from filename
            if len(height_list):
                try:
                    self.data[_filename]['Camera']['metadata']['height_mm'] = float(height_list[0].replace(",","."))*self.data[_filename]['Camera']['metadata']['ROI size'][1]/self.data[_filename]['Camera']['metadata']['shape'][1]/self.data[_filename]['Camera']['metadata']['Binning'][1]
                    self.data[_filename]['Camera']['metadata']['width_mm'] = self.data[_filename]['Camera']['metadata']['height_mm']/self.data[_filename]['Camera']['metadata']['aspect']
                except:
                    self.print("Warning: found 'imageheight-' keyword, but unable to get image height from: {}".format(_filename))
        else:
            self.data[_filename]['Camera']['metadata']['height_mm'] = float(_height_mm)*self.data[_filename]['Camera']['metadata']['ROI size'][1]/self.data[_filename]['Camera']['metadata']['shape'][1]/self.data[_filename]['Camera']['metadata']['Binning'][1]
            self.data[_filename]['Camera']['metadata']['width_mm'] = self.data[_filename]['Camera']['metadata']['height_mm']/self.data[_filename]['Camera']['metadata']['aspect']

        self.data[_filename]['Camera']['metadata']['vertical_mm'] = np.linspace(-0.5*self.data[_filename]['Camera']['metadata']['height_mm']+_offset[1],0.5*self.data[_filename]['Camera']['metadata']['height_mm']+_offset[1],self.data[_filename]['Camera']['metadata']['shape'][1])
        self.data[_filename]['Camera']['metadata']['horizontal_mm'] = np.linspace(-0.5*self.data[_filename]['Camera']['metadata']['width_mm']+_offset[0],0.5*self.data[_filename]['Camera']['metadata']['width_mm']+_offset[0],self.data[_filename]['Camera']['metadata']['shape'][0])

                                                                                                                # Define arrays
        self.data[_filename]['Camera']['signal_array'] = np.zeros(self.data[_filename]['Camera']['metadata']['shape'])
        self.data[_filename]['Camera']['background_array'] = np.full_like(self.data[_filename]['Camera']['signal_array'], 0)
        self.data[_filename]['Camera']['corrected_array'] = np.full_like(self.data[_filename]['Camera']['signal_array'], 0)

        if _read_calibration and not single_image:
            self.data[_filename]['Camera']['calibration_array'] = np.full_like(self.data[_filename]['Camera']['signal_array'], 0)

        if _per_shot:                                                                                           # Define arrays per shot
            self.data[_filename]['Camera']['signal_arrays'] = np.zeros((int(self.data[_filename]['Camera']['metadata']['shots']), self.data[_filename]['Camera']['metadata']['shape'][0], self.data[_filename]['Camera']['metadata']['shape'][1]))
            self.data[_filename]['Camera']['background_arrays'] = np.full_like(self.data[_filename]['Camera']['signal_arrays'],0)
            self.data[_filename]['Camera']['corrected_arrays'] = np.full_like(self.data[_filename]['Camera']['signal_arrays'],0)

            if _read_calibration and not single_image:
                self.data[_filename]['Camera']['calibration_arrays'] = np.full_like(self.data[_filename]['Camera']['signal_arrays'],0)
        
        if not single_image:
            calibration_index = np.argmin([np.sum(fits_data[i]) for i in range(3)])                                 # Find order of trigger windows in file (Calibration, signal and background)
            signal_index = (_order.find('s') - _order.find('c') + calibration_index)%3
            background_index = (_order.find('b') - _order.find('c') + calibration_index)%3

        for frame in range(self.data[_filename]['Camera']['metadata']['frames']):
            if single_image:
                self.data[_filename]['Camera']['signal_array'] = np.add(self.data[_filename]['Camera']['signal_array'], fits_data[frame])
                if _per_shot:
                    self.data[_filename]['Camera']['signal_arrays'][int(frame)] = fits_data[frame]*_intensity_rescale+_intensity_offset
            else:
                if (frame - calibration_index)%3 == 0 and _read_calibration:
                    self.data[_filename]['Camera']['calibration_array'] = np.add(self.data[_filename]['Camera']['calibration_array'], fits_data[frame])
                    if _per_shot:
                        self.data[_filename]['Camera']['calibration_arrays'][int(frame/3)] = fits_data[frame]*_intensity_rescale+_intensity_offset
                elif (frame - signal_index)%3 == 0:
                    self.data[_filename]['Camera']['signal_array'] = np.add(self.data[_filename]['Camera']['signal_array'], fits_data[frame])
                    if _per_shot:
                        self.data[_filename]['Camera']['signal_arrays'][int(frame/3)] = fits_data[frame]*_intensity_rescale+_intensity_offset
                elif (frame - background_index)%3 == 0:
                    self.data[_filename]['Camera']['background_array'] = np.add(self.data[_filename]['Camera']['background_array'], fits_data[frame])
                    if _per_shot:
                        self.data[_filename]['Camera']['background_arrays'][int(frame/3)] = fits_data[frame]*_intensity_rescale+_intensity_offset

        # Divide averages array by number of shots

        self.data[_filename]['Camera']['signal_array'] = np.divide(self.data[_filename]['Camera']['signal_array'], self.data[_filename]['Camera']['metadata']['shots'])*_intensity_rescale
        self.data[_filename]['Camera']['background_array'] = np.divide(self.data[_filename]['Camera']['background_array'], self.data[_filename]['Camera']['metadata']['shots'])*_intensity_rescale

        if 'x' in _flip:
            self.data[_filename]['Camera']['signal_array'] = np.flip(self.data[_filename]['Camera']['signal_array'], axis = 1)
            self.data[_filename]['Camera']['background_array'] = np.flip(self.data[_filename]['Camera']['background_array'], axis = 1)

        if 'y' in _flip:
            self.data[_filename]['Camera']['signal_array'] = np.flip(self.data[_filename]['Camera']['signal_array'])
            self.data[_filename]['Camera']['background_array'] = np.flip(self.data[_filename]['Camera']['background_array'])

        if _read_calibration and not single_image:
            self.data[_filename]['Camera']['calibration_array'] = np.divide(self.data[_filename]['Camera']['calibration_array'], self.data[_filename]['Camera']['metadata']['shots'])*_intensity_rescale
            if 'x' in _flip:
                self.data[_filename]['Camera']['calibration_array'] = np.flip(self.data[_filename]['Camera']['calibration_array'], axis = 1)

            if 'y' in _flip:
                self.data[_filename]['Camera']['calibration_array'] = np.flip(self.data[_filename]['Camera']['calibration_array'])
            
        if _background_filename == None:
            self.data[_filename]['Camera']['corrected_array'] = self.data[_filename]['Camera']['signal_array'] - self.data[_filename]['Camera']['background_array'] + _intensity_offset
        else:
            self.data[_filename]['Camera']['corrected_array'] = self.data[_filename]['Camera']['signal_array'] - self.data[_filename]['Camera']['background_array'] - self.MinderViezeOorSmear(self.data[_background_filename]['Camera']['corrected_array'], _background_smear, _background_smear)*self.data[_filename]['Camera']['metadata']['shots']/self.data[_background_filename]['Camera']['metadata']['shots'] + _intensity_offset
        if _per_shot:
            if _background_filename == None:
                for shot in range(self.data[_filename]['Camera']['metadata']['shots']):
                    self.data[_filename]['Camera']['corrected_arrays'][shot] = self.data[_filename]['Camera']['signal_arrays'][shot] - self.data[_filename]['Camera']['background_arrays'][shot]
            else:
                smeared_background = self.MinderViezeOorSmear(self.data[_background_filename]['Camera']['corrected_array'], _background_smear, _background_smear)
                for shot in range(self.data[_filename]['Camera']['metadata']['shots']):
                    self.data[_filename]['Camera']['corrected_arrays'][shot] = self.data[_filename]['Camera']['signal_arrays'][shot] - self.data[_filename]['Camera']['background_arrays'][shot] - smeared_background/self.data[_background_filename]['Camera']['metadata']['shots']
        
        
        return self.data[_filename]
    def read_simulated_data(self, filename, overwrite=False, _directory=''):
        Data = np.load(file=filename)
        
        # Initialize the key if it doesn't exist
        if filename not in self.data:
            self.data[filename] = {}
            
        # Initialize the 'Simulation' dictionary if it doesn't exist
        if "Simulation" not in self.data[filename]:
            self.data[filename]["Simulation"] = {}
            
        self.data[filename]["Simulation"]['x'] = Data[0]
        self.data[filename]["Simulation"]['y'] = Data[1]
        
        return self.data[filename]
    # Set aspect ratio for _filename to _aspect
    #
    def set_aspect(self, _filename, _aspect = 1):
        found_filename = self.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        found_camera_filename = self.find_filename(_filename)                                                       # If index is given instead of _filename find the corresponding _filename
        if found_camera_filename != "":
            _filename = found_camera_filename

        if _filename not in self.data:                                                                              # If it does not yet exist, add this _filename to the data dictionary
            self.read_camera_data(_filename, _overwrite = True, _aspect = _aspect)
        
        return self.data[_filename]
        
    #
    # Bin camera data in _array with bins of size (_binfactor, _binfactor)
    #
    def bin_camera_array(self, _array, _binfactor):
        if _binfactor == 1:
            return _array
        elif not _array.shape[0]%_binfactor and not _array.shape[0]%_binfactor:
            m_bins = _array.shape[0]//_binfactor
            n_bins = _array.shape[1]//_binfactor
            _array = _array.reshape(m_bins, _binfactor, n_bins, _binfactor).sum(3).sum(1)/_binfactor**2
            print(m_bins, " x ", n_bins, "bins in camera data")
            return np.repeat(np.repeat(_array, _binfactor, axis=0), _binfactor, axis=1)
        else:
            print("Unable to bin a {} array with {}x{} bins".format(_array.shape, _binfactor, _binfactor))
            return _array

    # 
    # Change the _camera_filename under which it is stored in the data directory to _filename. (To link to PicoScope file)
    #
    def rename_camera_data(self, _camera_filename, _filename, _overwrite = False):
        found_filename = self.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        found_camera_filename = self.find_filename(_camera_filename)                                                 # If index is given instead of _filename find the corresponding _filename
        if found_camera_filename != "":
            _camera_filename = found_camera_filename

        if _camera_filename not in self.data:
            self.print("Can not rename _camera ", _camera_filename," entry to ", _filename, ", entry not found in data dictionary.")
            return
        if _filename in self.data:
            if 'Camera' in self.data[_filename] and not _overwrite:
                self.print("Can not rename ", _camera_filename," entry to ", _filename, ", already contains camera data. Call with _overwrite = True to rename anyway.")
                return
        else:
            self.data[_filename] = {}

        self.data[_filename] = self.data[_camera_filename]
        self.data.pop(_camera_filename)
        
        return self.data[_filename]
        
    #
    # Combine camera data in _filenames and store it under the _combined_filename in the data dictionary. The filenames are without .fits, _overwrite if was already loaded. 
    # All metadata is read from the first in _filenames. It is assumed that data is combined for files of the same structure.
    # Camera data is combined weighted by the number of shots in each file.
    # If _filenames = [] it combines all read camera data.
    # Ik _keep_original = False it removes the originals from the data dicionary after combination.
    # For other arguments see read_camera_data, they are used when the camera data was not read yet.
    # If _contributions != None it should be an array on with what weigths to add each of the images
    # _offsets_pixel provides a list of offset in pixels in (x,y) for each _filenames added. Usually the first entry would be (0,0) as the base reference point of the first image. If empty list take all to be (0,0).
    #

    def combine_camera_data(self, _combined_filename = 'combined', _filenames = [], _keep_original = False, _overwrite = False, _directory = '', _order = 'scb', _per_shot = False, _read_calibration = False, _aspect = None, _height_mm = None, _background_filename = None, _background_smear = 8, _contributions = None, _offsets_pixels = []):
        if _filenames == []:                                                                            # If _filenames is empty add all read camera data
            for f in self.data:
                if 'Camera' in self.data[f]:
                    _filenames.append(f)

        for _filenamei, _filename in enumerate(_filenames):
            found_filename = self.find_filename(_filename)                                              # If index is given instead of _filename find the corresponding _filename
            if found_filename != "":
                _filenames[_filenamei] = found_filename
        
        if isinstance(_contributions, type(None)):
            _contributions = np.ones(len(_filenames))
            
        if len(_offsets_pixels) == 0:
            _offsets_pixels = [(0,0)]*len(_filenames)
        _offsets_pixels = np.array(_offsets_pixels)
        
        if _filename not in self.data or 'Camera' not in self.data[_filename]:                          # If it does not yet exist, add this _filename to the data dictionary
            self.read_camera_data(_filename, _overwrite = _overwrite, _directory = _directory, _order = _order, _per_shot = _per_shot, _read_calibration = _read_calibration, _aspect = _aspect, _height_mm = _height_mm, _background_filename = _background_filename, _background_smear = _background_smear)

        if _combined_filename in self.data and not _overwrite:
            self.print("Combined camera data " + _combined_filename + " already exists. If you want to overwrite, call combine_camera_data() with _overwrite = True as an argument.")
            return self.data[_combined_filename]
        
        
        self.data[_combined_filename] = {}
        self.data[_combined_filename]['Camera'] = {}
        self.data[_combined_filename]['Camera']['metadata'] = self.data[_filenames[0]]['Camera']['metadata']
        
        shots = np.array([self.data[_filename]['Camera']['metadata']['shots'] for _filename in _filenames])
        total_shots = sum(shots)
        weights = shots*_contributions # /total_shots
        
        original_shape = np.array(self.data[_filenames[0]]['Camera']['signal_array'].shape)
        new_shape = original_shape + [np.max(_offsets_pixels[:,1])-np.min(_offsets_pixels[:,1]), np.max(_offsets_pixels[:,0])-np.min(_offsets_pixels[:,0])]
        reference_point = np.array([abs(min(np.min(_offsets_pixels[:,1]),0)), abs(min(np.min(_offsets_pixels[:,0]),0))])
        
        self.data[_combined_filename]['Camera']['metadata']['aspect'] = self.data[_filenames[0]]['Camera']['metadata']['aspect']/new_shape[1]*original_shape[1]/original_shape[0]*new_shape[0]
        self.data[_combined_filename]['Camera']['metadata']['height_mm'] = self.data[_filenames[0]]['Camera']['metadata']['height_mm']*new_shape[0]/original_shape[0]
        self.data[_combined_filename]['Camera']['metadata']['width_mm'] = self.data[_filenames[0]]['Camera']['metadata']['width_mm']*new_shape[1]/original_shape[1]
        self.data[_combined_filename]['Camera']['metadata']['shape'] = new_shape
        self.data[_combined_filename]['Camera']['metadata']['horizontal_mm'] = np.linspace(-0.5*self.data[_combined_filename]['Camera']['metadata']['width_mm'],0.5*self.data[_combined_filename]['Camera']['metadata']['width_mm'],self.data[_combined_filename]['Camera']['metadata']['shape'][1])
        self.data[_combined_filename]['Camera']['metadata']['vertical_mm'] = np.linspace(-0.5*self.data[_combined_filename]['Camera']['metadata']['height_mm'],0.5*self.data[_combined_filename]['Camera']['metadata']['height_mm'],self.data[_combined_filename]['Camera']['metadata']['shape'][0])
        
        shots_array = np.zeros(new_shape)
        
        self.data[_combined_filename]['Camera']['signal_array'] = np.zeros(new_shape) #np.sum([weights[_filenamei]*self.data[_filename]['Camera']['signal_array'] for _filenamei, _filename in enumerate(_filenames)], axis = 0)
        self.data[_combined_filename]['Camera']['background_array'] = np.zeros(new_shape)
        self.data[_combined_filename]['Camera']['corrected_array'] = np.zeros(new_shape)

        #print([self.data[_filename]['Camera']['signal_array'].shape for _filename in _filenames])

        for _filenamei, _filename in enumerate(_filenames):
            shots_array[(reference_point[1]+_offsets_pixels[_filenamei,1]):(reference_point[1]+_offsets_pixels[_filenamei,1]+original_shape[1]),(reference_point[0]+_offsets_pixels[_filenamei,0]):(reference_point[0]+_offsets_pixels[_filenamei,0]+original_shape[0])] = np.add(shots_array[(reference_point[1]+_offsets_pixels[_filenamei,1]):(reference_point[1]+_offsets_pixels[_filenamei,1]+original_shape[1]),(reference_point[0]+_offsets_pixels[_filenamei,0]):(reference_point[0]+_offsets_pixels[_filenamei,0]+original_shape[0])],np.ones(original_shape)*shots[_filenamei])
            #print(shots_array)
            #print(_filenamei, _filename, new_shape, original_shape, (self.data[_combined_filename]['Camera']['signal_array'][(reference_point[1]+_offsets_pixels[_filenamei,1]):(reference_point[1]+_offsets_pixels[_filenamei,1]+original_shape[1]),(reference_point[0]+_offsets_pixels[_filenamei,0]):(reference_point[0]+_offsets_pixels[_filenamei,0]+original_shape[0])]).shape, (self.data[_filename]['Camera']['signal_array']*weights[_filenamei]).shape)
            self.data[_combined_filename]['Camera']['signal_array'][(reference_point[1]+_offsets_pixels[_filenamei,1]):(reference_point[1]+_offsets_pixels[_filenamei,1]+original_shape[1]),(reference_point[0]+_offsets_pixels[_filenamei,0]):(reference_point[0]+_offsets_pixels[_filenamei,0]+original_shape[0])] = np.add(self.data[_combined_filename]['Camera']['signal_array'][(reference_point[1]+_offsets_pixels[_filenamei,1]):(reference_point[1]+_offsets_pixels[_filenamei,1]+original_shape[1]),(reference_point[0]+_offsets_pixels[_filenamei,0]):(reference_point[0]+_offsets_pixels[_filenamei,0]+original_shape[0])],self.data[_filename]['Camera']['signal_array']*weights[_filenamei])
            self.data[_combined_filename]['Camera']['background_array'][(reference_point[1]+_offsets_pixels[_filenamei,1]):(reference_point[1]+_offsets_pixels[_filenamei,1]+original_shape[1]),(reference_point[0]+_offsets_pixels[_filenamei,0]):(reference_point[0]+_offsets_pixels[_filenamei,0]+original_shape[0])] = np.add(self.data[_combined_filename]['Camera']['background_array'][(reference_point[1]+_offsets_pixels[_filenamei,1]):(reference_point[1]+_offsets_pixels[_filenamei,1]+original_shape[1]),(reference_point[0]+_offsets_pixels[_filenamei,0]):(reference_point[0]+_offsets_pixels[_filenamei,0]+original_shape[0])],self.data[_filename]['Camera']['background_array']*weights[_filenamei])
            self.data[_combined_filename]['Camera']['corrected_array'][(reference_point[1]+_offsets_pixels[_filenamei,1]):(reference_point[1]+_offsets_pixels[_filenamei,1]+original_shape[1]),(reference_point[0]+_offsets_pixels[_filenamei,0]):(reference_point[0]+_offsets_pixels[_filenamei,0]+original_shape[0])] = np.add(self.data[_combined_filename]['Camera']['corrected_array'][(reference_point[1]+_offsets_pixels[_filenamei,1]):(reference_point[1]+_offsets_pixels[_filenamei,1]+original_shape[1]),(reference_point[0]+_offsets_pixels[_filenamei,0]):(reference_point[0]+_offsets_pixels[_filenamei,0]+original_shape[0])],self.data[_filename]['Camera']['corrected_array']*weights[_filenamei])
            
        self.data[_combined_filename]['Camera']['signal_array'] = np.divide(self.data[_combined_filename]['Camera']['signal_array'], shots_array, out=np.zeros_like(self.data[_combined_filename]['Camera']['signal_array']), where=~np.isclose(shots_array,np.zeros_like(shots_array)))
        self.data[_combined_filename]['Camera']['background_array'] = np.divide(self.data[_combined_filename]['Camera']['background_array'], shots_array, out=np.zeros_like(self.data[_combined_filename]['Camera']['background_array']), where=~np.isclose(shots_array,np.zeros_like(shots_array)))
        self.data[_combined_filename]['Camera']['corrected_array'] = np.divide(self.data[_combined_filename]['Camera']['corrected_array'], shots_array, out=np.zeros_like(self.data[_combined_filename]['Camera']['corrected_array']), where=~np.isclose(shots_array,np.zeros_like(shots_array)))

        if 'calibration_array' in self.data[_filenames[0]]['Camera']:
            self.data[_combined_filename]['Camera']['calibration_array'] = np.zeros(new_shape)
            for _filenamei, _filename in enumerate(_filenames):
                self.data[_combined_filename]['Camera']['calibration_array'][(reference_point[1]+_offsets_pixels[_filenamei,1]):(reference_point[1]+_offsets_pixels[_filenamei,1]+original_shape[1]),(reference_point[0]+_offsets_pixels[_filenamei,0]):(reference_point[0]+_offsets_pixels[_filenamei,0]+original_shape[0])] = np.add(self.data[_combined_filename]['Camera']['calibration_array'][(reference_point[1]+_offsets_pixels[_filenamei,1]):(reference_point[1]+_offsets_pixels[_filenamei,1]+original_shape[1]),(reference_point[1]+_offsets_pixels[_filenamei,0]):(reference_point[0]+_offsets_pixels[_filenamei,0]+original_shape[0])],self.data[_filename]['Camera']['calibration_array']*weights[_filenamei])
            self.data[_combined_filename]['Camera']['calibration_array'] = np.divide(self.data[_combined_filename]['Camera']['calibration_array'], shots_array, out=np.zeros_like(self.data[_combined_filename]['Camera']['calibration_array']), where=~np.isclose(shots_array,np.zeros_like(shots_array)))

            #self.data[_combined_filename]['Camera']['calibration_array'] = np.sum([weights[_filenamei]*self.data[_filename]['Camera']['calibration_array'] for _filenamei, _filename in enumerate(_filenames)], axis = 0)

        #self.data[_combined_filename]['Camera']['corrected_array'] = np.sum([weights[_filenamei]*self.data[_filename]['Camera']['corrected_array'] for _filenamei, _filename in enumerate(_filenames)], axis = 0)

        #if 'corrected_arrays' in self.data[_filenames[0]]['Camera']:
        #    self.data[_combined_filename]['Camera']['corrected_arrays'] = np.concatenate([weights[_filenamei]*self.data[_filename]['Camera']['corrected_arrays'] for _filenamei, _filename in enumerate(_filenames)], axis = 0)
        
        if not _keep_original:
            for _filename in _filenames:                                                                    # Remove camera data from _filename
                self.data[_filename].pop('Camera')
                if not self.data[_filename]:                                                                # Remove if empty
                    self.data.pop(_filename)
        return self.data[_combined_filename]
        
        
    def MinderViezeOorSmear(self, array, width, height):     
        """ Same vieze smear in less time (factor ~1000), note that this function also correctly handles the edges
            time0 = time.time()
            corrected_array2 = HeleViezeOorSmear(corrected_array, 2, 2)
            print("Smear_time", time.time()-time0)
            time0 = time.time()
            corrected_array_mindervies = MinderViezeOorSmear(corrected_array, 2, 2)
            print("Smear_time_minder_vies", time.time()-time0)
            print(np.sum(np.abs(corrected_array2[3:-3,3:-3]-corrected_array_mindervies[3:-3,3:-3])))
        """
        if width == 0 and height ==0:
            return array
        conv_array = np.ones((2*width+1,2*height+1))
        new_array = scipy.signal.convolve2d(array, conv_array, mode="full", boundary="symm")[width:-width,height:-height]      
        return new_array/np.sum(conv_array)

    #
    # Integrate camera data. Sum all counts together in the camera data in the channel _channel of the _filename, weighted by the _mask. If _per_shot the camera data will be stored per shot aswell.
    # Can integrate over selected axis: _axis = None: Both directions, _axis = 0: vertical, _axis = 1: horizontal
    # Stored in the dictionary as data[_filename]['Camera']['integrated_signal'] (and if _per_shot, also data[_filename]['Camera']['integrated_signals'])
    #
    def integrate_camera_data(self, _filename, _overwrite = False, _mask = None, _per_shot = False, _channel = 'corrected_array', _axis = None):
        found_filename = self.find_filename(_filename)                                                              # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        if _filename not in self.data or 'Camera' not in self.data[_filename] or (_per_shot and 'integrated_signals' not in self.data[_filename]['Camera']):                                                                              # If it does not yet exist, add this _filename to the data dictionary
            self.read_camera_data(_filename, _per_shot=_per_shot, _overwrite=True)

        if 'integrated_signal' in self.data[_filename]['Camera'] and not _overwrite:
            self.print("There already is an integrated_signal for " + _filename + ". If you want to overwrite, call integrate_camera_data() with _overwrite = True as an argument.")
            return

        try:
            if _mask == None:
                _mask = np.full_like(self.data[_filename]['Camera'][_channel], 1)
                print(_mask)
        except ValueError:
            pass

        if _mask.shape != self.data[_filename]['Camera'][_channel].shape:
            self.print("Shape of the _mask", _mask.shape, "does not match the shape of the camera data", self.data[_filename]['Camera'][_channel].shape, ", can not integrate_camera_data.")

        self.data[_filename]['Camera']['integrated_signal'] = np.sum(np.multiply(self.data[_filename]['Camera'][_channel], _mask), axis=_axis)

        if _per_shot:
            self.data[_filename]['Camera']['integrated_signals'] = [np.sum(np.multiply(i, _mask), axis=_axis) for i in self.data[_filename]['Camera'][_channel+'s']]
        
        if self.verbose:
            self.print("integrate_camera_data: Integrated camera data for " + str(_filename) + " over axis " + str(_axis) + ".")
        
        return self.data[_filename]

    #
    # Create a mask for the integration of camera data. 
    # _type can be 'ellipse' or 'rectangle'
    # The mask is centered at _position [x,y] and has _dimension [x,y] as radius of half width away from the _position in each direction.
    # _unit can be "mm" for physical dimensions of "pixel" for number of pixels. If it is taken as "mm" the _position and _dimension are taken to be in the physical dimensions of the image in _filename.
    # _invert 
    #
    def create_camera_mask(self, _filename, _type = 'ellipse', _position = [0,0], _dimension = [2, 2], _unit = "mm", _invert = False):
        found_filename = self.find_filename(_filename)                                                  # If index is given instead of _filename find the corresponding _filename
        if found_filename != "":
            _filename = found_filename

        if _filename not in self.data:
            self.read_camera_data(_filename)

        self.data[_filename]['Camera']['mask'] = np.zeros(self.data[_filename]['Camera']['metadata']['shape'])
        if _unit == "mm":
            _position = [(_position[0]- self.data[_filename]['Camera']['metadata']['horizontal_mm'][0])/self.data[_filename]['Camera']['metadata']['width_mm']*self.data[_filename]['Camera']['metadata']['shape'][1], (_position[1] - self.data[_filename]['Camera']['metadata']['vertical_mm'][0])/self.data[_filename]['Camera']['metadata']['height_mm']*self.data[_filename]['Camera']['metadata']['shape'][0]]
            _dimension = [_dimension[0]/self.data[_filename]['Camera']['metadata']['width_mm']*self.data[_filename]['Camera']['metadata']['shape'][1], _dimension[1]/self.data[_filename]['Camera']['metadata']['height_mm']*self.data[_filename]['Camera']['metadata']['shape'][0]]
        
        #print(self.data[_filename]['Camera']['metadata']['shape'])
        if _type == 'ellipse':
            for x in range(self.data[_filename]['Camera']['metadata']['shape'][1]):
                for y in range(self.data[_filename]['Camera']['metadata']['shape'][0]):
                    self.data[_filename]['Camera']['mask'][y,x] = (((x - _position[0])**2/_dimension[0]**2+(y - _position[1])**2/_dimension[1]**2 < 1) ^ _invert)
        elif _type == 'rectangle':
            for x in range(self.data[_filename]['Camera']['metadata']['shape'][1]):
                for y in range(self.data[_filename]['Camera']['metadata']['shape'][0]):
                    self.data[_filename]['Camera']['mask'][y,x] = ((abs(x - _position[0]) < _dimension[0] and abs(y - _position[1]) < _dimension[1])  ^ _invert)

        return self.data[_filename]['Camera']['mask']       
        
    #
    # Average over _number_entries consecutive values in the _array and return. If len(_array) is not a multiple of _number_entries the last are dropped
    # Rebinned to return array of length floor(len(_array)/_number_entries)
    #
    def average_consecutive(self, _array, _number_entries):
        if _number_entries == 1:
            return _array
        else:
            return sum(_array[:(len(_array)-len(_array)%_number_entries)][x::_number_entries] for x in range(_number_entries)) / _number_entries

    #
    # Average over _number_entries consecutive values in the _array and return. 
    # Takes running average to return array of length len(_array) - _number_entries + 1
    #    
    def running_average(self, _array, _number_entries):
        if _number_entries == 1:
            return np.array(_array)
        else:
            return_array = np.empty(len(_array) - _number_entries + 1)
            for i in range(len(_array) - _number_entries + 1):
                return_array[i] = np.mean(_array[i:i+_number_entries])
            return return_array
        
   # def read_simulation_file(self,_filename):
   #     Data=np.load(_filename)
   #     self.data=Data
   #     return self.data


# Default function to test the program
'''
if __name__ == '__main__':
    dataManager = DataManager()
    dataManager.read_all()
    self.print(dataManager.data)
    self.print(dataManager.waveforms)

'''