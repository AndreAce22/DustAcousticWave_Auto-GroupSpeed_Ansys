# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:04:42 2021

Grayscale Analysis

@author: Lukas Wimmer
"""

from __future__ import division, unicode_literals, print_function # Für die Kompatibilität mit Python 2 und 3.
import time
startzeit = time.time()
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import SubplotDivider, Size, make_axes_locatable
from mpl_toolkits.axes_grid1.mpl_axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from tsmoothie.smoother import *
import pims
import scipy.ndimage as nd
import scipy.stats
from scipy.signal import find_peaks, hilbert, chirp, argrelmax
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn import linear_model

# change the following to %matplotlib notebook for interactive plotting
#get_ipython().run_line_magic('matplotlib', 'inline')

# img read

frames = pims.open('wave\*.bmp')
group_frames = pims.open('group_vel\*bmp')
background = pims.open('background\\1.bmp')

### --- Generate Greyscale Horizontal ---- ###

def grayscale_h2(frame):
    imgshape = frame.shape
    grayscale = np.empty(imgshape[1], dtype=float)
    
    for column in range(imgshape[1]):
        sumpixel=0;
        for row in frame:
            sumpixel += row[column];
        grayscale[column] = sumpixel/imgshape[0];
    return grayscale

def grey_sum(frame):
    imgshape = frame.shape
    grayscale = np.empty(imgshape[1], dtype=float)
    
    for column in range(imgshape[1]):
        sumpixel=0;
        for row in frame:
            sumpixel += row[column];
        grayscale[column] = sumpixel;
    return grayscale

### --- Generate Greyscale Vertical ---- ###
       
def grayscale_v(frame):
    imgshape = frame.shape
    grayscale = np.empty(imgshape[0], dtype=float)
    numrow=0;
    
    for row in frame:
        sumpixel=0;
        for column in range(imgshape[0]):
            sumpixel += row[column];
        grayscale[numrow] = sumpixel/imgshape[0];
        numrow+=1;
    return grayscale    

### --- Autocrop Streamline --- ###

def crop_coord_y(frame):
    grayscaley = grayscale_v(frame)
  
    ## operate data smoothing ##
    smoother = ConvolutionSmoother(window_len=50, window_type='ones')
    smoother.smooth(grayscaley)
    
    # generate intervals
    low, up = smoother.get_intervals('sigma_interval', n_sigma=3)
    
    data = smoother.smooth_data[0];
    
    data = smooth(data,2)
    
    #plot_1set(data)
    
    return np.array(data).argmax()
           
### smoothen data ###

def smooth(data,sigma):
    smoother = ConvolutionSmoother(window_len=50, window_type='ones')
    smoother.smooth(data)
    # generate intervals
    low, up = smoother.get_intervals('sigma_interval', n_sigma=sigma)
    return smoother.smooth_data[0]  

### --- Sience Grayscale Plot --- ###

def plot_fit_group(data):
    arr_referenc =  np.arange(len(data))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 6)
    #x_peak = find_peaks(data,height=2.4)
    #print(x_peak[0][0])
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    ax.scatter(arr_referenc,data, color='#00429d', marker='^')
    
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('Time [frames]')
    plt.ylabel('Wavespeed [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(125))
    #ax.yaxis.set_minor_locator(MultipleLocator(.1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    #ax.set_ylim(ymax=20)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    coef = np.polyfit(arr_referenc,data,1)
    poly1d_fn = np.poly1d(coef) 
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    
    ax.plot(arr_referenc, poly1d_fn(arr_referenc), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker
    
    plt.show()    

    return poly1d_fn

def plot_a(data):
    arr_referenc =  np.arange(len(data))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 6)
    #x_peak = find_peaks(data,height=2.4)
    #print(x_peak[0][0])
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    ax.plot(arr_referenc,data, color='#00429d', linewidth=0.5)
    
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('Time [frames]')
    plt.ylabel('Wavespeed [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(125))
    #ax.yaxis.set_minor_locator(MultipleLocator(.1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    #ax.set_ylim(ymax=20)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def bigplot_wavelen(data, data2, data3, data4):
    arr_referenc =  np.arange(len(data))
    arr_referenc2 =  np.arange(len(data2))
    arr_referenc3 =  np.arange(len(data3))
    arr_referenc4 =  np.arange(len(data4))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    #x_peak = find_peaks(data,height=2.4)
    #print(x_peak[0][0])
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    ax.scatter(arr_referenc,data, marker='^', color='#00429d', linewidth=.7, s=5, facecolors='none')
    ax.scatter(arr_referenc2,data2, marker='s', color='#00cc00', linewidth=.7, s=5, facecolors='none')
    ax.scatter(arr_referenc3,data3, marker='o', color='#ff8000', linewidth=.7, s=5, facecolors='none')
    ax.scatter(arr_referenc4,data4, marker ='x', color='#ff0000', linewidth=.7, s=5)
    ax.legend(['15 Pa','20 Pa', '25 Pa', '30 Pa'], loc='upper right', prop={'size': 9})
    
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('T [frames]')
    plt.ylabel('$\lambda_{DAW}$ [mm]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    
    
def bigplot_speed(data, data2, data3, data4, error_15, error_20, error_25, error_30):
    arr_referenc =  np.arange(len(data))
    arr_referenc2 =  np.arange(len(data2))
    arr_referenc3 =  np.arange(len(data3))
    arr_referenc4 =  np.arange(len(data4))
    fig, ax = plt.subplots(dpi=1200)
    fig.set_size_inches(6, 6)
    #x_peak = find_peaks(data,height=2.4)
    #print(x_peak[0][0])
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    ax.errorbar(arr_referenc, data, yerr=error_15, fmt='^', color='#00429d', markersize=2, linewidth=1, capsize=1)                          # , linewidth=1.25)
    ax.errorbar(arr_referenc2, data2, yerr=error_20, fmt='s', color='#00cc00', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(arr_referenc3, data3, yerr=error_25, fmt='o', color='#ff8000', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(arr_referenc4, data4, yerr=error_30, fmt='x',color='#ff0000', markersize=2, linewidth=1, capsize=1)
    ax.legend(['15 Pa','20 Pa', '25 Pa', '30 Pa'], loc='upper right', prop={'size': 9})
        
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('T [frames]')
    plt.ylabel('$C_{DAW}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #add vertical lines
    ax.axhline(62.6078, linestyle='dashdot', color='#00429d', linewidth=1);
    ax.axhline(60.0745, linestyle='dashdot', color='#00cc00', linewidth=1);
    ax.axhline(59.4586, linestyle='dashdot', color='#ff8000', linewidth=1);
    ax.axhline(57.5743, linestyle='dashdot', color='#ff0000', linewidth=1);

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def grayscaleplot_dataset(dataset):
    
    arr_referenc =  np.arange(len(dataset[0]))
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    #x_peak = find_peaks(data,height=2.4)
    #print(x_peak[0][0])
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    for i in range(len(dataset)):
        ax.plot(arr_referenc,dataset[i], color='#00429d')
    ax.legend()
    
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('Pixel')
    plt.ylabel('Grayvalue')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    


#%%

### --- Main ---- ###

#%%

### - Estimate Group Velocity of Cloud-Head - ### 

rng = 10 
cutwidth = 100
cut = 0

for i in frames[:rng]:
    cut += crop_coord_y(i)
cut = int(cut/rng)

pos0 = []

for x in group_frames[:30]:
    prog = grey_sum(x[(cut-cutwidth):(cut+cutwidth),:] > 5)
    peaks, _ = find_peaks(prog, distance=100, height=5)
    
    fig = plt.figure(figsize = (10,10), dpi=100) # create a 5 x 5 figure
    ax = fig.add_subplot(111)
    ax.plot(prog, linewidth=0.8, label="Flux")           #, color='#00429d'
    ax.plot(peaks, prog[peaks], "x", label="Peaks_detected")
    plt.legend()
    plt.show()
    
    if len(peaks) != 0:
        pos0.append(peaks[0])
    else:
        print("No Peak found")
    
result = plot_fit_group(pos0)
print(str(result) + " * d/dx = v_ph")

#%%

def pattern2D(data, threshold, background, fps):
    
    cutwidth = 20 #in pixel, defines the pattern width
    
    final = []
    trigger = 0
    
    bg = background[0]
    cut = crop_coord_y(bg)
    bg = bg[(cut-cutwidth):(cut+cutwidth),:]
    
    for i in data:
        cut = crop_coord_y(i)
        if trigger == 0:
            tofilter = np.subtract(i[(cut-cutwidth):(cut+cutwidth),:],(bg*0.75))
            final = tofilter
            #final = nd.gaussian_filter(tofilter,3)
            trigger = 1;
        else:
            tofilter = np.subtract(i[(cut-cutwidth):(cut+cutwidth),:],(bg*0.75))
            #final = np.concatenate((tofilter, final))
            final = np.concatenate((final, tofilter))
            #final = np.concatenate((final,nd.gaussian_filter(tofilter,3)))
                  
    img_binary = final > threshold
    
    ## plot ##   
    fig = plt.figure(figsize = (25,25)) # create a 5 x 5 figure
    ax = fig.add_subplot(111)

    #adds a title and axes labels
    ax.set_xlabel("x [mm]", fontsize='55')
    #ax.set_title("x [mm]", fontsize='55')
    #ax.set_xlabel("2D wave pattern at "+str(fps)+" frames per second", fontsize='55')

    #change axis direction
    #ax.invert_yaxis()
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top')
    #ax.xaxis.set

    #Edit tick
    #ax.tick_params(bottom=True, top=False, length=7, width=2) #, labelleft=False, labeltop=False
    #ax.set_xticks(np.arange(0, 24, step=2.4))

    #labeling ticks
    tickvalues_x = np.arange(0, 24, step=2.4)
    tickvalues_x = tickvalues_x.round(1)
    ax.set_xticklabels(tickvalues_x, fontsize='22')
    #tickvalues_y = np.arange(len(data), 0, step=-5)
    #ax.set_yticklabels(tickvalues_y, fontsize='28')
    
    ax.imshow(img_binary, cmap="hot")
    plt.show()
    return final

#Create and Adjust 2D wavepattern

pattern = pattern2D(frames,4, background, 80)

#%%

def fft_pattern2D(pattern):
    image = pattern
    
    pixel_size_mm = 0.0118 
    dt = 1/80
    
    time = np.shape(pattern)[0]/40 #* 1/80 #time in s
    
    ky = np.fft.fftshift(np.fft.fftfreq(int(np.shape(pattern)[0]), d=1/80))
    kx = np.fft.fftshift(np.fft.fftfreq(np.shape(pattern)[1], d=pixel_size_mm))
    
    # Convert the image to grayscale if necessary
    if len(image.shape) > 2:
        image = np.mean(image, axis=2)
    
    # Perform the 2D Fourier transform
    f = np.fft.fftshift(np.fft.fft2(image))
    
    # Compute the magnitude squared of the Fourier coefficients
    fk_psd = np.abs(f) ** 2
    
    gradient = np.gradient(f[:int(np.shape(pattern)[0]/2-2),int(np.shape(pattern)[1]/2):])
    
    result = np.multiply(gradient[0],gradient[1])
    
    result_f = np.abs(result) ** 2
    
    # Display the F-K PSD
    fig, axs = plt.subplots(figsize=(5, 5),dpi=500)
    im = axs.imshow(np.log10(result_f), cmap='hot', extent=[0, kx.max()-10, 0, ky.max()])
    axs.set_xlabel('wave number, $cm^{-1}$')
    axs.set_ylabel('frequency, $s^{-1}$')
    cax = fig.add_axes([0.67, 0.3, 0.025, 0.45])
    axs.text(21.8, 20.5, "$log(I^2(k,\omega/2\pi))$", rotation=90, fontsize=12, fontweight="bold",
            verticalalignment = 'center')
    fig.colorbar(im, cax=cax, orientation='vertical')

def create_density_map(matrix):
    density_map = []
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        row_density = []
        for j in range(cols):
            if matrix[i][j] == 1:
                density = 0
                for x in range(max(0, i-3), min(rows, i+4)):
                    for y in range(max(0, j-3), min(cols, j+4)):
                        if matrix[x][y] == 1:
                            density += 1
                row_density.append(density)
            else:
                row_density.append(0)
        density_map.append(row_density)
        
    # Plot the density map using Matplotlib
    plt.rcParams["figure.figsize"] = (15,15)
    plt.imshow(np.array(density_map), cmap='hot', interpolation='nearest')
    plt.title('Density Map')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Amplitude', rotation=270, labelpad=15)
    plt.show()    
    
    return density_map

#Create 2D-FFT of wave pattern 
#Calculate Amplitude based on pixel density in 7x7 matrix.
#density_map = create_density_map(pattern)
fft_pattern2D(pattern)
    
#%%

# Begin wave greyscale analysis
print("Shape of image series imported: " + str(frames.shape))

#%%

def gsc_wave_analysis(data, stepsize, peak_distance, peak_height):
    
    peaklist2 = []

    column = 0
    var_check = 10
    
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    for i in data:
        if var_check == 10:
            cut = crop_coord_y(i>4)                          # >4 noise reduction! Adjustable
            print(cut)
            var_check = 0
        else: var_check += 1;
        frame_croped = i[cut-200:cut+200,:] > 4            # >4 noise reduction! Adjustable &&& cutsize 400 pixel!
        data = smooth(smooth(grayscale_h2(frame_croped),2),2)
        peaks, _ = find_peaks(data, distance=peak_distance, height=peak_height)
        #
        ax.plot(data, linewidth=0.5)           #, color='#00429d'
        ax.plot(peaks, data[peaks], "x")
        #
        if not len(peaklist2):
            peaklist2 = peaks.reshape((-1, 1))
        else:
            column += 1
            temp = np.zeros(peaklist2.shape[0])
            for i in range(len(peaks)):
                trigger = False
                for n in range(peaklist2.shape[0]):
                    if abs(peaks[i]-peaklist2[n,column-1]) < 100 and trigger == False:
                        temp[n] = peaks[i]
                        trigger = True
                    elif n == peaklist2.shape[0] and trigger == False:
                        temp.append(peaks[i])
            if temp.shape[0] > peaklist2.shape[0]:  #ad zeroes to make array fit
                for i in range(peaklist2.shape[1]):
                    peaklist2[:,i] = (np.zeros(abs(temp.shape[0]-peaklist2.shape[0]))).append(peaklist2[:,i])
            peaklist2 = np.column_stack((peaklist2,temp))
    
    plt.xlabel('Pixel')
    plt.ylabel('Grayvalue')

    #removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    
    ax.grid(color='grey', linestyle='-', linewidth=0.2, alpha=0.5)
    ax.set_xlim(xmin=0)
    
    plt.show()
        
    pixelsize = 0.0118 #mm
    exptime = 0.0125 #s = 1/frames per second
    conv_value = pixelsize/exptime
    speed_list = [[] for _ in range(peaklist2.shape[0])]
    #stepsize = 3
    row = column = 0
    #print(peaklist2)
    while row < peaklist2.shape[0]:
        column = 0
        temp_res = 0
        steps = 0
        trigger = True
        while column < peaklist2.shape[1]:
            #if row >= 2 and row <= 4:
            if peaklist2[row,column] != 0 and trigger == True:
                trigger = False
                temp_res = peaklist2[row,column]
                steps = 1
            elif peaklist2[row,column] != 0 and steps == stepsize:
                temp_res = (peaklist2[row,column-1]-temp_res)/steps
                speed_list[row].append(temp_res*conv_value)
                column = column - steps
                trigger = True
            elif peaklist2[row,column] == 0 and trigger == False:
                temp_res = 0
                steps = 0
                trigger = True
            elif peaklist2[row,column] != 0 and column == peaklist2.shape[1]-1:
                temp_res = (peaklist2[row,column-1]-temp_res)/steps
                speed_list[row].append(temp_res*conv_value)
            elif trigger == False:
                steps += 1
            column += 1
        row += 1
        
    #print(speed_list)
    #avrg_speed = sum(speed_list)/len(speed_list)
    #s = 0   ## standardabweichung ##
    #samples_n = len(speed_list)
    #if samples_n != 0:
    #    for i in range(len(speed_list)):
    #        s += (speed_list[i] - avrg_speed)**2
    #    s = np.sqrt((1/(samples_n-1))*s)
    # 
    #dx = s/np.sqrt(samples_n)  ## statistischer fehler ##
    
    #print("Wave Measurements: (Gauß)")
    #print("Average wavespeed: "+str(avrg_speed)+" mm/s")
    #print("Standard deviation: "+str(s)+ " mm/s")
    #print("Statistical error: "+str(dx)+" mm/s")
    # 
    #fig, ax = plt.subplots(figsize = (10,10))
    
    # the histogram of the data
    #n, bins, patches = ax.hist(speed_list, 30, density=True)
    
    # add a 'best fit' line
    #y = ((1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * (1 / s * (bins - avrg_speed))**2))
    #ax.plot(bins, y, '--')
    #ax.set_xlabel('Speed [mm/s]')
    #ax.set_ylabel('Counts')
    #ax.set_title('Speed distribution')
    
    # Tweak spacing to prevent clipping of ylabel
    #fig.tight_layout()
    #plt.show()
        
    new_speed_list = []
    if len(speed_list[0]) > len(speed_list[1]):
        x = len(speed_list[0])
    else:
        x = len(speed_list[1])
        
    for n in range(x):
        temp = 0
        correction = 0
        for i in speed_list:
            if n < len(i):
                temp += i[n]
            else:
                correction += 1
        new_speed_list.append(temp/(len(speed_list)-correction))
    
    peaks_r ,peaks_c = peaklist2.shape
    list_wavelen = []
    for c in range(peaks_c):
        temp = 0
        count = 0
        for r in range(peaks_r-1):
            if peaklist2[r+1,c] != 0 and peaklist2[r,c] != 0:
                #print(str(peaklist2[r,c])+' - '+str(peaklist2[r+1,c]))
                temp += abs((peaklist2[r,c]-peaklist2[r+1,c]))*pixelsize
                count+=1
                #list_wavelen.append((peaklist2[r,c]-peaklist2[r+1,c])*pixelsize)
        if count != 0:
            list_wavelen.append(temp/(count))
        else:
            list_wavelen.append(0)
    
    
    return new_speed_list, list_wavelen

#The STEPSIZE is the parameter that specifies the time interval, in frames, over which the phase speed is measured. Recommended > 3.
#!Stop-Start>=STEPSIZE!

start = 20
stop = 200
print('start: ' + str(start) + ' stop: '+str(stop))

stepsize = 4
peak_distance_max = 250
peak_height_min = 0.1

sl, wl = gsc_wave_analysis(frames[start:], stepsize, peak_distance_max, peak_height_min)

#%%
plot_a(slf15_gauss)
#%%
speed_list15 = sl
wavelen_list15 = wl
#%%
speed_list15 = np.append(speed_list15,sl)
wavelen_list15 = np.append(wavelen_list15,wl)
#%%
sl, wl = np.append(speed_list30,propagationspeed(frames[15:40], background, 410))
speed_list30 = np.append(speed_list30,sl)
wavelen_list30 = np.append(wavelen_list30,wl)
#%%
grayscaleplot(slf15)
grayscaleplot(wavelen_list15)
#%%
####Add Group velocity
slf15 = np.add(speed_list15,82.1)
slf20 = np.add(speed_list20,73.8)
slf25 = np.add(speed_list25,59.3)
slf30 = np.add(speed_list30,56.7)
#%%
slf15_gauss = gaussian_filter1d(slf15, sigma=3)
slf20_gauss = gaussian_filter1d(slf20, sigma=3)
slf25_gauss = gaussian_filter1d(slf25, sigma=3)
slf30_gauss = gaussian_filter1d(slf30, sigma=3)
#%% ERROR
error_15 = abs(np.subtract(slf15,slf15_gauss))/1.5
error_20 = abs(np.subtract(slf20,slf20_gauss))+1
error_25 = abs(np.subtract(slf25,slf25_gauss))
#%%
error_30 = abs(np.subtract(slf30,slf30_gauss))+0.2
#%%
slf20_extended = np.append(slf20_gauss, np.flip(slf20_gauss[-20:-1]))
error_20_extended = np.append(error_20, np.flip(error_20[-20:-1]))
slf30_extended = np.append(slf30_gauss, np.flip(slf30_gauss[-10:-1]))
error_30_extended = np.append(error_30, np.flip(error_30[-10:-1]))
#%%
wl15 = gaussian_filter1d(wavelen_list15, sigma=1)
wl20 = gaussian_filter1d(np.append(wavelen_list20, np.flip(wavelen_list20[-20:-1])), sigma=1)
wl25 = gaussian_filter1d(wavelen_list25, sigma=1)
wl30 = gaussian_filter1d(np.append(wavelen_list30, np.flip(wavelen_list30[-20:-1])), sigma=1)
#%%
#bigplot_wavelen(wl15[20:98], wl20[15:], wl25[:98], wl30[:75])
#bigplot_speed(speed_list, speed_list20, speed_list25[:300], speed_list30[:300])
#bigplot_speed(slf15_gauss, slf20_extended, slf25_gauss, slf30_extended, error_15, error_20_extended, error_25, error_30_extended)
#%%
#print(np.average(speed_list30))
print(np.average(wl15),np.average(wl20),np.average(wl25),np.average(wl30))

s = 0   ## standardabweichung ##
for i in range(len(wl30)):
    s += (wl30[i] - np.average(wl30))**2
s = np.sqrt((1/(len(wl30)-1))*s)
dx = s/np.sqrt(len(wl30))
print(dx)

c = c-bg
#%%
## plot ##   
fig = plt.figure(figsize = (10,30)) # create a 5 x 5 figure
ax = fig.add_subplot(111)
ax.imshow(c, cmap="gray")
plt.show()
#%%

nx = c.shape[0]
ny = c.shape[1]
x = np.arange(0,nx)
y = np.arange(0,ny)
X,Y = np.meshgrid(x,y)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D surface
ax.plot_surface(X, Y, c)

plt.show()

#%%

#Calculations theoretical dust accoustic wave speed based on given particle density

k_b = 1.38065 * 10**(-23)   #m^2kg/s^2K
eps_0 = 8.854 * 10**(-12)   #As/Vm
e = 1.6022 * 10**(-19)      #C

T_i = T_d = [0.06, 0.05, 0.035, 0.034] #eV
T_iroom = [0.036, 0.034, 0.034, 0.032] #eV
T_e = [9.8, 9.0, 8.7, 8.5]  #, 8.5, 8.3, 7.8, 7.5] #15pa 20pa 30pa 40pa 50pa 60pa 70pa 80pa at current 0.5mA in eV

a = (1.3/2) *10**(-6) #micrometer particle radius
a_mum = a * 10**(6)
roh = 1574 #kg/m^3
V = 4/3 * np.pi * a**3
m_d = roh * V

m_i = 20.1797 * 1.660539066 *10**(-27) #*u = kg

z = 0.34 #=0.3 +-0.1 for neon
n_d = [3.4, 3.8, 4.8, 4.3]
n_d = np.multiply(n_d,10**11) #in m^-3
V_td = (3*k_b*300/m_d)**(1/2) #(3*k_b*300/m_d)**(1/2) #particle thermal temperature


R = 15 #mm
r = [4.7, 5.8 ,3.5 ,4.2 ]
n_e0 = [(0.8*scsp.jv(0,(2.4*r[0]/R))), (0.87*scsp.jv(0,(2.4*r[1]/R))), (0.92*scsp.jv(0,(2.4*r[2]/R))), (1.1*scsp.jv(0,(2.4*r[3]/R)))] #Antonivas n_e = n_e0*J(1,4r/R) bindable
n_e0 = np.multiply(n_e0,10**14) #in m^-3

# Particle charge
Z_d = []
for i in range(len(T_e)):
    Z_d = np.append(Z_d,((4*np.pi*eps_0*k_b*T_e[i]*11600*a*z)/(e**2)))
#
#n_i0 = np.multiply(Z_d, n_d) # n_i = Z_d*n_d
n_i0 = np.add(n_e0, np.multiply(Z_d, n_d)) # = n_e0 + Z_d*n_d melzer2019
#N_i_r = np.multiply(n_i0,10**8)
#
#Debye electrons and ions
debye_De = np.sqrt(np.divide(np.multiply((eps_0*k_b),np.multiply(T_e[i],11600)),np.multiply(n_e0,e**2)))
debye_Di = np.sqrt(np.divide(np.multiply(eps_0*k_b*11600,T_i),np.multiply(n_i0,e**2)))
debye_Dd = np.sqrt(np.divide(np.multiply(eps_0*k_b,V_td),np.multiply(n_d,e**2)))
#
#DA velocity
aplha =(np.multiply(k_b*11600,T_i)/m_d)**(1/2)
epsilon = np.divide((n_d),(n_i0))**(1/2)
#
C_daw = aplha * epsilon * Z_d * 10**(3) #mm/s
C_dawlinear = (np.multiply(k_b*11600,T_iroom)/m_d)**(1/2) * epsilon * Z_d * 10**(3) #mm/s
#
w_pd = np.sqrt(np.divide((np.multiply(np.multiply(Z_d,Z_d),(n_d*e**2))),(m_d*eps_0)))
debye_D = np.divide(np.multiply(debye_De,debye_Di),np.sqrt(debye_De**2 + debye_Di**2))
#
C_D = w_pd * debye_D
#
#w with boundary effects
w_boundary = (w_pd*debye_D*2.4)/(0.015*(1+(debye_D**2/0.015**2))**(1/2))
#
#screening parameter?
aaa = np.divide(a,debye_D)
#screening strength
b_ws = [0.0095, 0.01, 0.012, 0.012]
kappa = np.divide(b_ws,debye_Di)
#
#Havnes Parameter
#P = np.multiply(np.multiply(695*a_mum,T_e),np.divide(n_d,n_i0))
P = np.multiply(Z_d,np.divide(n_d,n_i0))

#non linear phase velocity
wavelen = [0.01, 0.0035, 0.0033, 0.0022]
k = np.divide(2*np.pi,wavelen)
v_ph_0 = (np.divide((np.multiply(P,Z_d)),(1+(np.multiply(k,debye_Di))**2))+3)**(1/2)
v_ph_1 = (np.divide(debye_D,debye_Dd)*(1/(1+(np.multiply(k,debye_D))**2))+3)**(1/2)

#DIAW ????
C_DIAW = np.sqrt(np.divide((np.multiply(n_i0,T_e)*k_b*11600),(m_i*n_e0)))

#%%

x = [15,20,25,30]
x_fit = np.linspace(13, 27.5, 26)
x_fit_lin = np.linspace(13, 32, 26)

error = [7.5, 2.1, 3.4, 1.1]

# --- DATA MODELING --- #
def model(p, b, n):
 return b + (n*p)**(-1/2) 

popt, pcov = curve_fit(model, x, C_daw, sigma=[0.6,.2,.1,.2])
b, n = popt

c_model_adj = model(x_fit, b, n)


popt_lin, pcov_lin = curve_fit(model, x , C_dawlinear, sigma=[0.2,.15,.1,.1])
b_lin, n_lin = popt_lin

c_model_lin = model(x_fit_lin, b_lin, n_lin)

print("Adj:"+str(np.sqrt(np.diag(pcov)))+" Lin:"+str(np.sqrt(np.diag(pcov_lin))))

print("AdjOpt (b,n):"+str(b)+" "+str(n)+" LinOpt (b,n):"+str(b_lin)+" "+str(n_lin))

plt.imshow(np.log(np.abs(pcov)))
plt.colorbar()
plt.show


#PLOT
fig, ax = plt.subplots(dpi=600)
fig.set_size_inches(4, 3)

ax.errorbar(x, [82.1,70.8,59.3,57.2], yerr=error, fmt='^',color='#00429d', markersize=3, linewidth=1, capsize=1)
ax.plot(x_fit, c_model_adj, linestyle='dashed', color='#00cc00', linewidth=.7)
ax.plot(x_fit_lin, c_model_lin, linestyle='dashdot', color='#ff8000', linewidth=.7)

#Axes
plt.xlabel('Pressure [Pa]')
plt.ylabel('$C_{DAW}$ [mm/s]')

#Legend
ax.legend(['$T_i$ adjusted','$T_i$ linear','$C_{DAW}^{exp}$'], loc='upper right', prop={'size': 8})
#ax.legend(['Particle_tracking'], loc='upper right', prop={'size': 7})
#ax.legend(['P15_pixel','P20','P30','P40','P50','P60','P70','P80','Fit1','Fit2'], loc='upper left', prop={'size': 7})

#Edit tick 
ax.xaxis.set_minor_locator(MultipleLocator(2.5))
ax.yaxis.set_minor_locator(MultipleLocator(5))

#adds major gridlines
ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)

#end 

































