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
from sklearn import linear_model

# change the following to %matplotlib notebook for interactive plotting
#get_ipython().run_line_magic('matplotlib', 'inline')

# img read
frames = pims.open('my_directory/30Pa140323/*.bmp')
background = pims.open('my_directory/group_vel/no/background.bmp')

#frame = frames[50]
#plt.imshow(frame)


### --- Generate Greyscale Horizontal ---- ###

def grayscale_h(frame):
    imgshape = frame.shape
    grayscale = np.empty(imgshape[1], dtype=float)
    
    for column in range(imgshape[1]):
        sumpixel=0;
        for row in frame:
            sumpixel += row[column];
        grayscale[column] = sumpixel/imgshape[0];
    grayscale = grayscale.reshape((1, -1))
    return grayscale

def grayscale_h2(frame):
    imgshape = frame.shape
    grayscale = np.empty(imgshape[1], dtype=float)
    
    for column in range(imgshape[1]):
        sumpixel=0;
        for row in frame:
            sumpixel += row[column];
        grayscale[column] = sumpixel/imgshape[0];
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
    
def crop_streamline(frame):
    grayscaley = grayscale_v(frame)
  
    ## operate data smoothing ##
    smoother = ConvolutionSmoother(window_len=30, window_type='ones')
    smoother.smooth(grayscaley)

    # generate intervals
    low, up = smoother.get_intervals('sigma_interval', n_sigma=3)
    
    data = smoother.smooth_data[0];
    
    ## -- find peak -- ## (locate streamline)
    ## integratie ##
    #integgration stepsize
    stepsize = 50
    len_dataset = len(data)/stepsize
    I_temp = np.arange(0.0, len_dataset)
    index = 0;
    temp = 0;
    trigger = 0;
    for i in range(len(data)):
        temp += data[i];
        trigger += 1;
        if trigger == stepsize:
            I_temp[index] = temp; 
            temp = 0
            trigger = 0
            index += 1
    if temp != 0:
        I_temp[index] = temp
        temp = 0

    ## find peak ##
    peak = 0
    for i in I_temp:
        if i > peak:
            peak = i;
    print("Streamline peak horizontal @ "+str(peak*stepsize))
   
     ## find valley ##
    templeft = 0;
    tempright = 0;
    counterleft = 0;
    counterright = 0;
    #the expansion valeu ist softening the cutwith of the streamline peak. (The higher the value the wider the cut, not linear!)
    #use small values for defined/narrow streamlines expansion <= 3 
    expansion = 3
 

    for i in range(int(len_dataset)-3):
        if I_temp[i] == peak:
            left = i-1
            right = i+1
            templeft = I_temp[i] - I_temp[i-1]
            tempright = I_temp[i] - I_temp[i+1]
            n = i
            while counterleft < expansion:
                compare = abs(I_temp[n-1] - I_temp[left-1])
                if templeft > compare:
                    counterleft += 1
                    n -= 1
                    left -= 1
                else:
                    templeft = compare
                    n -= 1
                    left -= 1
                    counterleft = 0
            n = i       
            while counterright < expansion:
                compare = abs(I_temp[n+1] - I_temp[right+1])
                if tempright > compare:
                    counterright += 1
                    n += 1
                    right += 1
                else:
                    tempright = compare
                    n += 1
                    right += 1
                    counterright = 0
    
    ## -- coordinates to crop streamline -- ##
    ## calculate actual pixel value ##
    firstcut = left * stepsize
    secondcut = (right * stepsize)
    
    ## -- crop and return img -- ##
    frame = frame[firstcut:secondcut,:] 
    return frame

def crop_coord(frame):
    grayscaley = grayscale_v(frame)
  
    ## operate data smoothing ##
    smoother = ConvolutionSmoother(window_len=30, window_type='ones')
    smoother.smooth(grayscaley)

    # generate intervals
    low, up = smoother.get_intervals('sigma_interval', n_sigma=3)
    
    data = smoother.smooth_data[0];
    
    ## -- find peak -- ## (locate streamline)
    ## integratie ##
    #integgration stepsize
    stepsize = 50
    len_dataset = len(data)/stepsize
    I_temp = np.arange(0.0, len_dataset)
    index = 0;
    temp = 0;
    trigger = 0;
    for i in range(len(data)):
        temp += data[i];
        trigger += 1;
        if trigger == stepsize:
            I_temp[index] = temp; 
            temp = 0
            trigger = 0
            index += 1
    if temp != 0:
        I_temp[index] = temp
        temp = 0

    ## find peak ##
    peak = 0
    for i in I_temp:
        if i > peak:
            peak = i;
   
    #grayscaleplot(I_temp)
    ## find valley ##
    templeft = 0;
    tempright = 0;
    counterleft = 0;
    counterright = 0;
    #the expansion valeu ist softening the cutwith of the streamline peak. (The higher the value the wider the cut, not linear!)
    #use small values for defined/narrow streamlines expansion <= 3 
    expansion = 3
 

    for i in range(int(len_dataset)-3):
        if I_temp[i] == peak:
            left = i-1
            right = i+1
            templeft = I_temp[i] - I_temp[i-1]
            tempright = I_temp[i] - I_temp[i+1]
            n = i
            while counterleft < expansion:
                compare = abs(I_temp[n-1] - I_temp[left-1])
                if templeft > compare:
                    counterleft += 1
                    n -= 1
                    left -= 1
                else:
                    templeft = compare
                    n -= 1
                    left -= 1
                    counterleft = 0
            n = i       
            while counterright < expansion:
                compare = abs(I_temp[n+1] - I_temp[right+1])
                if tempright > compare:
                    counterright += 1
                    n += 1
                    right += 1
                else:
                    tempright = compare
                    n += 1
                    right += 1
                    counterright = 0
    
    ## -- coordinates to crop streamline -- ##
    ## calculate actual pixel value ##
    firstcut = left * stepsize
    secondcut = (right * stepsize)
    
    cut = [firstcut,secondcut]
    return cut 
        
### smoothen data ###
def smooth(data,sigma):
    smoother = ConvolutionSmoother(window_len=50, window_type='ones')
    smoother.smooth(data)
    # generate intervals
    low, up = smoother.get_intervals('sigma_interval', n_sigma=sigma)
    return smoother.smooth_data[0]  

### --- creat 2D pattern --- ###
### slice ###
def sliceposition(data):
     ## integratie ##
    #integgration stepsize
    stepsize = 50
    len_dataset = len(data)/stepsize
    I_temp = np.arange(0.0, len_dataset)
    index = 0;
    temp = 0;
    trigger = 0;
    for i in range(len(data)):
        temp += data[i];
        trigger += 1;
        if trigger == stepsize:
            I_temp[index] = temp; 
            temp = 0
            trigger = 0
            index += 1
    if temp != 0:
        I_temp[index] = temp
        temp = 0

    ## locate peak ##
    peak = 0
    for i in range(len(I_temp)):
        if I_temp[i] > peak:
            peak = i*stepsize;
    return peak

def periodigram(data, x, background, fps):
    cutwidth = 250 #in pixel, defines the pattern width
    temp = data[0]
    cut = crop_coord(temp)
    final = []
    trigger = 0
    bg = background[0]
    bg = bg[cut[0]:cut[1],:]
    bg = bg[(x-cutwidth):(x+cutwidth),:]
    
    for i in data:
        temp = i[cut[0]:cut[1],:]
        if trigger == 0:
            tofilter = np.subtract(temp[(x-cutwidth):(x+cutwidth),:],(bg*0.75))
            final = grayscale_h(tofilter)
            #final = nd.gaussian_filter(tofilter,3)
            trigger = 1;
        else:
            tofilter = np.subtract(temp[(x-cutwidth):(x+cutwidth),:],(bg*0.75))
            tofilter = grayscale_h(tofilter)
            final = np.concatenate((final,tofilter))
            #final = np.concatenate((final,nd.gaussian_filter(tofilter,3)))
    
    fig = plt.figure(figsize = (10,30)) # create a 5 x 5 figure
    ax = fig.add_subplot(111)
    ax.imshow(final, cmap="gray")
    plt.show()
    
    return final              
    #img_binary = final > threshold
    
    """
    ## plot ##   
    fig = plt.figure(figsize = (30,30)) # create a 5 x 5 figure
    ax = fig.add_subplot(111)

    #adds a title and axes labels
    ax.set_title("x [mm]", fontsize='25')
    ax.set_xlabel("2D wave pattern at "+str(fps)+" frames per second", fontsize='25')
    ax.set_ylabel("frames", fontsize='25')

    #change axis direction
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top')
    ax.xaxis.set

    #Edit tick
    ax.tick_params(bottom=False, top=True, length=7, width=2) #, labelleft=False, labeltop=False
    ax.set_xticks(np.arange(0, 2040, step=100))
    ax.set_yticks(np.arange(0, 2*cutwidth*len(data), step=(4*cutwidth)))

    #labeling ticks
    tickvalues_x = np.arange(0, 25.2, step=1.2)
    tickvalues_x = tickvalues_x.round(1)
    ax.set_xticklabels(tickvalues_x, fontsize='13')
    tickvalues_y = np.arange(0, len(data), step=2)
    ax.set_yticklabels(tickvalues_y, fontsize='13')
    
    ax.imshow(img_binary, cmap="hot")
    plt.show()
    return img_binary
    """

def pattern2D(data, x, threshold, background, fps):
    cutwidth = 20 #in pixel, defines the pattern width
    temp = data[0]
    cut = crop_coord(temp)
    #temp = []
    final = []
    trigger = 0
    bg = background[0]
    bg = bg[cut[0]:cut[1],:]
    bg = bg[(x-cutwidth):(x+cutwidth),:]
    
    for i in data:
        temp = i[cut[0]:cut[1],:]
        if trigger == 0:
            tofilter = np.subtract(temp[(x-cutwidth):(x+cutwidth),:],(bg*0.75))
            final = tofilter
            #final = nd.gaussian_filter(tofilter,3)
            trigger = 1;
        else:
            tofilter = np.subtract(temp[(x-cutwidth):(x+cutwidth),:],(bg*0.75))
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
 
### --- Sience Grayscale Plot --- ###

def grayscaleplot(data):
    arr_referenc =  np.arange(len(data))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 6)
    #x_peak = find_peaks(data,height=2.4)
    #print(x_peak[0][0])
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    ax.plot(arr_referenc,data, color='#00429d')
    
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('Time [frames]')
    plt.ylabel('Wavespeed [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(125))
    ax.yaxis.set_minor_locator(MultipleLocator(.1))

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

def propagationspeed(data, background, y):
    peaklist2 = []
    cutwidth = 20
    temp = data[0] > 4
    cut = crop_coord(temp)
    print(cut)
    column = 0

    bg = background[0]
    bg = bg[cut[0]:cut[1],:]
    bg = bg[y-cutwidth:y+cutwidth,:]

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    for i in data:
        frame_croped = i[cut[0]+50:cut[1]-200,:] > 4
        #frame_croped = np.subtract(frame_croped[(y-cutwidth):(y+cutwidth),:],(bg*0.75))
        data = smooth(smooth(grayscale_h2(frame_croped),2),2)
        peaks, _ = find_peaks(data, distance=500, height=0.04)
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
    stepsize = 3
    row = column = 0
    print(peaklist2)
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
    for n in range(len(speed_list[0])):
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

    
def wavelength(data):
    triggervalue = 0.1 #min aplitude finding a peak, check plot
    cutwidth = 20
    frame_h = 2*cutwidth
    h, w = pattern.shape
    temp_gs = []
    temp = []
    peaklist = []

    for i in range(h):
        if i == frame_h:
            if i == (2*cutwidth):
                temp_gs = grayscale_h(pattern[:(2*cutwidth),:])
                frame_h = frame_h + (2*cutwidth)
                continue
            temp = grayscale_h(pattern[frame_h-(2*cutwidth):frame_h,:])
            temp_gs = np.concatenate((temp_gs,temp))
            frame_h = frame_h + (2*cutwidth)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    #fig.savefig('test2png.png', dpi=100)
        
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    for i in range(temp_gs.shape[0]):
        if i >= 0: #and i < 30:
            graph = smooth(smooth(smooth(smooth(smooth(smooth(temp_gs[i,:], 2),2),2),2),2),2)
            peaks, _ = find_peaks(graph, distance=500, height=triggervalue)
            ax.plot(graph, linewidth=0.5)           #, color='#00429d'
            ax.plot(peaks, graph[peaks], "x")
            if not len(peaklist):
                peaklist = peaks
            else:
                while peaks.shape[0] < peaklist.shape[0]:
                    peaks = np.append(peaks, 0)
                while peaks.shape[0] > peaklist.shape[0]:  
                    stack = np.zeros((1,peaklist.shape[1]))
                    peaklist = np.row_stack((peaklist,stack))
                peaklist = np.column_stack((peaklist,peaks))
    peaklist = np.transpose(peaklist)
    #print(peaklist)
    #ax.legend()
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('Pixel')
    plt.ylabel('Grayvalue')
     
    #removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
        
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(125))
    #ax.yaxis.set_minor_locator(MultipleLocator(.25))
    
    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');
    
    #adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.2, alpha=0.5)
        
    #ax limit
    ax.set_xlim(xmin=0)
        
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
        
    plt.show()

    ### --- calculate wavelength --- ###
    peaks_h ,peaks_w = peaklist.shape
    pixelsize = 0.0118 #mm
    samples_n = peaks_h * peaks_w
    avrg_wavelen = 0
    list_wavelen = []
    
    for h in range(peaks_h):
        for w in range(peaks_w-1):
            if peaklist[h,w+1] == 0:
                samples_n -= 1
            else:
                list_wavelen.append((peaklist[h,w+1]-peaklist[h,w])*pixelsize)
    for i in range(len(list_wavelen)):
        avrg_wavelen += list_wavelen[i]
    
    avrg_wavelen = avrg_wavelen/len(list_wavelen)
    
    s = 0   ## standardabweichung ##
    for i in range(len(list_wavelen)):
        s += (list_wavelen[i] - avrg_wavelen)**2
    s = np.sqrt((1/(samples_n-1))*s)
    
    dx = s/np.sqrt(samples_n)  ## statistischer fehler ##
    
    print("Wave Measurements: (Gauß)")
    print("Average wavelength: "+str(avrg_wavelen)+" mm")
    print("Standard deviation: "+str(s)+ " mm")
    print("Statistical error: "+str(dx)+" mm")
    
    fig, ax = plt.subplots(figsize = (10,10))
    
    # the histogram of the data
    n, bins, patches = ax.hist(list_wavelen, 30, density=True)
    
    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * (1 / s * (bins - avrg_wavelen))**2))
    ax.plot(bins, y, '--')
    ax.set_xlabel('Lambda [mm]')
    ax.set_ylabel('Counts')
    ax.set_title('Wavelength distribution')
    
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

def fft(channel):
    fft = np.fft.fft2(channel)
    fft *= 255.0 / fft.max()  # proper scaling into 0..255 range
    return np.absolute(fft)

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

def analyze_fft_amplitude(data):
    # Compute the FFT of the input data
   data_fft = np.fft.ifftshift(data)
   data_fft = np.fft.fft2(data_fft)
   data_fft_shifted = np.fft.fftshift(data_fft)
   freqs = np.fft.fftfreq(data.shape[0], 1/data.shape[0])
   freqs_shifted = np.fft.fftshift(freqs)
   wave_numbers = 2*np.pi*freqs_shifted

   result = np.abs(data_fft_shifted)

   # Find the frequency and wave number of maximum amplitude
   max_index = np.unravel_index(np.argmax(result, axis=None), result.shape)
   max_frequency = freqs_shifted[max_index[0]]
   max_wave_number = wave_numbers[max_index[1]]

   # Plot the amplitude of the FFT as a function of frequency and wave number
   plt.imshow(result, extent=(wave_numbers.min(), wave_numbers.max(), freqs_shifted.min(), freqs_shifted.max()), aspect='auto')
   plt.xlabel('Wave number')
   plt.ylabel('Frequency')
   plt.colorbar()
   plt.scatter(max_wave_number, max_frequency, color='red')
   plt.annotate(f'Maximum amplitude at ({max_wave_number:.2f}, {max_frequency:.2f})', xy=(max_wave_number, max_frequency), xytext=(max_wave_number, max_frequency+0.2), color='red')
   plt.show()


   #return np.abs(data_fft_shifted)
### --- Main ---- ###
#%%

frame = frames[25]
frame = frame[1050+100:1500-0,:] > 4
plt.imshow(frame)


#frames = frames[30:-60]

#%%

data = frames
y = 350
peaklist2 = []
cutwidth = 20
temp = data[0] > 4
cut = crop_coord(temp)
print(cut)
column = 0

bg = background[0]
bg = bg[cut[0]:cut[1],:]
bg = bg[y-cutwidth:y+cutwidth,:]

fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

for i in data[:3]:
    frame_croped = i[cut[0]+50:cut[1]-200,:] > 4
    #frame_croped = np.subtract(frame_croped[(y-cutwidth):(y+cutwidth),:],(bg*0.75))
    data = smooth(smooth(grayscale_h2(frame_croped),2),2)
    #
    dt = 1/80
    n = len(i)
    x = np.arange(0,n,1)
    fhat = np.fft.fft(data,n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1/(dt*n)) * np.arange(n)
    L = np.arange(1, np.floor(n/2), dtype='int')

    fig,axs = plt.subplots(dpi=200)
    plt.plot(freq[L],PSD[L])
    plt.show()

#%%
#period = periodigram(frames, 410, background, 80)
#speed_list15, wavelen_list15 = propagationspeed(frames[0:40], background, 350) #complete img is analysed, takes time to calculate but more acurate
#print(crop_coord(frames[0]))
#print(len(frames))
pattern = pattern2D(frames,435,4, background, 80) #propagating wave 375, screw 525, scan16 & pa20 455
#density_map = create_density_map(pattern)

#%%

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

#axs.set_title('Frequency-Wavenumber Resolved Power Spectral Density')
#data = analyze_fft_amplitude(np.array(period))
#wavelength(frames)
#%%

# Create a random space-time diagram (just for demonstration)
diagram = pattern

# Perform Fourier transformation on the space axis
fft_diagram_space = np.fft.fft(diagram, axis=0)
#fft_diagram_space_shifted = np.fft.fftshift(fft_diagram_space)
space_dia = np.abs(fft_diagram_space) ** 2

# Perform Fourier transformation on the time axis
fft_diagram_time = np.fft.fft(pattern, axis=1)
time_dia = np.abs(fft_diagram_time) ** 2
#fft_diagram_time_shifted = np.fft.fftshift(fft_diagram_time)

# Calculate the frequency wavenumbers
#kx = 2 * np.pi * np.fft.fftfreq(width, d=pixel_size_mm)
#ky = 2 * np.pi * np.fft.fftfreq(height, d=pixel_size_mm)
#kx_shifted = np.fft.fftshift(kx)
#ky_shifted = np.fft.fftshift(ky)

# Plot the original space-time diagram and the Fourier-transformed diagram
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(diagram, cmap='hot', aspect='auto', origin='lower')
axs[0].set_title('Space-Time Diagram')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Space')

axs[1].imshow(np.log(time_dia), cmap='viridis', aspect='auto', origin='lower')
axs[1].set_title('Frequency-Wavenumber Diagram')
axs[1].set_xlabel('Frequency')
axs[1].set_ylabel('Wavenumber')

plt.tight_layout()
plt.show()

#%%
# Define the dimensions of the space-time heatmap
height = 1000  # Number of pixels along the y-axis (space)
width = 2000   # Number of pixels along the x-axis (time)

# Define the physical properties
pixel_size_mm = 0.0118  # Size of each pixel in millimeters
timestep_seconds = 1/80  # Duration of each timestep in seconds

# Create a random space-time heatmap (just for demonstration)
heatmap = density_map
# Perform Fourier transformation
fft_heatmap = np.fft.fft2(heatmap)

# Shift the zero-frequency component to the center of the spectrum
fft_heatmap_shifted = np.fft.fftshift(fft_heatmap)

# Calculate the frequency wavenumbers
ky = 2 * np.pi * np.fft.fftfreq(width, d=pixel_size_mm)
kx = 2 * np.pi * np.fft.fftfreq(height, d=pixel_size_mm)
ky_shifted = np.fft.fftshift(ky)
kx_shifted = np.fft.fftshift(kx)

psf = np.abs(fft_heatmap) ** 2

# Plot the original heatmap and its frequency wavenumber spectrum
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(heatmap, cmap='hot', aspect='auto', origin='lower')
axs[0].set_title('Space-Time Heatmap')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Space')

axs[1].imshow(np.log10(psf), cmap='hot', aspect='auto', origin='lower',
              extent=[0, kx_shifted.max(), 0, ky_shifted.max()])
axs[1].set_title('Frequency Wavenumber Spectrum')
axs[1].set_xlabel('kx (Spatial frequency wavenumber)')
axs[1].set_ylabel('ky (Spatial frequency wavenumber)')
plt.tight_layout()
plt.show()
#%%
pattern_fft = np.fft.fft2(pattern)

fig = plt.figure(figsize = (10,30)) # create a 5 x 5 figure
ax = fig.add_subplot(111)
ax.imshow(pattern_fft, cmap="gray")
plt.show()
#%%
start = 100
stop = 100
print('start: ' + str(start) + ' stop: '+str(stop))
sl, wl = propagationspeed(frames[start:], background, 360)
#%%
speed_list15 = sl
wavelen_list15 = wl
#%%
speed_list15 = np.append(speed_list15,sl)
wavelen_list15 = np.append(wavelen_list15,wl[:20])
#%%
sl, wl = np.append(speed_list30,propagationspeed(frames[15:40], background, 410))
speed_list30 = np.append(speed_list30,sl)
wavelen_list30 = np.append(wavelen_list30,wl)
#%%
grayscaleplot(slf15)
grayscaleplot(wavelen_list15)
#%%
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
bigplot_wavelen(wl15[20:98], wl20[15:], wl25[:98], wl30[:75])
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
#%%
### Calculate Group Velocity 

#preprocessing

frame = frames[-16]
frame = frame [1150:1450,0:2000]
data = grayscale_h2(frame)
#grayscaleplot(data)
dt = 1/80
n = len(data)
x = np.arange(0,n,1)
fhat = np.fft.fft(data,n)
PSD = fhat * np.conj(fhat) / n
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(1, np.floor(n/2), dtype='int')

fig,axs = plt.subplots(dpi=200)
plt.plot(freq[L],PSD[L])
plt.show()

#print(PSD[0:20])

indices = PSD < 17.5
#indices = indices > 0.99
PSDclean = PSD * indices
fhat = indices * fhat
ffilt = np.fft.ifft(fhat)

fig,axs = plt.subplots(dpi=200)
plt.plot(x,ffilt, linewidth=0.5)
plt.show()

#%%
dataset=[]
t_step_Y = 0
bg = frames[-1]
bg = bg[1050:1550,:]
for i in range(len(frames._filepaths)):
    data = frames[i]
    data = data[1050:1550,:]
    excede = np.subtract(gaussian_filter(data,sigma=1),gaussian_filter(bg*0.4,sigma=1))
    dataset.append(np.add(t_step_Y,gaussian_filter1d(grayscale_v(excede),sigma=30)))
    t_step_Y -= 0.15

#%%
grayscaleplot_dataset(dataset[:-5])

#%%
peaks = []
for n in dataset:
    x = np.max(n)
    for index in range(len(n)):
        if n[index] == x:
            peaks = np.append(peaks,index)

#peaks = peaks[:13]
peaks = peaks[18:]
#PLOT and FIT
x =  np.arange(len(peaks))
#fit
coef = np.polyfit(x,peaks,1)
poly1d_fn = np.poly1d(coef)
#
fig, ax = plt.subplots(dpi=500)
ax.plot(x, peaks, 'X', x, poly1d_fn(x), '--k')

#%%

c_ph = (coef[0]*0.0118)/(1/80)

print('Group velocity = ', c_ph, 'mm/s')
#%%

"""
#%%
hH, hW = pattern.shape
r = 1100
#img_ft = np.real(np.fft.fft2(pattern))
fft = np.fft.fft2(pattern)
fft = np.fft.fftshift(fft)
#fft *= 255.0 / fft.max()  # proper scaling into 0..255 range
img_ft =  np.absolute(fft)

## plot ##   
fig = plt.figure(figsize = (20,20)) # create a 5 x 5 figure
ax = fig.add_subplot(111)

#adds a title and axes labels
ax.set_title("FT 2D pattern", fontsize='25')
#ax.set_xlabel("2D wave pattern at "+str(fps)+" frames per second", fontsize='25')
#ax.set_ylabel("frames", fontsize='25')

#change axis direction
#ax.invert_yaxis()
#ax.xaxis.tick_top()
#ax.xaxis.set_label_position('top')
#ax.xaxis.set

#Edit tick
#ax.tick_params(bottom=False, top=True, length=7, width=2) #, labelleft=False, labeltop=False
#ax.set_xticks(np.arange(0, 2040, step=100))
#ax.set_yticks(np.arange(0, 2*cutwidth*len(data), step=(4*cutwidth)))

#labeling ticks
#tickvalues_x = np.arange(0, 25.2, step=1.2)
#tickvalues_x = tickvalues_x.round(1)
#ax.set_xticklabels(tickvalues_x, fontsize='13')
#tickvalues_y = np.arange(0, len(data), step=2)
#ax.set_yticklabels(tickvalues_y, fontsize='13')
    
ax.imshow(img_ft[hH-r:hH+r,hW-r:hW+r], extent=[-r,r,-r,r], cmap="hot")
plt.show()

"""
#%%
### analysis ###

### --- cut --- ###

cutwidth = 20
frame_h = 2*cutwidth
h, w = pattern.shape
temp_gs = []
temp = []
temp_graph = []
peaklist = []

for i in range(h):
    if i == frame_h:
        if i == (2*cutwidth):
            temp_gs = grayscale_h(pattern[:(2*cutwidth),:])
            frame_h = frame_h + (2*cutwidth)
            continue
        temp = grayscale_h(pattern[frame_h-(2*cutwidth):frame_h,:])
        temp_gs = np.concatenate((temp_gs,temp))
        frame_h = frame_h + (2*cutwidth)

fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
#fig.savefig('test2png.png', dpi=100)
 
#color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green

### plot ##   
#fig_fft = plt.figure(figsize = (20,20)) # create a 5 x 5 figure
#ax_fft = fig_fft.add_subplot(111)
#adds a title and axes labels
#ax_fft.set_title("FT 2D pattern", fontsize='25')

### --- find ion waves --- ###

counter = 0
start = 0
finish = 70
for i in range(temp_gs.shape[0]):
    if i >= start: # and i < finish:
        graph = smooth(abs(hilbert(temp_gs[i,:])),2)
        if counter < 0:
            #hH, hW = pattern.shape
            #r = 1100
            #img_ft = np.real(np.fft.fft2(pattern))
            fft = np.fft.fft(graph)
            fft = np.fft.fftshift(fft)
            #fft *= 255.0 / fft.max()  # proper scaling into 0..255 range
            img_ft =  np.absolute(fft)
            ## plot ##   
            fig = plt.figure(figsize = (20,20)) # create a 5 x 5 figure
            ax = fig.add_subplot(111)
            ax.plot(img_ft)
            #ax.imshow(img_ft[hH-r:hH+r,hW-r:hW+r], extent=[-r,r,-r,r], cmap="hot")
            plt.show()
            
            counter += 1

        peaks, _ = find_peaks(-graph, distance=250, height=-0.1)
        ax.plot(graph, linewidth=0.5)           #, color='#00429d'
        ax.plot(peaks, graph[peaks], "x")
        
        if not len(peaklist):
            peaklist = peaks
        else:
            while peaks.shape[0] < peaklist.shape[0]:
                peaks = np.append(peaks, 0)
            while peaks.shape[0] > peaklist.shape[0]:  
                stack = np.zeros((1,peaklist.shape[1]))
                peaklist = np.row_stack((peaklist,stack))
            peaklist = np.column_stack((peaklist,peaks))
#peaklist = np.transpose(peaklist)

#print(peaklist)
#ax.legend()
#adds a title and axes labels
#ax.set_title('')
plt.xlabel('Pixel')
plt.ylabel('Grayvalue')
 
#removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
    
#Edit tick 
#ax.xaxis.set_minor_locator(MultipleLocator(125))
#ax.yaxis.set_minor_locator(MultipleLocator(.25))

#add vertical lines
#ax.axvline(left, linestyle='dashed', color='b');
#ax.axvline(right, linestyle='dashed', color='b');

#adds major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.2, alpha=0.5)
    
#ax limit
ax.set_xlim(xmin=0)
    
#legend
#ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
plt.show()
#%%
#manuele bearbeitung peaklist
peaklisttemp = peaklist

peaklisttemp[2:,116:] = 0
peaklisttemp[1:2,125:0] = 0
peaklisttemp[1:,60:70] = 0



for i in range(peaklisttemp.shape[1]):
    if i >= 9 and i <= 11:
        peaklisttemp[:,i] = np.roll(peaklisttemp[:,i],1)
    if i == 41:
        peaklisttemp[:,i] = np.roll(peaklisttemp[:,i],1)
   
"""        
#peaklist[:,8] = np.roll(peaklist[:,8],1)

peaklist = peaklist[1:,:]
peaklist[:,8] = np.roll(peaklist[:,8],-1)
peaklist[:,10:] = np.roll(peaklist[:,10:],-1)
peaklist[:,14] = np.roll(peaklist[:,14],-1)
peaklist[:,16] = np.roll(peaklist[:,16],-1)
"""
#%%
peaklisttemp = peaklist
strange_peak_list = []
max_pixdiff = 200



for row in range(peaklisttemp.shape[0]):
    for column in range(peaklisttemp.shape[1]-1):
        if peaklisttemp[row,column+1]!= 0 and peaklisttemp[row,column]!= 0 and peaklisttemp[row,column] - peaklisttemp[row,column+1] > 0:
            strange_peak_list.append((row,column+1))
            #print("row "+str(row)+"        column "+str(column+1))

n=0
while n < len(strange_peak_list):
    compare = strange_peak_list[n]
    templist = []
    for i in range(len(strange_peak_list)-1):
        compare2 = strange_peak_list[i]
        if compare[1] == compare2[1]:
            templist.append(compare2)
    mis_len = len(templist)
    compare_len = 0
    if mis_len != 0:                                    #Detektion Wellenübergang
        compare_p1 = templist[0]
        for i in range(peaklisttemp.shape[0]):
            if peaklisttemp[i,compare_p1[1]-1] != 0:
                print(peaklisttemp[i,compare_p1[1]-1])
                compare_len +=1
        if compare_len == mis_len:                          #Wellenübergang aus Fehlerliste löschen
            for i in templist:
                strange_peak_list.remove(i)
        else:
            for i in templist:
                if peaklisttemp[i[0],i[1]-1] - peaklisttemp[i] > max_pixdiff:   #zu große Messfehler aus den daten entfernen 
                    peaklisttemp[i] = 0
                strange_peak_list.remove(i)
    else:
        n+=1
"""
out = []
for n in range(len(strange_peak_list)-1):
    compare = strange_peak_list[n]
    templist = []
    for i in range(len(strange_peak_list)-1):
        compare2 = strange_peak_list[i]
        if compare[1] == compare2[1]:
            templist.append(compare2)
    mis_len = len(templist)
    print(templist)
    if len(templist) == 1:
        coord_temp = templist[0]
        if  peaklisttemp[coord_temp[0],coord_temp[1]-1] - peaklisttemp[coord_temp] > max_pixdiff:
            peaklisttemp[coord_temp] = 0
    
  
    if not len(out):
            out = templist
    else:
            while out.shape[0] < out.shape[0]:
                peaks = np.append(templist, 0)
            while out.shape[0] > out.shape[0]:
                stack = np.zeros((1,out.shape[1]))
                out = np.row_stack((out,stack))
            out = np.column_stack((out,compare))
"""    
#%%

### --- sort peaklist to peaklisttemp --- ###
#peaklisttemp = peaklist
max_pixdiff = 200
for row in range(peaklisttemp.shape[0]):
    for column in range(peaklisttemp.shape[1]-1):
        if peaklisttemp[row,column+1]!= 0 and abs(peaklisttemp[row,column+1] - peaklisttemp[row,column]) > max_pixdiff:
            stack = np.zeros((1,peaklisttemp.shape[1]))
            peaklisttemp = np.row_stack((stack,peaklisttemp))
            pointer_column = column
            pointer_row = row
            for pointer_column in range(peaklisttemp.shape[1]-1):
                flag = 0
                while abs(peaklisttemp[pointer_row+1,pointer_column+1] - peaklisttemp[pointer_row+1,pointer_column]) > max_pixdiff:
                    peaklisttemp[:,pointer_column+1] = np.roll(peaklisttemp[:,pointer_column+1],-1)
                    flag += 1
                if flag == peaklisttemp.shape[0]*2:
                    break;
    break;

#%%               

### --- AI - comparative value wavespeed --- ###
temp_list = []
for row in range(peaklist.shape[0]):
    for column in range(peaklist.shape[1]-1):
        if peaklist[row,column] != 0 and peaklist[row,column+1] != 0 and peaklist[row,column+1] - peaklist[row,column] > 0 and peaklist[row,column+1] - peaklist[row,column] < 100:
            temp_list.append(peaklist[row,column+1] - peaklist[row,column])
value_ai_wavespeed = sum(temp_list)/len(temp_list)
templistlen = len(temp_list)
print(value_ai_wavespeed)
#%%

### --- wavespeed AI --- ###
pixelsize = 0.0118 #mm
exptime = 0.0125 #s = 1/frames per second
conv_value = pixelsize/exptime 
list = []
percent = 0.1  #+10%
plus_ai_value = value_ai_wavespeed + (value_ai_wavespeed*percent)   
minus_ai_value = value_ai_wavespeed - (value_ai_wavespeed*percent)   
steps = 25
for row in range(peaklisttemp.shape[0]):
    for column in range(peaklisttemp.shape[1]-steps):
        if peaklisttemp[row,column] != 0 and peaklisttemp[row,column+steps] != 0 and peaklisttemp[row,column+steps] - peaklisttemp[row,column] < plus_ai_value*steps and peaklisttemp[row,column+steps] - peaklisttemp[row,column] > 0:
            list.append(((peaklisttemp[row,column+steps] - peaklisttemp[row,column])/steps)*conv_value)  
            #value_ai_wavespeed = ( value_ai_wavespeed  + (peaklisttemp[row,column+steps] - (peaklisttemp[row,column])/steps) )/2
            value_ai_wavespeed = ( value_ai_wavespeed *((templistlen-1)/templistlen) + (peaklisttemp[row,column+steps] - (peaklisttemp[row,column])/steps) *(1/templistlen))   #gewichtung des neuen werts gegenüber ai anpassen
            print(value_ai_wavespeed)
            templistlen +=1
            plus_ai_value = value_ai_wavespeed + (value_ai_wavespeed*percent)   
            minus_ai_value = value_ai_wavespeed - (value_ai_wavespeed*percent)    

avrg_speed = sum(list)/len(list)
s = 0   ## standardabweichung ##
samples_n = len(list)
for i in range(len(list)):
    s += (list[i] - avrg_speed)**2
s = np.sqrt((1/(samples_n-1))*s)

dx = s/np.sqrt(samples_n)  ## statistischer fehler ##

print("Wave Measurements:")
print("Average wavespeed: "+str(avrg_speed)+" mm/s")
print("Standard deviation: "+str(s)+ " mm/s")
print("Statistical error: "+str(dx)+" mm/s")

fig, ax = plt.subplots(figsize = (15,15))

# the histogram of the data
n, bins, patches = ax.hist(list, 30, density=True)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * (1 / s * (bins - avrg_speed))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Speed [mm/s]')
ax.set_ylabel('Counts')
ax.set_title('Speed distribution')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

#%%

### --- wavespeed --- ###
pixelsize = 0.0118 #mm
exptime = 0.0125 #s = 1/frames per second
conv_value = pixelsize/exptime
speed_list = []
speedrange = 25
                   
for row in range(peaklisttemp.shape[0]):
    for column in range(peaklisttemp.shape[1]):
        if peaklisttemp.shape[1]-column > speedrange and peaklisttemp[row,column] != 0  and peaklisttemp[row,column+speedrange] != 0 and peaklisttemp[row,column+speedrange]-peaklisttemp[row,column] > 0:
            speed_list.append((peaklisttemp[row,column+speedrange]-peaklisttemp[row,column])*conv_value/(speedrange))
for i in speed_list:
    if i > 18:
        speed_list.remove(i)
avrg_speed = sum(speed_list)/len(speed_list)
s = 0   ## standardabweichung ##
samples_n = len(speed_list)
for i in range(len(speed_list)):
    s += (speed_list[i] - avrg_speed)**2
s = np.sqrt((1/(samples_n-1))*s)

dx = s/np.sqrt(samples_n)  ## statistischer fehler ##

print("Wave Measurements: (Hilbert)")
print("Average wavespeed: "+str(avrg_speed)+" mm/s")
print("Standard deviation: "+str(s)+ " mm/s")
print("Statistical error: "+str(dx)+" mm/s")

fig, ax = plt.subplots(figsize = (10,10))

# the histogram of the data
n, bins, patches = ax.hist(speed_list, 30, density=True)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * (1 / s * (bins - avrg_speed))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Speed [mm/s]')
ax.set_ylabel('Counts')
ax.set_title('Speed distribution')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()


#%%

#
#
#
#
#
#

### --- calculate wavelength --- ###
peaklist = np.transpose(peaklist)
peaks_h ,peaks_w = peaklist.shape
pixelsize = 0.0118 #mm
samples_n = peaks_h * peaks_w
avrg_wavelen = 0
list_wavelen = []

for h in range(peaks_h):
    for w in range(peaks_w-1):
        if peaklist[h,w+1] == 0:
            samples_n -= 1
        else:
            list_wavelen.append((peaklist[h,w+1]-peaklist[h,w])*pixelsize)
for i in range(len(list_wavelen)):
    avrg_wavelen += list_wavelen[i]

avrg_wavelen = avrg_wavelen/len(list_wavelen)

s = 0   ## standardabweichung ##
for i in range(len(list_wavelen)):
    s += (list_wavelen[i] - avrg_wavelen)**2
s = np.sqrt((1/(samples_n-1))*s)

dx = s/np.sqrt(samples_n)  ## statistischer fehler ##

print("Wave Measurements: (Hilbert)")
print("Average wavelength: "+str(avrg_wavelen)+" mm")
print("Average wavelength / 2: "+str(avrg_wavelen/pixelsize)+" pixel") #442
print("Standard deviation: "+str(s)+ " mm")
print("Statistical error: "+str(dx)+" mm")

fig, ax = plt.subplots(figsize = (10,10))

# the histogram of the data
n, bins, patches = ax.hist(list_wavelen, 30, density=True)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * (1 / s * (bins - avrg_wavelen))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Lambda [mm]')
ax.set_ylabel('Counts')
ax.set_title('Wavelength distribution')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

#%%
### --- calculate wavespeed --- ### todo
pixelsize = 0.0118 #mm
exptime = 0.0125 #s = 1/frames per second
conv_value = pixelsize/exptime

peaks_h ,peaks_w = peaklist.shape
pointer = 0
wavetrigger = True
wavestarttrigger = True
temp = 0
pointer_wavestart = (0,0)
templist = []
results_wavespeed = []
values = []
histo = []


shaped = False
h = w = 0
while w < peaks_w:
    while h < peaks_h:
        if wavetrigger == True:
            templist = [peaklist[h,w]]
            wavetrigger = False 
            wavestarttrigger = True
            pointer_wavestart = (0,0)
        else:
            if abs(peaklist[h,w]-templist[-1]) < 210:  
                templist.append(peaklist[h,w])
            elif abs(peaklist[h,w]-templist[-1]) > 210 and peaklist[h,w] != 0:
                if wavestarttrigger == True and shaped == False:
                    pointer_wavestart = (h,w)
                    wavestarttrigger = False
                elif wavetrigger == True:
                    pointer_wavestart = (0,w+1)
                    wavestarttrigger = False
                w += 1
                h -= 1
            if h == peaks_h-1 or peaklist[h,w] == 0:
                results_wavespeed.append((templist[-1]-templist[0])*conv_value/len(templist))
                values.append(templist)
                if w == 0 and h == peaks_h-1:
                    shaped = True
                    h=0
                    w=1
                if peaklist[h,w] == 0 and pointer_wavestart == (0,0):
                    h,w = -1,w+1
                elif h == peaks_h-1 and pointer_wavestart == (0,0):
                    h = -1
                else:
                    h,w = pointer_wavestart
                wavetrigger = True
        if w == peaks_w:
            break
        h += 1
    h = 0          

avg_wavespeed = sum(results_wavespeed)/len(results_wavespeed)
print("Average Propagationspeed: "+str(avg_wavespeed)+" mm/s")
s2 = 0 #standardabweichung
for i in values:
    for n in range(len(i)-1):
        temp = (i[n+1]-i[n])*conv_value
        if temp <= 100 and temp >= 0:
            histo.append(temp)
            s2 += (temp-avg_wavespeed)**2
n = len(histo) #anz. messungen
#for i in values:
#    n = n + len(i)
s2 = np.sqrt((1/(n-1))*s2)
dx2 = 0 #statistischer fehler
dx2 = s2/np.sqrt(n)

print("Standard deviation: "+str(s2)+ " mm/s")
print("Statistical error: "+str(dx2)+" mm/s")   
print("Average frequency: "+str((avg_wavespeed*0.0001)/avrg_wavelen)+" Hz")

fig, ax = plt.subplots(figsize = (15,15))

# the histogram of the data
n, bins2, patches = ax.hist(histo, 30)

# add a 'best fit' line
y2 =  ((1 / (np.sqrt(2 * np.pi) * s2)) * np.exp(-0.5 * (1 / s2 * (bins2 - avg_wavespeed)**2)))*500          #((1 / (np.sqrt(2 * np.pi) * s2)) *
ax.plot(bins2, y2, '--')
ax.set_xlabel('v [mm/s]')
ax.set_ylabel('Counts')
ax.set_title('Speed distribution')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

#%%
#Tests Periodigram with grayscale analysis
final = []
trigger = 0
       
for i in frames:
    temp = grayscale_h2(i)
    if trigger == 0:
        final = temp
        final = np.column_stack((final,temp))
        final = np.column_stack((final,temp))
        final = np.column_stack((final,temp))
        final = np.column_stack((final,temp))
        trigger = 1;
    else:
        final = np.column_stack((final,temp))
        final = np.column_stack((final,temp))
        final = np.column_stack((final,temp))
        final = np.column_stack((final,temp))
        final = np.column_stack((final,temp))
#%%
bg = []
trigger = 0
temp = grayscale_h2(background[0])      
for i in range(101):
    if trigger == 0:
        bg = temp
        bg = np.column_stack((bg,temp))
        bg = np.column_stack((bg,temp))
        bg = np.column_stack((bg,temp))
        bg = np.column_stack((bg,temp))
        trigger = 1;
    else:
        bg = np.column_stack((bg,temp))
        bg = np.column_stack((bg,temp))
        bg = np.column_stack((bg,temp))
        bg = np.column_stack((bg,temp))
        bg = np.column_stack((bg,temp))
#%%
#final = np.transpose(final)
bg = np.transpose(bg)
bg = (255*(bg - np.min(bg))/np.ptp(bg)).astype(int)
#%%
test = final 
c = (255*(test - np.min(test))/np.ptp(test)).astype(int)
#%%
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

#end 

































