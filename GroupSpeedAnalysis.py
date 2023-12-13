# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:04:42 2021

Grayscale Analysis

@author: Lukas Wimmer
"""

from __future__ import division, unicode_literals, print_function # Für die Kompatibilität mit Python 2 und 3.
import time
from tqdm import tqdm
startzeit = time.time()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pims
import scipy.stats

from scipy.ndimage import gaussian_filter, gaussian_filter1d


import optuna
import optuna.visualization as vis
from functools import partial
import plotly





### --- Generate Greyscale Horizontal ---- ###

def grey_sum(frame):
    imgshape = frame.shape
    grayscale = np.empty(imgshape[1], dtype=float)
    
    for column in range(imgshape[1]):
        sumpixel=0;
        for row in frame:
            sumpixel += row[column];
        grayscale[column] = sumpixel;
    return grayscale

### --- Sience Grayscale Plot --- ###

def plot_fit_group(data, pix_size, fps):
    
    arr_referenc =  np.arange(len(data))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 6)
    
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    ax.scatter(arr_referenc,data, color='#00429d', marker='^')
    
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('Time [frames]')
    plt.ylabel('Wave position [mm]')

    #adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    #ax.set_ylim(ymax=20)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    coef = np.polyfit(arr_referenc,data,1)
    poly1d_fn = np.poly1d(coef)             # poly1d_fn is now a function which takes in x and returns an estimate for y
    
    ax.plot(arr_referenc, poly1d_fn(arr_referenc), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker
    
    plt.show()    

    return poly1d_fn



### - Estimate Group Velocity of Cloud-Head - ### 

def envelope(sig, distance):
    # split signal into negative and positive parts
    u_x = np.where(sig > 0)[0]
    l_x = np.where(sig < 0)[0]
    u_y = sig.copy()
    u_y[l_x] = 0
    l_y = -sig.copy()
    l_y[u_x] = 0
    
    # find upper and lower peaks
    u_peaks, _ = scipy.signal.find_peaks(u_y, distance=distance)
    l_peaks, _ = scipy.signal.find_peaks(l_y, distance=distance)
    
    # use peaks and peak values to make envelope
    u_x = u_peaks
    u_y = sig[u_peaks]
    l_x = l_peaks
    l_y = sig[l_peaks]
    
    # add start and end of signal to allow proper indexing
    end = len(sig)
    u_x = np.concatenate((u_x, [0, end]))
    u_y = np.concatenate((u_y, [0, 0]))
    l_x = np.concatenate((l_x, [0, end]))
    l_y = np.concatenate((l_y, [0, 0]))
    
    # create envelope functions
    u = scipy.interpolate.interp1d(u_x, u_y)
    l = scipy.interpolate.interp1d(l_x, l_y)
    
    return u, l


def cloudhead_pos_data(group_frames,threshold, gate, gauss_sigma, envelope_step, cut_width, cut, reverse_data, fps, pix_size):
    limit = []

    #cut = 1250 #int(group_frames[0].shape[0] - cut_width)  #cut out the bottom

    faktor = 1/(fps*pix_size)

    items = range(len(group_frames))
    for item in tqdm(items, desc="Processing items", unit="item"):
        frame = group_frames[item]
        prog = gaussian_filter1d(grey_sum(frame[int(cut-(cut_width/2)):int(cut+(cut_width/2)),:]>threshold),sigma=gauss_sigma)
        #
        u, l = envelope(prog, envelope_step)
        # 
        x = np.arange(len(prog))
        value = u(x)
        if reverse_data == True:    
            prog = prog[::-1]
            value = value[::-1]
            
        check = 0
        for i in range(len(value)):
            if value[i] > gate and check == 0:
                limit.append(i)
                check = i      
            
        ### PLOT ###
        fig = plt.figure(figsize = (10,10), dpi=50) # create a 5 x 5 figure
        ax = fig.add_subplot(111)
        plt.plot(x, value, label="envelope")
        ax.plot(prog, linewidth=0.8, label="Flux")           #, color='#00429d'
        #
        if check != 0:
            ax.axvline(check, linestyle='dashed', color='r');
        #
        plt.legend()
        plt.show()

    if reverse_data == True:    
        limit = limit[::-1]
    
    poly_result2 = plot_fit_group(limit, pix_size, fps)
    result2 = np.polyder(poly_result2).coeffs[0]    #first coefficient -> slope
    #print('wert von result2: '+str(result2))
    

    s2 = 0   ## standard deviation of x ##
    for i in range(len(limit)):
        s2 += (limit[i] - poly_result2(i))**2
    s2 = np.sqrt((1/(len(limit)-1))*s2)
    dx2 = s2/np.sqrt(len(limit))
    
    ### error of v ###
    
    dx2_in_mm = dx2 * pix_size
    v_in_mms = result2 * faktor
    s_in_mm = (limit[-1]-limit[0]) * pix_size
    dv_in_mms = v_in_mms*dx2_in_mm/s_in_mm
    
    print("Group speed v = " + str(v_in_mms) + " /pm " + str(dv_in_mms) + " /frac(mm)(s)")
    
    return dv_in_mms    # dv_in_mms is the error


# function to minimize
def objective(trial, group_frames, cut, reverse_data, fps, pix_size):
    try:    
        #----------Parameter Space----------
        threshold = trial.suggest_float('threshold', 10, 12)
        gate = trial.suggest_float('gate', 5, 20)
        gauss_sigma = trial.suggest_float('gauss_sigma', 5, 25)
        envelope_step = trial.suggest_int('envelope_step', 80, 280)
        cut_width = trial.suggest_int('cut_width', 20, 300)
        #-----------------------------------
        error = cloudhead_pos_data(
            group_frames[:-7], threshold, gate, gauss_sigma,
            envelope_step, cut_width, cut, reverse_data, fps, pix_size
        )

        return error
    except Exception as e:
        # Return infinity when an exception occurs (threshold or gate too high -> no data points for fit)
        print(f"Exception: {e}")
        return float('inf')


# Adjustables
fps = 60
pix_size = 0.0147  # mm (14,7 micrometers)
cut = 450
#cut_width = 100
reverse_data = True

iteration_number = 3     # number of iterations per dataset
number_of_datasets = 3  # number of all the datasets (e.g. 2 means Analysing Data_0 and Data_1)

### --- Main ---- ###

best_params = []
best_error = []

for x in range(number_of_datasets):
    
    print('\nLoading Data_'+str(x)+'\n')
    # Read in Images
    group_frames = pims.open('Data_'+str(x)+'/*bmp')
    # Foldernames: Data_0, Data_1, Data_2,...
    test = group_frames[20] > 14
    
    # Optionally, tweak styles.
    matplotlib.rc('figure',  figsize=(10, 5))
    matplotlib.rc('image', cmap='gray')
    plt.imshow(test)
    
    
    # Create a partial function to pass additional fixed parameters to the objective function
    objective_partial = partial(
        objective, group_frames=group_frames, cut=cut,
        reverse_data=reverse_data, fps=fps, pix_size=pix_size
    )
    
    #Test_1, Test_2, Test_Randomsampler, Data_0_Test, Data_1_Test
    study_name = "Data_"+str(x)+"_Test"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    
    # Bayesian optimization with optuna
    study = optuna.create_study(study_name=study_name, storage=storage_name, sampler=optuna.samplers.TPESampler()
    , load_if_exists=True, direction='minimize')
    print(f"Sampler is {study.sampler.__class__.__name__}")
    study.optimize(objective_partial, n_trials=iteration_number)      # n_trials: number of iterations
    
    # extract best parameters and corresponding error from the created study
    best_params.append(study.best_params)
    best_error.append(study.best_value)
    
    # Print the best parameters and error
    print('----------------------------------------------------------')
    print("Best Parameters for Data_"+str(x)+":", best_params[x])
    print("Best Error for Data_"+str(x)+":", best_error[x])
    print('----------------------------------------------------------')

# summary for every dataset
print('\n\nSUMMARY OF RESULTS:')
print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
for j in range(number_of_datasets):
    # Print the best parameters and error
    print('----------------------------------------------------------')
    print("Best Parameters for Data_"+str(j)+":", best_params[j])
    print("Best Error for Data_"+str(j)+":", best_error[j])
    print('----------------------------------------------------------')

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


#Test_2:
#Best Parameters: {'threshold': 10.730445589607204, 'gate': 14.459429768288892, 'gauss_sigma': 8.919739209910576, 'envelope_step': 81, 'cut_width': 240}
#Best Error: 0.17711427227461246























