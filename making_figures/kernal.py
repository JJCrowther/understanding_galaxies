import numpy as np
from scipy.stats import norm
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import scipy
import matplotlib.pyplot as plt
from numpy.core.numeric import full
import pandas as pd
import matplotlib.pyplot as plt
import csv

def kernal_create (x, x_val, sd, weights):
    kern_sum = 0
    for v in range(len(x_val)):
        exp_num = (x-x_val[v])**2
        exp_don = 2 * (sd[v])**2
        normalising_func = 1/(sd[v] * np.sqrt(2*np.pi))
        kern_sum_mid = weights[v] * normalising_func * np.exp(-exp_num/exp_don)
        kern_sum += kern_sum_mid
    return kern_sum

#frf for file opening and plotting functions
import functions_for_redshifting_figures as frf

delta_z = 0.008 #sets width of sample box
delta_p = 0.016 #sets height of smaple box
delta_mag = 0.5 #Vary to find better base value

full_data_array_first_cut = open('full_data_array_first_cut.csv', 'r')
full_data_array_first_cut_var = open('full_data_array_first_cut_var.csv', 'r')
full_data_array_first_cut_data = csv.reader(full_data_array_first_cut)
full_data_array_first_cut_var_data = csv.reader(full_data_array_first_cut_var)
rows = np.zeros((0,6))
for row in full_data_array_first_cut_data:
    rows = np.vstack((rows, row))
    
rows_var = np.zeros((0,6))
for row in full_data_array_first_cut_var_data:
    rows_var = np.vstack((rows_var, row))
    
full_data_array_first_cut.close()
full_data_array_first_cut_var.close()

test_sample_names = rows[469:470, 0] 

full_dataframe = pd.DataFrame(rows)
full_dataframe_var = pd.DataFrame(rows_var)
test_sample = pd.DataFrame(columns=full_dataframe.columns)

for name in test_sample_names:
    cond = full_dataframe[0] == name
    rows = full_dataframe.loc[cond, :]
    test_sample = test_sample.append(rows ,ignore_index=True)
    full_dataframe.drop(rows.index, inplace=True)
    full_dataframe_var.drop(rows.index, inplace=True)

print('Beginning predictions')
#If we want to operate over multiple galaxies, start a for loop here
for name in test_sample_names:

    test_galaxy = test_sample[test_sample[0] == name]
    gal_max_z = test_galaxy.loc[[test_galaxy[4].astype(float).idxmax()]]
    gal_min_z = test_galaxy.loc[[test_galaxy[4].astype(float).idxmin()]]
    test_z = gal_max_z[4].astype(float).to_numpy()[0]
    test_p = gal_max_z[2].astype(float).to_numpy()[0]
    pred_z = gal_min_z[4].astype(float).to_numpy()[0]
    actual_p = gal_min_z[2].astype(float).to_numpy()[0]
    test_mag = gal_max_z[5].astype(float).to_numpy()[0]

    #Set values for smapling 
    upper_z = test_z + delta_z
    lower_z = test_z - delta_z
    upper_p = test_p + delta_p
    lower_p =test_p - delta_p

    immediate_sub_sample = full_dataframe[(full_dataframe[4].astype(float) < upper_z) & (full_dataframe[4].astype(float) >= lower_z) & (full_dataframe[2].astype(float) >= lower_p) & (full_dataframe[2].astype(float) <= upper_p)]
    unique_names = pd.unique(immediate_sub_sample[0])
        
    sim_sub_set = pd.DataFrame()
    sim_sub_set_var = pd.DataFrame()
    for unique_name in unique_names:
        sim_sub_set = sim_sub_set.append(full_dataframe[full_dataframe[0] == unique_name])
        sim_sub_set_var = sim_sub_set_var.append(full_dataframe_var[full_dataframe_var[0] == unique_name])
    
    
    #Let's make some predictions

    prediction_list=[]
    weight_list = []
    sd_list = []


    for unique_name in unique_names:
        galaxy_data = sim_sub_set[sim_sub_set[0] == unique_name]
        galaxy_data_var = sim_sub_set_var[sim_sub_set_var[0] == unique_name]

        abs_diff_pred_z = abs(galaxy_data[4].astype(float) - pred_z)
        min_pos_pred = abs_diff_pred_z.idxmin()

        abs_diff_test_z = abs(galaxy_data[4].astype(float) - test_z)
        min_pos_test = abs_diff_test_z.idxmin()
        
        estimate_predictions = galaxy_data.loc[[min_pos_pred]]
        estimate_predictions_var = galaxy_data_var.loc[[min_pos_pred]]
        
        closest_vals = galaxy_data.loc[[min_pos_test]] #Possible could encounter edge case issues
        
        gaussain_p_variable = closest_vals[2].astype(float).to_numpy()[0]
        gaussian_z_variable = closest_vals[4].astype(float).to_numpy()[0]
        gaussian_mag_variable = closest_vals[5].astype(float).to_numpy()[0]

        proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p, test_z, delta_p/2, delta_z/2)
        mag_weight = frf.gaussian_weightings(gaussian_mag_variable, 0, test_mag, 0, delta_mag, 1)

        weight = proximity_weight * mag_weight

        #print('mag_weight is:', mag_weight, '\nprox_wieght is:', proximity_weight, '\nTotal Weight is:', weight)

        prediction_list.append(estimate_predictions[2].astype(float).to_numpy()[0])
        sd_list.append(estimate_predictions_var[2].astype(float).to_numpy()[0])
        weight_list.append(weight)
        
    
    mean_prediction = np.mean(prediction_list)
    mean_std = np.std(prediction_list)

    weighted_mean_numerator = np.sum(np.array(weight_list) * np.array(prediction_list))
    weighted_mean_denominator = np.sum(np.array(weight_list))
    weighted_mean = weighted_mean_numerator/weighted_mean_denominator

    weighted_std_numerator = np.sum(np.array(weight_list)*((np.array(prediction_list) - weighted_mean)**2))
    weighted_std_denominator = np.sum(np.array(weight_list))
    weighted_std = np.sqrt(weighted_std_numerator/weighted_std_denominator)
    
    """
    Copy from here
    
    Change x_val, sd and weights
    """
    
    x_val = prediction_list
    sd = np.sqrt(sd_list)
    weights = weight_list

    kern_sum_val = np.zeros(0)
    area_kern_sum_val = np.zeros(0)
    

    """
    
    kern_sum_val = np.zeros(0)
    area_kern_sum_val = np.zeros(0)
    x_val = prediction_list
    sd = np.sqrt(sd_list)
    weights = weight_list
    x_range = np.arange (0,1,0.001)
    for x in x_range:
        
        a = kernal_create(x, x_val, sd, weights)
        kern_sum_val = np.append(kern_sum_val, a)
        area_kern_sum_val = np.append(area_kern_sum_val, a * 0.001)

    total = np.vstack((x_range, kern_sum_val)).T

    above_array = np.zeros(0)
    below_array = np.zeros(0)
    
    mean_sum_array = np.zeros(0)
    var_sum_array = np.zeros(0)
    
    max_v = np.argmax(kern_sum_val)
    max_prob = total[max_v][0]
    max_val = np.max(kern_sum_val)
    half_max_val = max_val/2
    half_array = abs(kern_sum_val-half_max_val)
   
    for i in range(len(half_array)):
        
        mean_sum_val = x_range[i] * (kern_sum_val[i]/np.sum(kern_sum_val))
        mean_sum_array = np.append(mean_sum_array, mean_sum_val)
        var_sum_val = (x_range[i]**2) * (kern_sum_val[i]/np.sum(kern_sum_val))
        var_sum_array = np.append(var_sum_array, var_sum_val)
        
    new_mean = np.sum(mean_sum_array)
    variance = np.sum(var_sum_array) - new_mean**2
    sd = np.sqrt(variance)
    
    area_norm = np.sum(area_kern_sum_val)
    norm_kern_sum = kern_sum_val/area_norm
    
    peaks, _ = find_peaks(norm_kern_sum, height=0)
    
    if (len(peaks)> 1):
        peak_data = {"peak" : [], "ratio" : [], "kern" : [], "loc" : [], "mean" : []};
        minimums = np.zeros(0)
        for i in range(len(peaks)):
            peak_data["peak"].append(i)
            peak_data["mean"].append(peaks[i])
            peak_height_sum = 0
            for j in range(len(peaks)):
                peak_height_sum += norm_kern_sum[peaks[j]]
            ratio = norm_kern_sum[peaks[i]]/peak_height_sum
            peak_data["ratio"].append(ratio)
            if (i < (len(peaks)-1)):
                values = []
                for l in range(len(x_range)):
                    if (l>=peaks[i] and l<peaks[i+1]):
                        values.append(norm_kern_sum[l])
                values = np.asarray(values)
                min_val_id = np.where(values == np.amin(values)) + peaks[i]
                min_val_id = float(min_val_id[0])
                minimums = np.asarray(minimums)
                minimums = np.append(minimums, min_val_id)
                
        for i in range(len(peaks)):
            kern_data = []
            loc_data = []
            for m in range(len(x_range)):
                if (i == 0):
                    if (m<=minimums[i]):
                        kern_data.append(norm_kern_sum[m])
                        loc_data.append(x_range[m])
                if (i == (len(peaks)-1)):
                    if (m>minimums[i-1]):
                        kern_data.append(norm_kern_sum[m])
                        loc_data.append(x_range[m])
                else:
                    if ((m>minimums[i-1]) and (m<=minimums[i])):
                        kern_data.append(norm_kern_sum[m])
                        loc_data.append(x_range[m])
            kern_data = np.asarray(kern_data)
            loc_data = np.asarray(loc_data)
            peak_data["kern"].append(kern_data)
            peak_data["loc"].append(loc_data)
        df = pd.DataFrame(peak_data)
        sds = np.zeros(0)
        for i in range(len(peaks)):
            data = df.iloc[i]['kern']
            locs = df.iloc[i]['loc']
            mean = df.iloc[i]['mean']
            half_values = np.zeros(0)
            close_to_half_data = abs(data-peaks[i]/2)
            below_mean = False
            below_mean_data = np.zeros(0)
            above_mean = False
            above_mean_data = np.zeros(0)
            for m in range(len(data)):
                if m> np.argmax(data):
                    above_mean_data = np.append(above_mean_data, data[m])
                else:
                    below_mean_data = np.append(below_mean_data, data[m])
            if (np.min(above_mean_data) < (norm_kern_sum[mean]/2)):
                close_to_half_data = abs(above_mean_data-(norm_kern_sum[mean]/2))
                closest = np.argmin(close_to_half_data) + mean
                half_values = np.append(half_values, closest)
            if (np.min(below_mean_data) < norm_kern_sum[mean]/2):
                close_to_half_data = abs(below_mean_data-(norm_kern_sum[mean]/2))
                closest = mean - np.argmin(close_to_half_data)
                half_values = np.append(half_values, closest) 
            
            if len(half_values) == 2:
                FWHM = abs(half_values[1]-half_values[0])/1000
                sd = FWHM/(2*np.sqrt(2*np.log(2)))
            if len(half_values) == 1:
                diff = abs(half_values[0]-mean)/1000
                sd = diff/(np.sqrt(2*np.log(2)))
                
            sds = np.append(sds, sd)
        df["sd"] = sds
                
    if (len(peaks) == 1):      
        plt.plot(x_range,norm_kern_sum, label= 'Kerneled pdf')
        plt.xlabel("Featured probability")
        plt.ylabel("Normalised value")
        plt.title("Featured probability of {0}\nat z = {1:.3f} from z = {2:.3f} and p = {3:.3f} with N = {4} galaxies".format(name, pred_z, test_z, test_p, len(unique_names)))
        plt.axvline(actual_p, label='original prob = {0:.3f}'.format(actual_p), color='black')
        plt.axvline(new_mean, label='mean= {0:.3f}'.format(new_mean), color='red')
        
        plt.fill_between(x_range, norm_kern_sum, where=(sd > abs((new_mean-x_range))), color ='blue', alpha = 0.4, label = "Standard deviation = {0:.3f}".format(sd))
        plt.legend()
        plt.savefig('{0}_kernal_prediction_featured.png'.format(name))
        plt.close()
    
    else:
        plt.plot(x_range,norm_kern_sum, label= 'Kerneled pdf')
        plt.xlabel("Featured probability")
        plt.ylabel("Normalised value")
        plt.title("Featured probability of {0}\nat z = {1:.3f} from z = {2:.3f} and p = {3:.3f} with N = {4} galaxies".format(name, pred_z, test_z, test_p, len(unique_names)))
        plt.axvline(actual_p, label='original prob = {0:.3f}'.format(actual_p), color='black')
        
        for i in range(len(peaks)):
            plt.axvline(peaks[i]/1000, label='peak_{1} = {0:.3f}, with peak ratio = {2:.2f}'.format(df.iloc[i]['mean']/1000, i,df.iloc[i]['ratio'] ), color='red')
            plt.fill_between(x_range, norm_kern_sum, where=(df.iloc[i]['sd'] > abs(((df.iloc[i]['mean']/1000)-x_range))), color ='blue', alpha = 0.4, label = "Standard deviation_{1} = {0:.3f}".format(df.iloc[i]['sd'], i))
        plt.legend()
        plt.savefig('{0}_kernal_prediction_featured.png'.format(name))
        plt.close()
        
