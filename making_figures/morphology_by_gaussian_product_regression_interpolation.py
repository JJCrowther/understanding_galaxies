#import core data handling and plotting libraries
from numpy.core.numeric import full
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

#import the rectangle plot library
from matplotlib.patches import Rectangle

#frf for file opening and plotting functions
import functions_for_redshifting_figures as frf

#import the GPR libraries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern

print('\nStart')

if __name__ == '__main__':
    delta_z = 0.01 #sets width of sample box - Default optimised = 0.008
    delta_p = 0.02 #sets height of smaple box - Default optimised = 0.016
    delta_mag = 0.5 #Vary to find better base value - Default optimised = 0.5

    #Individual galaxy tunable test parameters
    #test_z = 0.2308291643857956
    #pred_z = 0.1154145821928978
    #actual_p = 0.5717100280287778
    #test_p = 0.4432387127766238
    #test_mag = -19.726

    #Set values for smapling 
    #upper_z = test_z + delta_z
    #lower_z = test_z - delta_z
    #upper_p = test_p + delta_p
    #lower_p =test_p - delta_p
    
    scale_factor_data={}
    full_data_array_first_cut=np.zeros((0, 6))
    full_data_array_first_cut_var=np.zeros((0, 6))
    chi_squared_list=[]

        # The data

    file_name_list = ['scaled_image_predictions_1.csv', 'scaled_image_predictions_1.2.csv', 'scaled_image_predictions_1.4.csv', 'scaled_image_predictions_1.6.csv', 'scaled_image_predictions_1.8.csv', 'scaled_image_predictions_2.csv', 'scaled_image_predictions_2.2.csv', 'scaled_image_predictions_2.4.csv', 'scaled_image_predictions_2.6.csv', 'scaled_image_predictions_2.8.csv', 'scaled_image_predictions_3.csv', 'scaled_image_predictions_4.csv', 'scaled_image_predictions_5.csv', 'scaled_image_predictions_6.csv']

    scale_factor_multiplier=[1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 4, 5, 6] #index used for scale facotr multiplication
    i=0 
    parquet_file = pd.read_parquet('nsa_v1_0_1_mag_cols.parquet', columns= ['iauname', 'redshift', 'elpetro_absmag_r'])

    for file_name in file_name_list:

        scale_factor_data[file_name] = frf.file_reader(file_name)

        scale_factor_dataframe = pd.DataFrame(scale_factor_data[file_name])
        scale_factor_dataframe.rename(columns={0: 'iauname', 1:'smooth-or-featured_smooth_pred', 2:'smooth-or-featured_featured-or-disk_pred', 3:'smooth-or-featured_artifact_pred'}, inplace=True) #rename the headers of the dataframe

        scale_factor_dataframe['iauname'] = scale_factor_dataframe.iauname.str.replace('/share/nas/walml/repos/understanding_galaxies/scaled_{0}/'.format(scale_factor_multiplier[i]), '', regex=False)
        scale_factor_dataframe['iauname'] = scale_factor_dataframe.iauname.str.replace('.png', '', regex=False)

        merged_dataframe = scale_factor_dataframe.merge(parquet_file, left_on='iauname', right_on='iauname', how='left') #in final version make the larger dataframe update itsefl to be smaller?
        merged_dataframe['redshift']=merged_dataframe['redshift'].clip(lower=1e-10) #removes any negative errors
        merged_dataframe['redshift']=merged_dataframe['redshift'].mul(scale_factor_multiplier[i]) #Multiplies the redshift by the scalefactor

        first_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -18 ) & (merged_dataframe["elpetro_absmag_r"] >= -24) & (merged_dataframe["redshift"] <= 0.25)]
        
        merged_numpy_first_cut = first_mag_cut.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation
        
        numpy_merged_probs_first_cut = frf.prob_maker(merged_numpy_first_cut)
        numpy_merged_var_first_cut = frf.variance_from_beta(merged_numpy_first_cut)

        numpy_merged_probs_first_cut = np.hstack((numpy_merged_probs_first_cut, merged_numpy_first_cut[:, -1:]))
        numpy_merged_var_first_cut = np.hstack((numpy_merged_var_first_cut, merged_numpy_first_cut[:, -1:]))

        full_data_array_first_cut=np.vstack((full_data_array_first_cut, numpy_merged_probs_first_cut)) #stacks all data from current redshift to cumulative array
        full_data_array_first_cut_var=np.vstack((full_data_array_first_cut_var, numpy_merged_var_first_cut))
        i+=1 

    print('Files appended, removing test sample')
    #Remove the test sample
    test_sample_names = full_data_array_first_cut[1:2, 0] 

    full_dataframe = pd.DataFrame(full_data_array_first_cut)
    full_dataframe_var = pd.DataFrame(full_data_array_first_cut_var)
    test_sample = pd.DataFrame(columns=full_dataframe.columns)

    for name in test_sample_names:
        cond = full_dataframe[0] == name
        rows = full_dataframe.loc[cond, :]
        test_sample = test_sample.append(rows ,ignore_index=True)
        full_dataframe.drop(rows.index, inplace=True)
        full_dataframe_var.drop(rows.index, inplace=True)

    print('Beginning predictions')
    #If we want to operate over multiple galaxies, start a for loop here
    for test_name in test_sample_names:
        #Assign empty variables for each loop
        prediction_list = []
        prediction_list_gp = []
        pred_sigma_list = []
        weight_list = []
        sd_list = []
    
        test_galaxy = test_sample[test_sample[0] == test_name]
        gal_max_z = test_galaxy.loc[[test_galaxy[4].astype(float).idxmax()]]
        gal_min_z = test_galaxy.loc[[test_galaxy[4].astype(float).idxmin()]]
        test_z = gal_max_z[4].astype(float).to_numpy()[0]
        test_p = gal_max_z[1].astype(float).to_numpy()[0]
        pred_z = gal_min_z[4].astype(float).to_numpy()[0]
        actual_p = gal_min_z[1].astype(float).to_numpy()[0]
        test_mag = gal_max_z[5].astype(float).to_numpy()[0]

        #Set values for smapling 
        upper_z = test_z + delta_z
        lower_z = test_z - delta_z
        upper_p = test_p + delta_p
        lower_p =test_p - delta_p

        immediate_sub_sample = full_dataframe[(full_dataframe[4].astype(float) < upper_z) & (full_dataframe[4].astype(float) >= lower_z) & (full_dataframe[1].astype(float) >= lower_p) & (full_dataframe[1].astype(float) <= upper_p)]
        unique_names = pd.unique(immediate_sub_sample[0])
            
        sim_sub_set = pd.DataFrame()
        sim_sub_set_var = pd.DataFrame()
        for name in unique_names:
            sim_sub_set = sim_sub_set.append(full_dataframe[full_dataframe[0] == name])
            sim_sub_set_var = sim_sub_set_var.append(full_dataframe_var[full_dataframe_var[0] == name])
    
        for name in unique_names:
            galaxy_data = sim_sub_set[sim_sub_set[0] == name]
            galaxy_data_var = sim_sub_set_var[sim_sub_set_var[0] == name]

            abs_diff_pred_z = abs(galaxy_data[4].astype(float) - pred_z)
            min_pos_pred = abs_diff_pred_z.nsmallest(2).index[0] #nsmallest pickest the n smallest values and puts them in df
            next_min_pos_pred = abs_diff_pred_z.nsmallest(2).index[1]

            abs_diff_test_z = abs(galaxy_data[4].astype(float) - test_z)
            min_pos_test = abs_diff_test_z.nsmallest(2).index[0] #nsmallest pickest the n smallest values and puts them in df
            nextmin_pos_test = abs_diff_test_z.nsmallest(2).index[1]

            estimate_predictions = galaxy_data.loc[[min_pos_pred]]
            estimate_predictions_var = galaxy_data_var.loc[[min_pos_pred]]
            grad_reference = galaxy_data.loc[[next_min_pos_pred]]

            diff_y = estimate_predictions[1].astype(float).to_numpy()[0] - grad_reference[1].astype(float).to_numpy()[0]
            diff_x = estimate_predictions[4].astype(float).to_numpy()[0] - grad_reference[4].astype(float).to_numpy()[0] #the astype and to numpy are to extract numbers from dataframe
            gradient = diff_y / diff_x #Finding the gradient between the two points closest to the test value

            minimum_point_seperation = pred_z - estimate_predictions[4].astype(float).to_numpy()[0]
            grad_correction = gradient * minimum_point_seperation
            grad_corrected_prediction = estimate_predictions[1].astype(float).to_numpy()[0] + grad_correction

            closest_vals = galaxy_data.loc[[min_pos_test]] #Possible could encounter edge case issues
            
            gaussain_p_variable = closest_vals[1].astype(float).to_numpy()[0]
            gaussian_z_variable = closest_vals[4].astype(float).to_numpy()[0]
            gaussian_mag_variable = closest_vals[5].astype(float).to_numpy()[0]

            proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p, test_z, delta_p/2, delta_z/2)
            mag_weight = frf.gaussian_weightings(gaussian_mag_variable, 0, test_mag, 0, delta_mag, 1)

            weight = proximity_weight * mag_weight

            #print('mag_weight is:', mag_weight, '\nprox_wieght is:', proximity_weight, '\nTotal Weight is:', weight)

            prediction_list.append(grad_corrected_prediction)
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

        #finding a chi squared value
        chi_squared = frf.chi_squared(weighted_mean, actual_p, (weighted_std)**2, 2)
        chi_squared_list.append(chi_squared)

        #Initiate and name the figure
        plt.figure(figsize=(10,6))
        plt.suptitle('{3} Morphology Near Test\nValue Parameters z={0:.3f} p={1:.3f} with N={2} Galaxies'.format(test_z, test_p, len(unique_names), test_name), fontsize=18)
        
        #Manipulate the weight list to turn into usable alphas
        weight_list_np = np.array(weight_list)
        #transform to interval [0, 1] using -1/log10(weight/10)
        logged_weights = np.log10(weight_list_np/10)
        alpha_per_gal = -1/logged_weights
        #Normalise the alphas to max at 0.8
        max_alpha = alpha_per_gal.max()
        norm_factor = 0.5/max_alpha
        norm_alphas_per_gal = alpha_per_gal * norm_factor

        #Open the subplot
        plt.subplot(111)
        weight_index=0
        for name in unique_names:
            # Instantiate a Gaussian Process model for Regression
            kernel = 1 * RBF(length_scale=0.01, length_scale_bounds=(1e-7, 1e7)) #Matern(length_scale=0.01, length_scale_bounds=(1e-7, 1e7), nu=1.5) #RBF(length_scale=0.001, length_scale_bounds=(1e-5, 1e5))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=12)

            #Find the x and y data for subset galaxy
            data_to_plot = sim_sub_set[sim_sub_set[0] == name]
            var_to_plot = sim_sub_set_var[sim_sub_set_var[0] == name]
            x_data = np.asarray(data_to_plot[4]).astype(float)
            y_data = np.asarray(data_to_plot[1]).astype(float)
            y_err = np.sqrt(np.asarray(var_to_plot[1]).astype(float))
            
                #Now to do regression - base of code taken from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
            
            # Fit to data using Maximum Likelihood Estimation of the parameters
            gp.fit(np.atleast_2d(x_data).T, y_data)

            # Mesh the input space for evaluations of the real function, the prediction and its MSE
            x = np.atleast_2d(np.linspace(0, 0.25, 100)).T     

            # Make the prediction on the meshed x-axis (ask for MSE as well)
            y_pred, sigma = gp.predict(x, return_std=True)

            target_index = pred_z/((0.25-0)/100)
            prediction_gp = y_pred[np.round(target_index).astype(int)]
            pred_sigma = sigma[np.round(target_index).astype(int)]
            prediction_list_gp.append(prediction_gp)
            pred_sigma_list.append(pred_sigma)

            #Plot the data points and the data fit
            plt.errorbar(x_data, y_data, marker ='x', alpha=norm_alphas_per_gal[weight_index])
            plt.plot(x[:], y_pred[:], "b-", alpha=norm_alphas_per_gal[weight_index]) #plot for smaller x range (drop the first 0.1 maybe?) - looks weird
            weight_index+=1

        prediction_average = np.mean(prediction_list_gp)
        sigma_average = np.mean(np.sqrt(pred_sigma_list))
        plt.errorbar(pred_z, prediction_average, sigma_average, marker='v', color = 'r', label='Prediction value: {0:.3f}\nStandard Deviation: {1:.3f}'.format(prediction_average, sigma_average))
        plt.errorbar(pred_z, actual_p, marker = 'v', alpha = 0.75,  color = 'black', label='Actual Test prediction: {0:.3f}\nTarget Redshift: {1:.3f}'.format(actual_p, pred_z))
        plt.errorbar(test_z, test_p, marker = 's', alpha = 0.75,  color = 'black', label='Original redshift prediction')
        plt.xlabel('Redshift')
        plt.ylabel('Prediction of Smoothness Liklihood')
        plt.xlim([0, 0.25])
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig('gp_interpolation_RBF_{0}_weightalpha.png'.format(test_name))

    plt.close('all')
    print('End')
    
    
    