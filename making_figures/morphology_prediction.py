import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#frf for file opening and plotting functions
import functions_for_redshifting_figures as frf

print('\nstart')

if __name__ == '__main__':
    
    delta_z = 0.01
    delta_p = 0.025
    
    test_z = 0.17
    test_p = 0.3
    pred_z = 0.15

    upper_z = test_z + delta_z
    lower_z = test_z - delta_z
    upper_p = test_p + delta_p
    lower_p =test_p - delta_p
    
    rounding=0.02
    scale_factor_data={}
    cut_threshold = 0.7
    full_data_array_first_cut=np.zeros((0, 5))

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

        first_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -18 ) & (merged_dataframe["elpetro_absmag_r"] >= -20) & (merged_dataframe["redshift"] <= 0.25)]
        #second_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -20 ) & (merged_dataframe["elpetro_absmag_r"] >= -21)]
        #third_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -21 ) & (merged_dataframe["elpetro_absmag_r"] >= -24)]

        merged_numpy_first_cut = first_mag_cut.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation
        #merged_numpy_second_cut = second_mag_cut.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation
        #merged_numpy_third_cut = third_mag_cut.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation

        numpy_merged_probs_first_cut = frf.prob_maker(merged_numpy_first_cut)
        #numpy_merged_probs_second_cut = frf.prob_maker(merged_numpy_second_cut)
        #numpy_merged_probs_third_cut = frf.prob_maker(merged_numpy_third_cut)

        full_data_array_first_cut=np.vstack((full_data_array_first_cut, numpy_merged_probs_first_cut)) #stacks all data from current redshift to cumulative array
        #full_data_array_second_cut=np.vstack((full_data_array_second_cut, numpy_merged_probs_second_cut)) #stacks all data from current redshift to cumulative array
        #full_data_array_third_cut=np.vstack((full_data_array_third_cut, numpy_merged_probs_third_cut)) #stacks all data from current redshift to cumulative array

        full_dataframe = pd.DataFrame(full_data_array_first_cut)
        
        immediate_sub_sample = pd.DataFrame(full_data_array_first_cut[(full_dataframe[4].astype(float) < upper_z) & (full_dataframe[4].astype(float) >= lower_z) & (full_dataframe[1].astype(float) >= lower_p) & (full_dataframe[1].astype(float) <= upper_p)])
        unique_names = pd.unique(immediate_sub_sample[0])
        
        sim_sub_set = pd.DataFrame()
        for name in unique_names:
            sim_sub_set = sim_sub_set.append(full_dataframe[full_dataframe[0] == name])
        
        i+=1 #will cause problems if scale factors aren't linearly listed in intervals of 1, should change

    #Let's make some predictions

    prediction_list=[]
    weight_list = []

    for name in unique_names:
        galaxy_data = sim_sub_set[sim_sub_set[0] == name]

        abs_diff_pred_z = abs(galaxy_data[4].astype(float) - pred_z)
        min_pos_pred = abs_diff_pred_z.idxmin()

        abs_diff_test_z = abs(galaxy_data[4].astype(float) - test_z)
        min_pos_test = abs_diff_test_z.idxmin()
        
        estimate_predictions = galaxy_data.loc[[min_pos_pred]]
        closest_vals = galaxy_data.loc[[min_pos_test]] #Possible could encounter edge case issues
        
        gaussain_p_variable = closest_vals[1].astype(float).to_numpy()[0]
        gaussian_z_variable = closest_vals[4].astype(float).to_numpy()[0]

        weight = frf.gaussian_wieghtings(gaussain_p_variable, gaussian_z_variable, test_p, test_z, delta_p, delta_z)

        prediction_list.append(estimate_predictions[1].astype(float).to_numpy()[0])
        weight_list.append(weight)

    mean_prediction = np.mean(prediction_list)
    mean_std = np.std(prediction_list)

    weighted_mean_numerator = np.sum(np.array(weight_list) * np.array(prediction_list))
    weighted_mean_denominator = np.sum(np.array(weight_list))
    weighted_mean = weighted_mean_numerator/weighted_mean_denominator

    weighted_std_numerator = np.sum(np.array(weight_list)*((np.array(prediction_list) - weighted_mean)**2))
    weighted_std_denominator = np.sum(np.array(weight_list))
    weighted_std = np.sqrt(weighted_std_numerator/weighted_std_denominator)

    plt.figure(figsize=(10,6))
    plt.suptitle('Individual Galaxy Morphology Near Test\nValue Parameters z={0} p={1}'.format(test_z, test_p), fontsize=18)
    
    plt.subplot(121)
    for name in unique_names:
        data_to_plot = sim_sub_set[sim_sub_set[0] == name]
        x_data = np.asarray(data_to_plot[4]).astype(float)
        y_data = np.asarray(data_to_plot[1]).astype(float)
        
        plt.errorbar(x_data, y_data, marker ='x', alpha=0.3)
    plt.errorbar(pred_z, mean_prediction, mean_std, marker ='x', alpha=1, label='unweighted mean = {0:.2f}\nunweighted std = {1:.2f}'.format(mean_prediction, mean_std)) #plotting raw average
    plt.xlabel('Redshift')
    plt.ylabel('Prediction of Smoothness Liklihood')
    plt.xlim([0, 0.25])
    plt.ylim([0, 1])
    plt.legend()

    plt.subplot(122)
    for name in unique_names:
        data_to_plot = sim_sub_set[sim_sub_set[0] == name]
        x_data = np.asarray(data_to_plot[4]).astype(float)
        y_data = np.asarray(data_to_plot[1]).astype(float)
        
        plt.errorbar(x_data, y_data, marker ='x', alpha=0.3)
    plt.errorbar(pred_z, weighted_mean, weighted_std, marker ='x', alpha=1, label='weighted mean = {0:.2f}\nweighted std = {1:.2f}'.format(weighted_mean, weighted_std)) #plotting average weighted by 2D gaussian
    plt.xlabel('Redshift')
    plt.xlim([0, 0.25])
    plt.ylim([0, 1])
    plt.legend()

    plt.savefig('subset_galaxies_with_prediction.png', dpi=200)
    plt.close()

    print('End')