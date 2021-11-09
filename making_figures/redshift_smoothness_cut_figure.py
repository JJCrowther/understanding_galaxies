import numpy as np
import functions_for_redshifting_figures as frf
import pandas as pd



print('Begin \n')
file_name_list = ['scaled_image_predictions_1.csv', 'scaled_image_predictions_2.csv', 'scaled_image_predictions_3.csv', 'scaled_image_predictions_4.csv', 'scaled_image_predictions_5.csv', 'scaled_image_predictions_6.csv']

i=0 #index used for scale facotr multiplication
rounding=0.02
scale_factor_data={}
cut_threshold = 0.7
full_data_array_first_cut=np.zeros((0, 5))
full_data_array_second_cut=np.zeros((0, 5))
full_data_array_third_cut=np.zeros((0, 5))
x_data_list = []
proportions_by_redshift_by_cut = []

parquet_file = pd.read_parquet('nsa_v1_0_1_mag_cols.parquet', columns= ['iauname', 'redshift', 'elpetro_absmag_r'])

for file_name in file_name_list:
    i+=1 #will cause problems if scale factors aren't linearly listed in intervals of 1, should change

    scale_factor_data[file_name] = frf.file_reader(file_name)

    scale_factor_dataframe = pd.DataFrame(scale_factor_data[file_name])
    scale_factor_dataframe.rename(columns={0: 'iauname', 1:'smooth-or-featured_smooth_pred', 2:'smooth-or-featured_featured-or-disk_pred', 3:'smooth-or-featured_artifact_pred'}, inplace=True) #rename the headers of the dataframe

    scale_factor_dataframe['iauname'] = scale_factor_dataframe.iauname.str.replace('/share/nas/walml/repos/understanding_galaxies/scaled_{0}/'.format(i), '', regex=False)
    scale_factor_dataframe['iauname'] = scale_factor_dataframe.iauname.str.replace('.png', '', regex=False)

    merged_dataframe = scale_factor_dataframe.merge(parquet_file, left_on='iauname', right_on='iauname', how='left') #in final version make the larger dataframe update itsefl to be smaller?
    merged_dataframe['redshift']=merged_dataframe['redshift'].clip(lower=1e-10) #removes any negative errors
    merged_dataframe['redshift']=merged_dataframe['redshift'].mul(i) #Multiplies the redshift by the scalefactor

    first_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -18 ) & (merged_dataframe["elpetro_absmag_r"] >= -20)]
    second_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -20 ) & (merged_dataframe["elpetro_absmag_r"] >= -21)]
    third_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -21 ) & (merged_dataframe["elpetro_absmag_r"] >= -24)]

   
    merged_numpy_first_cut = first_mag_cut.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation
    merged_numpy_second_cut = second_mag_cut.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation
    merged_numpy_third_cut = third_mag_cut.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation

    numpy_merged_probs_first_cut = frf.prob_maker(merged_numpy_first_cut)
    numpy_merged_probs_second_cut = frf.prob_maker(merged_numpy_second_cut)
    numpy_merged_probs_third_cut = frf.prob_maker(merged_numpy_third_cut)

    full_data_array_first_cut=np.vstack((full_data_array_first_cut, numpy_merged_probs_first_cut)) #stacks all data from current redshift to cumulative array
    full_data_array_second_cut=np.vstack((full_data_array_second_cut, numpy_merged_probs_second_cut)) #stacks all data from current redshift to cumulative array
    full_data_array_third_cut=np.vstack((full_data_array_third_cut, numpy_merged_probs_third_cut)) #stacks all data from current redshift to cumulative array

for cut in [full_data_array_first_cut, full_data_array_second_cut, full_data_array_third_cut]:
    #full_data_array[:, 4]=np.round(full_data_array[:, 4].astype(float), 2) #rounds the redshift values to 2 dp for binning
    for redshift in range(len(cut)):
        cut[redshift, 4] = round(cut[redshift, 4].astype(float)*(1/rounding))/(1/rounding)

    cut = cut[np.argsort(cut[:, 4])] #sorts all data based on ascending redshift
        
    split_by_redshift = np.split(cut, np.where(np.diff(cut[:,4].astype(float)))[0]+1) #creates a list with entries grouped by identical redshift
    x_data_temp=[]
    for i in range(len(split_by_redshift)):
        x_data_temp.append(split_by_redshift[i][0, 4].astype(float))

    proportion={}
    for entry in range(len(split_by_redshift)):
        #proportion[entry] = frf.proportion_over_threshold_using_full_total(split_by_redshift[entry][:, 1:], cut_threshold)
        proportion[entry] = frf.proportion_over_threshold_using_certain_total(split_by_redshift[entry][:, 1:], cut_threshold)

    x_data_list.append(x_data_temp)
    proportions_by_redshift_by_cut.append(proportion)

figure_save_names = ['smoothness_cut_18_20_{0}_graph_redshift_certain_classification.png'.format(cut_threshold), 
                    'smoothness_cut_20_21_{0}_graph_redshift_certain_classification.png'.format(cut_threshold),
                    'smoothness_cut_21_24_{0}_graph_redshift_certain_classification.png'.format(cut_threshold)]
i=0
for proportions in proportions_by_redshift_by_cut:
    y_data = np.zeros((0, 3))
    for index in proportions:
        y_data = np.vstack((y_data, proportions[index][0:3].astype(float)))

    frf.error_bar_smoothness_3(x_data_list[i], y_data[:, 0:1], y_data[:, 1:2], y_data[:, 2:3], save_name=figure_save_names[i], title='Galaxy Morphology with Redshift', xlabel='Redshift', ylabel='Proportion of expected predictions', ylimits=[0, 1], xlimits=[0.02, 0.25])
    i+=1

print('\n end')
