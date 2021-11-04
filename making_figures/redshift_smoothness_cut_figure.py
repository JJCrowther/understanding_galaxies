import numpy as np
import functions_for_redshifting_figures as frf
import pandas as pd
import parquet


print('Begin \n')
file_name_list = ['scaled_image_predictions_1.csv', 'scaled_image_predictions_2.csv', 'scaled_image_predictions_3.csv', 'scaled_image_predictions_4.csv', 'scaled_image_predictions_5.csv', 'scaled_image_predictions_6.csv']

i=0
scale_factor_data={}
proportion={}
cut_threshold = 0.8
full_data_array=np.zeros((0, 5))

parquet_file = pd.read_parquet('nsa_v1_0_1_mag_cols.parquet', columns= ['iauname', 'redshift'])

for file_name in file_name_list:
    i+=1

    scale_factor_data[file_name] = frf.file_reader(file_name)

    scale_factor_dataframe = pd.DataFrame(scale_factor_data[file_name])
    scale_factor_dataframe.rename(columns={0: 'iauname', 1:'smooth-or-featured_smooth_pred', 2:'smooth-or-featured_featured-or-disk_pred', 3:'smooth-or-featured_artifact_pred'}, inplace=True) #rename the headers of the dataframe

    scale_factor_dataframe['iauname'] = scale_factor_dataframe.iauname.str.replace('/share/nas/walml/repos/understanding_galaxies/scaled_{0}/'.format(i), '', regex=False)
    scale_factor_dataframe['iauname'] = scale_factor_dataframe.iauname.str.replace('.png', '', regex=False)

    merged_dataframe = scale_factor_dataframe.merge(parquet_file, left_on='iauname', right_on='iauname', how='left') #in final version make the larger dataframe update itsefl to be smaller?
    merged_dataframe['redshift']=merged_dataframe['redshift'].clip(lower=1e-10) #removes any negative errors
    merged_dataframe['redshift']=merged_dataframe['redshift'].mul(i) #Multiplies the redshift by the scalefactor

    merged_numpy = merged_dataframe.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation
    numpy_merged_probs = frf.prob_maker(merged_numpy)
    full_data_array=np.vstack((full_data_array, numpy_merged_probs)) #stacks all data from current redshift to cumulative array
    

full_data_array[:, 4]=np.round(full_data_array[:, 4].astype(float), 2) #rounds the redshift values to 2 dp for binning
    
full_data_array = full_data_array[np.argsort(full_data_array[:, 4])] #sorts all data based on ascending redshift
    
split_by_redshift = np.split(full_data_array, np.where(np.diff(full_data_array[:,4].astype(float)))[0]+1) #creates a list with entries grouped by identical redshift

for entry in range(len(split_by_redshift)):
    #proportion[entry] = frf.proportion_over_threshold_using_full_total(split_by_redshift[entry][:, 1:], cut_threshold)
    proportion[entry] = frf.proportion_over_threshold_using_certain_total(split_by_redshift[entry][:, 1:], cut_threshold)

x_data = np.arange(0, 0.91, 0.01)

y_data = np.zeros((0, 3))
for index in proportion:
    y_data = np.vstack((y_data, proportion[index][0:3].astype(float)))

frf.error_bar_smoothness_3(x_data, y_data[:, 0:1], y_data[:, 1:2], y_data[:, 2:3], save_name='smoothness_cut_graph_redshift_certain_classification.png', title='Galaxy Morphology with Redshift', xlabel='Redshift', ylabel='Proportion of expected predictions', ylimits=[0, 0.3], xlimits=[0.02, 0.25])


print('\n end')