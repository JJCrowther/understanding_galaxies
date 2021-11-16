import numpy as np
import functions_for_redshifting_figures as frf
import pandas as pd
import matplotlib.pyplot as plt

print('Begin \n')

file_name_list = ['scaled_image_predictions_1.csv', 'scaled_image_predictions_1.2.csv', 'scaled_image_predictions_1.4.csv', 'scaled_image_predictions_1.6.csv', 'scaled_image_predictions_1.8.csv', 'scaled_image_predictions_2.csv', 'scaled_image_predictions_2.2.csv', 'scaled_image_predictions_2.4.csv', 'scaled_image_predictions_2.6.csv', 'scaled_image_predictions_2.8.csv', 'scaled_image_predictions_3.csv', 'scaled_image_predictions_4.csv', 'scaled_image_predictions_5.csv', 'scaled_image_predictions_6.csv']

scale_factor_multiplier=[1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 4, 5, 6] #index used for scale facotr multiplication
i=0
rounding=0.02
scale_factor_data={}
cut_threshold = 0.7
full_data_array_probs={}
full_data_array_var={}
x_data_list = []
proportions_by_redshift_by_cut = []

parquet_file = pd.read_parquet('nsa_v1_0_1_mag_cols.parquet', columns= ['iauname', 'redshift', 'elpetro_absmag_r'])

print('Parquet file read\n')
for file_name in file_name_list:

    scale_factor_data[file_name] = frf.file_reader(file_name)[0:25, :] #produces (dict of) array of top 100 smoothness values

    scale_factor_dataframe = pd.DataFrame(scale_factor_data[file_name])
    scale_factor_dataframe.rename(columns={0: 'iauname', 1:'smooth-or-featured_smooth_pred', 2:'smooth-or-featured_featured-or-disk_pred', 3:'smooth-or-featured_artifact_pred'}, inplace=True) #rename the headers of the dataframe

    scale_factor_dataframe['iauname'] = scale_factor_dataframe.iauname.str.replace('/share/nas/walml/repos/understanding_galaxies/scaled_{0}/'.format(scale_factor_multiplier[i]), '', regex=False)
    scale_factor_dataframe['iauname'] = scale_factor_dataframe.iauname.str.replace('.png', '', regex=False) #isolate just the iauname for each galaxy from image save name

    merged_dataframe = scale_factor_dataframe.merge(parquet_file, left_on='iauname', right_on='iauname', how='left') #in final version make the larger dataframe update itsefl to be smaller?
    merged_dataframe['redshift']=merged_dataframe['redshift'].clip(lower=1e-10) #removes any negative errors, wont be plotted anyway due to lower boundary on redshifts
    merged_dataframe['redshift']=merged_dataframe['redshift'].mul(scale_factor_multiplier[i]) #Multiplies the redshift by the scalefactor (should happen before clip? But doesn't make difference here since clip is so small)

    merged_numpy = merged_dataframe.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation
    numpy_merged_probs = frf.prob_maker(merged_numpy)
    numpy_merged_variance = frf.variance_from_beta(merged_numpy)
    
    #numpy_merged_probs = numpy_merged_probs[np.argsort(numpy_merged_probs[:, 4])] #sorts all data based on ascending redshift
    i+=1

    full_data_array_probs[i]=numpy_merged_probs #stacks all data from current redshift to cumulative array
    full_data_array_var[i]=numpy_merged_variance

    print('Finsihed pass of redshift scale factor {0}\n'. format(scale_factor_multiplier[i-1]))
    
#data to plot is redshift on x and certainty on y

print('Beginning plotting\n')

for line in range(len(full_data_array_probs[1])):
    x_data_list = []
    y_data_list=[]
    y_err=[]
    for redshift in range(len(full_data_array_probs)):
        x_data_list.append(full_data_array_probs[redshift+1][line, 4].astype(float)) #4=redshift
        y_data_list.append(full_data_array_probs[redshift+1][line, 1].astype(float)) #1=smooth, 2=featured, 3=artifact
        y_err.append(np.sqrt(full_data_array_var[redshift+1][line, 1].astype(float))) #1=smooth_err, 2=featured_err, 3=artifact_err

    plt.errorbar(x_data_list, y_data_list, y_err, marker ='x', alpha=0.3)

plt.title('Individual Galaxy Morphology')
plt.xlabel('Redshift')
plt.ylabel('Prediction of Smoothness Liklihood')
plt.xlim([0.02, 0.15])
plt.ylim([0, 1])

plt.savefig('Individual_galaxy_predictions_smooth_extended_data.png', dpi=200)
plt.close()

print('End')