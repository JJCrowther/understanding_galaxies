import numpy as np
import average_galaxy_type_plot as agtp

print('Begin \n')
file_name_list = ['scaled_image_predictions_1.csv', 'scaled_image_predictions_2.csv', 'scaled_image_predictions_3.csv', 'scaled_image_predictions_4.csv', 'scaled_image_predictions_5.csv', 'scaled_image_predictions_6.csv']

scale_factor_data={}
averages={}

for file_name in file_name_list:

    scale_factor_data[file_name] = agtp.file_reader(file_name)

    scale_factor_data[file_name] = agtp.prob_maker(scale_factor_data[file_name])

    averages[file_name] = agtp.average_maker(scale_factor_data[file_name])

x_data = [1, 2, 3, 4, 5, 6]

y_data = np.zeros((0, 3))
for entry in averages:
    y_data = np.vstack((y_data, averages[entry]))

agtp.error_bar_smoothness_3(x_data, y_data[:, 0:1], y_data[:, 1:2], y_data[:, 2:3], save_name='smoothness_comparison_graph.png', 
                            title='Galaxy Morphology with Scalefactor', xlabel='Scalefactor', ylabel='Average Probability', ylimits=[0, 1])

print('finsihed iterating over filename \n')

#print('Factor_1 data:', scale_factor_data['scaled_image_predictions_1.csv'])
#print('Factor_1 averages:', averages['scaled_image_predictions_1.csv'])

#print(y_data)

#print('Factor_2 data:', scale_factor_data['scaled_image_predictions_2.csv'])
#print('Factor_2 averages:', averages['scaled_image_predictions_2.csv'])

print('\n end')    
