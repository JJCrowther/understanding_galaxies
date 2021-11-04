import numpy as np
import average_galaxy_type_plot as agtp

print('Begin \n')
file_name_list = ['scaled_image_predictions_1.csv', 'scaled_image_predictions_2.csv', 'scaled_image_predictions_3.csv', 'scaled_image_predictions_4.csv', 'scaled_image_predictions_5.csv', 'scaled_image_predictions_6.csv']

scale_factor_data={}
cut_number={}
cut_threshold = 0.8

for file_name in file_name_list:

    scale_factor_data[file_name] = agtp.file_reader(file_name)

    scale_factor_data[file_name] = agtp.prob_maker(scale_factor_data[file_name])

    smooth_count = np.count_nonzero(scale_factor_data[file_name][1:, 1:2].astype(float) >= cut_threshold)
    featured_count = np.count_nonzero(scale_factor_data[file_name][1:, 2:3].astype(float) >= cut_threshold)
    artifact_count = np.count_nonzero(scale_factor_data[file_name][1:, 3:4].astype(float) >= cut_threshold)

    null_count = len(scale_factor_data[file_name]) - np.add(smooth_count, np.add(featured_count, artifact_count))

    cut_number[file_name] = np.array((smooth_count, featured_count, artifact_count, null_count))
    cut_number[file_name] = cut_number[file_name]/len(scale_factor_data[file_name])

x_data = [1, 2, 3, 4, 5, 6]

y_data = np.zeros((0, 4))
for entry in cut_number:
    y_data = np.vstack((y_data, cut_number[entry]))

agtp.error_bar_smoothness_3(x_data, y_data[:, 0:1], y_data[:, 1:2], y_data[:, 2:3], save_name='morphology_cutoff_graph.png',
                            title='Faction of classifications above 0.8 threshold value', xlabel='Scalefactor', ylabel='Fraction of predictions above threshold', ylimits=[0, 0.2])

#print('cut_values:', cut_number['scaled_image_predictions_1.csv'])

print('\n end')

