#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:05:34 2021

@author: adamfoster
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

def main():
    return

def file_reader(file_name):
    input_file=open(file_name, 'r', encoding='utf-8-sig')

    data_array=csv.DictReader(input_file)

    smoothness=np.zeros((0, 4))

    for line in data_array:

        smoothness_line = np.array((line['image_loc'], line['smooth-or-featured_smooth_pred'], line['smooth-or-featured_featured-or-disk_pred'], line['smooth-or-featured_artifact_pred']))
        smoothness = np.vstack((smoothness, smoothness_line))
    
    for i in range(np.size(smoothness, 0)):
        for j in range(np.size(smoothness, 1)):
            smoothness[i, j] = smoothness[i, j].replace('[', '')
            smoothness[i, j] = smoothness[i, j].replace(']', '')
    
    return smoothness


def prob_maker(input_array):

    cropped_input = input_array[:, 1:4].astype(float)
    empty_probabilities = np.empty((0, np.size(cropped_input, 1)))
    for line in cropped_input:
        empty_row=np.empty((0, np.size(cropped_input, 1)))
        for row in line:

            smoothness_probability = row/sum(line)
            empty_row = np.append(empty_row, smoothness_probability)
        empty_probabilities = np.vstack((empty_probabilities, empty_row))
    
    #empty_probabilities = np.vstack((prob_headers, empty_probabilities)) #Re-add names for columns
    empty_probabilities = np.hstack((input_array[:, 0:1], empty_probabilities)) #Re-add the iaunames for each row
    empty_probabilities = np.hstack((empty_probabilities, input_array[:, 4:5]))

    return empty_probabilities

def pie_chart_maker(values, save_name, names):
    plt.pie(values, labels=names, autopct='%1.1f%%')
    plt.title('Smooth or featured predictions off 1628 images')
    plt.savefig(save_name, dpi=200)
    plt.close()
    return

def scatter_chart_maker(array, save_name):
    plt.scatter(array[:, 0], array[:, 1], s=2, marker ='x')
    plt.title('Smooth vs featured predictions off 1628 images')
    plt.xlabel('Smoothness prediction probability')
    plt.ylabel('Featured prediction probability')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(save_name, dpi=200)
    plt.close()
    return
       
def error_bar_smoothness_3(x_data, y_data_smooth, y_data_featured, y_data_artifact, save_name, title, xlabel, ylabel, ylimits, xlimits):

    plt.errorbar(x_data, y_data_smooth, marker='x', color='g', label='smooth')
    plt.errorbar(x_data, y_data_featured, marker='x', color='r', label='featured')
    plt.errorbar(x_data, y_data_artifact, marker='x', color='b', label='artifact')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.legend()
    plt.savefig(save_name, dpi=200)
    plt.close()
    return

def error_bar_smoothness_4(x_data, y_data_smooth, y_data_featured, y_data_artifact, y_data_unclassified, save_name):
    
    plt.errorbar(x_data, y_data_smooth, marker='x', color='g', label='smooth')
    plt.errorbar(x_data, y_data_featured, marker='x', color='r', label='featured')
    plt.errorbar(x_data, y_data_artifact, marker='x', color='b', label='artifact')
    plt.errorbar(x_data, y_data_unclassified, marker='x', color='y', label='unclassified')
    plt.title('Galaxy Morphology with Scalefactor')
    plt.xlabel('Scalefactor')
    plt.ylabel('Fraction of Prediction Above Threshold')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(save_name, dpi=200)
    plt.close()
    return

def average_maker(input_array):
    """
    input_array has form [iauname, smooth, featured, artifact, redshift] with no header
    """
    average_smooth = sum(input_array[:, 1].astype(float))/len(input_array)
    average_featured = sum(input_array[:, 2].astype(float))/len(input_array)
    average_artifact = sum(input_array[:, 3].astype(float))/len(input_array)
    
    averages = np.array(([average_smooth, average_featured, average_artifact, input_array[0, 4]]))
    
    return averages

def proportion_over_threshold(input_array, threshold):
    """
    input_array - array of floats size (x, 4) where x can vary
    threshold - single float value designating cut-off threshold
    """

    smooth_count = np.count_nonzero(input_array[0:, 0].astype(float) >= threshold)
    featured_count = np.count_nonzero(input_array[0:, 1].astype(float) >= threshold)
    artifact_count = np.count_nonzero(input_array[0:, 2].astype(float) >= threshold)

    null_count = len(input_array) - np.add(smooth_count, np.add(featured_count, artifact_count))

    over_threshold_counts = np.array((smooth_count, featured_count, artifact_count, null_count))
    proportions = over_threshold_counts/len(input_array)

    return proportions

if __name__ == '__main__':
    main()