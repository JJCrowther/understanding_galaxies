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

def file_reader_filtered(file_name):
    """
    file_name - str (name and csv file to read)
    #filter_columns - list (columns by whihc the intital data set will be selected)
    #data_columns - list (columns to be used for plotting)
    threshold_condition - float (value at which threshold cuttoff occurs)
    """
    input_file=open(file_name, 'r', encoding='utf-8-sig')

    data_array=csv.DictReader(input_file)

    smoothness=np.zeros((0, 7))

    for line in data_array:
        
        smoothness_line = np.array((line['image_loc'], line['bar_strong_pred'], line['bar_weak_pred'], line['bar_no_pred'], line['smooth-or-featured_smooth_pred'], line['smooth-or-featured_featured-or-disk_pred'], line['smooth-or-featured_artifact_pred']))
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

def prob_maker_filtered(input_array):
    """
    input_array - numpy array of str
    """
    cropped_input_filter = input_array[:, 1:4].astype(float)
    cropped_input_data = input_array[:, 4:7].astype(float)
    
    empty_probabilities_filter = np.empty((0, np.size(cropped_input_filter, 1)))
    empty_probabilities_data = np.empty((0, np.size(cropped_input_data, 1)))

    for line in cropped_input_filter:
        empty_row=np.empty((0, np.size(cropped_input_filter, 1)))
        for row in line:
            smoothness_probability = row/sum(line)
            empty_row = np.append(empty_row, smoothness_probability)
        empty_probabilities_filter = np.vstack((empty_probabilities_filter, empty_row))

    for line in cropped_input_data:
        empty_row=np.empty((0, np.size(cropped_input_data, 1)))
        for row in line:
            smoothness_probability = row/sum(line)
            empty_row = np.append(empty_row, smoothness_probability)
        empty_probabilities_data = np.vstack((empty_probabilities_data, empty_row))
    
    #empty_probabilities = np.vstack((prob_headers, empty_probabilities)) #Re-add names for columns
    empty_probabilities = np.hstack((input_array[:, 0:1], empty_probabilities_filter)) #Re-add the iaunames for each row
    empty_probabilities = np.hstack((empty_probabilities, empty_probabilities_data))

    return empty_probabilities

def pie_chart_maker(values, save_name, names):
    plt.pie(values, labels=names, autopct='%1.1f%%')
    plt.title('Smooth or featured predictions off 1628 images')
    plt.savefig(save_name, dpi=200)
    plt.close()
    return

def scatter_chart_maker(x_data, y_data, alpha, title, xlabel, ylabel, xlim, ylim,):
    plt.scatter(x_data, y_data, s=2, marker ='x', alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    #plt.savefig(save_name, dpi=200)
    #plt.close()
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

def proportion_over_threshold_using_full_total(input_array, threshold):
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

def proportion_over_threshold_using_certain_total(input_array, threshold):
    """
    input_array - array of floats size (x, 4) where x can vary
    threshold - single float value designating cut-off threshold

    finds the proportions taking into account only the galaxies for which the liklihood of one
    cassification is greater than the threshold value
    """

    smooth_count = np.count_nonzero(input_array[0:, 0].astype(float) >= threshold)
    featured_count = np.count_nonzero(input_array[0:, 1].astype(float) >= threshold)
    artifact_count = np.count_nonzero(input_array[0:, 2].astype(float) >= threshold)

    positive_classification_count = np.add(smooth_count, np.add(featured_count, artifact_count))

    #null_count = len(input_array) - positive_classification_count

    over_threshold_counts = np.array((smooth_count, featured_count, artifact_count))
    if positive_classification_count == 0:
        proportions = over_threshold_counts #Removes divide by 0 error
    else:
        proportions = over_threshold_counts/positive_classification_count

    return proportions

def variance_from_beta(input_array):
    """
    5 column array [name, smooth, featured, artifct, redshift]
    """
    cropped_input = input_array[:, 1:4].astype(float) #seperates only the wanted inputs
    empty_variance = np.empty((0, np.size(cropped_input, 1)))
    for line in cropped_input:
        empty_row=np.empty((0, np.size(cropped_input, 1)))
        for row in line:
            alpha = row
            beta = sum(line) - row

            numerator = (alpha * beta)
            denominator = ((alpha + beta)**2) * (alpha + beta + 1)
            smoothness_variance = numerator/denominator #calculation

            empty_row = np.append(empty_row, smoothness_variance)
        empty_variance = np.vstack((empty_variance, empty_row))

    empty_variance = np.hstack((input_array[:, 0:1], empty_variance)) #Re-add the iaunames for each row
    empty_variance = np.hstack((empty_variance, input_array[:, 4:5]))

    return empty_variance

def gaussian_weightings(p, z, p_0, z_0, delta_p, delta_z):
    """
    
    """
    #prefactor = (2*np.pi*((delta_z**2)+(delta_p)**2))**(-1/2)
    exponent = np.exp(-((((z - z_0)**2)/(2*(delta_z**2))) + (((p - p_0)**2)/(2*(delta_p**2)))))

    gaussian_factor = exponent #prefactor 
    return gaussian_factor

def chi_squared(observed, expected, variance):
    """
    """
    numerator = (expected - observed)**2
    denominator = variance
    chi_squared = numerator/denominator
    return chi_squared

if __name__ == '__main__':
    main()