#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:33:33 2021

@author: adamfoster
"""
import numpy as np
import matplotlib.pyplot as plt



input =np.array(([30.723865509033203, 11.890228271484375, 17.92437744140625], [90.44308471679688, 10.80966567993164, 20.582881927490234]))
names = ['smooth', 'featured', 'artifact']

empty_probabilities = np.empty((0, 3))
for line in input:
    empty_row=np.empty((0, 3))
    for row in line:
        probability = row/sum(line)
        
        empty_row = np.append(empty_row, probability)
    empty_probabilities = np.vstack((empty_probabilities, empty_row))
    

def pie_chart_maker(row, save_name):
    plt.pie(input[row, :], labels=names, autopct='%1.1f%%')
    plt.savefig(save_name, dpi=200)
    plt.close()
    
pie_chart_maker(0, 'J000000.80+004200.0.png')
pie_chart_maker(1, 'J000103.04+011605.0.png')
    
