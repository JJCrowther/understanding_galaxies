# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:09:31 2022

@author: r41331jc
"""

import glob
import os
import warnings
from astropy.io import fits
import argparse
import pandas as pd
import numpy as np

import creating_image_functions

import functions_for_redshifting_figures as frf

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fits-dir', dest='fits_dir', type=str)
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--max-redshift', dest='max_redshift', default = 0.2, type=float)
    parser.add_argument('--step-size', dest='step_size', default = 0.004, type=float)
    
    args = parser.parse_args()
    
    parquet_file = pd.read_parquet('nsa_v1_0_1_mag_cols.parquet', columns= ['iauname', 'redshift'])

    fits_dir =  args.fits_dir
    print('Loading images from {}'.format(fits_dir))
    assert os.path.isdir(fits_dir)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # '/**/*.fits', recursive=True):
    imgs = {} # Opens dictionary for storing images
    filenames = glob.iglob(f'{fits_dir}' + '/*.fits') #operates over all FIT's within the desired directory
    # print(filenames)
    # filenames = list(filenames)[:5]
    for filename in filenames:
        try:
            img, hdr = fits.getdata(filename, 0, header=True) #Extract FITs data
        except Exception:
            warnings.warn('Invalid fits at {}'.format(filename))
        imgs[filename] = img #Unsure if didctionary would be better here if required for a large quantity of data
        #imgs.append(img)

    print('All images loaded')
  
    final_data = {} # Create dictionary to append final data to

    for original_loc, original_img in imgs.items():
        
        min_redshift_df = parquet_file.where(parquet_file['iauname']==filename.replace('.png', ''))
        min_redshift_df = min_redshift_df.dropna(subset = ['iauname'])
        min_redshift = min_redshift_df.to_numpy()
        redshift_val = min_redshift[:,1]
        redshift_val = redshift_val[0]
        
        for scale_factor in np.arange(redshift_val, args.max_redshift, args.step_size):
            filename = os.path.basename(original_loc)
            filename_scale = filename.replace('.fits', '{}.png'.format(scale_factor))
            # file_loc = os.path.join('/share/nas/walml/repos/understanding_galaxies', output_dir_name[1], filename)
            file_loc = os.path.join(save_dir, filename_scale)

            _, _, img_scaled = creating_image_functions.photon_counts_from_FITS(original_img, scale_factor) # Second input is scale factor, changed in parser
            final_data[file_loc] = img_scaled
         

    print('All images scaled')

    for save_loc, scaled_image in final_data.items():
        #creating_image_functions.make_png_from_corrected_fits(final_data[entry_name][0], os.getcwd() + '/' + f'{output_dir_name[0]}' + '/Original_' + entry_name + '.png', 424) #Might want to remove the word Original in file name?
        creating_image_functions.make_png_from_corrected_fits(
            img=scaled_image,
            png_loc=save_loc,
            png_size=424)

    print('Successfully made images - exiting')