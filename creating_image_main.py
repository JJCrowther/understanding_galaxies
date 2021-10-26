import glob
import os
import warnings
from astropy.io import fits
import argparse

import creating_image_functions

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--FITS-dir', dest='fits_dir', type=str)
    parser.add_argument('--scale-factor', dest='scale_factor', default=1, type=float)

    args = parser.parse_args()

    input_dir_name=args.fits_dir
    #input_dir_name='FITSdata' #Sets the name of the folder FITs files are stored in
    output_dir_name = ['original_images', 'scaled_images'] #Sets the name of the folder which holds the input images for the CNN

    # '/**/*.fits', recursive=True):
    imgs = {} #Opens dictionary for storing images
    for filename in glob.iglob(f'{input_dir_name}' + '/*.fits'): #operates over all FIT's within the desired directory
        try:
            img, hdr = fits.getdata(filename, 0, header=True) #Extract FITs data
        except Exception:
            warnings.warn('Invalid fits at {}'.format(filename))
        imgs[filename] = img #Unsure if didctionary would be better here if required for a large quantity of data
        #imgs.append(img)
  
    final_data = {} #Create dictionary to append final data to

    for key in imgs.keys():
        final_data[key.replace('.fits', '').replace(os.getcwd() + '/' + f'{input_dir_name}' + '/', '')]=creating_image_functions.photon_counts_from_FITS(imgs[key], args.scale_factor) #Second input is scale factor, changed in parser

    for entry_name in final_data.keys():
        #creating_image_functions.make_png_from_corrected_fits(final_data[entry_name][0], os.getcwd() + '/' + f'{output_dir_name[0]}' + '/Original_' + entry_name + '.png', 424) #Might want to remove the word Original in file name?
        creating_image_functions.make_png_from_corrected_fits(final_data[entry_name][2], '/' + f'{output_dir_name[1]}' + '/Scaled_' + entry_name + '.png', 424)

