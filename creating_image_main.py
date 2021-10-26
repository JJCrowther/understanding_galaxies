import glob
import os
import warnings
from astropy.io import fits
import argparse

import creating_image_functions

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fits-dir', dest='fits_dir', type=str)
    parser.add_argument('--scale-factor', dest='scale_factor', default=1, type=float)

    args = parser.parse_args()

    input_dir_name=  args.fits_dir
    assert os.path.isdir(input_dir_name)

    #input_dir_name='FITSdata' #Sets the name of the folder FITs files are stored in
    output_dir_name = ['original_images', 'scaled_images'] # Sets the name of the folder which holds the input images for the CNN

    # '/**/*.fits', recursive=True):
    imgs = {} # Opens dictionary for storing images
    filenames = glob.iglob(f'{input_dir_name}' + '/*.fits') #operates over all FIT's within the desired directory
    # print(filenames)
    # filenames = list(filenames)[:5]
    for filename in filenames:
        try:
            img, hdr = fits.getdata(filename, 0, header=True) #Extract FITs data
        except Exception:
            warnings.warn('Invalid fits at {}'.format(filename))
        imgs[filename] = img #Unsure if didctionary would be better here if required for a large quantity of data
        #imgs.append(img)
  
    final_data = {} # Create dictionary to append final data to

    for original_loc, original_img in imgs.items():
        filename = os.path.basename(original_loc)
        filename = filename.replace('.fits', '.png')
        file_loc = os.path.join('/share/nas/walml/repos/understanding_galaxies', output_dir_name[1], filename)

        _, _, img_scaled = creating_image_functions.photon_counts_from_FITS(original_img, args.scale_factor) # Second input is scale factor, changed in parser

        final_data[file_loc] = img_scaled 

    for save_loc, scaled_image in final_data.items():
        #creating_image_functions.make_png_from_corrected_fits(final_data[entry_name][0], os.getcwd() + '/' + f'{output_dir_name[0]}' + '/Original_' + entry_name + '.png', 424) #Might want to remove the word Original in file name?
        creating_image_functions.make_png_from_corrected_fits(
            img=scaled_image,
            png_loc=save_loc,
            png_size=424)

    print('Successfully made images - exiting')
