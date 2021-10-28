import glob
import os
import warnings
from astropy.io import fits
import argparse

import creating_image_functions

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fits-dir', dest='fits_dir', type=str)
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--scale-factor', dest='scale_factor', default=1, type=float)

    args = parser.parse_args()

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
        filename = os.path.basename(original_loc)
        filename = filename.replace('.fits', '.png')
        # file_loc = os.path.join('/share/nas/walml/repos/understanding_galaxies', output_dir_name[1], filename)
        file_loc = os.path.join(save_dir, filename)

        _, _, img_scaled = creating_image_functions.photon_counts_from_FITS(original_img, args.scale_factor) # Second input is scale factor, changed in parser

        final_data[file_loc] = img_scaled 

    print('All images scaled to {}'.format(args.scale_factor))

    for save_loc, scaled_image in final_data.items():
        #creating_image_functions.make_png_from_corrected_fits(final_data[entry_name][0], os.getcwd() + '/' + f'{output_dir_name[0]}' + '/Original_' + entry_name + '.png', 424) #Might want to remove the word Original in file name?
        creating_image_functions.make_png_from_corrected_fits(
            img=scaled_image,
            png_loc=save_loc,
            png_size=424)

    print('Successfully made images - exiting')
