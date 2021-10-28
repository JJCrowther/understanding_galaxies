from PIL import Image
import numpy as np
import os


def photon_counts_from_FITS(imgs, scale_factor=1, bands='grz'):
    """
    Convert a FITS data file into a photon counts array. Begins by defining the softening parameters
    for each of the grz bands (necessary?) befroe creating an empty (3, x, x) array. Each band in 
    the image (grz) is run through the flux_to_counts function, outputting the number of photon counts
    for each pixel of the image in each band. This is appended to the empty array and pased through 
    the poisson_noise function to return a rescaled version of the image. Finally a reverse of the first 
    process is conducted to convert the scaled photon counts back into the original flux format of the
    FITs file so it can be processed into a .png later.

    Inputs:
        imgs - (3, x, x) numpy array of floats with flux values for a specific image in each band
        scale_factor - The factor by which the image is to be scaled (default set to 1)
        bands - Defines the bands over which the image is captured (default set to grz)
    
    Outputs:
        imgs - The original FITs input file of fluxes
        img_counts - The photon counts for the original FITs file before rescaling
        img_scaled - The output fluxes, rescaled for the new distance, returned to the original FITs
        formating 
    """
    
    size = imgs[0].shape[1]
    
    soft_params = dict(g=(0, 0.9e-10),
                     r=(1, 1.2e-10),
                     z=(2, 7.4e-10) #We think this is the right plane order...hopefully (Are these necessary anymore?)
                     )
                  
    img_counts = np.zeros((3, size, size), np.float32)
    for im, band in zip(imgs, bands):  #im is an (x, x)
        plane, softening_parameters = soft_params.get(band, (0, 1.))
        counts = flux_to_counts(im, softening_parameters, band) #im is a (3, x, x) array ; softening_parameters is an int
        img_counts[plane, :, :] = counts #counts is (x, x) in shape
        
    
    new_position_counts = poisson_noise(img_counts, scale_factor, size) #scaling by scale_factor (set 1 by default)
    #Should the scale factor be determined by some luminosity deistances or is a set value good enough?
    #Should the equivilent redshift be calculated here to determine if K-corrections are necessary?

    img_scaled = np.zeros((3, size, size), np.float32)
    for count, band in zip(new_position_counts, bands): #count is a (x, x)
        plane, softening_parameters = soft_params.get(band, (0, 1.))
        nMgys = counts_to_flux(count, band)
        img_scaled[plane, :, :] = nMgys #nMgys is (x, x) in shape
       
    clipped_img = np.clip(imgs, 1e-10, None) #Replaces any values in array less than 1e-10 with 1e-10.

    return clipped_img, img_counts, img_scaled

def flux_to_counts(im, softening_parameters, band, exposure_time_seconds = 90.): #Check exposure
    """
    Converts data in the form of fluxes in nanomaggies into equivilent photon counts. First defines
    the photon energy assuming a central average wavelength for each band. The FITs data is then 
    clipped to remove any negative values (problematic later when adding the noise - root(N) - and 
    poisson distribution - lambda = N). Nanomaggies are converted to fluxes via a conversion factor,
    and this is then multiplied by the pixel exposure time divided by the energy pf the appropriate band
    to obtain the photon count.

    Inputs:
        im - (x, x) array of FITs data in nMgy's 
        softening parameter - Necessary? (Used in making asinh magnitudes)
        band - The relevent band grz we are working in
        exposure_time_seconds - The collecting time for each pixel, set to 90 seconds by defualt.

    Outputs:
        img_photns - (x, x) array of floats representing photon counts within each pixel
    """
    photon_energy = dict(g=(0, 4.12e-19), #475nm
                         r=(1, 3.20e-19), #622nm
                         z=(2, 2.20e-19) #905nm
                         )# Using wavelength for each band better than a general overall wavelength
    
    size = im.shape[1]
    #img_nanomaggies_nonzero = np.clip(im, 1e-10, None) #Array with no values below 1e-9
    img_nanomaggies_nonzero = im
    img_photons = np.zeros((size, size), np.float32)

    energy = photon_energy.get(band, 1)[1] #.get has inputs (key, value) where value is returned if key deos not exist
    #flux = asinh_mag_to_flux(Im, softening_parameters)
    flux = img_nanomaggies_nonzero * 3.631e-6
    img_photons[:, :] = np.multiply(flux, (exposure_time_seconds / energy)) #the flux values reach the upper limits of float manipulation - need to be scaled by some value for operations to be conducted
    
    
    return img_photons

def counts_to_flux(counts, band, exposure_time_seconds = 90.):
    """
    Converts data from the form of photon counts into nanomaggy fluxes. Starts by defining the 
    energy for photons in each the 3 bands, using a central band wavelength. Then takes the counts 
    and divides through by the exposure time over the photon energy of the band. This igves the flux 
    in Jy which can then be converted to nMgy's via a scale factor 3.631e-6.

    Inputs:
        counts - (x, x) array of photon counts for each pixel (float)
        band - The relevent band grz we are working in
        exposure_time_seconds - The collecting time for each pixel, set to 90 seconds by defualt.

    Outputs:
        img_mgy - (x, x) array of floats representing the nMgy value for each pixel
    """
    photon_energy = dict(g=(0, 4.12e-19), #475nm
                         r=(1, 3.20e-19), #622nm
                         z=(2, 2.20e-19) #905nm
                         )# TODO assume 600nm mean freq. for gri bands, can improve this
    
    size = counts.shape[1]
    img_flux = np.zeros((size, size), np.float32)
    
    energy = photon_energy.get(band, 1)[1]
    img_flux[:, :] = counts / (exposure_time_seconds / energy)
    
    img_mgy = img_flux / 3.631e-6
        
    return img_mgy

def poisson_noise(photon_count, x, size):
    """
    Scales the photon count by 1/d^2 to account for decreased photon numbers at new position
    before adding a poissonly distributed random noise to each channel for each pixel.

    Inputs:
        photon_count - (x, x) array of photon counts for each pixel
        x - The scale factor we want to scale the image by

    Outputs:
        photon_with_poisson - (x, x) array of floats of the original input data, scaled to the new
        distance and with random poisson noise added.
    """
    photon_at_distance_scale_x = photon_count * (1/x)**2
    #photon_at_distance_scale_x = np.where(photon_count>5e10, photon_count * (1/x)**2, photon_count) #Only scales certain pixels with larger counts, imporves speckling
    photon_with_poisson = photon_at_distance_scale_x + np.random.poisson(np.sqrt(np.abs(photon_at_distance_scale_x)))
    #photon_with_poisson += np.random.poisson(5e12, (3, size, size))
    return photon_with_poisson

def make_png_from_corrected_fits(img, png_loc, png_size):
    '''
    Create png from multi-band fits
    Args:
        fits_loc (str): location of .fits to create png from
        png_loc (str): location to save png
    Returns:
        None
    '''
        # TODO wrap?

            # Set parameters for RGB image creation
    _scales = dict(
            g=(2, 0.008),
            r=(1, 0.014),
            z=(0, 0.019))
    _mnmx = (-0.5, 300)

    rgbimg = dr2_style_rgb(
            (img[0, :, :], img[1, :, :], img[2, :, :]),
            'grz',
            mnmx=_mnmx,
            arcsinh=1.,
            scales=_scales,
            desaturate=True)
    save_carefully_resized_png(png_loc, rgbimg, target_size=png_size)


def save_carefully_resized_png(png_loc, native_image, target_size):
    """
    # TODO
    Args:
        png_loc ():
        native_image ():
        target_size ():
    Returns:
    """
    native_pil_image = Image.fromarray(np.uint8(native_image * 255.), mode='RGB')
    nearest_image = native_pil_image.resize(size=(target_size, target_size), resample=Image.LANCZOS)
    nearest_image = nearest_image.transpose(Image.FLIP_TOP_BOTTOM)  # to align with north/east
    nearest_image.save(png_loc)
    #nearest_image.save(png_loc, path = Path(os.getcwd() + '/Images'))

def dr2_style_rgb(imgs, bands, mnmx=None, arcsinh=None, scales=None, desaturate=False):
    '''
    Given a list of image arrays in the given bands, returns a scaled RGB image.
    Originally written by Dustin Lang and used by Kyle Willett for DECALS DR1/DR2 Galaxy Zoo subjects
    Args:
        imgs (list): numpy arrays, all the same size, in nanomaggies
        bands (list): strings, eg, ['g','r','z']
        mnmx (min,max), values that will become black/white *after* scaling. Default is (-3,10)):
        arcsinh (bool): if True, use nonlinear scaling (as in SDSS)
        scales (str): Override preset band scaling. Dict of form {band: (plane index, scale divider)}
        desaturate (bool): If [default=False] desaturate pixels dominated by a single colour
    Returns:
        (np.array) of shape (H, W, 3) with values between 0 and 1 of pixel values for colour image
    '''

    bands = ''.join(bands)  # stick list of bands into single string

    # first number is index of that band
    # second number is scale divisor - divide pixel values by scale divisor for rgb pixel value
    grzscales = dict(
        g=(2, 0.0066),
        r=(1, 0.01385),
        z=(0, 0.025),
    )

    if scales is None:
        if bands == 'grz':
            scales = grzscales
        elif bands == 'urz':
            scales = dict(
                u=(2, 0.0066),
                r=(1, 0.01),
                z=(0, 0.025),
            )
        elif bands == 'gri':
            scales = dict(
                g=(2, 0.002),
                r=(1, 0.004),
                i=(0, 0.005),
            )
        else:
            scales = grzscales

    #  create blank matrix to work with
    h, w = imgs[0].shape
    rgb = np.zeros((h, w, 3), np.float32)

    # Copy each band matrix into the rgb image, dividing by band scale divisor to increase pixel values
    for im, band in zip(imgs, bands):
        plane, scale = scales[band]
        rgb[:, :, plane] = (im / scale).astype(np.float32)

    # TODO mnmx -> (min, max)
    # cut-off values for non-linear arcsinh map
    if mnmx is None:
        mn, mx = -3, 10
    else:
        mn, mx = mnmx

    if arcsinh is not None:
        # image rescaled by single-pixel not image-pixel, which means colours depend on brightness
        rgb = nonlinear_map(rgb, arcsinh=arcsinh)
        mn = nonlinear_map(mn, arcsinh=arcsinh)
        mx = nonlinear_map(mx, arcsinh=arcsinh)

    # lastly, rescale image to be between min and max
    rgb = (rgb - mn) / (mx - mn)

    # default False, but downloader sets True
    if desaturate:
        # optionally desaturate pixels that are dominated by a single
        # colour to avoid colourful speckled sky

        # reshape rgb from (h, w, 3) to (3, h, w)
        RGBim = np.array([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])
        a = RGBim.mean(axis=0)  # a is mean pixel value across all bands, (h, w) shape
        # putmask: given array and mask, set all mask=True values of array to new value
        np.putmask(a, a == 0.0, 1.0)  # set pixels with 0 mean value to mean of 1. Inplace?
        acube = np.resize(a, (3, h, w))  # copy mean value array (h,w) into 3 bands (3, h, w)
        bcube = (RGBim / acube) / 2.5  # bcube: divide image by mean-across-bands pixel value, and again by 2.5 (why?)
        mask = np.array(bcube)  # isn't bcube already an array?
        wt = np.max(mask, axis=0)  # maximum per pixel across bands of mean-band-normalised rescaled image
        # i.e largest relative deviation from mean
        np.putmask(wt, wt > 1.0, 1.0)  # clip largest allowed relative deviation to one (inplace?)
        wt = 1 - wt  # invert relative deviations
        wt = np.sin(wt*np.pi/2.0)  # non-linear rescaling of relative deviations
        temp = RGBim * wt + a*(1-wt) + a*(1-wt)**2 * RGBim  # multiply by weights in complicated fashion
        rgb = np.zeros((h, w, 3), np.float32)  # reset rgb to be blank
        for idx, im in enumerate((temp[0, :, :], temp[1, :, :], temp[2, :, :])):  # fill rgb with weight-rescaled rgb
            rgb[:, :, idx] = im

    clipped = np.clip(rgb, 0., 1.)  # set max/min to 0 and 1

    return clipped

def nonlinear_map(x, arcsinh=1.):
    """
    Apply non-linear map to input matrix. Useful to rescale telescope pixels for viewing.
    Args:
        x (np.array): array to have map applied
        arcsinh (np.float):
    Returns:
        (np.array) array with map applied
    """
    return np.arcsinh(x * arcsinh)