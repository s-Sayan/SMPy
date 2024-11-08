import numpy as np
import pandas as pd
import random
from SMPy.KaiserSquires import kaiser_squires

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS

def load_shear_data(shear_cat_path, ra_col, dec_col, g1_col, g2_col, weight_col):
    """ 
    Load shear data from a FITS file and return a pandas DataFrame.

    :param path: Path to the FITS file.
    :param ra_col: Column name for right ascension.
    :param dec_col: Column name for declination.
    :param g1_col: Column name for the first shear component.
    :param g2_col: Column name for the second shear component.
    :param weight_col: Column name for the weight.
    :return: pandas DataFrame with the specified columns.
    """
    # Read data from the FITS file
    shear_catalog = Table.read(shear_cat_path)

    # Convert to pandas DataFrame
    shear_df = pd.DataFrame({
        'ra': shear_catalog[ra_col],
        'dec': shear_catalog[dec_col],
        'g1': shear_catalog[g1_col],
        'g2': shear_catalog[g2_col],
        'weight': shear_catalog[weight_col]
    })

    return shear_df

def load_shear_data_v2(shear_cat_path, ra_col, dec_col, g1_col, g2_col, weight_col, mu_col, obj_class_col):
    """ 
    Load shear data from a FITS file and return a pandas DataFrame.

    :param path: Path to the FITS file.
    :param ra_col: Column name for right ascension.
    :param dec_col: Column name for declination.
    :param g1_col: Column name for the first shear component.
    :param g2_col: Column name for the second shear component.
    :param weight_col: Column name for the weight.
    :return: pandas DataFrame with the specified columns.
    """
    # Read data from the FITS file
    shear_catalog = Table.read(shear_cat_path)

    # Convert to pandas DataFrame
    shear_df = pd.DataFrame({
        'ra': shear_catalog[ra_col],
        'dec': shear_catalog[dec_col],
        'g1': shear_catalog[g1_col],
        'g2': shear_catalog[g2_col],
        'weight': shear_catalog[weight_col],
        'mu': shear_catalog[mu_col],
        'obj_class': shear_catalog[obj_class_col]
    })
    gal_idx = np.where(shear_df['obj_class'] == b'gal')[0]
    
    shear_df_c = []

    for i in gal_idx:
        shear_df_c.append(shear_df.iloc[i])
        
    shear_df_c = pd.DataFrame(shear_df_c)
    return shear_df_c

def correct_RA_dec(shear_df):
    shear_df_f = shear_df.copy()
    ra = shear_df['ra']
    dec = shear_df['dec']
    ra_0 = (np.max(ra) + np.min(ra))/2 # center of ra, set as a refernce for this transformation
    dec_0 = (np.max(dec) + np.min(dec))/2 # center of dec, set as a refernce for this transformation
    
    ra_flat, dec_flat = np.zeros(len(ra)), np.zeros(len(dec))
    for i in range(len(ra)):
        ra_flat[i] = (ra[i] - ra_0) * np.cos(np.deg2rad(dec[i]))
        dec_flat[i] = dec[i] - dec_0
    shear_df_f['ra'] = ra_flat
    shear_df_f['dec'] = dec_flat
    return shear_df_f, ra_0, dec_0

def calculate_field_boundaries(ra, dec):
    """
    Calculate the boundaries of the field in right ascension (RA) and declination (Dec).
    
    :param ra: Dataframe column containing the right ascension values.
    :param dec: Dataframe column containing the declination values.
    :param resolution: Resolution of the map in arcminutes.
    :return: A dictionary containing the corners of the map {'ra_min', 'ra_max', 'dec_min', 'dec_max'}.
    """
    # Calculate median RA and Dec
    med_ra = np.median(ra)
    med_dec = np.median(dec)
    
    # Calculate the range of RA and Dec values
    ra_range = np.max(ra) - np.min(ra)
    dec_range = np.max(dec) - np.min(dec)
    
    # Calculate the size of the field in degrees
    ra_size = ra_range
    dec_size = dec_range
    
    # Calculate RA and Dec extents and store in a dictionary
    boundaries = {
        'ra_min': med_ra - ra_size / 2,
        'ra_max': med_ra + ra_size / 2,
        'dec_min': med_dec - dec_size / 2,
        'dec_max': med_dec + dec_size / 2
    }
    
    return boundaries

def calculate_field_boundaries_v2(ra, dec):
    """
    Calculate the boundaries of the field in right ascension (RA) and declination (Dec).
    
    :param ra: Dataframe column containing the right ascension values.
    :param dec: Dataframe column containing the declination values.
    :param resolution: Resolution of the map in arcminutes.
    :return: A dictionary containing the corners of the map {'ra_min', 'ra_max', 'dec_min', 'dec_max'}.
    """
    boundaries = {
        'ra_min': np.min(ra),
        'ra_max': np.max(ra),
        'dec_min': np.min(dec),
        'dec_max': np.max(dec)
    }
    
    return boundaries

def create_shear_grid(ra, dec, g1, g2, weight, boundaries, resolution):
    '''
    Bin values of shear data according to position on the sky.
    '''
    ra_min, ra_max = boundaries['ra_min'], boundaries['ra_max']
    dec_min, dec_max = boundaries['dec_min'], boundaries['dec_max']
    
    # Calculate number of pixels based on field size and resolution
    npix_ra = int(np.ceil((ra_max - ra_min) * 60 / resolution))
    npix_dec = int(np.ceil((dec_max - dec_min) * 60 / resolution))
    
    ra_bins = np.linspace(ra_min, ra_max, npix_ra + 1)
    dec_bins = np.linspace(dec_min, dec_max, npix_dec + 1)
    
    if weight is None:
        weight = np.ones_like(ra)
    
    # Digitize the RA and Dec to find bin indices
    ra_idx = np.digitize(ra, ra_bins) - 1
    dec_idx = np.digitize(dec, dec_bins) - 1
    
    # Filter out indices that are outside the grid boundaries
    valid_mask = (ra_idx >= 0) & (ra_idx < npix_ra) & (dec_idx >= 0) & (dec_idx < npix_dec)
    ra_idx = ra_idx[valid_mask]
    dec_idx = dec_idx[valid_mask]
    g1 = g1[valid_mask]
    g2 = g2[valid_mask]
    weight = weight[valid_mask]
    
    # Initialize the grids
    g1_grid = np.zeros((npix_dec, npix_ra))
    g2_grid = np.zeros((npix_dec, npix_ra))
    weight_grid = np.zeros((npix_dec, npix_ra))
    
    # Accumulate weighted values using np.add.at
    np.add.at(g1_grid, (dec_idx, ra_idx), g1 * weight)
    np.add.at(g2_grid, (dec_idx, ra_idx), g2 * weight)
    np.add.at(weight_grid, (dec_idx, ra_idx), weight)
    
    # Normalize the grid by the total weight in each bin (weighted average)
    #try with commented out 
    nonzero_weight_mask = weight_grid != 0
    g1_grid[nonzero_weight_mask] /= weight_grid[nonzero_weight_mask]
    g2_grid[nonzero_weight_mask] /= weight_grid[nonzero_weight_mask]
    
    return g1_grid, g2_grid

def create_shear_grid_v2(ra, dec, g1, g2, resolution, weight=None, boundaries = None, verbose=False):
    '''
    Bin values of shear data according to position on the sky with an option of not having a specified boundary.
    
    Args:
    - ra, dec, g1, g2, weight: numpy arrays of the same length containing the shear data.
    - resolution: Resolution of the map in arcminutes.
    - boundaries: Dictionary containing 'ra_min', 'ra_max', 'dec_min', 'dec_max'.
    - verbose: If True, print details of the binning.
    Returns:
    - A tuple of two 2D numpy arrays containing the binned g1 and g2 values.
    '''
    
    if boundaries is not None:
        ra_min, ra_max = boundaries['ra_min'], boundaries['ra_max']
        dec_min, dec_max = boundaries['dec_min'], boundaries['dec_max']
    else:
        ra_min, ra_max = np.min(ra), np.max(ra)
        dec_min, dec_max = np.min(dec), np.max(dec)
        
    if weight is None:
        weight = np.ones_like(ra)
    #print(ra_min, ra_max, dec_min, dec_max)
    # Calculate number of pixels based on field size and resolution
    npix_ra = int(np.ceil((ra_max - ra_min) * 60 / resolution))
    npix_dec = int(np.ceil((dec_max - dec_min) * 60 / resolution))
    
    #print(npix_ra, npix_dec)
    
    # Initialize the grids
    wmap, xbins, ybins = np.histogram2d(ra, dec, bins=[npix_ra, npix_dec], range=[[ra_min, ra_max], [dec_min, dec_max]],
                                            weights=weight)
    
    wmap[wmap == 0] = np.inf
    # Compute mean values per pixel
    result = tuple((np.histogram2d(ra, dec, bins=[npix_ra, npix_dec], range=[[ra_min, ra_max], [dec_min, dec_max]],
                    weights=(vv * weight))[0] / wmap).T for vv in [g1, g2])
    
    if verbose:
        print("npix : {}".format([npix_ra, npix_dec]))
        print("extent : {}".format([xbins[0], xbins[-1], ybins[0], ybins[-1]]))
        print("(dx, dy) : ({}, {})".format(xbins[1] - xbins[0],
                                           ybins[1] - ybins[0]))
        
    return result


def save_convergence_fits(convergence, boundaries, config, output_name):
    """
    Save the convergence map as a FITS file with WCS information if configured to do so.

    Parameters:
    -----------
    convergence : numpy.ndarray
        The 2D convergence map.
    boundaries : dict
        Dictionary containing 'ra_min', 'ra_max', 'dec_min', 'dec_max'.
    config : dict
        Configuration dictionary containing output path and other settings.

    Returns:
    --------
    None
    """
    if not config.get('save_fits', False):
        return

    # Create a WCS object
    wcs = WCS(naxis=2)
    
    # Set up the WCS parameters
    npix_dec, npix_ra = convergence.shape
    wcs.wcs.crpix = [npix_ra / 2, npix_dec / 2]
    wcs.wcs.cdelt = [(boundaries['ra_max'] - boundaries['ra_min']) / npix_ra, 
                     (boundaries['dec_max'] - boundaries['dec_min']) / npix_dec]
    wcs.wcs.crval = [(boundaries['ra_max'] + boundaries['ra_min']) / 2, 
                     (boundaries['dec_max'] + boundaries['dec_min']) / 2]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Create a FITS header from the WCS information
    header = wcs.to_header()

    # Add some additional information to the header
    header['AUTHOR'] = 'SMPy'
    header['CONTENT'] = 'Convergence Map'

    # Create a primary HDU containing the convergence map
    hdu = fits.PrimaryHDU(convergence, header=header)

    # Create a FITS file
    hdul = fits.HDUList([hdu])

    # Save the FITS file
    hdul.writeto(output_name, overwrite=True)

    print(f"Convergence map saved as FITS file: {output_name}")


def _shuffle_ra_dec(shear_df):
    """
    Shuffle the 'ra' and 'dec' columns of the input DataFrame together.
    
    :param shear_df: Input pandas DataFrame.
    :return: A new pandas DataFrame with shuffled 'ra' and 'dec' columns.
    """
    # Make a copy to avoid modifying the original
    shuffled_df = shear_df.copy()

    # Combine RA and DEC into pairs
    ra_dec_pairs = list(zip(shuffled_df['ra'], shuffled_df['dec']))
    x_y_pairs = list(zip(shuffled_df['x'], shuffled_df['y']))
    
    # Shuffle the pairs
    random.shuffle(ra_dec_pairs)
    random.shuffle(x_y_pairs)
    
    # Unzip the shuffled pairs back into RA and DEC
    shuffled_ra, shuffled_dec = zip(*ra_dec_pairs)
    shuffled_x, shuffled_y = zip(*x_y_pairs)
    
    shuffled_df['ra'] = shuffled_ra
    shuffled_df['dec'] = shuffled_dec
    shuffled_df['x'] = shuffled_x
    shuffled_df['y'] = shuffled_y

    return shuffled_df

def _shuffle_galaxy_rotation(shear_df):
    """The function will shuffle the galaxy rotation in the input shear_df DataFrame.

    Args:
        shear_df (_type_): _description_
    """
    
    # Make a copy to avoid modifying the original
    shuffled_df = shear_df.copy()
    
    # Shuffle the galaxy rotation
    g1, g2 = shuffled_df['g1'], shuffled_df['g2']
    
    # Add a random angle to the galaxy rotation
    angle = np.random.uniform(0, 2 * np.pi, len(g1))
    g1g2_len = np.sqrt(np.array(g1)**2 + np.array(g2)**2)
    g1g2_angle  = np.arctan2(g2, g1) + angle
    g1_new = g1g2_len * np.cos(g1g2_angle)
    g2_new = g1g2_len * np.sin(g1g2_angle)
    
    shuffled_df['g1'] = g1_new
    shuffled_df['g2'] = g2_new
    
    return shuffled_df
    

def generate_multiple_shear_dfs(og_shear_df, num_shuffles=100, seed=42):
    """
    Generate a list of multiple data frames with shuffled RA and DEC columns by calling the load and shuffle functions.
    :return: A list of shuffled pandas DataFrames.
    """

    # List to store the shuffled data frames (not sure if a list of these data frames is the best format rn)
    shuffled_dfs = []
    
    #set a seed for reproducibility
    random.seed(seed)
        
    # Loop to generate multiple shuffled data frames
    for i in range(num_shuffles):
        shuffled_df = _shuffle_galaxy_rotation(og_shear_df)
        shuffled_dfs.append(shuffled_df)
    
    return shuffled_dfs

def shear_grids_for_shuffled_dfs(list_of_dfs, config, boundaries=None): 
    grid_list = []
    for shear_df in list_of_dfs: 
        g1map, g2map = create_shear_grid_v2(shear_df['ra'], 
                                           shear_df['dec'], 
                                           shear_df['g1'],
                                           shear_df['g2'],
                                           config['resolution'], 
                                           shear_df['weight'], 
                                           boundaries=boundaries)

        grid_list.append((g1map, g2map))

    return grid_list

def shear_grids_for_shuffled_dfs_xy(list_of_dfs, resolution_xy, boundaries=None): 
    grid_list = []
    for shear_df in list_of_dfs: 
        g1map, g2map = create_shear_grid_v2(shear_df['x'], 
                                           shear_df['y'], 
                                           shear_df['g1'],
                                           shear_df['g2'],
                                           resolution_xy, 
                                           shear_df['weight'], 
                                           boundaries=boundaries)

        grid_list.append((g1map, g2map))

    return grid_list

def ks_inversion_list(grid_list, key="ra_dec"):
    """
    Iterate through a list of (g1map, g2map) pairs and return a list of kappa_e values.
    Parameters:
    grid_list : list of tuples
        A list where each element is a tuple of (g1map, g2map)
    Returns:
    kappa_e_list, kappa_b_list : list
        A list containing the kappa_e_maps for each (g1map, g2map) pair, likewise for kappa_b_maps
    """
    kappa_e_list = []
    kappa_b_list = []
    
    for g1map, g2map in grid_list:
        # Call the ks_inversion function for each pair
        if key == "ra_dec":
            kappa_e, kappa_b = kaiser_squires.ks_inversion(g1map, -g2map)
        elif key == "xy":
            kappa_e, kappa_b = kaiser_squires.ks_inversion(g1map, g2map)
        else:
            raise ValueError("Unknown key, must be either 'ra_dec' or 'xy'")
        kappa_e_list.append(kappa_e)
        kappa_b_list.append(kappa_b)
    
    return kappa_e_list, kappa_b_list


def g1g2_to_gt_gc(g1, g2, ra, dec, ra_c, dec_c, pix_ra = 100):
    """
    Convert reduced shear to tangential and cross shear (Eq. 10, 11 in McCleary et al. 2023).
    args:
    - g1, g2: Reduced shear components.
    - ra, dec: Right ascension and declination of the catalogue,i.e. shear_df['ra'], shear_df['dec'].
    - ra_c, dec_c: Right ascension and declination of the cluster-centre.
    
    returns:
    - gt, gc: Tangential and cross shear components.
    - phi: Polar angle in the plane of the sky.
    """ 
    ra_max, ra_min, dec_max, dec_min = np.max(ra), np.min(ra), np.max(dec), np.min(dec)
    aspect_ratio = (ra_max - ra_min) / (dec_max - dec_min)
    pix_dec = int(pix_ra / aspect_ratio)
    ra_grid, dec_grid = np.meshgrid(np.linspace(ra_min, ra_max, pix_ra), np.linspace(dec_min, dec_max, pix_dec))

    phi = np.arctan2(dec_grid - dec_c, ra_grid - ra_c)
    
    # Calculate the tangential and cross components
    gt = -g1 * np.cos(2 * phi) - g2 * np.sin(2 * phi)
    gc = -g1 * np.sin(2 * phi) + g2 * np.cos(2 * phi)

    return gt, gc, phi