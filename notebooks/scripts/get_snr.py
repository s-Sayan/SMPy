import argparse
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import yaml
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from SMPy import utilsv2, utils
from SMPy.KaiserSquires import kaiser_squires, plot_kmap

# Function to read the YAML config file
def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Command-line argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a shear catalog and compute the SNR map.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('--plot_kappa', action='store_true', help='Flag to plot convergence map')
    parser.add_argument('--plot_error', action='store_true', help='Flag to plot the std map')
    parser.add_argument('--save_fits', action='store_true', help='Flag to save the convergence map as a FITS file')  # New flag
    return parser.parse_args()

# Function to load shear data
def load_shear_data(shear_cat_path, ra_col, dec_col, g1_col, g2_col, weight_col, x_col, y_col):
    shear_catalog = Table.read(shear_cat_path)
    shear_df = pd.DataFrame({
        'ra': shear_catalog[ra_col],
        'dec': shear_catalog[dec_col],
        'g1': shear_catalog[g1_col],
        'g2': shear_catalog[g2_col],
        'weight': shear_catalog[weight_col],
        'x': shear_catalog[x_col],
        'y': shear_catalog[y_col],
    })
    return shear_df

# Function to correct cluster center
def correct_center(center_cl, ra_0, dec_0):
    center_c = {}
    center_c["ra_center"] = (center_cl["ra_center"] - ra_0) * np.cos(np.deg2rad(center_cl["dec_center"]))
    center_c["dec_center"] = center_cl["dec_center"] - dec_0
    return center_c

# Function to save a FITS file
def save_fits(data, true_boundaries, filename):
    """
    Save a 2D array as a FITS file with proper WCS information.

    - data: 2D numpy array containing the map.
    - true_boundaries: Dictionary with 'ra_min', 'ra_max', 'dec_min', 'dec_max'.
    - filename: Output filename.
    """
    hdu = fits.PrimaryHDU(data)
    header = hdu.header

    ny, nx = data.shape
    ra_min, ra_max = true_boundaries['ra_min'], true_boundaries['ra_max']
    dec_min, dec_max = true_boundaries['dec_min'], true_boundaries['dec_max']

    pixel_scale_ra = (ra_max - ra_min) / nx
    pixel_scale_dec = (dec_max - dec_min) / ny

    header["CTYPE1"] = "RA---TAN"
    header["CUNIT1"] = "deg"
    header["CRVAL1"] = (ra_max + ra_min) / 2
    header["CRPIX1"] = nx / 2
    header["CD1_1"]  = -pixel_scale_ra
    header["CD1_2"]  = 0.0

    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT2"] = "deg"
    header["CRVAL2"] = (dec_max + dec_min) / 2
    header["CRPIX2"] = ny / 2
    header["CD2_1"]  = 0.0
    header["CD2_2"]  = pixel_scale_dec

    hdu.writeto(filename, overwrite=True)
    print(f"Saved FITS file: {filename}")


# Main script logic
if __name__ == "__main__":
    args = parse_arguments()
    config = read_config(args.config)

    # Load and preprocess shear data
    shear_df = load_shear_data(
        config['input_path'], 
        config['ra_col'], 
        config['dec_col'], 
        config['g1_col'], 
        config['g2_col'], 
        config['weight_col'], 
        config['x_col'], 
        config['y_col']
    )
    
    print(f"Loaded shear data from {config['input_path']}")

    true_boundaries = utils.calculate_field_boundaries_v2(shear_df['ra'], shear_df['dec'])
    shear_df, ra_0, dec_0 = utils.correct_RA_dec(shear_df)
    boundaries = utils.calculate_field_boundaries_v2(shear_df['ra'], shear_df['dec'])
    boundaries_xy = utils.calculate_field_boundaries_v2(shear_df['x'], shear_df['y'])

    if config['center'] is not None:
        center_cl = correct_center(config["center"], ra_0, dec_0)
    else:
        center_cl = None

    x_factor, y_factor = (
        (np.max(shear_df['x']) - np.min(shear_df['x'])) / (np.max(shear_df['ra']) - np.min(shear_df['ra'])), 
        (np.max(shear_df['y']) - np.min(shear_df['y'])) / (np.max(shear_df['dec']) - np.min(shear_df['dec']))
    )
    factor = (x_factor + y_factor) / 2
    resolution_xy = int(config["resolution"] * factor)

    g1map_og_2, g2map_og_2 = utils.create_shear_grid_v2(
        shear_df['x'], shear_df['y'], shear_df['g1'], shear_df['g2'], resolution_xy, shear_df['weight'], verbose=True
    )

    og_kappa_e_2, og_kappa_b_2 = kaiser_squires.ks_inversion(g1map_og_2, g2map_og_2)
    kernel = config['gaussian_kernel']
    og_kappa_e_2_smoothed = gaussian_filter(og_kappa_e_2, kernel)


    if args.plot_kappa:
        plot_kmap.plot_convergence_v4(
            og_kappa_e_2_smoothed, 
            boundaries, 
            true_boundaries, 
            config, 
            invert_map=False, 
            title="Convergence: "+config['cluster']+"_"+config['band'] + f" (Resolution: {config['resolution']:.2f} arcmin, Kernel: {kernel:.2f})",
            #vmax= 0.085,
            #vmin= -0.085, 
            #threshold=config['threshold'],
            center_cl=center_cl,
            save_path=config['output_path']+"kappa_"+config['cluster']+"_"+config['band']+".png"
        )
    if args.save_fits:
        fits_filename = config['output_path'] + f"kappa_{config['cluster']}_{config['band']}.fits"
        save_fits(og_kappa_e_2_smoothed, true_boundaries, fits_filename)
    
    if config["shuffle_type"] == "rotation":
        shuffled_dfs = utilsv2.generate_multiple_shear_dfs(shear_df, config['num_sims'], seed=config['seed_sims'])
    elif config["shuffle_type"] == "spatial":
        shuffled_dfs = utils.generate_multiple_shear_dfs(shear_df, config['num_sims'], seed=config['seed_sims'])
    else:
        KeyError("Invalid shuffle type. Must be either 'rotation' or 'spatial'.")

    shuffled_dfs = utilsv2.generate_multiple_shear_dfs(shear_df, config['num_sims'], seed=config['seed_sims'])
    g1_g2_map_list_xy = utils.shear_grids_for_shuffled_dfs_xy(shuffled_dfs, resolution_xy)
    shuff_kappa_e_list_xy, shuff_kappa_b_list_xy = utils.ks_inversion_list(g1_g2_map_list_xy, 'xy')

    kappa_e_stack_xy = np.stack(shuff_kappa_e_list_xy, axis=0)
    kappa_e_stack_smoothed_xy = np.zeros_like(kappa_e_stack_xy)
    for i in range(kappa_e_stack_xy.shape[0]):
        kappa_e_stack_smoothed_xy[i] = gaussian_filter(kappa_e_stack_xy[i], kernel)
    
    std_xy = np.std(kappa_e_stack_smoothed_xy, axis=0)
    snr_xy = gaussian_filter(og_kappa_e_2, kernel) / std_xy
    if args.plot_error:
        plot_kmap.plot_convergence_v4(
            std_xy,
            boundaries,
            true_boundaries,
            config,
            invert_map=False,
            title="Error: "+config['cluster']+"_"+config['band'] + f" (Resolution: {config['resolution']:.2f} arcmin, Kernel: {kernel:.2f})",
            #vmax=config['vmax'],
            #threshold=config['threshold'],
            center_cl=center_cl,
            save_path=config['output_path']+"error_"+config['cluster']+"_"+config['band']+".png"
        )

    # Plotting SNR map
    plot_kmap.plot_convergence_v4(
        snr_xy, 
        boundaries, 
        true_boundaries, 
        config, 
        invert_map=False, 
        title=config['plot_title']+config['cluster']+"_"+config['band'] + f" (Resolution: {config['resolution']:.2f} arcmin, Kernel: {kernel:.2f})",
        vmax=config['vmax'], 
        threshold=config['threshold'],
        center_cl=center_cl,
        save_path=config['output_path']+"snr_"+config['cluster']+"_"+config['band']+".png"
    )
    '''x_c, y_c, hx = utils.find_peaks_v2(og_kappa_e_2, boundaries_xy, smoothing=kernel, threshold=0.11)
    center_cl_xy = {'ra_center': x_c, 'dec_center': y_c }

    # Radial profiles for different kernels
    kernels = np.array(config['kernels'])
    plt.figure(figsize=(8, 6))

    for kernel in kernels:
        kappa_e_stack_smoothed_xy = np.zeros_like(kappa_e_stack_xy)
        for i in range(kappa_e_stack_xy.shape[0]):
            kappa_e_stack_smoothed_xy[i] = gaussian_filter(kappa_e_stack_xy[i], kernel)
        
        std_xy = np.std(kappa_e_stack_smoothed_xy, axis=0)
        snr_xy = gaussian_filter(og_kappa_e_2, kernel) / std_xy

        profile_convergence, error_convergence, r_bin_centers_arcmin = utils.profile_1D(
        snr_xy, boundaries_xy, resolution_xy, center_cl_xy, smoothing=0
        )
        
        plt.plot(r_bin_centers_arcmin, profile_convergence, label=f'Kernel = {kernel:.2f}')
        plt.fill_between(
            r_bin_centers_arcmin, 
            profile_convergence - error_convergence, 
            profile_convergence + error_convergence, 
            alpha=0.15
        )

    plt.xlabel('Radial Distance (arcmin)')
    plt.ylabel('Azimuthally Averaged Value')
    plt.title('Radial Profiles of SNR for Different Kernels')
    plt.legend(loc='upper right', fontsize='small', title='Smoothing Kernels')
    plt.grid(True)
    plt.xlim(0, 12)
    plt.tight_layout()
    #plt.show(block=True)
    plt.savefig(config['output_path']+"SNR_multiple_kernels.pdf")'''
