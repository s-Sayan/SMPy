import argparse
import numpy as np
import pandas as pd
from astropy.table import Table
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
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
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

    true_boundaries = utils.calculate_field_boundaries_v2(shear_df['ra'], shear_df['dec'])
    shear_df, ra_0, dec_0 = utils.correct_RA_dec(shear_df)
    boundaries = utils.calculate_field_boundaries_v2(shear_df['ra'], shear_df['dec'])
    boundaries_xy = utils.calculate_field_boundaries_v2(shear_df['x'], shear_df['y'])

    center_cl = correct_center(config["center"], ra_0, dec_0)

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

    shuffled_dfs = utilsv2.generate_multiple_shear_dfs(shear_df, config['num_sims'], seed=config['seed_sims'])
    g1_g2_map_list_xy = utils.shear_grids_for_shuffled_dfs_xy(shuffled_dfs, resolution_xy)
    shuff_kappa_e_list_xy, shuff_kappa_b_list_xy = utils.ks_inversion_list(g1_g2_map_list_xy, 'xy')

    kernel = config['gaussian_kernel']

    kappa_e_stack_xy = np.stack(shuff_kappa_e_list_xy, axis=0)
    kappa_e_stack_smoothed_xy = np.zeros_like(kappa_e_stack_xy)
    for i in range(kappa_e_stack_xy.shape[0]):
        kappa_e_stack_smoothed_xy[i] = gaussian_filter(kappa_e_stack_xy[i], kernel)
    
    std_xy = np.std(kappa_e_stack_smoothed_xy, axis=0)
    snr_xy = gaussian_filter(og_kappa_e_2, kernel) / std_xy

    # Plotting SNR map
    plot_kmap.plot_convergence_v4(
        snr_xy, 
        boundaries, 
        true_boundaries, 
        config, 
        invert_map=False, 
        title=config['plot_title'] + f" (Resolution: {config['resolution']:.2f} arcmin, Kernel: {kernel:.2f})",
        vmax=config['vmax'], 
        threshold=config['threshold']
    )
    x_c, y_c, hx = utils.find_peaks_v2(og_kappa_e_2, boundaries_xy, smoothing=kernel, threshold=0.11)
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
    plt.savefig("../plots/SNR_multiple_kernels.png")
