import argparse
import numpy as np
import pandas as pd
from astropy.table import Table
import yaml
import matplotlib.pyplot as plt

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
def load_shear_data(shear_cat_path, ra_col, dec_col, x_col, y_col, r11, r22, r12, r21):
    """Load shear data from a FITS file and return a pandas DataFrame."""
    shear_catalog = Table.read(shear_cat_path)
    shear_df = pd.DataFrame({
        'ra': shear_catalog[ra_col],
        'dec': shear_catalog[dec_col],
        'x': shear_catalog[x_col],
        'y': shear_catalog[y_col],
        'r11': shear_catalog[r11],
        'r22': shear_catalog[r22],
        'r12': shear_catalog[r12],
        'r21': shear_catalog[r21],
    })
    return shear_df

# Main script logic
if __name__ == "__main__":
    args = parse_arguments()
    config = read_config(args.config)
    
    # Load shear data
    shear_df1 = load_shear_data(
        config['input_path1'], 
        config['ra_col'], 
        config['dec_col'],
        config['x_col'],
        config['y_col'],
        config['r11'],
        config['r22'],
        config['r12'],
        config['r21']
    )
    shear_df2 = load_shear_data(
        config['input_path2'], 
        config['ra_col'], 
        config['dec_col'],
        config['x_col'],
        config['y_col'],
        config['r11'],
        config['r22'],
        config['r12'],
        config['r21']
    )

    # Ensure byte order compatibility
    shear_df1 = shear_df1.astype(shear_df1['r11'].dtype.newbyteorder('='))
    shear_df2 = shear_df2.astype(shear_df2['r11'].dtype.newbyteorder('='))
    print(f"Loaded first mcal data from {config['input_path1']}")
    print(f"Loaded second mcal data from {config['input_path2']}")
    
    # Print means
    print("==== First mcal file ====")
    print(f"Mean r11: {np.mean(shear_df1['r11'])}")
    print(f"Mean r22: {np.mean(shear_df1['r22'])}")
    print("==== Second mcal file ====")
    print(f"Mean r11: {np.mean(shear_df2['r11'])}")
    print(f"Mean r22: {np.mean(shear_df2['r22'])}")
    
    # Scatter plot of x, y from the two files
    plt.scatter(shear_df1['x'], shear_df1['y'], label='First mcal data', alpha=0.5)
    plt.scatter(shear_df2['x'], shear_df2['y'], label='Second mcal data', alpha=0.5)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot of x and y')
    plt.show()
    
    # Create histograms for r11 and r22 in subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Histogram for r11
    axs[0].hist(shear_df1['r11'], bins="auto", alpha=0.5, label='First mcal data', color='blue')
    axs[0].hist(shear_df2['r11'], bins="auto", alpha=0.5, label='Second mcal data', color='orange')
    axs[0].set_title('Histogram of r11')
    axs[0].set_xlabel('r11')
    axs[0].set_xlim(-3, 3)  # Set x-axis limits
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    # Histogram for r22
    axs[1].hist(shear_df1['r22'], bins="auto", alpha=0.5, label='First mcal data', color='green')
    axs[1].hist(shear_df2['r22'], bins="auto", alpha=0.5, label='Second mcal data', color='red')
    axs[1].set_title('Histogram of r22')
    axs[1].set_xlabel('r22')
    axs[1].set_xlim(-3, 3)  # Set x-axis limits
    axs[1].legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('histogram_r11_r22.pdf')
    plt.show()
