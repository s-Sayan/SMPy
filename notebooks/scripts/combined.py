from astropy.io import fits
from astropy.table import Table
import numpy as np
from tqdm import tqdm

# File paths
file_b = "/work/mccleary_group/saha/data/AbellS0592/b/out/AbellS0592_b_annular.fits"
file_g = "/work/mccleary_group/saha/data/AbellS0592/g/out/AbellS0592_g_annular.fits"
output_file = "/work/mccleary_group/saha/data/AbellS0592/combined_annular.fits"

# Define angular tolerance in degrees
tolerance_deg = 1e-3  # Adjust as needed
tolerance_rad = np.radians(tolerance_deg)

# Read the FITS files
data_b = Table.read(file_b, format='fits')
data_g = Table.read(file_g, format='fits')
print("File reading is done")

# Columns of interest
columns_to_keep = ['ra', 'dec', 'g1_Rinv', 'g2_Rinv', 'weight', 'X_IMAGE', 'Y_IMAGE', 'g_cov_noshear']

# Extract only the necessary columns
data_b = data_b[columns_to_keep]
data_g = data_g[columns_to_keep]

# Convert RA/DEC to radians
ra_b = np.radians(data_b['ra'])
dec_b = np.radians(data_b['dec'])
ra_g = np.radians(data_g['ra'])
dec_g = np.radians(data_g['dec'])

# Step 1: Initial Matching
print("Initial matching is starting...")
matches_b_to_g = -np.ones(len(data_b), dtype=int)  # To store matches
distance_b_to_g = np.full(len(data_b), np.inf)  # To store distances

for i, (ra1, dec1) in tqdm(enumerate(zip(ra_b, dec_b)), total=len(ra_b)):
    # Compute angular distances to all objects in `data_g`
    cos_distance = (
        np.sin(dec1) * np.sin(dec_g) +
        np.cos(dec1) * np.cos(dec_g) * np.cos(ra1 - ra_g)
    )
    angular_distance = np.arccos(np.clip(cos_distance, -1, 1))  # Ensure valid range

    # Find closest match within the tolerance
    closest_index = np.argmin(angular_distance)
    closest_distance = angular_distance[closest_index]

    if closest_distance < tolerance_rad:
        matches_b_to_g[i] = closest_index
        distance_b_to_g[i] = closest_distance

print(f"Number of matches after Initial matching:{len(np.where(matches_b_to_g != -1)[0])}")

# Step 2: Reassignment to Ensure Best Matches
print("Reassignment of matches is starting...")
best_match_for_g = {}  # Track the best match for each `data_g` object
for i, match in tqdm(enumerate(matches_b_to_g), total=len(matches_b_to_g)):
    if match != -1:  # If `data_b[i]` has a match
        if match not in best_match_for_g or distance_b_to_g[i] < distance_b_to_g[best_match_for_g[match]]:
            best_match_for_g[match] = i

# Create final matched indices
final_matches_b = list(best_match_for_g.values())
final_matches_g = list(best_match_for_g.keys())

# Step 3: Combine Matched Data
matched_data_b = data_b[final_matches_b]
matched_data_g = data_g[final_matches_g]

# Calculate combined weights and weighted averages
combined_weight = matched_data_b['weight'] + matched_data_g['weight']
g1_avg = (
    matched_data_b['g1_Rinv'] * matched_data_b['weight'] +
    matched_data_g['g1_Rinv'] * matched_data_g['weight']
) / combined_weight
g2_avg = (
    matched_data_b['g2_Rinv'] * matched_data_b['weight'] +
    matched_data_g['g2_Rinv'] * matched_data_g['weight']
) / combined_weight

# Extract variances from the catalogues
var_g1_b = matched_data_b['g_cov_noshear'][:, 0, 0]
var_g2_b = matched_data_b['g_cov_noshear'][:, 1, 1]
var_g1_g = matched_data_g['g_cov_noshear'][:, 0, 0]
var_g2_g = matched_data_g['g_cov_noshear'][:, 1, 1]

# Compute weights from the matched data
w_b = matched_data_b['weight']
w_g = matched_data_g['weight']

# Calculate the combined weight
numerator = w_b**2 * (var_g1_b + var_g2_b) + w_g**2 * (var_g1_g + var_g2_g)
denominator = w_b**2 + w_g**2
combined_weight1 = 1 / (0.26 + numerator / denominator)

# Add combined data to the table
combined_data = Table()
combined_data['ra'] = matched_data_b['ra']
combined_data['dec'] = matched_data_b['dec']
combined_data['X_IMAGE'] = matched_data_b['X_IMAGE']
combined_data['Y_IMAGE'] = matched_data_b['Y_IMAGE']
combined_data['g1'] = g1_avg
combined_data['g2'] = g2_avg
combined_data['weight'] = combined_weight1

# Save the combined data to a new FITS file
combined_data.write(output_file, format='fits', overwrite=True)

# Print object counts and confirmation
print(f"Number of objects in B-band file: {len(data_b)}")
print(f"Number of objects in G-band file: {len(data_g)}")
print(f"Number of matched objects: {len(combined_data)}")
print(f"Combined FITS file saved at: {output_file}")
