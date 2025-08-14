from ruamel.yaml import YAML
import argparse

# Set up argument parsing with short options
parser = argparse.ArgumentParser(description="Update cluster, band, and input_path in config.yaml")
parser.add_argument("-c", "--cluster", required=True, help="Cluster name (e.g., Abell3411)")
parser.add_argument("-b", "--band", required=True, help="Band name (e.g., g, r)")

args = parser.parse_args()
cluster_name = args.cluster
band_name = args.band

# Load the YAML file
yaml = YAML()
config_file = "config.yaml"

with open(config_file, 'r') as file:
    config = yaml.load(file)

# Update cluster, band, and input_path
config['cluster'] = cluster_name
config['band'] = band_name
config['input_path'] = f"/projects/mccleary_group/saha/data/{cluster_name}/{band_name}/out/{cluster_name}_{band_name}_annular.fits"

# Save the updated YAML file
with open(config_file, 'w') as file:
    yaml.dump(config, file)

print(f"Updated config.yaml:")
print(f"  cluster: {config['cluster']}")
print(f"  band: {config['band']}")
print(f"  input_path: {config['input_path']}")
