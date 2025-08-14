### Running the `get_snr.py` Script

To run the `get_snr.py` script, use the following command in your terminal:

```bash
python get_snr.py -c config.yaml --save_fits [--plot_kappa]
```

Make sure you have all the necessary dependencies installed before running the script.

Modify the `config_Abell3411.yaml` file to include the correct path for the catalogue. For example:

```yaml
input_path: /path/to/your/catalogue/file
```

Two plots, the SNR map and the radially averaged SNR, will be saved in the `../plots` folder.