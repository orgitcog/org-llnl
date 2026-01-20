""" 
This script generates PSD features and labels at the layer-level.
""" 

import os
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt

# parameter variables
DATA_DIR = '/Users/ciardi1/Development/sensor_fusion/data/processed_signal'
SAMPLE_DIRS = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])
OUTPUT_FILE = '/Users/ciardi1/Development/sensor_fusion/data/psd_layer.parquet'

# sample number - regime label dictionary
regime_dict = {6: 'conduction', 
               7: 'conduction',
               8: 'conduction',
               18: 'lack-of-fusion',
               19: 'lack-of-fusion',
               20: 'lack-of-fusion', 
               21: 'keyhole',
               22: 'keyhole',
               23: 'keyhole'}

sensors = ['pyrometer', 'elec', 'xarion', 'coaxial_pd', 'offaxis_pd']

def generate_psd(df, sensor_cols):
    """
    generate PSD features for given sample's sensor data
    """
    results = []
    df = df[df['layer_num'] <= 450]

    for layer in tqdm(df['layer_num'].unique()):
        df_layer = df[df['layer_num'] == layer].reset_index(drop=True)
        sample_num = df_layer.at[0, 'sample_num']
        layer_num = df_layer.at[0, 'layer_num']

        for sensor in sensor_cols:
            signal_data = df_layer[sensor]
            frequencies, psd = signal.welch(signal_data, fs=100000, nperseg=256)

            for freq, power in zip(frequencies, psd):
                results.append({
                    'sample_num': sample_num,
                    'layer_num': layer_num,
                    'regime': regime_dict[sample_num],
                    'sensor': sensor,
                    'freq': freq,
                    'psd': power
                })

    return pd.DataFrame(results)

def main():
    df = pd.DataFrame()

    for sample_dir in SAMPLE_DIRS:
        print(f'Generating PSD for sample {int(sample_dir) + 1}')
        df_sample = pd.read_parquet(os.path.join(DATA_DIR, sample_dir))
        df_psd = generate_psd(df_sample, sensors)
        df = pd.concat([df, df_psd])

    df_psd.to_parquet(OUTPUT_FILE)