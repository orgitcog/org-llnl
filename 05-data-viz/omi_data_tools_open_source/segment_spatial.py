# %%
"""
This script performs layer-wise spatial segmentation of a part given a set of parameters.
"""

import os
import pandas as pd
from tqdm import tqdm
from glob import glob
import concurrent.futures
from itertools import repeat

from layer_transform import x_y_rotation
from segment import assign_tracks, assign_segments_spatial

# parameter variables
SAMPLE_IDS = ['0005']
DATA_DIR = '/Users/ciardi1/Development/sensor_fusion/data/processed_signal'
OUTPUT_DIR = '/Users/ciardi1/Development/sensor_fusion/data/segment'

TIME_THRESHOLD = 0.0001    # threshold of time to segment linear tracks
SPLIT_FROM = 'middle'      # subtrack starting point for linear tracks
VOXEL_LENGTH = 0.5
TRACK_WIDTH = 3

df = pd.read_parquet('/Users/ciardi1/Development/sensor_fusion/data/processed_signal/0005/L0001_000.parquet')
df_transform = x_y_rotation(df)
df_tracks = assign_tracks(df_transform, TIME_THRESHOLD)

def segment_print(file_path: str, time_threshold: float, voxel_length: int, track_width: int, output_dir: str):
    try:
        df = pd.read_parquet(file_path)

        # layer rotation
        df_transform = x_y_rotation(df)

        # segment full linear tracks and sub-segments
        df_tracks = assign_tracks(df_transform, time_threshold)
        df_segment = assign_segments_spatial(df_tracks, 'center', voxel_length, track_width)

        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)

        # append to existing segmentation if file exists
        if os.path.exists(output_path):
            voxel_length = df_segment.at[0, 'voxel_length']
            track_width = df_segment.at[0, 'track_width']

            df_temp = pd.read_parquet(output_path)
            df_temp = df_temp[(df_temp['voxel_length'] != voxel_length) & (df_temp['track_width'] != track_width)]
            df_segment = pd.concat([df_temp, df_segment], axis=0)

        df_segment.to_parquet(output_path)

    except Exception as e:
        print(f'Error processing file {file_path}: {str(e)}')

def main():
    for sample_id in SAMPLE_IDS:
        sample_num = int(sample_id) + 1
        data_dir = os.path.join(DATA_DIR, sample_id)
        output_dir = os.path.join(OUTPUT_DIR, sample_id)

        # generate output directory if it does not already exist
        os.makedirs(output_dir, exist_ok=True)

        print(f'Processing sample number {sample_num} with voxel_length = {VOXEL_LENGTH} and track_width = {TRACK_WIDTH} \n')
        file_paths = sorted(glob(os.path.join(data_dir, '*[0-9].parquet')))[0:1]

        # generate output directory if it does not already exist
        os.makedirs(output_dir, exist_ok=True)

        # multithreading for accelerated layer extraction
        with concurrent.futures.ProcessPoolExecutor() as executor:
            data = list(tqdm(executor.map(segment_print, 
                                        file_paths,
                                        repeat(TIME_THRESHOLD),
                                        repeat(VOXEL_LENGTH),
                                        repeat(TRACK_WIDTH),
                                        repeat(output_dir)),
                                        total=len(file_paths)))
                            
if __name__ == '__main__':
    main()