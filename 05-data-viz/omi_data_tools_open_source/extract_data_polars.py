"""
This script extracts and transforms data from raw Aconity binary files into a clean set of layer-wise parquet files.
"""

import os
import numpy as np
import polars as pl
from glob import glob
from tqdm import tqdm
import concurrent.futures
from typing import Optional
from itertools import repeat
from scipy.spatial import distance

# parameter variables
SAMPLE_ID = '0017'
DATA_DIR = r'\\amdata.llnl.gov\data\Prints\Fraunhofer\Builds\20221120_DiagSupCalta\BuildSupPrint.dcn'
OUTPUT_DIR = r'\\amdata.llnl.gov\data\Gorgannejad\Sensor Fusion\Fraunhofer-2023\Sensor-parquet'
OUTPUT_FORMAT = 'parquet'  # csv or parquet: defaults to csv

def extract_laser_speed(file_path: str):
    """
    extract laser speed from PLog metadata file
    """
    col_names = ['folder', 'part_id', 'layer_id', 'laser_power', 'laser_speed', 'mode', 'NA', 'date', 'time']
    df = pl.read_csv(file_path, separator='\t', has_header=False, new_columns=col_names)
    laser_speed = int(df.item(0, 'laser_speed').split(' ')[1])
    return laser_speed

def read_binary_data(file_path: str):
    """
    read raw data binary file into a dataframe
    """
    col_names = ['laser_state', 'x', 'y', 'laser_pwr_commanded', 'pyrometer', 'nc1', 'elec', 
                 'xarion', 'offaxis_pd', 'coaxial_pd', 'nc4','laser_pwr_measured']

    data_type = '>f'
    schema = np.dtype([(field, data_type) for field in col_names])
    data = np.fromfile(file_path, dtype=schema)

    # values encoded as big endian, swap to little endian
    df = pl.DataFrame(data.byteswap().newbyteorder()) 
    # parse layer number from filename (e.g. L0001_000 -> 1)
    df = df.with_columns(pl.lit(int(os.path.basename(file_path).split('_')[0][1:])).alias('layer_num'))
    df = df.with_row_index()
    return df

def sensor_delay_shift(sample_num: int, sensor: str):
    """
    shift sensor data based on distance from sample location
    """
    sensor_dict = {'xarion': [-143.18, -0.6, 55.37],
                   'elec': [-131.07, -70.65, 53.16]}

    sample_locations = {
        0: (28.08, -23.03, 0),
        1: (28.31, -10.31, 0),
        2: (28.39, 2.61, 0),
        3: (28.42, 15.52, 0),
        4: (28.47, 28.46, 0),
        5: (15.49, -23.1, 0),
        6: (15.47, -10.24, 0),
        7: (15.48, 2.67, 0),
        8: (15.53, 15.57, 0),
        9: (15.59, 28.48, 0),
        10: (2.62, -23.08, 0),
        11: (2.6, -10.22, 0),
        12: (2.62, 2.68, 0),
        13: (2.65, 15.6, 0),
        14: (2.69, 28.51, 0),
        15: (-10.23, -23.08, 0),
        16: (-10.27, -10.21, 0),
        17: (-10.24, 2.69, 0),
        18: (-10.22, 15.62, 0),
        19: (-10.18, 28.53, 0),
        20: (-23.14, -22.93, 0),
        21: (-23.36, -10.13, 0),
        22: (-23.06, 2.71, 0),
        23: (-23.02, 15.6, 0),
        24: (-23.02, 28.48, 0)
    }
    # calculate 3D euclidean distance
    sensor_distance = distance.euclidean(sensor_dict[sensor.lower()], sample_locations[sample_num - 1])

    speed_argon = 323000                                           # mm per second
    time_microseconds = (sensor_distance / speed_argon) * 10**6    # time taken to travel the distance
    num_position = round(time_microseconds / 10)                   # shift for 100 kHZ frequency, timestep is 10 microsecond
    return -num_position

def filter_nonprint(df: pl.DataFrame, power_threshold: float=2.0, filter_modes: list=['retract']):  # To remove turnaround: list=['retract', 'turnaround']
    """
    filter retraction path and turnarounds
    """
    # last data point corresponding to true print
    last_high_pwr_idx = df.filter(pl.col('laser_pwr_measured') > power_threshold).select(pl.col('index').max())

    df = df.with_columns(
        pl.when(pl.col('laser_pwr_measured') > power_threshold)
        .then(pl.lit('print'))
        .otherwise(pl.lit('turnaround'))
        .alias('mode'))
    
    df = df.with_columns(
        pl.when(pl.col('mode').cum_count() > last_high_pwr_idx)
        .then(pl.lit('retract'))
        .otherwise(pl.col('mode'))
        .alias('mode'))

    df = df.filter(~pl.col('mode').is_in(filter_modes))
    return df.drop('mode')

def transform_data(df: pl.DataFrame, sample_num: int, laser_speed: Optional[int]=None):
    """
    clean and transform raw data
    """
    shift_dict = {900: 8, 700: 8, 500: 6, 300: 4, 100: 0}
    shift = shift_dict[laser_speed] if laser_speed else 0

    elec_shift = sensor_delay_shift(sample_num, 'elec')
    xarion_shift = sensor_delay_shift(sample_num, 'xarion')

    df = df.with_columns(
        pl.lit(sample_num).alias('sample_num'),
        pl.col('index').mul(1e-5).round(5).alias('time'),
        pl.col('x').mul((3.165 / 1000)).shift(shift, fill_value=0),
        pl.col('y').mul((3.425 / 1000)).shift(shift, fill_value=0),
        pl.col('layer_num').sub(1).mul(0.05).alias('z'),
        pl.col('elec').shift(elec_shift, fill_value=0),
        pl.col('xarion').shift(xarion_shift, fill_value=0))

    df = df.filter(pl.col('laser_state') != pl.col('laser_state').min())
    df = filter_nonprint(df, power_threshold=2)

    col_order = ['sample_num', 'layer_num', 'time', 'x', 'y', 'z', 'pyrometer', 'elec', 'xarion', 'offaxis_pd',
                 'coaxial_pd', 'laser_state', 'laser_pwr_commanded', 'laser_pwr_measured']
    return df.select(col_order)

def process_layer_data(file_path: str, output_dir: str, output_format: Optional[str], laser_speed: Optional[int]):
    """
    read, transform, and write binary layer data from Aconity
    """
    try:
        sample_num = int(os.path.basename(os.path.dirname(file_path))) + 1

        df = read_binary_data(file_path)
        df_transform = transform_data(df, sample_num, laser_speed)

        file_name = os.path.basename(file_path)
        if output_format == 'parquet':
            file_name = f'{file_name }.parquet'
            df_transform.write_parquet(os.path.join(output_dir, file_name))
        else:
            file_name = f'{file_name }.csv'
            df_transform.write_csv(os.path.join(output_dir, file_name))

    except Exception as e:
        print(f'Error processing file {file_path}: {str(e)}')

def main():
    output_dir = os.path.join(OUTPUT_DIR, SAMPLE_ID)
    # generate output directory if it does not already exist
    os.makedirs(output_dir, exist_ok=True)

    # extract laser speed
    data_dir = os.path.join(DATA_DIR, SAMPLE_ID)
    metadata_file = glob(os.path.join(data_dir, '*Log'))[0]
    laser_speed = extract_laser_speed(metadata_file)

    print(f'Processing sample ID: {SAMPLE_ID} \n')
    layer_files = sorted(glob(os.path.join(data_dir, 'L*')))

    # multithreading for accelerated layer extraction
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_layer_data,
                            layer_files,
                            repeat(output_dir), 
                            repeat(OUTPUT_FORMAT), 
                            repeat(laser_speed)), 
                            total=len(layer_files)))

if __name__ == '__main__':
    main()