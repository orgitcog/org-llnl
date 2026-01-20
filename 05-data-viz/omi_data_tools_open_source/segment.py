import numpy as np
import pandas as pd

def assign_tracks(df: pd.DataFrame, threshold: float):
    """
    segment a set of coordinates into linear tracks
    """
    diff = df['time'].diff(1)
    idx = diff[diff > threshold].index
    df.loc[idx, 'track_id'] = range(1, len(idx) + 1)
    df['track_id'] = df['track_id'].ffill()
    df['track_id'] = df['track_id'].fillna(0)
    df['track_id'] = df['track_id'].astype('int')
    return df

def assign_segments_temporal(df: pd.DataFrame, split_from: str='middle', window_length: int=50, stride: int=50):
    """
    subsegment a set of linear tracks into segments given a temporal length
    """        
    df_segment = pd.DataFrame()

    for track_id in df['track_id'].unique():
        df_track = df[df['track_id'] == track_id].reset_index(drop=True)
        num_samples = len(df_track)

        if window_length < num_samples:
            # set a start index to align consecutive segments with start, middle, or end of full linear track
            start_dict = {'middle': (num_samples % window_length) // 2, 'start': 0, 'end': num_samples % window_length}
            start_idx = start_dict[split_from]
            end_idx = num_samples - window_length + 1
            start_ids = range(start_idx, end_idx, stride)

            for idx, start_id in enumerate(start_ids):
                timestamps = df_track.iloc[start_id:start_id+window_length]['time'].tolist()

                sample_num = df_track['sample_num'].unique()[0]
                layer_num = df_track['layer_num'].unique()[0]
                segment_id = '_'.join(str(x) for x in [sample_num, layer_num, track_id, idx, window_length, stride])

                subtrack_info = {
                    'segment_id': [segment_id],
                    'sample_num': [sample_num],
                    'layer_num': [layer_num],
                    'track_id': [track_id],
                    'subtrack_id': [idx],
                    'window_length': [window_length],
                    'stride': [stride],
                    'start_time': [min(timestamps)],
                    'end_time': [max(timestamps)],
                    'time': [timestamps]
                }

                df_segment = pd.concat([df_segment, pd.DataFrame(subtrack_info)]).reset_index(drop=True)

    return df_segment

def assign_segments_spatial(df: pd.DataFrame, split_from: str='center', voxel_length: float=0.5, track_width: int=3):
    """ 
    subsegment a set of linear tracks into segments given a spatial length and track width
    """
    def _group_valid_tracks(df: pd.DataFrame, voxel_length: float, track_width: int):
        # determine length of each linear track
        df_group = df.groupby('track_id', as_index=False)['x_rotated'].agg(['min', 'max'])
        df_group['length'] = df_group['max'] - df_group['min']
        valid_tracks = df_group[df_group['length'] > voxel_length]['track_id'].unique()

        # group linear tracks into segments of n width
        num_tracks = len(valid_tracks)
        start_track = valid_tracks[(num_tracks % track_width) // 2]
        num_voxels = num_tracks // track_width
        end_track = valid_tracks[start_track + (num_voxels * track_width) - 1]

        df_voxel = pd.DataFrame({'track_id': valid_tracks})
        df_voxel['track_group'] = np.floor_divide(df_voxel.index, track_width)
        df_voxel['track_group'] = df_voxel['track_group'].shift(start_track, fill_value=-1)
        df_voxel.loc[end_track + 1:, 'track_group'] = -1

        df = pd.merge(df, df_voxel, how='left', on='track_id')
        df = df[df['track_group'] != -1]
        return df.reset_index(drop=True)
    
    df_segment = pd.DataFrame()
    df = _group_valid_tracks(df, voxel_length, track_width)

    for track_group, group in df.groupby('track_group'):
        # find max permissible segments given by minimum size linear track
        df_track = group.groupby('track_id', as_index=False)['x_rotated'].agg(['min', 'max'])
        x_min = df_track['min'].max()
        x_max = df_track['max'].min()
        x_length = x_max - x_min

        # set a start index to align consecutive segments with start, middle, or end of full linear track
        start_dict = {'center': (x_length % voxel_length) / 2, 'left': 0, 'right': x_length % voxel_length}
        start_x = x_min + start_dict[split_from]
        end_x = x_max - voxel_length
        start_vals = np.arange(start_x, end_x, voxel_length)

        for idx, x in enumerate(start_vals):
            timestamps = group.loc[(group['x_rotated'] >= x) & (group['x_rotated'] < (x + voxel_length))]['time'].tolist()

            sample_num = group['sample_num'].unique()[0]
            layer_num = group['layer_num'].unique()[0]
            center_track_id = group['track_id'].unique()[track_width // 2]
            segment_id = '_'.join(str(x) for x in [sample_num, layer_num, center_track_id, idx, voxel_length, track_width])

            subtrack_info = {
                'segment_id': [segment_id],
                'sample_num': [sample_num],
                'layer_num': [layer_num],
                'center_track_id': [center_track_id],
                'subtrack_id': [idx],
                'voxel_length': [voxel_length],
                'track_width': [track_width],
                'start_x': [group['x_rotated'].min()],
                'end_x': [group['x_rotated'].max()],
                'time': [timestamps]
            }

            df_segment = pd.concat([df_segment, pd.DataFrame(subtrack_info)]).reset_index(drop=True)
    return df_segment