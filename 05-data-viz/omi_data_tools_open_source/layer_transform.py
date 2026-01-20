import warnings
import numpy as np
import pandas as pd
from scipy.stats import mode

def _find_most_common_slope(df: pd.DataFrame, subset_size: int=50):
    """
    given a set of (x, y) coordinates determine the most frequent slope of subsets
    """
    X = df['x'].values
    Y = df['y'].values
    
    # create overlapping subsets of x and y
    X_subsets = np.lib.stride_tricks.sliding_window_view(X, subset_size)[::subset_size-1]
    Y_subsets = np.lib.stride_tricks.sliding_window_view(Y, subset_size)[::subset_size-1]
    
    # calculate slopes using linear regression
    slopes = []
    for x, y in zip(X_subsets, Y_subsets):
        with warnings.catch_warnings():
            # flag vertical lines
            warnings.simplefilter('error', np.RankWarning)
            try:
                slope, _ = np.polyfit(x, y, 1)
                slopes.append(slope.round(3)) # round to 3 for frequency counting
            except np.RankWarning:
                slopes.append(float('inf'))

    return mode(slopes).mode

def _rotate_points(x: float, y: float, angle: float=0):
    """
    rotate (x, y) coordinates given an angle
    """
    rad = np.deg2rad(angle)
    cos_angle = np.cos(rad)
    sin_angle = np.sin(rad)
    x_rotated = x * cos_angle - y * sin_angle
    y_rotated = x * sin_angle + y * cos_angle
    return x_rotated, y_rotated

def x_y_rotation(df: pd.DataFrame):
    """ 
    given an angled set of points, determine the angle, and rotate the points to make horizontal
    """
    common_slope = _find_most_common_slope(df)
    if common_slope == float('inf'):
        angle_to_horizontal = -90
    else:
        angle_to_horizontal = -np.rad2deg(np.arctan(common_slope))

    df['x_rotated'], df['y_rotated'] = zip(*df.apply(lambda row: 
        _rotate_points(row['x'], row['y'], angle_to_horizontal), axis=1))
    return df