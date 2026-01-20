import math
import numpy as np
from datetime import datetime
from prettytable import PrettyTable
from enum import StrEnum


WEATHER_MODELS_TABLE_FIELDS = [
    "Model",
    "Grid (Deg)",
    "Time Period",
    "Model Cycle",
    "Forecast Cycle",
    "Geographic Extent",
]


class Cycle(StrEnum):
    MODEL = "model"
    FORECAST = "forecast"
    FORECAST_ONLY = "forecast-only"


@staticmethod
def convert_datetime_to_str_for_filename(data: np.datetime64) -> str:
    sep = "_"
    if data is None:
        return str(None)

    return (
        np.datetime_as_string(data, unit="s")
        .replace(":", sep)
        .replace(".", sep)
        .replace("-", sep)
    )


@staticmethod
def closest_time(time_list: list[str], target_time: str) -> str:
    """Returns the closest time
    The function first checks if the target time is less than or equal to the first time in the list. If so, it returns the first time.
    It then checks if the target time is greater than or equal to the last time in the list. If so, it returns the last time.
    For other cases, it calculates the difference between the target time and each time in the list, keeping track of the closest time.
    Finally, it returns the closest time based on the specified logic.
    The time must be in the format \\'\\%H\\%M\\'.
    """
    # Convert target_time to a datetime object
    target_time = datetime.strptime(target_time, "%H:%M")

    closest_time = None
    smallest_difference = float("inf")  # Initialize with a large number

    for time_str in time_list:
        # Convert each time in the list to a datetime object
        current_time = datetime.strptime(time_str, "%H:%M")

        # Calculate the difference in minutes
        difference = (current_time - target_time).total_seconds() / 60

        # If the target time is less than or equal to the first time
        if target_time <= datetime.strptime(time_list[0], "%H:%M"):
            return time_list[0]

        # If the target time is greater than or equal to the last time
        if target_time >= datetime.strptime(time_list[-1], "%H:%M"):
            return time_list[-1]

        # Check if the current time is the closest
        if abs(difference) < smallest_difference:
            smallest_difference = abs(difference)
            closest_time = current_time.strftime("%H:%M")

    return closest_time


@staticmethod
def closest_time_downward(time_list: list[str], target_time: str) -> str:
    """Returns the closest time that is less than or equal to the target time.
    The time must be in the format '%H:%M'.
    """
    # Convert target_time to a datetime object
    target_time = datetime.strptime(target_time, "%H:%M")

    closest_time = None

    for time_str in time_list:
        # Convert each time in the list to a datetime object
        current_time = datetime.strptime(time_str, "%H:%M")

        # Check if the current time is less than or equal to the target time
        if current_time <= target_time:
            # If it's the first valid time found or is greater than the current closest
            if closest_time is None or current_time > datetime.strptime(
                closest_time, "%H:%M"
            ):
                closest_time = current_time.strftime("%H:%M")

    # If no valid time found, return None or an appropriate value
    return closest_time if closest_time is not None else "00:00"


@staticmethod
def available_weather_models_info_prettytable(
    sort_by: str = "Model", reverse_sort: bool = False
) -> str:
    table = PrettyTable()
    table.field_names = WEATHER_MODELS_TABLE_FIELDS
    table.sortby = sort_by
    table.reversesort = reverse_sort
    table.add_row(
        [
            "GFS",
            "0.25",
            "2021/02/26 - Present",
            "4/day: 00, 06, 12, 18 UTC",
            "1h (0h - 384h)",
            "Global",
        ]
    )
    table.add_row(
        [
            "GFS",
            "0.50",
            "2021/02/26 - Present",
            "4/day: 00, 06, 12, 18 UTC",
            "3h (0h - 384h)",
            "Global",
        ]
    )
    table.add_row(
        [
            "GFS",
            "1.00",
            "2021/02/26 - Present",
            "4/day: 00, 06, 12, 18 UTC",
            "3h (0h - 384h)",
            "Global",
        ]
    )
    table.add_row(
        [
            "GFS",
            "0.50",
            "2007/01/01 - Present",
            "4/day: 00, 06, 12, 18 UTC",
            "3h (0h - 6h)",
            "Global",
        ]
    )
    table.add_row(
        [
            "GFS",
            "1.00",
            "2004/03/02 - Present",
            "4/day: 00, 06, 12, 18 UTC",
            "3h (0h - 6h)",
            "Global",
        ]
    )
    table.add_row(
        [
            "RAP",
            "13km",
            "2012/05/01 - Present",
            "24/Day: 00, 01, 02, ... 23 UTC",
            "1h (0h - 21h)",
            "CONUS",
        ]
    )
    table.add_row(
        [
            "HRRR",
            "3km",
            "2014/07/30 - Present",
            "24/Day: 00, 01, 02, ... 23 UTC",
            "1h (0h - 15h)",
            "CONUS",
        ]
    )
    table.add_row(
        [
            "IFS",
            "0.25",
            "2023/01/18 - Present",
            "4/day: 00, 06, 12, 18 UTC",
            "3h (0 - 144h)",
            "Global",
        ]
    )
    return table


@staticmethod
def get_azimuth(sp1, sp2):
    phi1 = sp1[1]
    lambda1 = sp1[0]
    phi2 = sp2[1]
    lambda2 = sp2[0]
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    R = 6371 * 1000
    a = (
        math.sin(dphi / 2 * math.pi / 180) ** 2
        + math.cos(phi1 * math.pi / 180)
        * math.cos(phi2 * math.pi / 180)
        * math.sin(dlambda / 2 * math.pi / 180) ** 2
    )
    d = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    theta = (
        180
        / math.pi
        * math.atan2(
            math.sin(dlambda * math.pi / 180) * math.cos(phi2 * math.pi / 180),
            math.cos(phi1 * math.pi / 180) * math.sin(phi2 * math.pi / 180)
            - math.sin(phi1 * math.pi / 180)
            * math.cos(phi2 * math.pi / 180)
            * math.cos(dlambda * math.pi / 180),
        )
    )
    return theta
