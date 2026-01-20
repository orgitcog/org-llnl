from datetime import datetime
import pytest
import numpy as np
from apbuilder import utils


def test_valid_convert_datetime_to_str_for_filename():
    np_datetime = np.datetime64("2024-07-08T06:00:00.123000000")
    str_datetime = utils.convert_datetime_to_str_for_filename(np_datetime)
    assert str_datetime == "2024_07_08T06_00_00"


def test_none_convert_datetime_to_str_for_filename():
    str_datetime = utils.convert_datetime_to_str_for_filename(None)
    assert str_datetime == "None"


def test_closest_time():
    time_list = ["00:00", "06:00", "12:00", "18:00"]
    assert utils.closest_time(time_list, "00:00") == "00:00"
    assert utils.closest_time(time_list, "00:45") == "00:00"
    assert utils.closest_time(time_list, "02:00") == "00:00"
    assert utils.closest_time(time_list, "02:59") == "00:00"
    assert utils.closest_time(time_list, "03:00") == "00:00"
    assert utils.closest_time(time_list, "03:01") == "06:00"
    assert utils.closest_time(time_list, "05:00") == "06:00"
    assert utils.closest_time(time_list, "06:15") == "06:00"
    assert utils.closest_time(time_list, "12:00") == "12:00"
    assert utils.closest_time(time_list, "16:35") == "18:00"
    assert utils.closest_time(time_list, "21:14") == "18:00"
    assert utils.closest_time(time_list, "23:55") == "18:00"


def test_closest_time_downward():
    time_list = ["00:00", "06:00", "12:00", "18:00"]
    assert utils.closest_time_downward(time_list, "00:00") == "00:00"
    assert utils.closest_time_downward(time_list, "00:45") == "00:00"
    assert utils.closest_time_downward(time_list, "02:00") == "00:00"
    assert utils.closest_time_downward(time_list, "02:59") == "00:00"
    assert utils.closest_time_downward(time_list, "03:00") == "00:00"
    assert utils.closest_time_downward(time_list, "03:01") == "00:00"
    assert utils.closest_time_downward(time_list, "05:00") == "00:00"
    assert utils.closest_time_downward(time_list, "06:15") == "06:00"
    assert utils.closest_time_downward(time_list, "12:00") == "12:00"
    assert utils.closest_time_downward(time_list, "16:35") == "12:00"
    assert utils.closest_time_downward(time_list, "21:14") == "18:00"
    assert utils.closest_time_downward(time_list, "23:55") == "18:00"
