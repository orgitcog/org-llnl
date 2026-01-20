import datetime
import pytest
from apbuilder import main


def test_parse_date_only_datetime():
    args = [
        "build1d",
        "20240708",
        "37.1149",
        "243.9309",
        "37.1149",
        "243.9309",
        "-out",
        "out_dir",
    ]
    parser = main.parse_args(args)
    assert parser.datetime.year == 2024
    assert parser.datetime.month == 7
    assert parser.datetime.day == 8
    assert parser.datetime.hour == 0
    assert parser.datetime.minute == 0
    assert parser.datetime.second == 0
    assert parser.lat == 37.1149
    assert parser.lon == 243.9309
    assert parser.lat2 == 37.1149
    assert parser.lon2 == 243.9309
    assert parser.output_directory == "out_dir"


def test_parse_iso8601_datetime():
    args = [
        "build1d",
        "20240708T06:00:00",
        "37.1149",
        "243.9309",
        "37.1149",
        "243.9309",
    ]
    parser = main.parse_args(args)
    assert parser.datetime.year == 2024
    assert parser.datetime.month == 7
    assert parser.datetime.day == 8
    assert parser.datetime.hour == 6
    assert parser.datetime.minute == 0
    assert parser.datetime.second == 0
    assert parser.lat == 37.1149
    assert parser.lon == 243.9309
    assert parser.lat2 == 37.1149
    assert parser.lon2 == 243.9309


def test_parse_iso8601_with_tz_datetime():
    args = [
        "build1d",
        "20240708T06:00:00-07:00",
        "37.1149",
        "243.9309",
        "37.1149",
        "243.9309",
    ]
    parser = main.parse_args(args)
    assert parser.datetime.year == 2024
    assert parser.datetime.month == 7
    assert parser.datetime.day == 8
    assert parser.datetime.hour == 6
    assert parser.datetime.minute == 0
    assert parser.datetime.second == 0
    assert parser.lat == 37.1149
    assert parser.lon == 243.9309
    assert parser.lat2 == 37.1149
    assert parser.lon2 == 243.9309


def test_parse_iso8601_with_tz_ahead_of_utc_datetime():
    args = [
        "build1d",
        "20240708T06:00:00+05:00",
        "37.1149",
        "243.9309",
        "37.1149",
        "243.9309",
    ]
    parser = main.parse_args(args)
    assert parser.datetime.year == 2024
    assert parser.datetime.month == 7
    assert parser.datetime.day == 8
    assert parser.datetime.hour == 6
    assert parser.datetime.minute == 0
    assert parser.datetime.second == 0
    assert parser.lat == 37.1149
    assert parser.lon == 243.9309
    assert parser.lat2 == 37.1149
    assert parser.lon2 == 243.9309


def test_parse_epoch_datetime():
    args = ["build1d", "1720443600", "37.1149", "243.9309", "37.1149", "243.9309", "-e"]
    parser = main.parse_args(args)
    assert parser.datetime.year == 2024
    assert parser.datetime.month == 7
    assert parser.datetime.day == 8
    assert parser.datetime.hour == 13
    assert parser.datetime.minute == 0
    assert parser.datetime.second == 0
    assert parser.lat == 37.1149
    assert parser.lon == 243.9309
    assert parser.lat2 == 37.1149
    assert parser.lon2 == 243.9309


def test_parse_datetime_as_iso8601():
    args = ["build1d", "20201029", "37.1149", "243.9309", "37.1149", "243.9309"]
    parser = main.parse_args(args)
    assert parser.datetime.year == 2020
    assert parser.datetime.month == 10
    assert parser.datetime.day == 29
    assert parser.datetime.hour == 0
    assert parser.datetime.minute == 0
    assert parser.datetime.second == 0


def test_parse_datetime_as_epoch():
    args = ["build1d", "20201029", "37.1149", "243.9309", "37.1149", "243.9309", "-e"]
    parser = main.parse_args(args)
    assert parser.datetime.year == 1970
    assert parser.datetime.month == 8
    assert parser.datetime.day == 22
    assert parser.datetime.hour == 19
    assert parser.datetime.minute == 23
    assert parser.datetime.second == 49


def test_parse_invalid_datetime():
    args = ["build1d", "invalid", "37.1149", "243.9309", "37.1149", "243.9309"]
    with pytest.raises(SystemExit, match="Invalid date value"):
        main.parse_args(args)


def test_parse_future_datetime():
    future_date = datetime.datetime.now() + datetime.timedelta(days=1)
    args = ["build1d", str(future_date), "37.1149", "243.9309", "37.1149", "243.9309"]
    with pytest.raises(SystemExit, match="date cannot be in the future"):
        main.parse_args(args)


def test_parse_epoch_as_datetime():
    args = ["build1d", "1603756800", "37.1149", "243.9309", "37.1149", "243.9309", "-e"]
    parser = main.parse_args(args)
    assert parser.datetime.year == 2020
    assert parser.datetime.month == 10
    assert parser.datetime.day == 27
    assert parser.datetime.hour == 00
    assert parser.datetime.minute == 00
    assert parser.datetime.second == 00


def test_positive_float_with_dot_ending():
    args = ["build1d", "20240708", "37.", "120.", "37.", "243."]
    parser = main.parse_args(args)
    assert parser.datetime.year == 2024
    assert parser.datetime.month == 7
    assert parser.datetime.day == 8
    assert parser.datetime.hour == 0
    assert parser.datetime.minute == 0
    assert parser.datetime.second == 0
    assert parser.lat == 37.0
    assert parser.lon == 120.0
    assert parser.lat2 == 37.0
    assert parser.lon2 == 243.0


def test_negative_float_with_dot_ending():
    args = ["build1d", "20240708", "-37.", "-120.0", "-37.0", "-243.0"]
    # parser = main.parse_args(args)
    with pytest.raises(SystemExit):
        main.parse_args(args)


def test_parse_invalid_weather_model():
    args = [
        "run",
        "invalid",
        "1D",
        "37.1149",
        "243.9309",
        "37.1149",
        "243.9309",
        "--out",
        "out_dir",
        "--weather_model",
        "invalid",
    ]
    with pytest.raises(SystemExit):
        main.parse_args(args)


def test_parse_1d_plot_limits():
    args = [
        "build1d",
        "20240708",
        "37.",
        "120.",
        "37.",
        "243.",
        "-clim",
        "-10",
        "350",
        "-wlim",
        "-350",
        "180.3",
        "-dlim",
        "50.1",
        "-200.0",
    ]
    parser = main.parse_args(args)
    assert parser.clim[0] == -10.0
    assert parser.clim[1] == 350.0
    assert parser.dlim[0] == 50.1
    assert parser.dlim[1] == -200.0
    assert parser.wlim[0] == -350.0
    assert parser.wlim[1] == 180.3


def test_parse_2d_plot_limits():
    args = [
        "build2d",
        "--weather-model",
        "GFS",
        "20160815",
        "48.3200",
        "16.8700",
        "47.9183",
        "19.8908",
        "0.5",
        "--out",
        "out_dir",
        "-clim",
        "0",
        "350",
        "-wlim",
        "5.5",
        "180.3",
        "-dlim",
        "50.1",
        "200.0",
    ]
    parser = main.parse_args(args)
    assert parser.clim[0] == 0.0
    assert parser.clim[1] == 350.0
    assert parser.dlim[0] == 50.1
    assert parser.dlim[1] == 200.0
    assert parser.wlim[0] == 5.5
    assert parser.wlim[1] == 180.3


def test_parse_2d_clim_with_error():
    args = [
        "build2d",
        "--weather-model",
        "GFS",
        "20160815",
        "48.3200",
        "16.8700",
        "47.9183",
        "19.8908",
        "0.5",
        "-clim",
        "0",
        "error",
    ]
    with pytest.raises(SystemExit):
        main.parse_args(args)


def test_parse_2d_dlim_with_error():
    args = [
        "build2d",
        "--weather-model",
        "GFS",
        "20160815",
        "48.3200",
        "16.8700",
        "47.9183",
        "19.8908",
        "0.5",
        "-dlim",
        "0",
        "error",
    ]
    with pytest.raises(SystemExit):
        main.parse_args(args)


def test_parse_2d_wlim_with_error():
    args = [
        "build2d",
        "--weather-model",
        "GFS",
        "20160815",
        "48.3200",
        "16.8700",
        "47.9183",
        "19.8908",
        "0.5",
        "-wlim",
        "error",
        "360",
    ]
    with pytest.raises(SystemExit):
        main.parse_args(args)


def test_parse_output_directory():
    args = [
        "build1d",
        "20201029",
        "37.1149",
        "243.9309",
        "37.1149",
        "243.9309",
        "-out",
        "/tmp/out",
    ]
    parser = main.parse_args(args)
    assert parser.output_directory == "/tmp/out"


def test_parse_data_directory():
    args = [
        "build1d",
        "20201029",
        "37.1149",
        "243.9309",
        "37.1149",
        "243.9309",
        "-data",
        "/tmp/data",
    ]
    parser = main.parse_args(args)
    assert parser.data_directory == "/tmp/data"
