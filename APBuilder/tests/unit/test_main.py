import datetime
import pytest
from apbuilder import main
from apbuilder import exceptions


def test_extract_data_not_found_error():
    datetime_ = datetime.datetime.fromisoformat("19900101T00:00:00+00:00")
    weather_model = "GFS"
    profile_format = "1D"
    sp1 = [0, 0]
    sp2 = [0, 0]
    dr = 0
    with pytest.raises(exceptions.DataNotFoundError) as excinfo:
        main.extract(datetime_, weather_model, profile_format, sp1, sp2, dr)
    assert "Unable to find data" in str(excinfo.value)


def test_extract_unknown_model_cycle_error():
    datetime_ = datetime.datetime.fromisoformat("19900101")
    weather_model = "GFS"
    profile_format = "1D"
    sp1 = [0, 0]
    sp2 = [0, 0]
    dr = 0
    with pytest.raises(exceptions.UnknownModelCycle) as excinfo:
        main.extract(
            datetime_, weather_model, profile_format, sp1, sp2, dr, cycle="invalid"
        )
    assert "Unknown model" in str(excinfo.value)


def test_get_model_cycle_gfs_0600():
    datetime_ = datetime.datetime.fromisoformat("20240801T06:00:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T06:00:00")


def test_get_model_cycle_gfs_060510():
    datetime_ = datetime.datetime.fromisoformat("20240801T06:05:10")
    weather_model = "GFS"
    grid_degree = 0.25
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T06:00:00")


def test_get_model_cycle_gfs_0259():
    datetime_ = datetime.datetime.fromisoformat("20240801T02:59:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T00:00:00")


def test_get_model_cycle_gfs_0300():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:00:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T00:00:00")


def test_get_model_cycle_gfs_0301():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:01:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T06:00:00")


def test_get_model_cycle_rap_0259():
    datetime_ = datetime.datetime.fromisoformat("20240801T02:59:00")
    weather_model = "RAP"
    grid_degree = 13
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T03:00:00")


def test_get_model_cycle_rap_0300():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:00:00")
    weather_model = "RAP"
    grid_degree = 13
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T03:00:00")


def test_get_model_cycle_rap_0301():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:01:00")
    weather_model = "RAP"
    grid_degree = 13
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T03:00:00")


def test_get_model_cycle_rap_0329():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:29:00")
    weather_model = "RAP"
    grid_degree = 13
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T03:00:00")


def test_get_model_cycle_rap_0330():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:30:00")
    weather_model = "RAP"
    grid_degree = 13
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T03:00:00")


def test_get_model_cycle_rap_0331():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:31:00")
    weather_model = "RAP"
    grid_degree = 13
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T04:00:00")


def test_get_model_cycle_hrrr_0259():
    datetime_ = datetime.datetime.fromisoformat("20240801T02:59:00")
    weather_model = "HRRR"
    grid_degree = 3
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T03:00:00")


def test_get_model_cycle_hrrr_0300():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:00:00")
    weather_model = "HRRR"
    grid_degree = 3
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T03:00:00")


def test_get_model_cycle_hrrr_0301():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:01:00")
    weather_model = "HRRR"
    grid_degree = 3
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T03:00:00")


def test_get_model_cycle_hrrr_0329():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:29:00")
    weather_model = "HRRR"
    grid_degree = 3
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T03:00:00")


def test_get_model_cycle_hrrr_0330():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:30:00")
    weather_model = "HRRR"
    grid_degree = 3
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T03:00:00")


def test_get_model_cycle_hrrr_0331():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:31:00")
    weather_model = "HRRR"
    grid_degree = 3
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    assert model_datetime == datetime.datetime.fromisoformat("20240801T04:00:00")


def test_get_model_cycle_unknown_weather_model():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:31:00")
    weather_model = "UNK"
    grid_degree = 3
    model_datetime = main.get_model_cycle(datetime_, weather_model, grid_degree)
    # The method should return the provided datetime if the weather model is unknown
    assert model_datetime == datetime_


def test_get_forecast_cycle_gfs_0500():
    datetime_ = datetime.datetime.fromisoformat("20240801T05:00:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_cycle, forecast_cycle = main.get_forecast_cycle(
        datetime_, weather_model, grid_degree
    )
    assert forecast_cycle == 0
    assert model_cycle == datetime.datetime.fromisoformat("20240801T06:00:00")


def test_get_forecast_cycle_gfs_1500():
    datetime_ = datetime.datetime.fromisoformat("20240801T15:00:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_cycle, forecast_cycle = main.get_forecast_cycle(
        datetime_, weather_model, grid_degree
    )
    assert forecast_cycle == 3
    assert model_cycle == datetime.datetime.fromisoformat("20240801T12:00:00")


def test_get_forecast_cycle_gfs_1730():
    datetime_ = datetime.datetime.fromisoformat("20240801T17:30:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_cycle, forecast_cycle = main.get_forecast_cycle(
        datetime_, weather_model, grid_degree
    )
    assert forecast_cycle == 0
    assert model_cycle == datetime.datetime.fromisoformat("20240801T18:00:00")


def test_get_forecast_cycle_gfs_1845():
    datetime_ = datetime.datetime.fromisoformat("20240801T18:45:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_cycle, forecast_cycle = main.get_forecast_cycle(
        datetime_, weather_model, grid_degree
    )
    assert forecast_cycle == 1
    assert model_cycle == datetime.datetime.fromisoformat("20240801T18:00:00")


def test_get_forecast_cycle_gfs_0245_05():
    datetime_ = datetime.datetime.fromisoformat("20240801T02:45:00")
    weather_model = "GFS"
    grid_degree = 0.5
    model_cycle, forecast_cycle = main.get_forecast_cycle(
        datetime_, weather_model, grid_degree
    )
    assert forecast_cycle == 3
    assert model_cycle == datetime.datetime.fromisoformat("20240801T00:00:00")


def test_get_forecast_cycle_rap_0345():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:45:00")
    weather_model = "RAP"
    grid_degree = 13
    model_cycle, forecast_cycle = main.get_forecast_cycle(
        datetime_, weather_model, grid_degree
    )
    assert forecast_cycle == 0
    assert model_cycle == datetime.datetime.fromisoformat("20240801T04:00:00")


def test_get_forecast_cycle_rap_2042():
    datetime_ = datetime.datetime.fromisoformat("20240801T20:42:00")
    weather_model = "RAP"
    grid_degree = 13
    model_cycle, forecast_cycle = main.get_forecast_cycle(
        datetime_, weather_model, grid_degree
    )
    assert forecast_cycle == 0
    assert model_cycle == datetime.datetime.fromisoformat("20240801T21:00:00")


def test_get_forecast_cycle_unknown_weather_model():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:31:00")
    weather_model = "UNK"
    grid_degree = 3
    model_cycle, forecast_cycle = main.get_forecast_cycle(
        datetime_, weather_model, grid_degree
    )
    # The method should return the provided datetime if the weather model is unknown
    assert model_cycle == datetime_
    assert forecast_cycle == 0


def test_get_forecast_only_cycle_gfs_0500():
    datetime_ = datetime.datetime.fromisoformat("20240801T05:00:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_cycle, forecast_cycle = main.get_forecast_only_cycle(
        datetime_, weather_model, grid_degree
    )
    assert model_cycle.hour == 0
    assert forecast_cycle == 5


def test_get_forecast_only_cycle_gfs_1500():
    datetime_ = datetime.datetime.fromisoformat("20240801T15:00:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_cycle, forecast_cycle = main.get_forecast_only_cycle(
        datetime_, weather_model, grid_degree
    )
    assert model_cycle.hour == 12
    assert forecast_cycle == 3


def test_get_forecast_only_cycle_gfs_1730():
    datetime_ = datetime.datetime.fromisoformat("20240801T17:30:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_cycle, forecast_cycle = main.get_forecast_only_cycle(
        datetime_, weather_model, grid_degree
    )
    assert model_cycle.hour == 12
    assert forecast_cycle == 5


def test_get_forecast_only_cycle_gfs_1845():
    datetime_ = datetime.datetime.fromisoformat("20240801T18:45:00")
    weather_model = "GFS"
    grid_degree = 0.25
    model_cycle, forecast_cycle = main.get_forecast_only_cycle(
        datetime_, weather_model, grid_degree
    )
    assert model_cycle.hour == 18
    assert forecast_cycle == 1


def test_get_forecast_only_cycle_gfs_0245_05():
    datetime_ = datetime.datetime.fromisoformat("20240801T02:45:00")
    weather_model = "GFS"
    grid_degree = 0.5
    model_cycle, forecast_cycle = main.get_forecast_only_cycle(
        datetime_, weather_model, grid_degree
    )
    assert model_cycle.hour == 0
    assert forecast_cycle == 3


def test_get_forecast_only_cycle_rap_0345():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:45:00")
    weather_model = "RAP"
    grid_degree = 13
    model_cycle, forecast_cycle = main.get_forecast_only_cycle(
        datetime_, weather_model, grid_degree
    )
    assert model_cycle.hour == 3
    assert forecast_cycle == 1


def test_get_forecast_only_cycle_rap_2042():
    datetime_ = datetime.datetime.fromisoformat("20240801T20:42:00")
    weather_model = "RAP"
    grid_degree = 13
    model_cycle, forecast_cycle = main.get_forecast_only_cycle(
        datetime_, weather_model, grid_degree
    )
    assert model_cycle.hour == 20
    assert forecast_cycle == 1


def test_get_forecast_only_cycle_rap_grid_error():
    datetime_ = datetime.datetime.fromisoformat("20240801T20:42:00")
    weather_model = "RAP"
    grid_degree = 1
    with pytest.raises(exceptions.UnsupportedGridDegreeError) as excinfo:
        _, _ = main.get_forecast_only_cycle(datetime_, weather_model, grid_degree)
    assert "is unsupported" in str(excinfo.value)


def test_get_forecast_only_cycle_unknown_weather_model():
    datetime_ = datetime.datetime.fromisoformat("20240801T03:31:00")
    weather_model = "UNK"
    grid_degree = 3
    model_cycle, forecast_cycle = main.get_forecast_only_cycle(
        datetime_, weather_model, grid_degree
    )
    # The method should return the provided datetime if the weather model is unknown
    assert model_cycle == datetime_
    assert forecast_cycle == 0


def test_no_download_flag_turned_on(capsys):
    datetime_ = datetime.datetime.fromisoformat("20240801T05:28:00+00:00")
    weather_model = "GFS"
    profile_format = "1D"
    sp1 = [0, 0]
    sp2 = [0, 0]
    dr = 0
    with pytest.raises(exceptions.NoDownloadMode) as excinfo:
        main.extract(
            datetime_, weather_model, profile_format, sp1, sp2, dr, no_download=True
        )
    assert "Exiting due to no_download flag turned on" in str(excinfo.value)


def test_print_models_info(capsys):
    args = ["info", "-mi"]
    main.run_apb_with_args(args)
    captured = capsys.readouterr()
    assert "Global" in captured.out
    assert "CONUS" in captured.out
    assert "GFS" in captured.out
    assert "RAP" in captured.out
    assert "HRRR" in captured.out


def test_print_gmt_info(capsys):
    args = ["info", "-gmt"]
    main.run_apb_with_args(args)
    captured = capsys.readouterr()
    assert "PyGMT information" in captured.out
    assert "System information" in captured.out
    assert "Dependency information" in captured.out
    assert "GMT library information" in captured.out


def test_selfcheck(capsys):
    args = ["info", "--selfcheck"]
    main.run_apb_with_args(args)
    captured = capsys.readouterr()
    assert "Start selfcheck for ecCodes" in captured.out
    assert "Start selfcheck for cfgrib" in captured.out
    assert "Start selfcheck for PyGMT" in captured.out
    assert "Found: ecCodes" in captured.out
    assert "PyGMT information" in captured.out
    assert "System information" in captured.out
    assert "Dependency information" in captured.out
    assert "GMT library information" in captured.out
