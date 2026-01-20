import argparse
import logging
import logging.config
import yaml
import datetime
import xarray
import os
import pygmt
import requests
import sys
import concurrent.futures
from herbie import Herbie
from importlib.resources import files
from pathlib import Path

import xarray as xr

import apbuilder.exceptions
import apbuilder.utils
import apbuilder.weather_models
import apbuilder.profile_1d
import apbuilder.profile_2d

from apbuilder._version import __version__
from apbuilder.exceptions import HorizontalResolutionError


logger = logging.getLogger(__name__)

# https://herbie.readthedocs.io/en/stable/user_guide/configure.html#priority
# This configure option allows you to specify a different order to look for data or only look in certain locations.
# But beware; setting a default priority might prevent you from checking all available sources.
PRIORITY = [
    "local",
    "ncei_analysis",
    "ncei_forecast",
    "ncei_historical_analysis",
    "nomads",
    "ecmwf",
    "aws",
    "aws-old",
    "ftpprd",
    "azure",
    "azure-scda",
    "azure-waef",
]


def configure_logger():
    """
    Reads a YAML file to configure the logging module.
    """

    log_config_file = os.path.join("config", "logging.yaml")
    try:
        # Read the log file from the package location (mostly when installed via pip)
        if __package__:
            data_text = files(__package__).joinpath(log_config_file).read_text()
            config = yaml.safe_load(data_text)
        # Otherwise, read from local path (mostly when developing)
        else:
            log_path = os.path.join(os.path.dirname(__file__), log_config_file)
            with open(log_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f.read())
    except FileNotFoundError as e:
        print(f"Error configuring logger: {e}")
        exit(1)
    logging.config.dictConfig(config)


def convert_datetime(s: str, is_epoch: bool = False) -> datetime.datetime:
    """
    Convert the date string to ISO8601 format or epoch format
        s: str - the date/time string to convert
        is_epoch: bool - True to specify string is in epoch format

    Returns:
        datetime: string converted to datetime object
    """
    date = None
    try:
        if is_epoch:
            s_float = float(s)
            date = datetime.datetime.fromtimestamp(s_float, tz=datetime.timezone.utc)
        else:
            date = datetime.datetime.fromisoformat(s)
            if date.tzinfo is None:
                date = date.replace(tzinfo=datetime.timezone.utc)
    except:
        raise argparse.ArgumentTypeError(f"Invalid date value : {s}")

    if date > datetime.datetime.now(datetime.timezone.utc):
        raise argparse.ArgumentTypeError(f"date cannot be in the future: {date}")
    return date


def selfcheck() -> None:
    """
    Checks the APBuilder is ready to run successfully by checking the following:
    eccodes, pygmt
    """

    print("---- Starting selfcheck for APBuilder ----")

    print("---- Start selfcheck for ecCodes ----")
    from eccodes import __main__ as eccm

    eccm.selfcheck()
    print("---- End selfcheck for ecCodes ----")

    print("---- Start selfcheck for cfgrib ----")
    from cfgrib import messages as cfgm

    print(f"Found: ecCodes v{cfgm.eccodes_version}.")
    print("Your system is ready.")
    print("---- End selfcheck for ecCodes ----")

    print("---- Start selfcheck for PyGMT ----")
    pygmt.show_versions(file=sys.stdout)
    print("---- End selfcheck for PyGMT ----")

    print("---- Completed selfcheck for APBuilder ----")


def choices_multiline(choices):
    return "Allowed values: {" + ", ".join(f"{choice}" for choice in choices) + "} "


def parse_args(args) -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        any: The populated namespace as defined by parse.args().
    """
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, width=100
        ),
        description="Atmospheric Profile Builder (APBuilder)",
        epilog="LLNL-CODE-2012226",
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # The Parent Parser with arguments shared with run parsers
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "datetime",
        action="store",
        type=str,
        help="date of data in ISO 8601 or epoch format",
    )
    parent_parser.add_argument(
        "-wm",
        "--weather-model",
        action="store",
        type=str,
        choices=["GFS", "HRRR", "IFS", "RAP", "ERA5"],
        default="GFS",
        help="weather model",
    )
    parent_parser.add_argument(
        "-e",
        "--epoch",
        help="specify if datetime is in epoch format",
        action="store_true",
    )
    parent_parser.add_argument(
        "-nd",
        "--no-download",
        help="test out the parameters without downloading data",
        action="store_true",
    )
    parent_parser.add_argument(
        "-c",
        "--cycle",
        help="the model or forecast cycle type to download the data",
        type=apbuilder.utils.Cycle,
        choices=list(apbuilder.utils.Cycle),
        default=apbuilder.utils.Cycle.MODEL,
    )
    parent_parser.add_argument(
        "-lf",
        "--local-filename",
        help="name of the local file to read instead of downloading from remote server",
        type=str,
        metavar="filename",
        action="store",
    )
    parent_parser.add_argument(
        "-out",
        "--output-directory",
        help="specify full path directory to save output files",
        action="store",
        type=str,
        metavar="out_dir",
        default=os.path.join(Path.home(), "apbuilder", "output"),
    )
    parent_parser.add_argument(
        "-data",
        "--data-directory",
        help="specify full path directory to save the weather models data files",
        action="store",
        type=str,
        metavar="data_dir",
        default=os.path.join(Path.home(), "apbuilder", "data"),
    )
    parent_parser.add_argument(
        "-pof",
        "--prefix-output-file",
        help="Prefix for the output binary files",
        action="store",
        type=str,
        metavar="prefix",
        default=None,
    )
    # plot range
    parent_parser.add_argument(
        "-clim",
        action="store",
        nargs=2,
        type=float,
        metavar=("min", "max"),
        default=[0, 0],
        help="min and max values for sound speed profile plots",
    )

    parent_parser.add_argument(
        "-dlim",
        action="store",
        nargs=2,
        type=float,
        metavar=("min", "max"),
        default=[0, 0],
        help="min and max values for density profile plots",
    )

    parent_parser.add_argument(
        "-wlim",
        action="store",
        nargs=2,
        type=float,
        metavar=("min", "max"),
        default=[0, 0],
        help="min and max values for wind profile plots",
    )

    # The build1d command parser
    build1d_parser = subparsers.add_parser(
        "build1d",
        parents=[parent_parser],
        help="build 1D profiles",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, width=100
        ),
    )
    build1d_parser.add_argument(
        "lat",
        action="store",
        type=float,
        metavar="[-90, 90]",
        help="latitude of the profile",
    )
    build1d_parser.add_argument(
        "lon",
        action="store",
        type=float,
        metavar="[-180, 360]",
        help="longitude of the profile",
    )
    build1d_parser.add_argument(
        "lat2",
        action="store",
        type=float,
        metavar="[-90, 90]",
        help="latitude of the point to which direction the wind speed is calculated",
    )
    build1d_parser.add_argument(
        "lon2",
        action="store",
        type=float,
        metavar="[-180, 360]",
        help="longitude of the point to which direction the wind speed is calculated",
    )

    # The build 2D parser
    build2d_parser = subparsers.add_parser(
        "build2d",
        parents=[parent_parser],
        help="build 2D profiles",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, width=100
        ),
    )
    build2d_parser.add_argument(
        "lat",
        action="store",
        type=float,
        metavar="[-90, 90]",
        help="latitude of the starting point of the 2D section",
    )
    build2d_parser.add_argument(
        "lon",
        action="store",
        type=float,
        metavar="[-180, 360]",
        help="longitude of the starting point of the 2D section",
    )
    build2d_parser.add_argument(
        "lat2",
        action="store",
        type=float,
        metavar="[-90, 90]",
        help="latitude of the end point of the 2D section",
    )
    build2d_parser.add_argument(
        "lon2",
        action="store",
        type=float,
        metavar="[-180, 360]",
        help="longitude of the end point of the 2D section",
    )
    build2d_parser.add_argument(
        "dr",
        action="store",
        nargs="?",
        type=float,
        metavar="[0, 360]",
        default=0,
        help="horizontal resolution of 2D slice in degree",
    )
    build2d_parser.add_argument(
        "dh",
        action="store",
        nargs="?",
        type=float,
        metavar="[0, ]",
        default=100,
        help="vertical resolution of 2D slice in meter",
    )

    # Additional Information Parser
    info_parser = subparsers.add_parser(
        "info",
        help="additional information",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, width=100
        ),
    )
    info_parser.add_argument(
        "-sb",
        "--sort-by",
        help=f"column name to sort the available weather models table. {choices_multiline(apbuilder.utils.WEATHER_MODELS_TABLE_FIELDS)}",
        choices=apbuilder.utils.WEATHER_MODELS_TABLE_FIELDS,
        default=apbuilder.utils.WEATHER_MODELS_TABLE_FIELDS[0],
        metavar="COLUMN_NAME",
        action="store",
    )
    info_parser.add_argument(
        "-ro",
        "--reverse-order",
        help="Reverse the sorting order of the available weather models table",
        action="store_true",
    )
    group = info_parser.add_argument_group("exclusive options")
    exclusive_group = group.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument(
        "-mi",
        "--models-info",
        help="print information about the available weather models",
        default=argparse.SUPPRESS,
        action="store_true",
    )
    exclusive_group.add_argument(
        "-gmt",
        "--gmt-info",
        help="print information about GMT",
        default=argparse.SUPPRESS,
        action="store_true",
    )
    exclusive_group.add_argument(
        "-sc",
        "--selfcheck",
        help="check the installation of the tool and dependencies",
        default=argparse.SUPPRESS,
        action="store_true",
    )

    parsed_args = parser.parse_args(args)

    try:
        if hasattr(parsed_args, "datetime"):
            parsed_args.datetime = convert_datetime(
                parsed_args.datetime, parsed_args.epoch
            )
    except Exception as e:
        sys.exit(str(e))
    return parsed_args


def get_model_cycle(
    datetime_: datetime.datetime,
    weather_model: str,
    grid_degree: float,
    closest_upward=True,
) -> datetime.datetime:
    """
    Returns a model cycle datetime from a given datetime.
        datetime_: datetime.datetime - the date/time of the event to convert to a model cycle
        weather_model: str - the weather model to use for the model cycle conversion

    Returns:
        datetime.datetime - model cycle datetime
    """
    try:
        concrete_weather_model = apbuilder.weather_models.get_weather_model(
            weather_model
        )
    except apbuilder.exceptions.UnknownWeatherModel:
        logger.warning(
            f"Unable to get model cycle due to unknown {weather_model=}, using provided datetime."
        )
        return datetime_

    logger.debug(
        f"Converting datetime_={datetime_.isoformat()} to get {weather_model=}"
    )
    time_format = "%H:%M"
    closest_time = None
    time_list = concrete_weather_model.get_model_cycle(grid_degree)
    if time_list == None:
        return datetime_

    if closest_upward:
        closest_time = apbuilder.utils.closest_time(
            time_list=time_list, target_time=f"{datetime_:{time_format}}"
        )
    else:
        closest_time = apbuilder.utils.closest_time_downward(
            time_list=time_list, target_time=f"{datetime_:{time_format}}"
        )
    closest_time = datetime.datetime.strptime(closest_time, time_format)
    return datetime_.replace(
        hour=closest_time.time().hour, minute=closest_time.time().minute, second=0
    )


def get_forecast_cycle(
    datetime_: datetime, weather_model: str, grid_degree: float
) -> int:
    """
    Returns a forecast cycle from a given datetime.
        datetime_: datetime.datetime - the date/time of the event to convert to a model cycle
        weather_model: str - the weather model to use for the model cycle conversion

    Returns:
        (datetime.datetime, int) - the model cycle and forecast cycle
    """
    model_cycle_datetime = get_model_cycle(datetime_, weather_model, grid_degree)

    try:
        concrete_weather_model = apbuilder.weather_models.get_weather_model(
            weather_model
        )
    except apbuilder.exceptions.UnknownWeatherModel:
        logger.warning(
            f"Unable to get model cycle due to unknown {weather_model=}, using provided datetime."
        )
        return datetime_, 0

    time_format = "%H:%M"
    time_list = concrete_weather_model.get_forecast_cycle(grid_degree=grid_degree)
    if time_list == None:
        return model_cycle_datetime, 0

    closest_time = apbuilder.utils.closest_time(
        time_list=time_list, target_time=f"{datetime_:{time_format}}"
    )
    closest_time = datetime.datetime.strptime(closest_time, time_format)

    if model_cycle_datetime.hour >= closest_time.hour:
        forecast_cycle = 0
    else:
        forecast_cycle = closest_time.hour - model_cycle_datetime.hour

    return model_cycle_datetime, forecast_cycle


def get_forecast_only_cycle(
    datetime_: datetime.datetime, weather_model: str, grid_degree: float
) -> tuple[datetime.datetime, int]:
    """
    Returns a forecast cycle from a given datetime along with the model cycle going downward.
        datetime_: datetime.datetime - the date/time of the event to convert to a model cycle
        weather_model: str - the weather model to use for the model cycle conversion

    Returns:
        (datetime.datetime, int) - the model cycle and forecast cycle
    """
    model_cycle_datetime = get_model_cycle(
        datetime_, weather_model, grid_degree, closest_upward=False
    )

    try:
        concrete_weather_model = apbuilder.weather_models.get_weather_model(
            weather_model
        )
    except apbuilder.exceptions.UnknownWeatherModel:
        logger.warning(
            f"Unable to get model cycle due to unknown {weather_model=}, using provided datetime."
        )
        return datetime_, 0

    time_format = "%H:%M"
    time_list = concrete_weather_model.get_forecast_cycle(grid_degree=grid_degree)
    if time_list == None:
        return model_cycle_datetime, 0

    closest_time = apbuilder.utils.closest_time(
        time_list=time_list, target_time=f"{datetime_:{time_format}}"
    )
    closest_time = datetime.datetime.strptime(closest_time, time_format)

    if model_cycle_datetime.hour >= closest_time.hour:
        forecast_cycle = 0
    else:
        forecast_cycle = closest_time.hour - model_cycle_datetime.hour

    return model_cycle_datetime, forecast_cycle


def grib_and_index_found(H: Herbie):
    return H.idx is not None and H.grib is not None


def determine_Pa_dataset(low_InhPa, high_InPa):
    if len(high_InPa) > 0:
        high_InhPa = high_InPa.rename({"isobaricInPa": "isobaricInhPa"})
        high_InhPa.coords["isobaricInhPa"] = (
            high_InPa.coords["isobaricInPa"].values / 100
        )
        return xr.concat([low_InhPa, high_InhPa], dim="isobaricInhPa")
    return low_InhPa


def get_filtered_data_from_source(weather_model: str, filename: str) -> xarray.Dataset:
    """Returns the u and v components of wind, elevation and temperature datasets"""

    # To silence the FutureWarning about decode_timedelta in xarray,
    # which will default to False in a future version,
    # set the decode_timedelta parameter to True, False, or a CFTimedeltaCoder instance when opening your dataset
    # decode_timedelta=True: This will ensure that time units are decoded into timedelta objects,
    # as they are currently when decode_timedelta is None
    kwargs = {
        "engine": "cfgrib",
        "decode_timedelta": True,
        "indexpath": "{path}.idx",
        "backend_kwargs": {
            "filter_by_keys": {"cfVarName": "u", "typeOfLevel": "isobaricInhPa"}
        },
    }

    kwargs_InPa = {
        "engine": "cfgrib",
        "decode_timedelta": True,
        "indexpath": "{path}.idx",
        "backend_kwargs": {
            "filter_by_keys": {"cfVarName": "u", "typeOfLevel": "isobaricInPa"}
        },
    }

    kwargs["backend_kwargs"]["filter_by_keys"]["cfVarName"] = "u"
    kwargs_InPa["backend_kwargs"]["filter_by_keys"]["cfVarName"] = "u"
    du_low_InhPa = xr.open_dataset(filename, **kwargs)
    du_high_InPa = xr.open_dataset(filename, **kwargs_InPa)
    du = determine_Pa_dataset(du_low_InhPa, du_high_InPa)

    kwargs["backend_kwargs"]["filter_by_keys"]["cfVarName"] = "v"
    kwargs_InPa["backend_kwargs"]["filter_by_keys"]["cfVarName"] = "v"
    dv_low_InhPa = xr.open_dataset(filename, **kwargs)
    dv_high_InPa = xr.open_dataset(filename, **kwargs_InPa)
    dv = determine_Pa_dataset(dv_low_InhPa, dv_high_InPa)

    kwargs["backend_kwargs"]["filter_by_keys"]["cfVarName"] = "gh"
    kwargs_InPa["backend_kwargs"]["filter_by_keys"]["cfVarName"] = "gh"
    dgh_low_InhPa = xr.open_dataset(filename, **kwargs)
    dgh_high_InPa = xr.open_dataset(filename, **kwargs_InPa)
    dgh = determine_Pa_dataset(dgh_low_InhPa, dgh_high_InPa)

    kwargs["backend_kwargs"]["filter_by_keys"]["cfVarName"] = "t"
    kwargs_InPa["backend_kwargs"]["filter_by_keys"]["cfVarName"] = "t"
    dtmp_low_InhPa = xr.open_dataset(filename, **kwargs)
    dtmp_high_InPa = xr.open_dataset(filename, **kwargs_InPa)
    dtmp = determine_Pa_dataset(dtmp_low_InhPa, dtmp_high_InPa)

    if weather_model in ["ERA5"]:
        kwargs["backend_kwargs"]["filter_by_keys"]["cfVarName"] = "z"
        dgh = xr.open_dataset(filename, **kwargs)

    # If the model attribute is not on dtmp, then it does not exists on the other datasets neither
    if "model" not in dtmp.attrs:
        du.attrs["model"] = weather_model
        dv.attrs["model"] = weather_model
        dgh.attrs["model"] = weather_model
        dtmp.attrs["model"] = weather_model

    return (du, dv, dgh, dtmp)


def convert_sp_to_sp360(sp):
    sp_360 = sp
    if sp_360[0] <= 0.0:
        sp_360[0] = sp_360[0] + 360.0
    return sp_360


def get_atm(
    du: xarray.Dataset,
    dv: xarray.Dataset,
    dgh: xarray.Dataset,
    dtmp: xarray.Dataset,
    profile_format: str,
    sp1: float,
    sp2: float,
    dr: float,
    dh: float,
    save_dir: str,
    clim=[0, 0],
    dlim=[0, 0],
    wlim=[0, 0],
    out_file_prefix="",
):
    sp1 = convert_sp_to_sp360(sp1)
    sp2 = convert_sp_to_sp360(sp2)
    if profile_format == "1D":
        dspd1, dden1, dv1, du1, dr1 = mp_get_atm_1d(
            du, dv, dgh, dtmp, sp1, sp2, save_dir, clim, dlim, wlim, out_file_prefix
        )
    else:
        dspd1, dden1, dv1, du1, dr1 = mp_get_atm_2d(
            du,
            dv,
            dgh,
            dtmp,
            sp1,
            sp2,
            dr,
            dh,
            save_dir,
            clim,
            dlim,
            wlim,
            out_file_prefix,
        )
    return (dspd1, dden1, dv1, du1, dr1)


def mp_get_atm_1d(
    du: xarray.Dataset,
    dv: xarray.Dataset,
    dgh: xarray.Dataset,
    dtmp: xarray.Dataset,
    sp1: float,
    sp2: float,
    save_dir: str,
    clim=[0, 0],
    dlim=[0, 0],
    wlim=[0, 0],
    out_file_prefix="",
):
    logger.info("Processing 1D atmospheric speed, density, v_wind, and u_wind")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit independent calculations
        future_dspd1 = executor.submit(
            apbuilder.profile_1d.atm_speed,
            dtmp,
            dgh,
            sp1,
            save_dir=save_dir,
            clim=clim,
            out_file_prefix=out_file_prefix,
        )
        future_dden1 = executor.submit(
            apbuilder.profile_1d.atm_density,
            dtmp,
            dgh,
            sp1,
            save_dir=save_dir,
            dlim=dlim,
            out_file_prefix=out_file_prefix,
        )
        future_dv1 = executor.submit(
            apbuilder.profile_1d.atm_v_wind,
            dv,
            dgh,
            sp1,
            save_dir=save_dir,
            wlim=wlim,
            out_file_prefix=out_file_prefix,
        )
        future_du1 = executor.submit(
            apbuilder.profile_1d.atm_u_wind,
            du,
            dgh,
            sp1,
            save_dir=save_dir,
            wlim=wlim,
            out_file_prefix=out_file_prefix,
        )

        dspd1 = future_dspd1.result()
        logger.info("Processing atmospheric speed done")
        dden1 = future_dden1.result()
        logger.info("Processing atmospheric density done")
        dv1 = future_dv1.result()
        logger.info("Processing atmospheric v_wind done")
        du1 = future_du1.result()
        logger.info("Processing atmospheric u_wind done")

    # r_wind depends on du1 and dv1, so do it after
    logger.info("Processing atmospheric r_wind")
    dr1 = apbuilder.profile_1d.atm_r_wind(
        du,
        du1,
        dv1,
        sp1,
        sp2,
        save_dir=save_dir,
        wlim=wlim,
        out_file_prefix=out_file_prefix,
    )
    logger.info("Processing atmospheric r_wind done")
    return (dspd1, dden1, dv1, du1, dr1)


def mp_get_atm_2d(
    du: xarray.Dataset,
    dv: xarray.Dataset,
    dgh: xarray.Dataset,
    dtmp: xarray.Dataset,
    sp1: float,
    sp2: float,
    dr: float,
    dh: float,
    save_dir: str,
    clim=[0, 0],
    dlim=[0, 0],
    wlim=[0, 0],
    out_file_prefix="",
):
    logger.info("Processing 2D atmospheric speed, density, v_wind, and u_wind")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Start independent calculations in parallel
        future_dspd0 = executor.submit(
            apbuilder.profile_2d.atm_speed,
            dtmp,
            dgh,
            sp1,
            sp2,
            dr,
            save_dir=save_dir,
            clim=clim,
            out_file_prefix=out_file_prefix,
        )
        future_dden0 = executor.submit(
            apbuilder.profile_2d.atm_density,
            dtmp,
            dgh,
            sp1,
            sp2,
            dr,
            save_dir=save_dir,
            dlim=dlim,
            out_file_prefix=out_file_prefix,
        )
        future_dv0 = executor.submit(
            apbuilder.profile_2d.atm_v_wind,
            dv,
            dgh,
            sp1,
            sp2,
            dr,
            save_dir=save_dir,
            wlim=wlim,
            out_file_prefix=out_file_prefix,
        )
        future_du0 = executor.submit(
            apbuilder.profile_2d.atm_u_wind,
            du,
            dgh,
            sp1,
            sp2,
            dr,
            save_dir=save_dir,
            wlim=wlim,
            out_file_prefix=out_file_prefix,
        )

        dspd0 = future_dspd0.result()
        logger.info("Processing atmospheric speed done")
        dden0 = future_dden0.result()
        logger.info("Processing atmospheric density done")
        dv0 = future_dv0.result()
        logger.info("Processing atmospheric v_wind done")
        du0 = future_du0.result()
        logger.info("Processing atmospheric u_wind done")

    # Now that du0 and dv0 are ready, process r_wind (in main process)
    logger.info("Processing atmospheric r_wind")
    dr0 = apbuilder.profile_2d.atm_r_wind(
        du,
        du0,
        dv0,
        sp1,
        sp2,
        save_dir=save_dir,
        wlim=wlim,
        out_file_prefix=out_file_prefix,
    )
    logger.info("Processing atmospheric r_wind done")

    # Interpolations can also be done in parallel
    logger.info("Processing interpolations")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_dspd1 = executor.submit(apbuilder.profile_2d.atm_interp, dspd0, dr, dh)
        future_dden1 = executor.submit(apbuilder.profile_2d.atm_interp, dden0, dr, dh)
        future_dv1 = executor.submit(apbuilder.profile_2d.atm_interp, dv0, dr, dh)
        future_du1 = executor.submit(apbuilder.profile_2d.atm_interp, du0, dr, dh)
        future_dr1 = executor.submit(apbuilder.profile_2d.atm_interp, dr0, dr, dh)

        dspd1 = future_dspd1.result()
        logger.info("Processing atmospheric speed interpolation done")
        dden1 = future_dden1.result()
        logger.info("Processing atmospheric density interpolation done")
        dv1 = future_dv1.result()
        logger.info("Processing atmospheric v_wind interpolation done")
        du1 = future_du1.result()
        logger.info("Processing atmospheric u_wind interpolation done")
        dr1 = future_dr1.result()
        logger.info("Processing atmospheric r_wind interpolation done")
    return (dspd1, dden1, dv1, du1, dr1)


def extract(
    datetime_: datetime,
    weather_model: str,
    profile_format: str,
    sp1,
    sp2,
    dr=0,
    dh=100,
    out_dir="output",
    data_dir="data",
    no_download=False,
    cycle: apbuilder.utils.Cycle = "model",
    local_filename: str = None,
    clim=[0, 0],
    dlim=[0, 0],
    wlim=[0, 0],
    out_file_prefix="",
) -> str:
    """
    Downloads the grib data and saves it in NetCDF4 format.

    Args:
        datetime_: the date/time to search for the weather model
        weather_model: the weather model to search for

    Returns:
        str: path to NetCDF file
    """

    # Create directories if not exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    H = None
    if local_filename is None:
        logger.debug(
            f"Converting user provided datetime_={datetime_.isoformat()} to a weather model datetime"
        )
        try:
            concrete_weather_model = apbuilder.weather_models.get_weather_model(
                weather_model
            )
        except apbuilder.exceptions.UnknownWeatherModel as e:
            msg = f"Unable to get model cycle due to unknown {weather_model=}, using provided datetime."
            logger.warning(msg)
            raise e

        grid_degrees = concrete_weather_model.get_supported_grid_degrees()
        data_found = False
        for grid_degree in grid_degrees:
            match cycle:
                case apbuilder.utils.Cycle.MODEL:
                    model_time = get_model_cycle(datetime_, weather_model, grid_degree)
                    forecast_cycle = 0
                case apbuilder.utils.Cycle.FORECAST:
                    model_time, forecast_cycle = get_forecast_cycle(
                        datetime_, weather_model, grid_degree
                    )
                case apbuilder.utils.Cycle.FORECAST_ONLY:
                    model_time, forecast_cycle = get_forecast_only_cycle(
                        datetime_, weather_model, grid_degree
                    )
                case _:
                    raise apbuilder.exceptions.UnknownModelCycle(
                        f"Unknown model {cycle=}"
                    )

            logger.debug(
                f"The model_time={model_time.isoformat()} will be used as the weather model datetime"
            )
            try:
                product = concrete_weather_model.get_grid_degree_as_herbie_product(
                    grid_degree, model_time
                )
            except apbuilder.exceptions.UnsupportedGridDegreeError as e:
                continue

            logger.info(
                f"Extracting data for {weather_model=} for model_time={model_time.isoformat()} and {forecast_cycle=} with {grid_degree=} from user provided datetime={datetime_.isoformat()}"
            )

            try:
                H = Herbie(
                    model_time.replace(tzinfo=None),
                    model=weather_model,
                    fxx=forecast_cycle,
                    save_dir=data_dir,
                    priority=PRIORITY,
                    product=product,
                    overwrite=False,
                )
                if grib_and_index_found(H):
                    data_found = True
                    break
                else:
                    logger.info(
                        f"Unable to find grib and index data for {grid_degree=}"
                    )
            except Exception as e:
                logger.error(e)
                continue

        if not data_found:
            if H == None:
                msg = f"Unable to find data for model_time='{model_time.isoformat()}'"
            else:
                msg = f"Unable to find data for model_time='{model_time.isoformat()}' on any of the repositories={H.priority} for any of the resolutions={grid_degrees}"
            raise apbuilder.exceptions.DataNotFoundError(msg)

        if no_download:
            raise apbuilder.exceptions.NoDownloadMode(
                f"Exiting due to no_download flag turned on."
            )

        logger.info(f"Downloading grib data to {data_dir}...")
        _ = H.download(overwrite=False, save_dir=data_dir, verbose=True)

    filename = local_filename if local_filename != None else H.grib
    logger.info(f"Getting filtered data from {filename=}")
    du, dv, dgh, dtmp = get_filtered_data_from_source(weather_model, filename)

    logger.info(f"Extracted {weather_model=}")
    dspd1, dden1, dv1, du1, dr1 = get_atm(
        du,
        dv,
        dgh,
        dtmp,
        profile_format,
        sp1,
        sp2,
        dr,
        dh,
        out_dir,
        clim=clim,
        dlim=dlim,
        wlim=wlim,
        out_file_prefix=out_file_prefix,
    )

    return [dspd1, dden1, dv1, du1, dr1]


def transform(lat: float, lon: float, netcdf_file: str):
    logger.info("TODO: implement transformation from NetCDF to ASCII and AC2DR profile")
    netcdf_data = xarray.open_dataset(netcdf_file)
    # gmt project -C-115./42.5 -E-118./38.4 -G50k -Q
    points = pygmt.project(
        netcdf_data,
        center=[[-115.0, 42.5]],
        endpoint=[[-118.0, 38.4]],
        generate=10,
        unit=True,
        output_type="pandas",
    )
    logger.info(f"{points}")


def load():
    # logger.info("TODO: implement loading/saving AC2Dr atmosphere profile to file")
    pass


def print_models_info(sort_by: str = None, reverse_sort: bool = False):
    table = apbuilder.utils.available_weather_models_info_prettytable(
        sort_by=sort_by, reverse_sort=reverse_sort
    )
    print(table)


def print_info_commands(args: argparse.Namespace):
    if hasattr(args, "models_info") and args.models_info:
        print_models_info(args.sort_by, args.reverse_order)
        return True

    if hasattr(args, "gmt_info") and args.gmt_info:
        pygmt.show_versions(file=sys.stdout)
        return True

    if hasattr(args, "selfcheck") and args.selfcheck:
        selfcheck()
        return True


def run_apb() -> int:
    """
    The entry point to the application for testing.
    """
    return run_apb_with_args(sys.argv[1:])


def run_apb_with_args(user_args) -> int:
    """
    The entry point to the application.
    """

    configure_logger()
    args = parse_args(user_args)

    if print_info_commands(args):
        return 0

    if args.subcommand == "build1d":
        try:
            atm = extract(
                args.datetime,
                args.weather_model,
                "1D",
                [args.lon, args.lat],
                [args.lon2, args.lat2],
                data_dir=args.data_directory,
                out_dir=args.output_directory,
                no_download=args.no_download,
                cycle=args.cycle,
                local_filename=args.local_filename,
                clim=args.clim,
                dlim=args.dlim,
                wlim=args.wlim,
                out_file_prefix=args.prefix_output_file,
            )
        except requests.exceptions.SSLError as e:
            logger.error(
                f"Error downloading data from 3rd party source. If you are on a VPN connection, disconnect and try again. {e}"
            )
            return 1
        except EOFError as e:
            logger.error(f"Unable to parse data due to no data available: {e}")
            return 1
        except apbuilder.exceptions.NoDownloadMode as e:
            logger.warning(e)
            return 1
        except Exception as e:
            logger.error(f"{e}")
            return 1
        apbuilder.profile_1d.writebin_ac2dr(
            atm[0],
            "speed",
            save_dir=args.output_directory,
            out_file_prefix=args.prefix_output_file,
        )
        apbuilder.profile_1d.writebin_ac2dr(
            atm[1],
            "density",
            save_dir=args.output_directory,
            out_file_prefix=args.prefix_output_file,
        )
        apbuilder.profile_1d.writebin_ac2dr(
            atm[2],
            "Vwind",
            save_dir=args.output_directory,
            out_file_prefix=args.prefix_output_file,
        )
        apbuilder.profile_1d.writebin_ac2dr(
            atm[3],
            "Uwind",
            save_dir=args.output_directory,
            out_file_prefix=args.prefix_output_file,
        )
        apbuilder.profile_1d.writebin_ac2dr(
            atm[4],
            "Rwind",
            save_dir=args.output_directory,
            out_file_prefix=args.prefix_output_file,
        )
    elif args.subcommand == "build2d":
        atm = extract(
            args.datetime,
            args.weather_model,
            "2D",
            [args.lon, args.lat],
            [args.lon2, args.lat2],
            dr=args.dr,
            dh=args.dh,
            out_dir=args.output_directory,
            data_dir=args.data_directory,
            no_download=args.no_download,
            cycle=args.cycle,
            local_filename=args.local_filename,
            clim=args.clim,
            dlim=args.dlim,
            wlim=args.wlim,
            out_file_prefix=args.prefix_output_file,
        )
        # transform(args.lat, args.lon, netcdf_file)
        try:
            apbuilder.profile_2d.writebin_ac2dr(
                atm[0],
                "speed",
                save_dir=args.output_directory,
                out_file_prefix=args.prefix_output_file,
            )
            apbuilder.profile_2d.writebin_ac2dr(
                atm[1],
                "density",
                save_dir=args.output_directory,
                out_file_prefix=args.prefix_output_file,
            )
            apbuilder.profile_2d.writebin_ac2dr(
                atm[2],
                "Vwind",
                save_dir=args.output_directory,
                out_file_prefix=args.prefix_output_file,
            )
            apbuilder.profile_2d.writebin_ac2dr(
                atm[3],
                "Uwind",
                save_dir=args.output_directory,
                out_file_prefix=args.prefix_output_file,
            )
            apbuilder.profile_2d.writebin_ac2dr(
                atm[4],
                "Rwind",
                save_dir=args.output_directory,
                out_file_prefix=args.prefix_output_file,
            )
        except HorizontalResolutionError as e:
            logger.error(
                f"Error: {e} Arguments: horizontal_res={args.dr} start_lat={args.lat} end_lat={args.lat2}"
            )
            return 1
    else:
        raise RuntimeError(f"Unknown subcommand={args.subcommand}")

    load()
    logger.info(f"Completed with output saved to {args.output_directory}")
    return 0


if __name__ == "__main__":
    run_apb_with_args(sys.argv[1:])
