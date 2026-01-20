import pytest
import os

from PIL import Image, ImageChops
from apbuilder import main

OUTPUT_DIRECTORY = os.path.join("tests", "integration", "actual")
EXPECTED_DIRECTORY = os.path.join("tests", "integration", "expected")
DATA_DIRECTORY = os.path.join("tests", "data")


def test_era5_ecmwf_localfile_2020_08_04_1D(tmp_path):
    test_name = "2020_08_04_era5_ecmwf_localfile_1d"
    datetime_ = "20200804"
    lat = "50.6565"
    lon = "6.8033"
    lat2 = "53.2534"
    lon2 = "8.68980"
    weather_model = "ERA5"
    out_dir = os.path.join(OUTPUT_DIRECTORY, tmp_path)
    expected_dir = os.path.join(EXPECTED_DIRECTORY, test_name)
    local_filename = os.path.join(
        DATA_DIRECTORY,
        weather_model.lower(),
        datetime_,
        "ECMWF_ERA5_20200804_1200.grib",
    )
    args = [
        "build1d",
        datetime_,
        lat,
        lon,
        lat2,
        lon2,
        "-out",
        out_dir,
        "-wm",
        weather_model,
        "-lf",
        local_filename,
    ]
    main.run_apb_with_args(args)

    files = [file for file in os.listdir(expected_dir) if file.endswith(".png")]
    for file in files:
        # assign images
        expected_filename = os.path.join(EXPECTED_DIRECTORY, test_name, file)
        actual_filename = os.path.join(out_dir, file)
        expected_image = Image.open(expected_filename)
        actual_image = Image.open(actual_filename)

        # finding difference
        diff = ImageChops.difference(expected_image, actual_image)
        bbox = diff.getbbox(alpha_only=False)
        if bbox != None:
            diff.show()

        # assert images are the same
        assert (
            bbox == None
        ), f"actual image {actual_filename} is different than expected image {expected_filename}"


def test_ifs_ecmwf_localfile_2024_08_25_1D(tmp_path):
    test_name = "2024_08_25_ifs_ecmwf_localfile_1d"
    datetime_ = "20240825"
    lat = "37.1149"
    lon = "243.9309"
    lat2 = "37.1149"
    lon2 = "243.9309"
    weather_model = "IFS"
    out_dir = os.path.join(OUTPUT_DIRECTORY, tmp_path)
    expected_dir = os.path.join(EXPECTED_DIRECTORY, test_name)
    local_filename = os.path.join(
        DATA_DIRECTORY,
        weather_model.lower(),
        datetime_,
        "20240825000000-0h-oper-fc.grib2",
    )
    args = [
        "build1d",
        datetime_,
        lat,
        lon,
        lat2,
        lon2,
        "-out",
        out_dir,
        "-wm",
        weather_model,
        "-lf",
        local_filename,
    ]
    main.run_apb_with_args(args)

    files = [file for file in os.listdir(expected_dir) if file.endswith(".png")]
    for file in files:
        # assign images
        expected_filename = os.path.join(EXPECTED_DIRECTORY, test_name, file)
        actual_filename = os.path.join(out_dir, file)
        expected_image = Image.open(expected_filename)
        actual_image = Image.open(actual_filename)

        # finding difference
        diff = ImageChops.difference(expected_image, actual_image)
        bbox = diff.getbbox(alpha_only=False)
        if bbox != None:
            diff.show()

        # assert images are the same
        assert (
            bbox == None
        ), f"actual image {actual_filename} is different than expected image {expected_filename}"


def test_gfs_noaa_localfile_2014_01_03_1D(tmp_path):
    test_name = "2014_01_03_gfs_noaa_localfile_1d"
    datetime_ = "20140103"
    lat = "50.6565"
    lon = "6.8033"
    lat2 = "50.6565"
    lon2 = "6.8033"
    weather_model = "GFS"
    out_dir = os.path.join(OUTPUT_DIRECTORY, tmp_path, "sub_folder", "tmp")
    expected_dir = os.path.join(EXPECTED_DIRECTORY, test_name)
    local_filename = os.path.join(
        DATA_DIRECTORY, weather_model.lower(), datetime_, "gfs_4_20140103_1200_000.grb2"
    )
    args = [
        "build1d",
        datetime_,
        lat,
        lon,
        lat2,
        lon2,
        "-out",
        out_dir,
        "-wm",
        weather_model,
        "-lf",
        local_filename,
    ]
    main.run_apb_with_args(args)

    files = [file for file in os.listdir(expected_dir) if file.endswith(".png")]
    for file in files:
        # assign images
        expected_filename = os.path.join(EXPECTED_DIRECTORY, test_name, file)
        actual_filename = os.path.join(out_dir, file)
        expected_image = Image.open(expected_filename)
        actual_image = Image.open(actual_filename)

        # finding difference
        diff = ImageChops.difference(expected_image, actual_image)
        bbox = diff.getbbox(alpha_only=False)
        if bbox != None:
            diff.show()

        # assert images are the same
        assert (
            bbox == None
        ), f"actual image {actual_filename} is different than expected image {expected_filename}"


def test_gfs_noaa_localfile_2014_01_03_2D(tmp_path):
    test_name = "2014_01_03_gfs_noaa_localfile_2d"
    datetime_ = "20140103"
    lat = "50.6565"
    lon = "6.8033"
    lat2 = "51.6565"
    lon2 = "7.8033"
    weather_model = "GFS"
    out_dir = os.path.join(OUTPUT_DIRECTORY, tmp_path)
    expected_dir = os.path.join(EXPECTED_DIRECTORY, test_name)
    local_filename = os.path.join(
        DATA_DIRECTORY, weather_model.lower(), datetime_, "gfs_4_20140103_1200_000.grb2"
    )
    args = [
        "build2d",
        datetime_,
        lat,
        lon,
        lat2,
        lon2,
        "-out",
        out_dir,
        "-wm",
        weather_model,
        "-lf",
        local_filename,
    ]
    main.run_apb_with_args(args)

    files = [file for file in os.listdir(expected_dir) if file.endswith(".png")]
    for file in files:
        # assign images
        expected_filename = os.path.join(EXPECTED_DIRECTORY, test_name, file)
        actual_filename = os.path.join(out_dir, file)
        expected_image = Image.open(expected_filename)
        actual_image = Image.open(actual_filename)

        # finding difference
        diff = ImageChops.difference(expected_image, actual_image)
        bbox = diff.getbbox(alpha_only=True)
        if bbox != None:
            diff.show()

        # assert images are the same
        assert (
            bbox == None
        ), f"actual image {actual_filename} is different than expected image {expected_filename}"


def test_gfs_noaa_localfile_2014_01_03_2D_plot_limits(tmp_path):
    test_name = "2014_01_03_gfs_noaa_localfile_plot_limits_2d"
    datetime_ = "20140103"
    lat = "50.6565"
    lon = "6.8033"
    lat2 = "51.6565"
    lon2 = "7.8033"
    weather_model = "GFS"
    clim = ["-400.0", "400.0"]
    dlim = ["-400.0", "400.0"]
    wlim = ["-400.0", "400.0"]
    out_dir = os.path.join(OUTPUT_DIRECTORY, tmp_path)
    expected_dir = os.path.join(EXPECTED_DIRECTORY, test_name)
    local_filename = os.path.join(
        DATA_DIRECTORY, weather_model.lower(), datetime_, "gfs_4_20140103_1200_000.grb2"
    )
    args = [
        "build2d",
        datetime_,
        lat,
        lon,
        lat2,
        lon2,
        "-out",
        out_dir,
        "-wm",
        weather_model,
        "-lf",
        local_filename,
        "-clim",
        clim[0],
        clim[1],
        "-dlim",
        dlim[0],
        dlim[1],
        "-wlim",
        wlim[0],
        wlim[1],
    ]
    main.run_apb_with_args(args)

    files = [file for file in os.listdir(expected_dir) if file.endswith(".png")]
    for file in files:
        # assign images
        expected_filename = os.path.join(EXPECTED_DIRECTORY, test_name, file)
        actual_filename = os.path.join(out_dir, file)
        expected_image = Image.open(expected_filename)
        actual_image = Image.open(actual_filename)

        # finding difference
        diff = ImageChops.difference(expected_image, actual_image)
        bbox = diff.getbbox(alpha_only=True)
        if bbox != None:
            diff.show()

        # assert images are the same
        assert (
            bbox == None
        ), f"actual image {actual_filename} is different than expected image {expected_filename}"
