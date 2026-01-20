import pytest
import os

from PIL import Image, ImageChops
from apbuilder import main

OUTPUT_DIRECTORY = os.path.join("tests", "integration", "actual")
EXPECTED_DIRECTORY = os.path.join("tests", "integration", "expected")
DATA_DIRECTORY = os.path.join("tests", "data")


def test_ncei_analysis_2020_10_29_1D(tmp_path):
    test_name = "2020_10_29_ncei_analysis_1d"
    datetime_ = "20201029"
    lat = "37.1149"
    lon = "243.9309"
    lat2 = "37.1149"
    lon2 = "243.9309"
    out_dir = os.path.join(OUTPUT_DIRECTORY, tmp_path)
    expected_dir = os.path.join(EXPECTED_DIRECTORY, test_name)
    args = ["build1d", datetime_, lat, lon, lat2, lon2, "-out", out_dir]
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


def test_ncei_analysis_historical_2017_12_12_1D(tmp_path):
    test_name = "2017_12_12_ncei_historical_analysis_1d"
    datetime_ = "20171212"
    lat = "37.1149"
    lon = "243.9309"
    lat2 = "37.1149"
    lon2 = "243.9309"
    out_dir = os.path.join(OUTPUT_DIRECTORY, tmp_path)
    expected_dir = os.path.join(EXPECTED_DIRECTORY, test_name)
    args = ["build1d", datetime_, lat, lon, lat2, lon2, "-out", out_dir]
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


def test_ncei_analysis_historical_2016_08_15_2D(tmp_path):
    test_name = "2016_08_15_ncei_historical_analysis_2d"
    datetime_ = "20160815"
    lat1 = "48.3200"
    lon1 = "16.8700"
    lat2 = "47.9183"
    lon2 = "19.8908"
    res = "0.5"
    weather_model = "GFS"
    out_dir = os.path.join(OUTPUT_DIRECTORY, tmp_path)
    expected_dir = os.path.join(EXPECTED_DIRECTORY, test_name)
    args = [
        "build2d",
        datetime_,
        lat1,
        lon1,
        lat2,
        lon2,
        res,
        "-out",
        out_dir,
        "-wm",
        weather_model,
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


def test_ifs_ecmwf_2024_08_25_1D(tmp_path):
    test_name = "2024_08_25_ifs_ecmwf_1d"
    datetime_ = "20240825"
    lat = "37.1149"
    lon = "243.9309"
    lat2 = "37.1149"
    lon2 = "243.9309"
    weather_model = "IFS"
    out_dir = os.path.join(OUTPUT_DIRECTORY, tmp_path)
    expected_dir = os.path.join(EXPECTED_DIRECTORY, test_name)
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
