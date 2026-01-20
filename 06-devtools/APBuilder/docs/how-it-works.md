# How it works?

APBuilder uses `herbie-data` to download grib and inventory files from remote
servers. APBuilder also supports reading local grib files without an inventory
file. This application will check if the grib file exists locally, if not it
will download it from a remote server. Then process the file accordingly. The
next time you run this application with the datetime and weather model, it
will use the local copy of the file, saving time as there is no need to
download the file. You can also specify a local grib filename to use instead of
downloading from a remote server.

Then APBuilder will generate the atmospheric profile files to be used in AC2Dr.
In addition, it generates images to visualize the data. The output is saved to
the default location or the directory specified by the user with the `-out` parameter.

## Directory Structure

APBuilder uses a directory structure created by `herbie-data`, which is the following:

- `out_dir` is the output directory set by the user in the command
- `weather_model` is the weather model set by the user in the command, or default if not set
- `yyyymmdd` is the date of the files to search set by the user in the command
- `filename` is the filename of the grib2 file. This filename has a defined structure, see more info below.

```text
|-- out_dir
|   |-- weather_model
|       |-- yyyymmdd
|           |-- filename
```

Here is a concrete example.

```text
|-- data
|   |-- gfs
|       |-- 20201029
|           |-- gfs_4_20201029_0000_000.grb2
```

## Adding local files manually

You can use a local grib file instead of downloading from a remote server.
The file must be placed inside a specific folder as denoted above on the `Directory Structure` section.

To use the local file with APBuilder, make sure to specify the `-lf` or `--local-filename` flag with
the full path to the file.

For example:

```bash
apbuilder build1d 20200804T00:00:00 50.6565 6.8033 53.2534 8.68980 \
    C:\\Users\\myuser\\apbuilder\\data \
    -lf C:\\Users\\myuser\\apbuilder\\data\\ifs\\20200804\\ECMWF_ERA5_20200804_1200.grib \
    -wm ERA5
```
