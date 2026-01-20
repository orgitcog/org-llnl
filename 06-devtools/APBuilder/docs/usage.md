# Using APBuilder

## Validate Installation

You can run a selfcheck to make sure there are no errors with 3rd party dependencies.  
No errors should be reported. If there is any error, please double check you installed
everything correctly.

```bash
apbuilder info --selfcheck
```

## Test Run

You can run a basic 1D profile build with the following command:

````bash
apbuilder build1d 20240601 37.1149 243.9309 38.1149 244.9309
````

Here is the command to test a 2D profile build:

```bash
apbuilder build2d 20160815 48.3200 16.8700 47.9183 19.8908 0.5
```

## Usage

**If NCEI source does not work on VPN, please disconnect VPN and try again.**

```bash
$ apbuilder --help
usage: apbuilder [-h] [-v] {build1d,build2d,info} ...

Atmospheric Profile Builder (APBuilder)

positional arguments:
  {build1d,build2d,info}
    build1d             build 1D profiles
    build2d             build 2D profiles
    info                additional information

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit

LLNL
```

```bash
$ apbuilder build1d --help
usage: apbuilder build1d [-h] [-wm {GFS,HRRR,IFS,RAP,ERA5}] [-e] [-nd]
                         [-c {model,forecast,forecast-only}] [-lf filename] [-out out_dir]
                         [-data data_dir] [-pof prefix] [-clim min max] [-dlim min max]
                         [-wlim min max]
                         datetime [-90, 90] [-180, 360] [-90, 90] [-180, 360]

positional arguments:
  datetime              date of data in ISO 8601 or epoch format
  [-90, 90]             latitude of the profile
  [-180, 360]           longitude of the profile
  [-90, 90]             latitude of the point to which direction the wind speed is calculated
  [-180, 360]           longitude of the point to which direction the wind speed is calculated

options:
  -h, --help            show this help message and exit
  -wm {GFS,HRRR,IFS,RAP,ERA5}, --weather-model {GFS,HRRR,IFS,RAP,ERA5}
                        weather model (default: GFS)
  -e, --epoch           specify if datetime is in epoch format (default: False)
  -nd, --no-download    test out the parameters without downloading data (default: False)
  -c {model,forecast,forecast-only}, --cycle {model,forecast,forecast-only}
                        the model or forecast cycle type to download the data (default: model)
  -lf filename, --local-filename filename
                        name of the local file to read instead of downloading from remote server
                        (default: None)
  -out out_dir, --output-directory out_dir
                        specify full path directory to save output files (default:
                        C:\Users\myuser\apbuilder\output)
  -data data_dir, --data-directory data_dir
                        specify full path directory to save the weather models data files (default:
                        C:\Users\myuser\apbuilder\data)
  -pof prefix, --prefix-output-file prefix
                        Prefix for the output binary files (default: None)
  -clim min max         min and max values for sound speed profile plots (default: [0, 0])
  -dlim min max         min and max values for density profile plots (default: [0, 0])
  -wlim min max         min and max values for wind profile plots (default: [0, 0])
```

```bash
$ apbuilder build2d --help
usage: apbuilder build2d [-h] [-wm {GFS,HRRR,IFS,RAP,ERA5}] [-e] [-nd]
                         [-c {model,forecast,forecast-only}] [-lf filename] [-out out_dir]
                         [-data data_dir] [-pof prefix] [-clim min max] [-dlim min max]
                         [-wlim min max]
                         datetime [-90, 90] [-180, 360] [-90, 90] [-180, 360] [[0, 360]] [[0,]]

positional arguments:
  datetime              date of data in ISO 8601 or epoch format
  [-90, 90]             latitude of the starting point of the 2D section
  [-180, 360]           longitude of the starting point of the 2D section
  [-90, 90]             latitude of the end point of the 2D section
  [-180, 360]           longitude of the end point of the 2D section
  [0, 360]              horizontal resolution of 2D slice in degree (default: 0)
  [0, ]                 vertical resolution of 2D slice in meter (default: 100)

options:
  -h, --help            show this help message and exit
  -wm {GFS,HRRR,IFS,RAP,ERA5}, --weather-model {GFS,HRRR,IFS,RAP,ERA5}
                        weather model (default: GFS)
  -e, --epoch           specify if datetime is in epoch format (default: False)
  -nd, --no-download    test out the parameters without downloading data (default: False)
  -c {model,forecast,forecast-only}, --cycle {model,forecast,forecast-only}
                        the model or forecast cycle type to download the data (default: model)
  -lf filename, --local-filename filename
                        name of the local file to read instead of downloading from remote server
                        (default: None)
  -out out_dir, --output-directory out_dir
                        specify full path directory to save output files (default:
                        C:\Users\myuser\apbuilder\output)
  -data data_dir, --data-directory data_dir
                        specify full path directory to save the weather models data files (default:
                        C:\Users\myuser\apbuilder\data)
  -pof prefix, --prefix-output-file prefix
                        Prefix for the output binary files (default: None)
  -clim min max         min and max values for sound speed profile plots (default: [0, 0])
  -dlim min max         min and max values for density profile plots (default: [0, 0])
  -wlim min max         min and max values for wind profile plots (default: [0, 0])
```

```bash
$ apbuilder info --help
usage: apbuilder info [-h] [-sb COLUMN_NAME] [-ro] (-mi | -gmt | -sc)

options:
  -h, --help            show this help message and exit
  -sb COLUMN_NAME, --sort-by COLUMN_NAME
                        column name to sort the available weather models table. Allowed values:
                        {Model, Grid (Deg), Time Period, Model Cycle, Forecast Cycle, Geographic
                        Extent} (default: Model)
  -ro, --reverse-order  Reverse the sorting order of the available weather models table (default:
                        False)

exclusive options:
  -mi, --models-info    print information about the available weather models
  -gmt, --gmt-info      print information about GMT
  -sc, --selfcheck      check the installation of the tool and dependencies
```
