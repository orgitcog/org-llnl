import logging
import math
import numpy as np
import struct
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

import apbuilder.utils

logger = logging.getLogger(__name__)


def atm_u_wind(da, dgh, pt1, save_dir=".", wlim=[0, 0], out_file_prefix=""):
    return __atm_wind(
        da,
        dgh,
        pt1,
        "UComponentOfWind",
        save_dir=save_dir,
        wlim=wlim,
        out_file_prefix=out_file_prefix,
    )


def atm_v_wind(da, dgh, pt1, save_dir=".", wlim=[0, 0], out_file_prefix=""):
    return __atm_wind(
        da,
        dgh,
        pt1,
        "VComponentOfWind",
        save_dir=save_dir,
        wlim=wlim,
        out_file_prefix=out_file_prefix,
    )


def atm_r_wind(da, u1, v1, pt1, pt2, save_dir=".", wlim=[0, 0], out_file_prefix=""):
    theta = apbuilder.utils.get_azimuth(pt1, pt2)
    var1 = u1.copy()
    # unit vector to r (x, y)
    rv = [math.sin(theta * math.pi / 180), math.cos(theta * math.pi / 180)]
    # project wind, rw
    var1.values = u1.values * rv[0] + v1.values * rv[1]
    var1.attrs["long_name"] = "R component of wind"
    if wlim[0] == 0 and wlim[1] == 0:
        var1.plot(y="height")
    else:
        var1.plot(y="height", xlim=wlim)
    # plt.imshow(temp.T)
    # plt.show()
    datetime_str = apbuilder.utils.convert_datetime_to_str_for_filename(
        da.coords["time"].values.flatten()[0]
    )
    fn = "%s/%s%s_%s_1d_%s.png" % (
        save_dir,
        (out_file_prefix or ""),
        da.attrs["model"].lower(),
        datetime_str,
        "RComponentOfWind",
    )
    tlt1 = "%s %s" % (
        da.attrs["model"].upper(),
        da.coords["time"].values.astype("datetime64[s]"),
    )
    var_dphi = pt2[1] - pt1[1]
    var_dlam = pt2[0] - pt1[0]
    ER = 6371000
    var_th = math.atan2(
        math.sin(var_dlam * math.pi / 180) * math.cos(pt2[1] * math.pi / 180),
        math.cos(pt1[1] * math.pi / 180) * math.sin(pt2[1] * math.pi / 180)
        - math.sin(pt1[1] * math.pi / 180)
        * math.cos(pt2[1] * math.pi / 180)
        * math.cos(var_dlam * math.pi / 180),
    )
    # tlt2="from (N%s, E%s) to (N%s, E%s)"%(pt1[1],pt1[0],pt2[1],pt2[0])
    tlt2 = "%s degrees from North at (N%s, E%s)" % (
        round(var_th * 180 / math.pi, 1),
        pt1[1],
        pt1[0],
    )
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    return var1


def atm_speed(da, dgh, pt1, save_dir=".", clim=[0, 0], out_file_prefix=""):
    # da: variable data (temperature, v, u), hand over a xarrayArray with a single variable
    # dgh: height data (gh)
    gam = 1.4
    R0 = 8.3143
    M0 = 0.029
    Rd = R0 / M0
    # ceff1=sqrt(gam*Rd*tarry)-wyarry

    gh1, _ = __get_gh(da, dgh, pt1)

    # var1=pres1*100
    da_var = list(da.data_vars)[0]
    temp1 = da.sel(longitude=pt1[0], latitude=pt1[1], method="nearest")[da_var].values
    var1 = (gam * Rd * temp1) ** (1 / 2)
    coords = {"height": gh1}
    var1_xr = xr.DataArray(var1, coords, dims=["height"])
    var1_xr.attrs = {"long_name": "SoundSpeed", "units": "m/s"}
    var1_xr["height"].attrs = {"long_name": "Height", "units": "m"}
    if clim[0] == 0 and clim[1] == 0:
        var1_xr.plot(y="height")
    else:
        var1_xr.plot(y="height", xlim=clim)
    # plt.imshow(temp.T)
    # plt.show()
    datetime_str = apbuilder.utils.convert_datetime_to_str_for_filename(
        da.coords["time"].values.flatten()[0]
    )
    fn = "%s/%s%s_%s_1d_%s.png" % (
        save_dir,
        (out_file_prefix or ""),
        da.attrs["model"].lower(),
        datetime_str,
        "SoundSpeed",
    )
    tlt1 = "%s %s" % (
        da.attrs["model"].upper(),
        da.coords["time"].values.astype("datetime64[s]"),
    )
    tlt2 = "at (N%s, E%s)" % (pt1[1], pt1[0])
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    return var1_xr


def atm_density(da, dgh, pt1, save_dir=".", dlim=[0, 0], out_file_prefix=""):
    # da: variable data (temperature, v, u), hand over a xarrayArray with a single variable
    # dgh: height data (gh)
    gam = 1.4
    R0 = 8.3143
    M0 = 0.029
    Rd = R0 / M0

    gh1, pres1 = __get_gh(da, dgh, pt1)

    pres1_pa = pres1 * 100
    da_var = list(da.data_vars)[0]
    temp1 = da.sel(longitude=pt1[0], latitude=pt1[1], method="nearest")[da_var].values
    var1 = pres1_pa * M0 / (R0 * temp1)
    coords = {"height": gh1}
    var1_xr = xr.DataArray(var1, coords, dims=["height"])
    var1_xr.attrs = {"long_name": "Density", "units": "kg/m^3"}
    var1_xr["height"].attrs = {"long_name": "Height", "units": "m"}
    if dlim[0] == 0 and dlim[1] == 0:
        var1_xr.plot(y="height")
    else:
        var1_xr.plot(y="height", xlim=dlim)
    # plt.imshow(temp.T)
    # plt.show()
    datetime_str = apbuilder.utils.convert_datetime_to_str_for_filename(
        da.coords["time"].values.flatten()[0]
    )
    fn = "%s/%s%s_%s_1d_%s.png" % (
        save_dir,
        (out_file_prefix or ""),
        da.attrs["model"].lower(),
        datetime_str,
        "Density",
    )
    tlt1 = "%s %s" % (
        da.attrs["model"].upper(),
        da.coords["time"].values.astype("datetime64[s]"),
    )
    tlt2 = "at (N%s, E%s)" % (pt1[1], pt1[0])
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    return var1_xr


def writebin_ac2dr(da, var, save_dir=".", out_file_prefix=""):
    logger.info(f"Creating {var} 1D binary output file")
    file_path = f"{save_dir}/{out_file_prefix or ''}ac2dr_1d_prof_{var}.bin"
    with open(file_path, "wb") as file:
        ftype = 1
        s = struct.pack("i", ftype)
        file.write(s)
        for i in range(len(da.values)):
            s = struct.pack("d" * 1, da.coords["height"].values[i])
            file.write(s)
            s = struct.pack("d" * 1, da.values[i])
            file.write(s)
        file.close()


def __atm_wind(da, dgh, pt1, plot_name, save_dir=".", wlim=[0, 0], out_file_prefix=""):
    # da: variable data (temperature, v, u), hand over a xarrayArray with a single variable
    # dgh: height data (gh)

    gh1, pres1 = __get_gh(da, dgh, pt1)
    da_var = list(da.data_vars)[0]

    var1 = da.sel(longitude=pt1[0], latitude=pt1[1], method="nearest")[da_var].values
    coords = {"height": gh1}
    var1_xr = xr.DataArray(var1, coords, dims=["height"])
    var1_xr.attrs = {
        "long_name": da[da_var].attrs["long_name"],
        "units": da[da_var].attrs["units"],
    }
    var1_xr["height"].attrs = {"long_name": "Height", "units": "m"}
    if wlim[0] == 0 and wlim[1] == 0:
        var1_xr.plot(y="height")
    else:
        var1_xr.plot(y="height", xlim=wlim)
    # plt.imshow(temp.T)
    # plt.show()
    datetime_str = apbuilder.utils.convert_datetime_to_str_for_filename(
        da.coords["time"].values.flatten()[0]
    )
    fn = "%s/%s%s_%s_1d_%s.png" % (
        save_dir,
        (out_file_prefix or ""),
        da.attrs["model"].lower(),
        datetime_str,
        plot_name,
    )
    tlt1 = "%s %s" % (
        da.attrs["model"].upper(),
        da.coords["time"].values.astype("datetime64[s]"),
    )
    tlt2 = "at (N%s, E%s)" % (pt1[1], pt1[0])
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    return var1_xr


def __get_gh(da, dgh, pt1):
    da_var = list(da.data_vars)[0]
    dgh_var = list(dgh.data_vars)[0]
    # TODO: with weather model HRRR, this next line throws an exception because there is no
    # longitude nor latitude in the indexes of dgh.
    gh0 = dgh.sel(longitude=pt1[0], latitude=pt1[1], method="nearest")[dgh_var].values
    pres0 = dgh[dgh_var].coords["isobaricInhPa"].values
    pres1 = da[da_var].coords["isobaricInhPa"].values
    gh1 = np.interp(pres1, pres0[::-1], gh0[::-1])
    return gh1, pres1
