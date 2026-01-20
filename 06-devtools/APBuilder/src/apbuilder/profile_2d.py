import logging
import math
import numpy as np
import pandas as pd
import pygmt
import struct
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

import apbuilder.utils
from apbuilder.exceptions import HorizontalResolutionError

logger = logging.getLogger(__name__)


def atm_interp(da, dr1, dh1):
    orig_dr = abs(da.coords["distance"].values[1] - da.coords["distance"].values[0])
    if dr1 == 0:
        dr2 = orig_dr
    else:
        dr2 = dr1
    oldh = da.coords["height"].values
    newh = np.arange(min(oldh), max(oldh), dh1)
    oldd = da.coords["distance"].values
    if dr2 > (max(oldd) - min(oldd)):
        logger.warn(
            "The given resolution for 2-D section is larger than the model size. The model extent will be used    for the resolution of 2-D section."
        )
        dr2 = max(oldd) - min(oldd)

    newd = np.arange(min(oldd), max(oldd) + dr2, dr2)
    da2 = da.interp(
        height=newh,
        distance=newd,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    da2 = da2.assign_coords(height=newh - min(newh), distance=newd - min(newd))
    return da2


def atm_v_wind(da, dgh, pt1, pt2, dr0, save_dir=".", wlim=[0, 0], out_file_prefix=""):
    # da: variable data (temperature, v, u), hand over a xarrayArray with a single variable
    # dgh: height data (gh)
    # dr0 (user-defined resolution) is not used in this routine but kept in case of future use
    da_var = list(da.data_vars)[0]
    dgh_var = list(dgh.data_vars)[0]
    gh0 = dgh.sel(longitude=pt1[0], latitude=pt1[1], method="nearest")[dgh_var].values
    pres0 = dgh[dgh_var].coords["isobaricInhPa"].values
    pres1 = da[da_var].coords["isobaricInhPa"].values
    gh1 = np.interp(pres1, pres0[::-1], gh0[::-1])

    orig_dr = abs(
        da[da_var].coords["latitude"].values[1]
        - da[da_var].coords["latitude"].values[0]
    )
    pt1e = list(pt1)
    pt2e = list(pt2)
    for i in range(len(pt1)):
        if pt1[i] >= 180.0:
            pt1e[i] = pt1[i] - 360.0
    for i in range(len(pt2)):
        if pt2[i] >= 180.0:
            pt2e[i] = pt2[i] - 360.0
    str1 = "%f/%f" % (pt1e[0], pt1e[1])
    str2 = "%f/%f" % (pt2e[0], pt2e[1])
    str3 = "%f" % (orig_dr)
    track_df = pygmt.project(
        center=str1,  # Start point of survey line (longitude/latitude)
        endpoint=str2,  # End point of survey line (longitude/latitude)
        generate=str3,  # Output data in steps of 0.1 degrees
    )
    dist1 = track_df["p"]
    for i in range(len(list(track_df["r"]))):
        if track_df["r"][i] < 0:
            # track_df["r"][i] = list(track_df["r"][i])+360.0
            track_df.loc[i, "r"] = track_df["r"][i] + 360.0
    # Extract the elevation at the generated points from the downloaded grid
    # and add it as new column "i" to the pandas.DataFrame
    for i in range(len(pres1)):
        track_df2 = pygmt.grdtrack(
            grid=da[da_var][i, :, :],
            points=track_df,
            newcolname=i,
        )
        if i == 0:
            df_res = pd.DataFrame(track_df2.iloc[:, 3])
        else:
            df_res.insert(i, i, track_df2.iloc[:, 3], True)
    res2d = df_res.to_numpy()
    coords = {"distance": dist1, "height": gh1}
    res2d_xr = xr.DataArray(res2d, coords, dims=["distance", "height"])
    res2d_xr.attrs = {
        "long_name": da[da_var].attrs["long_name"],
        "units": da[da_var].attrs["units"],
    }
    res2d_xr["height"].attrs = {"long_name": "Height", "units": "m"}
    res2d_xr["distance"].attrs = {"long_name": "Distance", "units": "degree"}
    # temp.shape
    surf_xr = da[da_var][1, :, :]
    plt.plot(track_df.iloc[:, 0], track_df.iloc[:, 1], color="blue")
    surf_xr.plot(cmap="hot")
    datetime_str = apbuilder.utils.convert_datetime_to_str_for_filename(
        da.coords["time"].values.flatten()[0]
    )
    fn = "%s/%s%s_%s_2d_Surface_%s.png" % (
        save_dir,
        (out_file_prefix or ""),
        da.attrs["model"].lower(),
        datetime_str,
        "VComponentOfWind",
    )
    tlt1 = "%s %s" % (
        da.attrs["model"].upper(),
        da.coords["time"].values.astype("datetime64[s]"),
    )
    tlt2 = "Surface Level"
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    # plt.show()
    if wlim[0] == 0 and wlim[1] == 0:
        res2d_xr.plot(x="distance", y="height", cmap="hot")
    else:
        res2d_xr.plot(x="distance", y="height", cmap="hot", vmin=wlim[0], vmax=wlim[1])
    # plt.imshow(temp.T)
    fn = "%s/%s%s_%s_2d_%s.png" % (
        save_dir,
        (out_file_prefix or ""),
        da.attrs["model"].lower(),
        datetime_str,
        "VComponentOfWind",
    )
    tlt1 = "%s %s" % (
        da.attrs["model"].upper(),
        da.coords["time"].values.astype("datetime64[s]"),
    )
    tlt2 = "from (N%s, E%s) to (N%s, E%s)" % (pt1[1], pt1[0], pt2[1], pt2[0])
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    # plt.show()
    return res2d_xr


def atm_u_wind(da, dgh, pt1, pt2, dr0, save_dir=".", wlim=[0, 0], out_file_prefix=""):
    # da: variable data (temperature, v, u), hand over a xarrayArray with a single variable
    # dgh: height data (gh)
    # dr0 (user-defined resolution) is not used in this routine but kept in case of future use
    da_var = list(da.data_vars)[0]
    dgh_var = list(dgh.data_vars)[0]
    gh0 = dgh.sel(longitude=pt1[0], latitude=pt1[1], method="nearest")[dgh_var].values
    pres0 = dgh[dgh_var].coords["isobaricInhPa"].values
    pres1 = da[da_var].coords["isobaricInhPa"].values
    gh1 = np.interp(pres1, pres0[::-1], gh0[::-1])

    orig_dr = abs(
        da[da_var].coords["latitude"].values[1]
        - da[da_var].coords["latitude"].values[0]
    )
    pt1e = list(pt1)
    pt2e = list(pt2)
    for i in range(len(pt1)):
        if pt1[i] >= 180.0:
            pt1e[i] = pt1[i] - 360.0
    for i in range(len(pt2)):
        if pt2[i] >= 180.0:
            pt2e[i] = pt2[i] - 360.0
    str1 = "%f/%f" % (pt1e[0], pt1e[1])
    str2 = "%f/%f" % (pt2e[0], pt2e[1])
    str3 = "%f" % (orig_dr)
    track_df = pygmt.project(
        center=str1,  # Start point of survey line (longitude/latitude)
        endpoint=str2,  # End point of survey line (longitude/latitude)
        generate=str3,  # Output data in steps of 0.1 degrees
    )
    dist1 = track_df["p"]
    for i in range(len(list(track_df["r"]))):
        if track_df["r"][i] < 0:
            # track_df["r"][i] = list(track_df["r"][i])+360.0
            track_df.loc[i, "r"] = track_df["r"][i] + 360.0
    # Extract the elevation at the generated points from the downloaded grid
    # and add it as new column "i" to the pandas.DataFrame
    for i in range(len(pres1)):
        track_df2 = pygmt.grdtrack(
            grid=da[da_var][i, :, :],
            points=track_df,
            newcolname=i,
        )
        if i == 0:
            df_res = pd.DataFrame(track_df2.iloc[:, 3])
        else:
            df_res.insert(i, i, track_df2.iloc[:, 3], True)
    res2d = df_res.to_numpy()
    coords = {"distance": dist1, "height": gh1}
    res2d_xr = xr.DataArray(res2d, coords, dims=["distance", "height"])
    res2d_xr.attrs = {
        "long_name": da[da_var].attrs["long_name"],
        "units": da[da_var].attrs["units"],
    }
    res2d_xr["height"].attrs = {"long_name": "Height", "units": "m"}
    res2d_xr["distance"].attrs = {"long_name": "Distance", "units": "degree"}
    # temp.shape
    surf_xr = da[da_var][1, :, :]
    plt.plot(track_df.iloc[:, 0], track_df.iloc[:, 1], color="blue")
    surf_xr.plot(cmap="hot")
    datetime_str = apbuilder.utils.convert_datetime_to_str_for_filename(
        da.coords["time"].values.flatten()[0]
    )
    fn = "%s/%s%s_%s_2d_Surface_%s.png" % (
        save_dir,
        (out_file_prefix or ""),
        da.attrs["model"].lower(),
        datetime_str,
        "UComponentOfWind",
    )
    tlt1 = "%s %s" % (
        da.attrs["model"].upper(),
        da.coords["time"].values.astype("datetime64[s]"),
    )
    tlt2 = "Surface Level"
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    # plt.show()
    if wlim[0] == 0 and wlim[1] == 0:
        res2d_xr.plot(x="distance", y="height", cmap="hot")
    else:
        res2d_xr.plot(x="distance", y="height", cmap="hot", vmin=wlim[0], vmax=wlim[1])
    # plt.imshow(temp.T)
    fn = "%s/%s%s_%s_2d_%s.png" % (
        save_dir,
        (out_file_prefix or ""),
        da.attrs["model"].lower(),
        datetime_str,
        "UComponentOfWind",
    )
    tlt1 = "%s %s" % (
        da.attrs["model"].upper(),
        da.coords["time"].values.astype("datetime64[s]"),
    )
    tlt2 = "from (N%s, E%s) to (N%s, E%s)" % (pt1[1], pt1[0], pt2[1], pt2[0])
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    # plt.show()
    return res2d_xr


def atm_speed(da, dgh, pt1, pt2, dr0, save_dir=".", clim=[0, 0], out_file_prefix=""):
    # da: variable data (temperature, v, u), hand over a xarrayArray with a single variable
    # dgh: height data (gh)
    # dr0 (user-defined resolution) is not used in this routine but kept in case of future use
    gam = 1.4
    R0 = 8.3143
    M0 = 0.029
    Rd = R0 / M0
    da_var = list(da.data_vars)[0]
    dgh_var = list(dgh.data_vars)[0]
    gh0 = dgh.sel(longitude=pt1[0], latitude=pt1[1], method="nearest")[dgh_var].values
    pres0 = dgh[dgh_var].coords["isobaricInhPa"].values
    pres1 = da[da_var].coords["isobaricInhPa"].values
    gh1 = np.interp(pres1, pres0[::-1], gh0[::-1])

    orig_dr = abs(
        da[da_var].coords["latitude"].values[1]
        - da[da_var].coords["latitude"].values[0]
    )
    pt1e = list(pt1)
    pt2e = list(pt2)
    for i in range(len(pt1)):
        if pt1[i] >= 180.0:
            pt1e[i] = pt1[i] - 360.0
    for i in range(len(pt2)):
        if pt2[i] >= 180.0:
            pt2e[i] = pt2[i] - 360.0
    str1 = "%f/%f" % (pt1e[0], pt1e[1])
    str2 = "%f/%f" % (pt2e[0], pt2e[1])
    str3 = "%f" % (orig_dr)
    track_df = pygmt.project(
        center=str1,  # Start point of survey line (longitude/latitude)
        endpoint=str2,  # End point of survey line (longitude/latitude)
        generate=str3,  # Output data in steps of 0.1 degrees
    )
    dist1 = track_df["p"]
    for i in range(len(list(track_df["r"]))):
        if track_df["r"][i] < 0:
            # track_df["r"][i] = list(track_df["r"][i])+360.0
            track_df.loc[i, "r"] = track_df["r"][i] + 360.0
    # Extract the elevation at the generated points from the downloaded grid
    # and add it as new column "i" to the pandas.DataFrame
    for i in range(len(pres1)):
        track_df2 = pygmt.grdtrack(
            grid=da[da_var][i, :, :],
            points=track_df,
            newcolname=i,
        )
        if i == 0:
            df_res = pd.DataFrame((gam * Rd * track_df2.iloc[:, 3]) ** (1 / 2))
        else:
            df_res.insert(i, i, (gam * Rd * track_df2.iloc[:, 3]) ** (1 / 2), True)
    res2d = df_res.to_numpy()
    coords = {"distance": dist1, "height": gh1}
    res2d_xr = xr.DataArray(res2d, coords, dims=["distance", "height"])
    res2d_xr.attrs = {"long_name": "SoundSpeed", "units": "m/s"}
    res2d_xr["height"].attrs = {"long_name": "Height", "units": "m"}
    res2d_xr["distance"].attrs = {"long_name": "Distance", "units": "degree"}
    # temp.shape
    surf_xr = da[da_var][1, :, :]
    plt.plot(track_df.iloc[:, 0], track_df.iloc[:, 1], color="blue")
    surf_xr.plot(cmap="hot")
    datetime_str = apbuilder.utils.convert_datetime_to_str_for_filename(
        da.coords["time"].values.flatten()[0]
    )
    fn = "%s/%s%s_%s_2d_Surface_%s.png" % (
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
    tlt2 = "Surface Level"
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    # plt.show()
    if clim[0] == 0 and clim[1] == 0:
        res2d_xr.plot(x="distance", y="height", cmap="hot")
    else:
        res2d_xr.plot(x="distance", y="height", cmap="hot", vmin=clim[0], vmax=clim[1])
    # plt.imshow(temp.T)
    fn = "%s/%s%s_%s_2d_%s.png" % (
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
    tlt2 = "from (N%s, E%s) to (N%s, E%s)" % (pt1[1], pt1[0], pt2[1], pt2[0])
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    # plt.show()
    return res2d_xr


def atm_density(da, dgh, pt1, pt2, dr0, save_dir=".", dlim=[0, 0], out_file_prefix=""):
    # da: variable data (temperature, v, u), hand over a xarrayArray with a single variable
    # dgh: height data (gh)
    # dr0 (user-defined resolution) is not used in this routine but kept in case of future use
    gam = 1.4
    R0 = 8.3143
    M0 = 0.029
    Rd = R0 / M0
    da_var = list(da.data_vars)[0]
    dgh_var = list(dgh.data_vars)[0]
    gh0 = dgh.sel(longitude=pt1[0], latitude=pt1[1], method="nearest")[dgh_var].values
    pres0 = dgh[dgh_var].coords["isobaricInhPa"].values
    pres1 = da[da_var].coords["isobaricInhPa"].values
    gh1 = np.interp(pres1, pres0[::-1], gh0[::-1])

    orig_dr = abs(
        da[da_var].coords["latitude"].values[1]
        - da[da_var].coords["latitude"].values[0]
    )
    pt1e = list(pt1)
    pt2e = list(pt2)
    for i in range(len(pt1)):
        if pt1[i] >= 180.0:
            pt1e[i] = pt1[i] - 360.0
    for i in range(len(pt2)):
        if pt2[i] >= 180.0:
            pt2e[i] = pt2[i] - 360.0
    str1 = "%f/%f" % (pt1e[0], pt1e[1])
    str2 = "%f/%f" % (pt2e[0], pt2e[1])
    str3 = "%f" % (orig_dr)
    track_df = pygmt.project(
        center=str1,  # Start point of survey line (longitude/latitude)
        endpoint=str2,  # End point of survey line (longitude/latitude)
        generate=str3,  # Output data in steps of 0.1 degrees
    )
    dist1 = track_df["p"]
    for i in range(len(list(track_df["r"]))):
        if track_df["r"][i] < 0:
            # track_df["r"][i] = list(track_df["r"][i])+360.0
            track_df.loc[i, "r"] = track_df["r"][i] + 360.0
    # Extract the elevation at the generated points from the downloaded grid
    # and add it as new column "i" to the pandas.DataFrame
    for i in range(len(pres1)):
        track_df2 = pygmt.grdtrack(
            grid=da[da_var][i, :, :],
            points=track_df,
            newcolname=i,
        )
        if i == 0:
            df_res = pd.DataFrame(pres1[i] * 100 * M0 / (R0 * track_df2.iloc[:, 3]))
        else:
            df_res.insert(i, i, pres1[i] * 100 * M0 / (R0 * track_df2.iloc[:, 3]), True)
    res2d = df_res.to_numpy()
    coords = {"distance": dist1, "height": gh1}
    res2d_xr = xr.DataArray(res2d, coords, dims=["distance", "height"])
    res2d_xr.attrs = {"long_name": "Density", "units": "kg/m**3"}
    res2d_xr["height"].attrs = {"long_name": "Height", "units": "m"}
    res2d_xr["distance"].attrs = {"long_name": "Distance", "units": "degree"}
    # temp.shape
    surf_xr = da[da_var][1, :, :]
    plt.plot(track_df.iloc[:, 0], track_df.iloc[:, 1], color="blue")
    surf_xr.plot(cmap="hot")
    datetime_str = apbuilder.utils.convert_datetime_to_str_for_filename(
        da.coords["time"].values.flatten()[0]
    )
    fn = "%s/%s%s_%s_2d_Surface_%s.png" % (
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
    tlt2 = "Surface Level"
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    # plt.show()
    if dlim[0] == 0 and dlim[1] == 0:
        res2d_xr.plot(x="distance", y="height", cmap="hot")
    else:
        res2d_xr.plot(x="distance", y="height", cmap="hot", vmin=dlim[0], vmax=dlim[1])
    # plt.imshow(temp.T)
    fn = "%s/%s%s_%s_2d_%s.png" % (
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
    tlt2 = "from (N%s, E%s) to (N%s, E%s)" % (pt1[1], pt1[0], pt2[1], pt2[0])
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    # plt.show()
    return res2d_xr


def atm_r_wind(da, u1, v1, pt1, pt2, save_dir=".", wlim=[0, 0], out_file_prefix=""):
    theta = apbuilder.utils.get_azimuth(pt1, pt2)
    var1 = u1.copy()
    # unit vector to r (x, y)
    rv = [math.sin(theta * math.pi / 180), math.cos(theta * math.pi / 180)]
    # project wind, rw
    var1.values = u1.values * rv[0] + v1.values * rv[1]
    var1.attrs["long_name"] = "R component of wind"
    if wlim[0] == 0 and wlim[1] == 0:
        var1.plot(x="distance", y="height", cmap="hot")
    else:
        var1.plot(x="distance", y="height", cmap="hot", vmin=wlim[0], vmax=wlim[1])
    # plt.imshow(temp.T)
    datetime_str = apbuilder.utils.convert_datetime_to_str_for_filename(
        da.coords["time"].values.flatten()[0]
    )
    fn = "%s/%s%s_%s_2d_%s.png" % (
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
    tlt2 = "from (N%s, E%s) to (N%s, E%s)" % (pt1[1], pt1[0], pt2[1], pt2[0])
    plt.title("%s\n%s" % (tlt1, tlt2))
    plt.savefig(fn)
    plt.close()
    return var1


def writebin_ac2dr(da, var, save_dir=".", out_file_prefix=""):
    logger.info(f"Creating {var} 2D binary output file")
    file_path = f"{save_dir}/{out_file_prefix or ''}ac2dr_2d_section_{var}.bin"
    with open(file_path, "wb") as file:
        ftype = 2
        try:
            dx = da.coords["distance"].values[1] - da.coords["distance"].values[0]
        except IndexError as e:
            raise HorizontalResolutionError(
                f"The horizontal resolution of 2D slice in degree must be less or equal to the distance between the latitude start and end points."
            )
        dy = da.coords["height"].values[1] - da.coords["height"].values[0]
        nx = len(da.coords["distance"].values)
        ny = len(da.coords["height"].values)
        file.write(struct.pack("i", ftype))
        file.write(struct.pack("i", nx))
        file.write(struct.pack("i", ny))
        file.write(struct.pack("d", dx))
        file.write(struct.pack("d", dy))
        for i in range(nx):
            for j in range(ny):
                file.write(struct.pack("d", da.values[i, j]))
        file.close()
