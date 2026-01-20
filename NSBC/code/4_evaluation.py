'''
This figure is used to generate the figures.
step 1. Figure 1. mean, std,skewness,kurtosis
step 2. Figure 2. Q33,Q66,Q99
step 3. Figure 3. 1-day max, 3-day max, 5-day max
step 4. Figure 4. annual mean plot
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
import torch
import numpy as np
import torch.nn as nn
import torch, tqdm, copy, os, ssl, h5py
import torch.nn.functional as F
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import pearsonr
from cartopy.io import shapereader
import matplotlib.colors
from cartopy import crs as ccrs



def RMSE(x, y):
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    return np.sqrt(np.mean((y - x) ** 2))


def correlation(x, y):
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    corr = pearsonr(x, y)[0]
    return corr


#step 0. preparation load datasets for figure plot
mask_numpy = np.load("./data/mask.npy")
plat = np.load('./data/Precip_Latitude.npy')[1:]
plon = np.load('./data/Precip_Longitude.npy')
lon_p, lat_p = np.meshgrid(plon, plat)
extent = [-135, -56, 15.5, 52]


train_y_obs = np.load('./data/train_temp_gfdl_interp.npy')
test_y_obs = np.load('./data/test_temp_gfdl_interp.npy')
future_y_obs = np.load('./data/future_temp_gfdl_interp.npy')
all_y_obs = np.concatenate((train_y_obs,test_y_obs,future_y_obs),axis = 0)

train_y_e3sm = np.load('./data/train_temp_e3sm_interp.npy')
test_y_e3sm = np.load('./data/test_temp_e3sm_interp.npy')
future_y_e3sm = np.load('./data/future_temp_e3sm_interp.npy')
all_y_e3sm = np.concatenate((train_y_e3sm,test_y_e3sm,future_y_e3sm),axis = 0)


preds = np.concatenate((np.load('./data/tas_test_nsbc.npy'),np.load('./data/tas_future_nsbc.npy')),axis=0)
nP4GCM = all_y_e3sm[72*365:,:,:]
nP4Obser =all_y_obs[72*365:,:,:]

nP4GCM = np.array(nP4GCM).squeeze()
nP4Obser = np.array(nP4Obser).squeeze()
CESM2_Precip = copy.deepcopy(nP4GCM)
Obs_Precip = copy.deepcopy(nP4Obser)

qm= np.concatenate((np.load('./data/gcm_2022_2060_correct_edcdfm.npy'),np.load('./data/gcm_2061_2099_correct_edcdfm.npy')),axis = 0)
QM_Precip = copy.deepcopy(qm)

#step 1, figure 1 plot for the mean, std, skewness, kurtosis.

mean_obs = np.mean(Obs_Precip, axis=0)
mean_obs[mask_numpy == 0] = np.nan
std_obs = np.std(Obs_Precip, dtype=np.float64, axis=0)
std_obs[mask_numpy == 0] = np.nan
skew_obs = skew(Obs_Precip)
skew_obs[mask_numpy == 0] = np.nan
kur_obs = kurtosis(Obs_Precip)
kur_obs[mask_numpy == 0] = np.nan


mean_cesm = np.mean(CESM2_Precip, axis=0)
mean_cesm[mask_numpy == 0] = np.nan
mean_cesm_r = correlation(mean_obs, mean_cesm)
mean_cesm_rmse = RMSE(mean_obs, mean_cesm)
std_cesm = np.std(CESM2_Precip, dtype=np.float64, axis=0)
std_cesm[mask_numpy == 0] = np.nan
std_cesm_r = correlation(std_obs, std_cesm)
std_cesm_rmse = RMSE(std_obs, std_cesm)
skew_cesm = skew(CESM2_Precip)
skew_cesm[mask_numpy == 0] = np.nan
skew_cesm_r = correlation(skew_obs, skew_cesm)
skew_cesm_rmse = RMSE(skew_obs, skew_cesm)
kur_cesm = kurtosis(CESM2_Precip)
kur_cesm[mask_numpy == 0] = np.nan
kur_cesm_r = correlation(kur_obs, kur_cesm)
kur_cesm_rmse = RMSE(kur_obs, kur_cesm)

RADA_Precip = preds
mean_rada = np.mean(RADA_Precip, axis=0).squeeze()
mean_rada[mask_numpy == 0] = np.nan
mean_rada_r = correlation(mean_obs, mean_rada)
mean_rada_rmse = RMSE(mean_obs, mean_rada)

std_rada = np.std(RADA_Precip, dtype=np.float64, axis=0)
std_rada[mask_numpy == 0] = np.nan
std_rada_r = correlation(std_obs, std_rada)
std_rada_rmse = RMSE(std_obs, std_rada)

skew_rada = skew(RADA_Precip)
skew_rada[mask_numpy == 0] = np.nan
skew_obs[np.isnan(skew_rada)] = np.nan
skew_rada_r = correlation(skew_obs, skew_rada)
skew_rada_rmse = RMSE(skew_obs, skew_rada)

kur_rada = kurtosis(RADA_Precip)
kur_rada[mask_numpy == 0] = np.nan
kur_obs[np.isnan(kur_rada)] = np.nan
kur_rada_r = correlation(kur_obs, kur_rada)
kur_rada_rmse = RMSE(kur_obs, kur_rada)


mean_qm = np.mean(QM_Precip, axis=0).squeeze()
mean_qm[mask_numpy == 0] = np.nan
mean_qm_r = correlation(mean_obs, mean_qm)
mean_qm_rmse = RMSE(mean_obs, mean_qm)

std_qm = np.std(QM_Precip, dtype=np.float64, axis=0)
std_qm[mask_numpy == 0] = np.nan
std_qm_r = correlation(std_obs, std_qm)
std_qm_rmse = RMSE(std_obs, std_qm)

skew_qm = skew(QM_Precip)
skew_qm[mask_numpy == 0] = np.nan
skew_obs[np.isnan(skew_qm)] = np.nan
skew_qm_r = correlation(skew_obs, skew_qm)
skew_qm_rmse = RMSE(skew_obs, skew_qm)

kur_qm = kurtosis(QM_Precip)
kur_qm[mask_numpy == 0] = np.nan
kur_obs[np.isnan(kur_qm)] = np.nan
kur_qm_r = correlation(kur_obs, kur_qm)
kur_qm_rmse = RMSE(kur_obs, kur_qm)

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

extent = [-135, -56, 15.5, 52]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
                                                           ["#FFFFFF", "#14FFFF", "#FFB61F", "#FF5500", "#AB0000"])
blank = np.zeros((27, 62))
fname = './data/cb_2018_us_state_5m.shp'
state_shapes = list(shapereader.Reader(fname).geometries())
array_list = [mean_obs, mean_cesm, mean_rada,mean_qm,std_obs, std_cesm, std_rada, std_qm, skew_obs, skew_cesm, skew_rada,skew_qm, kur_obs,
              kur_cesm, kur_rada,kur_qm]
r_list = [mean_cesm_r, std_cesm_r, skew_cesm_r, kur_cesm_r]
r_list2 = [mean_rada_r, std_rada_r, skew_rada_r, kur_rada_r]
r_list3 = [mean_qm_r, std_qm_r, skew_qm_r, kur_qm_r]


rmse_list = [mean_cesm_rmse, std_cesm_rmse, skew_cesm_rmse, kur_cesm_rmse]
rmse_list2 = [mean_rada_rmse, std_rada_rmse, skew_rada_rmse, kur_rada_rmse]
rmse_list3 = [mean_qm_rmse, std_qm_rmse, skew_qm_rmse, kur_qm_rmse]


labels = ['Observation', 'E3SM', 'NSBC','EDCDFm']
cb_labels = ['Mean', 'Std', 'Skew', 'Kurt']
y_labels = ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis']
max_values = ['20', '15', '1', '1']
min_values = ['4', '4', '-1', '-2']
count = 0
extent = [-125, -67, 24, 47]
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(4, 49, hspace=0.17, wspace=0.2)
for i in range(4):
    for j in range(4):
        axs = fig.add_subplot(gs[i, (j * 12):(j + 1) * 12], projection=ccrs.PlateCarree())
        cs = axs.pcolormesh(lon_p[:, :], lat_p[:, :], array_list[count], cmap=cmap, vmin=min_values[i],
                            vmax=max_values[i], transform=ccrs.PlateCarree())
        axs.set_extent(extent)
        axs.add_geometries(state_shapes, ccrs.PlateCarree(), edgecolor='grey', facecolor='none', alpha=1)
        if j == 0:
            axs.text(-0.01, 0.5, y_labels[i], va='bottom', ha='center', rotation='vertical', rotation_mode='anchor',
                     transform=axs.transAxes, fontsize=13)
        if j == 1:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if j == 2:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list2[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list2[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if j == 3:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list3[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list3[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if j == 4:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list4[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list4[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if i == 0:
            axs.set_title(labels[j], fontsize=14)
        count += 1
    cb = plt.colorbar(cs, cax=plt.subplot(gs[i, 48:]), orientation='vertical', fraction=0.05, pad=0.01)
    cb.set_label(cb_labels[i])

plt.tight_layout()
out_path = ('./plots/figure1.png')
plt.savefig(out_path,dpi=100)


## step 2. figure 2. Q33,Q66.Q99
q33_preds = np.percentile(preds,33,axis=0)
q33_preds[mask_numpy == 0] = np.nan
q66_preds = np.percentile(preds,66,axis=0)
q66_preds[mask_numpy == 0] = np.nan
q99_preds = np.percentile(preds,99,axis=0)
q99_preds[mask_numpy == 0] = np.nan


q33_nP4GCM = np.percentile(nP4GCM,33,axis=0)
q33_nP4GCM[mask_numpy == 0] = np.nan
q66_nP4GCM = np.percentile(nP4GCM,66,axis=0)
q66_nP4GCM[mask_numpy == 0] = np.nan
q99_nP4GCM = np.percentile(nP4GCM,99,axis=0)
q99_nP4GCM[mask_numpy == 0] = np.nan



q33_nP4Obser = np.percentile(nP4Obser,33,axis=0)
q33_nP4Obser[mask_numpy == 0] = np.nan
q66_nP4Obser = np.percentile(nP4Obser,66,axis=0)
q66_nP4Obser[mask_numpy == 0] = np.nan
q99_nP4Obser = np.percentile(nP4Obser,99,axis=0)
q99_nP4Obser[mask_numpy == 0] = np.nan


q33_qm = np.percentile(qm,33,axis=0)
q33_qm[mask_numpy == 0] = np.nan
q66_qm = np.percentile(qm,66,axis=0)
q66_qm[mask_numpy == 0] = np.nan
q99_qm = np.percentile(qm,99,axis=0)
q99_qm[mask_numpy == 0] = np.nan



array_list = [q33_nP4Obser,q33_nP4GCM,q33_preds,q33_qm, q66_nP4Obser,q66_nP4GCM,q66_preds,q66_qm,q99_nP4Obser,q99_nP4GCM,q99_preds,q99_qm]


mean_cesm_r_33 = correlation(q33_nP4GCM, q33_nP4Obser)
mean_rada_r_33= correlation(q33_preds, q33_nP4Obser)
mean_qm_r_33 = correlation(q33_qm, q33_nP4Obser)


mean_cesm_r_66= correlation(q66_nP4GCM, q66_nP4Obser)
mean_rada_r_66= correlation(q66_preds, q66_nP4Obser)
mean_qm_r_66= correlation(q66_qm, q66_nP4Obser)



mean_cesm_r_99=correlation(q99_nP4GCM, q99_nP4Obser)
mean_rada_r_99=correlation(q99_preds, q99_nP4Obser)
mean_qm_r_99=correlation(q99_qm, q99_nP4Obser)

mean_cesm_rmse_33=RMSE(q33_nP4GCM, q33_nP4Obser)
mean_rada_rmse_33=RMSE(q33_preds, q33_nP4Obser)
mean_qm_rmse_33=RMSE(q33_qm, q33_nP4Obser)


mean_cesm_rmse_66=RMSE(q66_nP4GCM, q66_nP4Obser)
mean_rada_rmse_66=RMSE(q66_preds, q66_nP4Obser)
mean_qm_rmse_66=RMSE(q66_qm, q66_nP4Obser)


mean_cesm_rmse_99=RMSE(q99_nP4GCM, q99_nP4Obser)
mean_rada_rmse_99=RMSE(q99_preds, q99_nP4Obser)
mean_qm_rmse_99=RMSE(q99_qm, q99_nP4Obser)



r_list = [mean_cesm_r_33, mean_cesm_r_66, mean_cesm_r_99]
r_list2 = [mean_rada_r_33, mean_rada_r_66, mean_rada_r_99]
r_list3 = [mean_qm_r_33, mean_qm_r_66, mean_qm_r_99]


rmse_list = [mean_cesm_rmse_33, mean_cesm_rmse_66, mean_cesm_rmse_99]
rmse_list2 = [mean_rada_rmse_33, mean_rada_rmse_66, mean_rada_rmse_99]
rmse_list3 = [mean_qm_rmse_33, mean_qm_rmse_66, mean_qm_rmse_99]


labels = ['Observation', 'E3SM', 'NSBC','EDCDFm']
cb_labels = ['Q33', 'Q66', 'Q99']
y_labels = ['Q33', 'Q66', 'Q99']
max_values = ['16', '26', '40']
min_values = ['-3', '0', '8']
count = 0
extent = [-125, -67, 24, 47]
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(4, 49, hspace=0.17, wspace=0.2)
for i in range(3):
    for j in range(4):
        axs = fig.add_subplot(gs[i, (j * 12):(j + 1) * 12], projection=ccrs.PlateCarree())
        cs = axs.pcolormesh(lon_p[:, :], lat_p[:, :], array_list[count], cmap=cmap, vmin=min_values[i],
                            vmax=max_values[i], transform=ccrs.PlateCarree())
        axs.set_extent(extent)
        axs.add_geometries(state_shapes, ccrs.PlateCarree(), edgecolor='grey', facecolor='none', alpha=1)
        if j == 0:
            axs.text(-0.01, 0.5, y_labels[i], va='bottom', ha='center', rotation='vertical', rotation_mode='anchor',
                     transform=axs.transAxes, fontsize=13)
        if j == 1:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if j == 2:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list2[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list2[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if j == 3:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list3[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list3[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if j == 4:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list4[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list4[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if i == 0:
            axs.set_title(labels[j], fontsize=14)
        count += 1
    cb = plt.colorbar(cs, cax=plt.subplot(gs[i, 48:]), orientation='vertical', fraction=0.05, pad=0.01)
    cb.set_label(cb_labels[i])

plt.tight_layout()
out_path = ('./plots/figure2.png')
plt.savefig(out_path,dpi = 100)

#step3. figure 3 plot,1-day max, 3-day max, 5-day max
def Intensity(data):
    intensity = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            data2 = data[:, i, j]
            data_no_nan = data2[~np.isnan(data2)]
            N = len(data2)
            N2 = len(data_no_nan)
            if ((N2 / N) < 0.8):
                intensity[i, j] = np.nan
            else:
                SS = data2[data2 > 1]
                intensity[i, j] = np.nanmean(SS)
    return intensity


def day1max(data):
    day_max = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            day_max[i, j] = np.nanmax(data[:, i, j])
    return day_max


def day3max(data):
    day_max = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            data3day = np.convolve(data[:, i, j], np.ones(3, dtype=int), 'valid')
            day_max[i, j] = np.nanmax(data3day)/3
    return day_max


def day5max(data):
    day_max = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            data3day = np.convolve(data[:, i, j], np.ones(5, dtype=int), 'valid')
            day_max[i, j] = np.nanmax(data3day)/5
    return day_max

mask = np.repeat(mask_numpy[np.newaxis,:,:],preds.shape[0],axis=0)
nP4Obser[mask ==0] = np.nan
nP4GCM[mask ==0] = np.nan
preds[mask ==0] = np.nan
qm[mask ==0] = np.nan

Obs_Precip=nP4Obser
CESM2_Precip = nP4GCM
RADA_Data = preds
QM_Data = qm


intensity_obs = Intensity(Obs_Precip)
intensity_cesm = Intensity(CESM2_Precip)
intensity_cesm_r = correlation(intensity_obs, intensity_cesm)
intensity_cesm_rmse = RMSE(intensity_obs, intensity_cesm)

day1max_obs = day1max(Obs_Precip)
day1max_cesm = day1max(CESM2_Precip)
day1max_cesm_r = correlation(day1max_obs, day1max_cesm)
day1max_cesm_rmse = RMSE(day1max_obs, day1max_cesm)

day3max_obs = day3max(Obs_Precip)
day3max_cesm = day3max(CESM2_Precip)
day3max_cesm_r = correlation(day3max_obs, day3max_cesm)
day3max_cesm_rmse = RMSE(day3max_obs, day3max_cesm)

day5max_obs = day5max(Obs_Precip)
day5max_cesm = day5max(CESM2_Precip)
day5max_cesm_r = correlation(day5max_obs, day5max_cesm)
day5max_cesm_rmse = RMSE(day5max_obs, day5max_cesm)


lon_p, lat_p = np.meshgrid(plon, plat)
RADA_Precip = RADA_Data
intensity_rada = Intensity(RADA_Precip)
intensity_rada [mask_numpy ==0] = np.nan
intensity_rada_r = correlation(intensity_obs, intensity_rada)
intensity_rada_rmse = RMSE(intensity_obs, intensity_rada)
day1max_rada = day1max(RADA_Precip)
day1max_rada_r = correlation(day1max_obs, day1max_rada)
day1max_rada_rmse = RMSE(day1max_obs, day1max_rada)
day3max_rada = day3max(RADA_Precip)
day3max_rada_r = correlation(day3max_obs, day3max_rada)
day3max_rada_rmse = RMSE(day3max_obs, day3max_rada)
day5max_rada = day5max(RADA_Precip)
day5max_rada_r = correlation(day5max_obs, day5max_rada)
day5max_rada_rmse = RMSE(day5max_obs, day5max_rada)

QM_Precip = QM_Data
intensity_qm = Intensity(QM_Precip)
intensity_qm [mask_numpy ==0] = np.nan
intensity_qm_r = correlation(intensity_obs, intensity_qm)
intensity_qm_rmse = RMSE(intensity_obs, intensity_qm)
day1max_qm = day1max(QM_Precip)
day1max_qm_r = correlation(day1max_obs, day1max_qm)
day1max_qm_rmse = RMSE(day1max_obs, day1max_qm)
day3max_qm = day3max(QM_Precip)
day3max_qm_r = correlation(day3max_obs, day3max_qm)
day3max_qm_rmse = RMSE(day3max_obs, day3max_qm)
day5max_qm = day5max(QM_Precip)
day5max_qm_r = correlation(day5max_obs, day5max_qm)
day5max_qm_rmse = RMSE(day5max_obs, day5max_qm)

extent = [-135, -56, 15.5, 52]
import matplotlib.colors
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
                                                           ["#FFFFFF", "#14FFFF", "#FFB61F", "#FF5500", "#AB0000"])
from cartopy.io import shapereader
blank = np.zeros((27, 62))

fname = './data/cb_2018_us_state_5m.shp'
state_shapes = list(shapereader.Reader(fname).geometries())
array_list = [intensity_obs, intensity_cesm, intensity_rada, intensity_qm, day1max_obs, day1max_cesm, day1max_rada,day1max_qm,
              day3max_obs, day3max_cesm, day3max_rada,day3max_qm, day5max_obs, day5max_cesm, day5max_rada,day5max_qm]
r_list = [intensity_cesm_r, day1max_cesm_r, day3max_cesm_r, day5max_cesm_r]
r_list2 = [intensity_rada_r, day1max_rada_r, day3max_rada_r, day5max_rada_r]
r_list3 = [intensity_qm_r, day1max_qm_r, day3max_qm_r, day5max_qm_r]

rmse_list = [intensity_cesm_rmse, day1max_cesm_rmse, day3max_cesm_rmse, day5max_cesm_rmse]
rmse_list2 = [intensity_rada_rmse, day1max_rada_rmse, day3max_rada_rmse, day5max_rada_rmse]
rmse_list3 = [intensity_qm_rmse, day1max_qm_rmse, day3max_qm_rmse, day5max_qm_rmse]

labels = ['Observation', 'E3SM', 'NSBC','EDCDFm']
cb_labels = ['Int. (C/d)', r'$max_{1}$ (${\times}$C)', r'$max_{3}$ (${\times}$C)',
             r'$max_{5}$ (${\times}$C)']
y_labels = ['Average Intensity', '1-day max', '3-day max', '5-day max']
max_values = ['20', '55', '52', '50']
min_values = ['1.0', '20', '20', '20']
count = 0
extent = [-125, -67, 24, 47]
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(4, 49, hspace=0.17, wspace=0.2)
for i in range(4):
    for j in range(4):
        axs = fig.add_subplot(gs[i, (j * 12):(j + 1) * 12], projection=ccrs.PlateCarree())
        cs = axs.pcolormesh(lon_p[:, :], lat_p[:, :], array_list[count], cmap=cmap, vmin=min_values[i],
                            vmax=max_values[i], transform=ccrs.PlateCarree())
        axs.set_extent(extent)
        axs.add_geometries(state_shapes, ccrs.PlateCarree(), edgecolor='grey', facecolor='none', alpha=1)
        if j == 0:
            axs.text(-0.01, 0.5, y_labels[i], va='bottom', ha='center', rotation='vertical', rotation_mode='anchor',
                     transform=axs.transAxes, fontsize=13)
        if j == 1:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if j == 2:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list2[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list2[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if j == 3:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list3[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list3[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if j == 4:
            axs.text(0.025, 0.2, 'r' + r'$\approx$' + str(round(r_list4[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
            axs.text(0.025, 0.1, 'RMSE' + r'$\approx$' + str(round(rmse_list4[i], 2)), horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes)
        if i == 0:
            axs.set_title(labels[j], fontsize=14)
        count += 1
    cb = plt.colorbar(cs, cax=plt.subplot(gs[i, 48:]), orientation='vertical', fraction=0.05, pad=0.01)
    cb.set_label(cb_labels[i])

plt.tight_layout()
out_path = ('./plots/figure3.png')
plt.savefig(out_path,dpi = 100)






#step 4. Figure 4. annual mean plot

nP4GCM = np.array(nP4GCM).squeeze()
nP4Obser = np.array(nP4Obser).squeeze()

mask = np.repeat(mask_numpy[np.newaxis,:,:],preds.shape[0],axis=0)
preds [mask ==0] = 0
nP4GCM [mask ==0] = 0
nP4Obser [mask ==0] = 0
qm[mask ==0] = 0
add = np.zeros((2,))+np.nan

mean_obs = np.nanmean(np.nanmean(nP4Obser,axis = 1),axis = 1)
mean_cesm = np.nanmean(np.nanmean(nP4GCM,axis = 1),axis = 1)
mean_unet = np.nanmean(np.nanmean(preds,axis = 1),axis = 1)
mean_qm = np.nanmean(np.nanmean(qm,axis = 1),axis = 1)


def monthly_mean(day,data):
    a = data.reshape(-1,365)
    b = np.zeros((a.shape[0],12))
    for y in range(a.shape[0]):
        for m in range(12):
            b[y,m] = np.nanmean(a[y,day[m]:day[m+1]])
    return b


plat = np.load('./data/Precip_Latitude.npy')[1:]
plon = np.load('./data/Precip_Longitude.npy')
lon_p, lat_p = np.meshgrid(plon, plat)
extent = [-135, -56, 15.5, 52]



mean_obs= np.nanmean(np.nanmean(nP4Obser,axis = 1),axis = 1)
mean_cesm= np.nanmean(np.nanmean(nP4GCM,axis = 1),axis = 1)
mean_unet= np.nanmean(np.nanmean(preds,axis = 1),axis = 1)
mean_qm= np.nanmean(np.nanmean(qm,axis = 1),axis = 1)



fig = plt.figure(figsize=(10, 4))
obs_y = np.nanmean(mean_obs.reshape(-1,365),axis=1)
cesm_y = np.nanmean(mean_cesm.reshape(-1,365),axis=1)
unet_y =  np.nanmean(mean_unet.reshape(-1,365),axis=1)
qm_y = np.nanmean(mean_qm.reshape(-1,365),axis=1)
start = 2100 - obs_y.shape[0]
x= np.arange(start,2100)

plt.plot(x,obs_y.reshape(-1,),'r',label = 'Observation',)
plt.plot(x,cesm_y.reshape(-1,),'b',label = 'E3SM RMSE = '+str(np.round(RMSE(obs_y.reshape(-1,), cesm_y.reshape(-1,)),2)))
plt.plot(x,unet_y.reshape(-1,),'g',label = 'NSBC RMSE = '+str(np.round(RMSE(obs_y.reshape(-1,), unet_y.reshape(-1,)),2)))
plt.plot(x,qm_y.reshape(-1,),label = 'EDCDFm RMSE = '+str(np.round(RMSE(obs_y.reshape(-1,), qm_y.reshape(-1,)),2)))
plt.legend(frameon=False,fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=12)
plt.ylabel('yearly mean  (C)',fontsize = 14)
out_path = ('./plots/figure4.png')
plt.savefig(out_path,dpi=100)
plt.show()




