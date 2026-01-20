'''
This code is the figure plots, including the
figure 1. 1st-4th order moments
figure 2. Q33, Q66,99
figure 3. 1,3,5 days maximum precipitation
figure 4. winter and summer precipitation ratio
figure 5. monthly mean

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

## step 0. input the dataset

mask_numpy = np.load("./data/mask.npy")
plat = np.load('./data/Precip_Latitude.npy')[1:,]
plon = np.load('./data/Precip_Longitude.npy')
lon_p, lat_p = np.meshgrid(plon, plat)
preds_path = './data/preds_ep_unet.npy' # this is the result of the UNet
qm_path = './data/preds_ep_ufnet.npy' # this is the result of the UFNet
rada_path = './data/e3sm_qm_only_linear_interp.npy' # this is the result of E3SM QM mapping
nP4GCM_1 = np.load('./data/np4GCM_test.npy') # this is the E3SM
nP4Obser_1 = np.load('./data/test_obs.npy') # this is the observation

preds = np.load(preds_path)
preds[preds<0]=0
nP4GCM = nP4GCM_1
nP4Obser = nP4Obser_1[:,0,:,:]
qm = np.load(qm_path).squeeze()
yan_rada = np.load(rada_path).squeeze()


nP4GCM = np.array(nP4GCM).squeeze()
nP4Obser = np.array(nP4Obser).squeeze()
CESM2_Precip = copy.deepcopy(nP4GCM)
Obs_Precip = copy.deepcopy(nP4Obser)
QM_Precip = copy.deepcopy(qm)
YAN_Precip  = copy.deepcopy(yan_rada)


### step 1. 1st to 4th order moments
### Observation Eval
mean_obs = np.mean(Obs_Precip, axis=0)
mean_obs[mask_numpy == 0] = np.nan
std_obs = np.std(Obs_Precip, dtype=np.float64, axis=0)
std_obs[mask_numpy == 0] = np.nan
skew_obs = skew(Obs_Precip)
skew_obs[mask_numpy == 0] = np.nan
kur_obs = kurtosis(Obs_Precip)
kur_obs[mask_numpy == 0] = np.nan

### E3SM Eval
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

### UNET Eval
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

### UFNET Eval
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

### QM Eval
mean_yan = np.mean(YAN_Precip, axis=0).squeeze()
mean_yan[mask_numpy == 0] = np.nan
mean_yan_r = correlation(mean_obs, mean_yan)
mean_yan_rmse = RMSE(mean_obs, mean_yan)
std_yan = np.std(YAN_Precip, dtype=np.float64, axis=0)
std_yan[mask_numpy == 0] = np.nan
std_yan_r = correlation(std_obs, std_yan)
std_yan_rmse = RMSE(std_obs, std_yan)
skew_yan = skew(YAN_Precip)
skew_yan[mask_numpy == 0] = np.nan
skew_obs[np.isnan(skew_yan)] = np.nan
skew_yan_r = correlation(skew_obs, skew_yan)
skew_yan_rmse = RMSE(skew_obs, skew_yan)
kur_yan = kurtosis(YAN_Precip)
kur_yan[mask_numpy == 0] = np.nan
kur_obs[np.isnan(kur_yan)] = np.nan
kur_yan_r = correlation(kur_obs, kur_yan)
kur_yan_rmse = RMSE(kur_obs, kur_yan)


if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
extent = [-135, -56, 15.5, 52]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
                                                           ["#FFFFFF", "#14FFFF", "#FFB61F", "#FF5500", "#AB0000"])
blank = np.zeros((27, 62))
fname = './data/cb_2018_us_state_5m.shp'
state_shapes = list(shapereader.Reader(fname).geometries())

array_list = [mean_obs, mean_cesm, mean_rada,mean_qm,mean_yan, std_obs, std_cesm, std_rada, std_qm, std_yan,skew_obs, skew_cesm, skew_rada,skew_qm, skew_yan,kur_obs,
              kur_cesm, kur_rada,kur_qm,kur_yan]

r_list = [mean_cesm_r, std_cesm_r, skew_cesm_r, kur_cesm_r]
r_list2 = [mean_rada_r, std_rada_r, skew_rada_r, kur_rada_r]
r_list3 = [mean_qm_r, std_qm_r, skew_qm_r, kur_qm_r]
r_list4 = [mean_yan_r, std_yan_r, skew_yan_r, kur_yan_r]

rmse_list = [mean_cesm_rmse, std_cesm_rmse, skew_cesm_rmse, kur_cesm_rmse]
rmse_list2 = [mean_rada_rmse, std_rada_rmse, skew_rada_rmse, kur_rada_rmse]
rmse_list3 = [mean_qm_rmse, std_qm_rmse, skew_qm_rmse, kur_qm_rmse]
rmse_list4 = [mean_yan_rmse, std_yan_rmse, skew_yan_rmse, kur_yan_rmse]

labels = ['Observation', 'E3SM', 'U-Net','UFNet','QM']
cb_labels = ['Mean', 'Std', 'Skew', 'Kurt']
y_labels = ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis']
max_values = ['6', '10', '6.1', '43']
min_values = ['0', '0.3', '0.4', '-1']
count = 0
extent = [-125, -67, 24, 47]
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(4, 61, hspace=0.17, wspace=0.2)
for i in range(4):
    #if i > 1: continue
    for j in range(5):
        axs = fig.add_subplot(gs[i, (j * 12):(j + 1) * 12], projection=ccrs.PlateCarree())
        cs = axs.pcolormesh(lon_p[:, :], lat_p[:, :], array_list[count], cmap=cmap, vmin=min_values[i],
                            vmax=max_values[i], transform=ccrs.PlateCarree())
        axs.set_extent(extent)
        axs.add_geometries(state_shapes, ccrs.PlateCarree(), edgecolor='grey', facecolor='none', alpha=1)
        if j == 0:
            # axs.set_ylabel(cb_labels[i], fontsize=14)
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
    cb = plt.colorbar(cs, cax=plt.subplot(gs[i, 60:]), orientation='vertical', fraction=0.05, pad=0.01)
    cb.set_label(cb_labels[i])
plt.tight_layout()
out_path = ('./plots/figure1.png')
plt.savefig(out_path,dpi=300)


## step 2. Q33, Q66,99;day 1,3,5
q33_preds = np.percentile(preds,33,axis=0)
q33_preds[mask_numpy == 0] = np.nan
q66_preds = np.percentile(preds,66,axis=0)
q66_preds[mask_numpy == 0] = np.nan
q99_preds = np.percentile(preds,99,axis=0)
q99_preds[mask_numpy == 0] = np.nan
pp_preds = np.zeros((preds.shape[1],preds.shape[2]))
for w in range(preds.shape[1]):
    for h in range(preds.shape[2]):
        pp_preds[w,h] = np.where(preds[:,w,h]>1)[0].shape[0]/preds.shape[0]
pp_preds [mask_numpy == 0] = np.nan


q33_nP4GCM = np.percentile(nP4GCM,33,axis=0)
q33_nP4GCM[mask_numpy == 0] = np.nan
q66_nP4GCM = np.percentile(nP4GCM,66,axis=0)
q66_nP4GCM[mask_numpy == 0] = np.nan
q99_nP4GCM = np.percentile(nP4GCM,99,axis=0)
q99_nP4GCM[mask_numpy == 0] = np.nan
pp_nP4GCM = np.zeros((nP4GCM.shape[1],nP4GCM.shape[2]))
for w in range(nP4GCM.shape[1]):
    for h in range(nP4GCM.shape[2]):
        pp_nP4GCM[w,h] = np.where(nP4GCM[:,w,h]>1)[0].shape[0]/nP4GCM.shape[0]
pp_nP4GCM [mask_numpy == 0] = np.nan



q33_nP4Obser = np.percentile(nP4Obser,33,axis=0)
q33_nP4Obser[mask_numpy == 0] = np.nan
q66_nP4Obser = np.percentile(nP4Obser,66,axis=0)
q66_nP4Obser[mask_numpy == 0] = np.nan
q99_nP4Obser = np.percentile(nP4Obser,99,axis=0)
q99_nP4Obser[mask_numpy == 0] = np.nan
pp_nP4Obser = np.zeros((nP4Obser.shape[1],nP4Obser.shape[2]))
for w in range(nP4Obser.shape[1]):
    for h in range(nP4Obser.shape[2]):
        pp_nP4Obser[w,h] = np.where(nP4Obser[:,w,h]>1)[0].shape[0]/nP4Obser.shape[0]
pp_nP4Obser [mask_numpy == 0] = np.nan



q33_qm = np.percentile(qm,33,axis=0)
q33_qm[mask_numpy == 0] = np.nan
q66_qm = np.percentile(qm,66,axis=0)
q66_qm[mask_numpy == 0] = np.nan
q99_qm = np.percentile(qm,99,axis=0)
q99_qm[mask_numpy == 0] = np.nan
pp_qm = np.zeros((qm.shape[1],qm.shape[2]))
for w in range(qm.shape[1]):
    for h in range(qm.shape[2]):
        pp_qm[w,h] = np.where(qm[:,w,h]>1)[0].shape[0]/qm.shape[0]
pp_qm [mask_numpy == 0] = np.nan


yan  = YAN_Precip
q33_yan = np.percentile(yan,33,axis=0)
q33_yan[mask_numpy == 0] = np.nan
q66_yan = np.percentile(yan,66,axis=0)
q66_yan[mask_numpy == 0] = np.nan
q99_yan = np.percentile(yan,99,axis=0)
q99_yan[mask_numpy == 0] = np.nan

pp_yan = np.zeros((yan.shape[1],yan.shape[2]))
for w in range(yan.shape[1]):
    for h in range(yan.shape[2]):
        pp_yan[w,h] = np.where(yan[:,w,h]>1)[0].shape[0]/yan.shape[0]
pp_yan [mask_numpy == 0] = np.nan




array_list = [q33_nP4Obser,q33_nP4GCM,q33_preds,q33_qm,q33_yan, q66_nP4Obser,q66_nP4GCM,q66_preds,q66_qm,q66_yan,q99_nP4Obser,q99_nP4GCM,q99_preds,q99_qm,q99_yan,pp_nP4Obser,pp_nP4GCM,pp_preds,pp_qm,pp_yan]
mean_cesm_r_33 = correlation(q33_nP4GCM, q33_nP4Obser)
mean_rada_r_33= correlation(q33_preds, q33_nP4Obser)
mean_qm_r_33 = correlation(q33_qm, q33_nP4Obser)
mean_yan_r_33 = correlation(q33_yan, q33_nP4Obser)

mean_cesm_r_66= correlation(q66_nP4GCM, q66_nP4Obser)
mean_rada_r_66= correlation(q66_preds, q66_nP4Obser)
mean_qm_r_66= correlation(q66_qm, q66_nP4Obser)
mean_yan_r_66= correlation(q66_yan, q66_nP4Obser)


mean_cesm_r_99=correlation(q99_nP4GCM, q99_nP4Obser)
mean_rada_r_99=correlation(q99_preds, q99_nP4Obser)
mean_qm_r_99=correlation(q99_qm, q99_nP4Obser)
mean_yan_r_99=correlation(q99_yan, q99_nP4Obser)

mean_cesm_r_pp=correlation(pp_nP4GCM, pp_nP4Obser)
mean_rada_r_pp=correlation(pp_preds, pp_nP4Obser)
mean_qm_r_pp=correlation(pp_qm, pp_nP4Obser)
mean_yan_r_pp=correlation(pp_yan, pp_nP4Obser)

mean_cesm_rmse_33=RMSE(q33_nP4GCM, q33_nP4Obser)
mean_rada_rmse_33=RMSE(q33_preds, q33_nP4Obser)
mean_qm_rmse_33=RMSE(q33_qm, q33_nP4Obser)
mean_yan_rmse_33=RMSE(q33_yan, q33_nP4Obser)

mean_cesm_rmse_66=RMSE(q66_nP4GCM, q66_nP4Obser)
mean_rada_rmse_66=RMSE(q66_preds, q66_nP4Obser)
mean_qm_rmse_66=RMSE(q66_qm, q66_nP4Obser)
mean_yan_rmse_66=RMSE(q66_yan, q66_nP4Obser)

mean_cesm_rmse_99=RMSE(q99_nP4GCM, q99_nP4Obser)
mean_rada_rmse_99=RMSE(q99_preds, q99_nP4Obser)
mean_qm_rmse_99=RMSE(q99_qm, q99_nP4Obser)
mean_yan_rmse_99=RMSE(q99_yan, q99_nP4Obser)

mean_cesm_rmse_pp=RMSE(pp_nP4GCM, pp_nP4Obser)
mean_rada_rmse_pp=RMSE(pp_preds, pp_nP4Obser)
mean_qm_rmse_pp=RMSE(pp_qm, pp_nP4Obser)
mean_yan_rmse_pp=RMSE(pp_yan, pp_nP4Obser)


r_list = [mean_cesm_r_33, mean_cesm_r_66, mean_cesm_r_99,mean_cesm_r_pp]
r_list2 = [mean_rada_r_33, mean_rada_r_66, mean_rada_r_99,mean_rada_r_pp]
r_list3 = [mean_qm_r_33, mean_qm_r_66, mean_qm_r_99,mean_qm_r_pp]
r_list4 = [mean_yan_r_33, mean_yan_r_66, mean_yan_r_99,mean_yan_r_pp]

rmse_list = [mean_cesm_rmse_33, mean_cesm_rmse_66, mean_cesm_rmse_99,mean_cesm_rmse_pp]
rmse_list2 = [mean_rada_rmse_33, mean_rada_rmse_66, mean_rada_rmse_99,mean_rada_rmse_pp]
rmse_list3 = [mean_qm_rmse_33, mean_qm_rmse_66, mean_qm_rmse_99,mean_qm_rmse_pp]
rmse_list4 = [mean_yan_rmse_33, mean_yan_rmse_66, mean_yan_rmse_99,mean_yan_rmse_pp]

labels = ['Observation', 'E3SM', 'U-Net','UFNet','QM']
cb_labels = ['Q33', 'Q66', 'Q99','PoP']
y_labels = ['Q33', 'Q66', 'Q99','PoP']
max_values = ['0.8', '2', '40','1']
min_values = ['0', '0', '8','0']
count = 0
extent = [-125, -67, 24, 47]
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(4, 61, hspace=0.17, wspace=0.2)
for i in range(4):
    #if i > 1: continue
    for j in range(5):
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
    cb = plt.colorbar(cs, cax=plt.subplot(gs[i, 60:]), orientation='vertical', fraction=0.05, pad=0.01)
    cb.set_label(cb_labels[i])
plt.tight_layout()
out_path = ('./plots/figure2.png')
plt.savefig(out_path,dpi = 300)



### step 3. 1,3,5 days maximum precipitation
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
            day_max[i, j] = np.nanmax(data3day)
    return day_max


def day5max(data):
    day_max = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            data3day = np.convolve(data[:, i, j], np.ones(5, dtype=int), 'valid')
            day_max[i, j] = np.nanmax(data3day)
    return day_max


preds = np.load(preds_path).squeeze()
preds[preds<0]=0
nP4GCM = np.array(nP4GCM).squeeze()
nP4Obser = np.array(nP4Obser).squeeze()
qm = np.load(qm_path).squeeze()
yan = np.load(rada_path).squeeze()

Obs_Precip=nP4Obser
CESM2_Precip = nP4GCM
RADA_Data = preds
QM_Data = qm
YAN_Data = yan

intensity_obs = Intensity(Obs_Precip)
intensity_cesm = Intensity(CESM2_Precip)
intensity_cesm_r = correlation(intensity_obs, intensity_cesm)
intensity_cesm_rmse = RMSE(intensity_obs, intensity_cesm)

day1max_obs = day1max(Obs_Precip) / 10
day1max_cesm = day1max(CESM2_Precip) / 10  # per day
day1max_cesm_r = correlation(day1max_obs, day1max_cesm)
day1max_cesm_rmse = RMSE(day1max_obs, day1max_cesm)

day3max_obs = day3max(Obs_Precip) / 10
day3max_cesm = day3max(CESM2_Precip) / 10  # per day
day3max_cesm_r = correlation(day3max_obs, day3max_cesm)
day3max_cesm_rmse = RMSE(day3max_obs, day3max_cesm)

day5max_obs = day5max(Obs_Precip) / 10
day5max_cesm = day5max(CESM2_Precip) / 10  # per day
day5max_cesm_r = correlation(day5max_obs, day5max_cesm)
day5max_cesm_rmse = RMSE(day5max_obs, day5max_cesm)


lon_p, lat_p = np.meshgrid(plon, plat)
RADA_Precip = RADA_Data
intensity_rada = Intensity(RADA_Precip)
intensity_rada [mask_numpy ==0] = np.nan
intensity_rada_r = correlation(intensity_obs, intensity_rada)
intensity_rada_rmse = RMSE(intensity_obs, intensity_rada)
day1max_rada = day1max(RADA_Precip) / 10  # per day
day1max_rada_r = correlation(day1max_obs, day1max_rada)
day1max_rada_rmse = RMSE(day1max_obs, day1max_rada)
day3max_rada = day3max(RADA_Precip) / 10  # per day
day3max_rada_r = correlation(day3max_obs, day3max_rada)
day3max_rada_rmse = RMSE(day3max_obs, day3max_rada)
day5max_rada = day5max(RADA_Precip) / 10  # per day
day5max_rada_r = correlation(day5max_obs, day5max_rada)
day5max_rada_rmse = RMSE(day5max_obs, day5max_rada)

QM_Precip = QM_Data
intensity_qm = Intensity(QM_Precip)
intensity_qm [mask_numpy ==0] = np.nan
intensity_qm_r = correlation(intensity_obs, intensity_qm)
intensity_qm_rmse = RMSE(intensity_obs, intensity_qm)
day1max_qm = day1max(QM_Precip) / 10  # per day
day1max_qm_r = correlation(day1max_obs, day1max_qm)
day1max_qm_rmse = RMSE(day1max_obs, day1max_qm)
day3max_qm = day3max(QM_Precip) / 10  # per day
day3max_qm_r = correlation(day3max_obs, day3max_qm)
day3max_qm_rmse = RMSE(day3max_obs, day3max_qm)
day5max_qm = day5max(QM_Precip) / 10  # per day
day5max_qm_r = correlation(day5max_obs, day5max_qm)
day5max_qm_rmse = RMSE(day5max_obs, day5max_qm)

YAN_Precip = YAN_Data
intensity_yan = Intensity(YAN_Precip)
intensity_yan [mask_numpy ==0] = np.nan
intensity_yan_r = correlation(intensity_obs, intensity_yan)
intensity_yan_rmse = RMSE(intensity_obs, intensity_yan)
day1max_yan = day1max(YAN_Precip) / 10  # per day
day1max_yan_r = correlation(day1max_obs, day1max_yan)
day1max_yan_rmse = RMSE(day1max_obs, day1max_yan)
day3max_yan = day3max(YAN_Precip) / 10  # per day
day3max_yan_r = correlation(day3max_obs, day3max_yan)
day3max_yan_rmse = RMSE(day3max_obs, day3max_yan)
day5max_yan = day5max(YAN_Precip) / 10  # per day
day5max_yan_r = correlation(day5max_obs, day5max_yan)
day5max_yan_rmse = RMSE(day5max_obs, day5max_yan)

extent = [-135, -56, 15.5, 52]
import matplotlib.colors
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
                                                           ["#FFFFFF", "#14FFFF", "#FFB61F", "#FF5500", "#AB0000"])
from cartopy.io import shapereader
blank = np.zeros((27, 62))

fname = './data/cb_2018_us_state_5m.shp'
state_shapes = list(shapereader.Reader(fname).geometries())
array_list = [intensity_obs, intensity_cesm, intensity_rada, intensity_qm,intensity_yan, day1max_obs, day1max_cesm, day1max_rada,day1max_qm,day1max_yan,
              day3max_obs, day3max_cesm, day3max_rada,day3max_qm, day3max_yan,day5max_obs, day5max_cesm, day5max_rada,day5max_qm,day5max_yan]
r_list = [intensity_cesm_r, day1max_cesm_r, day3max_cesm_r, day5max_cesm_r]
r_list2 = [intensity_rada_r, day1max_rada_r, day3max_rada_r, day5max_rada_r]
r_list3 = [intensity_qm_r, day1max_qm_r, day3max_qm_r, day5max_qm_r]
r_list4 = [intensity_yan_r, day1max_yan_r, day3max_yan_r, day5max_yan_r]

rmse_list = [intensity_cesm_rmse, day1max_cesm_rmse, day3max_cesm_rmse, day5max_cesm_rmse]
rmse_list2 = [intensity_rada_rmse, day1max_rada_rmse, day3max_rada_rmse, day5max_rada_rmse]
rmse_list3 = [intensity_qm_rmse, day1max_qm_rmse, day3max_qm_rmse, day5max_qm_rmse]
rmse_list4 = [intensity_yan_rmse, day1max_yan_rmse, day3max_yan_rmse, day5max_yan_rmse]

labels = ['Observation', 'E3SM', 'U-Net','UFNet','QM']
cb_labels = ['Int. (mm/d)', r'$max_{1}$ (${\times}$10 mm)', r'$max_{3}$ (${\times}$10 mm)',
             r'$max_{5}$ (${\times}$10 mm)']
y_labels = ['Average Intensity', '1-day max', '3-day max', '5-day max']
max_values = ['11', '20', '40', '40']
min_values = ['1.0', '2.0', '2.0', '2.0']
count = 0
extent = [-125, -67, 24, 47]
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(4, 61, hspace=0.17, wspace=0.2)
for i in range(4):
    #if i > 1: continue
    for j in range(5):
        axs = fig.add_subplot(gs[i, (j * 12):(j + 1) * 12], projection=ccrs.PlateCarree())
        cs = axs.pcolormesh(lon_p[:, :], lat_p[:, :], array_list[count], cmap=cmap, vmin=min_values[i],
                            vmax=max_values[i], transform=ccrs.PlateCarree())
        axs.set_extent(extent)
        axs.add_geometries(state_shapes, ccrs.PlateCarree(), edgecolor='grey', facecolor='none', alpha=1)
        if j == 0:
            # axs.set_ylabel(cb_labels[i], fontsize=14)
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
    cb = plt.colorbar(cs, cax=plt.subplot(gs[i, 60:]), orientation='vertical', fraction=0.05, pad=0.01)
    cb.set_label(cb_labels[i])
plt.tight_layout()
out_path = ('./plots/figure3.png')
plt.savefig(out_path,dpi = 300)



### step 4. winter and summer precipitation ratio

def season_ratio (data):
    day = [0,31,59,90,120,151,181,212,243,273,304,334,365]
    year_number = int(preds.shape[0] / 365)
    prep = np.zeros((12*year_number,preds.shape[1],preds.shape[2]))
    month_number = np.arange(1,13)
    month_name = np.zeros((year_number*12))
    for y in range (year_number):
        for m in range(12):
            prep[y*12+m:,:] = np.sum(data[y*365+day[m]:y*365+day[m+1],:,:],axis=0)
            month_name[y*12+m] = month_number[m]
    winter = month_name[11:-1:12]
    summer = month_name[17:-1:12]
    prep_winter = np.concatenate((prep[11:-1:12,:,:],prep[12:-1:12,:,:],prep[13:-1:12,:,:]),axis=0)
    prep_summer = np.concatenate((prep[17:-1:12,:,:],prep[18:-1:12,:,:],prep[19:-1:12,:,:]),axis=0)
    prep_all = np.sum(data[11:-1],axis=0)
    winter_r = np.sum(prep_winter,axis=0) / prep_all
    summer_r = np.sum(prep_summer,axis=0) / prep_all
    return winter_r,summer_r


preds = np.load(preds_path).squeeze()
nP4GCM = np.array(nP4GCM).squeeze()
nP4Obser = np.array(nP4Obser).squeeze()
qm = np.load(qm_path).squeeze()
yan  = np.load(rada_path).squeeze()

winter_r_obs, summer_r_obs = season_ratio (nP4Obser)
winter_r_gcm, summer_r_gcm = season_ratio (nP4GCM)
winter_r_rada,summer_r_rada= season_ratio (preds)
winter_r_qm, summer_r_qm = season_ratio (qm)
winter_r_yan, summer_r_yan = season_ratio (yan)

winter_r_obs [mask_numpy == 0] = np.nan
summer_r_obs [mask_numpy == 0] = np.nan
winter_r_gcm [mask_numpy == 0] = np.nan
summer_r_gcm [mask_numpy == 0] = np.nan
winter_r_rada [mask_numpy == 0] = np.nan
summer_r_rada [mask_numpy == 0] = np.nan
winter_r_qm [mask_numpy == 0] = np.nan
summer_r_qm [mask_numpy == 0] = np.nan
winter_r_yan [mask_numpy == 0] = np.nan
summer_r_yan [mask_numpy == 0] = np.nan

array_list = [winter_r_obs,winter_r_gcm,winter_r_rada,winter_r_qm,winter_r_yan,summer_r_obs,summer_r_gcm,summer_r_rada,summer_r_qm,summer_r_yan]
mean_cesm_r_winter= correlation(winter_r_obs, winter_r_gcm)
mean_rada_r_winter= correlation(winter_r_obs, winter_r_rada)
mean_qm_r_winter= correlation(winter_r_obs, winter_r_qm)
mean_yan_r_winter= correlation(winter_r_obs, winter_r_yan)

mean_cesm_r_summer= correlation(summer_r_obs, summer_r_gcm)
mean_rada_r_summer= correlation(summer_r_obs, summer_r_rada)
mean_qm_r_summer= correlation(summer_r_obs, summer_r_qm)
mean_yan_r_summer= correlation(summer_r_obs, summer_r_yan)

mean_cesm_rmse_winter=RMSE(winter_r_obs, winter_r_gcm)
mean_rada_rmse_winter=RMSE(winter_r_obs, winter_r_rada)
mean_qm_rmse_winter=RMSE(winter_r_obs, winter_r_qm)
mean_yan_rmse_winter=RMSE(winter_r_obs, winter_r_yan)

mean_cesm_rmse_summer=RMSE(summer_r_obs, summer_r_gcm)
mean_rada_rmse_summer=RMSE(summer_r_obs, summer_r_rada)
mean_qm_rmse_summer=RMSE(summer_r_obs, summer_r_qm)
mean_yan_rmse_summer=RMSE(summer_r_obs, summer_r_yan)


r_list = [mean_cesm_r_winter, mean_cesm_r_summer]
r_list2 = [mean_rada_r_winter, mean_rada_r_summer]
r_list3 = [mean_qm_r_winter, mean_qm_r_summer]
r_list4 = [mean_yan_r_winter, mean_yan_r_summer]

rmse_list = [mean_cesm_rmse_winter, mean_cesm_rmse_summer]
rmse_list2 = [mean_rada_rmse_winter, mean_rada_rmse_summer]
rmse_list3 = [mean_qm_rmse_winter, mean_qm_rmse_summer]
rmse_list4 = [mean_yan_rmse_winter, mean_yan_rmse_summer]

labels = ['Observation', 'E3SM', 'U-Net','UFNet','QM']
cb_labels = ['winter ratio', 'summer ratio']
y_labels = ['winter', 'summer']
max_values = ['0.5', '0.6']
min_values = ['0', '0']
count = 0
extent = [-125, -67, 24, 47]
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(4, 61, hspace=0.17, wspace=0.2)
for i in range(2):
    #if i > 1: continue
    for j in range(5):
        axs = fig.add_subplot(gs[i, (j * 12):(j + 1) * 12], projection=ccrs.PlateCarree())
        cs = axs.pcolormesh(lon_p[:, :], lat_p[:, :], array_list[count], cmap=cmap, vmin=min_values[i],
                            vmax=max_values[i], transform=ccrs.PlateCarree())
        axs.set_extent(extent)
        axs.add_geometries(state_shapes, ccrs.PlateCarree(), edgecolor='grey', facecolor='none', alpha=1)
        if j == 0:
            # axs.set_ylabel(cb_labels[i], fontsize=14)
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
    cb = plt.colorbar(cs, cax=plt.subplot(gs[i, 60:]), orientation='vertical', fraction=0.05, pad=0.01)
    cb.set_label(cb_labels[i])
fig.suptitle("winter and summer prep ratio", fontsize=16, y=0.95)
plt.tight_layout()
out_path = ('./plots/figure4.png')
plt.savefig(out_path,dpi = 300)







### step 5. monthly mean
preds = np.load(preds_path)
nP4GCM = nP4GCM_1
nP4Obser = nP4Obser_1[:,0,:,:]
nP4GCM = np.array(nP4GCM).squeeze()
nP4Obser = np.array(nP4Obser).squeeze()
qm = np.load(qm_path).squeeze()
yan = np.load(rada_path).squeeze()
mask = np.repeat(mask_numpy[np.newaxis,:,:],preds.shape[0],axis=0)
preds [mask ==0] = 0
nP4GCM [mask ==0] = 0
nP4Obser [mask ==0] = 0
qm [mask ==0] = 0
yan [mask ==0] =0
add = np.zeros((2,))+np.nan

mean_obs = np.concatenate((np.nanmean(np.nanmean(nP4Obser,axis = 1),axis = 1),add),axis=0)
mean_cesm = np.concatenate((np.nanmean(np.nanmean(nP4GCM,axis = 1),axis = 1),add),axis=0)
mean_unet = np.concatenate((np.nanmean(np.nanmean(preds,axis = 1),axis = 1),add),axis=0)
mean_qm = np.concatenate((np.nanmean(np.nanmean(qm,axis = 1),axis = 1),add),axis=0)
mean_yan = np.concatenate((np.nanmean(np.nanmean(yan,axis = 1),axis = 1),add),axis=0)


def monthly_mean(day,data):
    a = data[0:365*16,].reshape(-1,365)
    b = np.zeros((16,12))
    for y in range(16):
        for m in range(12):
            b[y,m] = np.nanmean(a[y,day[m]:day[m+1]])
    return b

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
day = [0,31,59,90,120,151,181,212,243,273,304,334,365]
year_number = int(mean_obs.shape[0] / 365)
obs_m = monthly_mean(day,mean_obs)
cesm_m =  monthly_mean(day,mean_cesm)
unet_m =  monthly_mean(day,mean_unet)
qm_m =  monthly_mean(day,mean_qm)
yan_m =  monthly_mean(day,mean_yan)
x = np.arange(16*12)
plt.plot(x,obs_m.reshape(-1,),'red',label = 'Observation',)
plt.plot(x,cesm_m.reshape(-1,),'blue',label = 'E3SM cc = '+str(np.round(pearsonr(obs_m.reshape(-1,), cesm_m.reshape(-1,))[0],2)))
plt.plot(x,qm_m.reshape(-1,),'green',label = 'UFNet cc = '+str(np.round(pearsonr(obs_m.reshape(-1,), qm_m.reshape(-1,))[0],2)))
plt.legend(frameon=False,fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=14)
plt.xticks(np.linspace(0, 11*16+4, 16))
x= np.arange(1995,2011)
ax.set_xticklabels(x)
plt.ylim(0,2.8)
out_path = ('./plots/figure5.png')
plt.savefig(out_path,dpi = 300)



