# UFNet
Paper information.

Shuang Yu, Indrasis Chakraborty, Gemma J. Anderson, Donald D. Lucas, Yannic Lops, and Daniel Galea. UFNet: Joint U-Net and fully connected neural network to bias correct precipitation predictions from climate models. Artificial Intelligence for the Earth Systems, 2024.

Overview.

This work develops the UFNet methodology to correct historical precipitation projection bias. The UFNet deep learning framework consists of a two-part architecture: a U-Net convolutional network to capture the spatiotemporal distribution of precipitation and a fully connected network to capture the distribution of higher-order statistics. The joint network, termed UFNet, can simultaneously improve the spatial structure of the modeled precipitation and capture the distribution of extreme precipitation values. The detail structure of the UFNet is as follows.

<img width="1159" height="682" alt="image" src="https://github.com/user-attachments/assets/54c370f5-53d8-4983-9df0-c34980001cf4" />


Below we provide guidance for applying UFNet to correct the Energy Exascale Earth System Model (E3SM; Golaz et al. 2019) daily precipitation projection over the contiguous United States (CONUS).

Getting started

1. Obtain the historical climate simulation and observation data. The E3SM historical simulation data are available through https://aims2.llnl.gov/search/cmip6/. The CPC unified gauge-based analysis of daily precipitation can be found through https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html. The ECMWF atmospheric reanalysis of the 20th century (ERA-20C) data are available through https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era-20c. The spatial resolution of E3SM and observed datasets are both regridded to a common 1° resolution grid using conservative interpolation. The regridded E3SM, CPC and ERA-20C with 1° resolution can be found throught ./data/.

2. Train the fully connected network (DNN) Python train_dnn.py

3. Train the UFNet Python train_ufnet.py

4. Evaluation and compared with the baseline Python evaluation.py

License This project is licensed under the MIT License - see the LICENSE.md file for details

Release This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344 and was supported by the LLNL-LDRD Program under Project No. 22-SI-008, and it was released as LLNL-SM-869725 and CP03010.
