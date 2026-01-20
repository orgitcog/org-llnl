
###############################################################################################################################
# This code calculates mass-averaged quantities temperature and heat capacity ratios the combustion chamber, and also performances metrics
# such as heat release, IMEP and RI. It was written using R x64 3.1.1 with RStudio 0.98.978.

# Note: Make sure the output file from the cpp code (Outputs_cpp_10000pts.txt) and the pressure trace (pressure_120412b-bis.csv)is in the correct folder ("My Documents")

# Please use the following reference when using part or all of this program: G. Petitpas, M. McNenly and R. Whitesides, "A Framework
# for Quantifying Measurement Uncertainties and Uncertainty Propagation in HCCI/LTGC Engine Experiments", SAE International, 
# Warrendale, PA, SAE Technical Paper 2017-01-0736.

# Please do not hesitate to send questions/suggestions/comments at petitpas1@llnl.gov or whitesides1@llnl.gov

# The work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under contract DE-AC52-07NA27344.
################################################################################################################################

library(sensitivity)

set.seed(10304)

rm(list=ls()) # clear all objects
n_vector = 10000 # size of the vector that is statiscally significant

# ENGINE GEOMETRY (needs to be inputed)
Bore=rnorm(n_vector,mean=0.10223,sd=3.3e-6/2)   # Bore, m
Stroke=rnorm(n_vector,mean=0.12,sd=5*2.5e-5/2)  # Stroke, m
Conrod=rnorm(n_vector,mean=0.192,sd=5*2.5e-5/2) # connecting rod, m
Vd= pi*(Bore)^2*Stroke/4                        # displacement volume, m3
Vc=rnorm(n_vector,mean=0.000075771,sd=2.5e-7/2) # clearance volume, m3
CR=(Vd+Vc)/Vc                                   # calculated compression ratio, m3
RPM=rnorm(n_vector,mean=1200,sd=12)             # engine speed, rev/min

# COMPOSITION AT IVC AND EVO (needs to be inputed)
X_O2_IVC   = rnorm(n_vector, mean=	0.127922	,sd=	0.00277085	)
X_N2_IVC	 = rnorm(n_vector, mean=	0.788411	,sd=	0.0108572	)
X_CO2_IVC	 = rnorm(n_vector, mean=	0.0551264	,sd=	0.00111045	)
X_CO_IVC	 = rnorm(n_vector, mean=	0.000504137	,sd=	5.76E-05	)
X_H2O_IVC	 = rnorm(n_vector, mean=	0.0204024	,sd=	0.0137248	)
X_IC8H18_IVC	 = rnorm(n_vector, mean=	0.0037254	,sd=	0.000110292	)
X_nC7H16_IVC	 = rnorm(n_vector, mean=	0.001168	,sd=	3.46E-05	)
X_C6H5CH3_IVC	 = rnorm(n_vector, mean=	0.00233601	,sd=	6.92E-05	)
X_C5H10_2_IVC	 = rnorm(n_vector, mean=	0.000404602	,sd=	1.20E-05	)
X_O2_EVO	 = rnorm(n_vector, mean=	0.0454472	,sd=	0.00117022	)
X_N2_EVO	 = rnorm(n_vector, mean=	0.77329	,sd=	0.0103548	)
X_CO2_EVO	 = rnorm(n_vector, mean=	0.108093	,sd=	0.00158125	)
X_CO_EVO	 = rnorm(n_vector, mean=	0.000988455	,sd=	0.000111606	)
X_H2O_EVO	 = rnorm(n_vector, mean=	0.0720125	,sd=	0.0124345	)
X_IC8H18_EVO	 = rnorm(n_vector, mean=	8.26E-05	,sd=	2.37E-06	)
X_nC7H16_EVO	 = rnorm(n_vector, mean=	2.59E-05	,sd=	7.43E-07	)
X_C6H5CH3_EVO	 = rnorm(n_vector, mean=	5.18E-05	,sd=	1.49E-06	)
X_C5H10_2_EVO	 = rnorm(n_vector, mean=	8.97E-06	,sd=	2.57E-07	)

sum_products_IVC	 = rnorm(n_vector, mean=	29.7734	,sd=	0.164251	)
sum_products_EVO	 = rnorm(n_vector, mean=	29.2163	,sd=	0.149762	)

# MASS-AVERAGE TEMPERATURE AT IVC (needs to be inputed)
T_IVC = rnorm(n_vector,mean=385.799,sd=16.0652) 

aa= read.table("Outputs_cpp_10000pts.txt",header=TRUE) # overwrites previous inputs with output from cpp code. Please comment out this line if you want to use previous inputs 

X_O2_IVC  =	aa$X_O2_IVC	
X_N2_IVC	=	aa$X_N2_IVC	
X_CO2_IVC	=	aa$X_CO2_IVC	
X_CO_IVC	=	aa$X_CO_IVC	
X_H2O_IVC	=	aa$X_H2O_IVC	
X_IC8H18_IVC	=	aa$X_IC8H18_IVC	
X_nC7H16_IVC	=	aa$X_nC7H16_IVC	
X_C6H5CH3_IVC	=	aa$X_C6H5CH3_IVC	
X_C5H10_2_IVC	=	aa$X_C5H10_2_IVC	
X_O2_EVO	=	aa$X_O2_EVO	
X_N2_EVO	=	aa$X_N2_EVO	
X_CO2_EVO	=	aa$X_CO2_EVO	
X_CO_EVO	=	aa$X_CO_EVO	
X_H2O_EVO	=	aa$X_H2O_EVO	
X_IC8H18_EVO	=	aa$X_IC8H18_EVO	
X_nC7H16_EVO	=	aa$X_nC7H16_EVO	
X_C6H5CH3_EVO	=	aa$X_C6H5CH3_EVO	
X_C5H10_2_EVO	=	aa$X_C5H10_2_EVO	
sum_products_IVC	=	aa$sum_products_IVC	
sum_products_EVO	=	aa$sum_products_EVO	
T_IVC = aa$T_IVC



#FUNCTION TO CALCULATE CHAMBER'S VOLUME AS A FUNCTION OF THE CRANK ANGLE
Volume<-function(angle){  
 angle=rnorm(n_vector,mean=angle,sd=0.05/2)
 Volume=Vd/(CR-1)+0.5*Vd*(2*Conrod/Stroke+1-cos(angle*pi/(180))-((2*Conrod/Stroke)^2-(sin(angle*pi/(180)))^2)^(1/2))
 return(Volume)}

## THERMODYNAMICS DATAS
#####
## Values here assumed to have a negligible compared to experimental errors (~ 0)
# molar masses
M_IC8H18=114.22852
M_nC7H16=100.20194
M_C6H5CH3=92.13842
M_C5H10_2=70.1329
M_IC5H12=72.14878
M_NC5H12=72.14878
M_O2=31.9988
M_N2=28.0134
M_CO2=44.0095
M_CO=28.0151
M_H2O=18.01528

# surrogate composition: RD387 AKI87 fuel (molar basis), from : Mehl et al,
# "An Approach for Formulating Surrogates for Gasoline with Application toward 
# a Reduced Surrogate Mechanism for CFD Engine Modeling," Energy Fuels 25(11):52155223, 
# 2011, doi:10.1021/ef201099y.
X_IC8H18=48.8
X_nC7H16=15.3
X_C6H5CH3=30.6
X_C5H10_2=5.3

nC_fuel= (8*X_IC8H18 + 7*(X_nC7H16 +X_C6H5CH3) + 5 * X_C5H10_2)/100 # number of C atoms
nH_fuel= (18*X_IC8H18 + 16*X_nC7H16 + 8 *X_C6H5CH3 + 10 * X_C5H10_2)/100 # number of H atoms
M_fuel= nC_fuel*12.0107+nH_fuel*1.00794 # fuel molar mass

# NASA polynomials
# O2  
NASA_O2= matrix (c(3.212936,1.127486E-03,-5.756150E-07,1.313877E-09,-8.768554E-13,-1.005249E+03,3.697578E+00,6.135197E-04,-1.258842E-07,1.775281E-11,-1.136435E-15,-1.233930E+03),2,6,byrow = TRUE)
cutoff_temp_O2 = 1000
# N2  
NASA_N2= matrix (c(3.298677000E+00,1.408240000E-03,-3.963222000E-06,5.641515000E-09,-2.444855000E-12,-1.020900000E+03,2.926640000E+00,1.487977000E-03,-5.684761000E-07,1.009704000E-10,-6.753351000E-15,-9.227977000E+02),2,6,byrow = TRUE)
cutoff_temp_N2 = 1000
# CO2  
NASA_CO2= matrix (c(2.579304900E+00,8.246849870E-03,-6.427160470E-06,2.546370240E-09,-4.120304430E-13,-4.841628300E+04,8.811410410E+00,5.189530180E+00,2.060064760E-03,-7.335753240E-07,1.170043740E-10,-6.917292150E-15,-4.931789530E+04,-5.182893030E+00),2,7,byrow = TRUE)
cutoff_temp_CO2 = 1380
# CO  
NASA_CO= matrix (c(3.19036352E+00,8.94419972E-04, -3.24927563E-08,-1.04599967E-10, 2.41965693E-14, -1.42869054E+04,5.33277914E+00, 3.11216890E+00, 1.15948283E-03,-3.38480362E-07, 4.41403098E-11, -2.12862228E-15,-1.42718539E+04, 5.71725177E+00),2,7,byrow = TRUE)
cutoff_temp_CO = 1429
# H2O  
NASA_H2O= matrix (c(3.386842000E+00,3.474982000E-03,-6.354696000E-06,6.968581000E-09,-2.506588000E-12,-3.020811000E+04,2.590233000E+00,2.672146000E+00,3.056293000E-03,-8.730260000E-07,1.200996000E-10,-6.391618000E-15,-2.989921000E+04,6.862817000E+00),2,7,byrow = TRUE)
cutoff_temp_H2O = 1000
# IC8H18  
NASA_IC8H18= matrix (c(-4.208688930E+00,1.114405810E-01,-7.913465820E-05,2.924062420E-08,-4.437431910E-12,-2.994468750E+04,4.495217010E+01,2.713735900E+01,3.790048900E-02,-1.294373580E-05,2.007603720E-09,-1.164005800E-13,-4.079581770E+04,-1.232774950E+02),2,7,byrow = TRUE)
cutoff_temp_IC8H18 = 1396
# nC7H16 
NASA_nC7H16= matrix (c(-1.268361870E+00,8.543558200E-02,-5.253467860E-05,1.629457210E-08,-2.023949250E-12,-2.565865650E+04,3.537329120E+01,2.221489690E+01,3.476757500E-02,-1.184071290E-05,1.832984780E-09,-1.061302660E-13,-3.427600810E+04,-9.230401960E+01),2,7,byrow = TRUE)
cutoff_temp_nC7H16= 1391
# C6H5CH3  
NASA_C6H5CH3= matrix (c(-4.089822890E+00,6.864773740E-02,-4.747165660E-05,1.670012050E-08,-2.395780070E-12,4.499375420E+03,4.345825910E+01,1.630915420E+01,2.253316120E-02,-7.842818270E-06,1.232006300E-09,-7.206750430E-14,-2.758040950E+03,-6.667597740E+01),2,7,byrow = TRUE)
cutoff_temp_C6H5CH3 = 1389
# C5H10_2  
NASA_C5H10_2= matrix (c(-5.415605510E-01,5.396299180E-02,-3.235087380E-05,9.774160370E-09,-1.185346680E-12,-5.986061690E+03,2.971427480E+01,1.411092670E+01,2.283482720E-02,-7.786268350E-06,1.206274910E-09,-6.987959830E-14,-1.143365070E+04,-5.016011630E+01),2,7,byrow = TRUE)
cutoff_temp_C5H10_2 = 1389

# FUNCTION TO CALCULATE Cp/R
Cp_R<-function(NASA_species,cuttoff_temp, temperature){
  CpR=matrix(, nrow = length(temperature), ncol = 1)
  for (i in 1:length(temperature)) {
    if (temperature[i]<cuttoff_temp) {CpR[i] = NASA_species[1,1] + NASA_species[1,2] * temperature[i] + NASA_species[1,3] * (temperature[i])^2 + NASA_species[1,4] * (temperature[i])^3 + NASA_species[1,5] * (temperature[i])^4}
    else {CpR[i,] = NASA_species[2,1] + NASA_species[2,2] * temperature[i] + NASA_species[2,3] * (temperature[i])^2 + NASA_species[2,4] * (temperature[i])^3 + NASA_species[2,5] * (temperature[i])^4}
  }
  return(CpR)}

#####

# load pressure trace
Pressure<-read.csv("pressure_120412b-bis.csv",header=TRUE) # file located in "My Documents". 3 columns: CAD, mean pressure (in MPa) and standard uncertainty (in MPa)

rows_Pressure = nrow(Pressure) # number of Pressure observations
row_BDC = which(Pressure[,1]==-180,arr.ind=TRUE)
row_IVC = which(Pressure[,1]==-155,arr.ind=TRUE)
row_EVO = which(Pressure[,1]==120,arr.ind=TRUE)
row_BDC2 = which(Pressure[,1]==180,arr.ind=TRUE)


#################################
# IMEP (units : Joules)
#################################

IMEP_mat=matrix(, nrow = row_BDC2[1]-row_BDC[1]+1, ncol = 3) # matrix with CAD, mean(IMEP), and standard deviation (IMEP)
colnames(IMEP_mat)=cbind("CAD", "mean_IMEP", "sd_IMEP")
IMEP=matrix(0, nrow = n_vector, ncol = 1)

for (i in row_BDC:row_BDC2) {
      IMEP_mat[i-row_BDC+1,1]=Pressure[i,1]
      IMEP_mat[i-row_BDC+1,2]=mean(IMEP)
      IMEP_mat[i-row_BDC+1,3]=sd(IMEP)
      angle1=rnorm(n_vector, mean=  Pressure[i,1],sd=	0.05/2	)
      angle2=rnorm(n_vector, mean=  Pressure[i+1,1],sd=  0.05/2	)
      Volume1=Volume(Pressure[i,1])
      Volume2=Volume(Pressure[i+1,1])
      IMEP=0.5*(rnorm(n_vector,mean=Pressure[i,2],sd=Pressure[i,3])+(rnorm(n_vector,mean=Pressure[i+1,2],sd=Pressure[i,3])))*1e6*(Volume2-Volume1)*100/Vd/100000+IMEP
    }

mean(GrossIMEP)
sd(GrossIMEP)
plot(IMEP_mat[,1], IMEP_mat[,2],type="l",xlab="CAD",ylab="IMEP [Joules]", panel.first = grid())
lines(IMEP_mat[,1], IMEP_mat[,2]+1.94*IMEP_mat[,3],col="red")
lines(IMEP_mat[,1], IMEP_mat[,2]-1.94*IMEP_mat[,3],col="red")
title("IMEP between -180 and 180 CAD, Mean (black) and 95% Confidence Interval (red)")


#############################################################
# CUMULATIVE NET HEAT RELEASE  (units : Joules)
#############################################################

# Preliminary step : calculate heat capacity ratios both fresh and burnt gases at each crank angle
P_IVC = rnorm(n_vector,Pressure[row_IVC[1],2],Pressure[row_IVC[1],3]) # P_IVC in MPa
n_IVC = Volume(-155) * P_IVC *1e6/(8.3144621*T_IVC)    # number of moles at IVC
m_IVC = n_IVC * sum_products_IVC                       # mass at IVC (= mass at EVo)
n_EVO = m_IVC / sum_products_EVO                       # number of moles at EVO (mass IVC = mass EVO)
gammas_mat=matrix(, nrow = row_EVO[1]-row_IVC[1]+1 , ncol = 9)
colnames(gammas_mat)=cbind("CAD","T_fresh","sd_T_fresh","T_burnt", "sd_T_burnt", "gamma_fresh", "sd gamma_fresh", "gamma_burnt", "sd gamma_burnt")
for (i in row_IVC:row_EVO ) {
  T_fresh = rnorm(n_vector,Pressure[i,2],sd=Pressure[i,3])*1e6*Volume(Pressure[i,1])/(8.3144621*mean(n_IVC))
  T_burnt = rnorm(n_vector,Pressure[i,2],sd=Pressure[i,3])*1e6*Volume(Pressure[i,1])/(8.3144621*mean(n_EVO))
  
  CpR_fresh = X_O2_IVC * Cp_R(NASA_O2, cutoff_temp_O2, T_fresh) +
    X_CO2_IVC * Cp_R(NASA_CO2, cutoff_temp_CO2, T_fresh)+
    X_CO_IVC * Cp_R(NASA_CO, cutoff_temp_CO, T_fresh)+
    X_H2O_IVC * Cp_R(NASA_H2O, cutoff_temp_H2O, T_fresh)+
    X_N2_IVC * Cp_R(NASA_N2, cutoff_temp_N2, T_fresh) +
    X_IC8H18_IVC * Cp_R(NASA_IC8H18, cutoff_temp_IC8H18, T_fresh) +
    X_nC7H16_IVC * Cp_R(NASA_nC7H16, cutoff_temp_nC7H16, T_fresh) +
    X_C6H5CH3_IVC * Cp_R(NASA_C6H5CH3, cutoff_temp_C6H5CH3, T_fresh) +
    X_C5H10_2_IVC * Cp_R(NASA_C5H10_2, cutoff_temp_C5H10_2, T_fresh)
  
  CpR_burnt = X_O2_EVO * Cp_R(NASA_O2, cutoff_temp_O2, T_burnt) + 
    X_CO2_EVO * Cp_R(NASA_CO2, cutoff_temp_CO2, T_burnt) + 
    X_CO_EVO * Cp_R(NASA_CO, cutoff_temp_CO, T_burnt) + 
    X_H2O_EVO * Cp_R(NASA_H2O, cutoff_temp_H2O, T_burnt) +  
    X_N2_EVO * Cp_R(NASA_N2, cutoff_temp_N2, T_burnt)  + 
    X_IC8H18_EVO * Cp_R(NASA_IC8H18, cutoff_temp_IC8H18, T_burnt) + 
    X_nC7H16_EVO * Cp_R(NASA_nC7H16, cutoff_temp_nC7H16, T_burnt) + 
    X_C6H5CH3_EVO * Cp_R(NASA_C6H5CH3, cutoff_temp_C6H5CH3, T_burnt) + 
    X_C5H10_2_EVO * Cp_R(NASA_C5H10_2, cutoff_temp_C5H10_2, T_burnt)
  
  gamma_fresh = CpR_fresh / (CpR_fresh - 1)
  gamma_burnt = CpR_burnt / (CpR_burnt - 1)
  gammas_mat[i-row_IVC+1,1]=Pressure[i,1] 
  gammas_mat[i-row_IVC+1,2]=mean(T_fresh)
  gammas_mat[i-row_IVC+1,3]=sd(T_fresh)
  gammas_mat[i-row_IVC+1,4]=mean(T_burnt)
  gammas_mat[i-row_IVC+1,5]=sd(T_burnt)
  gammas_mat[i-row_IVC+1,6]=mean(gamma_fresh)
  gammas_mat[i-row_IVC+1,7]=sd(gamma_fresh)
  gammas_mat[i-row_IVC+1,8]=mean(gamma_burnt)
  gammas_mat[i-row_IVC+1,9]=sd(gamma_burnt)
  
}

cumHR_mat=matrix(, nrow = row_EVO-row_IVC+1, ncol = 5)
colnames(cumHR_mat)=cbind("CAD", "mean_HR", "sd_HR", "mean_dQdtheta","sd_dQdtheta")


interval_ = 20 #interval for derivative (must be an even nunber)
gg=interval_ / 2 # half-interval for derivative

cumHR=matrix(0, nrow = n_vector, ncol = 1)
dQdteta=matrix(0, nrow = n_vector, ncol = 1)
for (i in row_IVC:row_EVO) {
  cumHR_mat[i-row_IVC+1,1]=Pressure[i,1]
  cumHR_mat[i-row_IVC+1,2]=mean(cumHR)
  cumHR_mat[i-row_IVC+1,3]=sd(cumHR)
  cumHR_mat[i-row_IVC+1,4]=mean(dQdteta)
  cumHR_mat[i-row_IVC+1,5]=sd(dQdteta)
  angle1=rnorm(n_vector, mean=  Pressure[i-gg,1],sd=  0.05/2	)
  angle2=rnorm(n_vector, mean=  Pressure[i+gg,1],sd=  0.05/2	)
  Volume1=Volume(Pressure[i-gg,1])
  Volume2=Volume(Pressure[i+gg,1])
  
  dvdteta=(Volume2-Volume1)/(angle2-angle1)
  dpdteta=1e6*(rnorm(n_vector,mean=Pressure[i+gg,2],sd=Pressure[i+gg,3])-rnorm(n_vector,mean=Pressure[i-gg,2],sd=Pressure[i-gg,3]))/(angle2-angle1)
  
  vdp=0.5*(Volume1+Volume2)*dpdteta/(rnorm(n_vector,gammas_mat[i-row_IVC+1,8],gammas_mat[i-row_IVC+1,9])-1)
  pdv=rnorm(n_vector,gammas_mat[i-row_IVC+1,8],gammas_mat[i-row_IVC+1,9])*0.5*(rnorm(n_vector,mean=Pressure[i-gg,2],sd=Pressure[i-gg,3])+rnorm(n_vector,mean=Pressure[i+gg,2],sd=Pressure[i+gg,3]))*1e6*dvdteta/(rnorm(n_vector,gammas_mat[i-row_IVC+1,8],gammas_mat[i-row_IVC+1,9])-1)
  
  dQdteta=(pdv+vdp)
  cumHR=cumHR+dQdteta*(0.25)
}

plot(cumHR_mat[,1], cumHR_mat[,2],type="l",xlab="CAD",ylab="IMEP [Joules]", ylim=c(-400,2600),panel.first = grid())
lines(cumHR_mat[,1], cumHR_mat[,2]+1.94*cumHR_mat[,3],col="red")
lines(cumHR_mat[,1], cumHR_mat[,2]-1.94*cumHR_mat[,3],col="red")
title("Heat Release between -180 and 180 CAD, Mean (black) and 95% Confidence Interval (red)")

max(cumHR_mat[,4]) # maximum Heat Release Rate, in Joules/CAD
cumHR_mat[row_EVO[1]-row_IVC[1]+1,2] # total Heat Release at EVO, in Joules (mean)
cumHR_mat[row_EVO[1]-row_IVC[1]+1,3] # total Heat Release at EVO, in Joules (sd)
#############################################################
# Ringing Intensity (units : MW/m2)
#############################################################

# first: enter max dp/dtheta as an input, then convert it into dp/dt (in kPa/ min)
dpdthetamax=rnorm(n_vector,mean=1202,sd=139.6) # among each individual raw 100 pressure traces. Derivation over 1.25 CAD. Units: kPa/deg
dpdtmax=dpdthetamax * 6 * RPM * 1e3            # units: kPa/min

# second : calculate Pmax, Tmax (of burnt gas) 
Pmax=max(Pressure[,2])
row_Pmax = which(Pressure[,2]== Pmax,arr.ind=TRUE)
Pmax=rnorm(n_vector,mean=Pmax,sd=Pressure[row_Pmax[1],3])
Pmax=1e6*Pmax # convert MPa to Pa

Tmax=max(gammas_mat[,4])
row_Tmax = which(gammas_mat[,4]== Tmax,arr.ind=TRUE)
Tmax=rnorm(n_vector,mean=Tmax,sd=gammas_mat[row_Tmax[1],5])
    
# third : calculate gamma at Tmax
CpR_ = X_O2_EVO * Cp_R(NASA_O2, cutoff_temp_O2, Tmax) + 
  X_CO2_EVO * Cp_R(NASA_CO2, cutoff_temp_CO2, Tmax) + 
  X_CO_EVO * Cp_R(NASA_CO, cutoff_temp_CO, Tmax) + 
  X_H2O_EVO * Cp_R(NASA_H2O, cutoff_temp_H2O, Tmax) +  
  X_N2_EVO * Cp_R(NASA_N2, cutoff_temp_N2, Tmax)  + 
  X_IC8H18_EVO * Cp_R(NASA_IC8H18, cutoff_temp_IC8H18, Tmax) + 
  X_nC7H16_EVO * Cp_R(NASA_nC7H16, cutoff_temp_nC7H16, Tmax) + 
  X_C6H5CH3_EVO * Cp_R(NASA_C6H5CH3, cutoff_temp_C6H5CH3, Tmax) + 
  X_C5H10_2_EVO * Cp_R(NASA_C5H10_2, cutoff_temp_C5H10_2, Tmax)

gamma_ = CpR_ / (CpR_ - 1)

# fourth : calculate Ringing Intensity  
beta=0.05/1000
RI = (beta * dpdtmax)^2 * sqrt(gamma_ * 8.314 * Tmax * 1000/(sum_products_EVO))/(2 * gamma_ * Pmax) / 1e6
mean(RI)
sd(RI)

######################################################################
# WRITE IMEP, Temperatures, Gammas and Heat Release to a .csv file
#####################################################################

write.csv(IMEP_mat,"IMEP_.csv") 
write.csv(gammas_mat,"Temperatures&Gammas_.csv")
write.csv(cumHR_mat,"cumH01192017_interval20.csv")
