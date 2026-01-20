
//###############################################################################################################################
// This code calculates quantities such as EGR and %residuals through MC Bayesian inference, then computes the estimated mass-averaged 
// temperatures (Tresiduals, Texhaust, TIN,eff, TBDC, TIVC) using a mixing model.  This code is a parallel MPI code and was compiled 
// with gcc-4.9.3 and linked with OpenMPI-1.6.5 and trng-4.19.
//
// Note: Make sure the pressure trace (pressure_120412b-bis.csv) is in the correct folder.
//
// The mixing method used here to estimate the mass averaged temperatures has been adapted from : M. Sjöberg and J. E. Dec, 
// "An Investigation of the Relationship Between Measured Intake Temperature, BDC Temperature, and Combustion Phasing for 
// Premixed and DI HCCI Engines" SAE International, Warrendale, PA, SAE Technical Paper 2004-01-1900, June 2004.
//
// Please use the following reference when using part or all of this program: G. Petitpas, M. McNenly and R. Whitesides, "A Framework
// for Quantifying Measurement Uncertainties and Uncertainty Propagation in HCCI/LTGC Engine Experiments", SAE International, 
// Warrendale, PA, SAE Technical Paper 2017-01-0736.
//
// Please do not hesitate to send questions/suggestions/comments at petitpas1@llnl.gov or whitesides1@llnl.gov

// The work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under contract DE-AC52-07NA27344.
//
//################################################################################################################################

#include <vector>
#include <math.h> //sqrt
#include <iostream>
#include <fstream>
#include <assert.h>
#include <fenv.h> //for fpe trapping
#include "mpi.h"
#include "trng/yarn2.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_dist.hpp"
#include "trng/normal_dist.hpp"


void update_moments(long int n,
                    std::vector<double>& x,
                    std::vector<double>& m,
                    std::vector<double>& s,
                    std::vector<double>& sd)
{
    double denom=1.0/n;
    double denom1=1.0;
    if(n>1) denom1 = 1.0/(n-1);
    for(int i = 0; i < x.size(); ++i)
    {
        double delta = x[i] - m[i];
        m[i] += delta*denom;
        s[i] += delta*(x[i]-m[i]);
        sd[i] = sqrt(s[i]*denom1);
    }
}


void update_moments_global(long int n1,
                           std::vector<double>& m1,
                           std::vector<double>& s1,
                           long int n2,
                           std::vector<double>& m2,
                           std::vector<double>& s2,
                           std::vector<double>& sd)
{
    if(n1==0 && n2==0) return;
    double denom=1.0/(n1+n2);
    double denom1=1.0;
    if((n1+n2)>1) denom1 = 1.0/(n1+n2-1);
    for(int i = 0; i < m1.size(); ++i)
    {
        double delta = m2[i] - m1[i];
        m2[i] = (n1*m1[i] + n2*m2[i])/(n1+n2);
        s2[i] += s1[i] + delta*delta*n1*n2*denom;
        sd[i] = sqrt(s2[i]*denom1);
    }
}

double Cp_R(const double* NASA_species, double temperature, int nterms)
{
    double CpR;
    int length=2*nterms+1; //two ranges and cutoff temperature
    double cutoff_temp = NASA_species[length-1];
    const double* NASA_terms;
    if(temperature< cutoff_temp)
    {
        NASA_terms = &NASA_species[0];
    }
    else
    {
        NASA_terms = &NASA_species[nterms];
    }
    CpR = NASA_terms[0] 
        + NASA_terms[1] * temperature
        + NASA_terms[2] * pow(temperature,2)
        + NASA_terms[3] * pow(temperature,3)
        + NASA_terms[4] * pow(temperature,4);
    return CpR;
};

double trng_ran_triangular(trng::yarn2 &rng, double a, double b, double c)
{
    static trng::uniform01_dist<double> uniform;
    double U = uniform(rng);
    double F = (c - a) / (b - a);
    if (U <= F)
        return a + sqrt(U * (b - a) * (c - a));
    else
        return b - sqrt((1 - U) * (b - a) * (b - c));
}

void print_vector(std::vector<double> v)
{
    for(int i = 0; i < v.size(); ++i)
    {
        printf("%g",v[i]);
        if(i < v.size()-1) printf(", ");
    }
}

inline double sample_angle(trng::yarn2 &rng, double angle)
{
    trng::normal_dist<double> normal(angle,0.5/2);
    double as = normal(rng);
    return as;
}

inline double sample_angle_var(trng::yarn2 &rng, double angle, double variance)
{
    trng::normal_dist<double> normal(angle,variance/2);
    double as = normal(rng);
    return as;
}

double Volume(double angle, double Vd, double CR, double Conrod, double Stroke)
{  
    double V = Vd/(CR-1) + 0.5*Vd*(2*Conrod/Stroke+1-cos(angle*M_PI/(180))-sqrt(pow(2*Conrod/Stroke,2)-pow(sin(angle*M_PI/(180)),2)));
    return V;
}

std::vector<std::vector<double> > ReadPressureCSVFile(const char filename[], 
                                                      const char *ignore_chars,
                                                      const int num_lines_skipped)
{
    size_t ignore_pos;
    std::ifstream input_file(filename, std::ios_base::in);
    std::string line;
    std::string sub_line;
    double CA_read, P_mean_read, P_stddev_read;
    int num_values_read;
    int num_lines_read = 0;

    std::vector<std::vector<double> > pressure_data;

    if(!input_file.is_open()) {
        printf("ERROR: In VolumeFromFile::ReadFile(...),\n");
        printf("       could not open file %s.\n", filename);
        fflush(stdout);
        return pressure_data;
    }

    while(std::getline(input_file, line)) {
        ++num_lines_read;

        if(num_lines_read > num_lines_skipped) {
            ignore_pos = line.find_first_of(ignore_chars);
            sub_line = line.substr(0,ignore_pos);

            if(sub_line.size() > 2) { // minimum length for two values
                num_values_read = sscanf(sub_line.c_str(),
                                         "%lf,%lf,%lf",
                                         &CA_read,
                                         &P_mean_read,
                                         &P_stddev_read);
                if(num_values_read != 3) {
                    printf("INFO: Skipping line %d of %s, no data pair read by sscanf:\n",
                           num_lines_read,filename);
                    printf("      %s\n",line.c_str());
                } else {
                    std::vector<double> line_data(3);
                    line_data[0] = CA_read;
                    line_data[1] = P_mean_read;
                    line_data[2] = P_stddev_read;
                    pressure_data.push_back(line_data);
                }
            }
        } 
    }

    return pressure_data;
}

std::vector<double> get_rank_start(const int nprocs, const int batch_size)
{
    std::vector<double> rank_start(nprocs+1);
    rank_start[0] = 0;
    int batch_per_proc  = batch_size / nprocs;
    int batch_remainder = batch_size % nprocs;
    for(int iproc = 0; iproc < nprocs; ++iproc)
    {
        rank_start[iproc+1] = rank_start[iproc] + batch_per_proc;
        if(iproc < batch_remainder) rank_start[iproc+1] += 1;
    }
    return rank_start;
}

// molar masses
static const double M_IC8H18=114.22852;
static const double M_nC7H16=100.20194;
static const double M_C6H5CH3=92.13842;
static const double M_C5H10_2=70.1329;
static const double M_IC5H12=72.14878;
static const double M_NC5H12=72.14878;
static const double M_O2=31.9988;
static const double M_N2=28.0134;
static const double M_CO2=44.0095;
static const double M_CO=28.0151;
static const double M_H2O=18.01528;

// surrogate composition: RD387 AKI87 fuel (molar basis), from : Mehl et al,
// "An Approach for Formulating Surrogates for Gasoline with Application toward 
// a Reduced Surrogate Mechanism for CFD Engine Modeling," Energy Fuels 25(11):52155223, 
// 2011, doi:10.1021/ef201099y.
static const double X_IC8H18=0.488;
static const double X_nC7H16=0.153;
static const double X_C6H5CH3=0.306;
static const double X_C5H10_2=0.053;

static const double nC_fuel= (8*X_IC8H18 + 7*(X_nC7H16 +X_C6H5CH3) + 5 * X_C5H10_2); // number of C atoms
static const double nH_fuel= (18*X_IC8H18 + 16*X_nC7H16 + 8 *X_C6H5CH3 + 10 * X_C5H10_2); // number of H atoms
static const double M_fuel= nC_fuel*12.0107+nH_fuel*1.00794; // fuel molar mass

static const double AF_st     = (nC_fuel+nH_fuel*.25)*138.2796/(nC_fuel*12.0107+nH_fuel*1.00794); // air to fuel ratio assuming stochiometry
static const double air_ratio = 79.0/21.0;// ratio between Xn2 and Xo2

static const double CO2_exp_IVC =0.05588;
static const double CO2_exp_EVO =0.1138;
static const double O2_exp_EVO =0.0501; 
static const double CO_exp_EVO =1044.0/1.0e6; 
static const double THC_exp_EVO =1248.0/1.0e6;

static const double err_CO2_exp_IVC= 0.0017; // error in the CO2 analyzer at IVC
static const double err_CO2_exp_EVO= 0.0036;// error in the CO2 analyzer at EVO
static const double err_O2_exp_EVO= 0.0035;// error in the O2 analyzer at EVO
static const double err_CO_exp_EVO= 200.0/1.0e6;// error in the CO analyzer at EVO
static const double err_THC_exp_EVO=62.0/1.0e6;// error in the THC analyzer at EVO

// NASA polynomials
// O2  
static const double NASA_O2[] = {3.212936,1.127486E-03,-5.756150E-07,
    1.313877E-09,-8.768554E-13,-1.005249E+03,
    3.697578E+00,6.135197E-04,-1.258842E-07,
    1.775281E-11,-1.136435E-15,-1.233930E+03, 1000.0};
// N2  
static const double NASA_N2[] = {3.298677000E+00,1.408240000E-03,-3.963222000E-06,
    5.641515000E-09,-2.444855000E-12,-1.020900000E+03,
    2.926640000E+00,1.487977000E-03,
    -5.684761000E-07,1.009704000E-10,-6.753351000E-15,-9.227977000E+02, 1000.0};
// CO2  
static const double NASA_CO2[] = {2.579304900E+00,8.246849870E-03,-6.427160470E-06,
    2.546370240E-09,-4.120304430E-13,-4.841628300E+04,
    8.811410410E+00,5.189530180E+00,2.060064760E-03,
    -7.335753240E-07,1.170043740E-10,-6.917292150E-15,
    -4.931789530E+04,-5.182893030E+00, 1380.0};
// CO  
static const double NASA_CO[] = {3.19036352E+00, 8.94419972E-04, -3.24927563E-08, 
    -1.04599967E-10, 2.41965693E-14, -1.42869054E+04, 
    5.33277914E+00, 3.11216890E+00, 1.15948283E-03,
    -3.38480362E-07, 4.41403098E-11, -2.12862228E-15,
    -1.42718539E+04, 5.71725177E+00, 1429.0};
// H2O  
static const double NASA_H2O[] = {3.386842000E+00,3.474982000E-03,-6.354696000E-06,
    6.968581000E-09,-2.506588000E-12,-3.020811000E+04,
    2.590233000E+00,2.672146000E+00,3.056293000E-03,
    -8.730260000E-07,1.200996000E-10,-6.391618000E-15,
    -2.989921000E+04,6.862817000E+00, 1000.0};
// IC8H18  
static const double NASA_IC8H18[] = {-4.208688930E+00,1.114405810E-01,-7.913465820E-05,
    2.924062420E-08,-4.437431910E-12,-2.994468750E+04,
    4.495217010E+01,2.713735900E+01,3.790048900E-02,
    -1.294373580E-05,2.007603720E-09,-1.164005800E-13,
    -4.079581770E+04,-1.232774950E+02, 1396.0};
// nC7H16 
static const double NASA_nC7H16[] = {-1.268361870E+00,8.543558200E-02,-5.253467860E-05,
    1.629457210E-08,-2.023949250E-12,-2.565865650E+04,
    3.537329120E+01,2.221489690E+01,3.476757500E-02,
    -1.184071290E-05,1.832984780E-09,-1.061302660E-13,
    -3.427600810E+04,-9.230401960E+01, 1391.0};
// C6H5CH3  
static const double NASA_C6H5CH3[] = {-4.089822890E+00,6.864773740E-02,-4.747165660E-05,
    1.670012050E-08,-2.395780070E-12,4.499375420E+03,
    4.345825910E+01,1.630915420E+01,2.253316120E-02,
    -7.842818270E-06,1.232006300E-09,-7.206750430E-14,
    -2.758040950E+03,-6.667597740E+01, 1389.0};
// C5H10_2  
static const double NASA_C5H10_2[] = {-5.415605510E-01,5.396299180E-02,-3.235087380E-05,
    9.774160370E-09,-1.185346680E-12,-5.986061690E+03,
    2.971427480E+01,1.411092670E+01,2.283482720E-02,
    -7.786268350E-06,1.206274910E-09,-6.987959830E-14,
    -1.143365070E+04,-5.016011630E+01, 1389.0};

int main(int argc, char* argv[])
{
    // Trap floating point exceptions
    feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW );
    MPI_Init(&argc,&argv);
    int rank,nprocs,ierr;
    ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    // Load pressure trace and assume conservation of mass during the closed cycle
    // Pressure data are in MPa
    std::vector<std::vector<double> > Pressure = ReadPressureCSVFile("./pressure_120412b-bis.csv", "#", 1);
    int row_IVC = -1;
    int row_blowdown = -1;
    int row_after_blowdown = -1;
    int row_TDC = -1;  // TDC after ignition
    int row_intake_BDC = -1;
    double CA_IVC = -155; // IVC angle
    for(int i = 0; i < Pressure.size(); ++i)
    {
        if(fabs(Pressure[i][0]-CA_IVC) < 1.0e-3)
        {
            row_IVC = i;
        }
        if(fabs(Pressure[i][0]-120) < 1.0e-3)
        {
            row_blowdown = i;
        }
        if(fabs(Pressure[i][0]-359.75) < 1.0e-3)
        {
            row_after_blowdown = i;
        }
        if(fabs(Pressure[i][0]-180) < 1.0e-3)
        {
            row_TDC = i;
        }
        if(fabs(Pressure[i][0]- (-180)) < 1.0e-3)
        {
            row_intake_BDC = i;
        }
    }
    assert(row_IVC >= 0);
    assert(row_blowdown >= 0);
    assert(row_after_blowdown >= 0);
    assert(row_TDC >= 0);
    assert(row_intake_BDC >= 0);

    //These need to be updated if sampling changes.
    int random_numbers_per_loop = 11 + 12 + 7;
    int random_numbers_per_accepted = 12 + 7;

    trng::uniform01_dist<double> uniform01;
    trng::yarn2 rng;
    unsigned long seed = 141164;
    rng.seed(seed);

    long int n_samples_total = 0;
    long int n_samples_found = 0;
    long int n_samples_found_global = 0;
    long int n_batches = 0;

    long int n_samples_wanted = 1e4;
    long int batch_size = n_samples_wanted*1e1;
    std::vector<double> rank_start = get_rank_start(nprocs, batch_size);
    double min_sd_T_IVC = 1.0e30;
    int best_row = -1;

    std::vector<double> means(56,0.0);
    std::vector<double> variances(56,0.0);
    std::vector<double> stddevs(56,0.0);
    std::vector<double> sample(56);

    double mean_pct_res = 0.5;
    double stdev_pct_res = 0.1;

    double mean_pct_res_old = 1.0;
    double stdev_pct_res_old;
    
    int counter_res_loop = 0;// counter for convergence loop on pct_residual

    while ( std::abs(mean_pct_res_old-mean_pct_res) > (0.01 * mean_pct_res_old))// beginning of convergence loop on pct_residuals
    {
        counter_res_loop =  counter_res_loop + 1;
	if (counter_res_loop == 10)// if too many iteration, then exit
        {
            if(rank==0) {
              printf("Failed to converge on residual loop.\n Quitting.\n");
            }
            MPI_Finalize();
            exit(1);
        }

        mean_pct_res_old=mean_pct_res;
        stdev_pct_res_old=mean_pct_res;

	n_samples_found = 0;
        n_samples_found_global = 0;
        for(int k = 0; k < means.size(); ++k)
        {
            means[k] = 0.0;
            variances[k] = 0.0;
            stddevs[k] = 0.0;
            sample[k] = 0.0;
        }

        n_batches = 0;
        int failed_temp_iteration = 0;
        int failed_temp_iteration_global = 0;
        while(n_samples_found_global <  n_samples_wanted) //bayesian inference loop
        {
            n_batches += 1;
            rng.jump(rank_start[rank]*random_numbers_per_loop);
            for(long int n_vector = rank_start[rank]; n_vector < rank_start[rank+1]; n_vector+=1)
            {
                trng::normal_dist<double> q_air_dist(10.97,0.01); //experimentally measured q_air
                trng::normal_dist<double> q_fuel_dist(0.594,0.594 * 0.01);//experimentally measured q_fuel
                trng::uniform_dist<double> egr_dist(0.0,1.0);//initial guess for egr (uniform distribution)

                double q_air = q_air_dist(rng);
                double q_fuel = q_fuel_dist(rng);
                double egr=egr_dist(rng);

                trng::uniform_dist<double> res_dist1(0.0,1.0);//initial guess for res (uniform distribution)
                trng::normal_dist<double> res_dist2(mean_pct_res,stdev_pct_res);

                double res;

                if (counter_res_loop == 1)
                {
                    res =res_dist1(rng);
                }
                else
                {
                    res = res_dist2(rng);
                }


                if (res < 0.0) // when mean(res) is low (2-5%), negative value can be sampled. This if statement avoids res < 0
                {
                    res=0.0;
                }

                double wr=uniform01(rng); // water removal in the EGR loop
                double condenser_eff_fuel_IVC=uniform01(rng);  // IVC fuel condenser before the gas analyzer
                double condenser_eff_water_IVC=uniform01(rng);  // IVC water condenser before the gas analyzer
                double condenser_eff_fuel_EVO=uniform01(rng); // EVO fuel condenser before the gas analyzer
                double condenser_eff_water_EVO=uniform01(rng); // EVO water condenser before the gas analyzer
                double rCO=uniform01(rng) * 0.1;  // ratio between CO and CO2 in exhaust flow (limited to 0.1)
                double rTHC=uniform01(rng) * 0.1; // ratio between fuel in exhaust and at intake (limited to 0.1)

                double AF_fresh_inv  = q_fuel/q_air; // air to fuel (mass) ratio inverted
                double phi_fresh = AF_st * AF_fresh_inv; // equivalence ratio (mass-based)
                double b = (egr)/(1+(1-egr) * AF_fresh_inv); // molar ratio between Egr and exhaust
                double r = (res)/(1+(1-egr) * (1-res) * AF_fresh_inv); // molar ratio between residuals and exhaust
                double nO2_air = 1.0 * (nC_fuel+nH_fuel * .25)/phi_fresh; //number of moles of O2 from air
 
                double n_fuel_IVC = 1.0/(1.0-(b+r)*rTHC);
                double n_fuel_EVO = rTHC * n_fuel_IVC;

                double n_H2O_EVO  = nH_fuel * (n_fuel_IVC-n_fuel_EVO)/(2 * (1-(1-wr) * b-r));
                double n_CO2_EVO  = nC_fuel * (n_fuel_IVC-n_fuel_EVO)/(1-b-r);
                double n_CO_EVO   = rCO * n_CO2_EVO;
                double n_N2_EVO   = nO2_air * air_ratio/(1-r-b);

                double n_H2O_IVC  = ((1-wr) * b+r) * n_H2O_EVO;
                double n_CO2_IVC  = (b+r) * n_CO2_EVO;
                double n_CO_IVC   = (b+r) * n_CO_EVO;
                double n_N2_IVC   =  n_N2_EVO;

                double n_O2_EVO   = (nO2_air+n_CO2_IVC-n_CO2_EVO+0.5 * (n_H2O_IVC-n_H2O_EVO))/(1.0-b-r);
                double n_O2_IVC   = nO2_air+(b+r)*n_O2_EVO;

                double n_IVC      =  n_fuel_IVC + n_O2_IVC + n_N2_IVC + n_H2O_IVC + n_CO2_IVC + n_CO_IVC;
                double n_EVO      =  n_fuel_EVO + n_O2_EVO + n_N2_EVO + n_H2O_EVO + n_CO2_EVO + n_CO_EVO;

                double X_CO2_IVC_measured  = n_CO2_IVC / ((1-condenser_eff_fuel_IVC) * n_fuel_IVC + n_O2_IVC + n_N2_IVC + (1-condenser_eff_water_IVC) * n_H2O_IVC + n_CO2_IVC + n_CO_IVC);
                double X_O2_EVO_measured   = n_O2_EVO /  ((1-condenser_eff_fuel_EVO) * n_fuel_EVO + n_O2_EVO + n_N2_EVO + (1-condenser_eff_water_EVO) * n_H2O_EVO + n_CO2_EVO + n_CO_EVO);
                double X_CO2_EVO_measured  = n_CO2_EVO / ((1-condenser_eff_fuel_EVO) * n_fuel_EVO + n_O2_EVO + n_N2_EVO + (1-condenser_eff_water_EVO) * n_H2O_EVO + n_CO2_EVO + n_CO_EVO);
                double X_CO_EVO_measured   = n_CO_EVO /  ((1-condenser_eff_fuel_EVO) * n_fuel_EVO + n_O2_EVO + n_N2_EVO + (1-condenser_eff_water_EVO) * n_H2O_EVO + n_CO2_EVO + n_CO_EVO);
                double X_THC_EVO_measured  = nC_fuel * n_fuel_EVO /( n_fuel_EVO * nC_fuel + n_O2_EVO + n_N2_EVO + n_H2O_EVO + n_CO2_EVO + n_CO_EVO); // there is no condenser in front of the THC analyzer

                bool condA =fabs(X_CO2_IVC_measured-CO2_exp_IVC) < err_CO2_exp_IVC;
                bool condB =fabs(X_CO2_EVO_measured-CO2_exp_EVO) < err_CO2_exp_EVO;
                bool condC =fabs(X_O2_EVO_measured-O2_exp_EVO) < err_O2_exp_EVO;
                bool condD =fabs(X_CO_EVO_measured-CO_exp_EVO) < err_CO_exp_EVO;
                bool condE =fabs(X_THC_EVO_measured-THC_exp_EVO) < err_THC_exp_EVO;

                if ((b+r) > 1.0) // this condition gives n_N2< 0 and should not be considered
                {
                    condA = 0;
                    condB = 0;
                    condC = 0;
                    condD = 0;
                    condE = 0;
                }

                if(condA && condB && condC && condD && condE)
                {
                    // GEOMETRY
                    trng::normal_dist<double> Bore_dist(0.10223,3.3e-6/2); //  bore, m
                    trng::normal_dist<double> Stroke_dist(0.12,2.5e-5/2);  //  stroke, m
                    trng::normal_dist<double> Conrod_dist(0.192,2.5e-5/2);  //  connecting rod, m
                    trng::normal_dist<double> Vc_dist(0.0000757712088715733,2.5e-7/2);  // clearance volume, m3

                    double Bore=Bore_dist(rng);  //  bore, m
                    double Stroke=Stroke_dist(rng);  //  stroke, m
                    double Conrod=Conrod_dist(rng);  //  connecting rod, m
                    double Vc=Vc_dist(rng);  // clearance volume, m3

                    double Vd= M_PI*(Bore*Bore)*Stroke * 0.25; // displacement volume, m3
                    double CR=(Vd+Vc)/Vc; // calculated compression ratio, m3
                    double comb_eff = 1.0-(n_CO_EVO * M_CO * 10103.0+n_fuel_EVO * 13.82 * 42952.0)/(M_fuel * 42952.0); // combustion efficiency

                    // molar fractions at IVC and EVO
                    double X_fuel_IVC = n_fuel_IVC / n_IVC;
                    double X_O2_IVC   = n_O2_IVC / n_IVC;
                    double X_N2_IVC   = n_N2_IVC / n_IVC;
                    double X_H2O_IVC  = n_H2O_IVC / n_IVC;
                    double X_CO2_IVC  = n_CO2_IVC / n_IVC;
                    double X_CO_IVC   = n_CO_IVC / n_IVC;
                    double X_IC8H18_IVC  = X_IC8H18 * X_fuel_IVC;
                    double X_nC7H16_IVC  = X_nC7H16 * X_fuel_IVC;
                    double X_C6H5CH3_IVC = X_C6H5CH3 * X_fuel_IVC;
                    double X_C5H10_2_IVC = X_C5H10_2 * X_fuel_IVC;
                    double X_fuel_EVO = n_fuel_EVO / n_EVO;
                    double X_O2_EVO   = n_O2_EVO / n_EVO;
                    double X_N2_EVO   = n_N2_EVO / n_EVO;
                    double X_H2O_EVO  = n_H2O_EVO / n_EVO;
                    double X_CO2_EVO  = n_CO2_EVO / n_EVO;
                    double X_CO_EVO   = n_CO_EVO / n_EVO;
                    double X_IC8H18_EVO  = X_IC8H18 * X_fuel_EVO;
                    double X_nC7H16_EVO  = X_nC7H16 * X_fuel_EVO;
                    double X_C6H5CH3_EVO = X_C6H5CH3 * X_fuel_EVO;
                    double X_C5H10_2_EVO = X_C5H10_2 * X_fuel_EVO;

                    // mixture molar masses at IVC and EVO
                    double sum_products_IVC = X_fuel_IVC * M_fuel + X_O2_IVC * M_O2 + X_N2_IVC * M_N2 + X_H2O_IVC * M_H2O+ X_CO2_IVC * M_CO2 + X_CO_IVC * M_CO;
                    double sum_products_EVO = X_fuel_EVO * M_fuel + X_O2_EVO * M_O2 + X_N2_EVO * M_N2 + X_H2O_EVO * M_H2O+ X_CO2_EVO * M_CO2 + X_CO_EVO * M_CO;

                    // INPUTS
                    trng::normal_dist<double> RPM_dist(1200,12);  // engine speed
                    trng::normal_dist<double> T_residuals_dist(700,1);  // initial guess for T_residuals (will be refined through iterative loop)
                    trng::normal_dist<double> T_BDC_dist(370,1); // initial guess for T_BDC (will be refined through iterative loop)
                    trng::normal_dist<double> P_IVC_dist(Pressure[row_IVC][1],Pressure[row_IVC][2]);

                    double RPM=RPM_dist(rng);  // engine speed
                    double T_residuals=T_residuals_dist(rng);  // initial guess for T_residuals (will be refined through iterative loop)
                    double T_BDC=T_BDC_dist(rng); // initial guess for T_BDC (will be refined through iterative loop)
                    double P_IVC=P_IVC_dist(rng);

                    // Values from literature (SAE 2004-01-1900)
                    double vol_efficiency_base = 0.96; // Figures 3 and 4 (0.98 at 100 C, 0.96 at 60 C)
                    double sd_vol_efficiency_base = 0.001; // estimated
                    trng::normal_dist<double> vol_efficiency_dist(vol_efficiency_base,sd_vol_efficiency_base);
                    double vol_efficiency = vol_efficiency_dist(rng);
                    double T_IN_base = 273.15 + 107; // Figure 8
                    double sd_T_IN_base  = 1; // estimated
                    trng::normal_dist<double> T_IN_dist(T_IN_base,sd_T_IN_base);
                    double T_IN = T_IN_dist(rng);
                    double P_intake_base = 0.1; // MPa, intake pressure at which vol_efficiency_base and T_IN_base where calculated
                    double sd_P_intake_base = 2.8 * 0.001; // estimated, MPa
                    trng::normal_dist<double> P_intake_dist(P_intake_base,sd_P_intake_base);
                    double P_intake = P_intake_dist(rng);

                    double T_IN_eff, P_before_blowdown, P_after_blowdown, ca_sample_IVC, ca_sample_before_blowdown, ca_sample_after_blowdown, P_exhaust_TDC, P_BDC, pct_residuals, m_charge_measured, T_exhaust;
                    ca_sample_IVC = sample_angle_var(rng,CA_IVC,0.5);
                    ca_sample_before_blowdown = sample_angle_var(rng,Pressure[row_blowdown][0],0.5);
                    ca_sample_after_blowdown = sample_angle_var(rng,Pressure[row_after_blowdown][0],0.5);

                    trng::normal_dist<double> P_before_blowdown_dist(Pressure[row_blowdown][1],Pressure[row_blowdown][2]);
                    P_before_blowdown = P_before_blowdown_dist(rng);
                    trng::normal_dist<double> P_after_blowdown_dist(Pressure[row_after_blowdown][1],Pressure[row_after_blowdown][2]);
                    P_after_blowdown = P_after_blowdown_dist(rng);
                    trng::normal_dist<double> P_exhaust_TDC_dist(Pressure[row_TDC][1],Pressure[row_TDC][2]);
                    P_exhaust_TDC = (P_exhaust_TDC_dist(rng)); 
                    trng::normal_dist<double> P_BDC_dist(Pressure[row_intake_BDC][1],Pressure[row_intake_BDC][2]);
                    P_BDC = P_BDC_dist(rng);

                    int counter_T_IVC_loop = 0;

                    double mean_T_IVC_old = 470.0;
                    double st_T_IVC_old = 1.0 ;

                    double mean_T_IVC_new = 370.0;
                    double st_T_IVC_new = 1.0;
                    
                    trng::normal_dist<double> T_IVC_new_dist(mean_T_IVC_new,st_T_IVC_new);  // initial guess for T_IVC (will be refined through iterative loop)
                    trng::normal_dist<double> T_IVC_old_dist(mean_T_IVC_old,st_T_IVC_old);  // initial VALUE for T_IVC (will be refined through iterative loop)
                    double T_IVC_new=T_IVC_new_dist(rng);  // initial guess for T_IVC (will be refined through iterative loop)
                    double T_IVC_old=T_IVC_old_dist(rng);  // initial VALUE for T_IVC (will be refined through iterative loop)
                    double T_IVC;

                    while ( std::abs(T_IVC_old-T_IVC_new) > 0.01 * mean_T_IVC_old )// beginning of convergence loop on T_IVC
                    {
                        counter_T_IVC_loop =  counter_T_IVC_loop + 1;
	                if (counter_T_IVC_loop == 10)// if too many iteration, then exit
                        {
                           failed_temp_iteration = 1;
                           break;
                        }


                        T_IVC_old = T_IVC_new;

                        T_IVC = T_IVC_new;                     

                        // conservation of mass throughout the cycle
                        double n_IVC_real = Volume(ca_sample_IVC, Vd, CR, Conrod, Stroke) * P_IVC * 1e6/(8.3144621 * T_IVC); // number of moles at IVC
                        double m_IVC = n_IVC_real * sum_products_IVC;                       // mass at IVC (= mass at EVo)
                        double n_EVO_real = m_IVC / sum_products_EVO;                       // number of moles at EVO (mass IVC = mass EVO)

                        // Calculate T after blowdown (blowdown = EVO = 120 CAD)
                        double T_before_blowdown = P_before_blowdown * 1e6 * Volume(ca_sample_before_blowdown, Vd, CR, Conrod, Stroke)/(8.3144621 * n_EVO_real);
                        double CpR_before = X_O2_EVO * Cp_R(NASA_O2, T_before_blowdown, 6)
                            + X_CO2_EVO * Cp_R(NASA_CO2, T_before_blowdown, 7)
                            + X_CO_EVO * Cp_R(NASA_CO, T_before_blowdown, 7)
                            + X_H2O_EVO * Cp_R(NASA_H2O, T_before_blowdown, 7)
                            + X_N2_EVO * Cp_R(NASA_N2, T_before_blowdown, 6)
                            + X_IC8H18_EVO * Cp_R(NASA_IC8H18, T_before_blowdown, 7)
                            + X_nC7H16_EVO * Cp_R(NASA_nC7H16, T_before_blowdown, 7)
                            + X_C6H5CH3_EVO * Cp_R(NASA_C6H5CH3, T_before_blowdown,7)
                            + X_C5H10_2_EVO * Cp_R(NASA_C5H10_2, T_before_blowdown, 7);
                        double gamma_before = CpR_before / (CpR_before - 1);

                        double T_gamma = P_after_blowdown * 1e6 * Volume(ca_sample_after_blowdown, Vd, CR, Conrod, Stroke)/(8.3144621 * n_EVO_real); //T_gamma is an intermediate value created to calculate the gamma for the isentropic expansion
                        double CpR_after = X_O2_EVO * Cp_R(NASA_O2, T_gamma, 6)
                            + X_CO2_EVO * Cp_R(NASA_CO2, T_gamma, 7)
                            + X_CO_EVO * Cp_R(NASA_CO, T_gamma, 7)
                            + X_H2O_EVO * Cp_R(NASA_H2O, T_gamma, 7)
                            + X_N2_EVO * Cp_R(NASA_N2, T_gamma, 6)
                            + X_IC8H18_EVO * Cp_R(NASA_IC8H18, T_gamma, 7)
                            + X_nC7H16_EVO * Cp_R(NASA_nC7H16, T_gamma, 7)
                            + X_C6H5CH3_EVO * Cp_R(NASA_C6H5CH3, T_gamma,7)
                            + X_C5H10_2_EVO * Cp_R(NASA_C5H10_2, T_gamma, 7);

                        double gamma_after = CpR_after / (CpR_after - 1);
                        double gamma_blowdown = 0.5 * (gamma_after + gamma_before); // suggested by Jeremie
                        double T_after_blowdown = T_before_blowdown * pow( P_after_blowdown / P_before_blowdown,(gamma_blowdown - 1) / gamma_blowdown);

                        // calculate T residuals
                        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        // iteration on T_residuals
                        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        for(int i = 0; i < 5; ++i)
                        {
                            T_gamma =  (T_residuals + T_after_blowdown)/2; //T_gamma is an intermediate input created to calculate the gamma for the isentropic expansion

                            double CpR_exhaust = X_O2_EVO * Cp_R(NASA_O2, T_gamma, 6)
                                + X_CO2_EVO * Cp_R(NASA_CO2, T_gamma, 7)
                                + X_CO_EVO * Cp_R(NASA_CO, T_gamma, 7)
                                + X_H2O_EVO * Cp_R(NASA_H2O, T_gamma, 7)
                                + X_N2_EVO * Cp_R(NASA_N2, T_gamma, 6)
                                + X_IC8H18_EVO * Cp_R(NASA_IC8H18, T_gamma, 7)
                                + X_nC7H16_EVO * Cp_R(NASA_nC7H16, T_gamma, 7)
                                + X_C6H5CH3_EVO * Cp_R(NASA_C6H5CH3, T_gamma,7)
                                + X_C5H10_2_EVO * Cp_R(NASA_C5H10_2, T_gamma, 7);

                            double gamma_exhaust = CpR_exhaust / (CpR_exhaust - 1);
                            T_exhaust = T_after_blowdown * pow( P_IVC / P_after_blowdown , (gamma_exhaust - 1) / gamma_exhaust);  // suggested by Jeremie
                            T_residuals = 0.5 * (T_after_blowdown + T_exhaust);
                        } // end of "i" loop

                        // calculate rho_residuals, in mol/m3
                        double rho_residuals = P_exhaust_TDC * 1e6/(8.3144621 * T_residuals);

                        // Calculate m_residuals, in gramms
                        double m_residuals = rho_residuals * Vc * sum_products_EVO;

                        // Calculate m charge reference, in gramms
                        double m_charge_ref = vol_efficiency_base * Vd * 28.8 * P_intake * 1e6 / (8.3144621 * T_IN_base);

                        // fresh mass flowing into engine, in gramms
                        m_charge_measured = 2 * ((q_fuel + q_air)/(1-egr)) / (RPM/60); // updated equation for m_charge_measured

                        // calculate %residuals
                        pct_residuals = (m_residuals)/(m_charge_measured + m_residuals);

                        // Calculate T_IN,effective (Temperature at BDC assuming no residuals)
                        T_IN_eff = T_IN_base * m_charge_ref * sum_products_IVC * P_BDC /(m_charge_measured * 28.8 * P_intake);

                        // Calculate T_BDC (Temperature at BDC taking into account the residuals)
                        //!!!!!!!!!!!!!!!!!!!!!!!
                        // iteration on T_BDC
                        //!!!!!!!!!!!!!!!!!!!!!!!!
                        for (int i = 0; i < 5; ++i)
                        {
                            double T1 = (T_IN_eff +T_BDC)/2;
                            double CpR_airfuel = X_O2_IVC * Cp_R(NASA_O2, T1, 6)
                                + X_CO2_IVC * Cp_R(NASA_CO2, T1, 7)
                                + X_CO_IVC * Cp_R(NASA_CO, T1, 7)
                                + X_H2O_IVC * Cp_R(NASA_H2O, T1, 7)
                                + X_N2_IVC * Cp_R(NASA_N2, T1, 6)
                                + X_IC8H18_IVC * Cp_R(NASA_IC8H18, T1, 7)
                                + X_nC7H16_IVC * Cp_R(NASA_nC7H16, T1, 7)
                                + X_C6H5CH3_IVC * Cp_R(NASA_C6H5CH3, T1,7)
                                + X_C5H10_2_IVC * Cp_R(NASA_C5H10_2, T1, 7);

                            double T2 = (T_residuals +T_BDC)/2;
                            double CpR_residuals = X_O2_EVO * Cp_R(NASA_O2, T2, 6)
                                + X_CO2_EVO * Cp_R(NASA_CO2, T2, 7)
                                + X_CO_EVO * Cp_R(NASA_CO, T2, 7)
                                + X_H2O_EVO * Cp_R(NASA_H2O, T2, 7)
                                + X_N2_EVO * Cp_R(NASA_N2, T2, 6)
                                + X_IC8H18_EVO * Cp_R(NASA_IC8H18, T2, 7)
                                + X_nC7H16_EVO * Cp_R(NASA_nC7H16, T2, 7)
                                + X_C6H5CH3_EVO * Cp_R(NASA_C6H5CH3, T2,7)
                                + X_C5H10_2_EVO * Cp_R(NASA_C5H10_2, T2, 7);

                            double CV_airfuel  = CpR_airfuel - 1;    //  Cv at (Tin+Tbdc)/2  : correction by Jeremie. Ideal gas constant "R" is omitted here on purpose
                            double CV_residuals = CpR_residuals - 1; //  Cv at (Tres+Tbdc)/2 : correction by Jeremie. Ideal gas constant "R" is omitted here on purpose
                            T_BDC = (T_IN_eff * m_charge_measured * CV_airfuel+T_residuals * m_residuals * CV_residuals)/(m_charge_measured * CV_airfuel+m_residuals * CV_residuals);
                        }// end of "i" loop

                        // Calculate T_IVC (Temperature at IVC assuming isentropic compression from BDC)
                        double CpR_intake_BDC = X_O2_IVC * Cp_R(NASA_O2, T_BDC, 6)
                            + X_CO2_IVC * Cp_R(NASA_CO2, T_BDC, 7)
                            + X_CO_IVC * Cp_R(NASA_CO, T_BDC, 7)
                            + X_H2O_IVC * Cp_R(NASA_H2O, T_BDC, 7)
                            + X_N2_IVC * Cp_R(NASA_N2, T_BDC, 6)
                            + X_IC8H18_IVC * Cp_R(NASA_IC8H18, T_BDC, 7)
                            + X_nC7H16_IVC * Cp_R(NASA_nC7H16, T_BDC, 7)
                            + X_C6H5CH3_IVC * Cp_R(NASA_C6H5CH3, T_BDC,7)
                            + X_C5H10_2_IVC * Cp_R(NASA_C5H10_2, T_BDC, 7);

                        double gamma_intake_BDC = CpR_intake_BDC / (CpR_intake_BDC - 1);
                        T_IVC_new = T_BDC * pow(P_IVC/P_BDC,((gamma_intake_BDC-1)/gamma_intake_BDC));

                    }// end of convergence loop on T_IVC 
                    if(failed_temp_iteration) {
                      break;
                    }
                    printf("n_sample %d n_vector %d,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,%3f,\n", n_samples_found_global, n_samples_total, T_IN_eff,T_BDC,T_IVC,egr,res,pct_residuals,wr,condenser_eff_fuel_IVC,condenser_eff_water_IVC,condenser_eff_fuel_EVO,condenser_eff_water_EVO,phi_fresh, X_O2_IVC, X_N2_IVC, X_CO2_IVC, X_CO_IVC,X_H2O_IVC,X_O2_EVO,X_N2_EVO,X_CO2_EVO,X_CO_EVO,X_H2O_EVO,comb_eff,T_residuals,T_exhaust,m_charge_measured,n_CO_IVC,n_CO2_IVC,n_O2_IVC,n_N2_IVC,n_CO_EVO,n_CO2_EVO,n_O2_EVO,n_N2_EVO,RPM,b,r,n_fuel_IVC,n_fuel_EVO,rCO,rTHC,X_CO2_IVC_measured,X_O2_EVO_measured,X_CO2_EVO_measured,X_CO_EVO_measured,X_THC_EVO_measured,sum_products_IVC,sum_products_EVO, X_IC8H18_IVC,X_nC7H16_IVC,X_C6H5CH3_IVC,X_C5H10_2_IVC,X_IC8H18_EVO,X_nC7H16_EVO,X_C6H5CH3_EVO,X_C5H10_2_EVO);
                    //update means (running means and stddevs)
                    sample[0] = T_IN_eff;
                    sample[1] = T_BDC;
                    sample[2] = T_IVC;
                    sample[3] = egr;
                    sample[4] = res;
                    sample[5] = pct_residuals;
                    sample[6] = wr;
                    sample[7] = condenser_eff_fuel_IVC;
                    sample[8] = condenser_eff_water_IVC;
                    sample[9] = condenser_eff_fuel_EVO;
                    sample[10] = condenser_eff_water_EVO;
                    sample[11] = phi_fresh;
                    sample[12] = X_O2_IVC;
                    sample[13] = X_N2_IVC;
                    sample[14] = X_CO2_IVC;
                    sample[15] = X_CO_IVC;
                    sample[16] = X_H2O_IVC;
                    sample[17] = X_O2_EVO;
                    sample[18] = X_N2_EVO;
                    sample[19] = X_CO2_EVO;
                    sample[20] = X_CO_EVO;
                    sample[21] = X_H2O_EVO;
                    sample[22] = comb_eff;
                    sample[23] = T_residuals;            
                    sample[24] = T_exhaust;
                    sample[25] =  m_charge_measured;            
                    sample[26] = n_CO_IVC;
                    sample[27] = n_CO2_IVC;
                    sample[28] = n_O2_IVC;
                    sample[29] = n_N2_IVC;
                    sample[30] = n_CO_EVO;
                    sample[31] = n_CO2_EVO;
                    sample[32] = n_O2_EVO;
                    sample[33] = n_N2_EVO;
                    sample[34] = RPM;
                    sample[35] = b;
                    sample[36] = r;                  
                    sample[37] = n_fuel_IVC;
                    sample[38] = n_fuel_EVO;
                    sample[39] = rCO;
                    sample[40] = rTHC;                            
                    sample[41] = X_CO2_IVC_measured;
                    sample[42] = X_O2_EVO_measured;
                    sample[43] = X_CO2_EVO_measured;
                    sample[44] = X_CO_EVO_measured;
                    sample[45] = X_THC_EVO_measured;
                    sample[46] = sum_products_IVC;
                    sample[47] = sum_products_EVO;
                    sample[48] = X_IC8H18_IVC;
                    sample[49] = X_nC7H16_IVC;
                    sample[50] = X_C6H5CH3_IVC;
                    sample[51] = X_C5H10_2_IVC;
                    sample[52] = X_IC8H18_EVO;
                    sample[53] = X_nC7H16_EVO;
                    sample[54] = X_C6H5CH3_EVO;
                    sample[55] = X_C5H10_2_EVO;
		 
                    n_samples_found += 1;
                    update_moments(n_samples_found,sample,means,variances,stddevs);
                }// accepted sample, end of the if down selection
                else
                {
                    rng.jump(random_numbers_per_accepted);
                }
            } // n_vector
            rng.jump(random_numbers_per_loop*(batch_size-rank_start[rank+1]));
            MPI_Allreduce(&n_samples_found,&n_samples_found_global,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
            MPI_Allreduce(&failed_temp_iteration, &failed_temp_iteration_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if(failed_temp_iteration > 0) {
              if(rank == 0) {
                printf("Failed temperature convergence.\nQuitting.\n");
                exit(2);
              }
            }
        }// end of while(n_samples_found_global < n_samples_wanted) loop
        n_samples_total = n_batches*batch_size;
        int n_elems = means.size();
        //Update moments globally
        std::vector<long int> n_samples_per_rank;
        std::vector<double> means_per_rank;
        std::vector<double> variances_per_rank;
        std::vector<double> stddevs_per_rank;
        if(rank == 0)
        {
            n_samples_per_rank.resize(nprocs);
            means_per_rank.resize(n_elems*nprocs);
            variances_per_rank.resize(n_elems*nprocs);
            stddevs_per_rank.resize(n_elems*nprocs);
        }
        MPI_Gather(&n_samples_found, 1, MPI_LONG, &n_samples_per_rank[0], 1, MPI_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&means[0], n_elems, MPI_DOUBLE, &means_per_rank[0], n_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&variances[0], n_elems, MPI_DOUBLE, &variances_per_rank[0], n_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&stddevs[0], n_elems, MPI_DOUBLE, &stddevs_per_rank[0], n_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if(rank ==0)
        {
            for(int iproc = 1; iproc < nprocs; ++iproc)
            {
                std::vector<double> proc_means(&means_per_rank[n_elems*iproc],&means_per_rank[n_elems*(iproc+1)]);
                std::vector<double> proc_variances(&variances_per_rank[n_elems*iproc],&variances_per_rank[n_elems*(iproc+1)]);
                std::vector<double> proc_stddevs(&stddevs_per_rank[n_elems*iproc],&stddevs_per_rank[n_elems*(iproc+1)]);
                update_moments_global(n_samples_per_rank[iproc],proc_means,proc_variances,n_samples_found,means,variances,stddevs);
                n_samples_found += n_samples_per_rank[iproc];
            }
            assert(n_samples_found == n_samples_found_global);
            stdev_pct_res = stddevs[5];
            mean_pct_res = means[5];
        }
        MPI_Bcast(&mean_pct_res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(&stdev_pct_res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(&stddevs[0],stddevs.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    } // end of convergence loop on pct_residuals
    if(rank==0)
    {
        printf("N_SAMPLE %d, %ld, %ld",  n_samples_wanted, n_samples_found_global, n_samples_total);
        printf(" MEANS ");
        print_vector(means);
        printf("\n");
        printf("N_SAMPLE %d, %ld, %ld",  n_samples_wanted, n_samples_found_global, n_samples_total);
        printf(" STDDEVS ");
        print_vector(stddevs);
        printf("\n");
    }
    MPI_Finalize();
    return 0;
}