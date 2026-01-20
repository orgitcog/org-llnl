#include "sebInclude.h"

template <class T> 
void grow(T *array3d, int nx, int ny, int nz,
		  T *array3d_2, int nx_2, int ny_2, int nz_2, T skirt=1) throw(char *);


template <class T> 
void writeVol(const char *outfn, T *array3d, int nx, int ny, int nz);

template <class T> 
void readPointsToVol(const char *fn, 
 					 T *array3d, int nx, int ny, int nz,
 					 double xmin, double xside,
 					 double ymin, double yside,
 					 double zmin, double zside);

template <class T> 
void binarize(T *array3d, int nx, int ny, int nz, T threshold);

void medianer(float *array3d, 
			  int nx, int ny, int nz, 
			  float threshold,
			  float *med, float *mdedev);

void segmenter(
			   short **p_Out1dA, int *p_nxA, //oldLableOf
			   short **p_Out1dB, int *p_nxB, //newLableFor
			   
			   short *imgBIN, 
			   const int nx1, const int ny1, const int nz1,
			   short *lableIMG, 
			   const int nx2, const int ny2, const int nz2,
			   int *numObjs);

//  void fastwv(float *data, 
//  			int nx, int ny, int nz,
//  			float *step3, 
//  			int nx2, int ny2, int nz2,
//  			int ordx, int ordy, int ordz);
//  void fastwv2(float *data, 
//  			int nx, int ny, int nz,
//  			float *step3, 
//  			int nx2, int ny2, int nz2,
//  			int ordx, int ordy, int ordz);
//  void fastwv3(float *data, 
//  			int nx, int ny, int nz,
//  			float *step3, 
//  			int nx2, int ny2, int nz2,
//  			int ordx, int ordy, int ordz);
//  void fastwv4(float *data, 
//  			int nx, int ny, int nz,
//  			float *step3, 
//  			int nx2, int ny2, int nz2,
//  			int ordx, int ordy, int ordz);
//  void fastwv5(float *data, 
//  			 int nx, int ny, int nz,
//  			 float *step3, 
//  			 int nx2, int ny2, int nz2,
//  			 int ordx, int ordy, int ordz,
//  			 const bool prints=0);
void fastwv6(float *data, 
			 int nx, int ny, int nz,
			 float *step3, 
			 int nx2, int ny2, int nz2,
			 int ordx, int ordy, int ordz,
			 const bool prints=0);
//  void fastwv66(float *data, 
//  			 int nx, int ny, int nz,
//  			 float *step3, 
//  			 int nx2, int ny2, int nz2,
//  			 int ordx, int ordy, int ordz,
//  			 const bool prints=0);

void wave3dChp(float *data, 
			   int nx, int ny, int nz,
			   float *step3, 
			   int nx2, int ny2, int nz2,
			   int ordx, int ordy, int ordz);///ifChop  , const bool chop=1);

void discriminator(float *img, //inVol, 
				   int nx, int ny, int nz,
				   float *img2, //outVol, 
				   int nx2, int ny2, int nz2,
				   short *outLableIMG,
				   int nx3, int ny3, int nz3,
				   int order, float thrsh, float offset, 
				   bool zdis, 
				   double deltax = 1, double deltay = 1,  double deltaz = 1);


double willyStdDevMeasure(float *array3d,
						  int nx, int ny, int nz);

void writeWavletVTK(float *array3d, 
					int nx, int ny, int nz,
					int ordx, int ordy, int ordz,
					float dx, float dy, float dz, 
					char *unit,
					char *fn);

