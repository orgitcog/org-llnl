// -*- c++ -*-

%module willy

%include sebTypemaps.i

%{
#include "willy.h"
%}

template <class T> 
void grow(T *array3d, int nx, int ny, int nz,
		  T *array3d, int nx, int ny, int nz, T skirt=1)  throw(char *);

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

%template(binarize) binarize<Byte>;
%template(binarize) binarize<short>;
%template(binarize) binarize<float>;
%template(binarize) binarize<Word>;
%template(binarize) binarize<long>;

%template(grow) grow<Byte>;
%template(grow) grow<short>;
%template(grow) grow<float>;
%template(grow) grow<Word>;
%template(grow) grow<long>;

%template(writeVol) writeVol<Byte>;
%template(writeVol) writeVol<short>;
%template(writeVol) writeVol<float>;
%template(writeVol) writeVol<Word>;
%template(writeVol) writeVol<long>;

%template(readPointsToVol) readPointsToVol<Byte>;
%template(readPointsToVol) readPointsToVol<short>;
%template(readPointsToVol) readPointsToVol<float>;
%template(readPointsToVol) readPointsToVol<Word>;
%template(readPointsToVol) readPointsToVol<long>;



void medianer(float *array3d, 
			  int nx, int ny, int nz, 
			  float threshold,
			  float *OUTPUT, float *OUTPUT);
//			  float *med, float *meddev);

// TODO TODO TODO FIXME FIXME
//  //  //  void segmenter(
//  //  //  			   short **p_Out1d, int *p_nx, //oldLableOf
//  //  //  			   short **p_Out1d, int *p_nx, //newLableFor

//  //  //  			   short *array3d, 
//  //  //  			   int nx, int ny, int nz,
//  //  //  			   short *array3d, 
//  //  //  			   int nx, int ny, int nz,
//  //  //  			   int *OutValue);
//  void fastwv(float *array3d, 
//  			int nx, int ny, int nz,
//  			float *array3d, 
//  			int nx, int ny, int nz,
//  			int ordx, int ordy, int ordz);
//  void fastwv2(float *array3d, 
//  			int nx, int ny, int nz,
//  			float *array3d, 
//  			int nx, int ny, int nz,
//  			int ordx, int ordy, int ordz);
//  void fastwv3(float *array3d, 
//  			int nx, int ny, int nz,
//  			float *array3d, 
//  			int nx, int ny, int nz,
//  			int ordx, int ordy, int ordz);
//  void fastwv4(float *array3d, 
//  			int nx, int ny, int nz,
//  			float *array3d, 
//  			int nx, int ny, int nz,
//  			int ordx, int ordy, int ordz);
//  void fastwv5(float *array3d, 
//  			 int nx, int ny, int nz,
//  			 float *array3d, 
//  			 int nx, int ny, int nz,
//  			 int ordx, int ordy, int ordz,
//  			 const bool prints=0);
void fastwv6(float *array3d, 
			 int nx, int ny, int nz,
			 float *array3d, 
			 int nx, int ny, int nz,
			 int ordx, int ordy, int ordz,
			 const bool prints=0);
//  void fastwv66(float *array3d, 
//  			 int nx, int ny, int nz,
//  			 float *array3d, 
//  			 int nx, int ny, int nz,
//  			 int ordx, int ordy, int ordz,
//  			 const bool prints=0);


void wave3dChp(float *array3d, 
			   int nx, int ny, int nz,
			   float *array3d, 
			   int nx, int ny, int nz,
			   int ordx, int ordy, int ordz);///ifChop  , const bool chop=1);

void discriminator(float *array3d, //inVol, 
				   int nx, int ny, int nz,
				   float *array3d, //outVol, 
				   int nx, int ny, int nz,
				   short *array3d,
				   int nx, int ny, int nz,
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

