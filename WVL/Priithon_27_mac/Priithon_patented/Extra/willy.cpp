#include "willy.h"
#include <math.h>
#include <fstream>
#include <iostream>
using namespace std;


inline float sqr(const float d) { return d*d; }


template <class T>
void binarize(T *img3DStack, 
			  int nx, int ny, int nz,
			  T threshold)
{
  T *p = img3DStack;
  for(int i=0;i<nx*ny*nz;i++)
  {
	  T &f = *p++;
      f = (f >= threshold) ? 1 : 0;
    }
}
template void binarize<Byte> (Byte  *img, int nx, int ny, int nz, Byte  threshold);
template void binarize<short>(short *img, int nx, int ny, int nz, short threshold);
template void binarize<float>(float *img, int nx, int ny, int nz, float threshold);
template void binarize<Word> (Word  *img, int nx, int ny, int nz, Word  threshold);
template void binarize<long> (long  *img, int nx, int ny, int nz, long  threshold);


double willyStdDevMeasure(float *wvl, 
						 int nx, int ny, int nz)
{
  const int nxyz = nx * ny *nz;

  double sum = 0;
  int N = 0; // num of # > 0
  for(int i=0;i<nxyz;i++)
    {
      const double d = wvl[ i ];
      if(d>10e-10) {
	sum += d;
	N++;
      }
    }
  
  double avg = sum / N;
  double var = 0;
  
  {for(int i=0;i<nxyz;i++)
    {
      const double d = wvl[ i ];
      if(d>10e-10) {
	var += sqr(avg - d);
      }
  }}
  var /= N-1;
  
  return sum * sqrt( var );
}

#define IJK(i, j, k) ((i) + nx * ((j) + ny*(k)))

void writeWavletVTK(float *img3DStack, 
					int nx, int ny, int nz,
					int ordx, int ordy, int ordz,
					float dx, float dy, float dz, 
					char *unit,
					char *fn)
{
  const float origx = 1.+ordx/2.;
  const float origy = 1.+ordy*dy/(2.*dx);
  const float origz = 1.+ordz*dz/(2.*dx);
  
  ofstream outf(fn);
  
  outf << "# vtk DataFile Version 2.0" << endl;
  outf << "filename: " << fn << " dx,dy,dz: " << dx << " " << dy << " " << dz << unit << endl; 
  outf << "ASCII" << endl;
  outf << endl;
  outf << "DATASET STRUCTURED_POINTS" << endl;
  outf << "DIMENSIONS " << nx << " " << ny << " " << nz << endl;
  outf << "ORIGIN " << origx << " " << origy << " " << origz << endl;
  outf << "SPACING " << 1    << " " << dy/dx << " " << dz/dx << endl;
  outf << endl;
  outf << "POINT_DATA " << nx*ny*nz << endl;
  outf << "SCALARS scalars float" << endl;
  outf << "LOOKUP_TABLE default" << endl;
  outf << endl;
    
  for(int z=0;z<nz;z++)
	{
	  for(int y=0;y<ny;y++)
		{
		  for(int x=0;x<nx;x++)
			{
			  outf << img3DStack[ IJK(x,y,z) ] << " ";
		
			  if(!outf) {
				cerr << "write in error at x,y,z = " 
					 <<  x << " " << y << " " << z <<  endl;
				return;
			  }
			}
		}
	  outf << endl;
	}
}
