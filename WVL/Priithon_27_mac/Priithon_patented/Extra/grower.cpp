#include "sebInclude.h"

//#include <fstream.h>
//  #include <stdlib.h>
//  #include <stdio.h>
#include <fstream>
#include <iostream>
using namespace std;

#define IJK(i, j, k) ((i) + nx * ((j) + ny*(k)))

//  int nx=0;
//  int ny=0;
//  int nz=0;
//  int nxyz=0;


//  //  //  //  double xmin = -2.0;  // -1.4
//  //  //  //  double ymin = -2.0;
//  //  //  //  double zmin = -0.2;
//  //  //  //  double xside = 4.0;  // 2.8
//  //  //  //  double yside = 4.0;
//  //  //  //  double zside = 3.8; // 2.6

template <class T>
void readPointsToVol(const char *fn, 
					 T *vol, int nx, int ny, int nz,
					 double xmin, double xside,
					 double ymin, double yside,
					 double zmin, double zside)
{
  int nxyz=nx*ny*nz;
  for (int i=0; i< nxyz; i++)
    vol[ i ] = 0;
  
  ifstream inf(fn);
  if(!inf) {
    cerr << " could NOT open file \""<<fn<<"\"!" << endl;
    return;
  }
  cerr << "loading file...";
  
  int c_nx = 0;
  int c_ny = 0;
  int c_nz = 0;
  int c_x0 = 0;
  int c_y0 = 0;
  int c_z0 = 0;

  while(!inf.eof())
    {
      double x,y,z;
      inf >> x >> y >> z;
      
      int i = int( nx * (x - xmin) / xside );
      int j = int( ny * (y - ymin) / yside );
      int k = int( nz * (z - zmin) / zside );
      
      if(i>=nx) { i = nx-1; c_nx++; }
      else if(i<0) { i = 0; c_x0++; }
      if(j>=ny) { j = ny-1; c_ny++; }
      else if(j<0) { j = 0; c_y0++; }
      if(k>=nz) { k = nz-1; c_nz++; }
      else if(k<0) { k = 0; c_z0++; }
      
      vol[ IJK(i,j,k) ] = 1;
    }
  if(c_nx > 0) cerr << c_nx << " points beyond nx" << endl;
  if(c_ny > 0) cerr << c_ny << " points beyond ny" << endl;
  if(c_nz > 0) cerr << c_nz << " points beyond nz" << endl;
  if(c_x0 > 0) cerr << c_x0 << " points beyond x0" << endl;
  if(c_y0 > 0) cerr << c_y0 << " points beyond y0" << endl;
  if(c_z0 > 0) cerr << c_z0 << " points beyond z0" << endl;

  cerr << "done."<< endl;
}

template <class T>
void writeVol(const char *outfn, T *vol, int nx, int ny, int nz)
{
  ofstream outf(outfn);
  
  cerr << "saving file...";
  for(int z=0;z<nz;z++)
    {
      for(int y=0;y<ny;y++)
	{
	  for(int x=0;x<nx;x++)
	    {
	      outf << vol[ IJK(x,y,z) ] << " ";
	      
	      if(!outf) {
			cerr << "write in error at x,y,z = " <<
			  x << " " << y << " " << z <<  endl;
			return;
	      }
	    }
	}
      outf << endl;
    }
  cerr << "done."<< endl;
}

#include <stdio.h>

template <class T>
void grow(T *v1, int nx, int ny, int nz,
		  T *v2, int nx2, int ny2, int nz2, T skirt)
{
  if(skirt <=0)
	throw (char *)"skirt should _really_ be positive - your bad...";
  /////printf("skirt should _really_ be positive - your bad...\n");

  for(int z=0;z<nz;z++)
    {
      int zmin = z-1;
      if(zmin<0) zmin=0;
      int zmax = z+1;
      if(zmax==nz) zmax=nz-1;
	  
      for(int y=0;y<ny;y++)
		{
		  int ymin = y-1;
		  if(ymin<0) ymin=0;
		  int ymax = y+1;
		  if(ymax==ny) ymax=ny-1;
		  
		  for(int x=0;x<nx;x++)
			{
			  int xmin = x-1;
			  if(xmin<0) xmin=0;
			  int xmax = x+1;
			  if(xmax==nx) xmax=nx-1;
			  
			  T &centervox = v1[ IJK(x,y,z) ];
			  if(centervox > 0) 
				{
				  v2[ IJK(x,y,z) ] = centervox;
				  for(int zz=zmin;zz<=zmax;zz++)
					for(int yy=ymin;yy<=ymax;yy++)
					  for(int xx=xmin;xx<=xmax;xx++)
						{
						  T &potvox = v2[ IJK(xx,yy,zz) ];
						  if(potvox == 0 &&
							 !(xx==x && yy==y && zz==z)) {
							potvox = skirt;
						  }
						}
				}
			}
		}
    }
}

template void writeVol<Byte> (const char *outfn, Byte  *img, int nx, int ny, int nz);
template void writeVol<short>(const char *outfn, short *img, int nx, int ny, int nz);
template void writeVol<float>(const char *outfn, float *img, int nx, int ny, int nz);
template void writeVol<Word> (const char *outfn, Word  *img, int nx, int ny, int nz);
template void writeVol<long> (const char *outfn, long  *img, int nx, int ny, int nz);

template void grow<Byte> (Byte  *img, int nx, int ny, int nz,
						  Byte  *img2, int nx2, int ny2, int nz2, Byte skirt);
template void grow<short>(short *img, int nx, int ny, int nz,
						  short *img2, int nx2, int ny2, int nz2, short skirt);
template void grow<float>(float *img, int nx, int ny, int nz,
						  float *img2, int nx2, int ny2, int nz2, float skirt);
template void grow<Word> (Word  *img, int nx, int ny, int nz,
						  Word  *img2, int nx2, int ny2, int nz2, Word skirt);
template void grow<long> (long  *img, int nx, int ny, int nz,
						  long  *img2, int nx2, int ny2, int nz2, long skirt);


template void readPointsToVol<Byte> (const char *fn, 
									 Byte  *img, int nx, int ny, int nz,
									 double xm, double xs, double ym, double ys, double zm, double zs);
template void readPointsToVol<short>(const char *fn, 
									 short *img, int nx, int ny, int nz,
									 double xm, double xs, double ym, double ys, double zm, double zs);
template void readPointsToVol<float>(const char *fn, 
									 float *img, int nx, int ny, int nz,
									 double xm, double xs, double ym, double ys, double zm, double zs);
template void readPointsToVol<Word> (const char *fn, 
									 Word  *img, int nx, int ny, int nz,
									 double xm, double xs, double ym, double ys, double zm, double zs);
template void readPointsToVol<long> (const char *fn, 
									 long  *img, int nx, int ny, int nz,
									 double xm, double xs, double ym, double ys, double zm, double zs);

