#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

#include "wv3d.h"

// #include "mrc.h"

void wave3dChp(float *data, 
			   int nx, int ny, int nz,
			   float *shft, 
			   int nx2, int ny2, int nz2,
			   int ordx, int ordy, int ordz)///ifChop  , const bool chop)
{
  //TODO CHECK
  // check    data and step3 same size

  const int nxyz =  nx * ny * nz;

  float *wvl = new float[nxyz];
  float *step3 = new float[nxyz];

  //float *shft = new float[nxyz];  // moss addition
  const int nshftx = ordx/2;
  const int nshfty = ordy/2;
  const int nshftz = ordz/2;


  fprintf(stderr, "part1 (of 2; 0<=z<%d): ", nz);
  for(int zz=0;zz<nz;zz++) {
	fprintf(stderr, "%d ",zz);
    wvl3d_part1(wvl, nx, ny, ordx, ordy, zz, data);///ifChop  , chop);
  }

  fprintf(stderr, "\npart2 (of 2): \n");
  wvl3d_part2_3(step3, nx, ny, nz,  
		ordz, wvl);///ifChop  ,  chop);

  // shift by ord/2 (for x,y, and z) to get overlay correct 
  //TODO - FIXME for even ord 
  cerr << "shftx, shfty, shftz= " << nshftx << " "<< nshfty << " "<< nshftz << endl;
  {
  for(int z=0;z<nz;z++)
	for(int y=0;y<ny;y++)
	  for(int x=0;x<nx;x++)
		shft[ IJK(x,y,z) ] =0;
  }
  for(int z=0;z<nz-nshftz;z++)
	for(int y=0;y<ny-nshfty;y++)
	  for(int x=0;x<nx-nshftx;x++)
		shft[ IJK(x+nshftx,y+nshfty,z+nshftz) ] = step3[ IJK(x,y,z) ];
    

  
  delete [] wvl;
  delete [] step3;
  //delete [] shft;  // willy addition
}
