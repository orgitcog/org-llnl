#include "sebInclude.h"
#include <iostream.h>
#include <stdio.h>


/// ****************** same as fastwv6 BUT with periodic boundary cond. - instead of set border to 0


#define IJK(x, y, z) ((x) + nx * ((y) + ny*(z)))
inline float sqr(const float d) { return d*d; }

// using new definitino of psi
// but "special" insofar that I multiply psi by 2
//  and do the '*2' multipl. as 2 additions (see below)
inline int fpsi(const int ord, const int i, const int j) {
  const int ord2 = ord*2;
  if(j-ord  < i  && i <= j)     return  2;
  if(j-ord2 < i  && i <= j)     return -1;
  if(    j <= i  && i <= j+ord) return -1;
  return 0;
}

// same as 2D but having a (untouched) 3rd index in data
static 
void fwvl3d_part1(float *wvl,
				 const int nx, const int ny,
				 const int ordx, const int ordy, 
				 const int zz, 
				 const float * data)///ifChop  , const bool chop)
{
  const float ysfac = 1.0 / ordx / 2.0;  // 2 because of my special psi
  const float xsfac = 1.0 / ordy / 2.0;  // 2 because of my special psi
  const float wvlNorm = ysfac*xsfac;

  const int nshftx = ordx/2;
  const int nshfty = ordy/2;
  
  const int ordy3 = 3*ordy; // support size of wavelet kernel
  float *ysSumParts = new float[ordy3];

//66  
//    //zero first (not calculated) sects
//    for(int y=0;y<ordy+nshfty;y++) 
//  	for(int x=0;x<ordx+nshftx;x++) 
//  	  wvl[ IJK(x,y,zz) ] = 0;

//    //zero last (not calculated) sects
//    {for(int y=ny-2*ordy+nshfty;y<ny;y++) 
//  	for(int x=nx-2*ordx+nshftx;x<nx;x++) 
//  	  wvl[ IJK(x,y,zz) ] = 0;
//    }

  for(int x=0;x<nx;x++)//66                     // j: x-direction
	{
	  int xsmin =  x-ordx;
	  int xsmax =  x+2*ordx;
	  if(xsmin<0)   xsmin += nx; //66
	  if(xsmax>=nx) xsmax -= nx; //66
	  float yssum = 0; // y-sum: first full calc - then only updated

	  //first full calc of yssums
	  {
		int y = ordy;
		
		for(int ys=0;ys<ordy3;ys++)          // k: y-direction
		  {
			float xssum = 0;
			  
			int xs = xsmin;
xxxxxxxx			const float *dp = data + IJK(xs,ys,zz);              ///////////////// 66 FIXME from here
			{for(int i=0;i<ordx;i++)
			  xssum -= *dp++;
			}{for(int i=0;i<ordx;i++)
			  xssum += 2* *dp++;
			}{for(int i=0;i<ordx;i++)
			  xssum -= *dp++;
			}
			if(xssum < 0) xssum = 0;    //    moss CHOP addition
			ysSumParts[ys] = xssum;
			
			const int psi1 = fpsi(ordy ,y,ys);//, k,i);
			yssum += psi1 * xssum;
		  }

		//66
		int xxx= x+nshftx;
		if(xxx>=nx) xxx -=nx;
		int yyy= y+nshfty;
		if(yyy>=ny) yyy -=ny;

		if(yssum < 0) wvl[ IJK(xxx,yyy,zz) ] = 0;    //    moss CHOP addition
		else          wvl[ IJK(xxx,yyy,zz) ] = yssum * wvlNorm;
	  }

	  int ysSumPartsBottomIdx = 0;
	  //following y just update 

	  //66 for(int y=ordy+1;y<ny-2*ordy;y++)
	  for(int y=0;y<ny;y++)
		{
		  int ysspI = ysSumPartsBottomIdx;

		  yssum += ysSumParts[ysspI]; // update(/compensate) -1  to 0
		  ysspI+=ordy; if(ysspI >= ordy3) ysspI-=ordy3;
		  yssum -= 3*ysSumParts[ysspI]; // update(/compensate) +2  to -1
		  ysspI+=ordy; if(ysspI >= ordy3) ysspI-=ordy3;
		  yssum += 3*ysSumParts[ysspI]; // update(/compensate) -1  to +2
		  // ysspI+=ordy; if(ysspI >= ordy3) ysspI-=ordy3;
		  // true: ysspI = ysSumPartsBottomIdx
		  
		  // calc (the only) ONE new(!) xssum
		  //yssum += ysSumParts[ysspI]; // update(/compensate) 0  to -1
		  float xssum = 0;
		  const int xs = xsmin;
		  const int ys = y + 2*ordy -1;
		  const float *dp = data + IJK(xs,ys,zz);
		  {for(int i=0;i<ordx;i++)
			xssum -= *dp++;
		  }{for(int i=0;i<ordx;i++)
			xssum += 2* *dp++;
		  }{for(int i=0;i<ordx;i++)
			xssum -= *dp++;
		  }
		  if(xssum < 0) xssum = 0;    //    moss CHOP addition
		  ysSumParts[ysSumPartsBottomIdx] = xssum;

		  yssum -= xssum; 
		  if(yssum < 0) wvl[ IJK(x+nshftx,y+nshfty,zz) ] = 0;    //    moss CHOP addition
		  else          wvl[ IJK(x+nshftx,y+nshfty,zz) ] = yssum * wvlNorm;

		  // shift ysSum - range one 'up'
		  ysSumPartsBottomIdx++;
		  if(ysSumPartsBottomIdx >= ordy3) 
			ysSumPartsBottomIdx-=ordy3;
		}
	}
  delete [] ysSumParts;
}

static 
void fwvl3d_part2_3(float *wvl,
					const int nx, const int ny, const int nz,  
					const int ordx, const int ordy, const int ordz)
{
  const int zStride = nx*ny;
  const float zsfac = 1.0 / ordz / 2.0;  // 2 because of my special psi

  const int nshftx = ordx/2;
  const int nshfty = ordy/2;
  const int nshftz = ordz/2;

  // this is to allow 'step3'(Z-wavelet) inplace
 const int ordz3 = 3*ordz; // support size of wavelet kernel
 float *zsParts = new float[ordz3];

 for(int y=ordy+nshfty;y<ny-2*ordy+nshfty;y++)
   for(int x=ordx+nshftx;x<nx-2*ordx+nshftx;x++)
	  {
		float zssum = 0; // z-sum: first full calc - then only updated
		
		//first full calc of zssums
		{
		  const float *dp = wvl + IJK(x,y, 0);

		  float *zsPptr = zsParts;

		  {for(int i=0;i<ordz;i++)
			*zsPptr++= *dp, zssum -= *dp, dp+=zStride;
		  }{for(int i=0;i<ordz;i++)
			*zsPptr++= *dp, zssum += 2* *dp, dp+=zStride;
		  }{for(int i=0;i<ordz;i++)
			*zsPptr++= *dp, zssum -= *dp, dp+=zStride;
		  }
		  // zero first (not calculated) z sects
		  int z;
		  for(z=0;z<ordz+nshftz;z++)
			wvl[ IJK(x,y, z) ] = 0;
		  
		  if(zssum < 0)  //    moss CHOP addition
			wvl[ IJK(x,y, z) ] = 0;
		  else
			wvl[ IJK(x,y, z) ] = zssum * zsfac;
		}
		
		//following z just update 
		int zsPBottomIdx = 0;

		for(int z=1;z<nz-3*ordz;z++)
		{
		  int zsPidx = zsPBottomIdx;

		  zssum += zsParts[zsPidx];   // update(/compensate) -1  to 0
		  zsPidx+=ordz; if(zsPidx >= ordz3) zsPidx-=ordz3;
		  zssum -= 3* zsParts[zsPidx];   // update(/compensate) +2  to -1
		  zsPidx+=ordz; if(zsPidx >= ordz3) zsPidx-=ordz3;
		  zssum += 3* zsParts[zsPidx];   // update(/compensate) -1  to +2
		  //zsPidx+=ordz; if(zsPidx >= ordz3) zsPidx-=ordz3;
		  
		  // calc (the only) ONE new(!) xssum
		  // for z: actually just put val into zsParts list and subt. from zssum
		  zssum -= ( zsParts[zsPBottomIdx] = wvl[ IJK(x,y, z+3*ordz-1) ] );
		  
		  if(zssum < 0)  //    moss CHOP addition
			wvl[ IJK(x,y, z+ordz+nshftz) ] = 0;
		  else
			wvl[ IJK(x,y, z+ordz+nshftz) ] = zssum * zsfac;

		  zsPBottomIdx++;
		  if(zsPBottomIdx >= ordz3) zsPBottomIdx-=ordz3;
		}
		// zero last (not calculated) z sects
		{for(int z=nz-2*ordz+nshftz;z<nz;z++)
		  wvl[ IJK(x,y, z) ] = 0;
		}
	  }
 delete [] zsParts;
} 


void fastwv66(float *data, 
			 int nx, int ny, int nz,
			 float *wvl, 
			 int nx2, int ny2, int nz2,
			 int ordx, int ordy, int ordz,
			 const bool prints)
{
  //TODO CHECK
  // check    data and step3 same size

  const int nxyz =  nx * ny * nz;

  if(nx<3*ordx) {
	if(prints) fprintf(stderr, "nx(%d) < 3* ordx(%d) --> wvl all zero\n", nx, ordx);
	return ;
  }
  if(ny<3*ordy) {
	if(prints) fprintf(stderr, "ny(%d) < 3* ordy(%d) --> wvl all zero\n", ny, ordy);
	return ;
  }
  

  if(prints) fprintf(stderr, "part1 (of 2; 0<=z<%d): ", nz);
  for(int zz=0;zz<nz;zz++) {
	if(prints) fprintf(stderr, "%d ",zz);
    fwvl3d_part1(wvl, nx, ny, ordx, ordy, zz, data);///ifChop  , chop);
  }

  if(nz<3*ordz) {
	if(prints) fprintf(stderr, "\n2D wavelet !! nz < 3*ordz\n");
  } else {
	if(prints) fprintf(stderr, "\npart2 (of 2): \n");
	fwvl3d_part2_3(wvl, nx, ny, nz,  
				   ordx, ordy, ordz);
  }
}
