#include <stdlib.h>
#include <math.h>
#include "wv3d.h"


//  void wvl3d(float *step3, float * wvl,
//  	   const int nx, const int ny, const int nz,  
//  	   const int ordx, const int ordy, const int ordz,
//  	   const float * data)
//  {
//    for(int zz=0;zz<nz;zz++)
//      wvl3d_part1(wvl, nx, ny, ordx, ordy, zz, data);

  
//    wvl3d_part2_3(step3, nx, ny, nz,  
//  		ordz, wvl);
//  }



// same as 2D but having a (untouched) 3rd index in data
void wvl3d_part1(float *wvl,
				 const int nx, const int ny,
				 const int ordx, const int ordy, 
				 const int zz, 
				 const float * data)///ifChop  , const bool chop)
{
  const float kfac = 1.0 / ordx / 2.0;  // 2 because of my special psi
  const float lfac = 1.0 / ordy / 2.0;  // 2 because of my special psi
  
  for(int j=0;j<nx;j++)                     // j: x-direction
    for(int i=0;i<ny;i++)                   // i: y-direction
      {
		float &ksum = wvl[ IJK(j,i,zz) ];
		ksum=0;

		int kmin =  i-ordy;    //0;
		if(kmin<0)  kmin = 0;
		int kmax =  i+2*ordy;  //ny;
		if(kmax>ny) kmax = ny;
		
		int lmin =  j-ordx;    //0;
		if(lmin<0)  lmin = 0;
		int lmax =  j+2*ordx;  //nx;
		if(lmax>nx) lmax = nx;
		
		for(int k=kmin;k<kmax;k++)          // k: y-direction
		  {
			int psi1 = psi(ordy ,i,k);//, k,i);
			if(psi1 != 0)
			  {
				float lsum = 0;
				
				for(int l=lmin;l<lmax;l++)  // l: x-direction
				  {
					int psi2 = psi(ordx ,j,l);//, l, j);
					if(psi2<0) lsum -= data[ IJK(l,k,zz) ];
					if(psi2>0) {
					  const float d = data[ IJK(l,k,zz) ];
					  lsum += 2*d;
					  //*2  lsum += d;
					}
				  }
				lsum *= lfac;
                if(lsum < 0) lsum = 0;    //    moss CHOP addition
				else {
				  if(psi1 < 0) ksum -= lsum;
				  if(psi1 > 0) {
					ksum += 2* lsum;
					//*2  ksum += lsum;
				  }
				}
			  }
		  }
		ksum *= kfac;
        if(ksum < 0) ksum = 0;    //    moss CHOP addition
      }
}

void wvl3d_part2_3(float *step3,
				   const int nx, const int ny, const int nz,  
				   const int ordz,
				   const float * wvl)///ifChop  , const bool chop)
{
  
  
  for(int zz=0;zz<nz;zz++)   ///   
    {
      const float kfac = 1.0 / ordz / 2.0;  // 2 because of my special psi
      
      for(int j=0;j<ny;j++)
		for(int i=0;i<nx;i++)
		  {
			float &ksum = step3[ IJK(i,j,zz) ];
			ksum=0;
			int kmin = zz-ordz;//0;
			if(kmin<0) kmin = 0;
			int kmax = zz+2*ordz;//nz;
			if(kmax>nz) kmax = nz;
			
			for(int k=kmin;k<kmax;k++)
			  {
				int psi1 = psi(ordz ,zz, k);//, k,zz);
				if(psi1 > 0)
				  {
					const float d = wvl[ IJK(i,j,k) ];
					ksum += 2*d;
					//*2  ksum += d;
				  }
				if(psi1 < 0)
				  {
					ksum -= wvl[ IJK(i,j,k) ];
				  }
			  }
			ksum *= kfac;
            if(ksum < 0) ksum = 0;    //    moss CHOP addition
		  }
    }  
}
