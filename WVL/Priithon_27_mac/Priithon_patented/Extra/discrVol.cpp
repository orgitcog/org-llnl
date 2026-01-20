#include <iostream>
using namespace std;
#include <math.h>
#include <assert.h>

const float smallWilli = 1e-10;//pow(2.0,-12.0);

#define XYZ(x,y,z) ((x)+ nx*((y) + (z) *ny))



void
discrVol(float *inVol, 
		 int nx, int ny, int nz,
		 float *outVol, 
		 int nx2, int ny2, int nz2,
		 const int order, int dx, int dy, int dz, double delta)
{
  dx = (dx==0) ? 0: 1;
  dy = (dy==0) ? 0: 1;
  dz = (dz==0) ? 0: 1;

  // order HAS TO BE _ODD_

  const int beg = (order-1) /2 ;// (integer devision) 
  
  for(int z=(dz?beg:0);z < nz-(dz?beg:0);z++) {
    cerr << z << " ";

    for(int y=(dy?beg:0);y < ny-(dy?beg:0);y++) {
      for(int x=(dx?beg:0);x < nx-(dx?beg:0);x++) 
	{
	  float f1=0;
	  float f2=0;
	  float f3=0;
	  

	  for(int a=-beg ; a<=beg ; a++)
	    {
	      f1 += a*a * inVol[ XYZ(x+a*dx, y+a*dy, z+a*dz) ];
	      f2 += a   * inVol[ XYZ(x+a*dx, y+a*dy, z+a*dz) ];
	      f3 +=       inVol[ XYZ(x+a*dx, y+a*dy, z+a*dz) ];
	    }
	  
	  float deriv1, deriv2;
	  switch(order) {
	  case 5:
	    deriv2 = 2 * (f1 - 2 * f3) / 14.0;
	    deriv1 = f2 / 10.0;
	    break;
	  case 9:
	    deriv2 = 2 * (3*f1 - 20 * f3) / 924.0;
	    deriv1 = f2 / 60.0;
	    break;
	  case 15:
	    deriv2 = 2 * (3*f1 - 56 * f3) / 12376.0;
	    deriv1 = f2 / 280.0;
	    break;
	  case 21:
	    deriv2 = 2 * (3*f1 - 110 * f3) / 67298.0;
	    deriv1 = f2 / 770.0;
	    break;
	    
	  default:
	    assert(1==0);
	  }

// note: June 4th, 2002: adding delta to indicate different pixel-spacing in x,y,z-direction
	  deriv1 /= delta;
	  deriv2 /= delta*delta;
	  
	  const float discrim = deriv2 / (fabsf(deriv1) + smallWilli);
	  
	  
	  outVol[ XYZ(x, y, z) ] += discrim;
	  
	  //
	  //  EDGE HANDLING  for DX, DY, DZ separately !!!
	  //


	  if(dx) {
	    if(x==beg) {  // handle left edge
	      for(int a=0;a<beg;a++) {
		
		const float deriv1PRE = delta * deriv2 * (-beg + a) + deriv1;
		
		const float discrimX = deriv2 / (fabsf(deriv1PRE) + smallWilli);
		outVol[ XYZ(a, y, z) ] += discrimX;
	      }
	    }
	    else if(x==nx-beg-1) {  // handle right edge
	      for(int a=0;a<beg;a++) {
		
		const float deriv1POST = delta * deriv2 * (1+ a) + deriv1;
		
		const float discrimX = deriv2 / (fabsf(deriv1POST) + smallWilli);
		outVol[ XYZ(nx-beg+a, y, z) ] += discrimX;
	      }
	    }
	  }
	  
  
	  if(dy) {
	    if(y==beg) {  // handle left edge
	      for(int a=0;a<beg;a++) {
		
		const float deriv1PRE = delta * deriv2 * (-beg + a) + deriv1;
		
		const float discrimX = deriv2 / (fabsf(deriv1PRE) + smallWilli);
		outVol[ XYZ(x, a, z) ] += discrimX;
	      }
	    }
	    else if(y==ny-beg-1) {  // handle right edge
	      for(int a=0;a<beg;a++) {
		
		const float deriv1POST = delta * deriv2 * (1+ a) + deriv1;
		
		const float discrimX = deriv2 / (fabsf(deriv1POST) + smallWilli);
		outVol[ XYZ(x, ny-beg+a, z) ] += discrimX;
	      }
	    }
	  }
	  
	  
	  if(dz) {
	    if(z==beg) {  // handle left edge
	      for(int a=0;a<beg;a++) {
		
		const float deriv1PRE = delta * deriv2 * (-beg + a) + deriv1;
		
		const float discrimX = deriv2 / (fabsf(deriv1PRE) + smallWilli);
		outVol[ XYZ(x, y, a) ] += discrimX;
	      }
	    }
	    else if(z==nz-beg-1) {  // handle right edge
	      for(int a=0;a<beg;a++) {
		
		const float deriv1POST = delta * deriv2 * (1+ a) + deriv1;
		
		const float discrimX = deriv2 / (fabsf(deriv1POST) + smallWilli);
		outVol[ XYZ(x, y, nz-beg+a) ] += discrimX;
	      }
	    }
	  }
	}
    }
  }
}
