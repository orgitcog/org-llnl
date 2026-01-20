#include <stdlib.h>
#include <iostream>
using namespace std;
#include <math.h> //fabs

int lessthan(const void *a, const void *b) { 
  const float x=*(float*)(a);
  const float y=*(float*)(b);
  return x<y? -1 : (x>y ? 1 : 0); 
}


void medianer(float *img, 
			  const int nx, const int ny, const int nz, 
			  float threshold,
  			  float *median, float *median_dev)
{
  int n=0;
  float *medianArray= new float[nx*ny*nz];

  float *p = img;
  for(int k=0;k<nz;k++)
	{
	  for(int j=0;j<ny;j++)
		for(int i=0;i<nx;i++)
		  {
			const float f = *p++;
			if(f>threshold)
			  medianArray[ n++ ] = f;
		  }
	}
  cout << "found " << n << " values greater than " << threshold << endl;
  
  //void qsort (void* base, size_t nel, size_t width, int (*compar) (const void *, const void *));
  qsort(medianArray, n, sizeof(float), lessthan);

  const float med= (n%2)? medianArray[(n-1)/2] : (medianArray[n/2]+medianArray[n/2-1])/2.0;
  
  for(int i=0;i<n;i++)
    {
      medianArray[i] = fabs(medianArray[i] - med);
    }
  qsort(medianArray, n, sizeof(float), lessthan);
  
  const float meddev= (n%2)? medianArray[(n-1)/2] : (medianArray[n/2]+medianArray[n/2-1])/2.0;
  
  //  mask[z] = med - 0.5*meddev ;
  
  cerr << "median: "  << med << endl;
  cerr << "median dev:" << meddev << endl;

  *median     = med;
  *median_dev = meddev;
  cerr << "done." << endl;

  delete [] medianArray;
}
