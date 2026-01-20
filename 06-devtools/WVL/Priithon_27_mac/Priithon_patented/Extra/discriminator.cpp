//  // compile on SGI:  CC -O2 discriminator.cpp -o discriminator  -lm 
//  // compile on MAC:  c++ -O4 discriminator.cpp -o discriminator

//  #include "myIVE.h"
//  #include <stdlib.h>
#include <iostream>
using namespace std;
//  #include <iomanip.h>
#include <stdio.h>
#include <stdlib.h>
//  #include <strings.h>

#include <math.h>
#include <assert.h>

static const float smallWilli = 1e-10;//pow(2.0,-12.0);

static int nz=0,ny=0,nx=0;
static int *objSizes;
static short *lableIMG;
static short *newLableFor;
static short *oldLableOf;
                                                //    z*nx*ny +  y*nx +x
                                                // == x+ nx*(y + z *ny) 
#define XYZ(x,y,z) ((x)+ nx*((y) + (z) *ny))

template <class T> void sw(T* a, T* b) { T h=*a; *a=*b; *b=h; }

static int lessthan(const void *a, const void *b) { 
  const float x=*(float*)(a);
  const float y=*(float*)(b);
  return x<y? -1 : (x>y ? 1 : 0); 
}


static int objOrder(const void *a, const void *b) { 
  const short x=*(short*)(a);
  const short y=*(short*)(b);
  return objSizes[x]>objSizes[y]? -1 : 
        (objSizes[x]<objSizes[y] ? 1 : 0); 
}



////float *loadPriismFile(const char *fn); // forward declaration
static short lable3Dimg(int * in, short *out); // forward declaration

// note: June 4th, 2002: adding delta to indicate different pixel-spacing in x,y,z-direction
void
static discrVol(float *inVol, float *outVol, const int order, int dx, int dy, int dz, double delta);


//  template <class T> void  writeTo(const T *image, const char *of) ; // forward declaration




  



void discriminator(float *img, //inVol, 
				   int nx1, int ny1, int nz1,
				   float *img2, //outVol, 
				   int nx2, int ny2, int nz2,
				   short *outLableIMG,
				   int nx3, int ny3, int nz3,
				   int order, float thrsh, float offset, 
				   bool zdis, 
				   double deltax, double deltay,  double deltaz)
{
  nx=nx1;
  ny=ny1;
  nz=nz1;
  //TODO - check on correct array sizes !!!!!
  fprintf(stderr, "smallWilli: %e\n", smallWilli);

// WE MAKE DELTAs IN UNITS OF DELTAX !!!!!
  deltaz /= deltax;
  deltay /= deltax;
  deltax = 1.0;
// WE MAKE DELTAs IN UNITS OF DELTAX !!!!!


  const bool debugMASK = (order < 0);
  if(order < 0) order = -order;   /// JUST out way to switch to OLD (wrong) mask usage (for debugging)



//  float *img= loadPriismFile(inFN);

  
  //    float *img2 = new float[nx*ny*nz];
  int *imgBIN  = new int[nx*ny*nz];
  int *imgBIN1 = new int[nx*ny*nz];
  int *imgBIN2 = new int[nx*ny*nz];
  int *imgBIN3 = new int[nx*ny*nz];

  
  float *medianArray = new float[nx*ny];

  // later for iautoback == 0 ..... 
  float *mask = new float [nz];

  cerr << "masking" << endl;
  
  for(int z=0;z<nz;z++) 
    {
      cerr << z << " ";
      if(thrsh >= 0) 
	{
	  mask[z] = thrsh; 
	} 
      else 
	{
	  int n=0;

	  for(int y=0;y<ny;y++) 
	    for(int x=0;x<nx;x++) 
	      {
		const float f = img[ XYZ(x,y,z) ];
		if(f > 0) medianArray[n++] = f;
	      }
	  
	  //void qsort (void* base, size_t nel, size_t width, int (*compar) (const void *, const void *));
	  qsort(medianArray, n, sizeof(float), lessthan);

	  const float med= (n%2)? medianArray[(n-1)/2] : (medianArray[n/2]+medianArray[n/2-1])/2.0;

	  for(int i=0;i<n;i++)
	    {
	      medianArray[i] = fabs(medianArray[i] - med);
	    }
	  qsort(medianArray, n, sizeof(float), lessthan);
	  
	  const float meddev= (n%2)? medianArray[(n-1)/2] : (medianArray[n/2]+medianArray[n/2-1])/2.0;

	  mask[z] = med - 0.5*meddev ;
	  
	  cerr << "(" << mask[z] << "|" << med << "|" << meddev << ") ";
	}
    }


  //  cerr << "\ndatsub/img2=0  " << endl;
  {{
    for(int z=0;z<nz;z++) {
      //      cerr << z << " ";
      for(int y=0;y<ny;y++) {
	for(int x=0;x<nx;x++) 
	  {
	    // this is what was done for 'iautoback == 0'
	    if(img[ XYZ(x,y,z) ] < mask[z])
	      img[ XYZ(x,y,z) ] = mask[z];      // = datsub
	    
	    // init img2 to '0'
	    img2[ XYZ(x,y,z) ] = 0.0;
	  }
      }
    }
  }}

  cerr << "\ndiscrvol x" << endl;
  discrVol( img, img2, order, 1,0,0, deltax );

  cerr << "\ndiscrvol y" << endl;
  discrVol( img, img2, order, 0,1,0, deltay  );

  if(zdis) {
    cerr << "\ndiscrvol z" << endl;
// note: June 4th, 2002: adding delta to indicate different pixel-spacing in x,y,z-direction
//      cerr << " ********************************************************** " <<  endl;
//      cerr << " WARNING :  discriminant in delta-Z is only defined if dz = dx " << endl;
//      cerr << " ********************************************************** " <<  endl;
    discrVol( img, img2, order, 0,0,1, deltaz  );
  }



  cerr << "\npredig.totsub.offset" << endl;
  {{
    for(int z=0;z<nz;z++) {

      cerr << z << " ";

      for(int y=0;y<ny;y++) {
	for(int x=0;x<nx;x++) 
	  {

	    img2[ XYZ(x,y,z) ] *=  -( img[ XYZ(x,y,z) ] - mask[z]);  // = totsub //// nov-8: 
	    if(debugMASK) {
	      img2[ XYZ(x,y,z) ] = img[ XYZ(x,y,z) ] >= (mask[z]+1)    // = predig //// SHOULD BE: '>= mask'
		                                                       //// JUST CHANGED temporarily 
                                                                       ////  for debugging w/ old results
		? img2[ XYZ(x,y,z) ]   
		: 0;
	    }
	    else {
	      img2[ XYZ(x,y,z) ] = img[ XYZ(x,y,z) ] >= mask[z]     // = predig
		//// JUST CHANGED temporarily 
		////  for debugging w/ old results
		? img2[ XYZ(x,y,z) ]   
		: 0;
	    }
			  



	    //binarize !!
	    imgBIN[ XYZ(x,y,z) ]  = img2[ XYZ(x,y,z) ] > offset ? 1 : 0;
	  }
      }
    }
  }}
  
  
  cerr << "\ndgtzm1" << endl;
  {{
    for(int z=0;z<nz;z++) {
      
      cerr << z << " ";
      
      for(int y=0;y<ny;y++) {
	for(int x=0;x<nx;x++) 
	  {
	    int sum = 0;

	    for(int a=x-1;a<=x+1;a++) {
	      if(a<0 || a>= nx) continue;
	      for(int b=y-1;b<=y+1;b++) {
		if(b<0 || b>= ny) continue;
		//todo c: for z

		sum += imgBIN[ XYZ(a,b,z) ];
	      }
	    }
	      
	    if(sum >= 3)  // keep if at-least itself and two neighbors
	      imgBIN1[ XYZ(x,y,z) ] = imgBIN[ XYZ(x,y,z) ];
	    else 
	      imgBIN1[ XYZ(x,y,z) ] = 0;
	  }
      }
    }
  }}  // imgBIN1 =  dgtzm1


  cerr << "\ndgtzm2" << endl;
  {{
    for(int z=0;z<nz;z++) {
      
      cerr << z << " ";
      
      for(int y=0;y<ny;y++) {
	for(int x=0;x<nx;x++) 
	  {
	    int sum = 0;

	    for(int a=x-1;a<=x+1;a++) {
	      if(a<0 || a>= nx) continue;
	      for(int b=y-1;b<=y+1;b++) {
		if(b<0 || b>= ny) continue;
		//todo c: for z

		sum += imgBIN1[ XYZ(a,b,z) ];
	      }
	    }
	      
	    if(sum >= 4)  // keep if at-least itself and three neighbors
	      imgBIN2[ XYZ(x,y,z) ] = imgBIN1[ XYZ(x,y,z) ];
	    else
	      imgBIN2[ XYZ(x,y,z) ] = 0;
	  }
      }
    }
  }}  // imgBIN =  dgtzm2


  cerr << "\ndgtzm3" << endl;
  {{
    for(int z=0;z<nz;z++) {

      cerr << z << " ";

      for(int y=0;y<ny;y++) {
	for(int x=0;x<nx;x++) 
	  {
	    int sum = 0;

	    for(int a=x-1;a<=x+1;a++) {
	      if(a<0 || a>= nx) continue;
	      for(int b=y-1;b<=y+1;b++) {
		if(b<0 || b>= ny) continue;
		//todo c: for z

		sum += imgBIN2[ XYZ(a,b,z) ];
	      }
	    }
	      
	    if(sum >= 2)  // keep if at-least itself and one neighbors
	      imgBIN3[ XYZ(x,y,z) ] = imgBIN2[ XYZ(x,y,z) ];
	    else
	      imgBIN3[ XYZ(x,y,z) ] = 0;
	  }
      }
    }
  }}  // imgBIN =  dgtzm3



/*

  char of[80];


//      cerr << "\nwrite BIN" << endl;
//      sprintf(of , "%s_imgBIN.txt", outFN);
//      writeTo(imgBIN, of);

//      cerr << "\nwrite BIN1" << endl;
//      sprintf(of , "%s_imgBIN1.txt", outFN);
//      writeTo(imgBIN1, of);

//      cerr << "\nwrite BIN2" << endl;
//      sprintf(of , "%s_imgBIN2.txt", outFN);
//      writeTo(imgBIN2, of);

//      cerr << "\nwrite IMG" << endl;
//      sprintf(of , "%s_img.txt", outFN);
//      writeTo(img, of);

  cerr << "\nwrite discriminant (predig)" << endl;
  sprintf(of , "%s_discr.txt", outFN);
  writeTo(img2, of);
*/
  //python: here img2 is outVol ;-)


//  cerr << "\nwrite BIN3" << endl;
//  sprintf(of , "%s_bin.txt", outFN);
//  writeTo(imgBIN3, of);



  lableIMG = outLableIMG;//new short[nx*ny*nz];

  int numObjs=lable3Dimg(imgBIN3, lableIMG); // see also #define IMG_PTR() !!!!



//    cerr << "\nwrite orig lable" << endl;
//    sprintf(of , "%s_origlable.txt", outFN);
//    writeTo(lableIMG, of);



//  int *objSizes = new int[ numObjs ];
  objSizes = new int[ numObjs ];

  cerr <<"\ncounting objs ";
  {{
    for(int i=0;i<numObjs;i++) 
      objSizes[i] = 0;

    short *p = lableIMG;
    for(int z=0;z<nz;z++) {

      cerr << z << " ";

      for(int y=0;y<ny;y++)
	for(int x=0;x<nx;x++) 
	  {
	    
	    objSizes[ *p++ ]++;
	  }
    }
  }}

  oldLableOf  = new short[ numObjs ];
  newLableFor = new short[ numObjs ];
  {{
    for(short i=0;i<numObjs;i++) {
      oldLableOf[i] = i;
    }
  }}


  qsort(oldLableOf, numObjs, sizeof(short), objOrder);

  {{
    for(short i=0;i<numObjs;i++) {
      newLableFor[ oldLableOf[i] ] = i;
    }
  }}

/*TODO: python
  {{
    sprintf(of , "%s_objSizes", outFN);
    cerr << of << endl;

    ofstream o(of);

    for(short i=0;i<numObjs;i++) {
      o << i << " " << objSizes[oldLableOf[i]] << endl;
//      o << objSizes[i] << " " << i << " " << newLableFor[i] << endl;
//      o << objSizes[newLableFor[i]] << " " << i << " " << newLableFor[i] << endl;
    }
  }}
*/

  cerr <<"\nrelabling objs ";
  {{
    short *p = lableIMG;
    for(int z=0;z<nz;z++) {

      cerr << z << " ";

      for(int y=0;y<ny;y++)
	for(int x=0;x<nx;x++) 
	  {
	    if(*p > 0) // only relable foreground
	    *p = newLableFor[*p];
	    p++;
	  }
    }
  }}


/*TODO -> Python 
  cerr << "\nwrite relabled" << endl;
  sprintf(of , "%s_lable.txt", outFN);
  writeTo(lableIMG, of);
*/

/*
  {
    sprintf(of , "%s_lable.dat", outFN);
    ofstream  priismOutFile(of);
    if(pf.header.nlab<10) {
      strcpy(pf.header.label[pf.header.nlab++], "discriminant lables");
    }
    pf.header.mode = 1;
    pf.header.inbsym = 0;
    pf.header.nDVID = 0xc0a0; // no swapped !!! CHECK
    
    priismOutFile << pf.header;

    for(int z=0;z<nz; z++) 
      priismOutFile.write((char*)&lableIMG[z*nx*ny] , sizeof(lableIMG[0])*nx*ny);	
  }
  cerr << "\ndone." << endl;
*/
  //python: here lableIMG is out-lableIMG ;-)

  delete [] mask;
  delete [] medianArray;

  //    delete [] img2;
  delete [] imgBIN;
  delete [] imgBIN1;
  delete [] imgBIN2;
  delete [] imgBIN3;
  
  //    delete [] img;

  return;
}


#if 0
template <class T>
void  writeTo(const T *image, const char *of)
{
  ofstream o(of);

  int x,y,z;


  cerr << "nx: " << nx << endl;  

  for(z=0;z<nz;z++)
    {
      cerr << z << " ";

      for(y=0;y<ny;  /*CVS y++ */ )
	{
	  for(x=0;x<nx;   /*CVS x++ */ )
	    {	      
	      T s = image[ XYZ(x,y,z) ];
//	      o << setw(2) << s;
	      o << s;
//	      o.write((char*) &s, sizeof(s));
	      
	      if(++x != nx) o << " ";       // csv
	      //else  o <<"\n";     ; // x and y in one row ...

	      //if(++x == nx) o << "}";
	      //else o << ", ";
	    }

	  if(++y != ny) o << " ";
	  else o << "\r"; // csv  new line on each z
	}
    }
}
#endif


/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////



//typedef int pixel_t; /* mathematica uses int ; not short :-( */
typedef short pixel_t; 


static short *neighbor_stack; /* entries: short row, short col */
static short *top;            /* points to top of stack */

static short cur_obj_ID;



/* help functions */

void inline pushIF(const short i, const short j, const short k) 
{
  pixel_t *p = &lableIMG[ XYZ(i,j,k) ];
  
  if(*p < 0) {
    *p = cur_obj_ID;   /* p is part of 'current' object */

    *top++ = i;
    *top++ = j;
    *top++ = k;
  }
}

void inline pushSlice(const short i, const short j, const short k) 
{
  if(j>0) {                  // comments have i and j meaning  switched ... 
      pushIF(i,   j-1, k);     /* left   */
    if(i<nx-1) {
      pushIF(i+1, j-1, k);     /* bottom - left */
    }
    if(i>0) {
      pushIF(i-1, j-1, k);     /* top - left */
    }
  }
  
  if(j<ny-1) {
      pushIF(i,   j+1, k);     /* right  */
    if(i<nx-1) 
      pushIF(i+1, j+1, k);     /* bottom - right */
    if(i>0)
      pushIF(i-1, j+1, k);     /* top - right  */
  }
  
  if(i>0)
    pushIF(  i-1,   j, k);         /* top    */	
  if(i<nx-1)
    pushIF(  i+1,   j, k);         /* bottom */
}


void inline push_all_neighbor_of(const short i, const short j, const short k) 
{
  pushSlice(   i,j, k);

  if(k>0)  {
    pushIF(    i,j, k-1); 
    pushSlice( i,j, k-1);
  }

  if(k<nz-1) {
    pushIF(    i,j, k+1); 
    pushSlice( i,j, k+1);
  }
}

void inline workStack() 
{
  while(top > neighbor_stack) {
    const short k = *(--top);	
    const short j = *(--top);
    const short i = *(--top);
    
    pixel_t *p = &lableIMG[ XYZ(i,j,k) ];
    
    if(*p > 0) {  /* here NOT '< 0' ; since neighbor are immediately labled 
		     to prevent double-entries on stack */
      push_all_neighbor_of(i, j, k);
    }
  }
}

/* these should be enough ... */ 


// return number of labled (found) objects
short lable3Dimg( int * in, pixel_t *out)
{
  // count foreground pixels  -- this is max depth of stack !
  int max =0;   // (third of) max hight of stack

/*
  {{
    for(int z=0;z<nz;z++)
      for(int y=0;y<ny;y++)
	for(int x=0;x<nx;x++)
	  {
	    if( in[ XYZ(x,y,z) ] ) {
	      max++;
	      *IMG_PTR(x,y,z) = -1;
	    } else 	      
	      *IMG_PTR(x,y,z) = 0;
	  }
  }}
*/

  int len = nx*ny *nz;
  int  *p=in;
  pixel_t *q=out;

  for(long l=0;l<len;l++) {
    if(*p++ > 0) {
       max++;
       *q++ = -1;
    }
    else *q++ = 0;
  }

  
  top = 
    neighbor_stack = (short*) malloc(max * 3 * sizeof(short)); 
  
  if(!top) {
    cerr <<" cannot allocate mem for labling stack  -42, 0, 42 \n"; exit(1);
    //    int a[3] = { -42, 0, 42 };
    //    MLPutIntegerList(stdlink, a, 3);
    
    return 0;
  }


  //  top = 
  //    neighbor_stack = (short*) malloc(n*m*z  * 3 * sizeof(short));
   
  cur_obj_ID = 0; /* first object ID will be '1' */
  //img = list;



  for(int k=0;k<nz;k++) /* slice */
    {
      for(int j=0;j<ny;j++) /* row   (top--->down) */
	{
	  for(int i=0;i<nx;i++) /* col (left--->right) */
	    {
	      pixel_t  *p = &lableIMG[ XYZ(i,j,k) ];//IMG_PTR(i,j,k);
	      if(*p < 0) {      /* found new object */
		*p = ++cur_obj_ID;
		
		
		push_all_neighbor_of(i, j, k);
		
		workStack();
	      }
	    }
	}
    }

  free(neighbor_stack);

  return cur_obj_ID+1;

  //  MLPutIntegerList(stdlink, list, len);
}




/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////


void
discrVol(float *inVol, float *outVol, const int order, int dx, int dy, int dz, double delta)
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
