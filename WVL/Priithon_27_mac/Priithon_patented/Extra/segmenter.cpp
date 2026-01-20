// from thrshAndSeg.cpp  -- like lable3Dimg

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;


//TODO
// FIXME  FIXME  FIXME  FIXME  FIXME 
//   no global vars - please !!!!
// FIXME  FIXME  FIXME  FIXME  FIXME 

int nz=0,ny=0,nx=0;
int *objSizes;
short *lableIMG;
short *newLableFor;
short *oldLableOf;
                                                //    z*nx*ny +  y*nx +x
                                                // == x+ nx*(y + z *ny) 
#define XYZ(x,y,z) ((x)+ nx*((y) + (z) *ny))

template <class T> void sw(T* a, T* b) { T h=*a; *a=*b; *b=h; }

int objOrder(const void *a, const void *b) { 
  const short x=*(short*)(a);
  const short y=*(short*)(b);
  return objSizes[x]>objSizes[y]? -1 : 
        (objSizes[x]<objSizes[y] ? 1 : 0); 
}



////float *loadPriismFile(const char *fn); // forward declaration
short lable3Dimg(short * in, short *out); // forward declaration


void segmenter(short **p_Out1dA, int *p_nxA, //oldLableOf
			   short **p_Out1dB, int *p_nxB, //newLableFor

			   short *imgBIN, 
			   const int nx1, const int ny1, const int nz1,
			   short *THElableIMG, 
			   const int nx2, const int ny2, const int nz2,
			   int *numObjs_ptr)
{
  //TODO: CHECK for same size of imgBIN & lableIMG
  nx=nx1;
  ny=ny1;
  nz=nz1;
  lableIMG = THElableIMG;
  const int nxyz=nx * ny * nz;
  
  //imgBIN = img3DStack

  ///lableIMG = new short[nxyz];

  //printf("seb1: %d %d %d\n", nx, ny, nz);

  const int numObjs=lable3Dimg(imgBIN, lableIMG); // see also #define IMG_PTR() !!!!

  *numObjs_ptr = numObjs;

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

  *p_Out1dA = oldLableOf  = new short[ *p_nxA = numObjs ];
  *p_Out1dB = newLableFor = new short[ *p_nxB = numObjs ];
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


//TODO return objSizes as Python array
//  //    {{
//  //      sprintf(of , "%s_objSizes", outFN);
//  //      cerr << of << endl;

//  //      ofstream o(of);

//  //      for(short i=0;i<numObjs;i++) {
//  //        o << i << " " << objSizes[oldLableOf[i]] << endl;
//  //  //      o << objSizes[i] << " " << i << " " << newLableFor[i] << endl;
//  //  //      o << objSizes[newLableFor[i]] << " " << i << " " << newLableFor[i] << endl;
//  //      }
//  //    }}


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

//HACK - who calls delete !????
//    delete [] oldLableOf;
//    delete [] newLableFor;

  
//  //    cerr << "\nwrite relabled" << endl;
//  //    sprintf(of , "%s_lable.txt", outFN);
//  //    writeTo(lableIMG, of);


//  //    {
//  //      sprintf(of , "%s_lable.dat", outFN);
//  //      ofstream  priismOutFile(of);
//  //      if(pf.header.nlab<10) {
//  //        strcpy(pf.header.label[pf.header.nlab++], "discriminant lables");
//  //      }
//  //      pf.header.mode = 1;
//  //      pf.header.inbsym = 0;
//  //      pf.header.nDVID = 0xc0a0; // no swapped !!! CHECK
    
//  //      priismOutFile << pf.header;

//  //      for(int z=0;z<nz; z++) 
//  //        priismOutFile.write((char*)&lableIMG[z*nx*ny] , sizeof(lableIMG[0])*nx*ny);	
//  //    }
  
//  //    cerr << "\ndone." << endl;

}



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


short *neighbor_stack; /* entries: short row, short col */
short *top;            /* points to top of stack */

short cur_obj_ID;



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
short lable3Dimg( short * in, pixel_t *out)
{
  // count foreground pixels  -- this is max depth of stack !
  int max =0;   // (third of) max hight of stack


  int len = nx*ny *nz;
  short  *p=in;
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
}
