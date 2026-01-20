// -*- c++ -*-



////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
//  1D     ////
//  1D     ////
//  1D     ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
//  #include <complex.h>
//  typedef complex<float> complex32;

%typemap(python,in) 
  (complex32 *cmpArray, int n) 
  (PyArrayObject *temp=NULL) 
{
  PyArrayObject *NAimg  = NA_InputArray($input, tComplex32, NUM_C_ARRAY);

  if (!NAimg) {
	printf("**** no numarray *****\n");
	return 0;
  }
  temp = NAimg;
  printf("* complex - ndim: %d\n", NAimg->nd); 
  printf("* n: %d\n", NAimg->dimensions[0]); 

  $1 = (complex32 *) NAimg->data;
  $2 = NAimg->dimensions[0];
}


%typemap(python,freearg) 
  (complex32 *cmpArray, int n) 
{
  Py_XDECREF(temp$argnum);
}




////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
//  OUT array    ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

//TODO FIXME //TODO FIXME  who calls delete ?????????


////////////////////////////////////////////////////////////
// out int 2D

%typemap(in,numinputs=0) (int **p_Out2d, int *p_nx, int *p_ny) (int *tmpP, int tmpNx, int tmpNy) 
{
  
  debugPrintf("debug: (in,...)(int **Out2d,int *nx,int*ny): %p %p %p\n", tmpP, tmpNx, tmpNy);
  $1 = &tmpP;
  $2 = &tmpNx;
  $3 = &tmpNy;
}

%typemap(argout) (int **p_Out2d, int *p_nx, int *p_ny)
{
  debugPrintf("debug: (argout) (int **Out2d, int *nx, int *ny)\n");
  debugPrintf("debug: %p %d %d\n", tmpP$argnum, tmpNx$argnum, tmpNy$argnum);
  PyObject *o = (PyObject*) NA_New(tmpP$argnum, tInt32, 2, tmpNy$argnum, tmpNx$argnum);

  argout_Append_To_Result($result, o);
}

////////////////////////////////////////////////////////////
// out int 1D

%typemap(in,numinputs=0) (int **p_Out1d, int *p_nx) (int *tmpP, int tmpNx) 
{
  debugPrintf("debug: (in,...)(int **p_Out1d, int *p_nx): %p %p\n", tmpP, tmpNx);
  $1 = &tmpP;
  $2 = &tmpNx;
}
%typemap(argout) (int **p_Out1d, int *p_nx)
{
  debugPrintf("debug: (argout)(int **p_Out1d, int *p_nx): %p %p\n", tmpP$argnum, tmpNx$argnum);
  PyObject *o = (PyObject*) NA_New(tmpP$argnum, tInt32, 1, tmpNx$argnum);
  argout_Append_To_Result($result, o);
}


////////////////////////////////////////////////////////////
// out short 1D

%typemap(in,numinputs=0) (short **p_Out1d, int *p_nx) (short *tmpP, int tmpNx) 
{
  debugPrintf("debug: (in,...)(short **p_Out1d, int *p_nx): %p %p\n", tmpP, tmpNx);
  $1 = &tmpP;
  $2 = &tmpNx;
}
%typemap(argout) (short **p_Out1d, int *p_nx)
{
  //seb($argnum)
  //seb($0_name)
  //seb($1_name)
  //seb($2_name)

  debugPrintf("debug: (argout)(short **p_Out1d, int *p_nx): %p %p\n", tmpP$argnum, tmpNx$argnum);
  PyObject *o = (PyObject*) NA_New(tmpP$argnum, tInt16, 1, tmpNx$argnum);
  argout_Append_To_Result($result, o);
}

