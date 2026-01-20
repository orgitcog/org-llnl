// -*- c++ -*-

//included from sebTypemaps.i

//  ///////////////////////////////////////////////////////
//  //  checker to confirm what type of array we are talking about
//  ///////////////////////////////////////////////////////

#define SEBS_TYPECHECK2d(SWIG_TC , CTYPE, TYSTR, PYARR_TC)                 \
%typecheck(SWIG_TYPECHECK_##SWIG_TC) (CTYPE *array2d, int nx, int ny) {    \
  debugPrintf("debug: typecheck2d "TYSTR"\n");                             \
  if(!PyArray_Check($input)) $1=0;										   \
  else if(((PyArrayObject*)$input)->descr->type_num != PYARR_TC) $1=0;	   \
  else if(((PyArrayObject*)$input)->nd != 2) $1=0;						   \
  else $1=1;															   \
}

//  debugPrintf("debug: typecheck " TYSTR "\n");                           \
//  $1 = PyArray_Check($input)?((PyArrayObject*)$input)->descr->type_num == PYARR_TC : 0; }


SEBS_TYPECHECK2d(UINT8,   Byte,      "byte",    tUInt8)
SEBS_TYPECHECK2d(INT16,   short,     "short",   tInt16)
SEBS_TYPECHECK2d(FLOAT,   float,     "float",   tFloat32)
SEBS_TYPECHECK2d(COMPLEX, complex32, "complex", tComplex32)
SEBS_TYPECHECK2d(UINT16,  Word,      "Ushort",  tUInt16)
SEBS_TYPECHECK2d(UINT32,  long,      "long",    tInt32)
SEBS_TYPECHECK2d(DOUBLE,  double,    "double",  tFloat64)

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

#define MAP_SHAPE2D                                                        \
  switch(NAimg->nd) {          											   \
  case 1:                      											   \
	$2 = NAimg->dimensions[0]; 											   \
	$3=1;                    											   \
	break;																   \
  case 2: 																   \
	$2 = NAimg->dimensions[1]; 											   \
	$3=NAimg->dimensions[0]; 											   \
	break;																   \
  default: 																   \
    debugPrintf(" **** numarray dim >2 (ns=%d)\n", NAimg->nd); 			   \
    _SWIG_exception(SWIG_RuntimeError, "numarray dim > 2"); 			   \
    return 0; 															   \
}


#define SEBS_TYPEMAP2d(CTYPE, TYSTR, PYARR_TC)                             \
%typemap(in)														   \
  (CTYPE *array2d, int nx, int ny)										   \
  (PyArrayObject *temp=NULL)											   \
{																		   \
  debugPrintf("debug: "TYSTR" *array2d  -> NA_InputArray\n");			   \
  PyArrayObject *NAimg  = NA_InputArray($input, PYARR_TC, NUM_C_ARRAY);	   \
																		   \
  if (!NAimg) {															   \
	printf("**** no ("TYSTR") numarray *****\n");						   \
	return 0;															   \
  }																		   \
  temp = NAimg;															   \
																		   \
  $1 = (CTYPE *) (NAimg->data + NAimg->byteoffset);						   \
  MAP_SHAPE2D															   \
}																		   \
%typemap(freearg)												   \
  (CTYPE *array2d, int nx, int ny)										   \
{																		   \
  debugPrintf("debug: "TYSTR" *array2d  -> Py_XDECREF\n");				   \
  Py_XDECREF(temp$argnum);												   \
}

SEBS_TYPEMAP2d(Byte,      "byte",    tUInt8)
SEBS_TYPEMAP2d(short,     "short",   tInt16)
SEBS_TYPEMAP2d(float,     "float",   tFloat32)
SEBS_TYPEMAP2d(complex32, "complex", tComplex32)
SEBS_TYPEMAP2d(Word,      "Ushort",  tUInt16)
SEBS_TYPEMAP2d(long,      "long",    tInt32)
SEBS_TYPEMAP2d(double,    "double",  tFloat64)
