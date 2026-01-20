// -*- c++ -*-
//__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
//__license__ = "BSD license - see LICENSE file"

//included from sebTypemaps.i

//  ///////////////////////////////////////////////////////
//  //  checker to confirm what type of array we are talking about
//  ///////////////////////////////////////////////////////

%define SEBS_TYPECHECK00d(SWIG_TC , CTYPE, TYSTR, PYARR_TC)
%typecheck(SWIG_TYPECHECK_##SWIG_TC) (CTYPE *array00d) {
  debugPrintf("debug: typecheck00d "TYSTR" type_num: (%d)\n", PYARR_TC);
  debugPrintf("debug: is_array($input)%d\n", is_array($input));
  if(!is_array($input)) $1=0;
  else if( array_type($input) != PYARR_TC) $1=0;
  else $1=1;
}
%enddef
//20041105  else if(((PyArrayObject*)$input)->nd >  1) $1=0;

SEBS_TYPECHECK00d(UINT8,   Byte,      "byte",      NPY_UBYTE)
SEBS_TYPECHECK00d(INT16,   short,     "short",     NPY_SHORT)
SEBS_TYPECHECK00d(FLOAT,   float,     "float",     NPY_FLOAT)
SEBS_TYPECHECK00d(COMPLEX, complex32, "complex32", NPY_CFLOAT)
SEBS_TYPECHECK00d(COMPLEX, complex64, "complex64", NPY_CDOUBLE)
SEBS_TYPECHECK00d(UINT16,  Word,      "Ushort",    NPY_USHORT)
SEBS_TYPECHECK00d(INT32,   int,       "int",       NPY_INT)
SEBS_TYPECHECK00d(DOUBLE,  double,    "double",    NPY_DOUBLE)
SEBS_TYPECHECK00d(UINT32,  DWord,     "Uint",      NPY_UINT) 
// no NPY_ULONG
SEBS_TYPECHECK00d(INT64,   long,      "long",      NPY_LONG)

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

  /*20041105 debugPrintf(" **** numarray dim >1 (ns=%d)\n", NAimg->nd);
    20041105 SWIG_exception(SWIG_RuntimeError, "numarray dim > 1");
  */

//  #define MAP_SHAPE00D                                                 \
//    switch(NAimg->nd) {          										 \
//    case 1:                      										 \
//  	$2 = NAimg->dimensions[0]; 										 \
//  	break;															 \
//    default: 															 \
//   { int a=1; for(int i=0;i<NAimg->nd;i++) a*=NAimg->dimensions[i];    \
//  		$2 = a; }                                                    \
//  }


%define SEBS_TYPEMAP00d(CTYPE, TYSTR, PYARR_TC)
%typemap(in)
  (CTYPE *array00d)
  (PyArrayObject *temp=NULL)
{
  debugPrintf("debug: "TYSTR" *array00d  -> NA_InputArray\n");
  if($input == Py_None) { // 20060815
	$1 = NULL;
  } else {
	temp = obj_to_array_no_conversion($input, PYARR_TC);
	//PyArrayObject *NAimg  = NA_InputArray($input, PYARR_TC, NUM_C_ARRAY);
	
	if (!temp  || !require_contiguous(temp) || !require_native(temp)) SWIG_fail;
	
	$1 = (CTYPE *) temp->data;
  }
}
%typemap(freearg)
  (CTYPE *array00d)
{
  debugPrintf("debug: "TYSTR" *array00d  -> Py_XDECREF\n");
//CHECK#20060722     Py_XDECREF(temp$argnum);
}
%enddef

SEBS_TYPEMAP00d(Byte,      "byte",     NPY_UBYTE)
SEBS_TYPEMAP00d(short,     "short",    NPY_SHORT)
SEBS_TYPEMAP00d(float,     "float",    NPY_FLOAT)
SEBS_TYPEMAP00d(complex32, "complex32",NPY_CFLOAT)
SEBS_TYPEMAP00d(complex64, "complex64",NPY_CDOUBLE)
SEBS_TYPEMAP00d(Word,      "Ushort",   NPY_USHORT)
SEBS_TYPEMAP00d(int,       "int",      NPY_INT)
SEBS_TYPEMAP00d(double,    "double",   NPY_DOUBLE)
SEBS_TYPEMAP00d(DWord,     "Uint",     NPY_UINT)
// no NPY_ULONG
SEBS_TYPEMAP00d(long,      "long",     NPY_LONG)

SEBS_TYPEMAP00d(void ,     "void",     NPY_ANY)  // 20060630
