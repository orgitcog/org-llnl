// -*- c++ -*-
//__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
//__license__ = "BSD license - see LICENSE file"

//included from sebTypemaps.i

//  ///////////////////////////////////////////////////////
//  //  checker to confirm what type of array we are talking about
//  ///////////////////////////////////////////////////////

//%define SEBS_TYPECHECK1d(SWIG_TC , CTYPE, TYSTR, PYARR_TC)
%define SEBS_TYPECHECK1d(SWIG_TC , CTYPE, TYSTR, MY_NPY_ARRTYPECHECK)
%typecheck(SWIG_TYPECHECK_##SWIG_TC) (CTYPE *array1d, int n) {
  debugPrintf("debug: typecheck1d "TYSTR"\n");
  debugPrintf("debug: is_array($input)%d array_type:%d\n", is_array($input), array_type($input));
  if(!is_array($input)) $1=0;
  //20070216 else if(PyArray_ISBYTESWAPPED($input)) $1=0;
  //else if( array_type($input) != PYARR_TC) $1=0;
  else if( !(MY_NPY_ARRTYPECHECK)) $1=0;
  else $1=1;
}
%enddef
//20041105  else if(((PyArrayObject*)$input)->nd >  1) $1=0;

//20060822  "int"  is NPY_LONG on 32bit ..but.. NPY_INT on 64bit
//20060822 (PyArray_ISSIGNED(self) && PyArray_ITEMSIZE(self)==SIZEOF_INT)

SEBS_TYPECHECK1d(UINT8,   Byte,      "byte",      uintarray_type_isize($input, 1))
SEBS_TYPECHECK1d(INT16,   short,     "short",     intarray_type_isize($input, SIZEOF_SHORT))
SEBS_TYPECHECK1d(FLOAT,   float,     "float",     array_type($input) ==NPY_FLOAT)
SEBS_TYPECHECK1d(COMPLEX, complex32, "complex32", array_type($input) ==NPY_CFLOAT)
SEBS_TYPECHECK1d(COMPLEX, complex64, "complex64", array_type($input) ==NPY_CDOUBLE)
SEBS_TYPECHECK1d(UINT16,  Word,      "Ushort",    array_type($input) ==NPY_USHORT)
SEBS_TYPECHECK1d(INT32,   int,       "int",       intarray_type_isize($input, SIZEOF_INT))
SEBS_TYPECHECK1d(DOUBLE,  double,    "double",    array_type($input) ==NPY_DOUBLE)
SEBS_TYPECHECK1d(UINT32,  DWord,     "Uint",      uintarray_type_isize($input, SIZEOF_INT))
// no NPY_ULONG
SEBS_TYPECHECK1d(INT64,   long,      "long",      intarray_type_isize($input, SIZEOF_LONG))

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

  /*20041105 debugPrintf(" **** numarray dim >1 (ns=%d)\n", NAimg->nd);
    20041105 SWIG_exception(SWIG_RuntimeError, "numarray dim > 1");
  */

%define MAP_SHAPE1D
  switch(temp->nd) {
  case 1:
	$2 = temp->dimensions[0];
	break;
  default:
   debugPrintf(" **** array dim  (ns=%d)\n", temp->nd);
 { int a=1; for(int i=0;i<temp->nd;i++) a*=temp->dimensions[i];
		$2 = a; }
}
%enddef


%define SEBS_TYPEMAP1d(CTYPE, TYSTR, PYARR_TC)
%typemap(in)
  (CTYPE *array1d, int n)
  (PyArrayObject *temp=NULL)
{
  debugPrintf("debug: "TYSTR" *array1d  -> NA_InputArray\n");
  temp = obj_to_array_no_conversion($input, PYARR_TC);
  //PyArrayObject *NAimg  = NA_InputArray($input, PYARR_TC, NUM_C_ARRAY);

  if (!temp  || !require_contiguous(temp) || !require_native(temp)) SWIG_fail;

  $1 = (CTYPE *) temp->data;
  MAP_SHAPE1D
}
%typemap(freearg)
  (CTYPE *array1d, int n)
{
  debugPrintf("debug: "TYSTR" *array1d  -> Py_XDECREF\n");
//CHECK#20060722     Py_XDECREF(temp$argnum);
}
%enddef

SEBS_TYPEMAP1d(Byte,      "byte",     NPY_UBYTE)
SEBS_TYPEMAP1d(short,     "short",    NPY_SHORT)
SEBS_TYPEMAP1d(float,     "float",    NPY_FLOAT)
SEBS_TYPEMAP1d(complex32, "complex32",NPY_CFLOAT)
SEBS_TYPEMAP1d(complex64, "complex64",NPY_CDOUBLE)
SEBS_TYPEMAP1d(Word,      "Ushort",   NPY_USHORT)
SEBS_TYPEMAP1d(int,      "int",       NPY_INT)
SEBS_TYPEMAP1d(double,    "double",   NPY_DOUBLE)
SEBS_TYPEMAP1d(DWord,     "Uint",     NPY_UINT)
// no NPY_ULONG
SEBS_TYPEMAP1d(long,      "long",     NPY_LONG)
