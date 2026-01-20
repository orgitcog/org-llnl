// -*- c++ -*-
//__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
//__license__ = "BSD license - see LICENSE file"

//included from sebTypemaps.i

//  ///////////////////////////////////////////////////////
//  //  checker to confirm what type of array we are talking about
//  ///////////////////////////////////////////////////////

%define  SEBS_TYPECHECK3d(SWIG_TC, CTYPE, TYSTR, PYARR_TC)
%typecheck(SWIG_TYPECHECK_##SWIG_TC) (CTYPE *array3d, int nx, int ny, int nz) {
  debugPrintf("debug: typecheck3d "TYSTR" type_num: (%d)\n", PYARR_TC);
  debugPrintf("debug: is_array($input)%d\n", is_array($input));
  if(!is_array($input)) $1=0;
  else if( array_type($input) != PYARR_TC) $1=0;
  else if( array_dimensions($input) >  3) $1=0;
  else $1=1;
}
%enddef

SEBS_TYPECHECK3d(UINT8,   Byte,      "byte",      NPY_UBYTE)
SEBS_TYPECHECK3d(INT16,   short,     "short",     NPY_SHORT)
SEBS_TYPECHECK3d(FLOAT,   float,     "float",     NPY_FLOAT)
SEBS_TYPECHECK3d(COMPLEX, complex32, "complex32", NPY_CFLOAT)
SEBS_TYPECHECK3d(COMPLEX, complex64, "complex64", NPY_CDOUBLE)
SEBS_TYPECHECK3d(UINT16,  Word,      "Ushort",    NPY_USHORT)
SEBS_TYPECHECK3d(INT32,   int,       "int",       NPY_INT)
SEBS_TYPECHECK3d(DOUBLE,  double,    "double",    NPY_DOUBLE)
SEBS_TYPECHECK3d(UINT32,  DWord,     "Uint",      NPY_UINT)
// no NPY_ULONG
SEBS_TYPECHECK3d(INT64,   long,      "long",      NPY_LONG)

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

%define MAP_SHAPE3D
  switch(temp->nd) {
  case 1:
	$2 = temp->dimensions[0];
	$3=1;
	$4=1;
	break;
  case 2:
	$2 = temp->dimensions[1];
	$3=temp->dimensions[0];
	$4=1;
	break;
  case 3:
	$2 = temp->dimensions[2];
	$3=temp->dimensions[1];
	$4=temp->dimensions[0];
	break;
  default:
    debugPrintf(" **** array dim >3 (ns=%d)\n", temp->nd); 
    SWIG_exception(SWIG_RuntimeError, " array dim > 3");
    return 0;
}
%enddef

%define SEBS_TYPEMAP3d(CTYPE, TYSTR, PYARR_TC)
%typemap(in)
  (CTYPE *array3d, int nx, int ny, int nz)
  (PyArrayObject *temp=NULL)
{
  debugPrintf("debug: "TYSTR" *array3d  -> NA_InputArray\n");
  temp = obj_to_array_no_conversion($input, PYARR_TC);
  //PyArrayObject *NAimg  = NA_InputArray($input, PYARR_TC, NUM_C_ARRAY);

  if (!temp  || !require_contiguous(temp) || !require_native(temp)) SWIG_fail;

  $1 = (CTYPE *) temp->data;
  MAP_SHAPE3D
}
%typemap(freearg)
  (CTYPE *array3d, int nx, int ny, int nz)
{
  debugPrintf("debug: "TYSTR" *array3d  -> Py_XDECREF\n");
//CHECK#20060722      Py_XDECREF(temp$argnum);
}
%enddef

SEBS_TYPEMAP3d(Byte,      "byte",     NPY_UBYTE)
SEBS_TYPEMAP3d(short,     "short",    NPY_SHORT)
SEBS_TYPEMAP3d(float,     "float",    NPY_FLOAT)
SEBS_TYPEMAP3d(complex32, "complex32",NPY_CFLOAT)
SEBS_TYPEMAP3d(complex64, "complex64",NPY_CDOUBLE)
SEBS_TYPEMAP3d(Word,      "Ushort",   NPY_USHORT)
SEBS_TYPEMAP3d(int,      "int",       NPY_INT)
SEBS_TYPEMAP3d(double,    "double",   NPY_DOUBLE)
SEBS_TYPEMAP3d(DWord,     "Uint",     NPY_UINT)
// no NPY_ULONG
SEBS_TYPEMAP3d(long,      "long",     NPY_LONG)
