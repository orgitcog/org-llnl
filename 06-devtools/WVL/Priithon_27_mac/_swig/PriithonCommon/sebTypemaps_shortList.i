// -*- c++ -*-
//__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
//__license__ = "BSD license - see LICENSE file"

//included from sebTypemaps.i

%define SEBS_TYPEMAP_SL(CTYPE, TYSTR, CNV_O2C)
%typemap(in) CTYPE shortList[ANY] (CTYPE temp[$1_dim0]) {
  int i;
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_ValueError,
					"Expected a sequence of $1_dim0 "TYSTR" numbers");
    return NULL;
  }
  if (PySequence_Length($input) != $1_dim0) {
    PyErr_SetString(PyExc_ValueError,
					"Size mismatch. Expected $1_dim0 elements ("TYSTR" numbers)");
    return NULL;
  }
  for (i = 0; i < $1_dim0; i++) {
    PyObject *o = PySequence_GetItem($input,i);
    if (PyNumber_Check(o)) {
      temp[i] = (CTYPE) CNV_O2C;
    } else {
      PyErr_SetString(PyExc_ValueError,"Sequence elements must be ("TYSTR") numbers");
      return NULL;
    }
  }
  $1 = temp;
}
%enddef

SEBS_TYPEMAP_SL(Byte,      "byte",    PyInt_AsLong(o)          )
SEBS_TYPEMAP_SL(short,     "short",   PyInt_AsLong(o)          )
SEBS_TYPEMAP_SL(float,     "float",   PyFloat_AsDouble(o)      )
SEBS_TYPEMAP_SL(complex32, "complex32", PyComplex_AsCComplex(o)  )
SEBS_TYPEMAP_SL(complex64, "complex64", PyComplex_AsCComplex(o)  )
SEBS_TYPEMAP_SL(Word,      "Ushort",  PyInt_AsLong(o)          )
SEBS_TYPEMAP_SL(int,       "int",     PyInt_AsLong(o)          )
SEBS_TYPEMAP_SL(double,    "double",  PyFloat_AsDouble(o)      )
SEBS_TYPEMAP_SL(DWord,     "Uint",    PyInt_AsLong(o)          )




%define SEBS_TYPEMAP_SL_OF_ARR(CTYPE, TYSTR, PYARR_TC)
%typemap(in) CTYPE* shortList[ANY] (CTYPE *temp[$1_dim0]) {
  int i;
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"Expected a sequence of $1_dim0 "TYSTR"s");
    return NULL;
  }
  if (PySequence_Length($input) != $1_dim0) {
    PyErr_SetString(PyExc_ValueError,"Size mismatch. Expected $1_dim0 elements");
    return NULL;
  }
  for (i = 0; i < $1_dim0; i++) {
    PyObject *o = PySequence_GetItem($input,i);

	PyArrayObject *arrObj  = obj_to_array_no_conversion(o, PYARR_TC);
	if (!arrObj  || !require_contiguous(arrObj) || !require_native(temp)) SWIG_fail;
	temp[i] = (CTYPE *) arrObj->data;
  }
  $1 = temp;
}
%enddef

SEBS_TYPEMAP_SL_OF_ARR(Byte,      "byte",      NPY_UBYTE)
SEBS_TYPEMAP_SL_OF_ARR(short,     "short",     NPY_SHORT)
SEBS_TYPEMAP_SL_OF_ARR(float,     "float",     NPY_FLOAT)
SEBS_TYPEMAP_SL_OF_ARR(complex32, "complex32", NPY_CFLOAT)
SEBS_TYPEMAP_SL_OF_ARR(complex64, "complex64", NPY_CDOUBLE)
SEBS_TYPEMAP_SL_OF_ARR(Word,      "Ushort",    NPY_USHORT)
SEBS_TYPEMAP_SL_OF_ARR(int,       "int",       NPY_INT)
SEBS_TYPEMAP_SL_OF_ARR(double,    "double",    NPY_DOUBLE)
SEBS_TYPEMAP_SL_OF_ARR(DWord,     "Uint",      NPY_UINT)
// no NPY_ULONG
SEBS_TYPEMAP_SL_OF_ARR(long,       "long",       NPY_LONG)
