// -*- c++ -*-
//__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
//__license__ = "BSD license - see LICENSE file"

//20060722 %feature("autodoc", "3");
%feature("autodoc", "1");
//%feature("autodoc","extended");

%include exception.i
%include typemaps.i

%include sebInclude.h

%{
//check //#inlude "Python.h"
//what is this ??:  #  define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
%}

//see numpy.i
%init %{
import_array();
%}

/* Get the Numeric typemaps */
%include "numpy.i"
//#define array_type(a)          (int)(PyArray_TYPE(a))
//#define is_array(a)            ((a) && PyArray_Check(a))
//define array_dimensions(a)    (((PyArrayObject *)a)->nd)


//20060722
//these 3 defines appaer to be needed because I wrote the SWIG typemaps w/ shortcuts
//           SWIG appears to first apply typedef to functions, then the typemap would NOT bite
#define Byte unsigned char
#define Word unsigned short
#define DWord unsigned int

%include sebTypemaps1.i  // 1d array typemaps
%include sebTypemaps2.i  // 2d array typemaps
%include sebTypemaps3.i  // 3d array typemaps
%include sebTypemaps00.i  // nd array typemaps - no safety ! - just base-pointer
%include sebTypemaps_shortList.i  // shortList[n]
%include sebTypemaps_scalar.i  // for numpy scalars


// Grab a Python function object as a Python object.
%typemap(in) PyObject *pyfunc {
  if (!PyCallable_Check($input)) {
      PyErr_SetString(PyExc_TypeError, "Need a callable object!");
      return NULL;
  }
  $1 = $input;
}


//  ///////////////////////////////////////////////////////
//  //  helper function for SWIG typemaper ////////////////
//  ///////////////////////////////////////////////////////

%{
static void argout_Append_To_Result(PyObject *&RESULT, PyObject *o)
{
  // return o - if nothing to be returnd yet, only o
  //          -    otherwise append o to result tuple
  PyObject *o2, *o3;
  if ((!RESULT) || (RESULT == Py_None)) {
	RESULT = o;
  } else {
	if (!PyTuple_Check(RESULT)) {
	  PyObject *o2 = RESULT;
	  RESULT = PyTuple_New(1);
	  PyTuple_SetItem(RESULT,0,o2);
	}
	o3 = PyTuple_New(1);
	PyTuple_SetItem(o3,0,o);
	o2 = RESULT;
	RESULT = PySequence_Concat(o2,o3);
	Py_DECREF(o2);
	Py_DECREF(o3);
  }
}
%}


%include sebTypemapsTest.i

// SHORTCUT  DEFINES

%define SEBS_template_ALL_NONCPLX(fn)
%template(fn) fn<Byte >;
%template(fn) fn<short>;
%template(fn) fn<float>;
%template(fn) fn<Word>;
%template(fn) fn<int>;
%template(fn) fn<double>;
%template(fn) fn<DWord>;
%template(fn) fn<long>;
%enddef


%define SEBS_template_ALL_NONCPLX_pair(fn)
%template(fn) fn<Byte  ,Byte >; %template(fn) fn<Byte  ,short>;%template(fn) fn<Byte  ,float>;%template(fn) fn<Byte  ,Word>;%template(fn) fn<Byte  ,int >;%template(fn) fn<Byte  ,double >;%template(fn) fn<Byte  ,DWord>;%template(fn) fn<Byte  ,long>;
%template(fn) fn<short ,Byte >; %template(fn) fn<short ,short>;%template(fn) fn<short ,float>;%template(fn) fn<short ,Word>;%template(fn) fn<short ,int >;%template(fn) fn<short ,double >;%template(fn) fn<short ,DWord>;%template(fn) fn<short ,long>;
%template(fn) fn<float ,Byte >; %template(fn) fn<float ,short>;%template(fn) fn<float ,float>;%template(fn) fn<float ,Word>;%template(fn) fn<float ,int >;%template(fn) fn<float ,double >;%template(fn) fn<float ,DWord>;%template(fn) fn<float ,long>;
%template(fn) fn<Word  ,Byte >; %template(fn) fn<Word  ,short>;%template(fn) fn<Word  ,float>;%template(fn) fn<Word  ,Word>;%template(fn) fn<Word  ,int >;%template(fn) fn<Word  ,double >;%template(fn) fn<Word  ,DWord>;%template(fn) fn<Word  ,long>;
%template(fn) fn<int   ,Byte >; %template(fn) fn<int   ,short>;%template(fn) fn<int   ,float>;%template(fn) fn<int   ,Word>;%template(fn) fn<int   ,int >;%template(fn) fn<int   ,double >;%template(fn) fn<int   ,DWord>;%template(fn) fn<int   ,long>;
%template(fn) fn<double,Byte >; %template(fn) fn<double,short>;%template(fn) fn<double,float>;%template(fn) fn<double,Word>;%template(fn) fn<double,int >;%template(fn) fn<double,double >;%template(fn) fn<double,DWord>;%template(fn) fn<double,long>;
%template(fn) fn<DWord ,Byte >; %template(fn) fn<DWord ,short>;%template(fn) fn<DWord ,float>;%template(fn) fn<DWord ,Word>;%template(fn) fn<DWord ,int >;%template(fn) fn<DWord ,double >;%template(fn) fn<DWord ,DWord>;%template(fn) fn<DWord ,long>;
%template(fn) fn<long  ,Byte >; %template(fn) fn<long  ,short>;%template(fn) fn<long  ,float>;%template(fn) fn<long  ,Word>;%template(fn) fn<long  ,int >;%template(fn) fn<long  ,double >;%template(fn) fn<long  ,DWord>;%template(fn) fn<long  ,long>;
%enddef

%define SEBS_template_ALL_NONCPLX_CPLX_pair(fn)
%template(fn) fn<Byte  ,complex32 >;
%template(fn) fn<short ,complex32 >;
%template(fn) fn<float ,complex32 >;
%template(fn) fn<Word  ,complex32 >;
%template(fn) fn<int   ,complex32 >;
%template(fn) fn<double,complex32 >;
%template(fn) fn<DWord ,complex32 >;
%template(fn) fn<long  ,complex32 >;
// %template(fn) fn<Byte  ,complex64 >;
// %template(fn) fn<short ,complex64 >;
// %template(fn) fn<float ,complex64 >;
// %template(fn) fn<Word  ,complex64 >;
// %template(fn) fn<int   ,complex64 >;
// %template(fn) fn<double,complex64 >;
// %template(fn) fn<DWord ,complex64 >; 
%enddef

//  #define SEBS_RECURSIVE__ALL_NONCPLX( MACRO ) \
//  MACRO( Byte   )\
//  MACRO( short  )\
//  MACRO( float  )\
//  MACRO( Word   )\
//  MACRO( int   )\
//  MACRO( double )

//never used: #define __intNXYZ__ int nx, int ny, int nz

