// -*- c++ -*-
//__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
//__license__ = "BSD license - see LICENSE file"

//included from sebTypemaps.i

// %runtime %{
%wrapper %{

///////////////////////
/* ------------------------------------------------------------------------
 * for internal method declarations
 * ------------------------------------------------------------------------ */

#ifndef SWIGINTERN
#define SWIGINTERN static 
#endif

#ifndef SWIGINTERNSHORT
#ifdef __cplusplus
#define SWIGINTERNSHORT static inline 
#else /* C case */
#define SWIGINTERNSHORT static 
#endif /* __cplusplus */
#endif

/*
  Exception handling in wrappers
*/
//#define SWIG_fail                goto fail
//#define SWIG_arg_fail(arg)       SWIG_Python_ArgFail(arg)
//#define SWIG_append_errmsg(msg)   SWIG_Python_AddErrMesg(msg,0)
//#define SWIG_preppend_errmsg(msg) SWIG_Python_AddErrMesg(msg,1)
#define SWIG_type_error(type,obj) SWIG_Python_TypeError(type,obj)
//#define SWIG_null_ref(type)       SWIG_Python_NullRef(type)

// /* ----------------------------------------------------------------------
//  * Alloc. memory flags
//  * ---------------------------------------------------------------------- */
// #define SWIG_OLDOBJ  1
// #define SWIG_NEWOBJ  SWIG_OLDOBJ + 1
// #define SWIG_PYSTR   SWIG_NEWOBJ + 1

///////////////////////


///////////////////////////////////////////////////////
// for numpy scalars
///////////////////////////////////////////////////////

#include <numpy/arrayscalars.h>
#include <float.h>
#include <limits.h>

//////////////
SWIGINTERN int
  SEB_CheckDoubleInRange(double value, double min_value, 
			  double max_value, const char* errmsg)
{
  if (value < min_value) {
    if (errmsg) {
      PyErr_Format(PyExc_OverflowError, 
		   "value %g is less than %s minimum %g", 
		   value, errmsg, min_value);
    }
    return 0;
  } else if (value > max_value) {
    if (errmsg) {
      PyErr_Format(PyExc_OverflowError, 
		   "value %g is greater than %s maximum %g", 
		   value, errmsg, max_value);
    }
    return 0;
  }
  return 1;
}
SWIGINTERN int
  SEB_CheckLongInRange(long value, long min_value, long max_value,
			const char *errmsg)
{
  if (value < min_value) {
    if (errmsg) {
      PyErr_Format(PyExc_OverflowError, 
		   "value %ld is less than '%s' minimum %ld", 
		   value, errmsg, min_value);
    }
    return 0;    
  } else if (value > max_value) {
    if (errmsg) {
      PyErr_Format(PyExc_OverflowError,
		   "value %ld is greater than '%s' maximum %ld", 
		   value, errmsg, max_value);
    }
    return 0;
  }
  return 1;
}
SWIGINTERNSHORT int
  SEB_CheckUnsignedLongInRange(unsigned long value,
				unsigned long max_value,
				const char *errmsg) 
{
  if (value > max_value) {
    if (errmsg) {
      PyErr_Format(PyExc_OverflowError,
		   "value %lu is greater than '%s' minimum %lu",
		   value, errmsg, max_value);
    }
    return 0;
  }
  return 1;
 }



//////////////

SWIGINTERN int
  SEB_AsVal_double(PyObject *obj, double *val)
{
  if (PyFloat_Check(obj)) {
    if (val) *val = PyFloat_AS_DOUBLE(obj);
    return 1;
  }  
  if (PyInt_Check(obj)) {
    if (val) *val = PyInt_AS_LONG(obj);
    return 1;
  }
  if (PyLong_Check(obj)) {
    double v = PyLong_AsDouble(obj);
    if (!PyErr_Occurred()) {
      if (val) *val = v;
      return 1;
    } else {
      if (!val) PyErr_Clear();
      return 0;
    }
  }
  //seb numpy
  if (PyArray_IsScalar(obj, Number)) {
	PyArray_Descr *descr1;
	descr1 = PyArray_DescrFromTypeObject((PyObject *)(obj->ob_type));
 	if (PyArray_CanCastSafely(descr1->type_num, PyArray_DOUBLE)) {
	  if(val) PyArray_CastScalarDirect(obj, descr1, val, PyArray_DOUBLE);
	  Py_DECREF(descr1);
	  return 1;
	}
	else {
	  Py_DECREF(descr1);
	}
  }


  if (val) {
    SWIG_type_error("double", obj);
  }
  return 0;
}


SWIGINTERN int
  SEB_AsVal_float(PyObject *obj, float *val)
{
  const char* errmsg = val ? "float" : (char*)0;
  double v;
  if (SEB_AsVal_double(obj, &v)) {
    if (SEB_CheckDoubleInRange(v, -FLT_MAX, FLT_MAX, errmsg)) {
      if (val) *val = (float)(v);
      return 1;
    } else {
      return 0;
    }
  } else {
    PyErr_Clear();
  }
  if (val) {
    SWIG_type_error(errmsg, obj);
  }
  return 0;
}

SWIGINTERNSHORT float
SEB_As_float(PyObject* obj)
{
  float v;
  if (!SEB_AsVal_float(obj, &v)) {
    /*
      this is needed to make valgrind/purify happier. 
     */
    memset((void*)&v, 0, sizeof(float));
  }
  return v;
}

SWIGINTERNSHORT int
SEB_Check_float(PyObject* obj)
{
  return SEB_AsVal_float(obj, (float*)0);
}


#include <limits.h>



SWIGINTERN int
  SEB_AsVal_long(PyObject * obj, long* val)
{
  if (PyInt_Check(obj)) {
    if (val) *val = PyInt_AS_LONG(obj);
    return 1;
  }
  if (PyLong_Check(obj)) {
    long v = PyLong_AsLong(obj);
    if (!PyErr_Occurred()) {
      if (val) *val = v;
      return 1;
    } else {
      if (!val) PyErr_Clear();
      return 0;
    }
  }
  //seb numpy
  if (PyArray_IsScalar(obj, Number)) {
	PyArray_Descr *descr1;
	descr1 = PyArray_DescrFromTypeObject((PyObject *)(obj->ob_type));
 	if (PyArray_CanCastSafely(descr1->type_num, PyArray_LONG)) {
	  if(val) PyArray_CastScalarDirect(obj, descr1, val, PyArray_LONG);
	  Py_DECREF(descr1);
	  return 1;
	}
	else {
	  Py_DECREF(descr1);
	}
  }


  if (val) {
    SWIG_type_error("long", obj);
  }
  return 0;
 }


#if INT_MAX != LONG_MAX
SWIGINTERN int
  SEB_AsVal_int(PyObject *obj, int *val)
{ 
  const char* errmsg = val ? "int" : (char*)0;
  long v;
  if (SEB_AsVal_long(obj, &v)) {
    if (SEB_CheckLongInRange(v, INT_MIN,INT_MAX, errmsg)) {
      if (val) *val = (int)(v);
      return 1;
    } else {
      return 0;
    }
  } else {
    PyErr_Clear();
  }
  if (val) {
    SWIG_type_error(errmsg, obj);
  }
  return 0;    
}
#else
SWIGINTERNSHORT int
  SEB_AsVal_int(PyObject *obj, int *val)
{
  return SEB_AsVal_long(obj,(long*)val);
}
#endif


SWIGINTERN int
  SEB_AsVal_bool(PyObject *obj, bool *val)
{
  // //seb from numpy: PyArray_BoolConverter(obj, &bb);
  // 	if (PyObject_IsTrue(object))
  // 		*val=TRUE;
  // 	else *val=FALSE;
  // 	if (PyErr_Occurred())
  // 		return PY_FAIL;
  // 	return PY_SUCCEED;

  int bb = PyObject_IsTrue(obj);
  if (PyErr_Occurred()) {
	PyErr_Clear();
	if (val) {
	  SWIG_type_error("bool", obj);
	}
	return 0;
  } else {
	if (val) {
	  *val = (bool)bb;
	}
	return 1;
  }

/*
  } else {
	if (obj == Py_True) {
	  return 1;
	}
	if (obj == Py_False) {
	  return 1;
	}
  //numpy
  //if (PyArray_ISBOOL(obj))

	int res = 0;
	if (SEB_AsVal_int(obj, &res)) {    
	  if (val) *val = res ? true : false;
	  return 1;
	} else {
	  PyErr_Clear();
	}  
  if (val) {
    SWIG_type_error("bool", obj);
  }
  return 0;
*/
}


SWIGINTERNSHORT bool
SEB_As_bool(PyObject* obj)
{
  bool v;
  if (!SEB_AsVal_bool(obj, &v)) {
    /*
      this is needed to make valgrind/purify happier. 
     */
    memset((void*)&v, 0, sizeof(bool));
  }
  return v;
}


// /* returns SWIG_OLDOBJ if the input is a raw char*, SWIG_PYSTR if is a PyString */
// SWIGINTERN int
// SEB_AsCharPtrAndSize(PyObject *obj, char** cptr, size_t* psize)
// {
//   static swig_type_info* pchar_info = 0;
//   char* vptr = 0;
//   if (!pchar_info) pchar_info = SWIG_TypeQuery("char *");
//   if (SWIG_ConvertPtr(obj, (void**)&vptr, pchar_info, 0) != -1) {
//     if (cptr) *cptr = vptr;
//     if (psize) *psize = vptr ? (strlen(vptr) + 1) : 0;
//     return SWIG_OLDOBJ;
//   } else {
//     PyErr_Clear();
//     if (PyString_Check(obj)) {
//       if (cptr) {
// 	*cptr = PyString_AS_STRING(obj);
// 	if (psize) {
// 	  *psize = PyString_GET_SIZE(obj) + 1;
// 	}
//       }
//       return SWIG_PYSTR;
//     }
//   }
//   if (cptr) {
//     SWIG_type_error("char *", obj);
//   }
//   return 0;
// }


// SWIGINTERN int
// SEB_AsCharArray(PyObject *obj, char *val, size_t size)
// { 
//   char* cptr; size_t csize;  
//   if (SEB_AsCharPtrAndSize(obj, &cptr, &csize)) {
//     /* in C you can do:        

//          char x[5] = "hello"; 

//         ie, assing the array using an extra '0' char.
//     */
//     if ((csize == size + 1) && !(cptr[csize-1])) --csize;
//     if (csize <= size) {
//       if (val) {
// 	if (csize) memcpy(val, cptr, csize);
// 	if (csize < size) memset(val + csize, 0, size - csize);
//       }
//       return 1;
//     }
//   }
//   if (val) {
//     PyErr_Format(PyExc_TypeError,
// 		 "a char array of maximum size %lu is expected", 
// 		 (unsigned long) size);
//   }
//   return 0;
// }


// SWIGINTERN int
//   SEB_AsVal_char(PyObject *obj, char *val)
// {
//   const char* errmsg = val ? "char" : (char*)0;
//   long v;
//   if (SEB_AsVal_long(obj, &v)) {
//     if (SEB_CheckLongInRange(v, CHAR_MIN,CHAR_MAX, errmsg)) {
//       if (val) *val = (char)(v);
//       return 1;
//     } else {
//       return 0;
//     }
//   } else if (PyArray_IsScalar(obj, Byte)) { // numpy
// 	if (val) *val = (char) PyArrayScalar_VAL(obj, Byte);
// 	return 1;
//   } else {
//     PyErr_Clear();
//     return SEB_AsCharArray(obj, val, 1);
//   }
//  }


// SWIGINTERNSHORT char
// SEB_As_char(PyObject* obj)
// {
//   char v;
//   if (!SEB_AsVal_char(obj, &v)) {
//     /*
//       this is needed to make valgrind/purify happier. 
//      */
//     memset((void*)&v, 0, sizeof(char));
//   }
//   return v;
// }


SWIGINTERN int
  SEB_AsVal_short(PyObject *obj, short *val)
{ 
  const char* errmsg = val ? "short" : (char*)0;
  long v;
  if (SEB_AsVal_long(obj, &v)) {
    if (SEB_CheckLongInRange(v, SHRT_MIN, SHRT_MAX, errmsg)) {
      if (val) *val = (short)(v);
      return 1;
    } else {
      return 0;
    }    
  } else {
    PyErr_Clear();
  }
  if (val) {
    SWIG_type_error(errmsg, obj);
  }
  return 0;    
}


SWIGINTERNSHORT short
SEB_As_short(PyObject* obj)
{
  short v;
  if (!SEB_AsVal_short(obj, &v)) {
    /*
      this is needed to make valgrind/purify happier. 
     */
    memset((void*)&v, 0, sizeof(short));
  }
  return v;
}


SWIGINTERNSHORT int
SEB_As_int(PyObject* obj)
{
  int v;
  if (!SEB_AsVal_int(obj, &v)) {
    /*
      this is needed to make valgrind/purify happier. 
     */
    memset((void*)&v, 0, sizeof(int));
  }
  return v;
}


SWIGINTERNSHORT long
SEB_As_long(PyObject* obj)
{
  long v;
  if (!SEB_AsVal_long(obj, &v)) {
    /*
      this is needed to make valgrind/purify happier. 
     */
    memset((void*)&v, 0, sizeof(long));
  }
  return v;
}


SWIGINTERNSHORT double
SEB_As_double(PyObject* obj)
{
  double v;
  if (!SEB_AsVal_double(obj, &v)) {
    /*
      this is needed to make valgrind/purify happier. 
     */
    memset((void*)&v, 0, sizeof(double));
  }
  return v;
}


SWIGINTERN int
  SEB_AsVal_unsigned_SS_long(PyObject *obj, unsigned long *val) 
{
  if (PyInt_Check(obj)) {
    long v = PyInt_AS_LONG(obj);
    if (v >= 0) {
      if (val) *val = v;
      return 1;
    }   
  }
  if (PyLong_Check(obj)) {
    unsigned long v = PyLong_AsUnsignedLong(obj);
    if (!PyErr_Occurred()) {
      if (val) *val = v;
      return 1;
    } else {
      if (!val) PyErr_Clear();
      return 0;
    }
  } 
  if (val) {
    SWIG_type_error("unsigned long", obj);
  }
  return 0;
}




SWIGINTERN int
  SEB_AsVal_unsigned_SS_char(PyObject *obj, unsigned char *val)
{ 
  const char* errmsg = val ? "unsigned char" : (char*)0;
  unsigned long v;
  if (SEB_AsVal_unsigned_SS_long(obj, &v)) {
    if (SEB_CheckUnsignedLongInRange(v, UCHAR_MAX,errmsg)) {
      if (val) *val = (unsigned char)(v);
      return 1;
    } else {
      return 0;
    }
  } else {
    PyErr_Clear();
  }
  if (val) {
    SWIG_type_error(errmsg, obj);
  }
  return 0;
}


SWIGINTERNSHORT unsigned char
SEB_As_unsigned_SS_char(PyObject* obj)
{
  unsigned char v;
  if (!SEB_AsVal_unsigned_SS_char(obj, &v)) {
    /*
      this is needed to make valgrind/purify happier. 
     */
    memset((void*)&v, 0, sizeof(unsigned char));
  }
  return v;
}


SWIGINTERN int
  SEB_AsVal_unsigned_SS_short(PyObject *obj, unsigned short *val)
{ 
  const char* errmsg = val ? "unsigned short" : (char*)0;
  unsigned long v;
  if (SEB_AsVal_unsigned_SS_long(obj, &v)) {
    if (SEB_CheckUnsignedLongInRange(v, USHRT_MAX, errmsg)) {
      if (val) *val = (unsigned short)(v);
      return 1;
    } else {
      return 0;
    }
  } else {
    PyErr_Clear();
  }
  if (val) {
    SWIG_type_error(errmsg, obj);
  }
  return 0;
}


SWIGINTERNSHORT unsigned short
SEB_As_unsigned_SS_short(PyObject* obj)
{
  unsigned short v;
  if (!SEB_AsVal_unsigned_SS_short(obj, &v)) {
    /*
      this is needed to make valgrind/purify happier. 
     */
    memset((void*)&v, 0, sizeof(unsigned short));
  }
  return v;
}


#if UINT_MAX != ULONG_MAX
SWIGINTERN int
  SEB_AsVal_unsigned_SS_int(PyObject *obj, unsigned int *val)
{ 
  const char* errmsg = val ? "unsigned int" : (char*)0;
  unsigned long v;
  if (SEB_AsVal_unsigned_SS_long(obj, &v)) {
    if (SEB_CheckUnsignedLongInRange(v, INT_MAX, errmsg)) {
      if (val) *val = (unsigned int)(v);
      return 1;
    }
  } else {
    PyErr_Clear();
  }
  if (val) {
    SWIG_type_error(errmsg, obj);
  }
  return 0;    
}
#else
SWIGINTERNSHORT unsigned int
  SEB_AsVal_unsigned_SS_int(PyObject *obj, unsigned int *val)
{
  return SEB_AsVal_unsigned_SS_long(obj,(unsigned long *)val);
}
#endif


SWIGINTERNSHORT unsigned int
SEB_As_unsigned_SS_int(PyObject* obj)
{
  unsigned int v;
  if (!SEB_AsVal_unsigned_SS_int(obj, &v)) {
    /*
      this is needed to make valgrind/purify happier. 
     */
    memset((void*)&v, 0, sizeof(unsigned int));
  }
  return v;
}


SWIGINTERNSHORT unsigned long
SEB_As_unsigned_SS_long(PyObject* obj)
{
  unsigned long v;
  if (!SEB_AsVal_unsigned_SS_long(obj, &v)) {
    /*
      this is needed to make valgrind/purify happier. 
     */
    memset((void*)&v, 0, sizeof(unsigned long));
  }
  return v;
}

  
SWIGINTERNSHORT int
SEB_Check_bool(PyObject* obj)
{
  return SEB_AsVal_bool(obj, (bool*)0);
}

  
// SWIGINTERNSHORT int
// SEB_Check_char(PyObject* obj)
// {
//   return SEB_AsVal_char(obj, (char*)0);
// }

  
SWIGINTERNSHORT int
SEB_Check_short(PyObject* obj)
{
  return SEB_AsVal_short(obj, (short*)0);
}

  
SWIGINTERNSHORT int
SEB_Check_int(PyObject* obj)
{
  return SEB_AsVal_int(obj, (int*)0);
}

  
SWIGINTERNSHORT int
SEB_Check_long(PyObject* obj)
{
  return SEB_AsVal_long(obj, (long*)0);
}

  
SWIGINTERNSHORT int
SEB_Check_double(PyObject* obj)
{
  return SEB_AsVal_double(obj, (double*)0);
}

  
SWIGINTERNSHORT int
SEB_Check_unsigned_SS_char(PyObject* obj)
{
  return SEB_AsVal_unsigned_SS_char(obj, (unsigned char*)0);
}

  
SWIGINTERNSHORT int
SEB_Check_unsigned_SS_short(PyObject* obj)
{
  return SEB_AsVal_unsigned_SS_short(obj, (unsigned short*)0);
}

  
SWIGINTERNSHORT int
SEB_Check_unsigned_SS_int(PyObject* obj)
{
  return SEB_AsVal_unsigned_SS_int(obj, (unsigned int*)0);
}

  
SWIGINTERNSHORT int
SEB_Check_unsigned_SS_long(PyObject* obj)
{
  return SEB_AsVal_unsigned_SS_long(obj, (unsigned long*)0);
}



////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


%}


%define SEB_IN_TYPEMAP(TYPE, NAME)

%typemap(in) TYPE { 
$1 = (TYPE)(SEB_As_ ## NAME($input));
if (SWIG_arg_fail($argnum)) SWIG_fail;
}

%enddef

SEB_IN_TYPEMAP(bool, bool)
// SEB_IN_TYPEMAP(char, char)
SEB_IN_TYPEMAP(short, short)
SEB_IN_TYPEMAP(int, int)
SEB_IN_TYPEMAP(long, long)
SEB_IN_TYPEMAP(float, float)
SEB_IN_TYPEMAP(double, double)
SEB_IN_TYPEMAP(unsigned char,  unsigned_SS_char)
SEB_IN_TYPEMAP(unsigned short, unsigned_SS_short)
SEB_IN_TYPEMAP(unsigned int,   unsigned_SS_int)
SEB_IN_TYPEMAP(unsigned long,  unsigned_SS_long)
