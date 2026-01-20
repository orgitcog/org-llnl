// -*- c++ -*-
//__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
//__license__ = "BSD license - see LICENSE file"

//  The "argout" typemap is used to return values from arguments. This is most commonly used to write wrappers for C/C++ functions that need to return multiple values. The "argout" typemap is almost always combined with an "in" typemap---possibly to ignore the input value. For example: 
   //  /* Set the input argument to point to a temporary variable */
   //  %typemap(in, numinputs=0) int *out (int temp) {
   //     $1 = &temp;
   //  }
   
   //  %typemap(argout) int *out {
   //     // Append output value $1 to $result
   //     ...
   //  }
   
   //  The following special variables are available. 
   //  $result           - Result object returned to target language.
   //  $input            - The original input object passed.
   //  $symname          - Name of function/method being wrapped
   
   //  The code supplied to the "argout" typemap is always placed after the "out" typemap. If multiple return values are used, the extra return values are often appended to return value of the function. 

/*
%typemap(argout) (char **oStr, int *oN) {
  //seb($argnum)
  //seb($0_name)
  //seb($1_name)
  //seb($2_name)

  //  $n      		A C local variable corresponding to type n in the typemap pattern.  
  //  $argnum 		Argument number. Only available in typemaps related to argument conversion 
  //  $n_name 		Argument name 
  //  $n_type 		Real C datatype of type n. 
  //  $n_ltype 		ltype of type n 
  //  $n_mangle 		Mangled form of type n. For example _p_Foo 
  //  $n_descriptor 	Type descriptor structure for type n. For example SWIGTYPE_p_Foo. 
  //                  This is primarily used when interacting with the run-time type checker(described later 
  //  $*n_type 		Real C datatype of type n with one pointer removed. 
  //  $*n_ltype 		ltype of type n with one pointer removed. 
  //  $*n_mangle 		Mangled form of type n with one pointer removed.  
  //  $*n_descriptor 		Type descriptor structure for type n with one pointer removed.  
  //  $&n_type 		Real C datatype of type n with one pointer added. 
  //  $&n_ltype 		ltype of type n with one pointer added. 
  //  $&n_mangle 		Mangled form of type n with one pointer added. 
  //  $&n_descriptor 		Type descriptor structure for type n with one pointer added.  
  //  $n_basetype 		Base typename with all pointers and qualifiers stripped.  
  
  //  $0      		A C local variable corresponding to type n in the typemap pattern.  
  //  $argnum 		Argument number. Only available in typemaps related to argument conversion 
  //  $0_name 		Argument name 
  //  $0_type 		Real C datatype of type n. 
  //  $0_ltype 		ltype of type n 
  //  $0_mangle 		Mangled form of type n. For example _p_Foo 
  //  $0_descriptor 	Type descriptor structure for type n. For example SWIGTYPE_p_Foo. 
  //                  This is primarily used when interacting with the run-time type checker(described later 
  //  $*0_type 		Real C datatype of type n with one pointer removed. 
  //  $*0_ltype 		ltype of type n with one pointer removed. 
  //  $*0_mangle 		Mangled form of type n with one pointer removed.  
  //  $*0_descriptor 		Type descriptor structure for type n with one pointer removed.  
  //  $&0_type 		Real C datatype of type n with one pointer added. 
  //  $&0_ltype 		ltype of type n with one pointer added. 
  //  $&0_mangle 		Mangled form of type n with one pointer added. 
  //  $&0_descriptor 		Type descriptor structure for type n with one pointer added.  
  //  $0_basetype 		Base typename with all pointers and qualifiers stripped.  
  


  printf("%typemap(argout) (char **oStr, int *oN)\n");

  $result = PyString_FromStringAndSize(*$1, *$2);

  //  The following special variables are available. 
  //  $result      $ result     - Result object returned to target language.
  //  $input       $ input     - The original input object passed.
  //  $symname     $ symname     - Name of function/method being wrapped
}
*/
/*
%typemap(in) (char **oStr, int *oN) {
 printf("%typemap(in) (char **oStr, int *oN)\n");
}
*/
%typemap(out) (char **oStr, int *oN) {
  //seb($argnum)
  //seb($0_name)
  //seb($1_name)
  //seb($2_name)

  //  $n      		A C local variable corresponding to type n in the typemap pattern.  
  //  $argnum 		Argument number. Only available in typemaps related to argument conversion 
  //  $n_name 		Argument name 
  //  $n_type 		Real C datatype of type n. 
  //  $n_ltype 		ltype of type n 
  //  $n_mangle 		Mangled form of type n. For example _p_Foo 
  //  $n_descriptor 	Type descriptor structure for type n. For example SWIGTYPE_p_Foo. 
  //                  This is primarily used when interacting with the run-time type checker(described later 
  //  $*n_type 		Real C datatype of type n with one pointer removed. 
  //  $*n_ltype 		ltype of type n with one pointer removed. 
  //  $*n_mangle 		Mangled form of type n with one pointer removed.  
  //  $*n_descriptor 		Type descriptor structure for type n with one pointer removed.  
  //  $&n_type 		Real C datatype of type n with one pointer added. 
  //  $&n_ltype 		ltype of type n with one pointer added. 
  //  $&n_mangle 		Mangled form of type n with one pointer added. 
  //  $&n_descriptor 		Type descriptor structure for type n with one pointer added.  
  //  $n_basetype 		Base typename with all pointers and qualifiers stripped.  

  //  $0      		A C local variable corresponding to type n in the typemap pattern.  
  //  $argnum 		Argument number. Only available in typemaps related to argument conversion 
  //  $0_name 		Argument name 
  //  $0_type 		Real C datatype of type n. 
  //  $0_ltype 		ltype of type n 
  //  $0_mangle 		Mangled form of type n. For example _p_Foo 
  //  $0_descriptor 	Type descriptor structure for type n. For example SWIGTYPE_p_Foo. 
  //                  This is primarily used when interacting with the run-time type checker(described later 
  //  $*0_type 		Real C datatype of type n with one pointer removed. 
  //  $*0_ltype 		ltype of type n with one pointer removed. 
  //  $*0_mangle 		Mangled form of type n with one pointer removed.  
  //  $*0_descriptor 		Type descriptor structure for type n with one pointer removed.  
  //  $&0_type 		Real C datatype of type n with one pointer added. 
  //  $&0_ltype 		ltype of type n with one pointer added. 
  //  $&0_mangle 		Mangled form of type n with one pointer added. 
  //  $&0_descriptor 		Type descriptor structure for type n with one pointer added.  
  //  $0_basetype 		Base typename with all pointers and qualifiers stripped.  



  printf("%typemap(out) (char **oStr, int *oN)\n");

  $result = PyString_FromStringAndSize(*$1, *$2);

  //  The following special variables are available. 
  //  $result      $ result     - Result object returned to target language.
  //  $input       $ input     - The original input object passed.
  //  $symname     $ symname     - Name of function/method being wrapped
}
%typemap(freearg) (char **oStr, int *oN) {
 printf("%typemap(freearg) (char **oStr, int *oN)\n");
}

%typemap(newfree) (char **oStr, int *oN) {
 printf("%typemap(newfree (char **oStr, int *oN)\n");
}



////////////////////////////////////////////////////////
%typemap(out) (int *INTARRRET) {
  printf("%typemap(out) (int *INTARRRET)\n");
  char s[20];
  sprintf(s, "_seb_%p", $1);
  $result = PyString_FromString(s);

  //seb($argnum)
  //seb($0_name)
  //seb($1_name)
  //seb($2_name)

  //  $n      		A C local variable corresponding to type n in the typemap pattern.  
  //  $argnum 		Argument number. Only available in typemaps related to argument conversion 
  //  $n_name 		Argument name 
  //  $n_type 		Real C datatype of type n. 
  //  $n_ltype 		ltype of type n 
  //  $n_mangle 		Mangled form of type n. For example _p_Foo 
  //  $n_descriptor 	Type descriptor structure for type n. For example SWIGTYPE_p_Foo. 
  //                  This is primarily used when interacting with the run-time type checker(described later 
  //  $*n_type 		Real C datatype of type n with one pointer removed. 
  //  $*n_ltype 		ltype of type n with one pointer removed. 
  //  $*n_mangle 		Mangled form of type n with one pointer removed.  
  //  $*n_descriptor 		Type descriptor structure for type n with one pointer removed.  
  //  $&n_type 		Real C datatype of type n with one pointer added. 
  //  $&n_ltype 		ltype of type n with one pointer added. 
  //  $&n_mangle 		Mangled form of type n with one pointer added. 
  //  $&n_descriptor 		Type descriptor structure for type n with one pointer added.  
  //  $n_basetype 		Base typename with all pointers and qualifiers stripped.  

  //  $0      		A C local variable corresponding to type n in the typemap pattern.  
  //  $argnum 		Argument number. Only available in typemaps related to argument conversion 
  //  $0_name 		Argument name 
  //  $0_type 		Real C datatype of type n. 
  //  $0_ltype 		ltype of type n 
  //  $0_mangle 		Mangled form of type n. For example _p_Foo 
  //  $0_descriptor 	Type descriptor structure for type n. For example SWIGTYPE_p_Foo. 
  //                  This is primarily used when interacting with the run-time type checker(described later 
  //  $*0_type 		Real C datatype of type n with one pointer removed. 
  //  $*0_ltype 		ltype of type n with one pointer removed. 
  //  $*0_mangle 		Mangled form of type n with one pointer removed.  
  //  $*0_descriptor 		Type descriptor structure for type n with one pointer removed.  
  //  $&0_type 		Real C datatype of type n with one pointer added. 
  //  $&0_ltype 		ltype of type n with one pointer added. 
  //  $&0_mangle 		Mangled form of type n with one pointer added. 
  //  $&0_descriptor 		Type descriptor structure for type n with one pointer added.  
  //  $0_basetype 		Base typename with all pointers and qualifiers stripped.  





  //  The following special variables are available. 
  //  $result      $ result     - Result object returned to target language.
  //  $input       $ input     - The original input object passed.
  //  $symname     $ symname     - Name of function/method being wrapped
}
%typemap(freearg) (int *INTARRRET) {
 printf("%typemap(freearg) (int *INTARRRET)\n");
}

%typemap(newfree) (int *INTARRRET) {
 printf("%typemap(newfree (int *INTARRRET)\n");
}



%typemap(in) (int wwwt) {
 printf("%typemap(in) wwwt\n");
}
