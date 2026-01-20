//__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
//__license__ = "BSD license - see LICENSE file"

#ifndef __SEBs_INCLUDE
#define __SEBs_INCLUDE

//#define debugPrintf printf
#define debugPrintf //(X) // IGNORE
// #define IGNORE(X) 

typedef unsigned char  Byte;
typedef unsigned short Word;
typedef unsigned int   DWord;


#ifndef HAVE_ALREADY_COMPLEX32 

//20060722 #ifdef WIN32 // HACK
//20060722 #include <complex>
//20060722 //  using namespace std;
//20060722 typedef std::complex<float>  complex32;
//20060722 #else
//20060722 #include <complex.h>
//20060722 typedef complex<float>  complex32;
//20060722 #endif

#include <complex>
typedef std::complex<float>  complex32;

#endif
#ifndef HAVE_ALREADY_COMPLEX64

//20060722 #ifdef WIN32 // HACK
//20060722 #include <complex>
//20060722 //  using namespace std;
//20060722 typedef std::complex<double> complex64;
//20060722 #else
//20060722 #include <complex.h>
//20060722 typedef complex<double> complex64;
//20060722 #endif

#include <complex>
typedef std::complex<double> complex64;

#endif




//////////////////////////////////////////////////////////////////////////////
// helper shortcuts  for instantiating templated functions
// use:
//    first define:  '00' is a arbitrary 'tag'
// #define S_fn00(T) template void   clip<T >(T  *img, int nx);
//    then say:
// SEBS_FOR_ALL_NONCPLX(00)
//////////////////////////////////////////////////////////////////////////////

#define SEBS_FOR_ALL_NONCPLX(NN) S_fn##NN(Byte)  S_fn##NN(short)  S_fn##NN(float)    S_fn##NN(DWord) \
                                 S_fn##NN(Word)  S_fn##NN(int)    S_fn##NN(double)   S_fn##NN(long)


#define SEBS_FOR_ALL_NONCPLX_pair(NN) \
        S_fn##NN(Byte  ,Byte)  S_fn##NN(Byte  ,short)  S_fn##NN(Byte  ,float) S_fn##NN(Byte  ,Word)  S_fn##NN(Byte  ,int)    S_fn##NN(Byte  ,double)   S_fn##NN(Byte  ,DWord) S_fn##NN(Byte  ,long) \
        S_fn##NN(short ,Byte)  S_fn##NN(short ,short)  S_fn##NN(short ,float) S_fn##NN(short ,Word)  S_fn##NN(short ,int)    S_fn##NN(short ,double)   S_fn##NN(short ,DWord) S_fn##NN(short ,long) \
        S_fn##NN(float ,Byte)  S_fn##NN(float ,short)  S_fn##NN(float ,float) S_fn##NN(float ,Word)  S_fn##NN(float ,int)    S_fn##NN(float ,double)   S_fn##NN(float ,DWord) S_fn##NN(float ,long) \
        S_fn##NN(Word  ,Byte)  S_fn##NN(Word  ,short)  S_fn##NN(Word  ,float) S_fn##NN(Word  ,Word)  S_fn##NN(Word  ,int)    S_fn##NN(Word  ,double)   S_fn##NN(Word  ,DWord) S_fn##NN(Word  ,long) \
        S_fn##NN(int   ,Byte)  S_fn##NN(int   ,short)  S_fn##NN(int   ,float) S_fn##NN(int   ,Word)  S_fn##NN(int   ,int)    S_fn##NN(int   ,double)   S_fn##NN(int   ,DWord) S_fn##NN(int   ,long) \
        S_fn##NN(double,Byte)  S_fn##NN(double,short)  S_fn##NN(double,float) S_fn##NN(double,Word)  S_fn##NN(double,int)    S_fn##NN(double,double)   S_fn##NN(double,DWord) S_fn##NN(double,long) \
        S_fn##NN(DWord ,Byte)  S_fn##NN(DWord ,short)  S_fn##NN(DWord ,float) S_fn##NN(DWord ,Word)  S_fn##NN(DWord ,int)    S_fn##NN(DWord ,double)   S_fn##NN(DWord ,DWord) S_fn##NN(DWord ,long) \
        S_fn##NN(long  ,Byte)  S_fn##NN(long  ,short)  S_fn##NN(long  ,float) S_fn##NN(long  ,Word)  S_fn##NN(long  ,int)    S_fn##NN(long  ,double)   S_fn##NN(long  ,DWord) S_fn##NN(long  ,long)



#define SEBS_FOR_ALL_NONCPLX_CPLX_pair(NN) \
        S_fn##NN(Byte  ,complex32)  \
        S_fn##NN(short ,complex32)  \
        S_fn##NN(float ,complex32)  \
        S_fn##NN(Word  ,complex32)  \
        S_fn##NN(int   ,complex32)  \
        S_fn##NN(double,complex32)  \
        S_fn##NN(DWord ,complex32)  \
        S_fn##NN(long  ,complex32) 
//         S_fn##NN(Byte  ,complex64)  \
//         S_fn##NN(short ,complex64)  \
//         S_fn##NN(float ,complex64)  \
//         S_fn##NN(Word  ,complex64)  \
//         S_fn##NN(int   ,complex64)  \
//         S_fn##NN(double,complex64)  \
//         S_fn##NN(DWord ,complex64)  

#endif
