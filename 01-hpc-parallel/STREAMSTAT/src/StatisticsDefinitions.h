#ifndef STATS_DEFINITIONS_H
#define STATS_DEFINITIONS_H

#include <stdint.h>

static const double gStatisticsMaxValue = 10e34;
static const double gStatisticsMinValue = -10e34;


#define MIN(X,Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X,Y) (((X) > (Y)) ? (X) : (Y))


#ifndef NDEBUG

#include <cstdio>
#include <cstring>

#define stwarning(msg,...) {char error[200] = "WARNING: %s::%u:\n\t";strcat(error,msg);strcat(error,"\n");fprintf(stderr,error,__FILE__,__LINE__ , ## __VA_ARGS__);}
#define sterror(condition,msg,...) {if ((condition)) { char error[200] = "ERROR: %s::%u:\n\t";strcat(error,msg);strcat(error,"\n");fprintf(stderr,error,__FILE__,__LINE__ , ## __VA_ARGS__);assert(false);}}
#define stmessage(condition,msg,...)  {if ((condition)) { char error[200] = "WARNING: %s::%u:\n";strcat(error,msg);strcat(error,"\n");fprintf(stderr,error,__FILE__,__LINE__ , ## __VA_ARGS__);}}

#else

#define stwarning(msg,...) {;}
#define sterror(condition,msg,...) {;}
#define stmessage(condition,msg,...)  {;}

#endif


#endif
