#!/bin/sh
set -ex

CXX=g++
PYTHONINCLUDE=/usr/include/python2.6

# set this to the installation prefix for graphlib (where the /include and /lib 
# directories reside)
GRAPHLIBPREFIX=/nfs/tmp2/lee218/deleteme/stat-bgq

# set this to the installation prefix for stackwalker (where the /include and 
# /lib directories reside)
STACKWALKERPREFIX=/nfs/tmp2/lee218/deleteme/stat-bgq

# set this to the STAT source code directory
STATSRCDIR=/nfs/tmp2/lee218/deleteme/tmp/6-STAT

mkdir -p ../lib

$CXX -c -O0 -g  -fPIC STAT_GraphRoutines.C -I$GRAPHLIBPREFIX/include -I./ -I$STATSRCDIR -DSTAT_NO_STAT_H
$CXX -c -O0 -g  -fPIC -I$PYTHONINCLUDE -I$GRAPHLIBPREFIX/include -o STAT_merge.o STAT_merge.C -I$STATSRCDIR -DSTAT_NO_STAT_H
$CXX -shared -o ../lib/_STATmerge.so STAT_merge.o STAT_GraphRoutines.o -L$GRAPHLIBPREFIX/lib -Wl,-rpath=$GRAPHLIBPREFIX/lib -llnlgraph

# This is optional.  It provides better address resolution than the default
# addr2line and can be enabled in core_stack_merge with the -s option.
$CXX -O0 -g -I$STACKWALKERPREFIX/include -L$STACKWALKERPREFIX/lib -Wl,-rpath=$STACKWALKERPREFIX/lib symt_addr2line.C -lsymtabAPI -o ../bin/internals/symt_addr2line
