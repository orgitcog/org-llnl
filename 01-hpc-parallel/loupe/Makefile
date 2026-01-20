##############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
#
# Written by Emilio Castillo <ecastill@bsc.es>.
# LLNL-CODE-745958. All rights reserved.
#
# This file is part of Loupe. For details, see:
# https://github.com/LLNL/loupe
# Please also read the LICENSE file for the MIT License notice.
##############################################################################

MPI_DIST=/usr/tce/packages/mvapich2/mvapich2-2.2-intel-16.0.3/
PHDF5=/usr/tce/packages/hdf5/hdf5-parallel-1.8.18-intel-16.0.3-mvapich2-2.2/
WRAPPY=wrap/wrap.py

CFLAGS = -fPIC  -std=c++0x -O3 -I$(MPI_DIST)/include -Wall -g3
LDFLAGS= -L$(MPI_DIST)/lib -lmpi -L$(PHDF5)/lib

libmpidata: api.w mpid.cc util.cc
	python $(WRAPPY) api.w > api.cc
	g++ -I$(PHDF5)/include/ -c hdf5_dump.cc $(CFLAGS)
	g++ -c $(CFLAGS) api.cc mpid.cc util.cc
	g++ -shared -o libmpidata.so api.o mpid.o util.o hdf5_dump.o -lunwind -lhdf5 $(LDFLAGS)

clean:
	rm *.o libmpidata.so
