# Copyright 2020 Lawrence Livermore National Security, LLC and other
# minq developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

SLATEDIR = $(HOME)/files/codes/inq/slate
CUDADIR = /usr/local/cuda
CXX = mpic++
CXXFLAGS = -g -Ofast -fopenmp \
	-Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type \
	-I$(SLATEDIR)/include          \
	-I$(SLATEDIR)/blaspp/include   \
	-I$(SLATEDIR)/lapackpp/include \
	-I$(CUDADIR)/include

LDFLAGS = -L$(CUDADIR)/lib64/
LIBS = $(SLATEDIR)/lib/libslate.a $(SLATEDIR)/lapackpp/lib/liblapackpp.a $(SLATEDIR)/blaspp/lib/libblaspp.a -llapack -lblas -lcublas -lcudart -lgfortran

minq: minq.cpp
	$(CXX) $(CXXFLAGS) minq.cpp -o minq $(LDFLAGS) $(LIBS)

clean:
	rm -f minq

