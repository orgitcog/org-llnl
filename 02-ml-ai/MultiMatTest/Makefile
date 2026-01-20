GCC_FLAGS = -O3 -fopenmp -std=c99
GXX_FLAGS = -O3 -fopenmp -std=c++11
NVCC_FLAGS = -O3 -arch=sm_35 -Xcompiler "-O3" -ccbin g++

ifeq ($(DEBUG), yes)
  GCC_FLAGS = -O0 -g -fopenmp -std=c99
  GXX_FLAGS = -O0 -g -fopenmp -std=c++11
  NVCC_FLAGS = -O0 -g -G -arch=sm_35 -Xcompiler "-O0 -g" -ccbin g++
endif

CSRC  = $(wildcard *.c)
CXXSRC  = $(wildcard *.cpp)
OBJS = $(patsubst %.c, %.o, $(CSRC))
OBJS += $(patsubst %.cpp, %.o, $(CXXSRC))

mm: $(OBJS)
	g++ $(GXX_FLAGS) $(OBJS) -o multi

%.o: %.c Makefile
	g++ $(GXX_FLAGS) -c $< -o $@

%.o: %.cpp Makefile
	g++ $(GXX_FLAGS) -c $< -o $@

clean:
	rm -rf *.o multi
