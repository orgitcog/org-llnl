# Usage:
#  cd metall/tutorial/hands_on/
#  make

CXX = g++-8
CPPFLAGS = -pthread -std=c++17 -I../../include -I$(HOME)/boost
LDLIBS = -lstdc++fs

objects = t0 t1-1 t1-2 t2-1 t2-2 t3 t4-1 t4-2 t5-1 t5-2_create t5-2_open
all: $(objects)

$(objects): %: %.cpp
	$(CXX) $< $(CPPFLAGS) $(LDLIBS) -o $@

clean:
	/bin/rm -f $(objects)

.PHONY: all clean
