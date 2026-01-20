CFLAGS += $(pkg-config --cflags hwloc)
LDLIBS += $(pkg-config --libs hwloc)
HWLOC_ROOT = /g/g90/loussert/SOFTS/hwloc-2.0.4/INSTALL/
HWLOC_INCS = -I$(HWLOC_ROOT)/include
HWLOC_LIBS = -L$(HWLOC_ROOT)/lib -lhwloc

CFLAGS = -Wall $(HWLOC_INCS)
LDLIBS = $(HWLOC_LIBS)

all:
	gcc mpibind_launch.c $(CFLAGS) -o exe $(LDLIBS) -L/g/g90/loussert/Work/loussert-mpibind/bin -lmpibind
