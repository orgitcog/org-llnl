# -*- Makefile -*-
#__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
#__license__ = "BSD license - see LICENSE file"
# PYPATH_DIR ?= /cygdrive/c/PrCyg/Priithon
# >>> N.get_numpy_include()
# 'C:\PrWin24\numpy\core\include'
# >>> 

PYVER ?=2.5
PYVER_NODOT =$(subst .,,$(PYVER))

#2009-07-14 PYPATH_DIR ?= /cygdrive/c/Priithon_$(PYVER_NODOT)_win/Priithon
PYPATH_DIR?=.

ifeq ($(origin CXX), default)
	CXX = g++ -mno-cygwin 
endif
ifeq ($(origin CC), default)
	CC = gcc -mno-cygwin 
endif
ifeq ($(origin FC), default)
	FC  = g77 -mno-cygwin 
endif

#OPT_FLAGS = -O4
#EXTRA_LIBS=
#INCL=
#MODULE=
#OBJS= seb1.o #$(MODULE).o

SWIG_MODULE=$(MODULE)

ifeq ($(SWIG_INTERFACE),)
   SWIG_INTERFACE=$(SWIG_MODULE).i
endif

SWIG_CXX = $(SWIG_INTERFACE:.i=_wrap.cxx)
SWIG_OBJ = $(SWIG_INTERFACE:.i=_wrap.o)
SWIG_DLL = _$(MODULE).pyd
SWIG_PY  = $(MODULE).py
#LIBFILE = _$(MODULE)module.bundle
LIBFILE = $(PYPATH_DIR)/$(SWIG_DLL)

all: $(LIBFILE) $(PYPATH_DIR)/$(SWIG_PY) $(OBJS)


%.o: %.cpp
	$(CXX) -DUSE_DL_IMPORT $(OPT_FLAGS) $(CXXFLAGS) $(INCL) $(INC_PY) -c $<

%.o: %.c
	$(CC)  -DUSE_DL_IMPORT $(OPT_FLAGS) $(CFLAGS) $(INCL) $(INC_PY) -c $<

%.o: %.f
	$(FC)  -DUSE_DL_IMPORT $(OPT_FLAGS) $(FFLAGS) -c $<  


############################################################
############################################################

SWIG=swig -I${PRCOMMON} -I${PRCOMMON}/Include
##SWIGDIR=c:\SWIG-1.3.19
##SWIG=$(SWIGDIR)/swig -I$(SWIGDIR)/Lib -I$(SWIGDIR)/Lib/python \
#    -I${PRCOMMON}
#   -I$(SWIG_INCL_PRIITHON) -I$(SWIG_INCL_PRIITHON)/common


###INC_PY = -I/usr/include/python2.3 \
###         -I/cygdrive/c/Python22/include/
#WIN32 for #include <complex>
# the first two -D are for pyconfig.h PY_LONG_LONG 
#WITHOUT -mno-cygwin 
#20060722 INC_PY = -D__GNUC__ -D_WIN32  -DWIN32 -I/cygdrive/c/Python24/include/ -I${PRCOMMON} -I${PRCOMMON}/Include
INC_PY = -D_WIN32  -DWIN32 -I/cygdrive/c/Python$(PYVER_NODOT)/include/ -I${PRCOMMON} -I${PRCOMMON}/Include
#INC_PY = -D__GNUC__ -D_WIN32 -I/cygdrive/c/Python24/include/ -I${PRCOMMON} -I${PRCOMMON}/Include

############################################################


$(LIBFILE): $(SWIG_OBJ) $(OBJS)
	 @#$(CXX) $(OPT_FLAGS) -shared $^ -o $@ $(EXTRA_LIBS)
	 @#$(CC) -shared $^ -o $@ $(EXTRA_LIBS)
	 @#ld -shared $^ -o $@ $(EXTRA_LIBS)
	 @#xp	$(CXX) -shared -Wl,--enable-auto-image-base $^ -o $@ -L/usr/lib/python2.3/config -lpython2.3 $(EXTRA_LIBS)
	$(CXX) $(OPT_FLAGS) -shared -Wall $^ -o $@ \
              -L/cygdrive/c/python$(PYVER_NODOT)/libs \
                          -lpython$(PYVER_NODOT) $(EXTRA_LIBS)

	@#20070915 ##$(CXX) $(OPT_FLAGS) -shared -Wl,--enable-auto-image-base $^ -o $@

# 	g++ -shared $^ -o $(PYPATH_DIR)/cyg$(MODULE).dll \
# 	    -Wl,--out-implib=$(PYPATH_DIR)/lib$(MODULE).dll.a \
# 	    -Wl,--export-all-symbols \
# 	    -Wl,--enable-auto-import \
# 	    -Wl,--whole-archive ${old_lib} \
# 	    -Wl,--no-whole-archive ${dependency_libs}


$(SWIG_OBJ):  $(SWIG_CXX)
	@# $(CXX)  $(OPT_FLAGS) -fpic -c $< $(CXXFLAGS) $(INCL) $(INC_PY)
	$(CXX)  $(OPT_FLAGS) -c $< $(CXXFLAGS) $(INCL) $(INC_PY)

$(SWIG_CXX): $(SWIG_INTERFACE)
	$(SWIG) -Wall -python -shadow -globals v -c++ $<

ifneq ($(PYPATH_DIR),.)
$(PYPATH_DIR)/$(SWIG_PY): $(SWIG_PY)
	cp -p $(SWIG_PY) $(PYPATH_DIR)
endif

cleanall: 
	rm -f $(OBJS) $(SWIG_CXX) $(SWIG_OBJ) \
             $(LIBFILE)    $(SWIG_PY) $(SWIG_PY)c \
	     $(PYPATH_DIR)/$(SWIG_PY) $(PYPATH_DIR)/$(SWIG_PY)c

#clean all except swig wrapper
clean: 
	rm -f $(OBJS) $(SWIG_OBJ) $(LIBFILE)

dep:
	@#	/usr/X11R6/bin/makedepend.exe -f Makefile_$(shell uname).dep --
	makedepend -fMakefile_$(shell uname).dep -- \
	        $(CFLAGS) $(INC_PY) -- \
		*.{i,cpp,h,cxx,c,f}
	@echo " *** Don't forget make dep  AFTER make $(MODULE)_wrap.cxx ***"
