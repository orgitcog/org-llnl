#-*- Makefile -*-
#__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
#__license__ = "BSD license - see LICENSE file"

##?? why ist this varibale not used in INC_PY ????  PYVER ?= 2.5

#2015 ARCH_OPT ?= -arch i386 -arch ppc -mmacosx-version-min=10.4
ARCH_OPT ?= -arch i386 -arch x86_64 -mmacosx-version-min=10.8


#2009-07-14 PYPATH_DIR?=$(HOME)/Priithon_25_Mac
PYPATH_DIR?=.

# ifeq ($(shell uname -p),i386)
#    PYPATH_DIR?=/Users/${USER}/PrMacNInt/Priithon
# else
#    PYPATH_DIR?=/Users/${USER}/PrMacNPPC/Priithon
# endif

ifeq ($(origin CXX), default)
	CXX = c++
endif
ifeq ($(origin CC), default)
	CC = cc
endif
ifeq ($(origin FC), default)
   ifeq ($(shell uname -p),i386)
	FC  = gfortran
   else
	FC  = g77
   endif
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
SWIG_DLL = _$(MODULE)module.so
SWIG_PY  = $(MODULE).py
#LIBFILE = _$(MODULE)module.bundle
LIBFILE = $(PYPATH_DIR)/$(SWIG_DLL)

all: $(LIBFILE) $(PYPATH_DIR)/$(SWIG_PY) $(OBJS)


%.o: %.cpp
	$(CXX) $(ARCH_OPT) $(OPT_FLAGS) $(CXXFLAGS) $(INCL) $(INC_PY) -c $<

%.o: %.c
	$(CC) $(ARCH_OPT) $(OPT_FLAGS)  $(CFLAGS) $(INCL) $(INC_PY) -c $<

%.o: %.f
	$(FC) $(ARCH_OPT) $(OPT_FLAGS)  $(FFLAGS) -c $<  


############################################################
############################################################


SWIG?=${PRI}/_swig/swig


#10.3 location -I/Library/Frameworks/Python.framework/Headers
#INC_PY = \
#	 -I/System/Library/Frameworks/Python.framework/Headers \
#	 -I${PRCOMMON} \
#	 -I${PRCOMMON}/Include
INC_PY = \
	 -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
	 -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include \
     -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/numarray/include \
	 -I${PRCOMMON} \
	 -I${PRCOMMON}/Include

#2015 	 -I/Library/Frameworks/Python.framework/Versions/2.7/Headers

############################################################


$(LIBFILE): $(SWIG_OBJ) $(OBJS)
	$(CXX) $(ARCH_OPT) $(OPT_FLAGS) -bundle -flat_namespace -undefined suppress $^ -o $@ \
             $(EXTRA_LIBS)
#Erik /usr/local/lib/libg2c.a   
#	           $(EXTRA_LIBS) \  -L/sw/lib 



$(SWIG_OBJ):  $(SWIG_CXX)
	$(CXX)  $(ARCH_OPT) $(OPT_FLAGS) -c $< $(CXXFLAGS) $(INCL) $(INC_PY) # -fPIC 

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
	#makedepend -f Makefile_$(shell uname).dep --
	/usr/X11R6/bin/gccmakedep -f Makefile_$(shell uname).dep -- \
	        $(CFLAGS) $(INC_PY) -- \
		*.{i,cpp,h,cxx,c,f}
	@echo " *** Don't forget make dep  AFTER make $(MODULE)_wrap.cxx ***"
