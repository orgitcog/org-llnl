# -*- Makefile -*-
#__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
#__license__ = "BSD license - see LICENSE file"

PYVER ?= 2.5
PYVER_NODOT =$(subst .,,$(PYVER))

ifeq ($(shell uname -m),x86_64)
   #2009-07-14 PYPATH_DIR ?= $(HOME)/Priithon_$(PYVER_NODOT)_lin64/Priithon
   ifeq ($(origin CXX), default)
      CXX = g++ -fPIC
      FC = g77 -fPIC
   endif
endif

#2009-07-14 PYPATH_DIR ?= $(HOME)/Priithon_$(PYVER_NODOT)_lin/Priithon
PYPATH_DIR?=.

# test:
# 	echo "$(shell uname -m)" $(CXX) $(origin CXX)

# WXBASEDIR=/jws30/haase/wx2_cvs/wxWindows
# WXBUILDDIR=$(WXBASEDIR)/sebsWxGtk_build


##CXXFLAGS?=-Wno-deprecated
#OPT_FLAGS = -O4
#EXTRA_LIBS=
#INCL=
#MODULE=
#OBJS= seb1.o #$(MODULE).o


#PYPATH_DIR = $(HOME)/Pr/Linux

SWIG_MODULE=$(MODULE)

ifeq ($(SWIG_INTERFACE),)
   SWIG_INTERFACE=$(SWIG_MODULE).i
endif

SWIG_CXX = $(SWIG_INTERFACE:.i=_wrap.cxx)
SWIG_OBJ = $(SWIG_INTERFACE:.i=_wrap.o)
SWIG_DLL = _$(MODULE)module.so
SWIG_PY  = $(MODULE).py
LIBFILE = $(PYPATH_DIR)/$(SWIG_DLL)

all: $(LIBFILE) $(PYPATH_DIR)/$(SWIG_PY) $(OBJS)


%.o: %.cpp
	$(CXX) $(OPT_FLAGS) $(CXXFLAGS) $(INCL) $(INC_PY) -c $<

%.o: %.c
	$(CC) $(OPT_FLAGS)  $(CFLAGS) $(INCL) $(INC_PY) -c $<

%.o: %.f
	$(FC) $(OPT_FLAGS)  $(FFLAGS) -c $<  


############################################################
############################################################

SWIG?=${PRI}/_swig/swig

INC_PY = -I/usr/include/python$(PYVER) \
         -I${PRCOMMON}/Include/python$(PYVER) \
         -I${PRCOMMON} \
	 -I${PRCOMMON}/Include


############################################################


$(LIBFILE): $(SWIG_OBJ) $(OBJS)
	$(CXX) $(OPT_FLAGS) -shared $^ -o $@ $(EXTRA_LIBS)


$(SWIG_OBJ):  $(SWIG_CXX)
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
	makedepend -f Makefile_$(shell uname).dep -- \
                $(CFLAGS) $(INC_PY) -- \
		*.{i,cpp,h,cxx,c,f}
#                $(CFLAGS) -I/usr/include/g++-3 $(INC_PY) --
	@echo " *** Don't forget make dep  AFTER make $(MODULE)_wrap.cxx ***"
