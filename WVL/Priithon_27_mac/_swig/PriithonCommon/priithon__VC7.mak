# -*- Makefile -*-
#__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
#__license__ = "BSD license - see LICENSE file"

sebobj = $(objs:.o=.obj)
sebsrc = $(sebobj:.obj=.cpp)

SWIGTOP=c:\SWIG-1.3.24
SWIG=$(SWIGTOP)\swig

SWIGOPT=-c++ -python -shadow -globals v -I$(PRCOMMON) -I$(PRCOMMON)/Include


PYTHON_INCLUDE = /Ic:\python24\include

#INCL might be defined in "parent makefile"
INCL2=/I$(PRCOMMON) /I$(PRCOMMON)/Include $(PYTHON_INCLUDE) $(INCL)


#                           C/C++ COMPILER OPTIONS

#                               -OPTIMIZATION-

# /O1 minimize space                       /Op[-] improve floating-pt consistency
# /O2 maximize speed                       /Os favor code space
# /Oa assume no aliasing                   /Ot favor code speed
# /Ob<n> inline expansion (default n=0)    /Ow assume cross-function aliasing
# /Od disable optimizations (default)      /Ox maximum opts. (/Ogityb2 /Gs)
# /Og enable global optimization           /Oy[-] enable frame pointer omission
# /Oi enable intrinsic functions

#                              -CODE GENERATION-

# /G3 optimize for 80386                   /Gh enable _penter function call
# /G4 optimize for 80486                   /GH enable _pexit function call
# /G5 optimize for Pentium                 /GR[-] enable C++ RTTI
# /G6 optimize for PPro, P-II, P-III       /GX[-] enable C++ EH (same as /EHsc)
# /G7 optimize for Pentium 4 or Athlon     /EHs enable C++ EH (no SEH exceptions)
# /GB optimize for blended model (default) /EHa enable C++ EH (w/ SEH exceptions)
# /Gd __cdecl calling convention           /EHc extern "C" defaults to nothrow
# /Gr __fastcall calling convention        /GT generate fiber-safe TLS accesses
# /Gz __stdcall calling convention         /Gm[-] enable minimal rebuild
# /GA optimize for Windows Application     /GL[-] enable link-time code generation

# (press <return> to continue)
# /Gf enable string pooling                /QIfdiv[-] enable Pentium FDIV fix
# /GF enable read-only string pooling      /QI0f[-] enable Pentium 0x0f fix
# /Gy separate functions for linker        /QIfist[-] use FIST instead of ftol()
# /GZ Enable stack checks (/RTCs)          /RTC1 Enable fast checks (/RTCsu)
# /Ge force stack checking for all funcs   /RTCc Convert to smaller type checks
# /Gs[num] control stack checking calls    /RTCs Stack Frame runtime checking
# /GS enable security checks               /RTCu Uninitialized local usage checks
# /clr[:noAssembly] compile for the common language runtime
#     noAssembly - do not produce an assembly
# /arch:<SSE|SSE2> minimum CPU architecture requirements, one of:
#     SSE - enable use of instructions available with SSE enabled CPUs
#     SSE2 - enable use of instructions available with SSE2 enabled CPUs


#C:\Program Files\Microsoft Visual C++ Toolkit 2003\include\ostream(574) : warnin
#g C4530: C++ exception handler used, but unwind semantics are not enabled. Speci
#fy /EHsc
SEB_COMMON_CFLAGS=/EHsc /DWIN32 $(OPT_FLAGS)


CXXFLAGS=$(SEB_COMMON_CFLAGS) $(INCL2)
CPPFLAGS=$(SEB_COMMON_CFLAGS) $(INCL2)


LDSHARED = link /dll
LIBS=/LIBPATH:c:\python24\libs $(EXTRA_LIBS) kernel32.lib user32.lib python24.lib /nologo


INTERFACE=$(MODULE).i
ISRCS = $(INTERFACE:.i=_wrap.cxx)
IOBJS = $(INTERFACE:.i=_wrap.obj)

TARGET = _$(MODULE).dll


all: $(TARGET)

$(ISRCS): $(INTERFACE) 
	$(SWIG) $(SWIGOPT) $**

$(TARGET): $(IOBJS) $(sebobj)
        $(LDSHARED) $(LIBS) $** /out:$(TARGET)

# X:\Pr\glSeb>copy _glSeb.dll glSeb.py \PrWin24\Priithon
# The syntax of the command is incorrect.

# X:\Pr\glSeb>copy glSeb.py \PrWin24\Priithon
#         1 file(s) copied.

# X:\Pr\glSeb>copy _glSeb.dll \PrWin24\Priithon
#         1 file(s) copied.


cleanall:
        del *.cxx *.dll *.pyc *.exp *.lib *.obj

#clean all except swig wrapper
clean:
        del *.dll *.pyc *.exp *.lib *.obj
