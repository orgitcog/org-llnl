# vim:noet
INCLUDES=-Iinclude -iprefix ./ -iwithprefix include/thirdparty
LIBDIRS=-Llib -Llib/thirdparty
BOOST_DEBUG_BUILD=mgw102-mt-d-x32-1_66
BOOST_RELEASE_BUILD=mgw102-mt-x32-1_66
DEBUG_LIBS=${LIBDIRS} $(patsubst %, -lboost_%-${BOOST_DEBUG_BUILD}, log_setup log system thread chrono program_options filesystem) -lwsock32 -lws2_32
# RUNTIME_DEBUG_LIBS includes library versions compiled against the debugging
#	version of the standard and runtime support libraries as by the following
#	build invocation for Boost.Program_Options:
#
# ./b2 -a --with-program_options runtime-debugging=on address-model=32 
#		link=static threading=multi variant=debug debug stage 
#		cxxflags="-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC"
#
# While this tag is not added by Boost.Build by default for gcc, since the 
#	GLIBCXX flags above need to be provided, a *-gd-* tag for the library marks
#	the version with the added runtime checks, which can be much slower than the
#	standard debug instrumentation; thus, both library options can be selected
#	here
RUNTIME_DEBUG_LIBS=$(subst -mt-d-,-mt-gd-, $(DEBUG_LIBS))
DEBUGFLAGS=-g -Og -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -DENABLE_TEST_INJECT -DPERMISSIVE_CHECKSUM_HANDLING
RELEASELIBS=${LIBDIRS} $(patsubst %, -lboost_%-${BOOST_RELEASE_BUILD}, log_setup system thread chrono program_options filesystem) -lwsock32 -lws2_32
RELEASEFLAGS=-O2
WARNINGFLAGS=-Wall -Wextra -Wpedantic
CXXFLAGS_COMMON=${WARNINGFLAGS} --std=c++14 -DEIGEN_MPL2_ONLY ${INCLUDES}
CXXFLAGS_DEBUG=${DEBUGFLAGS} ${CXXFLAGS_COMMON}
CXXFLAGS_RELEASE=${RELEASEFLAGS} ${CXXFLAGS_COMMON}
CXXFLAGS=$(CXXFLAGS_DEBUG)
DEPFILE=tmonitor.dep
DEPFILE_TMP=$(addsuffix .tmp, $(DEPFILE))
CXXFLAGS_DEP=-MM -MF $(DEPFILE_TMP) $(INCLUDES)
TEST_LIBS_OBJECTS = src/test_libs.o src/parse_ubx.o
MAIN_OBJECTS = $(addprefix src/, main.o alerter.o alert_msg.o clock.o clock_desc.o clock_replay.o detector.o det_alg.o det_msg.o gm_model.o kf.o task.o task_container.o parse_ubx.o prog_state.o serial_dev.o spmc.o stream_clock.o tcp_alert_handler.o tdc.o time_msg.o ubx_base.o ubx_clock.o ubx_file_reader.o ubx_serial.o ws_server.o calib.o utility.o)
TEST_LIBS_RESULT = bin/test_libs_result.exe
UNIT_TEST_RESULT = bin/unit_test.exe
MAIN_RESULT = bin/tmonitor.exe
CXX = g++

VPATH = src:include

.PHONY: all clean dep tmonitor test_libs unit_test
.DEFAULT_GOAL := tmonitor
all: dep tmonitor test_libs
clean:
	rm -f $(MAIN_RESULT) $(MAIN_OBJECTS)
dep:
	@set -e; rm -f $(DEPFILE)
	$(foreach cxx_file, $(MAIN_OBJECTS:.o=.cxx), \
	  $(CXX) $(CXXFLAGS_DEP) $(cxx_file) && sed -e 's#^\(.*\.o\)#src/\1#; s#include/thirdparty\S*##g' $(DEPFILE_TMP) | grep -v '^\s*\\\s*$$' >> $(DEPFILE) && echo -e '\t$$(COMPILE.cc) $$< -o $$@' >> $(DEPFILE);)
	rm -f $(DEPFILE_TMP)
tmonitor: $(MAIN_RESULT)
test_libs: $(TEST_LIBS_RESULT)
# To build a unit test, make the unit_test target, also supplying a UNIT
#	variable (translation unit name); e.g., UNIT=parse_ubx
unit_test: $(UNIT_TEST_RESULT)

include $(DEPFILE)

${TEST_LIBS_RESULT}: ${TEST_LIBS_OBJECTS}
	$(CXX) ${CXXFLAGS_DEBUG} $^ ${DEBUGLIBS} -o $@

${UNIT_TEST_RESULT}: $(filter-out src/main.o,\
			$(subst $(UNIT).o,$(UNIT).cxx,$(MAIN_OBJECTS)))
	$(if $(filter undefined,$(origin UNIT)), \
		$(error No unit specified for test; use UNIT=<unit-name>))
	$(if $(findstring $(UNIT).o,$(MAIN_OBJECTS)),,\
		$(error No such unit as "$(UNIT)"))
	$(CXX) ${CXXFLAGS_DEBUG} -DUNIT_TEST $^ ${RUNTIME_DEBUG_LIBS} -o $@

${MAIN_RESULT}: $(MAIN_OBJECTS)
	$(CXX) ${CXXFLAGS_DEBUG} $^ ${RUNTIME_DEBUG_LIBS} -o $@

