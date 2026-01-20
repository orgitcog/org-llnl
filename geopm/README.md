Port of Global Extensible Open Power Manager (GEOPM) to IBM Power
=================================================================

INTRODUCTION
------------

This repository contains the code that ports GEOPM to IBM Power
microarchitecture implementation. We try to be in sync with [HEAD of GEOPM
dev branch](https://github.com/geopm/geopm).

BUILDING
--------

In order to build this code on IBM Power in addition to requirements
for GEOPM that you can find [here](README) (please note that for IBM
Power you do not need MSR driver) you would need to set for
`perf_event_paranoid` in `/proc/sys/kernel` to -1 because we are using
[`perf_event_open`](http://man7.org/linux/man-pages/man2/perf_event_open.2.html)
to measure activities of all processes and threads on CPUs. Also, you
would need to have NVIDIA Management Library (NVML) installed on your
system as we are using it to measure power consumption of GPUs that
are available on the system.

To build GEOPM on our local IBM Power 8 (Minsky) system we have used
the following `configure` after doing `./autogen.sh` as per [original
instructions](README):

    LDFLAGS="-L/usr/lib64/nvidia -lnvidia-ml" \
    CXXFLAGS="-I/usr/local/cuda/targets/ppc64le-linux/include" \
    ./configure

In `LDFLAGS` we have passed path where `libnvidia-ml.so` library and
in `CXXFLAGS` we have passed path where header files for the library
are. In addition to this you might also want to specify path where to
install binaries and libraries once GEOPM is compiled (`--prefix`
flag).

POWER CONSUMPTION MEASUREMENT
-----------------------------

As explained above to measure power of GPUS we use NVML library. For
measuring power consumption of IBM Power microprocessor we use sensors
that are exposed by [On Chip
Controller](http://hpm.ornl.gov/documents/HPM2015_Rosedahl.pdf). Here,
we assume that sensors are exposed via Linux kernel module that
creates a bunch of files in `/sys/devices/system/cpu/occ_sensors` that
we parse in order to report power consumption.