FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install \
    g++-5-multilib-i686-linux-gnu \
    g++-5-multilib-powerpc64-linux-gnu \
    g++-5-multilib-s390x-linux-gnu \
    g++-5-multilib-sparc64-linux-gnu \
    g++-5-mips-linux-gnu \
    g++-5-mips64-linux-gnuabi64 \
    g++-5-m68k-linux-gnu \
    # These two do not exist here
    #g++-5-hppa-linux-gnu \
    #g++-5-riscv64-linux-gnu \
    g++-5-sh4-linux-gnu \
    g++-5-arm-linux-gnueabi \
    g++-5-alpha-linux-gnu

WORKDIR /opt
RUN apt-get -y install wget g++ libssl-dev make
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2.tar.gz
RUN tar -xf cmake-3.24.2.tar.gz
WORKDIR /opt/cmake-3.24.2
RUN ./bootstrap
RUN make
RUN make install