FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install \
    g++-6-multilib-i686-linux-gnu \
    g++-6-multilib-powerpc64-linux-gnu \
    g++-6-multilib-s390x-linux-gnu \
    g++-6-multilib-sparc64-linux-gnu \
    g++-6-mips-linux-gnu \
    g++-6-mips64-linux-gnuabi64 \
    g++-6-m68k-linux-gnu \
    g++-6-hppa-linux-gnu \
    # No exist
    #g++-6-riscv64-linux-gnu \
    g++-6-sh4-linux-gnu \
    g++-6-arm-linux-gnueabi \
    g++-6-alpha-linux-gnu

WORKDIR /opt
RUN apt-get -y install wget g++ libssl-dev make
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2.tar.gz
RUN tar -xf cmake-3.24.2.tar.gz
WORKDIR /opt/cmake-3.24.2
RUN ./bootstrap
RUN make
RUN make install
