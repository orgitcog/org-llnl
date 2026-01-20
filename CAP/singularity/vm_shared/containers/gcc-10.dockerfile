FROM ubuntu:22.10

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install \
    g++-10-multilib-i686-linux-gnu \
    g++-10-multilib-powerpc64-linux-gnu \
    g++-10-multilib-s390x-linux-gnu \
    g++-10-multilib-sparc64-linux-gnu \
    g++-10-mips-linux-gnu \
    g++-10-mips64-linux-gnuabi64 \
    g++-10-m68k-linux-gnu \
    g++-10-hppa-linux-gnu \
    g++-10-riscv64-linux-gnu \
    g++-10-sh4-linux-gnu \
    g++-10-arm-linux-gnueabi \
    g++-10-alpha-linux-gnu

RUN apt-get -y install cmake make
