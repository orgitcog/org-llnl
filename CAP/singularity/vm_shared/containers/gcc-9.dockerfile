FROM ubuntu:22.10

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install \
    g++-9-multilib-i686-linux-gnu \
    g++-9-multilib-powerpc64-linux-gnu \
    g++-9-multilib-s390x-linux-gnu \
    g++-9-multilib-sparc64-linux-gnu \
    g++-9-mips-linux-gnu \
    g++-9-mips64-linux-gnuabi64 \
    g++-9-m68k-linux-gnu \
    g++-9-hppa-linux-gnu \
    g++-9-riscv64-linux-gnu \
    g++-9-sh4-linux-gnu \
    g++-9-arm-linux-gnueabi \
    g++-9-alpha-linux-gnu

RUN apt-get -y install cmake make
