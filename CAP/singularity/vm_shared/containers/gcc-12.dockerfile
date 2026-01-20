FROM ubuntu:22.10

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install \
    g++-12-multilib-i686-linux-gnu \
    g++-12-multilib-powerpc64-linux-gnu \
    g++-12-multilib-s390x-linux-gnu \
    g++-12-multilib-sparc64-linux-gnu \
    # mips doesn't exist for g++-12 yet on apt
    #g++-12-multilib-mips-linux-gnu \
    #g++-12-multilib-mips64-linux-gnuabi64 \
    g++-12-m68k-linux-gnu \
    g++-12-hppa-linux-gnu \
    g++-12-riscv64-linux-gnu \
    g++-12-sh4-linux-gnu \
    g++-12-arm-linux-gnueabi \
    g++-12-alpha-linux-gnu

RUN apt-get -y install cmake make strace
