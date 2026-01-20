FROM ubuntu:22.10

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install \
    g++-11-multilib-i686-linux-gnu \
    g++-11-multilib-powerpc64-linux-gnu \
    g++-11-multilib-s390x-linux-gnu \
    g++-11-multilib-sparc64-linux-gnu \
    # mips doesn't exist for g++-11 yet on apt
    #g++-11-multilib-mips-linux-gnu \
    #g++-11-multilib-mips64-linux-gnuabi64 \
    g++-11-m68k-linux-gnu \
    g++-11-hppa-linux-gnu \
    g++-11-riscv64-linux-gnu \
    g++-11-sh4-linux-gnu \
    g++-11-arm-linux-gnueabi \
    g++-11-alpha-linux-gnu

RUN apt-get -y install cmake make
