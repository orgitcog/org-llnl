FROM ubuntu:16.04

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install \
    g++-4.9-multilib-arm-linux-gnueabi \
    g++-4.9-multilib-powerpc-linux-gnu \
    g++-4.9-multilib-s390x-linux-gnu \
    g++-4.8-multilib-arm-linux-gnueabi \
    g++-4.8-multilib-powerpc-linux-gnu \
    g++-4.8-multilib-s390x-linux-gnu \
    g++-4.7-multilib-arm-linux-gnueabi \
    g++-4.7-multilib-powerpc-linux-gnu \
    g++-4.7-multilib-s390x-linux-gnu

WORKDIR /opt
RUN apt-get -y install wget g++ libssl-dev make
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2.tar.gz
RUN tar -xf cmake-3.24.2.tar.gz
WORKDIR /opt/cmake-3.24.2
RUN ./bootstrap
RUN make
RUN make install
