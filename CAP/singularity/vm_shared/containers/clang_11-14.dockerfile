FROM ubuntu:22.04

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install \
    clang-11 \
    clang-12 \
    clang-13 \
    clang-14 

RUN apt-get -y install cmake make
