FROM ubuntu:20.04

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install \
    clang-6.0 \
    clang-7 \
    clang-8 \
    clang-9 \
    clang-10 

RUN apt-get -y install cmake make
