FROM ubuntu:20.04

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install \
    clang-3.5 \
    clang-3.6 \
    clang-3.7 \
    clang-3.8 \
    clang-3.9 \
    clang-4.0 \
    clang-5.0

RUN apt-get -y install cmake make
