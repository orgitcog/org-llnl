FROM ubuntu:14.04

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get install -y openjdk-6-jdk

RUN echo __TT_COMPILER_VERSION__ $(java -version) __TT_COMPILER_VERSION__
