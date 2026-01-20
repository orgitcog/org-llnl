FROM ubuntu:22.10

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get install -y openjdk-20-jdk

RUN echo __TT_COMPILER_VERSION__ $(java -version) __TT_COMPILER_VERSION__