FROM python:3.8

RUN apt-get -y update
RUN apt-get -y install build-essential git git-lfs python3 python3-venv graphviz cmake libhdf5-dev

RUN mkdir slewpy

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
