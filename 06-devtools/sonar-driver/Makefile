PYTHON=/bin/python3.4

PYFILES := $(shell find . -name resource\*.py -print)

all: sonar-driver

venv: setup.py $(PYFILES)
	${PYTHON} -m venv venv/
	source ./venv/bin/activate \
		&& pip install --upgrade pip

sonar-driver: venv
	source ./venv/bin/activate \
		&& pip install --upgrade -e `pwd`
