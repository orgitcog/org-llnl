#!/bin/bash

sudo apt-get install libopenblas-dev
python3 -m venv aihwkit-venv
source aihwkit-venv/bin/activate
pip install aihwkit
python3 minimal_example.py
deactivate
