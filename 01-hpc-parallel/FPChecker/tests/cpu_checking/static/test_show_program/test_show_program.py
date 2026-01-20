#!/usr/bin/env python

import subprocess
import os
import sys

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def test_1():
    # --- compile code ---
    cmd = ["fpchecker-show"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    lines = cmdOutput.decode("utf-8").splitlines()
    pattern = "CFLAGS"
    flags = None
    for i in range(1, len(lines)):
        if pattern in lines[i - 1]:
            #print(">>>", lines[i])
            flags = lines[i]
    
    values = flags.split()
    header_file = values[2]

    header_exists = False
    # Check if the header file exists
    if os.path.isfile(header_file):
        header_exists = True

    library_file = values[3].split('=')[1]
    library_exists = False
    # Check if the header file exists
    if os.path.isfile(library_file):
        library_exists = True

    assert header_exists
    assert library_exists

