#!/usr/bin/env python

import subprocess
import os

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make -f Makefile_0 clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def run_command(cmd):
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()
    return cmdOutput

def test_0():
    # --- compile code ---
    cmd = ["make -f Makefile_0"]
    output = run_command(cmd).decode("utf-8")

    flag_1 = False
    flag_2 = False
    for line in output.splitlines():
        if "#FPCHECKER: Instrumented 0 @ main.cpp" in line:
            flag_1 = True
        if "#FPCHECKER: Instrumented 0 @ compute.cpp" in line:
            flag_2 = True
    assert flag_1
    assert flag_2

def test_1():
    # --- compile code ---
    cmd = ["make -f Makefile_1"]
    output = run_command(cmd).decode("utf-8")

    flag_1 = False
    flag_2 = False
    for line in output.splitlines():
        if "#FPCHECKER: Instrumented 0 @ main.cpp" in line:
            flag_1 = True
        if "#FPCHECKER: Instrumented 1 @ compute.cpp" in line:
            flag_2 = True
    assert flag_1
    assert flag_2

def test_2():
    # --- compile code ---
    cmd = ["make -f Makefile_2"]
    output = run_command(cmd).decode("utf-8")

    flag_1 = False
    flag_2 = False
    for line in output.splitlines():
        if "#FPCHECKER: Instrumented 0 @ main.cpp" in line:
            flag_1 = True
        if "#FPCHECKER: Instrumented 1 @ compute.cpp" in line:
            flag_2 = True
    assert flag_1
    assert flag_2

def test_3():
    # --- compile code ---
    cmd = ["make -f Makefile_3"]
    output = run_command(cmd).decode("utf-8")

    flag_1 = False
    flag_2 = False
    for line in output.splitlines():
        if "#FPCHECKER: Instrumented 0 @ main.cpp" in line:
            flag_1 = True
        if "#FPCHECKER: Instrumented 2 @ compute.cpp" in line:
            flag_2 = True
    assert flag_1
    assert flag_2

def test_4():
    # --- compile code ---
    cmd = ["make -f Makefile_4"]
    output = run_command(cmd).decode("utf-8")

    flag_1 = False
    flag_2 = False
    for line in output.splitlines():
        if "#FPCHECKER: Instrumented 0 @ main.cpp" in line:
            flag_1 = True
        if "#FPCHECKER: Instrumented 1 @ compute.cpp" in line:
            flag_2 = True
    assert flag_1
    assert flag_2