#!/usr/bin/env python

import subprocess
import os
import pytest
from dynamic import report

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make -f Makefile_0 clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

@pytest.fixture(scope="function", autouse=True)
def run_after_each_test():
    """A fixture that runs after each test function."""
    print("\nRunning test...")
    yield
    # Runs after each test function
    cmd = ["make -f Makefile_0 clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def run_command(cmd):
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()
    return cmdOutput

def foundNaN(line_number):
    found = False
    fileName = report.findReportFile('.fpc_logs')
    data = report.loadReport(fileName)
    for i in range(len(data)):
      print('i', i, data[i])
      if data[i]['file'].endswith('compute.cpp'):
        if data[i]['nan'] > 0:
          if data[i]['line'] == line_number:
            found = True
            break

    return found

def test_0():
    # --- compile code ---
    cmd = ["make -f Makefile_0"]
    output = run_command(cmd).decode("utf-8")

    # --- run code ---
    cmd = ["./main"]
    output = run_command(cmd).decode("utf-8")
    line_number = 64
    assert not foundNaN(line_number)

def test_1():
    # --- compile code ---
    cmd = ["make -f Makefile_1"]
    output = run_command(cmd).decode("utf-8")

    # --- run code ---
    cmd = ["./main"]
    output = run_command(cmd).decode("utf-8")
    line_number = 64
    assert not foundNaN(line_number)

def test_2():
    # --- compile code ---
    cmd = ["make -f Makefile_2"]
    output = run_command(cmd).decode("utf-8")

    # --- run code ---
    cmd = ["./main"]
    output = run_command(cmd).decode("utf-8")
    line_number = 64
    assert not foundNaN(line_number)

def test_3():
    # --- compile code ---
    cmd = ["make -f Makefile_3"]
    output = run_command(cmd).decode("utf-8")

    # --- run code ---
    cmd = ["./main"]
    output = run_command(cmd).decode("utf-8")
    line_number = 64
    assert foundNaN(line_number)

def test_4():
    # --- compile code ---
    cmd = ["make -f Makefile_4"]
    output = run_command(cmd).decode("utf-8")

    # --- run code ---
    cmd = ["./main"]
    output = run_command(cmd).decode("utf-8")
    line_number = 64
    assert not foundNaN(line_number)

def test_5():
    # --- compile code ---
    cmd = ["make -f Makefile_5"]
    output = run_command(cmd).decode("utf-8")

    # --- run code ---
    cmd = ["./main"]
    output = run_command(cmd).decode("utf-8")
    line_number = 64
    assert foundNaN(line_number)
