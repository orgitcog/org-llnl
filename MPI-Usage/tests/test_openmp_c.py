#!/usr/bin/env python

import test_config
import subprocess
import json
import sys

def main():
    t = "Testing OpenMP in C:"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    inputFile = "../code_examples/OpenMP/parallel.c"
    cmd = ["../mpiusage.py", inputFile]
    cmdOutput = subprocess.check_output(cmd)
    jsonOut = json.loads(cmdOutput)
    
    result = test_config.failed
    if jsonOut['OPENMP'] == 1:
        result = test_config.passed

    sys.stdout.write(result+"\n")
    
main()
