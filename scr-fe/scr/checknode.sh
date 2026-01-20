#!/bin/bash

################################################
# Generic checknode script
# it should be executed via pdsh
# this script should NOT be called by a user
################################################

# ARGUMENTS
# this script takes an arbitrary number of paths as arguments
# it checks that each directory is accessible

# TODO
# implement --free option to test amount of free space

TEST_FILE='testfile.txt'

for i in "$@"
do

    # check that we can access the directory
    `ls -lt $i &>/dev/null`
    if [[ $? != 0 ]]; then
        echo "FAIL: Could not access directory: $i"
        exit 1;
    fi

    # attempt to write to directory
    `touch $i/$TEST_FILE`
    if [[ $? != 0 ]]; then
        echo "FAIL: Could not touch test file: $i/$TEST_FILE"
        exit 1;
    fi

    # attempt to remove the test file
    `rm -f $i/$TEST_FILE`
    if [[ $? != 0 ]]; then
        echo "FAIL: Could not rm test file: $i/$TEST_FILE"
        exit 1;
    fi

done

echo "PASS"
exit 0
