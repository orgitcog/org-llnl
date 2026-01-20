#!/bin/bash

set -e

git remote set-url origin git@github.com:LLNL/serac_tests.git
git push
git remote set-url origin https://github.com/LLNL/serac_tests.git
