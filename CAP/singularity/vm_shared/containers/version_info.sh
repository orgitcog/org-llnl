#!/bin/bash

# Finds all of the gcc/clang versions and appends them to /mounted/compilers/txt
#for loc in /bin /usr/bin; do
#    ls $loc | grep -e '.*g[+][+]-[0-9]*$' -e '.*gcc-[0-9]*$' -e 'clang[+]*-[0-9]*' >> /mounted/compilers.txt;
#done

for loc in /bin /usr/bin; do
    for app in $(ls $loc | grep -e '.*g[+][+]-[0-9]\+$' -e '.*gcc-[0-9]\+$' -e 'clang[+]*-[0-9]\+$'); do
        version_str=$($app --version | head -n 1)
        version_array=($version_str)
        echo "$app ${version_array[3]}"
    done
done