#!/bin/bash

# Used like:
# ./getlibinfo.sh image_name lib_path
# Where:
#   image_name - name of singularity image to get lib info from. Should be full name within /vm_shared/images folder. EG: "gcc-12.sif"
#   lib_path - path to library folder within the image to get

# Paths:
# gcc-12.sif - all are within /usr, folders include: "i686-linux-gnu"

# Make sure the user passed the correct number of arguments
if [ $# -ne 2 ]; then
    printf "Error: invalid number of arguments. Expected: 2, got: %d\n" $#
fi

# Enter vagrant and execute all the commands needed
#vagrant up
echo "Copying library information from singularity container: $1"
vagrant ssh -c "singularity exec --bind /vm_shared:/vm_shared /vm_shared/images/$1 \
    strace cp -r -f $2 /vm_shared/libinfo/ 2> /vm_shared/err.log\
"

# Check if there was a non-zero exit status when copying the files
last_exit=$?
if [ $last_exit -ne 0 ]; then
    printf "Error: copying library information failed with exit code: %d\n" $last_exit
    exit 1
fi

echo "Library info copied!"