Directions to convert singularity to docker:

# Dump data/env variables
# The first one will print all the sif environment variables and build script, the second will copy all of the data
#   out of the singularity image and put it into a directory './image_data'
singularity sif dump 2 rose-binary-analysis.sif | sed 's/\\n/\n/g' > sif_info.txt
singularity sif dump 4 rose-binary-analysis.sif > data.squash

# Extract all the squashfs data
unsquashfs -dest data data.squash

# Build the docker container
docker build -t rose-binary-analysis .

# This is what should be in the dockerfile minimum:
FROM scratch
COPY data /
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Extract docker image to tarfile to transfer
docker save -o rose-binary-analysis.tar rose-binary-analysis

# Load the image on a new machine
docker load -i rose-binary-analysis.tar
