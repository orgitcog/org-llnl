#!/bin/bash

VERSION=`python -m setuptools_scm`
DOCKERFILE="Dockerfile"

GITLAB_TOKEN_FILENAME="secrets/token.txt"
GITLAB_TOKEN=$(head -n 1 $GITLAB_TOKEN_FILENAME)
export GITLAB_TOKEN

COMMIT_HASH=$(git log -1 --pretty=format:"%H")

echo "Building version $VERSION ..."
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
docker build --rm -f $DOCKERFILE \
    --progress=plain \
    --build-arg BUILD_DATE=$BUILD_DATE \
    --build-arg COMMIT_HASH=$COMMIT_HASH \
    --build-arg VERSION=$VERSION \
    --build-arg username=__token__ \
    --secret id=build_token,env=GITLAB_TOKEN \
    -t llnl/apbuilder:$VERSION .
    

rm -rf build_token

echo "Tagging $VERSION as 'latest'"
docker tag llnl/apbuilder:$VERSION llnl/apbuilder:latest

echo "Version $VERSION build completed"
