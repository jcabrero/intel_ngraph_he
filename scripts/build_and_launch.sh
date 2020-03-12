#!/bin/bash

echo "Restarting docker daemon [Because internet does not work]"
#sudo systemctl restart docker
echo
# list active docker containers
echo "Active docker containers..."
docker ps -a
echo
# clean up old docker containers
echo "Removing Exited docker containers..."
docker ps -a | grep Exited | cut -f 1 -d ' ' | xargs docker rm -f "${1}"
echo
#list docker images for he_transformer
echo "Docker images for he_transformer..."
docker images he_transformer
echo
# clean up docker images no longer in use
echo "Removing docker images for he_transformer..."
docker images -qa ubuntu:he_transformer* | xargs docker rmi -f "${1}"


echo "Building $1 dockerfile"
docker build -f ../docker/Dockerfile -t ubuntu:he_transformer .

echo "Launching and mounting jolly_gauss"
docker run --hostname=jolly_gauss --name=jolly_gauss -it --rm -v $(pwd):/root/mnt_dir ubuntu:he_transformer bash
