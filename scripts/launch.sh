#!/bin/bash

echo "Restarting docker daemon [Because internet does not work]"
sudo systemctl restart docker
echo


echo "Launching and mounting jolly_gauss"
docker run -p 10093:10093 --hostname=jolly_gauss --name=jolly_gauss -it --rm -v $(pwd)/../:/root/mnt_dir ubuntu:he_transformer bash
