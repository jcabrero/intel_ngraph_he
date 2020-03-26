#!/bin/bash
if [ "$#" -ne 1 ]; then
	echo "Illegal number of parameters"
	exit 2
fi
echo "Launching and mounting $1"
docker run --hostname=$1 --name=$1 -it --rm -v $(pwd)/../:/root/mnt_dir jcabrero/intel_ngraph_he:latest bash
