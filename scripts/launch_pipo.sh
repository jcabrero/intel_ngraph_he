#!/bin/bash

echo "Launching and mounting pipo"
docker run --hostname=pipo --name=pipo -it --rm --net=host -v $(pwd)/../:/root/mnt_dir jcabrero/intel_ngraph_he:latest bash
