#!/bin/bash

#echo "Restarting docker daemon [Because internet does not work]"
#sudo systemctl restart docker
#echo


echo "Launching and mounting intel_ngraph_he container"
docker run -p 10093:10093 --hostname=intel_ngraph_he --name=intel_ngraph_he -it --rm -v $(pwd)/../:/root/mnt_dir jcabrero/intel_ngraph_he:latest bash
