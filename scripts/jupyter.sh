#!/bin/bash

echo "Launching jupyter notebook on port 10093"
cd /root/
jupyter notebook --port=10093 --no-browser --ip=0.0.0.0 --allow-root
