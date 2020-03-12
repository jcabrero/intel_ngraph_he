# Intel nGraph + HE Transformer project
This repository includes an easy way to install and build a docker and some examples of [Intel nGraph HE](https://github.com/IntelAI/he-transformer).


## Installing Docker in CentOS 7 (CERN)
The main source of the `docker` installation is [Docker Documentation](https://docs.docker.com/install/linux/docker-ce/centos/).
First we remove previous installations of docker in case they are already installed:
```
sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine
```
Then, we install some prerequisites:
```
sudo yum install -y yum-utils \
  device-mapper-persistent-data \
  lvm2
```
Add the repository for docker:
```
sudo yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo
```
Then install docker:
```
sudo yum install docker-ce docker-ce-cli containerd.io
```
If the previous command fails, it might be because `containerd.io` is not installed with the commands above. You can manually install `containerd.io` and then install `docker` from the [repository](https://centos.pkgs.org/7/docker-ce-stable-x86_64/containerd.io-1.2.13-3.1.el7.x86_64.rpm.html):
```
wget https://centos.pkgs.org/7/docker-ce-stable-x86_64/containerd.io-1.2.13-3.1.el7.x86_64.rpm.html
rpm -i containerd.io-1.2.13-3.1.el7.x86_64.rpm
sudo yum install docker-ce docker-ce-cli
```
Then, we enable the `docker` daemon, in case we restart, and start it:
```
sudo systemctl enable docker
sudo systemctl start docker
sudo systemctl status docker # You should check that the status is active.
```
Once it is installed, and for the following commands to work, we need to add our user to the `docker` group. This will ensure that most of the commands can be executed appropriately.
```
sudo usermod -aG docker <your_username>
```

## Getting started with Intel nGraph HE-Transformer
In order to install Intel nGraph HE-Transformer, we provide two installation media. The first, by means of pulling the image from Docker Hub and making it from scratch. The image is based in Ubuntu 18.04.

###  [Option A] Downloading from Docker Hub
Before executing this step, note that the size of the image is approximately 14 GB.
First, pull the image from Docker Hub:
```
docker pull jcabrero/intel_ngraph_he
```
Then run the image:
```
cd scripts
bash launch.sh
```
Note that the only thing that `launch.sh` script does is executing the following command:
```
docker run --hostname=intel_ngraph_he \
	--name=intel_ngraph_he -it --rm -v \
	$(pwd):/root/mnt_dir jcabrero/intel_ngraph_he bash
```

### [Option B] Installing it from the scripts
We provide a Dockerfile and some scripts in charge of creating the image in your computer. They will download all the necessary dependencies, but you will have to compile each and every of the elements.
Installing the docker:
```
cd scripts
bash build_and_launch.sh
```
Once the image is created and launched, we need to do the following:
```
git clone https://github.com/IntelAI/he-transformer.git
cd he-transformer
export HE_TRANSFORMER=$(pwd)
mkdir build
cd $HE_TRANSFORMER/build
cmake .. -DCMAKE_CXX_COMPILER=clang++-9
```
Note that this last command will take a considerable amount of time:
```
make -j install
```
___
Author: @jcabrero

