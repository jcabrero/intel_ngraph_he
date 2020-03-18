FROM ubuntu:18.04
# Base image is Ubuntu
RUN apt-get update -y

RUN apt-get install -y build-essential software-properties-common bash-completion man  manpages-posix 
# Installing man pages
# Installing python
RUN apt-get install -y python3 python3-pip virtualenv python3-numpy python3-dev python3-wheel
# Install vim, optional, but just if needed
RUN apt-get install -y vim git wget unzip sudo
# Install several needed tools
RUN apt-get install -y patch diffutils libtinfo-dev autoconf graphviz yapf3 python3-yapf

# Install protobuf for serialization of the models
RUN apt install -y protobuf-compiler libprotobuf-dev


# Install clang-9
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
RUN apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main"
RUN apt-get update && apt install -y clang-9 clang-tidy-9 clang-format-9


# Install CMAKE
RUN mkdir -p /root/deps/cmake
RUN cd /root/deps
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.1/cmake-3.16.1-Linux-x86_64.sh 
RUN bash cmake-3.16.1-Linux-x86_64.sh --skip-license --prefix=/root/deps/cmake
RUN ln -s /root/deps/cmake/bin/cmake /bin/cmake
RUN ln -s /root/deps/cmake/bin/ccmake /bin/ccmake

# Getting bazel and instaling it.
RUN mkdir -p /root/deps/bazel
RUN cd /root/deps/bazel
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-installer-linux-x86_64.sh
RUN bash bazel-0.25.2-installer-linux-x86_64.sh --user
RUN echo "export PATH=$PATH:~/bin" >> /root/.bashrc

# Getting doxygen and graphviz
RUN apt-get install -y doxygen graphviz

# Getting net-tools and ping
RUN apt-get install -y net-tools iputils-ping dnsutils

# Run Jupyter in case, we want to develop from scratch.
RUN pip3 install jupyter
# Colored terminal
# RUN useradd -s /bin/bash -d /home/jocabrer -m -G sudo jocabrer
RUN echo "PS1='\[\033[01;31m\](docker) \[\e[01;36m\]\u\[\e[01;37m\]@\[\e[01;33m\]\H\[\e[01;37m\]:\[\e[01;32m\]\w\[\e[01;37m\]\$\[\033[0;37m\] '" >> /root/.bashrc
RUN echo "alias python=python3" >> /root/.bashrc
RUN echo "alias pip=pip3" >> /root/.bashrc
RUN echo "source /root/he-transformer/external/venv-tf-py3/bin/activate" >> /root/.bashrc
RUN echo "export HE_TRANSFORMER='/root/he-transformer'" >> /root/.bashrc
WORKDIR /root
CMD ['bash']
