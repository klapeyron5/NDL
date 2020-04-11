ARG CUDA_VERSION=10.1
ARG CUDNN_VERSION=7

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu18.04

WORKDIR /home/workdir

COPY predict.py /home/workdir
COPY exported_models /home/workdir/exported_models/

RUN apt-get update &&\
  apt-get install -y python3-pip python3-dev &&\
  cd /usr/local/bin &&\
  pip3 install --upgrade pip

RUN pip3 install numpy &&\
    pip3 install tensorflow==2.1 &&\
    mkdir /home/data &&\
    mkdir /home/predictions &&\
    chmod -R 777 /home

CMD [ "python3", "/home/workdir/predict.py"]