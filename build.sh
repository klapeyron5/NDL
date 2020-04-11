#!/bin/bash
docker stop ndl_c
docker rm ndl_c
docker rmi ndl_i:latest
docker build -t ndl_i .
