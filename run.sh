#!/bin/bash
docker stop ndl_с
docker rm ndl_с

readonly DATA_PATH=/home/klapeyron/NDL_test/data/
readonly PREDICTIONS_PATH=/home/klapeyron/NDL_test/predictions/

docker run \
-v $DATA_PATH:/home/data \
-v $PREDICTIONS_PATH:/home/predictions \
--name ndl_c ndl_i:latest
