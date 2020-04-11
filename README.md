# NDL
## task1
В этом же репо /task1/main.py
## task2
### Инструкции по запуску докера:
Сбилженный docker образ качать отсюда https://github.com/klapeyron5/NDL_docker_image  

При запуске образа необходимо указать два пути на машине хоста (см. run.sh):  
DATA_PATH - это путь к данным, на которых необходимо протестироваться  
PREDICTIONS_PATH - здесь будут лежать предсказания  
```bash
#!/bin/bash
readonly DATA_PATH=/path/to/data/
readonly PREDICTIONS_PATH=/where/to/save/predictions/

docker run \
-v $DATA_PATH:/home/data \
-v $PREDICTIONS_PATH:/home/predictions \
--name ndl_c ndl_i:latest
```
Если билдить образ, то см. Dockerfile и build.sh

### Что делает скрипт в докере:
Скрипт, который запускается в докере - это **predict.py**  

Структура PREDICTIONS_PATH:  
 -- PREDICTIONS_PATH  
 --- classification.json  
 --- /denoising  
 
##### classification.json - это словарь {filepath: label}  
label 0 - clean  
label 1 - noisy  
filepath - относительный путь к файлу в стартовой директории DATA_PATH
 
##### директория /denoising:  
здесь лежат denoised файлы  
структура каталога и имена файлов идентичны каталогу тестовых данных DATA_PATH
