# NDL
## task1
В этом же репо /task1/main.py
## task2
### Инструкции по запуску докера:
Сбилженный docker образ качать отсюда https://yadi.sk/d/AYySTnKB9FIe6Q  
(Там 5.6G, почему-то даже lfs не помог залить на гитхаб)  

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
Если билдить образ, то нужно запустить build.sh в корне этого репо

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

### Описание алгоритмов
Для обоих задач использовалась двуслойная LSTM-сеть одинаковой архитектуры  
Для каждой задачи сеть обучалась отдельно, если это классификация, то на выходе последней ячейки LSTM висит двуслойный fc-классификатор.  
Можно было обучить LSTM на denoising и отдельно на ней же обучить классификатор, такая интересная идея пришла поздно.  

При обучении подавал батчи размером 2.  
Для denoising это выглядит так: X=[noisy, clean], Y=[clean, clean]  
noisy и clean - это дорожки с одинаковыми именами и спикерами.  

Все параметры в нейросетях имеют тип float16  
Обычно используют float32, а я здесь как то машинально сделал float16  
В последний раз, когда я пытался на изображениях перейти с float32 на float16, метрики качества ощутимо упали :(

### Smth else:
predict.py не чистит PREDICTIONS_PATH, так что, возможно, лучше чистить PREDICTIONS_PATH вручную перед запуском оценок.  

Чтобы файл из DATA_PATH был включен в оценку алгоритмами, необходимо:  
- расширение файла .npy
- объект внутри файла имеет: type=np.ndarray, shape=(t, 80), dtype=np.float16
