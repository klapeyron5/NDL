import os
import numpy as np
import tensorflow as tf
import json
from time import time

script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

print('------------------gpus list here------------------')
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
print('------------------any gpus?------------------')

# здесь лежат данные, на которых нужно протестироваться
# предсказания будут получены для файлов с расширением .npy, если массив внутри файла имеет shape=(t, 80), dtype=float16
# предсказание для файла включает в себя и классификацию и денойзинг
data_path = '/home/data'

# здесь будут лежать предсказания для файлов из data_path
predictions_path = '/home/predictions'
# - /home/predictions
# -- classification.json  # {filepath: label}| label 0 is clean, label 1 is noisy
# -- /denoising
# --- ...  # denoised файлы; структура каталога и имена файлов идентичны каталогу тестовых данных /home/data

errmsg = "{} doesn't exist"
assert os.path.isdir(data_path), errmsg.format(data_path)
assert os.path.isdir(predictions_path), errmsg.format(predictions_path)

denoising_predictions_path = os.path.join(predictions_path, 'denoising')
classification_json_path = os.path.join(predictions_path, 'classification.json')
classification_json = {}


def load_model(model_path):
    """загрузка tf модели"""
    try:
        errmsg = "directory {} doesn't exist".format(model_path)
        assert os.path.isdir(model_path), errmsg
        model = tf.saved_model.load(model_path)
        return model
    except Exception as e:
        print("ERROR: can't load model {}".format(model_path))
        print(e)
        raise Exception


# LSTM для классификации: два слоя LSTM cells, выход ячеек такой же размерности (80,) как входной mel спектр фрейма
# выход последней ячейки передается на вход двухслойному классификатору:
# h_t(80,) -> fc_relu(80, 80) -> fc_relu_softmax(80, 2)
# TODO: в последний слой случайно попала relu, хотя должно быть просто fc_softmax.
#  Насколько успела пообучаться "исправленная" сетка, кажется что эта ошибка понизила точность :(
classification_model_path = os.path.join(script_dir, 'exported_models/classification/saved_model')
classification_model = load_model(classification_model_path)

# сразу протестируем, что модель работает:
errmsg = "classification_model doesn't work. Model was loaded from {}".format(classification_model_path)
test_data = tf.random.uniform((2, 555, 80), dtype=tf.float16)
prediction = classification_model(test_data)
assert prediction.shape == (2, 2), errmsg


# LSTM для denoising: такая же архитектура LSTM как для классификации, только без классификатора на последнем выходе
# TODO: можно было, скорее всего, обойтись одной LSTM:
#  обучить ее для denoising, а потом отдельно обучить классификатор для последнего выхода LSTM; идея пришла поздно
denoising_model_path = os.path.join(script_dir, 'exported_models/denoising/saved_model')
denoising_model = load_model(denoising_model_path)

# сразу протестируем, что модель работает:
errmsg = "denoising_model doesn't work. Model was loaded from {}".format(denoising_model_path)
test_data = tf.random.uniform((2, 555, 80), dtype=tf.float16)
prediction = denoising_model(test_data)
assert prediction.shape == (2, 555, 80), errmsg


def json_dump(json_filepath, json_obj, indent=4):
    try:
        with open(json_filepath, 'w', encoding='utf8') as f:
            ser = json.dumps(json_obj, indent=indent)
            f.write(ser)
    except Exception as e:
        print("ERROR: can't dump to json {}".format(json_filepath))
        print(e)
        raise Exception


def is_proper_sample(sample):
    """
    Проверяет формат считанного файла mel-спектров
    """
    try:
        assert isinstance(sample, np.ndarray)
        assert len(sample.shape) == 2 and sample.shape[-1] == 80
        assert sample.dtype == np.float16
        return True
    except Exception:
        return False


def read_sample(filepath):
    """
    Считывает файл mel-спектров
    Возвращает считанный массив или None, если что-то не так с форматом
    """
    try:
        assert filepath.endswith('.npy')
        sample = np.load(filepath)
        assert is_proper_sample(sample)
        return sample
    except Exception:
        return None


def classify(sample_batch):
    """sample_batch.shape == (1, t, 80)"""
    default_label = 0
    try:
        batch_probs = classification_model(sample_batch)
        probs = batch_probs.numpy()[0]
        label = int(np.argmax(probs))  # threshold = 0.5
        assert label in {0, 1}
        return label
    except Exception:
        return default_label


def denoise(sample_batch):
    """sample_batch.shape == (1, t, 80)"""
    defaul_denoised = np.zeros_like(sample_batch[0])
    try:
        denoised_batch = denoising_model(sample_batch)
        denoised = denoised_batch.numpy()[0]
        assert is_proper_sample(denoised)
        return denoised
    except Exception:
        return defaul_denoised

t1 = time()
for pardir, dirs, files in os.walk(data_path):
    for filename in files:
        filepath = os.path.join(pardir, filename)
        rel_filepath = os.path.relpath(filepath, data_path)
        sample = read_sample(filepath)
        if sample is not None:
            batch = np.reshape(sample, (1,) + sample.shape)

            sample_label = classify(batch)
            classification_json[rel_filepath] = sample_label

            denoised_sample = denoise(batch)
            denoised_dir = os.path.relpath(pardir, data_path)
            denoised_dir = os.path.join(denoising_predictions_path, denoised_dir)
            os.makedirs(denoised_dir, exist_ok=True)
            denoised_filepath = os.path.join(denoised_dir, filename)
            np.save(denoised_filepath, denoised_sample)
        else:
            print("WARNING: {} hasn't proper format as mel-processed sound signal".format(filepath))
print("wasted time={}".format(time()-t1))
json_dump(classification_json_path, classification_json)
