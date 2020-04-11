import tensorflow as tf
# -----------prepare tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
tf.get_logger().setLevel('ERROR')
# -----------

import os
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

import numpy as np
from data.get_dataset import PROBLEM_CLASSIFICATION, PROBLEM_DENOISING
from data.data_pipe import ProcessData, ProcessClassLabel
from data.data_manager import Data_manager
from models.metrics import mse as metric_mse

data_path = '/mnt/ssd1T_samsung860EVO/NDL'
preproc_X = ProcessData([ProcessData.np_load])
preproc_Y = ProcessData([ProcessData.np_load])
d = Data_manager(data_path=data_path, problem=PROBLEM_DENOISING,
                 preproc_trn_X=preproc_X, preproc_trn_Y=preproc_Y,
                 preproc_val_X=preproc_X, preproc_val_Y=preproc_Y)

l2_beta = 0.0001
dropout_drop_prob = 0.2

optimizer = tf.keras.optimizers.RMSprop
learning_rate = 0.0001
clipnorm = 1
clipvalue = None
optimizer_kwargs = {
    'learning_rate': learning_rate
}
if clipnorm is not None:
    optimizer_kwargs['clipnorm'] = clipnorm
if clipvalue is not None:
    optimizer_kwargs['clipvalue'] = clipvalue
optimizer = optimizer(**optimizer_kwargs)

from models.model_denoising import ModelDenoising
m = ModelDenoising(cell_shape=(80,), optimizer=optimizer, l2_beta=l2_beta, dropout_drop_prob=dropout_drop_prob)


def validate(part=1.0):
    mses = []
    for val_x, val_y in d.get_next_val_batch(batch_size=1, part=part):
        prediction = m(val_x)
        mse = metric_mse(val_y, prediction).numpy()
        mses.append(mse)
    mses = np.array(mses)
    MSE = np.mean(mses)
    return MSE


def save_m(dst_path):
    tf.saved_model.save(m, dst_path)


def load_m(src_path):
    return tf.saved_model.load(src_path)


best_mse = 10**1000


def validate_and_save(m, part=1.0, tmp_path=os.path.join(os.path.dirname(__file__), 'tmp'), best_mse=10**1000):
    mse = validate(part=part)
    if mse < best_mse:
        dst_path = os.path.join(tmp_path, 'saved_model_{}'.format(SCRIPT_NAME))
        save_m(dst_path)
        m = load_m(dst_path)
        best_mse = mse
    return mse, m, best_mse


mse, m, best_mse = validate_and_save(m, part=0.05, best_mse=best_mse)
print('val mse: {}'.format(round(mse, 4)))

eps = 1000
for ep in range(1, eps+1):
    for batch, labels, b in d.get_next_trn_pair_denoising():
        gt_loss = m.train_step(batch, labels)
        gt_loss = gt_loss.numpy()
        if b % 100 == 0:
            print('b={}; trn gt_loss={}'.format(b, gt_loss))
            if b % 3000 == 0:
                mse = validate(part=0.05)
                print('b={}; val mse={}'.format(b, round(mse, 4)))
    mse, m, best_mse = validate_and_save(m, part=1.0, best_mse=best_mse)
    print('ep={}; val mse={}; best_mse={}'.format(ep, mse, best_mse))
