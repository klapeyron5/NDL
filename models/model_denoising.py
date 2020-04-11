import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras import regularizers
from models.metrics import mse


class ModelDenoising(tf.Module):

    output_dim = (None,)
    cells_dim = (None,)
    batch_dim = (None,)

    NAME = 'ModelDenoising'

    def __init__(self, cell_shape, optimizer, l2_beta=0.00001, dropout_drop_prob=0.2):
        self.cell_shape = cell_shape

        lstm0_cell_shape = (self.cell_shape[0],)
        self.lstm0 = LSTM(units=self.cell_shape[0], return_sequences=True,
                          batch_input_shape=self.batch_dim+self.cells_dim+self.cell_shape,
                          dtype=tf.float16, kernel_regularizer=regularizers.l2(l2_beta))
        self.lstm1 = LSTM(units=self.cell_shape[0], return_sequences=True,
                          batch_input_shape=self.batch_dim+self.cells_dim+lstm0_cell_shape,
                          dtype=tf.float16, kernel_regularizer=regularizers.l2(l2_beta))

        self.optimizer = optimizer
        self.reg_l2_beta = tf.Variable(0.0, dtype=tf.float16, trainable=False)
        self.dropout_drop_prob = tf.Variable(0.0, dtype=tf.float16, trainable=False)

        self.__call__ = tf.function(
            self.__call__,
            input_signature=[
                tf.TensorSpec(self.batch_dim + self.cells_dim + self.cell_shape, tf.float16),
            ])
        self.train_step = tf.function(
            self.train_step,
            input_signature=[
                tf.TensorSpec(self.batch_dim + self.cells_dim + self.cell_shape, tf.float16),
                tf.TensorSpec(self.batch_dim + self.cells_dim + self.output_dim, tf.float16),
            ])
        self.loss = tf.function(
            mse,
            input_signature=[
                tf.TensorSpec(self.batch_dim + self.cells_dim + self.output_dim, tf.float16),
                tf.TensorSpec(self.batch_dim + self.cells_dim + self.output_dim, tf.float16),
            ])
        self.set_regularization_config = tf.function(
            self.set_regularization_config,
            input_signature=[
                tf.TensorSpec([], tf.float16),
                tf.TensorSpec([], tf.float16),
            ])

        self.set_regularization_config(l2_beta, dropout_drop_prob)


    @tf.function
    def __call__(self, x):
        return self.denoise(x, False)

    @tf.function
    def denoise(self, x, training):
        x = self.lstm0(x, training=training)
        return self.lstm1(x, training=training)

    @tf.function
    def train_step(self, data, gt):
        with tf.GradientTape() as tape:
            denoised = self.denoise(data, True)
            gt_loss = self.loss(gt, denoised)
        grads = tape.gradient(gt_loss, self.trainable_variables)
        self.optimizer.apply_gradients(list(zip(grads, self.trainable_variables)))
        return gt_loss

    @tf.function
    def reg_loss(self):
        l2_loss = tf.constant(0, dtype=tf.float16)
        l2_loss += self.fc0.get_reg_loss_l2()
        l2_loss += self.fc1.get_reg_loss_l2()
        return l2_loss

    @tf.function
    def set_regularization_config(self, l2_beta, dropout_drop_prob):
        self.reg_l2_beta.assign(l2_beta)
        self.dropout_drop_prob.assign(dropout_drop_prob)
        return 0
