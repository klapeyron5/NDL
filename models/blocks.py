import tensorflow as tf


class Dense(tf.Module):
    def __init__(self, input_shape, output_shape, name=None):
        super(Dense, self).__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal(input_shape + output_shape, stddev=0.1, dtype=tf.float16), name='w', dtype=tf.float16)
        self.b = tf.Variable(
            tf.zeros(output_shape, dtype=tf.float16), name='b', dtype=tf.float16)

    @tf.function
    def __call__(self, x, training):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

    @tf.function
    def get_reg_loss_l2(self):
        l2_loss = tf.nn.l2_loss(self.w)
        return l2_loss
