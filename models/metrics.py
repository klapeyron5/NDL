import tensorflow as tf


def loss_classification(labels, logits):
    out = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    out = tf.reduce_mean(out)
    return out


def acc(labels, predicts):
    out = tf.keras.metrics.binary_accuracy(labels, predicts, threshold=0.5)
    out = tf.reduce_mean(out)
    return out


def mse(gt, predicted):
    # shape=(batch, time, mel)
    return tf.reduce_mean(tf.keras.losses.MSE(gt, predicted))
