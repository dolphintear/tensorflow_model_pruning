from tensorflow.contrib.model_pruning.python.layers import layers
from tensorflow.python.ops import nn
import tensorflow as tf


class Network(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pruning_inference(self, inputs):
        net = layers.masked_conv2d(inputs, 64, 3)
        net = layers.masked_conv2d(net, 64, 3)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = layers.masked_conv2d(net, 128, 3)
        net = layers.masked_conv2d(net, 128, 3)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = layers.masked_conv2d(net, 256, 3)
        net = layers.masked_conv2d(net, 256, 3)
        net = layers.masked_conv2d(net, 256, 3)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = tf.layers.flatten(net)

        net = layers.masked_fully_connected(net, 1024)
        net = layers.masked_fully_connected(net, 1024)
        logits = layers.masked_fully_connected(net, self.num_classes, activation_fn=None)

        return tf.identity(logits, name='logits')

    def inference(self, inputs):
        net = tf.layers.conv2d(inputs, 64, 3, padding='same', activation=nn.relu)
        net = tf.layers.conv2d(net, 64, 3, padding='same', activation=nn.relu)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = tf.layers.conv2d(net, 128, 3, padding='same', activation=nn.relu)
        net = tf.layers.conv2d(net, 128, 3, padding='same', activation=nn.relu)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = tf.layers.conv2d(net, 256, 3, padding='same', activation=nn.relu)
        net = tf.layers.conv2d(net, 256, 3, padding='same', activation=nn.relu)
        net = tf.layers.conv2d(net, 256, 3, padding='same', activation=nn.relu)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = tf.layers.flatten(net)

        net = tf.layers.dense(net, 1024, activation=nn.relu)
        net = tf.layers.dense(net, 1024, activation=nn.relu)
        logits = tf.layers.dense(net, self.num_classes)

        return tf.identity(logits, name='logits')
