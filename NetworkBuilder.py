import tensorflow as tf
import numpy as np

class NetworkBuilder(object):

    # INPUT -> [[CONV -> ACT]*N -> POOL?]*M -> [FC -> ACT]*K -> FC
    def __init__(self, input_size, n_class):
        self._input_size = input_size
        self._n_class = n_class


    def build_network(self,
                      conv_filters=[8, 8, 8, 16, 16, 16, 32, 32, 32],
                      m=3,
                      fc_neurons=[100, 50],
                      conv_size=3,
                      conv_stride=1,
                      pool_size=2,
                      pool_stride=2):

        input = tf.placeholder(tf.float32, [None, self._input_size[0], self._input_size[1], self._input_size[2]], name='input')
        output = tf.placeholder(tf.float32, [None, self._n_class], name='output')

        network = tf.reshape(input, shape=[-1, self._input_size[0], self._input_size[1], self._input_size[2]])

        print(network.get_shape())

        n_filters = self._input_size[2]
        for i, val in enumerate(conv_filters):

            W = tf.Variable(tf.random_normal([conv_size, conv_size, n_filters, val]))
            b = tf.Variable(tf.random_normal([val]))
            network = tf.nn.conv2d(network, W, strides=[1, conv_stride, conv_stride, 1], padding='SAME')
            network = tf.nn.bias_add(network, b)
            network = tf.nn.relu(network)

            n_filters = val

            if (not i == 0) and (i % m == 0):
                network = tf.nn.max_pool(network, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_stride, pool_stride, 1], padding='SAME')

            print(network.get_shape())

        shape = network.get_shape().as_list()
        n_neurons = shape[1] * shape[2] * shape[3]
        network = tf.reshape(network, [-1, n_neurons])

        for i, val in enumerate(fc_neurons):
            W = tf.Variable(tf.random_normal([n_neurons, val]))
            b = tf.Variable(tf.random_normal([val]))
            network = tf.add(tf.matmul(network, W), b)
            network = tf.nn.relu(network)

            n_neurons = val

            print(network.get_shape())

        W = tf.Variable(tf.random_normal([n_neurons, self._n_class]))
        b = tf.Variable(tf.random_normal([self._n_class]))
        network = tf.add(tf.matmul(network, W), b)
        network = tf.nn.softmax(network)

        print(network.get_shape())

        return input, output, network
