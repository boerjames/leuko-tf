# Class that facilitates the building of CNNs
# General CNN structure: INPUT -> [[CONV -> ACT]*N -> POOL?]*M -> [FC -> ACT]*K -> FC

from __future__ import print_function
import tensorflow as tf

class NetworkBuilder(object):

    def __init__(self, input_shape, n_class, verbose=True):
        self._input_shape = input_shape
        self._n_class = n_class
        self.verbose = verbose

    def build_network(self,
                      conv_filters=[8, 8, 8],           # list of how many convolution filters to use in each layer, len(conv_filters) = n
                      conv_size=[3, 3, 3],              # size of the convolutions for each convolution layer
                      conv_stride=[1, 1, 1],            # stride of the convolutions for each convolution layer
                      m=1,                              # number of convolution layers before each pooling layer
                      fc_neurons=[25],                  # list of how many neurons are in each fully-connected layer, len(fc_neurons) = k
                      pool_size=2,                      # size of the pooling
                      pool_stride=2,                    # stride of the pooling
                      activation='elu'):                # activation function to use

        # input and output placeholders
        temp = [None]
        temp.extend(self._input_shape)
        x = tf.placeholder(tf.float32, temp)
        y = tf.placeholder(tf.float32, [None, self._n_class])

        # reshape network input to allow batches
        temp = [-1]
        temp.extend(self._input_shape)
        network = tf.reshape(x, shape=temp)

        if self.verbose:
            print("Input shape: {}".format(network.get_shape()))

        # add convolution layers
        n_filters = self._input_shape[2]
        for i, val in enumerate(conv_filters):
            W = tf.Variable(tf.random_normal([conv_size[i], conv_size[i], n_filters, val]))
            b = tf.Variable(tf.random_normal([val]))
            network = tf.nn.conv2d(network, W, strides=[1, conv_stride[i], conv_stride[i], 1], padding='SAME')
            network = tf.nn.bias_add(network, b)
            network = self._add_activation(network, activation)

            n_filters = val

            if self.verbose:
                print("Convolution layer {} shape: {}".format(i + 1, network.get_shape()))

            # add pooling layers if necessary
            if (i + 1) % m == 0:
                network = tf.nn.max_pool(network, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_stride, pool_stride, 1], padding='SAME')

        # prepare for fully-connected layers
        shape = network.get_shape().as_list()
        n_neurons = shape[1] * shape[2] * shape[3]
        network = tf.reshape(network, [-1, n_neurons])

        # add fully-connected layers
        for i, val in enumerate(fc_neurons):
            W = tf.Variable(tf.random_normal([n_neurons, val]))
            b = tf.Variable(tf.random_normal([val]))
            network = tf.add(tf.matmul(network, W), b)
            network = self._add_activation(network, activation)

            n_neurons = val

            if self.verbose:
                print("Fully-connected layer {} shape: {}".format(i + 1, network.get_shape()))

        # add fully-connected output layer
        W = tf.Variable(tf.random_normal([n_neurons, self._n_class]))
        b = tf.Variable(tf.random_normal([self._n_class]))
        network = tf.add(tf.matmul(network, W), b)

        if self.verbose:
            print("Output layer shape: {}".format(network.get_shape()), end='\n\n')

        network = {'x': x, 'y': y, 'prediction': network}

        return network

    # helper function for adding activation function
    def _add_activation(self, network, activation):
        if activation == 'relu':
            return tf.nn.relu(network)
        elif activation == 'relu6':
            return tf.nn.relu6(network)
        elif activation == 'elu':
            return tf.nn.elu(network)
        else:
            raise ValueError("Please provide a valid activation function.")