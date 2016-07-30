import tensorflow as tf

class NetworkBuilder(object):

    # INPUT -> [[CONV -> ACT]*N -> POOL?]*M -> [FC -> ACT]*K -> FC

    def __init__(self, input_shape, n_class, verbose=True):
        self._input_shape = input_shape
        self._n_class = n_class
        self.verbose = verbose

    def build_network(self,
                      conv_filters=[8, 12, 16],
                      m=1,
                      fc_neurons=[50],
                      conv_size=5,
                      conv_stride=1,
                      pool_size=2,
                      pool_stride=2,
                      activation='elu'):

        temp = [None]
        temp.extend(self._input_shape)
        x = tf.placeholder(tf.float32, temp)
        y = tf.placeholder(tf.float32, [None, self._n_class])

        temp = [-1]
        temp.extend(self._input_shape)
        network = tf.reshape(x, shape=temp)

        if self.verbose:
            print(network.get_shape())

        n_filters = self._input_shape[2]
        for i, val in enumerate(conv_filters):
            W = tf.Variable(tf.random_normal([conv_size, conv_size, n_filters, val]))
            b = tf.Variable(tf.random_normal([val]))
            network = tf.nn.conv2d(network, W, strides=[1, conv_stride, conv_stride, 1], padding='SAME')
            network = tf.nn.bias_add(network, b)
            network = self._add_activation(network, activation)

            n_filters = val

            if (not i == 0) and (i % m == 0):
                network = tf.nn.max_pool(network, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_stride, pool_stride, 1], padding='SAME')

            if self.verbose:
                print(network.get_shape())

        shape = network.get_shape().as_list()
        n_neurons = shape[1] * shape[2] * shape[3]
        network = tf.reshape(network, [-1, n_neurons])

        for i, val in enumerate(fc_neurons):
            W = tf.Variable(tf.random_normal([n_neurons, val]))
            b = tf.Variable(tf.random_normal([val]))
            network = tf.add(tf.matmul(network, W), b)
            network = self._add_activation(network, activation)

            n_neurons = val

            if self.verbose:
                print(network.get_shape())

        W = tf.Variable(tf.random_normal([n_neurons, self._n_class]))
        b = tf.Variable(tf.random_normal([self._n_class]))
        network = tf.add(tf.matmul(network, W), b)

        if self.verbose:
            print(network.get_shape())

        network = {'x': x, 'y': y, 'prediction': network}

        return network

    def _add_activation(self, network, activation):
        if activation == 'relu':
            return tf.nn.relu(network)
        elif activation == 'relu6':
            return tf.nn.relu6(network)
        elif activation == 'elu':
            return tf.nn.elu(network)
        else:
            raise ValueError("Please provide a valid activation function.")