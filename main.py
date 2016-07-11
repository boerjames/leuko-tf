from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn

### Download and load MNIST data.
mnist = learn.datasets.load_dataset('mnist')


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_model(X, y):
    # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and height
    # final dimension being the number of color channels
    X = tf.reshape(X, [-1, 28, 28, 1])

    with tf.variable_scope('conv_layer1'):
        h_conv1 = learn.ops.conv2d(X, n_filters=32, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('conv_layer2'):
        h_conv2 = learn.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    h_fc1 = learn.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
    return learn.models.logistic_regression(h_fc1, y)

# Training and predicting
classifier = learn.TensorFlowEstimator(model_fn=conv_model, n_classes=10, batch_size=100, steps=20000, learning_rate=0.001)
classifier.fit(mnist.train.images, mnist.train.labels)
score = metrics.accuracy_score(mnist.test.labels, classifier.predict(mnist.test.images))
print('Accuracy: {0:f}'.format(score))