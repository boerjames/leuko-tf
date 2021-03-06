# Class that facilitates the training of network
# todo train on multiple gpus

from __future__ import print_function
import tensorflow as tf
import numpy as np
import math

class NetworkTrainer(object):

    def __init__(self, data, verbose=True):
        self.data = data
        self.verbose = verbose

    def train_network(self, network, optimization_algorithm, learning_rate, training_epochs, early_stop, batch_size):
        x = network["x"]
        y = network["y"]
        prediction = network["prediction"]

        train_images = self.data["train_images"]
        train_labels = self.data["train_labels"]
        test_images = self.data["test_images"]
        test_labels = self.data["test_labels"]
        n_images = self.data["n_images"]
        n_train = train_images.shape[0]
        n_test = test_images.shape[0]

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
        optimizer = self._optimization_algorithm(optimization_algorithm=optimization_algorithm, learning_rate=learning_rate, cost=cost)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # prepare session
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        n_batch = int(math.ceil(n_train / batch_size))
        random_index = [i for i in xrange(n_train)]

        train_accuracy = 0.0
        test_accuracy = 0.0
        max_test_accuracy = 0.0
        max_test_accuracy_epoch = 0
        early_stopping_counter = 0

        # train for the giving number of epochs
        for epoch in xrange(training_epochs):
            average_cost = 0.0                  # reset the cost for each epoch
            np.random.shuffle(random_index)     # shuffle the data for each epoch

            # for each batch
            for batch in xrange(n_batch):
                batch_index = random_index[batch * batch_size : (batch + 1) * batch_size]
                batch_x = [train_images[i] for i in batch_index]
                batch_y = [train_labels[i] for i in batch_index]

                # todo augment these images on the fly
                # todo look into using a tensorflow 'pipeline' for dealing with augmented images

                # run the optimization algorithm and accumulate cost
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                average_cost += sess.run(cost, feed_dict={x: batch_x, y: batch_y}) / n_batch

            # find the accuracy for training data and testing data after this epoch training
            train_accuracy = sess.run(accuracy, feed_dict={x: train_images, y: train_labels})
            test_accuracy = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})

            if self.verbose:
                print("Epoch: {}/{}".format(epoch + 1, training_epochs))
                print("    Cost:           {:.1f}".format(average_cost))
                print("    Train Accuracy: {:.4f}".format(train_accuracy))
                print("    Test Accuracy:  {:.4f}".format(test_accuracy))

            # update max test accuracy and early stopping counter
            if test_accuracy > max_test_accuracy:
                max_test_accuracy = test_accuracy
                max_test_accuracy_epoch = epoch + 1
                early_stopping_counter = 0

                # todo save network when there is an improvement
            else:
                early_stopping_counter += 1

            # check if training should stop early
            if early_stop:
                if early_stopping_counter >= early_stop:
                    if self.verbose:
                        print("Stopping early at epoch {}".format(epoch + 1))
                    break

        if self.verbose:
            print("Done! Max test accuracy {:.4f} at epoch {}".format(max_test_accuracy, max_test_accuracy_epoch), end='\n\n')

        return train_accuracy, test_accuracy

    # helper function for adding an optimization algorithm
    def _optimization_algorithm(self, optimization_algorithm, learning_rate, cost):
        if optimization_algorithm == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        elif optimization_algorithm == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=learning_rate).minimize(cost)
        elif optimization_algorithm == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
        elif optimization_algorithm == 'adadelta':
            return tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
        elif optimization_algorithm == 'adam':
            return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        elif optimization_algorithm == 'ftrl':
            return tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(cost)
        elif optimization_algorithm == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)