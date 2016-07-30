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

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        n_batch = math.ceil(n_train / batch_size)
        random_index = [i for i in range(n_train)]

        train_accuracy = 0.0
        test_accuracy = 0.0
        max_test_accuracy = 0.0
        early_stopping_counter = 0

        for epoch in range(training_epochs):
            average_cost = 0.0
            np.random.shuffle(random_index)

            for batch in range(n_batch):
                batch_index = random_index[batch * batch_size : (batch + 1) * batch_size]
                batch_x = [train_images[i] for i in batch_index]
                batch_y = [train_labels[i] for i in batch_index]

                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                average_cost += sess.run(cost, feed_dict={x: batch_x, y: batch_y}) / n_batch

            train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            test_accuracy = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})

            if self.verbose:
                print("Epoch: {}/{}".format(epoch + 1, training_epochs))
                print("    Cost:           {}".format(average_cost))
                print("    Train Accuracy: {}".format(train_accuracy))
                print("    Test Accuracy:  {}".format(test_accuracy))

            if early_stop > 0:
                if test_accuracy > max_test_accuracy:
                    max_test_accuracy = test_accuracy
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stop:
                    if self.verbose:
                        print("Stopping early at epoch {}".format(epoch + 1))
                    break

        if self.verbose:
            print("Done! Test accuracy: {}".format(test_accuracy))

        return train_accuracy, test_accuracy

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