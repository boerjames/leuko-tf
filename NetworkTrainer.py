from Network import Network
import tensorflow as tf
import numpy as np

class NetworkTrainer(object):

    def __init__(self, network: Network):
        self.input = network.input
        self.target = network.output
        self.prediction = network.network

    def train(self, learning_rate, training_epochs):
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.target * tf.log(self.prediction), reduction_indices=1))   # cross entropy
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)                               # gradient descent

        correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.target))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        # temp!
        ntrain = 100
        batch_size = 32
        trainimg = []
        trainlabel = []
        display_step = 5
        for epoch in range(training_epochs):
            avg_cost = 0.0
            num_batch = int(ntrain / batch_size) + 1

            # loop over batches
            for i in range(num_batch):
                randidx = np.random.randint(ntrain, size=batch_size)
                batch_xs = trainimg[randidx, :]
                batch_ys = trainlabel[randidx, :]
                sess.run(self.optimizer, feed_dict={self.input: batch_xs, self.target: batch_ys})
                avg_cost += sess.run(self.cost, feed_dict={self.input: batch_xs, self.target: batch_ys}) / num_batch

            if epoch % display_step == 0:
                print("Epoch %03d/%03d cost: %.9f".format(epoch, training_epochs, avg_cost))
                train_acc = sess.run(accuracy, feed_dict={self.input: batch_xs, self.target: batch_ys})
                print("    Training accuracy: %.9f".format(train_acc))

        print("Done")