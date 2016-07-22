import os
import sys

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

class DataLoader(object):

    def __init__(self, path, input_shape, percent_train, verbose=True, save_data=False):
        self.path = path
        self.input_shape = input_shape
        self.percent_train = percent_train
        self.verbose = verbose
        self.save_data = save_data

    def load_images(self):
        # setup paths
        paths = set()
        for directory in os.listdir(self.path):
            absolute_path = os.path.join(self.path, directory)
            if os.path.isdir(absolute_path):
                paths.add(absolute_path)

        if self.verbose:
            print("Your images should be at: {}".format(paths))

        # allocate memory
        valid_extensions = [".jpg", ".jpeg", ".png"]
        n_images = 0
        n_class = len(paths)
        for path in paths:
            file_list = os.listdir(path)
            for file in file_list:
                if os.path.splitext(file)[1].lower() not in valid_extensions:
                    continue
                n_images += 1

        if self.verbose:
            print("Number of images: {}".format(n_images))
            print("Number of classes: {}".format(n_class))

        array_shape = [n_images]
        array_shape.extend(self.input_shape)
        images = np.ndarray(array_shape)

        array_shape = [n_images, n_class]
        labels = np.ndarray(array_shape)
        if self.verbose:
            print("Data shape is: {}".format(images.shape))
            print("Label shape is: {}".format(labels.shape))

        # load images
        image_counter = 0
        label_array = np.eye(n_class, n_class)
        for i, path in zip(range(n_class), paths):
            for file in os.listdir(path):

                if os.path.splitext(file)[1].lower() not in valid_extensions:
                    continue

                file_path = os.path.join(path, file)
                image = scipy.misc.imread(file_path)

                if len(image.shape) == 2:
                    image = [image, image, image]

                image = scipy.misc.imresize(image, self.input_shape) / 255

                images[image_counter, :] = image
                labels[image_counter, :] = label_array[i]
                image_counter += 1

        if self.verbose:
            print("Done loading images.")

        # divide into training and test set
        n_train = int(self.percent_train * n_images)
        random_index = np.random.randint(n_images, size=n_images)

        train_index = random_index[0 : n_train]
        test_index = random_index[n_train : n_images]

        train_images = images[train_index, :]
        train_labels = labels[train_index, :]
        test_images = images[test_index, :]
        test_labels = labels[test_index, :]

        if self.verbose:
            print("Train images shape: {}".format(train_images.shape))
            print("Train labels shape: {}".format(train_labels.shape))
            print("Test images shape:  {}".format(test_images.shape))
            print("Test labels shape:  {}".format(test_labels.shape))

        if self.save_data:
            save_path = os.path.join(os.getcwd(), "data.npz")
            np.savez(save_path, train_images=train_images, train_labels=train_labels, test_images=test_images, test_labels=test_labels, n_images=n_images)
            if self.verbose:
                print("Data saved: {}".format(save_path))

        data = {"train_images": train_images, "train_labels": train_labels, "test_images": test_images, "test_labels": test_labels, "n_images": n_images}

        # img = images[0, :]
        # plt.imshow(img)
        # plt.show()

        return data