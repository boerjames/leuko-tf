# Class the facilitates the loading of images into the proper data structure for training
# Images must be organized in the following way:
#   path/class1/
#   path/class2/
#   etc.

from __future__ import print_function
import os
import gc
import numpy as np
import scipy.misc

class DataLoader(object):

    def __init__(self, path, save_path, data_shape, percent_train=0.8, normalize=True, verbose=True, save_data=False):
        self.path = path
        self.save_path = save_path
        self.data_shape = data_shape
        self.percent_train = percent_train
        self.normalize = normalize
        self.verbose = verbose
        self.save_data = save_data

    def load_images(self):

        # prepare save data path
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

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
        array_shape.extend(self.data_shape)
        images = np.ndarray(array_shape)

        array_shape = [n_images, n_class]
        labels = np.ndarray(array_shape)
        if self.verbose:
            print("All images shape is: {}".format(images.shape))
            print("All labels shape is: {}".format(labels.shape))

        # load images
        image_counter = 0
        label_array = np.eye(n_class, n_class)
        for i, path in zip(xrange(n_class), paths):
            for file in os.listdir(path):

                if os.path.splitext(file)[1].lower() not in valid_extensions:
                    continue

                file_path = os.path.join(path, file)
                image = scipy.misc.imread(file_path)

                # convert greyscale to rgb, naive
                if len(image.shape) == 2:
                    image = [image, image, image]

                image = scipy.misc.imresize(image, self.data_shape) / 255

                images[image_counter, :] = image
                labels[image_counter, :] = label_array[i]
                image_counter += 1

        # divide into training and test set
        n_train = int(self.percent_train * n_images)
        random_index = np.random.randint(n_images, size=n_images)

        train_index = random_index[0 : n_train]
        test_index = random_index[n_train : n_images]

        train_images = images[train_index, :]
        train_labels = labels[train_index, :]
        test_images = images[test_index, :]
        test_labels = labels[test_index, :]

        # clear out old stuff
        del images
        del labels
        gc.collect()

        # normalize the data based on training data statistics
        if self.normalize:
            image_mean = np.mean(train_images, axis=0)
            image_std = np.std(train_images, axis=0)

            train_images = np.subtract(train_images, image_mean)
            train_images = np.divide(train_images, image_std)
            test_images = np.subtract(test_images, image_mean)
            test_images = np.divide(test_images, image_std)

            # if the network is trained on normalized data, the deployed model must normalize data as well
            if self.save_data:
                np.save(os.path.join(self.save_path, 'image_mean'), image_mean)
                np.save(os.path.join(self.save_path, 'image_std'), image_std)

        if self.verbose:
            print("Train images shape: {}".format(train_images.shape))
            print("Train labels shape: {}".format(train_labels.shape))
            print("Test images shape:  {}".format(test_images.shape))
            print("Test labels shape:  {}".format(test_labels.shape), end='\n\n')


        data = {"train_images": train_images, "train_labels": train_labels, "test_images": test_images, "test_labels": test_labels, "n_images": n_images, "n_class": n_class}

        return data