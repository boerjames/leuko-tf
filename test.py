import math
import numpy as np
import itertools

n_images = 25
batch_size = 10
n_epoch = 3

images = [[i, i+1, i+2, i+3] for i in range(25)]

n_batch = math.ceil(n_images / batch_size)
random_index = [i for i in range(n_images)]

for epoch in range(n_epoch):
    np.random.shuffle(random_index)
    print("Epoch {}".format(epoch))

    for batch in range(n_batch):
        batch_index = random_index[batch * batch_size : (batch + 1) * batch_size]
        batch_x = [images[i] for i in batch_index]
        print(batch_x)

    print()

#pprint.pprint(locals())