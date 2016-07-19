import glob
import os

import tensorflow as tf

def read_raw_images(path):
    images = []
    png_files = []
    jpg_files = []

    reader = tf.WholeFileReader()

    png_files_path = glob.glob(os.path.join(path, '*.[pP][nN][gG]'))
    jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))
    jpeg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][eE][gG]'))

    for filename in png_files_path:
        png_files.append(filename)
    for filename in jpg_files_path:
        jpg_files.append(filename)
    for filename in jpeg_files_path:
        jpg_files.append(filename)

    print(len(png_files) + len(jpg_files))