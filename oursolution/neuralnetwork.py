import os
# Set current working directory as this file's path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import mnist_reader
import numpy as np


def main():
    # Circles.txt training data
    circles_data = np.loadtxt('data/circles/circles.txt')

    # MNIST training data
    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # our_algorithm()


if __name__ == '__main__':
    main()
