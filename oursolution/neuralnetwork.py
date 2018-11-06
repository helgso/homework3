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


class Network(object):

    def __init__(self, num_neurons):
        """
        Initialization method

        :param num_neurons: A list of the amount of neurons in each layer [input, hidden_1, ..., hidden_n, output]
        """
        self.num_layers = len(num_neurons)
        self.layer_sizes = num_neurons
        self.biases = [np.zeros(layer_size) for layer_size in self.layer_sizes[1:]]
        self.weights = [
            [np.random.uniform(-1/np.sqrt(2), 1/np.sqrt(2), x) for i in range(0, y)]
            for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]


if __name__ == '__main__':
    main()
