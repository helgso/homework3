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

    def __init__(
        self,
        num_neurons
    ):
        """
        Initialization method

        :param num_neurons: A list of the amount of neurons in each layer [input, hidden_1, ..., hidden_n, output]
        """
        self.num_layers = len(num_neurons)
        self.layer_sizes = num_neurons
        self.biases = [np.zeros(layer_size) for layer_size in self.layer_sizes[1:]]
        self.weights = [
            [np.random.uniform(-1/np.sqrt(y), 1/np.sqrt(y), x) for i in range(0, y)]
            for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]

    def finite_difference_gradient_check(
        self,
        test_data
    ):
        """
        Used to estimate the gradient numerically. Used to check our gradient computation

        :param test_data: An array of examples+labels we want to verify our back propagation algorithm with
        """
        # 1: Calculate the loss function for the current parameter values (single example or mini batch)
        #loss_current_params = loss_function(test_data, current_params)

        # 2: Calculate the loss function for the current parameter values but add a small number ksi [10^-6; 10^-4] to each parameter first
        #ksi = np.random.uniform(10 ** -6, 10 ** -4)
        #loss_current_params_plus_ksi = loss_function(test_data, current_params + ksi)

        # 3: Divide the change of the loss function by ksi
        #finite_difference = np.absolute(loss_current_params - loss_current_params_plus_ksi)/ksi

        # 4: Validate that the ratio of our gradient computed by the back propagation and the finite difference is between 0.99 and 1.01
        #gradient_params = back_propagation(test_data)
        #loss_new_params = loss_function(test_data, gradient_params)
        #assert 0.99 <= finite_difference/loss_new_params <= 1.01


if __name__ == '__main__':
    main()
