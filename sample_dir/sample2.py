# Многослойный персептрон из матлаба
import numpy as np
import random
from math import sin, fabs

class NeuralNetwork():
    def __init__(self, NI, NL, NN):
        self.NL = NL
        self.NI = NI
        self.LW = [np.random.rand(NN[0], NI)]
        self.LW += [np.random.rand(NN[i], NN[i - 1]) for i in range(1, NL)]
        self.Lb = [np.random.rand(NN[i], 1) for i in range(NL)]
        # self.LW[0] = np.random.rand(NN[0], NI)
        # self.Lb[0] = np.random.rand(NN[0], 1)
        self.NN = NN
        self.activation_function = lambda x: np.arctan(x)
        self.func = lambda x: np.sin(x)


    def count(self, X):
        # count output signal
        curr_signal = np.array(X)
        curr_signal = self.activation_function(np.dot(self.LW[0], curr_signal) + self.Lb[0])
        for i in range(1, self.NL):
            curr_signal = self.activation_function(np.dot(self.LW[i], curr_signal) + self.Lb[i])
        return curr_signal

    def train(self, learning_rate, interval):
        random.seed()
        num_of_div = 100
        input_all = np.linspace(0, interval, num_of_div)
        random.shuffle(input_all)
        for input in input_all:
            input = np.array((input))
            targets = self.func(input)

            # count output signal for training: outputs
            # with signal inside neurons: inside
            outputs = [None for i in range(self.NL)]
            inside = [None for i in range(self.NL)]
            inside[0] = np.dot(self.LW[0], input) + self.Lb[0]
            outputs[0] = self.activation_function(inside[0])
            for i in range(1, self.NL):
                inside[i] = np.dot(self.LW[i], outputs[i - 1]) + self.Lb[i]
                outputs[i] = self.activation_function(inside[i])


            output_errors = 2*(outputs[self.NL - 1] - targets)
            curr_errors = output_errors

            for layer_num in range(self.NL - 1,  0, -1):
                diff_activation = 1 / (1 + inside[i] ** 2)
                diff_bias = curr_errors*diff_activation
                diff_weights = np.dot(curr_errors*diff_activation, outputs[i - 1].T)
                self.LW[i] = self.LW[i] - learning_rate * diff_weights
                self.Lb[i] = self.Lb[i] - learning_rate * diff_bias
                for_multiply_curr_errors = (np.dot(diff_activation, np.ones((diff_activation.shape[1], self.NN[i - 1])))*self.LW[i]).T
                curr_errors = np.dot(for_multiply_curr_errors, curr_errors)

            diff_activation = 1 / (1 + inside[0] ** 2)
            diff_bias = curr_errors*diff_activation
            diff_weights = np.dot(curr_errors*diff_activation, input.T)
            self.LW[0] = self.LW[0] - learning_rate * diff_weights
            self.Lb[0] = self.Lb[0] - learning_rate * diff_bias

# tests
a = NeuralNetwork(1, 2, (3, 1))

test_num = 1000
interval = np.pi
test_set = np.linspace(0, interval, test_num)

# опрос сети сразу после создания
mean_error = 0
for test in test_set:
    mean_error += fabs(sin(test) - a.count(test))
mean_error = mean_error / test_num
print('Mean error before train:', mean_error)

# опрос сети после каждой 100 тренировок

print('Mean error per every 100 iteration:')

for i in range(1, 10001):
    a.train(0.01, interval)

    if not(i % 100):
        mean_error = 0
        for test in test_set:
            mean_error += fabs(sin(test) - a.count(test))
        mean_error = mean_error / test_num
        print(i, ':', mean_error)