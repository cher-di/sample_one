# Многослойный персептрон из матлаба
import numpy as np
import random

class NeuralNetwork():
    def __init__(self, NI, NL, NN):
        self.NL = NL
        self.NI = NI
        self.LW = [np.random.rand(NN[i], NN[i - 1]) for i in range(NL)]
        self.Lb = [np.random.rand(NN[i], 1) for i in range(NL)]
        self.LW[0] = np.random.rand(NN[0], NI)
        self.Lb[0] = np.random.rand(NN[0], 1)
        # self.LW = [np.random.rand(NN[i - 1], NN[i]) for i in range(NL)]
        # self.Lb = [np.random.rand(1, NN[i]) for i in range(NL)]
        # self.LW[0] = np.random.rand(NI, NN[0])
        # self.Lb[0] = np.random.rand(1, NN[0])
        self.NN = NN
        self.activation_function = lambda x: np.arctan(x)
        self.func = lambda x: np.sin(x)

        # for i in range(1, NL):
        #     self.LW[i] = np.random.rand(NN[i], NN[i - 1])
        #     self.Lb[i] = np.random.rand(NN[i], 1)
        # self.LW[0] = np.random.rand(NN[0], NI)
        # self.Lb[0] = np.random.rand(NN[0], 1)


    def count(self, X):
        # count output signal
        curr_signal = np.array(X).T
        curr_signal = self.activation_function(np.dot(self.LW[0], curr_signal) + self.Lb[0])
        for i in range(1, self.NL):
            curr_signal = self.activation_function(np.dot(self.LW[i], curr_signal) + self.Lb[i])
        return curr_signal

    def count_for_train(self, X):
        # count output signal for training
        outputs = [0 for i in range(self.NL)]
        outputs[0] = np.array(X).T
        outputs[0] = self.activation_function(np.dot(self.LW[0], outputs[0]) + self.Lb[0])
        for i in range(1, self.NL):
            outputs[i] = self.activation_function(np.dot(self.LW[i], outputs[i - 1]) + self.Lb[i])
        return outputs

    def train(self, num_iter, learning_rate, interval):
        random.seed()
        for i in range(num_iter):
            inputs = [[k*0.1 for k in range(0, interval*10 + 1)]]
            inputs = np.array(inputs)
            random.shuffle(inputs)
            targets = self.func(inputs)

            # count output signal for training: outputs
            # with signal inside neurons: inside
            outputs = [0 for i in range(self.NL)]
            inside = [0 for i in range(self.NL)]
            inside[0] = np.dot(self.LW[0], inputs) + self.Lb[0]
            outputs[0] = self.activation_function(inside[0])
            for i in range(1, self.NL):
                inside[i] = np.dot(self.LW[i], outputs[i - 1]) + self.Lb[i]
                outputs[i] = self.activation_function(inside[i])


            output_errors = 2*(outputs[self.NL - 1] - targets)
            curr_errors = output_errors.T

            for layer_num in range(self.NL - 1, 0, -1):
                diff_activation = 1/(1 + inside[i] ** 2)
                diff_bias = np.dot(curr_errors, diff_activation)
                diff_weights = diff_bias*(outputs[i - 1].T)
                self.LW[i] = self.LW[i] - learning_rate * diff_weights
                self.Lb[i] = self.Lb[i] - learning_rate * diff_bias
                for_multiply_curr_errors = (np.dot(diff_activation, np.ones(1, self.NN[i - 1]))*self.LW[i]).T
                curr_errors = np.dot(for_multiply_curr_errors, curr_errors)

            diff_activation = 1 / (1 + inside[0] ** 2)
            diff_bias = np.dot(curr_errors, diff_activation)
            diff_weights = diff_bias * inputs
            self.LW[0] = self.LW[0] - learning_rate * diff_weights
            self.Lb[0] = self.Lb[0] - learning_rate * diff_bias

a = NeuralNetwork(1, 2, (10, 1))
a.train(1, 0.01, 1)