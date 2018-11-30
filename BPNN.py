import random
import math
import copy
import numpy as np

random.seed(0)
def rand(a, b):
    return (b - a) * random.random() + a
def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))
def logistic_derivative(x):
    return x * (1 - x)
def tanh(x):
    res = 2 * logistic(2*x) - 1
    return res
def tanh_derivative(x):
    return 1 - tanh(x)*tanh(x)
def relu(x):
    return max(x, 0)
def relu_derivative(x):
    return 1 if (x >0) else 0


def sigmoid(x, mode):
    if mode == 1:
        return logistic(x)
    elif mode == 2:
        return tanh(x)
    elif mode == 3:
        return relu(x)
    else:
        print("mode does not exist!")
        return 0
def sigmoid_derivative(x, mode):
    if mode ==1:
        return logistic_derivative(x)
    elif mode ==2:
        return tanh_derivative(x)
    elif mode == 3:
        return relu_derivative(x)
    else:
        print("mode doex not exist")

def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        i = i
        mat.append([fill] * n)
    return mat

class BPNN:
    def __init__(self, cases, expects, nh, mode, nlayer, limits = 100000, learn = 0.05):
        if len(cases) < 1:
            print("case can not be empty!!")
            exit(0)
        if len(cases) != len(expects):
            print("cases and expects not matching !")
            exit(0)
        if limits <= 0 :
            print("limits should be a positive number!")
            exit(0)
        if learn<0 or learn >0.1:
            print("learning-rate should be between [0, 1]!")
            exit(0)
        self.input_n = len(cases[0]) + 1 #include bias
        self.hidden_n = nh + 1#include bias
        self.output_n = len(cases[0])
        self.mode = mode
        self.layer_n = nlayer
        self.cases = cases
        self.expects = expects
        self.limits = limits
        self.learn = learn
        #init datas
        self.input_datas = [1.0] * self.input_n
        self.hidden_datas = [1.0] * self.hidden_n
        self.output_datas = [1.0] * self.output_n
        self.ih_weights = []
        self.ho_weights = []
        #init weights
        for i in range(self.input_n):
            tmp = []
            for h in range(self.hidden_n - 1):
                tmp.append(rand(-0.2, 0.2))
                self.ih_weights.append(tmp)
        for h in range(self.hidden_n):
            tmp = []
            for o in range(self.output_n):
                tmp.append(rand(-0.2, 0.2))
                self.ho_weights.append(tmp)
    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_datas[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n - 1):
            sum = 0.0
            for i in range(self.input_n):
                sum += self.input_datas[i] * self.ih_weights[i][j]
            self.hidden_datas[j] = sigmoid(sum, self.mode)
        # activate output layer
        for k in range(self.output_n):
            sum = 0.0
            for j in range(self.hidden_n - 1):
                sum += self.hidden_datas[j] * self.ho_weights[j][k]
            self.output_datas[k] = sigmoid(sum, self.mode)
        return self.output_datas
    def back_propagate(self, case, expect, learn ):
        #feed forward
        self.predict(case)
        #get output layer error
        output_errors = [0.0] * self.output_n
        for i in range(self.output_n):
            error = expect[i] - self.output_datas[i]
            output_errors[i] = sigmoid_derivative(self.output_datas[i], self.mode)*error
        #get hidden layer error
        hidden_errors = [0.0]*(self.hidden_n - 1)
        for i in range(self.hidden_n - 1):
            error = 0.0
            for j in range(self.output_n):
                error += output_errors[j] * self.ho_weights[i][j]
            hidden_errors[i] = sigmoid_derivative(self.hidden_datas[i], self.mode) * error
        #update ho_weights
        for i in range(self.hidden_n - 1):
            for j in range(self.output_n):
                change = output_errors[j] * self.hidden_datas[i]
                self.ho_weights[i][j] += learn * change
        #update ih_weights
        for i in range(self.input_n):
            for j in range(self.hidden_n - 1):
                change = hidden_errors[j] * self.input_datas[i] 
                self.ih_weights[i][j] += learn * change
    def train(self, cases, expects):
        res = []
        for i in range(len(cases)):
            for j in range(self.limits):
                self.back_propagate(cases[i], expects[i], self.learn)
            #print("input:", self.input_datas)
            #print("expect:", expects[i])
            #print("result:", self.output_datas)
            tmp = copy.deepcopy(self.output_datas)
            res.append(tmp)
        return res
                
    def test(self):
        cases = copy.deepcopy(self.cases)
        origin = copy.deepcopy(cases)
        expects = self.expects
        flag = self.layer_n
        while flag != 0 :
            flag -= 1
            res = self.train(cases, expects)
            #update the cases
            cases = copy.deepcopy(res)
            #print("new cases:", cases)
        print("cases:",origin)
        print("expect:", expects)
        for i in range(len(res)):
            print("result:", res[i])
        return res
def main():
    #init test data
    cases = [
            [0.5, 0.9, 0.1],
            [0.1, 0.7, 0.4],
            [0.99, 0.11, 0.3]
    ]
    expects =  [
        [0.1, 0.291, 0.7],
        [0.6, 0.8, 0.1],
        [0.1, 0.8, 0.81]
    ]
    number_of_neurons = 5
    function_mode = 3
    n_layers = 3
    limits = 100

    #init the main class with default learning-rate=0.05
    bn = BPNN(cases, expects, number_of_neurons, function_mode, n_layers, limits)
    bn.test()

if __name__ == '__main__':
    main()
