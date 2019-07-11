import numpy as np
from scipy import optimize as opt, special as sp
from matplotlib import pyplot as plt
import img_to_np as img
import math
import time


class TwoHLNN:
    def __init__(self, hl_1=500, hl_2=200, inp_sz=784, out_sz=10, initialize=0.5):
        self.hu_1 = hl_1
        self.hu_2 = hl_2
        self.inp_sz = inp_sz
        self.out_sz = out_sz
        # Random initialization
        self.Theta1 = (np.random.rand(hl_1, inp_sz+1) * (2 * initialize)) - (initialize * np.ones((hl_1, inp_sz+1)))
        self.Theta2 = (np.random.rand(hl_2, hl_1+1) * (2 * initialize)) - (initialize * np.ones((hl_2, hl_1+1)))
        self.Theta3 = (np.random.rand(out_sz, hl_2+1) * (2 * initialize)) - (initialize * np.ones((out_sz, hl_2+1)))
        self.accuracy = 0  # setting accuracy = 0
        self.layers = 2

    def train(self, train_data, train_labels, lr_rate=0.3, maxiter=150, lmda=0):
        m = train_data.shape[0]
        height = train_data.shape[1]  # height of image
        width = train_data.shape[2]  # width of image

        if not height * width == self.inp_sz:
            print("Invalid model input size!!")
            return

        X = train_data.reshape(m, height*width)  # creating 2D array
        X = np.hstack((np.ones((m, 1)), X))  # adding bias
        y = train_labels
        y_v = np.zeros((m, 10))

        # y_v is the vector implementation of y
        y_temp = y_v.tolist()
        for i in range(m):
            y_temp[i][y[i]%10] = 1
        y_v = np.array(y_temp)

        print("           ----Training----")
        strt = time.time()
        for i in range(maxiter):  # training for maxiter
            J = self.__grad_descent_reg(X, y_v, lr_rate, lmda)  # implementing gradient descent
            print('Iteration:', i + 1, 'Cost before ->', J, 'Time ->', time.time() - strt)
        end = time.time() - strt
        print('   Cost ->', self.__costfunction_reg(X, y_v, lmda), 'Total time elapsed ->', end)

    # def __grad_descent(self, X, y_v, lr_rate):
    #     m = X.shape[0]
    #
    #     A2 = sp.expit(np.matmul(X, self.Theta1.transpose()))
    #     A2_bias = np.hstack((np.ones((m, 1)), A2))
    #     A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))
    #     A3_bias = np.hstack((np.ones((m, 1)), A3))
    #     A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))
    #
    #     delta4 = A4 - y_v
    #     delta3 = np.matmul(delta4, self.Theta3) * A3_bias * (1 - A3_bias)
    #     delta2 = np.matmul(delta3[:, 1:], self.Theta2) * A2_bias * (1 - A2_bias)
    #
    #     Del1 = np.matmul(delta2[:, 1:].transpose(), X) / m
    #     Del2 = np.matmul(delta3[:, 1:].transpose(), A2_bias) / m
    #     Del3 = np.matmul(delta4.transpose(), A3_bias) / m
    #
    #     self.Theta1 = self.Theta1 - (lr_rate * Del1)
    #     self.Theta2 = self.Theta2 - (lr_rate * Del2)
    #     self.Theta3 = self.Theta3 - (lr_rate * Del3)
    #
    #     res = -(y_v * np.log(A4) + (1 - y_v) * np.log(1 - A4))
    #     res = np.sum(res) / m
    #     return res

    def __grad_descent_reg(self, X, y_v, lr_rate, lmda):
        m = X.shape[0]  # no. of training sets

        A2 = sp.expit(np.matmul(X, self.Theta1.transpose()))  # values at layer 1
        A2_bias = np.hstack((np.ones((m, 1)), A2))  # adding bias
        A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))  # values at layer 2
        A3_bias = np.hstack((np.ones((m, 1)), A3))  # adding bias
        A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))  # values at class layer

        delta4 = A4 - y_v
        delta3 = np.matmul(delta4, self.Theta3) * A3_bias * (1 - A3_bias)
        delta2 = np.matmul(delta3[:, 1:], self.Theta2) * A2_bias * (1 - A2_bias)

        # gradients
        Del1 = np.matmul(delta2[:, 1:].transpose(), X)
        Del1_reg = (Del1 + (np.hstack((np.zeros((self.hu_1, 1)), self.Theta1[:, 1:])) * lmda)) / m
        Del2 = np.matmul(delta3[:, 1:].transpose(), A2_bias) / m
        Del2_reg = (Del2 + (np.hstack((np.zeros((self.hu_2, 1)), self.Theta2[:, 1:])) * lmda)) / m
        Del3 = np.matmul(delta4.transpose(), A3_bias) / m
        Del3_reg = (Del3 + (np.hstack((np.zeros((self.out_sz, 1)), self.Theta3[:, 1:])) * lmda)) / m

        # subtracting gradients
        self.Theta1 = self.Theta1 - (lr_rate * Del1_reg)
        self.Theta2 = self.Theta2 - (lr_rate * Del2_reg)
        self.Theta3 = self.Theta3 - (lr_rate * Del3_reg)

        # calculating cost function
        res = -(y_v * np.log(A4) + (1 - y_v) * np.log(1 - A4))
        res = np.sum(res) / m
        reg = np.sum(self.Theta1[:, 1:] * self.Theta1[:, 1:])
        reg = reg + np.sum(self.Theta2[:, 1:] * self.Theta2[:, 1:])
        reg = reg + np.sum(self.Theta3[:, 1:] * self.Theta3[:, 1:])
        res = res + (reg * lmda * 0.5) / m
        return res

    def test(self, test_data, test_labels):
        test_m = test_data.shape[0]
        height = test_data.shape[1]
        width = test_data.shape[2]

        if not height * width == self.inp_sz:
            print("Invalid model input size!!")
            return

        test_mat = test_data.reshape(test_m, height * width)
        test_mat = np.hstack((np.ones((test_m, 1)), test_mat))

        A2 = sp.expit(np.matmul(test_mat, self.Theta1.transpose()))
        A2_bias = np.hstack((np.ones((test_m, 1)), A2))
        A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))
        A3_bias = np.hstack((np.ones((test_m, 1)), A3))
        A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))

        res = A4.argmax(axis=1)

        count = 0
        for i in range(test_m):
            if res[i] == test_labels[i]:
                count+=1
        self.accuracy = count * 100 / test_m

        print('Accuracy ->', self.accuracy)
        # print(res[:20], test_labels[:20])
        # plt.matshow(test_data[18])
        # plt.show()
        return self.accuracy

    # def __costfunction(self, X, y_v):
    #     m = X.shape[0]
    #
    #     A2 = sp.expit(np.matmul(X, self.Theta1.transpose()))
    #     A2_bias = np.hstack((np.ones((m, 1)), A2))
    #     A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))
    #     A3_bias = np.hstack((np.ones((m, 1)), A3))
    #     A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))
    #
    #     res = -(y_v * np.log(A4) + (1 - y_v) * np.log(1 - A4))
    #     res = np.sum(res) / m
    #     return res

    def __costfunction_reg(self, X, y_v, lmda):
        m = X.shape[0]

        A2 = sp.expit(np.matmul(X, self.Theta1.transpose()))
        A2_bias = np.hstack((np.ones((m, 1)), A2))
        A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))
        A3_bias = np.hstack((np.ones((m, 1)), A3))
        A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))

        res = -(y_v * np.log(A4) + (1 - y_v) * np.log(1 - A4))
        res = (np.sum(res)) / m
        reg = np.sum(self.Theta1[:, 1:] * self.Theta1[:, 1:])
        reg = reg + np.sum(self.Theta2[:, 1:] * self.Theta2[:, 1:])
        reg = reg + np.sum(self.Theta3[:, 1:] * self.Theta3[:, 1:])
        res = res + (reg * lmda * 0.5) / m
        return res

    def predict(self):
        if not math.sqrt(self.inp_sz)**2 == self.inp_sz:
            print("Model input not perfect square!!")
            return

        arr = img.image_to_np(math.sqrt(int(self.inp_sz)))
        arr = np.hstack((np.array(1).reshape(1, 1), arr.reshape(1, self.inp_sz)))

        A2 = sp.expit(np.matmul(arr, self.Theta1.T))
        A2_bias = np.hstack((np.array(1).reshape(1, 1), A2))
        A3 = sp.expit(np.matmul(A2_bias, self.Theta2.T))
        A3_bias = np.hstack((np.array(1).reshape(1, 1), A3))
        A4 = sp.expit(np.matmul(A3_bias, self.Theta3.T))

        return A4.argmax(axis=1)[0]


class ThreeHLNN:
    def __init__(self, hl_1=500, hl_2=200, hl_3=80, inp_sz=784, out_sz=10, initialize=0.5):
        self.hu_1 = hl_1
        self.hu_2 = hl_2
        self.hu_3 = hl_3
        self.inp_sz = inp_sz
        self.out_sz = out_sz
        # Random initialization
        self.Theta1 = (np.random.rand(hl_1, inp_sz+1) * (2 * initialize)) - (initialize * np.ones((hl_1, inp_sz+1)))
        self.Theta2 = (np.random.rand(hl_2, hl_1+1) * (2 * initialize)) - (initialize * np.ones((hl_2, hl_1+1)))
        self.Theta3 = (np.random.rand(hl_3, hl_2+1) * (2 * initialize)) - (initialize * np.ones((hl_3, hl_2+1)))
        self.Theta4 = (np.random.rand(out_sz, hl_3 + 1) * (2 * initialize)) - (initialize * np.ones((out_sz, hl_3 + 1)))
        self.accuracy = 0  # setting accuracy = 0
        self.layers = 3

    def train(self, train_data, train_labels, lr_rate=0.3, maxiter=150, lmda=0):
        m = train_data.shape[0]
        height = train_data.shape[1]  # height of image
        width = train_data.shape[2]  # width of image

        if not height * width == self.inp_sz:
            print("Invalid model input size!!")
            return

        X = train_data.reshape(m, height*width)  # creating 2D array
        X = np.hstack((np.ones((m, 1)), X))  # adding bias
        y = train_labels
        y_v = np.zeros((m, 10))

        # y_v is the vector implementation of y
        y_temp = y_v.tolist()
        for i in range(m):
            y_temp[i][y[i]%10] = 1
        y_v = np.array(y_temp)

        print("           ----Training----")
        strt = time.time()
        for i in range(maxiter):  # training for maxiter
            J = self.__grad_descent_reg(X, y_v, lr_rate, lmda)  # implementing gradient descent
            print('Iteration:', i + 1, 'Cost before ->', J, 'Time ->', time.time() - strt)
        end = time.time() - strt
        print('   Cost ->', self.__costfunction_reg(X, y_v, lmda), 'Total time elapsed ->', end)

    # def __grad_descent(self, X, y_v, lr_rate):
    #     m = X.shape[0]
    #
    #     A2 = sp.expit(np.matmul(X, self.Theta1.transpose()))
    #     A2_bias = np.hstack((np.ones((m, 1)), A2))
    #     A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))
    #     A3_bias = np.hstack((np.ones((m, 1)), A3))
    #     A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))
    #
    #     delta4 = A4 - y_v
    #     delta3 = np.matmul(delta4, self.Theta3) * A3_bias * (1 - A3_bias)
    #     delta2 = np.matmul(delta3[:, 1:], self.Theta2) * A2_bias * (1 - A2_bias)
    #
    #     Del1 = np.matmul(delta2[:, 1:].transpose(), X) / m
    #     Del2 = np.matmul(delta3[:, 1:].transpose(), A2_bias) / m
    #     Del3 = np.matmul(delta4.transpose(), A3_bias) / m
    #
    #     self.Theta1 = self.Theta1 - (lr_rate * Del1)
    #     self.Theta2 = self.Theta2 - (lr_rate * Del2)
    #     self.Theta3 = self.Theta3 - (lr_rate * Del3)
    #
    #     res = -(y_v * np.log(A4) + (1 - y_v) * np.log(1 - A4))
    #     res = np.sum(res) / m
    #     return res

    def __grad_descent_reg(self, X, y_v, lr_rate, lmda):
        m = X.shape[0]  # no. of training sets

        A2 = sp.expit(np.matmul(X, self.Theta1.transpose()))  # values at layer 1
        A2_bias = np.hstack((np.ones((m, 1)), A2))  # adding bias
        A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))  # values at layer 2
        A3_bias = np.hstack((np.ones((m, 1)), A3))  # adding bias
        A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))  # values at class layer
        A4_bias = np.hstack((np.ones((m, 1)), A4))
        A5 = sp.expit(np.matmul(A4_bias, self.Theta4.transpose()))

        delta5 = A5 - y_v
        delta4 = np.matmul(delta5, self.Theta4) * A4_bias * (1 - A4_bias)
        delta3 = np.matmul(delta4[:, 1:], self.Theta3) * A3_bias * (1 - A3_bias)
        delta2 = np.matmul(delta3[:, 1:], self.Theta2) * A2_bias * (1 - A2_bias)

        # gradients
        Del1 = np.matmul(delta2[:, 1:].transpose(), X)
        Del1_reg = (Del1 + (np.hstack((np.zeros((self.hu_1, 1)), self.Theta1[:, 1:])) * lmda)) / m
        Del2 = np.matmul(delta3[:, 1:].transpose(), A2_bias) / m
        Del2_reg = (Del2 + (np.hstack((np.zeros((self.hu_2, 1)), self.Theta2[:, 1:])) * lmda)) / m
        Del3 = np.matmul(delta4[:, 1:].transpose(), A3_bias) / m
        Del3_reg = (Del3 + (np.hstack((np.zeros((self.hu_3, 1)), self.Theta3[:, 1:])) * lmda)) / m
        Del4 = np.matmul(delta5.transpose(), A4_bias) / m
        Del4_reg = (Del4 + (np.hstack((np.zeros((self.out_sz, 1)), self.Theta4[:, 1:])) * lmda)) / m

        # subtracting gradients
        self.Theta1 = self.Theta1 - (lr_rate * Del1_reg)
        self.Theta2 = self.Theta2 - (lr_rate * Del2_reg)
        self.Theta3 = self.Theta3 - (lr_rate * Del3_reg)
        self.Theta4 = self.Theta4 - (lr_rate * Del4_reg)

        # calculating cost function
        res = -(y_v * np.log(A5) + (1 - y_v) * np.log(1 - A5))
        res = np.sum(res) / m
        reg = np.sum(self.Theta1[:, 1:] * self.Theta1[:, 1:])
        reg = reg + np.sum(self.Theta2[:, 1:] * self.Theta2[:, 1:])
        reg = reg + np.sum(self.Theta3[:, 1:] * self.Theta3[:, 1:])
        reg = reg + np.sum(self.Theta4[:, 1:] * self.Theta4[:, 1:])
        res = res + (reg * lmda * 0.5) / m
        return res

    def test(self, test_data, test_labels):
        test_m = test_data.shape[0]
        height = test_data.shape[1]
        width = test_data.shape[2]

        if not height * width == self.inp_sz:
            print("Invalid model input size!!")
            return

        test_mat = test_data.reshape(test_m, height * width)
        test_mat = np.hstack((np.ones((test_m, 1)), test_mat))

        A2 = sp.expit(np.matmul(test_mat, self.Theta1.transpose()))
        A2_bias = np.hstack((np.ones((test_m, 1)), A2))
        A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))
        A3_bias = np.hstack((np.ones((test_m, 1)), A3))
        A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))
        A4_bias = np.hstack((np.ones((test_m, 1)), A4))
        A5 = sp.expit(np.matmul(A4_bias, self.Theta4.transpose()))

        res = A5.argmax(axis=1)

        count = 0
        for i in range(test_m):
            if res[i] == test_labels[i]:
                count+=1
        self.accuracy = count * 100 / test_m

        print('Accuracy ->', self.accuracy)
        # print(res[:20], test_labels[:20])
        # plt.matshow(test_data[18])
        # plt.show()
        return self.accuracy

    # def __costfunction(self, X, y_v):
    #     m = X.shape[0]
    #
    #     A2 = sp.expit(np.matmul(X, self.Theta1.transpose()))
    #     A2_bias = np.hstack((np.ones((m, 1)), A2))
    #     A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))
    #     A3_bias = np.hstack((np.ones((m, 1)), A3))
    #     A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))
    #
    #     res = -(y_v * np.log(A4) + (1 - y_v) * np.log(1 - A4))
    #     res = np.sum(res) / m
    #     return res

    def __costfunction_reg(self, X, y_v, lmda):
        m = X.shape[0]

        A2 = sp.expit(np.matmul(X, self.Theta1.transpose()))
        A2_bias = np.hstack((np.ones((m, 1)), A2))
        A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))
        A3_bias = np.hstack((np.ones((m, 1)), A3))
        A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))
        A4_bias = np.hstack((np.ones((m, 1)), A4))
        A5 = sp.expit(np.matmul(A4_bias, self.Theta4.transpose()))

        res = -(y_v * np.log(A5) + (1 - y_v) * np.log(1 - A5))
        res = (np.sum(res)) / m
        reg = np.sum(self.Theta1[:, 1:] * self.Theta1[:, 1:])
        reg = reg + np.sum(self.Theta2[:, 1:] * self.Theta2[:, 1:])
        reg = reg + np.sum(self.Theta3[:, 1:] * self.Theta3[:, 1:])
        reg = reg + np.sum(self.Theta4[:, 1:] * self.Theta4[:, 1:])
        res = res + (reg * lmda * 0.5) / m
        return res
