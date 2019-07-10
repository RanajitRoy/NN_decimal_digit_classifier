import numpy as np
from scipy import optimize as opt, special as sp
from matplotlib import pyplot as plt
import time


class TwoHLNN:
    def __init__(self, hl_1=500, hl_2=200, inp_sz=784, out_sz=10, initialize=0.5):
        self.hu_1 = hl_1
        self.hu_2 = hl_2
        self.inp_sz = inp_sz
        self.out_sz = out_sz
        self.Theta1 = (np.random.rand(hl_1, inp_sz+1) * (2 * initialize)) - (initialize * np.ones((hl_1, inp_sz+1)))
        self.Theta2 = (np.random.rand(hl_2, hl_1+1) * (2 * initialize)) - (initialize * np.ones((hl_2, hl_1+1)))
        self.Theta3 = (np.random.rand(out_sz, hl_2+1) * (2 * initialize)) - (initialize * np.ones((out_sz, hl_2+1)))
        self.accuracy = 0

    def train(self, train_data, train_labels, lr_rate=0.3, iter=150):
        m = train_data.shape[0]
        height = train_data.shape[1]
        width = train_data.shape[2]

        X = train_data.reshape(m, height*width)
        X = np.hstack((np.ones((m, 1)), X))
        y = train_labels
        y_v = np.zeros((m, 10))

        y_temp = y_v.tolist()
        for i in range(m):
            y_temp[i][y[i]%10] = 1
        y_v = np.array(y_temp)

        print("      ----Training----")
        strt = time.time()
        for i in range(iter):
            J = self.__grad_descent(X, y_v, lr_rate)
            print('Iteration:', i+1, 'Cost before ->', J, 'Time ->', time.time()-strt)
        end = time.time() - strt
        print('   Cost ->', self.__costfunction(X, y_v), 'Total time elapsed ->', end)

        # plt.matshow(X.reshape(m_examples, height, width)[2])
        # plt.show()

    def __grad_descent(self, X, y_v, lr_rate):
        m = X.shape[0]

        A2 = sp.expit(np.matmul(X, self.Theta1.transpose()))
        A2_bias = np.hstack((np.ones((m, 1)), A2))
        A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))
        A3_bias = np.hstack((np.ones((m, 1)), A3))
        A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))

        delta4 = A4 - y_v
        delta3 = np.dot(delta4, self.Theta3) * A3_bias * (1 - A3_bias)
        delta2 = np.dot(delta3[:, 1:], self.Theta2) * A2_bias * (1 - A2_bias)

        # print(delta2.shape, delta3.shape, delta4.shape)
        # print(self.Theta1.shape, self.Theta2.shape, self.Theta3.shape)

        # print(np.sum(delta2[:, 1:], axis=0).reshape(-1, 1).shape)
        # print(np.sum(X, axis=0).shape)

        Del1 = np.matmul(delta2[:, 1:].transpose(), X) / m
        Del2 = np.matmul(delta3[:, 1:].transpose(), A2_bias) / m
        Del3 = np.matmul(delta4.transpose(), A3_bias) / m

        # print(Del1.shape, Del2.shape, Del3.shape)

        self.Theta1 = self.Theta1 - (lr_rate * Del1)
        self.Theta2 = self.Theta2 - (lr_rate * Del2)
        self.Theta3 = self.Theta3 - (lr_rate * Del3)

        res = -(y_v * np.log(A4) + (1 - y_v) * np.log(1 - A4))
        res = np.sum(res) / m
        return res

    def test(self, test_data, test_labels):
        test_m = test_data.shape[0]
        height = test_data.shape[1]
        width = test_data.shape[2]

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

    def __costfunction(self, X, y_v):
        m = X.shape[0]

        A2 = sp.expit(np.matmul(X, self.Theta1.transpose()))
        A2_bias = np.hstack((np.ones((m, 1)), A2))
        A3 = sp.expit(np.matmul(A2_bias, self.Theta2.transpose()))
        A3_bias = np.hstack((np.ones((m, 1)), A3))
        A4 = sp.expit(np.matmul(A3_bias, self.Theta3.transpose()))

        res = -(y_v * np.log(A4) + (1 - y_v) * np.log(1 - A4))
        res = np.sum(res) / m
        return res
