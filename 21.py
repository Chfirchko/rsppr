import numpy
import pandas as pd
import openpyxl
import re
import numpy as np
from numpy.random import randn


class RNN:
    # A many-to-one Vanilla Recurrent Neural Network.

    def __init__(self, input_size, output_size, hidden_size=6):
        # Weights
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        '''
        Perform a forward pass of the RNN using the given inputs.
        Returns the final output and hidden state.
        - inputs is an array of one hot vectors with shape (input_size, 1).
        '''
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}

        # Perform each step of the RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        # Compute the output
        y = self.Why @ h + self.by

        return y, h

    def backprop(self, d_y, learn_rate=2e-2):
        '''
        Perform a backward pass of the RNN.
        - d_y (dL/dy) has shape (output_size, 1).
        - learn_rate is a float.
        '''
        n = len(self.last_inputs)

        # Calculate dL/dWhy and dL/dby.
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Calculate dL/dh for the last h.
        # dL/dh = dL/dy * dy/dh
        d_h = self.Why.T @ d_y

        # Backpropagate through time.
        for t in reversed(range(n)):
            # An intermediate value: dL/dh * (1 - h^2)
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
            temp = np.reshape(6, 1)
            # dL/db = dL/dh * (1 - h^2)
            d_bh += temp

            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_Whh += temp @ self.last_hs[t].T
            temp = np.reshape(1, 10)
            # dL/dWxh = dL/dh * (1 - h^2) * x
            d_Wxh += temp @ np.array(self.last_inputs[t]).T

            # Next dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.Whh @ temp

        # Clip to prevent exploding gradients.
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Update weights and biases using gradient descent.
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


def Phrase_to_num(baza):
    tokens = []
    nums = []
    nums1 = []
    for i in baza:
        tokens.append(i[0].value.split())
    for i in tokens:
        for j in range(len(i)):
            i[j] = re.sub(r'[^\w\s]', '', i[j])
    for i in tokens:
        for j in range(len(i)):
            if i[j] not in nums:
                nums.append(i[j])
                nums1.append([len(nums1), i[j]])
    return nums1


def Tokenize(Phrase, phrases):
    batch = 10
    Bat = []
    for i in range(batch):
        Bat.append(0)
    for i in Phrase:
        i = re.sub(r'[^\w\s]', '', i)
    j = 0
    for i in phrases:
        if j >= len(Phrase):
            break
        if Phrase[j - 1] == i[1]:
            if len(Bat) <= 10:
                Bat[j] = i[0]
                j += 1
            else:
                break

    return Bat


bz = openpyxl.load_workbook('knowledge\\baza.xlsx', data_only=True)
baza = bz.active
phrases = Phrase_to_num(baza)
print(baza[2][0].value)
x_train = []
x_train1 = []
y_train = []
x_test = []
y_test = []
for i in range(1, baza.max_row - 2):
    y_train.append(int(baza[i + 1][1].value))
    x_train1.append(baza[i][0].value.split())
for i in x_train1:
    a = Tokenize(i, phrases)
    x_train.append(a)
print(len(x_train))
print(len(y_train))
x_test.append(Tokenize('Хохла', phrases))
y_test = [1]


# x_train1 = []
# for i in range(50, 40, -1):
#     y_train.append(int(baza[i][1].value))
#     x_train1.append(baza[i][0].value.split())
#
# for i in x_train1:
#     a = Tokenize(i, phrases)
#     x_train.append(a)
# for i in x_train:
#     print(i)
# print(y_train)
# print(len(x_train), len(y_train))

def softmax(xs):
    # Применение функции Softmax для входного массива
    return np.exp(xs) / sum(np.exp(xs))


rnn = RNN(10, 6)
zxc = 0
for x in x_train:
    target = y_train[zxc]
    out, h = rnn.forward([x])
    probs = softmax(out)
    print(probs)
    d_L_d_y = probs
    d_L_d_y[target] -= 1
    rnn.backprop(d_L_d_y)
