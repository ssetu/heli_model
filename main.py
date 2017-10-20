#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#csv parrsing includes
import pandas as pa
import numpy as np
from pathlib import Path
import os

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, inputT, hiddenT):
        combined = torch.cat((inputT, hiddenT), 1)
        hiddenT = self.i2h(combined)
        output = self.i2o(combined)
        return output, hiddenT

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

def train(outputT, inputT):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(inputT.size()[0]):
        output, hidden = rnn(inputT[i], hidden)
    loss = criterion(output, outputT.type(torch.FloatTensor))
    loss.backward()
    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def randomChoice(l):
    nRows = 10
    startRow = random.randint(2, len(l) - (nRows+1))
    inputVec = l.iloc[startRow:startRow+nRows]
    outputVec = l.iloc[startRow+nRows:startRow+nRows+1]
    inputVec = inputVec.values
    outputVec = outputVec.values
    inputVec = inputVec[:,[15,16,17,18,12, 13, 14, 3,4, 5, 6, 7]]
    inputVec = inputVec.reshape(nRows, 1, n_input)
    outputVec = outputVec[:,[3,4, 5, 12, 13, 14]]
    outputVec = outputVec.reshape(1, 1, n_output)
    return outputVec, inputVec

def randomTrainingExample(dataFrame):
    outputVec, inputVec = randomChoice(dataFrame)
    outputTensor = torch.from_numpy(outputVec)
    inputTensor = torch.from_numpy(inputVec)
    return outputTensor, inputTensor

#Get the data from csv file
fileName = '/home/sagar/logs/2017-09-29/11_47_27/250-300s.csv'
df = pa.read_csv(fileName)

#Set up the RNN
#inputs = [del_a, del_b, del_c, del_p, u, v, w, p, q, r, theta, phi]
n_input = 12
#hidden = [a, b, nu, beta_0]
n_hidden = 4
#output = [p_k+1, q_k+1 , r_k+1, u_k+1, v_k+1, w_k+1]
n_output = 6
rnn = RNN(n_input, n_hidden, n_output)
criterion = nn.L1Loss()
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

#Train the network
n_iters = 5000
print_every = 50
plot_every = 10

# Keep track of losses for plotting
current_loss = 0
all_losses = []

start = time.time()

for iter in range(1, n_iters + 1):
    outputTensor, inputTensor = randomTrainingExample(df)
    inT = Variable(inputTensor)
    outT = Variable(outputTensor)
    inT = inT.type(torch.FloatTensor)
    output, loss = train(outT[0], inT)
    current_loss += loss

    # Print iter number, loss
    if iter % print_every == 0:
        print('%d (%s) %.4f ' % (iter, timeSince(start), loss))
        print(output, outT)

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

#Plot the training progress
plt.figure()
plt.plot(all_losses)
