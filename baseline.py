from ast import Num
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import inspect
import torch
import torch.nn as nn
from utilis import *
from model import LSTM
import time

def ARIMA_alg(train, test, args):
    args.function_name = inspect.stack()[0][3]
    print(args.function_name)
    window = train[-args.backward_window_size:].tolist()
    predict = []
    real = window

    for i in range(len(test)):
        model = ARIMA(window, order=(13,1,5))
        model_fit = model.fit()
        predict = predict + model_fit.forecast(args.forward_window_size).tolist()
        real.append(test.iloc[i])
        
        if args.num_lagstep == 1:
            window = real[-args.backward_window_size:]
        else:
            if len(predict) < args.num_lagstep-1:
                window = real[-args.backward_window_size:-len(predict)] + predict[-len(predict):]
            else:
                window = real[-args.backward_window_size:-(args.num_lagstep-1)] + predict[-(args.num_lagstep-1):]

    predict = pd.Series(predict, index=test.index)

    return predict

def LSTM_alg(train, test, args):
    args.function_name = inspect.stack()[0][3]
    print(args.function_name)
    train_inout_seq = create_inout_sequences(torch.FloatTensor(train), args.backward_window_size, args.forward_window_size)
    input_size = 1
    lstm = LSTM(input_size, args.hidden_dim, args.forward_window_size)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    lstm.train()

    args.epoch = 50
    for i in range(args.epoch):
        epoch_loss = 0
        for seq, label in train_inout_seq:
            seq = seq.reshape(1, -1, 1)
            optimizer.zero_grad()
            y_pred = lstm(seq)
            loss = loss_function(y_pred, label)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss

    lstm.eval()

    window = train[-args.backward_window_size:].tolist()
    predict = []
    real = window

    for i in range(len(test)):
        seq = torch.FloatTensor(window).reshape(1, -1, 1)
        with torch.no_grad():
            predict = predict + [float(torch.squeeze(lstm(seq)))]
        real = real + test.iloc[i*args.forward_window_size:(i+1)*args.forward_window_size].tolist()
        if args.num_lagstep == 1:
            window = real[-args.backward_window_size:]
        else:
            if len(predict) < args.num_lagstep-1:
                window = real[-args.backward_window_size:-len(predict)] + predict[-len(predict):]
            else:
                window = real[-args.backward_window_size:-(args.num_lagstep-1)] + predict[-(args.num_lagstep-1):]

        args.online_training_epoch = 1
        train_inout_seq = create_inout_sequences(torch.FloatTensor(real[-(args.backward_window_size + args.forward_window_size):]), 
                                                    args.backward_window_size, args.forward_window_size)
        lstm.train()
        for i in range(args.online_training_epoch):
            for seq, label in train_inout_seq:
                seq = seq.reshape(1, -1, 1)
                optimizer.zero_grad()
                y_pred = lstm(seq)
                loss = loss_function(y_pred, label)
                loss.backward()
                optimizer.step()
        lstm.eval()

    predict = pd.Series(predict[-len(test):], index=test.index)

    return predict

