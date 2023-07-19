import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os

def load_traffic_data(args):
    df = pd.read_csv('Data/' + args.placement + '/' + args.RIDPID + '.csv')
    PID = args.RIDPID[-2:]
    dic_PIDtoFullName = {'L0': 'Local0', 'L1': 'Local1', 'L2': 'Local2', 'G0': 'Global0', 'G1': 'Global1', 'T0': 'Terminal0', 'T1': 'Terminal1'}
    data = df[dic_PIDtoFullName[PID]]
    data = normalize_minmax(data)
    train = data[0:args.num_train]

    # Delete unsteady data after 1380/1430 timepoint
    if args.placement == 'cont-adp':
        test = data[args.num_train:args.num_train + 1380]
    elif args.placement == 'rand_node-adp':
        test = data[args.num_train:args.num_train + 1430]

    return train, test

def normalize_minmax(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.values.reshape(-1,1))
    data = pd.Series(np.squeeze(data), index=range(len(data)))

    return data

def evaluate(y, pred_y):
    MSE = np.mean((y - pred_y) ** 2)
    MAE = np.mean(np.abs(y - pred_y))

    return MSE, MAE

def create_inout_sequences(input_data, backward_window_size, forward_window_size):
    inout_seq = []
    L = len(input_data)
    for i in range(L - backward_window_size - forward_window_size + 1):
        train_seq = input_data[i:i + backward_window_size]
        train_label = input_data[i + backward_window_size:i + backward_window_size + forward_window_size]
        inout_seq.append((train_seq ,train_label))

    return inout_seq

def plot(train, test, predict, args):
    plt.figure(figsize = (12,5), dpi = 100)
    size = 15
    font_label = {'family': 'Verdana', 'weight': 'normal', 'size': size}
    font_legend = {'family': 'Verdana', 'weight': 'normal', 'size': size}

    plt.ylim(-0.1, 1.3)
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(predict, label='Forecast')

    plt.xlabel('Time', fontdict=font_label)
    plt.ylabel('Traffic', fontdict=font_label)
    plt.tick_params(axis='x', labelsize=size)
    plt.tick_params(axis='y', labelsize=size)

    plt.legend(prop=font_legend, loc='best', mode="expand", ncol=3)
    folder_path = 'Figure/' + args.placement + '/' + args.RIDPID
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = folder_path + '/' + args.RIDPID + '_' + args.function_name + '.png'
    plt.savefig(path)
