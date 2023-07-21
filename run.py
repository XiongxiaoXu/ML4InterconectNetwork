# python run.py --RIDPID R0L0 --placement cont-adp --method ADPLSTM --backward_window_size 13 --forward_window_size 1
# python run.py --RIDPID R0L0 --placement cont-adp --method ARIMA --backward_window_size 200 --forward_window_size 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import *
from utilis import *
import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Network Prediction")
parser.add_argument('--RIDPID', type=str, help='port')
parser.add_argument('--placement', type=str, choices=['cont-adp', 'rand_node-adp'])
parser.add_argument('--num_lagstep', type=int, default=1, help='the number of lag step')
parser.add_argument('--backward_window_size', type=int, help='setting to predict')
parser.add_argument('--forward_window_size', type=int, help='strategy to forecast')
parser.add_argument('--method', type=str, choices = ['ADPLSTM', 'ARIMA'], help='method to predict')
parser.add_argument('--epoch', type=int, help='epochs to train')
parser.add_argument('--num_train', type=int, default=200, help='number of training data')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
parser.add_argument('--online_training_epoch', type=int, help='epochs to train')
parser.add_argument('--function_name', type=str, help='fucntion name')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train, test = load_traffic_data(args)
start = time.time()
if args.method == 'ARIMA':
    predict = myARIMA(train, test, args)
elif args.method == 'ADPLSTM':
    predict = ADPLSTM(train, test, args)
end = time.time()
elasped_time = end - start
MSE, MAE = evaluate(test, predict)
print('MSE: %.4f, MAE: %.4f, the elasped time: %.4f'%(MSE, MAE, elasped_time))
plot(train, test, predict, args)
