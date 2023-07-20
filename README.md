# Machine Learning for Interconnect Network Traffic Forecasting: Investigation and Exploitation. (SIGSIM-PADS 2023).
This repo is the official Pytorch implementation of paper: "[Machine Learning for Interconnect Network Traffic Forecasting: Investigation and Exploitation](https://xiongxiaoxu.github.io/publications/PADS23_ML.pdf)".

## Abstract
Interconnect networks play a key role in high-performance computing (HPC) systems. Parallel discrete event simulation (PDES) has been a long-standing pillar for studying large-scale networking systems by replicating the real-world behaviors of HPC facilities. However, the simulation requirements and computational complexity of PDES are growing at an intractable rate. An active research topic is to build a surrogate-ready PDES framework where an accurate surrogate model built on machine learning can be used to forecast network traffic for improving PDES. In this paper, we make the first attempt to introduce two representative time series methods, the Autoregressive Integrated Moving Average (ARIMA) and the Adaptive Long Short-Term Memory (ADP-LSTM), to forecast the traffic in interconnect networks, using the Dragonfly system as a representative example. The proposed ADP-LSTM can efficiently adapt to the ever-changing network traffic, facilitating the forecasting capability for intricate network traffic, by incorporating a novel online learning strategy. Our preliminary analysis demonstrates promising results and shows that ADP-LSTM can consistently outperform ARIMA with significantly less time overhead.

## Network Topology
The Dragonfly network (see the following figure) has a hierarchical design, consisting of the all-to-all inter-group connection and intra-group connection. In the network, the 72 compute nodes and 36 routers are averagely divided into 9 groups, which are all-to-all connected. Within a group, the routers are also all-to-all connected. Each router has 7 ports, including 2 terminal ports, 3 local ports, and 2 global ports. The total number of ports is 252.
![image](https://github.com/XiongxiaoXu/ML-SurrogateModel/assets/34889516/0801eaab-bcd6-4365-9131-2b1a2abb9b04)

## Environment
* python            3.8.13
* numpy             1.21.5
* pandas            1.4.3
* scikit-learn      1.1.2
* scipy             1.7.3
* statsmodels       0.13.2
* torch             1.9.0+cu111

## Code
The description of the code files.
* readdata.py: Read simulated traffic data
* model.py: The implementation ADP-LSTM model class
* algorithm.py: The implementation of ADP-LSTM and ARIMA algorithms
* run.py: run code
* utilis.py: utiliy functions

## Getting Started
### Read Data

First, you can read data from the raw simulated traffic data.

`python readdata.py --placement cont-adp`

### Run the Algorithms

To get the result of ADP-LSTM, run the following script:

`python run.py --RIDPID R0L0 --placement cont-adp --method ADPLSTM --backward_window_size 13 --forward_window_size 1`


To get the result of ARIMA, run the following script:

`python run.py --RIDPID R0L0 --placement cont-adp --method ARIMA --backward_window_size 200 --forward_window_size 1`

### Sentistity Analysis
To investigate the effect of the delay steps on forecast accuray, e.g., set the number of delay stpes as 3, run the following script:

`python run.py --RIDPID R0L0 --placement cont-adp --method ARIMA --backward_window_size 200 --forward_window_size 1 --num_lagstep 3`

## Experimental Results
![image](https://github.com/XiongxiaoXu/ML-SurrogateModel/assets/34889516/02bf5da7-80df-4493-8c6b-c716bca13334)

## Sentistity Analysis
![image](https://github.com/XiongxiaoXu/ML-SurrogateModel/assets/34889516/d49e53db-8d3c-45da-859c-b0f8e6e0e2e8)

![image](https://github.com/XiongxiaoXu/ML-SurrogateModel/assets/34889516/c1a696fb-7135-4a9b-b04c-920ea2b410c5)

## Cite
If you find this repository useful for your work, please consider citing it as follows:



