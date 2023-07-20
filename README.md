# Machine Learning for Interconnect Network Traffic Forecasting: Investigation and Exploitation. (SIGSIM-PADS 2023).
This repo is the official Pytorch implementation of paper: "[Machine Learning for Interconnect Network Traffic Forecasting: Investigation and Exploitation](https://xiongxiaoxu.github.io/publications/PADS23_ML.pdf)".

# Abstract
Interconnect networks play a key role in high-performance computing (HPC) systems. Parallel discrete event simulation (PDES) has been a long-standing pillar for studying large-scale networking systems by replicating the real-world behaviors of HPC facilities. However, the simulation requirements and computational complexity of PDES are growing at an intractable rate. An active research topic is to build a surrogate-ready PDES framework where an accurate surrogate model built on machine learning can be used to forecast network traffic for improving PDES. In this paper, we make the first attempt to introduce two representative time series methods, the Autoregressive Integrated Moving Average (ARIMA) and the Adaptive Long Short-Term Memory (ADP-LSTM), to forecast the traffic in interconnect networks, using the Dragonfly system as a representative example. The proposed ADP-LSTM can efficiently adapt to the ever-changing network traffic, facilitating the forecasting capability for intricate network traffic, by incorporating a novel online learning strategy. Our preliminary analysis demonstrates promising results and shows that ADP-LSTM can consistently outperform ARIMA with significantly less time overhead.




# Environment
* python            3.8.13
* numpy             1.21.5
* pandas            1.4.3
* scikit-learn      1.1.2
* scipy             1.7.3
* statsmodels       0.13.2
* torch             1.9.0+cu111

# Code
The description of the code files.
* readdata.py: Read simulated traffic data from the dally
* model.py: The implementation ADP-LSTM model class
* algorithm.py: The implementation of ADP-LSTM and ARIMA algorithms
* run.py: run code
* utilis.py: utiliy functions

#
