# Machine Learning for Interconnect Network Traffic Forecasting: Investigation and Exploitation. (ACM SIGSIM-PADS 2023).
The official code for paper: "[Machine Learning for Interconnect Network Traffic Forecasting: Investigation and Exploitation](https://xiongxiaoxu.github.io/publications/PADS23_ML.pdf)".

## Abstract
Interconnect networks play a key role in high-performance computing (HPC) systems. Parallel discrete event simulation (PDES) has been a long-standing pillar for studying large-scale networking systems by replicating the real-world behaviors of HPC facilities. However, the simulation requirements and computational complexity of PDES are growing at an intractable rate. An active research topic is to build a surrogate-ready PDES framework where an accurate surrogate model built on machine learning can be used to forecast network traffic for improving PDES. In this paper, we make the first attempt to introduce two representative time series methods, the Autoregressive Integrated Moving Average (ARIMA) and the Adaptive Long Short-Term Memory (ADP-LSTM), to forecast the traffic in interconnect networks, using the Dragonfly system as a representative example. The proposed ADP-LSTM can efficiently adapt to the ever-changing network traffic, facilitating the forecasting capability for intricate network traffic, by incorporating a novel online learning strategy. Our preliminary analysis demonstrates promising results and shows that ADP-LSTM can consistently outperform ARIMA with significantly less time overhead.

## Network Topology
The Dragonfly network (see the following figure) has a hierarchical design, consisting of the all-to-all inter-group connection and intra-group connection. In the network, the 72 compute nodes and 36 routers are averagely divided into 9 groups, which are all-to-all connected. Within a group, the routers are also all-to-all connected. Each router has 7 ports, including 2 terminal ports, 3 local ports, and 2 global ports. The total number of ports is 252.
![image](https://github.com/XiongxiaoXu/ML-SurrogateModel/assets/34889516/0801eaab-bcd6-4365-9131-2b1a2abb9b04)

## ML Surrogate Model
* ARIMA is a classical statistical time series forecasting method and has been widely adopted in traffic forecasting
* ADP-LSTM is a varaint of LSTM. It can dynamically update parameters and adapt to the evolving network traffic, by incorporating an online learning strategy into the standard offline learning 

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

`python run.py --RIDPID R0L0 --placement cont-adp --method ADPLSTM --backward_window_size 13 --forward_window_size 1 --num_lagstep 5`

## Experimental Results
![image](https://github.com/XiongxiaoXu/ML-SurrogateModel/assets/34889516/b6a1b346-0f44-4908-871e-7601a8e1c644)

## Sentistity Analysis
![image](https://github.com/XiongxiaoXu/ML-SurrogateModel/assets/34889516/1af1e324-296d-4447-b0a5-f5f2cc19e281)

![image](https://github.com/XiongxiaoXu/ML-SurrogateModel/assets/34889516/7b09dc26-90fd-4f1e-ab86-1d6bdd830b42)

## Cite
If you find this repository useful for your work, please consider citing the paper as follows:

```bibtex
@inproceedings{xu2023machine,
  title={Machine Learning for Interconnect Network Traffic Forecasting: Investigation and Exploitation},
  author={Xu, Xiongxiao and Wang, Xin and Cruz-Camacho, Elkin and D. Carothers, Christopher and A. Brown, Kevin and B. Ross, Robert and Lan, Zhiling and Shu, Kai},
  booktitle={Proceedings of the 2023 ACM SIGSIM Conference on Principles of Advanced Discrete Simulation},
  pages={133--137},
  year={2023}
}
```


