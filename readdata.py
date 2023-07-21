# python readdata.py --placement cont-adp
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='ReadData')
parser.add_argument('--placement', type=str, choices=['cont-adp', 'rand_node-adp'])

args = parser.parse_args()

dic_PlacementtoPath = {'cont-adp': 'cont-par-1d-jacobi_MILC-12339-1675222655', 
                        'rand_node-adp': 'rand_node0-par-1d-jacobi_MILC-22287-1675223028'}
placement_filename = dic_PlacementtoPath[args.placement]

df = pd.read_csv('dally/' + placement_filename + '/dragonfly-router-traffic-sample', sep=' ')
df.index = range(64800)

print(df)

for i in range(36):
    for j in range(7):
        valid_data_RP = df.iloc[i*1800 : (i + 1)*1800, j + 4]
        valid_data_RP.index = range(1800)

        RID = str(i)
        dic_ColtoPid = {0: 'L0', 1: 'L1', 2: 'L2', 3: 'G0', 4: 'G1', 5: 'T0', 6: 'T1'}
        PID = dic_ColtoPid[j]
        RIDPID = 'R' + RID + PID

        folder_path = 'Data/' + args.placement
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        valid_data_RP.to_csv(folder_path + '/' + RIDPID + '.csv', index=False)