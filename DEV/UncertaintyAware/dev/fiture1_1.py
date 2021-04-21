import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array


# Load the data
from loadData import raw
from EKF import EKF

import pickle
from tqdm import tqdm

from EKFNet import EKFNet_layers 
from EKFNet import EKFNet 
from EKFNet import EKF_solver
from EKFNet import config

data = pickle.load(open("../data_test/training.pickle", 'rb'))

dataset = data[0]['dataset']
EKF_filter = EKFNet.EKFNet()
EKF_filter.load_data_set(dataset)
#EKF_filter.add_some_noise()



parameters = {
            # Measurement Noise Sigma
            "sigma_GPS_x"        : 100.0,
            "sigma_GPS_y"        : 100.0,
            "sigma_GPS_h"        : (0.2)**2,

            # Process Noise Sigmas
            "max_acc"            : 1 ** 2,
            "max_sttering_rate"  : 0.1 ** 2,
            "sigma_x"            : 1.0,
            "sigma_y"            : 1.0,
            "sigma_h"            : 1.0,
            "sigma_v"            : 1.0,
            "sigma_p"            : 1.0,
        }
EKF_filter.set_paramters(parameters)   
EKF_filter.run_EKF_NET_forward()
EKF_filter.generate_grad()
loss = EKF_filter.totalLoss()
dR, dQ_acc, dQ_other = EKF_filter.run_backward()
print("dR is \n", dR)
print("dQ_acc is \n", dQ_acc)
print("dQ_other is \n", dQ_other)
#EKF_filter.plot_overview()
'''
4.342906865842658
          sigma_GPS_x 4.342906865842658
%s  : %f  sigma_GPS_y 4.236060598127525
%s  : %f  sigma_GPS_h 0.21190520630030937
%s  : %f  max_acc 5.000000000000001
%s  : %f  max_sttering_rate 0.3401658888943914
%s  : %f  sigma_x 0.01210004857369823

'''
parameters2 = {
            # Measurement Noise Sigma
            "sigma_GPS_x"        : 4.342906865842658,
            "sigma_GPS_y"        : 4.236060598127525,
            "sigma_GPS_h"        : 0.21190520630030937,

            # Process Noise Sigmas
            "max_acc"            : 5,
            "max_sttering_rate"  : 0.34016,
            "sigma_x"            : 1.0,
            "sigma_y"            : 1.0,
            "sigma_h"            : 1.0,
            "sigma_v"            : 1.0,
            "sigma_p"            : 1.0,
        }
dataset = data[0]['dataset']
EKF_filter2 = EKFNet.EKFNet()
EKF_filter2.load_data_set(dataset)
EKF_filter2.set_paramters(parameters2)   
EKF_filter2.run_EKF_NET_forward()

for k, v in loss.items():
    print(k, " : " ,v)
#EKF_filter.plot_overview()

fig = plt.figure(figsize=(12, 16))
ax = fig.add_subplot(111)

loc = np.array(EKF_filter.x)
plt.scatter(loc[:,0], loc[:,1], c='y', s=50, marker="1", 
        label="{}".format("EKF Before Learning"))
plt.plot(loc[:,0], loc[:,1], c='y')

loc = np.array(EKF_filter.bias_locs)
plt.scatter(loc[:,0], loc[:,1], c='r', s=20, marker="x", 
        label="{}".format("Measurements"))

loc = EKF_filter.datasets.afterSmooth.x[0::50]
loc = np.array(loc[0:len(EKF_filter.locs)])
plt.scatter(loc[:,0], loc[:,1], c='k', s=50, marker="2", 
        label="{}".format("Ground Truth"))
plt.plot(loc[:,0], loc[:,1], c='k')


loc = np.array(EKF_filter2.x)
plt.scatter(loc[:,0], loc[:,1], c='b', s=50, marker="+", 
        label="{}".format("EKF After Learning"))
plt.plot(loc[:,0], loc[:,1], c='b')

plt.title("Vehicle trajectory before and after learning", fontsize = 18)
plt.ylabel('x(m)', fontsize = 15)
plt.xlabel('y(m)', fontsize = 15)
ax.axis('equal')
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

ax.legend(fontsize = 15)
#plt.tight_layout()
plt.show()



