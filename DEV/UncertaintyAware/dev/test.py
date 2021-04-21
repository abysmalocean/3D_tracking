import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Load the data
from loadData import raw
from EKF import EKF


## Load the KITTI data
# Change this to the directory where you store KITTI data
basedir = '../data_test/'

# Specify the dataset to load
date = '2011_09_26'
drive = '0022'

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.raw(basedir, date, drive)
dataset      = raw(basedir, date, drive, frames=range(0, 8000, 50))

print("Total Lenght of the time data ", len(dataset))
EKF_filter = EKF.EKF()
EKF_filter.load_data_set(dataset)
EKF_filter.add_some_noise()
EKF_filter.run_kf()
EKF_filter.smoother()

EKF_filter.plot_overview()

