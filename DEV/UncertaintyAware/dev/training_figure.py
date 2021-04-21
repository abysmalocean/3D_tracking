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


testing_data     = pickle.load(open("../data_test/training.pickle", 'rb'))
validation_data  = pickle.load(open("../data_test/validation.pickle", 'rb'))
test_data        = pickle.load(open("../data_test/testing.pickle", 'rb'))

solver = EKF_solver.EKFSolver(
    data        = test_data,
    val_data    = testing_data, 
    test_data   = test_data, 
    update_rule = config.config["update_rule"],
    lr_decay    = config.config["lr_decay"],
    batch_size  = config.config["batch_size"],
    num_epochs  = config.config["num_epochs"], 
    print_every = config.config["print_every"], 
    verbose     = config.config["verbose"],
    lr          = config.config["lr"]
)

solver.training()

with open("../data_test/RMS_training", 'wb') as f:
            pickle.dump(solver, f)