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


data = pickle.load(open("../data_test/RMS_training_curve", 'rb'))


fig = plt.figure(figsize=(12, 16))
ax = fig.add_subplot(111)

size =  len(data.loss_history)

RMS_state = []
RMS_meas  = []

val_RMS_state = []
val_RMS_meas  = []

logLikelihood_state = []
logLikelihood_meas  = []


for i in range(0, size): 
    RMS_state.append(data.loss_history[i]["RMS_state"])
    RMS_meas.append(data.loss_history[i]["RMS_meas"] )
    logLikelihood_state.append(data.loss_history[i]["logLikelihood_state"] )
    logLikelihood_meas.append( data.loss_history[i]["logLikelihood_meas"]  )

    val_RMS_state.append(data.validation_history[i]["RMS_state"])
    val_RMS_meas.append( data.validation_history[i]["RMS_meas"] )

plt.plot(np.sqrt(RMS_state), c = 'b', label = "RMS (state)")
plt.plot(np.sqrt(RMS_meas),  c = 'r', label = "RMS (measurement)")
plt.plot(np.sqrt(val_RMS_state), linestyle='dashed', c = 'b', label = "RMS (state) Validation")
plt.plot(np.sqrt(val_RMS_meas), linestyle='dashed', c = 'r', label = "RMS (measurement), Validation")

plt.plot(logLikelihood_state, c = 'y', label = "-log likelihood (state)")
plt.plot(logLikelihood_meas, c = 'c',  label = "-log likelihood (measurement)")

#ax.axis('equal')
plt.ylabel('Training steps(m)', fontsize = 15)
plt.xlabel('RMS or Log Likelihood', fontsize = 15)
plt.title("Training and validatons loss scores", fontsize = 18)
ax.legend(fontsize = 15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
#plt.tight_layout()

plt.show()
