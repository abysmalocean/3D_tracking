from __future__ import print_function, division
from builtins import range
from builtins import object
from EKFNet import EKFNet 

import numpy as np
import random

from tqdm import tqdm

from cs231n import optim

class EKFSolver(object): 

    def __init__(self, data      = None, 
                       val_data  = None, 
                       test_data = None, **kwargs):

        #self.model = model
        self.training_data      = data
        self.validation_data    = val_data
        self.testing_data       = test_data
        # Unpack keyword arguments
        self.update_rule  = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay     = kwargs.pop('lr_decay', 1.0)
        self.batch_size   = kwargs.pop('batch_size', 10)
        self.num_epochs   = kwargs.pop('num_epochs', 100)
        self.lr           = kwargs.pop('lr', 1e-2)

        self.print_every  = kwargs.pop('print_every', 10)
        self.verbose      = kwargs.pop('verbose', True)
        self.update_rule  = getattr(optim, self.update_rule)
        self.training_len = len(self.training_data)
        self.testing_len  = len(self.testing_data)
        self.validation_len = len(self.validation_data)

        self.optim_config = {
            "learning_rate" : self.lr
        }

        self._reset()

        if self.verbose:
            print("Begin training, initial setup Finish total training ", 
                   self.training_len)

    def _reset(self):
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.validation_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.ramdom_index_training  = [i for i in range(0, self.training_len)]
        self.ramdom_index_validation = [i for i in range(0, self.validation_len)]
        
        random.shuffle(self.ramdom_index_training)
        random.shuffle(self.ramdom_index_validation)

        self.parameters = {
            # Measurement Noise Sigma
            "sigma_GPS_x"        : 100.0,
            "sigma_GPS_y"        : 100.0,
            "sigma_GPS_h"        : (2)**2,

            # Process Noise Sigmas
            "max_acc"            : 1 ** 2,
            "max_sttering_rate"  : 0.2,
            "sigma_x"            : 1.0,
            "sigma_y"            : 1.0,
            "sigma_h"            : 1.0,
            "sigma_v"            : 1.0,
            "sigma_p"            : 1.0,
        }
        self.best_parameters = self.parameters
        self.best_loss = 1e10

        self.grads = {
            # Measurement Noise Sigma
            "sigma_GPS_x"       : 0.0,
            "sigma_GPS_y"       : 0.0,
            "sigma_GPS_h"       : 0.0,

            # Process Noise Sigmas
            "max_acc"           : 0.0 ,
            "max_sttering_rate" : 0.0 ,
            "sigma_x"           : 0.0 ,
            "sigma_y"           : 0.0 ,
            "sigma_h"           : 0.0 ,
            "sigma_v"           : 0.0 ,
            "sigma_p"           : 0.0 ,
        }

        # optimization config
        self.optim_configs = {}
        for p , pp in self.parameters.items():
            d = {k: v for k, v in self.optim_config.items()}
            print(p)
            self.optim_configs[p] = d
    
    def draw(self):
        for index in range(0, self.training_len):
            dataset = self.training_data[index]['dataset']
            EKF_filter = EKFNet.EKFNet()
            EKF_filter.load_data_set(dataset)
            EKF_filter = EKFNet.EKFNet()
            EKF_filter.load_data_set(dataset)
            # Update the parameters
            print("best RMS Loss", self.best_loss)
            EKF_filter.set_paramters(self.best_parameters)
            EKF_filter.run_EKF_NET_forward()
            EKF_filter.plot_overview()



    def _step(self, counter):
        """
        training step, make a a single gradient update. 
        """
        # random select some data sets
        random_training_index = [counter + i for i in range(0 , self.batch_size)]
        #print(random_training_index)
        #print(random_training_index)
        dR = np.zeros((3,3))
        dQ_acc = np.zeros((2,2))
        dQ_other = np.zeros((5,5))

        loss = {
            "RMS_state"            :  0.0,
            "RMS_meas"             :  0.0,
            "logLikelihood_state"  :  0.0,
            "logLikelihood_meas"   :  0.0
        }

        #for training_index in random_training_index:
        random_training_index = [counter + i for i in range(0 , self.training_len)]
        for index in range(0, self.training_len):
            #index = self.ramdom_index_training[training_index]
            dataset = self.training_data[index]['dataset']
            
            EKF_filter = EKFNet.EKFNet()
            EKF_filter.load_data_set(dataset)
            # Update the parameters
            EKF_filter.set_paramters(self.parameters)
            EKF_filter.run_EKF_NET_forward()
            EKF_filter.generate_grad()

            loss_ = EKF_filter.totalLoss()
            dR_, dQ_acc_, dQ_other_ = EKF_filter.run_backward()
            loss["RMS_state"]           += loss_["RMS_state"]
            loss["RMS_meas"]            += loss_["RMS_meas"]
            loss["logLikelihood_state"] += loss_["logLikelihood_state"]
            loss["logLikelihood_meas"]  += loss_["logLikelihood_meas"]
            dR       += dR_
            dQ_acc   += dQ_acc_
            dQ_other += dQ_other_

            # perform a parameter update
        dR       /= len(random_training_index)
        dQ_acc   /= len(random_training_index)
        dQ_other /= len(random_training_index) 

        loss["RMS_state"]           /= len(random_training_index)  
        loss["RMS_meas"]            /= len(random_training_index) 
        loss["logLikelihood_state"] /= len(random_training_index) 
        loss["logLikelihood_meas"]  /= len(random_training_index) 
        self.loss_history.append(loss)
        if self.best_loss > loss["RMS_state"]:
            self.best_loss = loss["RMS_state"]
            self.best_parameters = self.parameters

        # perform a parameter update
        ## construct grad from dR, dQ_acc, dQ_other
        self.construct_grads(dR, dQ_acc, dQ_other)
        #print(self.parameters["max_acc"], " grad " , self.grads["max_acc"] )
        #print(self.parameters["max_sttering_rate"], " grad ", self.grads["max_sttering_rate"])

        for p, w in self.parameters.items():
            config = self.optim_configs[p]
            if self.grads[p] > 1 : 
                self.grads[p] =  1.0
            if self.grads[p] < -1:
                self.grads[p] = -1
            #grads[p] = grads[p] - 0.01 * self.parameters[p]

            self.grads[p] = self.grads[p] + 0.1 * self.parameters[p]
            
            next_w, next_config = self.update_rule(self.parameters[p], 
                                                   self.grads[p], 
                                                   config)
            if next_w > 0.01:
                self.parameters[p]    = next_w
            self.optim_configs[p] = next_config
        # fix some parameter for testing
        #self.parameters['sigma_GPS_x'] = 2.0
        #self.parameters['sigma_GPS_y'] = 2.0
        #self.parameters['sigma_GPS_h'] = (10 * np.pi / 180)**2

    def construct_grads(self, dR, dQ_acc, dQ_other):
        #dR = np.zeros((3,3))
        self.grads = {
            # Measurement Noise Sigma
            "sigma_GPS_x"          : dR[0][0],
            "sigma_GPS_y"          : dR[1][1],
            "sigma_GPS_h"          : dR[2][2],

            # Process Noise Sigmas
            "max_acc"              : dQ_acc[0][0],
            "max_sttering_rate"    : dQ_acc[1][1],

            "sigma_x"              : dQ_other[0][0],
            "sigma_y"              : dQ_other[1][1],
            "sigma_h"              : dQ_other[2][2],
            "sigma_v"              : dQ_other[3][3],
            "sigma_p"              : dQ_other[4][4],
        }
    
    def validation(self):
        loss = {
            "RMS_state"            :  0.0,
            "RMS_meas"             :  0.0,
            "logLikelihood_state"  :  0.0,
            "logLikelihood_meas"   :  0.0
        }

        for index in range(0, self.validation_len):
            #index = self.ramdom_index_training[training_index]
            dataset = self.validation_data[index]['dataset']
            
            EKF_filter = EKFNet.EKFNet()
            EKF_filter.load_data_set(dataset)
            # Update the parameters
            EKF_filter.set_paramters(self.parameters)
            EKF_filter.run_EKF_NET_forward()
            loss_ = EKF_filter.totalLoss()
            loss["RMS_state"]           += loss_["RMS_state"]
            loss["RMS_meas"]            += loss_["RMS_meas"]
            loss["logLikelihood_state"] += loss_["logLikelihood_state"]
            loss["logLikelihood_meas"]  += loss_["logLikelihood_meas"]

        loss["RMS_state"]           /= self.validation_len  
        loss["RMS_meas"]            /= self.validation_len 
        loss["logLikelihood_state"] /= self.validation_len 
        loss["logLikelihood_meas"]  /= self.validation_len 
        self.validation_history.append(loss)

    def training(self): 
        """
        run the optimization to train the model. 
        """
        iterations_per_epoch = max(self.training_len // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        counter = 0 

        for t in tqdm(range(num_iterations)):
            self._step(counter)
            self.validation()
            counter += 1
            # print the training loss and validation loss
            
            if self.verbose and t % self.print_every == 0: 
                print('(Iteration %d / %d) \n Post RMS: %f , \n Meas RMS: %f ,\n Post Like: %f, \n Meas Like %f' % (
                       t + 1, num_iterations, 
                       self.loss_history[-1]["RMS_state"], 
                       self.loss_history[-1]["RMS_meas"],
                       self.loss_history[-1]["logLikelihood_state"],
                       self.loss_history[-1]["logLikelihood_meas"],
                        
                       ))
                print('(Validation) \n Post RMS: %f , \n Meas RMS: %f ,\n Post Like: %f, \n Meas Like %f' % (
                       self.validation_history[-1]["RMS_state"], 
                       self.validation_history[-1]["RMS_meas"],
                       self.validation_history[-1]["logLikelihood_state"],
                       self.validation_history[-1]["logLikelihood_meas"],
                        
                       ))
                for k, v in self.parameters.items():
                    print("%s  : %f ", k ,v)
            
            epoch_end = (t + 1) % iterations_per_epoch == 0

            if epoch_end:
                counter = 0
                random.shuffle(self.ramdom_index_training)
                random.shuffle(self.ramdom_index_validation)
                self.epoch += 1
                if self.epoch % 50 == 0:
                    for k in self.optim_configs:
                        self.optim_configs[k]['learning_rate'] *= self.lr_decay





        



