import numpy as np
import time
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from murty import Murty
import copy

from .poisson import Poisson


class PMBM: 
    def __init__(self, config): 
        if config.motion_model_name != 'Mixed': 
            self.mixed = False
            self.measurement_model = config.measurement_model
            self.measurement_noise = config.measurement_noise
            self.meas_dim          = config.measurement_model.shape[0]
            self.unmeasurable_state_mean = config.unmeasurable_state_mean
            self.uniform_covariance = config.uniform_covariance
        else: 
            self.mixed = True
            self.measurement_models = config.measurement_models
            self.measurement_noises = config.measurement_noises
            self.meas_dims = 3
            self.unmeasurable_state_means = config.unmeasurable_state_means
            self.uniform_covariances = config.uniform_covariances
            
        self.state_dims = config.state_dims
        self.motion_model = config.motion_model
        self.gating_distance = config.gating_distance
        
        self.birth_gating_distance = config.birth_gating_distance
        
        self.poisson_birth_var = config.poisson_birth_var
        # TODO: implement the Poisson distribution
        
        
            
            



