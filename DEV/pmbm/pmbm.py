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
        
        self.poisson = Poisson(birth_state=config.poisson_birth_state,
                               birth_var=config.poisson_birth_var,
                               prune_threshold=config.prune_threshold_poisson,
                               birth_weight=config.birth_weight,
                               reduce_factor=config.poisson_reduce_factor,
                               merge_threshold=config.poisson_merge_threshold,
                               uniform_weight=config.uniform_weight,
                               uniform_radius=config.uniform_radius,
                               uniform_angle=config.uniform_angle,
                               uniform_adjust=config.uniform_adjust,
                               state_dim=self.state_dims)
        self.filt = config.filt
        self.prune_threshold_targets = config.prune_threshold_targets
        self.prune_global_hypo = config.prune_threshold_global_hypo
        self.prune_single_existence = config.prune_single_existence
        
        self.desired_nof_global_hypos = config.desired_nof_global_hypos
        self.max_nof_global_hypos = config.max_nof_global_hypos
        self.min_new_nof_global_hypos = config.min_new_nof_global_hypos
        self.max_new_nof_global_hypos = config.max_new_nof_global_hypos
        self.global_init_weight = config.global_init_weight
        self.clutter_intensity = config.clutter_intensity
        
        # internal states
        self.global_hypotheses = {}
        self.global_hypotheses_counter = 0 
        self.targets = {}
        self.new_targets = []
        self.target_counter = 0
        self.current_time  = 0
        self.estimated_targets = None
        self.show_predictions = config.show_predictions
        self.verbose = False
        
        self.dt = config.dt
        self.coord_transform = config.coord_transform
        
    def __repr__(self):
        return '< PMBM Class \nCurrent time: {}\nNumber of targets: {}\nGliobalHypos: {} \n#Poissons: {} \n'.format(
            self.current_time,
            len(self.targets),
            self.global_hypotheses,
            len(self.poisson.distributions))
        
    def run(self, 
            measurements, 
            classes, 
            imu_data,
            time_idx,
            verbose=False,
            verbose_time=False): 
        """
        Run the filter. 
        """
        if verbose: 
            print('Targets Before prediction: \n{}'.format(self.targets))
        if verbose: 
            print('GlobalHypos Before prediction: \n{}'.format(self.global_hypotheses))
        self.current_time = time_idx
        if verbose_time: 
            tic = time.time()
            self.predict(imu_data)
            toc_predict = time.time() - tic
            
            tic = time.time()
            self.update(measurements, classes)
            toc_update = time.time() - tic
            tic = time.time()
            
            self.target_estimation()
            toc_estimation = time.time() - tic
            tic = time.time()
            
            self.reduction()
            toc_reduction = time.time() - tic
            print('\t\tT_Prediction:\t {}\tms\n'
                  '\t\tT_Update:\t {}\tms\n'
                  '\t\tT_Estimation\t {}\tms\n'
                  '\t\tT_Reduction\t {}\tms\n'.format(round(toc_predict * 1000, 1),
                                                      round(toc_update * 1000, 1),
                                                      round(toc_estimation * 1000, 1),
                                                      round(toc_reduction * 1000, 1)))
        else: 
            self.predict(imu_data)
            self.update(measurements, classes)
            self.target_estimation()
            self.reduction()
        if verbose: 
            print('Targets After update and reduction: \n{}'.format(self.targets))
        if verbose: 
            print('GlobalHypos After update and reduction: \n{}'.format(self.global_hypotheses))

    # ! 4 Main functions
    def predict(self, imu_data): 
        """
        Prediciton, Poisson and MBM prediction
        """
        raise NotImplementedError()
    
    def update(self, measurement, classes): 
        """
        Update the MBM and update the global hypotheses
        Update the poisson also
        """
        raise NotImplementedError()
    
    def target_estimation(self): 
        """
        Estimate the target states according to the highest weight MB
        """
        raise NotImplementedError()
    
    def reduction(self):
        """
        post processing, reduce the computation burden
        """
        raise NotImplementedError()
    
    # !  Update Functions
    def update_global_hypotheses(self, measurement): 
        """
        Updates global hypotheses. If there is no global hypo since before. 
        it creates a new one. 
        If there are previous global hypothesis it creates new hypotheses using 
        Murty's algorithm. 
        
        """
        raise NotImplementedError()
    
    def create_cost(self, global_hypo, measurements): 
        """
        Create a cost matrix for a sepcific global hypothesis. 
        @ param: global_hypo: 
        @ meausmrenets: 
        """
        raise NotImplementedError()

    def new_targets_cost(self, measurements): 
        """
        New target cost Matrix
        """
        
        raise NotImplementedError()
    
    def generate_new_global_hypos(self, 
                                  cost_matrix, 
                                  global_hypo,
                                  row_col_2_tid_sid, 
                                  missed_meas_target_2_tid_sid): 
        """
        Generates new global hypotheses for a sepcific global hypo with 
            respective cost maxtix. 
        Uses Murty's algorithm to generate K different assignments for the musaruements. 
        
        """
        raise NotImplementedError()
    
    def get_global_weight(self, hypo): 
        raise NotImplementedError()
    
    def normalize_global_hypotheses(self): 
        raise NotImplementedError()
    
    
    # ! reduction Functions
    def prune_gloabl(self): 
        raise NotImplementedError()
    
    def recycle_targets(self): 
        raise NotImplementedError()
    
    def remove_unused_STH(self): 
        raise NotImplementedError()
    
    def cap_global_hypos(self): 
        raise NotImplementedError()
    
    def merge_global_duplicates(self): 
        """
        Merging global hyphtesese that are the same, the weights are summed
        """
        raise NotImplementedError()
    
    def grated_new_targets(self, meas, object_class): 
        raise NotImplementedError()
    
    def sum_likelihood(self, 
                       states, 
                       variances, 
                       weights, 
                       measurments): 
        raise NotImplementedError()
    
    def possible_new_targets(self, measurements, classes): 
        raise NotImplementedError()
    
    
    
    
    

        
            
            



