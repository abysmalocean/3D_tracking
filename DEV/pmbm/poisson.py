import numpy as np
from math import atan2, sqrt
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import cdist


class Distribution: 
    def __init__(self, 
                 state,
                 variance,
                 weight,
                 object_class, 
                 motion_model): 
        assert state.shape[1] == 1, "Input state is not column vector"
        assert state.shape[0] == variance.shape[0], "State vector not aligned with Covariance matrix"
        self.state_dim = state.shape[0]
        self.state     = state
        self.variacne  = variance
        self.weight    = weight
        self.object_class = object_class
        # TODO: the get_Q function still under develop
        self.motion_noise = motion_model.get_Q(self.object_class)
        
    def predict(self, 
                filt, 
                survival_probability): 
        """
        Prediction step
        Predict the distribution for the next step: 
            @ filt: filter
            @ survival_probability : surviviling probability
        """
        
        # selecting the motion model
        if self.motion_model.model == 0: 
            velocity = self.state[3]
        elif self.motion_model.model == 1: 
            velocity = np.linalg.norm(self.state[2:4, :])
        else: 
            velocity = 0
            
        if velocity != 0:
            # this is the important step for predicting the next step
            _state, _variance = filt.predict(self.state, 
                                             self.variance, 
                                             self.motion_model, 
                                             self.motion_niose, 
                                             self.object_class)
            self.state = _state
            self.variacne = _variance
            
        # updating the weight for the current Gaussian pdf
        self.weight = survival_probability * self.weight
    
    def update(self, detection_probability): 
        """
        updating step: 
            @ detection_probability : constant detection probability
        """
        # updating the weight
        self.weight = detection_probability * self.weight
    
    def __repr__(self): 
        return '<Distribution Class \n Weight: {} Object_class: {} \n'.format(self.weight, 
                                                                              self.object_class)
    
class Poisson: 
    """
    Class to hold all Gaussian Mixture poisson distribution. 
    Method include birth, prediction, merge, prune, recycle. 
    INPUT: 
    @ param: birth_state        : Where to birth new poisson distribution
    @ param: birth_var          : What covriances the new poisson should have. 
    @ param: birth_weight_factor: Weight of the new distribution
    @ param: prune_threshold    : Which weight threshold for pruning
    @ param: merge_threshold    : Which distance threshold for merge of poisson distribution
    @ param: reduce_factor      : How much should the weight be reduced for each timestep
    """
    
    def __init__(self, 
                 birth_state, 
                 birth_var, 
                 birth_weight, 
                 prune_threshold, 
                 merge_threshold, 
                 reduce_factor, 
                 uniform_weight,
                 uniform_radius, 
                 uniform_angle, 
                 uniform_adjust, 
                 state_dim): 
        
        # number of birth
        if birth_state is None: 
            self.number_of_births = 0
        else: 
            self.number_of_births = len(birth_state)
        
        self.birth_weight = birth_weight
        self.state_dim = state_dim
        self.distributions = []
        
        # define some parameters and threshold
        self.birth_state = birth_state
        self.birth_var = birth_var
        self.prune_threshold = prune_threshold
        self.merge_threshold = merge_threshold
        self.reduce_factor = reduce_factor

        self.uniform_radius = uniform_radius
        self.uniform_angle  = uniform_angle  # Tuple with min, max in radius
        self.uniform_adjust = uniform_adjust
        self.uniform_weight = uniform_weight
        self.uniform_area   = 0.5 * uniform_radius ** 2 * (uniform_angle[1] - uniform_angle[0])
        
        # Plot thingies
        self.x_range = [-50, 150]
        self.y_range = [-50, 150]
        self.window_min = np.array([self.x_range[0], self.y_range[0]])
        self.window_size = np.array([self.x_range[1] - self.x_range[0], 
                                     self.y_range[1] - self.y_range[0]])
        self.window_intensity = np.prod(self.window_size)
        
        self.grid_length = [200, 200]
        self.x_dim = np.linspace(self.x_range[0], 
                                 self.x_range[1], 
                                 self.grid_length[0])
        self.y_dim = np.linspace(self.y_range[0], 
                                 self.y_range[1], 
                                 self.grid_length[1])
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_dim, self.y_dim)
        self.grid = np.dstack((self.x_mesh, self.y_mesh))
        self.intensity = np.zeros((self.grid_length[1], 
                                   self.grid_length[0]))

    def __repr__(self): 
        return '<Poisson Class \n Distributions: \n {}>'.format(self.distributions)
    
    def give_birth(self): 
        """
        Give birth according to the defined birth distributions. 
        ! The birth distribution is defined in the begining when this object is constructed. 
        """
        for i in range(self.number_of_births): 
            self.distributions.append(Distribution(self.birth_state[0],
                                                   self.birth_var,
                                                   self.birth_birth_weight))
    
    def predict(self, filt, survival_probability): 
        """
        Predict all the filter in the self.distributions
        ! 1. Predict the existing Gaussian. 
        ! 2. Give birth according to the birth model
        @ param: survival_probability. constance survival probability. 
        """
        
        for dist in self.distributions: 
            dist.predict(filt, survival_probability)
        self.give_birth()
    
    def update(self, detection_probability): 
        """
        Update the existing distribution is self.disttributions
        """    
        
        for dist in self.distributions: 
            dist.update(detection_probability)
    
    def prune(self): 
        """
        prune the distributions which weight is small than self.prune_threshold
        """
        self.distributions[:] = [dis for dis in self.distributions if 
                                         dis.weight > self.prune_threshold]
        
    def within_uniform(self, point): 
        """
        ? Not sure what this function used for. 
        # TODO: implement this function
        """
        raise NotImplementedError()
        
    def merge(self): 
        """
        An important function to implement. 
        #TODO: implement this function 
        """
        raise NotImplementedError()
    
    def recycle(self, bernoulli, motion_model, object_class):
        """
        Recycle the small weight Bernoulli distribution. 
        Append the newly created Gaussian distribution to the self.distributions
        Please chech the material for this fucntion. 
        """
        _distr = Distribution(state=bernoulli.state,
                              variance=bernoulli.variance,
                              weight=bernoulli.existence,
                              object_class=object_class,
                              motion_model=motion_model)
        self.distributions.append(_distr)
    
    def reduce_weight(self, index): 
        self.distributions[index].weight *= self.reduce_factor
        
    def plot(self, measurement_model): 
        self.compute_intensity(measurement_model)
        plt.figure()
        
        # plot the center of the distribution
        for counter, distribution in enumerate(self.distributions):
            plt.plot(distribution.state[0], distribution.state[1], 'bo',
                     markersize=self.distributions[counter].weight * 10)
        plt.title("Poisson Distribution")
        
        # plot the intensity probability
        plt.contourf(self.x_dim, self.y_dim, self.intensity, alpha=0.5)
        plt.xlim(XLIM[0], XLIM[1])
        plt.ylim(ZLIM[0], ZLIM[1])
        plt.show()
        

    def compute_intensity(self, measurement_model): 
        self.intensity = np.zeros((self.grid_length[1], self.grid_length[0]))
        # increase intensity based on distributions in PPP
        for counter, distribution in enumerate(self.distributions): 
            measureable_states = measurement_model @ distribution.state
            measureable_variance  = measurement_model @ distribution.variance @ measurement_model.T
            # create the Gaussian distribution
            _rv = stats.multivariate_normal(mean=measurable_states.reshape(np.shape(measurement_model)[0]),
                                            cov=measurable_variance)
            # pribability with the weight
            sample_pdf = _rv.pdf(self.grid) * self.distributions[counter].weight
            # accumulate the probability
            self.intensity += sample_pdf
            
            
        