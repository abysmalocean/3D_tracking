# Tracker implementation

class KalmanBoxTracker(object): 
    """
    This class represents the internel state of individual tracked objects
    observed as bbox. 
    """
    count = 0
    def __init__(self, 
                 bbox3D, 
                 info, 
                 covariance_id = 0, 
                 track_score=None, 
                 tracking_name='car', 
                 use_angular_velocity=False):
        """
        Initialised a tracker using initial bounding box. 
        """
        # define the different model
        # with angular velocity or not
        if not use_angular_velocity: 
            self.kf = KalmanFilter(dim_x = 10, dim_z = 7)
            self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                                  [0,1,0,0,0,0,0,0,1,0],
                                  [0,0,1,0,0,0,0,0,0,1],
                                  [0,0,0,1,0,0,0,0,0,0],  
                                  [0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,0,0,1]])
            
            self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                                  [0,1,0,0,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0]])
        else: 
            # using the angluar velocity
            self.kf = KalmanFilter(dim_x=11, dim_z=7)       
            self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
                                  [0,1,0,0,0,0,0,0,1,0,0],
                                  [0,0,1,0,0,0,0,0,0,1,0],
                                  [0,0,0,1,0,0,0,0,0,0,1],  
                                  [0,0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,0,0,0,1]])
                                
            self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
                                  [0,1,0,0,0,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0,0]])
        # Initialize the covariance matrix, see covariance.py for more details
        if covariance_id == 0: # exactly the same as AB3DMOT baseline
            # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
            self.kf.P[7:,7:] *= 1000. #state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
            self.kf.P *= 10.
            # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
            self.kf.Q[7:,7:] *= 0.01
        elif covariance_id == 1: # for kitti car, not supported
            covariance = Covariance(covariance_id)
            self.kf.P = covariance.P
            self.kf.Q = covariance.Q
            self.kf.R = covariance.R
        elif covariance_id == 2: # for nuscenes
            # this is what we need
            covariance = Covariance(covariance_id)
            self.kf.P = covariance.P[tracking_name]
            self.kf.Q = covariance.Q[tracking_name]
            self.kf.R = covariance.R[tracking_name]
            if not use_angular_velocity:
              self.kf.P = self.kf.P[:-1,:-1]
              self.kf.Q = self.kf.Q[:-1,:-1]
        else:
            assert(False)
        
        # TODO: finish this fucntion
    def update(self, bbox3D, info): 
        """
        update the state vector with observed bbox. 
        """
        raise NotImplementedError
    def predict(self): 
        """
        Advances the state vector and returns the predicted bounding box estimates
        """
        raise NotImplementedError
    def get_state(self): 
        """
        Return the current bounding box estimate. 
        """
        return self.kf.x[:7].reshape((7, ))
    
    