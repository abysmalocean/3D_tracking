# Tracker implementation
from filterpy.kalman import KalmanFilter
from filter.covariance import Covariance
import numpy as np

def angle_in_range(angle):
      
    '''
    Input angle: -2pi ~ 2pi
    Output angle: -pi ~ pi
    '''
    if angle > np.pi:
          angle -= 2 * np.pi
    if angle < -np.pi:
          angle += 2 * np.pi
    return angle



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
        # change the x
        self.kf.x[:7] = bbox3D.reshape((7,1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        # number of total hits including the first detection
        self.hits = 1
        # number of continusing hit considering the first detection
        self.hit_streak = 1
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info # other information
        self.track_score = track_score
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity

    def update(self, bbox3D, info): 
        """
        update the state vector with observed bbox. 
            bbox3D: this is the detection function
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        # number of continusing hit
        self.hit_streak += 1
        if self.still_first: 
            # number of continusing hit in the first time
            self.first_continuing_hit += 1
        
        ###################### Orientation correction
        # normalize the theta ()
        # FIXME: use the angle normalization function need to change the whole
        # block. using the angle normalization code
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        # if the angle of two theta is not acute angle
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and \
           abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     
            self.kf.x[3] += np.pi       
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        
        # now the angle is acute: < 90 ir > 270, convert the case of > 270 to 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0: self.kf.x[3] += np.pi * 2
        else: self.kf.x[3] -= np.pi * 2

        ####################################
        # update the KF
        self.kf.update(bbox3D)
        
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        self.info = info

    def predict(self): 
        """
        Advances the state vector and returns the predicted bounding box estimates
        """
        raise NotImplementedError
        self.kf.predict()
        self.kf.x[3] = angle_in_range(self.kf.x[3])

        self.age += 1
        if (self.time_since_update > 0): 
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self): 
        """
        Return the current bounding box estimate. 
        """
        return self.kf.x[:7].reshape((7, ))
    
    