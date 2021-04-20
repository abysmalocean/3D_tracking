# Tracker implementation
# use the EKF, which is designed bu my self
from filterpy.kalman import KalmanFilter
from filter.covariance import Covariance
import numpy as np
from filter.EKFBoxTracker import EKFBoxTracker

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
                 timestemp = 0.0, 
                 track_score=None, 
                 tracking_name='car'):
        """
        Initialised a tracker using initial bounding box. 
        """
        # define the different model
        # with angular velocity or not
        self.EKF_tracker = EKFBoxTracker(
            bbox3D.reshape((7,1)),
            timestemp = timestemp, 
            tracking_name = tracking_name)

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

    def update(self, bbox3D, info): 
        """
        update the state vector with observed bbox. 
            bbox3D: this is the detection function
        """
        raise NotImplementedError
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
        print("update?")
        self.EKF_tracker.update(bbox3D)
        
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
        
        self.EKF_tracker.predict()

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
        return self.EKF_tracker.transfor_state()
    
    