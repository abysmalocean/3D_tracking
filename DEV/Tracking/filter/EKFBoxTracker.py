import numpy as np
from math import sin, sqrt, cos
from filter.covariance import Covariance
from utils.dict import *
from utils.utils import angle_difference
import pickle

class EKFBoxTracker():
    """
    This class represents the internal state of individual tracked
    objects observed bbox. Using the EKF algorithm. 
    """
    def __init__(
        self,
        detection,
        timestemp  = 0.0,
        tracking_name = 'car'):
        """
        Initialized a tracker using initial bounding box
        """
        self.tracking_name = tracking_name
        self.wheelbase_to_length_ratio = 0.7
        
        """
        x0 = np.array([x , 
                       y , 
                       self.headings[0], 
                       0.0, 
                       0.0, 
                       w[0], 
                       l[0],
                       h[0],
                       z])[:, None]
        > 
        box.size[2], 
        box.size[0],
        box.size[1],
        box.translation[0], 
        box.translation[1], 
        box.translation[2],
        rotation_to_yaw_agnle(box.rotation)
          
        """
        initials = pickle.load(open(processNoise_file_name , 'rb'))
        meas_var = pickle.load(open(MeasurementNoise_file_name, 'rb'))
        
        v = sqrt(initials['mean'][self.tracking_name][7] ** 2 +\
                 initials['mean'][self.tracking_name][8] ** 2)
        # initial assumprtions
        l = detection[4][0]
        th = detection[3][0]
        self.x = np.array([detection[0][0] - self.wheelbase_to_length_ratio * l * cos(th) / 2.0 , 
                           detection[1][0] - self.wheelbase_to_length_ratio * l * sin(th) / 2.0 , 
                           th, 
                           v, 
                           0.0, 
                           initials['mean'][self.tracking_name][1],
                           initials['mean'][self.tracking_name][2],
                           initials['mean'][self.tracking_name][0], 
                           initials['mean'][self.tracking_name][5]
                           ])[:, None]
        
        self.P = np.diag(np.array([1.**2, 
                                   1.**2, 
                                   (np.pi * 10 / 180.0)**2, 
                                   (10**2), 
                                   (0.1 * 0.1), 
                                   initials['var'][self.tracking_name][1] + 0.1,
                                   initials['var'][self.tracking_name][2] + 0.1,
                                   initials['var'][self.tracking_name][0] + 0.1, 
                                   initials['var'][self.tracking_name][5] + 0.1]))
        # build the measurement Noize
        self.Q = np.zeros((7, 7))
        self.Q[0][0] = meas_var[tracking_name][3]
        self.Q[1][1] = meas_var[tracking_name][4]
        self.Q[2][2] = meas_var[tracking_name][5]
        self.Q[3][3] = meas_var[tracking_name][6]
        self.Q[4][4] = meas_var[tracking_name][2]
        self.Q[5][5] = meas_var[tracking_name][1]
        self.Q[6][6] = meas_var[tracking_name][0]
        
        
        det = self.predict_measurment(self.x)
        y_ = detection - det
        y_.itemset(2, angle_difference(detection[3], det[3])% (2 * np.pi))
        dH = self.gererate_dH(self.x)
        self.S = np.dot(np.dot(dH, self.P), dH.transpose())
        self.K = np.dot(np.dot(self.P, dH.transpose()), np.linalg.inv(self.S))
        
        updated_x = self.x + np.dot(self.K, y_)
        
        # cap the heading
        if (updated_x.item(4) < -2 * np.pi / 6):
                updated_x.itemset(4, -2 * np.pi / 6)
        if (updated_x.item(4) > 2 * np.pi / 6):
                updated_x.itemset(4, 2 * np.pi / 6)
        
        self.x = updated_x
        
        

        
    def predict_measurment(self, state): 
        x   = state.item(0)
        y   = state.item(1)
        th  = state.item(2)
        v   = state.item(3)
        phi = state.item(4)
        w   = state.item(5)
        l   = state.item(6)
        h   = state.item(7)
        z   = state.item(8)
        
        return np.array([x + self.wheelbase_to_length_ratio * l * cos(th) / 2.0,
                         y + self.wheelbase_to_length_ratio * l * sin(th) / 2.0, 
                         z,
                         th,
                         l,
                         w,
                         h])[:, None]
        
        
    def predict(self, state, dt): 
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        w = state.item(5)
        l = state.item(6)
        h = state.item(7)
        z = state.item(8)
        
        wheelbase = self.wheelbase_to_length_ratio * l
        
        return np.array([x + cos(th) * cos(phi) * v * dt,
                         y + sin(th) * cos(phi) * v * dt,
                         th + sin(phi) * v * dt / wheelbase,
                         v,
                         phi,
                         w,
                         l,
                         h,
                         z])[:, None]
    
    def G(self, state, dt): 
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        w = state.item(5)
        l = state.item(6)
        h = state.item(7)
        z = state.item(8)
        wheelbase = self.wheelbase_to_length_ratio * l
        
        G_matrix = np.matrix([[1, 0, -sin(th) * cos(phi) * v * dt, 
                                      cos(th) * cos(phi) * dt, 
                                     -cos(th) * sin(phi) * v * dt, 0, 0, 0, 0],
                              [0, 1, cos(th) * cos(phi) * v * dt, 
                                     sin(th) * cos(phi) * dt, 
                                     -sin(th) * sin(phi) * v * dt, 0, 0, 0, 0],
                              [0, 0, 1, sin(phi) * dt / wheelbase, 
                                        cos(phi) * v * dt / wheelbase, 0, 
                                        - sin(phi) * v * dt / (wheelbase * l),
                                        0, 0], 
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0 ,0, 0, 0, 0], 
                              [0, 0, 0, 0, 0, 1 ,0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0 ,1, 0, 0, 0], 
                              [0, 0, 0, 0, 0, 0 ,0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0 ,0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0 ,0, 0, 0, 1]])
    def gererate_dH(self, state):
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        w = state.item(5)
        l = state.item(6)
        h = state.item(7)
        z = state.item(8)
        
        H = np.zeros((7,9))
        
        # for x
        H[0, 0] = 1.0
        H[0, 2] = self.wheelbase_to_length_ratio / 2.0 * l * cos(th)
        H[0, 6] = self.wheelbase_to_length_ratio / 2.0 * sin(th)
        # for y
        H[1, 1] = 1.0
        H[1, 2] = -self.wheelbase_to_length_ratio / 2.0 * l * sin(th)
        H[1, 6] = self.wheelbase_to_length_ratio / 2.0 * cos(th)
        
        # for z
        H[2, 8] = 1.0
        
        # for th
        H[3, 2] = 1.0
        
        # for l
        H[4, 6] = 1.0
        
        # for w 
        H[5, 5] = 1.0
        
        # for h
        H[6, 7] = 1.0
        
        return H
        
        
    
    def get_S(self): 
        """
        Get the residual covariance.
        """
        
    def R(self, dt, maxacc = 3.0, maxsteeringacc = 0.6):
        """
        Making the process noise
        """
        #maxacc = 3.0
        maxyawacc = 0.1
        #maxsteeringacc = 0.6
        return np.diag(np.array([(0.5 * maxacc * dt**2)**2     ,
                                 (0.5 * maxacc * dt**2)**2     ,
                                 (0.5 * maxyawacc * dt**2)**2  ,
                                 (maxacc * dt)**2              ,
                                 (maxsteeringacc * dt)**2      ,
                                 0.1 ** 2, 
                                 0.1 ** 2] ))
    def transfor_state(self): 
        return self.predict_measurment(self.x)
        
        
    
    def newBabylity_R(
        self, 
        state, 
        dt, 
        maxacc = 0.1, 
        maxsteeringacc = 0.03):
        """
        making the processing noise
        """
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        w = state.item(5)
        l = state.item(6)
        h = state.item(7)
        z = state.item(8)
        
        
        L = l

        max_linear_acc    = maxacc
        max_steering_rate = maxsteeringacc
        max_orthogonal_error = 0.1
        max_width_error      = 0.1
        max_length_error     = 0.1

        T = dt

        x     = state.item(0)
        y     = state.item(1)
        theta = state.item(2)
        v     = state.item(3)
        phi   = state.item(4)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        sin_cos = cos_t * sin_t
        tan_p = np.tan(phi)
        T2    = (dt * dt) / 2.0

        Q = np.eye(5)
        B = np.zeros((7,5))

        B[0][0] = T2*cos_t
        B[1][0] = T2*sin_t
        B[2][0] = -T2 * tan_p / L
        B[3][0] = T
        B[4][0] = 0.0

        #B[0][1] = -T2*v*v*tan_p*sin_t/L
        #B[1][1] = T2*v*v*tan_p*cos_t/L
        B[0][1] = 0.0
        B[1][1] = 0.0
        B[2][1] = T2*v*tan_p*tan_p/L
        B[3][1] = 0.0
        B[4][1] = T

        B[0][2] = -sin_t
        B[1][2] = cos_t
        B[2][2] = 0.0
        B[3][2] = 0.0
        B[4][2] = 0.0

        Q[0][0] = max_linear_acc ** 2
        Q[1][1] = max_steering_rate ** 2
        Q[2][2] = max_orthogonal_error ** 2
        Q[3][3] = max_width_error ** 2
        Q[4][4] = max_length_error ** 2

        cov = np.dot(np.dot(B, Q), B.T)
        # modify the width and length error
        cov[5][5] = (max_width_error *0.0001)** 2
        cov[6][6] = (max_length_error*0.0001) ** 2

        return cov
        
        
        
        
        
        
    