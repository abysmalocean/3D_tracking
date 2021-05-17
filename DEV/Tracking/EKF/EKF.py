import numpy as np
from math import sin, sqrt, cos
from scipy.stats import multivariate_normal

from utils.utils import angle_difference
from matplotlib import pyplot as plt

#np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

class After_filter(object):
    def __init__(self):
        super(After_filter, self).__init__()
        self.x        = []
        self.x_hat    = []
        self.W        = []
        self.W_meas   = []
        self.V        = []
        self.ftimes   = []
        self.locs     = []
        self.vels     = []
        self.P        = []
        self.headings = []
        self.lengths  = []
        self.widths   = []

        # Kalman Update Information
        self.S = []
        self.K = []
        

class EKF(): 
    def __init__(self, 
                 uncertainty_net = None,
                 tracking_name = 'car'):
        #super(EKF, self).__init__()
        self.maxacc = 3.0
        self.maxteeringacc = 0.6
        self.wheelbase_to_length_ratio = 0.7
        
        # the variance for w and l is from the original
        self.l_var = None
        self.w_var = None
        
        self.x = None    
        self.P = None
        
        self.post_x = []
        self.post_p = []
        
        # data used for smoother
        self.afterSmooth = After_filter()
        self.A = [] # predicted Jocobian for state transiton
        self.x_smooth = [] # updated states
        self.V = [] # updated covariance
        self.W = [] # predicted covariance
        
        # Unceratinty Net
        self.uncertainty_net = uncertainty_net
        
    # initialize the States
    def create_initial(
        self, 
        x_0 = None, 
        p_0 = None, 
        q_0 = None
        ): 
        self.x = x_0
        self.P = p_0
        self.Q = q_0
        self.w_var = p_0[5][5]
        self.l_var = p_0[6][6]
        #print("create Initial")
        # for smoother
        self.W.append(self.P)
        
    def G(self, state, dt):
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        w = state.item(5)
        l = state.item(6)
        #h = state.item(7)
        #z = state.item(8)
        wheelbase = self.wheelbase_to_length_ratio * l
        
        G_matrix = np.matrix([[1, 0, -sin(th) * cos(phi) * v * dt, 
                                      cos(th) * cos(phi) * dt, 
                                     -cos(th) * sin(phi) * v * dt, 0, 0],
                              [0, 1, cos(th) * cos(phi) * v * dt, 
                                     sin(th) * cos(phi) * dt, 
                                     -sin(th) * sin(phi) * v * dt, 0, 0],
                              [0, 0, 1, sin(phi) * dt / wheelbase, 
                                        cos(phi) * v * dt / wheelbase, 0, 
                                        - sin(phi) * v * dt / (wheelbase * l)], 
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0 ,0], 
                              [0, 0, 0, 0, 0, 1 ,0],
                              [0, 0, 0, 0, 0, 0 ,1]])
        return G_matrix
    
    def predict_measurment(self, state = None): 
        if state is None: 
            state = self.x
            
        x   = state.item(0)
        y   = state.item(1)
        th  = state.item(2)
        v   = state.item(3)
        phi = state.item(4)
        w   = state.item(5)
        l   = state.item(6)
        
        return np.array([x + (self.wheelbase_to_length_ratio - 0.5) * l * cos(th),
                         y + (self.wheelbase_to_length_ratio - 0.5) * l * sin(th), 
                         th,
                         w,
                         l])[:, None]
        
    def gererate_dH(self, state = None):
        if state is None: 
            state = self.x
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        w = state.item(5)
        l = state.item(6)
        
        H = np.zeros((5,7))
        
        # for x
        H[0, 0] = 1.0
        H[0, 2] = -(self.wheelbase_to_length_ratio - 0.5) * l * sin(th)
        H[0, 6] =  (self.wheelbase_to_length_ratio - 0.5) * cos(th)
        # for y
        H[1, 1] = 1.0
        H[1, 2] = (self.wheelbase_to_length_ratio - 0.5) * l * cos(th)
        H[1, 6] = (self.wheelbase_to_length_ratio - 0.5) * sin(th)
        
        # for th
        H[2, 2] = 1.0
        
        # for w
        H[3, 5] = 1.0
        
        # for l 
        H[4, 6] = 1.0
        
        return H
    
        
    def new_R(self, 
              state, 
              dt, 
              maxacc = 5.0, 
              maxsteeringacc = 0.3):
        
        max_linear_acc    = maxacc
        max_steering_rate = maxsteeringacc
        max_orthogonal_error = 0.1

        T = dt

        x     = state.item(0)
        y     = state.item(1)
        theta = state.item(2)
        v     = state.item(3)
        phi   = state.item(4)
        w = state.item(5)
        L = state.item(6)
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        sin_cos = cos_t * sin_t
        tan_p = np.tan(phi)
        T2    = (dt * dt) / 2.0

        Q = np.eye(3)
        B = np.zeros((7,3))

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

        cov = np.dot(np.dot(B, Q), B.T)
        # Maually add the variance for w and l
        
        cov[5][5] = self.w_var
        cov[6][6] = self.l_var
        return cov
    
    def predict(self, dt, state = None):
        if state is None:
            state = self.x
        #assert state, None
        x   = state.item(0)
        y   = state.item(1)
        th  = state.item(2)
        v   = state.item(3)
        phi = state.item(4)
        w   = state.item(5)
        l   = state.item(6)
        #h   = state.item(7)
        #z   = state.item(8)
        
        # predict the state 
        
        temp_x =  np.array([x + cos(th) * cos(phi) * v * dt,
                            y + sin(th) * cos(phi) * v * dt,
                            th + sin(phi) * v * dt / (l * self.wheelbase_to_length_ratio),
                            v, 
                            phi,
                            w,
                            l ])[:, None]
        
        # predict the covariance
        tmpG = self.G(state, dt)
        R_val = self.new_R(state, dt)
        PropCov = np.dot(np.dot(tmpG, self.P), tmpG.transpose()) + R_val
        self.x = temp_x
        self.P = PropCov
        
        # used for smoother
        self.A.append(tmpG)
        self.W.append(PropCov)
        return self.x
    
    def measurement_predict(self):
        self.z_pred = self.predict_measurment()
        self.H      = self.gererate_dH()
        self.S = np.dot(np.dot(self.H, self.P), self.H.transpose()) + self.Q
        self.K = np.dot(np.dot(self.P, self.H.transpose()), np.linalg.inv(self.S))
    
    def measurement_likelihood(self, detection): 
        """
        Out put the likelihood for current detections
        """
        y_ = detection - self.z_pred
        y_.itemset(2, angle_difference(detection.item(2),
                                      self.z_pred.item(2)))
        #print("heading difference -->", y_.item(2))
        mean = np.zeros(y_.shape[0])
        prob = multivariate_normal.logpdf(y_.T, 
                                          mean = mean, 
                                          cov = self.S,
                                          allow_singular = True)
        #print("pro ", prob)
        return prob
    
    def update(self, detection):
        #print("measurement ", detection[0], detection[1])
        y_ = detection - self.z_pred
        y_.itemset(2, angle_difference(detection.item(2),
                                      self.z_pred.item(2)% (2 * np.pi)))
        #print(y_.item(2))
        # TODO: Use the uncertainty net output! 
        self.P = np.dot((np.identity(7) - np.dot(self.K, self.H)), self.P)
        self.x = self.x + np.dot(self.K, y_)
        
        # not turning to quick
        if (self.x.item(4) < -2  * np.pi / 6):
            self.x.itemset(4, -2 * np.pi / 6)
        if (self.x.item(4) > 2   * np.pi / 6):
            self.x.itemset(4, 2  * np.pi / 6)
        
        # speed should greater than 0
        if self.x.item(3) < -10.0:
            self.x.itemset(3, -10.0)
        
        self.post_x.append(self.x)
        self.post_p.append(self.P)
        
        # for smoother
        self.x_smooth.append(self.x)
        self.V.append(self.P)
        
    def smoother(self, ftimes): 
            meas_times = ftimes
            NSteps = len(meas_times)
            kf_result_obj = None
            kf_result_obj = After_filter()

            x_T = []
            p_T = []
            
            x = self.x_smooth
            V = self.V
            W = self.W
            A = self.A
            
            #print(len(self.x_smooth), len(V), len(W), len(A))
            for i in range(NSteps - 1 , -1, -1):
                if i == NSteps - 1:
                    x_T.insert(0, self.x_smooth[i])
                    p_T.insert(0, self.V[i])
                else:
                    Jt = np.dot(np.dot(V[i], A[i].transpose()), np.linalg.inv(W[i+1]))

                    x_T.insert(0, x[i] + np.dot(Jt, (x_T[0] - self.predict
                                     (meas_times[i + 1] - meas_times[i], state=x[i]))))

                    p_T.insert(0, V[i] + np.dot(np.dot(Jt, (p_T[0] - W[i+1])), \
                                                Jt.transpose()))
            kf_result_obj.x = x_T
            kf_result_obj.V = p_T
            self.afterSmooth = kf_result_obj        





