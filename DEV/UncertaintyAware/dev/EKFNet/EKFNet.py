import numpy as np

from EKFNet import EKFNet_layers
from math import sin, sqrt, cos
from utility import angle_difference
from matplotlib import pyplot as plt
import EKFNet.config as config
from scipy.stats import multivariate_normal





class EKFNet(object): 

    """ 
    This is the EKFNet, implementation
    """ 

    def __init__(self): 
        super(EKFNet, self).__init__()
        self.ftimes    = []
        self.locs      = []
        self.bias_locs = []
        self.headings  = []
        self.w         = None
        self.h         = None
        self.imu_off   = None
        self.imu_off_l = None
        self.imu_alpha = None
        
        self.afterKF = None
        self.afterSmooth = None

        self.n = 5 


        # Measurement Noise Sigma
        self.sigma_GPS_x = 16.0
        self.sigma_GPS_y = 16.0
        #self.sigma_GPS_h = (10 * np.pi / 180)**2
        self.sigma_GPS_h = 0.04

        # Process Noise Sigmas
        self.max_acc = 3.0 **2
        self.max_sttering_rate = 0.6**2
        self.sigma_x = 5.0
        self.sigma_y = 5.0
        self.sigma_h = 5.0
        self.sigma_v = 5.0
        self.sigma_p = 5.0


        self.Q = np.diag(np.array([(1.0)**2, 
                                   (1.0)**2, 
                                   (5 * np.pi / 180)**2]))
        self.R = np.eye(3)
        max_linear_acc = 3.0
        max_steering_rate = 0.6
        max_orthogonal_error = 0.1
        self.R[0][0] = max_linear_acc ** 2
        self.R[1][1] = max_steering_rate ** 2
        self.R[2][2] = max_orthogonal_error ** 2
        self.x = []

        # Datasets
        self.datasets = None

        # Back_propogation Information
        self.forward_cache = {}
        self.dH            = {}

        # Configration
        self.config = config.config
        
    def set_paramters(self, parameters = None): 
        if parameters is not None: 
            # Measurement Noise Sigma
            self.sigma_GPS_x       = parameters["sigma_GPS_x"]
            self.sigma_GPS_y       = parameters["sigma_GPS_y"]
            self.sigma_GPS_h       = parameters["sigma_GPS_h"]

            # Process Noise Sigmas
            self.max_acc           = parameters["max_acc"]
            self.max_sttering_rate = parameters["max_sttering_rate"]
            #self.max_acc           = 0.0 
            #self.max_sttering_rate = 0.0 


            self.sigma_x           = parameters["sigma_x"]
            self.sigma_y           = parameters["sigma_y"]
            self.sigma_h           = parameters["sigma_h"]
            self.sigma_v           = parameters["sigma_v"]
            self.sigma_p           = parameters["sigma_p"]
            self.sigma_x           = 0.0
            self.sigma_y           = 0.0
            self.sigma_h           = 0.0
            self.sigma_v           = 0.0
            self.sigma_p           = 0.0

    def load_data_set(self, datasets): 
        self.datasets = datasets

        self.ftimes = datasets.timestamps_second
        self.w = datasets.w
        self.h = datasets.h
        self.imu_off = datasets.offset
        
        self.imu_alpha = np.arctan(self.imu_off[0] / self.imu_off[1])
        self.imu_off_l = np.sqrt(self.imu_off[0] ** 2 + self.imu_off[1] ** 2)

        for items in datasets.oxts:
            heading = items.heading
            locs = items.Trans
            #self.bias_locs.append([locs[0], locs[1]])
            x = locs[0] + self.imu_off_l * np.cos(heading - self.imu_alpha)
            y = locs[1] + self.imu_off_l * np.sin(heading - self.imu_alpha)
            self.locs.append([x,y])
            #self.headings.append(heading)
        self.bias_locs = datasets.locs
        self.headings   = datasets.heading
        #print(datasets.heading[0])
        
    
    def add_some_noise(self, sigma_x = 1.0, sigma_y= 1.0, sigma_theta = 0.05):
        n = len(self.bias_locs)
        noise_x = np.random.normal(0, sigma_x     , n)
        noise_y = np.random.normal(0, sigma_y     , n)
        noise_t = np.random.normal(0, sigma_theta , n)
        self.bias_locs = np.array(self.bias_locs)
        self.bias_locs[:, 0] += noise_x
        self.bias_locs[:, 1] += noise_y
        self.headings += noise_t

    def predict(self, state, dt): 
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        wheelbase = self.h
        return np.array([x + np.cos(th) * np.cos(phi) * v * dt,
                         y + np.sin(th) * np.cos(phi) * v * dt,
                         th + sin(phi) * v * dt / wheelbase,
                         v,
                         phi])[:, None]

    def G(self, state, dt):
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)

        wheelbase = self.h

        G_matrix = np.matrix([[1, 0, -sin(th) * cos(phi) * v * dt, 
                                      cos(th) * cos(phi) * dt, 
                                     -cos(th) * sin(phi) * v * dt],
                              [0, 1, cos(th) * cos(phi) * v * dt, 
                                     sin(th) * cos(phi) * dt, 
                                     -sin(th) * sin(phi) * v * dt],
                              [0, 0, 1, sin(phi) * dt / wheelbase, 
                                        cos(phi) * v * dt / wheelbase],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]])
        return G_matrix

    def new_R(self, state, dt, maxacc = 3.0, maxsteeringacc = 0.6):
        L = self.h
        max_linear_acc    = maxacc
        max_steering_rate = maxsteeringacc
        max_orthogonal_error = 0.1

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

        Q = np.eye(3)
        B = np.zeros((5,3))

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

        return cov

    def gen_B_simple(self, state, dt):
        T = dt

        x     = state.item(0)
        y     = state.item(1)
        theta = state.item(2)
        v     = state.item(3)
        phi   = state.item(4)
        T2    = (dt * dt) / 2.0
        
        B = np.zeros((5,2))
        B[0][0] = T2 * cos(theta)
        B[1][0] = T2 * sin(theta)
        B[3][0] = T

        B[2][1] = T2
        B[4][1] = T
        return B


    def gen_B(self, state, dt):
        L = self.h

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

        B = np.zeros((5,3))

        B[0][0] = T2*cos_t
        B[1][0] = T2*sin_t
        B[2][0] = -T2 * tan_p / L
        B[3][0] = T
        B[4][0] = 0.0

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

        return B

    def convertLocs(self, state):
        x = state.item(0)
        y = state.item(1)
        return ((x , y))
    
    def convertHeading(self, state):
    
        return state.item(2) % (2 * np.pi)

    def convertVels(self, state):
        
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        return (v * cos(phi) * cos(th), v * cos(phi) * sin(th))

    def gen_h(self, x_hat): 
        z_predict = np.array([0.0, 0.0, 0.0])[:, None]
        z_predict[0] = x_hat[0] - self.imu_off_l * cos(x_hat[2] - self.imu_alpha)
        z_predict[1] = x_hat[1] - self.imu_off_l * sin(x_hat[2] - self.imu_alpha)
        z_predict[2] = x_hat[2]
        return z_predict

    def gen_H(self , x_hat):
        l = self.imu_off_l
        alpha = self.imu_alpha
        th = x_hat[2]
        H_matrix = np.matrix([[1, 0, l * sin(th - alpha), 0, 0],
                              [0, 1,-l * cos(th - alpha), 0, 0],
                              [0, 0, 1, 0, 0]])
        return H_matrix

    def run_EKF_NET_forward(self): 
        x0 = np.array([self.bias_locs[0][0] , 
                       self.bias_locs[0][1] , 
                       self.headings[0], 
                       0.0, 
                       0.0])[:, None]
        
        if len(self.ftimes) > 1: 
            # better estimation of the initial velocity 
            x0.itemset(3, sqrt((self.locs[1][0] - self.locs[0][0])**2 + \
                (self.locs[1][1] - self.locs[0][1])**2))

        # Initial the Process Noise in the begining
        #
        P0 = np.diag(np.array([1.**2, 
                               1.**2, 
                               (np.pi * 10 / 180.0)**2, 
                               (10**2), 
                               (0.1 * 0.1)]))
        meas_times = self.ftimes
        n = x0.shape[0]

        NSteps = len(meas_times)
        x_pred = x0
        p_pred = P0

        R = np.diag(np.array([self.sigma_GPS_x, 
                              self.sigma_GPS_y, 
                              self.sigma_GPS_h]))

        Q_acc = np.diag(np.array([self.max_acc, 
                                  self.max_sttering_rate]))

        Q_other = np.diag(np.array([self.sigma_x, 
                                    self.sigma_y,
                                    self.sigma_h,
                                    self.sigma_v,
                                    self.sigma_p]))

        cache_pred = None; 

        for i in range(NSteps): 
            # Measurements
            z = [self.bias_locs[i][0], self.bias_locs[i][1], self.headings[i]]
            # Meaurement prediction
            y_, S, H, cache_meas = self.measurement_forward(x_pred, 
                                                       p_pred, z, R)

            # Update
            x_post, p_post, cache_update = self.update_forward(x_pred,
                                                               p_pred,
                                                               H, S, y_)
            self.x.append(x_post)
            self.forward_cache[i] = {
                'x_pred'      : x_pred,
                'p_pred'      : p_pred,
                'z'           : z,
                'cache_meas'  : cache_meas,
                'x_post'      : x_post,
                'p_post'      : p_post,
                'cache_update': cache_update,
                'cache_pred'  : cache_pred,
                'y_'          : y_,
                'S'           : S
            }
            # prediction
            if i < NSteps - 1: 
                dt = meas_times[i+1] - meas_times[i]
                x_pred, p_pred, cache_pred = self.prediction_forward(x_post, 
                                                                     p_post,
                                                                     dt,
                                                                     Q_acc,
                                                                     Q_other)
        
    def generate_fake_grad(self):
        T = len(self.forward_cache)

        for i in range(0 , T):
            self.dH[i] = {
                "dz_meas" : np.zeros((3,1)),
                "dp_meas" : np.zeros((3,3)), 
                "dx_post" : np.zeros((5,1)),
                "dp_post" : np.zeros((5,5)) 
            }

    def run_backward(self):
        '''
        dCache contains all the gradients produced by the each step
        "meas_pred" :
                      dz_meas: 
                      dp_meas:
        "state_post": 
                     dx_post:
                     dp_post: 

        return: 
            dQ_acc : deriative to acc
            dQ_other: deriative to Q_other
            dR     : deriative to measurements
        '''
        dQ_acc   = np.zeros((2,2))
        dQ_other = np.zeros((5,5))
        dR       = np.zeros((3,3))
        T = len(self.forward_cache)

        dx_post_prev = np.zeros((5,1))
        dp_post_prev = np.zeros((5,5))

        counter = 0
        for i in range(T - 10, 10, -1): 
            counter += 1
            #dQ_acc_       = np.zeros((2,2))
            #dQ_other_     = np.zeros((5,5))
            #dR_           = np.zeros((3,3))
            dz_meas       = self.dH[i]["dz_meas"]
            dp_meas       = self.dH[i]["dp_meas"]
            dx_post       = self.dH[i]["dx_post"]
            dp_post       = self.dH[i]["dp_post"]
            update_cache  = self.forward_cache[i]["cache_update"]
            meas_cache    = self.forward_cache[i]["cache_meas"]
            predict_cache = self.forward_cache[i]["cache_pred"]
            
            # Back update
            dx_pred_u, dp_pred_u, dH, dS, dy_ = self.\
                 update_backward(dx_post + dx_post_prev, 
                                 dp_post + dp_post_prev, 
                                 update_cache)
            # Back Meas
            dx_pred_m, dp_pred_m, dR_ = self.\
                measurement_backward(dy_ + dz_meas, 
                                     dS  + dp_meas,
                                     dH, 
                                     meas_cache)
            # back Prediction   
            dx_post_prev, dp_post_prev, dQ_acc_, dQ_other_ = self.\
                prediction_backward(dx_pred_u + dx_pred_m, 
                                    dp_pred_u + dp_pred_m, 
                                    predict_cache)

            dQ_acc   += dQ_acc_
            dQ_other += dQ_other_ 
            dR       += dR_

        dR     = dR         
        dQ_acc = dQ_acc     
        dQ_other = dQ_other 

        return dR, dQ_acc, dQ_other

    def run_kf(self):
        print("Running KF Liang XU")
        # initial state estimation
        x0 = np.array([self.bias_locs[0][0] , 
                       self.bias_locs[0][1] , 
                       self.headings[0], 
                       0.0, 
                       0.0])[:, None]
        
        if len(self.ftimes) > 1: 
            # better estimation of the initial velocity 
            x0.itemset(3, sqrt((self.locs[1][0] - self.locs[0][0])**2 + \
                (self.locs[1][1] - self.locs[0][1])**2))

        # Initial the Process Noise in the begining
        #
        P0 = np.diag(np.array([1.**2, 
                               1.**2, 
                               (np.pi * 10 / 180.0)**2, 
                               (10**2), 
                               (0.1 * 0.1)]))
        meas_times = self.ftimes
        n = x0.shape[0]

        NSteps = len(meas_times)
        Pred = x0
        PropCov = P0

        Q = np.diag(np.array([(0.1)**2, 
                              (0.1)**2, 
                              (5 * np.pi / 180)**2]))


        for i in range(NSteps): 
            # Update
            z = [self.bias_locs[i][0], self.bias_locs[i][1], self.headings[i]]
            x_post, p_post, cache = EKFNet_layers.ekf_net_update_forward(Pred, 
                                                 PropCov, 
                                                 self.gen_h, 
                                                 self.gen_H, 
                                                 self.Q, 
                                                 z)
            self.x.append(x_post)
            # prediction
            if i < NSteps - 1: 
                dt = meas_times[i+1] - meas_times[i]
                Pred, PropCov, cache = EKFNet_layers.efk_net_predict_forward(
                    x_post, p_post, dt, self.predict, self.G, self.gen_B, self.Q
                )

    def update_forward(self, x_pred, p_pred, H, S, y_): 
        PHT    = np.dot(p_pred, H.T)
        S_inv  = np.linalg.inv(S)
        K      = np.dot(PHT, S_inv)
        KH     = np.dot(K, H)
        IKH    = np.identity(5) - KH
        x_post = x_pred + np.dot(K, y_)
        p_post = np.dot(IKH, p_pred)
        cache = (x_pred, p_pred, H, S, y_, PHT, S_inv, K, KH, IKH)
        return x_post, p_post, cache

    def update_backward(self, dx_post, dp_post, cache):
        '''
        need to return
            dx_pred
            dp_pred
            dH
            dS
            dy_
        '''
        x_pred, p_pred, H, S, y_, PHT, S_inv, K, KH, IKH = cache
 
        """
        1.
        x_post = x_hat + np.dot(K, y_)
        """
        # dx_pred final
        dx_pred = dx_post
        dK_1      = np.dot(dx_post, y_.T)
        # dy_final
        dy_     = np.dot(K.T, dx_post)

        """
        2. 
        P = (I - KH)P
        """
        dp_pred_2 = np.dot(IKH.T, dp_post)
        dIKH      = np.dot(dp_post, p_pred.T)
        dK_2      = -np.dot(dIKH, H.T)
        dH_2      = -np.dot(K.T, dIKH)

        dK = dK_1 + dK_2
        """
        3. 
        K = pHTS-1
        """
        dS_inv = np.dot(PHT.T, dK)
        S_inv_T = np.linalg.inv(S.T)
        tmp = -np.kron(S_inv_T, S_inv)
        flat = dS_inv.reshape(1,9)
        # dS Final
        dS = (np.dot(tmp, flat.T)).reshape((3,3))

        d_PHT = np.dot(dK, S_inv.T)

        dp_pred_3 = np.dot(d_PHT, H)
        dH_3    = (np.dot(p_pred.T, d_PHT)).T

        dH = dH_2 + dH_3
        dp_pred = dp_pred_2 + dp_pred_3

        return dx_pred, dp_pred, dH, dS, dy_

    def measurement_forward(self,
                           x_pred, 
                           p_pred, 
                           z, 
                           R):
    
        H = self.gen_H(x_pred)
        PHT = np.dot(p_pred, H.transpose())  
        S   = np.dot(H, PHT) + R

        y_ = np.array([z[0], 
                       z[1],
                       z[2]])[:, None] - self.gen_h(x_pred)

        y_.itemset(2, angle_difference(z[2], (self.gen_h(x_pred).item(2)) % (2 * np.pi)))

        cache = (R, H, PHT, S, y_, x_pred, p_pred, z)
        return y_, S, H, cache

    def measurement_backward(self, dy_, dS, dH, cache): 
        '''
        Need to return
        dx_pred
        dp_pred
        dR    
        '''
        R, H, PHT, S, y_, x_pred, p_pred, z = cache
        dR = dS
        dPHT = np.dot(H.T, dS)
        dp_pred = np.dot(dPHT, H)

        dHT = np.dot(p_pred.T, dPHT)
        dH += np.dot(dS, PHT.T) + dHT.T

        dx_pred   = np.zeros(5)
        """
        l = EKF_filter.imu_off_l
        alpha = EKF_filter.imu_alpha
        """
        l = self.imu_off_l
        alpha = self.imu_alpha

        dx_pred[2] +=  l * cos(x_pred[2] - alpha) * dH[0,2]
        dx_pred[2] +=  l * sin(x_pred[2] - alpha) * dH[1,2]

        dx_pred = np.add(-np.dot(H.T, dy_).T, dx_pred).T
        #print(dx_pred)
        return dx_pred, dp_pred, dR 

    def prediction_forward(self, x_post, p_post, dt, Q_acc, Q_other):
        '''
        x_pred = f(x_post)
        p_pred = FPF + Q
        Q = B * sigma * B + Q_other
        '''
        B = self.gen_B_simple(x_post, dt)
        F = self.G(x_post, dt)
        FP    = np.dot(F, p_post)
        FPFt  = np.dot(FP, F.transpose())

        QBT = np.dot(Q_acc, B.T)
        Q   = np.dot(B, QBT) + Q_other
        p_pred = np.add(FPFt, Q)
        x_pred = self.predict(x_post, dt)

        cache = (B, F, FP, FPFt, QBT, Q, x_post, p_post, dt, Q_acc, Q_other)

        return x_pred, p_pred, cache

    def prediction_backward(self, dx_pred, dp_pred, cache):
        '''
        should output the deraitive
        dx_post : state 
        dp_post : cov
        dQ_acc  : acceleration
        dQ_other: process Noise
        '''

        B, F, FP, FPFt, QBT, Q, x_post, p_post, dt, Q_acc, Q_other = cache

        # Get dF
        dFP = np.dot(dp_pred, F)
        dF_1 = (np.dot(FP.T, dp_pred)).T
        dF_2 = np.dot(dFP, p_post.T)
        dF = dF_1 + dF_2

        # Get the dp_post
        dp_post = np.dot(F.T, dFP)
        dx_post = self.get_dx_post_int_prediction(x_post, dF, dt)

        # Get dQ
        dQ = dp_pred
        dQ_other = dQ
        dQBT      = np.dot(B.T, dQ)
        dQ_acc    = np.dot(dQBT, B)
        dBT       = np.dot(Q_acc.T, dQBT)
        dB        = np.dot(dQ, QBT.T) + dBT.T

        dx_post[2] += -0.5 * (dt**2) * sin(x_post[2]) * dB[0,0]
        dx_post[2] +=  0.5 * (dt**2) * cos(x_post[2]) * dB[1,0]

        ## From Eq1
        dx_post += np.dot(F.T ,dx_pred)


        return dx_post, dp_post, dQ_acc, dQ_other

    def get_dx_post_int_prediction(self, x_post, dF, dt):
        dx_post = np.zeros(x_post.shape)
        wheelbase = self.h

        state = x_post
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        # dF[0,2]
        dth_F02  = -cos(th) * cos(phi) * v * dt * dF[0,2]
        dphi_F02 =  sin(th) * sin(phi) * v * dt * dF[0,2]
        dv_F02   = -sin(th) * cos(phi) *     dt * dF[0,2]
        # dF[0,3]
        dth_F03  =  -sin(th) * cos(phi) * dt * dF[0,3]
        dphi_F03 =  -cos(th) * sin(phi) * dt * dF[0,3]
        dv_F03   =  0.0 
        # dF[0,4]
        dth_F04  =   sin(th) * sin(phi) * v * dt * dF[0,4]
        dphi_F04 =  -cos(th) * cos(phi) * v * dt * dF[0,4]
        dv_F04   =  -cos(th) * sin(phi)     * dt * dF[0,4]
        # dF[1, 2]
        dth_F12  =  -sin(th) * cos(phi) * v * dt * dF[1,2]
        dphi_F12 =  -cos(th) * sin(phi) * v * dt * dF[1,2]
        dv_F12   =   cos(th) * cos(phi)     * dt * dF[1,2]

        # dF[1, 3]
        dth_F13  =  cos(th) * cos(phi) * dt * dF[1,3]
        dphi_F13 = -sin(th) * sin(phi) * dt * dF[1,3]
        dv_F13   =  0.0

        # dF[1, 4]
        dth_F14  = -cos(th) * sin(phi) * v * dt * dF[1,4]
        dphi_F14 = -sin(th) * cos(phi) * v * dt * dF[1,4]
        dv_F14   = -sin(th) * sin(phi) *     dt * dF[1,4]

        # dF[2, 3]
        dth_F23  = 0.0
        dphi_F23 = (cos(phi) * dt / wheelbase) * dF[2,3]
        dv_F23   = 0.0

        # dF[2, 4]
        dth_F24  = 0.0
        dphi_F24 = (-sin(phi) * v * dt / wheelbase) * dF[2,4]
        dv_F24   = ( cos(phi)     * dt / wheelbase) * dF[2,4]

        dx_post[2] = dth_F02 + dth_F03 + dth_F04 + dth_F12 + dth_F13 + \
                     dth_F14 + dth_F23 + dth_F24 
        dx_post[4] = dphi_F02 + dphi_F03 + dphi_F04 + \
                     dphi_F12 + dphi_F13 + dphi_F14 + \
                     dphi_F23 + dphi_F24
        dx_post[3] = dv_F02 + dv_F03 + dv_F04 + \
                     dv_F12 + dv_F13 + dv_F14 +\
                     dv_F23 + dv_F24
        dx_post_tmp = np.random.randn(5,1)
        dx_post_tmp[0,0] = 0.0
        dx_post_tmp[1,0] = 0.0
        dx_post_tmp[2,0] = dx_post[2]
        dx_post_tmp[3,0] = dx_post[3]
        dx_post_tmp[4,0] = dx_post[4]

        return dx_post_tmp

    # Use this function to genreate the back propagation Loss Function
    def generate_grad(self):
        #np.set_printoptions(precision=4, suppress=True)
        
        T = len(self.headings)
        #print("Liang Calculate the gradients", str(T))
        # Ground Truth 
        x_gt = self.datasets.afterSmooth.x[0::self.config["interval"]]
        x_gt = np.array(x_gt[0:len(self.locs)])

        for i in range(0 , T):
            self.dH[i] = {
                "dz_meas" : np.zeros((3,1)),
                "dp_meas" : np.zeros((3,3)), 
                "dx_post" : np.zeros((5,1)),
                "dp_post" : np.zeros((5,5)) 
            }

            """
            self.forward_cache[i] = {
                'x_pred' : x_pred,
                'p_pred' : p_pred,
                'z'      : z,
                'cache_meas' : cache_meas,
                'x_post' : x_post,
                'p_post' : p_post,
                'cache_update': cache_update,
                'cache_pred'  : cache_pred
                'y_'          : y_,
                'S'           : S
            }
            """
            # Output from the Filter
            y_     = self.forward_cache[i]["y_"]
            S      = self.forward_cache[i]["S"]
            x_post = self.forward_cache[i]["x_post"]
            p_post = self.forward_cache[i]["p_post"]

            ## Error corrention        

            if self.config["Likelihood_post"] == True: 
                #print("Liang Calculate the gradients")
                gt_loc   = x_gt[i]
                post_loc = x_post
                residual = gt_loc - post_loc
                residual[2] = angle_difference(gt_loc[2], post_loc[2])
                post_sigma_inv = np.linalg.inv(p_post)
                # dx_post_current step
                dx_post_current = - np.dot(post_sigma_inv, residual)
                # dp_post current step
                insider = np.dot(post_sigma_inv.dot(residual.dot(residual.T)),post_sigma_inv)
                dp_post_current = 0.5 * (post_sigma_inv - insider )
                """
                print(residual)
                print(dx_post_current)
                print(dp_post_current)
                """
                self.dH[i]["dx_post"] += dx_post_current
                self.dH[i]["dp_post"] += dp_post_current

            if self.config["Likelihood_measurement"] == True: 
                residual = y_
                meas_sigma_inv = np.linalg.inv(S)
                # dz_meas
                dz_meas_current = meas_sigma_inv.dot(residual)
                insider = np.dot(meas_sigma_inv.dot(residual.dot(residual.T)), 
                                meas_sigma_inv)
                dp_meas_current = - 0.5 * (meas_sigma_inv - insider)

                self.dH[i]["dz_meas"] += dz_meas_current
                self.dH[i]["dp_meas"] += dp_meas_current

            if self.config["RMS_measurement"] == True:
                self.dH[i]["dz_meas"] +=  y_

            if self.config["RMS_post"] == True: 
                gt_loc   = x_gt[i]
                post_loc = x_post
                residual = gt_loc - post_loc
                residual[2] = angle_difference(gt_loc[2], post_loc[2])
                self.dH[i]["dx_post"] += -residual

    def totalLoss(self):
        T = len(self.headings)
        # Ground Truth 
        x_gt = self.datasets.afterSmooth.x[0::self.config["interval"]]
        x_gt = np.array(x_gt[0:len(self.locs)])
        RMS_state = 0.0
        RMS_meas  = 0.0
        logLikelihood_state = 0.0
        logLikelihood_meas  = 0.0 

        counter = 0 
        for i in range(5 , T):
            counter += 1
            # Output from the Filter
            y_     = self.forward_cache[i]["y_"]
            S      = self.forward_cache[i]["S"]
            x_post = self.forward_cache[i]["x_post"]
            p_post = self.forward_cache[i]["p_post"]

            gt_loc   = x_gt[i]
            post_loc = x_post
            residual = gt_loc - post_loc
            residual[2] = angle_difference(gt_loc[2], post_loc[2])
            RMS_state += ((residual.dot(residual.T)).trace()).item()
            RMS_meas  += (y_.dot(y_.T)).trace()

            zeros = np.zeros(5)
            like_states = multivariate_normal.logpdf(residual.T, 
                                              mean = zeros.T, 
                                              cov = p_post,
                                              allow_singular = False)
            logLikelihood_state += like_states
            zeros = np.zeros(3)
            like_meas = multivariate_normal.logpdf(y_.T, 
                                              mean = zeros.T, 
                                              cov = S,
                                              allow_singular = False)
            #if like_states < -100 :
            #    print(residual)
            logLikelihood_meas += like_meas
        
        loss = {
            "RMS_state"            :  RMS_state / counter,
            "RMS_meas"             :  RMS_meas / counter,
            "logLikelihood_state"  :  -logLikelihood_state / counter,
            "logLikelihood_meas"   :  -logLikelihood_meas / counter
        }
        return loss

    def plot_overview(self):
        fig = plt.figure(figsize=(12, 16))
        ax = fig.add_subplot(111)
        

        loc = np.array(self.x)
        plt.scatter(loc[:,0], loc[:,1], c='y', s=50, marker="1", 
                label="{}".format("After Kalman Filter"))
        plt.plot(loc[:,0], loc[:,1], c='y')        

        loc = np.array(self.bias_locs)
        plt.scatter(loc[:,0], loc[:,1], c='r', s=50, marker="x", 
                label="{}".format("Noise Measurements"))

        # Print Ground Truth Position
        if self.datasets.afterSmooth is not None:
            loc = self.datasets.afterSmooth.x[0::self.config["interval"]]
            loc = np.array(loc[0:len(self.locs)])
            plt.scatter(loc[:,0], loc[:,1], c='k', s=50, marker="2", 
                    label="{}".format("Ground Truth"))
            plt.plot(loc[:,0], loc[:,1], c='k')
        else: 
            loc = np.array(self.locs)
            plt.scatter(loc[:,0], loc[:,1], c='b', s=50, marker="+", 
                    label="{}".format("Center Measurements"))
            plt.plot(loc[:,0], loc[:,1], c='b')


        ax.axis('equal')
        ax.legend()
        plt.show()



