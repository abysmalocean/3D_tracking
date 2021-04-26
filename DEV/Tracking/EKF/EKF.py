import numpy as np
from math import sin, sqrt, cos
from scipy.stats import multivariate_normal

from utils.utils import angle_difference
from matplotlib import pyplot as plt


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

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
    def __init__(self, tracking_name = 'car'):
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
        
        return np.array([x + self.wheelbase_to_length_ratio * l * cos(th) / 2.0,
                         y + self.wheelbase_to_length_ratio * l * sin(th) / 2.0, 
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
        H[0, 2] = self.wheelbase_to_length_ratio / 2.0 * l * cos(th)
        H[0, 6] = self.wheelbase_to_length_ratio / 2.0 * sin(th)
        # for y
        H[1, 1] = 1.0
        H[1, 2] = -self.wheelbase_to_length_ratio / 2.0 * l * sin(th)
        H[1, 6] = self.wheelbase_to_length_ratio / 2.0 * cos(th)
        
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
              maxacc = 3.0, 
              maxsteeringacc = 0.6):
        
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
    
    def predict(self, dt):
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
        temp_x =  np.array([x + self.wheelbase_to_length_ratio * l * cos(th) / 2.0,
                            y + self.wheelbase_to_length_ratio * l * sin(th) / 2.0,
                            th + sin(phi) * v * dt / (l * self.wheelbase_to_length_ratio),
                            v, 
                            phi,
                            l,
                            w ])[:, None]
        
        # predict the covariance
        tmpG = self.G(state, dt)
        R_val = self.new_R(state, dt)
        PropCov = np.dot(np.dot(tmpG, self.P), tmpG.transpose()) + R_val
        self.x = temp_x
        self.P = PropCov
        
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
        mean = np.zeros(y_.shape[0])
        prob = multivariate_normal.logpdf(y_.T, 
                                          mean = mean, 
                                          cov = self.S,
                                          allow_singular = True)
        
        return prob
        
    
    def update(self, detection):
        #print("measurement ", detection[0], detection[1])
        y_ = detection - self.z_pred
        y_.itemset(2, angle_difference(detection.item(2),
                                      self.z_pred.item(2)))
        #print(y_)
        # dubug proposes
        #if (abs(y_.item(2)) > 1.0):
        #    print(y_.item(2))
        
        self.P = np.dot((np.identity(7) - np.dot(self.K, self.H)), self.P)
        self.x = self.x + np.dot(self.K, y_)
        self.post_x.append(self.x)
        self.post_p.append(self.P)

class EKF_old(object): 
    def __init__(self):
        super(EKF, self).__init__()
        self.ftimes   = []
        self.locs     = []
        self.headings = []
        self.shapes   = []
        self.angle    = []
        self.dist     = []
        self.n_lidar_points = []
         

    def load_data_set(self, datasets): 
        self.ftimes = datasets.timestamps_second
        self.w = datasets.w
        self.h = datasets.h
        self.imu_off = datasets.offset

        self.imu_alpha = np.arctan(self.imu_off[0] / self.imu_off[1])
        self.imu_off_l = np.sqrt(self.imu_off[0] ** 2 + self.imu_off[1] ** 2)
        for items in datasets.oxts:
            heading = items.heading
            locs = items.Trans
            self.bias_locs.append([locs[0], locs[1]])
            x = locs[0] + self.imu_off_l * np.cos(heading - self.imu_alpha)
            y = locs[1] + self.imu_off_l * np.sin(heading - self.imu_alpha)
            self.locs.append([x,y])

        self.bias_locs = datasets.locs
        self.headings  = datasets.heading

        # Update the measurement covariance
        self.GPS_x  = datasets.GPS_x 
        self.GPS_y  = datasets.GPS_y 
        self.GPS_h  = datasets.GPS_h 
    
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

    def R(self, dt, maxacc = 3.0, maxsteeringacc = 0.6):

        #maxacc = 3.0
        maxyawacc = 0.1
        #maxsteeringacc = 0.6
        return np.diag(np.array([(0.5 * maxacc * dt**2)**2     ,
                                 (0.5 * maxacc * dt**2)**2     ,
                                 (0.5 * maxyawacc * dt**2)**2  ,
                                 (maxacc * dt)**2              ,
                                 (maxsteeringacc * dt)**2]))
    
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


    def run_kf(self, maxacc = 3.0, maxteeringacc = 0.6): 
        """
        This functions is run the KF with the predefined R
        """

        kf_result_obj = After_filter()

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
        #p = H.shape[0]

        V = []
        W = []
        W_meas = []
        x = []
        x_hat = []
        A = []
        Factor = []


        Pred = x0
        PropCov = P0

        NSteps = len(meas_times)

        Q = np.diag(np.array([(self.GPS_x)**2, 
                              (self.GPS_y)**2, 
                              (self.GPS_h)**2]))

        for i in range(NSteps): 
            y_ = np.array([self.bias_locs[i][0], 
                           self.bias_locs[i][1],
                           self.headings[i]])[:, None] - self.gen_h(Pred)
            H = self.gen_H(Pred)
            
            y_.itemset(2, angle_difference(self.headings[i],
                                (self.gen_h(Pred).item(2)) % (2 * np.pi)))

            S = np.dot(np.dot(H, PropCov), H.transpose()) + Q
            K = np.dot(np.dot(PropCov, H.transpose()), np.linalg.inv(S))

            # keep a record
            kf_result_obj.S.append(S)
            kf_result_obj.K.append(K)

            updated_x = Pred + np.dot(K, y_)

            x_hat.append(Pred)
            x.append(updated_x)
            V.append(np.dot((np.identity(n) - np.dot(K, H)), PropCov))
            W.append(PropCov)
            W_meas.append(S)

        # prediction phase

            if i < NSteps - 1: 
                dt = meas_times[i+1] - meas_times[i]
                Pred = self.predict(x[-1], dt)
                tmpG = self.G(x[-1], dt)
                #R_val = self.R(dt)
                R_val = self.new_R(x[-1], dt)
                PropCov = np.dot(np.dot(tmpG, V[-1]), tmpG.transpose()) + R_val
                A.append(tmpG)
        
        for i in range(NSteps):
            kf_result_obj.locs.append(self.convertLocs(x[i]))
            kf_result_obj.vels.append(self.convertVels(x[i]))
            kf_result_obj.headings.append(self.convertHeading(x[i]))
        kf_result_obj.x = x
        kf_result_obj.W = W
        kf_result_obj.W_meas = W_meas
        kf_result_obj.V = V
        kf_result_obj.x_hat = x_hat
        self.A = A
        self.afterKF = kf_result_obj

    def smoother(self): 
        meas_times = self.ftimes
        NSteps = len(meas_times)
        kf_result_obj = None
        kf_result_obj = After_filter()

        x_T = []
        p_T = []
        x = self.afterKF.x
        V = self.afterKF.V
        W = self.afterKF.W
        A = self.A
        for i in range(NSteps - 1 , -1, -1):
            if i == NSteps - 1:
                x_T.insert(0, self.afterKF.x[i])
                p_T.insert(0, self.afterKF.V[i])
            else: 
                Jt = np.dot(np.dot(V[i], A[i].transpose()), np.linalg.inv(W[i + 1]))
                
                x_T.insert(0, x[i] + np.dot(Jt, (x_T[0] - self.predict
                                 (x[i], meas_times[i + 1] - meas_times[i]))))
            
                p_T.insert(0, V[i] + np.dot(np.dot(Jt, (p_T[0] - W[i+1])), \
                                            Jt.transpose()))
        kf_result_obj.x = x_T
        kf_result_obj.V = p_T
        self.afterSmooth = kf_result_obj

    def plot_overview(self, date = '2011_09_26', drive = '0015'):
        fig = plt.figure(figsize=(12, 16),num='date {} drive {}'.format(date, drive))
        ax = fig.add_subplot(111)
        loc = np.array(self.locs)
        plt.scatter(loc[:,0], loc[:,1], c='b', s=50, marker="+", 
                label="{}".format("Center Measurements"))
        plt.plot(loc[:,0], loc[:,1], c='b')

        loc = np.array(self.bias_locs)
        plt.scatter(loc[:,0], loc[:,1], c='r', s=50, marker="x", 
                label="{}".format("Biased Measurements"))
        
        loc = np.array(self.afterKF.x)
        plt.scatter(loc[:,0], loc[:,1], c='y', s=50, marker="1", 
                label="{}".format("After Kalman Filter"))
        plt.plot(loc[:,0], loc[:,1], c='y')


        loc = np.array(self.afterSmooth.x)
        plt.scatter(loc[:,0], loc[:,1], c='k', s=50, marker="2", 
                label="{}".format("After Smooth Filter"))
        plt.plot(loc[:,0], loc[:,1], c='k')
        ax.axis('equal')
        ax.legend()
        plt.show()
        





