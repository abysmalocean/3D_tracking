import numpy as np
from math import sin, sqrt, cos
import sys
sys.path.append("..")

from yaml_data import *
from utils.time import utime_to_ftime 
from plot import error_ellipse
from plot import error_ellipse_std

from utility import wheelbase_to_length_ratio, rawrtc_to_l_ratio
from utility import get_angle, angle_difference
from utility import find_closest_corner, two_point_length
from utility import Car_state
import utility 

from matplotlib import pyplot as plt
from matplotlib import patches 
from matplotlib.patches import Ellipse
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from matplotlib.widgets import RadioButtons
from scipy.linalg import cholesky
import ukf

import copy


import IPython as ipy


#matplotlib.use('TkAgg')
np.set_printoptions(precision=5, suppress=True)


from limites import *


from tqdm import tqdm

import IPython as ipy

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


class EKF(object):
    def __init__(self):
        super(EKF, self).__init__()
        self.ftimes   = [] 
        self.locs     = []
        self.headings = []
        self.w        = []
        self.h        = []
        self.dected_corner = []
        self.meas_cov = []
        self.L_shape_ends = []

        self.truths   = {}
        self.lshapes  = {}
        self.utime    = []

        self.afterKF = None
        self.afterSmooth = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.global_plot_index = 0
        self.ground_truth = None
        self.current_lines = []
        self.roation_slider = None
        self.error_play_meas = [0.0, 0.0, 0.0]
        self.error_play_lines = []


    def build_measurments(self, truth, lshape):
        self.truths  = truth
        self.lshapes = lshape
        length = len(self.truths)
        self.locs = np.zeros((length, 2))
        self.headings = np.zeros(length)
        self.w = np.zeros(length)
        self.h = np.zeros(length)
        self.dected_corner = np.zeros(length, dtype=int)
        self.L_shape_ends  = np.zeros(length,  dtype=int)
        index = 0

        for utime, l_meas in self.lshapes.items():
            self.utime.append(utime)
            self.ftimes.append(utime_to_ftime(utime))

            # First build un-biased measurements x y
            bias = l_meas.bias
            corners = l_meas.corners
            self.locs[index][0] = corners[1][0] + bias[0]
            self.locs[index][1]=  corners[1][1] + bias[1]
            
            # Build the heading
            ## 1. get the truth heading from the ground truth
            point_4 = self.truths[utime].corners[3,0:2]
            point_1 = self.truths[utime].corners[0,0:2]
            heading_truth = get_angle(point_4, point_1)
            
            ## 2. Find the closest corner on the box
            corners = l_meas.corners
            box     =self.truths[utime].corners
            cloest_corner_index = find_closest_corner(corners, box)
            self.dected_corner[index] = cloest_corner_index
            
            if cloest_corner_index < 2:
                heading_meas_h1 = get_angle(corners[2], corners[1])
                heading_meas_h2 = get_angle(corners[0], corners[1])
            else:
                heading_meas_h1 = get_angle(corners[1], corners[2])
                heading_meas_h2 = get_angle(corners[1], corners[0])

            angle_diff1 = np.abs(angle_difference(heading_truth, 
                                                  heading_meas_h1))
            angle_diff2 = np.abs(angle_difference(heading_truth, 
                                                  heading_meas_h2))
            
            if (angle_diff1 > angle_diff2):
                self.headings[index] = heading_meas_h2
            else:
                self.L_shape_ends[index] = 2
                self.headings[index] = heading_meas_h1

            # build the w and h
            h = two_point_length(box[0], box[3]) 
            w = two_point_length(box[0], box[1])
            self.w[index] = w
            self.h[index] = h

            # build the measurements uncertainty
            self.meas_cov.append(l_meas.cov)

            # increase the index
            index += 1

    def predict(self, state, dt): 
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        w = state.item(5)
        h = state.item(6)

        wheelbase = wheelbase_to_length_ratio() * self.h[0]

        return np.array([x + cos(th) * cos(phi) * v * dt,
                         y + sin(th) * cos(phi) * v * dt,
                         th + sin(phi) * v * dt / wheelbase,
                         v,
                         phi,
                         w,
                         h])[:, None]
    
    def convertLocs(self, state):
        x = state.item(0)
        y = state.item(1)
        return ((x , y))
    
    def convertHeading(self, state):
    
        return state.item(2) % (2 * math.pi)

    def convertVels(self, state):
        
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        return (v * cos(phi) * cos(th), v * cos(phi) * sin(th))



    def generate_H(self, states ,corner):
        """
        measurement function, and corner is the association corner. 
        
        """
        if corner > 3: 
            raise("corner should less than 3")
        
        # Cheacting Use the annotation W and H
        x = states[0]
        y = states[1]

        theta = states[2]
        w = states[5]
        h = states[6]
        
        l_1 = np.sqrt((0.5 * w)**2 + (wheelbase_to_length_ratio() * h)**2)
        l_2 = np.sqrt((0.5 * w)**2 + ((rawrtc_to_l_ratio()) * h)**2)

        alpha = np.arctan((0.5 * w)/(wheelbase_to_length_ratio() * h))
        beta  = np.arctan((0.5 * w)/(rawrtc_to_l_ratio() * h))

        if corner == 0:
            l = l_1
            angle = theta - alpha
        elif corner == 1: 
            l = l_1
            angle = theta + alpha
        elif corner == 2:
            l = l_2
            angle = theta - np.pi - beta
        elif corner == 3: 
            l = l_2
            angle = theta - np.pi + beta
        
        project_x = 1 + (l * np.cos(angle)) / x
        project_y = 1 + (l * np.sin(angle)) / y
        H = np.zeros((5,7))
        H[0,0] = project_x
        H[1,1] = project_y
        H[2,2] = 1.0
        H[3,5] = 1.0
        H[4,6] = 1.0

        return H

    def generate_dH(self, states ,corner):
        """
        measurement function, and corner is the association corner. 
        
        """
        if corner > 3: 
            raise("corner should less than 3")
        
        # Cheacting Use the annotation W and H
        x = states[0]
        y = states[1]

        theta = states[2]
        w = states[5]
        h = states[6]
        
        l_1 = np.sqrt((0.5 * w)**2 + (wheelbase_to_length_ratio() * h)**2)
        l_2 = np.sqrt((0.5 * w)**2 + ((rawrtc_to_l_ratio()) * h)**2)
        
        alpha = np.arctan((0.5 * w)/(wheelbase_to_length_ratio() * h))
        beta  = np.arctan((0.5 * w)/(rawrtc_to_l_ratio() * h))

        if corner == 0:
            l = l_1
            angle = theta - alpha
        elif corner == 1: 
            l = l_1
            angle = theta + alpha
        elif corner == 2:
            l = l_2
            angle = theta - np.pi - beta
        elif corner == 3: 
            l = l_2
            angle = theta - np.pi + beta
        
        project_x = 1 + (l * np.cos(angle)) / x
        project_y = 1 + (l * np.sin(angle)) / y
        H = np.zeros((5,7))
        
        H[0,0] = 1.0
        H[0,2] = -l * np.sin(angle)
        #H[0,3] = (0.5)**2 * (x / l) * np.cos(angle)
        #H[0,4] = (0.7)**2 * (y / l) * np.cos(angle)
        H[1,1] = 1.0
        H[1,2] = l * np.cos(angle)
        #H[1,3] = (0.5)**2 * (x / l) * np.sin(angle)
        #H[1,4] = (0.7)**2 * (y / l) * np.sin(angle)
        
        H[2,2] = 1.0
        H[3,5] = 1.0
        H[4,6] = 1.0

        return H
    
    def G(self, state, dt):
        x = state.item(0)
        y = state.item(1)
        th = state.item(2)
        v = state.item(3)
        phi = state.item(4)
        w = state.item(5)
        h = state.item(6)


        wheelbase = wheelbase_to_length_ratio() * h

        G_matrix = np.matrix([[1, 0, -sin(th) * cos(phi) * v * dt, 
                                      cos(th) * cos(phi) * dt, 
                                     -cos(th) * sin(phi) * v * dt, 0, 0],
                              [0, 1, cos(th) * cos(phi) * v * dt, 
                                     sin(th) * cos(phi) * dt, 
                                     -sin(th) * sin(phi) * v * dt, 0, 0],
                              [0, 0, 1, sin(phi) * dt / wheelbase, 
                                        cos(phi) * v * dt / wheelbase, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0 ,0], 
                              [0, 0, 0, 0, 0, 1 ,0],
                              [0, 0, 0, 0, 0, 0 ,1]])
        return G_matrix

    def newBabylity_R(self, state, dt, maxacc = 0.1, maxsteeringacc = 0.03):
        
        L = self.h[0]

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


    def R(self, dt, maxacc = 3.0, maxsteeringacc = 0.6):
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


    def run_kf(self, maxacc = 3.0, maxsteeringacc = 0.6, newR = False):
        """
        This function is run the KF with the predefined R
        Result are saved in a object
        """
        kf_result_obj = None
        kf_result_obj = After_filter()

        # initial state estimation
        ## 1. transfer the corner to the rare center
        ## Center is initialzed in a simple way, it initialized using the 
        ##  center of the L shape
        x0_x0 = self.lshapes[self.utime[0]].corners[0]
        x0_x2 = self.lshapes[self.utime[0]].corners[2]
        x0_x  = (x0_x0[0] + x0_x2[0]) / 2.0 
        x0_y  = (x0_x0[1] + x0_x2[1]) / 2.0
        # State x ,y, theta, speed, steering_speed, width, height # cheating a little bit by using the width and height from annotation # 
        x0 = np.array([x0_x , 
                       x0_y , 
                       self.headings[0], 
                       0.0, 0.0, 
                       self.w[0], 
                       self.h[0]])[:, None] 
        
        if len(self.ftimes) > 1: 
            # better estimation of the initial velocity 
            x0.itemset(3, sqrt((self.locs[1][0] - self.locs[0][0])**2 + \
                (self.locs[1][1] - self.locs[0][1])**2))

        # Initial the Process Noise in the begining
        #
        P0 = np.diag(np.array([1.**2, 
                               1.**2, 
                               (math.pi * 10 / 180.0)**2, 
                               (10**2), 
                               (0.1 * 0.1), 
                               (0.001) ** 2,
                               (0.001) ** 2]))
        
                            
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

        for i in range(NSteps):
            # H is created from the predicted states and the 
            #   corner it associated

            H = self.generate_H(Pred, self.dected_corner[i])
            y_ = np.array([self.locs[i][0], 
                           self.locs[i][1],
                           self.headings[i], 
                           self.w[0],
                           self.h[0]])[:, None] - np.dot(H, Pred)
            y_.itemset(2, angle_difference(self.headings[i],
                                (((np.dot(H, Pred)).item(2)) % (2 * math.pi))))
            Q = np.zeros((5, 5))
            Q[0:3, 0:3] = self.meas_cov[i]
            Q[3][3] = 0.1 ** 2
            Q[4][4] = 0.1 ** 2
            dH =self.generate_dH(Pred, self.dected_corner[i])

            S = np.dot(np.dot(dH, PropCov), dH.transpose()) + Q
            K = np.dot(np.dot(PropCov, dH.transpose()), np.linalg.inv(S))

            # keep a record
            kf_result_obj.S.append(S)
            kf_result_obj.K.append(K)

            updated_x = Pred + np.dot(K, y_)

            if (updated_x.item(4) < -2 * math.pi / 6):
                updated_x.itemset(4, -2 * math.pi / 6)
            if (updated_x.item(4) > 2 * math.pi / 6):
                updated_x.itemset(4, 2 * math.pi / 6)
            
            x_hat.append(Pred)
            x.append(updated_x)
            V.append(np.dot((np.identity(n) - np.dot(K, dH)), PropCov))
            W.append(PropCov)
            W_meas.append(S)
            #W.append(S)
            Factor.append((np.identity(n) - np.dot(K, dH)))

            # prediction phase
            if i < NSteps - 1: 
                dt = meas_times[i+1] - meas_times[i]
                Pred = self.predict(x[-1], dt)
                tmpG = self.G(x[-1], dt)
                
                R_val = None
                #ipy.embed()
                if newR == False:
                    R_val = self.R(dt, 
                                   maxacc=maxacc, 
                                   maxsteeringacc = maxsteeringacc)
                else:
                    # TODO: Need to check this part

                    R_val = self.newBabylity_R(x[-1], 
                                               dt,
                                               maxacc = maxacc, 
                                               maxsteeringacc = maxsteeringacc)

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
    
    def run_IEKF(self, maxacc = 3.0, maxsteeringacc = 0.6, newR = False):
        """
        This function is run the KF with the predefined R
        Result are saved in a object
        """
        kf_result_obj = None
        kf_result_obj = After_filter()

        # initial state estimation
        ## 1. transfer the corner to the rare center
        ## Center is initialzed in a simple way, it initialized using the 
        ##  center of the L shape
        x0_x0 = self.lshapes[self.utime[0]].corners[0]
        x0_x2 = self.lshapes[self.utime[0]].corners[2]
        x0_x  = (x0_x0[0] + x0_x2[0]) / 2.0 
        x0_y  = (x0_x0[1] + x0_x2[1]) / 2.0
        # State x ,y, theta, speed, steering_speed, width, height # cheating a little bit by using the width and height from annotation # 
        x0 = np.array([x0_x , 
                       x0_y , 
                       self.headings[0], 
                       0.0, 0.0, 
                       self.w[0], 
                       self.h[0]])[:, None] 
        
        if len(self.ftimes) > 1: 
            # better estimation of the initial velocity 
            x0.itemset(3, sqrt((self.locs[1][0] - self.locs[0][0])**2 + \
                (self.locs[1][1] - self.locs[0][1])**2))

        # Initial the Process Noise in the begining
        #
        P0 = np.diag(np.array([1.**2, 
                               1.**2, 
                               (math.pi * 10 / 180.0)**2, 
                               (10**2), 
                               (0.1 * 0.1), 
                               (0.001) ** 2,
                               (0.001) ** 2]))
        
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

        for i in range(NSteps):
            # H is created from the predicted states and the 
            #   corner it associated
            Prev_pred = copy.deepcopy(Pred) + np.Inf
            x_opt     = copy.deepcopy(Pred)
            num_iter = 0
            while (utility.state_error(Prev_pred, x_opt) > \
                  utility.iterative_threshold()) and num_iter < 10:
                H = self.generate_H(x_opt, 
                                self.dected_corner[i])
                dH = self.generate_dH(x_opt, 
                                    self.dected_corner[i])

                y_ = np.array([self.locs[i][0], 
                               self.locs[i][1],
                               self.headings[i], 
                               self.w[0],
                               self.h[0]])[:, None] - np.dot(H, x_opt)
                y_.itemset(2, angle_difference(self.headings[i],
                                (((np.dot(H, x_opt)).item(2)) % (2 * math.pi))))

                Q = np.zeros((5, 5))
                Q[0:3, 0:3] = self.meas_cov[i]
                Q[3][3] = 0.0001 ** 2
                Q[4][4] = 0.0001 ** 2
                S = np.dot(np.dot(dH, PropCov), dH.transpose()) + Q
                K = np.dot(np.dot(PropCov, dH.transpose()), np.linalg.inv(S))
                bias = np.dot(dH, (Pred - x_opt))
                updated_x = Pred + np.dot(K, (y_ - bias))
                
                Prev_pred = x_opt
                x_opt = updated_x
                num_iter += 1
            #print("Number Iteration ", num_iter)
            # keep a record
            kf_result_obj.S.append(S)
            kf_result_obj.K.append(K)

            if (updated_x.item(4) < -2 * math.pi / 6):
                updated_x.itemset(4, -2 * math.pi / 6)
            if (updated_x.item(4) > 2 * math.pi / 6):
                updated_x.itemset(4, 2 * math.pi / 6)
            
            x_hat.append(Pred)
            x.append(updated_x)
            V.append(np.dot((np.identity(n) - np.dot(K, dH)), PropCov))
            W.append(PropCov)
            W_meas.append(S)
            #W.append(S)
            Factor.append((np.identity(n) - np.dot(K, dH)))

            # prediction phase
            if i < NSteps - 1: 
                dt = meas_times[i+1] - meas_times[i]
                Pred = self.predict(x[-1], dt)
                tmpG = self.G(x[-1], dt)
                
                R_val = None
                #ipy.embed()
                if newR == False:
                    R_val = self.R(dt, 
                                   maxacc=maxacc, 
                                   maxsteeringacc = maxsteeringacc)
                else:
                    # TODO: Need to check this part

                    R_val = self.newBabylity_R(x[-1], 
                                               dt,
                                               maxacc = maxacc, 
                                               maxsteeringacc = maxsteeringacc)

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

    
    def plot_one_frame(self, box, corners, cluster, ax = None):
        
        if ax is None:
            ax = plt.gca()

        h3, = ax.plot(cluster[:, 0], cluster[:, 1], 'k.')
        box = box[range(-1, box.shape[0]), :]
        h1, = ax.plot(box[:, 0], box[:, 1], 'b', label = "Ground Truth")
        h2, = ax.plot(corners[:, 0], 
                       corners[:, 1], 
                       'r', 
                       label = "L Measurements")
        return h1, h2, h3
    
    def next(self, event): 
        tmp = self.global_plot_index + 1
        self.global_plot_index = min(len(self.afterKF.x) -1, tmp)
        print("current index ", self.global_plot_index)
        if self.global_plot_index == tmp: 
            self.draw()
    
    def prev(self, event):
        self.global_plot_index = max(0, self.global_plot_index - 1)
        print("current index ", self.global_plot_index)
        self.draw()
    
    def close(self, event): 
        plt.close()
    
    def print_out_states(self): 
        print("_______________One Step Update__________________________________")
        print("current States ------->")
        print(self.afterKF.x[self.global_plot_index])


        print("prediction States ---->")
        print(self.afterKF.x_hat[self.global_plot_index])

        print("current updated Covariance-->")
        print(self.afterKF.V[self.global_plot_index])

        print("Prediction Covariance ---> ")
        print(self.afterKF.W[self.global_plot_index])        
        #if self.global_plot_index <= 1:

        print("K------------------------> ")
        print(self.afterKF.K[self.global_plot_index])   
        print("S------------------------> ")
        print(self.afterKF.S[self.global_plot_index])   

    def draw(self): 
        self.car_state_display.clean_drawing(plt, self.ax1)
        self.draw_clean_list()
        
        #self.car_state_display.remove_drawing(self.car_state_display.remove_drawing_list)
        #self.car_state_display.remove_drawing_list = []

        index = self.global_plot_index

        time_diff = 0.0
        if index > 0:
            time_diff = self.ftimes[index] - self.ftimes[index-1]
            pt1_u = [self.afterKF.locs[index-1][0], 
                     self.afterKF.locs[index][0]]
            pt2_u = [self.afterKF.locs[index-1][1], 
                     self.afterKF.locs[index][1]]
            self.ax1.plot(pt1_u, pt2_u, 'r')
            print("Time Difference {:5.5f}".format(time_diff))
        
            if self.ground_truth is not None:
                pt1_u = [self.ground_truth[index-1][0], 
                         self.ground_truth[index][0]]
                pt2_u = [self.ground_truth[index-1][1], 
                         self.ground_truth[index][1]]
                self.ax1.plot(pt1_u, pt2_u, 'y')
                plt_ground_truth, = self.ax1.plot(self.ground_truth[index][0], 
                                            self.ground_truth[index][1], 
                                            'ys', 
                                            label = "Ground_truth")
                self.car_state_display.remove_label.append(plt_ground_truth)

        utime = self.utime[index]
        # print out the states inforamtion
        if self.verbose == True:
            self.print_out_states()


        self.car_state_display.draw(self.afterKF.x[index], self.ax1)
        self.car_state_display.draw_uncertainty(self.afterKF.x[index], 
                                           self.afterKF.V[index], 
                                           self.ax1
                                           )
        
        self.car_state_display.draw(self.afterKF.x[index], self.ax3)
        self.car_state_display.draw_uncertainty(self.afterKF.x[index], 
                                           self.afterKF.V[index], 
                                           self.ax3
                                           )
        box = self.truths[utime].corners
        


        self.ax1.set_title('Frame Index={:3}, Time Diff={:5.5}'.
                                            format(self.global_plot_index,
                                                      time_diff))
        # Update the measurements plot
        self.measurments_update(utime)
        self.main_diaplay_update(utime)

        self.ax3.set_ylim(min(box[:,1]) - 2, max(box[:,1]) + 2)
        self.ax3.set_xlim(min(box[:,0]) - 2, max(box[:,0]) + 2)
        self.ax2.set_ylim(min(box[:,1]) - 2, max(box[:,1]) + 2)
        self.ax2.set_xlim(min(box[:,0]) - 2, max(box[:,0]) + 2)

        self.ax3.legend()
        self.ax2.legend()
        self.ax1.legend()
        plt.draw()
    
    def error_play_ground(self, event = None):
        """
        A new image for testing the current state error analysis
        """
        fig, ax_ro = plt.subplots()
        self.error_play_car = Car_state()
        self.error_play_car_updated = Car_state()
        self.ax_ro = ax_ro

        ax_ro.set_title('Measurement Error Play ground')
        fig.set_figheight(15)
        fig.set_figwidth(15)
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        axcolor = 'lightgoldenrodyellow'
        
        ax_x = plt.axes([0.11, 0.15, 0.75, 0.02], facecolor=axcolor)
        ax_y = plt.axes([0.11, 0.13, 0.75, 0.02], facecolor=axcolor)
        ax_r = plt.axes([0.11, 0.11, 0.75, 0.02], facecolor=axcolor)
        
        pred_rot = plt.axes([0.11, 0.08, 0.75, 0.02], facecolor=axcolor)

        #ax_ux = plt.axes([0.51, 0.15, 0.45, 0.02], facecolor=axcolor)
        #ax_uy = plt.axes([0.51, 0.10, 0.45, 0.02], facecolor=axcolor)
        #ax_ur = plt.axes([0.51, 0.05, 0.45, 0.02], facecolor=axcolor)

        rax = plt.axes([0.90, 0.02, 0.10, 0.10], facecolor=axcolor)
        self.radio = RadioButtons(rax, ('EKF', 'Head_first', 'IEKF'))

        self.error_x_slider = Slider(ax_x, 
                                     'x_error', 
                                     x_slider_range[0], 
                                     x_slider_range[1], 
                                     valinit=0, 
                                     valstep=0.01)
        self.error_y_slider = Slider(ax_y, 
                                     'y_error', 
                                     x_slider_range[0], 
                                     x_slider_range[1], 
                                     valinit=0, 
                                     valstep=0.01)
        self.roation_slider = Slider(ax_r, 
                                     'rot_error', 
                                     r_slider_range[0], 
                                     r_slider_range[1], 
                                     valinit=0, 
                                     valstep=0.01)
        self.pred_roation_slider = Slider(pred_rot, 
                                          'Predion_Rot', 
                                          r_slider_range[0], 
                                          r_slider_range[1], 
                                          valinit=0, 
                                          valstep=0.01)

        self.error_x_slider.on_changed(self.error_play_ground_value_change)
        self.error_y_slider.on_changed(self.error_play_ground_value_change)
        self.roation_slider.on_changed(self.error_play_ground_value_change)
        self.radio.on_clicked(self.error_play_ground_value_change)

        self.pred_roation_slider.on_changed(self.error_play_ground_prediction_value_change)

        # Default draw the prediction states
        prediction_states = self.afterKF.x_hat[self.global_plot_index]
        self.prediction_states = prediction_states
        self.pred_cov          = self.afterKF.W[self.global_plot_index]
        self.error_play_car.draw(prediction_states, 
                                 self.ax_ro, 
                                 linestyle = '--', 
                                 label = "Prediction States")
        self.error_play_car.draw_uncertainty(prediction_states, 
                                             self.afterKF.W[self.global_plot_index],
                                             self.ax_ro,
                                             label = "Predicted States", 
                                             linestyle='--'
                                            )
        
        # Prediction Measurement and Uncertainty
        Pred = self.afterKF.x_hat[self.global_plot_index]
        H = self.generate_H(Pred, self.dected_corner[self.global_plot_index])
        pred_meas = np.dot(H, Pred)
        center_plot_pred_meas,  = self.ax_ro.plot(pred_meas[0], 
                                          pred_meas[1], 
                                           'gs', 
                                           label = "Predicted Measurements")
        ### Angle
        meas_end_x = pred_meas[0] + uncertainty_line_scale * np.cos(pred_meas[2])
        meas_end_y = pred_meas[1] + uncertainty_line_scale * np.sin(pred_meas[2])
        angle_x = [pred_meas[0], meas_end_x]
        angle_y = [pred_meas[1], meas_end_y]

        angle_plot_pred_meas,  = self.ax_ro.plot(angle_x, 
                                                  angle_y, 
                                                 'g')

        ### A line
        measurments_line_center = [prediction_states[0], pred_meas[0]]
        measurments_line_end    = [prediction_states[1], pred_meas[1]]
        plt_measurments_line,  = self.ax_ro.plot(measurments_line_center, 
                                                 measurments_line_end,
                                                 'g', linestyle = '--')

        ### Predicted uncertainty
        elp_pred_meas = error_ellipse([pred_meas[0], pred_meas[1]], 
                                       self.afterKF.W_meas[
                                            self.global_plot_index][0:2,0:2], 
                                       ax = self.ax_ro, 
                                       factor=plot_std, 
                                       edgecolor = 'g',
                                       linestyle='--')
        heading_uncertainty = plot_std * \
                                  np.sqrt(self.afterKF.W_meas[
                                         self.global_plot_index][2,2])
        heading_up   = pred_meas[2] + heading_uncertainty
        heading_down = pred_meas[2] - heading_uncertainty
        length_scale = uncertainty_line_scale
        pt_x = [pred_meas[0], pred_meas[0] + length_scale * np.cos(heading_up)]
        pt_y = [pred_meas[1], pred_meas[1] + length_scale * np.sin(heading_up)]
        head_plot_u, = self.ax_ro.plot(pt_x, pt_y, 'g')
        pt_x = [pred_meas[0], pred_meas[0] + length_scale * np.cos(heading_down)]
        pt_y = [pred_meas[1], pred_meas[1] + length_scale * np.sin(heading_down)]
        head_plot_d, = self.ax_ro.plot(pt_x, pt_y, 'g')

            
        cor_meas = self.ax_ro.annotate(self.dected_corner[self.global_plot_index]+1, 
                                       [pred_meas[0], pred_meas[1]],
                                       xytext=([pred_meas[0], pred_meas[1]]), 
                                       size=20,
                                       color = 'g')

        self.error_play_car.current_lines.append(center_plot_pred_meas)
        self.error_play_car.current_lines.append(angle_plot_pred_meas)
        self.error_play_car.current_lines.append(plt_measurments_line)
        self.error_play_car.current_lines.append(elp_pred_meas)
        self.error_play_car.current_lines.append(head_plot_u)
        self.error_play_car.current_lines.append(head_plot_d)
        self.error_play_car.current_lines.append(cor_meas)


        # Ground Truth
        utime = self.utime[self.global_plot_index]
        box = self.truths[utime].corners

        box = box[range(-1, box.shape[0]), :]
        h1, = self.ax_ro.plot(box[:, 0], box[:, 1], 'b', label = "Ground Truth")

        self.error_play_ground_value_change()

        self.ax_ro.axis('equal')
        self.ax_ro.set_xlim(prediction_states[0] - 8, prediction_states[0] + 8)
        self.ax_ro.set_ylim(prediction_states[1] - 8, prediction_states[1] + 8)        
        self.ax_ro.legend()
        plt.show(block=True)
    
    def error_play_ground_prediction_value_change(self, even=None): 
        self.prediction_states = copy.deepcopy(self.afterKF.\
                                 x_hat[self.global_plot_index])
        self.prediction_states[2] = self.afterKF.x_hat[self.global_plot_index][2] + \
                                    self.pred_roation_slider.val * (np.pi / 180.0)

        if self.verbose == True:
            print(self.afterKF.x_hat[self.global_plot_index][2])

        self.error_play_car.remove_center()
        self.error_play_car.clean_drawing(plt, self.ax_ro)
        self.error_play_car.draw(self.prediction_states, 
                                 self.ax_ro, 
                                 linestyle = '--', 
                                 label = "Prediction States")
        # TODO: Rotate the prediction Covariance
        pred_cov = self.afterKF.W[self.global_plot_index]
        ang_     = self.pred_roation_slider.val * (np.pi / 180.0)
        rot_mat  = utility.generate_rotation_matrix(ang_, True)
        pred_cov = np.dot(np.dot(rot_mat,pred_cov), rot_mat.T)
        dH =self.generate_dH(self.prediction_states, 
                             self.dected_corner[self.global_plot_index])
        Q = np.zeros((5, 5))
        Q[0:3, 0:3] = self.error_play_cov
        Q[3][3] = 0.0001 ** 2
        Q[4][4] = 0.0001 ** 2 
        S = np.dot(np.dot(dH, pred_cov), dH.transpose()) + Q
        self.pred_cov = pred_cov
        self.pred_meas_cov = S

        self.error_play_car.draw_uncertainty(self.prediction_states, 
                                             #self.afterKF.W[self.global_plot_index],
                                             pred_cov,
                                             self.ax_ro,
                                             label = "Predicted States", 
                                             linestyle='--'
                                            )
        #print(event)
        # Prediction Measurement and Uncertainty
        Pred = self.prediction_states
        H = self.generate_H(Pred, self.dected_corner[self.global_plot_index])
        pred_meas = np.dot(H, Pred)
        center_plot_pred_meas,  = self.ax_ro.plot(pred_meas[0], 
                                          pred_meas[1], 
                                           'gs', 
                                           label = "Predicted Measurements")
        ### Angle
        meas_end_x = pred_meas[0] + uncertainty_line_scale * np.cos(pred_meas[2])
        meas_end_y = pred_meas[1] + uncertainty_line_scale * np.sin(pred_meas[2])
        angle_x = [pred_meas[0], meas_end_x]
        angle_y = [pred_meas[1], meas_end_y]

        angle_plot_pred_meas,  = self.ax_ro.plot(angle_x, 
                                                  angle_y, 
                                                 'g')

        ### A line
        measurments_line_center = [self.prediction_states[0], pred_meas[0]]
        measurments_line_end    = [self.prediction_states[1], pred_meas[1]]
        plt_measurments_line,  = self.ax_ro.plot(measurments_line_center, 
                                                 measurments_line_end,
                                                 'g', linestyle = '--')

        ### Predicted uncertainty
        elp_pred_meas = error_ellipse([pred_meas[0], pred_meas[1]], 
                                       #self.afterKF.W_meas[
                                       #     self.global_plot_index][0:2,0:2],
                                       self.pred_meas_cov[0:2, 0:2], 
                                       ax = self.ax_ro, 
                                       factor=plot_std, 
                                       edgecolor = 'g',
                                       linestyle='--')
        heading_uncertainty = plot_std * \
                                  np.sqrt(self.pred_meas_cov[2,2])
        heading_up   = pred_meas[2] + heading_uncertainty
        heading_down = pred_meas[2] - heading_uncertainty
        length_scale = uncertainty_line_scale
        pt_x = [pred_meas[0], pred_meas[0] + length_scale * np.cos(heading_up)]
        pt_y = [pred_meas[1], pred_meas[1] + length_scale * np.sin(heading_up)]
        head_plot_u, = self.ax_ro.plot(pt_x, pt_y, 'g')
        pt_x = [pred_meas[0], pred_meas[0] + length_scale * np.cos(heading_down)]
        pt_y = [pred_meas[1], pred_meas[1] + length_scale * np.sin(heading_down)]
        head_plot_d, = self.ax_ro.plot(pt_x, pt_y, 'g')

            
        cor_meas = self.ax_ro.annotate(self.dected_corner[self.global_plot_index]+1, 
                                       [pred_meas[0], pred_meas[1]],
                                       xytext=([pred_meas[0], pred_meas[1]]), 
                                       size=20,
                                       color = 'g')

        self.error_play_car.current_lines.append(center_plot_pred_meas)
        self.error_play_car.current_lines.append(angle_plot_pred_meas)
        self.error_play_car.current_lines.append(plt_measurments_line)
        self.error_play_car.current_lines.append(elp_pred_meas)
        self.error_play_car.current_lines.append(head_plot_u)
        self.error_play_car.current_lines.append(head_plot_d)
        self.error_play_car.current_lines.append(cor_meas)
        
        # Update the updated value
        self.error_play_ground_value_change()

        
    
    def error_play_ground_value_change(self, event = None): 
        """
        Slider bar value change
        """
        print("Current Value x {:5.5f}, y {:5.5f}, anglge {:5.5}".format( 
                self.error_x_slider.val, self.error_y_slider.val, 
                self.roation_slider.val))
        if len(self.error_play_lines) > 0:
            utility.clean_lines(self.error_play_lines)
            self.error_play_lines = []
        
        ### Use real measurement here

        #Pred = self.afterKF.x_hat[self.global_plot_index]
        #H = self.generate_H(Pred, self.dected_corner[self.global_plot_index])
        #pred_meas = np.dot(H, Pred)
        utime = self.utime[self.global_plot_index]
        pred_meas = [self.locs[self.global_plot_index][0], 
                     self.locs[self.global_plot_index][1], 
                     self.headings[self.global_plot_index]]
        meas_x = self.error_x_slider.val + pred_meas[0]
        meas_y = self.error_y_slider.val + pred_meas[1]
        meas_r = self.roation_slider.val * (np.pi / 180.0) + pred_meas[2]

        # Change the covariance
        angle_theta = self.roation_slider.val * (np.pi / 180.0)
        rotation_matrix = utility.generate_rotation_matrix(angle_theta)
        meas_cov  =  self.meas_cov[self.global_plot_index]
        self.error_play_cov  = np.dot(np.dot(rotation_matrix, meas_cov),
                                     rotation_matrix.T)
        self.error_play_meas = [meas_x, meas_y, meas_r]

        # Display this measurements
        center_plot_pred_meas,  = self.ax_ro.plot(meas_x, 
                                                  meas_y, 
                                                 'ms', 
                                           label = "Measurements")
                                            
        elp_pred_meas = error_ellipse([meas_x, meas_y], 
                                       self.error_play_cov[0:2,0:2], 
                                       ax = self.ax_ro, 
                                       factor=plot_std, 
                                       edgecolor = 'm',
                                       linestyle='--')
        ## Measurements states center
        l = np.sqrt((wheelbase_to_length_ratio() * self.h[0])**2 + 
                    (0.5 * self.w[0])**2)
        phi = np.arctan( (0.5 * self.w[0]) / (wheelbase_to_length_ratio() * self.h[0]))

        l2 = np.sqrt((rawrtc_to_l_ratio() * self.h[0])**2 + 
                    (0.5 * self.w[0])**2)
        gama = np.arctan( (0.5 * self.w[0]) / (rawrtc_to_l_ratio() * self.h[0]))

        if (self.dected_corner[self.global_plot_index] == 0):
            meas_state_x = meas_x - l *  np.cos(meas_r - phi)
            meas_state_y = meas_y - l *  np.sin(meas_r - phi)
        elif (self.dected_corner[self.global_plot_index] == 1):
            meas_state_x = meas_x - l *  np.cos(meas_r + phi)
            meas_state_y = meas_y - l *  np.sin(meas_r + phi)
        elif (self.dected_corner[self.global_plot_index] == 2):
            meas_state_x = meas_x - l2 *  np.cos(meas_r - np.pi - gama)
            meas_state_y = meas_y - l2 *  np.sin(meas_r - np.pi - gama)
        elif (self.dected_corner[self.global_plot_index] == 3):
            meas_state_x = meas_x - l2 *  np.cos(meas_r - np.pi + gama)
            meas_state_y = meas_y - l2 *  np.sin(meas_r - np.pi + gama)
        else: 
            meas_state_x = 0.0
            meas_state_y = 0.0
        center_plot_pred_meas_state,  = self.ax_ro.plot(meas_state_x, 
                                                  meas_state_y, 
                                                 'mo', 
                                           label = "Measurements vel center")

        ## Angle
        meas_end_x = meas_x + 2 * np.cos(meas_r)
        meas_end_y = meas_y + 2 * np.sin(meas_r)
        angle_x = [meas_x, meas_end_x]
        angle_y = [meas_y, meas_end_y]

        angle_plot_pred_meas,  = self.ax_ro.plot(angle_x, 
                                                 angle_y, 
                                                 'm') 
        heading_uncertainty = plot_std * \
                                  np.sqrt(self.error_play_cov[2,2])
        heading_up   = meas_r + heading_uncertainty
        heading_down = meas_r - heading_uncertainty
        length_scale = uncertainty_line_scale
        pt_x = [meas_x, meas_x + length_scale * np.cos(heading_up)]
        pt_y = [meas_y, meas_y + length_scale * np.sin(heading_up)]
        head_plot_u, = self.ax_ro.plot(pt_x, pt_y, 'm', linestyle = "--")

        pt_x = [meas_x, meas_x + length_scale * np.cos(heading_down)]
        pt_y = [meas_y, meas_y + length_scale * np.sin(heading_down)]
        head_plot_d, = self.ax_ro.plot(pt_x, pt_y, 'm', linestyle = '--')

        #pt_x = [pred_meas[0], pred_meas[0] + length_scale * np.cos(heading_down)]
        #pt_y = [pred_meas[1], pred_meas[1] + length_scale * np.sin(heading_down)]
        #head_plot_d, = self.ax_ro.plot(pt_x, pt_y, 'g')
        
        # **** One step Kalman Update
        
        if self.radio.value_selected == 'EKF':
            self.one_step_EKF_update()
        elif self.radio.value_selected == 'Head_first':
            self.two_step_EKF_update_angle_first()
        else:
            self.iterative_EKF_update()
        #self.two_step_EKF_update_angle_first()

        self.error_play_lines.append(center_plot_pred_meas_state)

        self.error_play_lines.append(center_plot_pred_meas)
        self.error_play_lines.append(angle_plot_pred_meas)
        self.error_play_lines.append(head_plot_d)
        self.error_play_lines.append(head_plot_u)
        self.error_play_lines.append(elp_pred_meas)

        self.ax_ro.legend()
    
    def iterative_EKF_update(self):

        self.error_play_car_updated.remove_center()
        self.error_play_car_updated.clean_drawing(plt, self.ax_ro)

        Pred = self.prediction_states
        Prev_pred = copy.deepcopy(Pred) + np.Inf
        #self.pred_cov = pred_cov
        #self.pred_meas_cov = S
        PropCov = self.pred_cov
        x_opt = copy.deepcopy(Pred)
        all_middle_states = []
        
        while utility.state_error(Prev_pred, x_opt, useAll=True) >\
            utility.iterative_threshold():
            H = self.generate_H(x_opt, 
                                self.dected_corner[self.global_plot_index])
            dH = self.generate_dH(x_opt, 
                                self.dected_corner[self.global_plot_index])
            y_ = np.array([self.error_play_meas[0], 
                           self.error_play_meas[1],
                           self.error_play_meas[2], 
                           self.w[0],
                           self.h[0]])[:, None] - np.dot(H, x_opt)
            y_.itemset(2, angle_difference(self.error_play_meas[2],
                                (((np.dot(H, x_opt)).item(2)) % (2 * math.pi))))
            Q = np.zeros((5, 5))
            Q[0:3, 0:3] = self.error_play_cov
            Q[3][3] = 0.0001 ** 2
            Q[4][4] = 0.0001 ** 2
            S = np.dot(np.dot(dH, PropCov), dH.transpose()) + Q
            K = np.dot(np.dot(PropCov, dH.transpose()), np.linalg.inv(S))
            bias = np.dot(dH, (Pred - x_opt))
            updated_x = Pred + np.dot(K, (y_ - bias))
            all_middle_states.append(updated_x)
            
            Prev_pred = x_opt
            x_opt = updated_x
            #ipy.embed()
        
        for i in range(len(all_middle_states)-1):
            self.error_play_car_updated.draw(all_middle_states[i], 
                                             self.ax_ro,
                                             linestyle=':',
                                             label = "Middle Process"
                                             )
        if (updated_x.item(4) < -2 * math.pi / 6):
            updated_x.itemset(4, -2 * math.pi / 6)
        if (updated_x.item(4) > 2 * math.pi / 6):
            updated_x.itemset(4, 2 * math.pi / 6)
        n = len(updated_x)
        cov = np.dot((np.identity(n) - np.dot(K, dH)), PropCov)
        self.error_play_car_updated.draw(updated_x, self.ax_ro)
        self.error_play_car_updated.draw_uncertainty(updated_x,cov,self.ax_ro)



    def one_step_EKF_update(self): 
        """
        One step kalman filter update
        """
        self.error_play_car_updated.remove_center()
        self.error_play_car_updated.clean_drawing(plt, self.ax_ro)
        Pred = self.prediction_states
        H = self.generate_H(Pred, self.dected_corner[self.global_plot_index])
        y_ = np.array([self.error_play_meas[0], 
                       self.error_play_meas[1],
                       self.error_play_meas[2], 
                       self.w[0],
                       self.h[0]])[:, None] - np.dot(H, Pred)
        y_.itemset(2, angle_difference(self.error_play_meas[2],
                                (((np.dot(H, Pred)).item(2)) % (2 * math.pi))))
        #y_[2] = 0
        Q = np.zeros((5, 5))
        #Q[0:3, 0:3] = self.meas_cov[self.global_plot_index]
        Q[0:3, 0:3] = self.error_play_cov
        
        Q[3][3] = 0.0001 ** 2
        Q[4][4] = 0.0001 ** 2
        PropCov = self.pred_cov
        dH = self.generate_dH(Pred, self.dected_corner[self.global_plot_index])
        S = np.dot(np.dot(dH, PropCov), dH.transpose()) + Q
        K = np.dot(np.dot(PropCov, dH.transpose()), np.linalg.inv(S))
        if self.verbose == True:
            print("I is\n", y_)
            print("K is\n", K)
        updated_x = Pred + np.dot(K, y_)

        if (updated_x.item(4) < -2 * math.pi / 6):
            updated_x.itemset(4, -2 * math.pi / 6)
        if (updated_x.item(4) > 2 * math.pi / 6):
            updated_x.itemset(4, 2 * math.pi / 6)
        n = len(updated_x)
        cov = np.dot((np.identity(n) - np.dot(K, dH)), PropCov)

        self.error_play_car_updated.draw(updated_x, self.ax_ro)
        self.error_play_car_updated.draw_uncertainty(updated_x,cov,self.ax_ro)

    def Sigma_point_update(self): 
        """
        One step kalman filter update
        """
        self.error_play_car_updated.remove_center()
        self.error_play_car_updated.clean_drawing(plt, self.ax_ro)
        Pred = self.prediction_states[0:5]
        w    = self.prediction_states[5]
        h    = self.prediction_states[6]
        PropCov = self.afterKF.W[self.global_plot_index][0:5, 0:5]
        Q = np.zeros((3, 3))
        Q[0:3, 0:3] = self.meas_cov[self.global_plot_index]

        H = self.generate_H(self.prediction_states, 
                            self.dected_corner[self.global_plot_index])
        H = H[0:3, 0:5]
        sigma_points = ukf.generate_sigma_points(Pred, PropCov, Q)
        response = ukf.generate_response(sigma_points, H)
        u_y, cov_yy, cov_xy = ukf.generate_update_parameters(response, 
                                                             Pred, 
                                                             sigma_points,
                                                             len(Pred)+len(Q))


        K = np.dot(cov_xy, np.linalg.inv(cov_yy))
        P = PropCov - np.dot(K , cov_xy.T)
        y_ = self.error_play_meas - u_y
        ang = angle_difference(self.error_play_meas[2], u_y[2])
        y_[2] = ang

        updated_x_ = Pred.T - np.dot(K , (y_))
        updated_x_ = updated_x_.T
        updated_x = self.prediction_states
        updated_x[0:5] = updated_x_
        updated_x[5] = w
        updated_x[6] = h
        

        if (updated_x.item(4) < -2 * math.pi / 6):
            updated_x.itemset(4, -2 * math.pi / 6)
        if (updated_x.item(4) > 2 * math.pi / 6):
            updated_x.itemset(4, 2 * math.pi / 6)

        self.error_play_car_updated.draw(updated_x, self.ax_ro)
        self.error_play_car_updated.draw_uncertainty(updated_x,P,self.ax_ro)
    
    def two_step_EKF_update(self): 
        """
        Two step EKF update, one for the position and one for the angle
        """
        self.error_play_car_updated.remove_center()
        self.error_play_car_updated.clean_drawing(plt, self.ax_ro)
        Pred = self.prediction_states
        H = self.generate_H(Pred, self.dected_corner[self.global_plot_index])
        dH = self.generate_dH(Pred, self.dected_corner[self.global_plot_index])
        y_ = np.array([self.error_play_meas[0], 
                       self.error_play_meas[1],
                       self.error_play_meas[2], 
                       self.w[0],
                       self.h[0]])[:, None] - np.dot(H, Pred)
        y_.itemset(2, angle_difference(self.error_play_meas[2],
                                (((np.dot(H, Pred)).item(2)) % (2 * math.pi))))
        y_[2] = 0
        Q = np.zeros((5, 5))
        Q[0:2, 0:2] = self.meas_cov[self.global_plot_index][0:2,0:2]
        Q[3][3] = 0.0001 ** 2
        Q[4][4] = 0.0001 ** 2
        PropCov = self.afterKF.W[self.global_plot_index]
        S = np.dot(np.dot(dH, PropCov), dH.transpose()) + Q
        K = np.dot(np.dot(PropCov, dH.transpose()), np.linalg.inv(S))
        if self.verbose == True:
            print("I is\n", y_)
            print("K is\n", K)
        updated_x = Pred + np.dot(K, y_)

        if (updated_x.item(4) < -2 * math.pi / 6):
            updated_x.itemset(4, -2 * math.pi / 6)
        if (updated_x.item(4) > 2 * math.pi / 6):
            updated_x.itemset(4, 2 * math.pi / 6)
        n = len(updated_x)
        cov = np.dot((np.identity(n) - np.dot(K, dH)), PropCov)

        self.error_play_car_updated.draw(updated_x, 
                                        self.ax_ro,
                                        linestyle='-.',
                                        label="Only use the X Y pose")
        self.error_play_car_updated.draw_uncertainty(updated_x, cov, self.ax_ro)

        Pred = updated_x
        H = self.generate_H(Pred, self.dected_corner[self.global_plot_index])
        dH = self.generate_dH(Pred, self.dected_corner[self.global_plot_index])
        y_ = np.array([self.error_play_meas[0], 
                       self.error_play_meas[1],
                       self.error_play_meas[2], 
                       self.w[0],
                       self.h[0]])[:, None] - np.dot(H, Pred)
        y_.itemset(2, angle_difference(self.error_play_meas[2],
                                (((np.dot(H, Pred)).item(2)) % (2 * math.pi))))
        
        y_[0] = 0.0
        y_[1] = 0.0
        Q = np.zeros((5,5))
        Q[2,2] = self.meas_cov[self.global_plot_index][2][2]
        #PropCov = cov
        S = np.dot(np.dot(dH, PropCov), dH.transpose()) + Q
        K = np.dot(np.dot(PropCov, dH.transpose()), np.linalg.inv(S))
        if self.verbose == True:
            print("I is\n", y_)
            print("K is\n", K)
        updated_x = Pred + np.dot(K, y_)

        if (updated_x.item(4) < -2 * math.pi / 6):
            updated_x.itemset(4, -2 * math.pi / 6)
        if (updated_x.item(4) > 2 * math.pi / 6):
            updated_x.itemset(4, 2 * math.pi / 6)
        n = len(updated_x)
        cov = np.dot((np.identity(n) - np.dot(K, dH)), PropCov)

        self.error_play_car_updated.draw(updated_x, self.ax_ro)
        self.error_play_car_updated.draw_uncertainty(updated_x,cov,self.ax_ro)

    def two_step_EKF_update_angle_first(self): 
        """
        Two step EKF update, one for the position and one for the angle
        """
        self.error_play_car_updated.remove_center()
        self.error_play_car_updated.clean_drawing(plt, self.ax_ro)
        Pred = self.prediction_states
        H = self.generate_H(Pred, self.dected_corner[self.global_plot_index])
        dH = self.generate_dH(Pred, self.dected_corner[self.global_plot_index])
        y_ = np.array([self.error_play_meas[0], 
                       self.error_play_meas[1],
                       self.error_play_meas[2], 
                       self.w[0],
                       self.h[0]])[:, None] - np.dot(H, Pred)
        y_.itemset(2, angle_difference(self.error_play_meas[2],
                                (((np.dot(H, Pred)).item(2)) % (2 * math.pi))))
        y_[0] = 0.0
        y_[1] = 0.0
        Q = np.zeros((5,5))
        Q[2,2] = self.error_play_cov[2][2]

        Q[3][3] = 0.0001 ** 2
        Q[4][4] = 0.0001 ** 2
        PropCov = self.pred_cov
        S = np.dot(np.dot(dH, PropCov), dH.transpose()) + Q
        K = np.dot(np.dot(PropCov, dH.transpose()), np.linalg.inv(S))
        if self.verbose == True:
            print("I is\n", y_)
            print("K is\n", K)
        updated_x = Pred + np.dot(K, y_)

        if (updated_x.item(4) < -2 * math.pi / 6):
            updated_x.itemset(4, -2 * math.pi / 6)
        if (updated_x.item(4) > 2 * math.pi / 6):
            updated_x.itemset(4, 2 * math.pi / 6)
        n = len(updated_x)
        cov = np.dot((np.identity(n) - np.dot(K, dH)), PropCov)

        self.error_play_car_updated.draw(updated_x, 
                                        self.ax_ro,
                                        linestyle='-.',
                                        label="Only use the theta pose")
        self.error_play_car_updated.draw_uncertainty(updated_x, cov, self.ax_ro)

        Pred = updated_x
        H = self.generate_H(Pred, self.dected_corner[self.global_plot_index])
        dH = self.generate_dH(Pred, self.dected_corner[self.global_plot_index])
        y_ = np.array([self.error_play_meas[0], 
                       self.error_play_meas[1],
                       self.error_play_meas[2], 
                       self.w[0],
                       self.h[0]])[:, None] - np.dot(H, Pred)
        y_.itemset(2, angle_difference(self.error_play_meas[2],
                                (((np.dot(H, Pred)).item(2)) % (2 * math.pi))))
        
        y_[2] = 0.0
        Q = np.zeros((5,5))
        Q[2,2] = self.error_play_cov[2][2]

        Q = np.zeros((5, 5))
        Q[0:2, 0:2] = self.error_play_cov[0:2,0:2]
        Q[3][3] = 0.0001 ** 2
        Q[4][4] = 0.0001 ** 2
        #PropCov = cov
        S = np.dot(np.dot(dH, PropCov), dH.transpose()) + Q
        K = np.dot(np.dot(PropCov, dH.transpose()), np.linalg.inv(S))
        if self.verbose == True:
            print("I is\n", y_)
            print("K is\n", K)
        updated_x = Pred + np.dot(K, y_)

        if (updated_x.item(4) < -2 * math.pi / 6):
            updated_x.itemset(4, -2 * math.pi / 6)
        if (updated_x.item(4) > 2 * math.pi / 6):
            updated_x.itemset(4, 2 * math.pi / 6)
        n = len(updated_x)
        cov = np.dot((np.identity(n) - np.dot(K, dH)), PropCov)

        self.error_play_car_updated.draw(updated_x, self.ax_ro)
        self.error_play_car_updated.draw_uncertainty(updated_x,cov,self.ax_ro)


    def rotate_play_ground(self, event = None):
        """
        A new image for testing the rotate
        """
        fig, ax_ro = plt.subplots()
        self.rotation_play_car = Car_state()
        self.ax_ro = ax_ro
        
        #ax.plot(t, s)
        ax_ro.set_title('Rotation Play ground')
        fig.set_figheight(15)
        fig.set_figwidth(15)

        plt.subplots_adjust(bottom=0.25)
        axcolor = 'lightgoldenrodyellow'
        axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        self.roation_slider = Slider(axamp, 
                                     'Rotate Degree', 
                                     1., 
                                     180.0, 
                                     valinit=0, 
                                     valstep=0.1)
        self.roation_slider.on_changed(self.rotate_play_ground_update)
        self.rotate_play_ground_update()
        plt.show(block=True)

    def rotate_play_ground_update(self, event=None):
        print("Dispaly Rotate Play Ground")
        angle = (self.roation_slider.val) * (np.pi / 180.0)
        #print("Current angle ", angle)
        ## 1. State Transformation
        states = copy.deepcopy(self.afterKF.x_hat[self.global_plot_index])
        n = len(states)
        T = np.eye(n)
        T[0,0] = np.cos(angle)
        T[0,1] = -np.sin(angle)
        T[1,0] = np.sin(angle)
        T[1,1] = np.cos(angle)
        E = np.zeros(n)
        E[2] = angle
        states = (T @ states) 
        states[2] += angle

        ## 2. State Covariance Transforamtion
        state_cov = copy.deepcopy(self.afterKF.W[self.global_plot_index])

        state_cov = T @ state_cov @ T.T
        
        self.rotation_play_car.clean_drawing(plt, self.ax_ro)
        self.rotation_play_car.draw(states, 
                                    self.ax_ro, 
                                    linestyle='--',
                                    label = "Predicted")
        self.rotation_play_car.draw_uncertainty(states, 
                                                state_cov, 
                                                self.ax_ro, 
                                                label = "Predicted States",
                                                linestyle= '--')
        
        ## 3. Rotate the Measurements
        ### Position
        T = T[0:3, 0:3]
        meas = np.zeros(3)
        meas[0:2] = self.locs[self.global_plot_index]
        meas[2]   = self.headings[self.global_plot_index]
        meas = T @ meas
        meas[2] += angle
        ### Plot them on the map
        plt_bia_corner, = self.ax_ro.plot(meas[0], 
                                        meas[1], 
                                        'r.', 
                                        label = "measurements")
        self.rotation_play_car.current_lines.append(plt_bia_corner)

        heading = meas[2]
        pt_x = [meas[0], 
                meas[0] + uncertainty_line_scale * np.cos(heading)]
        pt_y = [meas[1], 
                meas[1] + uncertainty_line_scale * np.sin(heading)]
        head_plot, = self.ax_ro.plot(pt_x, 
                                     pt_y, 
                                     '#8B2252', 
                                     label = "Heading")

        self.rotation_play_car.current_lines.append(head_plot)

        ### Measurement covariance
        utime = self.utime[self.global_plot_index]
        meas_cov = self.lshapes[utime].cov
        meas_cov = T @ meas_cov @ T.T
        
        ### Plot the position uncertainty
        elp = error_ellipse(meas[0:2], 
                            meas_cov[0:2,0:2], 
                            ax = self.ax_ro, 
                            factor=3.0, 
                            edgecolor = 'r',
                            label = "Measurement Uncertainty")
        self.rotation_play_car.current_lines.append(elp)

        ### Plot the heading angle uncertainty
        heading_uncertainty = plot_std * np.sqrt(meas_cov[2,2])
        length_scale = uncertainty_line_scale

        heading_up   = meas[2] + heading_uncertainty
        heading_down = meas[2] - heading_uncertainty
        pt_x = [meas[0], 
                meas[0] + length_scale * np.cos(heading_up)]
        pt_y = [meas[1], 
                meas[1] + length_scale * np.sin(heading_up)]
        head_plot_u_pred, = self.ax_ro.plot(pt_x, pt_y, 'r', linestyle = '--')
        pt_x = [meas[0], 
                meas[0] + length_scale * np.cos(heading_down)]
        pt_y = [meas[1], 
                meas[1] + length_scale * np.sin(heading_down)]
        head_plot_d_pred, = self.ax_ro.plot(pt_x, pt_y, 'r', linestyle = '--')

        self.rotation_play_car.current_lines.append(head_plot_u_pred)
        self.rotation_play_car.current_lines.append(head_plot_d_pred)

        # EKF Update
        Pred = states
        H = self.generate_H(Pred, self.dected_corner[self.global_plot_index])
        dH = self.generate_dH(Pred, self.dected_corner[self.global_plot_index])
        y_ = np.array([meas[0], 
                       meas[1],
                       meas[2], 
                       self.w[0],
                       self.h[0]])[:, None] - np.dot(H, Pred)
        y_.itemset(2, angle_difference(meas[2],
                                (((np.dot(H, Pred)).item(2)) % (2 * math.pi))))
        
        Q = np.zeros((5, 5))
        Q[0:3, 0:3] = meas_cov
        Q[3][3] = 0.0001 ** 2
        Q[4][4] = 0.0001 ** 2

        PropCov = state_cov
        S = np.dot(np.dot(dH, PropCov), dH.transpose()) + Q
        K = np.dot(np.dot(PropCov, dH.transpose()), np.linalg.inv(S))

        updated_x = Pred + np.dot(K, y_)

        if (updated_x.item(4) < -2 * math.pi / 6):
            updated_x.itemset(4, -2 * math.pi / 6)
        if (updated_x.item(4) > 2 * math.pi / 6):
            updated_x.itemset(4, 2 * math.pi / 6)
        
        n = len(updated_x)
        cov = np.dot((np.identity(n) - np.dot(K, dH)), PropCov)

        # Draw the updated states
        self.rotation_play_car.draw(updated_x, 
                                    self.ax_ro, 
                                    linestyle='-',
                                    label = "Updated")

        self.rotation_play_car.draw_uncertainty(updated_x, 
                                                cov, 
                                                self.ax_ro, 
                                                label = "Updated States",
                                                linestyle= '-')

        
        dis = two_point_length(updated_x[0:2], states[0:2])
        self.ax_ro.set_title(f"Relative ditance {dis}")

        
        self.ax_ro.axis('equal')
        self.ax_ro.set_xlim(states[0] - 10, states[0] + 10)
        self.ax_ro.set_ylim(states[1] - 10, states[1] + 10)
        
        self.ax_ro.legend()
        plt.draw()


    def main_diaplay_update(self, utime):
        
        index = self.global_plot_index

        # 1. Ground truth and uncertainty
        box = self.truths[utime].corners
        corners = self.lshapes[utime].corners
        box = box[range(-1, box.shape[0]), :]
        h1, = self.ax1.plot(box[:, 0], box[:, 1], 'b', label = "Ground Truth")
        h2, = self.ax1.plot(corners[:, 0], 
                       corners[:, 1], 
                       'r', 
                       label = "L Measurements")
        self.current_lines.append(h1)
        self.current_lines.append(h2)

        # 2. Draw the predicted states and uncertainty
        center_x_pred = self.afterKF.x_hat[index][0]
        center_y_pred = self.afterKF.x_hat[index][1]
        center_plot_pred,  = self.ax1.plot(center_x_pred, 
                                     center_y_pred, 
                                     'ms', 
                                     label = "Predicted States")
        ### Draw Predicted state uncertainty
        elp_pred = error_ellipse([center_x_pred, center_y_pred], 
                                 self.afterKF.W[index][0:2,0:2], 
                                 ax = self.ax1, 
                                 factor=plot_std, 
                                 edgecolor = 'm',
                                 linestyle='--')

        ### Draw the predicted heading
        ### Length propotion to the speed
        heading = self.afterKF.x_hat[index][2]
        speed   = self.afterKF.x_hat[index][3] / 2

        pt_x_pred = [center_x_pred, center_x_pred + speed * np.cos(heading)]
        pt_y_pred = [center_y_pred, center_y_pred + speed * np.sin(heading)]
        head_plot_pred, = self.ax1.plot(pt_x_pred, 
                                        pt_y_pred, 
                                        'm')
        self.current_lines.append(head_plot_pred)
        self.current_lines.append(elp_pred)
        self.current_lines.append(center_plot_pred)

        ### Draw Heading uncertainty
        heading_uncertainty = plot_std * \
                              np.sqrt(self.afterKF.W[index][2,2])
        length_scale = uncertainty_line_scale

        heading_up   = heading + heading_uncertainty
        heading_down = heading - heading_uncertainty
        pt_x = [center_x_pred, 
                center_x_pred + length_scale * np.cos(heading_up)]
        pt_y = [center_y_pred, 
                center_y_pred + length_scale * np.sin(heading_up)]
        head_plot_u_pred, = self.ax1.plot(pt_x, pt_y, 'm', linestyle = '--')
        pt_x = [center_x_pred, 
                center_x_pred + length_scale * np.cos(heading_down)]
        pt_y = [center_y_pred, 
                center_y_pred + length_scale * np.sin(heading_down)]
        head_plot_d_pred, = self.ax1.plot(pt_x, pt_y, 'm', linestyle = '--')

        self.current_lines.append(head_plot_u_pred)
        self.current_lines.append(head_plot_d_pred)

        # 3. Unbiased Measurement and uncertainty
        # draw the unbiased Uncertainty
        plt_bia_corner, = self.ax1.plot(self.locs[index][0], 
                                        self.locs[index][1], 
                                        'r.', 
                                        label = "measurements")
        self.current_lines.append(plt_bia_corner)
        # draw the measurement ellipse
        cov = self.lshapes[utime].cov[0:2, 0:2]

        elp = error_ellipse([self.locs[index][0], self.locs[index][1]], 
                                cov, 
                                ax = self.ax1, 
                                factor=3.0, 
                                edgecolor = 'r')
        self.current_lines.append(elp)

        # Draw the heading and uncertainty
        heading_uncertainty = plot_std *\
                              np.sqrt(self.lshapes[utime].cov[2,2])
        heading = self.afterKF.headings[self.global_plot_index]
        pt_x = [self.locs[index][0],  
                self.locs[index][0] + 
                        uncertainty_line_scale * np.cos(heading)]
        pt_y = [self.locs[index][1], 
                self.locs[index][1] + 
                        uncertainty_line_scale * np.sin(heading)]
        head_plot, = self.ax1.plot(pt_x, 
                             pt_y, 
                             '#8B2252', 
                             label = "Heading")
        self.current_lines.append(head_plot)
        heading_up   = heading + heading_uncertainty
        heading_down = heading - heading_uncertainty
        length_scale = uncertainty_line_scale
        pt_x = [self.locs[index][0], 
                self.locs[index][0] + length_scale * np.cos(heading_up)]
        pt_y = [self.locs[index][1], 
                self.locs[index][1] + length_scale * np.sin(heading_up)]

        head_plot_u, = self.ax1.plot(pt_x, pt_y, '#8B2252', linestyle='--')
        pt_x = [self.locs[index][0], 
                self.locs[index][0] + length_scale * np.cos(heading_down)]
        pt_y = [self.locs[index][1], 
                self.locs[index][1] + length_scale * np.sin(heading_down)]
        head_plot_d, = self.ax1.plot(pt_x, pt_y, '#8B2252',linestyle='--')

        self.current_lines.append(head_plot_d)
        self.current_lines.append(head_plot_u)


        ## 4. Draw the predicted Measurements and uncertainty
        ### The center of measurements
        Pred = self.afterKF.x_hat[index]
        H = self.generate_H(Pred, self.dected_corner[index])
        pred_meas = np.dot(H, Pred)
        center_plot_pred_meas,  = self.ax1.plot(pred_meas[0], 
                                                pred_meas[1], 
                                                 'gs', 
                                                 label = "Predicted Measurements")

        ### A line
        measurments_line_center = [center_x_pred, pred_meas[0]]
        measurments_line_end    = [center_y_pred, pred_meas[1]]
        plt_measurments_line,  = self.ax1.plot(measurments_line_center, 
                                               measurments_line_end,
                                               'g', linestyle = '--')
        #### Predicted uncertainty
        elp_pred_meas = error_ellipse([pred_meas[0], pred_meas[1]], 
                                       self.afterKF.W_meas[index][0:2,0:2], 
                                       ax = self.ax1, 
                                       factor=plot_std, 
                                       edgecolor = 'g',
                                       linestyle='--')

        cor_meas = self.ax1.annotate(self.dected_corner[index]+1, 
                                     [pred_meas[0], pred_meas[1]],
                                     xytext=([pred_meas[0], pred_meas[1]]), 
                                     size=20,
                                     color = 'g')
        
        self.current_lines.append(center_plot_pred_meas)
        self.current_lines.append(plt_measurments_line)
        self.current_lines.append(elp_pred_meas)
        self.current_lines.append(cor_meas)

        if index < (len(self.afterKF.x_hat)-1):
                self.car_state_display.\
                        draw_next_prediction(self.afterKF.x_hat[index+1],
                                             self.ax1) 

    def measurments_update(self, utime):
        
        # Measurements
        index = self.global_plot_index

        box = self.truths[utime].corners
        corners = self.lshapes[utime].corners
        cluster = self.lshapes[utime].cluster
        h1, h2, h3 = self.plot_one_frame(box, corners, cluster, self.ax2)
        self.current_lines.append(h1)
        self.current_lines.append(h2)
        self.current_lines.append(h3)

        cloest_corner_index = self.dected_corner[index]

        box_corner = [
                      box[cloest_corner_index][0],
                      box[cloest_corner_index][1]
                      ]
        plt_box_cor,  = self.ax2.plot(box_corner[0], 
                                      box_corner[1], 
                                      'bs', 
                                      label = "Associated Detected Corner")
        self.current_lines.append(plt_box_cor)

        # Plot the annotate 
        cor1 = self.ax2.annotate('1', 
                           box[0,0:2],
                           xytext=(box[0,0]+0.1, box[0,1]+0.1), 
                           size=15,
                           color = 'r')
        cor2 = self.ax2.annotate('2', 
                           box[1,0:2],
                           xytext=(box[1,0]+0.1, box[1,1]+0.1), 
                           size=15,
                           color = 'r')

        cor3 = self.ax2.annotate('3', 
                           box[2,0:2],
                           xytext=(box[2,0]-0.1, box[2,1]-0.1), 
                           size=15,
                           color = 'r')

        cor4 = self.ax2.annotate('4', 
                           box[3,0:2],
                           xytext=(box[3,0]-0.1, box[3,1]-0.1), 
                           size=15,
                           color = 'r')
        self.current_lines.append(cor1)
        self.current_lines.append(cor2)
        self.current_lines.append(cor3)
        self.current_lines.append(cor4)

        # draw the unbiased Uncertainty
        plt_bia_corner, = self.ax2.plot(self.locs[index][0], 
                                        self.locs[index][1], 
                                        'r.', 
                                        label = "measurements")
        self.current_lines.append(plt_bia_corner)

        # draw the measurement ellipse
        cov = self.lshapes[utime].cov[0:2, 0:2]

        elp = error_ellipse([self.locs[index][0], self.locs[index][1]], 
                                cov, 
                                ax = self.ax2, 
                                factor=3.0,
                                edgecolor = 'r')
        self.current_lines.append(elp)

        # Draw the heading and uncertainty
        heading_uncertainty = plot_std *\
                              np.sqrt(self.lshapes[utime].cov[2,2])
        heading = self.headings[self.global_plot_index]
        pt_x = [self.locs[index][0], 
                self.locs[index][0] + 
                        uncertainty_line_scale * np.cos(heading)]
        pt_y = [self.locs[index][1], 
                self.locs[index][1] + 
                        uncertainty_line_scale * np.sin(heading)]
        head_plot, = self.ax2.plot(pt_x, 
                             pt_y, 
                             '#8B2252', 
                             label = "Heading")
        self.current_lines.append(head_plot)

        heading_up   = heading + heading_uncertainty
        heading_down = heading - heading_uncertainty
        length_scale = uncertainty_line_scale
        pt_x = [self.locs[index][0], 
                self.locs[index][0] + length_scale * np.cos(heading_up)]
        pt_y = [self.locs[index][1], 
                self.locs[index][1] + length_scale * np.sin(heading_up)]

        head_plot_u, = self.ax2.plot(pt_x, pt_y, '#8B2252',linestyle='--')
        pt_x = [self.locs[index][0], 
                self.locs[index][0] + length_scale * np.cos(heading_down)]
        pt_y = [self.locs[index][1], 
                self.locs[index][1] + length_scale * np.sin(heading_down)]
        head_plot_d, = self.ax2.plot(pt_x, pt_y, '#8B2252',linestyle='--')

        self.current_lines.append(head_plot_d)
        self.current_lines.append(head_plot_u)


    def draw_clean_list(self):
        for item in self.current_lines:
            item.remove()
        self.current_lines = []

    def quite_program(self, event):
        exit()

    def plot_overview(self, 
                      trackIndex, 
                      ground_truth = None, 
                      track_id = 0,
                      verbose = True):
        

        #f, ax = plt.subplots(figsize=(12, 16))
        self.ground_truth = ground_truth
        self.verbose = verbose
        f = plt.figure(figsize=(12, 16),num='trackIndex {:2}'.format(trackIndex))
        #f.suptitle('figure title')
        gs0 = gridspec.GridSpec(1, 1, figure=f)
        gs00 = gridspec.GridSpecFromSubplotSpec(6, 4, subplot_spec=gs0[0])
        ax1 = f.add_subplot(gs00[0:4, :])
        ax2 = f.add_subplot(gs00[4:6, 0:2])
        ax3 = f.add_subplot(gs00[4:6, 2:4])
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax1.set_title("Information Center")
        self.ax2.set_title("Measurement")
        self.ax3.set_title("Vehicle States")
        self.ax1.axis('equal')
        self.ax2.axis('equal')
        self.ax3.axis('equal')
        
        quite_ax = plt.axes([0.02, 0.00, 0.1,  0.03])
        error_play  = plt.axes([0.44, 0.00, 0.1,  0.03])
        rotate_play = plt.axes([0.56, 0.00, 0.1,  0.03])
        axprev = plt.axes([0.68, 0.00, 0.1,  0.03])
        axnext = plt.axes([0.80, 0.00, 0.1, 0.03])
        axclose = plt.axes([0.91, 0.00, 0.1, 0.03])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next)
        
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev)
        
        bclose = Button(axclose, 'Close')
        bclose.on_clicked(self.close)

        rotate_b = Button(rotate_play, "Rot Play")
        rotate_b.on_clicked(self.rotate_play_ground)
        
        error_b = Button(error_play, "Error Play")
        error_b.on_clicked(self.error_play_ground)

        quite_b = Button(quite_ax, "quite")
        quite_b.on_clicked(self.quite_program)

        self.car_state_display = Car_state()

        f.tight_layout()
        plt.subplots_adjust(bottom=0.05)
        self.draw()

        plt.show()

    def plot_ekf_frame_by_frame(self, ground_truth = None, track_id = 0):
        _, ax = plt.subplots(figsize=(12, 16))
        
        car_state_display = Car_state()
        for index in range(len(self.utime)):
            time_diff = 0.0
            if index > 0:
                time_diff = self.ftimes[index] - self.ftimes[index-1]
                pt1_u = [self.afterKF.locs[index-1][0], 
                         self.afterKF.locs[index][0]]
                pt2_u = [self.afterKF.locs[index-1][1], 
                         self.afterKF.locs[index][1]]
                ax.plot(pt1_u, pt2_u, 'r')
                print("Time Difference {:5.5f}".format(time_diff))
            utime = self.utime[index]
            
            plt.title('gid={:3}, Time Diff={:5.5}'.format(track_id,time_diff))
            plt.tight_layout()

            car_state_display.draw(self.afterKF.x[index], ax)
            car_state_display.draw_uncertainty(self.afterKF.x[index], 
                                               self.afterKF.V[index], 
                                               ax
                                               )
            # Plot the gorund truth state
            plt_ground_truth = None
            if ground_truth is not None: 
                plt_ground_truth, = ax.plot(ground_truth[index][0], 
                                            ground_truth[index][1], 
                                            'ys', 
                                            label = "Ground_truth")
            if index < (len(self.afterKF.x_hat)-1):
                car_state_display.\
                        draw_next_prediction(self.afterKF.x_hat[index+1],
                                                   ax)
        

            box = self.truths[utime].corners
            corners = self.lshapes[utime].corners
            cluster = self.lshapes[utime].cluster
            h1, h2, h3 = self.plot_one_frame(box, corners, cluster)

            # plot the bias on the plot
            plt_bia_corner, = ax.plot(self.locs[index][0], 
                                      self.locs[index][1], 
                                      'r.', 
                                      label = "measurements")

            # Plot the clostest corner
            cloest_corner_index = self.dected_corner[index]

            box_corner = [
                          box[cloest_corner_index][0],
                          box[cloest_corner_index][1]
                          ]
            plt_box_cor,  = ax.plot(box_corner[0],
                                     box_corner[1], 
                                     'bs', 
                                     label = "Associated Detected Corner")

            # Plot the annotate 
            cor1 = ax.annotate('1', 
                               box[0,0:2],
                               xytext=(box[0,0]+0.1, box[0,1]+0.1), 
                               size=20,
                               color = 'r')
            cor2 = ax.annotate('2', 
                               box[1,0:2],
                               xytext=(box[1,0]+0.1, box[1,1]+0.1), 
                               size=20,
                               color = 'r')

            cor3 = ax.annotate('3', 
                               box[2,0:2],
                               xytext=(box[2,0]-0.1, box[2,1]-0.1), 
                               size=20,
                               color = 'r')

            cor4 = ax.annotate('4', 
                               box[3,0:2],
                               xytext=(box[3,0]-0.1, box[3,1]-0.1), 
                               size=20,
                               color = 'r')

            # Draw result for EKF 
            ## 1. Draw the Updated states and Uncertainty.
            center_x = self.afterKF.locs[index][0]
            center_y = self.afterKF.locs[index][1]
            center_plot,  = ax.plot(center_x, 
                                     center_y, 
                                     'rs', 
                                     label = "Updated States")
            
            ### draw Heading
            ### Length propotion to the speed
            heading = self.afterKF.headings[index]
            speed   = self.afterKF.x[index][3] / 2

            pt_x = [center_x, center_x + speed * np.cos(heading)]
            pt_y = [center_y, center_y + speed * np.sin(heading)]
            head_plot, = ax.plot(pt_x, 
                                 pt_y, 
                                 '#8B2252', 
                                 label = "Heading")
            
            ### Draw Heading uncertainty

            heading_uncertainty = plot_std * \
                                  np.sqrt(self.afterKF.V[index][2,2])
            length_scale = min(speed, 1)
            heading_up   = heading + heading_uncertainty
            heading_down = heading - heading_uncertainty
            pt_x = [center_x, center_x + length_scale * np.cos(heading_up)]
            pt_y = [center_y, center_y + length_scale * np.sin(heading_up)]
            head_plot_u, = plt.plot(pt_x, pt_y, '#8B2252')
            pt_x = [center_x, center_x + length_scale * np.cos(heading_down)]
            pt_y = [center_y, center_y + length_scale * np.sin(heading_down)]
            head_plot_d, = plt.plot(pt_x, pt_y, '#8B2252')

            ### Draw Updated state uncertainty
            elp = error_ellipse([center_x, center_y], 
                                self.afterKF.V[index][0:2,0:2], 
                                ax = ax, 
                                factor=plot_std, 
                                edgecolor = 'r')

            ## 2. Draw the predicted states and uncertainty

            center_x_pred = self.afterKF.x_hat[index][0]
            center_y_pred = self.afterKF.x_hat[index][1]
            #center_plot_pred,  = ax.plot(center_x_pred, 
            #                        center_y_pred, 
            #                         'ms', 
            #                         label = "Predicted States")
            ### Draw Predicted state uncertainty
            elp_pred = error_ellipse([center_x_pred, center_y_pred], 
                                     self.afterKF.W[index][0:2,0:2], 
                                     ax = ax, 
                                     factor=plot_std, 
                                     edgecolor = 'm',
                                     linestyle='--')
            ### Draw the predicted heading
            ### Length propotion to the speed
            heading = self.afterKF.x_hat[index][2]
            speed   = self.afterKF.x_hat[index][3] / 2

            pt_x_pred = [center_x_pred, center_x_pred + speed * np.cos(heading)]
            pt_y_pred = [center_y_pred, center_y_pred + speed * np.sin(heading)]
            head_plot_pred, = ax.plot(pt_x_pred, 
                                      pt_y_pred, 
                                      'm',
                                      linestyle='--')
            
            ### Draw Heading uncertainty
            heading_uncertainty = plot_std * \
                                  np.sqrt(self.afterKF.W[index][2,2])
            length_scale = min(speed, 1)
            heading_up   = heading + heading_uncertainty
            heading_down = heading - heading_uncertainty
            pt_x = [center_x_pred, 
                    center_x_pred + length_scale * np.cos(heading_up)]
            pt_y = [center_y_pred, 
                    center_y_pred + length_scale * np.sin(heading_up)]
            head_plot_u_pred, = plt.plot(pt_x, pt_y, 'm', linestyle = '--')
            pt_x = [center_x_pred, 
                    center_x_pred + length_scale * np.cos(heading_down)]
            pt_y = [center_y_pred, 
                    center_y_pred + length_scale * np.sin(heading_down)]
            head_plot_d_pred, = plt.plot(pt_x, pt_y, 'm', linestyle = '--')

            ## 3. Draw the L shape Heading Corner end
            plt_box_cor_head, = ax.plot(corners[self.L_shape_ends[index]][0], 
                                        corners[self.L_shape_ends[index]][1], 
                                        'rs')

            ## 4. Draw the predicted Measurements and uncertainty
            ### The center of measurements
            Pred = self.afterKF.x_hat[index]
            H = self.generate_H(Pred, self.dected_corner[index])
            pred_meas = np.dot(H, Pred)
            center_plot_pred_meas,  = ax.plot(pred_meas[0], 
                                              pred_meas[1], 
                                               'gs', 
                                               label = "Predicted Measurements")

            ### A line
            measurments_line_center = [center_x_pred, pred_meas[0]]
            measurments_line_end    = [center_y_pred, pred_meas[1]]
            plt_measurments_line,  = ax.plot(measurments_line_center, 
                                             measurments_line_end,
                                             'g', linestyle = '--')
            ### Predicted uncertainty
            elp_pred_meas = error_ellipse([pred_meas[0], pred_meas[1]], 
                                           self.afterKF.W_meas[index][0:2,0:2], 
                                           ax = ax, 
                                           factor=plot_std, 
                                           edgecolor = 'g',
                                           linestyle='--')
            
            cor_meas = ax.annotate(self.dected_corner[index]+1, 
                                   [pred_meas[0], pred_meas[1]],
                                   xytext=([pred_meas[0], pred_meas[1]]), 
                                   size=20,
                                   color = 'g')

            plt.ylim(min(box[:,1]) - 3, max(box[:,1]) + 3)
            plt.xlim(min(box[:,0]) - 3, max(box[:,0]) + 3)
            plt.legend()
            
            plt.waitforbuttonpress()
            #plt.show()

            cor1.remove()
            cor2.remove()
            cor3.remove()
            cor4.remove()
            cor_meas.remove()
            
            plt.gca().lines.remove(h3)
            plt.gca().lines.remove(h2)
            plt.setp(h1, color='gray', linestyle='-.', label = None)
            ax.lines.remove(plt_bia_corner)
            ax.lines.remove(plt_box_cor)
            #ax.lines.remove(center_plot)
            plt.setp(center_plot, label = None)
            ax.lines.remove(head_plot)
            ax.lines.remove(head_plot_u)
            ax.lines.remove(head_plot_d)
            ax.patches.remove(elp)
            ax.patches.remove(elp_pred)
            if plt_ground_truth is not None:
                plt_ground_truth.remove()


            #ax.lines.remove(center_plot_pred)
            ax.lines.remove(head_plot_pred)
            ax.lines.remove(head_plot_u_pred)
            ax.lines.remove(head_plot_d_pred)
            ax.lines.remove(plt_box_cor_head)
            ax.lines.remove(center_plot_pred_meas)
            ax.lines.remove(plt_measurments_line)
            ax.patches.remove(elp_pred_meas)

            car_state_display.clean_drawing(plt, ax)


            

        
        plt.close()
