import numpy as np
from math import sin, sqrt, cos
from pyquaternion import Quaternion
import pickle
import copy



from utils.utils import angle_difference
from utils.utils import quaternion_yaw, yaw2quaternion
from utils.utils import normailized_heading, log_sum
from EKF.EKF import *
from utils.dict import *

DEBUG = False

class EKF_GT(object): 
    def __init__(self , tracking_name = 'car'):
        super(EKF_GT, self).__init__()
        self.ftimes   = []
        self.locs     = [] # center_x, center_y, center_z.
        self.headings = []
        self.shapes   = [] #  width, length, height.
        self.angle    = []
        self.dist     = []
        self.n_lidar_points = []
        
        self.tracking_name = tracking_name
        self.initials = pickle.load(open(processNoise_file_name , 'rb'))
        self.meas_var = pickle.load(open(MeasurementNoise_file_name, 'rb'))
        self.weights       = []
        
        self.angle_arguments = [0.0, np.pi / 2, -np.pi / 2, np.pi]
        self.measurement_weights = [0.9, 0.1, 0.1, 0.1]
        
        self.wheelbase_to_length_ratio = 0.7
        
        self.after_filter_states = []
        
        self.KalmanFilter = EKF()
    
    def correct_heading(self, datasets): 
        """
        This functions is used for correct the heading angles with the
        groudnd truth data. 
        Input is one track of the training data. 
        """
        corrected_data = []
        for index in range(len(datasets)): 
            current_frame = copy.copy(datasets[index])
            measruementHeading = quaternion_yaw(Quaternion(
                        current_frame['detection'].rotation))
            GT_heading         = quaternion_yaw(Quaternion(
                        current_frame['sample_annotation']['rotation']))
            #diff_ang = angle_difference(measruementHeading, GT_heading)
            angle_diff = np.zeros(4)
            angles     = np.zeros(4)
            angles     = [measruementHeading, 
                          measruementHeading + np.pi, 
                          measruementHeading + np.pi / 2, 
                          measruementHeading - np.pi / 2]
            angle_diff[0]  = angle_difference(measruementHeading          ,  GT_heading)
            angle_diff[1]  = angle_difference(measruementHeading + np.pi  ,  GT_heading)
            angle_diff[2]  = angle_difference(measruementHeading + np.pi/2,  GT_heading)
            angle_diff[3]  = angle_difference(measruementHeading - np.pi/2,  GT_heading)
            diff_ang_index = np.argmin(np.abs(angle_diff))
            
            corrected_angle = angles[diff_ang_index]
            corrected_q = yaw2quaternion(corrected_angle)
            
            # correct the detection data to the GT data
            #current_frame['detection'].rotation = current_frame['sample_annotation']['rotation']
            #current_frame['detection'].translation = current_frame['sample_annotation']['translation']
            #current_frame['detection'].shapes = current_frame['sample_annotation']['size']
            
            # corrent the heading of the detection data
            current_frame['detection'].rotation = corrected_q
            
            
            self.locs.append(current_frame['sample_annotation']['translation'])
            self.headings.append(GT_heading)
            self.shapes.append(current_frame['sample_annotation']['size'])
            self.ftimes.append(current_frame['sample_data_lidar']['timestamp'] / 1000000)
            
            corrected_data.append(current_frame)
        return corrected_data
    
    def generate_detection(self, 
                           locs, 
                           heading, 
                           shape): 
        det = np.array([locs[0], 
                        locs[1],
                        heading,
                        shape[0],
                        shape[1]])[:, None]
        return det
    
    def run_ekf(self):
        self.initial_kf()
        #print("runing the EKF")
        pre_time = self.ftimes[0]
        for step in range(len(self.ftimes)): 
            dt = self.ftimes[step] - pre_time
            if step != 0: 
                self.KalmanFilter.predict(dt)
            self.KalmanFilter.measurement_predict()
            
            detection = self.generate_detection(self.locs[step], 
                                                self.headings[step], 
                                                self.shapes[step])
            self.KalmanFilter.update(detection)
            self.after_filter_states.append(self.KalmanFilter.x)
        self.KalmanFilter.afterSmooth.x = self.KalmanFilter.x_smooth
        #self.KalmanFilter.smoother(self.ftimes)
    
    def initial_kf(self):
        
        v = sqrt(self.initials['mean'][self.tracking_name][7] ** 2 +\
                 self.initials['mean'][self.tracking_name][8] ** 2)
        v = 5.0
        l = self.shapes[0][1]
        th = self.headings[0]
        
        x_0  = np.array([self.locs[0][0] - self.wheelbase_to_length_ratio * l * cos(th) / 2.0 , 
                         self.locs[0][1] - self.wheelbase_to_length_ratio * l * sin(th) / 2.0 , 
                         th, 
                         v, 
                         0.0, 
                         self.initials['mean'][self.tracking_name][1],
                         self.initials['mean'][self.tracking_name][2]
                         #self.initials['mean'][self.tracking_name][0], 
                         #self.initials['mean'][self.tracking_name][5]
                        ])[:, None]
        
        P_0 = np.diag(np.array([2.**2, 
                      2.**2, 
                      (np.pi * 15 / 180.0)**2, 
                      (20**2), 
                      (0.2 * 0.2), 
                      self.initials['var'][self.tracking_name][1] + 0.1,
                      self.initials['var'][self.tracking_name][2] + 0.1
                      #self.initials['var'][self.tracking_name][0] + 0.1, 
                      #self.initials['var'][self.tracking_name][5] + 0.1
                      ]))
    
        Q_0 = np.zeros((5, 5))
        Q_0[0][0] = self.meas_var[self.tracking_name][3]
        Q_0[1][1] = self.meas_var[self.tracking_name][4]
        Q_0[2][2] = self.meas_var[self.tracking_name][5] 
        
        Q_0[3][3] = self.meas_var[self.tracking_name][2]    
        Q_0[4][4] = self.meas_var[self.tracking_name][6]
        #Q_0[5][5] = self.meas_var[self.tracking_name][1]
        #Q_0[6][6] = self.meas_var[self.tracking_name][0]
        
        
        # should create multiple EKF with different headings.
        self.weights = [0.35, 0.2, 0.2, 0.2]
        
        #print("original angle ", x_0.item(2) * (180.0 / np.pi))
        
        self.KalmanFilter.create_initial(x_0 = x_0, 
                                         p_0 = copy.copy(P_0), 
                                         q_0 = copy.copy(Q_0))
            
    def plot_result(self):
        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111)
        loc = np.array(self.locs)
        plt.scatter(loc[0,0], loc[0,1], c='r', s=500, marker="x", 
                label="{}".format("begining"))
        
        plt.scatter(loc[1:,0], loc[1:,1], c='b', s=50, marker="+", 
                label="{}".format("Center Measurements"))
        
        loc = np.array(self.KalmanFilter.afterSmooth.x)[:,0:2]
        headings = np.array(self.KalmanFilter.afterSmooth.x)[:,2]
        
        plt.scatter(loc[:,0], loc[:,1], c='y', s=50, marker="1", 
                label="{}".format("After Kalman Filter"))
        plt.plot(loc[:,0], loc[:,1], c='y')
        
        # plot heading
        #headings = self.headings
        ax.quiver(list(list(zip(*loc))[0]),
                  list(list(zip(*loc))[1]),
                  [np.cos(h) for h in headings],
                  [np.sin(h) for h in headings],
                  angles='xy',
                  scale_units='xy',
                  scale= 0.5 ,
                  color='g')
        
        ax.quiver(loc[0][0],
                  loc[0][1],
                  np.cos(headings[0]),
                  np.sin(headings[0]),
                  angles='xy',
                  scale_units='xy',
                  scale= 0.5 ,
                  color='r')
        
        ax.axis('equal')
        ax.legend()
        plt.show()
            
            
            
            
                
            
            
        
        
        
        
        



