import numpy as np
from math import sin, sqrt, cos
from pyquaternion import Quaternion
import pickle

from utils.utils import angle_difference
from utils.utils import quaternion_yaw
from EKF.EKF import *
from utils.dict import *




class EKF_wraper(object): 
    def __init__(self , tracking_name = 'car'):
        super(EKF_wraper, self).__init__()
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
        
        self.KalmanFilters = []
        self.weights       = []
        
        self.wheelbase_to_length_ratio = 0.7
        
        
    
    def load_data_set(self, datasets): 
        """
        Load the Nuscense dataset. Internal data type
        Transform one track to the EKF format
        """
        #print(len(datasets))
        for index in range(len(datasets)): 
            current_frame = datasets[index]
            self.locs.append(current_frame['detection'].translation)
            self.headings.append(quaternion_yaw(Quaternion(
                        current_frame['detection'].rotation)))
            self.shapes.append(current_frame['detection'].size)
            # time frame
            self.ftimes.append(current_frame['sample_data_lidar']['timestamp'] / 1000000)
            
            # angle to the ego vehicle
            ego_angle = quaternion_yaw(Quaternion(current_frame['ego_pose']['rotation']))
            #print("ego_angle ", ego_angle, " detected vehicle th ", self.headings[-1])
            angle_dif1 = angle_difference(ego_angle, self.headings[-1])
            angle_dif2 = angle_difference(ego_angle, self.headings[-1] + np.pi)
            self.angle.append(min([angle_dif1, angle_dif2], key=abs))
            
            # distance to ego_vehicle
            location_ego = current_frame['ego_pose']['translation'][0:2]
            dist = np.sqrt((self.locs[-1][0] - location_ego[0])**2 +\
                           (self.locs[-1][1] - location_ego[1])**2)
            self.dist.append(dist)
            
            # number of Lidar Points
            n_lidar_point = current_frame['sample_annotation']['num_lidar_pts']
            self.n_lidar_points.append(n_lidar_point)
    
    def predict(self, dt): 
        for ekf in self.KalmanFilters:
            ekf.predict(dt)
    
    def measurement_predict(self): 
        for ekf in self.KalmanFilters:
            ekf.measurement_predict()
    def update(self, locs, heading, shape): 
        
        for ekf in self.KalmanFilters:
            detection = self.generate_detection(locs, 
                                                heading, 
                                                shape)
            ekf.update(detection)
    
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
        print("runing the EKF")
        
        pre_time = self.ftimes[0]
        for step in range(len(self.ftimes)): 
            dt = self.ftimes[step] - pre_time
            # first predict then update
            if step != 0: 
                self.predict(dt)
            # measurement prediction
            self.measurement_predict()
            
            # Update
            self.update(locs=self.locs[step], 
                        heading=self.headings[step],
                        shape=self.shapes[step])
            
            
    
    def initial_kf(self):
        ekf_tmp = EKF()
        v = sqrt(self.initials['mean'][self.tracking_name][7] ** 2 +\
                 self.initials['mean'][self.tracking_name][8] ** 2)
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
        P_0 = np.diag(np.array([1.**2, 
                      1.**2, 
                      (np.pi * 10 / 180.0)**2, 
                      (10**2), 
                      (0.1 * 0.1), 
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
        ekf_tmp.create_initial(x_0=x_0, p_0 = P_0, q_0 = Q_0)
        self.KalmanFilters.append(ekf_tmp)
        
        
        
            