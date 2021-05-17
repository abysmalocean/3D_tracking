import numpy as np
from math import sin, sqrt, cos
from pyquaternion import Quaternion
import pickle
import copy



from utils.utils import angle_difference
from utils.utils import quaternion_yaw
from utils.utils import normailized_heading, log_sum
from EKF.EKF import *
from utils.dict import *

DEBUG = False


class EKF_wraper(object): 
    def __init__(self , 
                 uncertainty_net = None, 
                 tracking_name = 'car'):
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
        
        self.KalmanFilters = [EKF(), EKF(), EKF(), EKF()]
        self.weights       = []
        
        self.angle_arguments = [0.0, np.pi / 2, -np.pi / 2, np.pi]
        self.measurement_weights = [0.9, 0.1, 0.1, 0.1]
        
        self.wheelbase_to_length_ratio = 0.7
        
        self.after_filter_states = []
        
        # load the uncertainty_net.
        self.uncertainty_net = uncertainty_net
        
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
        
        # should create different heading angles
        new_weight = copy.copy(self.weights)
        
        # TODO: add the uncertatinty here.
        
        for filter_index, ekf in enumerate(self.KalmanFilters):
            # use to update the different angels
            hypothesis_weights = [0.0, 0.0, 0.0, 0.0]
            detections         = []
            #print("Hypothesis index ", filter_index)
            #print("current predicted heading ", ekf.x.item(2))
            # step 1, find the hypothesis
            for index in range(4):
                psudo_heading = copy.copy(heading + self.angle_arguments[index])
                psudo_heading = normailized_heading(psudo_heading)
                #print("measurment heading ", psudo_heading)
                detection = self.generate_detection(locs, 
                                                    psudo_heading, 
                                                    shape)
                detections.append(detection)
                hypothesis_weights[index] = ekf.measurement_likelihood(detection)
                #print("\n")
                hypothesis_weights[index] = ekf.measurement_likelihood(detection) +\
                                            np.log(self.measurement_weights[index])
            #print("each heading weight ", hypothesis_weights)
            update_detection_index = np.argmax(hypothesis_weights)
            # setp 2: selecting the most possible hypothesis then update
            ekf.update(detections[update_detection_index])
            new_weight[filter_index] += hypothesis_weights[update_detection_index]
        
        log_weights = log_sum(copy.copy(new_weight))
        self.weights = new_weight - log_weights
        
        # setp 3. get the posterior
        max_weight_hypothesis_index = np.argmax(self.weights)
        self.after_filter_states.append(self.KalmanFilters[max_weight_hypothesis_index].x)
        

        #print("index going to removed ", remove_indexs)
    
    def reduce_kf(self):
        # step 4. remove the additional KF if the weight too small
        #print("weights ", self.weights)
        remove_indexs = []
        new_list_filter = []
        new_list_weight = []
        
        for index in range(len(self.KalmanFilters)): 
            if self.weights[index] < np.log(0.00001): # TODO: change this number after figure out the updating
                remove_indexs.append(index)
            else: 
                new_list_filter.append(self.KalmanFilters[index])
                new_list_weight.append(self.weights[index])
        self.KalmanFilters = new_list_filter
        self.weights = new_list_weight
            
    
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
    
    def run_ekf(self, run_smoother = False):
        self.initial_kf()
        #print("runing the EKF")
        
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
            #print("current weights ", np.exp(self.weights))
            if len(self.KalmanFilters) > 1 and step != 0:
                if DEBUG:
                    for ekf_i, ekf in enumerate(self.KalmanFilters):
                        print( "[ ", ekf_i, "] heading : ", ekf.x[2])
                self.reduce_kf()
            elif step == 0: 
                self.weights = np.log([0.35, 0.20, 0.2, 0.25])
            
            pre_time = self.ftimes[step]
            if DEBUG: 
                print("step ", step, self.locs[step], " dt ", dt)
            
                print("detection x:{}, y:{}, heading:{}".format(self.locs[step][0], 
                                                                               self.locs[step][1],
                                                                               self.headings[step]) )

                print("x {}, y {}, th{}, v{}, ph:{}, w:{}, l:{}".format(self.after_filter_states[-1][0],
                                                                        self.after_filter_states[-1][1],
                                                                        self.after_filter_states[-1][2],
                                                                        self.after_filter_states[-1][3],
                                                                        self.after_filter_states[-1][4],
                                                                        self.after_filter_states[-1][5],
                                                                        self.after_filter_states[-1][6]))
                print("Hypotheses weights ", np.exp(self.weights))
                print("\n")
        if run_smoother: 
            for ekf in self.KalmanFilters:
                ekf.smoother(self.ftimes)
            
        #print("Size of posterior", len(self.KalmanFilters[0].post_x))
            
    
    def initial_kf(self):
        
        v = sqrt(self.initials['mean'][self.tracking_name][7] ** 2 +\
                 self.initials['mean'][self.tracking_name][8] ** 2)
        v = 5.0
        l = self.shapes[0][1]
        th = self.headings[0]
        
        x_0  = np.array([self.locs[0][0] - (self.wheelbase_to_length_ratio - 0.5) * l * cos(th), 
                         self.locs[0][1] - (self.wheelbase_to_length_ratio - 0.5) * l * sin(th), 
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
        Q_0[0][0] = self.meas_var[self.tracking_name][3] + .5
        Q_0[1][1] = self.meas_var[self.tracking_name][4] + .5
        Q_0[2][2] = self.meas_var[self.tracking_name][5] + 0.1
        
        Q_0[3][3] = self.meas_var[self.tracking_name][2]    
        Q_0[4][4] = self.meas_var[self.tracking_name][6]
        #Q_0[5][5] = self.meas_var[self.tracking_name][1]
        #Q_0[6][6] = self.meas_var[self.tracking_name][0]
        
        
        
        # should create multiple EKF with different headings.
        self.weights = [0.35, 0.2, 0.2, 0.2]
        
        #print("original angle ", x_0.item(2) * (180.0 / np.pi))
        
        for index in range(4):
            ang = normailized_heading(x_0.item(2) + self.angle_arguments[index])
            x = copy.copy(x_0)
            x.itemset(2, ang)
            self.KalmanFilters[index].create_initial(x_0=x, 
                                                     p_0 = copy.copy(P_0), 
                                                     q_0 = copy.copy(Q_0))
    def plot_result(self, smoother = False):
        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111)
        loc = np.array(self.locs)
        plt.scatter(loc[0,0], loc[0,1], c='r', s=500, marker="x", 
                label="{}".format("begining"))
        
        plt.scatter(loc[1:,0], loc[1:,1], c='b', s=50, marker="+", 
                label="{}".format("Center Measurements"))
        
        if smoother:
            loc = np.array(self.KalmanFilters[0].afterSmooth.x)[:,0:2]
            headings = np.array(self.KalmanFilters[0].afterSmooth.x)[:,2]
            string_out = "After Kalman Smoother"
        else: 
            loc = np.array(self.after_filter_states)
            headings = np.array(self.after_filter_states)[:,2]
            string_out = "After Kalman Filter"
            
        plt.scatter(loc[:,0], loc[:,1], c='y', s=50, marker="1", 
                label="{}".format(string_out))
        plt.plot(loc[:,0], loc[:,1], c='y')
        
        # plot heading
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
        
        
        
            