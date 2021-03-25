from __future__ import print_function
import os.path, copy, numpy as np, time, sys
from numba import jit
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from filterpy.kalman import KalmanFilter
from scipy.spatial import ConvexHull
import json
# load the nuscense library
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox 
from nuscenes.eval.detection.data_classes import DetectionBox
from pyquaternion import Quaternion
from tqdm import tqdm

# load the self libaray 
from utils.dict import *
from utils.utils import mkdir_if_missing

def track_nuscenes(data_split, 
                   covariance_id, 
                   match_distance, 
                   match_threshold, 
                   match_algorithm, 
                   save_root, 
                   use_angular_velocity):
    # TODO: implementing this fuction for tracking
    print("working on the track_nuscense")
    save_dir = os.path.join(save_root, data_split)
    print(save_dir)
    mkdir_if_missing(save_dir)
    if 'train' in data_split:
        detection_file = os.path.join(utils.dict.detection_path , 'megvii_train.json')
        data_root = '/juno/u/hkchiu/dataset/nuscenes/trainval'
        version='v1.0-trainval'
        output_path = os.path.join(save_dir, 'results_train_probabilistic_tracking.json')
    elif 'val' in data_split:
        detection_file = os.path.join(detection_path , 'megvii_val.json')
        data_root = '/juno/u/hkchiu/dataset/nuscenes/trainval'
        version='v1.0-trainval'
        output_path = os.path.join(save_dir, 'results_val_probabilistic_tracking.json')
    elif 'test' in data_split:
        detection_file = '/juno/u/hkchiu/dataset/nuscenes_new/megvii_test.json'
        data_root = '/juno/u/hkchiu/dataset/nuscenes/test'
        version='v1.0-test'
        output_path = os.path.join(save_dir, 'results_test_probabilistic_tracking.json')
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    result = {}
    total_time = 0.0
    total_frames = 0


    

if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv)!=9:
        print("Usage: python main.py data_split(train, val, test) covariance_id(0, 1, 2) match_distance(iou or m) match_threshold match_algorithm(greedy or h) use_angular_velocity(true or false) dataset save_root")
        sys.exit(1)

    data_split = sys.argv[1]
    covariance_id = int(sys.argv[2])
    match_distance = sys.argv[3]
    match_threshold = float(sys.argv[4])
    match_algorithm = sys.argv[5]
    use_angular_velocity = sys.argv[6] == 'True' or sys.argv[6] == 'true'
    dataset = sys.argv[7]
    save_root = os.path.join('' + sys.argv[8])

    # TODO: implement the track_nuscenes
    track_nuscenes(data_split, 
                   covariance_id, 
                   match_distance, 
                   match_threshold, 
                   match_algorithm, 
                   save_root, 
                   use_angular_velocity)


