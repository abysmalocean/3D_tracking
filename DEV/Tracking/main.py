from __future__ import print_function
import os.path, copy, numpy as np, time, sys
from numba import jit
import argparse
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
import pickle

# some common paths
from utils.dict import *
from utils.utils import quaternion_yaw, format_sample_result
from AB3DMOT import AB3DMOT

def run_tracking(
    args,
    nusc, 
    all_results): 
    print("Liang Xu in the tracking algorithm")
    
    # build the result dict
    results = {}
    total_time = 0.0
    total_frames = 0 
    
    processed_scene_tokens = set()
    for sample_token_idx in tqdm(range(len(all_results.sample_tokens))):
        sample_token = all_results.sample_tokens[sample_token_idx]
        # from the sample_token find the according scene
        scene_token  = nusc.get('sample', sample_token)['scene_token']
        if scene_token in processed_scene_tokens:
            continue
        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        current_sample_token = first_sample_token
        
        
        covariance_id = 2
        use_angular_velocity = False
        # build AB3DMOT for different class {car, bus ....}
        mot_trackers = {tracking_name: 
             AB3DMOT(covariance_id, 
             tracking_name=tracking_name, 
             use_angular_velocity=use_angular_velocity, 
             tracking_nuscenes=True) 
           for tracking_name in NUSCENES_TRACKING_NAMES}
        while current_sample_token != '':
            # extract all the detections for each class for current sample
            results[current_sample_token] = []
            dets = {tracking_name: [] 
                        for tracking_name in NUSCENES_TRACKING_NAMES}
            info = {tracking_name: [] 
                        for tracking_name in NUSCENES_TRACKING_NAMES}
            # all detections
            for box in all_results.boxes[current_sample_token]: 
                if box.detection_name not in NUSCENES_TRACKING_NAMES: 
                    continue
                q = Quaternion(box.rotation)
                angle = quaternion_yaw(q)
                # detection format [h, w, l, x, y, z, rot_y]
                detection = np.array(
                    [
                        box.size[2], 
                        box.size[0], 
                        box.size[1], 
                        box.translation[0],  
                        box.translation[1], 
                        box.translation[2],
                        angle
                    ]
                )
                information = np.array([box.detection_score])
                dets[box.detection_name].append(detection)
                info[box.detection_name].append(information)
            # build all the detection to a dict 
            dets_all = {
                tracking_name: {
                    'dets' : np.array(dets[tracking_name]), 
                    'info' : np.array(info[tracking_name])
                } for tracking_name in NUSCENES_TRACKING_NAMES
            }
            
            total_frames += 1
            start_time = time.time()
            # fed each trackers the detection data
            for tracking_name in NUSCENES_TRACKING_NAMES: 
                # if we detect anything in this class
                if dets_all[tracking_name]['dets'].shape[0] > 0: 
                    # run tracker updates
                    trackers_results = mot_trackers[tracking_name].update(
                        dets_all[tracking_name], 
                        match_distance = 'iou',
                        match_threshold = 0.1, 
                        match_algorithm = 'h', 
                        seq_name = scene_token 
                    )
                    # process the results
                    # result foramt (N, 9)
                    # (h, w, l, x, y, z, rot_y), tracking_id, tracking_score 
                    # change the format to the result needed by the algorihtm
                    for i in range(trackers_results.shape[0]): 
                        sample_result = format_sample_result(
                            current_sample_token, 
                            tracking_name, 
                            trackers_results[i]
                        )
                        results[current_sample_token].append(sample_result)
            cycle_time = time.time() - start_time
            total_time += cycle_time
            # tracking programming should above this code
            # get next frame and continue the while loop
            current_sample_token = nusc.get('sample', current_sample_token)['next']
                
        # code cor tracking inside the scene should above this code
        # finish processing this scene, go to next scene
        processed_scene_tokens.add(scene_token)
    
    
    

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Parameters")
    parser.add_argument(
        "--data_split",
        default="val"
        )
    parser.add_argument(
        "--saved_root",
        default='results/lastrun'
    )
    args = parser.parse_args()
    return args 

def main(): 
    print("Working on the tracking")
    args = parse_args()
    
    if args.data_split == 'train': 
        detection_file = center_point_detection_train
        output_path = os.path.join(args.saved_root, 'results_train_probabilistic_tracking.json')
        nusc_file   = nusc_train_val_pickle_file
    elif args.data_split == 'val':
        detection_file = center_point_detection_val
        output_path = os.path.join(args.saved_root, 'results_val_probabilistic_tracking.json')
        nusc_file   = nusc_train_val_pickle_file
    elif args.data_split == 'test':
        detection_file = center_point_detection_test
        output_path = os.path.join(args.saved_root, 'results_test_probabilistic_tracking.json')
        nusc_file   = nusc_test_pickle_file
    
    # load the nuscenes file
    nusc = pickle.load(open(nusc_file , 'rb'))
    # load the detection data
    with open(detection_file) as f:
        data = json.load(f)
    all_results = EvalBoxes.deserialize(data['results'], DetectionBox)
    meta = data['meta']
    print('meta: ', meta)
    print("Loaded results from {}. Found detections for {} samples."
        .format(detection_file, len(all_results.sample_tokens)))

if __name__ == "__main__":
    main()