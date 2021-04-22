import os
import sys

import numpy as np
#from main import iou3d, convert_3dbox_to_8corner
from sklearn.utils.linear_assignment_ import linear_assignment

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.loaders import create_tracks

from pyquaternion import Quaternion
from tqdm import tqdm

import argparse
import pickle

from utils.utils import angle_difference

NUSCENES_TRACKING_NAMES = [
  #'bicycle',
  'bus',
  'car',
  #'motorcycle',
  #'pedestrian',
  #'trailer',
  'truck'
]

NAMES = [
    #'vehicle.truck', 
    'vehicle.car', 
    #'vehicle.bus'
]

def training_data_creation(nusc,
                           pred_boxes, 
                           gt_boxes, 
                           tracks_gt):
    '''
    For each sample token, find matches of pred_boxes and gt_boxes, then get stats.
    tracks_gt has the temporal order info for each sample_token
    '''
    tracks = []
    scene_count = 0
    print("Number of instance ", len(nusc.instance))
    """
    cat_rec = nusc.get('category', nusc.instance[0]['category_token'])
    print("Instance ", cat_rec)
    """
    tracks_set = []
    for index in tqdm(range(len(nusc.instance))): 
        if(nusc.instance[index]['nbr_annotations'] > 25):
            cat_rec = nusc.get('category', nusc.instance[index]['category_token'])
            #print(cat_rec)
            if(cat_rec['name'] in NAMES): 
                tracks_set.append(index)
    print("total number of data sets ", len(tracks_set))
    
    for index in tqdm(tracks_set): 
        """
        instance 
        {
            "token":                    <str> -- Unique record identifier.
             "category_token":          <str> -- Foreign key pointing to the 
                                                 object category.
             "nbr_annotations":         <int> -- Number of annotations of this 
                                                 instance.
             "first_annotation_token":  <str> -- Foreign key. Points to the first 
                                                 annotation of this instance.
            "last_annotation_token":    <str> -- Foreign key. Points to the 
                                                 last annotation of this instance.
        }
        
        sample_annotation {
           "token":                   <str> -- Unique record identifier.
           "sample_token":            <str> -- Foreign key. NOTE: this points to a sample NOT a sample_data since annotations are done on the sample level taking all relevant sample_data into account.
           "instance_token":          <str> -- Foreign key. Which object instance is this annotating. An instance can have multiple annotations over time.
           "attribute_tokens":        <str> [n] -- Foreign keys. List of attributes for this annotation. Attributes can change over time, so they belong here, not in the instance table.
           "visibility_token":        <str> -- Foreign key. Visibility may also change over time. If no visibility is annotated, the token is an empty string.
           "translation":             <float> [3] -- Bounding box location in meters as center_x, center_y, center_z.
           "size":                    <float> [3] -- Bounding box size in meters as width, length, height.
           "rotation":                <float> [4] -- Bounding box orientation as quaternion: w, x, y, z.
           "num_lidar_pts":           <int> -- Number of lidar points in this box. Points are counted during the lidar sweep identified with this sample.
           "num_radar_pts":           <int> -- Number of radar points in this box. Points are counted during the radar sweep identified with this sample. This number is summed across all radar sensors without any invalid point filtering.
           "next":                    <str> -- Foreign key. Sample annotation from the same object instance that follows this in time. Empty if this is the last annotation for this object.
           "prev":                    <str> -- Foreign key. Sample annotation from the same object instance that precedes this in time. Empty if this is the first annotation for this object.
        }
        
        sample_data {
            "token":                   <str> -- Unique record identifier.
            "sample_token":            <str> -- Foreign key. Sample to which this sample_data is associated.
            "ego_pose_token":          <str> -- Foreign key.
            "calibrated_sensor_token": <str> -- Foreign key.
            "filename":                <str> -- Relative path to data-blob on disk.
            "fileformat":              <str> -- Data file format.
            "width":                   <int> -- If the sample data is an image, this is the image width in pixels.
            "height":                  <int> -- If the sample data is an image, this is the image height in pixels.
            "timestamp":               <int> -- Unix time stamp.
            "is_key_frame":            <bool> -- True if sample_data is part of key_frame, else False.
            "next":                    <str> -- Foreign key. Sample data from the same sensor that follows this in time. Empty if end of scene.
            "prev":                    <str> -- Foreign key. Sample data from the same sensor that precedes this in time. Empty if start of scene.
         }


        """
        current_instance = nusc.instance[index]
        first_annotation_token = current_instance['first_annotation_token']
        last_annotation_token  = current_instance['last_annotation_token']
        current_annotation_token = first_annotation_token
        num_annotation = current_instance['nbr_annotations']
        #print("first Ann token ", first_annotation_token)
        #print("Last Ann token ", last_annotation_token)
        count = 0
        tmp_tracks = []
        
        update_interval = 0 
        while count < num_annotation and update_interval < 4:
            #print(current_annotation_token)
            count += 1
            update_interval += 1
            current_annotation = nusc.get('sample_annotation', current_annotation_token)
            current_annotation_token = current_annotation['next']
            current_sample_token = current_annotation['sample_token']
            det_all = [box for box in pred_boxes.boxes[current_sample_token] if box.detection_name == 'car']
            if len(det_all) == 0: 
                continue
            x,y,z = current_annotation['translation']
            dets = np.stack([np.array([
                             box.translation[0], 
                             box.translation[1]
                             ]) for box in det_all], axis=0)
            distance = np.sqrt((dets[:, 0] - x)**2 + (dets[:, 1] - y)**2)
            
            min_index = np.argmin(distance)
            tmp_track_ele = {}
            if (distance[min_index]) < 2.0 :
                # Append the data to the tracks
                update_interval = 0
                tmp_track_ele['sample_annotation'] = current_annotation
                tmp_track_ele['sample'] = nusc.get('sample', current_annotation['sample_token'])
                sensor_token = tmp_track_ele['sample']['data']['LIDAR_TOP']
                sample_data_tmp = nusc.get('sample_data', sensor_token)
                tmp_track_ele['sample_data_lidar'] = sample_data_tmp
                tmp_track_ele['ego_pose'] = nusc.get('ego_pose', sample_data_tmp['ego_pose_token'])
                tmp_track_ele['detection'] =  det_all[min_index]
                tmp_tracks.append(tmp_track_ele)
        #print("Length of the created track", len(tmp_tracks))
        if len(tmp_tracks) > 25: 
            tracks.append(tmp_tracks)
        assert current_annotation, last_annotation_token
    
    training_file_name = "/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/training_data"
    with open(training_file_name, 'wb') as f:
        pickle.dump(tracks, f)
    print("Total Number of tracks ", len(tracks))
        
            
        
    
    for scene_token in tqdm(tracks_gt.keys()):
        size_of_sample_in_scene = len(tracks_gt[scene_token].keys())
        print("In [ " , 
              scene_count , 
              " ] scene_token ", 
              scene_token, 
              " has [ ", 
              size_of_sample_in_scene , 
              " ] sample")
        scene_count += 1
        
    

