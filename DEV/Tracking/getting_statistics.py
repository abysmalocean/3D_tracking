import os
import sys

import numpy as np
#`from main import iou3d, convert_3dbox_to_8corner
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

import argparse

# load the self libaray 
from utils.dict import *


NUSCENES_TRACKING_NAMES = [
  #'bicycle',
  'bus',
  'car',
  #'motorcycle',
  #'pedestrian',
  #'trailer',
  'truck'
]

def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

if __name__ == '__main__':
      # Settings.
    parser = argparse.ArgumentParser(description='Get nuScenes stats.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eval_set', type=str, default='train',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the NIPS 2019 configuration will be used.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--matching_dist', type=str, default='2d_center',
                        help='Which distance function for matching, 3d_iou or 2d_center.')
    args = parser.parse_args()

    eval_set_ = args.eval_set
    config_path = args.config_path
    verbose_ = bool(args.verbose)
    matching_dist = args.matching_dist

    if config_path == '':
        cfg_ = config_factory('tracking_nips_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))
    
    if 'train' in eval_set_:
        detection_file = os.path.join(detection_path , 'megvii_train.json')
        data_root      = os.path.join(nuscense_path , 'trainval')
        version='v1.0-trainval'
    elif 'val' in eval_set_:
        detection_file = os.path.join(detection_path , 'megvii_val.json')
        data_root      = os.path.join(nuscense_path , 'trainval')
        version='v1.0-trainval'
    elif 'test' in eval_set_:
        detection_file = os.path.join(detection_path , 'megvii_test.json')
        data_root      = os.path.join(nuscense_path , 'test')
        version='v1.0-test'
    
    #nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    # load the prediction box
    print("starting loading the detection box")
    pred_boxes, _ = load_prediction(detection_file, 10000, DetectionBox)
    print("starting loading the groudh truth box")
    gt_boxes = load_gt(nusc, eval_set_, TrackingBox)


