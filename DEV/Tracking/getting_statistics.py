import os
import sys

import numpy as np
from main import iou3d, convert_3dbox_to_8corner
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

  