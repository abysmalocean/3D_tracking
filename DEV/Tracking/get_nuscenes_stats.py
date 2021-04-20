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
  'bicycle',
  'bus',
  'car',
  'motorcycle',
  'pedestrian',
  'trailer',
  'truck'
]

def quaternion_yaw(q: Quaternion):
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def rotation_to_yaw_agnle(rotation): 
  q = Quaternion(rotation)
  return quaternion_yaw(q)

def rotation_to_positive_z_angle(rotation):
  q = Quaternion(rotation)
  angle = q.angle if q.axis[2] > 0 else -q.angle
  
  return angle

def get_mean(tracks):
  '''
  Input:
    tracks: {scene_token:  {t: [TrackingBox]}}
  '''
  print('len(tracks.keys()): ', len(tracks.keys()))

  # gt_trajectory_map to compute residual or velocity
  # tracking_name: {scene_token -> {tracking_id: {t_idx -> det_data}}
  # [h, w, l, x, y, z, yaw] #x_dot, y_dot, z_dot, yaw_dot]
  gt_trajectory_map = {tracking_name: {scene_token: {} for scene_token in tracks.keys()} for tracking_name in NUSCENES_TRACKING_NAMES}

  # store every detection data to compute mean and variance
  gt_box_data = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

  for scene_token in tqdm(tracks.keys()):
    #print('scene_token: ', scene_token)
    #print('tracks[scene_token].keys(): ', tracks[scene_token].keys())
    for t_idx in range(len(tracks[scene_token].keys())):
      #print('t_idx: ', t_idx)
      t = sorted(tracks[scene_token].keys())[t_idx]
      for box_id in range(len(tracks[scene_token][t])):
        #print('box_id: ', box_id)
        box = tracks[scene_token][t][box_id]
        #print('box: ', box)
       
        if box.tracking_name not in NUSCENES_TRACKING_NAMES:
          continue
        # box:  {'sample_token': '6a808b09e5f34d33ba1de76cc8dab423', 'translation': [2131.657, 1108.874, 3.453], 'size': [3.078, 6.558, 2.95], 'rotation': [0.8520240186812739, 0.0, 0.0, 0.5235026949216329], 'velocity': array([-0.01800415,  0.0100023 ]), 'ego_dist': 54.20556415873658, 'num_pts': 4, 'tracking_id': 'cbaabbf2a83a4177b2145ab1317e296e', 'tracking_name': 'truck', 'tracking_score': -1.0}
        # [h, w, l, x, y, z, ry, 
        #  x_t - x_{t-1}, ...,  for [x,y,z,ry]
        #  (x_t - x_{t-1}) - (x_{t-1} - x_{t-2}), ..., for [x,y,z,ry]
        box_data = np.array([
          box.size[2], box.size[0], box.size[1], 
          box.translation[0], box.translation[1], box.translation[2],
          rotation_to_positive_z_angle(box.rotation),
          0, 0, 0, 0, 
          0, 0, 0, 0])


        if box.tracking_id not in gt_trajectory_map[box.tracking_name][scene_token]:
          gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id] = {t_idx: box_data}
        else: 
          gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data

        # if we can find the same object in the previous frame, get the velocity
        if box.tracking_id in gt_trajectory_map[box.tracking_name][scene_token] and t_idx-1 in gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id]:
          residual_vel = box_data[3:7] - gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][3:7]
          box_data[7:11] = residual_vel
          gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data
          # back fill
          if gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][7] == 0:
            gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][7:11] = residual_vel

          # if we can find the same object in the previous two frames, get the acceleration
          if box.tracking_id in gt_trajectory_map[box.tracking_name][scene_token] and t_idx-2 in gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id]:
            residual_a = residual_vel - (gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][3:7] - gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-2][3:7])
            box_data[11:15] = residual_a
            gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data
            # back fill
            if gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][11] == 0:
              gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][11:15] = residual_a
            if gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-2][11] == 0:
              gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-2][11:15] = residual_a

        #print(det_data)
        gt_box_data[box.tracking_name].append(box_data)
        

  gt_box_data = {tracking_name: np.stack(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}

  mean = {tracking_name: np.mean(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  std = {tracking_name: np.std(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  var = {tracking_name: np.var(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  saved_to_file = { 'mean' : mean, 
                   'std' : std, 
                   'var' : var}
  processNoise_file_name = "/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/process_noise_estimate_mean"
  with open(processNoise_file_name, 'wb') as f:
    pickle.dump(saved_to_file, f)

  return mean, std, var


def matching_and_get_diff_stats(pred_boxes, gt_boxes, tracks_gt, matching_dist):
  '''
  For each sample token, find matches of pred_boxes and gt_boxes, then get stats.
  tracks_gt has the temporal order info for each sample_token
  '''


  diff = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [h, w, l, x, y, z, a]
  diff_vel = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x_dot, y_dot, z_dot, a_dot]
  miss_detections_rates = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
  clutters_rates        = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} 
  # similar to main.py class AB3DMOT update()
  reorder = [3, 4, 5, 6, 2, 1, 0]
  reorder_back = [6, 5, 4, 0, 1, 2, 3]

  for scene_token in tqdm(tracks_gt.keys()):
    #print('scene_token: ', scene_token)
    #print('tracks[scene_token].keys(): ', tracks[scene_token].keys())
    # {tracking_name: t_idx: tracking_id: det(7) }
    match_diff_t_map = {tracking_name: {} for tracking_name in NUSCENES_TRACKING_NAMES}
    for t_idx in range(len(tracks_gt[scene_token].keys())):
      #print('t_idx: ', t_idx)
      t = sorted(tracks_gt[scene_token].keys())[t_idx]
      #print(len(tracks_gt[scene_token][t]))
      if len(tracks_gt[scene_token][t]) == 0:
        continue
      box = tracks_gt[scene_token][t][0]
      sample_token = box.sample_token

      for tracking_name in NUSCENES_TRACKING_NAMES:
    
        #print('t: ', t)
        gt_all = [box for box in gt_boxes.boxes[sample_token] if box.tracking_name == tracking_name]
        if len(gt_all) == 0:
          continue
        gts = np.stack([np.array([
          box.size[2], 
          box.size[0],
          box.size[1],
          box.translation[0], 
          box.translation[1], 
          box.translation[2],
          rotation_to_yaw_agnle(box.rotation)
          ]) for box in gt_all], axis=0)
        gts_ids = [box.tracking_id for box in gt_all]
        det_all = [box for box in pred_boxes.boxes[sample_token] if box.detection_name == tracking_name]
        if len(det_all) == 0:
          continue
        dets = np.stack([np.array([
          box.size[2], 
          box.size[0],
          box.size[1],
          box.translation[0], 
          box.translation[1], 
          box.translation[2],
          rotation_to_yaw_agnle(box.rotation)
          ]) for box in det_all], axis=0)
        

        dets = dets[:, reorder]
        gts = gts[:, reorder]

        if matching_dist == '3d_iou':
          dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
          gts_8corner = [convert_3dbox_to_8corner(gt_tmp) for gt_tmp in gts]
          iou_matrix = np.zeros((len(dets_8corner),len(gts_8corner)),dtype=np.float32)
          for d,det in enumerate(dets_8corner):
            for g,gt in enumerate(gts_8corner):
              iou_matrix[d,g] = iou3d(det,gt)[0]
          #print('iou_matrix: ', iou_matrix)
          distance_matrix = -iou_matrix
          threshold = -0.1
        elif matching_dist == '2d_center':
          distance_matrix = np.zeros((dets.shape[0], gts.shape[0]),dtype=np.float32)
          for d in range(dets.shape[0]):
            for g in range(gts.shape[0]):
              distance_matrix[d][g] = np.sqrt((dets[d][0] - gts[g][0])**2 + (dets[d][1] - gts[g][1])**2) 
          threshold = 2
        elif matching_dist == 'advanced':
          threshold = 2
          test_distance_matrix = np.zeros((gts.shape[0], dets.shape[0]),dtype=np.float32)
          distance_matrix = np.zeros((gts.shape[0], dets.shape[0] + gts.shape[0]),dtype=np.float32)
          for g in range(gts.shape[0]):
            for d in range(dets.shape[0]):
              distance_matrix[g][d] = np.sqrt((dets[d][0] - gts[g][0])**2 + (dets[d][1] - gts[g][1])**2)
              test_distance_matrix[g][d] = distance_matrix[g][d]
          for g in range(gts.shape[0]):
            for d in range(gts.shape[0]):
              if g == d: 
                distance_matrix[g][d + dets.shape[0]] = 2.0
              else: 
                distance_matrix[g][d + dets.shape[0]] = 1e5
        else:
              print(matching_dist)
              assert(False)

        matched_indices = linear_assignment(distance_matrix)
        
        """
        used for debuging purpose
        matched_indices_test = linear_assignment(test_distance_matrix) 
        print("N trakcs {} Num of detections M {}".format(gts.shape[0], dets.shape[0]))
        print("Match include missdetections")
        print(matched_indices.T)
        dist = []
        for pair_id in range(matched_indices.shape[0]):
          index_i = matched_indices[pair_id][0]
          index_j = matched_indices[pair_id][1]
          dist.append(distance_matrix[index_i][index_j])
        print(dist)
        print("Match without Missdetections")
        print(matched_indices_test.T)
        dist = []
        for pair_id in range(matched_indices_test.shape[0]):
          index_i = matched_indices_test[pair_id][0]
          index_j = matched_indices_test[pair_id][1]
          dist.append(test_distance_matrix[index_i][index_j])
        print(dist)
          
        print("\n")
        """
        
        #print('matched_indices: ', matched_indices)
        dets = dets[:, reorder_back]
        gts = gts[:, reorder_back]
        
        miss_detections = 0
        clutters        = 0
        num_detected_objects = 0 
        for pair_id in range(matched_indices.shape[0]):
          if matched_indices[pair_id][1] < dets.shape[0]:
            num_detected_objects += 1
            index_i = matched_indices[pair_id][0] # gt index
            index_j = matched_indices[pair_id][1] # detection index
            ddisc   =  distance_matrix[index_i][index_j]
            
            if (ddisc > 2.0): 
              print("THe distance is greater than miss detection ", ddisc)
            diff_value = dets[index_j] - gts[index_i]
            #print("original value ", diff_value[6])
            angle_diff = np.zeros(4)
            angle_diff_0 = angle_difference(dets[index_j][6],  gts[index_i][6])
            angle_diff_1 = angle_difference((dets[index_j][6] + np.pi)  ,  gts[index_i][6])
            angle_diff_2 = angle_difference((dets[index_j][6] + np.pi/2),  gts[index_i][6])
            angle_diff_3 = angle_difference((dets[index_j][6] - np.pi/2),  gts[index_i][6])
            
            diff_value[6] = min(angle_diff_0, angle_diff_1, angle_diff_2, angle_diff_3, key=abs)
            #print("Optimized Angle ", diff_value[6], " -> {}, {}".format(np.degrees(dets[index_j][6]), 
            #                                                      np.degrees(gts[index_i][6])))
            #print("\n")
            diff[tracking_name].append(diff_value)
        miss_detections = gts.shape[0] - num_detected_objects
        clutters        = dets.shape[0] - num_detected_objects
        miss_detections_rates[tracking_name].append(float(miss_detections) /  gts.shape[0])
        clutters_rates[tracking_name].append(float(clutters) / dets.shape[0])
        """
        if t_idx not in match_diff_t_map[tracking_name]: 
          match_diff_t_map[tracking_name][t_idx] = {gt_track_id: diff}
        else: 
          match_diff_t_map[tracking_name][t_idx][gt_track_id] = diff_
        diff_value = dets[matched_indices[pair_id][0]] - gts[matched_indices[pair_id][1]]
        diff[tracking_name].append(diff_value)
        gt_track_id = gts_ids[matched_indices[pair_id][1]]
        if t_idx not in match_diff_t_map[tracking_name]:
          match_diff_t_map[tracking_name][t_idx] = {gt_track_id: diff_value}
        else:
          match_diff_t_map[tracking_name][t_idx][gt_track_id] = diff_value
        # check if we have previous time_step's matching pair for current gt object
        #print('t: ', t)
        #print('len(match_diff_t_map): ', len(match_diff_t_map))
        if t_idx > 0 and t_idx-1 in match_diff_t_map[tracking_name] and gt_track_id in match_diff_t_map[tracking_name][t_idx-1]:
          diff_vel_value = diff_value - match_diff_t_map[tracking_name][t_idx-1][gt_track_id]
          diff_vel[tracking_name].append(diff_vel_value)
          """
  
  tracking_diff_file_name = "/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/tracking_diff"
  miss_detection_rate_file_name = "/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/miss_detection_rate"
  clutter_rate_file_name = "/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/clutter_rate"
  with open(tracking_diff_file_name, 'wb') as f:
    pickle.dump(diff, f)
  with open(miss_detection_rate_file_name, 'wb') as f:
    pickle.dump(miss_detections_rates, f)
  with open(clutter_rate_file_name, 'wb') as f:
    pickle.dump(clutters_rates, f)
    

  diff = {tracking_name: np.stack(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  mean = {tracking_name: np.mean(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  std = {tracking_name: np.std(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  var = {tracking_name: np.var(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  
  #diff_vel = {tracking_name: np.stack(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  #mean_vel = {tracking_name: np.mean(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  #std_vel = {tracking_name: np.std(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  #var_vel = {tracking_name: np.var(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}

  return mean, std, var

if __name__ == '__main__':
  
  # Settings.
  parser = argparse.ArgumentParser(
    description='Get nuScenes stats.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
    '--eval_set', 
    type=str,
    default='train',
    help='Which dataset split to evaluate on, train, val or test.')

  parser.add_argument(
    '--config_path', 
    type=str, 
    default='',
    help='Path to the configuration file.'
         'If no path given, the NIPS 2019 configuration will be used.')
  
  parser.add_argument(
    '--verbose', 
    type=int,
    default=1,
    help='Whether to print to stdout.')

  parser.add_argument(
    '--matching_dist', 
    type=str,
    default='2d_center',
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
  '''
  if 'train' in eval_set_:
    detection_file = '/juno/u/hkchiu/dataset/nuscenes_new/megvii_train.json'
    data_root = '/juno/u/hkchiu/dataset/nuscenes/trainval'
    version='v1.0-trainval'
  elif 'val' in eval_set_:
    detection_file = '/juno/u/hkchiu/dataset/nuscenes_new/megvii_val.json'
    data_root = '/juno/u/hkchiu/dataset/nuscenes/trainval'
    version='v1.0-trainval'
  elif 'test' in eval_set_:
    detection_file = '/juno/u/hkchiu/dataset/nuscenes_new/megvii_test.json'
    data_root = '/juno/u/hkchiu/dataset/nuscenes/test'
    version='v1.0-test'
  '''
  detection_path = "/media/liangxu/ArmyData/nuscenes/Tracking_result/"
  nuscense_path  = "/media/liangxu/ArmyData/nuscenes/"
  
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

  nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

  # load the predictions and the gt boxes
  pred_boxes, _ = load_prediction(detection_file, 10000, DetectionBox)
  gt_boxes      = load_gt(nusc, eval_set_, TrackingBox)

  assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

  # Add center distances.
  pred_boxes = add_center_dist(nusc, pred_boxes)
  gt_boxes   = add_center_dist(nusc, gt_boxes)
  print('len(pred_boxes.sample_tokens): ', len(pred_boxes.sample_tokens))
  print('len(gt_boxes.sample_tokens): ', len(gt_boxes.sample_tokens))


  tracks_gt = create_tracks(gt_boxes, nusc, eval_set_, gt=True)

  mean, std, var = get_mean(tracks_gt)
  print('GT: Global coordinate system')
  print('h, w, l, x, y, z, a, x_dot, y_dot, z_dot, a_dot, x_dot_dot, y_dot_dot, z_dot_dot, a_dot_dot')
  print('mean: ', mean)
  print('std: ', std)
  print('var: ', var)

  # for observation noise covariance
  mean, std, var, mean_vel, std_vel, var_vel = matching_and_get_diff_stats(pred_boxes, gt_boxes, tracks_gt, matching_dist)
  print('Diff: Global coordinate system')
  print('h, w, l, x, y, z, a')
  print('mean: ', mean)
  print('std: ', std)
  print('var: ', var)
  print('h_dot, w_dot, l_dot, x_dot, y_dot, z_dot, a_dot')
  print('mean_vel: ', mean_vel)
  print('std_vel: ', std_vel)
  print('var_vel: ', var_vel)

