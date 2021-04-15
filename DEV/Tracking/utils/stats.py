from utils.dict import NUSCENES_TRACKING_NAMES
from utils.utils import rotation_to_positive_z_angle


from tqdm import tqdm
import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox

from sklearn.utils.linear_assignment_ import linear_assignment
import pickle


def add_center_dist(nusc: NuScenes,
                    eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
        

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (box.translation[0] - pose_record['translation'][0],
                               box.translation[1] - pose_record['translation'][1],
                               box.translation[2] - pose_record['translation'][2])
            ego_rotation    = pose_record['rotation']
            if isinstance(box, DetectionBox) or isinstance(box, TrackingBox):
                box.ego_translation = ego_translation
                box.ego_rotation    = ego_rotation
            else:
                raise NotImplementedError

    return eval_boxes

'''
A bounding box defining the position of an object seen in a sample. 
All location data is given with respect to the global coordinate system.

sample_annotation {
   "token":            <str> -- Unique record identifier.
   "sample_token":     <str> -- Foreign key. 
                                NOTE: this points to a sample NOT 
                                a sample_data since annotations are done on the 
                                sample level taking all relevant sample_data 
                                into account.
   "instance_token":   <str> -- Foreign key. 
                                Which object instance is this 
                                annotating. An instance can have multiple 
                                annotations over time.
   "attribute_tokens": <str> [n] -- Foreign keys. List of attributes for this 
                                annotation. Attributes can change over time, 
                                so they belong here, not in the instance table.
   "visibility_token": <str> -- Foreign key. Visibility may also change over time. 
                                If no visibility is annotated, 
                                the token is an empty string.
   "translation":      <float> [3] -- Bounding box location in meters 
                                      as center_x, center_y, center_z.
   "size":             <float> [3] -- Bounding box size in meters as 
                                      width, length, height.
   "rotation":         <float> [4] -- Bounding box orientation as 
                                      quaternion: w, x, y, z.
                                      
   "num_lidar_pts":    <int> -- Number of lidar points in this box. 
                                Points are counted during the lidar 
                                sweep identified with this sample.
                                
   "num_radar_pts":    <int> -- Number of radar points in this box. 
                                Points are counted during the radar sweep 
                                identified with this sample. 
                                This number is summed across all radar 
                                sensors without any invalid point filtering.
                                
   "next":             <str> -- Foreign key. Sample annotation from the same 
                                object instance that follows this in time. 
                                Empty if this is the last annotation for this 
                                object.
   "prev":             <str> -- Foreign key. 
                                Sample annotation from the same object instance 
                                that precedes this in time. 
                                Empty if this is the first annotation for this 
                                object.
}
'''
    
def get_mean(tracks):
    '''
     Input:
        tracks: {scene_token:  {t: [TrackingBox]}}
    '''
    '''
    print("Liang Xu in the getting mean")
    # TODO: finish this functions
    print("Length of the tracking scene {}".format(len(tracks)))
    print('len(tracks.keys()): ', len(tracks.keys()))
    #print("Example Data format")
    print(len(tracks['e631037169574c67944176bf079ee75c']))
    print(tracks['e631037169574c67944176bf079ee75c'][1542798831948182][0])
    #for k in tracks.keys():
    #    print(k)
    '''
    # get_trajectory_map to compute residual or velocity
    # tracking_name: {scene_token -> {tracking_id: {t_idx -> det_data}}
    # [h, w, l, x, y, z, yaw] #x_dot, y_dot, z_dot, yaw_dot]
    gt_trajectory_map = {tracking_name: 
        {scene_token: {} for scene_token in tracks.keys()} 
        for tracking_name in NUSCENES_TRACKING_NAMES}

    print("Length of gt_trajectory_map {}".format(len(gt_trajectory_map)))
    for scene_token, v in gt_trajectory_map.items():
        print("Scene token {}, and length {}".format(scene_token, len(v)))
    
    # store every detection data to compute mean and variance
    gt_box_data = {tracking_name: 
        [] for tracking_name in NUSCENES_TRACKING_NAMES}
    
    for scene_token in tqdm(tracks.keys()): 
        """
        tracks is : Dict[str, Dict[int, List[TrackingBox]]]
        """
        #print('scene_token: ', scene_token)
        #print('tracks[scene_token].keys(): ', len(tracks[scene_token].keys()))
        for t_idx in range(len(tracks[scene_token].keys())):
            t = sorted(tracks[scene_token].keys())[t_idx]
            for box_id in range(len(tracks[scene_token][t])):
                #print('box_id: ', box_id)
                box = tracks[scene_token][t][box_id]
                if box.tracking_name not in NUSCENES_TRACKING_NAMES:
                    continue
                # [h, w, l, x, y, z, ry, 
                #  x_t - x_{t-1}, ...,  for [x,y,z,ry]
                #  (x_t - x_{t-1}) - (x_{t-1} - x_{t-2}), ..., for [x,y,z,ry]
                box_data = np.array([
                    box.size[2],        # h
                    box.size[0],        # w
                    box.size[1],        # l
                    box.translation[0], # x
                    box.translation[1], # y
                    box.translation[2], # z
                    rotation_to_positive_z_angle(box.rotation), # ry
                    0,                  #  d_x
                    0,                  #  d_y
                    0,                  #  d_z
                    0,                  #  d_ry
                    0,                  #  dd_x
                    0,                  #  dd_y
                    0,                  #  dd_z
                    0                   #  dd_ry
                    ])
                
                if box.tracking_id not in gt_trajectory_map[box.tracking_name][scene_token]:
                    gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id] = {t_idx: box_data}
                else:
                    gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data

                # if we can find the same object in the previous frame, get the velocity
                if box.tracking_id in gt_trajectory_map[box.tracking_name][scene_token] and\
                   t_idx-1 in gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id]:
                        residual_vel = box_data[3:7] -\
                          gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][3:7]
                        box_data[7:11] = residual_vel
                        gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data
                        # back fill
                        if gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][7] == 0:
                            gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][7:11] = residual_vel
                        
                        # if we can find the same object in the previouse two frames, get the acceleration
                        if box.tracking_id in gt_trajectory_map[box.tracking_name][scene_token] and\
                            t_idx-2 in gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id]:
                            residual_a = residual_vel -\
                                (gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][3:7] -\
                                 gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-2][3:7])
                            box_data[11:15] = residual_a
                            gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data
                            # back fill
                            if gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][11] == 0:
                                gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1][11:15] = residual_a
                            if gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-2][11] == 0:
                                gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-2][11:15] = residual_a
                        gt_box_data[box.tracking_name].append(box_data)
                        
    gt_box_data = {tracking_name: np.stack(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    mean = {tracking_name: np.mean(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    std = {tracking_name: np.std(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    var = {tracking_name: np.var(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    return mean, std, var

def matching_and_get_diff_stats(
    pred_boxes, 
    gt_boxes, 
    tracks_gt, 
    matching_dist):
    """
    For each sample token, find matches of pred_boxes and gt_boxes, the get stats. 
    tracks_gt has the temporal order info for each sample_token
    """
    # [h, w, l, x, y, z, a]
    diff = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
    # [x_dot, y_dot, z_dot, a_dot]
    diff_vel = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} 
    
    # similar to main.py class
    reorder = [3, 4, 5, 6, 2, 1, 0]
    reorder_back = [6, 5, 4, 0, 1, 2, 3]
    
    #for scene_token in tracks_gt.keys():
    
    for scene_token in tqdm(tracks_gt.keys()): 
        """
        tracks is : Dict[str, Dict[int, List[TrackingBox]]]
        """
        match_diff_t_map = {tracking_name: {} for tracking_name in NUSCENES_TRACKING_NAMES}
        for t_idx in range(len(tracks_gt[scene_token].keys())):
            t = sorted(tracks_gt[scene_token].keys())[t_idx]
            if len(tracks_gt[scene_token][t]) == 0: 
                continue
            box = tracks_gt[scene_token][t][0]
            sample_token = box.sample_token

            for tracking_name in NUSCENES_TRACKING_NAMES:
                gt_all = [box for box in\
                          gt_boxes.boxes[sample_token] if\
                              box.tracking_name == tracking_name]
                if len(gt_all) == 0:
                    continue
                # 1. get all the ground truth boxes for current classes
                gts = np.stack(
                    [np.array([
                        box.size[2], 
                        box.size[0], 
                        box.size[1],
                        box.translation[0], 
                        box.translation[1], 
                        box.translation[2],
                        rotation_to_positive_z_angle(box.rotation)
                    ]) for box in gt_all], axis = 0)
                gts_ids = [box.tracking_id for box in gt_all]
                
                # 2. Get all the detections for current classes current time
                det_all = [
                    box for box in pred_boxes.boxes[sample_token] 
                    if box.detection_name == tracking_name
                    ]
                if len(det_all) == 0: 
                    continue
                dets = np.stack([np.array([
                  box.size[2], box.size[0], box.size[1],
                  box.translation[0], box.translation[1], box.translation[2],
                  rotation_to_positive_z_angle(box.rotation)
                  ]) for box in det_all], axis=0)
                
                # TODO: for new only using the "2d_center distance"
                distance_matrix = np.zeros((dets.shape[0], gts.shape[0]),dtype=np.float32)
                # FIXME: the assignment matrix should have n x (m + n) 
                # n is the ground truth and m is the detections
                for d in range(dets.shape[0]):
                    for g in range(gts.shape[0]):
                        distance_matrix[d][g] = np.sqrt((dets[d][0] - gts[g][0])**2 +\
                                                        (dets[d][1] - gts[g][1])**2)
                threshold = 2
                # FIXME: the new version should have different output
                matched_indices = linear_assignment(distance_matrix)
                # ! should count miss detections for each frames
                for pair_id in range(matched_indices.shape[0]): 
                    index_i = matched_indices[pair_id][0]
                    index_j = matched_indices[pair_id][1]
                    if distance_matrix[index_i][index_j] < threshold: 
                        diff_value = dets[index_i] - gts[index_j]
                        diff[tracking_name].append(diff_value)
                        gt_track_id = gts_ids[index_j]
                        if t_idx not in match_diff_t_map[tracking_name]:
                            match_diff_t_map[tracking_name][t_idx] = {gt_track_id: diff_value}
                        else:
                            match_diff_t_map[tracking_name][t_idx][gt_track_id] = diff_value
                        
                        # the the speed error. I do not think this is relevet
                        # ! not useful in my case
                        if t_idx > 0 and t_idx - 1 in match_diff_t_map[tracking_name] and\
                          gt_track_id in match_diff_t_map[tracking_name][t_idx-1]:
                            diff_vel_value = diff_value - match_diff_t_map[tracking_name][t_idx-1][gt_track_id]
                            diff_vel[tracking_name].append(diff_vel_value)
    # saving the tracking diff
    tracking_diff_file_name = "/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/tracking_diff"
    tracking_diff_vel_file_name = "/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/tracking_diff_vel"
    with open(tracking_diff_file_name, 'wb') as f:
        pickle.dump(diff, f)
        
    with open(tracking_diff_vel_file_name, 'wb') as f:
        pickle.dump(diff_vel, f)
    
    diff = {tracking_name: np.stack(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    mean = {tracking_name: np.mean(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    std = {tracking_name: np.std(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    var = {tracking_name: np.var(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  
    diff_vel = {tracking_name: np.stack(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    mean_vel = {tracking_name: np.mean(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    std_vel = {tracking_name: np.std(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    var_vel = {tracking_name: np.var(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}

"""
box data type
{'sample_token': 'd33ad33a66b94f77af92075762ece116', 
'translation': [175.697, 928.349, 0.855], 
'size': [2.092, 4.481, 1.943], 
'rotation': [0.6967771148546221, 0.0, 0.0, 0.7172877053281123], 
'velocity': array([ 0.08192941, -0.02397934]), 
'ego_translation': (7.231621651173384, 6.467850195518054, 0.855), 
'num_pts': 433, 
'tracking_id': '4c4c7d36c5494ab2a5cffb0c40420583', 
'tracking_name': 'car', 
'tracking_score': -1.0}
the box should have rotation, however, it not printing out some how
"""
    