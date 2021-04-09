import numpy as np
import copy
import copy 
import importlib
import sys 

#from track_utils import greedy_assignment

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]

# 99.9 percentile of the l2 velocity error distribution (per clss / 0.5 second)
# This is an earlier statistcs and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
  'car':4,
  'truck':4,
  'bus':5.5,
  'trailer':3,
  'pedestrian':1,
  'motorcycle':13,
  'bicycle':3,  
}

# assignment algorithm
def greedy_assignment(dist): 
    """
    Used of assign each tracker with detections
        input is the distances matrics
    @ dist: is M x N matrix
    """
    matched_indices = []
    if dist.shape[1] == 0: 
        # No item is the matrices
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            # disable the possibility match with other objects.
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    # ! out put is a [matched x 2] matrix, the [detection, tracks]
    return np.array(matched_indices, np.int32).reshape(-1, 2)

class PubTracker(object): 
    
    def __init__(self, hungarian=False, max_age=0):
        self.hungarian = hungarian
        self.max_age   = max_age
        print("Use hungarian: {}".format(hungarian))
        self.NUSCENE_CLS_VELOCITY_ERROR = NUSCENE_CLS_VELOCITY_ERROR
        self.reset()

    def reset(self):
        self.id_count = 0
        self.tracks = []
    
    def step_centertrack(self, results, time_lag): 
        if len(results) == 0:
            # FIXME: detection is 0 not means nothing in the scene
            self.tracks = []
            return []

        # bgein extract the data from detection
        temp = []
        for det in results: 
            # finter out classes not evaluated for tracking
            if det['detection_name'] not in NUSCENES_TRACKING_NAMES:
              continue

            det['ct'] = np.array(det['translation'][:2])
            det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
            det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])
            temp.append(det)
        # store the detections
        # N is the number of detections
        # M is the number of the current tracks
        results = temp
        N = len(results)
        M = len(self.tracks)

        # N x 2
        if 'tracking' in results[0]: 
            dets = np.array(
               [det['ct'].astype(np.float32) + 
                det['tracking'].astype(np.float32) for det in results], 
                np.float32)
        else: 
            dets = np.array(
               [det['ct'].astype(np.float32) + 
                det['tracking'].astype(np.float32) for det in results], 
                np.float32)
        
        item_cat = np.array([item['label_preds'] for item in results], np.int32) # N
        track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32) # M

        max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] 
                             for box in results], np.float32)
        tracks = np.array(
            [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2
        
        if len(tracks) > 0: # NOT FIRST FRAME
            # TODO: do it later
            # FIXME: Can treat the objects as different addon distance. 
            # For example, car base distance is 100, and human is 10000 something
            # this can be implemented in the humgrian algorithm. 

            # First calculate the distance between the objects
            # No matter what is the objects
            dist = (((tracks.reshape(1, -1, 2) - \
                dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
            dist = np.sqrt(dist) # absolute distance in meter

            # Mathch the distance with the belonging class
            invalid = ((dist > max_diff.reshape(N, 1)) + \
                (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
            dist = dist + invalid * 1e18
            
            # assignment algorihtm
            # for now, not using the humgarian algorithm for now
            matched_indices = greedy_assignment(copy.deepcopy(dist))

        else: # First few frame
            assert M == 0
            matched_indices = np.array([], np.int32).reshape(-1, 2)
        
        # unmatched detections (Item is not in the first colum of matched index)
        unmatched_dets = [d for d in range(dets.shape[0]) \
            if not (d in matched_indices[:, 0])]
        # unmathced tracks
        unmatched_tracks = [d for d in range(tracks.shape[0]) \
            if not (d in matched_indices[:, 1])]

        # Not using the hungarian algorithm, for simplicity
        # The original algorithm using both the greedy and hungarian algorithm
        matches = matched_indices
        
        ret = []
        # process the already tracked items
        for m in matches:
            # find the track in results array
            track = results[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            track['age'] = 1
            track['active'] = self.tracks[m[1]]['active'] + 1
            ret.append(track)
            
        # process the unmathced detections, build a new track
        for i in unmatched_dets:
            track = results[i]
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] =  1
        
        # still store unmatched tracks if its age doesn't exceed max_age, 
        # however, we shouldn't output
        # ! this is vired, this could be a miss detections, or something. 
        # ? this could be improved
        for i in unmatched_tracks: 
            track = self.tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ct = track['ct']
                
                # assume the track can move it self
                if 'tracking' in track: 
                    offset = track['tracking'] * -1 # move forward 
                    track['ct'] = ct + offset
                ret.append(track)
        
        self.tracks = ret
        return ret
        
            
            
            
            
        
        



        



        







        
        




