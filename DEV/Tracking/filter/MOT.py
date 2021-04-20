from utils.utils import format_sample_result
from utils.association import associate_detections_to_trackers
from utils.utils import convert_3dbox_to_8corner
from filter.KalmanBoxTracker import KalmanBoxTracker
from filter.covariance import Covariance

import numpy as np

class Mot_tracker(object): 
    def __init__(self, 
                 covariance_id = 0, 
                 max_age       = 0,
                 min_hits      = 3, 
                 tracking_name = 'car', 
                 use_angular_velocity = False, 
                 tracking_nuscenes    = False):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.covariance_id = covariance_id
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity
        self.tracking_nuscenes = tracking_nuscenes
    
    def update(
        self, 
        dets_all, 
        match_distance,
        match_threshold, 
        match_algorithm, 
        seq_name):
        """
        trackers_results = mot_trackers[tracking_name].update(
                        dets_all[tracking_name], 
                        match_distance = 'iou',
                        match_threshold = 0.1, 
                        match_algorithm = 'h', 
                        seq_name = scene_token )
        detes_all : dict
                    1. dets
                    a numpy array of detections in the format. 
                    [[x,y,z, theta, l, w, h],[x,y,z, theta, l, w, h] ...]
                    2. info: 
                    a array of other info for each detection.
        seq_name : scene_token
        
        Requires: 
                  This method must be called once for each frame even with empty
                  detections.
        Note: the number of objects returned may differ from the number of
              detections of detections proveided.
        """ 
        dets, info = dets_all['dets'], dets_all['info']
        # FIXME: do not need to reorder the detections
        dets = dets[:, self.reorder]
        
        self.frame_count += 1
        print_debug = False
        
        # N x 7, get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers),7))
        to_del = []
        ret    = []
        
        for t, trk in enumerate(trks):
            # This should do the measurement prediction
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # remove sthe tracker, if there is nan in the prediction
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # getting the S
        
            
        
        