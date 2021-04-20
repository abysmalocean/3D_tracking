
from utils.utils import format_sample_result
from utils.association import associate_detections_to_trackers
from utils.utils import convert_3dbox_to_8corner
from filter.KalmanBoxTracker import KalmanBoxTracker
from filter.covariance import Covariance

import numpy as np


class AB3DMOT(object): 
    def __init__(self, 
                 covariance_id = 0, 
                 max_age       = 0,
                 min_hits      = 3, 
                 tracking_name = 'car', 
                 use_angular_velocity = False, 
                 tracking_nuscenes    = False,
                 current_time_stemp   = 0.0):
        """
        Observation: 
            before reorder: [h, w, l, x, y ,z, rot_y]
            after reorder : [x, y, z, rot_y, l, w, h]
        state: 
            [x, y ,z, rot_y, l, w, h, x_dot, y_dot, z_dot]
        """
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
        self.current_time_stemp = current_time_stemp 
    
    def update(self, 
               dets_all, 
               match_distance,
               match_threshold,
               match_algorithm, 
               seq_name): 
        """
        Params: 
            dets_all: dict
                dets -> a numpy array of detections in the format. 
                    [[x,y,z, theta, l, w , h],[x,y,z, theta, l, w , h]... ]
                info: a array of other info for each det
            Requires: this method must be called once for each frame even with
                      empty detections
            Returns the similar array, where the last columb is the object ID

            NOTE: The number of objects returned may differ from the number of 
                  detections of detections provided.
        """
        dets, info = dets_all['dets'], dets_all['info']
        timstemp = dets_all['timestemp']
        dets = dets[:, self.reorder]

        self.frame_count += 1
        print_debug = False

        # N x 7 , get predicted locations from existign trackers. 
        trks = np.zeros((len(self.trackers),7))
        to_del = []
        ret    = []
        print("length of the detection", len(dets))
        
        for t, trk in enumerate(trks):
            print("Predict the update") 
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # remove the tracker, somehow
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # print the debug information
        if print_debug:
          for trk_tmp in self.trackers:
            print('trk_tmp.id: ', trk_tmp.id)
        
        # the following translation is for the detection.
        dets_8corner = [convert_3dbox_to_8corner(det_tmp, 
                                                 match_distance == 'iou' and self.tracking_nuscenes) 
                                                for det_tmp in dets]
        if len(dets_8corner) > 0 : dets_8corner = np.stack(dets_8corner, axis=0)
        else: dets_8corner = []

        # predictions
        trks_8corner = [convert_3dbox_to_8corner(trk_tmp, 
                                                 match_distance == 'iou' and self.tracking_nuscenes) 
                                                 for trk_tmp in trks]
        
        # getting the S
        trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in self.trackers]
        
        # Association algorithm
        if match_distance == 'iou':
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, 
                                                                                     trks_8corner, 
                                                                                     iou_threshold=match_threshold, 
                                                                                     print_debug=print_debug, 
                                                                                     match_algorithm=match_algorithm)
        else:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, 
                                                                                       trks_8corner, 
                                                                                       use_mahalanobis=True, 
                                                                                       dets=dets, 
                                                                                       trks=trks, 
                                                                                       trks_S=trks_S, 
                                                                                       mahalanobis_threshold=match_threshold, 
                                                                                       print_debug=print_debug, 
                                                                                       match_algorithm=match_algorithm)
        
        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers): 
            if t not in unmatched_trak:
                print("Update the track")
                # FIXME: I am not sure this is corrected, they are using the 
                # predicted value to update. should we just leave this blank? 
                d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
                trk.update(dets[d,:][0], info[d, :][0])
                detection_score = info[d, :][0][-1]
                trk.track_score = detection_score
        
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            detection_score = info[i][-1]
            track_score = detection_score
            # TODO: implement the Kalman Filter class
            trk = KalmanBoxTracker(dets[i,:], 
                                   info[i, :], 
                                   timestemp=timstemp, 
                                   track_score=track_score,
                                   tracking_name=self.tracking_name)
            # append the new tracker to the existing tracker
            self.trackers.append(trk)
        print("Current tracker for ", self.tracking_name, " is ", len(self.trackers))
        
        # summarying the result and output the result
        i = len(self.trackers)
        for trk in reversed(self.trackers): 
            d = trk.get_state() # get the bbox location
            d = d[self.reorder_back]
            if((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
                ret.append(np.concatenate((d, [trk.id+1], trk.info[:-1], [trk.track_score])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1

            #remove dead tracklet
            if(trk.time_since_update >= self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            # x, y, z, theta, l, w, h, ID, other info, confidence
            return np.concatenate(ret)
        return np.empty((0, 15 + 7))




        




        
        

