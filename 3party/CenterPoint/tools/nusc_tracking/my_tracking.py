from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import copy
import json
import os
import numpy as np
from nuscenes import NuScenes
import json 
import time
from nuscenes.utils import splits
import pickle
from tqdm import tqdm

#from pub_tracker import PubTracker as Tracker
from my_tracker import PubTracker as Tracker



def parse_args(): 
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument("--checkpoint", 
                        help="the dir to checkpoint which the model read from")
    parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--root", type=str, default="data/nuScenes")
    parser.add_argument("--version", type=str, default='v1.0-trainval')
    parser.add_argument("--max_age", type=int, default=3)
    args = parser.parse_args()

    return args

def save_first_frame(): 
    args = parse_args()
    nusc_file_name = os.path.join(args.root , 'nusc.pkl')
    print(nusc_file_name)
    if not os.path.exists(nusc_file_name):
        nusc = NuScenes(version=args.version, dataroot=args.root, verbose=True)
        with open(nusc_file_name, 'wb') as f:
            pickle.dump(nusc, f)
    else:
        print("Find the nusc file, saved time")
        nusc = pickle.load(open(nusc_file_name , 'rb'))
    print("finishe loading")

    if args.version == 'v1.0-trainval':
        # Using the validation data
        scenes = splits.val
    elif args.version == 'v1.0-test':
        # Using the training data
        scenes = splits.test 
    else:
        raise ValueError("unknown")
    
    frames = []
    for sample in nusc.sample: 
        scene_name = nusc.get("scene", sample['scene_token'])['name']
        if scene_name not in scenes: 
            continue
        
        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {}
        frame['token'] = token
        frame['timestamp'] = timestamp 

        # start of a sequence
        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True 
        else:
            frame['first'] = False
        
        frames.append(frame)
    
    del nusc
    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)
    
def main(): 
    args = parse_args()
    print("Program start")
    # TODO: implement the tracker, not finished yet
    tracker = Tracker(max_age=args.max_age, hungarian=args.hungarian)

    # open the detection result file
    try: 
        with open(args.checkpoint, 'rb') as f: 
            predictions = json.load(f)['results']
    except:
        print("The detection result file not found ->", args.checkpoint)
    
    # open the frame data which is created from using save_first_frame()
    try:
        with open(os.path.join(args.work_dir, 'frames_meta.json'), 'rb') as f:
            frames=json.load(f)['frames']
    except:
        print("can not find the frames data make sure run the save_first_frame()")
    
    nusc_annos = {
        "results": {},
        "meta": None,
    }
    assert(len(frames) == len(predictions), "size of frames and prediction are not equal")
    size = len(frames)
    print("Prediction size ", len(predictions))
    print("Start the tracking")

    start_timestemp = time.time()
    counter_new_scene = 0 
    for i in tqdm(range(size)): 
        token = frames[i]['token']

        # reset tracking after one video sequence
        if frames[i]['first']: 
            # use this for sanity check to ensure token order is correct
            tracker.reset()
            last_time_stamp = frames[i]['timestamp']
            counter_new_scene += 1
        
        time_lag = (frames[i]['timestamp'] - last_time_stamp)
        last_time_stamp = last_time_stamp = frames[i]['timestamp']

        # get the detection result for current frame
        preds = predictions[token]

        # get the tracking result for current frame
        # TODO: implement this result
        outputs = tracker.step_centertrack(preds, time_lag)
        annos = []
        
        for item in outputs: 
            if item['active'] == 0: 
                continue
            nusc_anno = {
                "sample_token": token,
                "translation": item['translation'],
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "tracking_id": str(item['tracking_id']),
                "tracking_name": item['detection_name'],
                "tracking_score": item['detection_score'],
            }
            annos.append(nusc_anno)
        nusc_annos["results"].update({token: annos})

    end_timestep = time.time()
    second = (end_timestep-start_timestemp)
    
    speed=size / second
    print("The speed is {} FPS".format(speed))
    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }
    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    with open(os.path.join(args.work_dir, 'tracking_result.json'), "w") as f:
        json.dump(nusc_annos, f)
    return speed
    
    print("Total number of scene is ", counter_new_scene)


if __name__ == '__main__':
    # Step 1, need to run the save first frame then the program save data
    #save_first_frame()
    main()
    # test_time()
    # TODO: the evaluation code is for later
    #eval_tracking()
    print("finish the program")

