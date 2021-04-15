NUSCENES_TRACKING_NAMES = [
  'bicycle',
  'bus',
  'car',
  'motorcycle',
  'pedestrian',
  'trailer',
  'truck'
]

detection_path = "/media/liangxu/ArmyData/nuscenes/Tracking_result/"
nuscense_path  = "/media/liangxu/ArmyData/nuscenes/"

center_point_detection_train = '/home/liangxu/gt/3D_tracking/3party/CenterPoint/work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_trainSet_new/infos_train_10sweeps_withvelo_filter_True.json'
center_point_detection_val   = '/home/liangxu/gt/3D_tracking/3party/CenterPoint/work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_validsation/infos_val_10sweeps_withvelo_filter_True.json'
center_point_detection_test  = ''

nusc_train_val_pickle_file = '/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/nusc.pkl'
nusc_test_pickle_file  = '/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/nusc_test.pkl'


tracking_diff_file_name = "/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/tracking_diff"
miss_detection_rate_file_name = "/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/miss_detection_rate"
clutter_rate_file_name = "/media/liangxu/ArmyData/nuscenes/Tracking_result/tracking_tmp/clutter_rate"
