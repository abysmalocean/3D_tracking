from det3d.datasets import build_dataloader, build_dataset
import pickle 
from det3d.torchie import Config
import copy


new_prediction_file = "/home/liangxu/gt/3D_tracking/3party/CenterPoint/work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_trainSet_new/prediction.pkl"

config = 'configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_dcn_flip.py'
work_dir = 'work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_trainSet_new'

cfg = Config.fromfile(config)
predictions = pickle.load(open(new_prediction_file , 'rb'))

dataset = build_dataset(cfg.data.train_test)
result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), 
                                    output_dir=work_dir, 
                                    testset=False)