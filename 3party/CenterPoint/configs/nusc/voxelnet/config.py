import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor
DOUBLE_FLIP = True 

task = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

tracking_task = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=1, class_names=["truck"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=1, class_names=["pedestrian"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))
tracking_class_names = list(itertools.chain(*[t["class_names"] for t in tracking_task]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings 

model = dict(
    type="VoxelNet", 
    pretrained=None, 
    reader=dict(
        type="VoxelFeatureExtractorV3",
        # type='SimpleVoxel',
        num_input_features=5,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8
    )
    
)
