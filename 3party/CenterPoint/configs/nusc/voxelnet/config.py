import itertools
import logging
import numpy as np

#from det3d.utils.config_tool import get_downsample_factor
DOUBLE_FLIP = True 

def get_downsample_factor(model_config):
    try:
        neck_cfg = model_config["neck"]
    except:
        model_config = model_config['first_stage_cfg']
        neck_cfg = model_config['neck']
    downsample_factor = np.prod(neck_cfg.get("ds_layer_strides", [1]))
    if len(neck_cfg.get("us_layer_strides", [])) > 0:
        downsample_factor /= neck_cfg.get("us_layer_strides", [])[-1]

    backbone_cfg = model_config['backbone']
    downsample_factor *= backbone_cfg["ds_factor"]
    downsample_factor = int(downsample_factor)
    assert downsample_factor > 0
    return downsample_factor

tasks = [
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

print(class_names)
print(tracking_class_names)

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
        type="SpMiddleResNetFHD", 
        num_input_features=5, 
        ds_factor=8
    ),
    # TODO: Need to under stand the RPN layers
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN")
    ), 
    bbox_head = dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)},
        share_conv_channel=64,
        dcn_head=True
    )
)
test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-54, -54],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.075, 0.075],
    double_flip=DOUBLE_FLIP
)
print(test_cfg)


assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)




