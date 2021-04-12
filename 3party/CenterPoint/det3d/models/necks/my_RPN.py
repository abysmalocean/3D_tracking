import time
import numpy as np
import math

import torch

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.torchie.trainer import load_checkpoint
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args

from .. import builder
from ..registry import NECKS
from ..utils import build_norm_layer

"""
Example:

neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),

"""

@NECKS.register_module
class RPN(nn.Module): 
    def __init__(
        self, 
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ): 
    super(RPN, self).__init__()
    self._layer_strides        = ds_layer_strides
    self._num_filters          = ds_num_filters
    self._layer_nums           = layer_nums
    self._upsample_strides     = us_layer_strides
    self._num_upsample_filters = us_num_filters
    self._num_input_features   = num_input_features

    if norm_cfg is None:
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
    self._norm_cfg = norm_cfg
    assert len(self._layer_strides)        == len(self._layer_nums)
    assert len(self._num_filters)          == len(self._layer_nums)
    assert len(self._num_upsample_filters) == len(self._upsample_strides)
    self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

    must_equal_list = []
    for i in range(len(self._upsample_strides)):
        # print(upsample_strides[i])
        must_equal_list.append(
            self._upsample_strides[i]
            / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
        )

    for val in must_equal_list:
        assert val == must_equal_list[0]