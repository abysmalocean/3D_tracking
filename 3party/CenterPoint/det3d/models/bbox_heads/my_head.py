# Mimic Center head programming

import logging
from collections import defaultdict
import torch
from torch import double, nn
import copy 


from det3d.core import box_torch_pos
from det3d.torchie.cnn import kaiming_init
from det3d.models.utils import Sequential
from ..registry import HEADS

from det3d.core.utils.circle_nms_jit import circle_nms

# TODO: Understant the functions
from det3d.ops.dcn import DeformConv

# TODO: Figure out those functions
from det3d.moduls.losses.centernet_loss import FastFocalLoss, RegLoss


@HEADS.register_module
class CenterHead(nn.Module):
    def __init__(
        self,
        in_channels        = [128,], 
        tasks              = [],
        dataset            = 'nuscenes', 
        weight             = 0.25, 
        code_weights       = [], 
        common_heads       = dict(), 
        logger             = None, 
        init_bias          = -2.19, 
        share_conv_channel = 64, 
        num_hm_conv        = 2, 
        dcn_head           = False
    ): 

    """
    A class used to represent the detections
    ...

    Attributes
    ----------
    in_channels        = [128,],
        : 
    tasks              = [],
        : used for representing the number of classes.
    dataset            = 0.25, 
    weight             = 0.25, 
    code_weights       = [], 
    common_heads       = dict(), 
    logger             = None, 
    init_bias          = -2.19, 
    share_conv_channel = 64, 
    num_hm_conv        = 2, 
    dcn_head           = False

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """
    super(CenterHead, self).__init__()
