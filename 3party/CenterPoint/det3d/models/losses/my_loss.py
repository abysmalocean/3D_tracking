import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat


# regression loss and the Heatmap loss

class RegLoss(nn.Module): 
    """
    Regression loss for an output tensor
    """
    
    
class FastFocalLoss(nn.Module): 
    """
    Reimplemneted focal loss, exactly the same as the CornerNet version. 
    Faster and cost much less memory
    """
    