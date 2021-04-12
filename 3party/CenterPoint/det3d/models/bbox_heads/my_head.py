# Mimic Center head programming

import logging
from collections import defaultdict
import torch
from torch import double, nn
import copy 
import numpy as np

from det3d.core import box_torch_pos
from det3d.torchie.cnn import kaiming_init
from det3d.models.utils import Sequential
from ..registry import HEADS

from det3d.core.utils.circle_nms_jit import circle_nms

from det3d.ops.dcn import DeformConv

# TODO: Figure out those functions
from det3d.moduls.losses.centernet_loss import FastFocalLoss, RegLoss

class FeatureAdaption(nn.Module): 
    """
    Feature Adaption Module. 
    
    Feature Adaption Module is implemented based on DCN v1. 
    It uses anchor shape prediciton rather than feature map to 
    predict offsets of deformable conv layers. 
    
    Args: 
        @ in_channels  (int): Number of channels in the input feature map. 
        @ out_channles (int): Number of channels in the output feature map. 
        @ kernel_size  (int): Deformable onv kernel size. 
        @ deformable_grou (int): Deformable conv group size. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        supre(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        
        self.conv_offset = nn.Conv2d(
            in_channels, 
            deformable_groups * offset_channels, 
            1, 
            bias=True)
        # Deformable convolutional neural network 
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)

        self.relu = nn.ReLU(inplace=True)
        self.init_offset()
        
    def init_offset(self):
        self.conv_offset.weight.data.zero_()
    
    def forward(self, x): 
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x

class SepHead(nn.Module): 
    def __init__(
        self,
        in_channels, 
        heads, 
        head_conv=64, 
        final_kernel=1, 
        bn=False, 
        init_bias=-2.19, 
        **kwargs
    ): 
        super(SepHead, self).__init__(**kwargs)
        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]
            fc = Sequential()
            for i in range(num_conv-1): 
                fc.add(nn.Conv2d(in_channels,   
                                 head_conv,
                                 kernel_size = final_kernel,
                                 stride=1,
                                 padding=final_kernel // 2,
                                 bias=True))
                if bn: 
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())
            fc.add(nn.Conv2d(head_conv, 
                             classes,
                             kernel_size=final_kernel,
                             stride=1,
                             padding=final_kernel // 2,
                             bias=True))
            if 'hm' in head: 
                fc[-1].bias.data.fill_(init_bias)
            else: 
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d): 
                        kaiming_init(m)
            self.__setattr__(head, fc)
    
    def forward(self, x): 
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)
        return ret_dict

class DCNSepHead(nn.Module):
    def __init(
        self,
        in_channels, 
        num_cls, 
        heads, 
        head_conv = 64, 
        final_kernel = 1, 
        bn = False, 
        init_bias = -2.19
        **kwargs
    ): 
        super(DCNSepHead, self).__init__(**kwargs)
        
        # Feature adaption with dcn
        # use separate features for classification / regression
        # TODO: understand the feature adaption head
        self.feature_adapt_cls = FeatureAdaption(
            in_channels,
            in_channels, 
            kernel_size = 3, 
            deformable_groups=4)
        

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
        num_classes       = [len(t["class_names"]) for t in tasks]
        self.class_names  = [t["class_names"] for t in tasks]
        self.code_weights = code_weights
        self.weight       = weight # weight between hm loss and loc loss
        self.dataset      = dataset

        self.in_channels  = in_channels
        self.num_classes  = num_classes

        # TODO: get into the loss functions
        # ! 1. The FocalLoss
        # ! 2. The regress Loss
        self.crit  = FastFocalLoss()
        self.crit  = RegLoss()

        # check if valocity is in the dict
        self.box_n_dim    = 9 if 'val' in common_heads else 7
        self.use_direction_classifier = False

        if not logger: 
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(
                f"num_classes: {num_classes}"
            )

        # a shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 
                      share_conv_channel, 
                      kernel_size=3, 
                      padding = 1
                      bias = True), 
            nn.BatchNorm2d(share_conv_channel), 
            nn.ReLU(inplace=True)
        )

        # Open a list for adding all the heads.
        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        if dcn_head:
            print("Use Deformable Convolution in the CenterHead!")

        for num_cls in num_classes: 
            heads = copy.deepcopy(common_heads)
            if not dcn_head: 
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                self.tasks.append(
                    SepHead(share_conv_channel, 
                            heads, 
                            bn=True, 
                            init_bias=init_bias, 
                            final_kernel=3)
                )
            else:
                self.tasks.append(
                    DCNSepHead(share_conv_channel, 
                               num_cls, 
                               heads, 
                               bn=True, 
                               init_bias=init_bias, 
                               final_kernel=3)
                )
        logger.info("Finish CenterHead Initializations")
    
    def forward(self, x, *kwargs): 
        ret_dicts = []
        
        x = self.shared_conv(x)
        
        for task in self.tasks:
            ret_dicts.append(task(x))
        return ret_dicts
    
    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y
    
    def loss(self, example, preds_dicts, **kwargs): 
        """
        Create the loss functions
        hm -> heat map
        ? what is example in the input, it is the GT label.
        """
        rets = []
        # iterative all the predictions
        
        for task_id, preds_dicts in enumerate(preds_dicts): 
            # generate the heat map and create the focal loss
            preds_dicts['hm'] - self._sigmoid(preds_dicts['hm'])
            # use the focal loss for the generated heat map
            
            # TODO: understand the heat map
            hm_loss = self.crit(
                preds_dicts['hm'], 
                example['hm'][task_id], 
                example['ind'][task_id], 
                example['mask'][task_id],
                example['cat'][task_id]
            )
            
            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple regression heads
            if self.dataset in ['waymo', 'nuscenes']:
                if 'vel' in preds_dicts:
                    # have the velocity case
                    preds_dicts['anno_box'] = torch.cat(
                        (
                            preds_dicts['reg'], 
                            preds_dicts['height'], 
                            preds_dicts['dim'], 
                            preds_dicts['vel'], 
                            preds_dicts['rot']
                        ), dim = 1
                    )
                else: 
                    preds_dict['anno_box'] = torch.cat(
                        (
                            preds_dict['reg'], 
                            preds_dict['height'], 
                            preds_dict['dim'],
                            preds_dict['rot']
                        ), dim=1
                    ) 
            else:
                raise NotImplementedError("Not supported dataset")

            ret = {}
            # regression loss for diemnsion, offset, height, rotation
            box_loss = self.crit_reg(
                preds_dicts['anno_box'], 
                example['mask'][task_id], 
                example['ind'][task_id], 
                target_box # the ground truth value
            )
            # create the loss, and balance the weight with self.code_weights
            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()
            
            # construct the total loss
            # self.weight is a wieght distribution between the different 
            # regression value 
            # For example
            # code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
            loss = hm_loss + self.weight * loc_loss
            ret.update({'loss': loss, 
                        'hm_loss': hm_loss.detach().cpu(), 
                        'loc_loss':loc_loss, 
                        'loc_loss_elem': box_loss.detach().cpu(), 
                        'num_positive': example['mask'][task_id].float().sum()})
            # append the result to the big list
            rets.append(ret)
        """
        Convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)
        
        return rets_merged
    
    @torch.no_grad()
    def predict(self, 
                example,
                predict_dicts,
                test_cfg, 
                **kwargs): 
        """
        decode, nms, then return the detection result,
        Additioanly support double flip testing
        ? what is the example in this case, 
        ? what is the difference between this example with the on in the loss
        ? what is the predict_dicts? 
        """
        # get loss info
        rets = []
        matas = []
        
        double_flip  = test_cfg.get('double_flip', False)
        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) = torch.tensor(
            post_center_range, 
            dtype=preds_dicts[0]['hm'].dtype,
            device=preds_dicts[0]['hm'].device
        )
        
        for task_id, predict_dict in enumerate(predict_dicts): 
            # Convert N C H W to N H W C
            for key, val in predict_dicts.items(): 
                predict_dict[key] = val.permute(0, 2, 3, 1).contiguous()
            
            batch_size = preds_dict['hm'].shape[0]
            
            
            # dataset argumentation
            if double_flip: 
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                
                # data argumentation
                for k in preds_dict.keys(): 
                    """
                    Transform the prediction map back to their original coordinate
                    before flipping. The flipped predictions are gordered in a 
                    group of 4. 
                    1. The first one is the original pointcloud,
                    2. X flip pointcloud(y = -y)
                    3. Y flip pointcloud(x = -x)
                    4. x and y flip pointcloud (x = -x, y = -y)
                    Also please note that pytorch's flip function is defiend
                    on higher dimensional sapce, so dims = [2] means that it
                    is flipping along the axis with H length (which is normally
                    the y axis), however, in our traditional word, it is flipping 
                    the X axis. The below flip follows pytorch's definiton
                    yflip(y = -y) xflip(x = -x)
                    """
                    _, H, W, C = preds_dict[k].shape
                    preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, H, W, C)
                    preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[1]) 
                    preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[2])
                    preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[1, 2])
            
            # ? what is metadata in the example? 
            if "metadata" not in example or len(example['metadata']) == 0: 
                meta_list = [None] * batch_size
            else: 
                meta_list = example["metadata"]
                if double_flip:
                    # increase the number of samples
                    meta_list = meta_list[:4*int(batch_size):4]
            
            batch_hm = torch.sigmoid(preds_dict['hm'])
            batch_dim = torch.exp(predict_dict['dim'])
            
            # get the regression score
            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']
            
            if double_flip:
                batch_hm = batch_hm.mean(dim=1)
                batch_hei = batch_hei.mean(dim=1)
                batch_dim = batch_dim.mean(dim=1)

                # y = -y reg_y = 1-reg_y
                batch_reg[:, 1, ..., 1] = 1 - batch_reg[:, 1, ..., 1]
                batch_reg[:, 2, ..., 0] = 1 - batch_reg[:, 2, ..., 0]

                batch_reg[:, 3, ..., 0] = 1 - batch_reg[:, 3, ..., 0]
                batch_reg[:, 3, ..., 1] = 1 - batch_reg[:, 3, ..., 1]
                batch_reg = batch_reg.mean(dim=1)

                # first yflip 
                # y = -y theta = pi -theta
                # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta)
                # batch_rots[:, 1] the same
                batch_rotc[:, 1] *= -1

                # then xflip x = -x theta = 2pi - theta
                # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta)
                # batch_rots[:, 2] the same
                batch_rots[:, 2] *= -1

                # double flip 
                batch_rots[:, 3] *= -1
                batch_rotc[:, 3] *= -1

                batch_rotc = batch_rotc.mean(dim=1)
                batch_rots = batch_rots.mean(dim=1)
            
            # get the result.
            batch_rot = torch.atan2(batch_rots, batch_rotc)
            batch, H, W, num_cls = batch_hm.size()
            
            
            # -------------- Regression Result and Heat Map result ---- #
            # location of the detection 2 + 1
            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)
            
            # rotation(2)
            batch_rot = batch_rot.reshape(batch, H*W, 1)
            # size (3)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            
            # Heat map result
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)
            
            # Refine the detection location with the pillar distance
            # ys and xs are the lication of the pillars
            ys, xs = torch.meshgrid([torch.arange(0, H), 
                                     torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            
            # getting the offset with the orginal location
            """
            pc_range=[-54, -54],
            out_size_factor=get_downsample_factor(model),
            voxel_size=[0.075, 0.075],
            """
            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]
            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]
            
            # Deal with the valocity
            if 'vel' in preds_dict: 
                batch_vel = preds_dict['vel']
                
                # in the case with flip detection
                if double_flip:
                    # flip vy
                    batch_vel[:, 1, ..., 1] *= -1
                    # flip vx
                    batch_vel[:, 2, ..., 0] *= -1
                    batch_vel[:, 3] *= -1
                    batch_vel = batch_vel.mean(dim=1)
                
                batch_vel = batch_vel.reshape(batch, H*W, 2)
                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim = 2
                )
            else: 
                # the case without the velocity
                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_rot], dim = 2
                )
            metas.append(meta_list)
            
            if test_cfg.get('per_class_nms', False): 
                pass
            else: 
                # TODO: implement the post_processing function
                rets.append(
                    self.post_processing(
                        batch_box_preds, 
                        batch_hm, 
                        test_cfg, 
                        post_center_limit_range, 
                        task_id
                    )
                )
        
        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])
        
        # build the return list
        ret_list = []
        for i in range(num_samples): 
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]: 
                    ret[k] = torch.cat([ret[i][k]] for ret in rets)
                elif k in ["label_preds"]: 
                    flag = 0
                    for j, num_class in enumerate(self.num_classes): 
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k]] for ret in rets)
            
            ret['metadata'] = metas[0][i]
            ret_list.append(ret)
        return ret_list
        
        
    
    def post_processing(
        self, 
        batch_box_preds, 
        bathc_hm, 
        test_cfg, 
        post_center_limit_range,
        task_id
    ):
        """
        TODO: the post_processing functions
        """
        batch_size = len(batch_hm)
        prediciton_dicts = []
        for i in range(batch_size): 
            box_preds = batch_box_preds[i]
            hm_preds  = batch_hm[i]
            # ! this is the place where tracking can help
            scores, labels = torch.max(hm_pred, dim=-1)
            # score_threshold=0.1,
            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)
            
            # mask creation
            mask = distance_mask & score_mask
            
            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]
            
            if test_cfg.get('circular_nms', False): 
                centers = boxes_for_nms[:, [0,1]]
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = _circle_nms(
                    boxes, 
                    min_radius=test_cfg.min_radius[task_id], 
                    post_max_size=test_cfg.nms.nms_post_max_size
                )
            else: 
                selected = box_torch_ops.rotate_nms_pcdet(
                    boxes_for_nms.float(), 
                    scores.float(), 
                    thresh=test_cfg.nms.nms_iou_threshold,
                    pre_maxsize=test_cfg.nms.nms_pre_max_size,
                    post_max_size=test_cfg.nms.nms_post_max_size)
            
            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]
            
            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }
            
            prediction_dicts.append(prediction_dict)
        
        return prediciton_dicts
                

def _circle_nms(boxes, min_radius, post_max_size=83): 
    """
    NMS according to center distance
    """
    keep = np.array(
        circle_nms(boxes.cpu().numpy(), 
                   thresh=min_radius)
        )[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)
    return keep

        
        
    
    
    