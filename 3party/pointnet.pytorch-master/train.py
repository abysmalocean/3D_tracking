from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm


# from model net
#from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pppnet.dataset import ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', 
                    type=int, 
                    default=32, 
                    help='input batch size')
parser.add_argument('--num_points', 
                    type=int, 
                    default=2500, 
                    help='input batch size')
parser.add_argument('--workers', 
                    type=int, 
                    help='number of data loading workers', 
                    default=4)
parser.add_argument('--nepoch', 
                    type=int, 
                    default=250, 
                    help='number of epochs to train for')
parser.add_argument('--outf', 
                    type=str, 
                    default='cls', 
                    help='output folder')
parser.add_argument('--model', 
                    type=str, 
                    default='', 
                    help='model path')
parser.add_argument('--dataset', 
                    type=str, 
                    default = '/media/liangxu/ArmyData/data/modelnet40_ply', 
                    help="dataset path")
parser.add_argument('--dataset_type', 
                    type=str, 
                    default='modelnet40', 
                    help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', 
                    action='store_true', 
                    help="use feature transform")

opt = parser.parse_args()
print(opt)
blue = lambda x: '\033[94m' + x + '\033[0m'

# generate random seed
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
# using the random seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)

elif opt.dataset_type == 'modelnet40':
    print("using the modelnet40")
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')

dataset[10]

