from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement

from os import listdir
from os.path import isfile, join




# inplementing the ModelNetDataset

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train', 
                 data_augmentation=True):
        self.npoints = npoints
        self.root    = root
        self.split   = split
        self.data_augmentation = data_augmentation
        self.fns     = []
        self.cat     = {}

        # open the file name
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                  '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])
        # build the list
        for cat_name in self.cat.keys():
            path = os.path.join(root, cat_name + "/" + split)
            onlyfiles = [path + "/" + f for f in listdir(path) if isfile(join(path, f))]
            self.fns.extend(onlyfiles)
        print(str(split) , "Dataset length", len(self.fns))
        self.classes = list(self.cat.keys())
    
    def __getitem__(self, index):
        fn = self.fns[index]
        cls = fn.split('/')
        cls = self.cat[cls[6]]
        with open(fn, 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale
        
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
        
        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array(cls))
        return point_set, cls



    
    def __len__(self): 
        return len(self.fns)
    

"""
def read_off(filename):
    points = []
    faces = []
    with open(filename, 'r') as f:
        first = f.readline()
        if (len(first) > 4): 
            n, m, c = first[3:].split(' ')[:]
        else:
            n, m, c = f.readline().rstrip().split(' ')[:]
        n = int(n)
        m = int(m)
        for i in range(n):
            value = f.readline().rstrip().split(' ')
            points.append([float(x) for x in value])
        for i in range(m):
            value = f.readline().rstrip().split(' ')
            faces.append([int(x) for x in value])
    points = np.array(points)
    faces = np.array(faces)
    return points, faces
"""


