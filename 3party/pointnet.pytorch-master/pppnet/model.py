from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class STN3d(nn.Module): 
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        # the fc layers are used in the feature layers
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512 , 256)
        self.fc3   = nn.Linear(256 , 9)

        # BathNormalization Layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x): 
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # learn the transforamtion matrix
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        # return the transormation matrix to the original point cloud
        return x

class STNkd(nn.Module): 
    def __init__(self, k = 64): 
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k
    
    def forward(self, x): 
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # learn the k * k transformation
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # transform the feature from the k * k matrix
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        # return the k * k matrix as feature transformation
        return x



class PointNetfeat(nn.Module):
    def __init__(self, 
                 global_feat = True, 
                 feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x): 
        n_pts = x.size()[2]
        # use the first transformation to make the transformation
        trans = self.stn(x)
        x = x.transpose(2, 1)
        # matrix multiplicaiton
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        # apply the first MPL 3 -> 64
        x = F.relu(self.bn1(self.conv1(x)))
        trans_feat = None

        if self.feature_transform: 
            trans_feat = slef.fstn(x)
            x = x.transpose(2, 1)
            # matrix multiplicaiton
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        # make a copy for future use
        pointfeat = x
        # second MPL 64 -> 128
        x = F.relu(self.bn2(self.conv2(x)))
        # Third MPL 128 -> 1024
        x = self.bn3(self.conv3(x))
        # max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # return depends on the global features
        # output the trans matrix used for regulization
        if self.global_feat: 
            return x, trans, trans_feat
        else: 
            # this is used in the segmentation
            x = x.view(-1, 1024, 1).repeat(1,1,n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
        





class PointNetCls(nn.Module): 
    def __init__(self, 
                 k=2, 
                 feature_transform = False): 
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        # MPL Layers
        self.fc1     = nn.Linear(1024, 512)
        self.fc2     = nn.Linear(512, 256)
        self.fc3     = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1     = nn.BatchNorm1d(512) 
        self.bn2     = nn.BatchNorm1d(256)
        self.relu    = nn.ReLU()
        # feature net Layers
        self.feat = PointNetfeat(global_feat = True, 
                                 feature_transform = feature_transform)
        
    def forward(self, x): 
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


def feature_transform_regularizer(trans): 
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda: 
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))

    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

