# pointnet code taken from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
# MIT License

# Copyright (c) 2019 benny

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Code based on code from https://github.com/ShunChengWu/3DSSG/tree/cvpr21
# Copyright (c) 2021, ShunChengWu
# All rights reserved.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from typing import Optional
from .util import InitWeightsMixin

__all__ = ["PointNetEncoder"]


class STN3d(nn.Module):
    def __init__(self, point_size: int, norm: str = "batch") -> None:
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(point_size, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        if norm == "batch":
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)
        elif norm == "layer" or norm == "instance":
            # We're using a mix of Instance and Layer Norm here, since InstanceNorm requires a 2D input (channels, points)
            # to normalize each channel accross all points
            # After the third convolution, we have 1024 channels flat, so LayerNorm is used
            self.bn1 = nn.InstanceNorm1d(64, affine=True)
            self.bn2 = nn.InstanceNorm1d(128, affine=True)
            self.bn3 = nn.InstanceNorm1d(1024, affine=True)
            self.bn4 = nn.LayerNorm(512, elementwise_affine=True)
            self.bn5 = nn.LayerNorm(256, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.eye(3).view(1, 9).repeat(batchsize, 1))
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k: int = 64, norm: str = "batch") -> None:
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        if norm == "batch":
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)
            # We're using a mix of Instance and Layer Norm here, since InstanceNorm requires a 2D input (channels, points)
            # to normalize each channel accross all points
            # After the third convolution, we have 512 channels flat, so LayerNorm is used
        elif norm == "layer" or norm == "instance":
            self.bn1 = nn.InstanceNorm1d(64, affine=True)
            self.bn2 = nn.InstanceNorm1d(128, affine=True)
            self.bn3 = nn.InstanceNorm1d(1024, affine=True)
            self.bn4 = nn.LayerNorm(512, elementwise_affine=True)
            self.bn5 = nn.LayerNorm(256, elementwise_affine=True)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.eye(self.k).view(1, self.k * self.k).repeat(batchsize, 1))
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module, InitWeightsMixin):
    def __init__(
        self,
        global_feat: bool = True,
        input_transform: bool = True,
        feature_transform: bool = False,
        point_size: int = 3,
        output_size: int = 1024,
        norm="batch",
        init_weights=True,
        point_descriptor_str: Optional[str] = None,
    ):
        """PointNet feature extractor module.

        Args:
            global_feat: Compute a global feature vector for the input point cloud. Defaults to True.
            input_transform: Compute a transform for the input point cloud. Defaults to True.
            feature_transform: Compute a transform the feature vectors of the point cloud as well. Defaults to False.
            nchannels: Number of point channels. Defaults to 3.
            output_size: The size of the output vector. Defaults to 1024.
            batch_norm: Use batch norm. Defaults to True.
            init_weights: Initialize the weights of the Module. Defaults to True.
            point_descriptor_str: A string to determine the point cloud structure in the format 'pnxxx'.
                                  Each entry is treated as a vector of 3. Required if input transform is to be used.
                                  Defaults to None.
        """
        super().__init__()
        self.norm = norm

        self.relu = nn.ReLU()
        self.point_size = point_size
        self.output_size = output_size

        self.conv1 = torch.nn.Conv1d(point_size, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, output_size, 1)
        if self.norm == "batch":
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(output_size)
        elif self.norm == "instance" or self.norm == "layer":
            # We use instance Norm here rather than batch norm since we expect small batches in the incremental case
            self.bn1 = nn.InstanceNorm1d(64, affine=True)
            self.bn2 = nn.InstanceNorm1d(128, affine=True)
            self.bn3 = nn.InstanceNorm1d(output_size, affine=True)

        self.global_feat = global_feat
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if input_transform:
            assert point_descriptor_str is not None
            self.point_descriptor_str = point_descriptor_str
            self.stn = STN3d(point_size=point_size, norm=self.norm)
        if self.feature_transform:
            self.fstn = STNkd(k=64, norm=self.norm)

        if init_weights:
            if self.norm == "batch":
                self.init_weights("constant", 1, target_op="BatchNorm")
                self.init_weights("xavier_normal", 1)
            else:
                self.init_weights("constant", 1)
                self.init_weights("normal", 1)

    def forward(self, x: torch.Tensor, return_meta: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim > 2
        npoints = x.size()[2]

        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if self.point_descriptor_str is None and self.point_size == 3:
                x[:, :, :3] = torch.bmm(x[:, :, :3], trans)
            elif self.point_size > 3:
                for i, p in enumerate(self.point_descriptor_str):
                    offset = i * 3
                    offset_ = (i + 1) * 3
                    if p == "p" or p == "n":  # transform points and normals
                        x[:, :, offset:offset_] = torch.bmm(x[:, :, offset:offset_], trans)
            x = x.transpose(2, 1)
        else:
            trans = None

        x = self.conv1(x)
        if self.norm:
            self.bn1(x)
        x = self.relu(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = torch.zeros([1])  # for tracing

        pointfeat = x
        x = self.conv2(x)
        if self.norm:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        if self.norm:
            x = self.bn3(x)
        x = self.relu(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_size)

        if self.global_feat:
            if return_meta:
                return x, trans, trans_feat
            else:
                return x
        else:
            x = x.view(-1, self.output_size, 1).repeat(1, 1, npoints)
            if return_meta:
                return torch.cat([x, pointfeat], 1), trans, trans_feat
            return torch.cat([x, pointfeat], 1)


class PointNetDecoder(nn.Module, InitWeightsMixin):
    def __init__(
        self,
        k: int = 2,
        input_size: int = 1024,
        batch_norm: bool = True,
        drop_out: bool = True,
        init_weights: bool = True,
        multilabel: bool = False,
    ) -> None:
        """PointNet classification module. Performs a classification of the input vector into k classes.
        Args:
            k: The number of classes to classifiy. Defaults to 2.
            input_size: The size of the input vector. Defaults to 1024.
            batch_norm: Use batch norm. Defaults to True.
            drop_out: Use drop out with p=0.3. Defaults to True.
            init_weights: Initialize the weights of this module. Defaults to True.
            multi_class: Use Multi-class classification.
                         The returned result may have multiple classes.
                         Defaults to False.
        """
        super().__init__()

        self.input_size = input_size
        self.k = k
        self.use_batch_norm = batch_norm
        self.use_drop_out = drop_out
        self.multi_class = multilabel

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        if drop_out:
            self.dropout = nn.Dropout(p=0.3)

        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

        if init_weights:
            self.init_weights("constant", 1, target_op="BatchNorm")
            self.init_weights("xavier_normal", 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)

        if self.multi_class:
            x = torch.sigmoid(x)
        else:
            x = F.log_softmax(x, dim=1)
        return x


def feature_transform_reguliarzer(trans: torch.Tensor) -> float:
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss
