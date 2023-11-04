from itertools import count
import torch
import torch.nn as nn
from torch.autograd.function import Function
import numpy


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        num_label = label.size(1)
        ctx.save_for_backward(feature, label, centers, batch_size)  
        loss = 0
        bsize = label.size(0)
        for i in range(bsize):  
            for j in range(num_label):   
                t = torch.Tensor([j])
                centers_batch = centers.index_select(0, t.long().cuda())
                lo = label[i][j].item() *(feature[i] - centers_batch).pow(2).sum() / 2.0 / batch_size
                loss+= lo
        return loss/bsize

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        bsize = label.size(0)
        num_label = label.size(1)
        for i in range(bsize):
            for j in range(num_label):
                t = torch.Tensor([j])
                centers_batch = centers.index_select(0, t.long().cuda())
                diff = centers_batch - feature[i]
        # init every iteration  
        counts = centers.new_ones(centers.size(0))  
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones) 
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None

class CenterLoss2(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss2, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):
        # margin =0.5
        batch_size = feat.size(0)
        num_label = label.size(1) 
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        loss = 0
        bsize = label.size(0)
        for i in range(bsize):  
            for j in range(num_label):   
                t = torch.Tensor([j])
                centers_batch = self.centers.index_select(0, t.long().cuda())
                lo = label[i][j].item() *(feat[i] - centers_batch).pow(2).sum() / 2.0 / batch_size
                loss+= lo
        return loss/num_label

