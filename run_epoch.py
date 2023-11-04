import argparse,math,time,warnings,copy, numpy as np, os.path as path
from re import L 
import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
import random
from torch.autograd import Variable
from center_loss import CenterLoss2


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid               
        xs_neg = 1 - x_sigmoid                        

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)         

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))     
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg                   # L = yL+  +  (1-y)L-

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y) 
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)  
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()



def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def cal_cl_loss(representations,label, T):
        n = len(label)

        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
        mask_no_sim = torch.ones_like(mask) - mask

        mask_dui_jiao_0 = torch.ones(n, n) - torch.eye(n, n)
        similarity_matrix = torch.exp(similarity_matrix / T)
        similarity_matrix = similarity_matrix * mask_dui_jiao_0.cuda()

        sim = mask * similarity_matrix
        no_sim = similarity_matrix - sim

        no_sim_sum = torch.sum(no_sim, dim=1)

        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum = sim + no_sim_sum_expend
        loss = torch.div(sim, sim_sum)

        loss = mask_no_sim + loss + torch.eye(n, n).cuda()

        loss = -torch.log(loss)  
        loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss))) 
        return loss

def run_epoch(args,model,data,optimizer,epoch,desc,train=False,warmup_scheduler=None):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_targets = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_masks = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_image_ids = []

    max_samples = args.max_samples

    batch_idx = 0
    loss_total = 0
    unk_loss_total = 0

    device = torch.device("cuda")
    if (args.modelsize == 'B'):
        loss_5 = CenterLoss2(num_classes=args.num_labels, feat_dim=1024).to(device) 
    else:
        loss_5 = CenterLoss2(num_classes=args.num_labels, feat_dim=1536).to(device)
    optimizer_centloss = torch.optim.SGD(loss_5.parameters(), lr=0.5)

    for batch in tqdm(data,mininterval=0.5,desc=desc,leave=False,ncols=50):
        if batch_idx == max_samples:
            break

        labels = batch['labels'].float()
        images = batch['image'].float()
        mask = batch['mask'].float()
        all_image_ids += batch['imageIDs']

        
        mask_in = mask.clone()

        if train:
            pred, score2, features= model(images.cuda(),tid=1) # tid=1
            # pred, features= model(images.cuda())
        else:
            with torch.no_grad():
                pred, score2, features= model(images.cuda(),tid=0)
                # pred, features= model(images.cuda())


        # loss0 =  F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels.cuda(),reduction='none')
        
        loss_1 = AsymmetricLoss(gamma_neg=1, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)  # 1;0;0.05
        loss1 = loss_1(pred.view(labels.size(0),-1),labels.cuda())       
        
        loss5 = loss_5(features, labels.cuda())
        
        
        view = batch['viewid'].int()
        view = torch.tensor(view, dtype=torch.int64)
        loss8 = cal_cl_loss(score2.cuda(), view.cuda(), T= 0.1)
        
        loss_out = loss1  + 0.5* loss5.sum()  +  1.0* loss8

        # loss_out =  loss0.sum() 
        # loss_out = loss1


        if train:
            optimizer_centloss.zero_grad()
            loss_out.backward()
            ## for param in loss_5.parameters():
            ##     param.grad.data *= (1./1)
            optimizer_centloss.step()

            # Grad Accumulation
            if ((batch_idx+1)%args.grad_ac_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()

        ## Updates ##
        loss_total += loss_out.item()
        unk_loss_total += loss_out.item()
        start_idx,end_idx=(batch_idx*data.batch_size),((batch_idx+1)*data.batch_size)
        
        if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            pred = pred.view(labels.size(0),-1)
        
        all_predictions[start_idx:end_idx] = pred.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()
        all_masks[start_idx:end_idx] = mask.data.cpu()
        batch_idx +=1

    loss_total = loss_total/float(all_predictions.size(0))
    unk_loss_total = unk_loss_total/float(all_predictions.size(0))

    return all_predictions,all_targets,all_masks,all_image_ids,loss_total,unk_loss_total


