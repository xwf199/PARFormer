from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from tools.function import get_pkl_rootpath

from dataloaders.data_utils import image_loader
from PIL import Image
from dataloaders.data_utils import get_unk_mask_indices,image_loader

class rap1Dataset(Dataset):
    def __init__(self, split, args, transform=None, target_transform=None,known_labels=0,attr_group_dict=None,testing=False,n_groups=1):
        print("rap1 dataset")
        data_path = get_pkl_rootpath(args.dataset)
        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.img_idx = dataset_info.partition[split]


        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]

        view = dataset_info.view
        vlabel = dataset_info.vlabel
        self.view = [view[i] for i in self.img_idx]
        self.vlabel = [vlabel[i] for i in self.img_idx]

        ##########################
        self.epoch = 1
        self.known_labels = known_labels
        self.testing=testing
        self.num_labels = len(self.attr_id)
        self.split=split
        self.img_root = img_id

    
    def __getitem__(self, index):
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        view = self.view[index]
        vlabel = self.vlabel[index]

        # print(imgname, vlabel, view)

        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)
        gt_label = torch.from_numpy(gt_label)
    
        sample = {}
        sample['image'] = img
        sample['labels'] = gt_label
        sample['imageIDs'] = imgname

        sample['viewid'] = view
        sample['vlabel'] = vlabel

        unk_mask_indices = get_unk_mask_indices(img,self.testing,self.num_labels,self.known_labels)
        mask = gt_label.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)
        sample['mask'] = mask

        return sample

    def __len__(self):
        return len(self.img_id)
