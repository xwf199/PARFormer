import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pdb import set_trace as stop
import os, random
from dataloaders.peta_dataset import petaDataset
from dataloaders.pa100k_dataset import pa100kDataset
from dataloaders.rap1_dataset import rap1Dataset
import warnings
warnings.filterwarnings("ignore")

from AttrDataset import AttrDataset, get_transform


def get_data(args):
    dataset = args.dataset
    data_root=args.dataroot
    batch_size=args.batch_size

    rescale=args.scale_size
    random_crop=args.crop_size
    attr_group_dict=args.attr_group_dict
    workers=args.workers
    n_groups=args.n_groups

    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_size = rescale
    crop_size = random_crop
    if args.test_batch_size == -1:
        args.test_batch_size = batch_size
    
    trainTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.RandomChoice([
                                        transforms.RandomCrop(640),
                                        transforms.RandomCrop(576),
                                        transforms.RandomCrop(512),
                                        transforms.RandomCrop(384),
                                        transforms.RandomCrop(320)
                                        ]),
                                        transforms.Resize((crop_size, crop_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normTransform])

    testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform])

    test_dataset = None
    test_loader = None
    drop_last = False
    
    if dataset == 'PETA':
        vg_root = os.path.join(data_root,'PETA')
        train_tsfm, valid_tsfm = get_transform(args)
        train_dataset = petaDataset(args=args, split=args.train_split, transform=train_tsfm,known_labels=args.train_known_labels)
        valid_dataset = petaDataset(args=args, split=args.valid_split, transform=valid_tsfm,known_labels=args.test_known_labels)
    
    elif dataset == 'PA100k':
        vg_root = os.path.join(data_root,'PA100K')
        train_tsfm, valid_tsfm = get_transform(args)
        train_dataset = pa100kDataset(args=args, split=args.train_split, transform=train_tsfm,known_labels=args.train_known_labels)
        valid_dataset = pa100kDataset(args=args, split=args.valid_split, transform=valid_tsfm,known_labels=args.test_known_labels)
    
    elif dataset == 'RAP1':
        vg_root = os.path.join(data_root,'RAP1')
        train_tsfm, valid_tsfm = get_transform(args)
        train_dataset = rap1Dataset(args=args, split=args.train_split, transform=train_tsfm,known_labels=args.train_known_labels)
        valid_dataset = rap1Dataset(args=args, split=args.valid_split, transform=valid_tsfm,known_labels=args.test_known_labels)


    else:
        print('no dataset avail')
        exit(0)

    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=workers,drop_last=drop_last)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers)

    return train_loader,valid_loader,test_loader
