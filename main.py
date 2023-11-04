from ast import arg
from email.policy import default
import torch
import torch.nn as nn
import argparse,math,numpy as np
from load_data import get_data
from config_args import get_args
from pdb import set_trace as stop
from optim_schedule import WarmupLinearSchedule
from run_epoch import run_epoch
import random
from tools.function import get_model_log_path, get_pedestrian_metrics
from vit.build import build_model
from vit.swin_config import get_config
from timm.scheduler import create_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
# import sys
# sys.exit(0)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

args = get_args(argparse.ArgumentParser())

print('Labels: {}'.format(args.num_labels))
train_loader,valid_loader,test_loader = get_data(args)

if(args.modelsize == 'B'):
    swin_model_size = 'base'  
else:
    swin_model_size = 'large'

print("model: ", swin_model_size, "mask ratio: ", args.maskratio)
## setup_seed(args.seed)

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file',\
        default='weights/swin_'+swin_model_size+'_patch4_window7_224_22k.yaml' )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight',
                        default='weights/swin_'+swin_model_size+'_patch4_window7_224_22k.pth')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps", default=2)
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel',\
        default= 9 )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

_, config = parse_option()
model = build_model(config, args.num_labels, args.maskratio)

def load_saved_model(saved_model_name,model):
    checkpoint = torch.load(saved_model_name)
    model.load_state_dict(checkpoint['state_dict'])
    return model

print(args.model_name)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.cuda()

if args.inference:
    model = load_saved_model(args.saved_model_name,model)
    if test_loader is not None:
        data_loader =test_loader
    else:
        data_loader =valid_loader
    
    all_preds,all_targs,all_masks,all_ids,test_loss,test_loss_unk = run_epoch(args,model,data_loader,None,1,'Testing')
    # test_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,test_loss,test_loss_unk,0,args.test_known_labels)

    exit(0)

if args.freeze_backbone:
    for p in model.module.backbone.parameters():
        p.requires_grad=False
    for p in model.module.backbone.base_network.layer4.parameters():
        p.requires_grad=True

if args.optim == 'adam':
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr)#, weight_decay=0.0004) 
else:
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)

if args.warmup_scheduler:
    step_scheduler = None
    scheduler_warmup = WarmupLinearSchedule(optimizer, 1, 300000)
else:
    scheduler_warmup = None

    num_steps = int(100 * len(train_loader))  # 100 * 2813
    warmup_steps = int(20 * len(train_loader))
    step_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, eta_min=5e-6, last_epoch=-1)


maximum = 0
maxaccu = 0
maxprec = 0
maxrecall = 0
maxf1 = 0

for epoch in range(1,args.epochs+1):
    print('======================== {} ========================'.format(epoch))
    for param_group in optimizer.param_groups:
        print('LR: {}'.format(param_group['lr']))

    train_loader.dataset.epoch = epoch
    ################### Train #################
    print("1. traning:")
    all_preds,all_targs,all_masks,all_ids,train_loss,train_loss_unk = run_epoch(args,model,train_loader,optimizer,epoch,'Training',train=True,warmup_scheduler=scheduler_warmup)

    ################### Valid #################
    print("2. Validating:")
    all_preds,all_targs,all_masks,all_ids,valid_loss,valid_loss_unk = run_epoch(args,model,valid_loader,None,epoch,'Validating')

    #########################
    valid_result = get_pedestrian_metrics(all_targs, all_preds)
    print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))
    cur_metric = valid_result.ma
    cur_accu = valid_result.instance_acc
    cur_prec = valid_result.instance_prec
    cur_recall = valid_result.instance_recall
    cur_f1 = valid_result.instance_f1

    ################### Test #################
    if test_loader is not None:
        all_preds,all_targs,all_masks,all_ids,test_loss,test_loss_unk = run_epoch(args,model,test_loader,None,epoch,'Testing')
        # test_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,test_loss,test_loss_unk,0,args.test_known_labels)
    #else:
    #    test_loss,test_loss_unk,test_metrics = valid_loss,valid_loss_unk,valid_metrics
    #loss_logger.log_losses('test.log',epoch,test_loss,test_metrics,test_loss_unk)

    if step_scheduler is not None:
        if args.scheduler_type == 'step':
            step_scheduler.step(epoch)
        elif args.scheduler_type == 'plateau':
            step_scheduler.step(valid_loss_unk)

    ############## Log and Save ##############
    # best_valid,best_test = metrics_logger.evaluate(train_metrics,valid_metrics,test_metrics,epoch,0,model,valid_loss,test_loss,all_preds,all_targs,all_ids,args)

