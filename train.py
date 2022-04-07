import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from scripts.utils import * 
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau 
from scripts.metrics import *
from scripts.cat_loader import CatLoader as loader
import cv2
from torchvision import transforms
import torch.nn.functional as F

from model.pspnet import *

parser = argparse.ArgumentParser(description="CaSSeD data segmentation")
parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
parser.add_argument("--epochs", type=int, default=81, help="Number of training epochs")

parser.add_argument("--milestone", type=int, default=[30, 60, 90,120], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=0.01, help="initial learning rate")

parser.add_argument("--save_path", type=str, default="./logs/cat/", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=10,help='save intermediate model')
##CAT data path: "/home/ece_tech_5323/Desktop/cavs_forest/CAT/pack/mixed" 
parser.add_argument("--data_path",type=str, default="/home/ece_tech_5323/Desktop/cavs_forest/CAT/pack/CAT/mixed",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')

opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr*((1-(epoch/opt.epochs))**0.9)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def main():  
    print('Loading dataset ...\n')
    data_loader = loader(root = opt.data_path, is_transform=True, img_size=(1025, 649), augmentations=False, phase='train')
    loader_train = DataLoader(data_loader, num_workers=4, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    
    print("# of training samples: %d\n" % int(len(loader_train)))

    ##data_loader_val = loader(root = opt.data_path, is_transform=False, img_size=(1025, 649),augmentations=False, phase='val')
    ##loader_val = DataLoader(data_loader_val, num_workers=4, batch_size = 4, shuffle=False)

    CAT_class_weights = torch.tensor([0.2825,0.6302,0.9800,0.5398]).cuda() ##median frequency based class to avoid the effect of
    ##imbalance classes

    class_freq = CAT_class_weights
    num_classes = 4 
    
    # Build model
    ##options for layers = [18, 34, 50, 101]
    model = PSPNet(layers = 101, classes = num_classes, class_weights = class_freq)
##    print_network(model)
    
    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
    
    # Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
##    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.1)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    ##Running metrics
    running_metrics_val = runningScore(n_classes = num_classes)
    val_loss_meter = averageMeter()

    # start training
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        adjust_learning_rate(optimizer, epoch, opt.lr)
##        scheduler.step(epoch) #update learning rate
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (images,labels) in enumerate(loader_train, 0):
##            print("Original Shape:",images.shape)
            model.train() #training mode of model
            model.zero_grad()
            optimizer.zero_grad()

            images, labels = Variable(images), Variable(labels)

            if opt.use_gpu:
                images, labels = images.cuda(), labels.cuda()

            out_img, main_loss, aux_loss = model(images, labels)
            loss = main_loss + 0.4*aux_loss ##auxiliary loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                print("[epoch %d/%d], train_loss: %.4f"%(epoch+1, i, loss.item()))

            step += 1
        ## epoch training end

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":

    main()
