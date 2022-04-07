import sys,os
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time
import collections
from torch.autograd import Variable
from torch.utils import data
import cv2
from torch.utils.data import DataLoader
from scripts.cat_loader import CatLoader as loader
from scripts.utils import *
from numpy import argmax
from model.pspnet import *

from scripts.metrics import *

##Give the directory of where you saved the training and testing data
cat_path = "/home/ece_tech_5323/Desktop/cavs_forest/CAT/pack/CAT/mixed"

parser = argparse.ArgumentParser(description = 'Test_CaT_data')
parser.add_argument("--data_path", type=str, default = cat_path,
                    help="validation images directory")
parser.add_argument("--save_path", type=str, default ="./results/cat/",
                    help="segmented images saving directory")
parser.add_argument("--logdir", type=str, default = "./logs/cat",
                    help="trained model directory")
parser.add_argument("--batch_size", type = int, default = 1, help = "Test batch size")
parser.add_argument("--num_classes", type = int, default = 4, help = "Total number of classes in segmentation")

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main():
    data_loader = loader(root = opt.data_path, is_transform=False, img_size=(1025,649),augmentations=False, phase='val')
    test_loader = DataLoader(data_loader, num_workers=1, batch_size=opt.batch_size, shuffle=False)

    im_paths = data_loader.im_paths()
    n_classes = opt.num_classes

    running_metrics_val = runningScore(n_classes = n_classes)
    
    #model
    print('Loading trained model ...\n')
    model = PSPNet(layers = 101, classes = opt.num_classes, training = False)
##    print_network(model)
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir,'net_latest.pth')), strict = False)
    model.eval()

    for i, (tr_imgs,imgs_o,tr_lbls,lbls_o) in enumerate(test_loader):
        im_name_splits= im_paths[i].split('/')[-1].split('.')[0].split('_')
        orig_h, orig_w = imgs_o.shape[1:3]

        print('processing %d-th image' %i)
        t0 = time.time()
        with torch.no_grad():
            image = Variable(tr_imgs).cuda()
            label = Variable(lbls_o).cuda()
##            print("Original label shape:", label.size())
            outputs = model(image)
##            outputs = outputs.cpu.numpy.transpose()
            outputs = F.interpolate(outputs, size = [orig_h, orig_w], mode='bilinear', align_corners = False)
##            print("Predicted shape:", outputs.shape)
            pred = outputs.data.max(1)[1].cpu().numpy()
            output_ind = np.squeeze(pred, axis=0)
            
            ##Calculate metrics
            gt = label.data.cpu().numpy()
            running_metrics_val.update(gt, pred)

            ##Save Output images in RGB
            gt = np.squeeze(gt, axis=0)
            
            ##Generate RGB predictions
            decoded = data_loader.decode_segmap(output_ind.astype(np.float32)) 
            decoded = cv2.cvtColor(decoded.astype(np.float32), cv2.COLOR_RGB2BGR)
##            print("Decoded Shape:", decoded.shape)

            print("Total classes in the image:", np.unique(output_ind))
            
            if len(im_name_splits) == 2:
                cv2.imwrite(opt.save_path + 'pred' + '_' + im_name_splits[1] + '.png', decoded)
            else:
                cv2.imwrite(opt.save_path + 'pred' + '_' + im_name_splits[1] + '_' + im_name_splits[2] +'.png', decoded)
                
            print('write to' + opt.save_path + im_name_splits[0] + '_' + im_name_splits[1] + '.png')
    print('===========Final Metrices===========')
    score, class_iou = running_metrics_val.get_scores()
    for k,v in score.items():
        print(k,v)
    for i in range(n_classes):
        print(i, class_iou[i])
########################################################################################
                                                                                            
if __name__ == "__main__":
    main()  
    
    
