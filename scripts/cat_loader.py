import os
import torch
import numpy as np
import cv2

from torch.utils import data

from .utils import recursive_glob
from .augmentations import *
import random

class CatLoader(data.Dataset):
    """Cassed data loader

    """
    mean_rgb = [107.549, 99.734, 94.627] # mean RGB value in the pool of training dataset

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(1025, 649), augmentations=None, version='CAT', phase='train'):
        """__init__

        :param root:
        :param split:
        :param is_transform: (not used)
        :param img_size: (not used)
        :param augmentations  (not used)
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 4
        self.img_size = img_size 
        self.mean = np.array(self.mean_rgb)
        self.files = {}

        if phase == 'train':
            self.images_base = os.path.join(self.root, 'Train', 'imgs')
            self.annotations_base = os.path.join(self.root, 'Train', 'annos', 'int_maps')
##          for unified labels comment above line and uncomment the following line           
##            self.annotations_base = '/home/ece_tech_5323/Desktop/cavs_forest/CAT/for_paper/mixed_unify/CaT/Train/int_maps/'
            
            self.im_files = recursive_glob(rootdir=self.images_base, suffix='.png')
        else:
            self.images_base = os.path.join(self.root, 'Test', 'imgs')
            self.annotations_base = os.path.join(self.root, 'Test', 'annos', 'int_maps')
            ## for unified labels comment above line and uncomment the following
##            self.annotations_base = '/home/ece_tech_5323/Desktop/cavs_forest/CAT/for_paper/mixed_unify/CaT/Test/int_maps/'
            
            self.split = 'test'

            self.im_files = recursive_glob(rootdir=self.images_base, suffix='.png')
            self.im_files = sorted(self.im_files)

        self.data_size = len(self.im_files)
        self.phase = phase

        print("Found %d %s images" % (self.data_size, self.split))

    def __len__(self):
        """__len__"""
        return self.data_size

    def im_paths(self):
        return self.im_files

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.im_files[index].rstrip()
        im_name_splits = img_path.split(os.sep)[-1].split('.')[0].split('_')

        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        
        if len(im_name_splits) == 2:
            lbl_path = os.path.join(self.annotations_base,
                                        'mask_'+ im_name_splits[1] + '.png')
        else:
##            print("Length of im_name_splits:", len(im_name_splits))
            lbl_path = os.path.join(self.annotations_base,
                                        'anno_'+ im_name_splits[1] + '_' + im_name_splits[2] + '.png')
            
##        print("label path:", lbl_path)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        lbl = np.array(lbl, dtype=np.uint8)

        tr_img, tr_lbl = self.transform(img, lbl)

        if self.phase == 'train':
            return np.array(tr_img), np.array(tr_lbl)
        else:
            return np.array(tr_img), np.array(img), np.array(tr_lbl), np.array(torch.from_numpy(lbl).long()) ##use tr_img and tr_lbl while training and another pair while testing


##        return np.array(tr_img), np.array(img), np.array(tr_lbl), np.array(lbl) ##use tr_img and tr_lbl while training and another pair while testing
    
    def transform(self, img, lbl=None):
        """transform

        :param img:
        :param lbl:
        """
        img = img.astype(np.float64)
        img -= self.mean

        img = cv2.resize(img, self.img_size)
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        if lbl is not None:
            lbl = cv2.resize(lbl, (int(self.img_size[0]), int(self.img_size[1])), interpolation=cv2.INTER_NEAREST)
            lbl = torch.from_numpy(lbl).long()
            return img, lbl
        else:
            return img

    def decode_segmap(self, temp, plot=False):
        Sedan = [27, 122, 235]
        Pickup = [56, 162, 4]
        Offroad = [245, 34, 45]
        Background = [0, 0, 0]

        label_colours = np.array(
            [
                Background,
                Sedan,
                Pickup,
                Offroad,
                
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r #/ 255.0
        rgb[:, :, 1] = g #/ 255.0
        rgb[:, :, 2] = b #/ 255.0
##        print ("In decode_segmap rgb size:", rgb.shape)
        return rgb

