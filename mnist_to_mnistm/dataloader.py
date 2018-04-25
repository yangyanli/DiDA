from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable, Function
from scipy.misc import imread, imresize, imsave
from torch.utils.data import DataLoader
from skimage import data_dir, io, transform, color
from os import listdir
from os.path import join
import numpy as np


def get_training_set(root_dir):
    train_list = join(root_dir, 'mnist_train.txt')
    return DatasetFromFolder(root_dir, train_list)

def preprocess_img(img):
    # [0,255] image to [0,1]
    #print(img.max())
    imin = 0
    imax = 255
    img = img * (1.0 / (imax - imin))- imin
    # [0,1] to [-1,1]
    img = img * 2 -1
    img = torch.FloatTensor(img).view(-1, 28, 28)
    
    return img

def load_img(filepath):
    img = imread(filepath)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    img = np.transpose(img, (2, 0, 1))
    #numpy.ndarray to FloatTensor
    img = torch.from_numpy(img).float()
    img = preprocess_img(img)
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, root_dir, train_list):
        super(DatasetFromFolder, self).__init__()
        self.img_root = join(root_dir, 'mnist_train')
        self.images = []
        self.wins = []
        #self.views = []
        file_to_read = open(train_list, 'r')
        #i = 1
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            terms = lines.split(':')
           
            self.images.append(terms[0])

            label = torch.FloatTensor(map(float, terms[1].split()))
            label = label[0]

            #labelu = torch.LongTensor(1).zero_() + torch.LongTensor(label)
           

            if label == 0.0:
		labelu = torch.LongTensor(1).zero_()
		
            if label == 1.0:
		labelu = torch.LongTensor(1).zero_() + 1
		
            if label == 2.0:
		labelu = torch.LongTensor(1).zero_() + 2

            if label == 3.0:
		labelu = torch.LongTensor(1).zero_() + 3
		
            if label == 4.0:
		labelu = torch.LongTensor(1).zero_() + 4
		
            if label == 5.0:
		labelu = torch.LongTensor(1).zero_() + 5

            if label == 6.0:
		labelu = torch.LongTensor(1).zero_() + 6
		
            if label == 7.0:
		labelu = torch.LongTensor(1).zero_() + 7
		
            if label == 8.0:
		labelu = torch.LongTensor(1).zero_() + 8

            if label == 9.0:
		labelu = torch.LongTensor(1).zero_() + 9
		
 
                            
            self.wins.append(labelu)
            
            #i = i+1

        file_to_read.close()

    def __getitem__(self, index):
        # Load Image
        image = load_img(join(self.img_root, self.images[index]))
        win = self.wins[index]
        #print(type(win))
        name = self.images[index]
       # view = self.views[index]
        
        return image, win, name

    def __len__(self):
        return len(self.images)

