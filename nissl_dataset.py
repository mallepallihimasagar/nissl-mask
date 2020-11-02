import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
import numpy as np
from skimage import io
import cv2
from torch.utils.data import Dataset,DataLoader

class Nissl_mask_dataset(Dataset):
    def __init__(self,root_dir='nissl_unet_data',transform=None,multiclass=True):
        self.root_dir = root_dir
        self.transform = transform
        self.multiclass = multiclass

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        #item is basically index value form 0 to length of dataset minus 1
        file_id = os.listdir(self.root_dir)[item]
        file_path = os.path.join(self.root_dir,file_id)
        background = np.zeros((512,512),dtype=np.int8)
        try:
            cell1_mask = io.imread(file_path+f'/{file_id}_cell1.png')[:,:,3]/255
            cell1_mask = cv2.resize(cell1_mask,(512,512))
            cell1_loc = np.where(cell1_mask >= 0.3)
            background = (background==0) & (cell1_mask < 0.3)

        except:
            cell1_loc=[]
        try:
            cell2_mask = io.imread(file_path + f'/{file_id}_cell2.png')[:, :, 3] /255
            cell2_mask = cv2.resize(cell2_mask, (512, 512))
            cell2_loc = np.where(cell2_mask >= 0.3)
            background = (background == 0) & (cell2_mask < 0.3)
        except:
            cell2_loc=[]
        try:
            cell3_mask = io.imread(file_path + f'/{file_id}_cell3.png')[:, :, 3]/255
            cell3_mask = cv2.resize(cell3_mask, (512, 512))
            cell3_loc = np.where(cell3_mask >= 0.3)
            background = (background == 0) & (cell3_mask < 0.3)
        except:
            cell3_loc=[]
        original = io.imread(file_path + f'/{file_id}_original.png')
        original = cv2.resize(original, (512, 512))
        #combining masks
        # cell1_loc = np.where(cell1_mask>=0.3)
        # cell2_loc = np.where(cell2_mask>=0.3)
        # cell3_loc = np.where(cell3_mask>=0.3)



        if self.multiclass:
            final_mask = np.zeros((4, 512, 512),dtype=np.int8)
            img = np.zeros((512, 512), dtype=np.int8)
            final_mask[0] = background
            img[cell1_loc]=1
            final_mask[1]=img
            img = np.zeros((512,512),dtype=np.int8)
            img[cell2_loc]=1
            final_mask[2] = img
            img = np.zeros((512,512),dtype=np.int8)
            img[cell3_loc] = 1
            final_mask[3] = img

        else :
            final_mask = np.zeros((1, 512, 512),dtype=np.int8)
            img = np.zeros((512, 512), dtype=np.int8)
            img[cell1_loc] = 1
            final_mask[0] = img
            #img = np.zeros((512, 512), dtype=np.bool)
            img[cell2_loc] = 1
            final_mask[0] = img
            #img = np.zeros((512, 512), dtype=np.bool)
            img[cell3_loc] = 1
            final_mask[0] = img
        # original = torch.tensor(original)
        # final_mask = torch.tensor(final_mask)
        original = np.swapaxes(np.swapaxes(original, 1, -1), 0, 1)
        if self.transform :
            original = self.transform(original)
            final_mask = self.transform(final_mask)


        # original : from 0-255 numpy array
        #final_maskk : from 0-1 numpy array
        sample = (original,final_mask)
        return sample


