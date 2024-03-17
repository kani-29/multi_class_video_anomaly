import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

class data_loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test , Aclass <-
    """
    def __init__(self, is_train=1,path="/home/kani/Documents/project_phase2/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/UCF-Crime/"):
        super(data_loader, self).__init__()
        self.is_train = is_train
        self.path = path
        # self.Aclass=Aclass
        
                
        if self.is_train == 1:
            data_list = os.path.join(path, 'train.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()        
        else:
            data_list = os.path.join(path, 'test.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        class_details={0:"Normal_Videos_event",1:"Robbery",2:"Stealing"}
        if self.is_train == 1:
            name,n_class = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1])

            rgb_npy = np.load(os.path.join(self.path+'all_rgbs',class_details[n_class], self.data_list[idx][:-1].split("|")[0]))
            flow_npy = np.load(os.path.join(self.path+'all_flows', class_details[n_class],self.data_list[idx][:-1].split("|")[0]))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return name,concat_npy,n_class
        else:
            name,n_class = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1])

            rgb_npy = np.load(os.path.join(self.path+'all_rgbs',class_details[n_class], self.data_list[idx][:-1]).split("|")[0])
            flow_npy = np.load(os.path.join(self.path+'all_flows', class_details[n_class],self.data_list[idx][:-1]).split("|")[0])
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return name,concat_npy,n_class
