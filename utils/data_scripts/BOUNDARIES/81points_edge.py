'''
    #face_full_boudnary
    a1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,79,75,80,74,73,81,72,71,70,69,77,76,78]
    #face_left_eyebow
    a2 = [18,19,20,21,22]
    #face_right_eyebow
    a3 = [23,24,25,26,27]
    #nose bridge
    a4 = [28,29,30,31,32,33,34]
    #nose_round
    a5 = [32,33,34,35,36]
    #left_eye
    a6 = [37,38,39,40,41,42]
    #right_eye
    a7 = [43,44,45,46,47,48]
    # upper mouth
    a8 = [49,50,51,52,53,54,55]
    # upper mouth down
    a9 = [61,62,63,64,65]
    # lower mouth up
    a10 = [61,68,67,66,65]
    # lower mouth down
    a11 = [49,60,59,58,57,56,55]
    a = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11]
'''

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2


def read_img(full_filename):
    img = cv2.imread(full_filename)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return img_gray.shape

shape = read_img("/home/cydia/文档/毕业设计/直接采用GT_parsing/Dataset/demo/0.jpg")
print(shape)

img_face_full_boudnary = np.zeros(shape)
img_face_left_eyebow = np.zeros(shape)
img_face_right_eyebow = np.zeros(shape)
img_nose_bridge = np.zeros(shape)
img_nose_round = np.zeros(shape)
img_left_eye = np.zeros(shape)
img_right_eye = np.zeros(shape)
img_upper_mouth = np.zeros(shape)
img_upper_mouth_down = np.zeros(shape)
img_lower_mouth_up = np.zeros(shape)
img_lower_mouth_down = np.zeros(shape)


class FaceLandmarksDataset(Dataset):
    """MY FACE LANDMARK DATASET"""
    def __init__(self,txt_file,root_dir,transform=None):
        self.landmarks_frame = pd.read_csv(txt_file,sep=' ',header =None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self,idx):
        img_name =  os.path.join(self.root_dir,self.landmarks_frame.ix[idx,0])
        image = (img_name)
        landmarks = self.landmarks_frame.ix[idx,1:].as_matrix().astype('float')
        landmarks = landmarks.reshape(-1,2)
        sample = {'image':image, 'landmarks':landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

face_dataset = FaceLandmarksDataset(txt_file = dir + 'landmark.txt',root_dir= dir + "picture")
