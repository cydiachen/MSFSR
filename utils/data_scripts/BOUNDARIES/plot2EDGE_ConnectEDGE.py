# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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
    a11 = [60,59,58,57,56,55]
    a = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11]
'''
# 源地址
dir = "/home/cydia/Desktop/FSR/DATASET/CASIA_Webface/CASIA_Webface_washed/"
# 目标地址
dest_dir_final = "/home/cydia/Desktop/FSR/DATASET/CASIA_Webface/BOUNDARY/"



point_dst = [0,0]
point_src = [0,0]

face_full_boudnary_start = [0,0]
face_full_boudnary_end = [0,0]

face_left_eyebow_start = [0,0]
face_left_eyebow_end = [0,0]

face_right_eyebow_start = [0,0]
face_right_eyebow_end = [0,0]

nose_bridge_start = [0,0]
nose_bridge_end = [0,0]

nose_round_start = [0,0]
nose_round_end = [0,0]

left_eye_start = [0,0]
left_eye_end = [0,0]

right_eye_start = [0,0]
right_eye_end = [0,0]

upper_mouth_start = [0,0]
upper_mouth_end = [0,0]

upper_mouth_down_start = [0,0]
upper_mouth_down_end = [0,0]

lower_mouth_up_start = [0,0]
lower_mouth_up_end = [0,0]

lower_mouth_down_start = [0,0]
lower_mouth_down_end = [0,0]


class FaceLandmarksDataset(Dataset):
    """MY FACE LANDMARK DATASET"""
    def __init__(self,txt_file,root_dir,transform=None):
        self.landmarks_frame = pd.read_csv(txt_file,sep=',',header =None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,self.landmarks_frame.ix[idx,0])
        image = (img_name)
        landmarks = self.landmarks_frame.ix[idx,1:].as_matrix().astype('float')
        landmarks = landmarks.reshape(-1,2)
        sample = {'image':image, 'landmarks':landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

face_dataset = FaceLandmarksDataset(txt_file ='/home/cydia/Desktop/FSR/DATASET/CASIA_Webface/landmarks.CSV',root_dir= dir)

num_len = len(face_dataset)

for i in range(num_len):
    sample = face_dataset[i]
    print(i,sample['image'])
    (filepath,tempfilename) = os.path.split(sample['image'])
    (final_filename,extension) = os.path.splitext(tempfilename)

    count = 0
    img = cv2.imread(sample['image'])

    img_face_full_boudnary = np.zeros(img.shape)

    img_face_left_eyebow = np.zeros(img.shape)
    img_face_right_eyebow = np.zeros(img.shape)

    img_nose_bridge = np.zeros(img.shape)
    img_nose_round = np.zeros(img.shape)

    img_left_eye = np.zeros(img.shape)
    img_right_eye = np.zeros(img.shape)

    img_upper_mouth = np.zeros(img.shape)
    img_upper_mouth_down = np.zeros(img.shape)

    img_lower_mouth_up = np.zeros(img.shape)
    img_lower_mouth_down = np.zeros(img.shape)

    point_list = sample['landmarks']

    # 把点练成线的过程
    for point in point_list:

        count = count + 1

        # 这里的话拟采用四舍五入的方法，使得我们的生成的点更接近实际
        point[0] = round(point[0])
        point[1] = round(point[1])
        point = tuple(point)

        #cv2.circle(img,(int(point[0]),int(point[1])),1,(0,0,255),2)
        if count % 2 == 0:
            point_dst[0] = point[0]
            point_dst[1] = point[1]
        else:
            point_src[0] = point[0]
            point_src[1] = point[1]

        if count == 1:
            face_full_boudnary_start[0] = point[0]
            face_full_boudnary_start[1] = point[1]
            continue

        # extract face contour (from point 1 to 33)
        if count == 17:
            face_full_boudnary_end[0] = point[0]
            face_full_boudnary_end[1] = point[1]
            print("face contour end",face_full_boudnary_end)
            continue

        if count >= 1 and count < 17:
            cv2.line(img_face_full_boudnary, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)

        # extract left_eyebrow (from 18 to 22)
        if count == 18: #
            face_left_eyebow_start[0] = point[0]
            face_left_eyebow_start[1] = point[1]
            print("this si the left eyebow start",face_left_eyebow_start)
            #cv2.circle(img_blank, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            continue

        if count == 22:
            face_left_eyebow_end[0] = point[0]
            face_left_eyebow_end[1] = point[1]
            print("this is the left eyebow end",face_left_eyebow_end)
            #cv2.circle(img_blank, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            cv2.line(img_face_left_eyebow, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)
            cv2.line(img_face_left_eyebow, (int(face_left_eyebow_start[0]), int(face_left_eyebow_start[1])), (int(face_left_eyebow_end[0]), int(face_left_eyebow_end[1])),
                     (255, 255, 255), 2)
            continue

        if count > 18 and count < 22:
            cv2.line(img_face_left_eyebow, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)

        # extract right_eyebrow (from 23 to 27)
        if count == 23:
            face_right_eyebow_start[0] = point[0]
            face_right_eyebow_start[1] = point[1]
            print("this is the right eyebow start",face_right_eyebow_start)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            continue

        if count >= 23 and count <= 27:
            cv2.line(img_face_right_eyebow, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)

        if count == 27:
            face_right_eyebow_end[0] = point[0]
            face_right_eyebow_end[1] = point[1]
            print("this is the right eyebow end",face_right_eyebow_end)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            cv2.line(img_face_right_eyebow, (int(face_right_eyebow_start[0]), int(face_right_eyebow_start[1])), (int(face_right_eyebow_end[0]), int(face_right_eyebow_end[1])),
                     (255, 255, 255), 2)
            continue

        # extract nose structure (from 28 to 34, Keep in mind of point 55-56)
        if count == 28:
            nose_bridge_start[0] = point[0]
            nose_bridge_start[1] = point[1]
            print("this is nose start",nose_bridge_start)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (255, 0, 0), 2)
            continue

        if count >= 28 and count <= 34:
            cv2.line(img_nose_bridge, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)

        if count == 34:
            nose_bridge_end[0] = point[0]
            nose_bridge_end[1] = point[1]
            print("this is the nose end",nose_bridge_end)
            #cv2.circle(img_blank, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            cv2.line(img_nose_bridge, (int(nose_bridge_start[0]), int(nose_bridge_start[1])), (int(nose_bridge_end[0]), int(nose_bridge_end[1])),
                     (255, 255, 255), 2)
            continue

        # extract nose round (from 32 to 36)
        if count == 28:
            nose_round_start[0] = point[0]
            nose_round_start[1] = point[1]
            print("this is nose start",nose_round_start)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (255, 0, 0), 2)
            continue

        if count >= 28 and count <= 34:
            cv2.line(img_nose_round, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)

        if count == 34:
            nose_round_end[0] = point[0]
            nose_round_end[1] = point[1]
            print("this is the nose end",nose_round_end)
            #cv2.circle(img_blank, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            cv2.line(img_nose_round, (int(nose_round_start[0]), int(nose_round_start[1])), (int(nose_round_end[0]), int(nose_round_end[1])),
                     (255, 255, 255), 2)
            continue


        # extract left eye (from 37 to 42)
        if count == 37:
            left_eye_start[0] = point[0]
            left_eye_start[1] = point[1]
            print("this is nose start",left_eye_start)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (255, 0, 0), 2)
            continue

        if count >= 37 and count <= 42:
            cv2.line(img_left_eye, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)

        if count == 42:
            left_eye_end[0] = point[0]
            left_eye_end[1] = point[1]
            print("this is the nose end",left_eye_end)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            cv2.line(img_left_eye, (int(left_eye_start[0]), int(left_eye_start[1])), (int(left_eye_end[0]), int(left_eye_end[1])),
                     (255, 255, 255), 2)
            continue

        # extract right eye (from 43 to 48)

        if count == 43:
            right_eye_start[0] = point[0]
            right_eye_start[1] = point[1]
            print("this is right eye start",right_eye_start)
            continue

        if count >= 43 and count <= 48:
            cv2.line(img_right_eye, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)

        if count == 48:
            right_eye_end[0] = point[0]
            right_eye_end[1] = point[1]
            print("this is right eye start",right_eye_start)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (255, 0, 0), 2)
            cv2.line(img_right_eye, (int(right_eye_start[0]), int(right_eye_start[1])), (int(right_eye_end[0]), int(right_eye_end[1])),
                     (255, 255, 255), 2)
            continue

        # extract upper mouth up(from 49 to 55)
        if count == 49:
            upper_mouth_start[0] = point[0]
            upper_mouth_start[1] = point[1]
            print("this is the outer lips_start",upper_mouth_start)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            continue

        if count >= 49 and count <= 55:
            cv2.line(img_upper_mouth, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)

        if count == 55:
            upper_mouth_end[0] = point[0]
            upper_mouth_end[1] = point[1]
            print("this is the outer lips_end",upper_mouth_end)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            cv2.line(img_upper_mouth, (int(upper_mouth_start[0]), int(upper_mouth_start[1])), (int(upper_mouth_end[0]), int(upper_mouth_end[1])),
                     (255, 255, 255), 2)
            continue

        # extract upper mouth down(from 61 to 65)
        if count == 61:
            upper_mouth_down_start[0] = point[0]
            upper_mouth_down_start[1] = point[1]
            print("this is the inner_lips_start",upper_mouth_down_start)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            continue

        if count >= 61 and count <= 65:
            cv2.line(img_upper_mouth_down, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)

        if count == 65:
            upper_mouth_down_end[0] = point[0]
            upper_mouth_down_end[1] = point[1]
            print("this is the inner_lips_end",upper_mouth_down_end)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            cv2.line(img_upper_mouth_down, (int(upper_mouth_down_start[0]), int(upper_mouth_down_start[1])), (int(upper_mouth_down_end[0]), int(upper_mouth_down_end[1])),
                     (255, 255, 255), 2)
            continue

        # extract lower mouth up(from 61 to 65)
        if count == 61:
            lower_mouth_up_start[0] = point[0]
            lower_mouth_up_start[1] = point[1]
            print("this is the inner_lips_start",lower_mouth_up_start)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            continue

        if count == 65:
            lower_mouth_up_end[0] = point[0]
            lower_mouth_up_end[1] = point[1]
            print("this is the inner_lips_end",lower_mouth_up_end)
            #cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            cv2.line(img_lower_mouth_up, (int(lower_mouth_up_start[0]), int(lower_mouth_up_start[1])), (int(lower_mouth_up_end[0]), int(lower_mouth_up_end[1])),
                     (255, 255, 255), 2)
            continue

        if count >= 65 and count <= 68:
            cv2.line(img_lower_mouth_up, (int(point_dst[0]), int(point_dst[1])), (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)

        # extract lower mouth down(from 55 to 60)


        if count == 60:
            lower_mouth_down_end[0] = point[0]
            lower_mouth_down_end[1] = point[1]
            print("this is the inner_lips_end", lower_mouth_down_end)
            # cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
            cv2.line(img_lower_mouth_down, (int(lower_mouth_down_start[0]), int(lower_mouth_down_start[1])),
                     (int(lower_mouth_down_end[0]), int(lower_mouth_down_end[1])),
                     (255, 255, 255), 2)
            continue

        if count > 55 and count <= 60:
            cv2.line(img_lower_mouth_down, (int(point_dst[0]), int(point_dst[1])),
                     (int(point_src[0]), int(point_src[1])),
                     (255, 255, 255), 2)

    print("Outputing edging map")
    cv2.imwrite(dest_dir_final+final_filename+"_face_full_boundary.jpg",img_face_full_boudnary)
    cv2.imwrite(dest_dir_final+final_filename+"_face_left_eyebow.jpg",img_face_left_eyebow)
    cv2.imwrite(dest_dir_final+final_filename+"_face_right_eyebow.jpg",img_face_right_eyebow)
    cv2.imwrite(dest_dir_final+final_filename+"_face_nose_bridge.jpg",img_nose_bridge)
    cv2.imwrite(dest_dir_final+final_filename+"_face_nose_round.jpg",img_nose_round)
    cv2.imwrite(dest_dir_final+final_filename+"_face_left_eye.jpg",img_left_eye)
    cv2.imwrite(dest_dir_final+final_filename+"_face_right_eye.jpg",img_right_eye)
    cv2.imwrite(dest_dir_final+final_filename+"_face_upper_mouth.jpg",img_upper_mouth)
    cv2.imwrite(dest_dir_final+final_filename+"_face_upper_mouth_down.jpg",img_upper_mouth_down)
    cv2.imwrite(dest_dir_final+final_filename+"_face_lower_mouth_up.jpg",img_lower_mouth_up)
    cv2.imwrite(dest_dir_final+final_filename+"_face_lower_mouth_down.jpg",img_lower_mouth_down)



    #print(i,sample['image'].shape,sample['landmarks'].shape)


