# -*- coding:utf-8 -*-

import numpy as np
import cv2
import dlib
import pandas as pd
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks.dat')

data_dir = "/home/cydiachen/Desktop/DEMO/LR/"


def read_img(full_filename):
    img = cv2.imread(full_filename)
    img = cv2.resize(img,(128,128))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img, img_gray

def count_faces(img):
    rects = detector(img, 0)
    count = len(rects)

    return rects, count


def save_imgs(img, dir, filename):
    img = cv2.imwrite(dir + filename, img)


# 先根据下面定义的Boundary Line作出11张处理过后的Face Boundary Line


# main

filename = os.listdir(data_dir)
filename.sort()
for name in filename:
    full_path = os.path.join(data_dir, name)
    img, img_gray = read_img(full_path)

    rects, num_faces = count_faces(img_gray)
    print(num_faces)
    # 这一部分来保存提取到的81个人脸特征点
    if num_faces == 1:
        with open("/media/cydiachen/DATASET/FSR/visualize//" + "landmarks.CSV", "a+") as f:
            f.write(name)
            for i in range(num_faces):
                landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
                for idx, point in enumerate(landmarks):
                    pos = (point[0, 0], point[0, 1])
                    f.write("," + str(pos[0]) + "," + str(pos[1]))
                    print(idx, pos)
            f.write("\n")

            cv2.imwrite("/media/cydiachen/DATASET/FSR/visualize/" + name, img)
