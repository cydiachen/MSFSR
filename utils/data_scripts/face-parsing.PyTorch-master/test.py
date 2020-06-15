#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt



#        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
#                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
#        For FSRNet, we choose
#        skin, lbrow,rbrow,leye,reye,lear,rear,nose,mouth,ulip,llip
#

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    part_of_interest = [1,2,3,4,5,7,8,10,11,12,13]
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    face_parsing_skin = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    face_parsing_lbrow = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    face_parsing_rbrow = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    face_parsing_leye = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    face_parsing_reye = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    face_parsing_lear = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    face_parsing_rear = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    face_parsing_nose = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    face_parsing_mouth = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    face_parsing_ulip = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    face_parsing_llip = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))


    num_of_class = np.max(vis_parsing_anno)

    for i in range(1,num_of_class +1):
        if i in part_of_interest:
            index = np.where(vis_parsing_anno == i)

            if i == 1:
                face_parsing_skin[index[0], index[1]] = 255
            if i == 2:
                face_parsing_lbrow[index[0], index[1]] = 255
            if i == 3:
                face_parsing_rbrow[index[0],index[1]] = 255
            if i == 4:
                face_parsing_leye[index[0],index[1]] = 255
            if i == 5:
                face_parsing_reye[index[0],index[1]] = 255
            if i == 7:
                face_parsing_lear[index[0],index[1]] = 255
            if i == 8:
                face_parsing_rear[index[0],index[1]] = 255
            if i == 10:
                face_parsing_nose[index[0],index[1]] = 255
            if i == 11:
                face_parsing_mouth[index[0],index[1]] = 255
            if i == 12:
                face_parsing_ulip[index[0],index[1]] = 255
            if i == 13:
                face_parsing_llip[index[0],index[1]] = 255

    if save_im:
        cv2.imwrite(save_path[:-4] +'_skin.png', face_parsing_skin)
        cv2.imwrite(save_path[:-4] +'_lbrow.png', face_parsing_lbrow)
        cv2.imwrite(save_path[:-4] +'_rbrow.png', face_parsing_rbrow)
        cv2.imwrite(save_path[:-4] +'_leye.png', face_parsing_leye)
        cv2.imwrite(save_path[:-4] +'_reye.png', face_parsing_reye)
        cv2.imwrite(save_path[:-4] +'_lear.png', face_parsing_lear)
        cv2.imwrite(save_path[:-4] +'_rear.png', face_parsing_rear)
        cv2.imwrite(save_path[:-4] +'_nose.png', face_parsing_nose)
        cv2.imwrite(save_path[:-4] +'_mouth.png', face_parsing_mouth)
        cv2.imwrite(save_path[:-4] +'_ulip.png', face_parsing_ulip)
        cv2.imwrite(save_path[:-4] +'_llip.png', face_parsing_llip)


def evaluate(respth='/home/lab216/Pictures/UnCropped/CelebAMask-HQ/CelebAMask-HQ/Parsing_Maps/', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            img = img.resize((48,48),Image.BILINEAR)
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))



if __name__ == "__main__":
    evaluate(dspth='/home/cydiachen/Desktop/DEMO/HR', respth = "/home/cydiachen/Desktop/DEMO/1/", cp='79999_iter.pth')


