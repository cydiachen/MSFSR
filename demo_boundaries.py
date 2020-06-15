import models.FSRNet as FSRNet
import models.OFSR as OFSR
import models.MSFSR as MSFSR
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt


def get_boundary(test_dir,output_dir,model_path_1,model_path_2,model_path_3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X2_SR_net_stg1 = MSFSR.defineThreeStageGenerator(input_nc=3, output_nc=3)
    X2_SR_net_stg2 = MSFSR.defineThreeStageGenerator(input_nc=3, output_nc=3)
    X2_SR_net_stg3 = MSFSR.defineThreeStageGenerator(input_nc=3, output_nc=3)

    weights1 = torch.load(model_path_1)
    weights2 = torch.load(model_path_2)
    weights3 = torch.load(model_path_3)


    X2_SR_net_stg1.load_state_dict(weights1['model'].state_dict())
    X2_SR_net_stg2.load_state_dict(weights2['model'].state_dict())
    X2_SR_net_stg3.load_state_dict(weights3['model'].state_dict())


    X2_SR_net_stg1.cuda()
    X2_SR_net_stg1.eval()
    X2_SR_net_stg2.cuda()
    X2_SR_net_stg2.eval()
    X2_SR_net_stg3.cuda()
    X2_SR_net_stg3.eval()


    to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    to_image = transforms.Compose([
        transforms.ToPILImage()
    ])


    with torch.no_grad():
        for image_path in os.listdir(test_dir):
            img_name = osp.join(test_dir, image_path)
            img = Image.open(img_name)
            img = to_tensor(img)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out1 = X2_SR_net_stg1(img)[2]
            out2 = X2_SR_net_stg2(out1)[2]
            out3 = X2_SR_net_stg3(out2)[1]

            output = torch.zeros([1,1,64,64])

            # evaluate image info
            for i in range(2):
                output += out3[:,i,:,:].cpu().clone()

            for i in range(3,11):
                output += out3[:,i,:,:].cpu().clone()


            output = output.squeeze(0)
            output = to_image(output)
            output.save(osp.join(output_dir,image_path))

            # save images


if __name__ == "__main__":
    get_boundary(test_dir = "./input/",
                   output_dir = "./result/",
                   model_path_1="./pretrained_weights/MSFSR/model_stg1.pth",
                   model_path_2="./pretrained_weights/MSFSR/model_stg2.pth",
                   model_path_3="./pretrained_weights/MSFSR/model_stg3.pth")
    # evaluate_URDGN(test_dir = "", output_dir = "", model_path = "")


