import cv2
import dlib
import numpy as np

path = "/home/cydiachen/Desktop/DEMO/HR/270.jpg"
img = cv2.imread(path)
img = cv2.resize(img,(16,16))
img = cv2.resize(img,(128,128))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
landmark = np.zeros(img.shape)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

dets = detector(gray,1)

for face in dets:
    shape = predictor(img,face)

    for pt in shape.parts():
        pt_pos = (pt.x,pt.y)
        cv2.circle(landmark,pt_pos,2,(255,255,255),thickness=-1)

cv2.imwrite("/media/cydiachen/DATASET/FSR/visualize/landmark.jpg",landmark)