import os
import cv2 as cv
import numpy as np


people = ['KINGCHARLES','EMMAWATSON','JOEBIDEN']
directory =r"C:\Users\sanji\Desktop\OPENCV\Faces"
features = []
labels = []

def prepare_data():
    for x in people:
        path = os.path.join(directory,x)
        label = people.index(x)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            read_img = cv.imread(img_path)
            gray = cv.cvtColor(read_img,cv.COLOR_BGR2GRAY)
            haar_cascade = cv.CascadeClassifier('simple projects/haar_face.xml')
            faces_rect =haar_cascade.detectMultiScale(gray,scaleFactor =2.1,minNeighbors =4 )
            for (x,y,z,h) in faces_rect:
                face_index = gray[y:y+h,x:x+z]
                features.append(face_index)
                labels.append(label)
prepare_data()
features = np.array(features,dtype ='object')
labels = np.array(labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

face_recognizer.save('facereco.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)