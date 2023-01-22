import numpy as np
import cv2 as cv


people = ['KINGCHARLES','EMMAWATSON','JOEBIDEN']
haar_cascade = cv.CascadeClassifier('simple projects/haar_face.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('facereco.yml')

img = cv.imread(r'C:\Users\sanji\Desktop\OPENCV\Faces\VALIDATION\JOEBIDEN\joe1.jpeg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('unknown person',gray)

face_detect = haar_cascade.detectMultiScale(gray,1.1,4)
for (w,x,y,z) in face_detect:
    face_point = gray[x:x+z,w:w+y]

    label, acc = face_recognizer.predict(face_point)
    print(f'Name:{people[label]} with an accuracy {acc}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0), thickness = 2)
    cv.rectangle(img,(w,x),(w+y,x+z),(0,255,0), thickness=2)
cv.imshow('Detected Face',img)

cv.waitKey(0)