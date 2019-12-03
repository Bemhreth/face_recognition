# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 01:14:02 2018

@author: bemhret gezahegn
"""
import os
import pickle
import numpy as np
from PIL import Image
import cv2
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR,"image")
face_cascade=cv2.CascadeClassifier('data\haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_ids={}
y_lables=[]
x_train=[]
for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()  
            #print(label,path)
            if not label in label_ids:
            
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
           # print(label_ids)
            pil_image=Image.open(path).convert("L")
            image_array=np.array(pil_image,"uint8")
           # print(image_array)
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            for(x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_lables.append(id_)
#print(y_labels)
# print(x_train)
with open("lables.pickles",'wb') as f:
    pickle.dump(label_ids,f)
recognizer.train(x_train,np.array(y_lables))
recognizer.save("trainner.yml")