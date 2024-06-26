
import cv2
import numpy as np
with open('data/train.txt') as f:
    lines=f.readlines()
for line in lines[:10]:
    line=line.split(' ')
    img=cv2.imread('data/'+line[0])
    box = np.array(list(map(int, line[1].split(','))))
    img=cv2.rectangle(img,box[:2],box[2:4],[255,0,0])
    cv2.imshow('a',img)
    cv2.waitKey(2000)