import cv2
import numpy as np
from patch import Patch

x = np.zeros((400000,27,27,3))
y = np.zeros((400000,2))

for i in range(20):
    I = cv2.imread('DRIVE/training/images/'+format(i+21,'02d')+'_training.tif')
    J = cv2.imread('DRIVE/training/1st_manual/'+format(i+21,'02d')+'_manual1.tif',0)
    mask = cv2.imread('DRIVE/training/mask/'+format(i+21,'02d')+'_training_mask.tif',0)
    print(i)
    for j in range(20000):
        index = i*20000+j
        try:
            patch = Patch(I,J,mask)
            x[index] = patch.data
            y[index,patch.label] = 1
        except ValueError:
            j -= 1

np.save('data/x_train.npy',x)
np.save('data/y_train.npy',y)
