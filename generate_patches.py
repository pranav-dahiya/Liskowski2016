import cv2
import numpy as np
from patch import Patch

x = np.memmap('data/x_test.npy',dtype=np.uint8,mode='write',shape=(400000,27,27,3))
y = np.memmap('data/y_test.npy',dtype=np.uint8,mode='write',shape=(400000,2))

for i in range(20):
    I = cv2.imread('DRIVE/test/images/'+format(i+1,'02d')+'_test.tif')
    J = cv2.imread('DRIVE/test/1st_manual/'+format(i+1,'02d')+'_manual1.tif',0)
    mask = cv2.imread('DRIVE/test/mask/'+format(i+1,'02d')+'_test_mask.tif',0)
    print(i)
    for j in range(20000):
        index = i*20000+j
        try:
            patch = Patch(I,J,mask)
            x[index] = patch.data
            y[index,patch.label] = 1
        except ValueError:
            j -= 1

del x
del y
