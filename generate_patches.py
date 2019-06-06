import cv2
import numpy as np
from patch import Patch

x = np.zeros((400000,27,27,3))
y = np.zeros((400000,2))

for i in range(20):
    I = cv2.imread('DRIVE/test/images/'+format(i+1,'02d')+'_test.tif')
    J = cv2.imread('DRIVE/test/1st_manual/'+format(i+1,'02d')+'_manual1.tif',0)
    mask = cv2.imread('DRIVE/test/mask/'+format(i+1,'02d')+'_test_mask.tif',0)
    print(i)
    for j in range(20000):
        try:
            patch = Patch(I,J,mask)
            x[i*20+j] = patch.data
            y[i*20+j,patch.label] = 1
        except ValueError:
            j -= 1

np.save('data/x_test.npy',x)
np.save('data/y_test.npy',y)
