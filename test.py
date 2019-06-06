from patch import Patch
from preprocessing import GCN, ZCA_whitening
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

i=0
I = cv2.imread('DRIVE/test/images/'+format(i+1,'02d')+'_test.tif')
J = cv2.imread('DRIVE/test/1st_manual/'+format(i+1,'02d')+'_manual1.tif',0)
mask = cv2.imread('DRIVE/test/mask/'+format(i+1,'02d')+'_test_mask.tif',0)
patch = Patch(I,J,mask)

X = np.zeros((1,27,27,3))
X[0] = patch.data

datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True) #GCN
datagen.fit(X)

iterator = datagen.flow(X)
for I in iterator:
    print(I.shape)
    cv2.imshow('patch',I[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cv2.imshow('patch',patch.data)
cv2.waitKey(0)
cv2.destroyAllWindows()

gcn = GCN(patch.data)
zca = ZCA_whitening(gcn)

cv2.imshow('patch',gcn)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('patch',zca)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(patch.label)
