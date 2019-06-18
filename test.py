import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def normalize(I):
    return (I-np.min(I))/(np.max(I)-np.min(I))
'''
i=0
I = cv2.imread('DRIVE/test/images/'+format(i+1,'02d')+'_test.tif')
J = cv2.imread('DRIVE/test/1st_manual/'+format(i+1,'02d')+'_manual1.tif',0)
mask = cv2.imread('DRIVE/test/mask/'+format(i+1,'02d')+'_test_mask.tif',0)
patch = Patch(I,J,mask)

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

hist = cv2.calcHist(gcn.astype(np.float32),[0],None,[256],[0,256])
hist,bins = np.histogram(gcn.ravel(),256,[0,256])
plt.hist(gcn.ravel(),256,[0,256]); plt.show()

cv2.imshow('patch',gcn.astype(np.float32))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('patch',zca)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(patch.label)
'''
x = np.memmap('data/x_test.npy', dtype=np.uint8, mode='r', shape=(400000, 27, 27, 3))
gcn = np.memmap('data/x_gcn_test.npy', dtype=np.float32, mode='r', shape=(400000, 27, 27, 3))
zca = np.memmap('data/x_zca_test.npy', dtype=np.float32, mode='r', shape=(400000, 27, 27, 3))
plt.figure()
plt.subplot(321)
plt.imshow(x[1])
plt.subplot(322)
plt.imshow(x[18])
plt.subplot(323)
plt.imshow(normalize(gcn[1]))
plt.subplot(324)
plt.imshow(normalize(gcn[18]))
plt.subplot(325)
plt.imshow(normalize(zca[1]))
plt.subplot(326)
plt.imshow(normalize(zca[18]))
plt.show()
