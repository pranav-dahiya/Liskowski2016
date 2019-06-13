import numpy as np
import cv2
from numpy import linalg
from numpy.random import rand
from tqdm import tqdm


def GCN(I):
    GCN = np.zeros(I.shape)
    for channel in range(I.shape[2]):
        Ic = I[:,:,channel].astype(np.float32)
        Ic = (Ic-np.mean(Ic))/max(np.std(Ic), 0.0000001)
        GCN[:,:,channel] = Ic
    return GCN


def ZCA_whitening(X, W_ZCA):
    shape = (27*27*3, 27*27*3)
    sigma = np.memmap('data/sigma.npy', dtype=np.float32, mode='w+', shape=shape)
    sigma = np.dot(X.T, X)/X.shape[0]
    del sigma
    print(1)
    sigma = np.memmap('data/sigma.npy', dtype=np.float32, shape=shape)
    u = np.memmap('data/u.npy', dtype=np.float32, mode='w+', shape=shape)
    u, s, _ = linalg.svd(sigma)
    S = np.memmap('data/s.npy', dtype=np.float32, mode='w+', shape=shape)
    S = np.diag(1/np.sqrt(s+10e-7))
    del u
    del S
    del sigma
    print(2)
    u = np.memmap('data/u.npy', dtype=np.float32, shape=shape)
    a = np.memmap('data/s.npy', dtype=np.float32, shape=shape)
    principal_components = np.memmap('data/principal_components.npy', dtype=np.float32, mode='w+', shape=shape)
    principal_components = np.dot(np.dot(u, s), u.T)
    del u
    del s
    del principal_components
    print(3)
    principal_components = np.memmap('data/principal_components.npy', dtype=np.float32, shape=shape)
    W_ZCA = np.dot(X, principal_components)
    del principal_components
    print(4)


def augmentation(I):
    out = np.zeros((11,27,27,3))
    out[0] = I[11:38,11:38,:]
    for i in range(10):
        #scale
        scale_factor = (rand()*0.5)+0.7
        I_ = cv2.resize(I,(int(I.shape[0]*scale_factor),int(I.shape[1]*scale_factor)))
        #rotate
        M = cv2.getRotationMatrix2D((I.shape[0]/2,I.shape[1]/2),(rand()*180)-90,1)
        I_ = cv2.warpAffine(I_,M,I.shape[:2])
        #flip vertically and horizontally
        if rand() < 0.5:
            I_ = cv2.flip(I_,1)
        if rand() < 0.5:
            I_ = cv2.flip(I_,0)
        #gamma correction
        hsv = cv2.cvtColor(I_,cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = hsv[:,:,1]**(rand()*((rand()*3.75)+0.25))
        hsv[:,:,2] = hsv[:,:,2]**(rand()*((rand()*3.75)+0.25))
        hsv[hsv>255] = 255
        I_= cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        out[i] = I_[11:38,11:38,:]
    return out

shape = (400000, 27, 27, 3)

x = np.memmap('data/x_test.npy', dtype=np.uint8, shape=shape)
x_gcn = np.memmap('data/x_gcn_test.npy', dtype=np.float32,mode='w+', shape=shape)

print("Applying GCN:")
for i, I in tqdm(enumerate(x)):
    x_gcn[i] = GCN(I)
del x
del x_gcn

x_gcn = np.memmap('data/x_gcn_test.npy', dtype=np.float32, shape=shape)
x_zca = np.memmap('data/x_zca_test.npy', dtype=np.float32, mode='w+', shape=shape)
W_ZCA = np.memmap('data/W_ZCA.npy', dtype=np.float32, mode='w+', shape=(400000, 27*27*3))
x_gcn_flat = np.memmap('data/x_gcn_flat.npy', dtype=np.float32, mode='w+', shape=(400000, 27*27*3))
x_gcn_flat = np.reshape(x_gcn, (400000, 27*27*3))
ZCA_whitening(x_gcn_flat, W_ZCA)
x_zca = np.reshape(W_ZCA, shape)
del W_ZCA
del x_zca
