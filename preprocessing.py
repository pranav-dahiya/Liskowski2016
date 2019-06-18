import numpy as np
import os
from time import sleep
from PIL import Image
from numpy import linalg
from numpy.random import rand
from tqdm import tqdm


def augmentation(I):
    out = np.zeros((11,27,27,3))
    out[0] = I[11:38,11:38,:]
    for i in range(10):
        #scale
        J = Image.fromarray(I)
        scale_factor = (rand()*0.5)+0.7
        J = J.resize((int(I.shape[0]*scale_factor),int(I.shape[1]*scale_factor)))
        #rotate
        J = J.rotate(int(np.random.rand()*180-90))
        #flip vertically and horizontally
        if rand() < 0.5:
            J = J.transpose(Image.FLIP_LEFT_RIGHT)
        if rand() < 0.5:
            J = J.transpose(Image.FLOP_TOP_BOTTOM)
        #gamma correction
        hsv = np.array(J.convert('HSV'))
        hsv[:,:,1] = hsv[:,:,1]**(rand()*((rand()*3.75)+0.25))
        hsv[:,:,2] = hsv[:,:,2]**(rand()*((rand()*3.75)+0.25))
        hsv[hsv>255] = 255
        J = Image.fromarray(hsv, 'HSV')
        out[i] = np.array(J, dtype=np.float32)
    return out

shape = (400000, 27, 27, 3)
flat_shape = (400000, shape[1]*shape[2]*shape[3])

if (os.path.exists('data/x_gcn_test.npy')):
    if(input("File exists. Do you want to overwrite?[y,N] ") != "y"):
        exit()

#Perform Global Contrast Normalization
x = np.memmap('data/x_test.npy', dtype=np.uint8, mode='r', shape=shape)
x_gcn = np.memmap('data/x_gcn_test.npy', dtype=np.float32, mode='w+', shape=shape)

print("Applying GCN:")
for i, I in tqdm(enumerate(x)):
    for channel in range(I.shape[2]):
        Ic = I[:,:,channel].astype(np.float32)
        Ic = (Ic-np.mean(Ic))/max(np.std(Ic), 0.0000001)
        x_gcn[i,:,:,channel] = Ic
del x
del x_gcn


#Perform ZCA whitening
x_flat = np.memmap('data/x_gcn_test.npy', dtype=np.float32, mode='r', shape=flat_shape)
x_zca = np.memmap('data/x_zca_test.npy', dtype=np.float32, mode='w+', shape=flat_shape)

print(1)
sigma = np.dot(x_flat.T, x_flat)/x_flat.shape[0]
print(2)
u, s, _ = linalg.svd(sigma)
print(3)
principal_components = np.dot(np.dot(u, np.diag(1/np.sqrt(s+10e-7))), u.T)
np.save('data/principal_components.npy', principal_components)
print(4)
for i in tqdm(range(100)):
    x_zca[i*4000:(i+1)*4000,:] = np.dot(x_flat[i*4000:(i+1)*4000,:], principal_components)
print(5)
del x_flat
del x_zca
