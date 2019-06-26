import numpy as np
import os
import matplotlib.pyplot as plt
from time import sleep
from PIL import Image
from numpy import linalg
from numpy.random import rand
from tqdm import tqdm

shape = (400000, 27, 27, 3)
flat_shape = (400000, shape[1]*shape[2]*shape[3])


def GCN():
    #Perform Global Contrast Normalization
    x = np.memmap('Data-3/x_test.npy', dtype=np.uint8, mode='r', shape=shape)
    x_gcn = np.memmap('Data-3/x_gcn_test.npy', dtype=np.float32, mode='w+', shape=shape)

    for i, I in tqdm(enumerate(x)):
        for channel in range(I.shape[2]):
            Ic = I[:,:,channel].astype(np.float32)
            Ic = (Ic-np.mean(Ic))/max(np.std(Ic), 0.0000001)
            x_gcn[i,:,:,channel] = Ic
    del x
    del x_gcn


def ZCA_whitening():
    #Perform ZCA whitening
    x_flat = np.memmap('Data-3/x_gcn_test.npy', dtype=np.float32, mode='r', shape=flat_shape)
    x_zca = np.memmap('Data-3/x_zca_test.npy', dtype=np.float32, mode='w+', shape=flat_shape)

    print(1)
    sigma = np.dot(x_flat.T, x_flat)/x_flat.shape[0]
    print(2)
    u, s, _ = linalg.svd(sigma)
    print(3)
    principal_components = np.dot(np.dot(u, np.diag(1/np.sqrt(s+10e-7))), u.T)
    print(4)
    for i in tqdm(range(100)):
        x_zca[i*4000:(i+1)*4000,:] = np.dot(x_flat[i*4000:(i+1)*4000,:], principal_components)
    print(5)
    del x_flat
    del x_zca


print("Performing GCN")
if (os.path.exists('Data-3/x_gcn_test.npy')):
    if(input("File exists. Do you want to overwrite?[y,N] ") == "y"):
        GCN()
else:
    GCN()

print("Performing ZCA whitening")
if (os.path.exists('Data-3/x_zca_test.npy')):
    if(input("File exists. Do you want to overwrite?[y,N] ") == "y"):
        ZCA_whitening()
else:
    ZCA_whitening()
