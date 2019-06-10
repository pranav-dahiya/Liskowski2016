import numpy as np
from numpy import linalg
from tqdm import tqdm

def GCN(I):
    I = I.astype(np.float32)
    for channel in range(3):
        mean = np.sum(I[:,:,channel])/(I.shape[0]*I.shape[1])
        sigma = (np.sum(np.square(I[:,:,channel]-mean))/(I.shape[0]*I.shape[1]))**0.5
        I[:,:,channel] -= mean
        I[:,:,channel] /= sigma
    return I

def ZCA_whitening(I):
    W_ZCA = np.zeros(I.shape)
    for channel in range(3):
        sigma = np.dot(I[:,:,channel].T,I[:,:,channel])/I.shape[0]
        u,s,_ = linalg.svd(sigma)
        principal_components = np.dot(np.dot(u,np.diag(1./np.sqrt(s+10e-7))),u.T)
        W_ZCA[:,:,channel] = np.dot(I[:,:,channel],principal_components)
    return W_ZCA

def augmentation(I):
    I = ZCA_whitening(I)


shape = (400000,27,27,3)

x = np.memmap('data/x_train.npy',dtype=np.uint8,shape=shape)
x_gcn = np.memmap('data/x_gcn_train.npy',dtype=np.uint8,mode='write',shape=shape)
x_zca = np.memmap('data/x_zca_train.npy',dtype=np.uint8,mode='write',shape=shape)
for i, I in tqdm(enumerate(x)):
    x_gcn[i] = GCN(I)
    x_zca[i] = ZCA_whitening(x_gcn[i])

del x
del x_gcn
del x_zca
