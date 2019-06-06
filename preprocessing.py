import numpy as np

def GCN(I):
    for channel in range(3):
        mean = np.sum(I[:,:,chaneel])/(I.shape[0]*I.shape[1])
        sigma = (np.sum(np.square(I[:,:,channel]-mean))/(I.shape[0]*I.shape[1]))**0.5
        I[:,:,channel] -= mean
        I[:,:,channel] /= sigma
    return I

def ZCA_whitening(I):
