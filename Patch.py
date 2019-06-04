import cv2
import numpy as np

class Patch:
    def __init__(self,fname):
        while True:
            x = np.random.randint(920)
            y = np.random.randint(920)
            if (x-460)**2 + (y-460)**2 < 444**2:
                break
        I = cv2.imread(fname+".jpg")
        self.data = I[x+20:x+47,y+8:y+35,:]
        J = cv2.imread(fname+"_1stHO.png")
        self.label = np.sum(J[x+20:x+47,y+8:y+35,:])>0
