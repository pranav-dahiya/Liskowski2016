import cv2
import numpy as np

class Patch:
    def __init__(self,I,J,mask):
        while True:
            x = np.random.randint(584)
            y = np.random.randint(565)
            if mask[x,y] == 255:
                break
        self.data = I[x-13:x+14,y-13:y+14,:]
        self.label = int(J[x,y]==255)
