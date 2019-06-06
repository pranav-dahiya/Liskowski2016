import cv2
import numpy as np

class Patch:
    def __init__(self,I,J,mask,size):
        while True:
            x = np.random.randint(584-(size/2))
            y = np.random.randint(565-(size/2))
            if mask[x+(size/2),y+(size/2)] == 255:
                break
        self.data = I[x-(size/2):x+(size-size/2),y-(size/2):y+(size-size/2),:]
        self.label = int(J[x,y]==255)
