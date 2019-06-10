import cv2
import numpy as np

class Patch:
    def __init__(self,I,J,mask,size=27):
        while True:
            x = np.random.randint(584-(size/2))
            y = np.random.randint(565-(size/2))
            if mask[x+int(size/2),y+int(size/2)] == 255:
                break
        self.data = I[x:x+size,y:y+size,:]
        self.label = int(J[x+int(size/2),y+int(size/2)]==255)
        if(not(self.label == 0 or self.label == 1)):
            print(self.label)
