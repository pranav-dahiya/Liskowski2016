import cv2
import numpy as np

class Patch:
    def __init__(self,I,J,mask,size=27,label=None):
        while True:
            x = np.random.randint(584-size)
            y = np.random.randint(565-size)
            if mask[x+int(size/2),y+int(size/2)] == 255:
                if label != None:
                    if int(J[x+int(size/2),y+int(size/2)]==255) == label:
                        break
        self.data = I[x:x+size,y:y+size,:]
        self.label = int(J[x+int(size/2),y+int(size/2)]==255)
        if(not(self.label == 0 or self.label == 1)):
            print(self.label)
