from patch import Patch
import numpy as np
import cv2
import matplotlib.pyplot as plt

patch = Patch("Image_01L")

print(np.max(patch.data))

cv2.imshow('patch',patch.data)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(patch.label)
