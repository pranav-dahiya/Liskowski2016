import numpy as np
from tqdm import tqdm
from PIL import Image
from patch import Patch

x = np.memmap('data/x_balanced_train.npy',dtype=np.uint8,mode='w+',shape=(400000,27,27,3))
y = np.memmap('data/y_balanced_train.npy',dtype=np.uint8,mode='w+',shape=(400000,2))

#x = np.memmap('data/x_test.npy',dtype=np.uint8,mode='write',shape=(400000,27,27,3))
#y = np.memmap('data/y_test.npy',dtype=np.uint8,mode='write',shape=(400000,2))

for i in range(20):
    #I = np.array(Image.open('DRIVE/training/images/'+format(i+21,'02d')+'_training.tif'))
    #J = np.array(Image.open('DRIVE/training/1st_manual/'+format(i+21,'02d')+'_manual1.tif'))
    #mask = np.array(Image.open('DRIVE/training/mask/'+format(i+21,'02d')+'_training_mask.tif'))
    I = np.array(Image.open('DRIVE/test/images/'+format(i+1,'02d')+'_test.tif'))
    J = np.array(Image.open('DRIVE/test/1st_manual/'+format(i+1,'02d')+'_manual1.tif'))
    mask = np.array(Image.open('DRIVE/test/mask/'+format(i+1,'02d')+'_test_mask.tif'))
    print(i)
    for j in tqdm(range(20000)):
        index = i*20000+j
        try:
            patch = Patch(I,J,mask,label=int(np.random.rand()<0.5))
            x[index] = patch.data
            y[index,patch.label] = 1
        except ValueError:
            j -= 1

del x
del y
