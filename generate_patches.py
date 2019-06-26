import numpy as np
from tqdm import tqdm
from PIL import Image
from patch import Patch


def generate_patches(x, y, split, balanced=False):
    for i in tqdm(range(20)):
        if split == 'train':
            I = np.array(Image.open('DRIVE/training/images/'+format(i+21,'02d')+'_training.tif'))
            J = np.array(Image.open('DRIVE/training/1st_manual/'+format(i+21,'02d')+'_manual1.tif'))
            mask = np.array(Image.open('DRIVE/training/mask/'+format(i+21,'02d')+'_training_mask.tif'))
        elif split == 'test':
            I = np.array(Image.open('DRIVE/test/images/'+format(i+1,'02d')+'_test.tif'))
            J = np.array(Image.open('DRIVE/test/1st_manual/'+format(i+1,'02d')+'_manual1.tif'))
            mask = np.array(Image.open('DRIVE/test/mask/'+format(i+1,'02d')+'_test_mask.tif'))
        for j in range(20000):
            index = i*20000+j
            try:
                label = None
                if balanced:
                    label = int(np.random.rand()<0.5)
                patch = Patch(I, J, mask, size=x.shape[1], label=label)
                x[index] = patch.data
                y[index,patch.label] = 1
            except ValueError:
                j -= 1


print("Generating Training Patches:")
x = np.memmap('data/x_train.npy', dtype=np.uint8, mode='w+', shape=(400000, 49, 49, 3))
y = np.memmap('data/y_train.npy', dtype=np.uint8, mode='w+', shape=(400000, 2))
generate_patches(x, y, 'train')
del x
del y

print("Generating Test Patches:")
x = np.memmap('data/x_test.npy', dtype=np.uint8, mode='w+', shape=(400000, 27, 27, 3))
y = np.memmap('data/y_test.npy', dtype=np.uint8, mode='w+', shape=(400000, 2))
generate_patches(x, y, 'test')
del x
del y

print("Generating Balanced Patches:")
x = np.memmap('data/x_balanced_train.npy', dtype=np.uint8, mode='w+', shape=(400000, 27, 27, 3))
y = np.memmap('data/y_balanced_train.npy', dtype=np.uint8, mode='w+', shape=(400000, 2))
generate_patches(x, y, 'train', True)
del x
del y
