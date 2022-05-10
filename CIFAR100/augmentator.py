import numpy as np 
from PIL import Image 

def augmentate(pics, labels = None, augmentation = False, 
                cut = False, mix = False, rotate = 15):
    '''Simple data augmentator'''
    pics2 = []
    n = len(pics)

    if not augmentation:
        pics2 = [np.array(pic) for pic in pics]
        return np.array(pics2), labels
    
    angles = (np.random.random(n) - .5) * (rotate * 2.)
    flips = np.random.randint(0, 2, n)
    for i in range(n):
        pic = pics[i]
        if flips[i]: pic = pic.transpose(Image.FLIP_LEFT_RIGHT)
        pic = pic.rotate(angles[i])
        pic = np.array(pic)
        pics2.append(pic)
    shape = pics2[0].shape 

    if cut and not mix:
        # cutout
        cutx = np.random.randint(cut, shape[0], n)
        cuty = np.random.randint(cut, shape[1], n)
        for i in range(n):
            pics2[i][cutx[i]-cut : cutx[i], 
                        cuty[i]-cut : cuty[i], : ] = 0
        labels *= (1. - cut * cut / pics2[0].size)

    elif mix and not cut:
        # mixup
        couple = np.arange(n)
        np.random.shuffle(couple)
        proportion = np.random.random(n)

        # warning: do not modify in-place
        pics3 = [proportion[i] * pics2[i] + (1 - proportion[i]) * pics2[couple[i]] 
                    for i in range(n)]

        # pointer
        pics2 = pics3 

        labels = labels + (labels[couple] - labels) * proportion.reshape((n,1))

    elif cut and mix:
        # cutmix
        couple = np.arange(n)
        np.random.shuffle(couple)
        proportion = np.random.random(n)
        rw = (shape[0] * np.sqrt(proportion)).astype('uint8')
        rh = ((shape[1] / shape[0]) * rw).astype('uint8')

        rx = np.random.randint(65535, size = n) % (shape[0] - rw)
        ry = np.random.randint(65535, size = n) % (shape[1] - rh)

        # warning: do not modify in-place
        pics3 = np.array(pics2)
        for i in range(n):
            pics3[i][rx[i] : rx[i]+rw[i], ry[i] : ry[i]+rh[i]] = \
                pics2[couple[i]][rx[i] : rx[i]+rw[i], ry[i] : ry[i]+rh[i]] 

        # pointer
        pics2 = pics3

        labels = labels + (labels[couple] - labels) * proportion.reshape((n,1))
            
    return np.array(pics2), labels
 

def resizer(pics, size):
    '''
    resize pictures to the given size
    '''
    if issubclass(type(pics[0]), np.ndarray):
        pics = [Image.fromarray(pic) for pic in pics]
    
    for i in range(len(pics)):
        pics[i] = pics[i].resize(size, Image.ANTIALIAS)
    
    return pics