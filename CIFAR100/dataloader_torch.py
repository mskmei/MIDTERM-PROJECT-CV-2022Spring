import numpy as np
import torch
from tqdm import tqdm 
from augmentator import *

def dataloader(data, labels, batch_size = 128, shuffle = True, verbose = True,
                resize = (224,224), augmentation = True, onehot = False,
                cut = False, mix = False, rotate = 15):
    
    order = np.arange(labels.shape[0], dtype = 'int32')
    if shuffle: np.random.shuffle(order)
    eye = np.eye(np.max(labels) + 1)

    gen = range(labels.shape[0] // batch_size)
    if verbose: gen = tqdm(gen)
    for i in gen:
        x = [data[order[j]] for j in range(i * batch_size, (i+1) * batch_size)]
        l = labels[order[i * batch_size: i * batch_size + batch_size]]

        # extract the onehot encoding
        l = eye[l]

        # augmentate the data
        x , l = augmentate(x, l, augmentation = augmentation, 
                            cut = cut, mix = mix, rotate = rotate)

        # transpose the dimensions to match (N,C,H,W)
        x = x.transpose((0,3,1,2))
        x = torch.tensor(x, dtype = torch.float32) / 255.

        # retrieve the non-onehot encoding if asked
        if not onehot:
            l = labels[order[i * batch_size: i * batch_size + batch_size]]
        yield x , l


def topk(y, labels, top):
    '''compute the number of hits in the top-k prediction'''
    topy = np.argpartition(y, -top, axis=-1)[:,-top:]
    topy -= labels.reshape((labels.shape[0], 1))
    return topy.size - np.count_nonzero(topy)


def test(net, data, labels, batch_size = 100, tops = [1,5],
            verbose = False, augmentation = False, cuda = 'cpu'):
    accs = [0] * len(tops)
    tops.sort()
    with torch.no_grad():
        for x, label in dataloader(data, labels, batch_size = batch_size, 
                            shuffle = False, verbose = verbose, augmentation = augmentation):
            y = net(x.to(cuda)).cpu().detach().numpy()
            label = np.array(label)
            for i in range(len(tops)):
                accs[i] += topk(y, label, tops[i])
    return np.array(accs) / labels.shape[0]
    

def preprocessor(data_file, resize = (32,32), only_test = False):
    from torchvision.datasets import CIFAR100

    try:
        data = CIFAR100(root = data_file, train = False, download = True)
    except:
        print('Data file not found.')
        data_file = './'
        data = CIFAR100(root = data_file, train = False, download = True)

    test_y = np.array([data[i][1] for i in range(10000)])
    test_x = [data[i][0] for i in range(10000)]
    test_x = resizer(test_x, size = resize)
    del data 


    if not only_test:
        data = CIFAR100(root = data_file, train = True, download = True)
        train_y = np.array([data[i][1] for i in range(50000)])
        train_x = [data[i][0] for i in range(50000)]
        train_x = resizer(train_x, size = resize)
        del data 
    else:
        train_x, train_y = 0, 0
 
    return (train_x, train_y) , (test_x, test_y)

