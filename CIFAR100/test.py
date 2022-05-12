from configs import CONFIG

path = input('Input the path to your Resnet-18 model here\n >> ')
device = 0
path = path.strip().strip("'").strip('"')
if path.endswith('.pdparams'):
    import paddle
    if paddle.fluid.is_compiled_with_cuda():
        device = input('Input the index of GPU you want to test on:')
        if not device.startswith('gpu:'):
            device = 'gpu:' + device 
    else:
        device = 'cpu'
    paddle.set_device(device)
    from network import network_paddle as network 
    net = network('resnet18', False)
    net.set_state_dict(paddle.load(path))

    from dataloader_paddle import preprocessor, test

elif path.endswith('.pth'):
    import torch 
    if torch.cuda.is_available():
        device = input('Input the index of CUDA you want to test on:')
        if not device.startswith('cuda:'):
            device = 'cuda:' + device 
    else:
        device = 'cpu'
    from network import network_torch as network 
    net = network('resnet18', False, cuda = device)
    net.load_state_dict(torch.load(path, map_location = device))
    
    from dataloader_torch import preprocessor, test
else:
    raise ValueError('Model format not supported or model directory not found.')




print('Loading Dataset......')
(train_x, train_y) , (test_x, test_y) = preprocessor(CONFIG.data_file, 
                                        resize = CONFIG.resize, only_test = True)

acc = test(net, test_x, test_y, CONFIG.test_size, verbose = True, cuda = device)

print('Top 1 Acc = %.2f%%\nTop 5 Acc = %.2f%%'%(acc[0] * 100, acc[1] * 100))