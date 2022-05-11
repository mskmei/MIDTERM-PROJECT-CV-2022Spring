class CONFIG:
    data_file = None
    log_file  = './log'
    
    # # example for paddle 
    # save_path = './models/model.pdparams'
    # device    = 'gpu:0'

    # # example for torch
    save_path   = './models/model.pth'
    device      = 'cuda:0'


    # data augmentation
    resize        = (224, 224)
    augmentation  = True
    cut           = 56            # size of Cutout, set to zero to disable
    mix           = True          # enable (True) / disable (False) Mixup
                                  # When cut > 0 and mix == True, then it activates CutMix

    # training
    net           = 'resnet18'    # support: resnet18, resnet34, resnet50, resnet101
    pretrained    = True          # whether or not use the pretrained parameters
    train_size    = 64            # training batch size
    test_size     = 100           # testing batch size
    epochs        = 30
    learning_rate = 3e-3          # learning rate for optimizer
    weight_decay  = 5e-4          # weight-decay for regularization
    optimizer     = 'sgd'         # support 'adam' / 'sgd'
    decay         = (11, 21)      # learning rate decays at these epochs
    decay_rate    = .1            # learning rate decays by the rate
