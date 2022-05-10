from dataloader_torch import * 
from configs import CONFIG
from network import network_torch as network
import torch

cuda = CONFIG.device

# Data Preparation
(train_x, train_y) , (test_x, test_y) = preprocessor(CONFIG.data_file, resize = CONFIG.resize)

# Network Definition
net = network(CONFIG.net, pretrained = CONFIG.pretrained, cuda = cuda)
optim = torch.optim.Adam(net.parameters(), lr = CONFIG.learning_rate)

losses = []
accs = []

# Logwriter Initialization
from torch.utils.tensorboard import SummaryWriter
import os 
try:
    os.makedirs(CONFIG.log_file)
except FileExistsError:
    pass
writer = SummaryWriter(CONFIG.log_file)


######################################
#              TRAINING
######################################
n = 50000
epochs = CONFIG.epochs
for epoch in range(len(accs) + 1, epochs + len(accs) + 1):
    for x , labels in dataloader(train_x, train_y, CONFIG.train_size,
                                 cut = CONFIG.cut, mix = CONFIG.mix, onehot = True):
        y = net(x.to(cuda)) 
        labels = torch.tensor(labels, dtype = torch.float32, device = cuda) 
        loss = torch.nn.BCEWithLogitsLoss()(y, labels)
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()

    # learning rate decays at certain epochs
    if epoch in CONFIG.decay:
        optim.param_groups[0]['lr'] *= CONFIG.decay_rate
        
    # compute the accuracy on the testing data
    accs.append(test(net, test_x, test_y, CONFIG.test_size, cuda = cuda))
    for i in range(len(losses) - n // CONFIG.train_size, len(losses)):
        writer.add_scalar(tag="train/loss", global_step = i, scalar_value = losses[i])  
    writer.add_scalar(tag="valid/top1 acc", global_step = len(accs), scalar_value = accs[-1][0]) 
    writer.add_scalar(tag="valid/top5 acc", global_step = len(accs), scalar_value = accs[-1][1]) 

    # save the model if better
    if accs[-1][0] >= max([i[0] for i in accs]):
        torch.save(net.state_dict(), CONFIG.save_path)


writer.close()
