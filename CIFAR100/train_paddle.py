from dataloader_paddle import * 
from configs import CONFIG
from network import network_paddle as network
import paddle 

paddle.set_device(CONFIG.device)

# Data Preparation
(train_x, train_y) , (test_x, test_y) = preprocessor(CONFIG.data_file, resize = CONFIG.resize)

# Network Definition
net = network(CONFIG.net, pretrained = CONFIG.pretrained)

#lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.00375,
#                                                T_max=50000 // 8 * CONFIG.epochs)
#optim = paddle.optimizer.Momentum(parameters = net.parameters(), 
#                                    learning_rate = lr,
#                                    weight_decay=paddle.regularizer.L2Decay(1e-6))


if CONFIG.optimizer.lower() == 'adam':
    optim = paddle.optimizer.Adam(parameters = net.parameters(),
                                  learning_rate = CONFIG.learning_rate,
                                  weight_decay = CONFIG.weight_decay)
else: 
    optim = paddle.optimizer.SGD(parameters = net.parameters(),
                                 learning_rate = CONFIG.learning_rate,
                                 weight_decay = CONFIG.weight_decay)

losses = []
accs = []

# Logwriter Initialization
from visualdl import LogWriter
import os 
try:
    os.makedirs(CONFIG.log_file)
except FileExistsError:
    pass
writer = LogWriter(CONFIG.log_file)


######################################
#              TRAINING
######################################
n = 50000
epochs = CONFIG.epochs
for epoch in range(len(accs) + 1, epochs + len(accs) + 1):
    for x , labels in dataloader(train_x, train_y, CONFIG.train_size,
                                 cut = CONFIG.cut, mix = CONFIG.mix, onehot = True):
        y = net(x) 
        labels = paddle.to_tensor(labels, dtype = 'float32')
        loss = paddle.nn.CrossEntropyLoss(soft_label = True)(y, labels)
        losses.append(loss.item())
        optim.clear_grad()
        loss.backward()
        optim.step()

    # learning rate decays at certain epochs
    if epoch in CONFIG.decay:
        optim.set_lr(optim.get_lr() * CONFIG.decay_rate)
        
    # compute the accuracy on the testing data
    accs.append(test(net, test_x, test_y, CONFIG.test_size))
    for i in range(len(losses) - n // CONFIG.train_size, len(losses)):
        writer.add_scalar(tag="train/loss", step = i, value = losses[i])  
    writer.add_scalar(tag="valid/top1 acc", step = len(accs), value = accs[-1][0]) 
    writer.add_scalar(tag="valid/top5 acc", step = len(accs), value = accs[-1][1]) 

    # save the model if better
    if accs[-1][0] >= max([i[0] for i in accs]):
        paddle.save(net.state_dict(), CONFIG.save_path)

        
writer.close()
