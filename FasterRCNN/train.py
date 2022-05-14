import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from nets.frcnn import FasterRCNN
from nets.frcnn_training import FasterRCNNTrainer, weights_init
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils_fit import fit_one_epoch
from torch.utils.tensorboard import SummaryWriter
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--cuda', dest='cuda', help='Whether to use GPU or not.', default=True)
parser.add_option('--backbone', dest='backbone', help='The backbone to use in feature extraction.', default='resnet50')
parser.add_option('--pretrained', dest='pretrained', help='Whether to use the pretrained weights of the backbone.',
				  default=True)

(options, args) = parser.parse_args()


if options.backbone != 'resnet50' and options.backbone != 'vgg':
	parser.error('Error:the backbone of network should be either resnet50 or vgg.')

writer = SummaryWriter('logs/tensorboard')

root = '/SSD_DISK/users/yuanjunhao/work'

Cuda = bool(options.cuda)

classes_path = os.path.join(root, 'model_data/voc_classes.txt')
class_names = []
with open(classes_path) as f:
	for line in f:
		class_names.append(line.strip())
num_classes = len(class_names)

model_path = ''
# ------------------------------------------------------#
#   输入的shape大小
# ------------------------------------------------------#
input_shape = [600, 600]
# ---------------------------------------------#
#   vgg或者resnet50
# ---------------------------------------------#
backbone = options.backbone
# ----------------------------------------------------------------------------------------------------------------------------#
#   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
#   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
#   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
#   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
# ----------------------------------------------------------------------------------------------------------------------------#
pretrained = bool(options.pretrained)
# ------------------------------------------------------------------------#
#   anchors_size用于设定先验框的大小，每个特征点均存在9个先验框。
#   anchors_size每个数对应3个先验框。
#   当anchors_size = [8, 16, 32]的时候，生成的先验框宽高约为：
#   [90, 180] ; [180, 360]; [360, 720]; [128, 128];
#   [256, 256]; [512, 512]; [180, 90] ; [360, 180];
#   [720, 360]; 详情查看anchors.py
#   如果想要检测小物体，可以减小anchors_size靠前的数。
#   比如设置anchors_size = [4, 16, 32]
# ------------------------------------------------------------------------#
anchors_size = [8, 16, 32]

# ----------------------------------------------------#
#   训练分为两个阶段，分别是冻结阶段和解冻阶段。
#   显存不足与数据集大小无关，提示显存不足请调小batch_size。
# ----------------------------------------------------#
# ----------------------------------------------------#
#   冻结阶段训练参数
#   此时模型的主干被冻结了，特征提取网络不发生改变
#   占用的显存较小，仅对网络进行微调
# ----------------------------------------------------#
Init_Epoch = 0
Freeze_Epoch = 30
Freeze_batch_size = 4
Freeze_lr = 1e-4
# ----------------------------------------------------#
#   解冻阶段训练参数
#   此时模型的主干不被冻结了，特征提取网络会发生改变
#   占用的显存较大，网络所有的参数都会发生改变
# ----------------------------------------------------#
UnFreeze_Epoch = 70
Unfreeze_batch_size = 2
Unfreeze_lr = 1e-5

Freeze_Train = True

num_workers = 4
# ----------------------------------------------------#
#   获得图片路径和标签
# ----------------------------------------------------#
train_annotation_path = os.path.join(root, 'trainSet.txt')
val_annotation_path = os.path.join(root, 'valSet.txt')

# ----------------------------------------------------#
#   获取classes和anchor
# ----------------------------------------------------#

model = FasterRCNN(num_classes, anchor_scales=anchors_size, backbone=backbone, pretrained=pretrained)
if not pretrained:
	weights_init(model)
if model_path != '':
	print('Load weights {}.'.format(model_path))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_dict = model.state_dict()
	pretrained_dict = torch.load(model_path, map_location=device)
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)

model_train = model.train()
if Cuda:
	model_train = torch.nn.DataParallel(model)
	cudnn.benchmark = True
	model_train = model_train.cuda()

# ---------------------------#
#   读取数据集对应的txt
# ---------------------------#
with open(train_annotation_path) as f:
	train_lines = f.readlines()
with open(val_annotation_path) as f:
	val_lines = f.readlines()
num_train = len(train_lines)
num_val = len(val_lines)

if True:
	batch_size = Freeze_batch_size
	lr = Freeze_lr
	start_epoch = Init_Epoch
	end_epoch = Freeze_Epoch

	epoch_step = num_train // batch_size
	epoch_step_val = num_val // batch_size

	if epoch_step == 0 or epoch_step_val == 0:
		raise ValueError("The dataset is too small, please expand the dataset.")

	optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

	train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
	val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
	gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
					 drop_last=True, collate_fn=frcnn_dataset_collate)
	gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
						 drop_last=True, collate_fn=frcnn_dataset_collate)

	if Freeze_Train:
		for param in model.extractor.parameters():
			param.requires_grad = False

	# freeze the BN
	model.freeze_bn()

	train_util = FasterRCNNTrainer(model, optimizer)

	for epoch in range(start_epoch, end_epoch):
		results = fit_one_epoch(model, train_util, optimizer, epoch, epoch_step, epoch_step_val, gen,
								gen_val, end_epoch, Cuda, backbone)

		writer.add_scalar('loss/rpn_cls_loss', results[0], epoch + 1)
		writer.add_scalar('loss/rpn_loc_loss', results[1], epoch + 1)
		writer.add_scalar('loss/roi_cls_loss', results[2], epoch + 1)
		writer.add_scalar('loss/roi_loc_loss', results[3], epoch + 1)
		writer.add_scalar('loss/train_loss', results[4], epoch + 1)
		writer.add_scalar('loss/val_loss', results[5], epoch + 1)
		writer.add_scalar('evaluation/lr', results[6], epoch + 1)
		writer.add_scalar('evaluation/mAP', results[7], epoch + 1)
		writer.add_scalar('evaluation/mIOU', results[8], epoch + 1)
		writer.add_scalar('evaluation/acc', results[9], epoch + 1)

		lr_scheduler.step()

if True:
	batch_size = Unfreeze_batch_size
	lr = Unfreeze_lr
	start_epoch = Freeze_Epoch
	end_epoch = UnFreeze_Epoch

	epoch_step = num_train // batch_size
	epoch_step_val = num_val // batch_size

	if epoch_step == 0 or epoch_step_val == 0:
		raise ValueError("The dataset is too small, please expand the dataset.")

	optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

	train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
	val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
	gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
					 drop_last=True, collate_fn=frcnn_dataset_collate)
	gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
						 drop_last=True, collate_fn=frcnn_dataset_collate)

	if Freeze_Train:
		for param in model.extractor.parameters():
			param.requires_grad = True

	# freeze the BN
	model.freeze_bn()

	train_util = FasterRCNNTrainer(model, optimizer)

	for epoch in range(start_epoch, end_epoch):
		results = fit_one_epoch(model, train_util, optimizer, epoch, epoch_step, epoch_step_val, gen,
								gen_val, end_epoch, Cuda, backbone)

		writer.add_scalar('loss/rpn_cls_loss', results[0], epoch + 1)
		writer.add_scalar('loss/rpn_loc_loss', results[1], epoch + 1)
		writer.add_scalar('loss/roi_cls_loss', results[2], epoch + 1)
		writer.add_scalar('loss/roi_loc_loss', results[3], epoch + 1)
		writer.add_scalar('loss/train_loss', results[4], epoch + 1)
		writer.add_scalar('loss/val_loss', results[5], epoch + 1)
		writer.add_scalar('evaluation/lr', results[6], epoch + 1)
		writer.add_scalar('evaluation/mAP', results[7], epoch + 1)
		writer.add_scalar('evaluation/mIOU', results[8], epoch + 1)
		writer.add_scalar('evaluation/acc', results[9], epoch + 1)

		lr_scheduler.step()

	writer.close()
