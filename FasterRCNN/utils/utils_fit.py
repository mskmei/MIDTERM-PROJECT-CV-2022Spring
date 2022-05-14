import torch
from tqdm import tqdm

from utils.utils import get_lr
from evaluation import evaluate


def fit_one_epoch(model, train_util, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
				  cuda, backbone):
	total_loss, rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, val_loss = 0, 0, 0, 0, 0, 0

	print('Start Train')
	with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
		for iteration, batch in enumerate(gen):
			if iteration >= epoch_step:
				break
			images, boxes, labels = batch[0], batch[1], batch[2]
			with torch.no_grad():
				images = torch.from_numpy(images).type(torch.FloatTensor)
				if cuda:
					images = images.cuda()

			rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, boxes, labels, 1)
			total_loss += total.item()
			rpn_loc_loss += rpn_loc.item()
			rpn_cls_loss += rpn_cls.item()
			roi_loc_loss += roi_loc.item()
			roi_cls_loss += roi_cls.item()

			pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1)})
			pbar.update(1)

	print('Finish Train')

	rpn_cls_loss = rpn_cls_loss / len(gen)
	rpn_loc_loss = rpn_loc_loss / len(gen)
	roi_cls_loss = roi_cls_loss / len(gen)
	roi_loc_loss = roi_loc_loss / len(gen)
	total_loss = total_loss / len(gen)
	lr = get_lr(optimizer)

	print('Start Validation')
	with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
		for iteration, batch in enumerate(gen_val):
			if iteration >= epoch_step_val:
				break
			images, boxes, labels = batch[0], batch[1], batch[2]
			with torch.no_grad():
				images = torch.from_numpy(images).type(torch.FloatTensor)
				if cuda:
					images = images.cuda()

				train_util.optimizer.zero_grad()
				_, _, _, _, val_total = train_util.forward(images, boxes, labels, 1)

				val_loss += val_total.item()
				pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
				pbar.update(1)

	print('Finish Validation')
	print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
	print('Train Loss: %.3f || Val Loss: %.3f ' % (total_loss, val_loss / epoch_step_val))

	weights_name = 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss, val_loss / epoch_step_val)

	torch.save(model.state_dict(), weights_name)

	val_loss = val_loss / epoch_step_val

	mAP, mIoU, acc = evaluate(getGT=False, weights_name=weights_name, backbone=backbone, cuda=cuda)

	return rpn_cls_loss, rpn_loc_loss, roi_cls_loss, roi_loc_loss, total_loss, val_loss, lr, mAP, mIoU, acc
