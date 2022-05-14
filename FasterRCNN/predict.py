import time
import cv2
import numpy as np
from PIL import Image
from frcnn import FRCNN
import os
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--img', dest='img', help='The path to the current file.')
parser.add_option('--save', dest='save', help='The path to save the predicted image.')
parser.add_option('--name', dest='name', help='The name of the predicted image.')
parser.add_option('--backbone', dest='backbone', help='The backbone to be used when predicting.', default='resnet50')
parser.add_option('--weights', dest='weights', help='The trained weights of the model.')

(options, args) = parser.parse_args()

root_path = '/SSD_DISK/users/yuanjunhao/work'

if not options.img:
	parser.error('Error:path to the image to predict must be specified. Pass --img to the command line')

img = options.img

if not options.save:
	print('The predicted image will be saved at the same location as the original image.')
	save_path = os.path.dirname(img)
else:
	save_path = options.save

if not options.name:
	print('The predicted result is out.jpg.')
	save_name = 'out.jpg'
else:
	save_name = options.name

if not options.weights:
	parser.error(
		'Error:the trained weights of the model when predicting must be specified. Path --weights to the command line')

weights = options.weights

backbone = options.backbone

frcnn = FRCNN(root=root_path, weights_name=weights, backbone=backbone)

save_name = os.path.join(save_path, save_name)

image = Image.open(img)
r_image = frcnn.detect_image(image)
r_image.save(save_name)
print('The predicted image has been saved.')
