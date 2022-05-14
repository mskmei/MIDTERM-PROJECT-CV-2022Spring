import os
import xml.etree.ElementTree as ET
import shutil
from PIL import Image
from tqdm import tqdm

from frcnn import FRCNN
from utils.utils import get_classes
from utils.utils_evaluate import get_coco_map, get_map
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--cuda', dest='cuda', help='Whether to use GPU or not', default=True)
parser.add_option('--backbone', dest='backbone', help='The backbone to use in feature extraction.', default='resnet50')
parser.add_option('--weights', dest='weights', help='The trained weights of the model.',
				  default='model_data/resnet50_faster.pth')

(options, args) = parser.parse_args()

weights_name = options.weights
backbone = options.backbone
cuda = options.cuda
# -------------------------------------------------------#
#   此处的classes_path用于指定需要测量VOC_map的类别
#   一般情况下与训练和预测所用的classes_path一致即可
# -------------------------------------------------------#
root_path = '/SSD_DISK/users/yuanjunhao/work'
classes_path = os.path.join(root_path, 'model_data/voc_classes.txt')
# -------------------------------------------------------#
#   MINOVERLAP用于指定想要获得的mAP0.x
#   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
# -------------------------------------------------------#
MINOVERLAP = 0.5
# -------------------------------------------------------#
#   map_vis用于指定是否开启VOC_map计算的可视化
# -------------------------------------------------------#
# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
VOCdevkit_path = os.path.join(root_path, 'VOCdevkit')

# -------------------------------------------------------#
#   结果输出的文件夹，默认为map_out
# -------------------------------------------------------#
map_out_path = os.path.join(root_path, 'map_out_test')
image_ids = open(os.path.join(VOCdevkit_path, "VOC2012/ImageSets/Main/val.txt")).read().strip().split()

if not os.path.exists(map_out_path):
	os.makedirs(map_out_path)
if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
	os.makedirs(os.path.join(map_out_path, 'ground-truth'))
if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
	os.makedirs(os.path.join(map_out_path, 'detection-results'))
if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
	os.makedirs(os.path.join(map_out_path, 'images-optional'))

class_names, _ = get_classes(classes_path)

if True:
	for image_id in tqdm(image_ids):
		with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
			root = ET.parse(os.path.join(VOCdevkit_path, "VOC2012/Annotations/" + image_id + ".xml")).getroot()
			for obj in root.findall('object'):
				difficult_flag = False
				if obj.find('difficult') != None:
					difficult = obj.find('difficult').text
					if int(difficult) == 1:
						difficult_flag = True
				obj_name = obj.find('name').text
				if obj_name not in class_names:
					continue
				bndbox = obj.find('bndbox')
				left = bndbox.find('xmin').text
				top = bndbox.find('ymin').text
				right = bndbox.find('xmax').text
				bottom = bndbox.find('ymax').text

				if difficult_flag:
					new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
				else:
					new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

if True:
	frcnn = FRCNN(confidence=0.05, root=root_path, weights_name=weights_name, backbone=backbone, cuda=cuda)

	for image_id in tqdm(image_ids):
		image_path = os.path.join(VOCdevkit_path, "VOC2012/JPEGImages/" + image_id + ".jpg")
		image = Image.open(image_path)
		frcnn.get_map_txt(image_id, image, class_names, map_out_path)

if True:
	mAP, mIOU, acc = get_map(MINOVERLAP, False, path=map_out_path)
	shutil.rmtree(os.path.join(map_out_path, 'detection-results'), ignore_errors=True)
	shutil.rmtree(os.path.join(map_out_path, 'images-optional'), ignore_errors=True)

	print("mAP = {0:.3f}".format(mAP))
	print('mIOU = {:.3f}'.format(mIOU))
	print('acc = {:.3f}'.format(acc))
	os.remove(os.path.join(map_out_path, 'results.txt'))
