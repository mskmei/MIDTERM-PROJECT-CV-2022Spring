import glob
import json
import math
import operator
import os
import shutil
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch


def box_iou(box1, box2):
	"""
	Return intersection-over-union (Jaccard index) of boxes.
	Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
	Arguments:
		box1 (Tensor[N, 4])
		box2 (Tensor[M, 4])
	Returns:
		iou (Tensor[N, M]): the NxM matrix containing the pairwise
			IoU values for every element in boxes1 and boxes2
	"""

	def box_area(box):
		# box = 4xn
		return (box[2] - box[0]) * (box[3] - box[1])

	area1 = box_area(box1.T)
	area2 = box_area(box2.T)

	# inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
	inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
	return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def log_average_miss_rate(precision, fp_cumsum, num_images):
	"""
		log-average miss rate:
			Calculated by averaging miss rates at 9 evenly spaced FPPI points
			between 10e-2 and 10e0, in log-space.

		output:
				lamr | log-average miss rate
				mr | miss rate
				fppi | false positives per image

		references:
			[1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
			   State of the Art." Pattern Analysis and Machine Intelligence, IEEE
			   Transactions on 34.4 (2012): 743 - 761.
	"""

	if precision.size == 0:
		lamr = 0
		mr = 1
		fppi = 0
		return lamr, mr, fppi

	fppi = fp_cumsum / float(num_images)
	mr = (1 - precision)

	fppi_tmp = np.insert(fppi, 0, -1.0)
	mr_tmp = np.insert(mr, 0, 1.0)

	ref = np.logspace(-2.0, 0.0, num=9)
	for i, ref_i in enumerate(ref):
		j = np.where(fppi_tmp <= ref_i)[-1][-1]
		ref[i] = mr_tmp[j]

	lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

	return lamr, mr, fppi


"""
 throw error and exit
"""


def error(msg):
	print(msg)
	sys.exit(0)


"""
 check if the number is a float between 0.0 and 1.0
"""


def is_float_between_0_and_1(value):
	try:
		val = float(value)
		if val > 0.0 and val < 1.0:
			return True
		else:
			return False
	except ValueError:
		return False


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""


def voc_ap(rec, prec):
	"""
	--- Official matlab code VOC2012---
	mrec=[0 ; rec ; 1];
	mpre=[0 ; prec ; 0];
	for i=numel(mpre)-1:-1:1
			mpre(i)=max(mpre(i),mpre(i+1));
	end
	i=find(mrec(2:end)~=mrec(1:end-1))+1;
	ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
	"""
	rec.insert(0, 0.0)  # insert 0.0 at begining of list
	rec.append(1.0)  # insert 1.0 at end of list
	mrec = rec[:]
	prec.insert(0, 0.0)  # insert 0.0 at begining of list
	prec.append(0.0)  # insert 0.0 at end of list
	mpre = prec[:]
	"""
	 This part makes the precision monotonically decreasing
		(goes from the end to the beginning)
		matlab: for i=numel(mpre)-1:-1:1
					mpre(i)=max(mpre(i),mpre(i+1));
	"""
	for i in range(len(mpre) - 2, -1, -1):
		mpre[i] = max(mpre[i], mpre[i + 1])
	"""
	 This part creates a list of indexes where the recall changes
		matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
	"""
	i_list = []
	for i in range(1, len(mrec)):
		if mrec[i] != mrec[i - 1]:
			i_list.append(i)  # if it was matlab would be i + 1
	"""
	 The Average Precision (AP) is the area under the curve
		(numerical integration)
		matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
	"""
	ap = 0.0
	for i in i_list:
		ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
	return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""


def file_lines_to_list(path):
	# open txt file lines to a list
	with open(path) as f:
		content = f.readlines()
	# remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content]
	return content


"""
 Draws text in image
"""


def draw_text_in_image(img, text, pos, color, line_width):
	font = cv2.FONT_HERSHEY_PLAIN
	fontScale = 1
	lineType = 1
	bottomLeftCornerOfText = pos
	cv2.putText(img, text,
				bottomLeftCornerOfText,
				font,
				fontScale,
				color,
				lineType)
	text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
	return img, (line_width + text_width)


"""
 Plot - adjust axes
"""


def adjust_axes(r, t, fig, axes):
	# get text width for re-scaling
	bb = t.get_window_extent(renderer=r)
	text_width_inches = bb.width / fig.dpi
	# get axis width in inches
	current_fig_width = fig.get_figwidth()
	new_fig_width = current_fig_width + text_width_inches
	propotion = new_fig_width / current_fig_width
	# get axis limit
	x_lim = axes.get_xlim()
	axes.set_xlim([x_lim[0], x_lim[1] * propotion])


"""
 Draw plot using Matplotlib
"""


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
				   true_p_bar):
	# sort the dictionary by decreasing value, into a list of tuples
	sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
	# unpacking the list of tuples into two lists
	sorted_keys, sorted_values = zip(*sorted_dic_by_value)
	#
	if true_p_bar != "":
		"""
		 Special case to draw in:
			- green -> TP: True Positives (object detected and matches ground-truth)
			- red -> FP: False Positives (object detected but does not match ground-truth)
			- orange -> FN: False Negatives (object not detected but present in the ground-truth)
		"""
		fp_sorted = []
		tp_sorted = []
		for key in sorted_keys:
			fp_sorted.append(dictionary[key] - true_p_bar[key])
			tp_sorted.append(true_p_bar[key])
		plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
		plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
				 left=fp_sorted)
		# add legend
		plt.legend(loc='lower right')
		"""
		 Write number on side of bar
		"""
		fig = plt.gcf()  # gcf - get current figure
		axes = plt.gca()
		r = fig.canvas.get_renderer()
		for i, val in enumerate(sorted_values):
			fp_val = fp_sorted[i]
			tp_val = tp_sorted[i]
			fp_str_val = " " + str(fp_val)
			tp_str_val = fp_str_val + " " + str(tp_val)
			# trick to paint multicolor with offset:
			# first paint everything and then repaint the first number
			t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
			plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
			if i == (len(sorted_values) - 1):  # largest bar
				adjust_axes(r, t, fig, axes)
	else:
		plt.barh(range(n_classes), sorted_values, color=plot_color)
		"""
		 Write number on side of bar
		"""
		fig = plt.gcf()  # gcf - get current figure
		axes = plt.gca()
		r = fig.canvas.get_renderer()
		for i, val in enumerate(sorted_values):
			str_val = " " + str(val)  # add a space before
			if val < 1.0:
				str_val = " {0:.2f}".format(val)
			t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
			# re-set axes to show number inside the figure
			if i == (len(sorted_values) - 1):  # largest bar
				adjust_axes(r, t, fig, axes)
	# set window title
	fig.canvas.set_window_title(window_title)
	# write classes in y axis
	tick_font_size = 12
	plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
	"""
	 Re-scale height accordingly
	"""
	init_height = fig.get_figheight()
	# comput the matrix height in points and inches
	dpi = fig.dpi
	height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
	height_in = height_pt / dpi
	# compute the required figure height
	top_margin = 0.15  # in percentage of the figure height
	bottom_margin = 0.05  # in percentage of the figure height
	figure_height = height_in / (1 - top_margin - bottom_margin)
	# set new height
	if figure_height > init_height:
		fig.set_figheight(figure_height)

	# set plot title
	plt.title(plot_title, fontsize=14)
	# set axis titles
	# plt.xlabel('classes')
	plt.xlabel(x_label, fontsize='large')
	# adjust size of window
	fig.tight_layout()
	# save the plot
	fig.savefig(output_path)
	# show image
	if to_show:
		plt.show()
	# close the plot
	plt.close()


def get_map(MINOVERLAP, draw_plot, path='./map_out'):
	GT_PATH = os.path.join(path, 'ground-truth')
	DR_PATH = os.path.join(path, 'detection-results')
	IMG_PATH = os.path.join(path, 'images-optional')
	TEMP_FILES_PATH = os.path.join(path, '.temp_files')
	RESULTS_FILES_PATH = os.path.join(path, 'results')

	show_animation = False
	if os.path.exists(IMG_PATH):
		for dirpath, dirnames, files in os.walk(IMG_PATH):
			if not files:
				show_animation = False
	else:
		show_animation = False

	if not os.path.exists(TEMP_FILES_PATH):
		os.makedirs(TEMP_FILES_PATH)

	if os.path.exists(RESULTS_FILES_PATH):
		shutil.rmtree(RESULTS_FILES_PATH)

	if draw_plot:
		os.makedirs(os.path.join(RESULTS_FILES_PATH, "AP"))
		os.makedirs(os.path.join(RESULTS_FILES_PATH, "F1"))
		os.makedirs(os.path.join(RESULTS_FILES_PATH, "Recall"))
		os.makedirs(os.path.join(RESULTS_FILES_PATH, "Precision"))
	if show_animation:
		os.makedirs(os.path.join(RESULTS_FILES_PATH, "images", "detections_one_by_one"))

	ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
	if len(ground_truth_files_list) == 0:
		error("Error: No ground-truth files found!")
	ground_truth_files_list.sort()
	gt_counter_per_class = {}  # the number of objects for each classes. (TP + FN)
	counter_images_per_class = {}

	for txt_file in ground_truth_files_list:
		file_id = txt_file.split(".txt", 1)[0]
		file_id = os.path.basename(os.path.normpath(file_id))
		# os.path.basename returns the last filename in the path.
		# os.path.normpath is used to normalize the specified path.
		temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
		if not os.path.exists(temp_path):
			error_msg = "Error. File not found: {}\n".format(temp_path)
			error(error_msg)
		lines_list = file_lines_to_list(txt_file)  # get the info of the ground-truth from .txt to list.
		bounding_boxes = []
		is_difficult = False
		already_seen_classes = []  # just for an image.
		for line in lines_list:
			try:
				if "difficult" in line:
					class_name, left, top, right, bottom, _difficult = line.split()
					is_difficult = True
				else:
					class_name, left, top, right, bottom = line.split()
			except:
				if "difficult" in line:
					line_split = line.split()
					_difficult = line_split[-1]
					bottom = line_split[-2]
					right = line_split[-3]
					top = line_split[-4]
					left = line_split[-5]
					class_name = ""
					for name in line_split[:-5]:
						class_name += name + " "
					class_name = class_name[:-1]
					is_difficult = True
				else:
					line_split = line.split()
					bottom = line_split[-1]
					right = line_split[-2]
					top = line_split[-3]
					left = line_split[-4]
					class_name = ""
					for name in line_split[:-4]:
						class_name += name + " "
					class_name = class_name[:-1]

			bbox = left + " " + top + " " + right + " " + bottom
			if is_difficult:
				bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
				is_difficult = False
			else:
				bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
				if class_name in gt_counter_per_class:
					gt_counter_per_class[class_name] += 1
				else:
					gt_counter_per_class[class_name] = 1

				if class_name not in already_seen_classes:  # just for an image.
					if class_name in counter_images_per_class:
						counter_images_per_class[class_name] += 1
					else:
						counter_images_per_class[class_name] = 1
					already_seen_classes.append(class_name)
		# count the already seen classes in an image.

		with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
			json.dump(bounding_boxes, outfile)
	# dump all the objects in an image.

	gt_classes = list(gt_counter_per_class.keys())  # count the number of objects in all images in
	gt_classes = sorted(gt_classes)
	n_classes = len(gt_classes)  # number of classes in the ground-truth (20).
	confusion_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

	# get the information of the detection results.
	dr_files_list = glob.glob(DR_PATH + '/*.txt')
	dr_files_list.sort()

	iou_thres = 0.5
	nc = 20

	for txt_file in dr_files_list:
		file_id = txt_file.split('.txt', 1)[0]
		file_id = os.path.basename(os.path.normpath(file_id))
		gt_path = os.path.join(GT_PATH, (file_id + '.txt'))
		detections = []
		labels = []
		lines = file_lines_to_list(txt_file)
		for line in lines:
			try:
				tmp_class_name, confidence, left, top, right, bottom = line.split()  # the predicted results.
			except:
				line_split = line.split()
				bottom = line_split[4]
				right = line_split[3]
				top = line_split[2]
				left = line_split[1]
				confidence = line_split[-5]
				tmp_class_name = ""
				for name in line_split[:-5]:
					tmp_class_name += name + " "
				tmp_class_name = tmp_class_name[:-1]

			if tmp_class_name in gt_classes:
				tmp_class_id = gt_classes.index(tmp_class_name)
				detections.append([int(left), int(top), int(right), int(bottom), float(confidence), tmp_class_id])

		lines = file_lines_to_list(gt_path)
		for line in lines:
			try:
				tmp_class_name, left, top, right, bottom = line.split()  # the predicted results.
			except:
				line_split = line.split()
				bottom = line_split[4]
				right = line_split[3]
				top = line_split[2]
				left = line_split[1]
				tmp_class_name = ""
				for name in line_split[:-5]:
					tmp_class_name += name + " "
				tmp_class_name = tmp_class_name[:-1]

			if tmp_class_name in gt_classes:
				tmp_class_id = gt_classes.index(tmp_class_name)
				labels.append([tmp_class_id, int(left), int(top), int(right), int(bottom)])
		if len(labels) > 0 and len(detections) > 0:
			detections = torch.Tensor(detections)
			labels = torch.Tensor(labels)
			gtClasses = labels[:, 0].int()
			detectionClasses = detections[:, 5].int()
			iou = box_iou(labels[:, 1:], detections[:, :4])

			x = torch.where(iou > iou_thres)
			if x[0].shape[0]:
				matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
				if x[0].shape[0] > 1:
					matches = matches[matches[:, 2].argsort()[::-1]]
					matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
					matches = matches[matches[:, 2].argsort()[::-1]]
					matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
			else:
				matches = np.zeros((0, 3))

			n = matches.shape[0] > 0
			m0, m1, _ = matches.transpose().astype(np.int16)
			for i, gc in enumerate(gtClasses):
				j = m0 == i
				if n and sum(j) == 1:
					confusion_matrix[detectionClasses[m1[j]], gc] += 1  # correct
				else:
					confusion_matrix[nc, gc] += 1  # background FP

			if n:
				for i, dc in enumerate(detectionClasses):
					if not any(m1 == i):
						confusion_matrix[dc, nc] += 1  # background FN

	for class_index, class_name in enumerate(gt_classes):  # count the detection results for each class.
		bounding_boxes = []  # belongs to a single class.
		for txt_file in dr_files_list:
			file_id = txt_file.split(".txt", 1)[0]
			file_id = os.path.basename(os.path.normpath(file_id))  # file_id is of the detection results.
			temp_path = os.path.join(GT_PATH, (file_id + ".txt"))  # find the related ground-truth.
			if class_index == 0:
				if not os.path.exists(temp_path):
					error_msg = "Error. File not found: {}\n".format(temp_path)
					error(error_msg)
			lines = file_lines_to_list(txt_file)  # get the info of the detection results.
			for line_id, line in enumerate(lines):
				# for line in lines:
				try:
					tmp_class_name, confidence, left, top, right, bottom = line.split()  # the predicted results.
				except:
					line_split = line.split()
					bottom = line_split[-1]
					right = line_split[-2]
					top = line_split[-3]
					left = line_split[-4]
					confidence = line_split[-5]
					tmp_class_name = ""
					for name in line_split[:-5]:
						tmp_class_name += name + " "
					tmp_class_name = tmp_class_name[:-1]

				if tmp_class_name == class_name:
					bbox = left + " " + top + " " + right + " " + bottom  # the predicted.
					bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})

		# sort by confidence in descending order.
		bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)

		# write the detection results of a single class into the temp file.
		with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
			json.dump(bounding_boxes, outfile)

	for txt_file in dr_files_list:
		file_id = txt_file.split('.txt', 1)[0]
		file_id = os.path.basename(os.path.normpath(file_id))
		tmp_path = os.path.join(DR_PATH, (file_id + '.txt'))
		lines = file_lines_to_list(txt_file)
		for line_id, line in enumerate(lines):
			try:
				tmp_class_name, confidence, left, top, right, bottom = line.split()  # the predicted results.
			# if mask[file_id][line_id] == 0:
			# confusion_matrix[-1][gt_classes.index(tmp_class_name)] += 1
			except:
				line_split = line.split()
				bottom = line_split[-1]
				right = line_split[-2]
				top = line_split[-3]
				left = line_split[-4]
				confidence = line_split[-5]
				tmp_class_name = ""
				for name in line_split[:-5]:
					tmp_class_name += name + " "
				tmp_class_name = tmp_class_name[:-1]

	# pprint.pprint(confusion_matrix)

	confusion_matrix = np.array([t[:20] for t in confusion_matrix[:20]])
	# pprint.pprint(confusion_matrix)

	detection_sum = np.sum(confusion_matrix, axis=0)
	ground_truth_sum = np.sum(confusion_matrix, axis=1)
	intersection = np.diag(confusion_matrix)

	IOU = intersection / (detection_sum + ground_truth_sum - intersection)
	mIOU = np.nanmean(IOU)

	acc = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

	sum_AP = 0.0
	ap_dictionary = {}
	lamr_dictionary = {}

	with open(path + "/results.txt", 'w') as results_file:
		count_true_positives = {}  # contains the TP of each class.

		for class_index, class_name in enumerate(gt_classes):
			count_true_positives[class_name] = 0
			dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"  # open the statistical info of the detection results.
			dr_data = json.load(open(dr_file))

			nd = len(dr_data)  # number of detection results.
			tp = [0] * nd
			fp = [0] * nd
			score = [0] * nd
			for idx, detection in enumerate(dr_data):  # judge the detection results with all classes.
				file_id = detection["file_id"]
				score[idx] = float(detection["confidence"])
				if score[idx] > 0.5:
					score05_idx = idx

				gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"  # get the ground truth info.
				ground_truth_data = json.load(open(gt_file))
				ovmax = -1
				gt_match = -1
				bb = [float(x) for x in detection["bbox"].split()]  # bounding box of the detection results.
				for obj in ground_truth_data:  # the ground truth and the detection results are corresponding.
					if obj["class_name"] == class_name:  # there are several objects in the ground truth.
						bbgt = [float(x) for x in obj["bbox"].split()]  # the bounding box of the ground truth.
						bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
						iw = bi[2] - bi[0] + 1
						ih = bi[3] - bi[1] + 1
						if iw > 0 and ih > 0:
							ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
																			  + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
							ov = iw * ih / ua
							if ov > ovmax:  # find the highest iou with the objects in the gt.
								ovmax = ov
								gt_match = obj  # record the 'best' object in the gt.

				min_overlap = MINOVERLAP
				if ovmax >= min_overlap:  # min_overlap=0.5 by default.
					if "difficult" not in gt_match:
						if not bool(gt_match["used"]):  # 'used'=False from the beginning.
							tp[idx] = 1
							gt_match["used"] = True  # the object is used only once.
							count_true_positives[class_name] += 1
							with open(gt_file, 'w') as f:
								f.write(json.dumps(
									ground_truth_data))  # the changed state of 'used' is written into the file.
							if show_animation:
								status = "MATCH!"
						else:
							fp[idx] = 1
							if show_animation:
								status = "REPEATED MATCH!"
				else:
					fp[idx] = 1
					if ovmax > 0:
						status = "INSUFFICIENT OVERLAP"

			cumsum = 0
			for idx, val in enumerate(fp):
				fp[idx] += cumsum
				cumsum += val

			cumsum = 0
			for idx, val in enumerate(tp):
				tp[idx] += cumsum
				cumsum += val

			rec = tp[:]
			for idx, val in enumerate(tp):
				rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)  # (TP / (TP + FN))

			prec = tp[:]
			for idx, val in enumerate(tp):
				prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)  # (TP / (TP + FP))

			ap, mrec, mprec = voc_ap(rec[:], prec[:])
			sum_AP += ap
			ap_dictionary[class_name] = ap

			n_images = counter_images_per_class[class_name]
			lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
			lamr_dictionary[class_name] = lamr

		mAP = sum_AP / n_classes

	shutil.rmtree(TEMP_FILES_PATH)

	return mAP, mIOU, acc


def preprocess_gt(gt_path, class_names):
	image_ids = os.listdir(gt_path)
	results = {}

	images = []
	bboxes = []
	for i, image_id in enumerate(image_ids):
		lines_list = file_lines_to_list(os.path.join(gt_path, image_id))
		boxes_per_image = []
		image = {}
		image_id = os.path.splitext(image_id)[0]
		image['file_name'] = image_id + '.jpg'
		image['width'] = 1
		image['height'] = 1
		image['id'] = str(image_id)

		for line in lines_list:
			difficult = 0
			if "difficult" in line:
				line_split = line.split()
				left, top, right, bottom, _difficult = line_split[-5:]
				class_name = ""
				for name in line_split[:-5]:
					class_name += name + " "
				class_name = class_name[:-1]
				difficult = 1
			else:
				line_split = line.split()
				left, top, right, bottom = line_split[-4:]
				class_name = ""
				for name in line_split[:-4]:
					class_name += name + " "
				class_name = class_name[:-1]

			left, top, right, bottom = float(left), float(top), float(right), float(bottom)
			cls_id = class_names.index(class_name) + 1
			bbox = [left, top, right - left, bottom - top, difficult, str(image_id), cls_id,
					(right - left) * (bottom - top) - 10.0]
			boxes_per_image.append(bbox)
		images.append(image)
		bboxes.extend(boxes_per_image)
	results['images'] = images

	categories = []
	for i, cls in enumerate(class_names):
		category = {}
		category['supercategory'] = cls
		category['name'] = cls
		category['id'] = i + 1
		categories.append(category)
	results['categories'] = categories

	annotations = []
	for i, box in enumerate(bboxes):
		annotation = {}
		annotation['area'] = box[-1]
		annotation['category_id'] = box[-2]
		annotation['image_id'] = box[-3]
		annotation['iscrowd'] = box[-4]
		annotation['bbox'] = box[:4]
		annotation['id'] = i
		annotations.append(annotation)
	results['annotations'] = annotations
	return results


def preprocess_dr(dr_path, class_names):
	image_ids = os.listdir(dr_path)
	results = []
	for image_id in image_ids:
		lines_list = file_lines_to_list(os.path.join(dr_path, image_id))
		image_id = os.path.splitext(image_id)[0]
		for line in lines_list:
			line_split = line.split()
			confidence, left, top, right, bottom = line_split[-5:]
			class_name = ""
			for name in line_split[:-5]:
				class_name += name + " "
			class_name = class_name[:-1]
			left, top, right, bottom = float(left), float(top), float(right), float(bottom)
			result = {}
			result["image_id"] = str(image_id)
			result["category_id"] = class_names.index(class_name) + 1
			result["bbox"] = [left, top, right - left, bottom - top]
			result["score"] = float(confidence)
			results.append(result)
	return results


def get_coco_map(class_names, path):
	from pycocotools.coco import COCO
	from pycocotools.cocoeval import COCOeval

	GT_PATH = os.path.join(path, 'ground-truth')
	DR_PATH = os.path.join(path, 'detection-results')
	COCO_PATH = os.path.join(path, 'coco_eval')

	if not os.path.exists(COCO_PATH):
		os.makedirs(COCO_PATH)

	GT_JSON_PATH = os.path.join(COCO_PATH, 'instances_gt.json')
	DR_JSON_PATH = os.path.join(COCO_PATH, 'instances_dr.json')

	with open(GT_JSON_PATH, "w") as f:
		results_gt = preprocess_gt(GT_PATH, class_names)
		json.dump(results_gt, f, indent=4)

	with open(DR_JSON_PATH, "w") as f:
		results_dr = preprocess_dr(DR_PATH, class_names)
		json.dump(results_dr, f, indent=4)

	cocoGt = COCO(GT_JSON_PATH)
	cocoDt = cocoGt.loadRes(DR_JSON_PATH)
	cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()
