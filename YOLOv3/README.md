## <div align="center">Introduction</div>
This project is aimed at training YOLOv3 on the VOC dataset. As for the code, we mainly use the code given by ultralytics and modify some parts of it. For example, we add the function to calculate mIoU(mean interaction over union) and accuracy, which are of vital importance in the object detecting field. In the following parts, I will show you a quick start(or a recommended way to train) and specific details concerning the training and testing process.
## <div align="center">Dataset</div>
In this experiment, we use VOC2007(both train data and validation data) to train our model, use VOC2007(test data) to validate, and use VOC2012(only validation data) to test our model. If you follow the following training steps, **you don't need to download the dataset by yourself**. In the training step, the needed dataset will be automatically downloaded from offcial website of VOC. Nevertheless, you can modify the split of train-validation-test set in files "voc.yaml" and "voc_2012test.yaml" in "data" directory.
## <div align="center">A quick start(Strongly recommended)</div>
PS: We have used tensorboard to visualize our results but we **only carry out tensorborad on colab**. Hence you may not get the visualization results provided by tensorboard on your own computer. We will offer you a more common way to visualize your results on your own computer

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mskmei/MIDTERM-PROJECT-CV-2022Spring/blob/main/YOLOv3/yolov3_pytorch_cv.ipynb)  
<details open>
 <summary>Train</summary>   
As we implement the ed training and test process on colab with free GPU, we strongly recommend you get a quick start on colab. Please click the "open in colab" button above, and you can have a view of the training and validation process quickly.(The specific guidance for how to train on colab can be seen after you click the button)

 Besides, you can also open the file "yolov3_pytorch_cv.ipynb" in this directory. The file provides you with the results we got after training and validating on the VOC dataset.
</details>

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mskmei/MIDTERM-PROJECT-CV-2022Spring/blob/main/YOLOv3/yolov3_test.ipynb) 
 
<details  open>
 <summary>Test</summary>   
The test process is similar to the training process, click the button above to get access to more specific details concerning the test process. You can also open the file "yolov3_test.ipynb" to have an overview first.
 
 As for the test process, we simply download the pre-trained model which is produced in the training process. The model was stored in my Google Drive and anyone can download it with the link below:
 
* [YOLOv3 pretrained Model](https://drive.google.com/file/d/1Pglfi0Y8poLzsEPrNgKb0GYXpVjwKtn1/view?usp=sharing)
</details>

## <div align="center">Train on own computer</div>
I still have to highlight that as we carried out the experiment smoothly on colab, we regard it as a better way to carry out the experiment in colab. However, we also make sure that the program runs properly on our computers.

<details open>
<summary>install</summary>
To start with, you need to get access to our files. You can try either

```bash
git clone https://github.com/mskmei/MIDTERM-PROJECT-CV-2022Spring.git
cd MIDTERM-PROJECT-CV-2022Spring/YOLOv3
```
or download our project as a zip and unzip it locally and turn to "YOLOv3" directory, you may try order like this:
 
```bash
cd path/to/YOLOv3
```
 
To construct the environment, you simply need to install the package in requirements.txt or just:
```bash
pip install -r requirements.txt
```
</details>

<details open>
<summary>train</summary>
Make sure you have installed packages listed in requirements.txt and you already cd into the directory "YOLOv3". To start running the process, you need to run:
 
```bash
python train.py --batch 32 --weights yolov3.pt --data voc.yaml  --epochs 75 --img 416  --project VOC --name 'yolov3' --cache --hyp hyp.VOC.yaml 
```
 
If you have a wandb account, you can log in and visualize the results through wandb online. If you want to visualize your results on tensorboard, you may follow the steps in the part "A quick start".
 
Furthermore, we offer a relatively simple method to visualize results. You can use the function in plot_results(the function was adapted from ultralytics) like:
 
```python
from utils.plots import plot_results
import matplotlib.pyplot as plt
path_to_result = '/content/yolov3/VOC/yolov3/results.csv'
plot_results(path_to_result)  # plot 'results.csv' as 'results.png'
from google.colab.patches import cv2_imshow,cv2
img=cv2.imread("/content/yolov3/VOC/yolov3/results.png")
cv2_imshow(img)
```
Ideally, it will be similar to the image below:

![image](https://raw.githubusercontent.com/mskmei/MIDTERM-PROJECT-CV-2022Spring/main/YOLOv3/results.png)
 
If you want to see more detailed results(such as results in tensorboard and analysis of the results), please turn to our experiment report.
</details>

<details open>
<summary>test</summary>
In this process, we will use pre-trained weights to test our model. First, prepare the environment like the training process. Then you need to download pre-trained weights through this link:
 
* [YOLOv3 pretrained Model](https://drive.google.com/file/d/1Pglfi0Y8poLzsEPrNgKb0GYXpVjwKtn1/view?usp=sharing)

After cd into directorty "YOLOv3", just try:
```bash
python val.py --weights /path/to/best.pt --data voc_2012test.yaml --img 416 --iou 0.5
```
It's almost the same operation as the validation process, but the test process operates on different datasets (no overlap). It may take some time to download the dataset if you run this for the first time.
</details>
<details open>
<summary>inference</summary>
This part can help you detect and visualize detected results. You need to download pre-trained weights first. Then simply put images into your input directory(which is set as data/images by default). Run the following order:
 
```bash
python detect.py --weights /path/to/best.pt --img 416 --conf 0.25 --source data/images
```
 
The results will be saved into runs/detect/exp. Ideally, the results will be similar to the images below.
![image](https://raw.githubusercontent.com/mskmei/MIDTERM-PROJECT-CV-2022Spring/main/YOLOv3/person.jpg)
