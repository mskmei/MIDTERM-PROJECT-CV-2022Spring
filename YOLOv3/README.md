## <div align="center">Introduction</div>
This project is aimed at training YOLOv3 on the VOC dataset. As for the code, we mainly use the code given by ultralytics and modify some parts of it. For example, we add the function to calculate mIoU(mean interaction over union) and accuracy, which are of vital importance in the object detecting field. In the following parts, I will show you a quick start(or a recommended way to train) and specific details concerning the training and testing process.
## <div align="center">Dataset</div>
In this experiment, we use VOC2007(both train data and validation data) to train our model, use VOC2007(test data) to validate, and use VOC2012(only validation data) to test our model. If you want to change the data used for each part, you can manually modify the file "voc.yaml" and "voc_2012test.yaml" in "data" directory.
## <div align="center">A quick start(Strongly recommended)</div>
PS: We have used tensorboard to visualize our results but we **only carry out tensorborad on colab** . Hence you may not get the visualization results provided by tensorboard on your own computer. We will offer you a more common way to visualize your results on your own computer

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
 
 https://drive.google.com/file/d/1Pglfi0Y8poLzsEPrNgKb0GYXpVjwKtn1/view?usp=sharing
</details>

## <div align="center">Train on own computer</div>
I still have to highlight that as we carried out the experiment smoothly on colab, we regard it as a better way to carry out the experiment in colab. However, we also make sure that the program runs properly on our computers.

<details open>
<summary>install</summary>
To start with, you need to get access to our files. You can try either

```bash
git clone https://github.com/mskmei/yolov3.git
cd YOLOv3
```
or download our project as a zip and unzip it locally and turn to YOLOv3 directory, you may try order like this:
 
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
