## <div align="center">Introduction</div>
This project is about experimenting data augmentation methods, Cutout, Mixup and CutMix. 

## <div align="center">Dataset</div>
In this experiment, we choose CIFAR-100 dataset to experiment on. It contains 50,000 natural images in the training dataset while 10,000 for testing. Each of them is of size 32 x 32, categorized in one of the 100 classes.

## <div align="center">A quick start</div>
Here we offer two notebooks on different platforms to play around. In AIStudio we implement with paddlepaddle while in Colab we implement with PyTorch. The training is light as a simple illustration of how things work. Make sure to run the notebook in GPU environment!
<div align="center">
[Open In AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/3824197?contributionType=1&shared=1)

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mskmei/MIDTERM-PROJECT-CV-2022Spring/blob/main/CIFAR100/notebook/CIFAR100_torch.ipynb)</div>
<details open>
  
## <div align="center">Train on your own</div>
We have offered two entrances for training, **PaddlePaddle** and **PyTorch**. One can train with either package that he or she is familiar with. 
 
<summary>Install</summary>
To start with, you need to get access to our files. You can try either

```bash
git clone https://github.com/mskmei/MIDTERM-PROJECT-CV-2022Spring.git
cd MIDTERM-PROJECT-CV-2022Spring/CIFAR100
```
or download our project as a zip and unzip it locally and turn to "CIFAR100" directory. Then 
 
<summary>PaddlePaddle</summary>
Dependencies:
1. paddlepaddle-gpu
2. tqdm
3. PIL
4. visualdl
 
<summary>PyTorch</summary>
Dependencies:
1. pytorch 
2. tqdm
3. PIL
4. tensorboard
 
 
<summary>train</summary>
Make sure you have installed packages listed in requirements.txt and you already cd into the directory "YOLOv3". To start running the process, you need to run:
 
```bash
python train.py --batch 32 --weights yolov3.pt --data voc.yaml  --epochs 75 --img 416  --project VOC --name 'yolov3' --cache --hyp hyp.VOC.yaml 
```
 
 
<summary>test</summary>
We have prepared **test.py** for one to easily test a ResNet-18 model trained on 224 x 224 images on CIFAR-100! Just type in 

```bash
python test.py
 ```
 
or manually run the file, and the program will ask for inputting the path to your model. Both **.pdparams** and **.pth** files are supported. Then, it will ask for the index of cuda to run your model on (the process will be omitted if no cuda is available). 
 
 Lastly, wait for a few seconds to obtain the result!
