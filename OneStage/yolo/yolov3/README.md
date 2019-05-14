# YOLOv3
### Download the source code
    git clone https://github.com/pjreddie/darknet
    cd darknet
    
    vim Makefile
    ...
    GPU=1
    CUDNN=1
    NVCC=/usr/local/cuda-9.0/bin/nvcc
    OPENCV=1
    
    make
### Download pre-training weights
    wget https://pjreddie.com/media/files/yolov3.weights
### Pre-trainnig model testing
    ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
    
***
# Train a model on own images
## 1. Image labeling 
#### LabelImg is a graphical image annotation tool - [labelImg](https://github.com/tzutalin/labelImg)
__Ubuntu Linux__
Python 3 + Qt5   
    
    git clone https://github.com/tzutalin/labelImg.git
    sudo apt-get install pyqt5-dev-tools
    sudo pip3 install -r requirements/requirements-linux-python3.txt
    make qt5py3
    cd labelImg
    python3 labelImg.py
    python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

* __[JPEGImages](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/JPEGImages) [Put all img in this folder]__
* __[Annotations](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/Annotations) [Put all labeled .xml file in this folder]__
* __[labels](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/labels) [Put all labeled .txt file in this folder]__

## 2. Make .txt file

* train.txt:写入用于训练图片的名字，每行一个名字（不带后缀.jpg）。

* val.txt:写入用于验证图片的名字，每行一个名字（不带后缀.jpg）。


__Run voc_label.py can get below file__

* obj_train.txt:写入用于训练图片的绝对路径，每行一个路径。

* obj_val.txt:写入用于验证图片的绝对路径，每行一个路径。

## 3. Make .names .cgf and .data file 
* __.names [classes name]__
*data folder voc.names*

      people 
      fire_extinguisher
      fireplug
      car
      bicycle
      motorcycle  

* __.data__ 
*cfg folder voc.data*
     
      classes= 6  #类别数
      train = /home/cai/Desktop/yolo_dataset/objectdetection/object_train.txt #obj_train.txt路径
      valid = /home/cai/Desktop/yolo_dataset/objectdetection/object_val.txt  #obj_val.txt路径
      names = /home/cai/Desktop/yolo_dataset/objectdetection/yolo3_object.names #obj_voc.names路径
      backup = /home/cai/Desktop/yolo_dataset/objectdetection/backup/ #建一个backup文件夹用于存放weights结果
 
 * __.cgf__
 *cfg folder yolov3-voc.cfg - __[yolov3-voc.cfg](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolov3/yolov3-voc.cfg)__*
       
       [convolutional]
       ...
       filters = 3*(classes + 5) #修改filters数量
       [yolo]
       ...
       classes=5 #修改类别数
       [具体修改可见cfg文件]
       
## 4. Download pre-taining weights
    wget https://pjreddie.com/media/files/darknet53.conv.74
## 5. Training
    ./darknet detector train obj_detect/obj_voc.data obj_detect/yolov3-voc.cfg darknet53.conv.74 2>1 | tee visualization/train_yolov3.log 

#### visualization log
    
    python3 extract_log.py
    python3 visualization_loss.py
    python3 visualization_iou.py

## 6. Testing
### ImgTesting
    ./darknet detector test ./obj_detect/obj_voc.data ./obj_detect/yolov3-voc.cfg ./obj_detect/backup/yolov3-voc_30000.weights ./obj_detect/test_data/test_img.jpg
### VideoTesting
    ./darknet detector demo ./obj_detect/obj_voc.data ./obj_detect/yolov3-voc.cfg ./obj_detect/backup/yolov3-voc_30000.weights ./obj_detect/test_data/obj_test.mp4
    