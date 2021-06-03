# 打标签训练测试
目录地址 8000class
### 1. 相关的有用的库文件目录 （git的地址，是否做了修改）
无
### 2. 有用的ipynb, 或 py (training,test case)
8000class/dataset-Copy1.ipynb（训练测试代码）
### 3. 数据文件（/home/data 或 data3）下的某个目录，说明下什么数据集
/home/data2/cr/download_img/explore_word/
下面每个文件夹是一个发现词，文件夹里是相应的图片
### 4. 当前目录的相关配置文件，数据文件的说明。
findwords_1.xlsx
整理后的，删除了部分发现词
dicts_8544.npy
8544个分类id对应的名称
### 5. 说明下部署的时候需要做啥？
将模型文件拷到tf2_mobilenet_no_pruned_dataset_8544目录下，保证文件名相同即可。
### 6. 返回过去的样本数据？
{'words': ['太空人', '宇航员', '公共设施', '入户柜', '虚拟', '柱状图', '微地形景观', '科技界面', '漫画', '海洋动物']}

———————————————————————————————————————

# 空间家具识别训练测试
目录地址 furniture
### 1. 相关的有用的库文件目录 （git的地址，是否做了修改）
无
### 2. 有用的ipynb, 或 py (training,test case)
furniture/tf2_train_first_EfficientNetB0.ipynb（训练测试代码）
训练第一层时，将目录改为/home/data/cr/furniture_pre_train
训练第二层时，将目录改为/home/data/cr/furniture_pre_train单品/
训练第三层时，将目录改为/home/data/cr/furniture_sec_train
### 3. 数据文件（/home/data 或 data3）下的某个目录，说明下什么数据集
/home/data/cr/furniture_pre_train/单品/
/home/data/cr/furniture_pre_train/
训练第三层时，将目录改为/home/data/cr/furniture_sec_train
下面每个文件夹是一个分类，文件夹里是相应的图片
### 4. 当前目录的相关配置文件，数据文件的说明。
/home/data/cr/msyh.ttc
字体文件，测试时用到
### 5. 说明下部署的时候需要做啥？
将模型文件拷到inception_v4_first、inception_v4_second、inception_v4_thirdly目录下，保证文件名相同即可。
### 6. 返回过去的样本数据？
{"type": "单品", "names": [{"label": “椅”, "boxes": []}]} 
{"names":[{"boxes":[0.419,0.061,0.856,0.581],"label":"架"}],"type":"室内场景"}   
{"names":["办公楼"],"type":"户外场景"}  
{"type": "负例", "names": []} 
{"type": "平面图", "names": []} 

———————————————————————————————————————

# 粗分类训练测试
目录地址 image_classify
### 1. 相关的有用的库文件目录 （git的地址，是否做了修改）
无
### 2. 有用的ipynb, 或 py (training,test case)
image_classify/mobilenet.ipynb（训练测试代码）
### 3. 数据文件（/home/data 或 data3）下的某个目录，说明下什么数据集
/home/data2/cr/image_classify
下面每个文件夹是一个分类，文件夹里是相应的图片
### 4. 当前目录的相关配置文件，数据文件的说明。
/home/data/cr/msyh.ttc
字体文件，测试时用到
### 5. 说明下部署的时候需要做啥？
将模型文件拷到tf2_mobilenet目录下，保证文件名相同即可。
### 6. 返回过去的样本数据？
[{"bid": 25459876, "label": ["平面设计"], "image": [{"img_id": 1095040846, "label": "平面设计"}, {"img_id": 1095040809, "label": "平面设计"}, {"img_id": 1095040779, "label": "平面设计"}, {"img_id": 1095024242, "label": "平面设计"}, {"img_id": 1094932218, "label": "平面设计"}, {"img_id": 1094931936, "label": "平面设计"}, {"img_id": 1094931895, "label": "平面设计"}, {"img_id": 1094931517, "label": "平面设计"}, {"img_id": 1093188873, "label": "平面设计"}, {"img_id": 1093188699, "label": "平面设计"}, {"img_id": 1093188669, "label": "平面设计"}, {"img_id": 1093188557, "label": "平面设计"}, {"img_id": 1093188545, "label": "平面设计"}, {"img_id": 1093188529, "label": "平面设计"}, {"img_id": 1093188520, "label": "平面设计"}, {"img_id": 1093188460, "label": "平面设计"}, {"img_id": 1093188415, "label": "平面设计"}, {"img_id": 1093188370, "label": "平面设计"}, {"img_id": 1093188289, "label": "平面设计"}, {"img_id": 1093188212, "label": "平面设计"}, {"img_id": 1093188180, "label": "平面设计"}, {"img_id": 1093188155, "label": "平面设计"}, {"img_id": 1093188153, "label": "平面设计"}, {"img_id": 1093188143, "label": "平面设计"}, {"img_id": 1093188122, "label": "平面设计"}, {"img_id": 1093188104, "label": "平面设计"}, {"img_id": 1093187909, "label": "平面设计"}, {"img_id": 1093187822, "label": "平面设计"}, {"img_id": 1093075560, "label": "平面设计"}, {"img_id": 1093075548, "label": "平面设计"}, {"img_id": 1093075518, "label": "平面设计"}, {"img_id": 1093075471, "label": "平面设计"}, {"img_id": 1093075425, "label": "平面设计"}, {"img_id": 1093053665, "label": "平面设计"}, {"img_id": 1093053594, "label": "平面设计"}, {"img_id": 1093053591, "label": "平面设计"}, {"img_id": 1093053587, "label": "平面设计"}, {"img_id": 1093053324, "label": "平面设计"}, {"img_id": 1093053313, "label": "平面设计"}, {"img_id": 1093053300, "label": "平面设计"}, {"img_id": 1093053279, "label": "平面设计"}, {"img_id": 1093053256, "label": "平面设计"}, {"img_id": 1093013888, "label": "平面设计"}, {"img_id": 1093013883, "label": "平面设计"}, {"img_id": 1093013873, "label": "平面设计"}, {"img_id": 1093013872, "label": "平面设计"}, {"img_id": 1093013866, "label": "平面设计"}, {"img_id": 1093013865, "label": "平面设计"}, {"img_id": 1093013864, "label": "平面设计"}, {"img_id": 1093013863, "label": "平面设计"}], "img_count": 99, "type": 0}]

———————————————————————————————————————

# 鉴黄训练测试
目录地址 
### 1. 相关的有用的库文件目录 （git的地址，是否做了修改）
无
### 2. 有用的ipynb, 或 py (training,test case)
identify-sex/MobileNet_tf2.0_299.ipynb （训练测试代码）
### 3. 数据文件（/home/data 或 data3）下的某个目录，说明下什么数据集
/home/data/nfsw/shtjh
下面分黄图和非黄图两个文件夹
### 4. 当前目录的相关配置文件，数据文件的说明。
无
### 5. 说明下部署的时候需要做啥？
复制模型文件到tf2_mobilenet_v2_399目录下
### 6. 返回过去的样本数据？
{"id": 20818418079, "rate": "-11.501033"}

———————————————————————————————————————

# 抠图训练测试
目录地址 matting
### 1. 相关的有用的库文件目录 （git的地址，是否做了修改）
https://github.com/xuebinqin/U-2-Net
做过简单的修改
### 2. 有用的ipynb, 或 py (training,test case)
matting/U-2-Net-master/train.ipynb （训练代码）
matting/U-2-Net-master/test.ipynb （测试代码）
以及1中下载的库文件中若干文件
### 3. 数据文件（/home/data 或 data3）下的某个目录，说明下什么数据集
/home/data3/cr/remove_bg
下面两个文件夹，分别是原图，以及抠出来的alpha图片，两张图片文件名相同，后缀分别为jpg、png
### 4. 当前目录的相关配置文件，数据文件的说明。
/home/data/cr/msyh.ttc
字体文件，测试时用到
### 5. 说明下部署的时候需要做啥？
复制相应的模型文件到saved_models目录下，模型文件有两个，一个人人像抠图，一个是通用抠图。模型文件名需与先前的一致
### 6. 返回过去的样本数据？
{"status": 200, "taskId": 111, "oss_url_matting":"http://......"} 

———————————————————————————————————————


# 框图训练测试
目录地址 ObjectDetection_new
### 1. 相关的有用的库文件目录 （git的地址，是否做了修改）
https://github.com/bubbliiiing/efficientdet-keras
做了些许修改
### 2. 有用的ipynb, 或 py (training,test case)
ObjectDetection_new/efficientdet-keras-master/train_50class.ipynb
ObjectDetection_new/efficientdet-keras-master/predict_and_evaluate.ipynb
ObjectDetection_new/data_process_50class.ipynb
以及1中下载的库文件中若干文件
### 3. 数据文件（/home/data 或 data3）下的某个目录，说明下什么数据集
/home/data3/Objects365
/home/data3/openimage_v6
### 4. 当前目录的相关配置文件，数据文件的说明。
/home/data/cr/msyh.ttc
字体文件，测试时用到
./ObjectDetection_new/2007_train_50class.txt
经处理以后的图片目录，以及该图片上的若干框，格式：目录 x1 y1 x2 y2 label 。示例： /home/111.jpg 100 100 500 500 0
./ObjectDetection_new/180-classes0202.xlsx或./ObjectDetection_new/180-classes1119.xlsx
经Pluto整理后的，各类别的名称
### 5. 说明下部署的时候需要做啥？
将模型文件复制到./ObjectDetection_new/efficientdet-keras-master/logs目录下，保证文件名相同。如果有修改分类信息，需替换掉180-classes0202.xlsx文件。
### 6. 返回过去的样本数据？
{"img_id": 20059762017, "boxes": [{"box": [0.141, 0.293, 0.953, 0.903]}, {"box": [0.011, 0.282, 0.288, 0.866]}]}

———————————————————————————————————————

# 人物识别训练
目录地址 person_claassify_new
### 1. 相关的有用的库文件目录 （git的地址，是否做了修改）
无
### 2. 有用的ipynb, 或 py (training,test case)
person_claassify_new/tf2_EfficientNetB0_gender.ipynb
性别识别
person_claassify_new/tf2_EfficientNetB0_2class.ipynb
插画、摄影识别
person_claassify_new/tf2_EfficientNetB0_age.ipynb
年龄识别
### 3. 数据文件（/home/data 或 data3）下的某个目录，说明下什么数据集
插画摄影分类数据集路径 /home/data2/cr/二分 /home/data2/cr/大兔
性别分类数据集路径 /home/data2/cr/gender_train
年龄分类数据集路径 /home/data2/cr/human_head_age
### 4. 当前目录的相关配置文件，数据文件的说明。
/home/data/cr/msyh.ttc
字体文件，测试时用到
### 5. 说明下部署的时候需要做啥？
替换相对应模型，二分类：ep018-val_loss0.08642-val_accuracy0.96961.h5
年龄：EfficientNetB0_age/ep025-val_loss0.73843-val_accuracy0.71467.h5
性别：EfficientNetB0_gender/ep028-val_loss0.34983-val_accuracy0.88631.h5

因为使用到别人的框图模型，需下载：https://github.com/tensorflow/models 等文件、见阿里云上部署说明。
### 6. 返回过去的样本数据？
{"label": "摄影", "person_count": 1, "person": [{"label": "Man", "boxes": [0.155, 0.313, 0.299, 0.552]}], "face_count": 0, "face": [], "img_id": 20254591637}


———————————————————————————————————————

jupyter notebook，可以pip安装，也可以用Anaconda，一般用Anaconda 
Anaconda教程
https://docs.continuum.io/
另需tensorflow、pytorch等GPU版本，参考各自的官网教程。https://tensorflow.google.cn/install/gpu， https://pytorch.org/get-started/locally/
需安装显卡驱动、CUDA、cudnn
可参考https://blog.csdn.net/qq_36653505/article/details/83932941，https://blog.csdn.net/smartxiong_/article/details/109045621

———————————————————————————————————————

# 虚拟环境创建
### 鉴黄
conda create -n py369new python=3.6.9  
conda activate py369new  
>pip install ipykernel  
>python -m ipykernel install --user --name py36new --display-name py369new  
>pip install scikit-learn==0.24.2 pandas==1.1.5 pillow==8.2.0 matplotlib==3.3.4  
>pip install tensorflow-model-optimization==0.2.1 tensorflow-gpu==2.1.0  
### 框图
conda create -n py369ob python=3.6.9  
conda activate py369ob  
>pip install ipykernel  
>python -m ipykernel install --user --name py369ob --display-name py369ob  
>pip install tensorflow-gpu==2.3.0 keras==2.4.3 tensorflow-model-optimization==0.2.1  
>pip install opencv-python==4.5.2.52 tqdm==4.61.0 pandas==1.1.5 xlrd==1.2.0 pillow==8.2.0 matplotlib==3.3.4 scikit-learn==0.24.2  
### 抠图
conda create -n py369mt python=3.6.9  
conda activate py369mt  
>pip install ipykernel  
>python -m ipykernel install --user --name py369mt --display-name py369mt  
>pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html  
>pip install opencv-python==4.5.2.52 pandas==1.1.5 pillow==8.2.0 scikit-image==0.17.2 scikit-learn==0.24.2  

### 打标签 ——>同鉴黄

### 空间家具识别训练测试 ——>同框图

### 粗分类训练测试 ——>同鉴黄

### 人物识别训练 ——>同框图




