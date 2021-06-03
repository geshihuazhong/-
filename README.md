# 打标签训练测试
目录地址 8000class
## 1. 相关的有用的库文件目录 （git的地址，是否做了修改）
无
## 2. 有用的ipynb, 或 py (training,test case)
8000class/dataset-Copy1.ipynb（训练测试代码）
## 3. 数据文件（/home/data 或 data3）下的某个目录，说明下什么数据集
/home/data2/cr/download_img/explore_word/
下面每个文件夹是一个发现词，文件夹里是相应的图片
## 4. 当前目录的相关配置文件，数据文件的说明。
findwords_1.xlsx
整理后的，删除了部分发现词
dicts_8544.npy
8544个分类id对应的名称
## 5. 说明下部署的时候需要做啥？
将模型文件拷到tf2_mobilenet_no_pruned_dataset_8544目录下，保证文件名相同即可。
## 6. 返回过去的样本数据？
{'words': ['太空人', '宇航员', '公共设施', '入户柜', '虚拟', '柱状图', '微地形景观', '科技界面', '漫画', '海洋动物']}



# 空间家具识别训练测试
目录地址 furniture
## 1. 相关的有用的库文件目录 （git的地址，是否做了修改）
无
## 2. 有用的ipynb, 或 py (training,test case)
furniture/tf2_train_first_EfficientNetB0.ipynb（训练测试代码）
训练第一层时，将目录改为/home/data/cr/furniture_pre_train
训练第二层时，将目录改为/home/data/cr/furniture_pre_train单品/
训练第三层时，将目录改为/home/data/cr/furniture_sec_train
## 3. 数据文件（/home/data 或 data3）下的某个目录，说明下什么数据集
/home/data/cr/furniture_pre_train/单品/
/home/data/cr/furniture_pre_train/
训练第三层时，将目录改为/home/data/cr/furniture_sec_train
下面每个文件夹是一个分类，文件夹里是相应的图片
## 4. 当前目录的相关配置文件，数据文件的说明。
/home/data/cr/msyh.ttc
字体文件，测试时用到
## 5. 说明下部署的时候需要做啥？
将模型文件拷到inception_v4_first、inception_v4_second、inception_v4_thirdly目录下，保证文件名相同即可。
## 6. 返回过去的样本数据？
{"type": "单品", "names": [{"label": “椅”, "boxes": []}]} 
{"names":[{"boxes":[0.419,0.061,0.856,0.581],"label":"架"}],"type":"室内场景"}   
{"names":["办公楼"],"type":"户外场景"}  
{"type": "负例", "names": []} 
{"type": "平面图", "names": []} 


image_classify/mobilenet.ipynb
粗分类训练测试
1. 相关的有用的库文件目录 （git的地址，是否做了修改）
无
2. 有用的ipynb, 或 py (training,test case)
image_classify/mobilenet.ipynb（训练测试代码）
3. 数据文件（/home/data 或 data3）下的某个目录，说明下什么数据集
/home/data2/cr/image_classify
下面每个文件夹是一个发现词，文件夹里是相应的图片
4. 当前目录的相关配置文件，数据文件的说明。
/home/data/cr/msyh.ttc
字体文件，测试时用到
5. 说明下部署的时候需要做啥？
将模型文件拷到inception_v4_first、inception_v4_second、inception_v4_thirdly目录下，保证文件名相同即可。
6. 返回过去的样本数据？
{"type": "单品", "names": [{"label": “椅”, "boxes": []}]} 
{"names":[{"boxes":[0.419,0.061,0.856,0.581],"label":"架"}],"type":"室内场景"}   
{"names":["办公楼"],"type":"户外场景"}  
{"type": "负例", "names": []} 
{"type": "平面图", "names": []} 

identify-sex/MobileNet_tf2.0_299.ipynb
鉴黄训练测试

matting/U-2-Net-master/train.ipynb
matting/U-2-Net-master/test.ipynb
抠图训练测试

ObjectDetection_new/efficientdet-keras-master/train_50class.ipynb
ObjectDetection_new/efficientdet-keras-master/predict_and_evaluate.ipynb
框图训练测试

person_claassify_new/tf2_EfficientNetB0_gender.ipynb
性别识别
person_claassify_new/tf2_EfficientNetB0_2class.ipynb
插画、摄影识别
person_claassify_new/tf2_EfficientNetB0_age.ipynb
年龄识别
人物识别训练




jupyter notebook，可以pip安装，也可以用Anaconda，一般用Anaconda 
Anaconda教程
https://docs.continuum.io/
另需tensorflow、pytorch等GPU版本，参考各自的官网教程。https://tensorflow.google.cn/install/gpu， https://pytorch.org/get-started/locally/
需安装显卡驱动、CUDA、cudnn
可参考https://blog.csdn.net/qq_36653505/article/details/83932941，https://blog.csdn.net/smartxiong_/article/details/109045621




# 虚拟环境创建
## 鉴黄
conda create -n py369new python=3.6.9
conda activate py369new
	pip install ipykernel
	python -m ipykernel install --user --name py36new --display-name py369new
	pip install scikit-learn==0.24.2 pandas==1.1.5 pillow==8.2.0 matplotlib==3.3.4
	pip install tensorflow-model-optimization==0.2.1 tensorflow-gpu==2.1.0
框图
conda create -n py369ob python=3.6.9
conda activate py369ob
	pip install ipykernel
	python -m ipykernel install --user --name py369ob --display-name py369ob
	pip install tensorflow-gpu==2.3.0 keras==2.4.3 tensorflow-model-optimization==0.2.1
	pip install opencv-python==4.5.2.52 tqdm==4.61.0 pandas==1.1.5 xlrd==1.2.0 pillow==8.2.0 matplotlib==3.3.4 scikit-learn==0.24.2
抠图
conda create -n py369mt python=3.6.9
conda activate py369mt
	pip install ipykernel
	python -m ipykernel install --user --name py369mt --display-name py369mt
	pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
	pip install opencv-python==4.5.2.52 pandas==1.1.5 pillow==8.2.0 scikit-image==0.17.2 scikit-learn==0.24.2

打标签 ——>同鉴黄

空间家具识别训练测试 ——>同框图

粗分类训练测试 ——>同鉴黄

人物识别训练 ——>同框图




