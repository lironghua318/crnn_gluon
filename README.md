### CRNN_GLUON
__crnn__ ocr识别算法mxnet-gluon实现
#### 依赖
* mxnet >=1.3.1
* gluon
* gluoncv
* python3.5

#### 组织结构
* crnn_gluon: 根目录
    * train_crnn_gluon.py: 训练脚本
    * predictor_gluon.py: 推理脚本
    * README.md
* crnn: 
    * config.py 配置文件
    * data_gluon.py 数据读取类
    * fit
        * ctc_metrics.py 计算acc类
    * gluons
        * crnn.py 网络结构
        * simplenet.py 特征提取网络，网络结构目前参照insight的symbol实现

#### 训练集
* 生成的数字图片
训练集和模型文件 https://pan.baidu.com/s/1Pl-CC7CYtogd1FZHpzg0PQ   
提取码: bskf     
* train_data/images/ 存放图片   
  train_data/train.txt 格式：xxx.img 0 5 9 1   
  train_data/test.txt  格式：xxx.img 0 6 0 1
  
#### 训练
python3 train_crnn_gluon.py
#### 推理
python3 predictor_gluon.py


