# [YOLO](https://github.com/tfzoo/YOLO) 

[![sites](tfzoo/tfzoo.png)](http://www.tfzoo.com)

## [简介](https://github.com/tfzoo/YOLO/wiki) 

YOLO将全图划分为SXS的格子，每个格子负责中心在该格子的目标检测，采用一次性预测所有格子所含目标的bbox、定位置信度以及所有类别概率向量来将问题一次性解决(one-shot)。

#### YOLO缺点

- 对小物体及邻近特征检测效果差：当一个小格中出现多于两个小物体或者一个小格中出现多个不同物体时效果欠佳。原因：B表示每个小格预测边界框数，而YOLO默认同格子里所有边界框为同种类物体。

- 图片进入网络前会先进行resize为448 x 448，降低检测速度(it takes about 10ms in 25ms)，如果直接训练对应尺寸会有加速空间。

- 基础网络计算量较大，yolov2使用darknet-19进行加速。

### 参考资源

#### [keras YOLOv3](https://github.com/qqwweee/keras-yolo3) 

#### [keras YOLOv2](https://github.com/experiencor/keras-yolo2) 


---

###  www.tfzoo.com 
####  qitas@qitas.cn
