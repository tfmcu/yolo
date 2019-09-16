# [YOLO](https://github.com/tfzoo/YOLO) 

### [YOLO简介](https://github.com/tfzoo/YOLO/wiki) 

继RCNN，fast-RCNN 和 faster-RCNN之后对DL目标检测速度问题提出的另外一种框架。YOLO V1其增强版本在GPU上能跑45fps，简化版本155fps。YOLO 使用的是 GoogLeNet 架构，比 VGG-16 快，YOLO 完成一次前向过程只用 85.2 亿次运算，而 VGG-16 要 306.9 亿次，但是 YOLO 精度稍低于 VGG-16。

YOLO 的核心思想就是利用整张图作为网络的输入，直接在输出层回归 bounding box（边界框） 的位置及其所属的类别。

faster-RCNN 中也直接用整张图作为输入，但是 faster-RCNN 整体还是采用了RCNN 那种 proposal+classifier 的思想，只不过是将提取 proposal 的步骤放在 CNN 中实现了，而 YOLO 则采用直接回归的思路。

YOLO将全图划分为SXS的格子，每个格子负责中心在该格子的目标检测，采用一次性预测所有格子所含目标的bbox、定位置信度以及所有类别概率向量来将问题一次性解决(one-shot)。

* YOLO 方法模型训练依赖于物体识别标注数据，因此，对于非常规的物体形状或比例，YOLO 的检测效果并不理想。
* YOLO 采用了多个下采样层，网络学到的物体特征并不精细，因此也会影响检测效果。
* YOLO 的损失函数中，大物体 IOU 误差和小物体 IOU 误差对网络训练中 loss 贡献值接近（虽然采用求平方根方式，但没有根本解决问题）。因此，对于小物体，小的 IOU 误差也会对网络优化过程造成很大的影响，从而降低了物体检测的定位准确性。

YOLOv2：代表着目前业界最先进物体检测的水平，它的速度要快过其他检测系统（FasterR-CNN，ResNet，SSD），使用者可以在它的速度与精确度之间进行权衡。

YOLO v2 基于一个新的分类模型，有点类似于 VGG。YOLO v2 使用 3*3 的 filter，每次池化之后都增加一倍 Channels 的数量。YOLO v2 使用全局平均池化，使用 Batch Normilazation 来让训练更稳定，加速收敛，使模型规范化。

最终的模型–Darknet19，有 19 个卷积层和 5 个 maxpooling 层，处理一张图片只需要 55.8 亿次运算，在 ImageNet 上达到 72.9% top-1 精确度，91.2% top-5 精确度。

YOLO9000：这一网络结构可以实时地检测超过 9000 种物体分类，这归功于它使用了WordTree，通过 WordTree 来混合检测数据集与识别数据集之中的数据，并使用联合优化技术同时在 ImageNet 和 COCO 数据集上进行训练，YOLO9000 进一步缩小了监测数据集与识别数据集之间的大小代沟。

YOLOv3 在 Pascal Titan X 上处理 608x608 图像速度可以达到 20FPS，在 COCO test-dev 上 mAP@0.5 达到 57.9%，与RetinaNet（FocalLoss论文所提出的单阶段网络）的结果相近，并且速度快 4 倍。YOLO v3 的模型比之前的模型复杂了不少，可以通过改变模型结构的大小来权衡速度与精度。

YOLOv3 的先验检测（Prior detection）系统将分类器或定位器重新用于执行检测任务。他们将模型应用于图像的多个位置和尺度。而那些评分较高的区域就可以视为检测结果。此外，相对于其它目标检测方法，我们使用了完全不同的方法。我们将一个单神经网络应用于整张图像，该网络将图像划分为不同的区域，因而预测每一块区域的边界框和概率，这些边界框会通过预测的概率加权。我们的模型相比于基于分类器的系统有一些优势。它在测试时会查看整个图像，所以它的预测利用了图像中的全局信息。与需要数千张单一目标图像的 R-CNN 不同，它通过单一网络评估进行预测。这令 YOLOv3 非常快，一般它比 R-CNN 快 1000 倍、比 Fast R-CNN 快 100 倍。


#### YOLO缺点

* 对小物体及邻近特征检测效果差：当一个小格中出现多于两个小物体或者一个小格中出现多个不同物体时效果欠佳。原因：B表示每个小格预测边界框数，而YOLO默认同格子里所有边界框为同种类物体。因为一个网格中只预测了两个框，并且只属于一类。

* 图片进入网络前会先进行resize为448 x 448，降低检测速度(it takes about 10ms in 25ms)，如果直接训练对应尺寸会有加速空间。

* 基础网络计算量较大，yolov2使用darknet-19进行加速。


###  [天府动物园 tfzoo：tensorflow models zoo](http://www.tfzoo.com)
####   qitas@qitas.cn
[![sites](tfzoo/tfzoo.png)](http://www.tfzoo.com)