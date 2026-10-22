下水道排水管道缺陷实例分割数据集  基于YOLOv11下水管道排水管缺陷实例分割系统 变形  沉积错口破裂树根障碍物
下水道排水管道缺陷实例分割数据集，1925张，yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0d3cff98506a42019309155f68a09897.png)

数据集统计信息
====================
图像尺寸:1024*1024
类别数量: 7 类
训练集图像:1636; 验证集图像:269; 测试集图像:20
总图像数: 1925

类别分布:![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ca99bbdd7157463f8927bd4c71b5161e.png)

类别名称 | 包含该类别的图像数 | 总标注数
----------------------------------------![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b8c06121a95e4e5b852953b761bf4e3c.png)

BX 变形 | 455 | 620
CJ 沉积| 365 | 426
CK 错口| 35 | 38
OBB | 92 | 115
PL 破裂| 210 | 318
SG 树根| 438 | 671
ZW 障碍物| 392 | 603
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bc286d28e45540fcb98918e2e99efe43.png)


模型代码：模型训练使用yolov11n-seg训练，80个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/426b7125e2ba4d43ae6d653d85ef3401.png)

qt界面：运行界面采用pyqt5编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ca8855787d5c4522af60358b174a436b.png)

4.支持windows和linux系统）![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9b341e6d272b40eca097d65bad204c73.png)
# 基于YOLOv11 下水道缺陷实例分割系统完整工程代码
## 环境安装
```bash
pip install ultralytics opencv-python pyqt5 numpy pillow
```

## 1、数据集配置文件 sewer.yaml
```yaml
# 下水道缺陷分割数据集
path: ./sewer_dataset
train: images/train
val: images/val
test: images/test

names:
  0: BX  #变形
  1: CJ  #沉积/淤积
  2: CK  #穿孔
  3: OBB #障碍物
  4: PL  #破裂
  5: SG  #树根
  6: ZW  #杂物
```

## 2、模型训练代码 train_sewer.py
```python
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载yolov11分割模型
    model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")

    results = model.train(
        data="sewer.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project="sewer_defect_seg",
        name="yolo11n_sewer",
        amp=True,
        mosaic=1.0
    )
```

## 3、推理预测代码 predict.py
```python
from ultralytics import YOLO
import cv2

model = YOLO("./sewer_defect_seg/yolo11n_seg/weights/best.pt")

def predict_img(img_path):
    res = model(img_path, conf=0.3)
    for r in res:
        img = r.plot()
        cv2.imwrite("result.jpg", img)
    return img

if __name__ == "__main__":
    predict_img("test.jpg")
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/afd24222c5d54a588938f60248be4170.png)

## 4、PyQt5可视化界面代码 gui_sewer.py（对应你截图里的系统）
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,QFileDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class SewerSegWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLO11的下水道缺陷实例分割系统")
        self.setGeometry(100,100,1400,800)
        self.model = YOLO("./sewer_defect_seg/yolo11n_seg/weights/best.pt")

        #按钮
        self.btn_img = QPushButton("选择图片",self)
        self.btn_img.move(20,20)
        self.btn_img.clicked.connect(self.load_image)

        #图像显示label
        self.label_ori = QLabel("原始图像",self)
        self.label_ori.setGeometry(20,80,650,600)
        self.label_res = QLabel("分割结果",self)
        self.label_res.setGeometry(700,80,650,600)

    def load_image(self):
        fname,_ = QFileDialog.getOpenFileName(self,"选择图片",".","*.jpg;*.png")
        if not fname:
            return
        img = cv2.imread(fname)
        #推理
        pred = self.model(fname,conf=0.3)[0]
        res_img = pred.plot()
        #展示原图
        self.show_cvimg(img,self.label_ori)
        #展示结果
        self.show_cvimg(res_img,self.label_res)

    def show_cvimg(self,cvimg,label):
        rgb = cv2.cvtColor(cvimg,cv2.COLOR_BGR2RGB)
        h,w,c = rgb.shape
        qimg = QImage(rgb.data,w,h,w*c,QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.width(),label.height(),Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SewerSegWindow()
    win.show()
    sys.exit(app.exec_())
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5f88877c78f74014af53751c08bb3023.png)

## 应用场景
1. 城市市政排水管道CCTV内窥检测，自动识别管道内部7类缺陷：**变形BX、淤积CJ、穿孔CK、障碍物OBB、破裂PL、树根SG、杂物ZW**
2. 替代人工回看管道视频，自动标注缺陷位置、类别，输出分割掩码与缺陷报表
3. 管网健康评估、管道病害等级判定、管道修复工程量统计

## 数据集说明（配套）
> 下水道管道CCTV内窥缺陷实例分割数据集，包含7类管道病害，YOLO/VOC/COCO三种标注，图片来自管道机器人内窥视频分帧，覆盖不同管径、积水、暗光场景。
类别对照：
| 缩写 | 中文名称 |
| ---- | ---- |
| BX | 管道变形 |
| CJ | 管内淤积 |
| CK | 管壁穿孔 |
| OBB | 管内障碍物 |
| PL | 管道破裂 |
| SG | 树根侵入 |
| ZW | 管内杂物 |

运行：`python gui_sewer.py` 即可启动你截图中的可视化分割系统，支持图片检测，可扩展视频、摄像头实时检测。


