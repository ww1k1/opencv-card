# 💳 信用卡号码识别系统（Credit Card OCR）

## 📌 项目简介

本项目基于 OpenCV 实现了一个信用卡号码识别系统，采用传统计算机视觉方法（非深度学习）对信用卡图像中的数字进行自动识别。

系统能够从输入图片中检测出信用卡号码区域，对数字进行分割，并通过模板匹配的方法识别每一位数字，最终输出完整卡号以及卡类型。

---

## 🚀 项目效果

**输入：** 信用卡图片
**输出：**

<img width="303" height="221" alt="image" src="https://github.com/user-attachments/assets/048bfb60-78bc-4acd-9ed4-e4cf11a98f38" />

---

## 🛠 技术栈

* Python
* OpenCV
* NumPy
* imutils

---

## 📂 项目结构

```text
project/
│── ocr_template_match.py   # 主程序（信用卡识别）
│── myutils.py              # 工具函数
│── images/                 # 测试图片
│── template/               # 数字模板
│── README.md
```

---

## ⚙️ 运行方法（debug软件为eclipse）：

### 1️⃣ 克隆项目

git clone https://github.com/你的用户名/项目名.git
cd 项目名

### 2️⃣ 安装依赖

pip install opencv-python numpy imutils

### 3️⃣ 运行程序

python ocr_template_match.py -i images/credit_card.png -t template/ocr_a.png

参数说明：

* `-i`：输入信用卡图像路径
* `-t`：OCR模板路径

---

## 🎯 核心功能

* 信用卡数字区域检测
* 数字分组提取（4位一组）
* 单个数字分割
* 模板匹配识别
* 信用卡映射

---

## 🔍 算法流程

1. 模板预处理（提取0-9数字模板）
2. 输入图像灰度化处理
3. 礼帽操作增强数字区域
4. Sobel算子提取梯度信息
5. 形态学操作连接数字区域
6. 轮廓检测定位数字区域
7. 数字分割
8. 模板匹配识别
9. 输出卡号与类型

---

## 💻 核心代码展示

### 1️⃣ 模板提取

img = cv2.imread(args["template"])
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

refCnts, _ = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]

digits = {}
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y+h, x:x+w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi

---

### 2️⃣ 图像预处理

image = cv2.imread(args["image"])
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

gradX = cv2.Sobel(tophat, cv2.CV_32F, 1, 0)
gradX = np.absolute(gradX)
gradX = (255 * (gradX - np.min(gradX)) / (np.max(gradX) - np.min(gradX)))
gradX = gradX.astype("uint8")

---

### 3️⃣ 数字识别（模板匹配）

scores = []
for (digit, digitROI) in digits.items():
    result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
    (_, score, _, _) = cv2.minMaxLoc(result)
    scores.append(score)

predicted_digit = np.argmax(scores)

---

## 🌟 项目亮点

* 使用传统计算机视觉方法实现OCR，无需深度学习模型
* 多阶段图像处理（Top-hat + Sobel + 形态学）
* 模板匹配实现高效识别
* 基于轮廓筛选的区域检测策略
* 代码结构清晰，模块化设计

---

## ⚠️ 不足与改进

### 当前不足

* 对复杂背景适应性较弱
* 对光照变化敏感
* 模板匹配泛化能力有限

### 改进方向

* 使用CNN进行数字识别
* 引入目标检测模型（如YOLO）
* 增强数据集，提高鲁棒性

---

## 📬 联系方式

wechat：WangS040122
