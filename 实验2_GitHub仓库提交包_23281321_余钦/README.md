# 实验2：图像检索与文字检测

学生：余钦  
学号：23281321

本项目完成北京交通大学校园 landmark 图像检索与文字区域检测。图像检索部分使用传统图像特征和相似度排序，文字检测部分使用阈值分割与形态学处理，并结合数据集标注进行结果对比。

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── experiment2_pipeline.py
├── docs/
│   └── sample_results/
├── demo_videos/
├── dataset/      # 本地数据集目录
└── outputs/      # 程序运行输出目录
```

## 数据集

课程数据集解压后放在项目目录的 `dataset/` 下，目录结构如下：

```text
dataset/
  image_retrieval/
    base/
      BJTU/
      util_pic/
    query/
  object_detection/
    data/
```

本次实验使用的数据包括：

- `image_retrieval/base/BJTU`：2665 张图片
- `image_retrieval/base/util_pic`：5063 张图片
- `image_retrieval/query`：135 张查询图片
- `object_detection/data`：1494 张图片及对应 LabelMe 标注文件

## 运行环境

Python 版本使用 3.12，依赖安装方式如下：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 运行方法

```bash
source .venv/bin/activate
python src/experiment2_pipeline.py --dataset dataset --out outputs
```

程序运行后会完成以下内容：

- 扫描 `base` 和 `query` 图片；
- 提取 HSV 颜色直方图、空间颜色块、HOG 和 ORB 特征；
- 对每张 query 检索 base 中的 Top-60 图片；
- 计算每类 landmark 的 P@20、P@40、P@60；
- 生成各类别 Precision@TopK 折线图；
- 生成各类别检索结果 contact sheet；
- 生成文字检测可视化样例。

## 方法说明

图像检索流程：

1. 根据文件名前缀解析类别标签，例如 `sy`、`tsg`、`nm`。
2. 对图像提取 HSV 颜色直方图、16×16 空间颜色特征和 HOG 形状纹理特征。
3. 使用余弦相似度完成初始排序。
4. 对候选短列表使用 ORB 特征匹配进行重排序。
5. 根据查询图片类别计算 P@20、P@40、P@60。

文字检测流程：

1. 对候选图像进行灰度化、增强和阈值处理。
2. 使用形态学闭运算连接文字笔画区域。
3. 根据面积、长宽比、边界位置等规则过滤候选框。
4. 可视化中黄色框表示检测结果，绿色框表示 LabelMe 标注框。

## 输出结果

```text
outputs/
  retrieval/
    top60_results.csv
    precision_by_class.csv
    precision_overall.csv
    plots/
    contact_sheets/
  text_detection/
    visual_samples/
    clear_examples/
```

本次运行的总体检索结果：

| K | Precision | Query 数 |
|---:|---:|---:|
| 20 | 0.2674 | 135 |
| 40 | 0.1994 | 135 |
| 60 | 0.1696 | 135 |

类别结果保存在 `outputs/retrieval/precision_by_class.csv`。其中 `fhy`、`jx`、`sjz`、`zx` 等类别表现较好；`kx`、`yf` 等类别受样本数量和视角差异影响，精度偏低。

## 演示视频

`demo_videos/` 目录下包含 4 个测试过程演示视频：

```text
demo_videos/demo_fhy.mp4
demo_videos/demo_nm.mp4
demo_videos/demo_sy.mp4
demo_videos/demo_zx.mp4
```

视频内容展示程序测试结果和样例查看过程，不包含训练过程。
