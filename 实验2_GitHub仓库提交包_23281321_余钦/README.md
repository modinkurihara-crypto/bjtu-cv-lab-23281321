# 实验2：图像检索与文字检测

学生：余钦  
学号：23281321

本仓库实现北京交通大学校园 landmark 图像检索与文字区域检测的测试流程。检索部分使用传统图像特征，不训练深度模型；文字检测部分使用传统 CV 阈值与形态学方法，并叠加数据集中 LabelMe 标注用于主观对比。

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── experiment2_pipeline.py
│   └── make_demo_videos.py
├── docs/
│   └── sample_results/
├── demo_videos/
├── dataset/      # 本地数据集，不上传 GitHub
└── outputs/      # 运行输出，不上传 GitHub
```

## 数据集

请将课程数据集解压到本目录的 `dataset/` 下，形成如下结构：

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

本地测试时，数据集包含：

- `image_retrieval/base/BJTU`：2665 张
- `image_retrieval/base/util_pic`：5063 张
- `image_retrieval/query`：135 张
- `object_detection/data`：1494 张图片及 1494 个 LabelMe JSON

## 运行环境

推荐使用 Python 3.12+。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 一键运行

```bash
source .venv/bin/activate
python src/experiment2_pipeline.py --dataset dataset --out outputs
```

脚本会完成：

- 扫描 `base` 和 `query` 图片；
- 提取 HSV 颜色直方图、空间颜色块、HOG 和 ORB 特征；
- 对每张 query 检索 base 中的 Top-60 图片；
- 计算每类 landmark 的 P@20、P@40、P@60；
- 生成每类 P@K 折线图，共 12 张；
- 生成每类检索结果 contact sheet；
- 生成 24 组文字检测可视化样例。

## 方法说明

图像检索流程：

1. 从文件名前缀解析类别标签，例如 `sy`、`tsg`、`nm`。
2. 对图像提取全局特征：HSV 三维颜色直方图、16×16 空间颜色特征、HOG 形状纹理特征。
3. 使用余弦相似度完成初始排序。
4. 对候选短列表使用 ORB 特征匹配进行重排序。
5. 依据文件名前缀计算 P@20、P@40、P@60。

文字检测流程：

1. 对候选图像进行灰度化、增强和边缘/阈值处理。
2. 使用形态学闭运算连接文字笔画区域。
3. 根据面积、长宽比、边界位置等规则过滤候选框。
4. 可视化中黄色框表示检测结果，绿色框表示 LabelMe 标注框。

## 主要输出

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

PPT 要求提交 3-5 个测试过程演示样例。本仓库已生成 4 个短视频：

```text
demo_videos/demo_fhy.mp4
demo_videos/demo_nm.mp4
demo_videos/demo_sy.mp4
demo_videos/demo_zx.mp4
```

如需重新生成演示视频：

```bash
source .venv/bin/activate
python src/make_demo_videos.py --outputs outputs --out demo_videos --assets docs/sample_results
```

## GitHub 上传说明

建议上传代码、README、`docs/sample_results/` 和 `demo_videos/`。不要上传 `dataset/`、`outputs/`、`.venv/`，这些目录已在 `.gitignore` 中排除。
