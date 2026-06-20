# 计算机视觉基础实验2：图像检索与文本检测

本仓库为计算机视觉基础实验2的提交材料，主要完成校园 landmark 图像检索、Precision@TopK 评价和文字区域检测可视化。

## 文件位置

项目文件位于仓库中的 `实验2_GitHub仓库工作包_23281321_余钦/` 目录下，主要内容如下：

- `src/experiment2_pipeline.py`：图像检索和文本检测主程序
- `src/make_demo_videos.py`：演示视频生成脚本
- `README.md`：项目运行说明
- `requirements.txt`：Python 依赖
- `docs/sample_results/`：检索和文字检测示例结果图
- `demo_videos/`：4 个测试过程演示视频

## 运行方法

先将课程提供的数据集解压到项目目录中的 `dataset/` 文件夹，然后进入项目目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/experiment2_pipeline.py --dataset dataset --out outputs
```

程序运行后会在 `outputs/` 目录下生成检索结果、Precision@TopK 统计文件和文本检测可视化图片。

## 演示视频

`demo_videos/` 目录下提供了 4 个测试过程演示视频：

- `demo_fhy.mp4`
- `demo_nm.mp4`
- `demo_sy.mp4`
- `demo_zx.mp4`

视频只展示测试与结果查看过程，不包含训练过程。
