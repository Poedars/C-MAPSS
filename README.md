# 6.2 智能制造领域的应用案例：最终配套代码

对应教材三部分核心代码：模型构建、模型训练、预测与评估。完整流程还包含数据预处理、结果保存和绘图。开源函数的位置、固定提交与改动见 [SOURCES.md](SOURCES.md)，原始源码和MIT许可证保存在 upstream/。

## 快速运行

先完整解压，在本README所在目录打开终端。建议使用Python 3.12及独立虚拟环境。

```bash
python -m venv .venv
```

Windows命令提示符激活：`.venv\Scripts\activate.bat`；Linux/macOS激活：`source .venv/bin/activate`。

```bash
python -m pip install -r requirements.txt
python run_case.py --data-dir CMAPSSData --out new_output
```

数据下载入口：
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

本包不含NASA原始数据。请下载C-MAPSS数据集，并将 train_FD001.txt、test_FD001.txt、RUL_FD001.txt 三个文件直接放入 CMAPSSData/。这是涡扇发动机退化仿真数据，并非真实工厂采集数据。教材实验使用的数据文件SHA256见 output/metrics.json。

运行命令完成10轮训练、预测、评估，并自动调用 plot_results.py 生成四幅图；新结果保存在 new_output/。请使用上述 --out 参数，避免覆盖 output/ 中教材原始实验记录。默认输出目录是 output/。

## 文件说明

| 文件或目录 | 用途 |
|---|---|
| core.py | 六个沿用的开源核心函数，包括模型、学习率调度、评分和数据处理 |
| run_case.py | 完整运行入口，衔接数据处理、训练、预测、评估和输出 |
| plot_results.py | 绘图辅助模块，由run_case.py调用；单独执行该文件不会自动出图 |
| requirements.txt | 本次实验使用的依赖版本 |
| SOURCES.md | 逐项代码来源、固定版本链接及改动说明 |
| upstream/ | 上游源码快照及两份MIT许可证 |
| output/model.keras | 本次10轮训练后的模型 |
| output/scaler.npz | 仅由训练设备拟合的标准化参数和特征顺序 |
| output/history.csv | 各轮训练与验证损失 |
| output/predictions.csv | 100台测试发动机的标签、预测值和误差 |
| output/metrics.json | 指标、数据划分、版本及数据文件哈希 |
| output/training.log | 教材实验的训练日志 |
| output/*.png | sensor、labels、training、evaluation四幅教材图 |
| validation.txt | 前次完整实验及来源核对记录 |
| SHA256SUMS.txt | 最终包各文件的完整性校验值 |

## 教材结果与复现口径

训练设备80台，验证设备20台；窗口30×14；训练/验证标签上限125；随机种子42；10轮训练，batch128。测试每台设备仅取最后窗口，官方测试标签不截断，预测不裁剪。使用最后一轮模型，不恢复最佳验证轮次。

保存的真实运行结果：RMSE=16.809561799858994周期，Score=715.6395484256846。数据预处理、运行环境或随机数调用顺序变化都可能导致复现数值不同。这些结果不是直接抄录上游项目的指标。

完整程序在数据准备前设置种子，划分数据后不要再次重置种子。修改窗口长度需同步修改 run_case.py 中 WINDOW 和 core.py 中 window_length。

中文图表需要系统安装中文字体，例如Noto Sans CJK、微软雅黑或黑体。若出现方框，请安装字体后重新运行绘图流程。

本次最终整理只更新说明文件与打包清单，未改变已核验的模型代码和实验输出，也未重新训练模型。run_case.py、plot_results.py为教材新增整合和辅助代码，不声称来自上游。
