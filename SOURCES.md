# 代码逐项来源与改动

本版以两份MIT项目的固定提交为依据。core.py的六个函数均通过AST逐函数比对；除了窗口函数的np.int→int，无算法改写。最新版教材正文保留模型构建、模型训练、预测与评估三部分代码；数据处理函数由配套程序调用。正文增加中文注释、变量衔接及必要导入；训练使用tf.data封装，预测直接调用模型。MIT许可证全文见upstream，发布或再分发时请保留。

| 本版函数 | 上游精确位置 | 改动 |
|---|---|---|
| `calc_training_rul` | [data_util.py L60–67](https://github.com/sabderra/predictive-maintenance-lstm/blob/6580c16d1c63d8d69ff9e40d721b088b87921171/data_util.py#L60-L67) | 原样摘录 |
| `split_data` | [data_generator.py L6–34](https://github.com/sabderra/predictive-maintenance-lstm/blob/6580c16d1c63d8d69ff9e40d721b088b87921171/data_generator.py#L6-L34) | 原样摘录 |
| `process_input_data_with_targets` | [FD001笔记本 In[5]](https://github.com/biswajitsahoo1111/rul_codes_open/blob/6744186f62f6ca04118d8d5fb88d428b343f66ff/notebooks/cmapss_notebooks/CMAPSS_FD001_LSTM_piecewise_linear_degradation_model.ipynb) | 仅np.int→int |
| `create_compiled_model` | [FD001笔记本 In[9]](https://github.com/biswajitsahoo1111/rul_codes_open/blob/6744186f62f6ca04118d8d5fb88d428b343f66ff/notebooks/cmapss_notebooks/CMAPSS_FD001_LSTM_piecewise_linear_degradation_model.ipynb) | 原样摘录 |
| `scheduler` | [FD001笔记本 In[10]](https://github.com/biswajitsahoo1111/rul_codes_open/blob/6744186f62f6ca04118d8d5fb88d428b343f66ff/notebooks/cmapss_notebooks/CMAPSS_FD001_LSTM_piecewise_linear_degradation_model.ipynb) | 原样摘录 |
| `compute_s_score` | [FD001笔记本 In[16]](https://github.com/biswajitsahoo1111/rul_codes_open/blob/6744186f62f6ca04118d8d5fb88d428b343f66ff/notebooks/cmapss_notebooks/CMAPSS_FD001_LSTM_piecewise_linear_degradation_model.ipynb) | 原样摘录 |

## 新增与替换的部分

- `run_case.py`：本教材新增整合程序。数据命名、校验、接口衔接、float32转换、tf.data输入、线程及种子设置、输出保存为新增代码，不声称摘自上游。
- `plot_results.py`：本教材新增绘图代码，所有图对应实际FD001记录或本次运行输出。
- Sahoo原笔记本先生成窗口再随机划分。本版改用sabderra的设备级split_data，并仅在80台训练设备上拟合StandardScaler。
- 沿用Sahoo的14传感器、窗口30、标签上限125、原模型、学习率调度、10轮和batch128；测试只取最后窗口，与其最后窗口评价口径一致。官方测试RUL不截断、预测不裁剪。
- 标签基础计算来自sabderra，clip125由整合程序应用；没有沿用sabderra完整网络或训练流程。
- 因划分、标准化范围与运行环境有变，本版结果不是对原作者指标的严格复现。
- 第七节大语言模型为教学拓展，未实际调用LLM/RAG，也不归属于以上项目实现。

## 许可证与源码快照

- `upstream/LICENSE-rul_codes_open.txt`：原MIT许可，Copyright (c) 2022 Biswajit Sahoo。
- `upstream/LICENSE-sabderra.txt`：原MIT许可，原文版权行保留原样。
- upstream中的两个.py与.ipynb是源文件快照，仅供核对出处；请运行根目录run_case.py，不要直接运行旧环境笔记本。
