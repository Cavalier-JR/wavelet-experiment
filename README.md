# 小波分析及应用课程实验

本项目是我的课程实验，研究电力终端监测信号的小波消噪与突变分析。实验使用 Python 生成三类仿真信号，比较不同消噪方法，并分析小波参数和阶跃定位效果。

## 项目说明

仓库包含程序、实验数据、绘图结果和复现步骤。环境安装、程序运行和结果核对方法见 **实验复现教程.md**。

- `verify_results.py`：核对全部汇总数值，并从逐次CSV独立重算消噪和参数实验统计。
- `show_results.py`：打印与正文表2、表3及图4对应的数据。
- `reference/results_expected.json`：数值核对基准，实验运行不覆盖。
- `supplementary.py`：独立噪声验证集选择移动平均窗口和FFT低通截止频率，补充局部RMSE及成对差值区间。
- `verify_supplementary.py`：从新增CSV独立复算统计，核对验证集、测试集与主实验的对应关系。

主实验数值及图1至图5的定义见课程报告；补充实验结果见 `data/supplementary_results.json`。

## 项目文件

- `experiment.py`：生成信号、添加噪声、比较消噪方法、参数分析、突变分析及绘图。
- `requirements.txt`：本次实验实际运行的依赖版本。
- `figures/`：正文图1至图6。
- `data/denoising_raw.csv`：270条含噪记录对应4种方法，共1080条逐次指标。
- `data/denoising_summary.csv`：按信号、噪声档、方法汇总的均值与样本标准差。
- `data/sensitivity_raw.csv` 和 `sensitivity_summary.csv`：小波基与层数分析。
- `data/detection_raw.csv` 和 `detection_summary.csv`：阶跃定位逐次结果与汇总。
- `data/example_signals.npz`：图2对应的无噪、含噪及重建序列。
- `data/results.json`：运行环境与全部汇总结果。
- `data/baseline_validation_raw.csv` 与 `baseline_validation_summary.csv`：30条独立验证观测、10个常规滤波候选，共300条指标。
- `data/supplementary_raw.csv` 与 `supplementary_summary.csv`：5种方法在原有30条S3测试记录上的全程与局部指标，共150条指标；这些输出不另计为独立观测。
- `data/supplementary_results.json` 与 `figures/fig6_supplementary.png`：补充比较的表4、图6及成对差值区间。

## 复现方法

实验环境为 Python 3.12。克隆仓库后，在项目目录创建虚拟环境：

```bash
python -m venv .venv
```

以下是通用命令。Windows 下按《实验复现教程》直接调用 `.venv\Scripts\python.exe`；Linux 或 macOS 下可使用 `.venv/bin/python`。

```bash
python -m pip install -r requirements.txt
python experiment.py
python supplementary.py
python verify_results.py
python verify_supplementary.py
```

运行后，程序会重新生成 `data` 与 `figures` 中的文件。基础随机种子为20261104；在相同依赖环境下可以复算报告中的统计值。绘图字体可能因操作系统不同而略有差异，不影响数值结果。若中文显示为方框，可安装微软雅黑、黑体或 Noto Serif CJK SC，并调整脚本中的字体设置。

## 实验口径

采样率为1024 Hz，每条信号2048点。实验构造三类仿真信号，在5、10、15 dB三档标称输入噪声下各重复30次。噪声按设定标准差独立抽取为高斯噪声，因此逐次实际信噪比会有小幅波动；各方法使用同一条含噪观测进行比较。主消噪配置为db4、5层、对称延拓和通用阈值，并保留最粗近似系数。

阶跃定位使用S2记录检测阶跃，并在独立的S1背景记录上统计误报。背景记录的噪声功率以S2信号功率为参考，以保持噪声幅度可比。Haar窗每侧长度为8、16、32，标准化峰值阈值为6，峰间距为128点，匹配容差为16点，两端各64点不参与检测。标准化同时考虑原始噪声和本尺度MAD；定位误差只对成功匹配的事件计算。F1根据事件记录中的真正例、误报和漏报计算；独立背景的告警数另行列出。

实验数据由程序根据数学定义生成。信号阶跃表示形态变化；原始观测用于位置复核，消噪输出用于重建质量评价和趋势观察。

## 补充比较口径

补充比较的验证集为S3、10 dB下30次独立噪声，种子20322104至20322133；测试集复用主实验S3、10 dB下的种子20282104至20282133，两组种子不重叠。移动平均候选窗长为5、9、17、33、65点，FFT截止频率为16、32、64、96、128 Hz，均按验证集平均输出SNR选参。FFT处理前在两端各对称延拓2048点，避免直接连接原序列首尾。

测试结果中，33点移动平均和16 Hz FFT低通的输出SNR分别为18.85和18.38 dB；db4五层硬阈值为20.48 dB。暂态局部RMSE在[1,1.12)秒的123个采样点上相对无噪S3计算，硬阈值为0.0909，33点移动平均为0.1384。CSV中的 `transient_gain` 是扣除无噪S2背景后向振荡模板投影的系数。

成对t区间基于同一组30条含噪观测上的SNR差值。
