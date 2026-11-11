"""教材新增的实验绘图辅助代码，不归为上游代码。"""
from pathlib import Path
import numpy as np
CAP=125
def draw_source_figures(raw, history, truth, pred, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})
    def save(fig, name):
        fig.tight_layout(); fig.savefig(out/name, dpi=180); plt.close(fig)
    g = raw[raw.id == 1]
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
    for a, sensor in zip(ax, ["s4", "s7"]):
        a.plot(g.cycle, g[sensor], lw=0.8, label="原始读数")
        a.plot(g.cycle, g[sensor].rolling(10, min_periods=1).mean(),
               lw=1.5, label="过去10周期均值")
        a.set(xlabel="运行周期", ylabel=f"{sensor} 原始读数")
        a.legend(fontsize=8); a.grid(alpha=.2)
    save(fig, "sensor.png")
    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    raw_rul = g.cycle.max() - g.cycle
    ax.plot(g.cycle, raw_rul, label="未截断标签")
    ax.plot(g.cycle, np.minimum(raw_rul, CAP), "--", label="截断标签 上限125")
    ax.set(xlabel="运行周期", ylabel="RUL（周期）"); ax.legend();ax.grid(alpha=.2)
    save(fig, "labels.png")
    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    ax.plot(np.arange(1, len(history['loss'])+1), history['loss'], label='训练集')
    ax.plot(np.arange(1, len(history['val_loss'])+1), history['val_loss'], label='验证集')
    ax.set(xlabel='训练轮次', ylabel='MSE（周期²）')
    ax.legend(); ax.grid(alpha=.2)
    save(fig, "training.png")
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax[0].scatter(truth, pred, s=14, alpha=.7)
    lim=max(truth.max(), pred.max())+10
    ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="理想预测")
    ax[0].set(xlabel="真实末端RUL（周期）", ylabel="预测末端RUL（周期）")
    ax[0].legend(fontsize=8)
    ax[1].hist(pred-truth, bins=15, edgecolor="white")
    ax[1].axvline(0, color="black", linestyle="--", lw=1)
    ax[1].set(xlabel="预测值 − 真实值（周期）", ylabel="发动机数量")
    save(fig, "evaluation.png")
