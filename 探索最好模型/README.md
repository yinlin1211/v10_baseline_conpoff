# 探索最好模型

这个目录记录的是一次独立的“checkpoint 批量探索”实验，用来观察：

- 在 `run/20260422_201016_COnP/checkpoints/` 中
- 不同 epoch 的 best checkpoint
- 经过快速两阶段 `val` 阈值搜索后
- 在 `MIR-ST500 test set` 上的最终表现差异

## 采用的流程

对每个 checkpoint：

1. 固定第一轮 `frame=0.40`
2. 在 `val40` 上搜索 `onset`
3. 固定选出的 `onset`，在 `val40` 上搜索 `frame`
4. 固定前两个阈值，在 `val40` 上搜索 `offset`
5. 最后在 `test100` 上推理并记录结果

这是一个快速近似搜索流程，不是完整二维 `onset x frame` 全网格穷举。

## 文件说明

- `run_fast_checkpoint_eval_cpu_priority.py`
  单个 checkpoint 的快速搜索脚本。
- `run_batch_fast_100_240.sh`
  批量脚本模板。当前实际跑的是其中自动扩展后的 `80~500` 范围。
- `plot_batch_fast_results.py`
  将批量结果日志解析成汇总表和折线图。
- `batch_fast_80_500.log`
  `80~500` 区间所有 best checkpoint 的批量运行日志。
- `batch_fast_80_500_summary.csv`
  从日志中抽取出的结构化结果表。
- `batch_fast_80_500_metrics.png`
  三个指标随 checkpoint 变化的可视化结果。

## 当前结果摘要

批量结果中：

- `COn` 最好：`epoch128`
- `COnP` 最好：`epoch128`
- `COnPOff` 最好：`epoch244`

详细数值请直接查看 `batch_fast_80_500_summary.csv`。
