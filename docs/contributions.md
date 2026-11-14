# 工作范围与证据

## 上游提供

MiniMind-V 作者提供模型结构、推理实现、tokenizer、权重和示例入口；视觉编码器及其资源来自注明的模型仓库。本项目不是从零实现或训练视觉语言模型。[来源与许可](../THIRD_PARTY_NOTICES.md)

## 本人参与

亲自执行环境隔离与依赖准备、GPU 检查、权重获取和单图运行；反馈真实终端错误，参与排错；阅读上游关键代码，参与中文注释并理解图片预处理、占位符替换、模型前向、自回归生成及 KV cache；亲自运行形状采集，结合输出讨论机制。关键形状来自实际运行，不是按架构猜测。

## AI 辅助

AI 助手协助环境检查、脚本编写、源码解释、错误定位、记录核验和文档编排。中文注释由学习过程中双方陆续补充，无法据 diff 精确归属每一行，不宣称全部由本人独立编写。能运行脚本不自动等于掌握全部实现；展示时应能够解释输入到输出的路径和记录的边界。

## 精选证据

| 事项 | 文件 | 可证明的范围 |
| --- | --- | --- |
| CUDA 基础运算 | [gpu-source-env.json](../results/evidence/gpu-source-env.json) | 原有源环境 GPU 运算通过；不是隔离环境的独立日志 |
| 编码错误 | [m1-encoding-error.txt](../results/evidence/m1-encoding-error.txt) | 真实失败点与报错 |
| 单图回答 | [m1-single-image.txt](../results/evidence/m1-single-image.txt) | 可读图片问答已完成，保留错误与遗漏 |
| 前两次前向 | [m2-shapes.json](../results/evidence/m2-shapes.json) | 实际形状、缓存长度、配置、token ID与本地diff |
| 教学注释 | [annotated-code.md](annotated-code.md) | 含注释函数快照，歧义在正文澄清 |
| 数据转换 | [provenance.json](../results/evidence/provenance.json) | 公开副本与本地原件的哈希和变换范围 |

公开副本脱敏机器目录并规范编码/换行；没有修改生成回答、形状、配置、token ID 或测量值。原脚本与原始记录中的脚本哈希可核验；当前公开 JSON 中的 image 路径含 `<PROJECT_ROOT>`，是脱敏标记而非可直接执行路径。后续运行会产生新的本地原始记录。
