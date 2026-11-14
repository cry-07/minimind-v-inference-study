# 来源、修改与许可说明

本成果仓库新增代码和文档使用根目录 LICENSE 中的 Apache License 2.0。此许可不授予上游模型权重或图片的额外权利。

## MiniMind-V

- 仓库：https://github.com/jingyaogong/minimind-v
- 固定提交：041d528e037a8a373c3c8aa7278a3ba3bd26b6dc
- 原项目许可副本：[third_party/minimind-v-LICENSE](third_party/minimind-v-LICENSE)
- docs/annotated-code.md、docs/inference-path.md 中的代码摘录及 patches/m2-observed-comments.patch 来自该版本推理实现；本地修改是教学注释与空白。摘录保留原有说明，不将上游实现归属本项目作者。补丁精确保留历史注释，部分措辞不准确之处在推理路径文档中明确澄清。
- 固定提交根目录未发现 NOTICE 文件；若未来替换版本，需要重新核对。

## 模型与图片

- 语言/视觉投影权重：https://huggingface.co/jingyaogong/minimind-3v-pytorch
- 视觉编码器：https://huggingface.co/jingyaogong/siglip2-base-p32-256-ve
- revision、文件名和哈希见 docs/reproduction.md。
- README 图片通过固定上游 raw URL 引用，没有随仓库打包；原始摄影/生成来源与单图许可未独立核验。该图片只用于说明已有上游例子的运行输入，不宣称原创或实拍。

## 学习背景

此前学习《从零构建大模型》，参考代码：https://github.com/rasbt/LLMs-from-scratch 。本发布目录未复制该书代码。
