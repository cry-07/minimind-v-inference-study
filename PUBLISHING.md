# 发布说明

目标仓库：https://github.com/cry-07/minimind-v-inference-study 。本地目录已初始化 Git 并暂存精选文件；以下初始化命令仅适用于未初始化的新副本。

此目录是精选发布版本，不包含模型、环境、缓存、内部交接或第三阶段数据。先在本地 GitHub 风格 Markdown 预览中检查 README 的图片和 Mermaid；远端图片或渲染服务不可用时，正文表格仍可阅读。

本地静态校验结果见 RELEASE_CHECKS.md。准备好后在 GitHub 创建名为 minimind-v-inference-study 的空仓库，建议描述为“MiniMind-V 本地 GPU 推理复现、图文融合与 KV cache 实测分析”。不要在网页端额外初始化 README。

在本目录运行（将远端 URL 替换为自己的空仓库地址）：

```powershell
git init -b main
git add .
git diff --cached --stat
git commit -m "Document MiniMind-V inference reproduction and tensor tracing"
git remote add origin https://github.com/cry-07/minimind-v-inference-study.git
git push -u origin main
```

不要向公开仓库提交访问令牌或账户凭据；使用现有 Git 认证流程。若本目录已初始化或已有 origin，不重复执行初始化/添加远端命令，先检查状态。git commit 如提示身份未设置，使用自己的真实 Git 姓名和邮箱配置本仓库；不由助手虚构身份。

仓库发布后建议把 README 作为导师阅读入口。本人能解释一条完整推理路径、两个实测形状和一个失败案例，比扩展尚未完成的能力描述更重要。
