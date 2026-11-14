# 复现说明

这份说明将已执行的 M1/M2 路径整理为相对路径命令。原实验使用已有 Conda 环境克隆，未在全新系统中重装验证；以下不是跨设备一键安装承诺。GPU、软件版本与依赖快照见 README 和 [environment-observed.txt](../configs/environment-observed.txt)。该快照是实际环境清单，不是保证可从公共源直接安装的通用锁文件。

## 1. 准备隔离环境

原实验从现有、已验证 CUDA 运算的环境克隆为项目 `.venv`，保留源环境。若已有匹配环境，可在仓库根目录使用：

```powershell
# 将 SOURCE_ENV_PATH 替换为你自己的已验证 Conda 环境路径。
conda create --prefix .\.venv --clone 'SOURCE_ENV_PATH' --offline
& .\.venv\python.exe -m pip install 'transformers==4.57.6'
& .\.venv\python.exe -m pip check
& .\.venv\python.exe -X utf8 .\scripts\check_gpu.py
```

`SOURCE_ENV_PATH` 是需要替换的参数，不是本仓库内文件。未有现成环境时，先按当前 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)准备支持自己 GPU 的构建，再检查本项目依赖；不要只按旧版上游 requirements 降级 PyTorch。本项目采用的完整版本仅证明本机已有环境可用。

## 2. 固定上游代码

在本成果仓库根目录执行。只针对新目录操作，不覆盖已有 vendor 检出。

```powershell
git clone https://github.com/jingyaogong/minimind-v.git vendor/minimind-v
git -C vendor/minimind-v checkout --detach 041d528e037a8a373c3c8aa7278a3ba3bd26b6dc
git -C vendor/minimind-v rev-parse HEAD
```

模型推理实现不随本仓库重新打包。M2 观察时的上游教学注释 diff 已保存在 [m2-observed-comments.patch](../patches/m2-observed-comments.patch)；仅作历史对照，含原注释歧义，阅读时结合 [澄清说明](inference-path.md)。不必应用补丁即可运行观察脚本；本地静态检查确认其相对固定提交仅增加注释/空白，AST 相同。

## 3. 获取匹配资源

必要文件共 329,181,648 bytes（约 314 MiB），不包含上游代码；需要联网下载。仅执行以下指定资源，不下载训练数据。

```powershell
$env:HF_HOME = "$PWD\.cache\huggingface"
& .\.venv\Scripts\hf.exe download jingyaogong/minimind-3v-pytorch sft_vlm_768.pth --revision 83ae67a06ec29dc1acf2294d8edacbdef12f4013 --local-dir vendor/minimind-v/out
& .\.venv\Scripts\hf.exe download jingyaogong/siglip2-base-p32-256-ve config.json preprocessor_config.json model.safetensors --revision 9465d1dc89db6bc6227c5b6b0e0ca9b940325d62 --local-dir vendor/minimind-v/model/siglip2-base-p32-256-ve
```

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| sft_vlm_768.pth | 140051548 | `0d3fee0970c2174d22226de25d273e2f375bda8ed39b8f1673e1519c94264f9e` |
| model.safetensors | 189129296 | `c1e9cc19ed6704b87353ee00b9ff5d6191886d741898339984364f789c62810d` |

Tokenizer 来自固定上游 `model/` 中的文件，不另外替换。视觉配置为 image=256、patch=32、hidden=768。模型资源各自许可与使用条件请见对应来源，本仓库的代码许可不替代模型或图片的许可。

## 4. 单张图片推理

本仓库外链展示上游图片；复现时使用固定检出自带图片。

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
New-Item -ItemType Directory -Force .\work\single | Out-Null
Copy-Item .\vendor\minimind-v\dataset\eval_images\image-01-golden-dog-balloons.jpg .\work\single\
New-Item -ItemType Directory -Force .\results\raw | Out-Null
$runName = 'm1-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8) + '.txt'
$logPath = Join-Path (Resolve-Path .\results\raw).Path $runName
Push-Location .\vendor\minimind-v
try {
    & ..\..\.venv\python.exe -X utf8 -u .\eval_vlm.py --load_from model --weight sft_vlm --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --device cuda --image_dir ../../work/single --max_new_tokens 64 2>&1 | Tee-Object -FilePath $logPath
} finally {
    Pop-Location
}
```

上游入口使用采样（temperature=0.7、top_p=0.85、每图随机种子），因此回答可以变化。新日志不会覆盖精选历史记录。上游统计不作为正式性能基准。

## 5. 观察前两次前向

```powershell
& .\.venv\python.exe -X utf8 -u .\scripts\trace_vlm_shapes.py
```

脚本离线加载已下载资源，最多生成 2 token，输出独立 `results/raw/m2-shapes-*.json`。检查 status=completed，以及 input/past_length 为 93/0、1/93；图像/投影各64位置，融合序列仍93位置。该长度依赖本说明中的默认图片与提示；更换问题可能改变文本长度。

历史运行脚本与公开脚本字节一致；历史资源与固定提交可追溯。本轮发布整理只做静态和证据核验，没有再次运行 GPU 推理，也没有全新系统安装测试。
