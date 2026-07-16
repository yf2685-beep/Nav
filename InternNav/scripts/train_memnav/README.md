# MemNav 任务提交检查清单

本目录中的任何 Slurm 任务在提交前都必须完成本清单。这里的“依赖”同时包括：

- Slurm 前置任务及其 `afterok` 依赖；
- 容器、Conda、Python 包和 CUDA；
- 数据、LingBot 缓存、模型权重及配置几何；
- 日志、checkpoint 和 W&B 输出路径。

**长任务不得直接裸提交。必须先运行与正式任务同环境、同数据、同权重、同脚本的零步预检，并让正式任务通过 `afterok` 依赖该预检。**

## 1. 固定并验证代码

- 工作树无意外修改；需要运行的代码已经 commit 并推送。
- 记录完整 commit SHA，集群部署目录中的 `DEPLOY_COMMIT` 与其一致。
- 本地和部署端的关键脚本 SHA256 一致。
- 所有相关脚本通过语法和编译检查：

```bash
git diff --check
bash -n scripts/train_memnav/<job>.sbatch
python -m py_compile <changed-python-files>
```

不要从正在被其他人修改的工作树直接运行。使用独立、不可变的部署目录，并在任务记录中保存其绝对路径。

## 2. 验证运行时依赖

以下检查必须在任务使用的同一个 Apptainer 镜像、overlay 和 Conda 环境中执行：

```bash
command -v python
python -V
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

- 使用 `command -v python`，不要使用可能被集群 shell 包装的 `which python`。
- 确认实际 Python 路径就是预期 Conda 环境。
- 确认关键 import、CUDA 和模型构造成功。
- 确认镜像、overlay、Conda 初始化脚本和环境目录均可读。
- `bash -n` 通过不代表容器内依赖正确，容器内检查不能省略。

## 3. 验证数据和权重依赖

- `MEMNAV_ROOT_DIR`、`MEMNAV_FEATURE_ROOT`、`LINGBOT_REPO`、`LINGBOT_WEIGHTS` 和 `MEMNAV_DINO_WEIGHTS` 都是预期的绝对路径。
- 记录 LingBot、DINO 等关键权重的 SHA256；加载结果应为预期的 missing/unexpected keys。
- 严格特征覆盖必须开启，不能为了让任务启动而临时关闭：

```bash
MEMNAV_STRICT_FEATURE_COVERAGE=1
MEMNAV_REQUIRE_GENERATED_POSE_CONVENTION=1
```

- 每个 source-ready episode 必须同时具有 `lingbot_cache.npz` 和 `lingbot_cam_cache.npz`。
- 缓存帧数、parquet/RGB 帧数和 goal 索引必须一致。
- `MEMNAV_WINDOW`、`MEMNAV_NUM_SCALE` 和 `MEMNAV_MAX_FRAME_NUM` 必须与预计算一致；`MAX_FRAME_NUM` 必须覆盖最长轨迹。
- 预计算日志必须显示 `errors=0`，且 `sacct` 状态为 `COMPLETED`、`ExitCode=0:0`。脚本内部打印错误但 Slurm 显示完成，不能视为成功。
- 输出、日志和 checkpoint 目录必须在 `sbatch` 前创建且可写；Slurm 会在脚本正文执行前打开 stdout/stderr 文件。

## 4. 强制零步预检

预检必须复用正式训练的 `.sbatch`，只缩短训练本身，例如：

```bash
NAME=memnav_preflight_<commit> \
MEMNAV_REPORT_TO=none \
BATCH_SIZE=2 EPOCHS=0 NUM_WORKERS=0 \
sbatch --time=00:30:00 --export=ALL \
  scripts/train_memnav/train_memnav_mp3d.sbatch
```

预检至少要完成以下过程：

- 容器挂载和 Conda 激活；
- Python、PyTorch 和 CUDA 检查；
- train/validation 数据集的严格扫描及 fingerprint 生成；
- LingBot 和 novel backbone 权重加载；
- Trainer/model 构造并以零训练步正常退出。

只有预检最终为 `COMPLETED`、`ExitCode=0:0` 才算通过。仅看到 `RUNNING`、环境检查的前几行，或 Slurm 已分配 GPU，都不算通过。

## 5. 用 `afterok` 提交长任务

可以先把长任务排入队列，但它必须依赖预检成功：

```bash
PREFLIGHT_JOB=$(sbatch --parsable <preflight-options> <script>)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}

TRAIN_JOB=$(sbatch --parsable \
  --dependency="afterok:${PREFLIGHT_JOB}" \
  --time=8:00:00 \
  <training-options> <script>)
TRAIN_JOB=${TRAIN_JOB%%;*}

scontrol show job "${TRAIN_JOB}"
```

提交后必须从 `scontrol show job` 核对：

- `Dependency=afterok:<正确的预检 JobID>`；
- `TimeLimit`、`WorkDir`、`StdOut`、`StdErr`；
- account、partition、GPU、CPU 和内存；
- 实际导出的数据、权重、run name 和 resume 参数。

若还有缓存预计算等前置任务，依赖链应为：

```text
cache/precompute --afterok--> zero-step preflight --afterok--> long training
```

不要使用 `afterany` 代替 `afterok`。

## 6. 提交后验证

使用以下命令区分“排队”“运行”和“成功”：

```bash
squeue -j <job_ids> -o '%.18i %.24j %.10T %.10M %.9l %.26R'
sacct -j <job_ids> -X --format=JobID,State,Elapsed,ExitCode,NodeList -P
squeue --start -j <job_id>
```

- `PENDING (Dependency)`：正在正确等待前置任务。
- `PENDING (QOSGrpGRES)`：等待 GPU 配额，不是代码依赖失败。
- `RUNNING`：仍需检查日志是否已经进入 Python、数据集和模型前向。
- `COMPLETED`：仍需核对应用日志、checkpoint 和 W&B；不能只看 Slurm 状态。

正式训练启动后还要确认：

- 日志中的 Python/torch/CUDA、数据 fingerprint 和权重加载结果正确；
- GPU 确实有显存占用和计算利用率；
- 训练 loss、验证指标和 W&B 已开始更新；
- 到达保存步后生成完整 checkpoint（模型、optimizer、scheduler、RNG 和 metadata）；
- resume 指向同一数据与评测 fingerprint。

## 7. 每次提交必须留下的记录

```text
commit:
deployment directory:
dataset/feature root and fingerprint:
weight paths and SHA256:
container / Conda / Python:
preflight JobID, final state, exit code:
long JobID and exact afterok dependency:
time limit and resources:
W&B run ID:
stdout / stderr paths:
first valid checkpoint:
```

如果任一项未知或尚未通过，应明确报告“等待预检”或“被依赖阻塞”，不能表述为“训练已经正常运行”。

## 已知事故教训

- 逐 episode 异常曾被预计算脚本捕获后继续运行，导致 Slurm 显示成功但缓存不完整；现在要求 `errors=0` 和非零失败退出。
- 集群容器中的 `which` 包装函数曾因不兼容参数使任务在 Python 前退出；统一使用 `command -v`。
- 数据未补齐时直接训练会改变数据集组成；必须开启 strict coverage，并通过 `afterok` 串联补算、预检和训练。
- `QOSGrpGRES` 只是 GPU 配额等待；应查看预计开始时间，不能重复提交相同任务来“解决”。
