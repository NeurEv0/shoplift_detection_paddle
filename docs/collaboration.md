# 多人协作开发指南

本文档说明 `shoplift_detection_paddle` 的多人协作方式：**代码走 Git、算力走 GPU 主机、数据/权重走局域网 NFS 共享**。

## 0. 角色与主机

| 主机 | 角色 | 说明 |
|---|---|---|
| `10.200.10.10` | GPU 主机（唯一） | RTX 5070 Ti 16G；训练/GPU 推理、NFS 服务端 |
| 各协作者主机 | 开发机（CPU） | Ubuntu 或 Windows；编码、单测，通过 NFS 只读访问数据/权重 |

三条原则：

- **代码**：通过 GitHub 协作（feature 分支 + Pull Request）。协作者无 GPU 也能完成开发与单测。
- **算力**：集中在 GPU 主机执行。只有训练/GPU 推理才提交到 GPU 机。
- **数据/权重/输出**：通过 NFS 共享，只存一份，不复制、不进 Git。

## 快速上手（新协作者）

一次性准备，按顺序执行：

1. **权限**：获得 `NeurEv0/shoplift_detection_paddle` 访问权；主机在 `10.200.0.0/16` 网段内。

2. **代码与环境**：

   ```bash
   git clone https://github.com/NeurEv0/shoplift_detection_paddle.git
   cd shoplift_detection_paddle
   conda env create -f environment.yml && conda activate shoplift-paddle
   pip install -e src/PaddleDetection-release-2.9
   cp shoplift/configs/env.example.yml shoplift/configs/env.local.yml   # Windows: Copy-Item
   python scripts/check_env.py --config shoplift/configs/env.local.yml
   pip install pre-commit && pre-commit install
   ```

3. **挂载共享数据（NFS）**：Ubuntu 见 2.2，Windows 见 2.3；`datasets`/`models`/`outputs` 只读，`datasets_annotation` 可写。

4. **设按人输出**：

   ```bash
   export SHOPLIFT_USER=你的GitHub用户名   # 写入 ~/.bashrc / 系统环境变量
   ```

5. **日常开发**：`git checkout -b feature/xxx` → 开发 → `python -m pytest shoplift/tests -q` + `ruff check shoplift scripts` → `git push` → 开 PR（CI 自动检查）→ 合入 `main`。

## 1. 代码协作

### 1.1 分支模型

- `main` 始终可运行，所有改动通过 feature 分支 + Pull Request 合入。
- 合并前 CI 自动跑：环境自检 + `ruff` + 单元测试。

### 1.2 克隆仓库

```bash
git clone https://github.com/NeurEv0/shoplift_detection_paddle.git
cd shoplift_detection_paddle
```

### 1.3 环境准备（CPU 开发机）

```bash
conda env create -f environment.yml
conda activate shoplift-paddle
pip install -e src/PaddleDetection-release-2.9
```

环境自检：

```bash
cp shoplift/configs/env.example.yml shoplift/configs/env.local.yml   # Windows 用: Copy-Item
python scripts/check_env.py --config shoplift/configs/env.local.yml
```

> `env.local.yml` 已被 `.gitignore` 排除，可按本机路径自由修改，不会进 Git。

### 1.4 启用 pre-commit（首次 clone 后执行一次）

```bash
pip install pre-commit
pre-commit install
```

### 1.5 提交前自检

```bash
python -m pytest shoplift/tests -q
ruff check shoplift scripts
```

GitHub Actions 会在每次 PR 自动执行同样的检查，本地先跑一遍能减少往返。

## 2. 数据/权重共享（NFS）

### 2.1 共享清单

NFS 服务端：`10.200.10.10`，仅对局域网 `10.200.0.0/16` 开放。
导出根：`/home/ubuntu/data_1t/shoplift_detection_paddle/<目录>`。

| 目录 | 权限 | 用途 |
|---|---|---|
| `datasets` | 只读 | 原始数据集 |
| `models` | 只读 | 模型权重 |
| `outputs` | 只读 | 推理输出/调试结果 |
| `datasets_annotation` | 读写 | 标注工作区（写操作统一映射为 `ubuntu` 用户） |

### 2.2 Ubuntu 挂载

```bash
# 1) 安装 NFS 客户端
sudo apt-get update && sudo apt-get install -y nfs-common

# 2) 进入仓库 clone，挂到被 gitignore 的目录（相对路径配置可直接使用）
cd /path/to/shoplift_detection_paddle
sudo mkdir -p datasets models outputs datasets_annotation
sudo mount -t nfs 10.200.10.10:/home/ubuntu/data_1t/shoplift_detection_paddle/datasets  datasets
sudo mount -t nfs 10.200.10.10:/home/ubuntu/data_1t/shoplift_detection_paddle/models   models
sudo mount -t nfs 10.200.10.10:/home/ubuntu/data_1t/shoplift_detection_paddle/outputs  outputs
sudo mount -t nfs 10.200.10.10:/home/ubuntu/data_1t/shoplift_detection_paddle/datasets_annotation datasets_annotation
```

开机自动挂载：把下面内容加入 `/etc/fstab`（`<clone>` 换成你的仓库绝对路径）：

```
10.200.10.10:/home/ubuntu/data_1t/shoplift_detection_paddle/datasets            <clone>/datasets            nfs ro,nofail,x-systemd.automount 0 0
10.200.10.10:/home/ubuntu/data_1t/shoplift_detection_paddle/models             <clone>/models             nfs ro,nofail,x-systemd.automount 0 0
10.200.10.10:/home/ubuntu/data_1t/shoplift_detection_paddle/outputs            <clone>/outputs            nfs ro,nofail,x-systemd.automount 0 0
10.200.10.10:/home/ubuntu/data_1t/shoplift_detection_paddle/datasets_annotation <clone>/datasets_annotation nfs rw,nofail,x-systemd.automount 0 0
```

### 2.3 Windows 挂载

要求：Windows 10/11 **Pro / Enterprise / Education**（家庭版 Home 不含 NFS 客户端，见 2.4）。

1）启用「NFS 客户端」功能（管理员 PowerShell，可能需要重启）：

```powershell
Enable-WindowsOptionalFeature -Online -FeatureName ServicesForNFS-ClientOnly -NoRestart
```

> 或：控制面板 → 程序和功能 → 启用或关闭 Windows 功能 → 勾选「NFS 服务」→「NFS 客户端」。

2）以**管理员**身份打开 PowerShell，映射盘符：

```powershell
mount -o anon \\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\datasets Z:
mount -o anon \\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\models Y:
mount -o anon \\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\outputs X:
mount -o anon \\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\datasets_annotation W:
```

3）查看/卸载：

```powershell
mount     # 查看已挂载
umount Z: # 卸载盘符
```

注意：

- Windows NFS 客户端默认以匿名身份访问；服务端对 `datasets_annotation` 已配置 `all_squash → ubuntu`，匿名访问即可读写。
- 只读共享（`datasets`/`models`/`outputs`）由服务端强制只读。
- Windows NFS 挂载**重启后不保留**，需要重新 `mount`（可写一个登录启动脚本）。

### 2.4 Windows 家庭版（Home）替代方案

Windows Home 无内置 NFS 客户端，可选：

- 使用 WSL2（Ubuntu）按 2.2 方式挂载；
- 或安装第三方 NFS 客户端。

### 2.5 输出目录约定（按人分目录）

为避免多人互相覆盖，所有产出统一按 `outputs/<用户>/<任务>/` 组织：

- `<用户>` 默认取运行环境的 `$USER`（GPU 机上是 `ubuntu`、`ljh` 等系统账号）；也可用 `SHOPLIFT_USER` 显式指定（推荐用 GitHub 用户名）：

  ```bash
  export SHOPLIFT_USER=your_github_name
  ```

- `<任务>` 由各配置里的 `outputs:` 路径决定（如 `shoplift`、`inference_visualization/test_videos`）。

`offline_analyze` 默认输出示例（`$USER=ubuntu`）：

```text
outputs/ubuntu/shoplift/frame_results.jsonl
outputs/ubuntu/shoplift/events.json
outputs/ubuntu/shoplift/debug/
```

`--output <dir>` 会完全覆盖默认，直接写到指定目录：

```bash
python -m shoplift.cli.offline_analyze --output outputs/<你的名字>/<任务> ...
```

其它 CLI（`video_infer_visualize`、`hand_crop_visualize`、`dcsass_eval`）同样用 `--output outputs/<你的名字>/<任务>` 指定。后续 GPU 任务排队会按作业自动设置 `SHOPLIFT_USER`。

## 3. GPU 算力约定

- 只有**训练 / GPU 推理**才在 GPU 主机执行；日常编码、单测在各自开发机完成。

## 4. 常见问题

- 挂载报 `access denied by server`：确认本机 IP 在 `10.200.0.0/16` 网段内，且能 ping 通 `10.200.10.10`。
- 读写共享里新建文件属主是 `ubuntu`：这是服务端 `all_squash` 的预期行为，属正常。
- 改了 `shoplift/configs/env.local.yml`：该文件已被 `.gitignore` 排除，不会进 Git。
