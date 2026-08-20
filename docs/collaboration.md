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

> 本小节已在 Windows 11 专业版（Build 26200）实测验证通过（2026-08）。与 Ubuntu 一样：`datasets`/`models`/`outputs` 只读、`datasets_annotation` 可写（服务端 `all_squash → ubuntu`）。

要求：Windows 10/11 **Pro / Enterprise / Education**（家庭版 Home 不含 NFS 客户端，见 2.4）。

1）以**管理员**身份打开 PowerShell，启用 NFS 客户端相关功能（**三个都要启用**）：

```powershell
Enable-WindowsOptionalFeature -Online -FeatureName ServicesForNFS-ClientOnly -NoRestart
Enable-WindowsOptionalFeature -Online -FeatureName ClientForNFS-Infrastructure -All -NoRestart
Enable-WindowsOptionalFeature -Online -FeatureName NFS-Administration -All -NoRestart
```

> 或：控制面板 → 程序和功能 → 启用或关闭 Windows 功能 → 勾选「NFS 服务」相关项。
>
> **实测要点**：仅启用 `ServicesForNFS-ClientOnly` 是不够的——Windows 11 25H2 上 `mount.exe`/`umount.exe` 及 NFS 客户端驱动由 `ClientForNFS-Infrastructure` 提供（`showmount`/`nfsadmin` 由 `NFS-Administration` 提供）。三个功能启用后**无需重启**，`nfsclnt` 服务与 `NfsRdr` 驱动即自动运行。验证是否装全：

```powershell
Get-Command mount.exe   # 存在即正常
```

2）映射盘符（管理员 PowerShell）。**注意必须写 `mount.exe`**——PowerShell 内置别名把 `mount` 指向 `New-PSDrive`，直接写 `mount -o ...` 会报 `parameter name 'o' is ambiguous`。**outputs（X:）必须带 `fileaccess=777`**（原因见 2.5 权限模型；否则 Windows 客户端读不回/覆盖不了自己建的文件）：

```powershell
mount.exe -o anon \\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\datasets Z:
mount.exe -o anon \\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\models Y:
mount.exe -o anon,fileaccess=777 \\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\outputs X:
mount.exe -o anon \\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\datasets_annotation W:
```

3）查看/卸载：

```powershell
mount.exe     # 查看已挂载
umount.exe Z: # 卸载盘符
```

4）**建立仓库内路径（关键步骤）**：仓库配置（`env.local.yml`）与 CLI 全部使用**相对路径**（`./models`、`./outputs/<用户>/<任务>` 等），与服务器布局一致。Windows NFS 客户端**只能挂载到盘符**（实测 `mount.exe` 用法为 `mount ... <devicename|*>`，不支持目录挂载点；NTFS junction 也不能指向 NFS/远程卷）。因此需要把盘符**链接**进仓库目录，仓库才会按服务器同布局运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\link_nfs_to_repo.ps1
```

该脚本（幂等，可重复执行）：

- 在仓库根创建目录符号链接：`datasets → Z:\`、`models → Y:\`、`outputs → X:\`、`datasets_annotation → W:\`（此后 `./models/paddledetection/person_mot` 等相对路径直接可用）；
- 对 `datasets/`、`models/` 下 7 个 git 跟踪的轻量文件设置 `git update-index --skip-worktree`，并把 4 个符号链接名加入 `.git/info/exclude`（两者均为**本仓库本地设置，不进 Git**），保证 `git status` 依然干净。

> 符号链接创建需管理员权限（或用开发者模式）：首次以**管理员** PowerShell 运行本脚本一次即可，符号链接会持久保留；`git` 相关设置（skip-worktree / exclude）无需管理员。`datasets`/`models` 目录里被符号链接"遮住"的 git 跟踪文件内容与服务器共享内一致（服务器仓库与共享同源），`git status` 不会出现删除/修改。

5）（可选）开机自动挂载：注册计划任务 `ShopliftNFSMount`。**任务必须以非提权（RunLevel Limited）运行**——Windows NFS 盘符按登录会话隔离，提权会话里挂载的盘符在资源管理器/普通程序里看不到（点开仓库目录会报"位置不可用"）。`mount.exe` 无需管理员权限，因此任务非提权即可正常挂载（符号链接已由第 4 步建好，任务只负责重挂盘符与重放 git 设置）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\register_nfs_mount_task.ps1
# 手动执行挂载/验证（普通 PowerShell 即可）：
powershell -ExecutionPolicy Bypass -File scripts\windows\mount_nfs_shares.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\verify_nfs_mount.ps1
```

> 若本机已存在旧版（提权）任务导致注册报"拒绝访问"，用管理员 PowerShell 重跑一次注册脚本覆盖即可。

注意：

- Windows NFS 客户端默认以匿名身份访问（`UID=-2/GID=-2`）；服务端对 `datasets_annotation` 已配置 `all_squash → ubuntu`，匿名访问即可读写。
- 只读共享（`datasets`/`models`/`outputs`）由服务端强制只读（客户端写入报"介质受写入保护"属正常）。
- Windows NFS 挂载**重启后不保留**，需要重新挂载（用第 5 步的计划任务即可；符号链接本身会保留，盘符挂上后自动恢复解析）。
- 若符号链接因盘符未挂载暂时不可解析属正常：先执行挂载，再访问仓库相对路径。
- 若 `mount` 报"此命令的另一个实例已在运行"：有残留的 `mount.exe`/`umount.exe` 进程（常见于卸载时仍有文件句柄），结束该进程后重试。

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

### 2.6 outputs 专属目录权限模型（Windows 协作者）

每位协作者在 `outputs/` 下有自己的专属目录（**名字缩写**，如 `outputs/wzf/`）：**本人对自己的目录有读写权限，对其余目录（含 `ubuntu/` 与其他协作者的目录）只有只读权限**；服务器（ubuntu）拥有全部管理权。该模型已在 Win11 客户端 + Linux NFS 服务端实测验证。

**开通方式**（GPU 主机上，管理员执行一次；脚本自动完成建账号/建目录/ACL/导出规则）：

```bash
# 在服务器仓库目录下
sudo bash scripts/nfs/add_collaborator.sh <缩写> <客户端IP>   # 如：sudo bash scripts/nfs/add_collaborator.sh wzf 10.200.2.244
```

脚本完成：

- 创建系统账号 `<缩写>`（如 `wzf`，UID 自动分配；用于 NFS 身份映射，无登录需求）；
- 创建 `outputs/<缩写>/`：属主 `ubuntu:ubuntu`、权限 775 + ACL（`<缩写>` rwx、`ubuntu` rwx、others **r-x**，default ACL 继承）；
- 在 `/etc/exports` 的 outputs 行追加该客户端 IP 的 **rw** 规则（`all_squash, anonuid=<UID>, anongid=1000`），**未登记 IP 仍走 ro 兜底（内核级只读）**；
- `exportfs -ra` 生效。

**权限模型要点**（实测）：

- **本人读写自己的目录**：`outputs/<缩写>/` 775 + ACL（本人 rwx）；文件由客户端 `fileaccess=777` 创建（777 对 others 开放是 Windows NFS 客户端本地权限模拟的需要——匿名凭据落到 "other" 类，`mount_nfs_shares.ps1` 已内置）；
- **其余目录只读（双层保证）**：① 未登记 IP 挂载 outputs 为 **ro 导出**（内核级，写入直接报 "Read-only file system"）；② 目录权限 others=r-x（`outputs/` 根 755、`outputs/ubuntu/` 755、各协作者目录 775），服务器拒绝任何非本人的写入；
- **ubuntu 管理权**：ubuntu 是各目录属主，无需 sudo 即可增删改任何协作者目录下的内容；root 当然全权；
- **已知限制**（Windows NFS 客户端）：协作者可创建/重命名/读写删除**文件**与创建子目录，但**无法删除子目录**（客户端本地检查拦截 RMDIR）——需要删目录时请 ubuntu 管理员在服务器执行 `rm -rf outputs/<缩写>/<目录>`；
- **服务器本地注意**：`/home/ubuntu` 是 750，普通协作者账号无法在服务器本地进入数据目录（NFS 访问不受影响），如需服务器本地访问请管理员调整。

## 3. GPU 算力约定

- 只有**训练 / GPU 推理**才在 GPU 主机执行；日常编码、单测在各自开发机完成。

## 4. 常见问题

- 挂载报 `access denied by server`：确认本机 IP 在 `10.200.0.0/16` 网段内，且能 ping 通 `10.200.10.10`。也可用 `showmount.exe -e 10.200.10.10` 确认服务端导出列表。
- Windows 上执行 `mount -o ...` 报 `parameter name 'o' is ambiguous`：PowerShell 的 `mount` 是 `New-PSDrive` 别名，必须写 `mount.exe`（或 `cmd /c "mount -o ..."`）。
- Windows 上功能已启用但找不到 `mount.exe`：还缺 `ClientForNFS-Infrastructure` 功能（Win11 25H2 实测必需），按 2.3 第 1 步全部启用。
- Windows 上仓库相对路径（如 `./models/paddledetection/...`）访问不到数据：Windows NFS 只能挂盘符，需按 2.3 第 4 步执行 `link_nfs_to_repo.ps1` 建立 `models` 等目录符号链接；`git status` 的干净由 `skip-worktree` + `.git/info/exclude` 保证（本机设置，不影响他人）。
- Windows 资源管理器点击仓库的 `datasets`/`models` 等目录报"位置不可用"：挂载发生在**提权会话**里，普通会话看不到 NFS 盘符（盘符按登录会话隔离）。用第 5 步的**非提权**计划任务（或普通 PowerShell 手动执行 `mount_nfs_shares.ps1`）重新挂载即可；若手动挂载报"另一个实例已在运行"，先结束残留的 `mount.exe`/`umount.exe` 进程。
- Windows 上 `mklink /J` 报"完成该操作需要本地卷"、或 junction 访问报"重分析点缓冲区中的数据无效"：NTFS junction 不能指向 NFS/远程卷，属系统限制，请改用 `mklink /D` 符号链接（`link_nfs_to_repo.ps1` 已自动处理）。
- `git clone` / `pip install` 超时或 SSL 报错、但浏览器能打开 GitHub：本机有系统代理时 git/pip 不读注册表代理，需显式配置：`git config --global http.proxy http://127.0.0.1:7897`（端口按本机代理改），pip 侧设环境变量 `HTTP_PROXY`/`HTTPS_PROXY`。
- `pre-commit run --all-files` 在 `src/PaddleDetection-release-2.9` 下报大量 ruff 错误：该目录是 vendored 第三方代码，不在协作 lint 范围；提交前自检用 `ruff check shoplift scripts`，pre-commit 钩子只检查本次暂存的文件。
- 读写共享里新建文件属主是 `ubuntu`：这是服务端 `all_squash` 的预期行为，属正常。
- 改了 `shoplift/configs/env.local.yml`：该文件已被 `.gitignore` 排除，不会进 Git。
