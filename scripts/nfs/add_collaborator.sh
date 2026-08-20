#!/usr/bin/env bash
# add_collaborator.sh - 在 GPU 主机（NFS 服务端）为 Windows 协作者开通 outputs 专属目录。
#
# 用法（服务器上，需要 sudo）：
#   sudo bash scripts/nfs/add_collaborator.sh <缩写> <客户端IP> [UID]
# 例：
#   sudo bash scripts/nfs/add_collaborator.sh wzf 10.200.2.244
#
# 完成的操作：
#   1. 创建系统账号 <缩写>（UID 自动分配或指定），无 home、无 shell 登录需求；
#   2. 创建 outputs/<缩写>/ 目录：属主 ubuntu:ubuntu、770，
#      ACL：<缩写> rwx、ubuntu rwx、other 无（default ACL 继承到新文件）；
#   3. 在 /etc/exports 的 outputs 行追加该 IP 的 rw 规则
#      （all_squash, anonuid=<UID>, anongid=1000），网段 ro 兜底不变；
#   4. exportfs -ra 生效。
#
# 权限模型（实测于 Win11 客户端 + Linux NFS 服务端）：
#   - 目录是"门"：770 + ACL 只允许本人与 ubuntu，其他协作者被服务器拒绝进入；
#   - 文件层面 Windows NFS 客户端有本地权限模拟（匿名凭据落到 other 类），
#     因此客户端挂载 outputs 必须用 fileaccess=777（见 mount_nfs_shares.ps1），
#     新文件对 other 可读写；隔离由目录门保证，ubuntu 通过目录属主管理一切。
set -euo pipefail

NAME="${1:?usage: add_collaborator.sh <缩写> <客户端IP> [UID]}"
CLIENT_IP="${2:?usage: add_collaborator.sh <缩写> <客户端IP> [UID]}"
UID_REQ="${3:-}"

ROOT="/home/ubuntu/data_1t/shoplift_detection_paddle"
OUT="$ROOT/outputs"
EXPORTS="/etc/exports"
EXPORT_LINE_BASE="$OUT 10.200.0.0/16(ro,sync,no_subtree_check,insecure)"

# --- 0) 校验 ---
[[ "$NAME" =~ ^[a-zA-Z0-9_]{1,16}$ ]] || { echo "缩写只能含字母数字下划线"; exit 1; }
[[ "$CLIENT_IP" =~ ^[0-9.]+$ ]] || { echo "IP 格式不对"; exit 1; }

# --- 1) 账号 ---
if id "$NAME" &>/dev/null; then
  echo "账号 $NAME 已存在"
else
  if [[ -n "$UID_REQ" ]]; then
    useradd --no-create-home -u "$UID_REQ" "$NAME"
  else
    useradd --no-create-home "$NAME"
  fi
  echo "已创建账号 $NAME"
fi
UID_NUM=$(id -u "$NAME")

# --- 2) 专属目录 + ACL（门：本人+ubuntu，other 无；default 继承写权限给文件） ---
mkdir -p "$OUT/$NAME"
chown ubuntu:ubuntu "$OUT/$NAME"
chmod 770 "$OUT/$NAME"
setfacl -m u:"$NAME":rwx -m u:ubuntu:rwx -m m::rwx "$OUT/$NAME"
setfacl -m d:u:"$NAME":rwx -m d:u:ubuntu:rwx -m d:other::rwx -m d:m::rwx "$OUT/$NAME"
echo "目录 $OUT/$NAME 就绪（770 + ACL）"

# --- 3) exports：把该 IP 的 rw 规则插入 outputs 行（放在 ro 兜底之前） ---
if grep -qF "$CLIENT_IP(rw" "$EXPORTS"; then
  echo "exports 已包含 $CLIENT_IP 的 rw 规则"
else
  sed -i "s#^\($OUT \)\(10.200.0.0/16(ro,sync,no_subtree_check,insecure)\)\$#\1$CLIENT_IP(rw,sync,no_subtree_check,insecure,all_squash,anonuid=$UID_NUM,anongid=1000) \2#" "$EXPORTS"
  echo "exports 已加入 $CLIENT_IP -> anonuid=$UID_NUM"
fi

# --- 4) 生效 + 验证 ---
exportfs -ra
showmount -e 127.0.0.1 | grep outputs
echo "完成：$NAME (uid=$UID_NUM) 的 outputs 专属目录已开通。"
echo "协作者 Windows 侧请使用 scripts/windows/mount_nfs_shares.ps1（X: 已带 fileaccess=777）。"
