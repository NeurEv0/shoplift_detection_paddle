<#
.SYNOPSIS
    启动 labelImg 进行容器检测（container_det）标注：图片目录 / 类别文件 / 保存目录 一键配好。

.DESCRIPTION
    pip 版 labelImg 支持启动参数：
        labelImg <图片目录> <类别文件> <保存目录>
    本脚本自动传入：
        图片目录 = <序列根>\images\full
        类别文件 = datasets\container_det\label_list.txt（9 类，与转换脚本共用同一份，保持同步）
        保存目录 = <序列根>\annotations（不存在则自动创建，PascalVOC XML 输出）
    内置 PyQt5 >= 5.15.10 兼容补丁：该版本 QSpinBox/QScrollBar.setValue 严格要求 int，
    而 labelImg 的缩放/滚动代码传 float 会直接崩溃，脚本在启动前包一层 int() 修复。
    标注规则见 docs/container_labeling/manual_labeling_spec.md。

.PARAMETER Sequence
    序列目录名，位于 datasets_annotation\container_det\<Sequence>，默认 yulong_store_stride90。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\launch_labelimg_container_det.ps1
    powershell -ExecutionPolicy Bypass -File scripts\launch_labelimg_container_det.ps1 -Sequence renmin_store_stride60
#>

[CmdletBinding()]
param(
    [string]$Sequence = "yulong_store_stride90",
    [string]$CondaEnv = "C:\Users\YLHP\AppData\Local\miniconda3\envs\shoplift-paddle"
)

$ErrorActionPreference = "Stop"

# ---------- 用户配置区（Workspace 由脚本位置自动推导，机器无关） ----------
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LabelList = "$Workspace\datasets\container_det\label_list.txt"
# --------------------------------------------------------------------

$LabelImgExe   = "$CondaEnv\Scripts\labelImg.exe"
$SequenceRoot  = "$Workspace\datasets_annotation\container_det\$Sequence"
$ImageDir      = "$SequenceRoot\images\full"
$SaveDir       = "$SequenceRoot\annotations"

# 前置检查（挂载盘临时不可用时 Test-Path 会抛错，用 SilentlyContinue 转为友好提示）
if (-not (Test-Path -LiteralPath $LabelImgExe -ErrorAction SilentlyContinue)) {
    Write-Error "未找到 labelImg：$LabelImgExe`n请先安装：python -m pip install labelImg"
}
if (-not (Test-Path -LiteralPath $ImageDir -ErrorAction SilentlyContinue)) {
    Write-Error "图片目录不存在：$ImageDir`n请检查 -Sequence 参数，或确认原始帧已就位。"
}
if (-not (Test-Path -LiteralPath $LabelList -ErrorAction SilentlyContinue)) {
    Write-Error "类别文件不存在：$LabelList"
}

# 标注输出目录不存在则创建
if (-not (Test-Path -LiteralPath $SaveDir -ErrorAction SilentlyContinue)) {
    New-Item -ItemType Directory -Path $SaveDir -Force | Out-Null
    Write-Host "[创建] 标注输出目录：$SaveDir"
}

$classCount = (Get-Content -LiteralPath $LabelList | Where-Object { $_.Trim() -ne "" }).Count

Write-Host ""
Write-Host "== 启动 labelImg 容器标注（container_det）=="
Write-Host "  图片目录 : $ImageDir"
Write-Host "  类别文件 : $LabelList（$classCount 类）"
Write-Host "  保存目录 : $SaveDir"
Write-Host "  标注规则 : docs/container_labeling/manual_labeling_spec.md"
Write-Host ""

$Python = "$CondaEnv\python.exe"

# 内置 PyQt5 >= 5.15.10 兼容补丁：QSpinBox/QScrollBar.setValue 严格要求 int，
# 而 labelImg 的缩放/滚动代码传 float 会崩溃（TypeError: ... unexpected type 'float'）。
# 启动前把两个类的 setValue 包一层 int()（对已有 int 参数无影响）。
$code = @'
import sys
from PyQt5.QtWidgets import QSpinBox, QScrollBar

_orig_spin = QSpinBox.setValue
_orig_scroll = QScrollBar.setValue
QSpinBox.setValue = lambda self, value: _orig_spin(self, int(value))
QScrollBar.setValue = lambda self, value: _orig_scroll(self, int(value))

from labelImg import labelImg
sys.exit(labelImg.main())
'@

& $Python -c $code $ImageDir $LabelList $SaveDir
