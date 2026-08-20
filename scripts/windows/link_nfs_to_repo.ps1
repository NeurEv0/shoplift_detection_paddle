# link_nfs_to_repo.ps1
# Windows NFS client can only mount to DRIVE LETTERS (directory mounts are not
# supported; NTFS junctions cannot target remote/NFS volumes; git does not
# traverse symlinked dirs). To give the repo the same relative layout as the
# GPU server (datasets/models/outputs/datasets_annotation as repo subdirs, as
# referenced by env.local.yml and CLI relative paths), this script:
#   1. creates directory SYMLINKS  repo\<dir> -> <mounted drive>\
#   2. marks the 7 git-tracked lightweight files under datasets/models as
#      skip-worktree (git then ignores their working-tree state, so git status
#      stays clean even though the symlinks hide them)
#   3. adds the symlink names to .git/info/exclude (local-only, not committed)
# Run ELEVATED after mount_nfs_shares.ps1. Idempotent.
$ErrorActionPreference = 'Stop'
$log = Join-Path $env:TEMP 'nfs_link.log'
function Log($msg) { $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg; $line | Tee-Object -FilePath $log -Append }

# Repo root = parent of scripts/windows/ (derive, do not hardcode)
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$links = @(
    @{ Name = 'datasets';            Target = 'Z:\' },
    @{ Name = 'models';              Target = 'Y:\' },
    @{ Name = 'outputs';             Target = 'X:\' },
    @{ Name = 'datasets_annotation'; Target = 'W:\' }
)

Log "=== link NFS drives into repo ($repo) ==="

# 1) Create directory symlinks (remove any existing reparse point or dir first)
foreach ($l in $links) {
    $path = Join-Path $repo $l.Name
    $item = Get-Item $path -Force -ErrorAction SilentlyContinue
    if ($item) {
        $isSymlink = ($item.LinkType -eq 'SymbolicLink') -and ([bool](Test-Path $path))
        if ($isSymlink) {
            Log "SKIP $($l.Name): already a working symlink"
            continue
        }
        Log "REMOVE $($l.Name): removing existing $(if ($item.LinkType) { $item.LinkType } else { 'directory' }) to create symlink"
        Remove-Item $path -Recurse -Force
    }
    if (-not (Test-Path $l.Target)) {
        Log "WARN $($l.Name): target $($l.Target) not mounted - symlink created anyway"
    }
    try {
        cmd /c mklink /D "$path" "$($l.Target)" 2>&1 | ForEach-Object { Log "  $_" }
        Log "LINK $($l.Name) -> $($l.Target) : resolves=$([bool](Test-Path $path))"
    } catch {
        # Symlink creation needs admin (or Developer Mode); mounts themselves do not.
        # If symlinks already exist (created once elevated), this is harmless.
        Log "WARN $($l.Name): could not create symlink without elevation: $($_.Exception.Message)"
    }
}

# 2) Mark git-tracked lightweight files under datasets/models as skip-worktree
$tracked = @(
    'datasets/container_det/README.md',
    'datasets/container_det/container_labeling_spec.md',
    'datasets/container_det/label_list.txt',
    'datasets/person_attribute/README.md',
    'models/pretrained/person_attribute/README.md',
    'models/shoplift/item_container/README.md',
    'models/shoplift/person_attribute/README.md'
)
foreach ($f in $tracked) {
    $skip = git -C $repo ls-files -v -- $f 2>&1
    if ($skip -match '^S') {
        Log "SKIP-WORKTREE already: $f"
    } else {
        git -C $repo update-index --skip-worktree -- $f 2>&1 | ForEach-Object { Log "  $_" }
        Log "SKIP-WORKTREE set: $f"
    }
}

# 3) Exclude symlink names from untracked scan (local-only)
$exclude = Join-Path $repo '.git\info\exclude'
$patterns = @('datasets', 'models', 'outputs', 'datasets_annotation')
$content = if (Test-Path $exclude) { Get-Content $exclude -Raw } else { '' }
foreach ($p in $patterns) {
    if ($content -notmatch "(?m)^\s*$([regex]::Escape($p))\s*$") {
        Add-Content -Path $exclude -Value $p -Encoding utf8
        Log "EXCLUDE added: $p"
    } else {
        Log "EXCLUDE already: $p"
    }
}

Log "=== git status after linking ==="
git -C $repo status --short 2>&1 | ForEach-Object { Log "  $_" }
Log "=== DONE ==="
