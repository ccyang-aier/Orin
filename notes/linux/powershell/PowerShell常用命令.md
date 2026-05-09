## 在PowerShell中查看和设置环境变量

### 查看环境变量

```powershell
# 查看单个环境变量
$env:ANTHROPIC_MODEL

# 或者用 Get-Item
Get-Item Env:ANTHROPIC_MODEL

# 查看所有环境变量
Get-ChildItem Env:
```

### 设置环境变量

#### 1. 仅当前会话有效（临时）

关闭PowerShell后失效：
```powershell
$env:ANTHROPIC_MODEL = "claude-sonnet-4-5"
```

#### 2. 永久设置（用户级别）
写入注册表，对当前用户永久生效：
```powershell
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_MODEL", "claude-sonnet-4-5", "User")
```

#### 3. 永久设置（系统级别）
对所有用户生效，**需要管理员权限**：
```powershell
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_MODEL", "claude-sonnet-4-5", "Machine")
```

> 永久设置后，需要**重新打开** PowerShell 窗口才能在新会话中生效。

### 删除环境变量

```powershell
# 删除当前会话中的变量
Remove-Item Env:ANTHROPIC_MODEL

# 永久删除（用户级别）
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_MODEL", $null, "User")
```

## 查看永久设置的环境变量

### 查看用户级别环境变量

```powershell
# 查看单个用户级变量
[System.Environment]::GetEnvironmentVariable("ANTHROPIC_MODEL", "User")

# 查看所有用户级变量
[System.Environment]::GetEnvironmentVariables("User")
```

### 查看系统级别环境变量

```powershell
# 查看单个系统级变量
[System.Environment]::GetEnvironmentVariable("ANTHROPIC_MODEL", "Machine")

# 查看所有系统级变量
[System.Environment]::GetEnvironmentVariables("Machine")
```

### 通过图形界面查看（更直观）

**方法一：系统设置**
```
Win + R → 输入 sysdm.cpl → 高级 → 环境变量
```

**方法二：直接打开环境变量窗口**
```powershell
# 在 PowerShell 中直接打开 GUI
rundll32 sysdm.cpl,EditEnvironmentVariables
```

### 区别说明

| 范围  | 存储位置（注册表）                                                            |
| --- | -------------------------------------------------------------------- |
| 用户级 | `HKCU:\Environment`                                                  |
| 系统级 | `HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment` |

```powershell
# 也可以直接从注册表读取
# 用户级
Get-ItemProperty "HKCU:\Environment"

# 系统级（需管理员）
Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
```

> **注意**：`$env:ANTHROPIC_MODEL` 读取的是当前会话合并后的值（系统级 + 用户级），无法区分来源，所以要确认来源必须用上面的方法。

