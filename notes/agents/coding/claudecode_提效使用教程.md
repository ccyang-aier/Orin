# Claude Code 提效使用教程

> 基于 Claude Code 官方文档整理，涵盖安装、常用命令、技巧、环境变量、Claude Code UI 等内容。
> 最后更新：2026-05-10

---

## 一、安装 Claude Code

### 推荐方式：原生安装（支持自动更新）

```bash
# macOS / Linux / WSL
curl -fsSL https://claude.ai/install.sh | bash

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex
```

### 其他安装方式

```bash
# npm（需要 Node.js 18+）
npm install -g @anthropic-ai/claude-code

# Homebrew（不自动更新，需手动 brew upgrade）
brew install --cask claude-code

# WinGet
winget install Anthropic.ClaudeCode
```

### 安装特定版本 / 切换频道

```bash
claude install 2.1.89    # 安装指定版本
claude install stable    # 切换到 stable 频道
claude update            # 更新到最新版本
```

### 系统要求

- macOS 13+、Windows 10 1809+、Ubuntu 20.04+
- 4GB+ RAM，x64 或 ARM64
- 需要网络连接

### 登录认证

```bash
claude auth login          # 登录（浏览器 OAuth）
claude auth login --console  # 无浏览器环境登录
claude auth status         # 查看认证状态
claude auth logout         # 登出
```

---

## 二、常用 CLI 命令

### 启动方式

| 命令 | 说明 |
|------|------|
| `claude` | 启动交互式会话 |
| `claude "query"` | 带初始提示启动 |
| `claude -p "query"` | 非交互模式，输出后退出 |
| `cat file \| claude -p "query"` | 处理管道内容 |
| `claude -c` | 继续当前目录最近的会话 |
| `claude -r "session-name" "query"` | 按名称恢复会话 |

### 常用 CLI Flags

| Flag | 说明 |
|------|------|
| `--model <alias>` | 设置模型，如 `--model opus` |
| `--effort <level>` | 推理努力级别：`low/medium/high/xhigh/max` |
| `--permission-mode <mode>` | 权限模式：`default/acceptEdits/plan/auto` |
| `-p, --print` | 非交互模式 |
| `-c, --continue` | 继续最近的会话 |
| `-r, --resume <session>` | 按 ID 或名称恢复会话 |
| `-w, --worktree <name>` | 在隔离的 git worktree 中启动 |
| `--add-dir <path>` | 添加额外工作目录 |
| `--max-turns <N>` | 限制代理轮次（print mode） |
| `--max-budget-usd <N>` | 限制最大花费（print mode） |
| `--output-format <fmt>` | 输出格式：`text/json/stream-json` |
| `--debug` | 启用调试模式 |
| `--bare` | 最小模式，跳过 hooks/skills/MCP/CLAUDE.md |
| `--dangerously-skip-permissions` | 跳过所有权限提示（慎用） |

### 模型别名

| 别名 | 说明 |
|------|------|
| `sonnet` | 最新 Sonnet（日常编码，推荐） |
| `opus` | 最新 Opus（复杂推理） |
| `haiku` | 快速高效（简单任务） |
| `best` | 当前最强模型 |
| `default` | 恢复账号推荐模型 |
| `sonnet[1m]` | Sonnet + 100 万 token 上下文 |

---

## 三、会话内 Slash 命令

输入 `/` 可查看所有可用命令，输入 `/` 加字母可过滤。

### 项目初始化

| 命令 | 说明 |
|------|------|
| `/init` | 生成 CLAUDE.md 项目指南 |
| `/memory` | 编辑 CLAUDE.md，管理自动记忆 |
| `/permissions` | 管理工具权限规则 |
| `/mcp` | 管理 MCP 服务器连接 |
| `/hooks` | 查看 hook 配置 |

### 任务执行中

| 命令 | 说明 |
|------|------|
| `/plan [description]` | 进入计划模式 |
| `/model [model]` | 切换模型 |
| `/effort [level]` | 设置推理努力级别 |
| `/fast [on\|off]` | 切换快速模式 |
| `/context [all]` | 可视化上下文使用情况 |
| `/compact [instructions]` | 压缩对话历史释放上下文 |
| `/btw <question>` | 提问侧边问题（不加入对话历史） |
| `/add-dir <path>` | 添加工作目录 |

### 代码审查

| 命令 | 说明 |
|------|------|
| `/diff` | 查看未提交的变更 |
| `/review [PR]` | 本地审查 PR |
| `/security-review` | 安全漏洞分析 |
| `/simplify [focus]` | 审查并优化最近修改的文件 |

### 会话管理

| 命令 | 说明 |
|------|------|
| `/clear [name]` | 开始新对话（保留项目记忆） |
| `/resume [session]` | 恢复会话 |
| `/branch [name]` | 分叉当前对话 |
| `/rewind` | 回滚代码和对话到检查点 |
| `/rename [name]` | 重命名当前会话 |
| `/export [filename]` | 导出对话为纯文本 |
| `/recap` | 生成当前会话摘要 |

### 工具和配置

| 命令 | 说明 |
|------|------|
| `/config` | 打开设置界面 |
| `/status` | 查看版本、模型、账号状态 |
| `/usage` | 查看会话花费和用量统计 |
| `/doctor` | 诊断安装和设置问题 |
| `/theme` | 更改颜色主题 |
| `/keybindings` | 打开快捷键配置文件 |

### 自动化

| 命令 | 说明 |
|------|------|
| `/loop [interval] [prompt]` | 定期重复执行（如 `/loop 5m check deploy`） |
| `/batch <instruction>` | 大规模并行代码变更 |
| `/tasks` | 列出和管理后台任务 |

---

## 四、关键环境变量

### 认证和 API

| 变量 | 说明 |
|------|------|
| `ANTHROPIC_API_KEY` | API 密钥（覆盖订阅登录） |
| `ANTHROPIC_BASE_URL` | 覆盖 API 端点（代理/网关） |

### 模型选择

| 变量 | 说明 |
|------|------|
| `ANTHROPIC_MODEL` | 使用的模型名称或别名 |
| `CLAUDE_CODE_EFFORT_LEVEL` | 努力级别：`low/medium/high/xhigh/max/auto` |
| `CLAUDE_CODE_SUBAGENT_MODEL` | 子代理使用的模型 |

### 超时和限制

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_TIMEOUT_MS` | 600000 | API 请求超时（ms） |
| `BASH_DEFAULT_TIMEOUT_MS` | 120000 | Bash 命令默认超时 |
| `BASH_MAX_TIMEOUT_MS` | 600000 | Bash 命令最大超时 |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | 模型相关 | 每次请求最大输出 token |
| `CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY` | 10 | 最大并行工具调用数 |

### 禁用功能

| 变量 | 禁用内容 |
|------|---------|
| `CLAUDE_CODE_DISABLE_THINKING` | 扩展思考 |
| `CLAUDE_CODE_DISABLE_AUTO_MEMORY` | 自动记忆 |
| `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` | 后台任务 |
| `CLAUDE_CODE_DISABLE_CRON` | 定时任务 |
| `DISABLE_TELEMETRY` | 遥测数据上报 |
| `DISABLE_AUTOUPDATER` | 自动更新检查 |

### 上下文压缩

| 变量 | 说明 |
|------|------|
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | 触发压缩的上下文容量百分比（1-100） |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | 覆盖假设的上下文窗口大小 |

---

## 五、设置与配置（settings.json）

### 配置文件位置

| 文件 | 作用域 | 说明 |
|------|--------|------|
| `~/.claude/settings.json` | 用户级 | 所有项目的个人偏好 |
| `.claude/settings.json` | 项目级 | 通过 git 与团队共享 |
| `.claude/settings.local.json` | 本地级 | 个人项目覆盖，gitignore |

优先级：本地级 > 项目级 > 用户级

### 常用配置示例

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "model": "sonnet",
  "effortLevel": "high",
  "autoMemoryEnabled": true,
  "permissions": {
    "allow": [
      "Bash(npm run lint)",
      "Bash(npm run test *)",
      "Bash(git log *)",
      "Bash(git diff *)"
    ],
    "deny": [
      "Bash(curl *)",
      "Read(./.env)"
    ]
  },
  "env": {
    "DISABLE_TELEMETRY": "1"
  }
}
```

### 权限规则语法

```
Tool                          # 匹配工具的所有使用
Bash(npm run *)               # 匹配以 "npm run" 开头的命令
Read(./.env)                  # 匹配读取 .env 文件
WebFetch(domain:example.com)  # 匹配对特定域名的请求
MCP(server_name:tool_name)    # 匹配特定 MCP 工具
```

拒绝规则 > 询问规则 > 允许规则。

---

## 六、CLAUDE.md 最佳实践

CLAUDE.md 是给 Claude 的持久化指令文件，每次会话开始时自动加载。

### 文件位置

| 路径 | 作用域 |
|------|--------|
| `./CLAUDE.md` | 项目级（提交到 git） |
| `~/.claude/CLAUDE.md` | 用户级（所有项目） |
| `./CLAUDE.local.md` | 本地个人偏好（gitignore） |

### 有效写法示例

```markdown
# 项目概述
这是一个 React + TypeScript 项目。

# 构建命令
- 开发：`npm run dev`
- 测试：`npm run test`
- 构建：`npm run build`

# 代码规范
- 使用 2 空格缩进
- 所有函数必须有 TypeScript 类型注解

# 重要约定
- 提交前运行 `npm run lint`
- 使用 pnpm，不用 npm
```

**技巧：**
- 目标 200 行以内，过长会降低遵从度
- 使用 `@path/to/file` 语法导入其他文件内容
- HTML 注释 `<!-- -->` 会被过滤，不消耗 token
- 用 `/init` 命令让 Claude 自动生成初始 CLAUDE.md

---

## 七、Hooks（钩子）

Hooks 是在 Claude Code 生命周期特定时间点自动执行的 shell 命令。

### 常用 Hook 事件

| 事件 | 触发时机 |
|------|---------|
| `PostToolUse` | 工具执行后 |
| `PreToolUse` | 工具执行前 |
| `Notification` | Claude 等待输入时 |
| `Stop` | Claude 停止时 |
| `SessionStart` | 会话开始时 |

### 配置示例

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write"
          }
        ]
      }
    ],
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "osascript -e 'display notification \"Claude Code needs your attention\"'"
          }
        ]
      }
    ]
  }
}
```

**常见用途：**
- 文件编辑后自动格式化（Prettier、Black 等）
- Claude 等待输入时发送桌面通知
- 阻止对受保护文件的编辑
- 压缩后重新注入上下文

---

## 八、MCP 服务器集成

MCP（Model Context Protocol）让 Claude 连接外部工具、数据库和 API。

### 添加 MCP 服务器

```bash
# 添加远程 HTTP 服务器
claude mcp add my-server --transport http https://example.com/mcp

# 添加本地 stdio 服务器
claude mcp add my-server --transport stdio -- npx -y @modelcontextprotocol/server-github

# 带环境变量
claude mcp add github --transport stdio \
  --env GITHUB_TOKEN=your_token \
  -- npx -y @modelcontextprotocol/server-github
```

### 作用域

```bash
claude mcp add my-server --scope user    # 用户级（所有项目）
claude mcp add my-server --scope project # 项目级（.mcp.json，提交到 git）
claude mcp add my-server --scope local   # 本地级（仅本机）
```

### 常用 MCP 服务器

```bash
# GitHub
claude mcp add github --transport http https://api.githubcopilot.com/mcp/ \
  --header "Authorization: Bearer YOUR_GITHUB_PAT"

# PostgreSQL
claude mcp add postgres --transport stdio \
  --env DATABASE_URL=postgresql://localhost/mydb \
  -- npx -y @modelcontextprotocol/server-postgres
```

### 管理命令

```bash
claude mcp list          # 列出所有服务器
claude mcp remove <name> # 删除服务器
claude mcp get <name>    # 查看服务器详情
```

会话内使用 `/mcp` 命令管理连接和 OAuth 认证。

---

## 九、提效技巧

### 非交互模式（脚本集成）

```bash
# 管道输入
cat error.log | claude -p "分析这些错误"

# JSON 输出（适合脚本解析）
claude -p "列出所有函数" --output-format json

# 限制轮次和预算
claude -p --max-turns 5 --max-budget-usd 1.00 "重构认证模块"

# 最小模式（更快启动）
claude --bare -p "快速查询"
```

### 并行代理和 Worktrees

```bash
# 在隔离的 worktree 中启动 Claude（互不干扰）
claude -w feature-auth

# 带 tmux 的 worktree
claude -w feature-auth --tmux

# 批量并行变更
/batch migrate src/ from Solid to React
```

### 上下文管理

```bash
/context          # 查看上下文使用情况
/compact          # 压缩对话历史
/compact 保留关于认证模块的所有信息   # 带焦点指令的压缩
/clear            # 开始新对话（保留项目记忆）
```

### 自动记忆

Claude 会自动将学到的内容保存到 `~/.claude/projects/<project>/memory/`。

```bash
/memory           # 查看和管理记忆
```

在 settings.json 中禁用：`"autoMemoryEnabled": false`

### 后台任务

按 `Ctrl+B` 将 Bash 命令放到后台运行，用 `/tasks` 查看状态。

### 检查点和回滚

```bash
/rewind           # 回滚代码和对话到上一个检查点
/branch my-exp    # 分叉当前对话做实验
```

---

## 十、Claude Code UI（cloudcli）

Claude Code UI 是一个基于 Web 的图形界面，让你通过浏览器使用 Claude Code，适合不习惯纯命令行的场景。

### 简介

- 提供聊天界面、文件浏览器、Git 浏览器等可视化功能
- 支持多 Agent（Claude Code、Cursor CLI、Codex、Gemini CLI 等）
- 会话与 CLI 完全同步，`~/.claude` 配置共用
- 支持移动端访问（自托管或云端）

### 安装方式

**方式一：npm 全局安装（推荐）**

```bash
npm install -g @siteboon/claude-code-ui
cloudcli                    # 启动，默认 http://localhost:3001
cloudcli -p 8080            # 自定义端口
cloudcli status             # 查看配置和数据位置
cloudcli update             # 更新到最新版本
```

**方式二：源码安装（获取最新特性）**

```bash
git clone https://github.com/siteboon/claudecodeui.git
cd claudecodeui
npm install
cp .env.example .env        # 编辑 .env 配置端口等
npm run dev                 # 启动，支持热重载
```

**方式三：后台服务（PM2）**

```bash
npm install -g pm2
pm2 start cloudcli --name "cloudcli-ui"
pm2 startup && pm2 save     # 开机自启
```

### 主要功能

**聊天界面（Chat Mode）**
- 实时流式响应（WebSocket）
- 会话历史持久化，带时间戳
- 文件引用高亮、代码块语法高亮
- 支持 Claude 的 Thinking Mode
- 内联工具审批/拒绝

**Shell 模式**
- 点击聊天工具栏的 Shell 图标进入
- 完整终端，等同于直接运行 CLI
- 支持传递 flags、运行长任务、监控原始输出

**文件浏览器**
- 可视化浏览项目文件结构
- 支持文件编辑

**Git 浏览器**
- 可视化查看 Git 状态和历史

**会话管理**
- 按项目目录自动组织会话
- 支持恢复、重命名、删除会话
- 跨设备访问（自托管通过局域网 IP，云端任意设备）

### 配置

**工具权限**
- 默认所有工具禁用，需手动开启
- 推荐从只读工具开始（Read、Glob、Grep），按需添加写入/执行权限
- UI 中的配置变更直接写入 `~/.claude/settings.json`，CLI 和 UI 自动同步

**MCP 服务器**
- 通过 UI 的齿轮图标 → 选择 Agent → MCP Servers → Add Server 添加
- 配置写入 `~/.claude.json`，CLI 和 UI 共用
- MCP 服务器在会话开始时初始化，添加后需重启会话生效

**环境变量**
- npm 全局安装：在 shell profile（`.bashrc`/`.zshrc`）中设置
- 源码安装：在 `.env` 文件中设置

### 常见问题

| 问题 | 解决方案 |
|------|---------|
| 无项目或会话显示 | 确保 Claude Code 至少运行过一次，`~/.claude/projects/` 存在且有权限 |
| 端口被占用（EADDRINUSE） | 用 `--port 8080` 换端口，或 `lsof -i :3001` 找到并结束占用进程 |
| Windows node-pty 报错 | 推荐使用 WSL；或安装 Visual Studio Build Tools |
| MCP 服务器未连接 | 检查命令和环境变量，重启会话（MCP 在会话开始时初始化） |
| 文件浏览器为空 | 检查目录权限，查看服务端控制台日志 |

---

## 十一、cc-switch（模型切换工具）

cc-switch 是社区工具，用于在多个 Claude Code 配置（API Key、模型、Base URL）之间快速切换，适合同时使用多个账号或代理的场景。

### 安装

```bash
npm install -g cc-switch
```

### 基本用法

```bash
cc-switch list              # 列出所有配置
cc-switch use <profile>     # 切换到指定配置
cc-switch add <profile>     # 添加新配置
cc-switch remove <profile>  # 删除配置
```

### 配置示例

```json
{
  "profiles": {
    "personal": {
      "ANTHROPIC_API_KEY": "sk-ant-xxx",
      "ANTHROPIC_MODEL": "claude-sonnet-4-6"
    },
    "work-proxy": {
      "ANTHROPIC_BASE_URL": "https://your-proxy.example.com",
      "ANTHROPIC_API_KEY": "sk-ant-yyy"
    }
  }
}
```

> 注：也可以直接在 settings.json 的 `env` 字段中配置 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_MODEL` 实现类似效果，无需额外工具。

---

## 参考链接

- [Claude Code 官方文档](https://docs.anthropic.com/claude-code)
- [Claude Code UI (cloudcli)](https://cloudcli.ai/docs)
- [MCP 协议官网](https://modelcontextprotocol.io)
- [Claude Code GitHub Issues](https://github.com/anthropics/claude-code/issues)
