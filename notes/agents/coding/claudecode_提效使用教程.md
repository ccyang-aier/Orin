---
tags:
  - ClaudeCode
  - AI编程
  - 效率工具
  - CLI
  - MCP
updated: 2026-05-10
---
## Claude Code基本使用

Claude Code 是 Anthropic 的 agentic coding 工具，可以在终端、IDE、桌面端和网页端使用。它能读取代码库、编辑文件、运行命令、管理上下文，并通过 MCP 接入外部工具。日常使用建议优先把它当成“项目内协作代理”：先让它理解项目，再让它计划、修改、验证和总结。

### 安装与登录

官方当前推荐原生安装方式，优点是自动更新、权限问题更少。npm 安装仍可用，但不建议使用 `sudo npm install -g`，否则容易带来权限和安全问题。

```bash
# macOS / Linux / WSL
curl -fsSL https://claude.ai/install.sh | bash

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex

# Homebrew
brew install --cask claude-code

# npm，需要 Node.js 18+
npm install -g @anthropic-ai/claude-code

# WinGet
winget install Anthropic.ClaudeCode
```

安装后运行诊断、更新与版本切换：

```bash
claude doctor
claude update
claude install stable
claude install 2.1.118
```

Windows 用户可以选择 WSL 或 Git for Windows。若使用原生 Windows 且 Claude Code 没有找到 Git Bash，可设置：

```powershell
$env:CLAUDE_CODE_GIT_BASH_PATH="C:\Program Files\Git\bin\bash.exe"
```

登录与状态检查：

```bash
claude auth login
claude auth login --console
claude auth status
claude auth logout
```

### CLI 常用命令

日常启动方式优先记住这几类：

| 命令 | 使用场景 |
| --- | --- |
| `claude` | 在当前项目启动交互式会话 |
| `claude "explain this project"` | 带初始问题进入交互式会话 |
| `claude -p "query"` | 非交互模式，适合脚本和一次性任务 |
| `cat logs.txt \| claude -p "explain"` | 处理管道输入 |
| `claude -c` | 继续当前目录最近一次会话 |
| `claude -r "session-name"` | 按 ID 或名称恢复会话 |
| `claude -w feature-auth` | 在隔离的 git worktree 中工作 |

常用 flags：

| Flag | 说明 |
| --- | --- |
| `--model sonnet` / `--model opus` | 临时指定模型或模型别名 |
| `--effort high` | 设置推理强度，常见值为 `low`、`medium`、`high`、`xhigh`、`max` |
| `--permission-mode plan` | 以计划模式启动，适合复杂改动前先确认方案 |
| `--permission-mode auto` | 让 Claude Code 自动判断权限模式 |
| `--add-dir ../lib` | 给当前会话增加可访问目录 |
| `--max-turns 3` | print mode 下限制代理轮次 |
| `--max-budget-usd 5.00` | print mode 下限制预算 |
| `--output-format json` | print mode 下输出结构化结果 |
| `--bare` | 最小模式，跳过 hooks、skills、plugins、MCP、auto memory 和 `CLAUDE.md` 加载 |
| `--dangerously-skip-permissions` | 跳过权限提示，风险高，仅在隔离环境中谨慎使用 |

### 会话命令与提效工作流

在会话中输入 `/` 可以查看可用命令，输入 `/` 加关键词可以过滤。常用命令按工作流记忆即可：

| 阶段 | 命令 | 用途 |
| --- | --- | --- |
| 项目初始化 | `/init`、`/memory`、`/mcp`、`/agents`、`/permissions` | 生成项目记忆、配置 MCP/子代理/权限 |
| 执行任务 | `/plan`、`/model`、`/effort`、`/add-dir` | 规划任务、切换模型、调整推理强度、增加目录 |
| 管理上下文 | `/context`、`/compact`、`/btw`、`/clear` | 查看上下文、压缩历史、提出旁路问题、开启新任务 |
| 提交前检查 | `/diff`、`/simplify`、`/review`、`/security-review` | 查看改动、简化代码、代码审查、安全审查 |
| 会话恢复 | `/resume`、`/branch`、`/rewind`、`/rename` | 恢复、分叉、回滚和命名会话 |
| 排障 | `/doctor`、`/debug`、`/status` | 诊断安装、调试运行问题、查看状态 |

常见工作流：

| 场景 | 推荐做法 |
| --- | --- |
| 新项目接入 | 运行 `/init` 生成初稿，再手动精简 `CLAUDE.md` |
| 大改动 | 先 `/plan`，确认边界后再让 Claude 修改 |
| 长任务 | 用 `/context` 观察上下文，用 `/compact` 在关键阶段压缩 |
| 自动化脚本 | 使用 `claude -p`、`--output-format json`、`--max-turns`、`--max-budget-usd` |
| 并行探索 | 使用 `claude -w <name>` 在隔离 worktree 中执行 |
| 提交前 | 先 `/diff`，再 `/review` 或 `/simplify`，最后运行项目测试 |

### 配置、权限与上下文

Claude Code 的配置按作用域组织。优先级大致是：托管策略最高，其次命令行参数、本地配置、项目配置、用户配置。

| 位置 | 作用 |
| --- | --- |
| `~/.claude/settings.json` | 用户级配置，影响所有项目 |
| `.claude/settings.json` | 项目级配置，可提交到 git，适合团队共享 |
| `.claude/settings.local.json` | 本地项目配置，通常 gitignore，适合个人实验 |

常用配置示例：

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "model": "sonnet",
  "effortLevel": "high",
  "permissions": {
    "allow": [
      "Bash(npm run lint)",
      "Bash(npm run test *)",
      "Bash(git log *)",
      "Bash(git diff *)",
      "Read"
    ],
    "deny": [
      "Read(./.env)",
      "Bash(curl *)"
    ]
  },
  "env": {
    "DISABLE_TELEMETRY": "1"
  }
}
```

高频环境变量：

| 变量 | 说明 |
| --- | --- |
| `ANTHROPIC_API_KEY` | API key。设置后会优先于 Claude Pro/Max/Team/Enterprise 登录态，可能产生 API 计费 |
| `ANTHROPIC_BASE_URL` | 覆盖 API endpoint，常用于代理或网关 |
| `ANTHROPIC_MODEL` | 指定默认模型 |
| `CLAUDE_CODE_EFFORT_LEVEL` | 设置推理强度 |
| `API_TIMEOUT_MS` | API 请求超时，默认 10 分钟 |
| `BASH_DEFAULT_TIMEOUT_MS` | Bash 工具默认超时，默认 2 分钟 |
| `BASH_MAX_TIMEOUT_MS` | Bash 工具最大超时，默认 10 分钟 |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | 调整自动压缩触发比例 |
| `CLAUDE_CODE_DISABLE_AUTO_MEMORY` | 禁用 auto memory |
| `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` | 禁用后台任务能力 |
| `CLAUDE_CODE_DISABLE_THINKING` | 禁用 extended thinking |
| `DISABLE_AUTOUPDATER` | 禁用自动更新 |
| `DISABLE_TELEMETRY` | 禁用遥测 |

`CLAUDE.md` 是给 Claude 的项目说明和长期规则，适合写架构背景、开发命令、代码规范、提交要求和项目禁忌。Auto memory 则是 Claude 根据你的纠正和偏好自动积累的项目记忆。两者都会作为上下文加载，但它们不是强制规则，所以越具体、越短、越可执行，效果越稳定。

建议 `CLAUDE.md` 控制在清晰可读的范围内，优先写这些内容：

```markdown
# Project Overview
This is a React + TypeScript project.

# Commands
- Dev: `npm run dev`
- Test: `npm run test`
- Build: `npm run build`

# Conventions
- Use 2-space indentation.
- Prefer existing utilities before adding new abstractions.
- Run `npm run lint` before committing.
```

MCP 用于把 Claude Code 连接到 GitHub、数据库、监控、设计工具等外部系统。适合“我本来要把外部信息复制进聊天”的场景。

```bash
# 添加远程 HTTP MCP
claude mcp add my-server --transport http https://example.com/mcp

# 添加本地 stdio MCP
claude mcp add github --transport stdio \
  --env GITHUB_TOKEN=your_token \
  -- npx -y @modelcontextprotocol/server-github

# 管理 MCP
claude mcp list
claude mcp get github
claude mcp remove github
```

## cc-switch 介绍与使用

`cc-switch` 是社区提供的 Claude Code 配置切换工具，用于在多套 API key、Base URL、模型配置之间快速切换。它适合同时使用个人账号、工作代理、第三方网关或不同模型策略的人。它不是 Claude Code 官方工具，使用前要确认包来源、配置文件位置和密钥安全。

### 适用场景

如果你经常在个人账号、工作代理、第三方网关之间切换，`cc-switch` 可以降低手动改环境变量的成本。如果你只有一套 Claude 订阅账号，或者只是在项目内固定使用一套配置，直接用 `~/.claude/settings.json`、项目级 `.claude/settings.json` 和 shell profile 就够了。

### 安装与命令

npm 上常见的包名是 `@hobeeliu/cc-switch`，安装命令通常为：

```bash
npm install -g cc-switch
cc-switch init
```

常见操作：

```bash
cc-switch list
cc-switch init
cc-switch new personal
cc-switch new work-proxy
cc-switch use personal
cc-switch current
```

另一个相近工具是 `@dingpx/claude-code-switch`，命令名为 `ccs`：

```bash
npm install -g @dingpx/claude-code-switch
ccs current
ccs list
ccs switch
ccs switch personal
```

两者定位相似，都是“多配置管理与切换”。如果你的目标只是切换 `ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`、`ANTHROPIC_MODEL`，可以先用其中一个，不要同时混用，避免不知道当前环境到底由谁写入。

### 使用建议

把配置切换当成“运行前的显式动作”，不要在同一个终端里频繁来回切。切换后建议立刻检查：

```bash
claude auth status --text
claude -p "print current model and provider assumptions"
```

同时注意 `ANTHROPIC_API_KEY` 的优先级：只要环境变量存在，Claude Code 可能优先使用 API key 而不是 Claude 订阅登录态。为了避免误计费，日常订阅使用场景可以保持 `ANTHROPIC_API_KEY` 未设置；需要 API key 时再临时启用。

```bash
# macOS / Linux
unset ANTHROPIC_API_KEY

# Windows PowerShell
Remove-Item Env:ANTHROPIC_API_KEY
```

## Claude Code UI 介绍与使用

Claude Code UI 现在对应 CloudCLI UI，是一个基于浏览器的图形界面，用来访问 Claude Code、Codex、Cursor CLI、Gemini CLI 等 coding agent。它把聊天、Shell、文件浏览、Git、会话管理、MCP 和权限配置放到一个 Web UI 中，适合不想一直留在纯终端、需要移动端访问、或希望可视化审查 agent 改动的场景。

### 安装与部署

全局 npm 安装适合日常使用：

```bash
npm install -g @siteboon/claude-code-ui
cloudcli
cloudcli -p 8080
cloudcli status
cloudcli update
cloudcli version
```

默认服务地址是 `http://localhost:3001`。如果端口冲突，可以用 `cloudcli --port 8080` 或 `cloudcli -p 8080`。

源码安装适合贡献代码、二次开发或体验 npm 包尚未发布的最新功能：

```bash
git clone https://github.com/siteboon/claudecodeui.git
cd claudecodeui
npm install
cp .env.example .env
npm run dev
```

需要常驻后台时可用 PM2：

```bash
npm install -g pm2
pm2 start cloudcli --name "cloudcli-ui"
pm2 start cloudcli --name "cloudcli-ui" -- --port 8080
pm2 startup
pm2 save
pm2 logs cloudcli-ui
pm2 restart cloudcli-ui
```

### 配置、权限与 MCP

CloudCLI UI 默认禁用 Claude Code 工具，需要在 Tools Settings 中显式开启。建议从只读能力开始，再按需打开写入和命令执行权限。

| 配置项 | 说明 |
| --- | --- |
| Tools & Permissions | 在侧边栏齿轮中开启工具权限；Read、Glob、Grep 风险较低，Edit/Write/Bash 风险更高 |
| Scoped Bash | 优先允许 `Bash(npm run lint)`、`Bash(git diff *)` 这类具体命令，而不是开放全部 Bash |
| 同步机制 | UI 修改会写入本地 Claude 配置，CLI 与 UI 共享配置 |
| Project scope | 项目级 `.claude/settings.json` 可提交到仓库，适合团队共享规则 |

CloudCLI UI 的环境变量读取顺序是：CLI flag 优先，其次进程环境变量、`.env` 文件、内置默认值。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `3001` | UI 服务端口 |
| `WORKSPACES_ROOT` | `~` | 项目发现根目录 |
| `ENABLE_HTTPS` | `false` | 是否启用 HTTPS |

全局安装时通常在 shell profile 或进程管理器中设置环境变量；源码安装时可写入项目根目录 `.env`。修改 `.env` 后需要重启服务。

MCP 可以在 UI 中通过“齿轮图标 -> 选择 Agent -> MCP Servers -> Add Server”添加。UI 写入 `~/.claude.json`，所以通过 UI 添加的 MCP 会出现在 Claude Code CLI 中；通过终端 `claude mcp add` 添加的 MCP 也会显示在 UI 中。MCP 通常在会话开始时初始化，添加或修改后如果工具未出现，先重启会话。

### 核心功能

| 功能 | 适合场景 |
| --- | --- |
| Chat Interface | 用聊天方式启动、恢复和继续 agent 会话，支持实时流式响应、文件引用、代码块、Thinking Mode 和工具审批 |
| Shell Mode | 从聊天工具栏进入完整终端，适合传 CLI flags、看长任务原始输出、执行 UI 未暴露的操作 |
| File Explorer & Editor | 在 UI 中浏览目录、打开文件、创建/重命名/删除文件，编辑器支持语法高亮和即时保存 |
| Agent Changes | agent 修改文件后可在 UI 中查看 diff，并按文件接受或拒绝 |
| Git Explorer | 查看变更文件、审查 diff、stage、commit、切换分支，适合把 agent 改动纳入正式提交前做人工检查 |
| Session Management | 按项目目录组织会话，支持恢复、重命名、删除和查看历史 |
| Mobile Access | 同一局域网中可用 `http://[your-machine-ip]:3001` 访问；移动端有底部导航、折叠侧栏和 PWA 支持 |

推荐工作流：

1. 在本机启动 `cloudcli`，打开 `http://localhost:3001`。
2. 先到 Tools Settings 开启只读工具，确认项目能被发现。
3. 在 Chat 中让 agent 解释项目或执行小任务。
4. agent 修改文件后，到 File Explorer 和 Git Explorer 查看 diff。
5. 确认无误后在 Git Explorer stage/commit，或回到终端执行测试与提交。
6. 需要手机查看长任务时，在同一局域网打开 `http://[your-machine-ip]:3001`，或将其添加为 PWA。

### 常见问题

| 问题 | 处理方式 |
| --- | --- |
| 没有项目或会话 | 确保 Claude Code 至少在项目中运行过一次，检查 `~/.claude/projects/` 和 `WORKSPACES_ROOT` |
| 文件浏览为空或权限错误 | 检查项目目录权限，确认运行 CloudCLI UI 的用户能访问该目录 |
| `EADDRINUSE :::3001` | 换端口启动：`cloudcli --port 8080` |
| Windows `node-pty` 安装报错 | 优先使用 WSL；原生 Windows 需安装 Visual Studio Build Tools |
| MCP server 未连接 | 检查命令、参数和环境变量，运行 `claude mcp list`，并重启会话 |
| `.env` 修改不生效 | 服务只在启动时读取 `.env`，需要重启 `cloudcli` 或 PM2 服务 |
