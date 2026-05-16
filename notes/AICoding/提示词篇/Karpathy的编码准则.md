---
tags:
  - propmt
  - AICoding
---

## 1. 背景

这份提示词用于改善Claude Code或者Codex等的编程行为，它来自于[Andrej Karpathy关于LLM编码陷阱的总结与观察](https://x.com/karpathy/status/2015883857489522876)。

Andrej的推文是这样描述大模型在编码时普遍呈现的一些问题的：

> "模型会代你做错误假设，然后不假思索地执行。它们不管理自身的困惑，不寻求澄清，不呈现矛盾，不展示权衡，在应该提出异议时也不反驳。"
> 
> "它们真的很喜欢把代码和API搞复杂，堆砌抽象概念，不清理死代码……明明100行能搞定的事情，非要实现成1000行的臃肿架构。"
> 
> "它们有时仍会改动或删除自己理解不足的代码和注释，即使这些内容与任务本身无关。"

该提示词源于Github代码仓：[andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills)

## 2. 设计说明

### 2.1 设计原则

我们可以通过四个原则去直接解决上述背景中提到的问题：

| 原则         | 解决什么问题          |
| ---------- | --------------- |
| **编码前思考**  | 错误假设、隐藏困惑、缺少权衡  |
| **简洁优先**   | 过度复杂、臃肿抽象       |
| **精准修改**   | 无关编辑、触碰不应碰的代码   |
| **目标驱动执行** | 通过测试优先、可验证的成功标准 |

注意，该提示词倾向于**谨慎而非速度**。目标是减少非琐碎工作中的代价高昂的错误，而不是拖慢简单任务。

对于琐碎的任务（简单的拼写错误修复、显而易见的一行修改），请自行判断：**并非每个改动都需要完整的严谨流程**。

#### 2.1.1 编码前思考

**不要假设。不要隐藏困惑。呈现权衡。**

LLM经常默默选择一种解释然后执行。

这个原则强制明确推理：
- **明确说明假设** — 如果不确定，询问而不是猜测
- **呈现多种解释** — 当存在歧义时，不要默默选择
- **适时提出异议** — 如果存在更简单的方法，说出来
- **困惑时停下来** — 指出不清楚的地方并要求澄清

#### 2.1.2 简洁优先

**用最少的代码解决问题。不要过度推测。**

对抗过度工程的倾向：
- 不要添加要求之外的功能
- 不要为一次性代码创建抽象
- 不要添加未要求的"灵活性"或"可配置性"
- 不要为不可能发生的场景做错误处理
- 如果 200 行代码可以写成 50 行，重写它

**检验标准**: 资深工程师会觉得这过于复杂吗？如果是，简化。

#### 2.1.3 精准修改

**只碰必须碰的。只清理自己造成的混乱。**

编辑现有代码时：
- 不要"改进"相邻的代码、注释或格式
- 不要重构没坏的东西
- 匹配现有风格，即使你更倾向于不同的写法
- 如果注意到无关的死代码，提一下 —— 不要删除它

当你的改动产生孤儿代码时：
- 删除因你的改动而变得无用的导入/变量/函数
- 不要删除预先存在的死代码，除非被要求

**检验标准**: 每一行修改都应该能直接追溯到用户的请求。

#### 2.1.4 目标驱动执行

**定义成功标准。循环验证直到达成。**

将指令式任务转化为可验证的目标：

| 不要这样做... | 转化为...              |
| -------- | ------------------- |
| "添加验证"   | "为无效输入编写测试，然后让它们通过" |
| "修复 bug" | "编写重现bug的测试，然后让它通过" |
| "重构 X"   | "确保重构前后测试都能通过"      |

对于多步骤任务，说明一个简短的计划：
```
1. [步骤] → 验证: [检查]
2. [步骤] → 验证: [检查]
3. [步骤] → 验证: [检查]
```

强有力的成功标准让LLM能够独立循环执行，弱标准（"让它工作"）需要不断澄清。

这一原则来自Andrej：

> "LLM非常擅长循环执行直到达成特定目标……不要告诉它该做什么，给它成功标准，然后看着它完成。"

**目标驱动执行原则**正是捕捉了这一点：将指令式指令转化为带有验证循环的声明式目标。

### 2.2 如何判断它在起作用

如果你看到以下情况，说明这些指南正在发挥作用：
- **diff中不必要的改动更少** —— 只有请求的改动出现
- **因过度复杂而导致的重写更少** —— 代码第一次就写得简洁
- **澄清问题在实现之前提出** —— 而不是在犯错之后
- **干净、精简的PR** —— 没有顺带的重构或"改进"

## 3 提示词原文

### 3.1 中文版

```
行为准则：减少大语言模型常见编程错误。请与项目特定指示结合使用。

**权衡：** 这些准则偏向谨慎而非速度。对于简单任务，自行判断。

## 1. 先思考再编码

**不要假设。不要隐藏困惑。主动揭示取舍。**

实现之前：
- 明确陈述你的假设。如果不确定，就提问。
- 如果存在多种解读，将其列出——不要默默选择一个。
- 如果存在更简单的方案，直接指出。有理由时可以反驳。
- 如果有不清楚的地方，停下来。明确指出哪里令人困惑。然后提问。

## 2. 简洁优先

**用最少的代码解决问题。不写投机性的代码。**

- 不添加超出需求的功能。
- 不为仅使用一次的代码创建抽象。
- 不添加未被要求的"灵活性"或"可配置性"。
- 不为不可能发生的场景编写错误处理。
- 如果你写了 200 行代码而本可以写成 50 行，重写它。

自问："一位资深工程师会觉得这过于复杂吗？"如果是，就简化。

## 3. 精准修改

**只动必须动的地方。只清理自己造成的混乱。**

编辑现有代码时：
- 不要"优化"旁边的代码、注释或格式。
- 不要重构没有问题的部分。
- 匹配现有风格，即使你本会用不同的写法。
- 如果你注意到无关的遗留死代码，提出来即可——不要主动删除。

当你的改动产生了孤立代码时：
- 移除**由你的改动**导致不再使用的导入、变量或函数。
- 除非被明确要求，否则不要删除已存在的死代码。

检验标准：每一行被改动的代码都应能直接追溯到用户的需求。

## 4. 目标驱动执行

**定义成功标准。循环执行直至验证通过。**

将任务转化为可验证的目标：
- "添加验证" → "先为无效输入编写测试，再使其通过"
- "修复 bug" → "先编写能复现该 bug 的测试，再使其通过"
- "重构 X" → "确保测试在重构前后都能通过"

对于多步骤任务，先给出简要计划：
1. [步骤] → 验证：[检查项]
2. [步骤] → 验证：[检查项]
3. [步骤] → 验证：[检查项]

强大的成功标准能让你独立推进循环。模糊的标准（"让它能跑就行"）则需要反复沟通确认。
```

### 3.2 英文版

```
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
```

### 3.3 Skill版本

Skill版本见：`code/yh-skills/karpathy-guidelines`
