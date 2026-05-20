---
tags:
  - AICoding
  - vibecoding
  - Codex
  - skill
  - frontend
updated: 2026-05-21
source:
  - "C:\\Users\\17335\\.agents\\skills\\brandkit\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\design-taste-frontend\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\gpt-taste\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\high-end-visual-design\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\image-to-code\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\imagegen-frontend-mobile\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\imagegen-frontend-web\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\impeccable\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\industrial-brutalist-ui\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\minimalist-ui\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\redesign-existing-projects\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\stitch-design-taste\\SKILL.md"
status: draft
---

# taste-skill的详细使用教程

这篇笔记只解决几个核心问题：taste-skill 怎么安装、每个子 skill 是干嘛的、什么场景该用哪个、提示词应该怎么写。

先说明边界：`taste-skill` 不是 OpenAI 官方固定能力名，而是这里对一组本地审美/前端/视觉方向 skill 的统称。当前机器上它们实际位于 `C:\Users\17335\.agents\skills\`。Codex 识别的是每个带 `SKILL.md` 的子目录，而不是一个外层父目录。

## 1. 如何安装taste-skill

### 1.1 安装位置

在当前环境中，推荐把每个子 skill 安装成这个结构：

```text
C:\Users\<用户名>\.agents\skills\
  brandkit\SKILL.md
  design-taste-frontend\SKILL.md
  gpt-taste\SKILL.md
  high-end-visual-design\SKILL.md
  image-to-code\SKILL.md
  imagegen-frontend-mobile\SKILL.md
  imagegen-frontend-web\SKILL.md
  impeccable\SKILL.md
  industrial-brutalist-ui\SKILL.md
  minimalist-ui\SKILL.md
  redesign-existing-projects\SKILL.md
  stitch-design-taste\SKILL.md
```

关键点：不要只把整个 `taste-skill` 外层文件夹复制成 `C:\Users\<用户名>\.agents\skills\taste-skill\...`，除非你的代理明确支持嵌套扫描。当前 Codex 会话里可见的是 `.agents\skills` 下的顶层子目录。

### 1.2 手动安装

假设你下载或克隆到本地的 taste-skill 包在 `D:\Downloads\taste-skill`，可以用 PowerShell 复制所有带 `SKILL.md` 的子目录：

```powershell
$src = "D:\Downloads\taste-skill"
$dst = "$env:USERPROFILE\.agents\skills"

New-Item -ItemType Directory -Force $dst
Get-ChildItem $src -Directory |
  Where-Object { Test-Path (Join-Path $_.FullName "SKILL.md") } |
  Copy-Item -Destination $dst -Recurse -Force
```

如果你只有单个 skill，例如 `gpt-taste`，也可以只复制这一个目录：

```powershell
Copy-Item "D:\Downloads\taste-skill\gpt-taste" "$env:USERPROFILE\.agents\skills\" -Recurse -Force
```

### 1.3 验证是否安装成功

先检查目录：

```powershell
Get-ChildItem "$env:USERPROFILE\.agents\skills" -Directory |
  Where-Object { Test-Path (Join-Path $_.FullName "SKILL.md") } |
  Select-Object Name
```

再抽查某个 skill：

```powershell
Get-Content -Raw "$env:USERPROFILE\.agents\skills\gpt-taste\SKILL.md"
```

最后重启 Codex 或新开一个会话，看当前可用 skills 列表里是否出现这些名称。调用时不需要运行 `npm install`；这些 skill 本身只是本地 Markdown 工作流规则。只有当生成的前端项目需要 `gsap`、`framer-motion`、`@phosphor-icons/react` 等依赖时，才需要在目标项目里另外安装依赖。

### 1.4 怎么调用

最简单的调用方式是在提示词里直接点名：

```text
请使用 gpt-taste 设计一个高视觉 landing page。
```

或：

```text
请使用 design-taste-frontend 改造这个产品设置页。
```

如果一个任务可能适合多个 skill，优先指定一个主 skill，避免规则互相打架。例如产品工具界面用 `design-taste-frontend`，营销落地页用 `gpt-taste`，移动 app 概念图用 `imagegen-frontend-mobile`。

## 2. 子skill总览

| 子 skill                      | 一句话用途                                   | 主要输出    |
| ---------------------------- | --------------------------------------- | ------- |
| `brandkit`                   | 做品牌视觉系统图、Logo 概念、品牌板                    | 图片      |
| `design-taste-frontend`      | 写或改产品级前端 UI                             | 代码      |
| `gpt-taste`                  | 做强视觉、强动效、Awwwards 感网页                   | 代码      |
| `high-end-visual-design`     | 做高端 agency 风格界面                         | 代码或设计方向 |
| `image-to-code`              | 先生成网页视觉图，再按图实现代码                        | 图片 + 代码 |
| `imagegen-frontend-mobile`   | 生成高质量移动 app 屏幕图                         | 图片      |
| `imagegen-frontend-web`      | 生成网站每个 section 的参考图                     | 图片      |
| `impeccable`                 | 项目级 UI 设计工作流，支持 teach/shape/craft/audit | 文档 + 代码 |
| `industrial-brutalist-ui`    | 工业/战术/终端风格高密度界面                         | 代码或设计方向 |
| `minimalist-ui`              | 克制、文档感、低噪声工具界面                          | 代码或设计方向 |
| `redesign-existing-projects` | 改造已有网站/app，不推翻重写                        | 代码      |
| `stitch-design-taste`        | 为 Google Stitch 生成可用的 `DESIGN.md`       | 文档      |

下面逐个说明。

## 3. brandkit

### 3.1 它是干嘛的

`brandkit` 用来生成品牌视觉系统图，例如 Logo 概念板、品牌规范总览、颜色/字体/应用场景展示。它适合“先定品牌世界观”，不是用来写前端代码。

### 3.2 适用场景

- 新产品还没有 Logo、配色和品牌气质；
- 想生成一张高端品牌规范板；
- 想为开发者工具、AI 产品、消费 app、游戏、奢侈品牌等做视觉方向；
- 需要给 Figma、Canva、网站设计提供品牌参考；

### 3.3 怎么用

```text
请使用 brandkit 为 Orin 生成一张品牌系统图。
定位：个人 AI 第二大脑；
气质：克制、长期主义、知识沉淀、可信赖；
需要包含 Logo 概念、配色、字体、UI 应用片段和一句 tagline。
```

### 3.4 实例

如果你要做一个 AI 知识库产品，先用 `brandkit` 生成品牌板，再让 `imagegen-frontend-web` 按这个品牌板生成网站 section 图，最后再用 `image-to-code` 或普通 Codex 实现页面。

## 4. design-taste-frontend

### 4.1 它是干嘛的

`design-taste-frontend` 是最适合真实产品 UI 的前端审美 skill。它强调依赖检查、React/Next/Tailwind 约束、状态完整性、响应式、性能和反 AI 味。

### 4.2 适用场景

- 产品 dashboard、settings、表单、列表、详情页；
- AI 工具、RAG 控制台、知识库工作台；
- 需要写真实代码，而不只是生成设计图；
- 希望 UI 高级但不夸张；

### 4.3 怎么用

```text
请使用 design-taste-frontend 优化当前设置页。
要求：先检查 package.json，不要导入不存在的库；
保留现有功能，补齐 loading、empty、error、focus、mobile 状态；
不要做成暗色控制台，不要紫蓝 AI 渐变。
```

### 4.4 实例

如果 Orin 有一个“知识卡片详情页”，可以让它重新组织标题、标签、来源、摘要、引用、关联笔记和操作按钮，让页面像一个长期可用的知识工具，而不是普通卡片堆叠。

## 5. gpt-taste

### 5.1 它是干嘛的

`gpt-taste` 偏向高强度创意网页和动效页面。它强调 AIDA 结构、`<design_plan>`、宽标题、gapless bento、GSAP ScrollTrigger、强视觉 section 和反模板化布局。

### 5.2 适用场景

- landing page、品牌官网、发布页；
- 想要 Awwwards 风格、强动效、强视觉冲击；
- 可以接受更大胆的排版、滚动叙事和高级动画；

### 5.3 怎么用

```text
请使用 gpt-taste 为一个 AI 开发者工具做 landing page。
先输出 <design_plan>；
页面遵循 AIDA：导航、Hero、功能、滚动叙事、CTA；
Hero 标题最多 2-3 行，避免三等分卡片和紫蓝 AI 渐变。
```

### 5.4 实例

适合做“Codex 插件市场”的官网首页：首屏强品牌表达，中段用 bento 展示插件能力，后段用 GSAP 做横向滚动案例展示，最后用大 CTA 收口。

## 6. high-end-visual-design

### 6.1 它是干嘛的

`high-end-visual-design` 偏高端 agency / 视觉导演风格。它强调高级字体、宏观留白、double-bezel 容器、按钮内部图标结构、细腻动效和“看起来像高预算设计稿”的质感。

### 6.2 适用场景

- 高端品牌页、产品宣传页、视觉作品集；
- 希望界面显得更贵、更有材质感；
- 需要强控制的 hero、CTA、卡片和动效细节；

### 6.3 怎么用

```text
请使用 high-end-visual-design 重新设计这个产品首页。
方向：高端、克制、有空间感；
要求：大留白、精致 CTA、统一字体层级，不要模板感卡片网格。
```

### 6.4 实例

如果你要为一个高端 AI 咨询服务做首页，它比 `minimalist-ui` 更适合，因为它会主动强化品牌质感、视觉张力和 CTA 结构。

## 7. image-to-code

### 7.1 它是干嘛的

`image-to-code` 是“先图后码”工作流。它要求先生成清晰的网页视觉参考图，再深度分析排版、颜色、按钮、间距和组件，最后按图实现代码。

### 7.2 适用场景

- 用户非常在意视觉还原；
- 做网站、hero、产品页、品牌页；
- 不想让 AI 直接写出一套普通模板；

### 7.3 怎么用

```text
请使用 image-to-code 为这个产品页工作。
先生成 4 张 section 参考图：Hero、Features、Workflow、CTA；
逐张分析文字、排版、颜色、按钮和间距；
确认后再实现代码，并尽量贴近生成图。
```

### 7.4 实例

如果你要做一套“AI 文件整理工具”的官网，可以先让它生成 6 个 section 的视觉图，再实现 React 页面。这样比直接写代码更容易得到清晰的视觉方向。

## 8. imagegen-frontend-mobile

### 8.1 它是干嘛的

`imagegen-frontend-mobile` 只生成移动 app 屏幕图，不写代码。它强调 iOS/Android 平台感、安全区、手机 mockup、流程一致性、文字可读性和多屏统一。

### 8.2 适用场景

- iOS / Android app 概念图；
- onboarding、登录、首页、详情、设置、聊天、购物、健康、金融等移动端流程；
- 需要多屏视觉探索，但暂时不写代码；

### 8.3 怎么用

```text
请使用 imagegen-frontend-mobile 生成一个 5 屏 iOS app 概念。
产品：个人 AI 知识库；
屏幕：onboarding、home、search、note detail、settings；
要求：统一手机 mockup，文字清晰，不要像网页塞进手机。
```

### 8.4 实例

如果 Orin 以后要做手机端，可以先用它生成 5-7 张 app screen，确认信息架构和视觉方向，再决定是否进入 React Native、SwiftUI 或 Flutter 实现。

## 9. imagegen-frontend-web

### 9.1 它是干嘛的

`imagegen-frontend-web` 只生成网站视觉参考图。它有一个关键规则：每个 section 单独生成一张横图，不要把整页压进一张长图或拼图。

### 9.2 适用场景

- landing page、marketing site、产品官网；
- 需要先确定每个 section 长什么样；
- 想让后续开发模型有清晰可还原的参考图；

### 9.3 怎么用

```text
请使用 imagegen-frontend-web 为 Orin 官网生成 6 张横向 section 图。
每个 section 一张：Hero、Problem、Features、Workflow、Use Cases、CTA；
保持同一套配色和字体，不要合成一张长图。
```

### 9.4 实例

先用 `brandkit` 定品牌，再用 `imagegen-frontend-web` 生成官网每个 section，最后用 `image-to-code` 做代码实现，是比较稳的三步链路。

## 10. impeccable

### 10.1 它是干嘛的

`impeccable` 是项目级前端设计工作流。它不是单一风格，而是一套命令：先建立产品上下文，再 shape、craft、audit、polish、harden 等。

### 10.2 适用场景

- 项目还没有 `PRODUCT.md` / `DESIGN.md`；
- 想先定产品定位和设计原则，再做 UI；
- 想让 Codex 先规划再实现；
- 需要对已有页面做审美、可用性、性能、响应式评审；

### 10.3 常用命令

| 命令 | 用途 |
| --- | --- |
| `teach` | 建立 `PRODUCT.md`，必要时补设计上下文 |
| `shape` | 只做 UX/UI 方案，不立刻写代码 |
| `craft` | 先 shape，再实现 |
| `audit` | 查可访问性、性能、响应式和技术质量 |
| `polish` | 发布前视觉和交互打磨 |
| `harden` | 补错误态、边界、i18n、生产状态 |
| `distill` | 去复杂度，回到核心 |
| `animate` | 增加有目的的动效 |
| `typeset` | 优化字体和排版层级 |
| `layout` | 修布局、节奏和视觉层级 |

### 10.4 怎么用

```text
请使用 impeccable teach，为这个项目建立 PRODUCT.md 和 DESIGN.md 的基础上下文。
重点写清楚：用户、产品目标、设计语气、反例、brand/product register。
```

```text
请使用 impeccable shape 规划这个知识库搜索页。
先不要写代码，只输出页面角色、信息架构、关键状态和验收标准。
```

### 10.5 实例

如果一个项目的 UI 方向一直摇摆，先跑 `impeccable teach`，再跑 `impeccable shape`，最后再进入 `craft`。它适合“从产品上下文开始”，而不是只修一个按钮。

## 11. industrial-brutalist-ui

### 11.1 它是干嘛的

`industrial-brutalist-ui` 做工业、战术、终端、蓝图风格界面。关键词是刚性网格、巨大字体、单色高对比、终端感、机械感、高密度信息。

### 11.2 适用场景

- 数据密集 dashboard；
- 安全、监控、遥测、作战室风格；
- 作品集或编辑页想要“解密档案/工业蓝图”气质；

### 11.3 怎么用

```text
请使用 industrial-brutalist-ui 设计一个安全态势面板。
方向：Tactical Telemetry；
要求：暗色、单一红色告警、monospace 数据、硬边框、无圆角、无渐变。
```

### 11.4 实例

适合网络安全威胁地图、日志遥测面板、硬核作品集首页。不适合日常知识库、消费 app、温和生产力工具。

## 12. minimalist-ui

### 12.1 它是干嘛的

`minimalist-ui` 做克制、文档感、低噪声、暖色单色的高级工具界面。它强调排版、信息层级、平面 bento、少阴影、少渐变。

### 12.2 适用场景

- 知识库、笔记、写作、阅读、搜索工具；
- 需要长时间使用的后台或工作台；
- 不想要视觉炫技，只想要高级、清爽、稳定；

### 12.3 怎么用

```text
请使用 minimalist-ui 设计 Orin 的笔记详情页。
要求：暖白背景、清晰排版、少阴影、少渐变；
重点突出标题、摘要、标签、来源、引用和关联笔记。
```

### 12.4 实例

Orin 这种“第二大脑”项目更适合先用 `minimalist-ui` 或 `design-taste-frontend`，通常不要一上来用 `gpt-taste` 做成炫酷官网。

## 13. redesign-existing-projects

### 13.1 它是干嘛的

`redesign-existing-projects` 专门升级已有网页或 app。它的核心流程是 scan、diagnose、fix：先读项目，再找问题，再在原技术栈里做有边界的改造。

### 13.2 适用场景

- 页面功能已经有了，但视觉普通；
- 想修字体、颜色、间距、卡片、按钮、响应式；
- 不想迁移框架或推翻重写；

### 13.3 怎么用

```text
请使用 redesign-existing-projects 改造当前首页。
保留现有功能和技术栈；
先 scan 并列出设计问题，再 targeted fix；
不要大改架构，最后用 Browser 验证桌面和移动端。
```

### 13.4 实例

如果已有页面是一堆默认 shadcn 卡片，它会优先从字体、配色、hover/active、布局间距、组件重复和空状态入手，而不是直接重建整个项目。

## 14. stitch-design-taste

### 14.1 它是干嘛的

`stitch-design-taste` 用来为 Google Stitch 生成语义化 `DESIGN.md`。它把审美规则写成 Stitch 更容易理解的自然语言设计系统，包括颜色、字体、组件、布局、动效和禁用项。

### 14.2 适用场景

- 使用 Google Stitch 生成 app 或网页屏幕；
- 想让 Stitch 不再输出普通 AI 风格界面；
- 需要把一个项目的设计语言沉淀成 `DESIGN.md`；

### 14.3 怎么用

```text
请使用 stitch-design-taste 为这个项目生成 DESIGN.md。
目标：让 Google Stitch 生成克制、高级、非模板化的产品 UI；
需要包含视觉氛围、颜色、字体、组件、布局、动效和明确禁用项。
```

### 14.4 实例

如果你准备用 Stitch 快速出一组 Orin 移动端界面，先用 `stitch-design-taste` 写 `DESIGN.md`，再把这份文档喂给 Stitch，会比直接说“做高级一点”稳定很多。

## 15. 怎么组合使用

### 15.1 品牌官网

```text
先用 brandkit 定品牌视觉；
再用 imagegen-frontend-web 生成每个 section 的横图；
最后用 image-to-code 按图实现网页。
```

### 15.2 产品工具界面

```text
先用 impeccable teach/shape 定产品上下文；
再用 design-taste-frontend 实现；
最后用 Browser 做桌面和移动端验收。
```

### 15.3 已有项目变高级

```text
用 redesign-existing-projects 做 scan -> diagnose -> fix；
如果是知识工具或长期使用界面，加入 minimalist-ui 约束；
如果是营销页，再考虑 gpt-taste 或 high-end-visual-design。
```

### 15.4 移动端概念设计

```text
用 imagegen-frontend-mobile 生成多屏 app flow；
确认后再进入 SwiftUI、React Native、Flutter 或 Web 实现。
```

## 16. 选型速查

| 你要做什么 | 优先用 |
| --- | --- |
| 安装整个 taste-skill 包 | 复制所有子目录到 `.agents\skills` |
| 检查当前有哪些 skill | `Get-ChildItem "$env:USERPROFILE\.agents\skills"` |
| 做品牌板/Logo 概念 | `brandkit` |
| 写产品 UI 代码 | `design-taste-frontend` |
| 做炫酷 landing page | `gpt-taste` |
| 做高端官网质感 | `high-end-visual-design` |
| 先生成网页图再写代码 | `image-to-code` |
| 生成移动 app 屏幕图 | `imagegen-frontend-mobile` |
| 生成网站 section 图 | `imagegen-frontend-web` |
| 建项目设计上下文 | `impeccable teach` |
| 先规划不写代码 | `impeccable shape` |
| 升级已有项目 | `redesign-existing-projects` |
| 做克制知识工具 | `minimalist-ui` |
| 做战术工业风界面 | `industrial-brutalist-ui` |
| 给 Google Stitch 写设计系统 | `stitch-design-taste` |

## 17. 最小可用提示词

```text
我想使用 taste-skill 做这个任务：<任务描述>。

请先判断应该使用哪个子 skill，并说明理由；
如果需要安装依赖，先检查 package.json；
如果是图片类 skill，只生成设计图，不写代码；
如果是代码类 skill，完成后运行可用校验并做浏览器验收。
```

如果你已经知道要用哪个，直接点名更好：

```text
请使用 design-taste-frontend 实现这个产品页面：<页面说明>。
```

```text
请使用 imagegen-frontend-mobile 生成这个 app 的 5 个核心屏幕。
```
