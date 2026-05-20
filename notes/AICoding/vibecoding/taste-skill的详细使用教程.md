---
tags:
  - AICoding
  - vibecoding
  - Codex
  - skill
  - frontend
updated: 2026-05-21
source:
  - "C:\\Users\\17335\\.agents\\skills\\gpt-taste\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\design-taste-frontend\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\impeccable\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\redesign-existing-projects\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\high-end-visual-design\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\image-to-code\\SKILL.md"
  - "C:\\Users\\17335\\.agents\\skills\\minimalist-ui\\SKILL.md"
status: draft
---

# taste-skill的详细使用教程

这篇教程记录如何在 Codex / Agent 前端工作流里使用 taste 类 skill，让 AI 不只是生成“能跑的页面”，而是更稳定地产出有审美判断、有产品上下文、有状态覆盖、有浏览器验收的界面。

先说明边界：这里的 `taste-skill` 不是 OpenAI 官方产品能力的固定名称，而是本文对本机一组本地设计审美类 skill 的统称。本文内容基于当前本机 `C:\Users\17335\.agents\skills\` 下可读到的 `SKILL.md` 文件，以及当前 Codex 会话中可见的 skills 列表；如果换机器、换代理、换版本或 skill 文件被修改，实际可用能力应以当前环境重新读取到的文件为准。

## 1. taste-skill是什么

### 1.1 它解决的问题

普通 AI 前端输出很容易出现这些问题：

- 页面能运行，但视觉像模板；
- hero 过度居中，标题换成 5-6 行；
- 卡片一排三个，图标加标题加说明，重复到没有记忆点；
- 紫蓝渐变、霓虹光效、暗色 dashboard 被滥用；
- 只实现成功态，没有 loading、empty、error、disabled、focus、mobile 等状态；
- 引入不存在的依赖，或者没有检查项目实际 Tailwind / React / Next.js 版本；
- 图像、按钮、动效和布局看起来不错，但一进浏览器就溢出、遮挡、空白或卡顿；

`taste-skill` 的作用不是替代产品判断，而是把“什么算廉价 AI 味”“什么是更高级的界面结构”“实现前要做哪些检查”“浏览器里要验收什么”写成可复用约束，让 Codex 在生成 UI、重设计页面、做视觉探索或修复前端时先进入更高标准的工作模式。

### 1.2 它不是魔法

使用 taste 类 skill 时要避免几个误解：

- 它不是 npm 包，也不是浏览器插件；多数 taste skill 本质上是 `SKILL.md` 中的设计、实现和验收规则；
- 它不会自动知道你的产品定位；如果没有 `PRODUCT.md`、`DESIGN.md`、页面截图或真实业务描述，它仍然可能做出漂亮但不贴题的界面；
- 它不能保证依赖存在；凡是 Framer Motion、GSAP、Phosphor、Radix、shadcn/ui、Three.js 等第三方库，都必须先检查项目依赖；
- 它不能替代浏览器验证；视觉任务最终必须看真实渲染，而不是只相信代码；
- 它不是越多越好；多个 taste skill 同时叠加可能互相冲突，应按任务类型选一个主 skill，再补一个辅助 skill；

### 1.3 本机可用的核心 taste 类 skill

当前本机可读到的 taste 相关 skill 主要分三类：

| 类型 | skill | 适合场景 | 主要特点 |
| --- | --- | --- | --- |
| 高强度创意前端 | `gpt-taste` | Awwwards 风格 landing page、强视觉单页、GSAP 动效页面 | 要求输出 `<design_plan>`、AIDA 结构、随机化布局、宽标题、gapless bento、GSAP 动效 |
| 产品级前端审美 | `design-taste-frontend` | 产品 UI、dashboard、settings、forms、组件、真实业务界面 | 强调依赖检查、React/Next/Tailwind 约束、状态覆盖、性能守卫、反 AI 味规则 |
| 项目级设计系统 | `impeccable` | 新项目定调、设计上下文建立、shape/craft/audit/polish 等流程 | 要先读取 `PRODUCT.md` / `DESIGN.md`，区分 brand/product register，有命令式子工作流 |
| 旧项目重设计 | `redesign-existing-projects` | 改造现有网页或 app，而不是从零重写 | scan、diagnose、fix；优先字体、配色、交互状态、布局、组件和最终 polish |
| 视觉 agency 风格 | `high-end-visual-design` | 高端官网、品牌页、视觉冲击强的页面 | 强约束字体、图标、动效、double-bezel、按钮结构和宏观留白 |
| 图生代码工作流 | `image-to-code` | 视觉质量非常关键的网站或页面实现 | 先生成设计图，再深度分析，再实现；强调一屏/一节一张清晰图，避免压缩成小拼图 |
| 克制极简界面 | `minimalist-ui` | Notion/Linear 类文档感、工具感、低噪声界面 | 暖色单色、编辑感排版、平面 bento、少阴影、少渐变、强调文字层级 |

## 2. 使用前的准备

### 2.1 先确认当前环境真的有这个 skill

在 Codex 当前会话里，如果 skills 列表已经包含 `gpt-taste`、`design-taste-frontend`、`impeccable` 等名称，通常可以直接在提示词中点名使用。

如果不确定，可以先让 Codex 检查本机目录：

```powershell
Get-ChildItem "$env:USERPROFILE\.agents\skills"
Get-Content -Raw "$env:USERPROFILE\.agents\skills\design-taste-frontend\SKILL.md"
```

需要注意：`C:\Users\<用户名>\.agents\skills\` 是当前机器上的本地事实，不是所有机器都一样。把教程或提示词交给别人时，应让对方先确认自己机器上是否安装了同名 skill。

### 2.2 先给产品上下文，不要只说“做得高级”

taste skill 最吃上下文。建议至少提供这些信息：

- 产品是什么，用户是谁，当前页面承担什么任务；
- 这是 brand/marketing 页面，还是 product/tool/dashboard 页面；
- 当前项目技术栈，例如 React、Next.js、Vue、Tailwind、shadcn/ui、CSS Modules；
- 是否已有 `PRODUCT.md`、`DESIGN.md`、Figma、截图、现有页面、品牌色；
- 你希望更克制、更大胆、更密集、更轻盈、更技术、更生活方式，还是更运营工具；
- 明确反例，例如“不要做成暗色控制台 dashboard”“不要紫蓝 AI 渐变”“不要 3 卡片模板”；
- 需要覆盖哪些状态，例如 loading、empty、error、success、disabled、mobile、focus；
- 最终验收方式，例如 Browser 打开 localhost、桌面和移动端截图、跑 typecheck/build/test；

### 2.3 先读项目，不要直接套模板

如果是在已有项目中改 UI，正确顺序是：

1. 读 `package.json`，确认框架、脚本和依赖；
2. 读现有页面、组件、样式系统和设计 token；
3. 识别当前页面的产品目标、主要路径和交互状态；
4. 选择一个主 taste skill；
5. 再进入设计和实现；

尤其要注意依赖边界：`design-taste-frontend` 明确要求导入第三方库前先检查 `package.json`。如果项目没有安装 `framer-motion`、`gsap`、`@phosphor-icons/react` 或 `@radix-ui/react-icons`，不能假装它们存在。

## 3. 怎么选择taste-skill

### 3.1 做新 landing page

优先组合：

- 主 skill：`gpt-taste`；
- 辅助 skill：`image-to-code` 或 `high-end-visual-design`；

适合场景：

- 需要强视觉、强动效、强记忆点；
- 页面主要是展示、转化、品牌表达；
- 可以接受更大胆的构图和滚动叙事；

提示词重点：

- 明确页面目标、品牌气质、受众和 section 数量；
- 要求先输出设计方案或 `<design_plan>`；
- 如果使用 `image-to-code`，要求先生成每个 section 的清晰参考图，再分析再写代码；
- 要求 hero 标题不要换成 4 行以上；
- 要求用浏览器检查首屏、移动端、按钮可读性和横向滚动；

### 3.2 做产品 UI 或工具界面

优先组合：

- 主 skill：`design-taste-frontend`；
- 辅助 skill：`impeccable`；

适合场景：

- app shell、dashboard、settings、表单、数据列表、工作台、AI 工具界面；
- 更重视稳定、清晰、可用、状态完整，而不是纯视觉冲击；
- 需要和已有工程结构保持一致；

提示词重点：

- 说明这是 product register，不是 marketing register；
- 要求不要默认暗色控制室或泛 AI dashboard；
- 要求覆盖 empty、loading、error、focus、mobile；
- 要求先检查依赖和 Tailwind 版本；
- 要求最后用 Browser 做桌面和移动端视觉验收；

### 3.3 重设计已有项目

优先组合：

- 主 skill：`redesign-existing-projects`；
- 辅助 skill：`design-taste-frontend`；

适合场景：

- 已有页面功能可用，但审美普通；
- 不希望大改架构；
- 希望在原有栈内提升字体、颜色、间距、状态、组件细节；

正确流程：

1. scan：读项目，识别框架、样式方法、当前设计模式；
2. diagnose：列出普通、廉价、缺状态、缺响应式、缺可访问性的地方；
3. fix：按最小风险顺序改；
4. verify：运行构建或测试，并用浏览器看真实效果；

不要一上来就重写整个项目。`redesign-existing-projects` 的规则是“working with the existing stack”，更适合有边界的升级，而不是推翻重建。

### 3.4 做高端但克制的知识工具

优先组合：

- 主 skill：`minimalist-ui`；
- 辅助 skill：`design-taste-frontend`；

适合场景：

- 个人知识库、笔记、搜索、RAG、文档工作台；
- 希望像 Notion、Linear、Arc、Raycast 那样清爽、克制、可长期使用；
- 不需要强烈视觉特效；

提示词重点：

- 使用暖色单色或低噪声中性色；
- 避免渐变、重阴影、霓虹和玻璃拟态；
- 让文字层级、表格、列表、空状态成为设计核心；
- 控制卡片数量，不要所有东西都装进卡片；

### 3.5 做视觉稿到代码

优先组合：

- 主 skill：`image-to-code`；
- 辅助 skill：`gpt-taste`、`high-end-visual-design` 或 `minimalist-ui`；

适合场景：

- 用户主要关心“长得好不好看”；
- 网站、hero、产品页、品牌页需要先看图；
- 需要实现尽量贴近视觉参考；

关键规则：

- 先生成设计图，再深度分析，再写代码；
- 在 Codex 中优先一节一张大图，而不是把多个 section 压进一张小拼图；
- 如果文字、按钮或间距看不清，应重新生成该 section 的清晰图，而不是裁剪旧图；
- 实现时要贴近图，不要写着写着回到默认模板；

## 4. 推荐提示词模板

### 4.1 产品 UI 从零实现

```text
请使用 design-taste-frontend 作为主 skill，帮我实现这个产品界面。

产品背景：
- 产品：
- 用户：
- 页面目标：
- 当前技术栈：
- 已有设计约束：

设计要求：
- 这是 product UI，不是 marketing landing page；
- 不要默认暗色控制台，不要紫蓝 AI 渐变，不要 3 等分卡片模板；
- 优先清晰、克制、高级、可长期使用；
- 覆盖 loading、empty、error、focus、mobile 状态；
- 导入任何第三方库前先检查 package.json；

执行要求：
1. 先读项目结构和目标文件；
2. 给出简短实现计划；
3. 修改代码；
4. 运行 typecheck/build/test 中当前项目可用的校验；
5. 用 Browser 检查桌面和移动端渲染。
```

### 4.2 已有页面重设计

```text
请使用 redesign-existing-projects 审视并升级这个页面：<页面路径或路由>。

目标：
- 保留现有功能和数据流；
- 不迁移技术栈；
- 不大范围重写；
- 重点提升字体、色彩、间距、层级、状态、响应式和交互反馈；

请按 scan -> diagnose -> fix -> verify 执行。
先列出主要设计问题，再做有边界的改动。
最后用浏览器截图或描述确认桌面端和移动端没有溢出、遮挡、空白。
```

### 4.3 高视觉 landing page

```text
请使用 gpt-taste 做一个高质量 landing page，并严格执行它的 design_plan 规则。

背景：
- 品牌/产品：
- 受众：
- 转化目标：
- 希望的情绪：
- 不要出现的风格：

要求：
- 先输出 <design_plan>，包括布局、字体、section 结构、bento 密度、按钮对比度检查；
- 页面遵循 AIDA：Navigation、Attention、Interest、Desire、Action；
- hero 标题最多 2-3 行，不要堆假标签和假数据；
- 卡片网格不能留空洞；
- 动效只用 transform 和 opacity，不要造成横向滚动；
- 写完后启动 dev server，用 Browser 检查。
```

### 4.4 先图后码的网站设计

```text
请使用 image-to-code 工作流。

我要做一个 <网站类型>，共 <section 数量> 个 section。
先为每个 section 生成单独、清晰、可分析的设计图；
然后逐张分析文字、排版、按钮、颜色、间距、组件结构；
如果某一张文字或按钮看不清，请重新生成该 section 的清晰版本；
最后再实现代码，并尽量贴近生成图，不要回退成通用模板。
```

### 4.5 克制型工具界面

```text
请使用 minimalist-ui 为这个知识工具界面定调。

要求：
- 温暖中性色，低噪声，少渐变，少阴影；
- 以排版、信息密度和导航清晰度取胜；
- 不要做成营销页，也不要做成暗色监控台；
- 卡片只在需要表达层级时使用；
- 页面要适合长时间阅读、检索和整理。
```

## 5. 实战工作流

### 5.1 最稳的一条链路

如果是一个真实项目，推荐这样走：

1. 建立产品上下文：写清楚用户、页面目标、品牌语气、反例和核心任务；
2. 选择主 skill：产品 UI 用 `design-taste-frontend`，landing page 用 `gpt-taste`，旧项目升级用 `redesign-existing-projects`；
3. 让 Codex 先读项目：包括 `package.json`、目标页面、组件和样式；
4. 先要一个短计划：不要让它直接输出一大坨代码；
5. 实现时保持边界：只改目标页面和必要组件；
6. 验证工程：运行当前项目已有的 typecheck、lint、test、build；
7. 验证视觉：用 Browser 打开页面，检查桌面和移动端；
8. 回收设计资产：如果方向稳定，把关键规则写入 `DESIGN.md` 或项目文档；

### 5.2 和 impeccable 一起用

`impeccable` 更像一个项目级设计工作流，而不是单个样式包。它强调先加载 `PRODUCT.md` / `DESIGN.md`，再区分 brand/product register，最后进入具体命令。

常见命令包括：

| 命令 | 用途 |
| --- | --- |
| `teach` | 建立或补全产品设计上下文 |
| `shape` | 先做 UX/UI 方案，不立刻写代码 |
| `craft` | 先 shape，再端到端实现 |
| `audit` | 查可访问性、性能、响应式和技术质量 |
| `polish` | 发布前质量打磨 |
| `distill` | 去掉复杂度，让界面回到本质 |
| `harden` | 补错误、边界、i18n、生产状态 |
| `animate` | 增加有目的的动效 |
| `typeset` | 改排版、层级、字体 |
| `layout` | 修间距、节奏、视觉层级 |

如果项目还没有产品上下文，先用：

```text
请使用 impeccable teach，帮这个项目建立 PRODUCT.md 和 DESIGN.md。
先询问或推断必要信息，再把产品目标、用户、设计原则、禁用风格和 register 写清楚。
```

如果已经有上下文，想先定方案：

```text
请使用 impeccable shape 规划这个页面：<页面或功能>。
先读取 PRODUCT.md / DESIGN.md，判断这是 brand 还是 product register；
只输出设计结构、页面角色、关键交互、状态和验收标准，先不要写代码。
```

如果要实现：

```text
请使用 impeccable craft 实现这个功能：<功能描述>。
先加载项目上下文，再 shape，再实现，最后做浏览器验收。
```

## 6. 验收清单

### 6.1 设计验收

- 页面第一屏是否清楚表达任务，而不是只展示空泛口号；
- hero 标题是否过长，是否被挤成 4 行以上；
- 是否出现默认 AI 紫蓝渐变、霓虹发光、暗色控制室、三等分卡片等惯性设计；
- 卡片是否真的承担层级，不是所有内容都被盒子包起来；
- 字体是否和场景匹配，技术工具是否避免不合适的花哨 serif；
- 颜色是否有明确策略，而不是多个高饱和色同时抢戏；
- 主要按钮、次级按钮、禁用态、hover、active、focus 是否可识别；
- 信息层级是否能被快速扫描；
- 移动端是否改成稳定单列，而不是把桌面不对称布局硬压下去；

### 6.2 工程验收

- `package.json` 中确实存在被导入的第三方库；
- Tailwind 版本和配置写法匹配，不把 v4 写法塞进 v3 项目；
- Next.js Server Component / Client Component 边界正确；
- 动效没有用 `top`、`left`、`width`、`height` 做高频动画；
- 大面积 blur、noise、grain 没有放在滚动容器上造成性能问题；
- 没有任意堆 `z-50`、`z-[9999]`；
- 图片有合理尺寸、alt 文本和加载策略；
- 表单有 label、helper、error、disabled 和 focus 状态；

### 6.3 浏览器验收

至少检查这些项：

- 页面是否非空白；
- 桌面端首屏是否完整，关键 CTA 是否可见；
- 移动端是否无横向滚动；
- 文本是否溢出按钮、卡片或侧栏；
- 弹窗、菜单、tab、输入框、tooltip 是否能正常交互；
- 图片和图标是否加载；
- loading、empty、error 等状态是否真实可见；
- 控制台是否有明显错误；

如果是视觉要求高的页面，最好要求 Codex 提供截图观察结论，而不是只说“已完成”。

## 7. 常见错误和修正方式

### 7.1 页面变漂亮了，但不像你的产品

原因通常是产品上下文不足。修正方式：

- 补充用户、场景、业务目标和操作路径；
- 明确这是 product register 还是 brand register；
- 写出反例，例如“不要做成数据大屏”“不要像加密货币官网”“不要过度游戏化”；
- 优先使用 `impeccable teach` 或先补 `PRODUCT.md`；

### 7.2 skill 之间互相打架

例如 `gpt-taste` 可能倾向强视觉、强动效、AIDA landing page；`minimalist-ui` 则倾向安静、低噪声、文档式工具界面。两者不适合无脑叠加。

修正方式：

- 只选一个主 skill；
- 把另一个 skill 当作局部约束，而不是全局规则；
- 对产品工具优先 `design-taste-frontend`；
- 对营销页面优先 `gpt-taste` 或 `image-to-code`；
- 对知识工作台优先 `minimalist-ui`；

### 7.3 代码用了不存在的库

这是很常见的 AI 前端错误。修正方式：

- 要求 Codex 先读 `package.json`；
- 如果缺依赖，先询问是否安装，或使用项目已有库；
- 如果不能新增依赖，用 CSS 和现有组件实现；
- 引入后必须运行构建或 typecheck；

### 7.4 视觉稿好看，代码实现普通

这通常是 image-to-code 的“实现漂移”。修正方式：

- 让 Codex 逐张分析视觉图；
- 把文字、颜色、间距、按钮、组件、section rhythm 明确提取出来；
- 不要允许它把设计简化成默认模板；
- 实现后用截图对比原视觉图；

### 7.5 页面过度炫技

过度使用动效、3D、玻璃拟态、巨型标题，会伤害工具界面的长期可用性。修正方式：

- 对产品 UI 使用 `design-taste-frontend` 或 `minimalist-ui`；
- 降低 motion intensity 和 visual density；
- 保留 hover、active、focus、loading 这类功能性反馈；
- 删除没有帮助的装饰性标签、伪系统指标和假状态；

## 8. 推荐工作习惯

### 8.1 把 taste 当作评审标准，而不是一次性提示词

真正有效的用法不是只在开头说一句“用 taste skill”，而是把它贯穿在每一步：

- 设计前：用 taste skill 约束方向；
- 实现中：用 taste skill 限制依赖、状态和组件结构；
- 实现后：用 taste skill 做 design audit；
- 发布前：用 `polish`、`audit`、`harden` 或浏览器截图做最后验收；

### 8.2 每个项目沉淀自己的 DESIGN.md

taste skill 提供的是通用审美规则，项目最终应该沉淀自己的设计规则。例如：

- 字体；
- 颜色；
- 间距；
- 圆角；
- 阴影；
- 表单；
- 空状态；
- 错误状态；
- icon 风格；
- 动效强度；
- 禁用风格；

这样后续 Codex 不需要每次重新猜，也能避免同一个项目不同页面看起来像不同产品。

### 8.3 不要把“高级”写成唯一目标

更好的提示方式是把“高级”拆成可验证标准：

- 信息层级清楚；
- 页面没有模板味；
- 交互状态完整；
- 移动端稳定；
- 字体、间距、颜色有一致系统；
- 不滥用卡片、渐变、玻璃和动效；
- 首屏能看出这是哪个产品、解决什么问题；
- 代码能构建，浏览器里无明显错误；

## 9. 一套可复用的完整提示词

```text
请把这个前端任务按 taste-skill 工作流执行。

任务：
<写清楚要做的页面/组件/重设计目标>

上下文：
- 产品：
- 用户：
- 页面目标：
- 当前技术栈：
- 目标路由/文件：
- 设计反例：

选择 skill：
- 如果这是产品工具界面，请以 design-taste-frontend 为主；
- 如果这是高视觉 landing page，请以 gpt-taste 为主；
- 如果这是已有项目重设计，请以 redesign-existing-projects 为主；
- 如果视觉稿质量最重要，请使用 image-to-code，先图后码；
- 如果界面需要克制、知识工具感，请加入 minimalist-ui 约束；

执行要求：
1. 先读项目和 package.json，确认依赖、框架和样式系统；
2. 不要导入 package.json 里不存在的第三方库；
3. 先给简短设计/实现计划；
4. 实现代码，保持改动范围清晰；
5. 覆盖 loading、empty、error、focus、mobile 等必要状态；
6. 运行当前项目可用的 typecheck/lint/test/build；
7. 启动页面，用 Browser 检查桌面和移动端；
8. 最后总结改了什么、验证了什么、还有什么风险。
```

## 10. 资料来源与边界

本文只把当前本机可读到的 taste 类 `SKILL.md` 作为主要事实来源，不把它们描述成官方稳定接口。

- `gpt-taste`：确认其定位为高强度 UX/UI 与 GSAP 动效工程 skill，包含 `<design_plan>`、AIDA、hero 行数、gapless bento、GSAP 动效和按钮对比度等规则；
- `design-taste-frontend`：确认其定位为高级前端 UI/UX skill，包含依赖检查、React/Next/Tailwind 约束、状态覆盖、颜色/排版/布局/性能守卫和反 AI 味规则；
- `impeccable`：确认其定位为项目级前端设计工作流，需要加载 `PRODUCT.md` / `DESIGN.md`，并支持 `teach`、`shape`、`craft`、`audit`、`polish`、`distill`、`harden` 等命令；
- `redesign-existing-projects`：确认其定位为已有项目升级，流程是 scan、diagnose、fix，并强调在现有技术栈内做有边界的改造；
- `high-end-visual-design`：确认其定位为高端 agency 级视觉设计约束，强调字体、图标、double-bezel、按钮结构、宏观留白和动效；
- `image-to-code`：确认其定位为先生成视觉参考图、深度分析、再实现代码的工作流，强调 Codex 中一节一张清晰图和防止实现漂移；
- `minimalist-ui`：确认其定位为克制、编辑感、暖色单色和低噪声的界面风格；

实际使用时，最可靠的方法是让 Codex 在当前会话中重新读取目标 skill 的 `SKILL.md`，再结合目标项目代码、依赖、截图和浏览器验证执行。不要把本文中的路径、skill 名称或行为当成跨机器、跨版本永远不变的事实。
