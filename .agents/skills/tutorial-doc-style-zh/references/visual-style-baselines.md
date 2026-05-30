# 视觉风格基线

当用户请求 `/tutorial-doc-style-zh handdrawn`、`/tutorial-doc-style-zh paper`、`/tutorial-doc-style-zh dark-paper`，或教程需要高质量教学图时，读取本文件。

本文件只补充视觉风格和 prompt 规则。最终教程图的生成流程以 `SKILL.md` 为准：所有基于本 skill 处理的教程文档，最终教学图必须使用 `imagegen`。

强制生成边界：教学图只能通过 `imagegen` skill 创建、修订和纠错。不要先生成粗糙或不完美的图，再用脚本、绘图工具、图像编辑器、SVG/HTML/canvas 重构、栅格或矢量清理、合成、重标注、裁剪式修复或任何其他生成后处理修补。若图未通过视觉、文字、布局或事实审查，应改进 `imagegen` prompt、简化标签、拆分图、增加或细化参考图，或从头重新生成。

当前基线包含三种正向风格和三类反模式：

1. `handdrawn`：精致手绘教学板；
2. `paper`：专业论文风格插图；
3. `dark-paper`：暗色系专业论文风格插图；
4. 反模式 A：文本不清晰，或高亮、色块、斜线填充等溢出真实语义对象；
5. 反模式 B：内容过度拥挤，边距不足，文字贴边或碰撞；
6. 反模式 C：图片内部出现大标题，挤占主体内容空间。

如果用户提供参考图，以参考图为最高优先级风格目标。若图像文件可落盘，复制到 `assets/style-baselines/`；若只在会话中可见，则把可复用视觉规则写入本文件，但写入前需要说明目的和内容并获得用户授权。

## 1. `handdrawn`

用于直觉建立、概念对比、机制步骤草图，以及需要“有人在认真讲解”的教程场景。

视觉 DNA：

- 像精心绘制的人类教学笔记，不像通用 UI dashboard，也不像随手 doodle；
- 使用干净浅色背景、可读的手绘线条、克制的语义颜色；
- 对象边界、箭头、标签和小例子在阅读尺寸下容易检查；
- 颜色绑定真实语义，不做装饰；
- 优先短标签和小 worked example，不在图内塞长段落；
- 面向读者的解释、note 和辅助标签应使用目标文档语言，同时标准英文技术术语、短语和标识符保留专业语境中的写法；
- 面板、箭头、标签和对象之间保留充足空间。

硬性反面：

- 不过度使用通用填充卡片、贴纸、dashboard chrome 或视觉特效作为主体语言；
- 不把 `handdrawn` 简化成“只有字体像手写，布局仍然机械”；
- 高亮、斜线填充、色块、箭头和标签不得逃出其语义对象；
- 不把长句挤进窄条，不让文字贴边；
- 不使用会损害结构可读性的装饰性潦草效果。

教程默认风格：

- 用户未指定视觉风格时，优先使用 `handdrawn`；
- 图应专业、信息丰富、可教学，不幼稚、不随意；
- 使用教学板隐喻：认真绘制的框、箭头、时间线、状态变化、小例子，以及必要时配合标准英文术语的简洁目标语言辅助标签；
- 面向读者的辅助标签、note 和解释使用目标文档语言；保留标准英文技术术语、短语、API、模型名和框架标识符；
- 每张图只承担一个教学任务。拥挤时拆成多张 `imagegen` 图，不压缩标签。

Prompt skeleton：

```text
Refined hand-drawn technical tutorial diagram. Clean warm-white teaching-board background, careful human-authored composition, readable ink-like linework, restrained semantic colors, generous margins, clear arrows and state boundaries. Use short reader-facing labels and helper notes in the target document language, while keeping standard English technical terms, phrases, APIs, model names, and framework identifiers in English. Show concrete objects, relationships, state ownership, before/after transitions, or small worked examples according to the teaching goal. Professional, information-rich, accurate, easy to understand, lively but not decorative. No large standalone title inside the image, no dense paragraphs, no tiny text, no cramped labels, no UI dashboard cards, no stickers, no watermark, no highlight or fill outside its semantic object.
```

## 2. `paper`

用于需要权威感、精确性和专业论文图气质的机制说明、结构对比、算法流程、数据布局、实验结果解释或检查型总结。该模式可稀疏也可高密度；密度由教学任务决定。

视觉 DNA：

- 白色或近白背景，细黑/灰线条，克制颜色，清晰面板结构；
- 形状、标签、公式、data-flow/state-flow 关系要足够精确，方便技术检查；
- 颜色只用于区分语义类别、边界、阶段或对比；
- 构图冷静、学术：面板平衡、小图例、短注释、无视觉戏剧化；
- 高密度 `paper` 图可包含公式摘要、shape table、legend、带标准英文术语的目标语言 note 或 recap strip，但必须服务慢速检查；
- 图内文本必须有明确用途，宽泛解释移到正文。

硬性反面：

- 不使用饱和海报色、营销 badge、装饰 icon、glossy card 或手写风；
- 不把多个无关机制混在一张图里；
- 不用密集 recap strip 替代正文解释；
- 新概念的第一张图不要密到压垮读者。

Prompt skeleton：

```text
Professional paper-style technical tutorial figure. White or near-white background, thin black and gray linework, restrained semantic colors, precise panel layout, clear labels, shape/table snippets only when useful, concise notes in the target document language with standard English technical terms preserved, accurate arrows and boundaries, calm academic composition. Use visual density appropriate to the teaching goal. No large standalone title inside the image, no saturated poster colors, no decorative icons, no glossy cards, no handwritten style, no cramped labels.
```

## 3. `dark-paper`

用于希望保留论文式严谨感，同时需要暗色视觉氛围的系统图、机制图、通信/拓扑图、部署图、性能代价图或综合对比图。它不是 cyberpunk 风格，也不是 dashboard 风格。

视觉 DNA：

- 深炭灰或近黑背景，线条和文字保持论文图式的克制与清晰；
- 结构可以是面板、pipeline、topology、timeline 或 comparison layout，由教学目的决定；
- 使用少量语义颜色区分路径、阶段、边界、瓶颈、风险或对比组；
- 色彩语义保持一致，例如 blue 表示稳定路径或 baseline，green 表示局部计算或成功状态，orange 表示协调边界或瓶颈，red 只表示风险；
- 可加入紧凑 legend、cost card、topology strip 或 recap strip，但不能压缩正文解释；
- 标准英文技术术语可更自由使用，关键教学点配目标文档语言的短辅助标签、note 和解释；
- 暗色背景下仍必须保证文字清晰、层级明确、留白充足。

硬性反面：

- 不做 cyberpunk decoration；
- 不堆 neon；
- 不使用随机代码纹理；
- 不使用 tiny labels；
- 不做 dense dashboard chrome；
- 不让暗色氛围压过技术结构。

Prompt skeleton：

```text
Dark paper-style technical tutorial figure. Charcoal or near-black background, precise academic linework, restrained semantic colors, clean panel or topology composition, readable labels, concise notes in the target document language with standard English technical terms preserved, accurate arrows and boundaries. Use blue for baseline or stable paths, green for local compute or successful state, orange for coordination boundary or bottleneck, red only for risk. Professional, rigorous, calm, high-contrast, generous spacing. No large standalone title inside the image, no cyberpunk decoration, no neon overload, no random code texture, no dashboard chrome, no tiny labels.
```

## 4. 反模式 A：文本不清晰或语义高亮溢出

问题：

- 文字模糊、过小、贴边或难以辨认；
- 高亮、填充、斜线、glow 或强调区域超出它应该代表的对象；
- 读者无法判断彩色区域是真实语义范围还是装饰。

修复：

- 缩短标签、增大画布、拆图，或把长解释移到正文；
- 让斜线和填充严格落在对象、区域、cell 或边界内部；
- 在 `imagegen` prompt 中明确要求 readable labels，并禁止 fill/hatching outside semantic boundaries；
- 优先使用稀疏内部斜线，少用大面积半透明覆盖；
- 生成后检查：“读者能否不用猜就读清标签，并识别视觉强调覆盖的确切对象、区域或状态？”

## 5. 反模式 B：布局拥挤

问题：

- 文字贴边；
- 标签与箭头、矩阵或面板碰撞；
- 底部条过薄，装不下内容；
- 卡片间距过窄，整张图显得紧绷和不专业。

修复：

- 增大画布、拆图或缩短标签；
- 1600x900 图中保留 24-32 px 以上内部 padding；
- 底部条必须足够高，否则删除；
- 长解释移到正文；
- 按最终显示尺寸检查，拒绝任何贴边或碰撞文本；
- 生成后检查：“这张图像认真教学笔记，还是像被模板硬塞进去？”

## 6. 反模式 C：图片内部大标题

问题：

- 图片顶部或内部出现一行占据视觉重心的大标题；
- 大标题承担了文档标题或章节标题功能，却没有增加图内教学信息；
- 大标题压缩主体图解空间，让读者先看到标题而不是机制、结构、流程或对比关系。

修复：

- 把标题放在正文标题、图注或图片前后的解释中，不放进最终教学图内部；
- 图片内部只保留必要的短标签、分区名、对象名、箭头说明和局部注释；
- 生成 prompt 中明确禁止 large standalone title / headline inside the image；
- 需要说明图的主题时，用正文图注完成，例如“图 X：GPU 内部并行执行结构”，而不是把这句话画进图片顶部；
- 生成后检查：“去掉图片顶部大标题后，主体内容是否仍然完整表达教学目的？”

## 7. 模式选择

- `handdrawn`：优先用于第一直觉、概念对比、步骤机制；
- `paper`：优先用于精确机制、论文风总结、检查型图；
- `dark-paper`：优先用于暗色论文风机制图、系统图、拓扑图、部署权衡和性能代价图；
- 不要把一种模式强行套到所有图上，但同一小节或图族内应保持风格一致。
