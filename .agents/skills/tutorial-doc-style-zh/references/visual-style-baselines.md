# 视觉风格基线

当用户请求 `/tutorial-doc-style-zh handdrawn`、`/tutorial-doc-style-zh paper`、`/tutorial-doc-style-zh dark-system`，或教程需要高质量 AI/LLM 教学图时，读取本文件。旧请求 `paper-detail`、`paper-detailed` 视为 `paper` 的高信息密度变体，不作为独立风格族。

本文件只补充视觉风格和 prompt 规则。最终教程图的工具路线和禁令以 `SKILL.md` 硬规则为准：最终资产必须使用 `imagegen`，不得用确定性/本地渲染或本地后处理替代。

当前基线包含三种正向风格和两类反模式：

1. `handdrawn`：精致手绘教学板；
2. `paper`：专业论文风机制图或检查图；
3. `dark-system`：暗色工程通信/拓扑图；
4. 反模式 A：高亮、斜线填充或色块溢出真实语义对象；
5. 反模式 B：内容过度拥挤，边距不足，文字贴边或碰撞。

如果用户提供参考图，以参考图为最高优先级风格目标。若图像文件可落盘，复制到 `assets/style-baselines/`；若只在会话中可见，则把可复用视觉规则写入本文件。

## 1. `handdrawn`

用于直觉建立、概念对比、机制步骤草图，以及需要“有人在认真讲解”的教程场景。

视觉 DNA：

- 像精心绘制的人类教学笔记，不像通用 UI dashboard，也不像随手 doodle；
- 使用干净浅色背景、可读的手绘线条、克制的语义颜色；
- 对象边界、箭头、标签和小例子在阅读尺寸下容易检查；
- 颜色必须绑定真实语义，不做装饰；
- 优先短标签和小 worked example，不在图内塞长段落；
- 中文承担读者辅助解释，稳定英文术语保留专业环境中的写法；
- 面板、箭头、标签和对象之间保留充足空间。

硬性反面：

- 不使用通用填充卡片、贴纸、dashboard chrome 或视觉特效作为主体语言；
- 不把 `handdrawn` 简化成“只有字体像手写，布局仍然机械”；
- 高亮、斜线填充、色块、箭头和标签不得逃出其语义对象；
- 不把长句挤进窄条，不让文字贴边；
- 不使用会损害结构可读性的装饰性潦草效果。

中文 AI/LLM 教程默认风格：

- 用户未指定视觉风格时，优先使用 `handdrawn`；
- 图应专业、信息丰富、可教学，不幼稚、不随意；
- 使用教学板隐喻：认真绘制的框、箭头、时间线、状态变化、小例子和简短双语标签；
- 中文用于读者辅助标签，保留 DP、TP、KV cache、all-reduce、replica、prefill、decode、router、shard 等稳定英文术语；
- 每张图只承担一个教学任务。拥挤时拆成多张 `imagegen` 图，不压缩标签。

Prompt skeleton：

```text
Refined hand-drawn Chinese technical tutorial diagram for an AI/LLM engineering article. Clean warm-white teaching-board background, careful human-authored composition, readable ink-like linework, restrained semantic colors, generous margins, clear arrows and state boundaries. Use short Chinese labels plus stable English technical terms. Show concrete data/state ownership and before/after transitions. Professional, information-rich, accurate, easy to understand, lively but not decorative. No dense paragraphs, no tiny text, no cramped labels, no UI dashboard cards, no stickers, no watermark, no highlight or fill outside its semantic object.
```

## 2. `paper`

用于需要权威感、精确性和专业论文图气质的机制说明。该模式同时覆盖稀疏机制图和高信息密度检查图；密度由教学任务决定，不另设风格族。

视觉 DNA：

- 白色或近白背景，细黑/灰线条，克制颜色，清晰面板结构；
- 形状、标签、公式、data-flow/state-flow 关系要足够精确，方便技术检查；
- 颜色只用于区分语义类别、边界、阶段或对比；
- 构图冷静、学术：面板平衡、小图例、短注释、无视觉戏剧化；
- 高密度 `paper` 图可包含公式摘要、shape table、legend、双语 note 或 recap strip，但必须服务慢速检查；
- 图内文本必须有明确用途，宽泛解释移到正文。

硬性反面：

- 不使用饱和海报色、营销 badge、装饰 icon、glossy card 或手写风；
- 不把多个无关机制混在一张图里；
- 不用密集 recap strip 替代正文解释；
- 新概念的第一张图不要密到压垮读者。

Prompt skeleton：

```text
Professional paper-style mechanism figure for a Chinese AI/LLM engineering tutorial. White background, thin black and gray linework, restrained semantic colors, precise panel layout, clear labels, shape/table snippets only when useful, concise bilingual notes, accurate arrows and boundaries, calm academic composition. No saturated poster colors, no decorative icons, no glossy cards, no handwritten style, no cramped labels.
```

## 3. `dark-system`

用于 runtime、deployment、topology、communication hotspot、observability 和 performance-cost 图。

视觉 DNA：

- 深炭灰或近黑背景，不做纯装饰霓虹；
- 水平 pipeline 或 topology layout，阶段卡片干净对齐；
- Blue = stable path / baseline flow / no-coordination stage；
- Green = local compute / local state / successful handoff；
- Orange = coordination boundary / bottleneck / communication hotspot；
- Red = risk only；
- 可加入紧凑 cost card 和 topology strip；
- 英文技术术语可更自由使用，关键教学点配短中文辅助标签；
- 每个 stage 要有呼吸空间。

硬性反面：

- 不做 cyberpunk decoration；
- 不堆 neon；
- 不使用随机代码纹理；
- 不使用 tiny labels；
- 不做 dense dashboard chrome。

Prompt skeleton：

```text
Dark engineering system topology diagram for a Chinese AI/LLM tutorial. Charcoal background, clean horizontal pipeline or topology layout, aligned stage cards, blue stable path, green local compute/state, orange coordination boundary or bottleneck, red only for risk. Short Chinese helper labels with stable English technical terms, compact cost card when useful, generous spacing, readable labels. No cyberpunk decoration, no neon overload, no random code texture, no dashboard chrome.
```

## 4. 反模式 A：语义高亮溢出

问题：

- 高亮、填充、斜线、glow 或强调区域超出它应该代表的对象；
- 读者无法判断彩色区域是真实语义范围还是装饰。

修复：

- 让斜线和填充严格落在对象、区域、cell 或边界内部；
- 在 `imagegen` prompt 中明确禁止 fill/hatching outside semantic boundaries；
- 优先使用稀疏内部斜线，少用大面积半透明覆盖；
- 生成后检查：“读者能否不用猜就识别该视觉强调覆盖的确切对象、区域或状态？”

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

## 6. 模式选择

- `handdrawn`：优先用于第一直觉、概念对比、步骤机制；
- `paper`：优先用于精确机制、论文风总结、检查型图；
- `dark-system`：优先用于通信、profiling、拓扑、部署权衡；
- 不要把一种模式强行套到所有图上，但同一小节或图族内应保持风格一致。
