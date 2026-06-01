---
tags: [automation, arxiv, papers, ai]
updated: 2026-06-01
description: Orin arXiv AI 论文日报自动化的抓取范围、执行流程、验证记录和状态文件约定。
---

# 1 Orin arXiv AI 论文日报自动化

## 1.1 目标

每天早上 10 点左右抓取前一天提交到 arXiv 的 AI、LLM 推理、AI 模型相关论文，生成一份高质量 Markdown 日报。日报只作为聚合报告保存，不创建单篇论文笔记，也不把论文正文长期保存到 `rag/`。

## 1.2 抓取范围

主查询使用 arXiv API 的前一天提交日期窗口：

- 日期窗口：`submittedDate:[YYYYMMDD0000 TO YYYYMMDD2359]`；
- 分类范围：`cs.AI`、`cs.CL`、`cs.LG`、`cs.CV`、`cs.NE`、`cs.IR`、`cs.DC`、`cs.PF`、`cs.AR`、`stat.ML`；
- 关键词范围：AI、artificial intelligence、machine learning、large language model、LLM、foundation model、generative AI、transformer、VLM、LLM reasoning、LLM inference、inference engine、model serving、continuous batching、speculative decoding、KV cache、KVCache、key-value cache、PagedAttention、FlashAttention、attention algorithm、long context、quantization、model compression、Mixture of Experts、MoE、RAG、AI agent 等；

当 arXiv API 被限流或失败时，脚本会继续使用 arXiv HTML 搜索页按关键词 OR 批次兜底，避免单一入口失败导致整日报中断。

## 1.3 输出约定

日报输出路径：

```text
notes/papers/arxiv-ai-daily/YYYY-MM-DD arxiv ai papers.md
```

状态文件：

```text
rag/arxiv_papers/processed_papers.json
```

运行期临时数据：

```text
code/arxiv_paper_radar/.tmp/YYYY-MM-DD-papers.json
```

`.tmp` 目录只用于当天自动化的中间数据，已加入 `.gitignore`。临时 JSON 中可以包含正文抽取结果，供 AI 在同一次运行中分析；该文件不进入 Git，不作为长期知识资产保存。

## 1.4 日报必须包含的字段

- 论文标题；
- 作者；
- arXiv 分类与自动标签；
- 论文自己的链接，即 `https://arxiv.org/abs/<id>`；
- PDF/HTML 正文抽取状态；
- 是否发现开源 Demo 或代码；
- 开源 Demo 或代码库链接；
- 论文核心问题；
- 原理与核心思想；
- 架构与流程；
- 实验与证据；
- 重要性评级，建议使用 `S/A/B/C`；
- 与 Orin 已有主题的连接，例如 vLLM、AIBrix、KVCache、调度、RAG、模型架构等；

## 1.5 推荐执行流程

1. 安装或刷新依赖；

```powershell
python -m pip install -r code\arxiv_paper_radar\requirements.txt
```

2. 计算目标日期：自动化在 2026-06-01 运行时，目标日期必须是 2026-05-31；

3. 抓取前一天候选论文，并写入临时 JSON；

```powershell
python code\arxiv_paper_radar\orin_arxiv_radar.py collect `
  --target-date 2026-05-31 `
  --state rag\arxiv_papers\processed_papers.json `
  --output code\arxiv_paper_radar\.tmp\2026-05-31-papers.json
```

4. 生成 Markdown 日报骨架；

```powershell
python code\arxiv_paper_radar\orin_arxiv_radar.py render-report `
  --input code\arxiv_paper_radar\.tmp\2026-05-31-papers.json `
  --output "notes\papers\arxiv-ai-daily\2026-05-31 arxiv ai papers.md"
```

5. 由 Codex 自动化读取临时 JSON 和日报骨架，对每篇新增论文补全 AI 分析、核心思想、架构、实验证据和重要性评级；

6. 日报确认完成后，更新去重状态；

```powershell
python code\arxiv_paper_radar\orin_arxiv_radar.py mark-analyzed `
  --input code\arxiv_paper_radar\.tmp\2026-05-31-papers.json `
  --state rag\arxiv_papers\processed_papers.json `
  --report "notes/papers/arxiv-ai-daily/2026-05-31 arxiv ai papers.md"
```

7. 按 Orin 规则执行 `git status`、`git add -A`、英文提交信息和 `git push origin <当前分支>`；

## 1.6 可行性验证记录

2026-06-01 验证结果：

- arXiv API 最小查询在当前运行环境中返回 `HTTP 429 Rate exceeded`，脚本因此必须保留重试/失败隔离和 HTML 兜底；
- arXiv 分类列表页可访问，`https://arxiv.org/list/cs.AI/recent?show=25` 成功返回论文列表；
- arXiv HTML 搜索页可访问，关键词查询 `KVCache OR "KV cache" OR "LLM inference"` 成功解析到论文条目、提交日期、标题、作者、摘要和 PDF 链接；
- arXiv 论文详情页可访问，`https://arxiv.org/abs/<id>` 可解析摘要、作者、分类、提交日期和 HTML 正文链接；
- PDF 下载可行但较慢，首 1MB 下载约 23 秒，因此正式流程优先使用 arXiv HTML 正文页，HTML 不存在时再回退 PDF；
- `https://arxiv.org/html/2605.31492` 成功抽取 HTML 正文，约 57,786 字符，说明 HTML 正文路径可支撑日常正文分析；

## 1.7 自动化提示词核心要求

自动化不应只复述摘要。它必须结合摘要、正文抽取文本、论文链接、代码链接和标签，对每篇新增论文给出可读、可检索、可复用的简要分析。若某篇论文正文抽取失败，必须在日报中标注“正文抽取失败”，并基于摘要给出较低置信度分析。

## 1.8 当前自动化配置

- 自动化 ID：`orin-arxiv-ai-paper-radar`；
- 名称：`Orin arXiv AI Paper Radar`；
- 运行时间：每天 10:00；
- 工作区：`C:\AIWorks\Orin`；
- 执行环境：local；
- 状态：ACTIVE；
