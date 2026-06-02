---
tags: [automation, arxiv, papers, ai]
updated: 2026-06-02
description: Orin arXiv AI 论文日报自动化的抓取范围、执行流程、验证记录和状态文件约定，当前版本以 OAI-PMH 为主采集入口。
---

# 1 Orin arXiv AI 论文日报自动化

## 1.1 目标

每天早上 10 点左右抓取前一天提交到 arXiv 的 AI、LLM 推理、AI 模型相关论文，生成一份高质量 Markdown 日报。日报只作为聚合报告保存，不创建单篇论文笔记，也不把论文正文长期保存到 `rag/`。

## 1.2 抓取范围

主查询改为 OAI-PMH 单日轮询，再用 `arXivRaw` 里的 `version v1` GMT 时间换算到 `Asia/Shanghai`：

- 轮询集合：`cs:cs:AI`、`cs:cs:CL`、`cs:cs:LG`、`cs:cs:CV`、`cs:cs:NE`、`cs:cs:IR`、`cs:cs:DC`、`cs:cs:PF`、`cs:cs:AR`、`stat:stat:ML`；
- 主过滤语义：`v1 GMT -> Asia/Shanghai date == target_date`；
- 查询窗口：默认轮询 `target_date` 到 `target_date + 2` 的单天 OAI 记录，并分别发起请求；
- 元数据补全：对入选论文访问 `https://arxiv.org/abs/<id>`，补充作者、摘要、分类、HTML 正文链接和显式外链；

这样做的原因是 arXiv API 与 HTML 搜索页容易限流，而 `recent/search` 更接近公告日语义，不适合精确满足“按上海自然日统计首次提交”的要求。

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

4. 生成 Markdown 日报；

```powershell
python code\arxiv_paper_radar\orin_arxiv_radar.py render-report `
  --input code\arxiv_paper_radar\.tmp\2026-05-31-papers.json `
  --output "notes\papers\arxiv-ai-daily\2026-05-31 arxiv ai papers.md"
```

5. `render-report` 会直接把每篇论文渲染成中文分析稿，包括核心问题、关键思路、架构、证据、评级、Orin 连接和后续问题；

6. 日报确认完成后，更新去重状态；

```powershell
python code\arxiv_paper_radar\orin_arxiv_radar.py mark-analyzed `
  --input code\arxiv_paper_radar\.tmp\2026-05-31-papers.json `
  --state rag\arxiv_papers\processed_papers.json `
  --report "notes/papers/arxiv-ai-daily/2026-05-31 arxiv ai papers.md"
```

7. 按 Orin 规则执行 `git status`、`git add -A`、英文提交信息和 `git push origin <当前分支>`；

## 1.6 可行性验证记录

2026-06-02 验证结果：

- arXiv API 与 HTML 搜索页都曾触发 `HTTP 429`，因此不再适合作为主采集入口；
- `https://oaipmh.arxiv.org/oai?verb=Identify` 与 `ListRecords` 在当前环境可稳定访问；
- OAI 的 `datestamp` 更适合做“公开可见记录增量轮询”，真正的“首次提交日”需要读取 `arXivRaw version v1 date` 后自行换算时区；
- `https://arxiv.org/abs/<id>` 可稳定解析摘要、作者、分类、提交日期和 HTML 正文链接；
- PDF 下载可行但较慢，因此正式流程只对前若干篇论文尝试全文抽取，其余条目保留摘要级分析并明确降低置信度；

## 1.7 自动化提示词核心要求

自动化不应只复述摘要。它必须结合摘要、正文抽取文本、论文链接、代码链接和标签，对每篇新增论文给出可读、可检索、可复用的简要分析。若某篇论文正文抽取失败，或因规模策略未抽取全文，必须在日报中明确标注，并基于摘要给出较低置信度分析。

## 1.8 当前自动化配置

- 自动化 ID：`orin-arxiv-ai-paper-radar`；
- 名称：`Orin arXiv AI Paper Radar`；
- 运行时间：每天 10:00；
- 工作区：`C:\AIWorks\Orin`；
- 执行环境：local；
- 状态：ACTIVE；
