---
tags: [papers, arxiv, ai, llm-inference]
updated: 2026-06-02
description: 2026-06-01 arXiv AI、LLM 推理与 AI 模型论文日报；本次运行因 arXiv 限流仅形成低置信度状态报告。
---

# 1 2026-06-01 arXiv AI 论文雷达

## 1.1 运行摘要

- 抓取目标日期：2026-06-01；
- 临时 JSON 输出：`code/arxiv_paper_radar/.tmp/2026-06-01-papers.json`；
- 脚本统计结果：候选论文 0 篇，新增待分析论文 0 篇，已跳过重复论文 0 篇；
- 结果解释：本次 `0` 篇不是“确认当天无论文”，而是当前运行环境访问 arXiv API 与 arXiv HTML 搜索页时均触发 `HTTP 429` 限流后的空结果；
- 日报置信度：低；

## 1.2 结论

- 事实：`collect` 与 `render-report` 已按既定流程执行并成功落盘；
- 事实：arXiv API 主查询返回 `HTTP 429`；
- 事实：4 组 HTML 搜索兜底查询全部返回 `HTTP 429`；
- 事实：因此临时 JSON 中没有任何可进入逐篇分析流程的新论文；
- 推断：当前 IP、请求频率或 arXiv 反滥用策略导致自动化窗口内的查询被整体限流；
- 未验证：2026-06-01 是否确实存在满足本自动化筛选条件的新论文，当前这次运行不能给出可信结论；

## 1.3 数据源状态

| 来源 | 状态 | 说明 |
|---|---|---|
| arXiv API | 失败 | `submittedDate:[202606010000 TO 202606012359]` 主查询返回 `HTTP 429`，未拿到候选集； |
| arXiv HTML 搜索批次 1 | 失败 | `large language model / LLM / foundation model / generative AI / transformer / VLM / multimodal model` 查询返回 `HTTP 429`； |
| arXiv HTML 搜索批次 2 | 失败 | `LLM reasoning / inference / model serving / speculative decoding / decoding` 查询返回 `HTTP 429`； |
| arXiv HTML 搜索批次 3 | 失败 | `KV cache / PagedAttention / FlashAttention / long context` 查询返回 `HTTP 429`； |
| arXiv HTML 搜索批次 4 | 失败 | `quantization / model compression / MoE / RAG / AI agent` 查询返回 `HTTP 429`； |

## 1.4 今日论文总表

| 评级 | arXiv ID | 标题 | 标签 | 论文链接 | 开源Demo/代码 | 一句话结论 |
|---|---|---|---|---|---|---|
| N/A | - | 本次自动化未获得可分析论文样本 | - | - | - | 当前输出仅能证明采集受限流阻断，不能证明 2026-06-01 没有新论文； |

## 1.5 论文详析

本次没有逐篇详析条目。

原因不是“没有新增论文”已被确认，而是采集链路在进入逐篇分析前就被 arXiv 限流拦截，因此：

- 没有论文级 arXiv 链接可列出；
- 没有可核验的开源 Demo 或代码库链接；
- 没有正文抽取成功或失败的逐篇记录；
- 没有可负责任给出的 S/A/B/C 重要性评级；

## 1.6 与 Orin 的知识连接

- 对 `code/arxiv_paper_radar/orin_arxiv_radar.py` 而言，这次运行暴露的不是主题判断问题，而是数据入口鲁棒性问题；
- 对 Orin 现有 `vLLM`、`AIBrix`、`KVCache`、`调度`、`RAG` 等主题沉淀而言，今天没有新增论文内容被成功纳入知识流水线；
- 这说明当前自动化离“稳定的知识摄入通道”还有一个明显缺口：需要把采集可用性本身视为知识生产前置依赖；

## 1.7 监控来源链接

- arXiv API 主查询：[https://export.arxiv.org/api/query?search_query=submittedDate%3A%5B202606010000+TO+202606012359%5D+AND+%28cat%3Acs.AI+OR+cat%3Acs.CL+OR+cat%3Acs.LG+OR+cat%3Acs.CV+OR+cat%3Acs.NE+OR+cat%3Acs.IR+OR+cat%3Acs.DC+OR+cat%3Acs.PF+OR+cat%3Acs.AR+OR+cat%3Astat.ML+OR+all%3AAI+OR+all%3A%22artificial+intelligence%22+OR+all%3A%22machine+learning%22+OR+all%3A%22deep+learning%22+OR+all%3A%22large+language+model%22+OR+all%3ALLM+OR+all%3ALLMs+OR+all%3A%22foundation+model%22+OR+all%3A%22generative+AI%22+OR+all%3Atransformer+OR+all%3A%22vision+language+model%22+OR+all%3AVLM+OR+all%3A%22multimodal+model%22+OR+all%3A%22reasoning+model%22+OR+all%3A%22LLM+reasoning%22+OR+all%3Ainference+OR+all%3A%22LLM+inference%22+OR+all%3A%22inference+engine%22+OR+all%3A%22model+serving%22+OR+all%3A%22serving+system%22+OR+all%3A%22continuous+batching%22+OR+all%3A%22speculative+decoding%22+OR+all%3Adecoding+OR+all%3A%22KV+cache%22+OR+all%3AKVCache+OR+all%3A%22key-value+cache%22+OR+all%3APagedAttention+OR+all%3AFlashAttention+OR+all%3A%22attention+algorithm%22+OR+all%3A%22long+context%22+OR+all%3A%22context+window%22+OR+all%3Aquantization+OR+all%3A%22model+compression%22+OR+all%3Apruning+OR+all%3Adistillation+OR+all%3A%22Mixture+of+Experts%22+OR+all%3AMoE+OR+all%3A%22sparse+model%22+OR+all%3A%22retrieval+augmented+generation%22+OR+all%3ARAG+OR+all%3A%22AI+agent%22%29&start=0&max_results=1000&sortBy=submittedDate&sortOrder=descending](https://export.arxiv.org/api/query?search_query=submittedDate%3A%5B202606010000+TO+202606012359%5D+AND+%28cat%3Acs.AI+OR+cat%3Acs.CL+OR+cat%3Acs.LG+OR+cat%3Acs.CV+OR+cat%3Acs.NE+OR+cat%3Acs.IR+OR+cat%3Acs.DC+OR+cat%3Acs.PF+OR+cat%3Acs.AR+OR+cat%3Astat.ML+OR+all%3AAI+OR+all%3A%22artificial+intelligence%22+OR+all%3A%22machine+learning%22+OR+all%3A%22deep+learning%22+OR+all%3A%22large+language+model%22+OR+all%3ALLM+OR+all%3ALLMs+OR+all%3A%22foundation+model%22+OR+all%3A%22generative+AI%22+OR+all%3Atransformer+OR+all%3A%22vision+language+model%22+OR+all%3AVLM+OR+all%3A%22multimodal+model%22+OR+all%3A%22reasoning+model%22+OR+all%3A%22LLM+reasoning%22+OR+all%3Ainference+OR+all%3A%22LLM+inference%22+OR+all%3A%22inference+engine%22+OR+all%3A%22model+serving%22+OR+all%3A%22serving+system%22+OR+all%3A%22continuous+batching%22+OR+all%3A%22speculative+decoding%22+OR+all%3Adecoding+OR+all%3A%22KV+cache%22+OR+all%3AKVCache+OR+all%3A%22key-value+cache%22+OR+all%3APagedAttention+OR+all%3AFlashAttention+OR+all%3A%22attention+algorithm%22+OR+all%3A%22long+context%22+OR+all%3A%22context+window%22+OR+all%3Aquantization+OR+all%3A%22model+compression%22+OR+all%3Apruning+OR+all%3Adistillation+OR+all%3A%22Mixture+of+Experts%22+OR+all%3AMoE+OR+all%3A%22sparse+model%22+OR+all%3A%22retrieval+augmented+generation%22+OR+all%3ARAG+OR+all%3A%22AI+agent%22%29&start=0&max_results=1000&sortBy=submittedDate&sortOrder=descending)；
- arXiv HTML 搜索批次 1：[https://arxiv.org/search/?query=%22large+language+model%22+OR+LLM+OR+LLMs+OR+%22foundation+model%22+OR+%22generative+AI%22+OR+transformer+OR+%22vision+language+model%22+OR+VLM+OR+%22multimodal+model%22&searchtype=all&abstracts=show&order=-announced_date_first&size=200&start=0](https://arxiv.org/search/?query=%22large+language+model%22+OR+LLM+OR+LLMs+OR+%22foundation+model%22+OR+%22generative+AI%22+OR+transformer+OR+%22vision+language+model%22+OR+VLM+OR+%22multimodal+model%22&searchtype=all&abstracts=show&order=-announced_date_first&size=200&start=0)；
- arXiv HTML 搜索批次 2：[https://arxiv.org/search/?query=%22LLM+reasoning%22+OR+%22reasoning+model%22+OR+%22chain+of+thought%22+OR+inference+OR+%22LLM+inference%22+OR+%22inference+engine%22+OR+%22model+serving%22+OR+%22continuous+batching%22+OR+%22speculative+decoding%22+OR+decoding&searchtype=all&abstracts=show&order=-announced_date_first&size=200&start=0](https://arxiv.org/search/?query=%22LLM+reasoning%22+OR+%22reasoning+model%22+OR+%22chain+of+thought%22+OR+inference+OR+%22LLM+inference%22+OR+%22inference+engine%22+OR+%22model+serving%22+OR+%22continuous+batching%22+OR+%22speculative+decoding%22+OR+decoding&searchtype=all&abstracts=show&order=-announced_date_first&size=200&start=0)；
- arXiv HTML 搜索批次 3：[https://arxiv.org/search/?query=%22KV+cache%22+OR+KVCache+OR+%22key-value+cache%22+OR+PagedAttention+OR+FlashAttention+OR+%22attention+algorithm%22+OR+%22long+context%22+OR+%22context+window%22&searchtype=all&abstracts=show&order=-announced_date_first&size=200&start=0](https://arxiv.org/search/?query=%22KV+cache%22+OR+KVCache+OR+%22key-value+cache%22+OR+PagedAttention+OR+FlashAttention+OR+%22attention+algorithm%22+OR+%22long+context%22+OR+%22context+window%22&searchtype=all&abstracts=show&order=-announced_date_first&size=200&start=0)；
- arXiv HTML 搜索批次 4：[https://arxiv.org/search/?query=quantization+OR+%22model+compression%22+OR+pruning+OR+distillation+OR+%22Mixture+of+Experts%22+OR+MoE+OR+%22sparse+model%22+OR+%22retrieval+augmented+generation%22+OR+RAG+OR+%22AI+agent%22&searchtype=all&abstracts=show&order=-announced_date_first&size=200&start=0](https://arxiv.org/search/?query=quantization+OR+%22model+compression%22+OR+pruning+OR+distillation+OR+%22Mixture+of+Experts%22+OR+MoE+OR+%22sparse+model%22+OR+%22retrieval+augmented+generation%22+OR+RAG+OR+%22AI+agent%22&searchtype=all&abstracts=show&order=-announced_date_first&size=200&start=0)；

## 1.8 后续问题

- 是否需要在采集脚本中加入指数退避与重试，以减少首次命中 429 后整批失效的问题；
- 是否需要新增 `list/<category>/recent` 页面的解析兜底，避免 API 与搜索页同时失效时整日报为空；
- 是否需要把“采集失败状态报告”和“真实零新增日报”分成两种不同模板，避免误读；
