---
tags:
  - vllm
  - llm-inference
  - tutorial-plan
  - architecture
updated: 2026-05-31
description: 本文规划 vLLM 架构与原理精讲系列在 01-06 之后的后续教程路线，围绕 GPU 执行、缓存复用、高级推理特性、分布式扩展、服务化与生产运维建立循序渐进的大纲。
---

# 00 vLLM 架构与原理精讲系列教程规划

这个系列已经完成了从推理系统基本问题到 vLLM V1 Scheduler 与 async scheduling 的核心铺垫。后续不应该继续横向罗列功能，而应该沿着一条更稳定的学习主线推进：

```text
请求进入系统
  -> EngineCore 与 Scheduler 决定本轮执行
  -> ModelRunner 把 SchedulerOutput 变成 GPU 可执行 batch
  -> Attention / Sampler / OutputProcessor 产生并回传 token
  -> Prefix cache / Spec decode / Structured output 等能力改变执行路径
  -> Executor / Worker / 并行策略把单机机制扩展到多进程和多节点
  -> OpenAI API / Metrics / Benchmark / Deployment 把引擎变成可运营服务
```

本规划基于本地 `code/opensource/vllm` 源码快照，源码分支为 `main`，短提交哈希为 `52a31ccec`。规划目标不是给出一次性功能清单，而是为后续每一章提供清楚的位置、边界、源码入口和教学重点。

## 1. 当前系列位置

| 已有章节 | 已覆盖的核心问题 | 后续依赖关系 |
| --- | --- | --- |
| 01 推理系统导论 | Prefill、Decode、KV Cache、TTFT、TPOT、吞吐、排队与推理系统职责 | 建立全系列性能语言和系统问题意识 |
| 02 初识 vLLM 推理引擎 | vLLM 定位、HF Transformers/TGI/vLLM 演进、PagedAttention 的动机 | 建立 vLLM 为什么把 KV Cache 管理放到中心 |
| 03 vLLM 引擎架构总览 | V1 架构地图、请求执行流、rank 坐标、TP/PP/DP/EP 部署视图 | 后续章节要把总览中的组件逐个落到源码和状态流 |
| 04 深入 KVCacheManager | block 化 KV Cache、KVCacheManager 分层、请求生命周期、Scheduler/Worker/PagedAttention 协作 | 后续可继续展开 prefix cache、hybrid KV、KV offload 与 disaggregated serving |
| 05 Scheduler 核心架构 | running/waiting 请求、token budget、chunked prefill、prefix cache、spec decode、preemption | 后续需要解释 SchedulerOutput 如何进入 ModelRunner 和 GPU 执行路径 |
| 06 Async Scheduling | CPU/GPU overlap、placeholder、一拍提前、batch queue、structured output、KV connector、MRV2 边界 | 后续最自然的切入点是 ModelRunner、persistent batch 与 GPU 输入准备 |

当前系列已经把“为什么要调度”和“调度器如何决定下一步”讲清楚了，但还没有系统解释“被调度出来的 batch 到底如何在 GPU 上执行”。因此第 07 章不宜直接跳到部署、LoRA 或 quantization，而应进入 ModelRunner。

## 2. 后续规划原则

1. 先闭合单次生成主链路，再展开高级能力；

   Scheduler 之后必须先讲 ModelRunner、Attention、Sampling 和 OutputProcessor。否则读者知道“系统决定跑什么”，却不知道“这些决定如何变成张量、kernel、token 和流式响应”。

2. 先讲改变执行语义的机制，再讲功能型扩展；

   Prefix caching、spec decode、structured output、multi-modal、LoRA 都不是孤立功能。它们会改变输入准备、KV 状态、采样约束、输出处理或进程边界，应放在主链路之后讲。

3. 先单机再分布式，先机制再部署；

   Executor、TP、PP、DP、DCP、EP、disaggregated prefill/decode 的解释依赖读者理解 EngineCore、Scheduler、ModelRunner 和 KV 状态。部署命令应该服务机制解释，而不是替代机制解释。

4. 源码入口优先选择稳定骨架；

   每章应从少量骨架文件进入，再扩展到辅助模块。避免一章同时追太多功能开关，导致文章变成源码索引。

5. 每章都要有可复用心智模型；

   后续章节不应只回答“这个类做什么”，还要回答“它为什么这样切边界、解决什么瓶颈、带来什么代价、调参或排障时该看什么”。

## 3. 推荐下一章

第 07 章建议写：

```text
07 从 SchedulerOutput 到 GPUModelRunner：一次 step 如何变成可执行 batch
```

它是 05 和 06 的自然下游，也是后续 Attention、Sampler、CUDA Graph、Spec Decode、Multi-Modal、LoRA 的共同基础。

### 3.1 第 07 章中心问题

第 07 章应回答：Scheduler 已经决定了本轮要推进哪些请求、每个请求推进多少 token、需要哪些 KV block 之后，vLLM 如何把这个逻辑计划变成 GPU 侧可执行的输入、模型前向和输出对象。

### 3.2 第 07 章建议边界

应覆盖：

- `EngineCore.step()` 如何产生 `SchedulerOutput` 并调用 executor 执行；
- `SchedulerOutput` 中的请求增删、token 数、KV block、grammar output、spec decode 信息如何进入 ModelRunner；
- `GPUModelRunner` 如何维护 batch state、block table、sampling metadata 和模型输入；
- `execute_model()` 的高层阶段：更新请求、准备输入、运行模型、采样、返回 `ModelRunnerOutput`；
- async scheduling 下 ModelRunner 为什么要避免 CPU/GPU 同步点；
- V1 ModelRunner 与 MRV2 的设计动机关系；

暂缓：

- Attention backend kernel 细节，放到第 08 章；
- Sampler 与 logits processor 细节，放到第 09 章；
- CUDA Graph 捕获和 torch.compile，放到性能阶段；
- LoRA、multi-modal、structured output 等能力只作为集成点点到为止；

### 3.3 第 07 章源码入口

| 主题 | 推荐入口 |
| --- | --- |
| EngineCore 主循环 | `vllm/v1/engine/core.py` |
| Scheduler 输出结构 | `vllm/v1/core/sched/output.py` |
| Scheduler 主体 | `vllm/v1/core/sched/scheduler.py` |
| GPU ModelRunner 主体 | `vllm/v1/worker/gpu/model_runner.py`、`vllm/v1/worker/gpu_model_runner.py` |
| GPU input batch 与 block table | `vllm/v1/worker/gpu/input_batch.py`、`vllm/v1/worker/gpu/block_table.py` |
| async output 边界 | `vllm/v1/worker/gpu/async_utils.py` |
| MRV2 设计文档 | `docs/design/model_runner_v2.md` |

### 3.4 第 07 章图表规划

| 图 | 教学目的 |
| --- | --- |
| SchedulerOutput 到 ModelRunner 的对象流 | 帮读者看到逻辑调度计划如何跨过 EngineCore/Executor/Worker 边界 |
| 一次 `execute_model()` 的阶段图 | 把 update requests、input prepare、forward、sample、output 串成单步生命周期 |
| Persistent batch before/after | 解释为什么连续 step 不应每次从零构建大张量 |
| Async scheduling 下的 CPU/GPU 数据危险区 | 展示为什么 pinned copy、staged write 和 no-sync 边界重要 |

## 4. 主线章节规划

### 4.1 第一阶段：闭合单机执行链路

这一阶段回答一个完整问题：一个请求已经进入 vLLM，调度器也已经做出决策，GPU 到底如何完成一次 token 推进，并把结果交还给前端。

| 序号 | 建议标题 | 中心问题 | 关键内容 | 源码与资料入口 |
| --- | --- | --- | --- | --- |
| 07 | 从 SchedulerOutput 到 GPUModelRunner：一次 step 如何变成可执行 batch | 调度计划怎样落到 GPU 执行 | EngineCore step、executor 调用、GPUModelRunner、request update、input batch、block table、ModelRunnerOutput | `vllm/v1/engine/core.py`、`vllm/v1/worker/gpu/model_runner.py`、`docs/design/model_runner_v2.md` |
| 08 | PagedAttention 的执行层：Attention Backend 如何读取 block table | KV block 地址怎样进入 attention kernel | backend selection、slot mapping、block table、prefill/decode attention、MLA backend、PagedAttention kernel 与不同硬件后端 | `vllm/v1/attention/`、`vllm/v1/worker/gpu/attn_utils.py`、`docs/design/attention_backends.md`、`docs/design/paged_attention.md` |
| 09 | 从 logits 到 token：Sampler 与 Logits Processor 的生成控制 | 模型输出 logits 后如何变成下一个 token | temperature、top-k/top-p、penalties、logprobs、prompt logprobs、custom logits processor、Triton sampler、rejection sampler 的边界 | `vllm/v1/sample/`、`vllm/v1/worker/gpu/sample/`、`docs/design/logits_processors.md` |
| 10 | OutputProcessor 与流式返回：token 如何回到用户 | GPU 输出怎样穿过 engine/frontend 边界 | detokenizer、output processor、streaming、finished reason、abort、logprobs、OpenAI response object 的映射 | `vllm/v1/engine/output_processor.py`、`vllm/v1/engine/detokenizer.py`、`vllm/v1/engine/async_llm.py` |
| 11 | 请求入口与输入处理：Prompt 如何进入 EngineCore | API 输入怎样变成内部 Request | tokenizer、chat template、multi-part input、InputProcessor、SamplingParams、Request、EngineCoreRequest、offline/online 差异 | `vllm/v1/engine/input_processor.py`、`vllm/v1/request.py`、`vllm/entrypoints/llm.py`、`vllm/entrypoints/openai/` |

阶段结束后，读者应能从一次请求的输入、调度、执行、采样、输出完整串起 vLLM V1 主路径。

### 4.2 第二阶段：缓存复用与长上下文能力

这一阶段回答：当请求不是一次性短 prompt，而是存在共享前缀、长上下文、多类 KV 布局或远端 KV 状态时，vLLM 如何保持吞吐和显存效率。

| 序号 | 建议标题 | 中心问题 | 关键内容 | 源码与资料入口 |
| --- | --- | --- | --- | --- |
| 12 | Automatic Prefix Caching：重复前缀如何变成可复用 KV | prefix hit 如何减少 prefill 成本 | block hash、cache hit、LRU eviction、cache salting、prefix cache metrics、与 Scheduler/KVCacheCoordinator 的关系 | `vllm/v1/core/kv_cache_coordinator.py`、`vllm/v1/core/kv_cache_utils.py`、`docs/design/prefix_caching.md`、`docs/features/automatic_prefix_caching.md` |
| 13 | Chunked Prefill 与长上下文调度：prefill 为什么要切块 | 长 prompt 如何避免压垮 decode | token budget、chunked prefill、mixed prefill/decode、TTFT/TPOT 权衡、长上下文工作负载 | `vllm/v1/core/sched/scheduler.py`、`docs/benchmarking/cli.md`、`docs/features/context_extension.md` |
| 14 | Hybrid KV Cache：不同 attention 类型如何共享缓存管理框架 | full/sliding/cross/mamba KV 如何统一调度 | SingleTypeKVCacheManager、HybridKVCacheCoordinator、sliding window、cross attention、Mamba、encoder cache | `vllm/v1/core/single_type_kv_cache_manager.py`、`vllm/v1/core/kv_cache_coordinator.py`、`docs/design/hybrid_kv_cache_manager.md` |
| 15 | KV Offload 与 KV Connector：KV 状态如何离开本机 GPU | 显存不够或分离式服务时 KV 如何搬迁 | CPU offload、simple/native KV offload、connector metadata、NIXL/P2P/Mooncake/MoRIIO 语义、lease 与一致性 | `vllm/v1/kv_offload/`、`vllm/distributed/kv_transfer/`、`vllm/v1/worker/gpu/kv_connector.py`、`docs/design/nixl_kv_cache_lease.md` |

阶段结束后，读者应理解 vLLM 的 KV 管理不止是“分配 block”，还包含复用、驱逐、异构布局、远端传输和一致性边界。

### 4.3 第三阶段：高级推理特性

这一阶段讲会改变生成语义或执行路径的高级能力。它们应在主链路之后讲，因为每个能力都要挂到 Scheduler、ModelRunner、Sampler 或 OutputProcessor 上。

| 序号 | 建议标题 | 中心问题 | 关键内容 | 源码与资料入口 |
| --- | --- | --- | --- | --- |
| 16 | Speculative Decoding：用草稿 token 换取 decode 加速 | draft、verify、accept/reject 如何协作 | draft model、EAGLE、Medusa、n-gram proposer、rejection sampler、metrics、与 scheduler token budget 的关系 | `vllm/v1/spec_decode/`、`vllm/v1/worker/gpu/spec_decode/`、`docs/features/speculative_decoding/` |
| 17 | Structured Outputs 与 Tool Calling：约束生成如何嵌入采样路径 | grammar 约束如何影响 token 选择 | xgrammar/outlines/guidance/lm-format-enforcer、tool parser、schema、reasoning outputs、async scheduling 下的延迟采样边界 | `vllm/v1/structured_output/`、`vllm/v1/worker/gpu/structured_outputs.py`、`docs/features/structured_outputs.md`、`docs/features/tool_calling.md` |
| 18 | LoRA Adapters：动态适配器如何进入模型执行 | 一个 base model 如何服务多个 adapter | LoRAConfig、LoRARequest、LoRAMapping、dynamic loading、resolver plugin、metrics、multi-modal LoRA 边界 | `vllm/lora/`、`vllm/v1/worker/lora_model_runner_mixin.py`、`docs/features/lora.md`、`docs/design/lora_resolver_plugins.md` |
| 19 | Multi-Modal 推理：图像、音频与文本如何共用引擎 | 非文本输入如何进入同一套调度与执行框架 | MultiModalConfig、processor output cache、encoder runner、encoder cache、placeholder tokens、multi-modal CUDA Graph | `vllm/multimodal/`、`vllm/v1/worker/gpu/mm/`、`docs/design/mm_processing.md`、`docs/features/multimodal_inputs.md` |
| 20 | Pooling、Embedding 与 Rerank：非生成任务如何复用 vLLM | vLLM 如何支持不逐 token 生成的任务 | pooling runner、embedding、scoring、classification、reward model、generative 与 pooling 输出差异 | `vllm/v1/pool/`、`vllm/v1/worker/gpu/pool/`、`docs/models/pooling_models/`、`vllm/entrypoints/pooling/` |

阶段结束后，读者应能判断一个新特性到底修改了输入、调度、KV、模型执行、采样还是输出，而不是把所有能力都看成 API 参数。

### 4.4 第四阶段：性能优化与内核工程

这一阶段讲 vLLM 如何减少 CPU overhead、GPU launch overhead、显存占用和通信空泡。它应该在读者理解主路径之后展开，否则容易变成工具名堆叠。

| 序号 | 建议标题 | 中心问题 | 关键内容 | 源码与资料入口 |
| --- | --- | --- | --- | --- |
| 21 | CUDA Graph 与 torch.compile：vLLM 如何减少 launch overhead | 静态图和动态请求如何共存 | CUDAGraphMode、BatchDescriptor、piecewise compile、full CUDA graph、warmup、capture size、compatibility | `vllm/compilation/`、`vllm/v1/cudagraph_dispatcher.py`、`vllm/v1/worker/gpu/cudagraph_utils.py`、`docs/design/cuda_graphs.md`、`docs/design/torch_compile.md` |
| 22 | Quantization 与 KV Cache Quantization：用精度换容量和吞吐 | 权重、激活与 KV 量化分别影响什么 | FP8、INT4/INT8、AWQ/GPTQ/BnB、online quantization、quantized KV cache、准确率/吞吐/显存权衡 | `vllm/model_executor/layers/quantization/`、`docs/features/quantization/` |
| 23 | Model Runner V2：vLLM 为什么重写 GPU 执行路径 | MRV2 解决 V1 哪些结构性问题 | persistent batch decoupling、async-first、StagedWriteTensor、GPU-native input prep、Triton sampler、explicit CUDA graph | `docs/design/model_runner_v2.md`、`vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/worker/gpu/` |
| 24 | Benchmark 与容量规划：如何读懂 vLLM 的性能结果 | 性能指标如何反推瓶颈 | latency/throughput benchmark、prefix workload、spec decode benchmark、multi-modal benchmark、TTFT/TPOT/throughput trade-off | `benchmarks/`、`vllm/benchmarks/`、`docs/benchmarking/` |

阶段结束后，读者应能把性能问题定位到 input preparation、attention kernel、sampling、CUDA Graph、KV cache、并行通信或服务排队，而不是只看吞吐数字。

### 4.5 第五阶段：分布式与并行扩展

这一阶段把前面单机机制扩展到多进程、多 GPU、多节点。03 已经讲过总览，后续应进入更精确的进程拓扑、rank 坐标、通信边界和失败模式。

| 序号 | 建议标题 | 中心问题 | 关键内容 | 源码与资料入口 |
| --- | --- | --- | --- | --- |
| 25 | Executor 与 Worker：vLLM 如何启动和管理执行进程 | EngineCore 如何驱动单进程、多进程和 Ray worker | UniProcExecutor、MultiprocExecutor、RayExecutor、WorkerProc、RPC、startup handshake、failure callback | `vllm/v1/executor/`、`vllm/v1/worker/worker_base.py`、`vllm/v1/worker/gpu_worker.py`、`docs/design/multiprocessing.md` |
| 26 | TP、PP、DP 与 DCP：rank 坐标如何决定计算和通信 | 不同 parallelism 拆的是模型、请求还是上下文 | ParallelConfig、parallel_state、TP group、PP group、DP coordinator、DCP group、external/hybrid LB | `vllm/config/parallel.py`、`vllm/distributed/parallel_state.py`、`vllm/v1/engine/coordinator.py`、`docs/serving/parallelism_scaling.md` |
| 27 | Expert Parallel、EPLB 与 DBO：MoE 服务为什么需要特殊调度 | MoE expert 如何分布、均衡与 overlap | EP、EPLB、expert distribution、DBO microbatch overlap、MoE kernel、communication/compute overlap | `vllm/config/parallel.py`、`vllm/distributed/eplb/`、`vllm/v1/worker/ubatching.py`、`docs/serving/expert_parallel_deployment.md`、`docs/design/dbo.md` |
| 28 | Disaggregated Prefill/Decode：把 prefill 与 decode 拆开部署 | 为什么分离 prefill 和 decode，KV 如何跨实例流动 | prefill/decode split、KV transfer、proxy/orchestration、P/D 角色、KV connector、benchmark 与适用负载 | `vllm/entrypoints/serve/disagg/`、`docs/features/disagg_prefill.md`、`docs/serving/expert_parallel_deployment.md` |

阶段结束后，读者应能画出 vLLM 在线服务的进程图，解释每种并行策略的资源切分对象、通信代价和部署边界。

### 4.6 第六阶段：服务化、可观测与生产运维

这一阶段把引擎机制连接到真实服务。它不应太早出现，因为 API、metrics 和部署只有挂在前面机制上才有解释力。

| 序号 | 建议标题 | 中心问题 | 关键内容 | 源码与资料入口 |
| --- | --- | --- | --- | --- |
| 29 | OpenAI-Compatible Server：vLLM 如何把引擎暴露成在线 API | `vllm serve` 如何连接 HTTP、AsyncLLM 与 EngineCore | CLI、api_server、chat/completion/responses、streaming、batch serving、health/profile/cache endpoints | `vllm/entrypoints/cli/serve.py`、`vllm/entrypoints/openai/api_server.py`、`vllm/entrypoints/openai/`、`docs/serving/online_serving/` |
| 30 | Metrics 与 Tracing：如何观察 vLLM 内部状态 | 指标如何从 engine/scheduler/frontend 汇总到 Prometheus | SchedulerStats、IterationStats、StatLoggerManager、Prometheus multiprocess、OpenTelemetry、KV cache metrics、prefix cache metrics | `vllm/v1/metrics/`、`vllm/entrypoints/serve/instrumentator/metrics.py`、`docs/design/metrics.md`、`docs/usage/metrics.md` |
| 31 | 部署与排障：从本机实验到生产服务 | 如何把机制理解转化为部署判断 | Docker、Kubernetes、Ray、Nginx、external LB、startup/readiness、distributed troubleshooting、network/NCCL/GPU memory | `docs/deployment/`、`docs/serving/distributed_troubleshooting.md`、`docs/serving/data_parallel_deployment.md` |

阶段结束后，读者应能从源码机制走向服务运营：知道 API 行为来自哪里、指标如何解释、部署问题可能落在哪个系统层。

## 5. 支线专题规划

主线之外，可以保留若干支线专题。支线适合在主线完成对应依赖之后穿插，不建议过早插入。

| 专题 | 适合插入位置 | 教学价值 | 注意边界 |
| --- | --- | --- | --- |
| Reasoning Outputs 与 tool parser 生态 | 第 17 章之后 | 解释推理模型输出结构、parser 与 OpenAI API 的关系 | 不要变成模型提示词教程 |
| Plugin system 与自定义扩展 | 第 18 或 29 章之后 | 解释 vLLM 如何加载外部能力 | 只讲与推理执行或服务入口相关的插件 |
| Realtime / Speech-to-Text | 第 19 或 29 章之后 | 展示 vLLM 服务形态从文本生成扩展到音频/实时场景 | 不抢 multi-modal 主线 |
| Model support 与自定义模型接入 | 第 19 或 21 章之后 | 帮助理解 `model_executor` 和模型接口 | 不做模型 zoo 罗列 |
| Security、multi-tenant 与成本控制 | 第 30 或 31 章之后 | 连接生产使用中的隔离、限制与可观测 | 以 vLLM 机制为中心，不扩展成云平台教程 |

## 6. 每章固定写作单元

后续每章建议保持相对稳定的交付结构，但不要让所有文章变成同一模板。

1. 开头用具体系统问题引入；

   例如第 07 章可以从“Scheduler 已经选出 batch，但 GPU 不能直接执行 Python 对象”切入。

2. 给出本章心智模型；

   例如 ModelRunner 是“把调度计划翻译成 GPU batch 的执行编译器”，Attention Backend 是“把逻辑 token 位置翻译成 KV block 访问的计算后端”。

3. 用一条主路径串联机制；

   每章都要选择一条主路径，不要一开始就展开所有分支。

4. 在关键位置补充源码阅读地图；

   源码地图应解释为什么从这些文件读，而不是只列路径。

5. 明确工程边界和误区；

   每章至少回答“什么时候它有用、什么时候不是瓶颈、常见误解是什么、指标或日志怎么看”。

6. 规划 3-5 张教学图；

   图应承担机制解释任务，如对象流、生命周期、memory layout、timeline、rank topology 或 before/after 对比。

7. 保留参考资料与学习测评；

   正式教程仍应在参考资料之后放置 `学习测评`，题目要覆盖概念、实现、边界和迁移判断。

## 7. 暂不建议优先写的主题

以下主题有价值，但不适合作为 06 之后立刻展开的主线。

1. 直接写生产部署大全；

   当前读者还缺 ModelRunner、Sampler、Executor 和 metrics 的细节，部署章节会缺少机制支撑。

2. 直接写 LoRA 或 Multi-Modal；

   这些能力依赖 ModelRunner 输入组织、batch state、输出路径和服务入口，过早写容易变成 API 参数说明。

3. 逐个量化后端罗列；

   Quantization 应先讲共同的精度、显存、kernel 与准确率权衡，再决定是否展开具体后端。

4. 逐个 attention backend 罗列；

   Attention chapter 应先讲 backend selection、block table、prefill/decode 语义，再把 FlashAttention、FlashInfer、Triton、MLA 等作为对比。

5. 过早深入插件生态；

   Plugin system 适合作为扩展机制专题，不适合作为架构主线。

## 8. 推荐推进顺序

若只考虑学习依赖，推荐未来优先写作顺序如下：

1. 07 从 SchedulerOutput 到 GPUModelRunner；
2. 08 PagedAttention 的执行层；
3. 09 从 logits 到 token；
4. 10 OutputProcessor 与流式返回；
5. 12 Automatic Prefix Caching；
6. 16 Speculative Decoding；
7. 17 Structured Outputs 与 Tool Calling；
8. 21 CUDA Graph 与 torch.compile；
9. 25 Executor 与 Worker；
10. 26 TP、PP、DP 与 DCP；
11. 29 OpenAI-Compatible Server；
12. 30 Metrics 与 Tracing；

这个顺序先完成“单机主路径”，再补“会改变主路径的高级机制”，最后进入“分布式、服务化与可观测”。如果后续目标是尽快服务生产部署，可以把 29、30、31 提前到 26 之后；如果目标是深入 vLLM 性能机制，则应优先完成 21、22、23、24。

## 9. 总结

当前系列已经完成了 vLLM 学习路径中最重要的前半段：读者知道推理系统为什么复杂，知道 vLLM 为什么围绕 KV Cache 和 continuous batching 设计，也理解了 Scheduler 与 async scheduling 的核心。

后续最重要的转折点是把视角从 Scheduler 移到 GPU 执行层。第 07-11 章应闭合请求主链路，第 12-15 章展开缓存与长上下文，第 16-20 章讲高级推理能力，第 21-24 章进入性能优化，第 25-28 章解释分布式扩展，第 29-31 章收束到服务化、可观测和生产运维。

如果这个顺序稳定执行，整套系列会从“推理系统入门”成长为一条完整的 vLLM 架构学习路线：既能读懂源码，也能解释生产服务中的性能、稳定性和扩展性问题。
