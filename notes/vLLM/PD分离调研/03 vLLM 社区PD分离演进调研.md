---
tags:
  - vLLM
  - PD分离
  - 调研
updated: 2026-08-07
description: vLLM 社区视角的 PD 分离演进调研，覆盖 GitHub RFC/issue/PR 讨论、关键设计争议与社区组织形态。
---

# 03 vLLM 社区 PD分离演进调研

本文沉淀 vLLM 社区（GitHub issues / RFC / PR）围绕 PD 分离的讨论脉络。社区材料比代码更能回答「为什么这样设计」：路线之争、抽象取舍、责任边界都在这里显性化。

## 1 RFC 时间线总览

| 时间 | 编号 | 标题 | 状态 |
| --- | --- | --- | --- |
| 2024-06 | #5557 | Implement disaggregated prefilling via KV cache transfer | 最初 RFC |
| 2024-11 | #10727 | Implement disaggregated prefilling using Mooncake | 已落地（#10884） |
| 2024-12 | #10818 | Disaggregated prefilling and KV cache transfer roadmap | 路线图文档 |
| 2025-02 | #13020 | Async KV Cache Transfer for Disaggregated Inference | 指向 v1 API |
| 2025-06 | #19329 | Graceful Error Handling for KV Connector Load Failures | 已落地（#19330/#26171） |
| 2025-08 | #22605 | Separated CPU KV Cache Offloading/Transfer Process | 未采纳（被 offloading connector 取代） |
| 2025-08 | #22817 | Disaggregated Everything - Token In/Token Out API Server | 开放 |
| 2026-02 | #34407/#35213 | Disaggregated Frontend | 开放 |
| 2026-03 | #36923 | KV push from Prefill to Decode using Nixl | 已落地（NixlPushConnector） |
| 2026-05 | #42109 | Disaggregated Speculative Decoding | 开放 |
| 2026-06 | #45036 | Mooncake Store Connector Roadmap | 开放 |
| 2026-07 | #49765 | Native Pull-Based Queue Worker + Heuristic Pull Router | 开放 |

## 2 起源：RFC #5557（2024-06，KuntaiDu）

这是整个方向的奠基讨论，比第一个实现 PR（#10502，2024-12）早了半年。19 条评论完整保留了方向的成型过程。

**动机**：不止 PD 分离一个用例——还包括「固定长文档集合的 KV 持久化与按需加载」（GPU+CPU 内存装不下所有文档 KV 时存到外部存储）。这个双用例动机解释了为什么抽象里一直保留「KV 存储」而不仅是「KV 传输」。

**最初提案的抽象**：

```
vllm <--> communicator <--> KV database
```

- communicator 负责在 src/dst 间搬数据（src/dst 可以是 vLLM 的 KV block 或 database 条目）；
- KV database 以 prefix caching 的 hash 为 key、KV 张量为 value。

**RFC 中显式列出的开放问题**（后来逐一成为工程课题）：

1. 如何利用 NVLink 等高速互连加速传输；
2. 如何流水线化 KV 传输；
3. 传输期间如何防止 block 被 swap out（→ 后来的租约/延迟释放机制）；
4. 传输中是否压缩 KV、谁来压缩。

**讨论原文还原**（19 条评论中最关键的几段）：

1. **KuntaiDu 次日收窄焦点并给出工作流草案**：「先聚焦 disaggregated prefilling」；草案为——请求以 max_tokens=1 发 prefill 实例 → decode 实例上以 preempt 方式占位预留 KV → prefill 每算完一层就 layer-wise 传输 → 完成后解除 preempt，decode 侧用 automatic prefix caching 取回 KV。这是后来全部协议的雏形；
2. **cadedaniel（维护者）的路线质疑**：「担心先建 infra 而非从有影响力的特性倒推——没有窄用例的 infra 很难排定设计取舍；PD 分离对 KV 传输有极紧的性能约束，若最终实现用不上这些抽象将是巨大浪费」。这条意见直接导致初版实现比 RFC 更保守——先做最小可用的 pipe/buffer/connector 三层；
3. **richardliaw 的抽象提议**：先做引擎级 `save/insert_state` 状态存取 API 再设计传输；KuntaiDu 同意方向，并定下粒度决策——**读写按 vLLM block 对齐**，因为 KV 读写时机由 block manager 的分配/换出决策触发；
4. **传输介质之争**：社区提问「nccl 还是 rdma」，初版选了 NCCL（StatelessProcessGroup），RDMA 路线由后来 Mooncake/NIXL 连接器补足；
5. **外部参照**：有用户引 Llumnix（arXiv 2406.03243）的 KV 迁移实现提问，社区辨析——Llumnix 是 decode 步间迁移，PD 分离是相位间迁移，两者的计算-传输重叠条件不同；
6. **2024-06-30 基线方案**：4 进程（prefill/decode 实例 + proxy），请求先 padding 到 block_size 整数倍，max_tokens=1 发 prefill，KV block 流式搬运后再发 decode——与最终落地的 #10502 一致。

**社区反馈的取向**：最终落地实现比 RFC 更保守——先做最小可用的 pipe/buffer/connector 三层，而不是上来就做 database 抽象；KV database 的理想后来由 LMCache/MooncakeStore/FlexKV 等外部系统承接。

## 3 路线图：RFC #10818（2024-12）

初版落地后一周发布的路线图，是 v0 时代社区共识的最完整快照。按主题归纳（保留原始决策）：

### 3.1 xPyD 路线之争：P2P 直连 vs 中心化 store

路线图中两条候选被显式划掉/保留：

1. ~~Xp 与 Yd 之间建立多条直连~~——被放弃，注明「We now go for KVCache-store-based design」；
2. **Xp 连接到一个 KV cache server，Yd 再从 server 取**（#12957 MooncakeStore）——被选定。

这是社区第一次明确表态：**扩展性上中心化 KV store 优于点对点连接矩阵**。但直连路线并未死亡——NIXL 时代以「D 直接 READ 任意 P 显存」的形式复活（xPyD 下连接数由 D 主动管理，且 RDMA 单边操作不需要持久连接），说明争议的本质是「连接管理成本 vs 传输路径效率」，答案随传输技术演进而变化。对直连有偏好的用户被引导到 Slack `#feat-prefill-disaggregation` 频道讨论。

### 3.2 兼容性清单

与 chunked prefill、prefix caching、pipeline parallel（#12301）、多模态兼容——这四项在 v1 时代花了整整一年逐个补齐（PP 兼容直到 2026 年 #43720/#44528 才完整）。

### 3.3 异步与流水线

KV prefetching 与 layer-by-layer pipelining（#12523）被列为一等目标，直接导向 2025-02 的 RFC #13020（Async KV Cache Transfer），并最终由 v1 Connector API 的 `start_load_kv`/异步调度模型实现。

### 3.4 容错与弹性

1. 「batch 中只收到部分 KV 时，仅对缺失 token 做 prefill」（#12285）——部分命中思想，v1 时代由 prefix caching + 外部 token 数天然实现；
2. 「一个 worker 可被重新用作相反角色」（#12957 提及）——角色漂移思想，至今仍是开放方向。

### 3.5 编排层归属

路线图明确列出：中心化 orchestrator、动态增删 worker、基于可观测性 API 的 worker 观察、初始路由。这一节在 vLLM 内部长期没有落地（只有示例 proxy），最终由 Dynamo/LMCache router 等外部系统承担，vLLM 自身收敛为「暴露 KV events + connector 接口」的数据面角色。

### 3.6 第三方集成

已落地：Mooncake（#10884）、LMCache（#12953）；流产：InfiniteStore（#9079）、Valkey（#8724），均因开发者无响应——侧面说明 connector 生态高度依赖背后团队的持续投入。

## 4 v1 时代的社区讨论主题

### 4.1 异步传输 RFC（#13020，2025-02）与 v1 API 诞生

v0 同步传输把 KV 传输时间完整计入 TTFT。RFC #13020 由 AWS Neuron 推理团队提出，是第一个系统性异步方案：LookupBuffer 层加 `async_drop_select`（入队即返回）与后台 `drop_select_requester` 线程；调度器层新增 `transfer queue` 与 `TRANSFERRING` 状态，把 `_schedule_prefills` 拆为 `_schedule_wait`（分配显存并触发异步传输）与 `_schedule_transferring`（传输完成移入 running queue）。这个方案证明：**异步化必须让调度器理解「传输中」状态与「外部 token」概念，在 v0 结构内只能打补丁**。结论导向 #15960（KV Connector API V1，2025-04）：把 disagg 实现埋在 v1 的 prefix caching 与 chunked prefill 语义之下，调度器算哪些 token 需要外部 KV，worker 只管执行；orchestration 留在 vLLM 之外。此后所有新能力（失败恢复、offloading、push）都长在 v1 API 上，v0 组件于 2025-08 起分两步删除（#21785、#29705）。

### 4.2 可靠性讨论（#19329，2025-06）

生产用户集中反馈 KV load 失败导致请求整体失败的问题。RFC 讨论出的分级策略成为现行设计：`kv_load_failure_policy` 可配 `fail`（默认，立即失败）或 `recompute`（标记坏 block、调度器重算）；worker 通过 `get_block_ids_with_load_errors` 上报。

### 4.3 push 模式 RFC（#36923，2026-03）

pull 模式运行一年后，NVIDIA 团队（snadampal）提出 push 补充。RFC 的核心论证：pull 的时序严格串行（P 计算 → proxy 转发参数 → D 分配+握手+READ），**D 在 P 计算期间完全空闲**，大 prompt 高 TP 下代价显著。RFC 列六条优势：降 TTFT（D 在 P 计算期预注册 block，省去 proxy 参数往返）；天然适配 layer-wise 流水（P 知道每层何时算完可立即 WRITE）；fan-out 下 P 掌控网卡调度；proxy 退出传输关键路径（P/D 可同时派发，协调走 ZMQ 点对点）；P 侧 WRITE 完成即释放显存；长上下文 GB 级 KV 的传输准备被藏进计算时间。讨论确认 push 作为 pull 的补充而非替代，实现为 NixlPushConnector（#35264），与 pull 共享握手与元数据路径，独立 writer 线程，不污染引擎主循环。

### 4.4 前端/编排归属的持续讨论（2026）

1. #22817「Disaggregated Everything」：主张把 tokenization/rendering 也拆为独立服务（token-in/token-out API），减少 P/D 两侧重复的模板渲染与 tokenization；
2. #34407/#35213「Disaggregated Frontend」：把在线服务前端与引擎分离，作为 disaggregated everything 的基础；近期已实现 `return_token_ids` + decode 侧跳过 tokenization 作为过渡方案；
3. #49765（2026-07）：原生 pull-based queue worker 接口与启发式 pull router——社区开始把「路由」也纳入引擎原生能力的讨论，与早期「orchestrator 交给外部」的共识形成有趣的回摆。

### 4.5 安全事件

CVE-2025-62164（vLLM 0.10.2+，Completions API 内存损坏，CVSS 8.8）与 PD 分离部署面相关，社区提醒生产环境及时升级；PD 分离把内部协调协议（kv_transfer_params、ZMQ 侧信道）暴露到网络面，安全边界是持续课题。

## 5 社区组织形态

1. **核心维护者与团队分工**：UChicago KuntaiDu 团队发起并长期主导抽象层；NVIDIA 主导 NIXL 集成与 push 模式；Mooncake/月之暗面团队主导 Mooncake 系连接器；LMCache 团队（与 UChicago 深度关联）主导 LMCache 集成；AMD 团队主导 MoRIIO；
2. **协作渠道**：GitHub RFC/PR 为主决策场所，Slack `#feat-prefill-disaggregation`（后改名相关频道）处理路线争议与快速对齐；
3. **接口稳定性承诺**：官方文档长期标注 disagg 为 experimental，KVConnectorBase_V1 初始化时会打印「API 尚在演进」警告；社区通过 connector registry + 动态加载（`kv_connector_module_path`，#18142）允许外部连接器不随主仓节奏破坏；
4. **测试文化**：nixl_integration accuracy 测试脚本、FakeNixlWrapper 单测、disagg 专用 CI 组（如 `NixlConnector PD accuracy tests`），ROCm/PP/混合模型兼容性问题大量通过 CI issue 暴露修复。

## 6 社区视角下的关键共识与教训

1. **抽象先行、实现保守**：RFC #5557 的 database 理想最终分解为多个专业系统，vLLM 本体只保留 connector 插槽；
2. **调度器一等公民是异步化的前提**：不进入调度器的传输无法被重叠、无法与 prefix cache 协同；
3. **生命周期管理最难**：block 何时释放（租约/心跳/watchdog）相关 bugfix 占 NIXL 提交的大头；
4. **兼容性是长尾**：PP、异构 TP、MLA/Mamba 混合模型、多模态，每一项都花了数月；
5. **编排层外置是当前稳态**：vLLM 专注数据面与控制面钩子，路由与容量规划交给 Dynamo 等框架，但原生化的讨论在 2026 年重新活跃。
