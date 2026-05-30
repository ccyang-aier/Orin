---
tags:
  - LLM
  - GPU
  - CUDA
  - NVIDIA
  - model-serving
updated: 2026-05-30
description: 面向大模型与推理全栈初学者，建立 GPU 设计哲学、存储层级、SM/Warp/Tensor Core 与 CUDA 执行模型的基础心智模型，为后续理解 LLM 性能与推理系统打底。
---

# 大模型与推理全栈精讲系列 01：理解 GPU 架构及其原理

> [!Quote] 本篇导读
> 如果你第一次接触大模型推理，很容易先看到模型名字、参数量和显卡型号。可真正让模型跑起来时，机器内部做的事情其实很朴素：大量数字被读进来、算一遍、再写回去；这个过程会重复很多次。
>
> GPU 之所以重要，不是因为它“更聪明”，而是因为它很擅长把许多形状相似的小工作同时铺开。可以先把 GPU 想成一个很大的计算工坊：里面有很多计算小组，适合处理成片、成批、规则排列的数字任务。
>
> 先别急着记一堆硬件名词。我们从一个简单问题开始：CPU 和 GPU 为什么不是同一种机器？等这个问题变清楚，再把一个小计算例子放进来，后面的数据搬运、计算单元和性能瓶颈就不再像凭空冒出来。

为了让后面的概念不飘在空中，本文会在第 1 节后半段正式引入一个大模型里常见的计算例子。它会从那时起承担贯穿样例的作用；在此之前，先把 CPU 和 GPU 的差异看清楚。

## 1. GPU的设计哲学

### 1.1 CPU 与 GPU 解决的是不同问题

CPU和GPU都能执行程序，但它们从一开始就不是为同一种工作负载设计的。

CPU追求的是低延迟和通用控制能力。它通常拥有少量复杂核心、较强的分支预测、乱序执行能力、多级缓存和复杂控制逻辑，适合操作系统调度、文件网络I/O、复杂业务逻辑、串行依赖强的程序，以及对单个任务响应时间敏感的场景。

GPU追求的是高吞吐。它不指望某一个线程像CPU核心那样强，而是准备了大量较轻量的执行资源，让成千上万个线程同时做形状相似的工作。

总的来说，可以把两者的差异压缩成一句话：**CPU擅长把少量复杂任务尽快做完，GPU擅长把海量相似任务同时铺开。**

如下图说明了CPU与GPU的架构差异。

![CPU 与 GPU 的设计哲学对比|900](imgs/gpu-cpu-design-philosophy-handdrawn-cn-v4.png)

这就是为什么大模型会大量依赖 GPU。模型运行时会反复处理成片排列的数字，许多工作不是“一个复杂线程做很多判断”，而是“同一种计算在大量位置上重复”。矩阵乘、向量加法、归一化、激活函数等都可以被拆成许多形状相似的小任务。GPU 的核心优势，首先是在这类规则数值计算上被释放出来的。

### 1.2 用矩阵乘看并行性从哪里来

上一小节说 GPU 喜欢“海量相似任务”，但这句话还比较抽象。下面引入一个具体例子：大模型里常见的 `Linear` 层。你可以先把它理解为“把一批输入数字乘上一块权重，得到一批新的数字”。

一个 `Linear` 层常常可以写成矩阵乘。设它满足：

$$
X \in \mathbb{R}^{B \times seq \times d_{model}}, \quad
W \in \mathbb{R}^{d_{model} \times d_{out}}, \quad
Y \in \mathbb{R}^{B \times seq \times d_{out}}
$$

通常可以先把 $X$ 展平成二维矩阵：

$$
X' \in \mathbb{R}^{(B \cdot seq) \times d_{model}}
$$

于是计算变成：

$$
Y = X'W
$$

如果第一次看到这些符号，可以先抓住直觉：$B$ 表示一次处理多少组输入，$seq$ 表示一次处理多少个位置，$d_{model}$ 和 $d_{out}$ 表示每个位置上的向量宽度。把 $X$ 展平以后，输出矩阵 $Y$ 的每一个元素，本质上都是一个长度为 $d_{model}$ 的点积。也就是说，输出里有：

$$
B \cdot seq \cdot d_{out}
$$

个可以并行组织的输出位置。

如果 $B=1$，$seq=4096$，$d_{out}=4096$，那么输出元素数量约为 1677 万。GPU 喜欢的正是这种问题：不是让一个线程把 1677 万个点积从头算到尾，而是把输出矩阵切成许多 tile，让大量计算单元分头处理。

![贯穿样例：Y=XW 如何被拆成输出 tile|900](imgs/gpu-yxw-tile-running-example-handdrawn-cn-v2.png)

初学 GPU 时最容易误解的一点是：GPU 的“快”不是简单来自频率更高，也不是来自某个线程更聪明，而是来自任务规模足够大、形状足够规则、数据访问足够友好时形成的吞吐优势。$Y=XW$ 之所以适合作为贯穿例子，是因为它同时牵出几个关键问题：工作怎么拆开，数据怎么搬运，计算单元怎么协作，最后又为什么可能遇到性能瓶颈。

到这里，$Y=XW$ 已经有了第一层含义：它不是一条公式，而是一大片可以被切分、分配和并行执行的输出空间。下一步要问的是，这些并行工作需要的数据从哪里来。

## 2. 存储层级

上一节只解决了“工作能不能拆开”。但拆开以后还有一个更实际的问题：每个小任务开始计算之前，都要先拿到自己需要的数据。计算单元再多，如果数据迟迟送不到，它们也只能等。

### 2.1 数据不是自动出现在计算单元旁边

GPU 的存储不是一个平坦的大空间，而是一个分层系统。越靠近计算单元，容量通常越小、延迟越低、访问越快；越远离计算单元，容量越大、延迟越高。对大模型来说，权重、激活、KV cache 和中间 buffer 大多先放在设备显存里；真正计算时，kernel 会尽量把马上要用、还会反复用的数据搬到更靠近计算单元的片上存储中。

| 层级 | 直觉 | 在 LLM 中的角色 |
| --- | --- | --- |
| HBM / Global Memory | 容量大、带宽高，但离计算单元远 | 存放权重、激活、KV cache、临时 buffer； |
| L2 Cache | 全 GPU 共享缓存 | 多个 SM 访问共享数据时的重要缓冲层； |
| Shared Memory / L1 | 靠近 SM，Block 内可协作复用 | 高性能矩阵乘会把 tile 搬进来反复使用； |
| Registers | 每个线程私有，最快但总量有限 | 存放局部变量、累加器和小片段中间值； |

这些存储层级和计算单元不是同一类东西。HBM、L2、Shared Memory/L1、Registers 负责保存或缓存数据；Tensor Core 和 CUDA Core 负责消费数据并执行计算。把二者分清楚，才能理解“算得快”为什么常常先取决于“喂得上”。

把这个层级放回 $Y=XW$，问题会更具体。$Y$ 里的一个元素需要读取 $X$ 的一段行数据和 $W$ 的一段列数据，然后做点积。相邻的输出元素并不是完全陌生的：同一行的多个输出会反复用到同一段 $X$，同一列方向的多个输出也会反复用到相邻的 $W$ 数据。

如果每个输出元素都独自从 HBM 里把自己需要的数据重新读一遍，很多带宽会浪费在“搬同样或相邻的数据”上。GPU 矩阵乘真正想做的是：先把一小块 $X$ 和一小块 $W$ 搬到更近的位置，让一批输出 tile 反复使用它们；中间累加结果尽量留在寄存器里，等一个 tile 算完后再写回 HBM。

![GPU 存储层级与 Y=XW 数据流动|900](imgs/gpu-memory-hierarchy-yxw-handdrawn-cn-v2.png)

因此，一个高性能矩阵乘 kernel 的核心努力，可以粗略理解为：

- 少从 HBM 重复读取已经能被复用的数据；
- 把 $X$ 和 $W$ 的小块 tile 搬到 Shared Memory / L1 或寄存器附近；
- 让同一块数据被尽可能多的乘加操作复用；
- 把中间累加结果留在寄存器里，最后再写回 HBM；

这也是为什么“存储层级”应该先于“SM 架构”理解。GPU 性能常常不是因为算术单元不够强，而是因为数据没有以合适的节奏、形状和复用方式送到算术单元。

### 2.2 带宽与算力的量级差异

上面的例子只是在讲直觉：重复搬数据会浪费时间。接下来用硬件峰值数字建立一点量级感。

以 NVIDIA A100 80GB SXM 产品实现为例，官方规格给出的 dense FP16 Tensor Core 峰值吞吐为 312 TFLOPS；如果使用结构化稀疏路径，标称峰值可到 624 TFLOPS。它的 HBM2e 带宽约为 2,039 GB/s；A100 80GB PCIe 版本的带宽则约为 1,935 GB/s。本文只用这些数字建立量级感，避免把不同产品形态、dense/sparse 路径和实际 kernel 性能混成一个数字。

注意，这里的数字是硬件峰值，不等于任意 PyTorch 代码都能达到。它们的作用是告诉我们：如果一个算子每从显存读入很少数据就能做大量计算，它更有机会接近计算上限；如果一个算子读写了大量数据却只做很少计算，它就更容易被 HBM 带宽限制。

这引出一个重要概念：Arithmetic Intensity，常译为计算强度。

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes moved}}
$$

它描述的是“每搬运 1 byte 数据，能做多少次浮点计算”。计算强度越高，越可能是 compute-bound；计算强度越低，越可能是 memory-bound。实际判断还要看硬件、精度、kernel、缓存复用和访问模式，但这个概念足以帮助初学者建立第一层判断。

### 2.3 同一个例子里，权重到底有多重

假设 $d_{model}=4096$，$d_{out}=4096$，权重矩阵 $W$ 的元素数量是：

$$
4096 \times 4096 = 16{,}777{,}216
$$

如果用 FP16 或 BF16 存储，每个元素 2 bytes，那么这个权重矩阵约为 32 MiB。它不是一个抽象矩阵，而是一块需要从 HBM 读入、经过缓存层级、再被计算单元消费的数据。

推理时可以先粗略区分两种形态。第一种是 prefill，也就是一次处理已有上下文中的多个 token。假设一次处理 $seq=4096$ 个 token，那么 $X'$ 的形状大约是 $[4096,4096]$，$W$ 是 $[4096,4096]$，输出也是 $[4096,4096]$。计算量约为：

$$
2 \times 4096^3 \approx 1374 \text{ 亿 FLOPs}
$$

此时同一份 $W$ 可以被许多 token 行复用，矩阵乘规模大，更容易把 Tensor Core 喂饱。

第二种是 decode，也就是模型逐步生成新 token。假设每次只生成一个 token，$X'$ 的形状接近 $[1,4096]$，$W$ 仍然是 $[4096,4096]$。计算量约为：

$$
2 \times 1 \times 4096 \times 4096 \approx 3355 \text{ 万 FLOPs}
$$

看起来计算量也不小，但权重矩阵仍然可能需要被大量读取，数据复用机会远低于 prefill。于是 decode 阶段更容易受到 HBM 带宽、KV cache 读取、batch 组织和 kernel launch 开销影响。这就是为什么 LLM 推理系统特别重视 batching、KV cache 管理和 attention kernel 优化。

这个例子先不用追求 profiler 级别的精确，只要记住一个稳定直觉：**大矩阵乘能否快，不只看 FLOPs，还要看这些 FLOPs 是否建立在足够高的数据复用之上。**

现在，$Y=XW$ 又多了一层含义：它不仅是一组并行点积，还是一场数据搬运与复用的组织问题。上一节回答了“数据从哪里来”；下一节回答“数据到达计算附近后，哪些硬件单元真正消耗它”。

## 3. SM 架构

### 3.1 SM 是 GPU 的基本计算单元

上一节讲的是数据怎么靠近计算单元。现在看真正消费这些数据的硬件。可以先把 GPU 想成由很多计算“车间”组成，每个主要车间就是一个 SM（Streaming Multiprocessor）。不同 GPU 代际的 SM 细节会变化，但初学者可以先抓住几个稳定组件：

| 组件 | 作用 | 学习重点 |
| --- | --- | --- |
| Warp Scheduler | 选择就绪 Warp 并发射指令 | 用切换 Warp 的方式隐藏访存延迟； |
| Register File | 为线程提供私有寄存器 | 寄存器用得太多会限制同时驻留的 Warp 数； |
| Shared Memory / L1 | SM 附近的片上存储 | Block 内线程协作复用数据的关键； |
| CUDA Core | 通用标量/向量计算单元 | 适合通用算术逻辑，不是所有计算都走 Tensor Core； |
| Tensor Core | 矩阵乘累加专用单元 | LLM 中 GEMM、QKV projection、MLP 等高度依赖它； |

以 A100 产品实现为例，它启用了 108 个 SM。这个数字意味着什么？它不是说只有 108 个任务能并行，而是说 GPU 有 108 个主要计算“车间”。每个 SM 内部又可以驻留多个 Block、多个 Warp，大量线程在这些 SM 上分批执行。更底层的 GA100 芯片规格与具体产品启用配置可能不同，所以教程里谈硬件数量时要尽量说明产品形态。

继续看 $4096 \times 4096$ 的输出矩阵。如果假设一个输出 tile 是 $128 \times 128$，那么输出矩阵可以被切成：

$$
32 \times 32 = 1024
$$

个输出 tile。若粗略假设一个 Block 负责一个或几个输出 tile，这些 Block 会分批分配到 108 个 SM 上执行。真实 cuBLAS 或 CUTLASS 风格 kernel 会更复杂：它们会沿 $M/N/K$ 维度选不同 tile 形状，使用流水、双缓冲和寄存器累加。但对于第一篇教程来说，1024 个 tile 足以说明 GPU 为什么能把一个矩阵乘变成许多并行工作。

### 3.2 Warp：32 个线程绑在一起执行

CUDA 编程里最小的显式执行实例是 Thread，但 NVIDIA GPU 的硬件调度常以 Warp 为单位。一个 Warp 通常包含 32 个 Thread。

NVIDIA 把这种模型称为 SIMT（Single Instruction, Multiple Threads）。可以把它理解为：同一个 Warp 内的线程通常执行同一条指令，但每个线程处理不同数据。Volta 之后的架构引入了更细粒度的 Independent Thread Scheduling，不过对性能直觉来说，同一 Warp 内控制流越一致、访存越规整，通常仍然越容易获得高效率。对矩阵乘来说，这很自然：一组线程可以同时处理不同输出元素、不同矩阵片段，执行路径高度相似。

Warp 模型带来两个关键后果。

第一，访存模式很重要。如果同一个 Warp 内相邻线程访问连续地址，硬件更容易把访问合并成高效的内存事务；如果线程访问地址很分散，就会浪费带宽。

第二，分支发散有代价。如果一个 Warp 内部分线程走 `if` 分支 A，另一部分线程走分支 B，硬件不能让同一 Warp 同时完整执行两条不同路径。通常会分段执行不同分支，未走当前路径的 lanes 被 mask 掉。于是实际有效吞吐下降。

这解释了为什么 LLM 的大矩阵乘天然适合 GPU：矩阵乘结构规则、数据布局可优化、控制分支少，非常适合 Warp 级 SIMT 执行。相反，如果一个算子有大量不规则分支、随机访存或稀疏控制逻辑，GPU 的高吞吐优势就会更难发挥。

### 3.3 Tensor Core：矩阵乘的专用加速器

CUDA Core 可以执行通用数值计算，但 LLM 中最重要的一类工作是矩阵乘累加。为此，NVIDIA GPU 提供了 Tensor Core 这样的专用矩阵计算单元。

Tensor Core 做的不是“理解神经网络”，而是高吞吐地执行类似下面的矩阵 tile 操作：

$$
D = A \times B + C
$$

其中 $A$、$B$、$C$、$D$ 是小矩阵 tile。大型 GEMM 会被拆成许多这样的 tile 级操作，再组合成完整输出。

![SM、Warp、CUDA Core 与 Tensor Core|900](imgs/gpu-sm-warp-tensorcore-handdrawn-cn-v2.png)

用 A100 的官方峰值做量级对比：FP32 CUDA Core 峰值约 19.5 TFLOPS，而 FP16 Tensor Core 峰值约 312 TFLOPS，不考虑稀疏加速时峰值差距约为 16 倍。这不是说任意 `float16` 代码都会自动快 16 倍，而是说明：当问题能被组织成 Tensor Core 友好的矩阵乘，并且数据供应、tile 形状、精度路径、kernel 实现都匹配时，硬件提供了远高于通用路径的矩阵吞吐上限。

常见精度路径可以只从硬件层面先这样理解：

| 精度路径 | 本篇需要记住什么 |
| --- | --- |
| FP32 | 通用高精度语义，具体可能走 CUDA Core FP32 路径，也可能在框架默认设置下使用 TF32 Tensor Core 近似加速；
| TF32 | Ampere 之后面向 FP32 矩阵乘代码的 Tensor Core 加速路径；
| FP16 | LLM 训练和推理中的经典半精度路径；
| BF16 | 指数范围接近 FP32，LLM 训练和推理常用；
| FP8 | Hopper/H100、Blackwell 等后续架构上的重要低精度矩阵计算方向，A100 不提供 FP8 Tensor Core 路径；
| INT8 / INT4 | 推理量化常见，但量化策略不属于本篇主线；

这里要克制边界：本篇只解释“为什么低精度矩阵路径能更快”，不展开 INT8/INT4 量化如何校准、哪些层要保护、反量化开销如何权衡。这些属于后续量化与推理优化专题。

### 3.4 矩阵乘 tile 如何喂给 Tensor Core

在 $Y=XW$ 中，输出矩阵的一个 tile 不是凭空算出来的。它需要读取 $X$ 的某些行块和 $W$ 的某些列块，沿着 $d_{model}$ 这个 K 维度不断做乘加累积。

可以粗略想象成三层切分：

- 输出矩阵 $Y$ 在 $M=(B \cdot seq)$ 和 $N=d_{out}$ 方向上被切成许多输出 tile；
- 每个输出 tile 沿 $K=d_{model}$ 方向分多轮读取 $X$ tile 和 $W$ tile；
- SM 内部的 Warp 和 Tensor Core 处理更小的矩阵片段，并把累加结果留在寄存器中；

这也是为什么高性能 GEMM kernel 会非常重视 tile 形状。tile 太小，不能充分利用 Tensor Core；tile 太大，寄存器和 Shared Memory 压力过高，反而减少可驻留 Warp 或导致调度效率下降。所谓“优化矩阵乘”，不是只把 for 循环改写得漂亮，而是在硬件存储层级、SM 资源、Warp 调度和 Tensor Core tile 之间做平衡。

此时，$Y=XW$ 已经从“很多输出元素”变成了“许多输出 tile 在多个 SM 上分批执行，每个 tile 又被拆成 Tensor Core 可消费的小矩阵片段”。硬件层面讲清楚后，还需要一层软件映射：PyTorch 或 CUDA 程序到底如何把这些工作交给 GPU。

## 4. CUDA 执行模型

### 4.1 从软件层次映射到硬件

大多数大模型学习者不需要一开始就手写 CUDA，但必须知道框架 API 最终会落到怎样的执行模型上。到这里我们已经知道硬件如何组织计算，但 PyTorch 代码不会直接操作 SM 和 Warp，所以还需要一层从软件任务到硬件执行的映射。CUDA 的基本层次是：

```text
Kernel -> Grid -> Block -> Thread
```

硬件执行时，还要理解：

```text
Thread -> Warp -> Block -> SM
```

它们的关系可以这样记：

| 层级 | 含义 | 直觉 |
| --- | --- | --- |
| Kernel | 一段在 GPU 上执行的函数 | 一次提交给 GPU 的并行任务；
| Grid | 一个 kernel 启动时的所有 Block | 整个任务网格；
| Block | 一组可以协作的 Thread | 同一 Block 内可共享 Shared Memory，并可做同步；
| Thread | 最小程序执行实例 | 处理一小份数据；
| Warp | 通常 32 个 Thread 组成的硬件调度单位 | 理解 SIMT、合并访存和分支发散的关键；
| SM | Block 被调度驻留的硬件执行单元 | 常规 thread block 语义下，同一个 Block 必须驻留在同一个 SM 上；

![CUDA 执行模型如何映射到 GPU 硬件|900](imgs/gpu-cuda-execution-mapping-handdrawn-cn-v2.png)

“同一个 Block 必须驻留在同一个 SM 上”是常规 thread block 语义下非常关键的边界。因为 Block 内线程能够共享 Shared Memory，并且可以通过同步原语协作。如果一个 Block 跨多个 SM，Shared Memory 和同步语义就很难成立。CUDA 的设计把 Block 作为一个局部协作单位，把 Grid 作为全局并行任务集合。Hopper 之后的 thread block cluster 和 distributed shared memory 属于更高级的协作模型，本篇不展开。

把 $Y=XW$ 交给 CUDA 时，一个直观映射是：Grid 覆盖整个输出矩阵；每个 Block 负责一个或多个输出 tile；Block 内的 Thread 被组织成若干 Warp；Warp 内线程协同加载数据、执行乘加、累加局部结果；最终把对应的 $Y$ tile 写回显存。真实高性能库会比这个模型复杂，但初学者先抓住这个映射，就能把 PyTorch 里的 `matmul` 和 GPU 硬件联系起来。

### 4.2 Kernel launch 与异步执行：为什么测量会错

CUDA kernel 启动通常是异步的。CPU 提交 kernel 后，不一定等 GPU 完成计算才继续执行下一行代码。框架会通过 stream、event 和同步点组织依赖。

这对初学者有一个直接影响：测 GPU 时间时，如果用 CPU 侧墙钟时间，需要在计时边界同步。

不严谨的写法是：

```python
import time
import torch

x = torch.randn(4096, 4096, device="cuda")
w = torch.randn(4096, 4096, device="cuda")

t0 = time.time()
y = x @ w
print(time.time() - t0)
```

这段代码可能主要测到 CPU 提交 kernel 的时间，而不是 GPU 真正完成矩阵乘的时间。更稳的写法是：

```python
import time
import torch

x = torch.randn(4096, 4096, device="cuda")
w = torch.randn(4096, 4096, device="cuda")

torch.cuda.synchronize()
t0 = time.time()
y = x @ w
torch.cuda.synchronize()
print(time.time() - t0)
```

更专业的 kernel 计时可以使用 CUDA events、PyTorch Profiler、Nsight Systems 或 Nsight Compute。这里先记住一点：GPU 不是 CPU 的同步函数调用，很多操作是排队提交、异步执行的。

### 4.3 Occupancy：并发不是越多越好

Occupancy 描述的是一个 SM 上实际驻留的 Warp 数量与理论最大可驻留 Warp 数量之间的比例。它受多种资源限制影响：

- 每个线程使用多少寄存器；
- 每个 Block 使用多少 Shared Memory；
- 每个 Block 有多少 Thread；
- 每个 SM 最多能驻留多少 Block 和 Warp；
- kernel 的实现是否有足够并行工作；

Occupancy 的意义在于 latency hiding。访问 HBM 的延迟很高，如果一个 SM 上只有很少 Warp，一旦它们都在等数据，计算单元就会空闲。更高的 occupancy 往往意味着有更多就绪 Warp 可供调度器切换，从而隐藏内存延迟。

但“occupancy 越高越好”也是错误的。它和性能的关系取决于算子类型。

| 算子类型 | Occupancy 的重要性 | 更关键的观察 |
| --- | --- | --- |
| Memory-bound | 通常更敏感 | LayerNorm、Softmax、element-wise、decode KV cache 读取等场景，需要更多并发来隐藏访存延迟；
| Compute-bound | 不一定越高越好 | 大 GEMM 更关键的是 Tensor Core 利用率、tile 形状、数据复用、流水是否充分；
| 混合型算子 | 需要 profiler 判断 | attention、fused kernel、采样等可能同时受计算、访存和调度影响；

高性能 GEMM kernel 有时会故意使用更多寄存器或 Shared Memory 来提高数据复用，导致 occupancy 不是满的，但整体更快。相反，一个逐元素算子如果 occupancy 太低，可能根本没有足够 Warp 来覆盖显存访问延迟。

因此，正确表述不是“occupancy 不重要”，而是：

**occupancy 是延迟隐藏能力的重要线索，但不是最终目标；最终目标是让当前算子的主要瓶颈被正确缓解。**

### 4.4 矩阵乘如何映射到 Grid 和 Block

如果 $Y$ 是 $4096 \times 4096$，一种教学上的粗略切分是把输出切成 $128 \times 128$ 的 tile，于是得到 1024 个输出 tile。每个 tile 可以由一个或多个 Block 协作完成，Block 内部再由多个 Warp 处理更小片段。

这个划分要同时考虑三件事：

- 输出 tile 要足够大，才能让 Tensor Core 做充分的矩阵 tile 计算；
- Block 使用的寄存器和 Shared Memory 不能过高，否则 occupancy 会过低；
- $X$ tile 和 $W$ tile 要能在片上存储中复用，否则 HBM 读写会拖慢整体；

对初学者来说，不需要马上设计最优 GEMM kernel。更重要的是建立映射：`torch.matmul(x, w)` 不是一个黑盒魔法，它最终会变成一批 kernel；kernel 把输出空间切成许多 Block；Block 被调度到 SM；Warp 执行 SIMT 指令；Tensor Core 消费小矩阵 tile；存储层级负责让数据尽量少从 HBM 重读。

这一章给 $Y=XW$ 加上的含义是“提交与调度”：同一个矩阵乘不仅要能切成 tile，还要被组织成 kernel、Grid、Block、Thread，并在 SM 上以 Warp 为单位执行。接下来，才能讨论这些工作为什么有时快、有时慢。

## 5. 三类性能瓶颈

### 5.1 先把瓶颈变成可判断的问题

现在前面的几件事已经连起来了：工作要能拆开，数据要能送到附近，计算单元要被喂饱，软件还要把任务排进 GPU。带着这些背景再看性能瓶颈，compute-bound、memory-bound、communication-bound 就不再只是三个英文标签。

现在可以把前面的内容归纳成一个诊断框架：

| 瓶颈类型 | 直觉 | 常见 LLM 场景 |
| --- | --- | --- |
| Compute-bound | 计算单元接近饱和，时间主要花在算 | 大 batch GEMM、prefill 中的大矩阵乘、训练中的 dense GEMM；
| Memory-bound | 计算单元在等数据，时间主要花在搬运 | decode KV cache 读取、LayerNorm、Softmax、element-wise、小 batch 推理；
| Communication-bound | 多设备之间等待数据交换或同步 | TP all-reduce、PP stage 边界、DP 梯度同步、多节点推理；

![GPU 性能瓶颈诊断框架|900](imgs/gpu-bottleneck-diagnosis-prefill-decode-handdrawn-cn-v3.png)

这三类不是互斥标签。一个 LLM 系统可能 prefill 更接近 compute-bound，decode 更接近 memory-bound，多卡 TP 又在某些层上受 communication-bound 影响。真正的判断通常需要 profiler，但这个框架能帮助你先问对问题。

### 5.2 Compute-bound：算力是主矛盾

Compute-bound 的直觉是：数据供应基本跟得上，主要时间花在计算上。典型例子是大规模 GEMM。此时继续减少一点显存读写未必是最主要收益，真正关键可能是：

- Tensor Core 是否被用上；
- 数据类型是否走到了合适精度路径；
- tile 形状是否适合硬件；
- batch 和序列长度是否足够大；
- kernel 是否有足够高的矩阵吞吐；

在 $Y=XW$ 中，prefill 阶段通常更容易接近 compute-bound。因为一次处理多个 token，$W$ 可以被许多 $X$ 行复用，矩阵形状大，Tensor Core 更容易保持忙碌。不过这不是绝对规律：如果 batch、seq、kernel 选择或硬件资源不同，仍然需要 profiler 验证。

### 5.3 Memory-bound：带宽是主矛盾

Memory-bound 的直觉是：计算单元并没有被喂饱，它们经常在等待数据。典型例子包括 LayerNorm、Softmax、element-wise 操作、采样、decode 阶段频繁读取 KV cache，以及小 batch 下的矩阵乘。

此时优化方向通常不是“再加一点计算单元”，而是：

- 减少 HBM 读写次数；
- 提高数据局部性和复用；
- 改善内存访问连续性；
- 使用 fused kernel 减少中间结果落回 HBM；
- 通过 batching 提高每次读入数据的计算利用率；

这也解释了 FlashAttention 一类 IO-aware 方法为什么重要：它的核心教学价值不是“换了一个 attention 公式”，而是通过 tiling 和重计算等策略减少 HBM 访问，把原本昂贵的数据搬运压力降下来。

在 $Y=XW$ 中，小 batch decode 阶段更容易 memory-bound。每次只处理少量新 token，权重和 KV cache 的读取压力难以被大量计算摊薄。即使 GPU 峰值 FLOPs 很高，也可能因为数据搬运跟不上而跑不满。

### 5.4 Communication-bound：多卡时通信是主矛盾

单卡 GPU 学明白之后，后续还会进入多卡推理和训练。多卡不是把 GPU 数量乘上去就自动线性加速，因为设备之间需要交换数据。

常见通信包括：

- Tensor Parallelism 中的 all-reduce 或 all-gather；
- Pipeline Parallelism 中 stage 之间传递激活；
- Data Parallelism 中训练梯度同步；
- Expert Parallelism 中 token dispatch 和 combine；

当通信时间占主导时，单卡 kernel 本身可能并不慢，但整体 step time 或 token latency 被跨卡链路拖住。此时需要关注 NVLink、PCIe、InfiniBand、NCCL collective、通信重叠、分片策略和拓扑。

本篇不展开多卡算法，只建立一个硬件直觉：**一旦张量被切到多张 GPU 上，性能就不只由每张卡的 SM 和 HBM 决定，还由卡与卡之间的数据交换决定。**

### 5.5 用 Arithmetic Intensity 做第一层判断

Arithmetic Intensity 可以帮助你把问题先归类。

对于矩阵乘：

$$
(M \times K) \cdot (K \times N) \rightarrow (M \times N)
$$

计算量大约是：

$$
2MKN
$$

如果 $M$、$N$、$K$ 都很大，且数据能被有效复用，那么每读入一批数据可以做大量 FLOPs，计算强度高，更可能 compute-bound。

如果 $M$ 很小，例如 decode 中 $M$ 接近 batch 或 token 数，而 $K$、$N$ 很大，那么读取权重和 KV cache 的代价很难被大量计算摊薄，更容易 memory-bound。

这不是最终判决，而是第一层提问方式：

- 如果 compute-bound，是否用上 Tensor Core，矩阵形状是否足够好；
- 如果 memory-bound，是否反复读写 HBM，是否能 fusion、tiling、batching 或减少 KV cache 搬运；
- 如果 communication-bound，是否有多卡 collective、stage 边界或跨节点通信在等待；

后续学习 TP、DP、PP、EP、KV cache、PagedAttention、FlashAttention、量化和 serving scheduler 时，都可以把这些问题带进去。很多“为什么这个优化有效”的答案，本质上都可以归入这三类瓶颈。

## 6. 本篇总结与系列导航

GPU 不是“更快的 CPU”，而是为高吞吐并行计算设计的处理器。它牺牲了许多单线程低延迟和复杂控制能力，换来大量线程、SM、Warp、片上存储和专用矩阵计算单元。大模型之所以适合 GPU，是因为 Transformer 中存在大量规则张量计算，尤其是 $Y=XW$ 这类矩阵乘，可以被拆成海量 tile 并行执行。

理解 GPU，要同时抓住两条线：一条是计算如何组织，另一条是数据如何移动。SM、Warp、CUDA Core 和 Tensor Core 解释“怎么算”；HBM、L2、Shared Memory、Registers 解释“数据从哪里来”；CUDA 的 Kernel、Grid、Block、Thread 解释“软件如何把任务交给硬件”。最后，compute-bound、memory-bound、communication-bound 则帮助我们把这些知识变成性能判断能力。

本篇之后，再学习 Attention、Transformer、KV cache、FlashAttention、PagedAttention、Tensor Parallelism、Pipeline Parallelism、Expert Parallelism、量化和推理服务时，就不会只看到一串框架名词。FlashAttention 可以先从减少 HBM 访问理解；PagedAttention 和 KV cache 管理可以先从显存分配与复用理解；TP、PP、EP 可以先从 communication-bound 理解；batching 和 scheduler 可以先从 compute/memory utilization 理解。你会开始追问：这个机制是在减少计算、减少 HBM 访问、提高 Tensor Core 利用率、改善 batching，还是在降低跨卡通信？这正是大模型与推理全栈学习中最重要的底层视角之一。

## 参考资料

1. NVIDIA A100 Tensor Core GPU Datasheet：A100 SM 数量、Tensor Core 峰值吞吐、HBM 带宽等硬件规格；https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf；
2. NVIDIA A100 Tensor Core GPU Architecture：Ampere 架构、第三代 Tensor Core、TF32/BF16/FP16 等能力说明；https://images.nvidia.com/aem-dam/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf；
3. NVIDIA H100 Tensor Core GPU：Hopper 架构、第四代 Tensor Core 与 FP8 Transformer Engine 背景；https://www.nvidia.com/en-us/data-center/h100/；
4. NVIDIA CUDA C++ Programming Guide：CUDA 编程模型、线程层级、内存层级、SIMT、Warp 与 Block 语义；https://docs.nvidia.com/cuda/cuda-c-programming-guide/；
5. NVIDIA CUDA C++ Best Practices Guide：occupancy、内存访问、性能优化和延迟隐藏相关建议；https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/；
6. NVIDIA Matrix Multiplication Background User Guide：矩阵乘的维度、tile、性能背景与深度学习中的 GEMM 形态；https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html；
7. NVIDIA Deep Learning Performance Guide：深度学习算子性能、Tensor Core 使用和通用性能分析入口；https://docs.nvidia.com/deeplearning/performance/index.html；
8. NVIDIA Nsight Compute Documentation：kernel 级 GPU 性能分析、occupancy、memory throughput、roofline 等观察方式；https://docs.nvidia.com/nsight-compute/；
9. NVIDIA Nsight Systems Documentation：系统时间线、CPU/GPU 并发、kernel launch 与多进程/多线程分析；https://docs.nvidia.com/nsight-systems/；
10. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness：IO-aware attention 与减少 HBM 读写的典型论文；https://arxiv.org/abs/2205.14135；
11. vLLM / PagedAttention：KV cache block 管理和推理 serving 中的内存管理背景；https://docs.vllm.ai/en/latest/design/paged_attention/；

## 学习测评

### 题目

1. 单选：CPU 与 GPU 设计哲学的核心差异，哪项最准确？
   A. CPU 主要追求少量复杂任务的低延迟处理，GPU 主要追求大量相似任务的高吞吐处理；
   B. CPU 和 GPU 都主要追求单线程低延迟，只是 GPU 的显存容量更大；
   C. GPU 的优势主要来自每个线程都比 CPU 核心更复杂；
   D. CPU 和 GPU 的差异主要是显存容量，不是执行方式；

2. 单选：把 `Linear` 层写成 $Y=XW$ 后，为什么它适合用来理解 GPU 并行？
   A. 因为输出矩阵中有大量位置可以被切成 tile 分配给不同计算资源；
   B. 因为权重矩阵只需要读取一次，后续不会再产生数据搬运压力；
   C. 因为输出元素之间存在强串行依赖，必须按顺序逐个完成；
   D. 因为只要有矩阵乘，就一定能达到硬件峰值；

3. 多选：以下哪些属于本文讨论的 GPU 存储或缓存层级，而不是计算单元？
   A. Registers；
   B. Shared Memory / L1；
   C. L2 Cache；
   D. Tensor Core；

4. 单选：为什么高性能矩阵乘会重视 tile 和片上复用？
   A. 因为 tile 可以让同一小块 $X$ 和 $W$ 被多个乘加操作复用，减少反复从 HBM 搬数据；
   B. 因为只要 tile 足够大，就一定能消除全部 HBM 访问；
   C. 因为 tile 只影响输出矩阵的编号方式，不影响数据搬运和复用；
   D. 因为 tile 越小越容易调度，所以可以不考虑 Tensor Core 利用率；

5. 单选：Arithmetic Intensity 的含义最接近哪一项？
   A. 每秒启动多少个 kernel；
   B. 每搬运 1 byte 数据能做多少 FLOPs；
   C. 每个 Warp 包含多少 Thread；
   D. 每个 Block 能否跨多个 SM；

6. 多选：关于 prefill 与 decode 的性能形态，哪些判断更合理？
   A. prefill 通常矩阵形状更大，更容易把 Tensor Core 喂饱；
   B. decode 每次 token 粒度小，更容易暴露 KV cache 和权重读取压力；
   C. decode 一定完全 compute-bound，因此不用关注 HBM 或 KV cache；
   D. prefill 与 decode 的瓶颈需要结合 batch、seq、kernel 和 profiler 判断；

7. 单选：SM（Streaming Multiprocessor）在初学阶段可以怎样理解？
   A. GPU 中主要的计算“车间”，Block 和 Warp 会被调度到 SM 上执行；
   B. 专门存放全部模型权重的显存区域；
   C. CPU 上负责提交 kernel 的线程；
   D. 只负责保存 Shared Memory，不参与 Warp 调度和算术执行；

8. 单选：理解 Warp 通常由 32 个 Thread 组成，最重要的学习价值是什么？
   A. 证明每个 Block 只能包含 32 个 Thread；
   B. 帮助理解 GPU 为什么偏好同一 Warp 内规则控制流和连续访存；
   C. 证明 Tensor Core 只能处理 $32 \times 32$ 矩阵；
   D. 说明 CUDA 程序不需要考虑 Block；

9. 单选：一个 Warp 中 20 个线程走 `if` 分支，12 个线程走 `else` 分支，最合理的性能直觉是什么？
   A. GPU 会把这个 Warp 自动拆成两个完全独立且无额外代价的 Warp；
   B. 分支路径通常需要分段推进，未走当前路径的 lanes 会被 mask，有效利用率下降；
   C. 只要使用 Tensor Core，Warp divergence 就完全不存在；
   D. 这种情况只影响 CPU，不影响 GPU；

10. 多选：Tensor Core 适合加速哪些类型的工作？
    A. 矩阵乘累加；
    B. LLM 中的 QKV projection；
    C. MLP 中的大 GEMM；
    D. 大量不规则分支控制逻辑；

11. 单选：以 A100 的峰值规格做教学对比时，FP16 Tensor Core 峰值远高于 FP32 CUDA Core 峰值，这个事实最应该如何理解？
    A. 任意 FP16 代码都会自动快 16 倍；
    B. 只要模型变成 FP16，就不会有内存瓶颈；
    C. 当工作负载、精度路径、tile 形状和 kernel 都适合 Tensor Core 时，硬件提供了更高的矩阵吞吐上限；
    D. CUDA Core 已经没有任何用途；

12. 多选：CUDA 执行模型中，哪些说法正确？
    A. Kernel 启动一个 Grid；
    B. Grid 包含多个 Block；
    C. 同一个 Block 内线程可以使用 Shared Memory 协作；
    D. 一个常规 Block 可以跨多个 SM 共享同一块 Shared Memory；

13. 单选：为什么用 CPU 侧 `time.time()` 测 GPU 操作时常需要 `torch.cuda.synchronize()`？
    A. 因为 CUDA kernel 启动通常是异步的；
    B. 因为 `time.time()` 会自动读取 Tensor Core 计数器；
    C. 因为 synchronize 会让 kernel 选择更快算法；
    D. 因为没有 synchronize 就无法创建 CUDA 张量；

14. 多选：关于 occupancy，哪些说法正确？
    A. 它与 SM 上可驻留 Warp/Block 的程度有关；
    B. 它受寄存器、Shared Memory 和 Block 大小等资源约束；
    C. 对 memory-bound 算子，高 occupancy 常有助于隐藏访存延迟；
    D. 对所有算子，occupancy 越高性能一定越好；

15. 多选：如果一个算子 memory-bound，哪些优化方向更可能有意义？
    A. 减少 HBM 读写；
    B. 使用 fused kernel 减少中间结果落回显存；
    C. 改善内存访问连续性；
    D. 优先增加不会复用数据的额外计算；

16. 单选：communication-bound 最可能出现在什么场景？
    A. 单卡上一个小 element-wise kernel 反复读写 HBM；
    B. 多卡 TP all-reduce、PP stage 传递或 DP 梯度同步；
    C. 单个 SM 内 Warp 发生分支发散；
    D. 一个 Block 内线程使用 Shared Memory 复用 tile；

### 答案与解析

1. 答案：A。CPU 更偏向低延迟和复杂控制，GPU 更偏向高吞吐和大量相似工作并行。B、C、D 都把差异说窄或说错了；

2. 答案：A。$Y=XW$ 的输出空间可以被切成大量 tile，这能自然引出并行、数据复用、硬件映射和瓶颈判断。B 忽略权重读取，C 把并行问题说成串行，D 把硬件峰值当成了自动结果；

3. 答案：A、B、C。Registers、Shared Memory/L1 和 L2 都属于存储或缓存层级；Tensor Core 是计算单元，不是存储层级；

4. 答案：A。tile 的价值在于让数据靠近计算单元并被多次复用。B 夸大了 tile 的能力，C 忽略了数据搬运，D 忽略了 tile 形状与硬件资源之间的平衡；

5. 答案：B。Arithmetic Intensity 衡量每搬运 1 byte 数据能做多少 FLOPs，是判断 compute-bound / memory-bound 的重要直觉工具；

6. 答案：A、B、D。prefill 通常矩阵规模大、复用更好；decode token 粒度小，KV cache 与权重读取压力更突出。但最终判断仍要结合实际配置和 profiler。C 把 decode 说成必然 compute-bound，是危险的绝对化；

7. 答案：A。SM 是 GPU 内部主要执行单元，可以先类比成计算车间。它不是显存、CPU 线程或通信链路；

8. 答案：B。Warp 大小不是为了死记硬背，而是帮助理解 SIMT、合并访存和分支发散。一个 Block 可以有多个 Warp，Tensor Core tile 也不是简单等同于 $32 \times 32$；

9. 答案：B。SIMT 模型下，同一 Warp 中不同分支路径通常要分段推进，未走当前路径的 lanes 会被 mask，导致有效利用率下降。Volta 之后调度更细，但分支一致性仍然是重要性能直觉；

10. 答案：A、B、C。Tensor Core 的核心价值是高吞吐矩阵乘累加，LLM 中 QKV projection、O projection、MLP 等大 GEMM 都高度相关。D 这类不规则分支控制逻辑即使出现在 GPU kernel 中，也不属于 Tensor Core 擅长的矩阵 tile 乘加工作；

11. 答案：C。峰值差距说明硬件为矩阵类低精度计算提供了更高上限，但是否兑现取决于工作负载、数据类型、kernel 和数据供应；

12. 答案：A、B、C。A、B、C 是 CUDA 基本层级和 Block 协作语义。D 错，常规 thread block 不跨多个 SM 共享同一块 Shared Memory；更高级的 thread block cluster 不属于本篇主线；

13. 答案：A。CUDA kernel launch 常异步返回，不同步就可能只测到 CPU 提交任务的时间；

14. 答案：A、B、C。Occupancy 是延迟隐藏的重要线索，但不是所有算子的最终目标。D 错，大 GEMM 可能为了更高数据复用牺牲部分 occupancy；

15. 答案：A、B、C。Memory-bound 的核心是数据搬运压力，减少 HBM 读写、fusion、改善访问连续性都可能有效。D 如果没有提高复用，只是增加无关计算，通常不能解决瓶颈；

16. 答案：B。Communication-bound 来自设备之间的数据交换或同步，多卡并行策略中的 collective 和 stage 边界是典型来源。A 更接近 memory-bound，C 是 Warp 内执行效率问题，D 是片上复用策略；
